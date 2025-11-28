import os
import sys
import zipfile
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
from time import time
from shapely.ops import linemerge, unary_union
from tqdm import tqdm
from shapely.ops import substring
from shapely.geometry import Point, LineString, MultiLineString, GeometryCollection

sys.path.append(r"D:\JACOBS\SERVICE\API")

from modules.table import sanitize_header, excel_styler
from modules.data import read_gdf
from modules.h3_route import identify_hexagon, retrieve_roads, build_graph
from modules.utils import route_path, dropwire_connection, create_topology
from modules.kml import read_kml, export_kml, sanitize_kml
from core.logger import create_logger
from core.config import settings
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA_DIR = settings.DATA_DIR
EXPORT_DIR = settings.EXPORT_DIR
MAINDATA_DIR = settings.MAINDATA_DIR
LOG_DIR = settings.LOG_DIR
MAX_WORKER = settings.MAX_WORKERS

# ------------------------------------------------------
# LOGGING CONFIGURATION
# ------------------------------------------------------
logger = create_logger(__file__)

# ------------------------------------------------------
# INSERT RING
# ------------------------------------------------------
def validate_insert(insert_sites:str | gpd.GeoDataFrame, kmz_data: str, sep="-"):
    # INSERT SITES
    if isinstance(insert_sites, str):
        insert_sites = read_gdf(str)
    elif isinstance(insert_sites, gpd.GeoDataFrame):
        insert_sites = sanitize_header(insert_sites, lowercase=True)
        insert_sites['long'] = insert_sites.geometry.to_crs(epsg=4326).x
        insert_sites['lat'] = insert_sites.geometry.to_crs(epsg=4326).y
    elif isinstance(insert_sites, pd.DataFrame):
        insert_sites = sanitize_header(insert_sites, lowercase=True)
        if 'long' not in insert_sites.columns:
            raise ValueError(f"Column long not defined in insert sites.")
        if 'lat' not in insert_sites.columns:
            raise ValueError(f"Column long not defined in insert sites.")
        insert_geom = gpd.points_from_xy(insert_sites['long'], insert_sites['lat'], crs='EPSG:4326')
        insert_sites = gpd.GeoDataFrame(insert_sites, geometry=insert_geom, crs="EPSG:4326")
    
    insert_sites['site_type'] = 'Site List'
    used_col = ['site_id', 'site_name', 'site_type','long', 'lat', 'geometry']
    for col in used_col:
        if col not in insert_sites.columns:
            raise ValueError(f"Column {col} not detected in Insert Sites data.")

    # KMZ DATA
    points_kmz, lines_kmz, _ = read_kml(kmz_data)  
    points_kmz = gpd.GeoDataFrame(points_kmz, geometry='geometry', crs='EPSG:4326') 
    lines_kmz = gpd.GeoDataFrame(lines_kmz, geometry='geometry', crs='EPSG:4326')  
    points_kmz = sanitize_header(points_kmz)
    lines_kmz = sanitize_header(lines_kmz)
    points_existing = points_kmz[~points_kmz['name'].str.lower().str.contains('connection')].copy()
    lines_existing = lines_kmz[lines_kmz['folder_name'].str.lower().str.contains('route')].copy()
    
    # POINT EXISTING
    points_existing['site_id'] = points_existing['name']
    points_existing['site_name'] = points_existing['name']
    points_existing['site_type'] = points_existing['folders'].str.split(";").str[-1]
    points_existing['site_type'] = np.where(points_existing['site_type'].str.lower().str.contains('hub'), "FO Hub", 'Site List')
    points_existing['long'] = points_existing.geometry.to_crs(epsg=4326).x
    points_existing['lat'] = points_existing.geometry.to_crs(epsg=4326).y
    points_existing['ring_name'] = points_existing['folders'].str.split(";").str[-2]
    points_existing['program'] = points_existing['folders'].str.split(";").str[-3]
    points_existing['geometry'] = points_existing.geometry.force_2d()
    points_existing['region'] = points_existing['folders'].str.extract(r'([A-Z]{3,6});')


    # LINES EXISTING
    lines_existing['segment'] = lines_existing['name']
    lines_existing['near_end'] = lines_existing['segment'].str.split(sep).str[0]
    lines_existing['far_end'] = lines_existing['segment'].str.split(sep).str[-1]
    lines_existing['geometry'] = lines_existing.geometry.force_2d()
    lines_existing['ring_name'] = lines_existing['folders'].str.split(";").str[-2]
    lines_existing['program'] = lines_existing['folders'].str.split(";").str[-3]
    lines_existing['region'] = lines_existing['folders'].str.extract(r'([A-Z]{3,6});')

    existing_col = ['site_id', 'site_name', 'site_type', 'long', 'lat', 'ring_name', 'program', 'region','geometry']
    for col in existing_col:
        if col not in points_existing.columns:
            raise ValueError(f"Column {col} not detected in Existing Point Sites data.")
    points_existing = points_existing[existing_col]

    existing_col = ['segment', 'near_end', 'far_end', 'ring_name', 'program', 'region','geometry']
    for col in existing_col:
        if col not in lines_existing.columns:
            raise ValueError(f"Column {col} not detected in Existing Lines Sites data.")
    lines_existing = lines_existing[existing_col]

    if points_existing.empty:
        raise ValueError(f"Point data in existing kmz is empty")

    if lines_existing.empty:
        raise ValueError(f"Lines data in existing kmz is empty")
    return insert_sites, points_existing, lines_existing


def identify_insert(insert_gdf:gpd.GeoDataFrame, lines_existing:gpd.GeoDataFrame, max_distance=3000):
    insert_gdf = insert_gdf.to_crs(epsg=3857)
    lines_existing = lines_existing.to_crs(epsg=3857)

    insert_reached = gpd.sjoin_nearest(insert_gdf, lines_existing[['geometry', 'near_end', 'far_end', 'ring_name']], how='inner', max_distance=max_distance, distance_col="dist_fiber").drop(columns='index_right')

    insert_not_reached = insert_gdf[~insert_gdf.index.isin(insert_reached.index)]
    insert_reached = insert_reached.reset_index(drop=True)
    insert_not_reached = insert_not_reached.reset_index(drop=True)
    return insert_reached, insert_not_reached


def build_connection(ring: str, to_insert:gpd.GeoDataFrame, target_fiber:gpd.GeoDataFrame, target_point:gpd.GeoDataFrame, max_member:int=None, start_column:str='near_end')-> tuple:
    if target_fiber.crs != 'EPSG:3857':
        target_fiber = target_fiber.to_crs(epsg=3857)
    if target_point.crs != 'EPSG:3857':
        target_point = target_point.to_crs(epsg=3857)
    if to_insert.crs != 'EPSG:3857':
        to_insert = to_insert.to_crs(epsg=3857)

    match start_column:
        case 'near_end':
            opposite_column = 'far_end'
        case 'far_end':
            opposite_column = 'near_end'
        case _:
            raise ValueError("start_column must be either 'near_end' or 'far_end'.")

    if max_member is not None and max_member < 4:
        raise ValueError("max_member must be at least 4 to form a valid ring.")
    
    # FO HUB AND SITELIST
    fo_hub = target_point[target_point['site_type'] == 'FO Hub'].reset_index(drop=True)
    site_list = target_point[target_point['site_type'] == 'Site List'].reset_index(drop=True)
    total_point = len(fo_hub) + len(site_list)

    # CHECK PREV CONNECTION
    hub_ids = fo_hub['site_id'].tolist()
    sitelist_ids = site_list['site_id'].tolist()
    start_hub = target_fiber[target_fiber[start_column].isin(hub_ids)][start_column].values
    
    if len(start_hub) == 0:
        start_hub = target_fiber[target_fiber[opposite_column].isin(hub_ids)][opposite_column].values
    if len(start_hub) == 0:
        print(f"❌ No FO Hub found in ring {ring}, cannot identify connection.")
        return None, None
    
    start_hub = start_hub[0]
    
    connection = []
    connection.append(start_hub)
    current = start_hub
    for idx in range(total_point - 1):
        current_site = target_fiber[target_fiber[start_column] == current]
        if current_site.empty:
            raise ValueError(f"No next site found connected to {current} in ring {ring}.")
        
        next_site = current_site.iloc[0][opposite_column]
        connection.append(next_site)
        current = next_site
        if current in hub_ids and current != start_hub:
            break
        if current in hub_ids and current == start_hub:
            break
    
    # CHECK INSERTION (NEW CONNECTION)
    to_insert = to_insert.sort_values(by='dist_fiber').reset_index(drop=True)
    to_insert = to_insert[~to_insert['site_id'].isin(connection)].reset_index(drop=True)
    near_dict = to_insert[start_column].value_counts().to_dict()

    # BUILD ROAD NETWORK
    hex_list = identify_hexagon(to_insert, type='convex')
    roads = retrieve_roads(hex_list, type='roads')
    nodes = retrieve_roads(hex_list, type='nodes')
    roads = roads.to_crs(epsg=3857)
    nodes = nodes.to_crs(epsg=3857)
    G = build_graph(roads, graph_type='route')

    if 'nearest_node' not in to_insert.columns:
        to_insert = gpd.sjoin_nearest(to_insert, nodes[['node_id', 'geometry']], how='left', distance_col='dist_to_node')
        to_insert = to_insert.rename(columns={'node_id': 'nearest_node'})

    if 'nearest_node' not in target_point.columns:
        target_point = gpd.sjoin_nearest(target_point, nodes[['node_id', 'geometry']], how='left', distance_col='dist_to_node')
        target_point = target_point.rename(columns={'node_id': 'nearest_node'})

    # MAPPING DISTANCE SEGMENT
    for idx, row in to_insert.iterrows():
        if pd.isna(row['nearest_node']):
            print(f"⚠️ No nearest node found for site {row['site_id']} in ring {ring}, skipping.")
            continue
        try:
            start_point = target_point[target_point['site_id'] == row[start_column]]
            start_node = start_point['nearest_node'].values[0]
            start_length = nx.shortest_path_length(
                G,
                source=row['nearest_node'],
                target=start_node,
                weight="length",
            )
            start_length = float(start_length) if not isinstance(start_length, (list, tuple, np.ndarray)) else float(start_length[0])
            to_insert.at[idx, f'distto_{start_column}'] = start_length
        except (nx.NetworkXNoPath, IndexError):
            to_insert.at[idx, f'distto_{start_column}'] = float('inf')
            print(f"⚠️ No path found from site {row['site_id']} to {row[start_column]} in ring {ring}, skipping.")
    
    to_insert = to_insert[to_insert[f'distto_{start_column}'] != float('inf')].reset_index(drop=True)
    to_insert = to_insert.sort_values(by=f'distto_{start_column}').reset_index(drop=True)

    new_connection = connection.copy()
    for near_id, count in near_dict.items():
        member_ids = [site for site in new_connection if site in sitelist_ids]
        member_count = len(member_ids)
        if max_member is not None and member_count >= max_member:
            print(f"⚠️ Maximum member limit of {max_member} reached for ring {ring}. Stopping insertion.")
            break
        
        insert_data = to_insert[to_insert[start_column] == near_id]
        if insert_data.empty:
            continue
        insert_ids = insert_data['site_id'].tolist()

        if near_id in new_connection:
            near_index = new_connection.index(near_id)
            for insert_id in insert_ids:
                if max_member is not None and len(new_connection) >= max_member:
                    print(f"⚠️ Maximum member limit of {max_member} reached during insertion for ring {ring}.")
                    break
                if insert_id not in new_connection:
                    new_connection.insert(near_index + 1, insert_id)
                    near_index += 1
        else:
            continue

    # ENSURE RING STARTS FROM START HUB
    if start_column == 'far_end':
        new_connection = new_connection[::-1]

    # UPDATE RING DATA
    points_new = []
    for site_id in new_connection:
        if site_id in fo_hub['site_id'].values:
            point_data = fo_hub[fo_hub['site_id'] == site_id].iloc[0]
        elif site_id in site_list['site_id'].values:
            point_data = site_list[site_list['site_id'] == site_id].iloc[0]
        elif site_id in to_insert['site_id'].values:
            point_data = to_insert[to_insert['site_id'] == site_id].iloc[0]
        else:
            print(f"⚠️ Site ID {site_id} still not found, skipping.")
            continue
        points_new.append(point_data)
    
    if not points_new:
        print(f"⚠️ No valid points found for ring {ring} after insertion.")
        return None, None
    points_new = gpd.GeoDataFrame(points_new, columns=points_new[0].index.tolist(), crs='EPSG:3857').reset_index(drop=True)
    points_new['note'] = points_new['site_id'].apply(lambda x: 'insert' if x in to_insert['site_id'].values else 'existing')
    return points_new, new_connection

# ---------------------------------------------------------
# ROUTING
# ---------------------------------------------------------
def relative_intersection(line_a: LineString | MultiLineString, line_b: LineString | MultiLineString, tolerance=0.0):
    """
    Detect overlapping portions of line_a and line_b based on
    projection distance (lowest → highest).
    
    Returns:
        overlap_geom: LineString / MultiLineString
        new_geom:     LineString / MultiLineString
    """

    if line_a.geom_type == "MultiLineString":
        line_a = linemerge(line_a)
        if line_a.geom_type == "MultiLineString":
            geoms = list(line_a.geoms)
            longest = geoms[0]
            for geom in geoms:
                if geom.length > longest.length:
                    longest = geom
        line_a = longest

    if line_b.geom_type == "MultiLineString":
        line_b = linemerge(line_b)
        if line_b.geom_type == "MultiLineString":
            geoms = list(line_b.geoms)
            longest = geoms[0]
            for geom in geoms:
                if geom.length > longest.length:
                    longest = geom
        line_b = longest


    if tolerance > 0:
        lineA = line_a.buffer(tolerance)
        lineB = line_b.buffer(tolerance)
    else:
        lineA = line_a
        lineB = line_b

    # ---- 1. Compute intersection ----
    inter = line_a.intersection(lineB)
    if inter is None:
        return None, line_a
    if inter.is_empty:
        return None, line_a

    # ---- 2. Extract all intersection endpoints ----
    pts = []

    # Intersection may be LineString, MultiLineString, or Points
    if isinstance(inter, LineString):
        pts.extend([inter.coords[0], inter.coords[-1]])

    elif isinstance(inter, MultiLineString):
        for seg in inter.geoms:
            pts.extend([seg.coords[0], seg.coords[-1]])

    else:
        for g in inter.geoms:
            pts.append((g.x, g.y))

    # ---- 3. Project intersection endpoints onto line_a ----
    dist_list = sorted([line_a.project(Point(p)) for p in pts])
    dist_list = sorted(list(set(dist_list)))

    if len(dist_list) < 2:
        return None, line_a

    start_d = dist_list[0]
    end_d   = dist_list[-1]
    overlap_geom = substring(line_a, start_d, end_d)

    # ---- 5. Compute remaining "new" segments ----
    new_segments = []
    if start_d > 0:
        new_segments.append(substring(line_a, 0, start_d))
    if end_d < line_a.length:
        new_segments.append(substring(line_a, end_d, line_a.length))

    if len(new_segments) == 1:
        new_geom = new_segments[0]
    else:
        new_geom = MultiLineString([s for s in new_segments if not s.is_empty])
    return overlap_geom, new_geom

def routing_insert(
    ring: str,
    new_connection: list,
    new_points: gpd.GeoDataFrame,
    target_fiber: gpd.GeoDataFrame,
    start_column: str = "near_end"
) -> gpd.GeoDataFrame:

    print(f"⚙️ Routing ring {ring} ...")

    # CRS normalizing
    if target_fiber.crs != "EPSG:3857":
        target_fiber = target_fiber.to_crs(3857)
    if new_points.crs != "EPSG:3857":
        new_points = new_points.to_crs(3857)

    # Opposing column
    opposite_column = "far_end" if start_column == "near_end" else "near_end"

    # -----------------------------
    # BUILD ROAD NETWORK
    # -----------------------------
    hex_list = identify_hexagon(new_points, type="convex")
    roads = retrieve_roads(hex_list, type="roads").to_crs(3857)
    nodes = retrieve_roads(hex_list, type="nodes").to_crs(3857)
    G = build_graph(roads, graph_type="fiber")

    # Nearest node assignment
    if "nearest_node" not in new_points.columns:
        new_points = gpd.sjoin_nearest( new_points, nodes[["node_id", "geometry"]], how="left", distance_col="dist_to_node").rename(columns={"node_id": "nearest_node"})

    # -----------------------------
    # START ROUTING
    # -----------------------------
    segments = []
    existing_cable = []
    new_cable = []
    for i in range(len(new_connection) - 1):
        start_id = new_connection[i]
        end_id = new_connection[i + 1]

        sp = new_points[new_points["site_id"] == start_id]
        ep = new_points[new_points["site_id"] == end_id]

        if sp.empty or ep.empty:
            print(f"🔴 Missing point data at {start_id} or {end_id}, skipping.")
            continue

        sp = sp.iloc[0]
        ep = ep.iloc[0]

        is_new = (sp["note"] == "insert") or (ep["note"] == "insert")

        # ----------------------------------------
        # NEW ROUTE — RUN DIJKSTRA
        # ----------------------------------------
        if is_new:
            start_node = sp["nearest_node"]
            end_node = ep["nearest_node"]
            if ep['note'] == "insert":
                startprev_id = start_id
                endprev_id = new_connection[i+2]
            else:
                startprev_id = new_connection[i-1]
                endprev_id = end_id
            
            sprev_id = new_points.loc[new_points["site_id"] == startprev_id, "site_id"].iloc[0]
            eprev_id = new_points.loc[new_points["site_id"] == endprev_id, "site_id"].iloc[0]

            prev_route = target_fiber[(target_fiber["near_end"] == sprev_id) & (target_fiber["far_end"] == eprev_id)].copy()
            prev_geom =  prev_route.geometry.union_all()

            if sp['note'] == 'insert' and ep['note'] == 'insert':
                prev_route = segments[-1]['geometry']
                exist_geom = target_fiber.geometry.union_all()
                prev_route = unary_union(exist_geom, prev_route)
            
            if prev_geom.geom_type == "MultiLineString":
                prev_geom = linemerge(prev_geom)

            try:
                path, path_geom, path_length = route_path(start_node, end_node, G, roads, merged=True)
                if path_length == 0:
                    path_geom = None
            except Exception:
                print(f"⚠️ No path between {start_id} → {end_id}")
                continue

            if path_geom is None:
                print(f"⚠️ Invalid geometry for segment {start_id} → {end_id}")
                continue

            # Dropwire adjustment
            path_geom, _ = dropwire_connection(path_geom, sp, ep, nodes, start_node, end_node)
            existing_line, new_line = relative_intersection(path_geom, prev_geom, tolerance=50)
            if new_line.geom_type == 'MultiLineString':
                insert_site = sp if sp['note'] =='insert' else ep
                insert_geom = insert_site.geometry
                insert_buff = insert_geom.buffer(5)
                new_line = linemerge(new_line)
                if new_line.geom_type == 'MultiLineString':
                    geoms = list(new_line.geoms)
                    intersect_insert = []
                    for geom in geoms:
                        if geom.intersects(insert_buff):
                            intersect_insert.append(geom)
                    if len(intersect_insert)  > 0:
                        intersect_insert = unary_union(intersect_insert)
                        new_line = intersect_insert
                    else:
                        new_line = LineString()
            if isinstance(existing_line, list):
                existing_line = LineString()
            if isinstance(new_line, list):
                new_line = LineString()
            
            # existing_line = path_geom.intersection(prev_geom)
            # new_line = path_geom.difference(existing_line)
            if not existing_line is None:
                if not existing_line.is_empty:
                    data = {"geometry": existing_line}
                    existing_cable.append(data)
            if not new_line is None:
                if not new_line.is_empty:
                    data = {"geometry": new_line}
                    new_cable.append(data)

            # new_length = new_line.length if new_line else 50
            # existing_length = existing_line.length if existing_line else 50
            new_length = new_line.length * 1.1 if new_line else 50
            existing_length = (existing_line.length * 1.1) + 500 if existing_line else 50
            total_length = new_length + existing_length
            segment_name = f"{start_id}-{end_id}"

            segments.append({
                "name": segment_name,
                start_column: start_id,
                opposite_column: end_id,
                "existing_length": round(existing_length, 2),
                "new_length": round(new_length, 2),
                "length": round(total_length, 2),
                "route_type": "New Route",
                "fo_note": "merged",
                "geometry": path_geom
            })
            print(f"🟢 NEW   {segment_name:<20} | {total_length:10.2f} m")

            # if not existing_line.is_empty:
            #     new_exist = existing_line.buffer(20)
            #     existing_buff = existing_buff.union(new_exist)

            # Reduce weight on used edges
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                if G.has_edge(u, v):
                    G[u][v]["weight"] *= 0.10
                if G.has_edge(v, u):
                    G[v][u]["weight"] *= 0.10

        # ----------------------------------------
        # EXISTING ROUTE
        # ----------------------------------------
        else:
            seg = target_fiber[
                (target_fiber["near_end"] == start_id) &
                (target_fiber["far_end"] == end_id)
            ]

            if seg.empty:
                seg = target_fiber[
                    (target_fiber["near_end"] == end_id) &
                    (target_fiber["far_end"] == start_id)
                ]

            if seg.empty:
                print(f"🟠 No existing segment for {start_id} ↔ {end_id}")
                continue
            
            union_line = seg.geometry.union_all()
            merged = linemerge(union_line) if union_line.geom_type == "MultiLineString" else union_line
            # seg_length = merged.length if merged else 0.0
            seg_length = merged.length * 1.1 if merged else 0.0

            segment_name = f"{start_id}-{end_id}"
            segments.append({
                "name": segment_name,
                start_column: start_id,
                opposite_column: end_id,
                "existing_length": 0.0,
                "new_length": 0.0,
                "length": seg_length,
                "route_type": "Existing Route",
                "fo_note": "merged",
                "geometry": merged
            })
            print(f"🟢 EXIST {segment_name:<20} | {seg_length:10.2f} m")

    # if len(existing_cable) > 0:
    #     existing_cable = gpd.GeoDataFrame(existing_cable, geometry='geometry', crs=3857)
    #     existing_cable.to_parquet(fr"D:\JACOBS\SERVICE\API\test\Trial Insert Ring TX Expansion 2026 V2\20251128\Identify Route\Existing Cable {ring}.parquet")

    # if len(new_cable) > 0:
    #     new_cable = gpd.GeoDataFrame(new_cable, geometry='geometry', crs=3857)
    #     new_cable.to_parquet(fr"D:\JACOBS\SERVICE\API\test\Trial Insert Ring TX Expansion 2026 V2\2025112\Identify Route\New Cable {ring}.parquet")

    # -----------------------------
    # RETURN GDF
    # -----------------------------
    if len(segments) == 0:
        return gpd.GeoDataFrame(columns=target_fiber.columns, geometry="geometry", crs=3857)

    df = gpd.GeoDataFrame(segments, geometry="geometry", crs=3857)
    df["length"] = df["length"].round(2)

    return df


def insert_algo(inserts:gpd.GeoDataFrame, lines:gpd.GeoDataFrame, points:gpd.GeoDataFrame, max_member:int=12):
    # FO HUB AND SITELIST
    fo_hub = points[points['site_type'] == 'FO Hub'].reset_index(drop=True).drop_duplicates('site_id')
    site_list = points[points['site_type'] == 'Site List'].reset_index(drop=True).drop_duplicates('site_id')
    ring_name = points['ring_name'].mode()[0]
    print(f"ℹ️ Ring: {ring_name:<15} -> Hub: {len(fo_hub):,} -> Site List: {len(site_list):,} -> Insert Points: {len(inserts):,}")

    if len(fo_hub) == 0:
        print(f"🔴 {ring_name} has no FO Hub in point data, cannot proceed.")
        print(f"Target Points: {points[['site_id', 'site_type']].to_dict(orient='records')}")
        return None, None
    
    # CHECK PREV CONNECTION
    hub_ids = fo_hub['site_id'].tolist()
    start_hub_candidates = lines[lines['near_end'].isin(hub_ids)]['near_end'].values
    start_hub = None
    start_column = None

    hub_ids = fo_hub['site_id'].tolist()
    start_hub_candidates = lines[lines['near_end'].isin(hub_ids)]['near_end'].values
    start_hub = None
    start_column = None

    if len(start_hub_candidates) > 0:
        start_hub = start_hub_candidates[0]
        start_column = 'near_end'
        print(f"ℹ️ Start Col 'near_end' | FO Hub {start_hub}")
    else:
        start_hub_candidates = lines[lines['far_end'].isin(hub_ids)]['far_end'].values
        if len(start_hub_candidates) > 0:
            start_hub = start_hub_candidates[0]
            start_column = 'far_end'
            print(f"ℹ️ Start Col 'far_end' | FO Hub {start_hub}")
        else:
            start_hub = hub_ids[0]
            start_column = 'near_end'
            print(f"⚠️ FO Hub {start_hub} not found in fiber data, defaulting to 'near_end'.")
            
    match len(fo_hub):
        case 1:
            print(f"ℹ️ {ring_name} has single FO Hub.")
        case 2:
            print(f"ℹ️ {ring_name} has multi FO Hub.")
        case _:
            print(f"⚠️ {ring_name} has {len(fo_hub):,} FO Hubs, check manually.")
            return None, None
    
    # PROJECT DETAILS
    region = fo_hub['region'].dropna().mode()[0]
    program = site_list['program'].dropna().mode()[0]

    # IDENTIFY CONNECTION
    new_points, new_connection = build_connection(ring_name, inserts, lines, points, max_member, start_column)
    if new_points is None or new_connection is None:
        print(f"⚠️ Skipping ring {ring_name} due to connection identification failure.")
        return None, None

    routed_segments = routing_insert(ring_name, new_connection, new_points, lines, start_column)
    if routed_segments.empty:
        print(f"🔴 No routed segments generated for ring {ring_name}.")
        return new_points, None
    
    new_points['region'] = region
    # new_points['program'] = program
    new_points['ring_name'] = ring_name
    routed_segments['region'] = region
    # routed_segments['program'] = program
    routed_segments['ring_name'] = ring_name
    return new_points, routed_segments

def parallel_insert(
    mapped_insert: gpd.GeoDataFrame,
    prev_fiber: gpd.GeoDataFrame,
    prev_point: gpd.GeoDataFrame,
    max_member: int = 20,
    ) -> tuple:

    ring_list = mapped_insert["ring_name"].dropna().unique().tolist()
    mapped_insert = mapped_insert.sort_values(by="dist_fiber")
    mapped_insert = mapped_insert[mapped_insert["dist_fiber"] > 0].reset_index(drop=True)
    mapped_insert["note"] = "Insert Site"

    # PREVIOUS DATA
    prev_fiber = prev_fiber.reset_index(drop=True)
    prev_point = prev_point.reset_index(drop=True)
    prev_point["note"] = "Existing Site"

    if "folders" in prev_fiber.columns:
        prev_fiber = prev_fiber.drop(columns=["folders"])
    if "folders" in prev_point.columns:
        prev_point = prev_point.drop(columns=["folders"])

    if "name" in prev_point.columns:
        prev_point = prev_point.rename(columns={"name": "site_id"})
    if "near_end" not in prev_fiber.columns or "far_end" not in prev_fiber.columns:
        prev_fiber["near_end"] = prev_fiber["name"].str.split("-").str[0].str.strip()
        prev_fiber["far_end"] = prev_fiber["name"].str.split("-").str[-1].str.strip()

    all_new_points = []
    all_new_segments = []
    with ProcessPoolExecutor(max_workers=MAX_WORKER) as executor:
        futures = {}
        for ring in ring_list:
            ring_insert = mapped_insert[mapped_insert["ring_name"] == ring].reset_index(drop=True)
            ring_fiber = prev_fiber[prev_fiber["ring_name"] == ring].reset_index(drop=True)
            ring_point = prev_point[prev_point["ring_name"] == ring].reset_index(drop=True)

            if ring_insert.empty:
                logger.warning(f"⚠️ No insert data for ring {ring}, skipping.")
                continue
            if ring_fiber.empty:
                logger.warning(f"⚠️ No fiber data for ring {ring}, skipping.")
                continue
            if ring_point.empty:
                logger.warning(f"⚠️ No point data for ring {ring}, skipping.")
                continue

            future = executor.submit(
                insert_algo,
                ring_insert,
                ring_fiber,
                ring_point,
                max_member,
            )
            futures[future] = ring

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing Rings"
        ):
            ring = futures[future]
            try:
                new_points, new_segments = future.result()
                if new_points is not None and not new_points.empty:
                    all_new_points.append(new_points)
                    logger.info(f"✅ Completed processing for ring {ring} with {len(new_points):,} new points.")
                if new_segments is not None and not new_segments.empty:
                    all_new_segments.append(new_segments)
                    logger.info(f"✅ Completed processing for ring {ring} with {len(new_segments):,} new segments.")
            except Exception as e:
                logger.error(f"❌ Error processing ring {ring}: {e}")
                all_new_points.append(ring_point)
                all_new_segments.append(ring_fiber)


    if all_new_points:
        all_new_points = pd.concat(all_new_points, ignore_index=True)
        print(f"ℹ️ Total new points collected: {len(all_new_points):,}")
    else:
        all_new_points = gpd.GeoDataFrame(columns=mapped_insert.columns, geometry="geometry", crs="EPSG:3857")
        print(f"ℹ️ No new points collected.")

    if all_new_segments:
        all_new_segments = pd.concat(all_new_segments, ignore_index=True)
        print(f"ℹ️ Total new segments collected: {len(all_new_segments):,}")
    else:
        all_new_segments = gpd.GeoDataFrame(columns=prev_fiber.columns, geometry="geometry", crs="EPSG:3857")
        print(f"ℹ️ No new segments collected.")
    return all_new_points, all_new_segments

def save_kml(
    points: gpd.GeoDataFrame,
    paths: gpd.GeoDataFrame,
    topology: gpd.GeoDataFrame,
    export_dir: str,
    method='Supervised'
):
    import simplekml

    date_today = datetime.now().strftime("%Y%m%d")
    kmz_path = os.path.join(export_dir, f"Intersite Design_{method}_{date_today}.kmz")
    logger.info(f"🧩 Exporting KML/KMZ to {kmz_path}")

    points = points.to_crs(epsg=4326).reset_index(drop=True)
    paths = paths.to_crs(epsg=4326).reset_index(drop=True)
    topology = topology.to_crs(epsg=4326).reset_index(drop=True)

    main_kml = simplekml.Kml()
    region_list = points['region'].dropna().unique().tolist()
    for region in region_list:
        logger.info(f"ℹ️ Exporting KML for region {region}")
        region_points = points[points['region'] == region].copy()
        region_paths = paths[paths['region'] == region].copy()
        topology_region = topology[topology['region'] == region].copy()

        if 'long' not in region_points.columns or 'lat' not in region_points.columns:
            region_points['long'] = region_points.geometry.to_crs(epsg=4326).x
            region_points['lat'] = region_points.geometry.to_crs(epsg=4326).y
        if 'vendor' not in region_points.columns:
            region_points['vendor'] = 'TBG'
        if 'program' not in region_points.columns:
            region_points['program'] = 'N/A'

        used_columns = {
            "ring_name": "Ring ID",
            "site_id": "Site ID",
            "site_name": "Site Name" if "site_name" in region_points.columns else "N/A",
            "long": "Long",
            "lat": "Lat",
            "region": "Region",
            "vendor": "Vendor" if "vendor" in region_points.columns else "TBG",
            "program": "Program" if "program" in region_points.columns else "N/A",
            "note": "Note",
            "geometry": "geometry",
        }
        available_col = [col for col in used_columns.keys() if col in region_points.columns]
        ring_list = region_points['ring_name'].dropna().unique().tolist()
        for ring in tqdm(ring_list, desc=f"Processing rings in {region}"):
            ring_points = region_points[region_points['ring_name'] == ring].copy()
            ring_paths = region_paths[region_paths['ring_name'] == ring].copy()
            topology_ring = topology_region[topology_region['ring_name'] == ring].copy()
            topology_ring = topology_ring.dissolve(by='ring_name').reset_index()
            topology_ring['geometry'] = topology_ring.geometry.apply(
                lambda geom: linemerge(geom) if geom.geom_type == "MultiLineString" else geom
            )
            topology_ring = topology_ring[['name', 'ring_name', 'region', 'geometry']]

            # FO HUB & SITELIST
            fo_hub = ring_points[ring_points['site_type'] == 'FO Hub'].copy().reset_index(drop=True)
            site_list = ring_points[ring_points['site_type'] != 'FO Hub'].copy().reset_index(drop=True)
            fo_hub = fo_hub[available_col]
            site_list = site_list[available_col]
            fo_hub = fo_hub.rename(columns=used_columns)
            site_list = site_list.rename(columns=used_columns)

            # PRIORITY SITES
            existing_sites = site_list[site_list['Note'].str.lower().str.contains('existing')]
            insert_sites = site_list[site_list['Note'].str.lower().str.contains('insert')]
            existing_sites = existing_sites.drop(columns='Note')
            insert_sites = insert_sites.drop(columns='Note')

            main_kml = export_kml(
                topology_ring,
                main_kml,
                folder_name=f"{region}_{ring}_Topology",
                subfolder=f"{region}/{ring}",
                name_col='name',
                color="#FF00FF",
                size=3,
                popup=False,
            )
            main_kml = export_kml(
                ring_paths,
                main_kml,
                folder_name=f"{region}_{ring}_Route",
                subfolder=f"{region}/{ring}/Route",
                name_col='name',
                color="#000FFF",
                size=3,
                popup=False,
            )
            main_kml = export_kml(
                existing_sites,
                main_kml,
                folder_name=f"{region}_{ring}_Site List",
                subfolder=f"{region}/{ring}/Site List",
                name_col='Site ID',
                color="#FFFF00",
                size=0.8,
            )
            main_kml = export_kml(
                insert_sites,
                main_kml,
                folder_name=f"{region}_{ring}_Site List",
                subfolder=f"{region}/{ring}/Site List",
                name_col='Site ID',
                color="#00FF00",
                icon='http://maps.google.com/mapfiles/kml/shapes/triangle.png',
                size=0.8,
            )
            main_kml = export_kml(
                fo_hub,
                main_kml,
                folder_name=f"{region}_{ring}_FO_Hub",
                subfolder=f"{region}/{ring}/FO Hub",
                name_col='Site ID',
                icon='http://maps.google.com/mapfiles/kml/paddle/A.png',
                size=0.8,
            )
    sanitize_kml(main_kml)
    main_kml.savekmz(kmz_path)
    logger.info(f"🏆 KML/KMZ export completed at {kmz_path}")


def save_intersite(
    points: gpd.GeoDataFrame,
    paths: gpd.GeoDataFrame,
    export_dir: str,
    method: str = "Insert"
):
    logger.info("🧩 Exporting insert outputs (parquet, KML, Excel).")
    topology = create_topology(points)

    # EXPORT PARQUET
    if not points.empty:
        points.to_crs(epsg=4326).to_parquet(os.path.join(export_dir, f"Points.parquet"), index=False)
        logger.info(f"🏆 Points parquet exported with {len(points):,} records.")
    if not paths.empty:
        paths.to_crs(epsg=4326).to_parquet(os.path.join(export_dir, f"Route.parquet"), index=False,)
        logger.info(f"🏆 Route parquet exported with {len(paths):,} records.")
    if not topology.empty:
        topology = topology.sort_values(by=['ring_name']).reset_index(drop=True)
        topology.to_crs(epsg=4326).to_parquet(os.path.join(export_dir, f"Topology.parquet"), index=False)
        logger.info(f"🏆 Topology parquet exported with {len(topology):,} records.")

    # EXPORT KML
    if not points.empty and not paths.empty and not topology.empty:
        save_kml(points, paths, topology, export_dir, method)

    # EXPORT EXCEL
    excel_path = os.path.join(export_dir, f"Summary Report_Intersite_{method}.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        if not points.empty:
            sheet_name = "Site Information"
            points_report = points.drop(columns="geometry")
            excel_styler(points_report).to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info(
                f"ℹ️ Excel sheet '{sheet_name}' written with {len(points_report):,} records."
            )
        if not paths.empty:
            sheet_name = "Route Information"
            paths_report = paths.drop(columns="geometry")
            paths_report = paths_report.pivot_table(
                index=['name', 'near_end', 'far_end', 'existing_length', 'new_length', 'region', 'ring_name'],
                columns='fo_note',
                values='length',
                aggfunc='sum',
                fill_value=0
            ).reset_index()
            paths_report = paths_report.rename(columns={'merged': 'Length'})
            # paths_report = paths_report.merge(
            #     paths[['ring_name', 'region', 'program']].drop_duplicates(),
            #     on='ring_name',
            #     how='left'
            # )
            paths_report = paths_report.sort_values(by=['ring_name', 'near_end']).reset_index(drop=True)
            paths_report.columns = paths_report.columns.str.replace(' ', '_').str.lower()
            excel_styler(paths_report).to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info(f"ℹ️ Excel sheet '{sheet_name}' written with {len(paths_report):,} records.")

def mark_insert(updated_points: gpd.GeoDataFrame, updated_paths:gpd.GeoDataFrame):
    grouped = updated_points.groupby('ring_name')
    dropped = []
    for idx, group in grouped:
        all_existing = (group['note'] == 'existing').all()
        no_insert = not (group['note'] == 'insert').any()

        if all_existing or no_insert:
            print(f"⚠️ Skipping {idx}: all sites are existing, no new insert required.")
            dropped.append(idx)
    
    inserted_points = updated_points[updated_points["note"].str.lower().str.contains("insert")].copy()
    updated_points = updated_points[~updated_points['ring_name'].isin(dropped)].reset_index(drop=True)
    updated_paths = updated_paths[~updated_paths['ring_name'].isin(dropped)].reset_index(drop=True)
    print(f"ℹ️ Total Inserted Points: {len(inserted_points):,}")
    return updated_points, updated_paths

# STYLIZE AND REPORT
def stylize_insert(df: pd.DataFrame):
    try:
        # Identify column groups by names
        red_cols = df.columns[
            df.columns.get_loc('Link Name') : df.columns.get_loc('Program') + 1
        ]
        origin_cols = df.columns[
            df.columns.get_loc('Origin Site ID') : df.columns.get_loc('Origin Note') + 1
        ]
        destination_cols = df.columns[
            df.columns.get_loc('Destination Site ID') : df.columns.get_loc('Destination Note') + 1
        ]

        # Masks
        p0_origin = df["Origin Priority"].astype(str) == "P0"
        p0_destination = df["Destination Priority"].astype(str) == "P0"
        new_cable = df['New Cable (m)'].fillna(0) > 0

        ring_col = "Ring ID"

        # ===============================
        # Base styling
        # ===============================
        styler = (
            df.style
            .set_properties(**{
                'text-align': 'center',
                'font-size': '10px',
                'border': '1px solid black'
            })
            # Highlight “Insert Site”
            .map(lambda v: "background-color: yellow" if str(v).lower() == "insert site" else "")
            # Highlight P0 origin
            .apply(
                lambda _: [
                    "background-color: #add8e6" if p0_origin[i] else ""
                    for i in range(len(df))
                ],
                axis=0,
                subset=origin_cols
            )
            # Highlight P0 destination
            .apply(
                lambda _: [
                    "background-color: #add8e6" if p0_destination[i] else ""
                    for i in range(len(df))
                ],
                axis=0,
                subset=destination_cols
            )
            .apply(
                lambda _: ["background-color: #fff2cc" if new_cable[i] else "" 
                           for i in range(len(df))],
                axis=0,
                subset=["New Cable (m)"],
            )
            # Highlight Ring ID column
            .apply(
                lambda _: ["background-color: #c6e0b4"] * len(df),
                axis=0,
                subset=[ring_col]
            )
            .format(precision=4, thousands=",")
            .set_table_attributes('style="width: 100%; border-collapse: collapse;"')
        )

        # ===============================
        # HEADER STYLING
        # ===============================
        styler = styler.map_index(
            lambda col: "background-color: red; color: white; font-weight: bold; font-size: 11px; text-align: center;" if col in origin_cols or col in destination_cols or col in red_cols else "background-color: #0070c0; color: white; font-weight: bold; font-size: 11px; text-align: center",
            axis=1)
        return styler

    except Exception as e:
        print(f"❌ Error in stylizing ring data: {e}")
        return df.style

    
def ioh_report(
    updated_points: gpd.GeoDataFrame,
    updated_paths: gpd.GeoDataFrame,
    export_loc: str,
    **kwargs
):
    vendor = kwargs.get("vendor", "TBG")
    program = kwargs.get("program", "Insert Site")
    admin = gpd.read_parquet(rf"{MAINDATA_DIR}/01. Admin/H3_Admin_2024_Kabkot.parquet")
    admin = admin.to_crs(epsg=3857)

    # ----------------------------
    # Column rename mapping
    # ----------------------------
    insert_column = {
        "no": "No",
        "ring_name": "Ring ID",
        "vendor": "Vendor",

        # ORIGIN
        "near_end": "Origin Site ID",
        "ne_name": "Origin Name",
        "ne_long": "Origin Long",
        "ne_lat": "Origin Lat",
        "ne_priority": "Origin Priority",
        "ne_note": "Origin Note",

        # DESTINATION
        "far_end": "Destination Site ID",
        "fe_name": "Destination Name",
        "fe_long": "Destination Long",
        "fe_lat": "Destination Lat",
        "fe_priority": "Destination Priority",
        "fe_note": "Destination Note",

        # SEGMENT
        "segment": "Link Name",
        "segment_ring": "Ring ID ",
        "ring_type": "RING/STAR",
        "ring_status": "Ring Status",
        "city": "City",
        "region": "Region",
        "existing_length": "Existing Cable (m)",
        "new_length": "New Cable (m)",
        "length": "Total Distance (m)",
        "segment_remark": "Remark",
        "program": "Program",

        # PEERS
        "peer_ring_name": "Peer Ring ID",
        "peer_1_id": "Peer 1 (SITE ID)",
        "peer_1_name": "Peer 1 Name",
        "peer_2_id": "Peer 2 (SITE ID)",
        "peer_2_name": "Peer 2 Name",
    }


    ringlist = updated_points["ring_name"].unique().tolist()
    sheet_paths = []

    # =====================================================
    # PROCESS RING BY RING
    # =====================================================
    for ring in ringlist:
        ring_paths = updated_paths[updated_paths["ring_name"] == ring].copy()
        ring_points = updated_points[updated_points["ring_name"] == ring].copy()
        city = gpd.sjoin(admin, ring_points[['geometry']]).drop(columns="index_right")
        if not city.empty:
            city = city['Kabkot'].mode()[0]
        else:
            city = None

        # Split points by category
        fo_hub = ring_points[
            (ring_points["site_type"].str.lower().str.contains("hub"))
            & (ring_points["note"].str.lower().str.contains("existing"))
        ]

        site_list_existing = ring_points[
            (ring_points["site_type"].str.lower().str.contains("site"))
            & (ring_points["note"].str.lower().str.contains("existing"))
        ]
        site_list_insert = ring_points[
            (ring_points["site_type"].str.lower().str.contains("site"))
            & (ring_points["note"].str.lower().str.contains("insert"))
        ]

        # PROGRAM
        prev_program = site_list_existing["program"].mode()[0] if not site_list_existing.empty else program
        insert_program = site_list_insert["program"].mode()[0] if not site_list_insert.empty else program

        # PEER HUB ASSIGNMENT
        if fo_hub.empty:
            raise ValueError(f"No FO Hub found for ring {ring}")

        if len(fo_hub) == 1:
            peer_1_id = fo_hub.iloc[0]["site_id"]
            peer_1_name = fo_hub.iloc[0]["site_name"]
            peer_2_id = None
            peer_2_name = None
        else:
            peer_1_id = fo_hub.iloc[0]["site_id"]
            peer_1_name = fo_hub.iloc[0]["site_name"]
            peer_2_id = fo_hub.iloc[-1]["site_id"]
            peer_2_name = fo_hub.iloc[-1]["site_name"]

        # =====================================================
        # ROUTE METADATA
        # =====================================================

        ring_paths["vendor"] = vendor
        ring_paths["program"] = program
        ring_paths["segment"] = ring_paths["near_end"] + "-" + ring_paths["far_end"]
        ring_paths["segment_ring"] = ring_paths["ring_name"]
        ring_paths["ring_type"] = "RING"
        ring_paths["ring_status"] = "New Ring"
        ring_paths["city"] = city

        # =====================================================
        # MERGE NEAR-END
        # =====================================================
        ne = ring_paths.merge(
            ring_points[["site_id", "site_name", "site_type", "long", "lat", "note"]],
            left_on="near_end",
            right_on="site_id",
            suffixes=("", "_ne"),
        )

        ne_conditions = [
            (ne["note"].str.lower().str.contains("existing")) & (ne["site_type"].str.lower().str.contains("hub")),
            (ne["note"].str.lower().str.contains("existing")) & (ne["site_type"].str.lower().str.contains("site")),
            (ne["note"].str.lower().str.contains("insert")) & (ne["site_type"].str.lower().str.contains("site")),
        ]
        ne_priority_choices = ["P0", "Access", "Insert Site"]
        ne_note_choices = ["FO Hub", "Existing Site", "New Site"]

        ne["ne_priority"] = np.select(ne_conditions, ne_priority_choices, default="Access")
        ne["ne_note"] = np.select(ne_conditions, ne_note_choices, default="Existing Site")

        ne = ne.rename(
            columns={
                "site_name": "ne_name",
                "site_type": "ne_type",
                "long": "ne_long",
                "lat": "ne_lat",
            }
        ).drop(columns=["note", "site_id"])

        # =====================================================
        # MERGE FAR-END
        # =====================================================
        fe = ne.merge(
            ring_points[
                ["site_id", "site_name", "site_type", "long", "lat", "note"]
            ],
            left_on="far_end",
            right_on="site_id",
            suffixes=("", "_fe"),
        )

        fe_conditions = [
            (fe["note"].str.lower().str.contains("existing")) & (fe["site_type"].str.lower().str.contains("hub")),
            (fe["note"].str.lower().str.contains("existing")) & (fe["site_type"].str.lower().str.contains("site")),
            (fe["note"].str.lower().str.contains("insert")) & (fe["site_type"].str.lower().str.contains("site")),
        ]
        fe_priority_choices = ["P0", "Access", "Insert Site"]
        fe_note_choices = ["FO Hub", "Existing Site", "New Site"]

        fe["fe_priority"] = np.select(fe_conditions, fe_priority_choices, default="Access")
        fe["fe_note"] = np.select(fe_conditions, fe_note_choices, default="Existing Site")

        fe = fe.rename(
            columns={
                "site_name": "fe_name",
                "site_type": "fe_type",
                "long": "fe_long",
                "lat": "fe_lat",
            }
        ).drop(columns=["note", "site_id"])

        # =====================================================
        # ASSIGN PROGRAM + REMARK
        # =====================================================
        fe["segment_remark"] = np.where(
            (fe["ne_note"].str.contains("New")) | (fe["fe_note"].str.contains("New")),
            "Insert Segment",
            None,
        )
        fe["program"] = np.where((fe["ne_note"].str.contains("New")), insert_program, prev_program)

        fe["peer_ring_name"] = fe["ring_name"]
        fe["peer_1_id"] = peer_1_id
        fe["peer_1_name"] = peer_1_name
        fe["peer_2_id"] = peer_2_id
        fe["peer_2_name"] = peer_2_name

        sheet_paths.append(fe)

    # =====================================================
    # CONCAT ALL RINGS
    # =====================================================
    sheet_paths = pd.concat(sheet_paths).reset_index(drop=True)
    sheet_paths['no'] = sheet_paths.index + 1
    insert_col = [col for col in insert_column.keys() if col in sheet_paths.columns]

    # REMOVE GEOMETRY
    if "geometry" in sheet_paths.columns:
        sheet_paths = sheet_paths.drop(columns="geometry")

    sheet_paths = sheet_paths[insert_col]
    sheet_paths = sheet_paths.rename(columns=insert_column)


    # =====================================================
    # SAVE EXCEL
    # =====================================================
    excel_path = os.path.join(export_loc, "IOH_Insert Site Report.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        styled_insert = stylize_insert(sheet_paths)
        styled_insert.to_excel(writer, sheet_name="Insert Ring", index=False)
    print(f"📊 Excel sheet 'Insert Ring' written with {len(sheet_paths):,} records.")


def main_insertring(
    insert_data: gpd.GeoDataFrame | str,
    kmz_data: str,
    export_dir: str,
    max_member: int = 12,
    max_distance: int = 3000,
    sep_segment: str = "-",
    method: str = "Insert",
    **kwargs
): 
    logger.info(f"🧩 Running Insert ring design")
    task_celery = kwargs.get("task_celery", False)
    program = kwargs.get("program", "Insert Intersite")
    vendor = kwargs.get("vendor", "TBG")

    # VALIDATE INPUT
    if isinstance(insert_data, str):
        insert_data = read_gdf(insert_data)

    insert_sites, points_existing, lines_existing = validate_insert(insert_sites=insert_data, kmz_data=kmz_data, sep=sep_segment)
    insert_reached, insert_not_reached = identify_insert(insert_sites, lines_existing, max_distance=max_distance)

    logger.info(f"ℹ️ Potential sites to insert: {len(insert_reached):,}")
    logger.info(f"ℹ️ Potential sites to new design: {len(insert_not_reached):,}")
    if not insert_reached.empty:
        insert_reached.to_parquet(os.path.join(export_dir, f"Reached_Points.parquet"))
    if not insert_not_reached.empty:
        insert_not_reached.to_parquet(os.path.join(export_dir, f"Not Reached_Points.parquet"))

    insert_reached = insert_reached.to_crs(epsg=3857)
    points_existing = points_existing.to_crs(epsg=3857)
    lines_existing = lines_existing.to_crs(epsg=3857)

    if 'program' not in insert_reached.columns:
        insert_reached['program'] = program
    if 'vendor' not in insert_reached.columns:
        insert_reached['vendor'] = vendor
    
    # PROCESS INSERT RING
    points_path = os.path.join(export_dir, f"Inserted_Points.parquet")
    paths_path = os.path.join(export_dir, f"Inserted_Lines.parquet")
    # if os.path.exists(points_path) and os.path.exists(paths_path):
    #     updated_points = gpd.read_parquet(points_path)
    #     updated_paths = gpd.read_parquet(paths_path)
    #     print(f"✅ Loaded existing processed data from {export_dir}.")
    # else:
    updated_points, updated_paths = parallel_insert(insert_reached, lines_existing, points_existing, max_member=max_member)

    if not updated_points.empty:
        updated_points.to_crs(epsg=4326).to_parquet(os.path.join(export_dir, f"Inserted_Points.parquet"), index=False)
        print(f"✅ Exported Updated Point Ring with {len(updated_points):,} records.")
    if not updated_paths.empty:
        updated_paths.to_crs(epsg=4326).to_parquet(os.path.join(export_dir, f"Inserted_Lines.parquet"), index=False)
        print(f"✅ Exported Updated Route Fiber with {len(updated_paths):,} records.")

    logger.info("🧩 Clean Not Contain New")
    updated_points, updated_paths = mark_insert(updated_points, updated_paths)

    logger.info("🧩 Save Design Information")
    save_intersite(updated_points, updated_paths, export_dir, method)

    logger.info("🧩 Save Excel IOH Format")
    ioh_report(updated_points, updated_paths, export_dir, program=program, vendor=vendor)
    print(f"✅ Export completed.")

if __name__ == "__main__":
    insert_sites = pd.read_excel(r"D:\JACOBS\PROJECT\TASK\NOVEMBER\Week 4\Insert Algorithm\Insert Site.xlsx")
    kmz_data = r"D:\JACOBS\PROJECT\TASK\NOVEMBER\Week 4\Insert Algorithm\20251119-Week47-TBG-v1.kmz"
    export_dir = r"D:\JACOBS\SERVICE\API\test\Trial Insert Ring TX Expansion 2026 V2"

    date_today = datetime.now().strftime("%Y%m%d")
    export_loc = f"{export_dir}/{date_today}"
    os.makedirs(export_loc, exist_ok=True)

    result = main_insertring(
        insert_data=insert_sites,
        kmz_data=kmz_data,
        export_dir=export_loc,
        max_member=12,
        max_distance=3000
    )

    zip_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_Insert_Task.zip"
    zip_filepath = os.path.join(export_loc, zip_filename)
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(export_loc):
            for file in files:
                if file != zip_filename and not file.endswith(".zip"):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_loc)
                    zipf.write(file_path, arcname)
    logger.info(f"🏆 Result files zipped at {zip_filepath}.")
