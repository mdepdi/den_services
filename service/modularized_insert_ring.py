import os
import geopandas as gpd
import pandas as pd
import networkx as nx
from datetime import datetime
from time import time
from shapely.ops import linemerge
from tqdm import tqdm
import sys

sys.path.append(r"D:\JACOBS\SERVICE\API")

from modules.data import fiber_utilization
from modules.utils import create_topology, route_path, dropwire_connection
from modules.h3_route import identify_hexagon, retrieve_roads, build_graph
from modules.table import excel_styler, find_best_match, detect_week
from modules.validation import input_insertring, identify_fiberzone, prepare_prevdata, identify_insertdata
from core.config import settings
from concurrent.futures import ProcessPoolExecutor, as_completed


DATA_DIR = settings.DATA_DIR
EXPORT_DIR = settings.EXPORT_DIR
LOG_DIR = settings.LOG_DIR

# =============================
# UTILITIES
# ============================
def remove_z(geom):
    from shapely.geometry import Point, LineString, Polygon
    if geom.has_z:
        if geom.geom_type == 'Point':
            return Point(geom.x, geom.y)
        elif geom.geom_type == 'LineString':
            return LineString([(x, y) for x, y, z in geom.coords])
        elif geom.geom_type == 'Polygon':
            exterior = [(x, y) for x, y, z in geom.exterior.coords]
            interiors = [[(x, y) for x, y, z in interior.coords] for interior in geom.interiors]
            return Polygon(exterior, interiors)
        elif geom.geom_type.startswith('Multi') or geom.geom_type == 'GeometryCollection':
            geoms = [remove_z(part) for part in geom.geoms]
            return type(geom)(geoms)
    return geom

def folders_identify(gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf['version'] = gdf['folders'].str.split(';').str[0].str.strip()
    gdf['region'] = gdf['folders'].str.split(';').str[1].str.strip()
    gdf['project'] = gdf['folders'].str.split(';').str[2].str.strip()
    gdf['ring_name'] = gdf['folders'].str.split(';').str[3].str.strip()
    gdf['site_type'] = gdf['folders'].str.split(';').str[-1].str.strip()
    return gdf

def data_preparation(
    mapped_insert: gpd.GeoDataFrame,
    prev_fiber: gpd.GeoDataFrame,
    prev_point: gpd.GeoDataFrame,
    export_folder: str = None,
    ) -> tuple:

    date_today = datetime.now().strftime("%Y%m%d")

    # PREPARE PREVIOUS DATA
    prev_fiber = prev_fiber[['name', 'folders', 'geometry']].copy()
    prev_point = prev_point[['name', 'folders', 'geometry']].copy()
    prev_fiber = folders_identify(prev_fiber)
    prev_point = folders_identify(prev_point)
    prev_fiber = prev_fiber.rename(columns={'site_type': 'route_type'})
    prev_fiber['geometry'] = prev_fiber.geometry.apply(lambda x: linemerge(x) if x.geom_type == 'MultiLineString' else x)
    prev_point['geometry'] = prev_point.geometry.apply(lambda x: remove_z(x) if x.has_z else x)

    # MAPPING NEAR END AND FAR END
    sitetype_candidate = ['Site List', 'FO Hub']
    prev_point['site_type'] = prev_point['site_type'].str.lower()
    prev_point['site_type'] = prev_point['site_type'].str.contains('hub').map({True: 'FO Hub', False: 'Site List'})
    prev_point['site_type'] = prev_point['site_type'].map(lambda name: find_best_match(name, sitetype_candidate, 0.85)[0] if pd.notna(name) else name)
    print(f"Summary Data Preparation:")
    print(prev_point.value_counts(subset=['site_type'], dropna=False))

    prev_fiber['near_end'] = prev_fiber['name'].str.split('-').str[0].str.strip()
    prev_fiber['far_end'] = prev_fiber['name'].str.split('-').str[-1].str.strip()
    mapped_insert['near_end'] = mapped_insert['nearest_segment'].str.split('-').str[0].str.strip()
    mapped_insert['far_end'] = mapped_insert['nearest_segment'].str.split('-').str[-1].str.strip()

    # EXPORT
    if export_folder is not None:
        mapped_insert.to_parquet(os.path.join(export_folder, f"insert_ring_mapped_{date_today}.parquet"), index=False)
        prev_fiber.to_parquet(os.path.join(export_folder, f"insert_ring_prev_fiber_{date_today}.parquet"), index=False)
        prev_point.to_parquet(os.path.join(export_folder, f"insert_ring_prev_point_{date_today}.parquet"), index=False)

    return mapped_insert, prev_fiber, prev_point

# =============================
# UPDATE CONNECTION
# =============================
def identify_connection(ring: str, to_insert:gpd.GeoDataFrame, target_fiber:gpd.GeoDataFrame, target_point:gpd.GeoDataFrame, max_member:int=None, start_column:str='near_end')-> tuple:
    if target_fiber.crs != 'EPSG:3857':
        target_fiber = target_fiber.to_crs(epsg=3857)
    if target_point.crs != 'EPSG:3857':
        target_point = target_point.to_crs(epsg=3857)
    if to_insert.crs != 'EPSG:3857':
        to_insert = to_insert.to_crs(epsg=3857)

    colopriming = gpd.read_parquet(f"{DATA_DIR}/Sitelist TBG_Aug 2025_v.1.0.parquet", columns=['SiteId TBG', 'Sitename TBG', 'geometry'])
    colopriming['SiteId TBG'] = colopriming['SiteId TBG'].astype(str)
    colopriming = colopriming.to_crs(epsg=3857)
    colopriming = colopriming.rename(columns={'SiteId TBG': 'site_id', 'Sitename TBG': 'site_name'})

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
    to_insert = to_insert.sort_values(by='distance_to_fiber').reset_index(drop=True)
    to_insert = to_insert[~to_insert['site_id'].isin(connection)].reset_index(drop=True)
    near_dict = to_insert[start_column].value_counts().to_dict()

    # BUILD ROAD NETWORK
    hex_list = identify_hexagon(to_insert, type='convex')
    roads = retrieve_roads(hex_list, type='roads')
    nodes = retrieve_roads(hex_list, type='nodes')
    roads = roads.to_crs(epsg=3857)
    nodes = nodes.to_crs(epsg=3857)
    G = build_graph(roads, graph_type='fiber')

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
            to_insert.at[idx, f'distto_{start_column}'] = start_length
        except (nx.NetworkXNoPath, IndexError):
            to_insert.at[idx, f'distto_{start_column}'] = float('inf')
            print(f"⚠️ No path found from site {row['site_id']} to {row[start_column]} in ring {ring}, skipping.")
    
    to_insert = to_insert[to_insert[f'distto_{start_column}'] != float('inf')].reset_index(drop=True)
    to_insert = to_insert.sort_values(by=f'distto_{start_column}').reset_index(drop=True)

    new_connection = connection.copy()
    for near_id, count in near_dict.items():
        member_count = len(new_connection)
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
            print(f"⚠️ Site ID {site_id} not found in any dataset, try colopriming.")

            point_data = colopriming[colopriming['site_id'] == site_id].iloc[0] if not colopriming[colopriming['site_id'] == site_id].empty else None
            if point_data is None:
                print(f"⚠️ Site ID {site_id} still not found after colopriming, skipping.")
                continue
        points_new.append(point_data)
    
    if not points_new:
        print(f"⚠️ No valid points found for ring {ring} after insertion.")
        return None, None
    points_new = gpd.GeoDataFrame(points_new, columns=points_new[0].index.tolist(), crs='EPSG:3857').reset_index(drop=True)
    points_new['note'] = points_new['site_id'].apply(lambda x: 'new' if x in to_insert['site_id'].values else 'existing')
    return points_new, new_connection

# =============================
# UPDATE ROUTE              
# =============================
def routing_insert(
    ring: str, 
    new_connection : list, 
    new_points: gpd.GeoDataFrame, 
    target_fiber: gpd.GeoDataFrame,
    start_column: str='near_end') -> gpd.GeoDataFrame:
    if target_fiber.crs != 'EPSG:3857':
        target_fiber = target_fiber.to_crs(epsg=3857)
    if new_points.crs != 'EPSG:3857':
        new_points = new_points.to_crs(epsg=3857)

    # START COLUMNS
    match start_column:
        case 'near_end':
            opposite_column = 'far_end'
        case 'far_end':
            opposite_column = 'near_end'
        case _:
            raise ValueError("start_column must be either 'near_end' or 'far_end'.")
        
    # BUILD NETWORK
    hex_list = identify_hexagon(new_points, type='convex')
    roads = retrieve_roads(hex_list, type='roads')
    nodes = retrieve_roads(hex_list, type='nodes')
    roads = roads.to_crs(epsg=3857)
    nodes = nodes.to_crs(epsg=3857)
    G = build_graph(roads, graph_type='fiber')

    # NEAREST NODE
    if 'nearest_node' not in new_points.columns:
        new_points = gpd.sjoin_nearest(new_points, nodes[['node_id', 'geometry']], how='left', distance_col='dist_to_node')
        new_points = new_points.rename(columns={'node_id': 'nearest_node'})
    
    # IDENTIFY PREVIOUS FIBER NODES
    prevfiber_buff = target_fiber.copy()
    prevfiber_buff['geometry'] = prevfiber_buff.geometry.buffer(20)
    node_prevfiber = gpd.sjoin(nodes, prevfiber_buff[['geometry', 'near_end', 'far_end']], how='inner', predicate='intersects').drop(columns=['index_right'])
    ref_prevfiber = set(node_prevfiber['node_id'])
    
    # # PENALTY EXISTING FIBER
    # for u, v, data in G.edges(data=True):
    #     if u in ref_prevfiber and v in ref_prevfiber:
    #         data['weight'] = 1e10  # High penalty for existing fiber edges
    #     else:
    #         data['weight'] = data.get('length', 1)

    # MAXIMIZE EXISTING FIBER
    for u, v, data in G.edges(data=True):
        if u in ref_prevfiber and v in ref_prevfiber:
            data['weight'] = data.get('weight', 1) / 3
        else:
            data['weight'] = data.get('weight', 1) 

    # ROUTING SEGMENTS
    segments = []
    for i in range(len(new_connection) - 1):
        start_id = new_connection[i]
        end_id = new_connection[i + 1]

        start_point = new_points[new_points['site_id'] == start_id].iloc[0]
        end_point = new_points[new_points['site_id'] == end_id].iloc[0]

        if start_point.empty or end_point.empty:
            print(f"🔴 {ring} missing point data for {start_id} or {end_id}, skipping segment.")
            continue

        is_newroute = (start_point['note'] == 'new') or (end_point['note'] == 'new')
        if is_newroute:
            start_node = start_point['nearest_node']
            end_node = end_point['nearest_node']
            
            try:
                path, path_geom, path_length = route_path(start_node, end_node, G, roads, merged=True)
                path_geom, path_length = dropwire_connection(path_geom, start_point, end_point, nodes, start_node, end_node)

                if not path_geom.is_empty:
                    segment_name = f"{start_id}-{end_id}" if start_column == 'near_end' else f"{end_id}-{start_id}"
                    route_type = 'New Route'

                    segment_record = {
                        'name': segment_name,
                        start_column: start_id,
                        opposite_column: end_id,
                        'length': path_length,
                        'route_type': route_type,
                        'fo_note': 'merged',
                        'geometry': path_geom
                    }

                    segments.append(segment_record)
                    print(f"🟢 Length: {path_length:10,.2f} m  | Routed segment    : {segment_name}")

                    ## UPDATE GRAPH
                    ## PENALTY EXISTING FIBER
                    # for j in range(len(path) - 1):
                    #     if G.has_edge(path[j], path[j + 1]):
                    #         G[path[j]][path[j + 1]]['weight'] = 1e10
                    #     if G.has_edge(path[j + 1], path[j]):
                    #         G[path[j + 1]][path[j]]['weight'] = 1e10

                    ## MAXIMIZE EXISTING FIBER
                    for j in range(len(path) - 1):
                        if G.has_edge(path[j], path[j + 1]):
                            G[path[j]][path[j + 1]]['weight'] = G[path[j]][path[j + 1]]['weight'] / 3
                        if G.has_edge(path[j + 1], path[j]):
                            G[path[j + 1]][path[j]]['weight'] = G[path[j + 1]][path[j]]['weight'] / 3
                else:
                    print(f"⚠️ No geometry found for segment {start_id} to {end_id}, skipping.")
            except nx.NetworkXNoPath:
                print(f"⚠️ No path found between {start_id} and {end_id}, skipping segment.")
        else:
            node_existing = node_prevfiber[(node_prevfiber[start_column] == start_id) & (node_prevfiber[opposite_column] == end_id)].copy()
            node_existing = node_existing.drop_duplicates(subset=['node_id']).reset_index(drop=True)
            path_geom = roads[roads['node_start'].isin(node_existing['node_id']) & roads['node_end'].isin(node_existing['node_id'])].reset_index(drop=True)
            path_length = path_geom['length'].sum()
            if not path_geom.empty:
                merged_line = linemerge(path_geom.geometry.union_all())
                segment_name = f"{start_id}-{end_id}" if start_column == 'near_end' else f"{end_id}-{start_id}"
                route_type = 'Existing Route'

                segment_record = {
                    'name': segment_name,
                    start_column: start_id,
                    opposite_column: end_id,
                    'length': path_length,
                    'route_type': route_type,
                    'fo_note': 'merged',
                    'geometry': merged_line
                }
                    
                segments.append(segment_record)
                print(f"🟢 Length: {path_length:10,.2f} m  | Existing segment  : {segment_name}")
            else:
                print(f"🟠 No geometry found for existing segment {start_id} to {end_id}, try reverse")
                node_existing = node_prevfiber[(node_prevfiber[start_column] == end_id) & (node_prevfiber[opposite_column] == start_id)].copy()
                node_existing = node_existing.drop_duplicates(subset=['node_id']).reset_index(drop=True)
                path_geom = roads[roads['node_start'].isin(node_existing['node_id']) & roads['node_end'].isin(node_existing['node_id'])].reset_index(drop=True)
                path_length = path_geom['length'].sum()
                if not path_geom.empty:
                    merged_line = linemerge(path_geom.geometry.union_all())
                    segment_name = f"{end_id}-{start_id}" if start_column == 'near_end' else f"{start_id}-{end_id}"
                    route_type = 'Existing Route'

                    segment_record = {
                        'name': segment_name,
                        start_column: end_id,
                        opposite_column: start_id,
                        'length': path_length,
                        'route_type': route_type,
                        'fo_note': 'merged',
                        'geometry': merged_line
                    }
                        
                    segments.append(segment_record)
                    print(f"🟢 Length: {path_length:10,.2f} m  | Existing segment  : {segment_name}")
                else:
                    print(f"⚠️ No geometry found for existing segment {start_id} to {end_id} in both directions, skipping.")

    if segments:
        segments_gdf = gpd.GeoDataFrame(segments, columns=segments[0].keys(), geometry='geometry', crs='EPSG:3857')
        segments_gdf['length'] = segments_gdf['length'].round(2)
        return segments_gdf
    else:
        return gpd.GeoDataFrame(columns=target_fiber.columns, geometry='geometry', crs=target_fiber.crs)

# =============================
# MAIN FUNCTION
# =============================
def insert_ring_processing(
    ring: str,
    to_insert: gpd.GeoDataFrame,
    target_fiber: gpd.GeoDataFrame,
    target_point: gpd.GeoDataFrame,
    max_member: int = None,
    route_type: str = 'merged',
) -> tuple:
    print(f"\n==============================")
    print(f"🔄 Processing Ring: {ring}")
    print(f"==============================")
    start_time = time()
    
    # FO HUB AND SITELIST
    fo_hub = target_point[target_point['site_type'] == 'FO Hub'].reset_index(drop=True)
    site_list = target_point[target_point['site_type'] == 'Site List'].reset_index(drop=True)
    total_point = len(fo_hub) + len(site_list)
    print(f"ℹ️ FO Hub(s)     : {len(fo_hub):,}")
    print(f"ℹ️ Site List(s)  : {len(site_list):,}")
    print(f"ℹ️ Total Points  : {total_point:,}")
    print(f"🔗 Insert Points : {len(to_insert):,}")

    # CHECK PREV CONNECTION
    hub_ids = fo_hub['site_id'].tolist()
    start_hub_candidates = target_fiber[target_fiber['near_end'].isin(hub_ids)]['near_end'].values
    start_hub = None
    start_column = None

    if len(fo_hub) == 0:
        print(f"🔴 {ring} has no FO Hub in point data, cannot proceed.")
        print(f"Target Points: {target_point[['site_id', 'site_type']].to_dict(orient='records')}")
        return None, None

    hub_ids = fo_hub['site_id'].tolist()
    start_hub_candidates = target_fiber[target_fiber['near_end'].isin(hub_ids)]['near_end'].values
    start_hub = None
    start_column = None

    if len(start_hub_candidates) > 0:
        start_hub = start_hub_candidates[0]
        start_column = 'near_end'
        print(f"ℹ️ Start Col 'near_end' | FO Hub {start_hub}")
    else:
        # Check far_end
        start_hub_candidates = target_fiber[target_fiber['far_end'].isin(hub_ids)]['far_end'].values
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
            print(f"ℹ️ {ring} has single FO Hub.")
        case 2:
            print(f"ℹ️ {ring} has multi FO Hub.")
        case _:
            print(f"⚠️ {ring} has {len(fo_hub):,} FO Hubs, check manually.")
            return None, None
    
    # PROJECT DETAILS
    date_today = datetime.now().strftime('%Y%m%d')
    week = detect_week(date_today)
    version = f"{date_today}_W{week}_v1.0"
    region = fo_hub[fo_hub['region'].notna()]['region'].mode()[0] if not fo_hub[fo_hub['region'].notna()]['region'].mode().empty else 'Unknown Region'
    project = to_insert['project'].mode()[0] if not to_insert['project'].mode().empty else 'Unknown Program'

    # IDENTIFY CONNECTION
    new_points, new_connection = identify_connection(ring, to_insert, target_fiber, target_point, max_member, start_column)
    if new_points is None or new_connection is None:
        print(f"⚠️ Skipping ring {ring} due to connection identification failure.")
        return None, None
    
    new_points['version'] = version
    new_points['region'] = region
    new_points['project'] = project
    new_points['ring_name'] = ring

    match route_type:
        case 'merged':
            routed_segments = routing_insert(ring, new_connection, new_points, target_fiber, start_column)
        case 'classified':
            routed_segments = routing_insert(ring, new_connection, new_points, target_fiber, start_column)
            routed_segments = fiber_utilization(routed_segments, target_fiber, overlap=True)
        case _:
            raise ValueError(f"Invalid route_type: {route_type}. Choose 'merged' or 'classified'.")

    if routed_segments.empty:
        print(f"🔴 No routed segments generated for ring {ring}.")
        return new_points, None
    
    routed_segments['version'] = version
    routed_segments['region'] = region
    routed_segments['project'] = project
    routed_segments['ring_name'] = ring
    print(f"ℹ️ Routed segments generated: {len(routed_segments):,}")
    
    end_time = time()
    elapsed_time = end_time - start_time
    print(f"⏱️ {ring} processed in {elapsed_time:,.2f} seconds")
    return new_points, routed_segments

def parallel_insert_processing(
    mapped_insert: gpd.GeoDataFrame,
    prev_fiber: gpd.GeoDataFrame,
    prev_point: gpd.GeoDataFrame,
    MAX_WORKERS: int = 4,
    MAX_MEMBER: int = 20,
    ROUTE_TYPE: str = 'merged',
    ) -> tuple:

    print(f"=================================")
    print(f"🚀 START PARALLEL INSERT RING")
    print(f"=================================")

    ring_list = mapped_insert["nearest_ring"].dropna().unique().tolist()
    print(f"🔄 Total Rings to Process: {len(ring_list):,}")

    mapped_insert = mapped_insert.sort_values(by="distance_to_fiber")
    mapped_insert = mapped_insert[mapped_insert["distance_to_fiber"] > 0].reset_index(drop=True)
    mapped_insert['site_type'] = mapped_insert.get('site_type', 'Site List')
    mapped_insert["note"] = "new"

    # PREVIOUS DATA
    prev_fiber = prev_fiber.reset_index(drop=True)
    prev_point = prev_point.reset_index(drop=True)
    prev_point["note"] = "existing"

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
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for ring in ring_list:
            ring_insert = mapped_insert[mapped_insert["nearest_ring"] == ring].reset_index(drop=True)
            ring_fiber = prev_fiber[prev_fiber["ring_name"] == ring].reset_index(drop=True)
            ring_point = prev_point[prev_point["ring_name"] == ring].reset_index(drop=True)

            if ring_insert.empty:
                print(f"⚠️ No insert data for ring {ring}, skipping.")
                continue
            if ring_fiber.empty:
                print(f"⚠️ No fiber data for ring {ring}, skipping.")
                continue
            if ring_point.empty:
                print(f"⚠️ No point data for ring {ring}, skipping.")
                continue

            future = executor.submit(
                insert_ring_processing,
                ring,
                ring_insert,
                ring_fiber,
                ring_point,
                MAX_MEMBER,
                ROUTE_TYPE,
            )
            futures[future] = ring

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing Rings"
        ):
            ring = futures[future]
            try:
                new_points, new_segments = future.result()
                if new_points is not None:
                    all_new_points.append(new_points)
                    print(f"✅ Completed processing for ring {ring} with {len(new_points):,} new points.")
                if new_segments is not None:
                    all_new_segments.append(new_segments)
                    print(f"✅ Completed processing for ring {ring} with {len(new_segments):,} new segments.")
            except Exception as e:
                print(f"❌ Error processing ring {ring}: {e}")

    if all_new_points:
        all_new_points = pd.concat(all_new_points, ignore_index=True)
        print(f"ℹ️ Total new points collected: {len(all_new_points):,}")
    else:
        all_new_points = gpd.GeoDataFrame(
            columns=mapped_insert.columns, geometry="geometry", crs="EPSG:3857"
        )
        print(f"ℹ️ No new points collected.")

    if all_new_segments:
        all_new_segments = pd.concat(all_new_segments, ignore_index=True)
        print(f"ℹ️ Total new segments collected: {len(all_new_segments):,}")
    else:
        all_new_segments = gpd.GeoDataFrame(
            columns=prev_fiber.columns, geometry="geometry", crs="EPSG:3857"
        )
        print(f"ℹ️ No new segments collected.")

    return all_new_points, all_new_segments

def export_kml_insertring(
    updated_points: gpd.GeoDataFrame,
    updated_paths: gpd.GeoDataFrame,
    topology: gpd.GeoDataFrame,
    not_inserted: gpd.GeoDataFrame,
    export_dir: str,
    route_type: str = 'merged',
):
    from modules.kml import export_kml, sanitize_kml
    import simplekml

    date_today = datetime.now().strftime("%Y%m%d")
    kmz_path = os.path.join(export_dir, f"Insert Ring_Fiberization_{date_today}.kmz")

    updated_points = updated_points.to_crs(epsg=4326).reset_index(drop=True)
    updated_paths = updated_paths.to_crs(epsg=4326).reset_index(drop=True)
    topology = topology.to_crs(epsg=4326).reset_index(drop=True)
    
    main_kml = simplekml.Kml()
    region_list = updated_points['region'].dropna().unique().tolist()
    for region in region_list:
        print(f"🔄 Exporting KML for Region: {region}")
        region_points = updated_points[updated_points['region'] == region]
        region_paths = updated_paths[updated_paths['region'] == region]
        topology_region = topology[topology['region'] == region]

        if 'long' not in region_points.columns or 'lat' not in region_points.columns:
            region_points['long'] = region_points.geometry.to_crs(epsg=4326).x
            region_points['lat'] = region_points.geometry.to_crs(epsg=4326).y
        if 'vendor' not in region_points.columns:
            region_points['vendor'] = 'TBG'
        if 'program' not in region_points.columns:
            if 'project' in region_points.columns:
                region_points['program'] = region_points['project']
            else:
                region_points['program'] = "N/A"
                

        used_columns = {
        "ring_name": "Ring ID",
        "site_id": "Site ID",
        "site_name": "Site Name" if "site_name" in region_points.columns else "N/A",
        "long": "Long",
        "lat": "Lat",
        "region": "Region",
        "vendor": "Vendor" if "vendor" in region_points.columns else "N/A",
        "program": "Program" if "program" in region_points.columns else "N/A",
        "note": "note",
        "fo_note": "fo_note",
        "geometry": "geometry",
        }
        available_col = [col for col in used_columns.keys() if col in region_points.columns]

        ring_list = region_points['ring_name'].dropna().unique().tolist()
        for ring in tqdm(ring_list, desc=f"Processing Rings in {region}"):
            ring_points = region_points[region_points['ring_name'] == ring]
            ring_paths = region_paths[region_paths['ring_name'] == ring]
            topology_ring = topology_region[topology_region['ring_name'] == ring]
            topology_ring = topology_ring.dissolve(by='ring_name').reset_index()
            topology_ring['geometry'] = topology_ring.geometry.apply(lambda geom: linemerge(geom) if geom.geom_type == 'MultiLineString' else geom)
            topology_ring = topology_ring[['ring_name', 'region', 'project', 'geometry']]
            topology_ring['name'] = "Connection"

            # FO HUB & SITELIST
            fo_hub = ring_points[ring_points['site_type'] == 'FO Hub'].reset_index(drop=True)
            site_list = ring_points[ring_points['site_type'] != 'FO Hub'].reset_index(drop=True)
            fo_hub = fo_hub[available_col]
            site_list = site_list[available_col]
            fo_hub = fo_hub.rename(columns=used_columns).drop(columns='note')
            site_list = site_list.rename(columns=used_columns)

            match route_type:
                case 'merged':
                    existing_points = site_list[site_list['note'] == 'existing'].reset_index(drop=True)
                    new_points = site_list[site_list['note'] == 'new'].reset_index(drop=True)
                    
                    existing_points = existing_points.drop(columns='note')
                    new_points = new_points.drop(columns='note')
                    main_kml = export_kml(topology_ring, main_kml, folder_name=f"{region}_{ring}_Topology", subfolder=f"{region}/{ring}", name_col='name', color="#FF00FF", size=3, popup=False)
                    main_kml = export_kml(ring_paths, main_kml, folder_name=f"{region}_{ring}_Route", subfolder=f"{region}/{ring}/Route", name_col='name', color="#000FFF", size=3, popup=False)
                    main_kml = export_kml(existing_points, main_kml, folder_name=f"{region}_{ring}_Existing_Points", subfolder=f"{region}/{ring}/Site List", name_col='Site ID', color="#FFFF00", size=0.8)
                    main_kml = export_kml(new_points, main_kml, folder_name=f"{region}_{ring}_New_Points", subfolder=f"{region}/{ring}/Site List", name_col='Site ID', color="#00ff00", icon='http://maps.google.com/mapfiles/kml/shapes/triangle.png', size=0.8)
                    main_kml = export_kml(fo_hub, main_kml, folder_name=f"{region}_{ring}_FO_Hub", subfolder=f"{region}/{ring}/FO Hub", name_col='Site ID', icon='http://maps.google.com/mapfiles/kml/paddle/A.png', size=0.8)

                case 'classified':
                    # EXISTING POINTS
                    existing_points = site_list[site_list['note'] == 'existing'].reset_index(drop=True)
                    new_points = site_list[site_list['note'] == 'new'].reset_index(drop=True)
                    existing_paths = ring_paths[ring_paths['fo_note'] == 'Existing'].reset_index(drop=True)
                    new_paths = ring_paths[ring_paths['fo_note'] == 'New'].reset_index(drop=True)

                    existing_points = existing_points.drop(columns='note')
                    new_points = new_points.drop(columns='note')
                    existing_paths = existing_paths.drop(columns='fo_note')
                    new_paths = new_paths.drop(columns='fo_note')
                    main_kml = export_kml(topology_ring, main_kml, folder_name=f"{region}_{ring}_Topology", subfolder=f"{region}/{ring}", name_col='name', color="#FF00FF", size=3, popup=False)
                    main_kml = export_kml(existing_paths, main_kml, folder_name=f"{region}_{ring}_Existing_Paths", subfolder=f"{region}/{ring}/Route/Existing Paths", name_col='name', color="#000FFF", size=3, popup=False)
                    main_kml = export_kml(new_paths, main_kml, folder_name=f"{region}_{ring}_New_Paths", subfolder=f"{region}/{ring}/Route/New Paths", name_col='name', color="#FF0000", size=3, popup=False)
                    main_kml = export_kml(existing_points, main_kml, folder_name=f"{region}_{ring}_Existing_Points", subfolder=f"{region}/{ring}/Site List/Existing Points", name_col='Site ID', color="#FFFF00", size=0.8)
                    main_kml = export_kml(new_points, main_kml, folder_name=f"{region}_{ring}_New_Points", subfolder=f"{region}/{ring}/Site List/New Points", name_col='Site ID', color='#00ff7f', icon='http://maps.google.com/mapfiles/kml/shapes/triangle.png', size=0.8)
                    main_kml = export_kml(fo_hub, main_kml, folder_name=f"{region}_{ring}_FO_Hub", subfolder=f"{region}/{ring}/FO Hub", name_col='Site ID', icon='http://maps.google.com/mapfiles/kml/paddle/A.png', size=0.8)
                case _:
                    raise ValueError(f"Invalid route_type: {route_type}. Choose 'merged' or 'classified'.")
    if not not_inserted.empty:
        not_inserted = not_inserted.to_crs(epsg=4326).reset_index(drop=True)
        main_kml = export_kml(not_inserted, main_kml, folder_name=f"Not_Inserted_Sites", subfolder=f"NOT INSERTED", name_col='site_id', color='#ff0000', icon='http://maps.google.com/mapfiles/kml/shapes/forbidden.png')
    
    sanitize_kml(main_kml)
    main_kml.savekmz(kmz_path)
    print(f"✅ Exported KML/KMZ file at {kmz_path}")

def export_insertring(
    updated_points: gpd.GeoDataFrame,
    updated_paths: gpd.GeoDataFrame,
    topology: gpd.GeoDataFrame,
    paths_utilization: gpd.GeoDataFrame,
    mapped_insert: gpd.GeoDataFrame,
    export_dir: str,
    route_type: str = 'merged',
):
    date_today = datetime.now().strftime("%Y%m%d")

    # NOT INSERTED DATA
    inserted = set(updated_points["site_id"]) if not updated_points.empty else set()
    not_inserted = mapped_insert[~mapped_insert["site_id"].isin(inserted)].reset_index(drop=True)
    if not not_inserted.empty:
        not_inserted.to_parquet(os.path.join(export_dir, f"Not Inserted_Mapped Insert Data_Fiberization.parquet"),index=False)
        print(f"⚠️ Exported Not Inserted data with {len(not_inserted):,} records.")

    # EXPORT PARQUET
    if not updated_points.empty:
        updated_points.to_crs(epsg=4326).to_parquet(os.path.join(export_dir, f"Updated Point Ring_Fiberization_{date_today}.parquet"), index=False)
        print(f"✅ Exported Updated Point Ring with {len(updated_points):,} records.")
    if not updated_paths.empty:
        updated_paths.to_crs(epsg=4326).to_parquet(os.path.join(export_dir, f"Updated Route Fiber_Fiberization_{date_today}.parquet"), index=False)
        print(f"✅ Exported Updated Route Fiber with {len(updated_paths):,} records.")
    if not paths_utilization.empty:
        paths_utilization = paths_utilization.sort_values(by=['ring_name']).reset_index(drop=True)
        paths_utilization.to_crs(epsg=4326).to_parquet(os.path.join(export_dir, f"Fiber Utilization_Fiberization_{date_today}.parquet"), index=False)
        print(f"✅ Exported Fiber Utilization with {len(paths_utilization):,} records.")
    if not not_inserted.empty:
        not_inserted = not_inserted.sort_values(by=['nearest_ring', 'distance_to_fiber']).reset_index(drop=True)
        not_inserted.to_crs(epsg=4326).to_parquet(os.path.join(export_dir, f"Not Inserted_Mapped Insert Data_Fiberization.parquet"), index=False)
        print(f"✅ Exported Not Inserted data with {len(not_inserted):,} records.")
    if not topology.empty:
        topology = topology.sort_values(by=['ring_name']).reset_index(drop=True)
        topology.to_crs(epsg=4326).to_parquet(os.path.join(export_dir, f"Topology Route_Fiberization_{date_today}.parquet"), index=False)
        print(f"✅ Exported Topology Route with {len(topology):,} records.")

    # EXPORT KML
    if not updated_points.empty and not updated_paths.empty and not topology.empty:
        export_kml_insertring(updated_points, updated_paths, topology, not_inserted, export_dir, route_type)

    # EXPORT EXCEL
    excel_path = os.path.join(export_dir, f"Summary Report_Fiberization_{date_today}.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        if not updated_points.empty:
            sheet_name = "Site Information"
            updated_points_report = updated_points.drop(columns="geometry")
            excel_styler(updated_points_report).to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"📊 Excel sheet '{sheet_name}' with {len(updated_points_report):,} records written.")
        if not updated_paths.empty:
            sheet_name = "Route Information"
            updated_paths_report = updated_paths.drop(columns="geometry")
            updated_paths_report = updated_paths_report.pivot_table(
                index=['name', 'near_end', 'far_end', 'ring_name'],
                columns='fo_note',
                values='length',
                aggfunc='sum',
                fill_value=0
            ).reset_index()

            match route_type:
                case 'merged':
                    updated_paths_report = updated_paths_report.rename(columns={'merged': 'Length'})
                case 'classified':
                    updated_paths_report['total_length'] = updated_paths_report.get('Existing', 0) + updated_paths_report.get('New', 0)
            
            updated_paths_report = updated_paths_report.merge(
                updated_paths[['ring_name', 'region', 'project']].drop_duplicates(),
                on='ring_name',
                how='left'
            )
            updated_paths_report = updated_paths_report.sort_values(by=['ring_name', 'near_end']).reset_index(drop=True)
            updated_paths_report.columns = updated_paths_report.columns.str.replace(' ', '_').str.lower()
            excel_styler(updated_paths_report).to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"📊 Excel sheet '{sheet_name}' with {len(updated_paths_report):,} records written.")
        if not paths_utilization.empty:
            sheet_name = "Fiber Utilization (No Overlap)"
            paths_utilization = paths_utilization.drop(columns="geometry")
            grouped_utilization = paths_utilization.pivot_table(
                index=['ring_name'],
                columns='fo_note',
                values='length',
                aggfunc='sum',
                fill_value=0
            ).reset_index()

            grouped_utilization = grouped_utilization.merge(
                paths_utilization[['ring_name', 'region', 'project']].drop_duplicates(),
                on='ring_name',
                how='left'
            )
            grouped_utilization['total_Length'] = grouped_utilization.get('Existing', 0) + grouped_utilization.get('New', 0)
            grouped_utilization = grouped_utilization.sort_values(by='ring_name').reset_index(drop=True)
            grouped_utilization.columns = grouped_utilization.columns.str.replace(' ', '_').str.lower()
            excel_styler(grouped_utilization).to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"📊 Excel sheet '{sheet_name}' with {len(grouped_utilization):,} records written.")
        if not not_inserted.empty:
            sheet_name = "Not Inserted"
            not_inserted_report = not_inserted.drop(columns="geometry")
            not_inserted_report = not_inserted_report.sort_values(by="distance_to_fiber").reset_index(drop=True)
            not_inserted_report['nearest_fiber_node'] = not_inserted_report['nearest_fiber_node'].fillna('N/A')
            not_inserted_report['distance_to_fiber'] = not_inserted_report['distance_to_fiber'].fillna(0).round(2)
            not_inserted_report['distance_to_fiber'] = not_inserted_report['distance_to_fiber'].replace('inf', 'N/A')
            excel_styler(not_inserted_report).to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"📊 Excel sheet '{sheet_name}' with {len(not_inserted_report):,} records written.")


def main_insertring(
    mapped_insert: gpd.GeoDataFrame,
    prev_fiber: gpd.GeoDataFrame,
    prev_point: gpd.GeoDataFrame,
    export_dir: str,
    MAX_WORKERS: int = 4,
    MAX_MEMBER: int = 20,
    ROUTE_TYPE: str = 'merged',
):
    # CRS

    if prev_fiber.crs != "EPSG:3857":
        prev_fiber = prev_fiber.to_crs(epsg=3857)

    if prev_point.crs != "EPSG:3857":
        prev_point = prev_point.to_crs(epsg=3857)

    if mapped_insert.crs != "EPSG:3857":
        mapped_insert = mapped_insert.to_crs(epsg=3857)

    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    
    # DATA PREPARATION
    mapped_insert, prev_fiber, prev_point = data_preparation(mapped_insert, prev_fiber, prev_point)

    # PROCESS INSERT RING
    date_today = datetime.now().strftime("%Y%m%d")
    points_path = os.path.join(export_dir, f"Updated Point Ring_Fiberization_{date_today}.parquet")
    paths_path = os.path.join(export_dir, f"Updated Route Fiber_Fiberization_{date_today}.parquet")
    if os.path.exists(points_path) and os.path.exists(paths_path):
        updated_points = gpd.read_parquet(points_path)
        updated_paths = gpd.read_parquet(paths_path)
        print(f"✅ Loaded existing processed data from {export_dir}.")
    else:
        updated_points, updated_paths = parallel_insert_processing(mapped_insert, prev_fiber, prev_point, MAX_WORKERS, MAX_MEMBER, ROUTE_TYPE)

    if not updated_points.empty:
        updated_points.to_crs(epsg=4326).to_parquet(os.path.join(export_dir, f"Updated Point Ring_Fiberization_{date_today}.parquet"), index=False)
        print(f"✅ Exported Updated Point Ring with {len(updated_points):,} records.")
    if not updated_paths.empty:
        updated_paths.to_crs(epsg=4326).to_parquet(os.path.join(export_dir, f"Updated Route Fiber_Fiberization_{date_today}.parquet"), index=False)
        print(f"✅ Exported Updated Route Fiber with {len(updated_paths):,} records.")

    # TOPOLOGY CHECK
    topology_paths = create_topology(updated_points)

    # IDENTIFY ROUTE UTILIZATION
    paths_utilization = fiber_utilization(updated_paths, target_fiber=prev_fiber, overlap=False)
    paths_utilization = paths_utilization[['ring_name','fo_note', 'length', 'geometry']]
    paths_utilization = paths_utilization.merge(
        updated_paths[["ring_name", "region", "project"]].drop_duplicates(),
        on="ring_name",
        how="left",
    )
    print(f"ℹ️ Total paths for utilization analysis: {len(paths_utilization):,}")

    # EXPORT
    export_insertring(updated_points, updated_paths, topology_paths, paths_utilization, mapped_insert, export_dir, ROUTE_TYPE)
    print(f"✅ Export completed.")

if __name__ == "__main__":
    # CONFIGURATION
    MAX_MEMBER = 12             # Maximum members in a ring
    ROUTE_TYPE = "merged"       # Options: 'merged' or 'classified'
    MAX_WORKERS = 8             # Number of parallel workers

    # DATE
    date_today = datetime.now().strftime("%Y%m%d")
    week = detect_week(date_today)

    # INPUT & OUTPUT PATH
    INPUT_DIR = f"{DATA_DIR}/data"
    EXCEL_FILE = os.path.join(INPUT_DIR, f"Template Fiberization IOH.xlsx")
    PREV_FIBER_PATH = r"D:\JACOBS\PROJECT\TASK\SEPTEMBER\Week 1\FIBERIZATION\data\Route Fiber Week 34.parquet"
    PREV_POINT_PATH = r"D:\JACOBS\PROJECT\TASK\SEPTEMBER\Week 1\FIBERIZATION\data\Point Ring Week 34.parquet"
    EXPORT_DIR = r"D:\JACOBS\PROJECT\TASK\SEPTEMBER\Week 3\FIBERISASI PAKNO\export\Insert Ring"
    MAPPED_DATA = r"D:\JACOBS\PROJECT\TASK\SEPTEMBER\Week 3\FIBERISASI PAKNO\20250912_110743_Identified_Insert_Data\Insert Data.xlsx"

    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)

    # PREPARE DATA
    print(f"🧩 DATA PREPARATION")
    if not os.path.exists(os.path.join(EXPORT_DIR, "prepared_data", f"Mapped Insert Data.parquet")):
        print(f"ℹ️ Preparing data from input files...")
        prev_fiber = gpd.read_parquet(PREV_FIBER_PATH)
        prev_point = gpd.read_parquet(PREV_POINT_PATH)
        prev_fiber, prev_point = prepare_prevdata(prev_fiber, prev_point)

        if MAPPED_DATA is not None:
            mapped_insert = gpd.read_file(MAPPED_DATA)
            geom_mapped = gpd.points_from_xy(mapped_insert['long'], mapped_insert['lat'], crs="EPSG:4326")
            mapped_insert = gpd.GeoDataFrame(mapped_insert, geometry=geom_mapped)
            print(f"ℹ️ Using provided MAPPED_DATA with {len(mapped_insert):,} records.")
        else:
            new_sites, existing_sites, hubs = input_insertring(EXCEL_FILE)
            newsites_within, newsites_outside = identify_fiberzone(new_sites, prev_fiber=prev_fiber, search_radius=2000)
            mapped_insert = identify_insertdata(newsites_within, prev_fiber=prev_fiber, prev_points=prev_point, search_radius=2000, max_member=MAX_MEMBER)
        mapped_insert = mapped_insert.sort_values(by="distance_to_fiber").reset_index(drop=True)
        
        # STRINGIFY COLUMNS
        for col in ['site_id', 'site_name','site_type', 'region', 'project', 'ring_name', 'nearest_ring', 'near_end', 'far_end']:
            if col in mapped_insert.columns:
                mapped_insert[col] = mapped_insert[col].astype(str)
            if col in prev_point.columns:
                prev_point[col] = prev_point[col].astype(str)
            if col in prev_fiber.columns:
                prev_fiber[col] = prev_fiber[col].astype(str)

        # EXPORT PREPARED DATA
        data_dir = os.path.join(EXPORT_DIR, "prepared_data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        prev_fiber.to_parquet(os.path.join(data_dir, f"Previous Fiber Data.parquet"), index=False)
        prev_point.to_parquet(os.path.join(data_dir, f"Previous Point Data.parquet"), index=False)
        # newsites_outside.to_parquet(os.path.join(data_dir, f"Outside Fiber Zone Insert Data.parquet"), index=False)
        mapped_insert.to_parquet(os.path.join(data_dir, f"Mapped Insert Data.parquet"), index=False)
        print(f"ℹ️ Data preparation completed.\n")
    else:
        print(f"ℹ️ Loading prepared data from {os.path.join(EXPORT_DIR, 'prepared_data')}")
        data_dir = os.path.join(EXPORT_DIR, "prepared_data")
        mapped_insert = gpd.read_parquet(os.path.join(data_dir, f"Mapped Insert Data.parquet"))
        prev_fiber = gpd.read_parquet(os.path.join(data_dir, f"Previous Fiber Data.parquet"))
        prev_point = gpd.read_parquet(os.path.join(data_dir, f"Previous Point Data.parquet"))
        print(f"ℹ️ Prepared data loaded.\n")


    # DETAIL INPUT
    print(f"==============================")
    print(f"📂 Mapped Insert Data   : {len(mapped_insert):,} features")
    print(f"📂 Previous Fiber Data  : {len(prev_fiber):,} features")
    print(f"📂 Previous Point Data  : {len(prev_point):,} features")
    print(f"==============================")

    # TRIAL RUNNING INSERT RING
    # =============================
    start_time = time()
    main_insertring(
        mapped_insert,
        prev_fiber,
        prev_point,
        export_dir=EXPORT_DIR,
        MAX_WORKERS=MAX_WORKERS,
        MAX_MEMBER=MAX_MEMBER,
        ROUTE_TYPE=ROUTE_TYPE,
    )
    end_time = time()
    elapsed_time = end_time - start_time
    print(f"⏱️ Elapsed time: {elapsed_time/60:.2f} minutes")
    print(f"==============================")
    # =============================