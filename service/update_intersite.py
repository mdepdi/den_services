import os
import sys
import numpy as np
import pickle
import simplekml
import geopandas as gpd
import networkx as nx
import pandas as pd
import zipfile
from collections import defaultdict

from shapely.geometry import Polygon
from shapely.ops import linemerge
from time import time
from glob import glob
from tqdm import tqdm
from datetime import datetime

from modules.data import fiber_utilization
from modules.table import excel_styler, find_best_match
from modules.utils import create_topology
from modules.kml import export_kml, sanitize_kml
from modules.h3_route import retrieve_roads, identify_hexagon


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
    gdf['region'] = gdf['folders'].str.split(';').str[0].str.strip()
    gdf['ring_name'] = gdf['folders'].str.split(';').str[1].str.strip()
    gdf['site_type'] = gdf['folders'].str.split(';').str[-1].str.strip()
    return gdf

# =============================
# UPDATE CONNECTION
# =============================
def identify_nearfar(point_gdf, route_gdf):
    ringlist = set(point_gdf['ring_name'])
    routes_identified = []
    for ring in tqdm(ringlist, total=len(ringlist), desc="Process Near End Far End"):
        point_ring = point_gdf[point_gdf['ring_name'] == ring].copy()
        route_ring = route_gdf[route_gdf['ring_name'] == ring].copy()
        point_ids = set(point_ring['site_id'].astype(str))
        route_ids = set(route_ring['name'].astype(str))

        # CLEAN POSSIBLE CONNECTION
        connection, score = find_best_match('Connection', route_ids, threshold=0.85)
        route, score = find_best_match('Route', route_ids, threshold=0.85)
        if connection is not None:
            route_ring = route_ring[route_ring["name"] != connection]
        if route is not None:
            route_ring = route_ring[route_ring["name"] != route]

        # fo_hub = point_ring[point_ring['site_type'].str.contains('hub')]
        # sitelist = point_ring[~(point_ring['site_type'].str.contains('hub'))]

        # NEAR END AND FAR END
        ne_checked = []
        fe_checked = []
        dropped = []
        for idx, row in route_ring.iterrows():
            name = row['name']
            parts = name.split("-")
            if len(parts) < 2:
                print(f"⚠️ Unexpected name format: {name}")
                dropped.append(name)
                continue

            near_end, far_end = parts[0], parts[-1]

            ne_selected = [i for i in point_ids if (near_end in i)]
            fe_selected = [i for i in point_ids if (far_end in i)]

            if len(ne_selected) > 0 and len(fe_selected) > 0:
                isdouble_ne = len(ne_selected) > 1
                isdouble_fe = len(fe_selected) > 1

                if isdouble_ne and not isdouble_fe:
                    fe_selected = fe_selected[0]
                    ne_selected = str(name).replace(fe_selected, "").rstrip("-")
                elif not isdouble_ne and isdouble_fe:
                    ne_selected = ne_selected[0]
                    fe_selected = str(name).replace(ne_selected, "").lstrip("-")
                else:
                    ne_selected = ne_selected[0]
                    fe_selected = fe_selected[0]

                route_ring.at[idx, 'near_end'] = ne_selected
                route_ring.at[idx, 'far_end'] = fe_selected
                ne_checked.append(ne_selected)
                fe_checked.append(fe_selected)
            else:
                print(f"🔴 Error in ring {ring} NE : {ne_selected} -> FE : {fe_selected}. Segment {name} dropped.")
                dropped.append(name)

        
        route_ring = route_ring[~(route_ring['name'].isin(dropped))]
        routes_identified.append(route_ring)
        
    routes_identified = pd.concat(routes_identified)
    routes_identified = routes_identified.reset_index(drop=True)

    return routes_identified

def identify_connection(ring: str, target_fiber:gpd.GeoDataFrame, target_point:gpd.GeoDataFrame, start_column:str='near_end')-> tuple:
    if target_fiber.crs != 'EPSG:3857':
        target_fiber = target_fiber.to_crs(epsg=3857)
    if target_point.crs != 'EPSG:3857':
        target_point = target_point.to_crs(epsg=3857)

    colopriming = gpd.read_parquet(f"D:/JACOBS/DATA/10. TBG Sitelist/Colopriming_Aug 2025/Sitelist TBG_Aug 2025_v.1.0.parquet", columns=['SiteId TBG', 'Sitename TBG', 'geometry'])
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
        candidates = target_fiber[start_column].values
        selected, score = find_best_match(current, candidates, threshold=0.1)
        current_site = target_fiber[target_fiber[start_column] == selected]
        if current_site.empty:
            print(target_fiber[['name', 'near_end', 'far_end']])
            raise ValueError(f"No next site found connected to {current} in ring {ring}.")
        
        next_site = current_site[opposite_column].values[0]
        connection.append(next_site)
        current = next_site
        if current in hub_ids and current != start_hub:
            break
        if current in hub_ids and current == start_hub:
            break

    # RING STARTS FROM START HUB
    if start_column == 'far_end':
        connection = connection[::-1]

    # UPDATE RING DATA
    points_sequential = []
    for site_id in connection:
        if site_id in fo_hub['site_id'].values:
            point_data = fo_hub[fo_hub['site_id'] == site_id].iloc[0]
        elif site_id in site_list['site_id'].values:
            point_data = site_list[site_list['site_id'] == site_id].iloc[0]
        else:
            print(f"⚠️ Site ID {site_id} not found in any dataset, try colopriming.")
            point_data = colopriming[colopriming['site_id'] == site_id].iloc[0] if not colopriming[colopriming['site_id'] == site_id].empty else None
            if point_data is None:
                print(f"⚠️ Site ID {site_id} still not found after colopriming, skipping.")
                continue
        points_sequential.append(point_data)
    
    if not points_sequential:
        print(f"⚠️ No valid points found for ring {ring} after insertion.")
        return None, None
    points_sequential = gpd.GeoDataFrame(points_sequential, columns=points_sequential[0].index.tolist(), crs='EPSG:3857').reset_index(drop=True)
    return points_sequential, connection
    

# =============================
# MAIN FUNCTION
# =============================
def routing_update(
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
        
        # PATH GEOMETRY
        path_geom = target_fiber[(target_fiber[start_column] == start_id) & (target_fiber[opposite_column] == end_id)].reset_index(drop=True)
        path_length = path_geom['length'].sum()
        if not path_geom.empty:
            union_geom = path_geom.geometry.union_all()
            merged_line = linemerge(union_geom) if union_geom.geom_type == "MultiLineString" else union_geom
            segment_name = f"{start_id}-{end_id}" if start_column == 'near_end' else f"{end_id}-{start_id}"
            segment_record = {
                'name': segment_name,
                start_column: start_id,
                opposite_column: end_id,
                'length': path_length,
                'fo_note': 'merged',
                'geometry': merged_line
            }
                
            segments.append(segment_record)
        else:
            print(f"🟠 No geometry found for existing segment {start_id} to {end_id}, try reverse")
            print(target_fiber[[start_column, opposite_column]])
            # print(target_fiber[start_column].unique().tolist())
            # print(target_fiber[opposite_column].unique().tolist())
            path_geom = target_fiber[(target_fiber[start_column] == end_id) & (target_fiber[opposite_column] == start_id)].reset_index(drop=True)
            path_length = path_geom['length'].sum()
            if not path_geom.empty:
                union_geom = path_geom.geometry.union_all()
                merged_line = linemerge(union_geom) if union_geom.geom_type == "MultiLineString" else union_geom
                segment_name = f"{end_id}-{start_id}" if start_column == 'near_end' else f"{start_id}-{end_id}"

                segment_record = {
                    'name': segment_name,
                    start_column: end_id,
                    opposite_column: start_id,
                    'length': path_length,
                    'fo_note': 'merged',
                    'geometry': merged_line
                }
                    
                segments.append(segment_record)
            else:
                print(f"⚠️ No geometry found for existing segment {start_id} to {end_id} in both directions, skipping.")

    if segments:
        segments_gdf = gpd.GeoDataFrame(segments, columns=segments[0].keys(), geometry='geometry', crs='EPSG:3857')
        segments_gdf['length'] = segments_gdf['length'].round(2)
        return segments_gdf
    else:
        return gpd.GeoDataFrame(columns=target_fiber.columns, geometry='geometry', crs=target_fiber.crs)
    
def update_intersite(
    ring: str,
    target_fiber: gpd.GeoDataFrame,
    target_point: gpd.GeoDataFrame,
) -> tuple:
    # print(f"🔄 Processing Ring: {ring}")
    start_time = time()
    # FO HUB AND SITELIST
    fo_hub = target_point[target_point['site_type'] == 'FO Hub'].reset_index(drop=True)
    site_list = target_point[target_point['site_type'] == 'Site List'].reset_index(drop=True)
    total_point = len(fo_hub) + len(site_list)
    # print(f"ℹ️ FO Hub(s)     : {len(fo_hub):,}")
    # print(f"ℹ️ Site List(s)  : {len(site_list):,}")
    # print(f"ℹ️ Total Points  : {total_point:,}")

    if len(fo_hub) == 0:
        print(f"🔴 {ring} has no FO Hub in point data, cannot proceed.")
        print(f"Target Points: {target_point[['site_id', 'site_type']].to_dict(orient='records')}")
        return None, None

    # CHECK PREV CONNECTION
    hub_ids = fo_hub['site_id'].tolist()
    start_hub_candidates = target_fiber[target_fiber['near_end'].isin(hub_ids)]['near_end'].values
    start_hub = None
    start_column = None

    if len(start_hub_candidates) > 0:
        start_hub = start_hub_candidates[0]
        start_column = 'near_end'
        # print(f"ℹ️ Start Col 'near_end' | FO Hub {start_hub}")
    else:
        # Check far_end
        start_hub_candidates = target_fiber[target_fiber['far_end'].isin(hub_ids)]['far_end'].values
        if len(start_hub_candidates) > 0:
            start_hub = start_hub_candidates[0]
            start_column = 'far_end'
            # print(f"ℹ️ Start Col 'far_end' | FO Hub {start_hub}")
        else:
            start_hub = hub_ids[0]
            start_column = 'near_end'
            print(f"⚠️ FO Hub {start_hub} not found in fiber data, defaulting to 'near_end'.")
            
    if len(fo_hub) == 0 or len(fo_hub) > 2:
            print(f"⚠️ {ring} has {len(fo_hub):,} FO Hubs, check manually.")
            return None, None
    
    # PROJECT DETAILS
    region = fo_hub[fo_hub['region'].notna()]['region'].mode()[0] if not fo_hub[fo_hub['region'].notna()]['region'].mode().empty else 'Unknown Region'
    # project = fo_hub['project'].mode()[0] if not fo_hub['project'].mode().empty else 'Unknown Program'

    # IDENTIFY CONNECTION
    points_sequential, connection = identify_connection(ring, target_fiber, target_point, start_column)
    if points_sequential is None or connection is None:
        print(f"⚠️ Skipping ring {ring} due to connection identification failure.")
        return None, None
    
    points_sequential['region'] = region
    # points_sequential['project'] = project
    points_sequential['ring_name'] = ring

    routed_segments = routing_update(ring, connection, points_sequential, target_fiber, start_column)

    if routed_segments.empty:
        print(f"🔴 No routed segments generated for ring {ring}.")
        return points_sequential, None
    
    routed_segments['region'] = region
    # routed_segments['project'] = project
    routed_segments['ring_name'] = ring
    # print(f"ℹ️ Routed segments generated: {len(routed_segments):,}")
    
    end_time = time()
    # elapsed_time = end_time - start_time
    # print(f"⏱️ {ring} processed in {elapsed_time:,.2f} seconds")
    return points_sequential, routed_segments

def export_update(
    updated_points: gpd.GeoDataFrame,
    updated_paths: gpd.GeoDataFrame,
    export_dir: str,
    route_type: str = 'merged',
):
    date_today = datetime.now().strftime("%Y%m%d")

    # EXPORT PARQUET
    if not updated_points.empty:
        updated_points.to_crs(epsg=4326).to_parquet(os.path.join(export_dir, f"Updated Point Ring_{date_today}.parquet"), index=False)
        print(f"✅ Exported Updated Point Ring with {len(updated_points):,} records.")
    if not updated_paths.empty:
        updated_paths.to_crs(epsg=4326).to_parquet(os.path.join(export_dir, f"Updated Route Fiber_{date_today}.parquet"), index=False)
        print(f"✅ Exported Updated Route Fiber with {len(updated_paths):,} records.")

    # EXPORT EXCEL
    excel_path = os.path.join(export_dir, f"Summary Report_Fiberization_{date_today}.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        if not updated_points.empty:
            sheet_name = "Site Information"
            updated_points_report = updated_points.drop(columns="geometry").reset_index(drop=True)
            excel_styler(updated_points_report).to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"📊 Excel sheet '{sheet_name}' with {len(updated_points_report):,} records written.")
        if not updated_paths.empty:
            sheet_name = "Route Information"
            updated_paths_report = updated_paths.drop(columns="geometry").reset_index(drop=True)
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
                updated_paths[['ring_name', 'region']].drop_duplicates(),
                # updated_paths[['ring_name', 'region', 'project']].drop_duplicates(),
                on='ring_name',
                how='left'
            )
            updated_paths_report = updated_paths_report.sort_values(by=['ring_name', 'near_end']).reset_index(drop=True)
            updated_paths_report.columns = updated_paths_report.columns.str.replace(' ', '_').str.lower()
            excel_styler(updated_paths_report).to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"📊 Excel sheet '{sheet_name}' with {len(updated_paths_report):,} records written.")

def main_update_intersite(point_gdf:gpd.GeoDataFrame, route_gdf:gpd.GeoDataFrame, export_dir:str, route_type="classified", target_fiber:gpd.GeoDataFrame=None):
    # CLEAN TOPOLOGY
    print(f"🌏 Update Intersite")
    route_gdf = route_gdf[route_gdf['name'].str.lower() != 'connection'].copy()

    # CRS
    if route_gdf.crs is None:
        route_gdf = route_gdf.set_crs(epsg=4326)
    route_gdf = route_gdf.to_crs(epsg=3857)
    point_gdf = point_gdf.to_crs(epsg=3857)

    # PREPARE DATA
    print(f"ℹ️ Prepare Data")
    route_gdf = route_gdf[['name', 'folders', 'geometry']].copy()
    point_gdf = point_gdf[['name', 'folders', 'geometry']].copy()
    route_gdf['name'] = route_gdf['name'].astype(str)
    point_gdf['name'] = point_gdf['name'].astype(str)
    point_gdf['site_id'] = point_gdf['name']
    point_gdf['long'] = point_gdf.geometry.x
    point_gdf['lat'] = point_gdf.geometry.y

    point_gdf = folders_identify(point_gdf)
    route_gdf = folders_identify(route_gdf)
    route_gdf = route_gdf.rename(columns={'site_type': 'route_type'})
    route_gdf['geometry'] = route_gdf.geometry.apply(lambda x: linemerge(x) if x.geom_type == 'MultiLineString' else x)
    route_gdf['length'] = route_gdf.geometry.length.round(3)

    point_gdf['geometry'] = point_gdf.geometry.apply(lambda x: remove_z(x) if x.has_z else x)
    route_gdf['geometry'] = route_gdf.geometry.apply(lambda x: remove_z(x) if x.has_z else x)

    # IDENTIFY NEAR END FAR END
    routes_identified = identify_nearfar(point_gdf, route_gdf)

    # PROCESSING UPDATE
    all_routes = []
    all_points = []
    ringlist = set(point_gdf['ring_name'])
    for ring in tqdm(ringlist, total=len(ringlist), desc="Process Ring"):
        point = point_gdf[point_gdf['ring_name'] == ring].copy()
        route = routes_identified[routes_identified['ring_name'] == ring].copy()

        # try:
        point, route = update_intersite(ring, route, point)
        if not point.empty:
            all_points.append(point)
        if not route.empty:
            all_routes.append(route)
        # except Exception as e:
            # print(f"🔴 Error in ring {ring}: {e}")

    if len(all_points) > 0 and len(all_routes) > 0:
        all_points = pd.concat(all_points)
        all_routes = pd.concat(all_routes)
    
    print(f"ℹ️ Generate Detailed Excel.")

    # TOPOLOGY CHECK
    # topology_paths = create_topology(all_points)

    # # IDENTIFY ROUTE UTILIZATION
    if route_type == 'classified':
        all_routes = fiber_utilization(all_routes, target_fiber=target_fiber, overlap=True)

    # paths_utilization = fiber_utilization(all_routes, overlap=False)
    # paths_utilization = paths_utilization[['ring_name','fo_note', 'length', 'geometry']]
    # paths_utilization = paths_utilization.merge(
    #     all_routes[["ring_name", "region", "project"]].drop_duplicates(),
    #     on="ring_name",
    #     how="left",
    # )
    # print(f"ℹ️ Total paths for utilization analysis: {len(paths_utilization):,}")

    # EXPORT
    export_update(all_points, all_routes, export_dir, route_type=route_type)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"{timestamp}_Update_Intersite.zip"
    zip_path = os.path.join(export_dir, zip_name)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(export_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.abspath(file_path) == os.path.abspath(zip_path):
                    continue  # ✅ skip the zip itself
                if os.path.splitext(file_path)[-1] == ".zip":
                    continue
                arcname = os.path.relpath(file_path, export_dir)
                zf.write(file_path, arcname)

    print(f"📦 Result files zipped.")
    print(f"✅ Export completed.")
    return zip_path

if __name__ == "__main__":
    export_dir = r"D:\JACOBS\PROJECT\TASK\OKTOBER\Week 1\UPDATE KMZ PAKNO\Export"
    os.makedirs(export_dir, exist_ok=True)

    point_gdf = gpd.read_parquet(r"D:\JACOBSPACE\TBIG Impact 2025\QCC Fiberisasi\Asessment\SMARTROUTE_Q1AOP2025_V2\Points_SmartRoute.parquet")
    route_gdf = gpd.read_parquet(r"D:\JACOBSPACE\TBIG Impact 2025\QCC Fiberisasi\Asessment\SMARTROUTE_Q1AOP2025_V2\Lines_SmartRoute.parquet")
    fiber_gdf = gpd.read_file(r"D:\JACOBSPACE\TBIG Impact 2025\QCC Fiberisasi\Data\Compile TBG FO Route Only (8 July 2024)\Compile TBG FO Route Only (8 July 2024).TAB").to_crs(epsg=4326)
    route_type = "classified"

    # PROCESS UPDATE
    result = main_update_intersite(point_gdf, route_gdf, export_dir=export_dir, route_type=route_type, target_fiber=fiber_gdf)
