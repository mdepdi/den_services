import os
import sys
import zipfile
import geopandas as gpd
import pandas as pd
import networkx as nx
from datetime import datetime
from time import time
from shapely.ops import linemerge
from tqdm import tqdm

sys.path.append(r"D:\JACOBS\SERVICE\API")

from service.intersite.boq_algorithm import main_boq
from service.intersite.ring_algorithm import save_intersite
from modules.table import sanitize_header
from modules.h3_route import identify_hexagon, retrieve_roads, build_graph
from modules.utils import route_path, spof_detection, dropwire_connection
from core.logger import create_logger
from core.config import settings
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------------------------------------
# SETTING
# ------------------------------------------------------
MAX_WORKERS = settings.MAX_WORKERS

# ------------------------------------------------------
# LOGGING CONFIGURATION
# ------------------------------------------------------
logger = create_logger(__file__)

# ------------------------------------------------------
# FIX ROUTE MODULES
# ------------------------------------------------------
def fixroute_algo(
    gdf_ne_ring: gpd.GeoDataFrame,
    gdf_fe_ring: gpd.GeoDataFrame,
    region: str,
    ring: str,
    export_loc: str = None,
) -> tuple:
    
    if export_loc is None:
        raise ValueError(f"Export location is not defined.")
    
    if not os.path.exists(export_loc):
        os.makedirs(export_loc, exist_ok=True)

    logger.info(f"🔄 Processing Ring: {ring}")
    result_route = os.path.join(export_loc, f"Route Ring_{region}_{ring}.parquet")
    result_point = os.path.join(export_loc, f"Points Ring_{region}_{ring}.parquet")
    
    if os.path.exists(result_route):
        logger.info(f"ℹ️ Ring {ring} already processed. Loading existing data...")
        routes = gpd.read_parquet(result_route)
        points = gpd.GeoDataFrame(columns=gdf_ne_ring.columns, geometry='geometry', crs=gdf_ne_ring.crs)
        return points, routes
    
    start_time = time()
    concated = pd.concat([gdf_ne_ring, gdf_fe_ring], ignore_index=True)
    concated = concated.drop_duplicates(subset=['geometry']).reset_index(drop=True)
    total_range = len(gdf_ne_ring)

    logger.info(f"ℹ️ Total NE points: {len(gdf_ne_ring)}, Total FE points: {len(gdf_fe_ring)}")
    if gdf_ne_ring.empty or gdf_fe_ring.empty:
        logger.info(f"⚠️ No NE or FE data for ring: {ring}. Skipping...")
    
    hex_list = identify_hexagon(concated, type="convex")
    logger.info(f"ℹ️ Total {len(hex_list)} hexagons for ring: {ring}")
    roads = retrieve_roads(hex_list, type="roads")
    nodes = retrieve_roads(hex_list, type="nodes")

    concated = concated.to_crs(epsg=3857).reset_index(drop=True)
    roads = roads.to_crs(epsg=3857).reset_index(drop=True)
    nodes = nodes.to_crs(epsg=3857).reset_index(drop=True)
    G = build_graph(roads, graph_type='full_weighted')

    concated = gpd.sjoin_nearest(concated, nodes[['node_id', 'geometry']], how='left', distance_col='dist_to_node')
    gdf_ne_ring = gpd.sjoin_nearest(gdf_ne_ring, nodes[['node_id', 'geometry']], how='left', distance_col='dist_to_node')
    gdf_fe_ring = gpd.sjoin_nearest(gdf_fe_ring, nodes[['node_id', 'geometry']], how='left', distance_col='dist_to_node')
    concated = concated.rename(columns={'node_id': 'nearest_node'})
    gdf_ne_ring = gdf_ne_ring.rename(columns={'node_id': 'nearest_node'})
    gdf_fe_ring = gdf_fe_ring.rename(columns={'node_id': 'nearest_node'})

    segments = []
    for i in range(total_range):
        ne_point = gdf_ne_ring.iloc[i]
        fe_point = gdf_fe_ring.iloc[i]
        start_id = ne_point['site_id']
        end_id = fe_point['site_id']
        node_start = ne_point['nearest_node']
        node_end = fe_point['nearest_node']

        logger.info(f"🔄 Routing Near End {start_id} -> Far End {end_id}")
        try:
            path, path_geom, path_length = route_path(node_start, node_end, G, roads, merged=True)
            path_geom, path_length = dropwire_connection(path_geom, ne_point, fe_point, nodes, node_start, node_end)

            if not path_geom.is_empty:
                segment_name = f"{start_id}-{end_id}"

                segment_record = {
                    'name': segment_name,
                    'near_end': start_id,
                    'far_end': end_id,
                    'algo': "Fix Route",
                    'fo_note': 'merged',
                    'ring_name': ring,
                    'region': region,
                    'length': round(path_length, 3),
                    'geometry': path_geom
                }

                segments.append(segment_record)
                logger.info(f"🟢 Length: {path_length:10,.2f} m  | Routed segment    : {segment_name}")

                ## UPDATE GRAPH
                # # PENALTY EXISTING FIBER
                # for j in range(len(path) - 1):
                #     if G.has_edge(path[j], path[j + 1]):
                #         G[path[j]][path[j + 1]]['weight'] = 1e10
                #     if G.has_edge(path[j + 1], path[j]):
                #         G[path[j + 1]][path[j]]['weight'] = 1e10

                ## MAXIMIZE EXISTING FIBER
                for j in range(len(path) - 1):
                    if G.has_edge(path[j], path[j + 1]):
                        G[path[j]][path[j + 1]]['weight'] = G[path[j]][path[j + 1]]['weight'] / 2
                    if G.has_edge(path[j + 1], path[j]):
                        G[path[j + 1]][path[j]]['weight'] = G[path[j + 1]][path[j]]['weight'] / 2
            else:
                logger.critical(f"⚠️ No geometry found for segment {start_id} to {end_id}, skipping.")
        except nx.NetworkXNoPath:
            logger.critical(f"⚠️ No path found between {start_id} and {end_id}, skipping segment.")
    ring_paths = gpd.GeoDataFrame(segments, geometry='geometry', crs="EPSG:3857")

    # SPOF CHECKING
    ring_paths = spof_detection(ring_paths, concated, G, roads, nodes, threshold_spof=3000, threshold_alt=25)

    # EXPORT
    if not concated.empty:
        result_point = os.path.join(export_loc, f"Points Ring_{region}_{ring}.parquet")
        concated.to_parquet(result_point)
    if not ring_paths.empty:
        result_route = os.path.join(export_loc, f"Route Ring_{region}_{ring}.parquet")
        ring_paths.to_parquet(result_route)
    logger.info(f"✅ Ring {ring} processing completed.\n")

    end_time = time()
    elapsed_time = end_time - start_time
    logger.info(f"⏱️ {ring} processed in {elapsed_time:,.2f} seconds")
    return concated, ring_paths

def parallel_fixroute(
    ne_data: gpd.GeoDataFrame,
    fe_data: gpd.GeoDataFrame,
    export_dir: str,
    **kwargs
    ) -> tuple:

    task_celery = kwargs.get("task_celery", None)
    ring_list = ne_data['ring_name'].dropna().unique().tolist()
    logger.info(f"🔄 Total Rings to Process: {len(ring_list):,}")

    all_new_points = []
    all_new_segments = []
    checkpoint_dir = os.path.join(export_dir, "Checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for ring in ring_list:
            ne_ring = ne_data[ne_data["ring_name"] == ring].reset_index(drop=True)
            fe_ring = fe_data[fe_data["ring_name"] == ring].reset_index(drop=True)
            region = ne_ring['region'].mode()[0]

            if ne_ring.empty:
                logger.info(f"⚠️ No NE data for ring {ring}, skipping.")
                continue
            if fe_ring.empty:
                logger.info(f"⚠️ No FE data for ring {ring}, skipping.")
                continue

            future = executor.submit(
                fixroute_algo,
                ne_ring,
                fe_ring,
                region,
                ring,
                checkpoint_dir,
            )
            futures[future] = ring

        total_process = len(futures)
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Rings"):
            ring = futures[future]
            try:
                new_points, new_segments = future.result()
                if new_points is not None:
                    all_new_points.append(new_points)
                if new_segments is not None:
                    all_new_segments.append(new_segments)
                
                if task_celery:
                    task_celery.update_state(
                        state="PROGRESS",
                        meta={"status": (f"Completed {len(all_new_points)}/{total_process} rings.")},
                    )
                    
            except Exception as e:
                logger.info(f"❌ Error processing ring {ring}: {e}")
                if task_celery:
                    task_celery.update_state(
                        state="FAILURE",
                        meta={"status": 
                                f"Error in ring {ring}: {e}. "
                              },
                    )

    if all_new_points:
        all_new_points = pd.concat(all_new_points, ignore_index=True)
        logger.info(f"ℹ️ Total new points collected: {len(all_new_points):,}")
    else:
        all_new_points = gpd.GeoDataFrame(columns=ne_data.columns, geometry="geometry", crs="EPSG:3857")
        logger.info(f"ℹ️ No new points collected.")

    if all_new_segments:
        all_new_segments = pd.concat(all_new_segments, ignore_index=True)
        logger.info(f"ℹ️ Total new segments collected: {len(all_new_segments):,}")
    else:
        raise ValueError(f"No new segments collected.")
    return all_new_points, all_new_segments


def validate_fixroute(df: pd.DataFrame):
    df = sanitize_header(df, lowercase=True)
    col_ne = df.columns[df.columns.str.contains("_a")].tolist() + ["ring_name", "region"]
    col_fe = df.columns[df.columns.str.contains("_b")].tolist() + ["ring_name", "region"]

    df_ne = df[col_ne].copy()
    df_fe = df[col_fe].copy()
    df_ne.columns = [col.replace("_a", "") for col in df_ne.columns]
    df_fe.columns = [col.replace("_b", "") for col in df_fe.columns]
    logger.info(f"ℹ️ Validating NE and FE data...")
    try:
        geom_ne = gpd.points_from_xy(df_ne["longitude"], df_ne["latitude"])
        geom_fe = gpd.points_from_xy(df_fe["longitude"], df_fe["latitude"])
    except:
        try:
            geom_ne = gpd.points_from_xy(df_ne["long"], df_ne["lat"])
            geom_fe = gpd.points_from_xy(df_fe["long"], df_fe["lat"])
        except Exception as e:
            raise ValueError(e)

    logger.info(f"ℹ️ Converting to GeoDataFrame...")
    gdf_ne = gpd.GeoDataFrame(df_ne, geometry=geom_ne, crs="EPSG:4326").to_crs(epsg=3857)
    gdf_fe = gpd.GeoDataFrame(df_fe, geometry=geom_fe, crs="EPSG:4326").to_crs(epsg=3857)
    gdf_ne["point_type"] = "NE"
    gdf_fe["point_type"] = "FE"
    logger.info(f"ℹ️ Validation completed.\n")
    return gdf_ne, gdf_fe

def main_fixroute(
    template_df: pd.DataFrame,
    export_dir: str,
    boq:bool = False,
    **kwargs
):
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    cable_cost = kwargs.get("cable_cost", 35000)
    vendor = kwargs.get("vendor", "TBG")
    program = kwargs.get("program", "Fiberization")
    method = kwargs.get("method", "Fix Route")
    task_celery = kwargs.get("task_celery", None)
    design_type = 'Bill of Quantity' if boq else 'Design'

    logger.info(f"🌏 Starting Intersite")
    logger.info(f"ℹ️ Method  : {method}")
    logger.info(f"ℹ️ Vendor  : {vendor}")
    logger.info(f"ℹ️ Program : {program}")
    logger.info(f"ℹ️ Design  : {design_type}")
    logger.info(f"ℹ️ Total Data  : {len(template_df):,}")

    # PROCESS FIXED ROUTING
    date_today = datetime.now().strftime("%d-%m-%Y")
    points_path = os.path.join(export_dir, f"Fixed Points_Intersite_{date_today}.parquet")
    routes_path = os.path.join(export_dir, f"Fixed Route_Intersite_{date_today}.parquet")
    if os.path.exists(points_path) and os.path.exists(routes_path):
        updated_points = gpd.read_parquet(points_path)
        updated_routes = gpd.read_parquet(routes_path)
        logger.info(f"✅ Loaded existing processed data from {export_dir}.")
    else:
        ne_data, fe_data = validate_fixroute(template_df)
        
        if task_celery:
            task_celery.update_state(state="PROGRESS", meta={"status": "Starting Parallel Fix Route"})
        updated_points, updated_routes = parallel_fixroute(ne_data, fe_data, export_dir, task_celery)

    if not updated_points.empty:
        logger.info(f"ℹ️ Total updated points: {len(updated_points):,}")
        updated_points.to_parquet(points_path)
    if not updated_routes.empty:
        logger.info(f"ℹ️ Total updated routes: {len(updated_routes):,}")
        updated_routes.to_parquet(routes_path)

    if 'program' not in updated_points.columns:
        updated_points['program'] = program

    if 'program' not in updated_routes.columns:
        updated_routes['program'] = program

    # EXPORT
    if boq:
        logger.info("🧩 Running BOQ Calculation...")
        main_boq(updated_points, updated_routes, export_dir=export_dir)
    else:
        # TOPOLOGY CHECK
        logger.info("🧩 Save Design Information")
        save_intersite(updated_points, updated_routes, export_dir, method)

    logger.info("🏆 Fix Route export completed.")
    logger.info(f"ℹ️ All files saved to: {export_dir}")

if __name__ == "__main__":
    excel_file = r"D:\JACOBS\SERVICE\API\data\template\Template_Fixed_Route.xlsx"
    export_dir = fr"D:\JACOBS\SERVICE\API\test\Fix Route"
    boq = False
    program ="Q1NewSite2026"


    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    template_df = pd.read_excel(excel_file)
    start_time = time()
    main_fixroute(
        template_df= template_df,
        export_dir=export_dir,
        boq=boq,
        program=program
    )
    end_time = time()
    elapsed_time = end_time - start_time
    logger.info(f"⏱️ Elapsed time: {elapsed_time/60:.2f} minutes")

    zip_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_Fix Route_Task.zip"
    zip_filepath = os.path.join(export_dir, zip_filename)
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(export_dir):
            for file in files:
                if file != zip_filename:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_dir)
                    zipf.write(file_path, arcname)
    logger.info(f"🏆 Result files zipped at {zip_filepath}.")
