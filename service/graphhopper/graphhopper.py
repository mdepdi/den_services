# ===============
# ROUTE EXTRACTOR
# ===============
import os
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import simplekml
import zipfile

from tqdm import tqdm
from time import time
from datetime import datetime
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import nearest_points
from shapely.ops import linemerge
from concurrent.futures import ProcessPoolExecutor, as_completed

from enum import Enum
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.append(root)

from modules.data import read_gdf, validate_longlat
from modules.table import excel_styler
from modules.utils import admin_information
from modules.geometry import geodesic_length
from modules.graphhopper import graphhopper_knn
from core.config import settings

from service.intersite.ring_algorithm import save_intersite

MAINDATA_DIR = settings.MAINDATA_DIR
DATA_DIR = settings.DATA_DIR

# ------------------------------------------------------
# LOGGER
# ------------------------------------------------------
from core.logger import create_logger
logger = create_logger(__file__)

def nearest_point_to_point(source_path: str, target_path: str, export_dir:str, k_final:int=1, sep:str=";", cutoff:int=100000, task_celery=False):
    source_gdf = read_gdf(source_path, geom_type='point')
    target_gdf = read_gdf(target_path, geom_type='point')

    if 'site_id' not in source_gdf.columns:
        if 'name' in source_gdf.columns:
            source_gdf['site_id'] = source_gdf['name']
        else:
            source_gdf['site_id'] = "Source" + source_gdf.index.astype(str)

    if 'site_name' not in source_gdf.columns:
        if 'name' in source_gdf.columns:
            source_gdf['site_name'] = source_gdf['name']
        else:
            source_gdf['site_name'] = "Source" + source_gdf.index.astype(str)

    if 'site_id' not in target_gdf.columns:
        if 'name' in target_gdf.columns:
            target_gdf['site_id'] = target_gdf['name']
        else:
            target_gdf['site_id'] = "Source" + target_gdf.index.astype(str)

    if 'site_name' not in target_gdf.columns:
        if 'name' in target_gdf.columns:
            target_gdf['site_name'] = target_gdf['name']
        else:
            target_gdf['site_name'] = "Source" + target_gdf.index.astype(str)

    if 'long' not in source_gdf.columns:
        source_gdf['long'] = source_gdf.geometry.x
        source_gdf['lat'] = source_gdf.geometry.y

    if 'long' not in target_gdf.columns:
        target_gdf['long'] = target_gdf.geometry.x
        target_gdf['lat'] = target_gdf.geometry.y

    # Validate
    source_gdf = validate_longlat(source_gdf)
    target_gdf = validate_longlat(target_gdf)
    source_gdf = admin_information(source_gdf)
    target_gdf = admin_information(target_gdf)

    source_gdf["site_id"] = source_gdf["site_id"].astype(str)
    target_gdf["site_id"] = target_gdf["site_id"].astype(str)
    source_gdf["site_type"] = "Site List"
    target_gdf["site_type"] = "FO Hub"

    source_gdf = source_gdf.reset_index(drop=True)
    target_gdf = target_gdf.reset_index(drop=True)

    logger.info(f"ℹ️ Running Graphhopper | Nearest Point to Point.")
    start_time = time()
    routing_gdf = graphhopper_knn(source_gdf, target_gdf, k_final=k_final, profile='custom_car', task_celery=task_celery)
    routing_gdf['length'] = routing_gdf.geometry.to_crs(epsg=4326).apply(geodesic_length)

    # Export
    os.makedirs(export_dir, exist_ok=True)
    excel_path = os.path.join(export_dir, f"Grapphopper_Routing_Result.xlsx")
    parquet_path = os.path.join(export_dir, f"Grapphopper_Routing_Result.parquet")

    # Cutoff
    routing_gdf = routing_gdf[routing_gdf['length'] <= cutoff].copy()
    if len(routing_gdf) == 0:
        raise ValueError(f"🔴 There is no routing result fulfill threshold {cutoff} m.")

    # Add Admin
    routing_gdf['name'] = routing_gdf["site_id_a"].astype(str) + str(sep) + routing_gdf["site_id_b"].astype(str)
    
    routing_gdf.to_parquet(parquet_path, index=False)
    excel_styler(routing_gdf.drop(columns='geometry')).to_excel(excel_path, sheet_name='DEN Graphhopper Routing', index=False)

    # Intersite Format
    points_list = []
    record_region = {}
    for idx, row in routing_gdf.iterrows():
        src_idx = row["src_idx"]
        tgt_idx = row["tgt_idx"]
        near_id = row['site_id_a']
        far_id = row['site_id_b']

        near_end = source_gdf.iloc[src_idx].copy()
        far_end = target_gdf.iloc[tgt_idx].copy()
        prov = near_end['Provinsi']
        city = near_end['Kabkot']

        if city in record_region.keys():
            record_region[city] += 1
        else:
            record_region[city] = 1
        
        num_record = record_region[city]
        ring_name = f"{city}_{num_record}"

        near_end['ring_name'] = ring_name
        near_end['region'] = prov
        far_end['ring_name'] = ring_name
        far_end['region'] = prov
        routing_gdf.at[idx, 'region'] = prov
        routing_gdf.at[idx, 'ring_name'] = ring_name

        points_list.append(near_end)
        points_list.append(far_end)

    points_df = pd.DataFrame(points_list)

    points_geom = gpd.points_from_xy(points_df['long'], points_df['lat'], crs="EPSG:4326")
    points_gdf = gpd.GeoDataFrame(points_df, geometry=points_geom)
    points_gdf = points_gdf.sort_values(by=['region', 'ring_name'])
    
    # Save Intersite
    save_intersite(
        points=points_gdf,
        paths=routing_gdf,
        export_dir=export_dir,
        sep=sep,
        method="Grapphopper Star"
    )
    
    end_time = time()
    process_time = end_time-start_time
    logger.info(f"✅ Grapphopper | Nearest Point to Point | Done in {process_time/60:.2f} minutes.")

if __name__ == "__main__":
    # PROCESS ROUTING
    source_path = r"D:\JACOBS\PROJECT\TASK\2026\FEB\W2\DISTANCE ODC TSEL\TRIAL ODC\Source.xlsx"
    target_path = r"D:\JACOBS\PROJECT\TASK\2026\FEB\W2\DISTANCE ODC TSEL\TRIAL ODC\Target.xlsx"
    k_final = 1

    export_dir = r"D:\JACOBS\PROJECT\TASK\2026\FEB\W2\DISTANCE ODC TSEL\TRIAL ODC"
    os.makedirs(export_dir, exist_ok=True)

    nearest_point_to_point(source_path, target_path, export_dir, k_final=1, cutoff=100000)

