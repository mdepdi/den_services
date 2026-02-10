# ===============
# ROUTE EXTRACTOR
# ===============
import os
import sys
import time
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import simplekml
import zipfile

from tqdm import tqdm
from datetime import datetime
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import nearest_points
from shapely.ops import linemerge
from concurrent.futures import ProcessPoolExecutor, as_completed

from enum import Enum
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.append(root)

from modules.kml import export_kml, sanitize_kml, validate_kmz_design, validate_kmz_ipl
from modules.data import read_gdf, validate_longlat
from modules.table import excel_styler
from modules.utils import auto_group, admin_information
from modules.geometry import geodesic_length
from modules.graphhopper import grapphopper_knn, distance_fiber
from core.config import settings
from service.intersite.report import boq_generation, boq_mmp

MAINDATA_DIR = settings.MAINDATA_DIR
DATA_DIR = settings.DATA_DIR

# ------------------------------------------------------
# LOGGER
# ------------------------------------------------------
from core.logger import create_logger
logger = create_logger(__file__)

def nearest_point_to_point(source_path: str, target_path: str, export_dir:str, k_final:int=1, cutoff:int=100000, task_celery=None):
    source_gdf = read_gdf(source_path, geom_type='point')
    target_gdf = read_gdf(target_path, geom_type='point')
    target_gdf['site_id'] = target_gdf['name']

    if 'long' not in target_gdf.columns:
        target_gdf['long'] = target_gdf.geometry.x
        target_gdf['lat'] = target_gdf.geometry.y

    source_gdf = validate_longlat(source_gdf)
    target_gdf = validate_longlat(target_gdf)
    routing_gdf = grapphopper_knn(source_gdf, target_gdf, k_final=k_final, profile='car')
    routing_gdf['length'] = routing_gdf['geometry'].apply(geodesic_length)

    # Cutoff
    routing_gdf = routing_gdf[routing_gdf['length'] <= cutoff].copy()
    
    # Export
    os.makedirs(export_dir, exist_ok=True)
    excel_path = os.path.join(export_dir, f"Routing_Result.xlsx")
    parquet_path = os.path.join(export_dir, f"Routing_Result.parquet")
    
    routing_gdf.to_parquet(parquet_path, index=False)
    excel_styler(routing_gdf).drop(columns='geometry').to_excel(excel_path, sheet_name='DEN Graphhopper Routing', index=False)

if __name__ == "__main__":
    # PROCESS ROUTING
    source_path = r"D:\JACOBS\PROJECT\TASK\2026\FEB\W2\DISTANCE ODC TSEL\Template Routing.xlsx"
    target_path = r"D:\JACOBS\PROJECT\TASK\2026\FEB\W2\DISTANCE ODC TSEL\ODC Telkom Sumatera.kmz"
    k_final = 1

    source_gdf = read_gdf(source_path)
    target_gdf = read_gdf(target_path, geom_type='point')
    target_gdf['site_id'] = target_gdf['name']
    
    if 'long' not in target_gdf.columns:
        target_gdf['long'] = target_gdf.geometry.x
        target_gdf['lat'] = target_gdf.geometry.y

    source_gdf = validate_longlat(source_gdf)
    target_gdf = validate_longlat(target_gdf)

    routing_gdf = grapphopper_knn(source_gdf, target_gdf, k_final=k_final, profile='car')

    # EXPORT DATA
    export_dir = fr"D:\JACOBS\PROJECT\TASK\2026\FEB\W2\DISTANCE ODC TSEL\Export"
    date_today = datetime.today().strftime("%Y-%m-%d")
    export_dir = os.path.join(export_dir, date_today)
    os.makedirs(export_dir, exist_ok=True)

    source_gdf.to_parquet(os.path.join(export_dir, "Source GDF.parquet"))
    target_gdf.to_parquet(os.path.join(export_dir, "Target GDF.parquet"))