import os
import sys
import numpy as np
import pickle
import simplekml
import geopandas as gpd
import networkx as nx
import pandas as pd
import datetime as dt
import zipfile
from tqdm import tqdm
from modules.data import read_gdf, get_unique_col
from modules.table import sanitize_header
from modules.utils import clutter_identification, admin_information

from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.append(root)

from core.config import settings

MAINDATA_DIR = settings.MAINDATA_DIR
poi_list = [
    f"{MAINDATA_DIR}/17. Point of Interest/Categorized/20260213_POI Enterprise/POI Classified Enterprise.parquet",
    f"{MAINDATA_DIR}/17. Point of Interest/Categorized/20260213_POI Minimarket/POI Classified Minimarket.parquet",
    f"{MAINDATA_DIR}/17. Point of Interest/Raw/POI Scrap Traditional Store Compile.parquet",
    f"{MAINDATA_DIR}/17. Point of Interest/Raw/POI Scrap Compile FnB.parquet",
    f"{MAINDATA_DIR}/17. Point of Interest/Raw/Scrap Compile Others.parquet",
]

def poi_remarking(sitelist: gpd.GeoDataFrame|pd.DataFrame|str) -> gpd.GeoDataFrame:
    sitelist = read_gdf(sitelist)
    sitelist = sanitize_header(sitelist)

    poi_compiled_path = os.path.join(f"{MAINDATA_DIR}/17. Point of Interest", "Compiled_POI.parquet")
    if os.path.exists(poi_compiled_path):
        print(f"ℹ️ Compiled POI already exists. Load data...")
        poi_compiled = gpd.read_parquet(poi_compiled_path)
    else:
        print(f"ℹ️ Compiled POI not available, create new compiled base on poi_list")
        poi_compiled = []
        for poi in poi_list:
            basename = os.path.basename(poi)
            print(f"ℹ️ Processing {basename}")
            df = gpd.read_parquet(poi)
            if 'sub_categories' in df.columns:
                df['poi_category'] = df['sub_categories']
            elif 'categories' in df.columns:
                df['poi_category'] = df['categories']
            else:
                df['poi_category'] = df['subclass']

            if df.crs is None:
                df = df.set_crs(epsg=4326)
            else:
                df = df.to_crs(epsg=4326)
            poi_compiled.append(df)

        poi_compiled = pd.concat(poi_compiled)
        poi_compiled = gpd.GeoDataFrame(poi_compiled, geometry='geometry', crs="EPSG:4326")
        poi_compiled = poi_compiled.drop_duplicates(subset=['geometry'])
        poi_compiled.to_parquet(poi_compiled_path, index=False)

    sitelist = sitelist.to_crs(epsg=3857)
    poi_compiled = poi_compiled.to_crs(epsg=3857)

    # Admin Information
    sitelist = admin_information(sitelist)

    # Clutter Mapping and POI Distance
    if 'poi_distance' not in sitelist.columns:
        if 'clutter' not in sitelist.columns:
            if 'site_id' not in sitelist.columns:
                unique_col = get_unique_col(sitelist)
                sitelist['site_id'] = sitelist[unique_col]

            sitelist['site_id'] = sitelist['site_id'].astype(str)
            sitelist = clutter_identification(sitelist, buffer=2000)

        clutter_mapping = {
            "dense urban": 300,
            "urban": 450,
            "sub urban": 700,
            "rural": 1000
        }

        sitelist['clutter'] = sitelist['clutter'].fillna("rural")
        sitelist['poi_distance'] = sitelist['clutter'].str.lower().map(clutter_mapping).fillna(1000)
    
    # Process POI Distance
    sitelist['geometry'] = sitelist.buffer(sitelist['poi_distance'])
    sitelist['site_id'] = sitelist['site_id'].astype(str)

    # Looping Each Site
    for idx, row in tqdm(sitelist.iterrows(), total=len(sitelist), desc="Process POI Remarking"):
        site_id = row['site_id']
        buf_area = row['geometry']
        intersected = poi_compiled[poi_compiled.intersects(buf_area)].copy()
        
        if intersected.empty:
            continue
        intersected['poi_category'] = intersected['poi_category'].str.upper()
        intersected = intersected[intersected['poi_category'].str.fullmatch(r"[A-Z ]+")].copy()
        categories = sorted(intersected['poi_category'].unique().tolist())
        sitelist.at[idx, 'poi_categories'] = (",").join(categories)

    sitelist['site_id'] = sitelist['site_id'].astype(str)
    sitelist['geometry'] = sitelist.geometry.centroid
    print(f"🟢 POI Remarking done.")
    return sitelist

if __name__ == "__main__":
    sitelist = r"D:\JACOBS\PROJECT\TASK\2026\FEB\W2\REQUEST POI\20260212 - Reseptor LRS Nokia_POI.xlsx"
    export_dir = r"D:\JACOBS\PROJECT\TASK\2026\FEB\W2\REQUEST POI\Export"
    os.makedirs(export_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(sitelist))[0]
    export_parquet = os.path.join(export_dir, f"Sitelist Remark_{basename}.parquet")
    export_xlsx = os.path.join(export_dir, f"Sitelist Remark_{basename}.xlsx")

    sitelist_remarked = poi_remarking(sitelist)
    sitelist_remarked.to_parquet(export_parquet, index=False)
    sitelist_remarked.to_excel(export_xlsx, index=False)