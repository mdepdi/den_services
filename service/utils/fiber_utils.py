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

from core.config import settings
from modules.utils import auto_group
from modules.geometry import geodesic_length
from modules.data import get_unique_col, read_gdf

MAINDATA_DIR = settings.MAINDATA_DIR
DATA_DIR = settings.DATA_DIR

# -----
# CLASS
# -----
class Separator(str, Enum):
    SEMICOLON = ";"
    HYPHEN = "-"

# ------------------------------------------------------
# LOGGER
# ------------------------------------------------------
from core.logger import create_logger
logger = create_logger(__file__)


# ------------------------------------------------------
# FIBER UTILIZATION
# ------------------------------------------------------
def clean_kmz(data_gdf: gpd.GeoDataFrame):
    remove_cols = ["description", "altitude", "alt_mode", "time_begin", "time_end", "time_when"]
    for col in remove_cols:
        if col in data_gdf.columns:
            data_gdf = data_gdf.drop(columns=col)
    data_gdf = data_gdf.dropna(axis=1)
    return data_gdf

def fiber_utilization(data_gdf: gpd.GeoDataFrame, target_fiber:gpd.GeoDataFrame | None = None, tolerance:int=20) -> gpd.GeoDataFrame:
    if target_fiber is None:
        target_fiber = gpd.read_parquet(f"{MAINDATA_DIR}/06. FO TBG/Compile TBG FO Route Only (22 Januari 2026)/Compile TBG FO Route Only (22 Januari 2026)-Add Unicom.parquet")
        target_fiber = target_fiber.to_crs(epsg=3857)
        data_gdf = data_gdf.to_crs(epsg=3857)

        target_fiber.columns = target_fiber.columns.str.lower()
        target_fiber = target_fiber[['name', 'remark', 'operator', 'geometry']]
        target_fiber = target_fiber.rename(columns={'name':'fiber'})
        target_fiber['geometry']  = target_fiber['geometry'].buffer(tolerance)
    else:
        target_fiber = target_fiber.copy()
        target_fiber = target_fiber.to_crs(epsg=3857)
        target_fiber['geometry']  = target_fiber['geometry'].buffer(tolerance)
        data_gdf = data_gdf.to_crs(epsg=3857)

        if 'name' in target_fiber.columns:
            target_fiber['fiber'] = target_fiber['name'].astype(str)

            # create incremental counter per name
            counts = target_fiber.groupby('fiber').cumcount()

            # only add suffix where duplicate (count > 0)
            mask = counts > 0
            target_fiber.loc[mask, 'fiber'] = (
                target_fiber.loc[mask, 'fiber']
                + "_"
                + counts[mask].astype(str)
            )
        else:
            target_fiber['fiber'] = (target_fiber.index + 1).astype(str)

    
    # ============
    # DATA GDF
    # ============
    data_gdf = data_gdf.reset_index(drop=True)
    data_gdf['num'] = data_gdf.index + 1

    if 'name' in data_gdf.columns:
        data_gdf['name'] = data_gdf['name'].astype(str)

        # create incremental counter per name
        counts = data_gdf.groupby('name').cumcount()

        # only add suffix where duplicate (count > 0)
        mask = counts > 0
        data_gdf.loc[mask, 'name'] = (data_gdf.loc[mask, 'name'] + "_" + counts[mask].astype(str))
    else:
        data_gdf['name'] = (data_gdf.index + 1).astype(str)

    # ========
    # FORCE 2D
    # ========
    data_gdf['geometry'] = data_gdf.geometry.force_2d()
    data_gdf['geometry'] = data_gdf.geometry.apply(lambda x: linemerge(x) if x.geom_type == "MultiLineString" else x)
    target_fiber['geometry'] = target_fiber.geometry.force_2d()

    data_gdf = data_gdf.to_crs(epsg=3857)
    target_fiber = target_fiber.to_crs(epsg=3857)

    # ============
    # JOIN REGION
    # ============
    group = auto_group(data_gdf)
    group = group.to_crs(epsg=3857)

    if "region" not in data_gdf.columns:
        data_gdf = gpd.sjoin(data_gdf, group[['region', 'geometry']], how="inner")

    existing = []
    new = []
    region_group = data_gdf.groupby('region')
    for region, region_data in region_group:
        if 'fo_note' in region_data.columns:
            region_data.drop(columns='fo_note')

        fo_intersects = gpd.overlay(region_data, target_fiber[['fiber', 'geometry']], how='intersection', keep_geom_type=False)
        fo_not_intersects = gpd.overlay(region_data, target_fiber[['fiber', 'geometry']], how='difference')

        if not fo_intersects.empty:
            fo_intersects['length'] = fo_intersects.geometry.to_crs(epsg=4326).apply(geodesic_length)
            fo_intersects = fo_intersects.sort_values('length', ascending=False)
            fo_intersects = fo_intersects.dissolve(by='name').reset_index()
            fo_intersects['length'] = fo_intersects.geometry.to_crs(epsg=4326).apply(geodesic_length)

        if not fo_not_intersects.empty:
            fo_not_intersects['length'] = fo_not_intersects.geometry.to_crs(epsg=4326).apply(geodesic_length)
            fo_not_intersects = fo_not_intersects.sort_values('length', ascending=False)
            fo_not_intersects = fo_not_intersects.dissolve(by='name').reset_index()
            fo_not_intersects['length'] = fo_not_intersects.geometry.to_crs(epsg=4326).apply(geodesic_length)

        existing.append(fo_intersects)
        new.append(fo_not_intersects)

    # Existing
    existing = pd.concat(existing, ignore_index=True)
    existing_gdf = gpd.GeoDataFrame(existing, geometry='geometry')
    existing_gdf['length'] = existing_gdf.geometry.to_crs(epsg=4326).apply(geodesic_length)
    existing_gdf['fo_note'] = "cable_existing"
    existing_gdf = existing_gdf.rename(columns={'length':'existing_length'})
    data_gdf = data_gdf.merge(existing_gdf[['name', 'existing_length']], on='name', how="left")

    
    # New Cable
    new = pd.concat(new, ignore_index=True)
    new_gdf = gpd.GeoDataFrame(new, geometry='geometry')
    new_gdf['length'] = new_gdf.geometry.to_crs(epsg=4326).apply(geodesic_length)
    new_gdf['fo_note'] = "cable_new"
    new_gdf = new_gdf.rename(columns={'length': 'new_length'})
    data_gdf = data_gdf.merge(new_gdf[['name', 'new_length']], on='name', how="left")

    compiled = pd.concat([existing_gdf, new_gdf])
    compiled = gpd.GeoDataFrame(compiled, geometry='geometry')
    logger.info(f"✅ Fiber Utilization done.")
    return compiled, data_gdf

if __name__ == "__main__":
    source_path = r"D:\JACOBS\PROJECT\TASK\2026\FEB\W3\FIBER UTILIZATION HUSEIN\Route Depok Dalam Border.parquet"
    target_path = r"D:\JACOBS\PROJECT\TASK\2026\FEB\W3\FIBER UTILIZATION HUSEIN\Unicom Route Only.kmz"

    source_gdf = read_gdf(source_path, geom_type="line")
    target_gdf = read_gdf(target_path, geom_type="line")
    source_gdf = clean_kmz(source_gdf)
    target_gdf = clean_kmz(target_gdf)

    filename = os.path.basename(source_path).split(".")[0]
    out_base = f"Fiber Utilization_{filename}"
    export_dir = r"D:\JACOBS\PROJECT\TASK\2026\FEB\W3\FIBER UTILIZATION HUSEIN"

    result_dir = os.path.join(export_dir, "Utils", "Fiber Utilization")
    os.makedirs(result_dir, exist_ok=True)
    zip_path = f"{result_dir}/{out_base}.zip"

    try:
        compiled_gdf, route_gdf = fiber_utilization(source_gdf, target_gdf, tolerance=20)
        print(compiled_gdf['fo_note'].value_counts())
        print(route_gdf.head())
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # XLSX
            xlsx_path = f"{result_dir}/{out_base}.xlsx"
            route_gdf.drop(columns="geometry").to_excel(xlsx_path, index=False)
            zf.write(xlsx_path, arcname=os.path.basename(xlsx_path))
            os.remove(xlsx_path)

            # PARQUET
            parquet_path = f"{result_dir}/{out_base}.parquet"
            compiled_gdf.to_parquet(parquet_path, index=False)
            zf.write(parquet_path, arcname=os.path.basename(parquet_path))
            os.remove(parquet_path)
    except Exception as e:
        raise ValueError(f"Erron in Fiber Utilziation: {e}")