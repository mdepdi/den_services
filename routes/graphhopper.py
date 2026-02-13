import geopandas as gpd
import pandas as pd
import os
import tempfile
import zipfile
import shutil

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from json import loads, dumps
from typing import List, Optional
from fastapi import UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.exceptions import HTTPException
from enum import Enum

from uuid import uuid4
from datetime import datetime
from time import time

from core.config import settings
from modules.data import read_gdf, validate_longlat, get_unique_col
from modules.table import sanitize_header
from modules.utils import auto_group
from modules.celery import report_state
from modules.geometry import point_coordinates

from tasks.graphhopper_celery import task_nearest_point

# ------------------------------------------------------
# LOGGER
# ------------------------------------------------------
from core.logger import create_logger
logger = create_logger(__file__)


# DATA DIR
MAINDATA_DIR = settings.MAINDATA_DIR
DATA_DIR = settings.DATA_DIR

# EXPORT DIR
UPLOAD_DIR = settings.UPLOAD_DIR
EXPORT_DIR = settings.EXPORT_DIR
DATA_DIR = settings.DATA_DIR

class Operator(str, Enum):
    IOH = "ioh"
    XL = "xl"
    SURGE = "surge"
    TSEL = "tsel"

class Separator(str, Enum):
    SEMICOLON = ";"
    HYPHEN = "-"

# ========
# ROUTER
# ========
router = APIRouter()


# ENDPOINT GRAPHHOPPER
@router.post("/nearest_point", tags=["GraphHopper"])
async def nearest_point(
    source_file: UploadFile = File(None, description="Source data to identify nearest point from. Defined as a sitelist."),
    target_file: UploadFile = File(None, description="Target data to identify nearest point to. Defined as a hub."),
    separator: Separator = Form(Separator.SEMICOLON, description="Separator for segment identify near end and far end."),
    k_final: int = Form(1, description="Number of nearest target to find."),
    cutoff: int = Form(100000, description="Cutoff distance, default to 100000m (100km)."),
):
    """
    Create Direct Routing based on **Graphhopper Services**.

    **Template Nearest Point**  
    [🟢 Download Here](http://10.83.10.16:8000/template/graphhopper/Template_Graphhopper_Routing.xlsx)

    **Note:**   
    - Make sure the latitude and longitude is not reversed.
    """

    # Read Excel file
    if source_file is None:
        return {"error": "Excel file is required."}

    date_today = datetime.now().strftime("%Y%m%d")
    nearest_dir = os.path.join(UPLOAD_DIR, date_today, "Graphhopper", "Nearest Point")
    os.makedirs(nearest_dir, exist_ok=True)

    logger.info(f"🌏 Nearest Point")
    logger.info(f"ℹ️ Separator   : {separator}")
    logger.info(f"ℹ️ K Final     : {k_final} nearest target")
    logger.info(f"ℹ️ Cutoff      : {cutoff} m")

    # Verify Source
    try:
        suffix = os.path.splitext(source_file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_source:
            tmp_source.write(source_file.file.read())
            tmp_source_path = tmp_source.name

        if suffix in [".xlsx", ".csv", ".kml", ".kmz", ".gpkg", ".parquet", ".shp"]:
            source_gdf = read_gdf(tmp_source_path, geom_type="point")
            source_gdf['long'] = source_gdf.geometry.to_crs(epsg=4326).x
            source_gdf['lat'] = source_gdf.geometry.to_crs(epsg=4326).y
            source_gdf = sanitize_header(source_gdf, lowercase=True)
            source_gdf = validate_longlat(source_gdf)

            if "site_id" not in source_gdf.columns:
                unique_col = get_unique_col(source_gdf)
                if unique_col is None:
                    raise ValueError(f"No unique column found in source data")
        else:
            raise ValueError(f"Unsupported format {suffix}")
    except Exception as e:
        return {f"Source excel file: {str(e)}"}
    
    # Verify Target
    try:
        suffix = os.path.splitext(target_file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_target:
            tmp_target.write(target_file.file.read())
            tmp_target_path = tmp_target.name

        if suffix in [".xlsx", ".csv", ".kml", ".kmz", ".gpkg", ".parquet", ".shp"]:
            target_gdf = read_gdf(tmp_target_path, geom_type="point")
            target_gdf['long'] = target_gdf.geometry.to_crs(epsg=4326).x
            target_gdf['lat'] = target_gdf.geometry.to_crs(epsg=4326).y
            target_gdf = sanitize_header(target_gdf, lowercase=True)
            target_gdf = validate_longlat(target_gdf)

            if "site_id" not in target_gdf.columns:
                unique_col = get_unique_col(target_gdf)
                if unique_col is None:
                    raise ValueError(f"No unique column found in target data")
        else:
            raise ValueError(f"Unsupported format {suffix}")
    except Exception as e:
        return {f"Target excel file: {str(e)}"}

    # SAVE DATA
    source_filename = str(source_file.filename).split(".")[0]
    target_filename = str(target_file.filename).split(".")[0]
    source_path = os.path.join(nearest_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{source_filename}.xlsx",)
    target_path = os.path.join(nearest_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{target_filename}.xlsx",)
    source_gdf.to_excel(source_path, index=False)
    target_gdf.to_excel(target_path, index=False)
    
    logger.info(f"📥 Source excel file saved to: {source_path}")
    logger.info(f"📥 Target excel file saved to: {target_path}")

    try:
        data = {
            "source_path": source_path,
            "target_path": target_path,
            "k_final": k_final,
            "cutoff": cutoff,
            "sep": separator.value,
        }
        data = dumps(data, default=str)
        celery_task = task_nearest_point.apply_async(args=[data])

        return {
            "message": "Graphhopper Nearest Point task has been initiated.",
            "task_id": celery_task.id,
            "task_status_url": f"/tasks/status/{celery_task.id}",
        }
    except Exception as e:
        return {"error": f"Failed to process data: {str(e)}"}
    
# ENDPOINT NEAREST TO LINE
@router.post("/nearest_line", tags=["GraphHopper"])
async def nearest_line(
    source_file: UploadFile = File(None, description="Source data to identify nearest point from. Defined as a sitelist."),
    linestring_file: UploadFile = File(None, description="Linestring data to identify nearest point to. Defined as a hub."),
    separator: Separator = Form(Separator.SEMICOLON, description="Separator for segment identify near end and far end."),
    k_final: int = Form(1, description="Number of nearest target to find."),
    cutoff: int = Form(100000, description="Cutoff distance, default to 100000m (100km)."),
):
    """
    Create Direct Routing to Linestring based on **Graphhopper Services**.

    **Template Nearest Point**
    [🟢 Download Here](http://10.83.10.16:8000/template/graphhopper/Template_Graphhopper_Routing.xlsx)

    **Note:**
    - Make sure the latitude and longitude is not reversed.
    """

    # Read Excel file
    if source_file is None:
        return {"error": "Excel file is required."}

    date_today = datetime.now().strftime("%Y%m%d")
    nearest_dir = os.path.join(UPLOAD_DIR, date_today, "Graphhopper", "Nearest Point")
    os.makedirs(nearest_dir, exist_ok=True)

    logger.info(f"🌏 Nearest Point")
    logger.info(f"ℹ️ Separator   : {separator}")
    logger.info(f"ℹ️ K Final     : {k_final} nearest target")
    logger.info(f"ℹ️ Cutoff      : {cutoff} m")

    # Verify Source
    try:
        suffix = os.path.splitext(source_file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_source:
            tmp_source.write(source_file.file.read())
            tmp_source_path = tmp_source.name

        if suffix in [".xlsx", ".csv", ".kml", ".kmz", ".gpkg", ".parquet", ".shp"]:
            source_gdf = read_gdf(tmp_source_path, geom_type="point")
            source_gdf['long'] = source_gdf.geometry.to_crs(epsg=4326).x
            source_gdf['lat'] = source_gdf.geometry.to_crs(epsg=4326).y

            source_gdf = sanitize_header(source_gdf, lowercase=True)
            source_gdf = validate_longlat(source_gdf)

            if "site_id" not in source_gdf.columns:
                unique_col = get_unique_col(source_gdf)
                if unique_col is None:
                    raise ValueError(f"No unique column found in source data")
        else:
            raise ValueError(f"Unsupported format {suffix}")
    except Exception as e:
        return {f"Source excel file: {str(e)}"}
    
    # Verify Linestring
    if isinstance(linestring_file, UploadFile):
        try:
            suffix = os.path.splitext(linestring_file.filename)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_linestring:
                tmp_linestring.write(linestring_file.file.read())
                tmp_linestring_path = tmp_linestring.name

            if suffix in [".kml", ".kmz", ".gpkg", ".parquet", ".shp"]:
                linestring_gdf = read_gdf(tmp_linestring_path, geom_type="line")
                linestring_gdf = sanitize_header(linestring_gdf, lowercase=True)

                if "site_id" not in linestring_gdf.columns:
                    unique_col = get_unique_col(linestring_gdf)
                    if unique_col is None:
                        raise ValueError(f"No unique column found in linestring data")
            else:
                raise ValueError(f"Unsupported format {suffix}")
        
            # Linestring to Points
            points_fiber = point_coordinates(linestring_gdf)
            group = auto_group(points_fiber, distance=300)
            points_fiber = gpd.sjoin(points_fiber, group[['geometry', 'region']]).drop(columns='index_right')
            points_fiber = points_fiber.drop_duplicates(subset='region')
            points_fiber.columns = points_fiber.columns.str.lower()
            points_fiber = points_fiber.to_crs(epsg=4326)
            
            if 'name' in points_fiber.columns:
                points_fiber['site_id'] = points_fiber['name'] + "_" + str(points_fiber.index + 1)
            else:
                points_fiber['site_id'] = str(points_fiber.index + 1)
                
            points_fiber['long'] = points_fiber.geometry.x
            points_fiber['lat'] = points_fiber.geometry.y
        except Exception as e:
            return {f"Target excel file: {str(e)}"}
    else:
        fiber = fr"{MAINDATA_DIR}\06. FO TBG\Compile FO Route Only June 2025\FO TBG Only_01062025.parquet"
        dirname = os.path.dirname(fiber)
        basename = os.path.basename(fiber).split(".")[0]
        point_path = os.path.join(dirname, f"Points_{basename}.parquet")

        logger.info(f"🌏 Checking Route to Fiber TBG")
        if os.path.exists(point_path):
            logger.info(f"ℹ️ FO Points already exist. Load exist.")
            points_fiber = gpd.read_parquet(point_path)
        else:
            logger.info(f"ℹ️ FO Points didn't exist. Process point coordinates.")
            fiber = gpd.read_parquet(fiber)
            points_fiber = point_coordinates(fiber)
            points_fiber.columns = points_fiber.columns.str.lower()
            points_fiber = points_fiber.drop_duplicates(subset=['name', 'operator', 'geometry']).reset_index(drop=True)
            points_fiber = points_fiber.to_crs(epsg=4326)
            points_fiber['site_id'] = points_fiber['name'] + points_fiber['operator']
            points_fiber['long'] = points_fiber.geometry.x
            points_fiber['lat'] = points_fiber.geometry.y
            points_fiber.to_parquet(point_path)

    # CUTOFF DISTANCE
    buff_source = source_gdf.copy()
    buff_source = buff_source.to_crs(epsg=3857)
    points_fiber = points_fiber.to_crs(epsg=3857)
    buff_source["geometry"] = buff_source.geometry.buffer(cutoff)
    points_fiber = gpd.sjoin(points_fiber, buff_source[['geometry']]).drop(columns="index_right")

    # SAVE DATA
    source_filename = str(source_file.filename).split(".")[0]
    linestring_filename = str(linestring_file.filename).split(".")[0]
    source_path = os.path.join(nearest_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{source_filename}.xlsx",)
    linestring_path = os.path.join(nearest_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{linestring_filename}.xlsx",)
    source_gdf.to_excel(source_path, index=False)
    points_fiber.to_excel(linestring_path, index=False)
    
    logger.info(f"📥 Source excel file saved to: {source_path}")
    logger.info(f"📥 Target excel file saved to: {linestring_path}")
    try:
        data = {
            "source_path": source_path,
            "target_path": linestring_path,
            "k_final": k_final,
            "cutoff": cutoff,
            "sep": separator.value,
        }
        data = dumps(data, default=str)
        celery_task = task_nearest_point.apply_async(args=[data])

        return {
            "message": "Graphhopper Nearest Point task has been initiated.",
            "task_id": celery_task.id,
            "task_status_url": f"/tasks/status/{celery_task.id}",
        }
    except Exception as e:
        return {"error": f"Failed to process data: {str(e)}"}