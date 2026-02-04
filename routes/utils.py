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

from uuid import uuid4
from datetime import datetime
from time import time
from enum import Enum

from core.config import settings
from modules.data import read_gdf, read_df
from modules.kml import validate_kmz_design
from service.utils.get_homepass import get_homepass
from service.utils.intersite_utils import takeout_ring

# EXPORT DIR
UPLOAD_DIR = settings.UPLOAD_DIR
EXPORT_DIR = settings.EXPORT_DIR
DATA_DIR = settings.DATA_DIR

# ========
# ROUTER
# ========
router = APIRouter()

# =====
# CLASS
# =====
class Separator(str, Enum):
    SEMICOLON = ";"
    HYPHEN = "-"

# =============================
# GET HOMEPASS
# =============================
@router.post("/get_homepass", tags=["Utils"])
async def task_homepass(
    boundary_file: UploadFile = File(..., description="GPKG, Parquet, or Shapefile containing boundary data."),
    one_unit: bool = Form(False, description="One unit from road")
    ):
    """
    Get homepass data from MDE geospatial.
    Returns a ZIP file with the results.
    """
    # Read boundary data
    print(f"🌏 Execute | Get Homepass")
    filename = os.path.basename(boundary_file.filename).split(".")[0]
    suffix = os.path.splitext(boundary_file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_boundary:
        tmp_boundary.write(boundary_file.file.read())
        tmp_boundary_path = tmp_boundary.name
    
    if suffix in ['kmz', 'kml', '.gpkg', '.parquet', '.shp']:
        boundary_gdf = read_gdf(tmp_boundary_path, geom_type="polygon")
        print(f"📥 Reading boundary file: {boundary_file.filename}")
    else:
        raise HTTPException(status_code=400, detail="Unsupported boundary file format. Supported formats are KMZ, KML, GPKG, Parquet, and Shapefile.")
    
    for geom_type in boundary_gdf.geom_type:
        if geom_type not in ['Polygon', 'MultiPolygon']:
            raise HTTPException(status_code=400, detail=f"Invalid file format {geom_type}")

    try:
        homepass_gdf = get_homepass(boundary_gdf, one_unit=one_unit)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {e}")
    
    # --- save outputs ---
    job_id = uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = f"Homepass_{filename}"

    result_dir = os.path.join(EXPORT_DIR, "Utils", "Homepass")
    os.makedirs(result_dir, exist_ok=True)
    zip_path = f"{result_dir}/{out_base}.zip"

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # CSV
            csv_path = f"{result_dir}/{out_base}.csv"
            homepass_gdf.drop(columns="geometry").to_csv(csv_path, index=False)
            zf.write(csv_path, arcname=os.path.basename(csv_path))
            os.remove(csv_path)

            # GPKG
            gpkg_path = f"{result_dir}/{out_base}.gpkg"
            homepass_gdf.to_file(gpkg_path, driver="GPKG")
            zf.write(gpkg_path, arcname=os.path.basename(gpkg_path))
            os.remove(gpkg_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build ZIP: {e}")

    # --- FileResponse ---
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=os.path.basename(zip_path),
    )

# ===================
# GENERATE DRM FORMAT
# ===================
@router.post("/intersite-takeout-ring", tags=["Utils"])
async def intersite_takeout_ring(
    design_file: UploadFile = File(None, description="Design file containing DEN intersite format (.kmz, .kml).",),
    ringlist_file: UploadFile = File(None, description="Ring List contain 'ring_name' column to takeout from design file.",),
    separator: Separator = Form(Separator.SEMICOLON, description="Separator for segment identify near end and far end."),
):
    """
    Create DRM Report based on Design KMZ.
    KMZ file must be containing ['Connection', 'Route', 'FO Hub', 'Site List'].

    **Input KMZ Design Sample**
    [🟢 Download Here](http://10.83.10.16:8000/download-template/BOQ_Design_Sample.kmz)
    """

    date_today = datetime.now().strftime("%Y%m%d")
    task_dir = os.path.join(UPLOAD_DIR, date_today, "Utils", "Takeout Ring")
    os.makedirs(task_dir, exist_ok=True)


    suffix = os.path.splitext(design_file.filename)[1].lower()
    filename = os.path.splitext(design_file.filename)[0]

    if suffix not in [".kml", ".kmz"]:
        return {"error": f"Unsupported format: {suffix}"}

    kmz_path = os.path.join(
        task_dir,
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}_{uuid4().hex}{suffix}",
    )

    with open(kmz_path, "wb") as buffer:
        shutil.copyfileobj(design_file.file, buffer)
        print(f"ℹ️ File copied into local storage.")

    # Process Ring List
    ring_list = read_df(ringlist_file)
    ring_list = set(ring_list['ring_name'].astype(str))

    date_today = datetime.now().strftime("%Y%m%d")
    export_loc = f"{EXPORT_DIR}/Utils/Takeout Ring/{date_today}"
    os.makedirs(export_loc, exist_ok=True)

    try:
        start_time = time.time()
        print(f"ℹ️ Takeout Ring Design Task Started")
        takeout_ring(
            kmz_path=kmz_path,
            ring_list=ring_list,
            export_dir=export_loc, 
            sep=separator,
        )
        end_time = time.time()
        excel_time = round((end_time - start_time) / 60, 2)
        print(f"ℹ️ Time Consumed:{excel_time:,} minutes")
        print(f"✅ All Takeout Ring Design Process Done.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build BOQ: {e}")

    try:
        out_base = f"Takeout Ring_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
        zip_path = os.path.join(export_loc, f"{out_base}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(export_loc):
                for export_file in files:
                    if export_file.endswith(".zip") or "Checkpoint" in export_file:
                        continue
                    export_file_path = os.path.join(root, export_file)
                    arcname = os.path.relpath(export_file_path, export_loc)
                    zipf.write(export_file_path, arcname)
        print(f"📦 Result files zipped.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build ZIP: {e}")

    # --- FileResponse ---
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=os.path.basename(zip_path),
    )