import geopandas as gpd
import pandas as pd
import os
import tempfile
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from json import loads, dumps
from typing import List, Optional
from fastapi import UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.exceptions import HTTPException

from uuid import uuid4
from datetime import datetime
import zipfile

from core.config import settings
from modules.data import read_gdf
from service.utilization import main_fiber_utilization, main_identify_centerline, main_polygonize_ring

# EXPORT DIR
UPLOAD_DIR = settings.UPLOAD_DIR
EXPORT_DIR = settings.EXPORT_DIR
DATA_DIR = settings.DATA_DIR

# ========
# ROUTER
# ========
router = APIRouter()

# =============================
# FIBER UTILIZATION
# =============================
@router.post("/fiber_utilization", tags=["Utils"])
async def process_fiber_utilization(
    linestring_file: UploadFile = File(
        ..., description="GPKG, Parquet, or Shapefile containing line data."
    ),
    overlap: bool = Form(True, description="Define the expected route overlay.")
):
    """
    Identify fiber utilization based on DEN existing fiber route data.
    Returns a ZIP file with the results.
    """
    # Read linestring data
    print(f"🌏 Execute | Fiber Utilization")
    suffix = os.path.splitext(linestring_file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_linestring:
        tmp_linestring.write(linestring_file.file.read())
        tmp_linestring_path = tmp_linestring.name
    
    if suffix in ['.gpkg', '.parquet', '.shp']:
        linestring_gdf = read_gdf(tmp_linestring_path)
        print(f"📥 Reading linesetring file: {linestring_file.filename}")
    else:
        raise HTTPException(status_code=400, detail="Unsupported linesetring file format. Supported formats are GPKG, Parquet, and Shapefile.")
    
    for geom_type in linestring_gdf.geom_type:
        if geom_type not in ['Linestring', 'MultiLineString']:
            raise HTTPException(status_code=400, detail=f"Invalid file format {geom_type}")

    try:
        identified_fiber = main_fiber_utilization(linestring_gdf, overlap=overlap)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {e}")
    
    # --- save outputs ---
    job_id = uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = f"fiber_utilization_{job_id}_{timestamp}"

    result_dir = os.path.join(EXPORT_DIR, "Utilization", "Fiber Utilization")
    os.makedirs(result_dir, exist_ok=True)
    zip_path = f"{result_dir}/{out_base}.zip"

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # CSV
            csv_path = f"{result_dir}/{out_base}.csv"
            identified_fiber.drop(columns="geometry").to_csv(csv_path, index=False)
            zf.write(csv_path, arcname=os.path.basename(csv_path))
            os.remove(csv_path)

            # GPKG
            gpkg_path = f"{result_dir}/{out_base}.gpkg"
            identified_fiber.to_file(gpkg_path, driver="GPKG")
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

# =============================
# IDENTIFY CENTERLINE
# =============================
@router.post("/identify_centerline", tags=["Utils"])
async def process_identify_centerline(
    linestring_file: UploadFile = File(
        ..., description="GPKG, Parquet, or Shapefile containing line data."
    ),
    overlap: bool = Form(True, description="Define the expected route overlay.")
):
    """
    Identify centerline of linestring data.
    Returns a ZIP file with the results.
    """
    # Read linestring data
    print(f"🌏 Execute | Identify Centerline")
    suffix = os.path.splitext(linestring_file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_linestring:
        tmp_linestring.write(linestring_file.file.read())
        tmp_linestring_path = tmp_linestring.name
    
    if suffix in ['.gpkg', '.parquet', '.shp']:
        linestring_gdf = read_gdf(tmp_linestring_path)
        print(f"📥 Reading linesetring file: {linestring_file.filename}")
    else:
        raise HTTPException(status_code=400, detail="Unsupported linesetring file format. Supported formats are GPKG, Parquet, and Shapefile.")
    
    for geom_type in linestring_gdf.geom_type:
        if geom_type not in ['Linestring', 'MultiLineString']:
            raise HTTPException(status_code=400, detail=f"Invalid file format {geom_type}")

    try:
        identified_centerline = main_identify_centerline(linestring_gdf, overlap=overlap)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {e}")
    
    # --- save outputs ---
    job_id = uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = f"identify_centerline_{job_id}_{timestamp}"

    result_dir = os.path.join(EXPORT_DIR, "Utilization", "Identify Centerline")
    os.makedirs(result_dir, exist_ok=True)
    zip_path = f"{result_dir}/{out_base}.zip"

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # CSV
            csv_path = f"{result_dir}/{out_base}.csv"
            identified_centerline.drop(columns="geometry").to_csv(csv_path, index=False)
            zf.write(csv_path, arcname=os.path.basename(csv_path))
            os.remove(csv_path)

            # GPKG
            gpkg_path = f"{result_dir}/{out_base}.parquet"         
            identified_centerline.to_parquet(gpkg_path)
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

# =============================
# POLIGONIZE RING
# =============================
@router.post("/polygonize_ring", tags=["Utils"])
async def polygonize_ring(
    excel_file: UploadFile = File(..., description="Excel file containing sitelist and hubs."),
    polygon_file: UploadFile = File(..., description="GPKG, Parquet, or Shapefile containing line data."),
    project_name: str = Form(None, description="Project name to define the ring name.")
):
    """
    Polygonize sitelist and hubs into valid **Supervised Algorithm** ready data. 
    """
    # Read linestring data
    print(f"🌏 Execute | Polygonize Ring")
    suffix = os.path.splitext(excel_file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_excel:
        tmp_excel.write(excel_file.file.read())
        tmp_excel_path = tmp_excel.name

    suffix = os.path.splitext(polygon_file.filename)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_polygon:
        tmp_polygon.write(polygon_file.file.read())
        tmp_polygon_path = tmp_polygon.name
    
    if suffix in ['.gpkg', '.parquet', '.shp']:
        polygon_gdf = read_gdf(tmp_polygon_path)
        print(f"📥 Reading polygon file: {polygon_file.filename}")
    else:
        raise HTTPException(status_code=400, detail="Unsupported polygon file format. Supported formats are GPKG, Parquet, and Shapefile.")
    
    for geom_type in polygon_gdf.geom_type:
        if geom_type not in ['Polygon', 'MultiPolygon']:
            raise HTTPException(status_code=400, detail=f"Invalid file format {geom_type}")

    try:
        poligonized = main_polygonize_ring(tmp_excel_path, tmp_polygon_path, project_name=project_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {e}")
    
    # --- save outputs ---
    job_id = uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = f"Polygonized_Ring_{project_name}_{job_id}_{timestamp}"

    result_dir = os.path.join(EXPORT_DIR, "Utilization", "Polygonized Ring")
    os.makedirs(result_dir, exist_ok=True)
    zip_path = f"{result_dir}/{out_base}.zip"

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # EXCEL
            xlsx_path = f"{result_dir}/{out_base}.xlsx"
            poligonized.drop(columns="geometry").to_excel(xlsx_path, index=False)
            zf.write(xlsx_path, arcname=os.path.basename(xlsx_path))
            os.remove(xlsx_path)

            # GPKG
            gpkg_path = f"{result_dir}/{out_base}.parquet"
            poligonized.to_parquet(gpkg_path)
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
