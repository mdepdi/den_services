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
from modules.table import sanitize_header
from tasks.fwa_celery import task_fwa
from service.fwa_alghoritm import validate_fwa

# EXPORT DIR
UPLOAD_DIR = settings.UPLOAD_DIR
EXPORT_DIR = settings.EXPORT_DIR
DATA_DIR = settings.DATA_DIR

# ========
# ROUTER
# ========
router = APIRouter()

# ==========================
# FIXED WIRELESS ACCESS (FWA)
# ==========================
@router.post("/fwa-celery", tags=["Fixed Wireless Access (FWA)"])
async def fixed_wireless_access(
    excel_file: UploadFile = File(None, description="Excel file containing FWA sitelist."),
    distance_fwa: int = Form(500, description="Distance in meters"),
    sector_angle: int = Form(120, description="Sector angle expectation"),
    sector_group: int = Form(120, description="Group sector angle based on."),
    threshold: int = Form(100, description="Threshold homepass for a valid sites."),
    method: str = Form('maximize', description="Input value should be: 'maximize'/'unique'"),
    extend_sector: bool = Form(False, description="Extend the sector"),
):
    """
    Create **Fixed Wireless Access (FWA)** asessment.  
    Excel file must be containing columns: 
    - site_id 
    - lat
    - long

    **Template FWA**  
    [🟢 Download Here](http://10.83.10.16:8000/download-template/Template_FWA.xlsx)

    **Note:**
    - Make sure the latitude and longitude is not reversed.
    - Make sure the CRS is in EPSG:4326
    """

    # Read Excel file
    if excel_file is None:
        return {"error": "Excel file is required."}
    
    upload_fwa = os.path.join(UPLOAD_DIR, "Fixed_Wireless_Access")
    os.makedirs(upload_fwa, exist_ok=True)
    
    try:
        site_data = pd.read_excel(excel_file.file)
    except Exception as e:
        return {"error": f"Failed to read Excel file: {str(e)}"}

    if "site_id" in site_data.columns:
        site_data["site_id"] = site_data["site_id"].astype(str)
    if 'index_right' in site_data.columns:
        site_data = site_data.drop(columns=['index_right'])

    # VALIDATION
    site_data = sanitize_header(site_data, lowercase=True)
    site_data = validate_fwa(site_data)
    if method not in ['maximize', 'unique']:
        raise ValueError("Method is not valid. Please only use 'maximize' or 'unique'")
    
    sites_geom = gpd.points_from_xy(site_data['long'], site_data['lat'], crs='EPSG:4326')
    site_data = gpd.GeoDataFrame(site_data, geometry=sites_geom)

    # SAVE AS PARQUET
    temp_fwa_path = os.path.join(upload_fwa, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_site_data_{uuid4().hex}.parquet")
    site_data.to_parquet(temp_fwa_path, index=False)
    print(f"📥 Temporary site data saved to: {temp_fwa_path}")

    # Process data
    try:
        data = {
            "data_path": temp_fwa_path,
            "distance_fwa": distance_fwa,
            "sector_angle": sector_angle,
            "sector_group": sector_group,
            "threshold": threshold,
            "method": method,
            "extend_sector": extend_sector
        }
        data = dumps(data, default=str)
        print("🚀 Initiating Fixed Wireless Access (FWA)...")
        celery_task = task_fwa.apply_async(args=[data])
        print(f"✅ Task submitted with ID: {celery_task.id}")

        return {
            "message": "Fixed Wireless Access (FWA) has been initiated.",
            "task_id": celery_task.id,
            "task_status_url": f"/tasks/status/{celery_task.id}"
        }
    except Exception as e:
        HTTPException(e)