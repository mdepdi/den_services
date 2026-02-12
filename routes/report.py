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
from modules.data import read_gdf
from modules.kml import read_kml, validate_kmz_design, validate_kmz_ipl

from service.intersite.report import drm_format, drm_xl
from service.intersite.boq_algorithm import boq_generation, boq_mmp

# EXPORT DIR
UPLOAD_DIR = settings.UPLOAD_DIR
EXPORT_DIR = settings.EXPORT_DIR
DATA_DIR = settings.DATA_DIR


class NewRingSchema(BaseModel):
    excel_file: UploadFile = File(
        None,
        description="Excel file containing ring data. Must be containing columns: 'site_id', 'site_name', 'site_type', 'lat', 'long', 'region', 'ring_name', 'flag'",
    )
    fiber_route: UploadFile = File(
        None, description="GPKG, Parquet, or Shapefile containing fiber route data."
    )
    method: str = Form(..., description="Method to use: 'supervised' or 'unsupervised'")


class InsertRingSchema(BaseModel):
    excel_file: UploadFile = (
        File(..., description="Excel file containing ring data to insert."),
    )
    previous_fiber: UploadFile = (
        File(
            ...,
            description="GPKG, Parquet, or Shapefile containing previous fiber data.",
        ),
    )
    previous_points: UploadFile = (
        File(
            ...,
            description="GPKG, Parquet, or Shapefile containing previous points data.",
        ),
    )
    max_member: int = Form(
        12, description="Maximum number of members to consider for insertion."
    )


class Operator(str, Enum):
    IOH = "ioh"
    XL = "xl"
    SURGE = "surge"
    TSEL = "tsel"

class Separator(str, Enum):
    SEMICOLON = ";"
    HYPHEN = "-"

class RoutePreference(str, Enum):
    FIBER = "existing_fiber"
    ROAD = "weighted_road"
    SHORTEST = "shortest_route"
    SURGE_763 = "surge_763"

class DeviceType(str, Enum):
    OTB = "OTB"
    ODP = "ODP"

class ConnectorType(str, Enum):
    SC = "SC"
    FC = "FC"

class IPLRoute(str, Enum):
    EXISTING_FIBER = "existing_fiber"
    SURGE_763 = "surge_763"

class BoQType(str, Enum):
    INTERSITE = "intersite"
    MMP = "mmp"


# ========
# ROUTER
# ========
router = APIRouter()

# ================
# GENERATE BOQ
# ================
# ===================
# GENERATE DRM FORMAT
# ===================
@router.post("/drm-intersite", tags=["Report"])
async def drm_intersite(
    design_file: UploadFile = File(None, description="Design file containing DEN intersite format (.kmz, .kml).",),
    operator: Operator = Form(Operator.XL, description="Operator to generate design report based on."),
    separator: Separator = Form(Separator.SEMICOLON, description="Separator for segment identify near end and far end."),
    project_name: str = Form(None, description="Project name to write in DRM Report."),
):
    """
    Create DRM Report based on Design KMZ.
    KMZ file must be containing ['Connection', 'Route', 'FO Hub', 'Site List'].

    **Input KMZ Design Sample**
    [🟢 Download Here](http://10.83.10.16:8000/download-template/BOQ_Design_Sample.kmz)
    """

    date_today = datetime.now().strftime("%Y%m%d")
    drm_dir = os.path.join(UPLOAD_DIR, date_today, "Intersite", "DRM")
    os.makedirs(drm_dir, exist_ok=True)


    suffix = os.path.splitext(design_file.filename)[1].lower()
    filename = os.path.splitext(design_file.filename)[0]

    if suffix not in [".kml", ".kmz"]:
        return {"error": f"Unsupported format: {suffix}"}

    kmz_path = os.path.join(
        drm_dir,
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}_{uuid4().hex}{suffix}",
    )
    with open(kmz_path, "wb") as buffer:
        shutil.copyfileobj(design_file.file, buffer)
        print(f"ℹ️ File copied into local storage.")

    extracted_design = validate_kmz_design(kmz_path, sep=separator.value)
    if extracted_design is not None:
        date_today = datetime.now().strftime("%Y%m%d")
        export_loc = f"{EXPORT_DIR}/Intersite/{date_today}/DRM/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}_{uuid4().hex}"
        os.makedirs(export_loc, exist_ok=True)

        try:
            start_time = time()
            print(f"ℹ️ DRM Format Task Started")
            match operator:
                case Operator.XL.value:
                    drm_xl(
                        kmz_path=kmz_path, 
                        export_dir=export_loc, 
                        sep=separator.value,
                        project_name=project_name
                    )
                case _:
                    drm_format(
                        kmz_path=kmz_path, 
                        export_dir=export_loc, 
                        sep=separator.value,
                        project_name=project_name
                    )
            end_time = time()
            excel_time = round((end_time - start_time) / 60, 2)
            print(f"ℹ️ Time Consumed:{excel_time:,} minutes")
            print(f"✅ All DRM Format Process Done.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to build DRM: {e}")

        try:
            out_base = f"DRM_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
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


@router.post("/boq-intersite", tags=["Report"])
async def boq_intersite_route(
    ipl_file: UploadFile = File(None, description="Implementation file containing DEN intersite format (.kmz, .kml).",),
    operator: Operator = Form(Operator.XL, description="Operator to generate BoQ report based on."),
    separator: Separator = Form(Separator.SEMICOLON, description="Separator for segment identify near end and far end."),
    program_name: Optional[str] =  Form("Intersite FO", description="Program name to write into BOQ"),
    interval_pole_m: Optional[int] = Form(80, description="Interval between pole in meters"),
    cable_percentage: Optional[int] = Form(10, description="Cable percentage (%) to calculate FO cable distance"),
    cable_multiplier: Optional[int] = Form(1, description="Multiplier for calculate FO cable distance"),
    device_in_site: Optional[DeviceType] = Form(DeviceType.OTB, description="Device to place in site, if BOQ is True."),
    device_in_branch: Optional[DeviceType] = Form(DeviceType.ODP, description="Device to place in branch, if BOQ is True."),
    sclc_enabled: Optional[bool] = Form(False, description="Set to True if SC LC enabled."),
    connector_in_site: Optional[ConnectorType] = Form(ConnectorType.SC, description="Connector to used in site"),
    connector_in_branch: Optional[ConnectorType] = Form(ConnectorType.SC, description="Connector to used in branch"),
):
    """
    Create BOQ Report based on Implementation KMZ.
    KMZ file must be containing ['Connection', 'Route', 'FO Hub', 'Site List', 'Route Backbone', 'Route Akses', 'Pole Eksisting', 'FO Existing', and so on].

    **Input KMZ Implementation Sample**
    [🟢 Download Here](http://10.83.10.16:8000/download-template/BOQ_Implementation_Sample.kmz)
    """

    date_today = datetime.now().strftime("%Y%m%d")
    boq_upload = os.path.join(UPLOAD_DIR, date_today, "Intersite", "BOQ")
    os.makedirs(boq_upload, exist_ok=True)


    suffix = os.path.splitext(ipl_file.filename)[1].lower()
    filename = os.path.splitext(ipl_file.filename)[0]

    if suffix not in [".kml", ".kmz"]:
        return {"error": f"Unsupported format: {suffix}"}

    kmz_path = os.path.join(boq_upload, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}_{uuid4().hex}{suffix}")
    with open(kmz_path, "wb") as buffer:
        shutil.copyfileobj(ipl_file.file, buffer)
        print(f"ℹ️ File copied into local storage.")

    extracted_ipl = validate_kmz_ipl(kmz_path, sep=separator.value)
    if extracted_ipl is not None:
        date_today = datetime.now().strftime("%Y%m%d")
        export_loc = f"{EXPORT_DIR}/Intersite/{date_today}/BOQ/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}_{uuid4().hex}"
        os.makedirs(export_loc, exist_ok=True)

        try:
            start_time = time()
            print(f"ℹ️ BOQ Generation Task Started")
            boq_generation(
                kmz_path=kmz_path, 
                export_dir=export_loc, 
                sep=separator.value, 
                operator=operator,  
                interval_pole_m = interval_pole_m,
                cable_percentage = cable_percentage,
                sclc_enabled = sclc_enabled,
                device_in_site = device_in_site,
                device_in_branch = device_in_branch,
                connector_in_site = connector_in_site,
                connector_in_branch = connector_in_branch,
                program_name = program_name
            )
            end_time = time()
            excel_time = round((end_time - start_time) / 60, 2)
            print(f"ℹ️ Time Consumed:{excel_time:,} minutes")
            print(f"✅ All BOQ Process Done.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to build BOQ: {e}")

        try:
            out_base = f"BOQ_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
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

@router.post("/boq-mmp", tags=["Report"])
async def boq_mmp_route(
    ipl_file: UploadFile = File(None, description="Implementation file containing DEN intersite format (.kmz, .kml).",),
    operator: Operator = Form(Operator.XL, description="Operator to generate implementation KMZ algorithm."),
    separator: Separator = Form(Separator.SEMICOLON, description="Separator for segment identify near end and far end."),
    program_name: Optional[str] =  Form("Intersite FO", description="Program name to write into BOQ"),
    interval_pole_m: Optional[int] = Form(60, description="Interval between pole in meters"),
    cable_percentage: Optional[int] = Form(15, description="Cable percentage (%) to calculate FO cable distance"),
    cable_multiplier: Optional[int] = Form(2, description="Multiplier for calculate FO cable distance"),
    device_in_site: Optional[DeviceType] = Form(DeviceType.ODP, description="Device to place in site."),
    device_in_branch: Optional[DeviceType] = Form(DeviceType.ODP, description="Device to place in branch."),
    connector_in_site: Optional[ConnectorType] = Form(ConnectorType.SC, description="Connector to used in site"),
    connector_in_branch: Optional[ConnectorType] = Form(ConnectorType.SC, description="Connector to used in branch"),
):
    """
    Create MMP BOQ Report based on Implementation KMZ.
    KMZ file must be containing ['Connection', 'Route', 'FO Hub', 'Site List', 'Route Backbone', 'Route Akses', 'Pole Eksisting', 'FO Existing', and so on].

    **Input KMZ Implementation Sample**
    [🟢 Download Here](http://10.83.10.16:8000/download-template/BOQ_Implementation_Sample.kmz)
    """

    date_today = datetime.now().strftime("%Y%m%d")
    boq_upload = os.path.join(UPLOAD_DIR, date_today, "Intersite", "BOQ")
    os.makedirs(boq_upload, exist_ok=True)


    suffix = os.path.splitext(ipl_file.filename)[1].lower()
    filename = os.path.splitext(ipl_file.filename)[0]

    if suffix not in [".kml", ".kmz"]:
        return {"error": f"Unsupported format: {suffix}"}

    kmz_path = os.path.join(
        boq_upload,
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}_{uuid4().hex}{suffix}",
    )
    with open(kmz_path, "wb") as buffer:
        shutil.copyfileobj(ipl_file.file, buffer)
        print(f"ℹ️ File copied into local storage.")

    extracted_ipl = validate_kmz_ipl(kmz_path, sep=separator.value)
    if extracted_ipl is not None:
        date_today = datetime.now().strftime("%Y%m%d")
        export_loc = f"{EXPORT_DIR}/Intersite/{date_today}/BOQ/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}_{uuid4().hex}"
        os.makedirs(export_loc, exist_ok=True)

        try:
            start_time = time()
            print(f"ℹ️ BOQ MMP Task Started")
            boq_mmp(
                kmz_path=kmz_path, 
                export_dir=export_loc, 
                sep=separator.value, 
                operator=operator,  
                interval_pole_m = interval_pole_m,
                cable_percentage = cable_percentage,
                cable_multiplier = cable_multiplier,
                device_in_site = device_in_site,
                device_in_branch = device_in_branch,
                connector_in_site = connector_in_site,
                connector_in_branch = connector_in_branch,
                program_name = program_name
            )
            end_time = time()
            excel_time = round((end_time - start_time) / 60, 2)
            print(f"ℹ️ Time Consumed:{excel_time:,} minutes")
            print(f"✅ All BOQ MMP Process Done.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to build BOQ: {e}")

        try:
            out_base = f"BOQ_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
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