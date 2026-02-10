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
from modules.table import sanitize_header

# ------------------------------------------------------
# LOGGER
# ------------------------------------------------------
from core.logger import create_logger
logger = create_logger(__file__)

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


# FIX ROUTE

@router.post("/nearest_point", tags=["Graphhopper"])
async def nearest_point(
    source_file: UploadFile = File(
        None, description="Source data to identify nearest point from. Defined as a sitelist"
    ),
    target_file: UploadFile = File(
        None, description="Target data to identify nearest point to. Defined as a hub"
    ),
    separator: Separator = Form(
        Separator.SEMICOLON, description="Separator for segment identify near end and far end."
    ),
):
    """
    Create Direct Routing based on **Graphhopper Services**.

    **Template Nearest Point**
    [🟢 Download Here](http://10.83.10.16:8000/download-template/graphhopper/Template_Routing.xlsx)

    **Note:**
    - Make sure the latitude and longitude is not reversed.
    """

    # Read Excel file
    if excel_file is None:
        return {"error": "Excel file is required."}

    date_today = datetime.now().strftime("%Y%m%d")
    fixroute_upload = os.path.join(UPLOAD_DIR, date_today, "Intersite", "Fix Route")
    os.makedirs(fixroute_upload, exist_ok=True)

    match route_preference:
        case RoutePreference.FIBER:
            graph_type = "full_weighted"
        case RoutePreference.ROAD:
            graph_type = "weighted_road"
        case RoutePreference.SURGE_763:
            graph_type = "surge_763"
        case _:
            graph_type = "route"

    logger.info(f"ℹ️ Separator         : {separator}")
    logger.info(f"ℹ️ Route Preference  : {route_preference}")

    try:
        # LOAD DATA
        with pd.ExcelFile(excel_file.file) as xls:
            fixroute_input = pd.read_excel(xls)
            gdf_ne, gdf_fe = validate_fixroute(fixroute_input)
    except Exception as e:
        return {"error": f"Failed to read Excel file: {str(e)}"}

    # SAVE DATA
    excel_path = os.path.join(
        fixroute_upload,
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_fixroute_{uuid4().hex}.xlsx",
    )
    fixroute_input.to_excel(excel_path, index=False)
    logger.info(f"📥 Temporary Excel data saved to: {excel_path}")

    try:
        data = {
            "template_path": excel_path,
            "spof_threshold": spof_threshold,
            "program": program,
            "sep": separator.value,
            "graph_type": graph_type,
        }
        data = dumps(data, default=str)
        celery_task = task_fixroute.apply_async(args=[data])

        return {
            "message": "Fix route fiberization task has been initiated.",
            "task_id": celery_task.id,
            "task_status_url": f"/tasks/status/{celery_task.id}",
        }
    except Exception as e:
        return {"error": f"Failed to process data: {str(e)}"}


# POLYGON BASED INTERSITE
@router.post("/polygon-intersite", tags=["Intersite"])
async def polygon_intersite(
    excel_file: UploadFile = File(
        None, description="Excel file containing sitelist and hubs sheet."
    ),
    polygon_file: UploadFile = File(
        None, description="Polygon file to process (.kmz, .kml, .parquet, .gpkg, etc)."
    ),
    spof_threshold: int = Form(3000, description="SPOF tolerance in meters."),
    program: Optional[str] = Form(
        "Fiberization", description="Program name if needed."
    ),
    separator: Separator = Form(
        Separator.SEMICOLON, description="Separator for segment identify near end and far end."
    ),
    route_preference: Optional[RoutePreference] = Form(
        RoutePreference.FIBER, description="Route preference for intersite design."
    ),
):
    """
    Create Intersite design **Polygon Based**.
    Excel file must be containing **'sitelist'** and **'hubs'** sheet.

    **Template Polygon Based**
    [🟢 Download Here](http://10.83.10.16:8000/download-template/Template_Polygon_Based.xlsx)

    **Sample Polygon**
    [🟢 Download Here](http://10.83.10.16:8000/download-template/Polygon_Sample.kmz)

    **Note:**
    - Make sure the latitude and longitude is not reversed.
    """

    # Read Excel file
    if excel_file is None:
        return {"error": "Excel file is required."}

    date_today = datetime.now().strftime("%Y%m%d")
    polygon_upload = os.path.join(UPLOAD_DIR, date_today, "Intersite", "Polygon Based")
    os.makedirs(polygon_upload, exist_ok=True)

    try:
        sitelist, hubs = validate_poligonize(excel_file.file)
    except Exception as e:
        return {"error": f"Failed to read Excel file: {str(e)}"}

    match route_preference:
        case RoutePreference.FIBER:
            graph_type = "full_weighted"
        case RoutePreference.ROAD:
            graph_type = "weighted_road"
        case RoutePreference.SURGE_763:
            graph_type = "surge_763"
        case _:
            graph_type = "route"

    logger.info(f"ℹ️ Separator         : {separator}")
    logger.info(f"ℹ️ Route Preference  : {route_preference}")

    try:
        suffix = os.path.splitext(polygon_file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_fiber:
            tmp_fiber.write(polygon_file.file.read())
            tmp_fiber_path = tmp_fiber.name

        if suffix in [".kml", ".kmz", ".gpkg", ".parquet", ".shp"]:
            polygon_gdf = read_gdf(tmp_fiber_path, geom_type="polygon")
            logger.info(f"📥 Reading polygon file: {polygon_file.filename}")
        else:
            return {
                "error": f"Unsupported polygon file format {suffix}. Supported formats are GPKG, Parquet, and Shapefile."
            }
    except Exception as e:
        return {"error": f"Failed to read polygon file: {str(e)}"}

    # SAVE DATA
    excel_path = os.path.join(
        polygon_upload,
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_template_{uuid4().hex}.xlsx",
    )
    polygon_path = os.path.join(
        polygon_upload,
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_polygon_{uuid4().hex}.parquet",
    )

    with pd.ExcelWriter(excel_path) as xls:
        hubs.to_excel(xls, sheet_name="hubs")
        sitelist.to_excel(xls, sheet_name="sitelist")

    polygon_gdf.to_parquet(polygon_path, index=False)
    logger.info(f"📥 Temporary Excel data saved to: {excel_path}")
    logger.info(f"📥 Temporary Polygon data saved to: {polygon_path}")

    try:
        data = {
            "excel_path": excel_path,
            "polygon_path": polygon_path,
            "spof_threshold": spof_threshold,
            "program": program,
            "sep": separator.value,
            "graph_type": graph_type,
        }
        data = dumps(data, default=str)
        celery_task = task_polygon_intersite.apply_async(args=[data])

        return {
            "message": "Polygon based intersite task has been initiated.",
            "task_id": celery_task.id,
            "task_status_url": f"/tasks/status/{celery_task.id}",
        }
    except Exception as e:
        return {"error": f"Failed to process data: {str(e)}"}


# TOPOLOGY BASED INTERSITE
@router.post("/topology-intersite", tags=["Intersite"])
async def topology_intersite(
    excel_file: UploadFile = File(None, description="Excel file containing sitelist."),
    topology_file: UploadFile = File(
        None, description="Topology file to process (.kmz, .kml, .parquet, .gpkg, etc)."
    ),
    spof_threshold: int = Form(3000, description="SPOF tolerance in meters."),
    distance_tolerance: int = Form(
        500,
        description="Distance tolerance in meters to identify point surrounding topology.",
    ),
    program: Optional[str] = Form(
        "Fiberization", description="Program name if needed."
    ),
    separator: Separator = Form(
        Separator.SEMICOLON, description="Separator for segment identify near end and far end."
    ),
    route_preference: Optional[RoutePreference] = Form(
        RoutePreference.FIBER, description="Route preference for intersite design."
    ),
):
    """
    Create Intersite design **Topology Based**.
    Excel file must be containing ['site_id', 'site_name','long', 'lat'].

    **Template Topology Based**
    [🟢 Download Here](http://10.83.10.16:8000/download-template/Template_Topology_Based.xlsx)

    **Sample Topology**
    [🟢 Download Here](http://10.83.10.16:8000/download-template/Topology_Sample.kmz)

    **Note:**
    - Make sure the latitude and longitude is not reversed.
    """

    match route_preference:
        case RoutePreference.FIBER:
            graph_type = "full_weighted"
        case RoutePreference.ROAD:
            graph_type = "weighted_road"
        case RoutePreference.SURGE_763:
            graph_type = "surge_763"
        case _:
            graph_type = "route"

    logger.info(f"ℹ️ Separator         : {separator}")
    logger.info(f"ℹ️ Route Preference  : {route_preference}")

    # Read Excel file
    if excel_file is None:
        return {"error": "Excel file is required."}

    date_today = datetime.now().strftime("%Y%m%d")
    topology_upload = os.path.join(
        UPLOAD_DIR, date_today, "Intersite", "Topology Based"
    )
    os.makedirs(topology_upload, exist_ok=True)

    try:
        sitelist = validate_topology(excel_file.file)
    except Exception as e:
        return {"error": f"Failed to read Excel file: {str(e)}"}

    try:
        suffix = os.path.splitext(topology_file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_fiber:
            tmp_fiber.write(topology_file.file.read())
            tmp_fiber_path = tmp_fiber.name

        if suffix in [".kml", ".kmz", ".gpkg", ".parquet", ".shp"]:
            topology_gdf = read_gdf(tmp_fiber_path, geom_type="line")
            logger.info(f"📥 Reading topology file: {topology_file.filename}")
        else:
            return {
                "error": f"Unsupported topology file format {suffix}. Supported formats are GPKG, Parquet, and Shapefile."
            }
    except Exception as e:
        return {"error": f"Failed to read topology file: {str(e)}"}

    # SAVE DATA
    excel_path = os.path.join(
        topology_upload,
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_template_{uuid4().hex}.xlsx",
    )
    topology_path = os.path.join(
        topology_upload,
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_topology_{uuid4().hex}.parquet",
    )

    with pd.ExcelWriter(excel_path) as xls:
        sitelist.to_excel(xls, sheet_name="sitelist")

    topology_gdf.to_parquet(topology_path, index=False)
    logger.info(f"📥 Temporary Excel data saved to: {excel_path}")
    logger.info(f"📥 Temporary Topology data saved to: {topology_path}")

    try:
        data = {
            "excel_path": excel_path,
            "topology_path": topology_path,
            "spof_threshold": spof_threshold,
            "distance_tolerance": distance_tolerance,
            "program": program,
            "sep": separator.value,
            "graph_type": graph_type,
        }
        data = dumps(data, default=str)
        celery_task = task_topology_intersite.apply_async(args=[data])

        return {
            "message": "Topology based intersite task has been initiated.",
            "task_id": celery_task.id,
            "task_status_url": f"/tasks/status/{celery_task.id}",
        }
    except Exception as e:
        return {"error": f"Failed to process data: {str(e)}"}


# ================
# GENERATE KMZ IPL
# ================
@router.post("/implementation-intersite", tags=["Intersite"])
async def implementation_intersite(
    design_file: UploadFile = File(None, description="Design file containing DEN intersite format (.kmz, .kml)."),
    program: Optional[str] = Form("Implementation", description="Program name if 'program' column not provided."),
    operator: Optional[Operator] = Form(Operator.IOH, description="Operator to generate implementation KMZ algorithm."),
    separator: Separator = Form(Separator.SEMICOLON, description="Separator for segment identify near end and far end."),
    device_in_site: DeviceType | None = Form(DeviceType.OTB, description="Device to place in site."),
    device_in_branch: DeviceType | None = Form(DeviceType.ODP, description="Device to place in branch."),
    ipl_route: Optional[IPLRoute] = Form(IPLRoute.EXISTING_FIBER, description="Route preference to identify as existing fiber.")
):
    """
    Create Intersite Implementation KMZ with BOQ Report.
    KMZ file must be containing ['Connection', 'Route', 'FO Hub', 'Site List'].

    **Input Design Sample**
    [🟢 Download Here](http://10.83.10.16:8000/download-template/BOQ_Design_Sample.kmz)
    """

    date_today = datetime.now().strftime("%Y%m%d")
    boq_upload = os.path.join(UPLOAD_DIR, date_today, "Intersite", "BOQ")
    os.makedirs(boq_upload, exist_ok=True)

    if (device_in_branch != DeviceType.ODP) and (device_in_site != DeviceType.ODP):
        raise ValueError("🔴 ODP must be enabled, either in branch or in site.")

    device_in_site = None if device_in_site == DeviceType.NONE else DeviceType(device_in_site)
    device_in_branch = None if device_in_branch == DeviceType.NONE else DeviceType(device_in_branch)

    try:
        suffix = os.path.splitext(design_file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_fiber:
            tmp_fiber.write(design_file.file.read())
            tmp_fiber_path = tmp_fiber.name

        if suffix in [".kml", ".kmz"]:
            point_kmz, line_kmz = validate_kmz_design(tmp_fiber_path, sep=separator.value)
        else:
            return {
                "error": f"Unsupported topology file format {suffix}. Supported formats are GPKG, Parquet, and Shapefile."
            }
    except Exception as e:
        return {"error": f"Failed to read topology file: {str(e)}"}

    # SAVE DATA
    points_path = os.path.join(boq_upload,f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_points_kmz_{uuid4().hex}.parquet",)
    lines_path = os.path.join(boq_upload,f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_lines_kmz_{uuid4().hex}.parquet",)

    point_kmz.to_parquet(points_path, index=False)
    line_kmz.to_parquet(lines_path, index=False)
    logger.info(f"📥 Temporary Points data saved to   : {points_path}")
    logger.info(f"📥 Temporary Lines data saved to    : {lines_path}")

    try:
        data = {
            "points_path": points_path,
            "lines_path": lines_path,
            "program": program,
            "operator": operator,
            "ipl_route": ipl_route,
            "sep": separator.value,
            "device_in_site": device_in_site,
            "device_in_branch": device_in_branch,
        }

        data = dumps(data, default=str)
        celery_task = task_ipl.apply_async(args=[data])

        return {
            "message": "Implementation KMZ and BOQ Report task has been initiated.",
            "task_id": celery_task.id,
            "task_status_url": f"/tasks/status/{celery_task.id}",
        }
    except Exception as e:
        return {"error": f"Failed to process data: {str(e)}"}