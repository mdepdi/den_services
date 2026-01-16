import geopandas as gpd
import pandas as pd
import os
import tempfile
import zipfile
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

from core.config import settings
from modules.data import read_gdf
from modules.kml import read_kml, validate_kmz_design
from modules.table import sanitize_header

from service.intersite.ring_algorithm import supervised_validation
from service.intersite.clustering_algorithm import unsupervised_validation
from service.intersite.fixroute_algorithm import validate_fixroute
from service.intersite.topology_algorithm import validate_topology
from service.intersite.poligonized_algorithm import validate_poligonize
from service.intersite.insert_algorithm import validate_insert
from tasks.intersite_celery import task_insertring, task_supervised, task_unsupervised, task_fixroute, task_polygon_intersite, task_topology_intersite, task_boq

# EXPORT DIR
UPLOAD_DIR = settings.UPLOAD_DIR
EXPORT_DIR = settings.EXPORT_DIR
DATA_DIR = settings.DATA_DIR

class NewRingSchema(BaseModel):
    excel_file: UploadFile = File(
        None, description="Excel file containing ring data. Must be containing columns: 'site_id', 'site_name', 'site_type', 'lat', 'long', 'region', 'ring_name', 'flag'"
    )
    fiber_route: UploadFile = File(
        None, description="GPKG, Parquet, or Shapefile containing fiber route data."
    )
    method: str = Form(..., description="Method to use: 'supervised' or 'unsupervised'")

class InsertRingSchema(BaseModel):
    excel_file: UploadFile = File(
        ..., description="Excel file containing ring data to insert."
    ),
    previous_fiber: UploadFile = File(
        ..., description="GPKG, Parquet, or Shapefile containing previous fiber data."
    ),
    previous_points: UploadFile = File(
        ..., description="GPKG, Parquet, or Shapefile containing previous points data."
    ),
    max_member: int = Form(12, description="Maximum number of members to consider for insertion.")

class Operator(str, Enum):
    IOH = "ioh"
    XL = "xl"
    SURGE = "surge"
    TSEL = "tsel"

class RoutePreference(str, Enum):
    FIBER = "existing_fiber"
    ROAD = "weigthed_road"
    SHORTEST = "shortest_route"

# ========
# ROUTER
# ========
router = APIRouter()

# =============================
# CHECK TASK STATUS
# =============================
@router.get("/tasks/status/{task_id}", tags=["Intersite"])
async def fiberization_task(task_id: str):
    """
    Checking Celery background task job status.  
    Paste the celery task id in here.
    """
    from celery.result import AsyncResult
    from celery_app import celery_app
    from fastapi.responses import FileResponse
    from fastapi.exceptions import HTTPException
    from pathlib import Path

    task_result = AsyncResult(task_id, app=celery_app)

    if task_result.state == 'PENDING':
        return {"task_id": task_id, "status": "PENDING", "info": "Task is pending."}

    elif task_result.state == 'PROGRESS':
        return {"task_id": task_id, "status": "PROGRESS", "info": task_result.info}

    elif task_result.state == 'SUCCESS':
        result = task_result.result
        if isinstance(result, dict) and "zip_file" in result:
            zip_file = Path(result["zip_file"])
            if not zip_file.is_file():
                raise HTTPException(status_code=404, detail="File not found")
            resp = FileResponse(zip_file, filename=zip_file.name)
            resp.headers["Cache-Control"] = "public, max-age=3600"
            return resp
        else:
            return {"task_id": task_id, "status": "SUCCESS", "result": result}

    elif task_result.state == 'FAILURE':
        return {"task_id": task_id, "status": "FAILURE", "error": str(task_result.result)}

    else:
        return {"task_id": task_id, "status": task_result.state, "info": str(task_result.info)}

# =============================
# INSERT RING
# =============================
@router.post("/insert_ring", tags=["Intersite"])
async def insert_ring(
    insert_list: UploadFile = File(..., description="Excel file containing potential sitelist to insert."),
    kmz_design: UploadFile = File(..., description="KMZ file containing existing design plan."),
    max_member: int = Form(12, description="Maximum number of members to consider for insertion."),
    max_distance: int = Form(3000, description="Maximum distance consider for insertion."),
    operator: Optional[Operator] = Form(Operator.IOH, description="Operator to define separator of near end far end from 'Route' folders."),
):
    """
    Create Intersite design based on **Insert Alghorithm**.  

    **Template Insert Ring**  
    [🟢 Download Here](http://10.83.10.16:8000/download-template/Template_Insert_Ring.xlsx)

    **Note:**
    - KMZ Data should be formatted as DEN intersite design rules.
    - Make sure the latitude and longitude is not reversed.

    Separator:
        - IOH Operator  : Separator will be '-'
        - XL Operator   : Separator will be ';'
    """
    date_today = datetime.now().strftime("%Y%m%d")
    upload_dir = os.path.join(UPLOAD_DIR, date_today, "Intersite", "Insert Ring")
    os.makedirs(upload_dir, exist_ok=True)

    # Operator
    match operator:
        case "xl":
            sep = ";"
        case _:
            sep = "-"
    
    print(f"ℹ️ Operator  : {operator}")
    print(f"ℹ️ Separator : {sep}")


    kmz_suffix = os.path.splitext(kmz_design.filename)[1].lower()
    if kmz_suffix not in [".kmz", ".kml"]:
        return {"error": "KMZ/KML only is supported for design plan."}

    kmz_path = os.path.join(
        upload_dir,
        f"{uuid4().hex}_design_{datetime.now().strftime('%H%M%S')}{kmz_suffix}"
    )

    with open(kmz_path, "wb") as f:
        f.write(await kmz_design.read())
    print(f"📥 Saved KMZ file → {kmz_path}")

    excel_suffix = os.path.splitext(insert_list.filename)[1].lower()
    if excel_suffix not in [".xlsx", ".xls", ".csv"]:
        return {"error": "Insert list must be Excel or CSV."}

    insert_path = os.path.join(
        upload_dir,
        f"{uuid4().hex}_insert_{datetime.now().strftime('%H%M%S')}{excel_suffix}"
    )
    with open(insert_path, "wb") as f:
        f.write(await insert_list.read())
    print(f"📥 Saved Insert List → {insert_path}")
    _, _, _ = validate_insert(insert_path, kmz_path)

    # Params
    params = dumps({
        "insert_list_path": insert_path,
        "kmz_path": kmz_path,
        "max_member": max_member,
        "max_distance": max_distance,
        "operator": operator,
        "sep": sep
    })

    celery_task = task_insertring.apply_async(args=[params])
    print(f"✅ Insert Task submitted with ID: {celery_task.id}")

    return {
        "message": "Insert ring task started!",
        "task_id": celery_task.id,
        "task_status_url": f"/tasks/status/{celery_task.id}"
    }

# =============================
# NEW RING
# =============================
# SUPERVISED
@router.post("/supervised", tags=["Intersite"])
async def supervised_ring(
    excel_file: UploadFile = File(None, description="Excel file containing ring data."),
    spof_threshold: int = Form(3000, description="SPOF tolerance in meters."),
    program: str = Form("Fiberization", description="Program name if needed."),
    boq:bool = Form(False, description="Output file to choose"),
    operator: Optional[Operator] = Form(Operator.IOH, description="Operator to define separator of near end far end from 'Route' folders."),
    route_preference: Optional[RoutePreference] = Form(RoutePreference.FIBER, description="Route preference for intersite design."),
):
    """
    Create Intersite design based on **Supervised Alghorithm**, you need to define the cluster first.  
    Excel file must be containing columns: 
    - site_id 
    - site_name
    - site_type
    - lat
    - long
    - region
    - ring_name
    - flag

    **Template Supervised Fiberization**  
    [🟢 Download Here](http://10.83.10.16:8000/download-template/Template_Supervised_Fiberization.xlsx)

    **Note:**
    - Site type should containing 'FO Hub' for interconnection source.
    - Each ring name must be on the same region.
    - Flag define the start hub or end hub.
    - Make sure the latitude and longitude is not reversed.

    Separator:
        - IOH Operator  : Separator will be '-'
        - XL Operator   : Separator will be ';'
    """

    # Read Excel file
    if excel_file is None:
        return {"error": "Excel file is required."}

    # Process data
    match operator:
        case "xl":
            sep = ";"
        case _:
            sep = "-"

    match route_preference:
        case RoutePreference.FIBER:
            graph_type = "full_weighted"
        case RoutePreference.ROAD:
            graph_type = "weighted_road"
        case _:
            graph_type = "route"
    
    print(f"ℹ️ Operator          : {operator}")
    print(f"ℹ️ Separator         : {sep}")
    print(f"ℹ️ Route Preference  : {route_preference}")
    
    date_today = datetime.now().strftime("%Y%m%d")
    supervised_upload = os.path.join(UPLOAD_DIR, date_today, "Intersite", "Supervised")
    os.makedirs(supervised_upload, exist_ok=True)
    
    try:
        site_data = pd.read_excel(excel_file.file)
        site_data = sanitize_header(site_data)
        site_data = supervised_validation(site_data)
    except Exception as e:
        return {"error": f"Failed to read Excel file: {str(e)}"}

    if "site_id" in site_data.columns:
        site_data["site_id"] = site_data["site_id"].astype(str)
    if 'index_right' in site_data.columns:
        site_data = site_data.drop(columns=['index_right'])

    # SAVE AS PARQUET
    temp_parquet_path = os.path.join(supervised_upload, f"{datetime.now().strftime('%H%M%S')}_site_data_{uuid4().hex}.parquet")
    site_data.to_parquet(temp_parquet_path, index=False)
    print(f"📥 Temporary site data saved to: {temp_parquet_path}")

    try:
        data = {
            "site_path": temp_parquet_path,
            "spof_threshold": spof_threshold,
            "program": program,
            "boq": boq,
            "sep": sep,
            "graph_type": graph_type
        }
        data = dumps(data, default=str)
        celery_task = task_supervised.apply_async(args=[data])
        print(f"✅ Supervised Task submitted with ID: {celery_task.id}")

        return {
            "message": "Supervised fiberization task has been initiated.",
            "task_id": celery_task.id,
            "task_status_url": f"/tasks/status/{celery_task.id}"
        }
    except Exception as e:
        return {"error": f"Failed to process data: {str(e)}"}
    
# UNSUPERVISED
@router.post("/unsupervised", tags=["Intersite"])
async def unsupervised_ring(
    excel_file: UploadFile = File(None, description="Excel file containing sitelist and hubs sheet."),
    member_expectation: int = Form(10, description="Member expectation in one ring."),
    max_distance: int = Form(10000, description="Maximum distance to route."),
    spof_threshold: int = Form(3000, description="SPOF tolerance in meters."),
    program: str = Form("Fiberization", description="Program name if needed."),
    drop_existings:bool = Form(False, description="Drop ring if not conatining new site."),
    boq:bool = Form(False, description="Output file to choose"),
    operator: Optional[Operator] = Form(Operator.IOH, description="Operator to define separator of near end far end from 'Route' folders."),
    route_preference: Optional[RoutePreference] = Form(RoutePreference.FIBER, description="Route preference for intersite design."),
):
    """
    Create Intersite design based on **Unsupervised Alghorithm**, the clustering based on our service.  
    Excel file must be containing **'sitelist'** and **'hubs'** sheet.

    **Template Unsupervised Fiberization**  
    [🟢 Download Here](http://10.83.10.16:8000/download-template/Template_Unsupervised_Fiberization.xlsx)

    **Note:**
    - Hubs should containing 'FO Hub' for interconnection source.
    - Each ring name must be on the same region.
    - Make sure the latitude and longitude is not reversed.

    Separator:
        - IOH Operator  : Separator will be '-'
        - XL Operator   : Separator will be ';'
    """

    # Read Excel file
    if excel_file is None:
        return {"error": "Excel file is required."}

    date_today = datetime.now().strftime("%Y%m%d")
    unsupervised_upload = os.path.join(UPLOAD_DIR, date_today, "Intersite", "Unsupervised")
    os.makedirs(unsupervised_upload, exist_ok=True)
    
    # Process data
    match operator:
        case "xl":
            sep = ";"
        case _:
            sep = "-"

    match route_preference:
        case RoutePreference.FIBER:
            graph_type = "full_weighted"
        case RoutePreference.ROAD:
            graph_type = "weighted_road"
        case _:
            graph_type = "route"
    
    print(f"ℹ️ Operator          : {operator}")
    print(f"ℹ️ Separator         : {sep}")
    print(f"ℹ️ Route Preference  : {route_preference}")

    try:
    # LOAD DATA
        with pd.ExcelFile(excel_file.file) as xls:
            used_sheets = ['sitelist', 'hubs']
            sheet_names = xls.sheet_names
            for sheet in sheet_names:
                if sheet not in used_sheets:
                    raise ValueError(f"Unexpected sheet name '{sheet}' found in the Excel file.")
                
            nr_sites = pd.read_excel(xls, 'sitelist')
            nr_hubs = pd.read_excel(xls, 'hubs')
            nr_sites, nr_hubs = unsupervised_validation(nr_sites, nr_hubs)
    except Exception as e:
        return {"error": f"Failed to read Excel file: {str(e)}"}
    
    site_data = nr_sites
    hubs_data = nr_hubs

    if "site_id" in site_data.columns:
        site_data["site_id"] = site_data["site_id"].astype(str)
    if 'index_right' in site_data.columns:
        site_data = site_data.drop(columns=['index_right'])

    # SAVE AS PARQUET
    temp_parquet_path = os.path.join(unsupervised_upload, f"{datetime.now().strftime('%H%M%S')}_site_data_{uuid4().hex}.parquet")
    temp_hub_path = os.path.join(unsupervised_upload, f"{datetime.now().strftime('%H%M%S')}_hub_data_{uuid4().hex}.parquet")
    site_data.to_parquet(temp_parquet_path, index=False)
    hubs_data.to_parquet(temp_hub_path, index=False)
    print(f"📥 Temporary site data saved to : {temp_parquet_path}")
    print(f"📥 Temporary hub data saved to  : {temp_hub_path}")

    try:
        data = {
            "site_path": temp_parquet_path,
            "hub_path": temp_hub_path,
            "member_expectation": member_expectation,
            "max_distance": max_distance,
            "drop_existings": drop_existings,
            "program": program,
            "spof_threshold": spof_threshold,
            "boq": boq,
            "sep": sep,
            "graph_type": graph_type
        }
        data = dumps(data, default=str)
        celery_task = task_unsupervised.apply_async(args=[data])
        print(f"✅ Unsupervised Task submitted with ID: {celery_task.id}")

        return {
            "message": "Unsupervised fiberization task has been initiated.",
            "task_id": celery_task.id,
            "task_status_url": f"/tasks/status/{celery_task.id}"
        }
    except Exception as e:
        return {"error": f"Failed to process data: {str(e)}"}
    
# FIX ROUTE
@router.post("/fixroute", tags=["Intersite"])
async def fixroute_ring(
    excel_file: UploadFile = File(None, description="Excel file containing fix route template."),
    spof_threshold: int = Form(3000, description="SPOF tolerance in meters."),
    program: Optional[str] = Form(None, description="Program name if not defined"),
    boq: Optional[bool] = Form(False, description="Output file to choose"),
    operator: Optional[Operator] = Form(Operator.IOH, description="Operator to define separator of near end far end from 'Route' folders."),
    route_preference: Optional[RoutePreference] = Form(RoutePreference.FIBER, description="Route preference for intersite design."),
):
    """
    Create Intersite design based on **Fix Route Alghorithm**.  
    Excel file must be containing **Near End (NE)** as source and **Far End (FE)** as target.  

    **Template Unsupervised Fiberization**  
    [🟢 Download Here](http://10.83.10.16:8000/download-template/Template_Fixed_Route.xlsx)

    **Note:**
    - Fix Route will running based on region and ring name. Make sure to order the ring from start hub to end hub.  
    - Each ring should containing 'FO Hub' for interconnection source.  
    - Each ring name must be on the same region.  
    - Make sure the latitude and longitude is not reversed.  
    """

    # Read Excel file
    if excel_file is None:
        return {"error": "Excel file is required."}

    date_today = datetime.now().strftime("%Y%m%d")
    fixroute_upload = os.path.join(UPLOAD_DIR, date_today, "Intersite", "Fix Route")
    os.makedirs(fixroute_upload, exist_ok=True)
    
    # Process data
    match operator:
        case "xl":
            sep = ";"
        case _:
            sep = "-"

    match route_preference:
        case RoutePreference.FIBER:
            graph_type = "full_weighted"
        case RoutePreference.ROAD:
            graph_type = "weighted_road"
        case _:
            graph_type = "route"
    
    print(f"ℹ️ Operator          : {operator}")
    print(f"ℹ️ Separator         : {sep}")
    print(f"ℹ️ Route Preference  : {route_preference}")

    try:
    # LOAD DATA
        with pd.ExcelFile(excel_file.file) as xls:
            fixroute_input = pd.read_excel(xls)
            gdf_ne, gdf_fe = validate_fixroute(fixroute_input)            
    except Exception as e:
        return {"error": f"Failed to read Excel file: {str(e)}"}


    # SAVE DATA
    excel_path = os.path.join(fixroute_upload, f"{datetime.now().strftime('%H%M%S')}_fixroute_{uuid4().hex}.xlsx")
    fixroute_input.to_excel(excel_path, index=False)
    print(f"📥 Temporary Excel data saved to: {excel_path}")

    try:
        data = {
            "template_path": excel_path,
            "spof_threshold": spof_threshold,
            "program": program,
            "boq": boq,
            "sep": sep,
            "graph_type": graph_type
        }
        data = dumps(data, default=str)
        celery_task = task_fixroute.apply_async(args=[data])

        return {
            "message": "Fix route fiberization task has been initiated.",
            "task_id": celery_task.id,
            "task_status_url": f"/tasks/status/{celery_task.id}"
        }
    except Exception as e:
        return {"error": f"Failed to process data: {str(e)}"}

# POLYGON BASED INTERSITE
@router.post("/polygon-intersite", tags=["Intersite"])
async def polygon_intersite(
    excel_file: UploadFile = File(None, description="Excel file containing sitelist and hubs sheet."),
    polygon_file: UploadFile = File(None, description="Polygon file to process (.kmz, .kml, .parquet, .gpkg, etc)."),
    spof_threshold: int = Form(3000, description="SPOF tolerance in meters."),
    program: Optional[str] = Form("Fiberization", description="Program name if needed."),
    boq: Optional[bool] = Form(False, description="Output file to choose"),
    operator: Optional[Operator] = Form(Operator.IOH, description="Operator to define separator of near end far end from 'Route' folders."),
    route_preference: Optional[RoutePreference] = Form(RoutePreference.FIBER, description="Route preference for intersite design."),
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

    Separator:
        - IOH Operator  : Separator will be '-'
        - XL Operator   : Separator will be ';'
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
    
    # Process data
    match operator:
        case "xl":
            sep = ";"
        case _:
            sep = "-"

    match route_preference:
        case RoutePreference.FIBER:
            graph_type = "full_weighted"
        case RoutePreference.ROAD:
            graph_type = "weighted_road"
        case _:
            graph_type = "route"
    
    print(f"ℹ️ Operator          : {operator}")
    print(f"ℹ️ Separator         : {sep}")
    print(f"ℹ️ Route Preference  : {route_preference}")

    try:
        suffix = os.path.splitext(polygon_file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_fiber:
            tmp_fiber.write(polygon_file.file.read())
            tmp_fiber_path = tmp_fiber.name
        
        if suffix in ['.kml', '.kmz','.gpkg', '.parquet', '.shp']:
            polygon_gdf = read_gdf(tmp_fiber_path, geom_type='polygon')
            print(f"📥 Reading polygon file: {polygon_file.filename}")
        else:
            return {"error": f"Unsupported polygon file format {suffix}. Supported formats are GPKG, Parquet, and Shapefile."}    
    except Exception as e:
        return {"error": f"Failed to read polygon file: {str(e)}"}

    # SAVE DATA
    excel_path = os.path.join(polygon_upload, f"{datetime.now().strftime('%H%M%S')}_template_{uuid4().hex}.xlsx")
    polygon_path = os.path.join(polygon_upload, f"{datetime.now().strftime('%H%M%S')}_polygon_{uuid4().hex}.parquet")
    
    with pd.ExcelWriter(excel_path) as xls:
        hubs.to_excel(xls, sheet_name='hubs')
        sitelist.to_excel(xls, sheet_name='sitelist')

    polygon_gdf.to_parquet(polygon_path, index=False)
    print(f"📥 Temporary Excel data saved to: {excel_path}")
    print(f"📥 Temporary Polygon data saved to: {polygon_path}")

    try:
        data = {
            "excel_path": excel_path,
            "polygon_path": polygon_path,
            "spof_threshold": spof_threshold,
            "program": program,
            "boq": boq,
            "sep": sep,
            "graph_type": graph_type
        }
        data = dumps(data, default=str)
        celery_task = task_polygon_intersite.apply_async(args=[data])

        return {
            "message": "Polygon based intersite task has been initiated.",
            "task_id": celery_task.id,
            "task_status_url": f"/tasks/status/{celery_task.id}"
        }
    except Exception as e:
        return {"error": f"Failed to process data: {str(e)}"}

# TOPOLOGY BASED INTERSITE
@router.post("/topology-intersite", tags=["Intersite"])
async def topology_intersite(
    excel_file: UploadFile = File(None, description="Excel file containing sitelist."),
    topology_file: UploadFile = File(None, description="Topology file to process (.kmz, .kml, .parquet, .gpkg, etc)."),
    spof_threshold: int = Form(3000, description="SPOF tolerance in meters."),
    program: Optional[str] = Form("Fiberization", description="Program name if needed."),
    boq:Optional[bool] = Form(False, description="Output file to choose"),
    operator: Optional[Operator] = Form(Operator.IOH, description="Operator to define separator of near end far end from 'Route' folders."),
    route_preference: Optional[RoutePreference] = Form(RoutePreference.FIBER, description="Route preference for intersite design."),
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
    
    Separator:
        - IOH Operator  : Separator will be '-'
        - XL Operator   : Separator will be ';'
    """

    # Process data
    match operator:
        case "xl":
            sep = ";"
        case _:
            sep = "-"

    match route_preference:
        case RoutePreference.FIBER:
            graph_type = "full_weighted"
        case RoutePreference.ROAD:
            graph_type = "weighted_road"
        case _:
            graph_type = "route"
    
    print(f"ℹ️ Operator          : {operator}")
    print(f"ℹ️ Separator         : {sep}")
    print(f"ℹ️ Route Preference  : {route_preference}")

    # Read Excel file
    if excel_file is None:
        return {"error": "Excel file is required."}

    date_today = datetime.now().strftime("%Y%m%d")
    topology_upload = os.path.join(UPLOAD_DIR, date_today, "Intersite", "Topology Based")
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
        
        if suffix in ['.kml', '.kmz','.gpkg', '.parquet', '.shp']:
            topology_gdf = read_gdf(tmp_fiber_path, geom_type='line')
            print(f"📥 Reading topology file: {topology_file.filename}")
        else:
            return {"error": f"Unsupported topology file format {suffix}. Supported formats are GPKG, Parquet, and Shapefile."}    
    except Exception as e:
        return {"error": f"Failed to read topology file: {str(e)}"}

    # SAVE DATA
    excel_path = os.path.join(topology_upload, f"{datetime.now().strftime('%H%M%S')}_template_{uuid4().hex}.xlsx")
    topology_path = os.path.join(topology_upload, f"{datetime.now().strftime('%H%M%S')}_topology_{uuid4().hex}.parquet")
    
    with pd.ExcelWriter(excel_path) as xls:
        sitelist.to_excel(xls, sheet_name='sitelist')

    topology_gdf.to_parquet(topology_path, index=False)
    print(f"📥 Temporary Excel data saved to: {excel_path}")
    print(f"📥 Temporary Topology data saved to: {topology_path}")


    try:
        data = {
            "excel_path": excel_path,
            "topology_path": topology_path,
            "spof_threshold": spof_threshold,
            "program": program,
            "boq": boq,
            "sep": sep,
            "graph_type": graph_type
        }
        data = dumps(data, default=str)
        celery_task = task_topology_intersite.apply_async(args=[data])

        return {
            "message": "Topology based intersite task has been initiated.",
            "task_id": celery_task.id,
            "task_status_url": f"/tasks/status/{celery_task.id}"
        }
    except Exception as e:
        return {"error": f"Failed to process data: {str(e)}"}


# ================
# GENERATE BOQ
# ================
@router.post("/boq", tags=["Intersite"])
async def boq_intersite(
    design_file: UploadFile = File(None, description="Design file containing DEN intersite format (.kmz, .kml)."),
    program: Optional[str] = Form("BOQ", description="Program name if 'program' column not provided."),
    operator: Optional[Operator] = Form(Operator.IOH, description="Operator to BOQ algorithm."),
):
    """
    Create Intersite Bill of Quantity .  
    KMZ file must be containing ['Topology', 'Route', 'FO Hub', 'Site List'].

    **Input Design Sample**  
    [🟢 Download Here](http://10.83.10.16:8000/download-template/BOQ_Design_Sample.kmz)
    
    **Note:**
    - IOH Operator  : Separator will be '-'
    - XL Operator   : Separator will be ';'
    """

    date_today = datetime.now().strftime("%Y%m%d")
    boq_upload = os.path.join(UPLOAD_DIR, date_today, "Intersite", "BOQ")
    os.makedirs(boq_upload, exist_ok=True)

    match operator:
        case "xl":
            sep = ";"
        case _:
            sep = "-"
    print(f"ℹ️ Operator  : {operator}")
    print(f"ℹ️ Separator : {sep}")
    
    try:
        suffix = os.path.splitext(design_file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_fiber:
            tmp_fiber.write(design_file.file.read())
            tmp_fiber_path = tmp_fiber.name
        
        if suffix in ['.kml', '.kmz']:
            point_kmz, line_kmz = validate_kmz_design(tmp_fiber_path, sep=sep)
        else:
            return {"error": f"Unsupported topology file format {suffix}. Supported formats are GPKG, Parquet, and Shapefile."}    
    except Exception as e:
        return {"error": f"Failed to read topology file: {str(e)}"}

    # SAVE DATA
    points_path = os.path.join(boq_upload, f"{datetime.now().strftime('%H%M%S')}_points_kmz_{uuid4().hex}.parquet")
    lines_path = os.path.join(boq_upload, f"{datetime.now().strftime('%H%M%S')}_lines_kmz_{uuid4().hex}.parquet")

    point_kmz.to_parquet(points_path, index=False)
    line_kmz.to_parquet(lines_path, index=False)
    print(f"📥 Temporary Points data saved to   : {points_path}")
    print(f"📥 Temporary Lines data saved to    : {lines_path}")

    try:
        data = {
            "points_path": points_path,
            "lines_path": lines_path,
            "program": program,
            "operator": operator,
            "sep": sep
        }
        data = dumps(data, default=str)
        celery_task = task_boq.apply_async(args=[data])

        return {
            "message": "BOQ task has been initiated.",
            "task_id": celery_task.id,
            "task_status_url": f"/tasks/status/{celery_task.id}"
        }
    except Exception as e:
        return {"error": f"Failed to process data: {str(e)}"}