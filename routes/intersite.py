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
from modules.validation import input_insertring, input_newring, input_unsupervised, identify_fiberzone, prepare_prevdata, identify_insertdata, validate_fixroute
from service.update_intersite import main_update_intersite
from tasks.intersite_celery import task_supervised, task_unsupervised, task_fixroute, task_insertring

# EXPORT DIR
UPLOAD_DIR = settings.UPLOAD_DIR
EXPORT_DIR = settings.EXPORT_DIR
DATA_DIR = settings.DATA_DIR

class NewRingSchema(BaseModel):
    excel_file: Optional[UploadFile] = File(
        None, description="Excel file containing ring data. Must be containing columns: 'site_id', 'site_name', 'site_type', 'lat', 'long', 'region', 'ring_name', 'flag'"
    )
    fiber_route: Optional[UploadFile] = File(
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
# IDENTIFY INSERT DATA
# =============================
@router.post("/identify_insert", tags=["Intersite"])
async def identify_insert(
    excel_file: UploadFile = File(
        ..., description="Excel file containing ring data to insert."
    ),
    previous_fiber: UploadFile = File(
        ..., description="GPKG, Parquet, or Shapefile containing previous fiber data."
    ),
    previous_points: UploadFile = File(
        ..., description="GPKG, Parquet, or Shapefile containing previous points data."
    ),
    max_member: int = Form(12, description="Maximum number of members to consider for insertion."),
    search_radius: int = Form(2000, description="Maximum radius(m) for search insert ring.")
):
    """
    Identified the insert ring data based on **Excel template, Existing Points, and Existing Fiber Route**.

    **Template Identify Insert**  
    [🟢 Download Here](http://localhost:8000/download-template/Template_Identify_Ring.xlsx)

    **Output:**  
    **Insert Data (.xlsx)** : Use this data to execute in **Insert Ring API**  
    **New Data (.xlsx)**    : Use this data to execute in **Unsupervised API**
    """
    # Read previous fiber file
    try:
        suffix = os.path.splitext(previous_fiber.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_fiber:
            tmp_fiber.write(previous_fiber.file.read())
            tmp_fiber_path = tmp_fiber.name
        
        if suffix in ['.gpkg', '.parquet', '.shp']:
            prev_fiber_gdf = read_gdf(tmp_fiber_path)
            print(f"📥 Reading previous fiber file: {previous_fiber.filename}")
        else:
            return {"error": "Unsupported previous fiber file format. Supported formats are GPKG, Parquet, and Shapefile."}
    except Exception as e:
        return {"error": f"Failed to read previous fiber file: {str(e)}"}
    finally:
        if os.path.exists(tmp_fiber_path):
            os.remove(tmp_fiber_path)

    # Read previous points file
    try:
        suffix = os.path.splitext(previous_points.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_points:
            tmp_points.write(previous_points.file.read())
            tmp_points_path = tmp_points.name
        
        if suffix in ['.gpkg', '.parquet', '.shp']:
            prev_points_gdf = read_gdf(tmp_points_path)
            print(f"📥 Reading previous points file: {previous_points.filename}")
        else:
            return {"error": "Unsupported previous points file format. Supported formats are GPKG, Parquet, and Shapefile."}
    except Exception as e:
        return {"error": f"Failed to read previous points file: {str(e)}"}
    
    finally:
        if os.path.exists(tmp_points_path):
            os.remove(tmp_points_path)

    # Process data
    try:
        suffix = os.path.splitext(excel_file.filename)[1].lower()
        filename = os.path.splitext(excel_file.filename)[0]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_excel:
            tmp_excel.write(excel_file.file.read())
            tmp_excel_path = tmp_excel.name
            result_paths = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
            result_paths = os.path.join(EXPORT_DIR, 'Identify Insert', result_paths)
            os.makedirs(result_paths, exist_ok=True)

            new_sites, existing_sites, hubs = input_insertring(tmp_excel_path)

            # PREPARE PREV DATA
            prev_fiber_gdf, prev_points_gdf = prepare_prevdata(prev_fiber_gdf, prev_points_gdf)
            newsites_within, newsites_outside = identify_fiberzone(new_sites, prev_fiber=prev_fiber_gdf, search_radius=search_radius)
            mapped_insert, dropped_insert = identify_insertdata(newsites_within, prev_fiber=prev_fiber_gdf, prev_points=prev_points_gdf, search_radius=search_radius, max_member=max_member)
            
            # INSERT DATA
            mapped_insert = mapped_insert.sort_values(by="distance_to_fiber").reset_index(drop=True)
            if 'index_right' in mapped_insert.columns:
                mapped_insert = mapped_insert.drop(columns=['index_right'])
            
            with pd.ExcelWriter(os.path.join(result_paths, f"Insert Data.xlsx")) as writer:
                if 'geometry' in mapped_insert.columns:
                    mapped_insert = mapped_insert.drop(columns=['geometry'])
                mapped_insert.to_excel(writer, sheet_name='mapped_insert', index=False)
            print(f"✅ Insert data identification completed.")

            # NEW RING DATA
            if 'index_right' in newsites_outside.columns:
                newsites_outside = newsites_outside.drop(columns=['index_right'])
            if 'index_right' in existing_sites.columns:
                existing_sites = existing_sites.drop(columns=['index_right'])
            if 'index_right' in dropped_insert.columns:
                dropped_insert = dropped_insert.drop(columns=['index_right'])

            newsites_outside = newsites_outside.reset_index(drop=True)
            existing_sites = existing_sites.reset_index(drop=True)
            dropped_insert = dropped_insert.reset_index(drop=True)

            compiled_newring = []
            if not existing_sites.empty:
                existing_sites = existing_sites.to_crs(epsg=4326)
                compiled_newring.append(existing_sites)
            if not newsites_outside.empty:
                newsites_outside = newsites_outside.to_crs(epsg=4326)
                compiled_newring.append(newsites_outside)
            if not dropped_insert.empty:
                dropped_insert = dropped_insert.to_crs(epsg=4326)
                compiled_newring.append(dropped_insert)
            
            compiled_newring = pd.concat(compiled_newring, ignore_index=True) if compiled_newring else pd.DataFrame()
            print(f"✅ Compiled new ring data. Total sites: {len(compiled_newring):,}")

            with pd.ExcelWriter(os.path.join(result_paths, f"New Ring Data.xlsx")) as writer:
                if 'geometry' in compiled_newring.columns:
                    compiled_newring = compiled_newring.drop(columns=['geometry'])
                if 'geometry' in hubs.columns:
                    hubs = hubs.drop(columns=['geometry'])
                compiled_newring.to_excel(writer, sheet_name='new_ring', index=False)
                hubs.to_excel(writer, sheet_name='hubs', index=False)

            # ZIPFILE
            zip_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_Identified_Insert_Data.zip"
            zip_filepath = os.path.join(result_paths, zip_filename)
            with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(result_paths):
                    for file in files:
                        if file != zip_filename:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, result_paths)
                            zipf.write(file_path, arcname)
            print(f"📦 Result files zipped.")
        return FileResponse(zip_filepath, filename=zip_filename, media_type='application/zip')
    except Exception as e:
        return {"error": f"Failed to process data: {str(e)}"}
    finally:
        if os.path.exists(tmp_excel_path):
            os.remove(tmp_excel_path)

# =============================
# INSERT RING
# =============================
@router.post("/insert_ring", tags=["Intersite"])
async def insert_ring(
    mapped_insert: UploadFile = File(
        ..., description="Excel file containing ring data to insert."
    ),
    previous_fiber: UploadFile = File(
        ..., description="GPKG, Parquet, or Shapefile containing previous fiber data."
    ),
    previous_points: UploadFile = File(
        ..., description="GPKG, Parquet, or Shapefile containing previous points data."
    ),
    max_member: int = Form(12, description="Maximum number of members to consider for insertion.")
):
    """
    Insert new rings into existing fiber network based on the provided Excel file and previous data.
    """
    # Read previous fiber file
    upload_insert_dir = os.path.join(UPLOAD_DIR, "Intersite", "Insert Ring")
    os.makedirs(upload_insert_dir, exist_ok=True)
    try:

        suffix = os.path.splitext(previous_fiber.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_fiber:
            tmp_fiber.write(previous_fiber.file.read())
            tmp_fiber_path = tmp_fiber.name
        
        if suffix in ['.gpkg', '.parquet', '.shp']:
            prev_fiber_gdf = read_gdf(tmp_fiber_path)
            print(f"📥 Reading previous fiber file: {previous_fiber.filename}")
        else:
            return {"error": "Unsupported previous fiber file format. Supported formats are GPKG, Parquet, and Shapefile."}
    except Exception as e:
        return {"error": f"Failed to read previous fiber file: {str(e)}"}
    finally:
        if os.path.exists(tmp_fiber_path):
            os.remove(tmp_fiber_path)

    # Read previous points file
    try:
        suffix = os.path.splitext(previous_points.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_points:
            tmp_points.write(previous_points.file.read())
            tmp_points_path = tmp_points.name
        
        if suffix in ['.gpkg', '.parquet', '.shp']:
            prev_points_gdf = read_gdf(tmp_points_path)
            print(f"📥 Reading previous points file: {previous_points.filename}")
        else:
            return {"error": "Unsupported previous points file format. Supported formats are GPKG, Parquet, and Shapefile."}
    except Exception as e:
        return {"error": f"Failed to read previous points file: {str(e)}"}

    finally:
        if os.path.exists(tmp_points_path):
            os.remove(tmp_points_path)

    # Process data
    try:
        suffix = os.path.splitext(mapped_insert.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_excel:
            tmp_excel.write(mapped_insert.file.read())
            tmp_excel_path = tmp_excel.name
        if suffix in ['.xlsx', 'csv']:
            mapped_insert_gdf = read_gdf(tmp_excel_path)
            print(f"📥 Reading mapped insert file: {mapped_insert.filename}")

    except Exception as e:
        return {"error": f"Failed to process data: {str(e)}"}
    finally:
        if os.path.exists(tmp_excel_path):
            os.remove(tmp_excel_path)
    
    # SAVE AS PARQUET
    temp_mapped_insert = os.path.join(upload_insert_dir, f"{uuid4().hex}_mapped_insert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet")
    mapped_insert_gdf.to_parquet(temp_mapped_insert, index=False)
    print(f"📥 Temporary site data saved to: {temp_mapped_insert}")

    temp_fiber_path = os.path.join(upload_insert_dir, f"{uuid4().hex}_prev_fiber_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet")
    prev_fiber_gdf.to_parquet(temp_fiber_path, index=False)
    print(f"📥 Temporary fiber data saved to: {temp_fiber_path}")

    temp_points_path = os.path.join(upload_insert_dir, f"{uuid4().hex}_prev_points_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet")
    prev_points_gdf.to_parquet(temp_points_path, index=False)
    print(f"📥 Temporary fiber data saved to: {temp_points_path}")

    # CELERY TASK
    try:
        insertring_params = {
            "mapped_insert_path": temp_mapped_insert,
            "prev_fiber_path": temp_fiber_path,
            "prev_points_path": temp_points_path,
            "max_member": max_member
        }
        insertring_params = dumps(insertring_params, default=str)
        print("🚀 Initiating insert ring task...")

        celery_task = task_insertring.apply_async(args=[insertring_params])
        return {
            "message": "Insert ring task has been initiated.",
            "task_id": celery_task.id,
            "task_status_url": f"/tasks/status/{celery_task.id}"
        }
    except Exception as e:
        return {"error": f"Failed to initiate insert ring task: {str(e)}"}

# =============================
# NEW RING
# =============================

# SUPERVISED
@router.post("/supervised", tags=["Intersite"])
async def supervised_ring(
    excel_file: UploadFile = File(None, description="Excel file containing ring data."),
    program: str = Form("Fiberization", description="Program name if needed."),
    boq:bool = Form(False, description="Output file to choose")
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
    [🟢 Download Here](http://localhost:8000/download-template/Template_Supervised_Fiberization.xlsx)

    **Note:**
    - Site type should containing 'FO Hub' for interconnection source.
    - Each ring name must be on the same region.
    - Flag define the start hub or end hub.
    - Make sure the latitude and longitude is not reversed.
    """

    # Read Excel file
    if excel_file is None:
        return {"error": "Excel file is required."}
    
    upload_newring_dir = os.path.join(UPLOAD_DIR, "Intersite", "Supervised")
    os.makedirs(upload_newring_dir, exist_ok=True)
    method = "supervised"
    
    try:
        site_data = pd.read_excel(excel_file.file)
    except Exception as e:
        return {"error": f"Failed to read Excel file: {str(e)}"}

    if "site_id" in site_data.columns:
        site_data["site_id"] = site_data["site_id"].astype(str)
    if 'index_right' in site_data.columns:
        site_data = site_data.drop(columns=['index_right'])

    # VALIDATION
    site_data = sanitize_header(site_data)
    site_data = input_newring(site_data, method=method)

    # SAVE AS PARQUET
    temp_parquet_path = os.path.join(upload_newring_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_site_data_{uuid4().hex}.parquet")
    site_data.to_parquet(temp_parquet_path, index=False)
    print(f"📥 Temporary site data saved to: {temp_parquet_path}")

    # Process data
    try:
        data = {
            "site_path": temp_parquet_path,
            "program": program,
            "boq": boq,
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
    
# SUPERVISED
@router.post("/unsupervised", tags=["Intersite"])
async def unsupervised_ring(
    excel_file: UploadFile = File(None, description="Excel file containing sitelist and hubs sheet."),
    member_expectation: int = Form(10, description="Member expectation in one ring."),
    max_distance: int = Form(10000, description="Maximum distance to route."),
    program: str = Form("Fiberization", description="Program name if needed."),
    drop_existings:bool = Form(False, description="Drop ring if not conatining new site."),
    boq:bool = Form(False, description="Output file to choose")
):
    """
    Create Intersite design based on **Unsupervised Alghorithm**, the clustering based on our service.  
    Excel file must be containing **'sitelist'** and **'hubs'** sheet.

    **Template Unsupervised Fiberization**  
    [🟢 Download Here](http://localhost:8000/download-template/Template_Unsupervised_Fiberization.xlsx)

    **Note:**
    - Hubs should containing 'FO Hub' for interconnection source.
    - Each ring name must be on the same region.
    - Make sure the latitude and longitude is not reversed.
    """

    # Read Excel file
    if excel_file is None:
        return {"error": "Excel file is required."}
    
    upload_newring_dir = os.path.join(UPLOAD_DIR, "Intersite", "Unsupervised")
    os.makedirs(upload_newring_dir, exist_ok=True)
    
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
            nr_sites, nr_hubs = input_unsupervised(nr_sites, nr_hubs)
            sites_geom = gpd.points_from_xy(nr_sites['long'], nr_sites['lat'], crs="EPSG:4326")
            hubs_geom = gpd.points_from_xy(nr_hubs['long'], nr_hubs['lat'], crs="EPSG:4326")
            
            nr_sites = gpd.GeoDataFrame(nr_sites, geometry=sites_geom)
            nr_hubs = gpd.GeoDataFrame(nr_hubs, geometry=hubs_geom)

    except Exception as e:
        return {"error": f"Failed to read Excel file: {str(e)}"}
    
    site_data = nr_sites
    hubs_data = nr_hubs

    if "site_id" in site_data.columns:
        site_data["site_id"] = site_data["site_id"].astype(str)
    if 'index_right' in site_data.columns:
        site_data = site_data.drop(columns=['index_right'])

    # VALIDATION
    site_data = sanitize_header(site_data)
    site_data = input_newring(site_data, method="unsupervised")

    # SAVE AS PARQUET
    temp_parquet_path = os.path.join(upload_newring_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_site_data_{uuid4().hex}.parquet")
    temp_hub_path = os.path.join(upload_newring_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_hub_data_{uuid4().hex}.parquet")
    site_data.to_parquet(temp_parquet_path, index=False)
    hubs_data.to_parquet(temp_hub_path, index=False)
    print(f"📥 Temporary site data saved to : {temp_parquet_path}")
    print(f"📥 Temporary hub data saved to  : {temp_hub_path}")

    # Process data
    try:
        data = {
            "site_path": temp_parquet_path,
            "hub_path": temp_hub_path,
            "member_expectation": member_expectation,
            "max_distance": max_distance,
            "drop_existings": drop_existings,
            "program": program,
            "boq": boq
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
async def create_fixroute(
    excel_file: UploadFile = File(None, description="Excel file containing fix route template."),
    program: str = Form(None, description="Program name if not defined")
):
    """
    Create Intersite design based on **Fix Route Alghorithm**.  
    Excel file must be containing **Near End (NE)** as source and **Far End (FE)** as target.  

    **Template Unsupervised Fiberization**  
    [🟢 Download Here](http://localhost:8000/download-template/Template_Fixed_Route.xlsx)

    **Note:**
    - Fix Route will running based on region and ring name. Make sure to order the ring from start hub to end hub.  
    - Each ring should containing 'FO Hub' for interconnection source.  
    - Each ring name must be on the same region.  
    - Make sure the latitude and longitude is not reversed.  
    """

    # Read Excel file
    if excel_file is None:
        return {"error": "Excel file is required."}
    
    upload_newring_dir = os.path.join(UPLOAD_DIR, "Fix Route")
    os.makedirs(upload_newring_dir, exist_ok=True)
    
    try:
    # LOAD DATA
        with pd.ExcelFile(excel_file.file) as xls:
            fixroute_input = pd.read_excel(xls)
            gdf_ne, gdf_fe = validate_fixroute(fixroute_input)            
    except Exception as e:
        return {"error": f"Failed to read Excel file: {str(e)}"}

    if 'index_right' in gdf_ne.columns:
        gdf_ne = gdf_ne.drop(columns=['index_right'])
    if 'index_right' in gdf_fe.columns:
        gdf_fe = gdf_fe.drop(columns=['index_right'])

    # SAVE AS PARQUET
    temp_ne = os.path.join(upload_newring_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_ne_data_{uuid4().hex}.parquet")
    temp_fe = os.path.join(upload_newring_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_fe_data_{uuid4().hex}.parquet")
    gdf_ne.to_parquet(temp_ne, index=False)
    gdf_fe.to_parquet(temp_fe, index=False)
    print(f"📥 Temporary ne data saved to: {temp_ne}")
    print(f"📥 Temporary fe data saved to: {temp_fe}")

    # Process data
    try:
        data = {
            "ne_path": temp_ne,
            "fe_path": temp_fe,
            "program": program,
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


# ================
# UPDATE INTERSITE
# ================

@router.post("/update_intersite", tags=["Intersite"])
async def update_intersite(
    point_gdf: UploadFile = File(
        ..., description="GPKG, Parquet, or Shapefile containing point data."
    ),
    route_gdf: UploadFile = File(
        ..., description="GPKG, Parquet, or Shapefile containing route data."
    ),
    separator: str = Form("-", description="Define separator to identify near end and far end.")
):
    """
    Generate excel file from updated intersite route and point data.
    """
    # Read linestring data
    print(f"🌏 Execute | Update Intersite")
    
    # Point data
    suffix = os.path.splitext(point_gdf.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_point:
        tmp_point.write(point_gdf.file.read())
        tmp_point_path = tmp_point.name
    
    if suffix in ['.gpkg', '.parquet', '.shp']:
        point_gdf = read_gdf(tmp_point_path)
        print(f"📥 Reading linesetring file: {point_gdf.filename}")
    else:
        raise HTTPException(status_code=400, detail="Unsupported linesetring file format. Supported formats are GPKG, Parquet, and Shapefile.")
    
    # Line data
    suffix = os.path.splitext(route_gdf.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_route:
        tmp_route.write(route_gdf.file.read())
        tmp_route_path = tmp_route.name
    
    if suffix in ['.gpkg', '.parquet', '.shp']:
        route_gdf = read_gdf(tmp_route_path)
        print(f"📥 Reading linesetring file: {route_gdf.filename}")
    else:
        raise HTTPException(status_code=400, detail="Unsupported linesetring file format. Supported formats are GPKG, Parquet, and Shapefile.")
    
    for geom_type in route_gdf.geom_type:
        if geom_type not in ['Linestring', 'MultiLineString']:
            raise HTTPException(status_code=400, detail=f"Invalid file format {geom_type}")

    try:
        result_dir = os.path.join(EXPORT_DIR, "Fiberization", "Update Intersite")
        os.makedirs(result_dir, exist_ok=True)
        zip_path = main_update_intersite(point_gdf=point_gdf, route_gdf=route_gdf, export_dir=result_dir)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {e}")

    # --- FileResponse ---
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=os.path.basename(zip_path),
    )