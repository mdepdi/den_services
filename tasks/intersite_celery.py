import os
import geopandas as gpd
import pandas as pd
import zipfile
from time import time
from json import loads, dumps
from datetime import datetime
from celery_app import celery_app
from service.modularized_insert_ring import main_insertring
from service.intersite.ring_algorithm import main_supervised
from service.intersite.clustering_algorithm import main_unsupervised
from service.modularized_fix_route import main_fixroute
from core.config import settings

EXPORT_DIR = settings.EXPORT_DIR
DOCKER = settings.DOCKER
if not os.path.exists(EXPORT_DIR):
    os.makedirs(EXPORT_DIR)

# TASK FOR INSERT RING
@celery_app.task(name="tasks.heavy.insert_ring", bind=True, max_retries=3, default_retry_delay=60)
def task_insertring(self, data: dict):
    try:
        print(f"🌏 Celery Fiberization| Insert Ring Task Started | Task ID: {self.request.id}")
        data = loads(data)

        self.update_state(
            state="PROGRESS", meta={"status": "Loading insert ring data"}
        )
        mapped_insert_path = data.get("mapped_insert_path")
        prev_fiber_path = data.get("prev_fiber_path")
        prev_points_path = data.get("prev_points_path")
        max_member = data.get("max_member", 12)
        route_type = data.get("route_type", "merged")
        program = data.get("program", 'N/A')

        date_today = datetime.now().strftime("%Y%m%d")
        export_loc = f"{EXPORT_DIR}/Insert_Ring/{date_today}/{self.request.id}"
        os.makedirs(export_loc, exist_ok=True)

        # LOAD DATA
        if DOCKER:
            if "/mnt/" not in mapped_insert_path:
                mapped_insert_path = mapped_insert_path.replace("uploads", "/mnt/uploads").replace("\\", "/")
            if "/mnt/" not in prev_fiber_path:
                prev_fiber_path = prev_fiber_path.replace("uploads", "/mnt/uploads").replace("\\", "/")
            if "/mnt/" not in prev_points_path:
                prev_points_path = prev_points_path.replace("uploads", "/mnt/uploads").replace("\\", "/")

        mapped_insert_gdf = gpd.read_parquet(mapped_insert_path)
        prev_fiber_gdf = gpd.read_parquet(prev_fiber_path)
        prev_points_gdf = gpd.read_parquet(prev_points_path)

        # IDENTIFY FIBERZONE & INSERT DATA
        self.update_state(
            state="PROGRESS", meta={"status": "Identifying fiber zone and insert data"}
        )

        # DETAIL INPUT
        print(f"==============================")
        print(f"📂 Mapped Insert Data   : {len(mapped_insert_gdf):,} features")
        print(f"📂 Previous Fiber Data  : {len(prev_fiber_gdf):,} features")
        print(f"📂 Previous Point Data  : {len(prev_points_gdf):,} features")
        print(f"==============================")


        # RUN INSERT RING PROCESSING
        self.update_state(
            state="PROGRESS", meta={"status": "Processing insert ring data"}
        )
        start_time = time()
        result = main_insertring(
            mapped_insert=mapped_insert_gdf,
            prev_fiber=prev_fiber_gdf,
            prev_point=prev_points_gdf,
            export_dir=export_loc,
            MAX_WORKERS=settings.MAX_WORKERS,
            MAX_MEMBER=max_member,
            ROUTE_TYPE=route_type
        )
        end_time = time()
        elapsed_time = end_time - start_time
        print(f"⏱️ Elapsed time: {elapsed_time/60:.2f} minutes")

        # ZIPFILE
        zip_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_Insert_Ring_Task.zip"
        zip_filepath = os.path.join(export_loc, zip_filename)
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(export_loc):
                for file in files:
                    if file != zip_filename and not file.endswith(".zip"):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, export_loc)
                        zipf.write(file_path, arcname)
        print(f"📦 Result files zipped.")
        
        self.update_state(
            state="SUCCESS",
            meta={"status": "Insert ring fiberization data processed successfully", "result": result, "zip_file": zip_filepath},
        )
        print(f"Celery | Insert Ring Task Completed | Task ID: {self.request.id}")

        # CLEAN UP TEMP FILES
        try:
            if os.path.exists(mapped_insert_path):
                os.remove(mapped_insert_path)
            if os.path.exists(prev_fiber_path):
                os.remove(prev_fiber_path)
            if os.path.exists(prev_points_path):
                os.remove(prev_points_path)
        except Exception as cleanup_error:
            print(f"Error during cleanup of temporary files: {str(cleanup_error)}")

        return result
    except Exception as e:
        self.retry(exc=e, countdown=60, max_retries=3)
        self.update_state(state="FAILURE", meta={"status": str(e)})
        print(f"Exception occurred during insert ring processing: {str(e)}")
        raise e


# TASK FOR SUPERVISED RING
@celery_app.task(name="tasks.heavy.supervised_ring", bind=True, max_retries=1, default_retry_delay=60)
def task_supervised(self, data: dict):
    try:
        print(f"🌏 Celery Intersite | Supervised Task Started | Task ID: {self.request.id}")
        parsed_data = loads(data)
        site_path = parsed_data.get("site_path")
        program = parsed_data.get("program", 'Fiberization')
        method = parsed_data.get("method", "supervised")
        area_col = parsed_data.get("area_col", 'region')
        cluster_col = parsed_data.get("cluster_col", 'ring_name')
        boq = parsed_data.get("boq", False)


        if DOCKER:
            if "/mnt/" not in site_path:
                site_path = site_path.replace("uploads", "/mnt/uploads").replace("\\", "/")

        # LOAD DATA
        site_data = gpd.read_parquet(site_path)

        date_today = datetime.now().strftime("%Y%m%d")
        export_loc = f"{EXPORT_DIR}/Intersite/Supervised/{date_today}/{self.request.id}"
        os.makedirs(export_loc, exist_ok=True)

        self.update_state(state="PROGRESS", meta={"status": "Processing supervised intersite process"})
        result = main_supervised(
            site_data=site_data,
            export_loc=export_loc,
            area_col=area_col,
            cluster_col=cluster_col,
            method=method,
            boq=boq,
            program=program,
            task_celery=self
        )

        # ZIPFILE
        zip_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_Supervised_Intersite.zip"
        zip_filepath = os.path.join(export_loc, zip_filename)
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(export_loc):
                for file in files:
                    if file != zip_filename and not file.endswith(".zip"):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, export_loc)
                        zipf.write(file_path, arcname)
        print(f"📦 Result files zipped.")
        
        self.update_state(
            state="SUCCESS",
            meta={"status": "Supervised fiberization data processed successfully", "zip_file": zip_filepath},
        )

        # CLEAN UP TEMP FILES
        try:
            if os.path.exists(site_path):
                os.remove(site_path)

        except Exception as cleanup_error:
            print(f"Error during cleanup of temporary files: {str(cleanup_error)}")
            
        return result

    except Exception as e:
        self.retry(exc=e, countdown=60, max_retries=3)
        self.update_state(state="FAILURE", meta={"status": str(e)})
        print(f"Exception occurred during supervised fiberization processing: {str(e)}")
        raise e

# TASK FOR UNSUPERVISED RING
@celery_app.task(name="tasks.heavy.unsupervised_ring", bind=True, max_retries=1, default_retry_delay=60)
def task_unsupervised(self, data: dict):
    try:
        print(f"🌏 Celery Intersite | Unsupervised Task Started | Task ID: {self.request.id}")
        parsed_data = loads(data)
        hubs_path = parsed_data.get("hub_path")
        site_path = parsed_data.get("site_path")
        member_expectation = parsed_data.get("member_expectation")
        max_distance = parsed_data.get("max_distance", 10000)
        boq = parsed_data.get("boq", False)
        area_col = parsed_data.get("area_col", 'region')
        cluster_col = parsed_data.get("cluster_col", 'ring_name')
        drop_existings = parsed_data.get("drop_existings", False)
        program = parsed_data.get("program", 'N/A')

        if DOCKER:
            if "/mnt/" not in site_path:
                site_path = site_path.replace("uploads", "/mnt/uploads").replace("\\", "/")
            if "/mnt/" not in hubs_path:
                hubs_path = hubs_path.replace("uploads", "/mnt/uploads").replace("\\", "/")

        # LOAD DATA
        site_data = gpd.read_parquet(site_path)
        hubs_data = gpd.read_parquet(hubs_path)
        
        date_today = datetime.now().strftime("%Y%m%d")
        export_loc = f"{EXPORT_DIR}/Intersite/Unsupervised/{date_today}/{self.request.id}"
        os.makedirs(export_loc, exist_ok=True)

        self.update_state(state="PROGRESS", meta={"status": "Processing unsupervised intersite data"})
        result = main_unsupervised(
            site_data=site_data,
            hubs_data=hubs_data,
            member_expectation=member_expectation,
            max_distance=max_distance,
            export_loc=export_loc,
            area_col=area_col,
            cluster_col=cluster_col,
            drop_existings=drop_existings,
            boq=boq,
            program=program,
            task_celery=self
        )
        
        # ZIPFILE
        zip_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_Unsupervised_Intersite.zip"
        zip_filepath = os.path.join(export_loc, zip_filename)
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(export_loc):
                for file in files:
                    if file != zip_filename and not file.endswith(".zip"):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, export_loc)
                        zipf.write(file_path, arcname)
        print(f"📦 Result files zipped.")
        self.update_state(
            state="SUCCESS",
            meta={"status": "Unsupervised fiberization data processed successfully", "zip_file": zip_filepath},
        )

        # CLEAN UP TEMP FILES
        try:
            if os.path.exists(site_path):
                os.remove(site_path)
        except Exception as cleanup_error:
            print(f"Error during cleanup of temporary files: {str(cleanup_error)}")
            
        return result

    except Exception as e:
        self.retry(exc=e, countdown=60, max_retries=3)
        self.update_state(state="FAILURE", meta={"status": str(e)})
        print(f"Exception occurred during unsupervised fiberization processing: {str(e)}")
        raise e
    
# TASK FOR FIX ROUTE
@celery_app.task(name="tasks.heavy.fixroute", bind=True, max_retries=1, default_retry_delay=60)
def task_fixroute(self, data: dict):
    try:
        print(f"🌏 Celery Fiberization | Fix Route Task Started | Task ID: {self.request.id}")
        parsed_data = loads(data)
        ne_path = parsed_data.get("ne_path")
        fe_path = parsed_data.get("fe_path")
        program_name = parsed_data.get("program_name", 'N/A')
        
        if DOCKER:
            if "/mnt/" not in ne_path:
                ne_path = ne_path.replace("uploads", "/mnt/uploads").replace("\\", "/")
            if "/mnt/" not in fe_path:
                fe_path = fe_path.replace("uploads", "/mnt/uploads").replace("\\", "/")

        # LOAD DATA
        ne_data = gpd.read_parquet(ne_path)
        fe_data = gpd.read_parquet(fe_path)
        
        date_today = datetime.now().strftime("%Y%m%d")
        export_loc = f"{EXPORT_DIR}/Intersite/Fix Route/{date_today}/{self.request.id}"
        os.makedirs(export_loc, exist_ok=True)

        self.update_state(state="PROGRESS", meta={"status": "Processing fix route data"})
        result = main_fixroute(
            ne_data=ne_data,
            fe_data=fe_data,
            export_dir=export_loc,
            max_workers=8,
            program_name=program_name,
            task_celery=self
        )

        # ZIPFILE
        zip_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_Fix_Route_Task.zip"
        zip_filepath = os.path.join(export_loc, zip_filename)
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(export_loc):
                for file in files:
                    if file != zip_filename and not file.endswith(".zip"):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, export_loc)
                        zipf.write(file_path, arcname)
        print(f"📦 Result files zipped.")
        
        self.update_state(
            state="SUCCESS",
            meta={"status": "Fix route fiberization data processed successfully", "result": result, "zip_file": zip_filepath},
        )

        # CLEAN UP TEMP FILES
        try:
            if os.path.exists(ne_path):
                os.remove(ne_path)
            if os.path.exists(fe_path):
                os.remove(fe_path)
        except Exception as cleanup_error:
            print(f"Error during cleanup of temporary files: {str(cleanup_error)}")
            
        return result

    except Exception as e:
        self.retry(exc=e, countdown=60, max_retries=3)
        self.update_state(state="FAILURE", meta={"status": str(e)})
        print(f"Exception occurred during fix route fiberization processing: {str(e)}")
        raise e