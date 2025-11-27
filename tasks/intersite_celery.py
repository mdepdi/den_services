import os
import geopandas as gpd
import pandas as pd
import zipfile
from time import time
from json import loads, dumps
from datetime import datetime
from celery_app import celery_app
from service.intersite.insert_algorithm import main_insertring
from service.intersite.ring_algorithm import main_supervised
from service.intersite.clustering_algorithm import main_unsupervised
from service.intersite.topology_algorithm import main_topology
from service.intersite.poligonized_algorithm import main_poligonized
from service.intersite.fixroute_algorithm import main_fixroute
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

        self.update_state(state="PROGRESS", meta={"status": "Loading insert ring data"})
        insert_list_path = data.get("insert_list_path")
        kmz_path = data.get("kmz_path")
        max_member = data.get("max_member", 12)
        max_distance = data.get("max_distance", 3000)
        program = data.get("program", 'Insert Ring')

        date_today = datetime.now().strftime("%Y%m%d")
        export_loc = f"{EXPORT_DIR}/Insert_Ring/{date_today}/{self.request.id}"
        os.makedirs(export_loc, exist_ok=True)

        # LOAD DATA
        if DOCKER:
            if "/mnt/" not in insert_list_path:
                insert_list_path = insert_list_path.replace("uploads", "/mnt/uploads").replace("\\", "/")
            if "/mnt/" not in kmz_path:
                kmz_path = kmz_path.replace("uploads", "/mnt/uploads").replace("\\", "/")

        insert_list_path = gpd.read_parquet(insert_list_path)
        kmz_path = gpd.read_parquet(kmz_path)

        # IDENTIFY FIBERZONE & INSERT DATA
        self.update_state(state="PROGRESS", meta={"status": "Identifying fiber zone and insert data"})


        # RUN INSERT RING PROCESSING
        self.update_state(state="PROGRESS", meta={"status": "Processing insert ring data"})
        start_time = time()
        result = main_insertring(
            insert_data = insert_list_path,
            kmz_data = kmz_path,
            export_dir = export_loc,
            max_member = max_member,
            max_distance = max_distance,
            task_celery = self
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
        boq = parsed_data.get("boq", False)
        method = parsed_data.get("method", "Supervised")
        area_col = parsed_data.get("area_col", 'region')
        cluster_col = parsed_data.get("cluster_col", 'ring_name')


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
            export_loc=export_loc,
            area_col=area_col,
            cluster_col=cluster_col,
            max_distance=max_distance,
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
        template_path = parsed_data.get("template_path")
        boq = parsed_data.get("boq", False)
        program = parsed_data.get("program", 'Fix Route Fiberization')
        
        if DOCKER:
            if "/mnt/" not in template_path:
                template_path = template_path.replace("uploads", "/mnt/uploads").replace("\\", "/")

        # LOAD DATA
        template_df = pd.read_excel(template_path)
        date_today = datetime.now().strftime("%Y%m%d")
        export_loc = f"{EXPORT_DIR}/Intersite/Fix Route/{date_today}/{self.request.id}"
        os.makedirs(export_loc, exist_ok=True)

        self.update_state(state="PROGRESS", meta={"status": "Processing fix route data"})
        result = main_fixroute(
            template_df=template_df,
            export_dir=export_loc,
            program=program,
            boq=boq,
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
            if os.path.exists(template_path):
                os.remove(template_path)
        except Exception as cleanup_error:
            print(f"Error during cleanup of temporary files: {str(cleanup_error)}")
            
        return result

    except Exception as e:
        self.retry(exc=e, countdown=60, max_retries=3)
        self.update_state(state="FAILURE", meta={"status": str(e)})
        print(f"Exception occurred during fix route fiberization processing: {str(e)}")
        raise e
    
# TASK POLYGON BASED INTERSITE
@celery_app.task(name="tasks.heavy.polygon_intersite", bind=True, max_retries=1, default_retry_delay=60)
def task_polygon_intersite(self, data: dict):
    try:
        print(f"🌏 Celery Fiberization | Polygon Based Task Started | Task ID: {self.request.id}")
        parsed_data = loads(data)
        excel_path = parsed_data.get("excel_path")
        polygon_path = parsed_data.get("polygon_path")
        boq = parsed_data.get("boq", False)
        program = parsed_data.get("program", 'Polygon Based Fiberization')
        
        if DOCKER:
            if "/mnt/" not in excel_path:
                excel_path = excel_path.replace("uploads", "/mnt/uploads").replace("\\", "/")
            if "/mnt/" not in polygon_path:
                polygon_path = polygon_path.replace("uploads", "/mnt/uploads").replace("\\", "/")

        # LOAD DATA
        date_today = datetime.now().strftime("%Y%m%d")
        export_loc = f"{EXPORT_DIR}/Intersite/Polygon Based/{date_today}/{self.request.id}"
        os.makedirs(export_loc, exist_ok=True)

        self.update_state(state="PROGRESS", meta={"status": "Processing Polygon Based data"})
        result = main_poligonized(
            excel_path=excel_path,
            polygon_file=polygon_path,
            export_dir=export_loc,
            program=program,
            boq=boq,
            task_celery=self
        )

        # ZIPFILE
        zip_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_Polygon_Task.zip"
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
            meta={"status": "Polygon based fiberization data processed successfully", "result": result, "zip_file": zip_filepath},
        )

        # CLEAN UP TEMP FILES
        try:
            if os.path.exists(polygon_path):
                os.remove(polygon_path)
            if os.path.exists(excel_path):
                os.remove(excel_path)
        except Exception as cleanup_error:
            print(f"Error during cleanup of temporary files: {str(cleanup_error)}")
            
        return result

    except Exception as e:
        self.retry(exc=e, countdown=60, max_retries=3)
        self.update_state(state="FAILURE", meta={"status": str(e)})
        print(f"Exception occurred during polygon based fiberization processing: {str(e)}")
        raise e

# TASK TOPOLOGY BASED INTERSITE
@celery_app.task(name="tasks.heavy.topology_intersite", bind=True, max_retries=1, default_retry_delay=60)
def task_topology_intersite(self, data: dict):
    try:
        print(f"🌏 Celery Fiberization | Topology Based Task Started | Task ID: {self.request.id}")
        parsed_data = loads(data)
        excel_path = parsed_data.get("excel_path")
        line_path = parsed_data.get("line_path")
        boq = parsed_data.get("boq", False)
        program = parsed_data.get("program", 'Topology Based Fiberization')
        
        if DOCKER:
            if "/mnt/" not in excel_path:
                excel_path = excel_path.replace("uploads", "/mnt/uploads").replace("\\", "/")
            if "/mnt/" not in line_path:
                line_path = line_path.replace("uploads", "/mnt/uploads").replace("\\", "/")

        # LOAD DATA
        date_today = datetime.now().strftime("%Y%m%d")
        export_loc = f"{EXPORT_DIR}/Intersite/Topology Based/{date_today}/{self.request.id}"
        os.makedirs(export_loc, exist_ok=True)

        self.update_state(state="PROGRESS", meta={"status": "Processing Topology Based data"})
        result = main_topology(
            excel_path=excel_path,
            line_file=line_path,
            export_dir=export_loc,
            program=program,
            boq=boq,
            task_celery=self
        )

        # ZIPFILE
        zip_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_Topology_Task.zip"
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
            meta={"status": "Topology Based fiberization data processed successfully", "result": result, "zip_file": zip_filepath},
        )

        # CLEAN UP TEMP FILES
        try:
            if os.path.exists(excel_path):
                os.remove(excel_path)
            if os.path.exists(line_path):
                os.remove(line_path)
        except Exception as cleanup_error:
            print(f"Error during cleanup of temporary files: {str(cleanup_error)}")
            
        return result

    except Exception as e:
        self.retry(exc=e, countdown=60, max_retries=3)
        self.update_state(state="FAILURE", meta={"status": str(e)})
        print(f"Exception occurred during Topology Based fiberization processing: {str(e)}")
        raise e