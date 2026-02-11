import os
import geopandas as gpd
import pandas as pd
import zipfile
from time import time
from json import loads, dumps
from datetime import datetime
from celery_app import celery_app

from modules.celery import report_state
from service.graphhopper.graphhopper import nearest_point_to_point
from core.config import settings

# DIRECTORY SETTING
EXPORT_DIR = settings.EXPORT_DIR
DOCKER = settings.DOCKER
if not os.path.exists(EXPORT_DIR):
    os.makedirs(EXPORT_DIR)

# TASK FOR GRAPHHOPPER
@celery_app.task(name="tasks.heavy.graphhopper", bind=True, max_retries=0, default_retry_delay=60)
def task_nearest_point(self, data: dict):
    try:
        print(f"Celery Graphhopper| Nearest Point Task Started | Task ID: {self.request.id}")
        data = loads(data)

        report_state(self, state="STARTED", message="Loading Nearest Point Data ...")
        source_path = data.get("source_path")
        target_path = data.get("target_path")
        k_final = data.get("k_final", 1)
        cutoff = data.get("cutoff", 100000)
        sep = data.get("sep", ";")

        date_today = datetime.now().strftime("%Y%m%d")
        export_loc = f"{settings.EXPORT_DIR}/{date_today}/Graphhopper/{self.request.id}"
        os.makedirs(export_loc, exist_ok=True)

        # LOAD DATA
        if DOCKER:
            if "/mnt/" not in source_path:
                source_path = source_path.replace("uploads", "/mnt/uploads").replace("\\", "/")
            if "/mnt/" not in target_path:
                target_path = target_path.replace("uploads", "/mnt/uploads").replace("\\", "/")

        # RUN INSERT RING PROCESSING
        report_state(self, state="PROGRESS", message="Processing Graphhopper Nearest Point ...")

        result = nearest_point_to_point(
            source_path=source_path,
            target_path=target_path,
            export_dir=export_loc,
            k_final=k_final,
            sep=sep,
            cutoff=cutoff,
            task_celery=self
        )


        # ZIPFILE
        zip_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_Graphhopper_Nearest_Point_Task.zip"
        zip_filepath = os.path.join(export_loc, zip_filename)
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(export_loc):
                for export_file in files:
                    if export_file.endswith(".zip") or "Checkpoint" in export_file:
                        continue
                    export_file_path = os.path.join(root, export_file)
                    arcname = os.path.relpath(export_file_path, export_loc)
                    zipf.write(export_file_path, arcname)
        print(f"📦 Result files zipped.")
        report_state(self, state="SUCCESS", message="Graphhopper Nearest Point Task processed successfully", extra={"zip_file": zip_filepath})

        return result
    except Exception as e:
        report_state(self, state="FAILURE", message=f"Error in Nearest Point: {e}")
        raise e