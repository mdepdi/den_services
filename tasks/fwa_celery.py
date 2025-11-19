import os
import geopandas as gpd
import pandas as pd
import zipfile
from time import time
from json import loads, dumps
from datetime import datetime
from celery_app import celery_app
from service.fwa_alghoritm import main_fwa
from core.config import settings

EXPORT_DIR = settings.EXPORT_DIR
DOCKER = settings.DOCKER
if not os.path.exists(EXPORT_DIR):
    os.makedirs(EXPORT_DIR)

# TASK FOR INSERT RING
@celery_app.task(name="tasks.heavy.fwa", bind=True, max_retries=2, default_retry_delay=60)
def task_fwa(self, data: dict):
    try:
        print(f"Celery FWA| FWA Task Started | Task ID: {self.request.id}")
        data = loads(data)

        self.update_state(
            state="PROGRESS", meta={"status": "Loading fixed wireless access (FWA) data"}
        )
        data_path = data.get("data_path")
        distance_fwa = data.get("distance_fwa", 500)
        sector_angle = data.get("sector_angle", 120)
        sector_group = data.get("sector_group", 120)
        threshold = data.get("threshold", 300)
        method = data.get("method", 'maximize')
        extend_sector = data.get("extend_sector", False)

        date_today = datetime.now().strftime("%Y%m%d")
        export_loc = f"{settings.EXPORT_DIR}/Fixed_Wireless_Access/{date_today}/{self.request.id}"
        os.makedirs(export_loc, exist_ok=True)

        # LOAD DATA
        if DOCKER:
            if "/mnt/" not in data_path:
                data_path = data_path.replace("uploads", "/mnt/uploads").replace("\\", "/")
        data_gdf = gpd.read_parquet(data_path)

        # DETAIL INPUT
        print(f"==============================")
        print(f"📂 Sitelist for FWA   : {len(data_gdf):,} features")
        print(f"==============================")

        # RUN INSERT RING PROCESSING
        self.update_state(
            state="PROGRESS", meta={"status": "Processing FWA"}
        )
        result = main_fwa(
            data_gdf= data_gdf,
            export_dir=export_loc,
            method=method,
            sector_angle=sector_angle,
            sector_group=sector_group,
            threshold=threshold,
            distance_fwa=distance_fwa,
            extend_sector=extend_sector,
            task_celery=self
        )

        print(f"Celery | FWA Task Completed | Task ID: {self.request.id}")

        # CLEAN UP TEMP FILES
        try:
            if os.path.exists(data_path):
                os.remove(data_path)
        except Exception as cleanup_error:
            print(f"Error during cleanup of temporary files: {str(cleanup_error)}")

        return result
    except Exception as e:
        self.retry(exc=e, countdown=60, max_retries=3)
        self.update_state(state="FAILURE", meta={"status": str(e)})
        print(f"Exception occurred during FWA processing: {str(e)}")
        raise e