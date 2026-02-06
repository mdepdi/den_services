# ===============
# ROUTE EXTRACTOR
# ===============
import os
import sys
import time
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import simplekml
import zipfile

from tqdm import tqdm
from datetime import datetime
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import nearest_points
from shapely.ops import linemerge
from concurrent.futures import ProcessPoolExecutor, as_completed

from enum import Enum
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.append(root)

from modules.kml import validate_kmz_design
from core.config import settings
from service.intersite.ring_algorithm import save_intersite

MAINDATA_DIR = settings.MAINDATA_DIR
DATA_DIR = settings.DATA_DIR

# -----
# CLASS
# -----
class Separator(str, Enum):
    SEMICOLON = ";"
    HYPHEN = "-"

# ------------------------------------------------------
# LOGGER
# ------------------------------------------------------
from core.logger import create_logger
logger = create_logger(__file__)

def takeout_ring(kmz_path: str, export_dir: str, ring_list:list, sep:Separator = Separator.SEMICOLON.value):
    filename = Path(kmz_path).stem
    qty_ring = len(ring_list)
    
    logger.info(f"ℹ️ Takeout Rings Executed")
    logger.info(f"ℹ️ Takeout {qty_ring:,} rings from {filename}")

    validated = validate_kmz_design(kmz_path, sep=sep)
    if validated is None:
        logger.info(f"❌ Invalid KMZ Design: {kmz_path}")
        return

    # ---------------------------
    # Inputs (GeoDataFrames)
    # ---------------------------
    points_kmz, lines_kmz = validated
    ring_list = [str(ring) for ring in ring_list]
    points_cleaned = points_kmz[~(points_kmz['ring_name'].astype(str).isin(ring_list))].copy()
    lines_cleaned = lines_kmz[~(lines_kmz['ring_name'].astype(str).isin(ring_list))].copy()


    os.makedirs(export_dir, exist_ok=True)
    logger.info("🧩 Save Design Information")
    save_intersite(
        points=points_cleaned,
        paths=lines_cleaned,
        export_dir=export_dir,
        method="Cleaned Ring"
    )

    logger.info("🏆 Clean KM Design export completed.")
    logger.info(f"ℹ️ All files saved to: {export_dir}")

if __name__ == "__main__":
    design_path = r"D:\JACOBS\PROJECT\TASK\2026\FEB\W1\DRM FORMAT\FWA COMPILE ADJUSTED\Compiled FWA Surge Adjusted.kmz"
    ringlist_file = r"D:\JACOBS\PROJECT\TASK\2026\FEB\W1\DRM FORMAT\Clean_Ring List.xlsx"

    export_dir = r"D:\JACOBS\PROJECT\TASK\2026\FEB\W1\DRM FORMAT\FWA COMPILE ADJUSTED"
    os.makedirs(export_dir, exist_ok=True)

    # ringlist = pd.read_excel(ringlist_file)
    ringlist = []

    takeout_ring(
        kmz_path=design_path,
        export_dir=export_dir,
        ring_list=ringlist,
        sep=Separator.SEMICOLON.value
    )

