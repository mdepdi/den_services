import os
import sys
import zipfile
import geopandas as gpd
import pandas as pd
import networkx as nx
import re
from datetime import datetime
from time import time
from shapely.ops import linemerge
from tqdm import tqdm

sys.path.append(r"D:\JACOBS\SERVICE\API")

from service.intersite.boq_algorithm import main_boq
from service.intersite.ring_algorithm import supervised_validation, main_supervised
from modules.table import sanitize_header
from modules.data import read_gdf, validate_longlat
from modules.utils import auto_group
from core.logger import create_logger
from core.config import settings
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------------------------------------
# SETTING
# ------------------------------------------------------
MAX_WORKERS = settings.MAX_WORKERS

# ------------------------------------------------------
# LOGGING CONFIGURATION
# ------------------------------------------------------
logger = create_logger(__file__)

# ------------------------------------------------------
# MAIN POLYGON
# ------------------------------------------------------
def validate_poligonize(excel_path:str):
    with pd.ExcelFile(excel_path) as excel:
        sheetname = excel.sheet_names
        used_sheet = ['sitelist', 'hubs']
        for sheet in used_sheet:
            if sheet not in sheetname:
                raise ValueError(f"Sheet {sheet} not found in Excel file. Check your input.")
        
        sitelist = pd.read_excel(excel, sheet_name='sitelist')
        hubs = pd.read_excel(excel, sheet_name='hubs')
        sitelist['site_type'] = "Site List"
        hubs['site_type'] = "FO Hub"

        print(f"ℹ️ Total sitelist: {len(sitelist):,}")
        print(f"ℹ️ Total hubs    : {len(hubs):,}")

        used_col = ['site_id','site_name', 'long', 'lat', 'site_type']
        for col in used_col:
            if col not in sitelist.columns:
                raise ValueError(f"Column {col} not found in Sitelist. Check your input.")
            if col not in hubs.columns:
                raise ValueError(f"Column {col} not found in Hubs. Check your input.")
        
        sitelist['site_id'] = sitelist['site_id'].astype(str) 
        hubs['site_id'] = hubs['site_id'].astype(str) 
        sitelist['site_name'] = sitelist['site_name'].astype(str) 
        hubs['site_name'] = hubs['site_name'].astype(str)
        sitelist = validate_longlat(sitelist)
        hubs = validate_longlat(hubs)
        sitelist_geom = gpd.points_from_xy(sitelist['long'], sitelist['lat'], crs="EPSG:4326")
        hubs_geom = gpd.points_from_xy(hubs['long'], hubs['lat'], crs='EPSG:4326')

        sitelist_gdf = gpd.GeoDataFrame(sitelist, geometry=sitelist_geom)
        hubs_gdf = gpd.GeoDataFrame(hubs, geometry=hubs_geom)
        
    return sitelist_gdf, hubs_gdf

def polygonize_algo(sitelist_gdf:gpd.GeoDataFrame, hubs_gdf:gpd.GeoDataFrame, polygon_gdf:gpd.GeoDataFrame, project_name:str):
    def get_code(hub_id:str):
        code = None
        pattern = re.compile(r"\b(?P<num>\d{2})(?P<code>[A-Z]{3,4})(?P<tail>\d+)\b")
        match = pattern.search(hub_id)
        if match:
            code = match.group("code")
        return code
        
    print(f"ℹ️ Total Polygon: {len(polygon_gdf):,}")
    polygon_gdf['name'] = polygon_gdf.index + 1

    if 'region' not in sitelist_gdf.columns:
        concated = pd.concat(sitelist_gdf[['site_id', 'geometry']], hubs_gdf[['site_id', 'geometry']])
        group = auto_group(concated, distance=20000)
        sitelist_gdf = gpd.sjoin(sitelist_gdf, group[['region', 'geometry']]).drop(columns='index_right')
        hubs_gdf = gpd.sjoin(hubs_gdf, group[['region', 'geometry']]).drop(columns='index_right')

    # CONVERT CRS
    sitelist_gdf = sitelist_gdf.to_crs(epsg=3857)
    hubs_gdf = hubs_gdf.to_crs(epsg=3857)
    polygon_gdf = polygon_gdf.to_crs(epsg=3857)

    # JOIN POLYGON
    sites_joined = gpd.sjoin(sitelist_gdf, polygon_gdf[['name', 'geometry']], predicate="intersects").drop(columns='index_right')
    hubs_joined = gpd.sjoin(hubs_gdf, polygon_gdf[['name', 'geometry']], predicate="intersects").drop(columns='index_right')
    sites_joined = sites_joined.rename(columns={'name':'group'})
    hubs_joined = hubs_joined.rename(columns={'name':'group'})

    identified_ring = []
    coded_dict = {}
    for idx, row in tqdm(polygon_gdf.iterrows(), total=len(polygon_gdf), desc="Polygonize Ring"):
        ring = row['name']
        sites_ring = sites_joined[sites_joined['group'] == ring].copy()
        hubs_ring = hubs_joined[hubs_joined['group'] == ring].copy()
        total_hub = len(hubs_ring)

        if total_hub == 0:
            print(f"🟠 Ring {ring} | No hubs found in this polygon. Try nearest")
            ring_pol = polygon_gdf.iloc[[idx]]
            hubs_ring = gpd.sjoin_nearest(hubs_joined, ring_pol, max_distance=5000, distance_col="dist_nearest").drop(columns='index_right')
            
            if not hubs_ring.empty:
                print(f"🟢 Found {len(hubs_ring)} nearest.")
        
        if "flag" not in hubs_ring.columns:
            hubs_ring["flag"] = None

        match total_hub:
            case 1:
                hubs_ring.iloc[0, hubs_ring.columns.get_loc("flag")] = "Start"
                start_hub = hubs_ring.iloc[[0]]
                ring_data = pd.concat([start_hub, sites_ring])

            case 2:
                hubs_ring.iloc[0, hubs_ring.columns.get_loc("flag")] = "Start"
                hubs_ring.iloc[-1, hubs_ring.columns.get_loc("flag")] = "End"
                start_hub = hubs_ring.iloc[[0]]
                end_hub = hubs_ring.iloc[[-1]]
                ring_data = pd.concat([start_hub, sites_ring, end_hub])

            case n if n > 2:
                print(f"🟠 Ring {ring} | Hubs more than 2, selecting first 2 hubs only")
                hubs_ring = hubs_ring.iloc[:2].copy()
                hubs_ring.iloc[0, hubs_ring.columns.get_loc("flag")] = "Start"
                hubs_ring.iloc[-1, hubs_ring.columns.get_loc("flag")] = "End"
                start_hub = hubs_ring.iloc[[0]]
                end_hub = hubs_ring.iloc[[-1]]
                ring_data = pd.concat([start_hub, sites_ring, end_hub])

            case _:
                print(f"🔴 Ring {ring} | No hubs found in this polygon.")
                ring_data = pd.DataFrame()
                continue
            
        region = hubs_ring['region'].mode().values[0]
        hub_id = hubs_ring['site_id'].iat[0]
        code = get_code(hub_id)
        
        if code in coded_dict:
            coded_dict[code] += 1
        else:
            coded_dict[code] = 1

        ring_num = coded_dict[code]
        ring_name = f"TBG-{code}-{project_name}-DF{str(ring_num).zfill(4)}"
        ring_data['region'] = region
        ring_data['code'] = code
        ring_data['ring_name'] = ring_name

        if not ring_data.empty:
            identified_ring.append(ring_data)

    if len(identified_ring) > 0:
        identified_ring = pd.concat(identified_ring)
        identified_ring = identified_ring.to_crs(epsg=4326).reset_index(drop=True)
        identified_ring['long'] = identified_ring.geometry.x
        identified_ring['lat'] = identified_ring.geometry.y
        print(f"✅ Ring poligonize completed.")
        return identified_ring
    else:
        print(f"🔴 Ring data empty.")
        return None

def main_poligonized(excel_path:str, polygon_file:str, export_dir:str, boq:bool=False, spof_threshold:int=3000, **kwargs):
    cable_cost = kwargs.get("cable_cost", 35000)
    vendor = kwargs.get("vendor", "TBG")
    program = kwargs.get("program", "Fiberization")
    method = kwargs.get("method", "Poligon Based")
    sep = kwargs.get("sep", "-")
    task_celery = kwargs.get("task_celery", None)
    design_type = 'Bill of Quantity' if boq else 'Design'

    logger.info(f"🌏 Starting Intersite")
    logger.info(f"ℹ️ Method  : {method}")
    logger.info(f"ℹ️ Vendor  : {vendor}")
    logger.info(f"ℹ️ Program : {program}")
    logger.info(f"ℹ️ Design  : {design_type}")

    sitelist_gdf, hubs_gdf = validate_poligonize(excel_path)
    polygon_gdf = read_gdf(polygon_file, geom_type='polygon')
    poligonized = polygonize_algo(sitelist_gdf, hubs_gdf, polygon_gdf, program)

    site_data = sanitize_header(poligonized)
    site_data = supervised_validation(site_data)

    design_dir = f"{export_dir}/{method.replace(' ', '_')}"
    os.makedirs(design_dir, exist_ok=True)

    if task_celery:
        task_celery.update_state(state="PROGRESS", meta={"status": "Starting Polygon Based Intersite"})

    result = main_supervised(
        site_data=site_data,
        export_loc=design_dir,
        program=program,
        vendor=vendor,
        boq=boq,
        method=method,
        sep=sep,
        spof_threshold=spof_threshold,
        task_celery=task_celery
    )

    return result

if __name__ == "__main__":
    excel_file = r"D:\JACOBS\SERVICE\API\data\template\Template_Polygon_Based.xlsx"
    poligon_file = r"D:\JACOBS\SERVICE\API\data\template\Polygon_Sample.kmz"
    export_dir = r"D:\JACOBS\PROJECT\TASK\DESEMBER\Week 1\Polygon Based"
    program = "Trial Poligonized"
    boq = False

    date_today = datetime.now().strftime("%Y%m%d")
    export_loc = f"{export_dir}/{date_today}"
    os.makedirs(export_loc, exist_ok=True)

    result = main_poligonized(
        excel_path=excel_file,
        polygon_file=poligon_file,
        export_dir=export_dir,
        boq=boq,
        program=program
    )

    zip_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_Poligonized_Task.zip"
    zip_filepath = os.path.join(export_loc, zip_filename)
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(export_loc):
            for file in files:
                if file != zip_filename:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_loc)
                    zipf.write(file_path, arcname)
    logger.info(f"🏆 Result files zipped at {zip_filepath}.")