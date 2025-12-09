import os
import sys
import zipfile
import geopandas as gpd
import pandas as pd
import networkx as nx
import re
from datetime import datetime
from time import time
from shapely.geometry import Point
from shapely.ops import linemerge
from tqdm import tqdm

sys.path.append(r"D:\JACOBS\SERVICE\API")

from service.intersite.boq_algorithm import main_boq
from service.intersite.fixroute_algorithm import main_fixroute, validate_fixroute
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
def validate_topology(excel_path:str):
    with pd.ExcelFile(excel_path) as excel:
        sheetname = excel.sheet_names
        sitelist = pd.read_excel(excel)
        print(f"ℹ️ Total sitelist: {len(sitelist):,}")

        used_col = ['site_id', 'long', 'lat']
        for col in used_col:
            if col not in sitelist.columns:
                raise ValueError(f"Column {col} not found in Sitelist. Check your input.")
        
        sitelist['site_id'] = sitelist['site_id'].astype(str) 
        sitelist['site_name'] = sitelist['site_name'].astype(str)

        # LONG AND LAT VALIDATION
        sitelist = validate_longlat(sitelist, lon_col="long", lat_col="lat")
        sitelist_geom = gpd.points_from_xy(sitelist['long'], sitelist['lat'], crs="EPSG:4326")
        sitelist_gdf = gpd.GeoDataFrame(sitelist, geometry=sitelist_geom, crs="EPSG:4326")
    return sitelist_gdf

def topology_algo(sitelist_gdf:gpd.GeoDataFrame, line_gdf:gpd.GeoDataFrame, vendor='TBG', program='Intersite'):
    line_gdf = line_gdf.to_crs(epsg=3857)
    sitelist_gdf = sitelist_gdf.to_crs(epsg=3857)

    # ----------------------------------------------------------------------
    # Ensure Required Columns
    # ----------------------------------------------------------------------
    if "site_id" not in sitelist_gdf.columns:
        sitelist_gdf = sitelist_gdf.reset_index(drop=True)
        sitelist_gdf['site_id'] = sitelist_gdf.index + 1

    if "site_name" not in sitelist_gdf.columns:
        sitelist_gdf['site_name'] = sitelist_gdf['site_id']

    if 'region' not in sitelist_gdf.columns:
        group = auto_group(sitelist_gdf)
        sitelist_gdf = gpd.sjoin(sitelist_gdf, group[['region', 'geometry']], how='left').drop(columns='index_right')
        sitelist_gdf['region'] = "Area" + "_" + sitelist_gdf['region'].astype(str)

    # ----------------------------------------------------------------------
    # Validate line geometries
    # ----------------------------------------------------------------------
    if not any(g in ['LineString', 'MultiLineString'] for g in line_gdf.geom_type):
        raise ValueError(f"Invalid Line Data (found: {line_gdf.geom_type.tolist()})")

    # Explode any MultiLines
    line_gdf = line_gdf.explode(ignore_index=True)

    # ----------------------------------------------------------------------
    # Extract topology points
    # ----------------------------------------------------------------------
    point_topology = []

    for idx, row in tqdm(line_gdf.iterrows(), total=len(line_gdf), desc="Extract Topology Coordinate"):
        geom = row.geometry

        if geom.geom_type == 'MultiLineString':
            merged = linemerge(geom)
            if merged.geom_type == "MultiLineString":
                for part in merged:
                    coords = list(part.coords)
                    for num, coord in enumerate(coords):
                        point = Point(coord)
                        last_idx = len(coords) - 1

                        flag = 'Start' if num == 0 else 'End' if num == last_idx else 'Middle'
                        site_type = "FO Hub" if flag in ['Start', 'End'] else "Site List"

                        point_topology.append({
                            'ring_id': idx,
                            'ring_name': f"{vendor}-{program}-{idx + 1}",
                            'num': num,
                            'site_type': site_type,
                            'flag': flag,
                            'long': point.x,
                            'lat': point.y,
                            'geometry': point
                        })
                continue
            else:
                geom = merged

        # Handle normal line
        coords = list(geom.coords)
        last_idx = len(coords) - 1

        for num, coord in enumerate(coords):
            point = Point(coord)
            flag = 'Start' if num == 0 else 'End' if num == last_idx else 'Middle'
            site_type = "FO Hub" if flag in ['Start', 'End'] else "Site List"

            point_topology.append({
                'ring_id': idx,
                'ring_name': f"{vendor}-{program}-{idx + 1}",
                'num': num,
                'site_type': site_type,
                'flag': flag,
                'long': point.x,
                'lat': point.y,
                'geometry': point
            })

    point_topology = gpd.GeoDataFrame(
        point_topology, geometry='geometry', crs=line_gdf.crs
    )

    # ----------------------------------------------------------------------
    # Map to nearest sitelist
    # ----------------------------------------------------------------------
    sitelist_gdf['centroid'] = sitelist_gdf.geometry.centroid
    mapped = gpd.sjoin_nearest(
        point_topology,
        sitelist_gdf[['site_id', 'site_name', 'region', 'centroid', 'geometry']],
        max_distance=2000,
        distance_col='dist_to_site'
    ).drop(columns='index_right')

    mapped['geometry'] = mapped['centroid']
    mapped = mapped.drop(columns='centroid')

    # Ensure sequential ordering
    mapped = mapped.sort_values(["ring_name", "ring_id", "site_type", "num"])
    mapped = mapped.drop_duplicates(['ring_name', 'site_id'])

    # Convert to lat/lon
    mapped_ll = mapped.to_crs(4326)
    mapped['long'] = mapped_ll.geometry.x
    mapped['lat'] = mapped_ll.geometry.y

    # ----------------------------------------------------------------------
    # Build sequential fiber segments
    # ----------------------------------------------------------------------
    mapped_fix = []
    for ring in mapped['ring_name'].unique():
        ring_data = mapped[mapped['ring_name'] == ring].copy()
        region = ring_data['region'].mode()[0]

        ring_data = ring_data.sort_values("num").reset_index(drop=True)
        total_data = len(ring_data)

        for i in range(total_data - 1):
            a = ring_data.iloc[i].drop(['geometry', 'region', 'ring_name'])
            b = ring_data.iloc[i+1].drop(['geometry', 'region', 'ring_name'])

            # Concatenate horizontally
            seg = pd.concat(
                [a.add_suffix('_a'), b.add_suffix('_b')],
                axis=0
            ).to_frame().T

            seg['ring_name'] = ring
            seg['region'] = region
            seg['sequence'] = i

            mapped_fix.append(seg)
    mapped_fix = pd.concat(mapped_fix, ignore_index=True)
    mapped = mapped.drop_duplicates(['ring_name', 'site_id'])
    mapped = mapped.drop_duplicates(['ring_name', 'geometry'])
    mapped = mapped.reset_index(drop=True)
    return mapped_fix

def main_topology(excel_path:str, line_file:str, export_dir:str, boq:bool=False, spof_threshold:int=3000, **kwargs):
    cable_cost = kwargs.get("cable_cost", 35000)
    vendor = kwargs.get("vendor", "TBG")
    program = kwargs.get("program", "Fiberization")
    method = kwargs.get("method", "Topology Based")
    sep = kwargs.get("sep", "-")
    task_celery = kwargs.get("task_celery", None)
    design_type = 'Bill of Quantity' if boq else 'Design'

    logger.info(f"🌏 Starting Intersite")
    logger.info(f"ℹ️ Method  : {method}")
    logger.info(f"ℹ️ Vendor  : {vendor}")
    logger.info(f"ℹ️ Program : {program}")
    logger.info(f"ℹ️ Design  : {design_type}")

    sitelist_gdf = validate_topology(excel_path)
    line_gdf = read_gdf(line_file, geom_type='line')
    if task_celery:
        task_celery.update_state(state="PROGRESS", meta={"status": "Mapping Topology to Sitelist"})

    site_data = topology_algo(sitelist_gdf, line_gdf, vendor, program)
    site_data = sanitize_header(site_data)
    # site_data = validate_fixroute(site_data)

    date_today = datetime.now().strftime("%Y%m%d")
    export_loc = f"{export_dir}/{date_today}"
    os.makedirs(export_loc, exist_ok=True)

    if task_celery:
        task_celery.update_state(state="PROGRESS", meta={"status": "Starting Fix Route Topology Based"})
    result = main_fixroute(
        template_df=site_data,
        export_dir=export_loc,
        program=program,
        vendor=vendor,
        spof_threshold=spof_threshold,
        boq=boq,
        sep=sep,
        method=method,
        task_celery=task_celery
    )

    return result

if __name__ == "__main__":
    excel_file = r"D:\JACOBS\PROJECT\TASK\DESEMBER\Week 2\Debug Fix Route\Sample\Template_Topology_Based.xlsx"
    line_file = r"D:\JACOBS\PROJECT\TASK\DESEMBER\Week 2\Debug Fix Route\Sample\TBG-001FWASG25-KTSRG-002A.kmz"
    export_dir = r"D:\JACOBS\PROJECT\TASK\DESEMBER\Week 2\Debug Fix Route\Sample\Export"
    program = "Trial Topology"
    boq = False

    date_today = datetime.now().strftime("%Y%m%d")
    export_loc = f"{export_dir}/{date_today}"
    os.makedirs(export_loc, exist_ok=True)

    result = main_topology(
        excel_path=excel_file,
        line_file=line_file,
        export_dir=export_loc,
        boq=boq,
        program=program
    )

    zip_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_Topology_Task.zip"
    zip_filepath = os.path.join(export_loc, zip_filename)
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(export_loc):
            for file in files:
                if file != zip_filename:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_loc)
                    zipf.write(file_path, arcname)
    logger.info(f"🏆 Result files zipped at {zip_filepath}.")