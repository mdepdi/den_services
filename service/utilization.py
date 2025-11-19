import pandas as pd
import geopandas as gpd
import re
import os
import sys

sys.path.append(r"D:\Data Analytical\SERVICE\API")

from tqdm import tqdm
from shapely.ops import linemerge
from concurrent.futures import ProcessPoolExecutor, as_completed
from modules.geometry import explode_lines, identify_centerline
from modules.h3_route import retrieve_roads, identify_hexagon
from modules.utils import auto_group
from modules.data import read_gdf
from core.config import settings

# =======
# MODULES
# =======
def fiber_utilization(data_region:gpd.GeoDataFrame, target_fiber: gpd.GeoDataFrame):
    # ROADS
    hex_list = identify_hexagon(data_region, type="convex")
    print(f"ℹ️ Identified {len(hex_list)} hexagons.")
    if not hex_list:
        raise ValueError("No hexagons identified. Please check the input data.")
    
    print("Retrieving roads and nodes...")
    roads = retrieve_roads(hex_list, type="roads")
    nodes = retrieve_roads(hex_list, type="nodes")

    # CRS
    print(f"ℹ️ Total Data Region: {len(data_region)}")
    data_region = data_region.to_crs(epsg=3857)
    target_fiber = target_fiber.to_crs(epsg=3857)
    roads = roads.to_crs(epsg=3857)
    nodes = nodes.to_crs(epsg=3857)

    data_region = data_region.reset_index(drop=True)
    data_region['num'] = data_region.index + 1
    print(f"ℹ️ Total Data Region: {len(data_region)}")

    # FO BUFFER
    if 'ref_fo' not in data_region.columns:
        target_fiber = target_fiber.copy()
        target_fiber['geometry'] = target_fiber.buffer(30)
        # data_buff = data_region.copy()
        # data_buff['geometry'] = data_buff.buffer(20)
        data_region = gpd.sjoin_nearest(data_region, roads[['geometry', 'node_start', 'node_end']]).drop(columns='index_right')

        #  REF FO
        print("ℹ️ Finding reference FO...")
        ref_fo = gpd.sjoin(nodes, target_fiber[['geometry']], how='inner', predicate='intersects')
        ref_fo = ref_fo['node_id'].unique().tolist()
        data_region['ref_fo'] = (data_region['node_start'].isin(ref_fo) & data_region['node_end'].isin(ref_fo)).astype(int)

    print(f"ℹ️ Total Data Region: {len(data_region)}")
    data_region['fo_note'] = data_region.apply(lambda x: 'Existing' if x['ref_fo'] == 1 else 'New', axis=1)
    data_region = data_region.dissolve(by=['num', 'fo_note']).explode(ignore_index=True)
    data_region['geometry'] = data_region.geometry.apply(lambda geom: linemerge(geom) if geom.geom_type == 'MultiLineString' else geom)
    data_region['length'] = data_region['geometry'].length.round(3)
    print(f"ℹ️ Total Data Region: {len(data_region)}")
    return data_region

def validate_poligonize(excel_path:str):
    with pd.ExcelFile(excel_path) as excel:
        sheetname = excel.sheet_names
        used_sheet = ['sitelist', 'hubs']
        for sheet in used_sheet:
            if sheet not in sheetname:
                raise ValueError(f"Sheet {sheet} not found in Excel file. Check your input.")
        
        sitelist = pd.read_excel(excel, sheet_name='sitelist')
        hubs = pd.read_excel(excel, sheet_name='hubs')

        print(f"ℹ️ Total sitelist: {len(sitelist):,}")
        print(f"ℹ️ Total hubs    : {len(hubs):,}")

        used_col = ['site_id', 'long', 'lat', 'site_type']
        for col in used_col:
            if col not in sitelist.columns:
                raise ValueError(f"Column {col} not found in Sitelist. Check your input.")
            if col not in hubs.columns:
                raise ValueError(f"Column {col} not found in Hubs. Check your input.")
        
        sitelist['site_id'] = sitelist['site_id'].astype(str) 
        hubs['site_id'] = hubs['site_id'].astype(str) 
        sitelist['site_name'] = sitelist['site_name'].astype(str) 
        hubs['site_name'] = hubs['site_name'].astype(str) 
        sitelist_geom = gpd.points_from_xy(sitelist['long'], sitelist['lat'], crs="EPSG:4326")
        hubs_geom = gpd.points_from_xy(hubs['long'], hubs['lat'], crs='EPSG:4326')

        sitelist_gdf = gpd.GeoDataFrame(sitelist, geometry=sitelist_geom)
        hubs_gdf = gpd.GeoDataFrame(hubs, geometry=hubs_geom)
        hubs_gdf['site_type'] = "FO Hub"
        
    return sitelist_gdf, hubs_gdf

def polygonize_ring(sitelist_gdf:gpd.GeoDataFrame, hubs_gdf:gpd.GeoDataFrame, polygon_gdf:gpd.GeoDataFrame, project_name:str):
    def get_code(hub_id:str):
        code = None
        pattern = re.compile(r"\b(?P<num>\d{2})(?P<code>[A-Z]{3})(?P<tail>\d+)\b")
        match = pattern.search(hub_id)
        if match:
            code = match.group("code")
        return code
        
    print(f"ℹ️ Total Polygon: {len(polygon_gdf):,}")
    polygon_gdf['name'] = polygon_gdf.index + 1

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
        print(f"✅ Ring polygonize completed.")
        return identified_ring
    else:
        print(f"🔴 Ring data empty.")
        return None

# ====
# MAIN
# ====
def main_fiber_utilization(data: gpd.GeoDataFrame, target_fiber: gpd.GeoDataFrame=None, overlap=True, nodes=None, roads=None) -> gpd.GeoDataFrame:
    print("🧩 Fiber Utilization Analysis ...")
    if target_fiber is None:
        DATA_DIR = settings.DATA_DIR
        target_fiber = gpd.read_parquet(f"{DATA_DIR}/FO TBG Only_01062025.parquet")

    # PREPARE DATA
    print("Preparing data...")
    if overlap:
        data = explode_lines(data)
    else:
        data, point_coords = identify_centerline(data, tolerance=0.5)
        data = data.drop_duplicates(subset='geometry')

    # CRS
    data = data.to_crs(epsg=3857)
    target_fiber = target_fiber.to_crs(epsg=3857)
    if 'region' in target_fiber.columns:
        target_fiber = target_fiber.drop(columns='region')

    # GROUPING
    data = auto_group(data)
    data = data.reset_index(drop=True)

    region_list = data['region'].unique().tolist()
    print(f"ℹ️ Total Region to process: {len(region_list)}")

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {}
        for region in region_list:
            region_data = data[data['region'] == region].copy()
            print(f"ℹ️ Total Data Input: {len(region_data)}")

            future = executor.submit(fiber_utilization, region_data, target_fiber)
            futures[future] = region

        compiled = []
        for future in as_completed(futures):
            region = futures[future]
            identified_region = future.result()

            if not identified_region.empty:
                compiled.append(identified_region)
                print(f"🟢 Region {region} done.")
            else:
                raise ValueError(f"🔴 Empty result in region {region}")
    compiled = pd.concat(compiled)
    print(f"✅ All data already processed.")
    return compiled

def main_identify_centerline(data_gdf:gpd.GeoDataFrame):
    print("🧩 Indentify Centerline ...")

    # PREPARE DATA
    print("Preparing data...")
    data_gdf = data_gdf.reset_index(drop=True)
    data_gdf['num'] = data_gdf.index + 1

    # GROUPING
    data_gdf = auto_group(data_gdf)
    data_gdf = data_gdf.reset_index(drop=True)

    region_list = data_gdf['region'].unique().tolist()
    print(f"ℹ️ Total Region to process: {len(region_list)}")

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {}
        for region in region_list:
            region_data = data_gdf[data_gdf['region'] == region].copy()
            future = executor.submit(identify_centerline, region_data, tolerance=1)
            futures[future] = region

        compiled_centerline = []
        for future in as_completed(futures):
            region = futures[future]
            identified_region = future.result()

            if not identified_region.empty:
                compiled_centerline.append(identified_region)
                print(f"🟢 Region {region} done.")
            else:
                raise ValueError(f"🔴 Empty result in region {region}")

    compiled_centerline = pd.concat(compiled_centerline)
    print(f"✅ All data already processed.")
    return compiled_centerline

def main_polygonize_ring(excel_file:str, polygon_file:str, project_name):
    sitelist_gdf, hubs_gdf = validate_poligonize(excel_file)
    polygon_gdf = read_gdf(polygon_file)
    poligonized = polygonize_ring(sitelist_gdf, hubs_gdf, polygon_gdf, project_name)
    print(f"✅ Polygonized success")
    return poligonized

if __name__=="__main__":
    
    # # POLYGONIZED RING
    # excel_file = r"D:\Data Analytical\PROJECT\TASK\SEPTEMBER\Week 5\IOH RING PROCESS\Template_Unsupervised_New site 2026 v1.2 - Combined.xlsx"
    # polygon_file = r"D:\Data Analytical\PROJECT\TASK\SEPTEMBER\Week 5\IOH RING PROCESS\Polygon Part 1.parquet"
    # project_name = "NewSite2026"
    # poligonized = main_polygonize_ring(excel_file, polygon_file, project_name)

    # export_dir = r"D:\Data Analytical\PROJECT\TASK\SEPTEMBER\Week 5\IOH RING PROCESS"
    # os.makedirs(export_dir, exist_ok=True)

    # poligonized.to_parquet(os.path.join(export_dir, f"Poligonized Ring {project_name}.parquet"))
    # poligonized.drop(columns='geometry').to_excel(os.path.join(export_dir, f"Poligonized Ring {project_name}.xlsx"), index=False)

    # FIBER UTILIZATION
    # # POLYGONIZED RING
    routes = gpd.read_parquet(r"D:\Data Analytical\PROJECT\TASK\OKTOBER\Week 4\Update BOQ\Export\Paths not Star.parquet")
    path_utilized = main_fiber_utilization(routes, overlap=True)

    path_utilized.to_parquet(r"D:\Data Analytical\PROJECT\TASK\OKTOBER\Week 4\Update BOQ\Export\Paths FO Utilized.parquet")