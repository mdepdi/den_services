import pandas as pd
import geopandas as gpd
import numpy as np
import requests
import os
from shapely.geometry import Point, LineString
from sklearn.neighbors import BallTree
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
from time import time
from tbg_modules.geometry import point_coordinates

def graphhopper_routing(start: Point, end: Point, endpoint="http://10.83.10.16:8989", profile='car'):
    url = f"{endpoint}/route"
    params = {
        "point": [f"{start.y},{start.x}", f"{end.y},{end.x}"],
        "profile": profile,
        "locale": "en",
        "points_encoded": "false"
    }

    max_retry = 5
    retry = 0

    while retry < max_retry:
        try:
            r = requests.get(url, params=params, timeout=30)
            data = r.json()

            if "message" in data and "Connection between locations not found" in data["message"]:
                print("⚠️ No route available between these coordinates.")
                return None

            if "paths" not in data or not data["paths"]:
                raise ValueError(f"Missing 'paths' in response: {data}")

            path = data["paths"][0]
            coords = path["points"]["coordinates"]
            distance = path["distance"]
            return LineString(coords)

        except Exception as e:
            print(f"🔴 GraphHopper Routing Error (retry {retry+1}/{max_retry}): {e}")
            retry += 1

    print("❌ Failed after maximum retries")
    return None

def nearest_candidates(source_gdf, target_gdf, k_candidates=10):
    """
    Finds k nearest straight-line candidates (Euclidean in WebMercator)
    Returns DataFrame of all source-target candidate pairs
    """
    source_gdf = source_gdf.to_crs(4326)
    target_gdf = target_gdf.to_crs(4326)
    src = source_gdf.to_crs(3857)
    tgt = target_gdf.to_crs(3857)

    src_coords = np.column_stack((src.geometry.y, src.geometry.x))
    tgt_coords = np.column_stack((tgt.geometry.y, tgt.geometry.x))

    tree = BallTree(tgt_coords, metric="euclidean")
    dist, ind = tree.query(src_coords, k=k_candidates)

    pairs = []
    for src_idx, ids in enumerate(ind):
        for pos, tgt_idx in enumerate(ids):
            pairs.append({
                "src_idx": src_idx,
                "tgt_idx": tgt_idx,
                "src_geom": source_gdf.iloc[src_idx].geometry,
                "tgt_geom": target_gdf.iloc[tgt_idx].geometry
            })

    return pd.DataFrame(pairs)

def graphhopper_parallel(pairs_df, workers=8, profile='car'):
    tasks = {}
    pairs_df = pairs_df.copy()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for idx, row in pairs_df.iterrows():
            source_geom = row['src_geom']
            target_geom = row['tgt_geom']
            task = ex.submit(graphhopper_routing, source_geom, target_geom, profile=profile)
            tasks[task] = idx

        for f in tqdm(as_completed(tasks), total=len(tasks), desc="Routing"):
            result = f.result()
            id = tasks[f]
            pairs_df.at[id, 'geometry'] = result
    route_gdf = gpd.GeoDataFrame(pairs_df, geometry="geometry", crs="EPSG:4326")
    route_gdf['distance'] = route_gdf.geometry.to_crs(epsg=3857).length
    if 'src_geom' in route_gdf.columns:
        route_gdf = route_gdf.drop(columns='src_geom')
    if 'tgt_geom' in route_gdf.columns:
        route_gdf = route_gdf.drop(columns='tgt_geom')
    return route_gdf

def best_route(routing_df:gpd.GeoDataFrame, k_final:int=1):
   final=[]
   for idx, group in routing_df.groupby("src_idx"):
      best = group.sort_values("distance").head(k_final).reset_index()
      best['num'] = best.index + 1
      final.append(best)
   final = pd.concat(final, ignore_index=True)
   return final

def grapphopper_knn(
      source_gdf:gpd.GeoDataFrame,
      target_gdf:gpd.GeoDataFrame,
      k_final:int=None,
      workers:int=10,
      profile:str='car'
):
    if k_final is None:
        k_final = 1
    
    print(f"ℹ️ Running Graphhopper KNN.")
    print(f"ℹ️ Profile: {profile}.")
    start_time = time()
    k_candidate = k_final*2
    pairs_df = nearest_candidates(source_gdf=source_gdf, target_gdf=target_gdf, k_candidates=k_candidate)
    route_gdf = graphhopper_parallel(pairs_df=pairs_df, workers=workers, profile=profile)
    route_gdf = route_gdf.merge(source_gdf[['site_id','lat','long']], left_on='src_idx', right_index=True, suffixes=[None, '_src'])
    route_gdf = route_gdf.merge(target_gdf[['site_id','lat','long']], left_on='tgt_idx', right_index=True, suffixes=[None, '_tgt'])
    best = best_route(route_gdf, k_final=k_final)

    gdf = gpd.GeoDataFrame(best, geometry="geometry", crs="EPSG:4326")
    gdf['length'] = gdf.geometry.to_crs(epsg=3857).length
    end_time = time()
    process_time = end_time-start_time

    print(f"✅ Graphhopper KNN Route Done in {process_time/60:.2f} minutes")
    return gdf

def save_routing(gdf:gpd.GeoDataFrame, export_dir):
    excel_path = os.path.join(export_dir, f"Routing_Result.xlsx")
    parquet_path = os.path.join(export_dir, f"Routing_Result.parquet")
    gdf.to_parquet(parquet_path)
    gdf.drop(columns='geometry').to_excel(excel_path, sheet_name='DEN Graphhopper Routing')
    return parquet_path

def verify_input(excel_file:str):
    basename = os.path.basename(excel_file)
    print(f"🔃 Verify Input for KNN Graphhopper Routing - {basename}")

    with pd.ExcelFile(excel_file) as excel:
        sheetnames = excel.sheet_names
        if 'Source' and 'Target' not in sheetnames:
            source_df = pd.read_excel(excel, sheet_name="Source")
            target_df = pd.read_excel(excel, sheet_name="Target")
        else:
            source_df = pd.read_excel(excel, sheet_name=sheetnames[0])
            target_df = pd.read_excel(excel, sheet_name=sheetnames[-1])

    source_df.columns = source_df.columns.str.lower()
    target_df.columns = target_df.columns.str.lower()
    used_col = ['site_id', 'lat', 'long']
    for col in used_col:
        if col not in source_df.columns:
            raise ValueError(f"Column {col} not found for Source GDF")
        if col not in target_df.columns:
            raise ValueError(f"Column {col} not found for Target GDF")
    source_geom = gpd.points_from_xy(source_df['long'], source_df['lat'], crs='EPSG:4326')
    target_geom = gpd.points_from_xy(target_df['long'], target_df['lat'], crs='EPSG:4326')
    source_gdf = gpd.GeoDataFrame(source_df, geometry=source_geom)
    target_gdf = gpd.GeoDataFrame(target_df, geometry=target_geom)
    print(f"✅ Input Data {basename} valid.")
    return source_gdf, target_gdf

def distance_fiber_tbg(source_gdf:gpd.GeoDataFrame, max_distance=10000):
    fiber = r"D:\JACOBS\DATA\06. FO TBG\Compile FO Route Only June 2025\FO TBG Only_01062025.parquet"
    dirname = os.path.dirname(fiber)
    basename = os.path.basename(fiber).split(".")[0]
    point_path = os.path.join(dirname, f"Points_{basename}.parquet")

    print(f"🌏 Checking Route to Fiber TBG")
    if os.path.exists(point_path):
        print(f"ℹ️ FO Points already exist. Load exist.")
        points_fiber = gpd.read_parquet(point_path)
    else:
        print(f"ℹ️ FO Points didn't exist. Process point coordinates.")
        fiber = gpd.read_parquet(fiber)
        points_fiber = point_coordinates(fiber)
        points_fiber.columns = points_fiber.columns.str.lower()
        points_fiber = points_fiber.drop_duplicates(subset=['name', 'operator', 'geometry']).reset_index(drop=True)
        points_fiber = points_fiber.to_crs(epsg=4326)
        points_fiber['site_id'] = points_fiber['name'] + points_fiber['operator']
        points_fiber['long'] = points_fiber.geometry.x
        points_fiber['lat'] = points_fiber.geometry.y
        points_fiber.to_parquet(point_path)

    source_gdf = source_gdf.to_crs(epsg=3857)
    points_fiber = points_fiber.to_crs(epsg=3857)

    k_final = 1
    source_buff = source_gdf.copy()
    source_buff['geometry'] = source_buff.geometry.buffer(max_distance*1.5)
    points_fiber = gpd.sjoin(points_fiber, source_buff[['geometry']]).drop(columns='index_right')
    points_fiber = points_fiber.reset_index(drop=True)
    
    # RUNNING ROUTING
    routing_gdf = grapphopper_knn(source_gdf, points_fiber, k_final=k_final, profile='pure_shortest')

    # EXPORT DATA
    export_dir = fr"D:\JACOBS\PROJECT\TASK\NOVEMBER\Week 2\Grapphopper Routing\Distance to Fiber"
    date_today = datetime.today().strftime("%Y-%m-%d")
    export_dir = os.path.join(export_dir, date_today)
    os.makedirs(export_dir, exist_ok=True)

    route_path = save_routing(routing_gdf, export_dir)
    source_gdf.to_parquet(os.path.join(export_dir, "Source GDF.parquet"))
    points_fiber.to_parquet(os.path.join(export_dir, "Target GDF.parquet"))
    return route_path

if __name__ == "__main__":
    # # PROCESS ROUTING
    # source_path = r"D:\JACOBS\PROJECT\TASK\NOVEMBER\Week 2\Grapphopper Routing\Template Routing.xlsx"
    # k_final = 5
    # source_gdf, target_gdf = verify_input(source_path)
    # routing_gdf = grapphopper_knn(source_gdf, target_gdf, k_final=k_final, profile='car')

    # EXPORT DATA
    # export_dir = fr"D:\JACOBS\PROJECT\TASK\NOVEMBER\Week 2\Grapphopper Routing\{basename}"
    # date_today = datetime.today().strftime("%Y-%m-%d")
    # export_dir = os.path.join(export_dir, date_today)
    # os.makedirs(export_dir, exist_ok=True)

    # save_routing(routing_gdf, export_dir)
    # source_gdf.to_parquet(os.path.join(export_dir, "Source GDF.parquet"))
    # target_gdf.to_parquet(os.path.join(export_dir, "Target GDF.parquet"))

    # PROCESS DATA OPERASIONAL AKSES INTERNET
    source_path = r"D:\JACOBS\PROJECT\TASK\NOVEMBER\Week 2\Grapphopper Routing\BLENDED DATA OPERASIONAL AKSES INTERNET TO FIBER TBG.xlsx"
    source_df = pd.read_excel(source_path, sheet_name='Total Lokasi')
    print(source_df.head())
    with pd.ExcelFile(source_path) as excel:
        print(excel.sheet_names)
    source_df.columns = source_df.columns.str.lower()
    source_df = source_df.rename(columns={'longitude':'long', 'latitude':'lat', 'site id':'site_id'})
    used_col = ['site_id', 'lat', 'long']
    for col in used_col:
        if col not in source_df.columns:
            raise ValueError(f"Column {col} not found for Source GDF")

    for col in source_df.select_dtypes(include=["object"]).columns:
        source_df[col] = source_df[col].astype(str)
    source_geom = gpd.points_from_xy(source_df['long'], source_df['lat'], crs='EPSG:4326')
    source_gdf = gpd.GeoDataFrame(source_df, geometry=source_geom)
    route_path = distance_fiber_tbg(source_gdf, max_distance=10000)