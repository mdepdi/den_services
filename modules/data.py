import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString, MultiLineString
from fastapi import HTTPException
from starlette.datastructures import UploadFile
from shapely.ops import nearest_points
from tqdm import tqdm
import sys
import os
from modules.geometry import explode_lines, identify_centerline
from modules.h3_route import identify_hexagon, retrieve_roads
from modules.table import find_best_match
from core.config import settings

MAINDATA_DIR = settings.MAINDATA_DIR
DATA_DIR = settings.DATA_DIR
EXPORT_DIR = settings.EXPORT_DIR

def connect_linestring(line_gdf, buffer=50):
    line_crs = line_gdf.crs

    if line_crs is None:
        raise ValueError("Input GeoDataFrame must have a valid CRS.")
    if line_crs != "EPSG:3857":
        line_gdf = line_gdf.to_crs("EPSG:3857")

    buffered = line_gdf.copy()
    buffered["geometry"] = buffered["geometry"].buffer(buffer)
    buffered = buffered.dissolve().explode(ignore_index=True)
    buffered["group"] = buffered.index
    grouped = gpd.sjoin(line_gdf[["geometry"]], buffered[["geometry", "group"]]).drop(
        columns=["index_right"]
    )
    grouped = grouped.dissolve(by="group").reset_index()
    grouped["length"] = grouped["geometry"].length

    # Convert back to original CRS
    if line_crs != "EPSG:3857":
        grouped = grouped.to_crs(line_crs)
    return grouped


def detect_endpoints(line_gdf):
    endpoints = []
    for i, row in line_gdf.iterrows():
        line_data = row.copy()
        geom = row["geometry"]
        if geom.geom_type == "MultiLineString":
            for linestring in geom.geoms:
                start_point = linestring.coords[0]
                line_data["geometry"] = Point(start_point)
                endpoints.append(line_data)
                end_point = linestring.coords[-1]
                line_data["geometry"] = Point(end_point)
                endpoints.append(line_data)
        elif geom.geom_type == "LineString":
            start_point = geom.coords[0]
            end_point = geom.coords[-1]
            line_data["geometry"] = Point(start_point)
            endpoints.append(line_data)
            line_data["geometry"] = Point(end_point)
            endpoints.append(line_data)
        else:
            continue

    endpoints_df = pd.DataFrame(endpoints)
    endpoints_df["geom_wkt"] = endpoints_df["geometry"].apply(lambda geom: geom.wkt)
    count_coords = endpoints_df.groupby("geom_wkt").size().reset_index(name="count")
    endpoints_df = endpoints_df.drop_duplicates(subset="geom_wkt")
    endpoints_df = endpoints_df.merge(count_coords, on="geom_wkt", how="left")
    endpoints_df = endpoints_df[endpoints_df["count"] == 1]
    endpoints_gdf = gpd.GeoDataFrame(
        endpoints_df, geometry="geometry", crs=line_gdf.crs
    )
    return endpoints_gdf


def retrieve_building(
    admin_kabkot,
    one_unit_from_road=True,
    area_building=True,
    aspect_ratio=True,
    parameters=None,
    centroid=False,
    admin = "2024",
):

    print("CONFIGURATION:")
    print(f"Kabkot: {admin_kabkot} | Admin: {admin}")
    print(f"One unit from road  : {one_unit_from_road}")
    print(f"Area building       : {area_building}")
    print(f"Aspect ratio        : {aspect_ratio}\n")

    if parameters is None:
        parameters = {
                "aspect_ratio_value": 0.25,
                "area_building_value": {
                    "min": 25,
                    "max": 500,
                },
            }
    
    building_dir = f"{MAINDATA_DIR}/02. Building/Adm {admin}"
    if one_unit_from_road:
        print("Filtering buildings by proximity to road...")
        file_path = f"{building_dir}/Preprocessed (mte25_lte50_one unit from road)/{admin_kabkot}_aspratio.parquet"
    else:
        file_path = f"{building_dir}/Aspect_Ratio/{admin_kabkot}_aspratio.parquet"

    city_building = gpd.read_parquet(file_path)
    city_building = city_building[~city_building["geometry"].is_empty]
    print(f"Total buildings: {len(city_building):,}")

    if area_building:
        print("Filtering buildings by area...")
        city_building = city_building[
            (city_building["area_in_meters"] > parameters["area_building_value"]["min"])
            & (city_building["area_in_meters"] < parameters["area_building_value"]["max"])
        ]

    if aspect_ratio:
        print("Filtering buildings by aspect ratio...")
        city_building = city_building[city_building["asp_ratio"] > parameters["aspect_ratio_value"]]

    if centroid:
        print("Calculating centroid for each building...")
        city_building["geometry"] = city_building["geometry"].centroid

    return city_building


def read_gdf(file: str = None, **kwargs):
    if isinstance(file, str):
        filename = file
        extension = filename.split(".")[-1].lower()
        try:
            if extension == "parquet":
                gdf = gpd.read_parquet(filename)
            elif extension in ["shp", "geojson", "gpkg", "tab"]:
                if extension == "shp":
                    gdf = gpd.read_file(filename, driver="ESRI Shapefile")
                elif extension == "geojson":
                    gdf = gpd.read_file(filename, driver="GeoJSON")
                elif extension == "gpkg":
                    gdf = gpd.read_file(filename, driver="GPKG")
                elif extension == "tab":
                    gdf = gpd.read_file(filename, driver="MapInfo File")
            elif extension == "xlsx":
                crs = kwargs.get("crs", "EPSG:4326")
                sheet_name = kwargs.get("sheet_name", 0)
                df = pd.read_excel(filename, sheet_name=sheet_name)
                long_col = find_best_match("long", df.columns.tolist(), 0.6)
                lat_col = find_best_match("lat", df.columns.tolist(), 0.6)
                if long_col and lat_col:
                    long_col = long_col[0]
                    lat_col = lat_col[0]
                    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[long_col], df[lat_col]), crs=crs)
                    print(f"Identified longitude column: {long_col}, latitude column: {lat_col}")
                else:
                    raise ValueError("DataFrame must contain 'long' and 'lat' columns.")
            elif extension == "csv":
                crs = kwargs.get("crs", "EPSG:4326")
                df = pd.read_csv(filename)
                long_col = find_best_match("long", df.columns.tolist())
                lat_col = find_best_match("lat", df.columns.tolist())
                if long_col and lat_col:
                    long_col = long_col[0]
                    lat_col = lat_col[0]
                    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[long_col], df[lat_col]), crs=crs)
                else:
                    raise ValueError("DataFrame must contain 'long' and 'lat' columns.")
            else:
                raise ValueError("Unsupported file format. Supported formats are: Parquet, GeoJSON, Shapefile, GPKG, and TAB.")
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")
    else:
        raise ValueError("File must be a string representing the file path.")

    return gdf


def read_df(file: UploadFile | str = None):

    if isinstance(file, str):
        filename = file
    elif isinstance(file, UploadFile):
        if not file.filename:
            raise HTTPException(status_code=400, detail="File must have a filename.")
        filename = file.filename
    else:
        raise HTTPException(
            status_code=400, detail="Invalid file type. Must be a string or UploadFile."
        )

    extension = filename.split(".")[-1].lower()
    if extension not in ["xlsx", "csv"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Supported formats are: XLSX and CSV.",
        )

    try:
        if extension == "xlsx":
            df = pd.read_excel(file.file)
        elif extension == "csv":
            df = pd.read_csv(file.file)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Supported formats are: XLSX and CSV.",
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    return df

# def fiber_utilization(data: gpd.GeoDataFrame, target_fiber: gpd.GeoDataFrame=None, overlap=True, nodes=None, roads=None) -> gpd.GeoDataFrame:
#     from modules.geometry import explode_lines
#     from shapely.ops import linemerge
#     from tqdm import tqdm

#     print("🧩 Fiber Utilization Analysis ...")
#     if target_fiber is None:
#         target_fiber = gpd.read_parquet(f"{DATA_DIR}/FO TBG Only_01062025.parquet")

#     # PREPARE DATA
#     print("Preparing data...")
#     data = data.reset_index(drop=True)
#     data['num'] = data.index + 1
#     if overlap:
#         data = explode_lines(data)
#     else:
#         data, point_coords = identify_centerline(data, tolerance=0.5)
#         data = data.drop_duplicates(subset='geometry')
#     data = data.reset_index(drop=True)

#     data = data.to_crs(epsg=3857)
#     target_fiber = target_fiber.to_crs(epsg=3857)
    
#     if 'ring_name' in data.columns:
#         ring_list = data['ring_name'].dropna().unique().tolist()
#     else:
#         data_buff = data.copy()
#         data_buff['geometry'] = data_buff.geometry.buffer(5000)
#         data_buff = data_buff.dissolve().explode(ignore_index=True)
#         data_buff['ring_name'] = data_buff.index.apply(lambda x: f"Ring_{x+1}")
#         data = gpd.sjoin(data, data_buff[['geometry', 'ring_name']], how='left', predicate='intersects').drop(columns=['index_right'])
#         ring_list = data['ring_name'].dropna().unique().tolist()
    
#     if 'ref_fo' not in target_fiber.columns:
#         fiber_buff = target_fiber.copy()
#         fiber_buff['geometry'] = fiber_buff.buffer(20)

#     calculated = []
#     for ring in tqdm(ring_list, desc=f'Fiber Utilization Analysis'):
#         ring_data = data[data['ring_name'] == ring].reset_index(drop=True)
        
#         # ROADS
#         hex_list = identify_hexagon(ring_data, type="convex")
#         if not hex_list:
#             raise ValueError("No hexagons identified. Please check the input data.")
        
#         # print("Retrieving roads and nodes...")
#         if roads is None:
#             roads = retrieve_roads(hex_list, type="roads")
#         if nodes is None:
#             nodes = retrieve_roads(hex_list, type="nodes")

#         # CRS
#         target_fiber = target_fiber.to_crs(epsg=3857)
#         roads = roads.to_crs(epsg=3857)
#         nodes = nodes.to_crs(epsg=3857)

#         # FO BUFFER
#         if 'ref_fo' not in ring_data.columns:
#             nodes_sindex = nodes.sindex
#             ring_data['nearest_node'] = ring_data['geometry'].apply(lambda geom: nodes.loc[nodes_sindex.nearest(geom)[1][0], 'node_id'])

#             #  REF FO
#             ref_fo = gpd.sjoin(nodes, fiber_buff[['geometry']], how='inner', predicate='intersects')
#             ref_fo = ref_fo['node_id'].unique().tolist()
#             ring_data['ref_fo'] = ring_data['nearest_node'].isin(ref_fo).astype(int)

#         ring_data['fo_note'] = ring_data.apply(lambda x: 'Existing' if x['ref_fo'] == 1 else 'New', axis=1)
#         ring_data = ring_data.dissolve(by=['num', 'fo_note']).reset_index()
#         ring_data['geometry'] = ring_data.geometry.apply(lambda geom: linemerge(geom) if geom.geom_type == 'MultiLineString' else geom)
#         ring_data['length'] = ring_data['geometry'].length.round(3)
#         calculated.append(ring_data)

#     calculated = pd.concat(calculated, ignore_index=True)
#     calculated_gdf = gpd.GeoDataFrame(calculated, geometry='geometry')
#     return calculated_gdf

def fiber_utilization(data_gdf: gpd, overlap:bool=True):
    target_fiber = gpd.read_parquet(f"{DATA_DIR}/FO TBG Only_01062025.parquet")
    data_gdf = data_gdf.reset_index(drop=True)
    if overlap:
        data_gdf = explode_lines(data_gdf)
    else:
        data_gdf, point_coords = identify_centerline(data_gdf, tolerance=0.5)
        data_gdf = data_gdf.drop_duplicates(subset='geometry')
    data_gdf = data_gdf.reset_index(drop=True)
    data_gdf = data_gdf.to_crs(epsg=3857)

    target_fiber.columns = target_fiber.columns.str.lower()
    target_fiber = target_fiber[['name', 'remark', 'operator', 'geometry']]
    target_fiber = target_fiber.rename(columns={'name':'fiber'})
    target_fiber = target_fiber.to_crs(epsg=3857)
    target_fiber['geometry']  = target_fiber['geometry'].buffer(20)
    
    data_gdf = data_gdf.reset_index(drop=True)
    data_gdf['num'] = data_gdf.index + 1

    existing = []
    new = []
    ringlist = data_gdf['ring_name'].unique().tolist()
    for num, ring in tqdm(enumerate(ringlist), total=len(ringlist), desc=f'Fiber Utilization Analysis'):
        ring_data = data_gdf[data_gdf['ring_name'] == ring].reset_index(drop=True)
        if 'fo_note' in ring_data.columns:
            ring_data.drop(columns='fo_note')

        fo_intersects = gpd.overlay(ring_data, target_fiber[['fiber', 'geometry']], how='intersection')
        fo_not_intersects = gpd.overlay(ring_data, target_fiber[['fiber', 'geometry']], how='difference')

        if not fo_intersects.empty:
            fo_intersects['length'] = fo_intersects.geometry.to_crs(epsg=3857).length
            fo_intersects = fo_intersects.sort_values('length', ascending=False)
            fo_intersects = fo_intersects.dissolve(by='fiber')
            fo_intersects['length'] = fo_intersects.geometry.to_crs(epsg=3857).length

        if not fo_not_intersects.empty:
            fo_not_intersects['length'] = fo_not_intersects.geometry.to_crs(epsg=3857).length
            fo_not_intersects = fo_not_intersects.sort_values('length', ascending=False)
            fo_not_intersects = fo_not_intersects.dissolve(by='fiber')
            fo_not_intersects['length'] = fo_not_intersects.geometry.to_crs(epsg=3857).length

        existing.append(fo_intersects)
        new.append(fo_not_intersects)
        # if num == 2:
        #     break

    existing = pd.concat(existing, ignore_index=True)
    existing_gdf = gpd.GeoDataFrame(existing, geometry='geometry')
    existing_gdf['fo_note'] = "Existing"

    new = pd.concat(new, ignore_index=True)
    new_gdf = gpd.GeoDataFrame(new, geometry='geometry')
    new_gdf['fo_note'] = "New"

    compiled = pd.concat([existing_gdf, new_gdf])
    compiled = gpd.GeoDataFrame(compiled, geometry='geometry')
    compiled = compiled.sort_values('ring_name')

    return compiled