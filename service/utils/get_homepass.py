import os
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from datetime import datetime
from modules.h3_route import identify_hexagon, retrieve_building
from modules.data import read_gdf

def get_homepass(data_gdf:gpd.GeoDataFrame=None, admin_dict:dict=None, one_unit=False, centroid=True):    
    
    # ======================
    # BY GDF AREA
    # ======================
    if data_gdf is not None:
        admin_dict = None
        print(f"ℹ️ Get Homepass | Geospatial File Based")

        # IDENTIFY HEXAGONS
        hex_list = identify_hexagon(data_gdf, type='single')
        print(f"ℹ️ Total Hex to be processed: {len(hex_list)}\n")

        # PROCESS BUILDING
        data_gdf = data_gdf.to_crs(epsg=3857)
        homepass_all = retrieve_building(hex_list, one_unit=one_unit, centroid=centroid)
        homepass_all = homepass_all.to_crs(epsg=3857)
        homepass_all = gpd.sjoin(homepass_all, data_gdf[['geometry']], how='inner').drop(columns=['index_right'])

    # ======================
    # PROCESS ADMIN BASED
    # ======================
    if admin_dict is not None:
        print(f"ℹ️ Get Homepass | Admin Based")
        provinsi = admin_dict.get('provinsi', [])
        kabkot = admin_dict.get('kabkot', [])
        kecamatan = admin_dict.get('kecamatan', [])

        if len(kecamatan) > 0:
            data_gdf = gpd.read_parquet(r"D:\JACOBS\DATA\01. Admin\Admin_2024_v3_Kecamatan.parquet")
            data_gdf = data_gdf.to_crs(epsg=3857).reset_index()
            print(data_gdf.columns)
        else:
            data_gdf = gpd.read_parquet(r"D:\JACOBS\DATA\01. Admin\Admin_2024_Kabkot.parquet")
            data_gdf = data_gdf.to_crs(epsg=3857)
        
        if len(provinsi) > 0:
            data_gdf = data_gdf[data_gdf['Provinsi'].isin(provinsi)].copy()
        if len(kabkot) > 0:
            data_gdf = data_gdf[data_gdf['Kabkot'].isin(kabkot)].copy()
        if len(kecamatan) > 0:
            data_gdf = data_gdf[data_gdf['Kecamatan'].isin(kecamatan)].copy()

        # IDENTIFY HEXAGONS
        hex_list = identify_hexagon(data_gdf, type='single', buffer=100)
        print(f"ℹ️ Total Hex to be processed: {len(hex_list)}\n")

        # PROCESS BUILDING
        homepass_all = retrieve_building(hex_list, type='single', one_unit=one_unit)
        homepass_all = homepass_all.to_crs(epsg=3857)
        homepass_all = gpd.sjoin(homepass_all, data_gdf[['geometry']], how='inner', predicate='within').drop(columns=['index_right'])
    return homepass_all

if __name__ == "__main__":
    data_excel = r"D:\JACOBS\PROJECT\TASK\DESEMBER\Week 2\Alfa Store PKP Identify Fasad\TBG Sitelist_Compile_Cek FWA_Dec 2025_v2.xlsx"
    data_gdf = read_gdf(data_excel, sheet_name='Alfa')
    homepass = get_homepass(data_gdf, centroid=False)
    homepass.to_parquet(r"D:\JACOBS\PROJECT\TASK\DESEMBER\Week 2\Alfa Store PKP Identify Fasad\Alfa Building.parquet")