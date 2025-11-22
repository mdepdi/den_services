import os
import time
import geopandas as gpd
import pandas as pd
import numpy as np

def validate_input(points:gpd.GeoDataFrame, routes:gpd.GeoDataFrame):
    point_columns = ['site_id','ring_name', 'site_name', 'region', 'vendor', 'program']
    paths_columns = ['name','near_end','far_end', 'region', 'vendor', 'program']

    points.columns = points.columns.str.lower()
    routes.columns = routes.columns.str.lower()

    for col in point_columns:
        if col not in points.columns:
            if col == 'site_id':
                raise ValueError(rf"Column {col} not in points data.")
            elif col == 'ring_name':
                raise ValueError(rf"Column {col} not in points data.")
            elif col == 'region':
                raise ValueError(rf"Column {col} not in points data.")
            else:
                print(rf"🟠 Column {col} not in points data.")
                points[col] = "Not Defined"
                

    for col in paths_columns:
        if col not in points.columns:
            if col == 'name':
                raise ValueError(rf"Column {col} not in points data.")
            elif col == 'near_end':
                raise ValueError(rf"Column {col} not in points data.")
            elif col == 'far_end':
                raise ValueError(rf"Column {col} not in points data.")
            elif col == 'region':
                raise ValueError(rf"Column {col} not in points data.")
            else:
                print(rf"🟠 Column {col} not in points data.")
                points[col] = "Not Defined"

    return points, routes

# FORMAT IOH
def report_ioh(points:gpd.GeoDataFrame, routes:gpd.GeoDataFrame):
    points = points.to_crs(epsg=4326)
    routes = routes.to_crs(epsg=4326)
    print(f"📗 Process Report | Indosat Sheet Format")
    print(f"ℹ️ Points Total to process: {len(points):,}")
    print(f"ℹ️ Routes Total to process: {len(routes):,}")

    sheet_point = ['Site ID','Site Name','Long','Lat','SoW','Vendor','Region','Existing/New Site','Segmen ID','Ring ID','Program Name','Program Ring','Program Status','Site Owner','Initial Site ID','Initial Site Name','insert/new ring']
    sheet_length = ["No", "Program", "Region", "Ring ID", "#of Site", "FO Distance (m)", "Vendor", "AVG Length", "Ring Status"]
    sheet_newring = ["No", "Ring ID", "Vendor", "Origin Site ID", "Origin Name", "Long_Origin", "Lat_Origin", "Priority_Origin", "Existing/New Site", "Destination Site ID", "Destination Name", "Long_Destination", "Lat_Destination", "Priority_Destination", "Existing/New Site" , "Link Name", "Ring ID", "RING/STAR", "Ring Status", "City", "Region", "Existing Cable (m)", "New Cable (m)", "Total Distance (m)", "Remark", "Program"]
    
    sheet_point = pd.DataFrame(columns=sheet_point)
    sheet_length = pd.DataFrame(columns=sheet_length)
    sheet_newring = pd.DataFrame(columns=sheet_newring)



