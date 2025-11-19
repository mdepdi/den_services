import os
import re
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString, MultiLineString

# ===============
# ROUTE
# ===============
def count_coordinates(paths):
    coord_count = {}
    
    for geom in paths.geometry:
        if geom.is_empty:
            print("Empty geometry found.")
            continue

        if isinstance(geom, MultiLineString):
            for line in geom.geoms:
                coords = line.coords
                for coord in coords:
                    if coord not in coord_count:
                        coord_count[coord] = 1
                    else:
                        coord_count[coord] += 1
        elif isinstance(geom, LineString):
            coords = geom.coords
            for coord in coords:
                if coord not in coord_count:
                    coord_count[coord] = 1
                else:
                    coord_count[coord] += 1
        else:
            try:
                coords = geom.coords
                for coord in coords:
                    if coord not in coord_count:
                        coord_count[coord] = 1
                    else:
                        coord_count[coord] += 1
            except Exception as e:
                print(f"Warning: Could not extract coordinates from geometry type {type(geom)}: {e}")
                continue
    
    counted_gdf = gpd.GeoDataFrame(
        {"coordinates": list(coord_count.keys()), "count": list(coord_count.values())},
        geometry=[Point(coord) for coord in coord_count.keys()],
        crs=paths.crs,
    )
    return counted_gdf


def duplicate_scores(paths):
    paths_coords = count_coordinates(paths)
    paths_double = paths_coords[paths_coords['count'] > 1].copy()

    duplicate_point = 0

    for coord, count in paths_double.iterrows():
        if count['count'] > 1:
            duplicate_point += count['count'] ** 2
    
    return duplicate_point

def fiber_utilization_score(roads_ref_fo, paths_nodes):
    if roads_ref_fo.crs != 'EPSG:3857':
        roads_ref_fo = roads_ref_fo.to_crs('EPSG:3857')

    fo_used = roads_ref_fo[roads_ref_fo['node_start'].isin(paths_nodes) & roads_ref_fo['node_end'].isin(paths_nodes)].copy()
    fo_used['length'] = fo_used.geometry.length
    total_length = fo_used['length'].sum()
    return total_length