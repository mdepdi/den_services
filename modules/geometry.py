import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Point, LineString, MultiLineString, Polygon, MultiPolygon

def explode_lines(gdf):
    """Explode lines into individual segments."""
    exploded = []

    for _, row in gdf.iterrows():
        row_data = row.to_dict()
        geom = row['geometry']
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            for i in range(len(coords) - 1):
                new_segment = LineString([coords[i], coords[i + 1]])
                exploded.append({**row_data, 'geometry': new_segment})
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                coords = list(line.coords)
                for i in range(len(coords) - 1):
                    new_segment = LineString([coords[i], coords[i + 1]])
                    exploded.append({**row_data, 'geometry': new_segment})
        else:
            exploded.append(row_data)
    return gpd.GeoDataFrame(exploded, crs=gdf.crs)

def point_coordinates(gdf):
    """Extract ceach coordinates geometry in the GeoDataFrame."""
    point_coords = []
    for idx, row in gdf.iterrows():
        geom = row['geometry']
        data = row.to_dict()
        if isinstance(geom, Point):
            coords = [(x, y) for x, y, *_ in geom.coords]
            point_coords.extend([{'x': x, 'y': y, **data, 'geometry': Point(x,y)} for x, y in coords])
        elif isinstance(geom, LineString):
            coords = [(x, y) for x, y, *_ in geom.coords]
            point_coords.extend([{'x': x, 'y': y, **data, 'geometry': Point(x,y)} for x, y in coords])
        elif isinstance(geom, MultiLineString):
            for line in geom:
                coords = [(x, y) for x, y, *_ in line.coords]
                point_coords.extend([{'x': x, 'y': y, **data, 'geometry': Point(x,y)} for x, y in coords])
        elif isinstance(geom, Polygon):
            exterior_coords = [(x, y) for x, y, *_ in geom.exterior.coords]
            point_coords.extend([{'x': x, 'y': y, **data, 'geometry': Point(x,y)} for x, y in exterior_coords])
            for interior in geom.interiors:
                interior_coords = [(x, y) for x, y, *_ in interior.coords]
                point_coords.extend([{'x': x, 'y': y, **data, 'geometry': Point(x,y)} for x, y in interior_coords])
        elif isinstance(geom, MultiPolygon):
            for poly in geom:
                exterior_coords = [(x, y) for x, y, *_ in poly.exterior.coords]
                point_coords.extend([{'x': x, 'y': y, **data, 'geometry': Point(x,y)} for x, y in exterior_coords])
                for interior in poly.interiors:
                    interior_coords = [(x, y) for x, y, *_ in interior.coords]
                    point_coords.extend([{'x': x, 'y': y, **data, 'geometry': Point(x,y)} for x, y in interior_coords])
    if point_coords:
        point_df = pd.DataFrame(point_coords)
        point_gdf = gpd.GeoDataFrame(point_df, geometry='geometry', crs=gdf.crs)
        return point_gdf
    else:
        # Return empty GeoDataFrame with same structure
        return gpd.GeoDataFrame(columns=['x', 'y', 'geometry'], geometry='geometry', crs=gdf.crs)
    
def identify_centerline(line_data, tolerance=0.5):
    line_data = line_data.explode(ignore_index=True)
    line_data = line_data.to_crs(epsg=3857)

    print(f"ℹ️ Number of lines: {len(line_data)}")
    line_data = line_data.drop_duplicates('geometry')
    line_data = line_data.reset_index(drop=True)
    print(f"ℹ️ Number of removed duplicates: {len(line_data)}")

    print("🧩 Exploding Lines.")
    line_data = explode_lines(line_data)
    print(f"ℹ️ Number of exploded lines: {len(line_data)}")


    print(f"🧩 Point coordinates")
    point_coords = point_coordinates(line_data)
    print(f"ℹ️ Number of point coordinates: {len(point_coords)}")


    # Drop the existing LENGTH column to avoid conflicts
    if 'LENGTH' in line_data.columns:
        line_data = line_data.drop(columns=['LENGTH'])
        
    line_data['length'] = line_data.geometry.length.round(2)
    line_data = line_data.sort_values(by='length', ascending=False).reset_index(drop=True)

    dropped_idx = []
    print(f"🧩 Identify center line")
    for i, row in line_data.iterrows():
        if i in dropped_idx:
            continue

        # print(f"Line {i+1}: Length = {row['length']} meters")
        geom = row.geometry
        line_within = line_data[line_data.geometry.within(geom.buffer(tolerance))]

        if len(line_within) > 1:
            # print(f"ℹ️ Found {len(line_within)} lines within 0.5 meter of this line.")
            for j, other_row in line_within.iterrows():
                if j != i:
                    # print(f"ℹ️ Other Line {j+1}: Length = {other_row['length']:,} meters")
                    dropped_idx.append(other_row.name)
                    # print(f"🔴 Dropping line {other_row.name} from the dataset.")

    print(f"\nℹ️ Total lines dropped: {len(dropped_idx)}")
    line_data = line_data.drop(index=dropped_idx).reset_index(drop=True)

    return line_data, point_coords

# def identify_centerline(line_data):
#     from tqdm import tqdm
#     import geopandas as gpd
#     from shapely.strtree import STRtree
    
#     print(f"🧩 Identify Centerline (Spatial Index)")
#     line_data = line_data.explode(ignore_index=True)
#     line_data = line_data.to_crs(epsg=3857)

#     print(f"ℹ️ Number of lines: {len(line_data)}")
#     line_data = line_data.drop_duplicates('geometry')
#     line_data = line_data.reset_index(drop=True)
#     print(f"ℹ️ Number of removed duplicates: {len(line_data)}")

#     print("🧩 Exploding Lines.")
#     line_data = explode_lines(line_data)
#     print(f"ℹ️ Number of exploded lines: {len(line_data)}")

#     point_coords = point_coordinates(line_data)
#     print(f"ℹ️ Number of point coordinates: {len(point_coords)}")

#     if 'LENGTH' in line_data.columns:
#         line_data = line_data.drop(columns=['LENGTH'])
        
#     line_data['length'] = line_data.geometry.length.round(2)
#     line_data = line_data.sort_values(by='length', ascending=False).reset_index(drop=True)

#     # ✅ SPATIAL INDEX: Much faster
#     print("🧩 Building spatial index...")
#     tree = STRtree(line_data.geometry)
    
#     dropped_idx = set()
#     for i, row in tqdm(line_data.iterrows(), total=len(line_data), desc="Cleaning Lines"):
#         if i in dropped_idx:
#             continue

#         geom = row.geometry
#         buffer_geom = geom.buffer(0.5)
        
#         # Use spatial index to find potential intersections
#         potential_matches = tree.query(buffer_geom)
        
#         for j in potential_matches:
#             if j != i and j not in dropped_idx:
#                 if line_data.iloc[j].geometry.within(buffer_geom):
#                     dropped_idx.add(j)

#     print(f"ℹ️ Total lines dropped: {len(dropped_idx):,}")
#     line_data = line_data.drop(index=list(dropped_idx)).reset_index(drop=True)

#     return line_data, point_coords
