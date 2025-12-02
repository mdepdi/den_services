import geopandas as gpd
import pandas as pd
import numpy as np
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

def detect_turn(nodes_gdf: gpd.GeoDataFrame,
                        edges_gdf: gpd.GeoDataFrame,
                        tol: float = 5.0,
                        min_cover_ratio: float = 0.8):
    """
    Detect turn nodes using an area-based approach.
    Adapted and improved from the reference implementation.
    Works in projected CRS (meters).

    Parameters
    ----------
    nodes_gdf : GeoDataFrame
        Node geometries (e.g., poles).
    edges_gdf : GeoDataFrame
        Line geometries (e.g., cables).
    tol : float, optional
        Buffer radius around each node in meters.
    min_cover_ratio : float, optional
        Ratio threshold (min_area/max_area). < this → likely a turn.

    Returns
    -------
    GeoDataFrame
        Same nodes_gdf with added columns:
        - 'turn_isec': 0=straight, 2=turn, >2=multi
        - 'turn_ratio': area ratio (min/max)
        - 'area_count': number of valid buffer overlap parts
    """
    if nodes_gdf.crs is None or edges_gdf.crs is None:
        raise ValueError("Both layers must have CRS defined (use meters).")

    nodes = nodes_gdf.copy()
    nodes = nodes.reset_index(drop=True)
    nodes["id"] = nodes.index
    nodes["turn_note"] = 'straight'
    nodes["turn_isec"] = -1.0
    nodes["turn_ratio"] = -1.0
    nodes["area_count"] = None
    
    # # --- group nodes ---
    # group = auto_group(nodes, distance=5)
    # group = group.rename(columns={'region':'group'})
    # nodes = nodes.sjoin(group[['geometry', 'group']]).drop(columns="index_right")
    
    # --- buffer nodes ---
    nodes_buff = nodes.copy()
    nodes_buff["geometry"] = nodes_buff.geometry.buffer(tol)

    edges_buff = edges_gdf.copy()
    edges_buff["geometry"] = edges_buff.geometry.buffer(0.5) 

    diff = gpd.overlay(nodes_buff[["id", "geometry"]], edges_buff, how="difference")
    diff = diff.explode(index_parts=True).reset_index(drop=True)
    diff["area"] = diff.geometry.area

    area_group = diff.groupby("id")["area"].apply(list).reset_index(name="area_list")

    # --- analyze ratio ---
    for idx, row in area_group.iterrows():
        areas = [a for a in row["area_list"] if a > 0]
        if len(areas) < 2:
            continue

        max_area = max(areas)
        min_area = min(areas)
        ratio = min_area / max_area if max_area > 0 else np.nan

        nodes.loc[row["id"], "turn_ratio"] = round(ratio, 3)
        nodes.loc[row["id"], "area_count"] = len(areas)

        # classify
        if len(areas) >= 3:
            nodes.loc[row["id"], "turn_isec"] = len(areas)
            nodes.loc[row["id"], "turn_note"] = "branch" # branch
        elif ratio < min_cover_ratio:
            nodes.loc[row["id"], "turn_isec"] = 2   # turn
            nodes.loc[row["id"], "turn_note"] = "turn"
        else:
            nodes.loc[row["id"], "turn_isec"] = 0   # straight
            nodes.loc[row["id"], "turn_note"] = "straight"

    return nodes

def route_preprocess(gdf: gpd.GeoDataFrame, tol: float = 5.0, decimals: int = 12):
    """
    Generate nodes and edges (u, v) from LineString/MultiLineString geometries.
    Each vertex is treated as a node.
    Automatically snaps close endpoints (within `tol`) across different lines.
    """
    # --- VALIDATE GEOMETRY ---
    geom_types = gdf.geom_type.unique().tolist()
    invalid = [gt for gt in geom_types if gt not in ["LineString", "MultiLineString"]]
    if invalid:
        raise ValueError(f"Unsupported geometry types: {invalid}")

    # --- PREPARE DATA ---
    crs_input = gdf.crs
    gdf["id_line"] = gdf.index + 1
    gdf = gdf.explode(ignore_index=True)
    gdf = gdf.drop_duplicates(subset="geometry").reset_index(drop=True)

    # --- EXTRACT EDGES ---
    edges = []
    for _, row in tqdm(gdf.iterrows(), desc="Extract Edges", total=len(gdf)):
        geom = row.geometry
        if geom.is_empty:
            continue
        lines = [geom] if geom.geom_type == "LineString" else geom.geoms
        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                u = Point(round(coords[i][0], decimals), round(coords[i][1], decimals))
                v = Point(round(coords[i + 1][0], decimals), round(coords[i + 1][1], decimals))
                edges.append({
                    "id_line": row["id_line"],
                    "geometry": LineString([u, v]),
                    "u": u,
                    "v": v,
                    **{k: v for k, v in row.items()}
                })
    edges_gdf = gpd.GeoDataFrame(edges, geometry="geometry", crs=gdf.crs)

    # --- BUILD NODES ---
    nodes = []
    for _, e in tqdm(edges_gdf.iterrows(), desc="Extract Nodes", total=len(edges_gdf)):
        nodes.append({"id_line": e["id_line"], "geometry": e["u"], **{k: v for k, v in edges_gdf.items()}})
        nodes.append({"id_line": e["id_line"], "geometry": e["v"], **{k: v for k, v in edges_gdf.items()}})
    nodes_gdf = gpd.GeoDataFrame(nodes, geometry="geometry", crs=gdf.crs)

    nodes_gdf["x"] = nodes_gdf.geometry.x.round(decimals)
    nodes_gdf["y"] = nodes_gdf.geometry.y.round(decimals)
    nodes_gdf["coord_key"] = list(zip(nodes_gdf["x"], nodes_gdf["y"]))
    node_counts = nodes_gdf.groupby("coord_key").size().rename("count")
    nodes_gdf = nodes_gdf.drop_duplicates("coord_key").merge(node_counts, left_on="coord_key", right_index=True, how="left").reset_index(drop=True)

    # --- MAP NODE IDs ---
    nodes_gdf["node_id"] = [f"N{i+1:07d}" for i in range(len(nodes_gdf))]
    def find_node(pt):
        key = (round(pt.x, decimals), round(pt.y, decimals))
        match = nodes_gdf[nodes_gdf["coord_key"] == key]
        return match["node_id"].values[0] if not match.empty else None

    edges_gdf["node_start"] = edges_gdf["u"].apply(find_node)
    edges_gdf["node_end"] = edges_gdf["v"].apply(find_node)
    edges_gdf["length"] = edges_gdf.geometry.length

    # --- TURN ---
    nodes_gdf = detect_turn(nodes_gdf, edges_gdf, tol=2)

    # --- CLEAN OUTPUT ---
    edges_gdf = edges_gdf.drop(columns=["u", "v"])
    # nodes_gdf = nodes_gdf[["node_id", "x", "y", "count", "turn_isec","turn_ratio", "turn_note", "geometry"]]

    # CRS
    nodes_gdf = nodes_gdf.to_crs(crs_input)
    edges_gdf = edges_gdf.to_crs(crs_input)

    return nodes_gdf, edges_gdf