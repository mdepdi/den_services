import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point, LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import substring, linemerge

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
            for line in geom.geoms:
                coords = [(x, y) for x, y, *_ in line.coords]
                point_coords.extend([{'x': x, 'y': y, **data, 'geometry': Point(x,y)} for x, y in coords])
        elif isinstance(geom, Polygon):
            exterior_coords = [(x, y) for x, y, *_ in geom.exterior.coords]
            point_coords.extend([{'x': x, 'y': y, **data, 'geometry': Point(x,y)} for x, y in exterior_coords])
            for interior in geom.interiors:
                interior_coords = [(x, y) for x, y, *_ in interior.coords]
                point_coords.extend([{'x': x, 'y': y, **data, 'geometry': Point(x,y)} for x, y in interior_coords])
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
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

def relative_intersection(line_a: LineString | MultiLineString, line_b: LineString | MultiLineString, tolerance=0.0):
    """
    Detect overlapping portions of line_a and line_b based on
    projection distance (lowest → highest).
    
    Returns:
        overlap_geom: LineString / MultiLineString
        new_geom:     LineString / MultiLineString
    """

    if line_a.geom_type == "MultiLineString":
        line_a = linemerge(line_a)
        if line_a.geom_type == "MultiLineString":
            geoms = list(line_a.geoms)
            longest = geoms[0]
            for geom in geoms:
                if geom.length > longest.length:
                    longest = geom
            line_a = longest

    if line_b.geom_type == "MultiLineString":
        line_b = linemerge(line_b)
        if line_b.geom_type == "MultiLineString":
            geoms = list(line_b.geoms)
            longest = geoms[0]
            for geom in geoms:
                if geom.length > longest.length:
                    longest = geom
            line_b = longest


    if tolerance > 0:
        lineA = line_a.buffer(tolerance)
        lineB = line_b.buffer(tolerance)
    else:
        lineA = line_a
        lineB = line_b

    # ---- 1. Compute intersection ----
    inter = line_a.intersection(lineB)
    if inter is None:
        return LineString(), line_a
    if inter.is_empty:
        return LineString(), line_a

    # ---- 2. Extract all intersection endpoints ----
    pts = []

    # Intersection may be LineString, MultiLineString, or Points
    if isinstance(inter, LineString):
        pts.extend([inter.coords[0], inter.coords[-1]])

    elif isinstance(inter, MultiLineString):
        for seg in inter.geoms:
            pts.extend([seg.coords[0], seg.coords[-1]])

    else:
        for g in inter.geoms:
            pts.append((g.x, g.y))

    # ---- 3. Project intersection endpoints onto line_a ----
    dist_list = sorted([line_a.project(Point(p)) for p in pts])
    dist_list = sorted(list(set(dist_list)))

    if len(dist_list) < 2:
        return None, line_a

    start_d = dist_list[0]
    end_d   = dist_list[-1]
    overlap_geom = substring(line_a, start_d, end_d)

    # ---- 5. Compute remaining "new" segments ----
    new_segments = []
    if start_d > 0:
        new_segments.append(substring(line_a, 0, start_d))
    if end_d < line_a.length:
        new_segments.append(substring(line_a, end_d, line_a.length))

    if len(new_segments) == 1:
        new_geom = new_segments[0]
    else:
        new_geom = MultiLineString([s for s in new_segments if not s.is_empty])
    return overlap_geom, new_geom


def detect_turn(nodes_gdf: gpd.GeoDataFrame,
                     edges_gdf: gpd.GeoDataFrame,
                     angle_thresh_deg: float = 150.0):
    """
    Faster turn detection using only topology and angles.

    - No buffers
    - No gpd.overlay
    - Classification:
        * turn_isec >= 3  -> 'branch'
        * turn_isec == 2 & angle < angle_thresh -> 'turn'
        * else -> 'straight'
    """
    nodes = nodes_gdf.copy().reset_index(drop=True)
    nodes["turn_note"] = "straight"
    nodes["turn_isec"] = 0
    nodes["turn_ratio"] = 1.0    # not really used anymore
    nodes["area_count"] = 1

    # build map: node_id -> list of direction vectors (dx, dy)
    node_dirs = {nid: [] for nid in nodes["node_id"]}

    def edge_dirs_at_node(edge_geom, node_point: Point):
        coords = list(edge_geom.coords)
        d0 = node_point.distance(Point(coords[0]))
        d1 = node_point.distance(Point(coords[-1]))
        if d0 <= d1:
            # direction from node to next coord
            if len(coords) >= 2:
                dx = coords[1][0] - coords[0][0]
                dy = coords[1][1] - coords[0][1]
            else:
                dx = dy = 0.0
        else:
            # direction from node to previous coord
            if len(coords) >= 2:
                dx = coords[-2][0] - coords[-1][0]
                dy = coords[-2][1] - coords[-1][1]
            else:
                dx = dy = 0.0
        return (dx, dy)

    # Lookup
    node_geom_map = dict(zip(nodes["node_id"], nodes.geometry))

    # iterate edges, add direction vectors to node_dirs
    for e in edges_gdf.itertuples():
        for side, nid in (("node_start", e.node_start), ("node_end", e.node_end)):
            if nid not in node_dirs:
                continue
            node_pt = node_geom_map.get(nid)
            if node_pt is None:
                continue
            dx, dy = edge_dirs_at_node(e.geometry, node_pt)
            norm = np.hypot(dx, dy)
            if norm == 0:
                continue
            node_dirs[nid].append((dx / norm, dy / norm))

    # classify each node
    for idx, row in nodes.iterrows():
        nid = row["node_id"]
        dirs = node_dirs.get(nid, [])
        k = len(dirs)

        if k == 0 or k == 1:
            # isolated or dead-end
            nodes.at[idx, "turn_isec"] = k
            nodes.at[idx, "turn_note"] = "straight"
            continue

        if k >= 3:
            nodes.at[idx, "turn_isec"] = k
            nodes.at[idx, "turn_note"] = "branch"
            continue

        # k == 2 -> check angle between two vectors
        (dx1, dy1), (dx2, dy2) = dirs[:2]
        dot = dx1 * dx2 + dy1 * dy2
        dot = max(-1.0, min(1.0, dot))
        angle_rad = np.arccos(dot)
        angle_deg = np.degrees(angle_rad)

        # if angle close to 180° => straight, else turn
        # angle here is the "inside" angle; 180 ~ straight, 90 ~ corner
        if angle_deg >= angle_thresh_deg:
            nodes.at[idx, "turn_isec"] = 0
            nodes.at[idx, "turn_note"] = "straight"
        else:
            nodes.at[idx, "turn_isec"] = 2
            nodes.at[idx, "turn_note"] = "turn"

    return nodes

def route_preprocess(gdf: gpd.GeoDataFrame, decimals: int = 12):
    """
    Faster version, attribute-aware:
    - Avoids per-row DataFrame scans to find nodes
    - Uses coord_key on edges to build nodes + mapping
    - Preserves all non-geometry columns from the input gdf on edges
    """

    # --- VALIDATE GEOMETRY ---
    geom_types = gdf.geom_type.unique().tolist()
    invalid = [gt for gt in geom_types if gt not in ["LineString", "MultiLineString"]]
    if invalid:
        raise ValueError(f"Unsupported geometry types: {invalid}")

    # --- PREPARE DATA ---
    crs_input = gdf.crs
    gdf = gdf.copy()

    geom_col = gdf.geometry.name          # usually "geometry"
    # kalau belum ada id_line, buat
    if "id_line" not in gdf.columns:
        gdf["id_line"] = np.arange(1, len(gdf) + 1)

    # kolom yang mau diwariskan ke edges (semua kecuali geometry)
    non_geom_cols = [c for c in gdf.columns if c != geom_col]

    # explode MultiLineString → LineString
    gdf = gdf.explode(ignore_index=True)
    # HATI-HATI: kalau kamu butuh duplicate geometry beda atribut, jangan drop di sini
    gdf = gdf.drop_duplicates(subset=[geom_col]).reset_index(drop=True)

    # --- EXTRACT EDGES ---
    print("Extract Edges")
    edge_records = []

    for row in gdf.itertuples():
        geom = getattr(row, geom_col)
        if geom.is_empty:
            continue

        lines = [geom] if geom.geom_type == "LineString" else geom.geoms
        base_attrs = {col: getattr(row, col) for col in non_geom_cols}

        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue

            for i in range(len(coords) - 1):
                x1, y1 = coords[i]
                x2, y2 = coords[i + 1]

                x1r, y1r = round(x1, decimals), round(y1, decimals)
                x2r, y2r = round(x2, decimals), round(y2, decimals)

                u_pt = Point(x1r, y1r)
                v_pt = Point(x2r, y2r)

                record = {
                    **base_attrs,
                    "geometry": LineString([u_pt, v_pt]),
                    "u_x": x1r,
                    "u_y": y1r,
                    "v_x": x2r,
                    "v_y": y2r,
                }
                edge_records.append(record)

    edges_gdf = gpd.GeoDataFrame(edge_records, geometry="geometry", crs=gdf.crs)

    # --- BUILD NODES ---
    print("Extract Nodes")
    edges_gdf["u_key"] = list(zip(edges_gdf["u_x"], edges_gdf["u_y"]))
    edges_gdf["v_key"] = list(zip(edges_gdf["v_x"], edges_gdf["v_y"]))

    stacked = edges_gdf[["u_key", "v_key"]].stack()
    node_counts = stacked.value_counts().rename("count")

    unique_keys = node_counts.index.to_list()
    node_x = [k[0] for k in unique_keys]
    node_y = [k[1] for k in unique_keys]

    nodes_gdf = gpd.GeoDataFrame(
        {
            "coord_key": unique_keys,
            "x": node_x,
            "y": node_y,
            "count": node_counts.values,
        },
        geometry=gpd.points_from_xy(node_x, node_y),
        crs=gdf.crs,
    ).reset_index(drop=True)

    nodes_gdf["node_id"] = [f"N{i+1:07d}" for i in range(len(nodes_gdf))]
    key_to_id = dict(zip(nodes_gdf["coord_key"], nodes_gdf["node_id"]))
    edges_gdf["node_start"] = edges_gdf["u_key"].map(key_to_id)
    edges_gdf["node_end"]   = edges_gdf["v_key"].map(key_to_id)
    edges_gdf["length"] = edges_gdf.geometry.length

    # --- TURN DETECTION ---
    nodes_gdf = detect_turn(nodes_gdf, edges_gdf, angle_thresh_deg=150)

    # --- CLEAN OUTPUT ---
    edges_gdf = edges_gdf.drop(columns=["u_x", "u_y", "v_x", "v_y", "u_key", "v_key"])
    nodes_gdf = nodes_gdf[[
        "node_id", "x", "y", "count",
        "turn_isec", "turn_ratio", "turn_note", "geometry"
    ]]

    # CRS
    if crs_input is not None:
        nodes_gdf = nodes_gdf.to_crs(crs_input)
        edges_gdf = edges_gdf.to_crs(crs_input)

    return nodes_gdf, edges_gdf