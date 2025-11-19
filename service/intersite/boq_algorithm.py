# ===============
# ROUTE EXTRACTOR
# ===============
import os
import time
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import simplekml
from tqdm import tqdm
from shapely.strtree import STRtree
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import nearest_points
from shapely.ops import linemerge
from tbg_modules.kml import export_kml, sanitize_kml
from tbg_modules.table import excel_styler

def snap_endpoints(line, points_union,tol=1):
    start, end = Point(line.coords[0]), Point(line.coords[-1])
    nearest_start = nearest_points(start, points_union)[1]
    nearest_end   = nearest_points(end, points_union)[1]
    new_start = nearest_start if start.distance(nearest_start) < tol else start
    new_end   = nearest_end if end.distance(nearest_end) < tol else end
    return LineString([new_start, new_end])    

def snap_line(line_geom, old_pt, new_pt):
    coords = list(line_geom.coords)
    if Point(coords[0]).equals(old_pt):
        coords[0] = (new_pt.x, new_pt.y)
    elif Point(coords[-1]).equals(old_pt):
        coords[-1] = (new_pt.x, new_pt.y)
    return LineString(coords)

def connect_line(line_geom, old_pt, new_pt):
    coords = list(line_geom.coords)
    if Point(coords[0]).equals(old_pt):
        coords[0] = (new_pt.x, new_pt.y)
        coords = [new_pt] + coords
    elif Point(coords[-1]).equals(old_pt):
        coords[-1] = (new_pt.x, new_pt.y)
        coords.append(new_pt)

    return LineString(coords)

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
    for _, row in gdf.iterrows():
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
    for _, e in edges_gdf.iterrows():
        nodes.append({"id_line": e["id_line"], "geometry": e["u"]})
        nodes.append({"id_line": e["id_line"], "geometry": e["v"]})
    nodes_gdf = gpd.GeoDataFrame(nodes, geometry="geometry", crs=gdf.crs)

    nodes_gdf["x"] = nodes_gdf.geometry.x.round(decimals)
    nodes_gdf["y"] = nodes_gdf.geometry.y.round(decimals)
    nodes_gdf["coord_key"] = list(zip(nodes_gdf["x"], nodes_gdf["y"]))
    node_counts = nodes_gdf.groupby("coord_key").size().rename("count")
    nodes_gdf = nodes_gdf.drop_duplicates("coord_key").merge(node_counts, left_on="coord_key", right_index=True, how="left").reset_index(drop=True)

    # # --- SNAP NEARBY ENDPOINTS ---
    # tree = STRtree(nodes_gdf.geometry)
    # geom_to_idx = {g.wkb: i for i, g in enumerate(nodes_gdf.geometry)}

    # snapped = {}
    # snap_updates = []
    # for geom in nodes_gdf.geometry:
    #     near_geoms = tree.query(geom, predicate="dwithin", distance=tol)
    #     if len(near_geoms) <= 1:
    #         continue
        
    #     idx = geom_to_idx[geom.wkb]
    #     source_id = nodes_gdf.at[idx, 'id_line']
    #     for jdx in near_geoms:
    #         near_geom = nodes_gdf.at[jdx, 'geometry']
    #         near_id = nodes_gdf.at[jdx, 'id_line']
    #         used_ids = set(i for ids in snapped.values() for i in ids)
    #         if idx == jdx:
    #             continue
    #         if source_id in snapped.keys():
    #             continue
    #         if jdx in used_ids:
    #             continue
    #         if source_id == near_id:
    #             continue
    #         if geom.distance(near_geom) == 0:
    #             continue

    #         if nodes_gdf.at[idx, "count"] <= nodes_gdf.at[jdx, "count"]:
    #             snapped.setdefault(source_id, []).append(idx)
    #             snap_updates.append((idx, near_geom))

    # for idx, geom_near in snap_updates:
    #     old_geom = nodes_gdf.at[idx, "geometry"]
    #     nodes_gdf.at[idx, "geometry"] = geom_near

    #     mask = (
    #         edges_gdf["u"].apply(lambda p: p.equals(old_geom)) |
    #         edges_gdf["v"].apply(lambda p: p.equals(old_geom))
    #     )
    #     for jdx, line in edges_gdf[mask].iterrows():
    #         edges_gdf.at[jdx, "geometry"] = snap_line(line.geometry, old_geom, geom_near)
    #         edges_gdf.at[jdx, "u"] = Point(list(edges_gdf.at[jdx, "geometry"].coords[0]))
    #         edges_gdf.at[jdx, "v"] = Point(list(edges_gdf.at[jdx, "geometry"].coords[-1]))

    # # --- REBUILD NODE REGISTRY AFTER SNAPPING ---
    # nodes_gdf["x"] = nodes_gdf.geometry.x.round(decimals)
    # nodes_gdf["y"] = nodes_gdf.geometry.y.round(decimals)
    # nodes_gdf["coord_key"] = list(zip(nodes_gdf["x"], nodes_gdf["y"]))
    # nodes_gdf = nodes_gdf.drop_duplicates("coord_key").reset_index(drop=True)
    # node_counts = nodes_gdf.groupby("coord_key").size().rename("count")
    # if 'count' in nodes_gdf.columns:
    #     nodes_gdf = nodes_gdf.drop(columns='count')
    # nodes_gdf = nodes_gdf.merge(node_counts, left_on="coord_key", right_index=True, how="left")

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
    nodes_gdf = nodes_gdf[["node_id", "x", "y", "count", "turn_isec","turn_ratio", "turn_note", "geometry"]]

    # CRS
    nodes_gdf = nodes_gdf.to_crs(crs_input)
    edges_gdf = edges_gdf.to_crs(crs_input)

    return nodes_gdf, edges_gdf


def snap_geom(g1:LineString|MultiLineString, g2:shapely.Geometry, threshold:float):
    from shapely.ops import nearest_points
    coordinates = []
    geom_type = g1.geom_type
    if geom_type == "LineString":
        for x, y in g1.coords:
            point = Point(x, y)
            p1, p2 = nearest_points(point, g2)
            if p1.distance(p2) <= threshold:
                coordinates.append(p2.coords[0])
            else:
                coordinates.append((x, y))
    elif geom_type == "MultiLineString":
        geoms = list(g1.geoms)
        for geom in geoms:
            for x, y in geom.coords:
                point = Point(x, y)
                p1, p2 = nearest_points(point, g2)
                if p1.distance(p2) <= threshold:
                    coordinates.append(p2.coords[0])
                else:
                    coordinates.append((x, y))
    return LineString(coordinates)


def substring_overlay(source_gdf: gpd.GeoDataFrame, ref_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    import geopandas as gpd
    from shapely.ops import substring
    from shapely.geometry import Point, LineString, MultiLineString, MultiPoint
    """
    For each source line, extract the full segment that lies between its first and last
    intersection with any reference line. Works with LineString/MultiLineString and
    both point and overlap intersections.

    Returns a GeoDataFrame of substring segments with source attributes + index_right.
    """
    if source_gdf.crs != ref_gdf.crs:
        ref_gdf = ref_gdf.to_crs(source_gdf.crs)
    if source_gdf.crs.to_epsg() != 3857:
        source_gdf = source_gdf.to_crs(3857)


    # CANDIDATE
    candidates = gpd.sjoin(source_gdf, ref_gdf, how="inner", predicate="intersects")
    candidates['length'] = candidates.geometry.length
    candidates = candidates.sort_values('length', ascending=False)
    candidates = candidates.drop_duplicates('geometry')
    if candidates.empty:
        return gpd.GeoDataFrame(columns=list(source_gdf.columns), crs=source_gdf.crs)

    out_rows = []
    ref_dict = ref_gdf.geometry.to_dict()

    for src_idx, row in candidates.iterrows():
        src_geom = row.geometry
        src_length = src_geom.length
        ref_idx = row["index_right"]
        ref_geom = ref_dict[ref_idx]

        inter = shapely.intersection(src_geom, ref_geom)
        inter_length = inter.length
        length_ratio = inter_length / src_length

        if inter.is_empty:
            continue

        anchors = []
        # POINT
        if isinstance(inter, Point):
            anchors.append(inter)
        elif isinstance(inter, MultiPoint):
            anchors.extend(list(inter.geoms))

        # LINESTRING
        if isinstance(inter, LineString):
            coords = list(inter.coords)
            if len(coords) >= 2:
                anchors.append(Point(coords[0]))
                anchors.append(Point(coords[-1]))
        elif isinstance(inter, MultiLineString):
            for seg in inter.geoms:
                coords = list(seg.coords)
                if len(coords) >= 2:
                    anchors.append(Point(coords[0]))
                    anchors.append(Point(coords[-1]))

        if len(anchors) < 2:
            continue
        
        # ANCHOR MAPPING DISTANCE
        dists = [src_geom.project(pt) for pt in anchors]
        start_d, end_d = min(dists), max(dists)
        if end_d - start_d <= 0:
            continue
        
        # SEGMENT
        seg = substring(src_geom, start_d, end_d)
        
        # DIFF 
        diff = shapely.difference(src_geom, ref_geom)
        if isinstance(diff, MultiLineString):
            for geom in diff.geoms:
                length = geom.length
                if length > 1000:
                    seg = shapely.difference(seg, geom)
        elif isinstance(diff, LineString):
            length = diff.length
            if length > 1000:
                seg = shapely.difference(seg, diff)

        if seg.is_empty or seg.length <= 0:
            continue

        attrs = {k: v for k, v in row.items() if k != "geometry"}
        attrs["geometry"] = seg
        out_rows.append(attrs)

    if not out_rows:
        return gpd.GeoDataFrame(columns=list(source_gdf.columns), crs=source_gdf.crs)

    result = gpd.GeoDataFrame(out_rows, crs=source_gdf.crs)
    result = result.explode(ignore_index=True)
    print(f"🟢 Substring overlay success")
    return result

def obstacle_detection(lines_gdf: gpd.GeoDataFrame):
    from tbg_modules.utils import auto_group
    
    lines_gdf = lines_gdf.copy()
    lines_gdf["line_id"] = lines_gdf.index

    ring_name = lines_gdf['ring_name'].mode()[0]
    osm_railway = gpd.read_parquet(r"D:\Data Analytical\DATA\03. Road Network\railway_osm.parquet")
    osm_toll = gpd.read_parquet(r"D:\Data Analytical\DATA\03. Road Network\toll.parquet")
    osm_railway = osm_railway.rename(columns={'name':'rail_name'})
    osm_toll = osm_toll.rename(columns={'name':'toll_name'})

    lines_gdf = lines_gdf.to_crs(epsg=3857)
    osm_railway = osm_railway.to_crs(epsg=3857)
    osm_toll = osm_toll.to_crs(epsg=3857)

    union_lines = lines_gdf.geometry.union_all().buffer(20)
    osm_railway = osm_railway[osm_railway.geometry.intersects(union_lines)].copy()
    osm_toll = osm_toll[osm_toll.geometry.intersects(union_lines)].copy()

    # RAILWAY
    if not osm_railway.empty:
        isec_rail = gpd.overlay(lines_gdf, osm_railway[['rail_name','geometry']], how="intersection", keep_geom_type=False)
        isec_rail['geometry'] = isec_rail.geometry.representative_point()
        isec_rail['remark'] = isec_rail['near_end'] + "-" + isec_rail['far_end']
        isec_rail['obstacle_railway'] = isec_rail.geometry.to_wkt()

        group = auto_group(isec_rail, distance=100)
        group = group.rename(columns={'region':'group'})
        isec_rail = gpd.sjoin(isec_rail, group[['group', 'geometry']]).drop(columns='index_right')
        isec_rail = isec_rail.drop_duplicates(subset='group')
        print(f"🟠 Found {len(isec_rail)} intersect with railway")
        lines_gdf = lines_gdf.merge(isec_rail[['line_id', 'rail_name', 'obstacle_railway']], how='left', on='line_id')
    else:
        lines_gdf['obstacle_railway'] = None
        print(f"🟢 No railway obstacle in {ring_name}.")

    # TOLL
    if not osm_toll.empty:
        isec_toll = gpd.overlay(lines_gdf, osm_toll[['toll_name','geometry']], how="intersection", keep_geom_type=False)
        isec_toll['geometry'] = isec_toll.geometry.representative_point()
        isec_toll['remark'] = isec_toll['near_end'] + "-" + isec_toll['far_end']
        isec_toll['obstacle_toll'] = isec_toll.geometry.to_wkt()

        group = auto_group(isec_toll, distance=100)
        group = group.rename(columns={'region':'group'})
        isec_toll = gpd.sjoin(isec_toll, group[['group', 'geometry']]).drop(columns='index_right')
        isec_toll = isec_toll.drop_duplicates(subset='group')
        print(f"🟠 Found {len(isec_toll)} intersect with highway")
        lines_gdf = lines_gdf.merge(isec_toll[['line_id','toll_name', 'obstacle_toll']], how='left', left_index=True, right_index=True)
    else:
        lines_gdf['obstacle_toll'] = None
        print(f"🟢 No highway obstacle in {ring_name}.")

    # # JOIN ADMIN
    # admin_2024 = admin_2024.to_crs(epsg=3857)
    # clean_col = ['kabkot', 'provinsi', 'kecamatan', 'desa']
    # intersected.columns = intersected.columns.str.lower()

    # for col in clean_col:
    #     if col in intersected.columns:
    #         intersected = intersected.drop(columns=col)
    # intersected = gpd.sjoin(intersected, admin_2024).drop(columns='index_right')

    # # EXPORT
    # intersected = intersected.to_crs(epsg=4326)
    # intersected['long'] = intersected.geometry.x
    # intersected['lat'] = intersected.geometry.y
    # intersected.to_parquet(fr"{export_dir}\Intersect Railway.parquet")

    return lines_gdf


def bill_of_quantity(points: gpd.GeoDataFrame, lines: gpd.GeoDataFrame):
    import shapely
    from shapely.ops import split, snap, linemerge
    from shapely.geometry import LineString
    import geopandas as gpd
    import numpy as np

    # =============================
    # LOAD FO REFERENCE GEOMETRY
    # =============================
    fo_route = r"D:\Data Analytical\DATA\06. FO TBG\Compile FO Route Only June 2025\FO TBG Only_01062025.parquet"
    fo_route = gpd.read_parquet(fo_route)
    fo_route = fo_route.to_crs(epsg=3857)
    fo_route.columns = fo_route.columns.str.lower()
    fo_route = fo_route.rename(columns={'name': 'fiber'})

    # =============================
    # PREPARE INPUT DATA
    # =============================
    points = points.copy().to_crs(epsg=3857)
    lines = lines.copy().to_crs(epsg=3857)
    ring_name = points['ring_name'].mode()[0]
    print(f"🌏 {ring_name} BOQ running ...")

    if points.empty:
        raise ValueError("Points data is empty.")
    if lines.empty:
        raise ValueError("Lines data is empty.")
    
    lines["id_line"] = lines.index + 1

    # =============================
    # ROUTE PREPROCESS
    # =============================
    nodes, edges = route_preprocess(lines)
    nodes = nodes.to_crs(epsg=3857)
    edges = edges.to_crs(epsg=3857)

    nodes.to_parquet(fr"D:\Data Analytical\PROJECT\TASK\NOVEMBER\Week 2\BoQ Intersite\Export\Trial BOQ\Nodes_{ring_name}.parquet")
    edges.to_parquet(fr"D:\Data Analytical\PROJECT\TASK\NOVEMBER\Week 2\BoQ Intersite\Export\Trial BOQ\Edges_{ring_name}.parquet")

    # =============================
    # IDENTIFY TURN / BRANCH POINTS
    # =============================
    turn_data = nodes[nodes['turn_note'].str.contains('turn|branch')].copy()
    turn_data = turn_data.rename(columns={'node_id': 'turn_id'})
    branch = turn_data[turn_data['turn_isec'] > 2].copy()
    branch = branch.rename(columns={'turn_id': 'branch_id'})

    points = gpd.sjoin_nearest(points, nodes[['geometry', 'node_id', 'turn_ratio']], how='left', exclusive=True).drop(columns='index_right')
    points = gpd.sjoin_nearest(points, turn_data[['geometry', 'turn_id']], how='left', distance_col='dist_turn', exclusive=True).drop(columns='index_right')
    points = gpd.sjoin_nearest(points, branch[['geometry', 'branch_id']], how='left', distance_col='dist_branch', exclusive=True, max_distance=500).drop(columns='index_right')
    points['dist_turn'] = points['dist_turn'].fillna(-1)
    points['dist_branch'] = points['dist_branch'].fillna(-1)

    # Assign OTB and ODP (nearest turn / branch)
    points['otb'] = points['node_id'].apply(lambda x: nodes.loc[nodes['node_id'] == x, 'geometry'].values[0].wkt)
    points['odp'] = points['turn_id'].apply(lambda x: turn_data.loc[turn_data['turn_id'] == x, 'geometry'].values[0].wkt)

    for idx, row in points.iterrows():
        node_id = row['node_id']
        turn_id = row['turn_id']
        dist_turn = row['dist_turn']
        branch_id = row['branch_id']
        dist_branch = row['dist_branch']
        branch_ratio = row['turn_ratio']

        # -- No nearest turn --
        if dist_turn > 500:
            geom_branch = row['otb']
            points.at[idx, 'odp'] = geom_branch

        # -- Nearest branch if exist --
        if dist_branch > 0 and branch_ratio < 0.5:
            geom_branch = branch.loc[branch['branch_id'] == branch_id, 'geometry'].values[0].wkt
            points.at[idx, 'odp'] = geom_branch

    # =============================
    # IDENTIFY ROUTE (ACCESS + BACKBONE)
    # =============================
    for idx, row in lines.iterrows():
        line_geom = row.geometry
        line_geom = linemerge(line_geom) if line_geom.geom_type == "MultiLineString" else line_geom
        ne = str(row['near_end'])
        fe = str(row['far_end'])
        ne_type = str(points.loc[points['site_id'].astype(str) == ne, 'site_type'].values[0]).lower()
        fe_type = str(points.loc[points['site_id'].astype(str) == fe, 'site_type'].values[0]).lower()

        odp_ne = points.loc[points["site_id"].astype(str) == ne, 'odp']
        odp_fe = points.loc[points["site_id"].astype(str) == fe, 'odp']

        access_ne = None
        access_fe = None

        # --- NEAR END ---
        if not odp_ne.empty and 'hub' not in ne_type:
            odp_ne_geom = shapely.from_wkt(odp_ne.values[0])
            odp_ne_geom = snap(odp_ne_geom, line_geom, tolerance=1)
            splitted_ne = split(line_geom, odp_ne_geom)
            splitted_ne = list(splitted_ne.geoms)
            splitted_ne = sorted(splitted_ne, key=lambda seg: seg.length)
            if len(splitted_ne) > 1:
                access_ne = splitted_ne[0]
                lines.at[idx, 'access_ne'] = access_ne.wkt

        # --- FAR END ---
        if not odp_fe.empty and 'hub' not in fe_type:
            odp_fe_geom = shapely.from_wkt(odp_fe.values[0])
            odp_fe_geom = snap(odp_fe_geom, line_geom, tolerance=1)
            splitted_fe = split(line_geom, odp_fe_geom)
            splitted_fe = list(splitted_fe.geoms)
            splitted_fe = sorted(splitted_fe, key=lambda seg: seg.length)
            if len(splitted_fe) > 1:
                access_fe = splitted_fe[0]
                lines.at[idx, 'access_fe'] = access_fe.wkt

        # --- BACKBONE ---
        backbone = line_geom
        if access_ne:
            backbone = shapely.difference(backbone, access_ne)
        if access_fe:
            backbone = shapely.difference(backbone, access_fe)
        lines.at[idx, 'backbone'] = backbone.wkt

    # =============================
    # IDENTIFY EXISTING FO ROUTES
    # =============================
    union_lines = lines.geometry.union_all().buffer(30)
    fo_route = fo_route[fo_route.geometry.intersects(union_lines)].copy()
    fo_route['geometry'] = fo_route.geometry.buffer(30)

    existing_route = substring_overlay(lines, fo_route)
    existing_route = gpd.overlay(lines, fo_route, how='intersection', keep_geom_type=True)

    if existing_route.empty:
        print("⚠️ No existing FO intersections found.")
        return points, lines

    existing_route = existing_route[['id_line', 'fiber', 'geometry']].reset_index(drop=True)
    existing_route = existing_route.dissolve(['id_line', 'fiber']).reset_index()
    existing_route["geometry"] = existing_route.geometry.apply(lambda g: linemerge(g) if g.geom_type == "MultiLineString" else g)
    existing_route['length'] = existing_route.geometry.length
    existing_route = existing_route.sort_values('length', ascending=False)

    dropped = []
    for idx, row in existing_route.iterrows():
        if idx in dropped:
            continue
        geom = row.geometry.buffer(5)
        within_idx = existing_route[(existing_route.index != idx) & (existing_route.within(geom))]
        if not within_idx.empty:
            within_idx = within_idx.index.to_list()
            dropped.extend(within_idx)
    
    if len(dropped) > 0:
        existing_route = existing_route.drop(index=dropped)
        print(f"ℹ️ Dropped {len(dropped)} overlapped lines.")
    existing_route = existing_route.drop_duplicates('geometry').reset_index(drop=True)
    existing_route.to_parquet(fr"D:\Data Analytical\PROJECT\TASK\NOVEMBER\Week 1\BoQ Intersite\Export\Trial BOQ\Existing Route_{ring_name}.parquet")

    lines['fo_exist'] = [{} for _ in range(len(lines))]
    lines['pole_exist'] = [{} for _ in range(len(lines))]
    lines['closure'] = [{} for _ in range(len(lines))]
    
    # =============================
    # CLASSIFY EXISTING & POLE EXISTING
    # =============================
    for idx, row in lines.iterrows():
        id_line = row['id_line']
        backbone = shapely.from_wkt(row['backbone'])
        fo_lines = existing_route[existing_route['id_line'] == id_line].copy()
        fo_exist_dict = {}
        pole_exist_dict = {}
        closure_dict = {}

        for _, fo_row in fo_lines.iterrows():
            fiber_name = fo_row['fiber']
            fo_geom = fo_row.geometry
            if fo_geom.is_empty:
                continue

            if fo_geom.length > 1000:
                print(f"ℹ️ FO Existing: {fiber_name} | Length: {fo_geom.length}")
                backbone = shapely.difference(backbone, fo_geom)
                closure = shapely.intersection(fo_geom, backbone)
                fo_exist_dict[fiber_name] = fo_geom.wkt

                if not closure.is_empty:
                    print(closure)
                    closure_dict[fiber_name] = closure.wkt
            elif fo_geom.length > 100 and fo_geom.length < 1000:
                print(f"ℹ️ Pole Existing: {fiber_name} | Length: {fo_geom.length}")
                pole_exist_dict[fiber_name] = fo_geom.wkt
            else:
                continue

        lines.at[idx, 'fo_exist'] = fo_exist_dict
        lines.at[idx, 'closure'] = closure_dict
        lines.at[idx, 'pole_exist'] = pole_exist_dict
        lines.at[idx, 'backbone'] = backbone.wkt

    # =============================
    # CLASSIFY OBSTACLE
    # =============================
    lines = obstacle_detection(lines)

    print(f"🟢 {ring_name} BOQ Processing complete.\n")
    return points, lines


def parallel_boq(points_gdf:gpd.GeoDataFrame, lines_gdf:gpd.GeoDataFrame):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ringlist = set(points_gdf['ring_name'])

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {}
        for ring in ringlist:
            points_ring = points_gdf[points_gdf['ring_name'] == ring].copy()
            lines_ring = lines_gdf[lines_gdf['ring_name'] == ring].copy()
            future = executor.submit(bill_of_quantity, points_ring, lines_ring)
            futures[future] = ring
        
        points_compiled = []
        lines_compiled = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Process BOQ..."):
            ring = futures[future]
            try:
                result = future.result()
                if result:
                    points_result, lines_result = result
                    points_compiled.append(points_result)
                    lines_compiled.append(lines_result)
            except Exception as e:
                raise(f"🔴 Error BOQ in {ring}: {e}")

        points_compiled = pd.concat(points_compiled)
        lines_compiled = pd.concat(lines_compiled)
    return points_compiled, lines_compiled

def compile_dict(data_gdf:gpd.GeoDataFrame, column:str):
    data_list = []
    for idx, row in data_gdf.iterrows():
        col_data = row[column]

        if not isinstance(col_data, dict):
            continue
        if len(col_data) < 1:
            continue

        for col_name, geom in col_data.items():
            segment = {
                **{k: v for k, v in row.items() if k != 'geometry'},
                column: col_name,
                'geometry': shapely.from_wkt(geom)
            }
            data_list.append(segment)

    if data_list:
        data_gdf = gpd.GeoDataFrame(data_list, geometry='geometry', crs=data_gdf.crs)
    else:
        data_gdf = gpd.GeoDataFrame(columns=data_gdf.columns, geometry='geometry', crs=data_gdf.crs)
    return data_gdf

def identify_connection(
    ring: str,
    target_fiber: gpd.GeoDataFrame,
    target_point: gpd.GeoDataFrame,
    start_column: str = 'near_end'
) -> tuple:
    
    import numpy as np
    import geopandas as gpd

    # --- CRS normalize ---
    if target_fiber.crs != 'EPSG:3857':
        target_fiber = target_fiber.to_crs(epsg=3857)
    if target_point.crs != 'EPSG:3857':
        target_point = target_point.to_crs(epsg=3857)

    # --- Flatten any list/array values ---
    for col in ['near_end', 'far_end']:
        if col in target_fiber.columns:
            target_fiber[col] = target_fiber[col].apply(
                lambda x: x[0] if isinstance(x, (list, tuple, np.ndarray)) else x
            )

    # --- Validate start column ---
    if start_column == 'near_end':
        opposite_column = 'far_end'
    elif start_column == 'far_end':
        opposite_column = 'near_end'
    else:
        raise ValueError("start_column must be either 'near_end' or 'far_end'.")

    # --- Separate hub and site list ---
    fo_hub = target_point[target_point['site_type'].str.lower().str.contains('hub')].drop_duplicates('geometry')
    site_list = target_point[~target_point['site_type'].str.lower().str.contains('hub')].drop_duplicates('geometry')
    total_point = len(fo_hub) + len(site_list)

    # --- Identify starting hub ---
    hub_ids = fo_hub['site_id'].astype(str).tolist()
    start_hub = target_fiber[target_fiber[start_column].astype(str).isin(hub_ids)][start_column].values
    if len(start_hub) == 0:
        start_hub = target_fiber[target_fiber[opposite_column].astype(str).isin(hub_ids)][opposite_column].values
    if len(start_hub) == 0:
        print(f"❌ No FO Hub found in ring {ring}")
        return None, None

    start_hub = start_hub[0]

    # --- Sequential connection search ---
    connection = [start_hub]
    visited = set([start_hub])
    frontier = [start_hub]  # support branching

    while frontier:
        current = frontier.pop(0)

        # find all fiber segments connected to this site
        matches = target_fiber[
            (target_fiber[start_column] == current) | (target_fiber[opposite_column] == current)
        ]

        for _, seg in matches.iterrows():
            if seg[start_column] == current:
                next_sites = [seg[opposite_column]]
            else:
                next_sites = [seg[start_column]]

            for next_site in next_sites:
                if next_site not in visited:
                    visited.add(next_site)
                    connection.append(next_site)
                    frontier.append(next_site)

    # --- Build ordered GeoDataFrame of connection points ---
    points_sequential = []
    for site_id in connection:
        site_id = str(site_id)
        if site_id in fo_hub['site_id'].astype(str).values:
            row = fo_hub[fo_hub['site_id'].astype(str) == site_id].iloc[0].to_dict()
        elif site_id in site_list['site_id'].astype(str).values:
            row = site_list[site_list['site_id'].astype(str) == site_id].iloc[0].to_dict()
        else:
            print(f"⚠️ Site {site_id} not found.")
            continue
        points_sequential.append(row)

    if not points_sequential:
        print(f"⚠️ No valid points found for ring {ring}")
        return None, None

    points_sequential = gpd.GeoDataFrame(points_sequential, crs='EPSG:3857').reset_index(drop=True)
    return points_sequential, connection

def auto_sorter(df: pd.DataFrame|gpd.GeoDataFrame, column: str, sort_list: list):
    from itertools import groupby
    
    sort_list = [key for key, _ in groupby(sort_list)]
    order_map = {str(v): i for i, v in enumerate(sort_list)}
    if not df.empty:
        df['order'] = df[column].astype(str).map(order_map)
        df = df.sort_values('order', na_position='last').drop(columns='order')
    return df

def create_topology(points_gdf: gpd.GeoDataFrame, merge: bool = True) -> gpd.GeoDataFrame:
    if points_gdf.crs != 'EPSG:3857':
        points_gdf = points_gdf.to_crs(epsg=3857)

    points_gdf = points_gdf[points_gdf.geometry.notnull() & ~points_gdf.geometry.is_empty].copy()
    points_gdf["geometry"] = points_gdf.geometry.force_2d()

    ring_list = points_gdf['ring_name'].unique().tolist()
    topology_records = []

    for ring in ring_list:
        ring_points = points_gdf[points_gdf['ring_name'] == ring].reset_index(drop=True)
        if ring_points.empty:
            continue

        region = next((x for x in ring_points.get('region', []) if pd.notna(x)), 'Unknown Region')
        project = next((x for x in ring_points.get('project', []) if pd.notna(x)), 'Unknown Project')
        fo_hub_count = len(ring_points[ring_points['site_type'] == 'FO Hub'])

        for i in range(len(ring_points)):
            start_point = ring_points.iloc[i]
            end_point = ring_points.iloc[(i + 1) % len(ring_points)]

            # skip bad geometries
            if start_point.geometry is None or end_point.geometry is None:
                print(f"⚠️ Skipping segment in ring {ring}: invalid geometry.")
                continue

            # handle FO hub cases
            match fo_hub_count:
                case 1:
                    pass
                case 2:
                    if (i + 1) % len(ring_points) == 0:
                        continue
                case _:
                    raise ValueError(f"Ring {ring} has {fo_hub_count} FO Hubs, which is not supported.")

            try:
                start_coords = list(start_point.geometry.coords)[0][:2]
                end_coords = list(end_point.geometry.coords)[0][:2]
                line_geom = LineString([start_coords, end_coords])
            except Exception as e:
                print(f"⚠️ Failed to create line in ring {ring}: {e}")
                continue

            record = {
                'name': f"{start_point['site_id']}-{end_point['site_id']}",
                'near_end': start_point['site_id'],
                'far_end': end_point['site_id'],
                'ring_name': ring,
                'region': region,
                'project': project,
                'length': line_geom.length,
                'route_type': 'Topology',
                'fo_note': 'topology',
                'geometry': line_geom
            }
            topology_records.append(record)

    if not topology_records:
        print("⚠️ No topology records created.")
        return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs='EPSG:3857')

    topology_gdf = gpd.GeoDataFrame(topology_records, geometry='geometry', crs='EPSG:3857')

    if merge and not topology_gdf.empty:
        topology_gdf = topology_gdf.dissolve(by='ring_name')
        topology_gdf = topology_gdf[['geometry', 'region', 'project']].reset_index()
        topology_gdf['name'] = 'Connection'
        topology_gdf['geometry'] = topology_gdf['geometry'].apply(
            lambda geom: linemerge(geom) if geom.geom_type == 'MultiLineString' else geom
        )
    return topology_gdf

def compile_boq(points_boq:gpd.GeoDataFrame, lines_boq:gpd.GeoDataFrame):
    # BILL OF QUANTITY
    backbone = lines_boq[['near_end', 'far_end', 'ring_name', 'backbone', 'geometry']].copy()
    backbone = backbone.dropna(subset=['backbone'])
    if not backbone.empty:
        backbone['geometry'] = backbone['backbone'].apply(lambda geom:shapely.from_wkt(geom))
        backbone['name'] = "BB " + backbone['near_end'] + "-" + backbone['far_end']
        backbone['geometry'] = backbone['geometry'].apply(lambda geom: linemerge(geom) if geom.geom_type == "MultiLineString" else geom)
        backbone = backbone.drop(columns='backbone')
        backbone = backbone.to_crs(epsg=4326)

    access_ne = lines_boq[['near_end', 'far_end', 'ring_name', 'access_ne', 'geometry']].copy()
    access_ne = access_ne.dropna(subset=['access_ne'])
    if not access_ne.empty:
        access_ne['geometry'] = access_ne['access_ne'].apply(lambda geom:shapely.from_wkt(geom))
        access_ne['name'] = "Akses " + access_ne['near_end'] + "-" + access_ne['far_end']
        access_ne['geometry'] = access_ne['geometry'].apply(lambda geom: linemerge(geom) if geom.geom_type == "MultiLineString" else geom)
        access_ne = access_ne.drop(columns='access_ne')
        access_ne = access_ne.to_crs(epsg=4326)

    access_fe = lines_boq[['near_end', 'far_end', 'ring_name', 'access_fe', 'geometry']].copy()
    access_fe = access_fe.dropna(subset=['access_fe'])
    if not access_fe.empty:
        access_fe['geometry'] = access_fe['access_fe'].apply(lambda geom:shapely.from_wkt(geom))
        access_fe['name'] = "Akses " + access_fe['near_end'] + "-" + access_fe['far_end']
        access_ne['geometry'] = access_ne['geometry'].apply(lambda geom: linemerge(geom) if geom.geom_type == "MultiLineString" else geom)
        access_fe = access_fe.drop(columns='access_fe')
        access_fe = access_fe.to_crs(epsg=4326)

    odp = points_boq[['site_id', 'ring_name','geometry', 'odp']].copy()
    odp = odp.dropna(subset=['odp'])
    if not odp.empty:
        odp['geometry'] = odp['odp'].apply(lambda geom:shapely.from_wkt(geom))
        odp['name'] = "ODP " + odp['site_id']
        odp = odp.drop(columns='odp')
        odp = odp.to_crs(epsg=4326)
        odp['long'] = odp.geometry.x
        odp['lat'] = odp.geometry.y

    otb = points_boq[['site_id', 'ring_name','geometry', 'otb']].copy()
    otb = otb.dropna(subset=['otb'])
    if not otb.empty:
        otb['geometry'] = otb['otb'].apply(lambda geom:shapely.from_wkt(geom))
        otb['name'] = "OTB " + otb['site_id']
        otb = otb.drop(columns='otb')
        otb = otb.to_crs(epsg=4326)
        otb['long'] = otb.geometry.x
        otb['lat'] = otb.geometry.y
    
    # DIVIDED BY INTERSECTION FIBER
    fo_exist = lines_boq[['near_end', 'far_end', 'ring_name', 'fo_exist', 'geometry']].copy()
    fo_exist = compile_dict(fo_exist, 'fo_exist')
    if not fo_exist.empty:
        fo_exist['name'] = fo_exist['near_end'] + "-" + fo_exist['far_end'] + "/" + fo_exist['fo_exist'].astype(str)
        fo_exist['geometry'] = fo_exist['geometry'].apply(lambda geom: linemerge(geom) if geom.geom_type == "MultiLineString" else geom)
        fo_exist = fo_exist.to_crs(epsg=4326)

    pole_exist = lines_boq[['near_end', 'far_end', 'ring_name', 'pole_exist', 'geometry']].copy()
    pole_exist = compile_dict(pole_exist, 'pole_exist')
    if not pole_exist.empty:
        pole_exist['name'] = pole_exist['near_end'] + "-" + pole_exist['far_end'] + "/POLE EXT"
        pole_exist['geometry'] = pole_exist['geometry'].apply(lambda geom: linemerge(geom) if geom.geom_type == "MultiLineString" else geom)
        pole_exist = pole_exist.to_crs(epsg=4326)


    closure = lines_boq[['near_end', 'far_end', 'ring_name', 'closure', 'geometry']].copy()
    closure = compile_dict(closure, 'closure')
    closure = closure.to_crs(epsg=4326)
    if not closure.empty:
        closure = closure.explode(ignore_index=True)
        closure['name'] = "Closure " + closure['near_end'] + "-" + closure['far_end']
        closure['long'] = closure.geometry.x
        closure['lat'] = closure.geometry.y

    obstacle_railway = lines_boq[['near_end', 'far_end', 'ring_name', 'obstacle_railway', 'geometry']].copy()
    obstacle_railway = obstacle_railway.dropna(subset=['obstacle_railway'])
    if not obstacle_railway.empty:
        obstacle_railway['geometry'] = obstacle_railway['obstacle_railway'].apply(lambda geom:shapely.from_wkt(geom))
        obstacle_railway = obstacle_railway.drop(columns='obstacle_railway')
        obstacle_railway = obstacle_railway.to_crs(epsg=4326)
        obstacle_railway['long'] = obstacle_railway.geometry.x
        obstacle_railway['lat'] = obstacle_railway.geometry.y
        obstacle_railway['name'] = "Obstacle Rail " + obstacle_railway['near_end'] + "-" + obstacle_railway['far_end']

    obstacle_toll = lines_boq[['near_end', 'far_end', 'ring_name', 'obstacle_toll', 'geometry']].copy()
    obstacle_toll = obstacle_toll.dropna(subset=['obstacle_toll'])
    if not obstacle_toll.empty:
        obstacle_toll['geometry'] = obstacle_toll['obstacle_toll'].apply(lambda geom:shapely.from_wkt(geom))
        obstacle_toll = obstacle_toll.drop(columns='obstacle_toll')
        obstacle_toll = obstacle_toll.to_crs(epsg=4326)
        obstacle_toll['long'] = obstacle_toll.geometry.x
        obstacle_toll['lat'] = obstacle_toll.geometry.y
        obstacle_toll['name'] = "Obstacle Toll " + obstacle_toll['near_end'] + "-" + obstacle_toll['far_end']
    return odp, otb, closure, backbone, access_ne, access_fe, fo_exist, pole_exist, obstacle_railway, obstacle_toll

def excel_boq(points_boq:gpd.GeoDataFrame, lines_boq:gpd.GeoDataFrame, export_dir:str, **kwargs):
    program = kwargs.get("program", "N/A")
    vendor = kwargs.get("vendor", "TBG")

    lines_boq = lines_boq.copy()
    points_boq = points_boq.copy()

    if 'long' not in points_boq.columns or 'lat' not in points_boq.columns:
        points_boq['long'] = points_boq.geometry.to_crs(epsg=4326).x
        points_boq['lat'] = points_boq.geometry.to_crs(epsg=4326).y
    if 'vendor' not in points_boq.columns:
        points_boq['vendor'] = vendor
    if 'program' not in points_boq.columns:
        points_boq['program'] = program

    # used_columns = {
    #     "ring_name": "Ring ID",
    #     "site_id": "Site ID",
    #     "site_name": "Site Name" if "site_name" in points_boq.columns else "N/A",
    #     "long": "Long",
    #     "lat": "Lat",
    #     "region": "Region",
    #     "vendor": "Vendor" if "vendor" in points_boq.columns else "N/A",
    #     "program": "Program" if "program" in points_boq.columns else "N/A",
    #     "geometry": "geometry",
    #     }

    # available_col = [col for col in used_columns.keys() if col in points_boq.columns]
    
    # -- Sitelist & Hub --
    sitelist = points_boq[points_boq["site_type"].str.lower().str.contains('site')].copy()
    hubs = points_boq[points_boq["site_type"].str.lower().str.contains('hub')].copy()

    # sitelist = sitelist[available_col].rename(columns=used_columns)
    # hubs = hubs[available_col].rename(columns=used_columns)

    sitelist = sitelist.drop_duplicates('geometry')
    hubs = hubs.drop_duplicates('geometry')
    sitelist = sitelist.to_crs(epsg=3246)
    hubs = hubs.to_crs(epsg=3246)

    # -- Route --
    route = lines_boq.copy()
    route_columns = [ "near_end", "far_end", "geometry", "ring_name", "length"]
    route = route[route_columns].copy()
    route["name"] = route["near_end"] + "-" + route["far_end"]

    # BOQ
    result_boq = compile_boq(points_boq, lines_boq)
    odp, otb, closure, backbone, access_ne, access_fe, fo_exist, pole_exist, obstacle_railway, obstacle_toll = result_boq

    # SUMMARY
    # Columns for Points: site_id, site_name, ring_name, long, lat, region, vendor, program, type, geometry
    # Columns for Lines: near_end, far_end, ring_name, length, region, vendor, program, type, geometry
    
    # -- Sitelist Summary --
    sitelist['type'] = "Sitelist"
    sitelist['long'] = sitelist.geometry.to_crs(epsg=4326).x
    sitelist['lat'] = sitelist.geometry.to_crs(epsg=4326).y
    
    hubs['type'] = "FO Hub"
    hubs['long'] = hubs.geometry.to_crs(epsg=4326).x
    hubs['lat'] = hubs.geometry.to_crs(epsg=4326).y
    cols_sitelist = ["site_id", "site_name", "site_type", "type", "algo", "region", "ring_name", "vendor", "program", "geometry"]
    
    valid_col = []
    for col in cols_sitelist:
        if col in sitelist.columns and col in hubs.columns:
            valid_col.append(col)
    hubs = hubs[valid_col]
    sitelist = sitelist[valid_col]


    sheet_sitelist = pd.concat([hubs, sitelist], join='inner')
    sheet_sitelist = sheet_sitelist.sort_values('ring_name')
    sheet_sitelist = sheet_sitelist.drop_duplicates(['ring_name', 'geometry'])
    sheet_sitelist = sheet_sitelist.drop(columns="geometry")
    sheet_sitelist.columns = sheet_sitelist.columns.str.lower().str.replace(" ", "_")

    # -- Device --
    odp['type'] = "ODP"
    odp = odp.to_crs(epsg=4326)
    if not odp.empty and 'geometry' in odp:
        odp['long'] = odp.geometry.to_crs(epsg=4326).x
        odp['lat'] = odp.geometry.to_crs(epsg=4326).y

    otb['type'] = "OTB"
    otb = otb.to_crs(epsg=4326)
    if not otb.empty and 'geometry' in otb:
        otb['long'] = otb.geometry.to_crs(epsg=4326).x
        otb['lat'] = otb.geometry.to_crs(epsg=4326).y

    closure['type'] = "CL"
    closure = closure.to_crs(epsg=4326)
    if not closure.empty and 'geometry' in closure:
        closure['long'] = closure.geometry.to_crs(epsg=4326).x
        closure['lat'] = closure.geometry.to_crs(epsg=4326).y

    sheet_devices = pd.concat([odp, otb, closure], join='inner')
    sheet_devices = sheet_devices.sort_values('ring_name')
    sheet_devices = sheet_devices.drop_duplicates(['ring_name', 'geometry'])
    sheet_devices = sheet_devices.drop(columns="geometry")
    sheet_devices.columns = sheet_devices.columns.str.lower().str.replace(" ", "_")
    
    # -- Lines --
    route['type'] = 'Route'
    route = route.to_crs(epsg=4326)
    if not route.empty and 'geometry' in route:
        route['length'] = route.geometry.to_crs(epsg=3857).length
        route['name'] = route["near_end"] + "-" + route["far_end"]
    
    backbone['type'] = "Backbone"
    backbone = backbone.to_crs(epsg=4326)
    if not backbone.empty and 'geometry' in backbone:
        backbone['length'] = backbone.geometry.to_crs(epsg=3857).length
        backbone['name'] = "BB " + backbone["near_end"] + "-" + backbone["far_end"]
    
    access_fe['type'] = "Access"
    access_fe = access_fe.to_crs(epsg=4326)
    if not access_fe.empty and 'geometry' in access_fe:
        access_fe['length'] = access_fe.geometry.to_crs(epsg=3857).length
        access_fe['name'] = "Akses " + access_fe["near_end"] + "-" + access_fe["far_end"]
    
    fo_exist['type'] = "FO Existing"
    fo_exist = fo_exist.to_crs(epsg=4326)
    if not fo_exist.empty and 'geometry' in fo_exist:
        fo_exist['length'] = fo_exist.geometry.to_crs(epsg=3857).length
        fo_exist['name'] = fo_exist["near_end"] + "-" + fo_exist["far_end"] + "/" + fo_exist['fo_exist'].astype(str)
    
    pole_exist['type'] = "Pole Existing"
    pole_exist = pole_exist.to_crs(epsg=4326)
    if not pole_exist.empty and 'geometry' in pole_exist:
        pole_exist['length'] = pole_exist.geometry.to_crs(epsg=3857).length
        pole_exist['name'] = pole_exist["near_end"] + "-" + pole_exist["far_end"] + "/POLE EXT"

    sheet_routes = pd.concat([route, backbone, access_fe, fo_exist, pole_exist])
    sheet_routes = sheet_routes.sort_values('ring_name')
    sheet_routes = sheet_routes.drop_duplicates(['ring_name', 'geometry'])
    sheet_routes = sheet_routes.drop(columns="geometry")
    sheet_routes.columns = sheet_routes.columns.str.lower().str.replace(" ", "_")

    # -- Obstacle --
    obstacle_railway['type'] = "Obstacle Railway"
    obstacle_railway = obstacle_railway.to_crs(epsg=4326)
    if not obstacle_railway.empty and 'geometry' in obstacle_railway:
        obstacle_railway['long'] = obstacle_railway.geometry.x
        obstacle_railway['lat'] = obstacle_railway.geometry.y

    obstacle_toll['type'] = "Obstacle Toll"
    obstacle_toll = obstacle_toll.to_crs(epsg=4326)
    if not obstacle_toll.empty and 'geometry' in obstacle_toll:
        obstacle_toll['long'] = obstacle_toll.geometry.x
        obstacle_toll['lat'] = obstacle_toll.geometry.y

    sheet_obstacle = pd.concat([obstacle_railway, obstacle_toll])
    sheet_obstacle = sheet_obstacle.sort_values('ring_name')
    sheet_obstacle = sheet_obstacle.drop_duplicates(['ring_name', 'geometry'])
    sheet_obstacle = sheet_obstacle.drop(columns="geometry")
    sheet_obstacle.columns = sheet_obstacle.columns.str.lower().str.replace(" ", "_")

    # -- Summary --
    summ_sitelist = sheet_sitelist.groupby(['ring_name', 'type']).size().unstack(fill_value=0)
    summ_devices = sheet_devices.groupby(['ring_name', 'type']).size().unstack(fill_value=0)
    summ_routes = sheet_routes.groupby(['ring_name', 'type'])['length'].sum().unstack(fill_value=0)
    summ_obstacle = sheet_obstacle.groupby(['ring_name', 'type']).size().unstack(fill_value=0)
    summary_compiled = summ_sitelist.copy()
    summary_compiled = (
        summ_sitelist
        .join(summ_devices)
        .join(summ_routes)
        .join(summ_obstacle)
        .fillna(0)
    )

    # EXPORT EXCEL
    excel_path = os.path.join(export_dir, f"BOQ Report.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        if not summary_compiled.empty:
            sheet_name = "Summary"
            summary_compiled = summary_compiled.reset_index()
            excel_styler(summary_compiled).to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"📊 Excel sheet '{sheet_name}' with {len(summary_compiled):,} records written.")
        if not sheet_sitelist.empty:
            sheet_name = "Sitelist Information"
            sheet_sitelist = sheet_sitelist.reset_index(drop=True)
            excel_styler(sheet_sitelist).to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"📊 Excel sheet '{sheet_name}' with {len(sheet_sitelist):,} records written.")
        if not sheet_devices.empty:
            sheet_name = "Devices Information"
            sheet_devices = sheet_devices.reset_index(drop=True)
            excel_styler(sheet_devices).to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"📊 Excel sheet '{sheet_name}' with {len(sheet_devices):,} records written.")
        if not sheet_routes.empty:
            sheet_name = "Routes Information"
            sheet_routes = sheet_routes.reset_index(drop=True)
            excel_styler(sheet_routes).to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"📊 Excel sheet '{sheet_name}' with {len(sheet_routes):,} records written.")
        if not sheet_obstacle.empty:
            sheet_name = "Obstacle"
            sheet_obstacle = sheet_obstacle.reset_index(drop=True)
            excel_styler(sheet_obstacle).to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"📊 Excel sheet '{sheet_name}' with {len(sheet_obstacle):,} records written.")
    print("✅ Save Excel file BOQ Done.")

def kmz_boq(main_kml, lines_boq:gpd.GeoDataFrame, points_boq:gpd.GeoDataFrame, folder:str, **kwargs):
    program = kwargs.get("program", "N/A")
    vendor = kwargs.get("vendor", "TBG")
    
    lines_boq = lines_boq.copy()
    points_boq = points_boq.copy()

    def safe_get_geometry(site_id):
        match = points_boq.loc[points_boq["site_id"].astype(str) == str(site_id), "geometry"]
        if not match.empty:
            return match.iloc[0]
        else:
            print(f"⚠️ Missing geometry for site_id: {site_id} in folder {folder}.")
            return None

    lines_boq["start"] = lines_boq["near_end"].astype(str).apply(safe_get_geometry)
    lines_boq["end"]   = lines_boq["far_end"].astype(str).apply(safe_get_geometry)

    lines_boq = lines_boq.reset_index(drop=True)
    filename = folder.replace("/", "-")
    if 'long' not in points_boq.columns or 'lat' not in points_boq.columns:
        points_boq['long'] = points_boq.geometry.to_crs(epsg=4326).x
        points_boq['lat'] = points_boq.geometry.to_crs(epsg=4326).y
    if 'vendor' not in points_boq.columns:
        points_boq['vendor'] = vendor
    if 'program' not in points_boq.columns:
        points_boq['program'] = program

    used_columns = {
        "ring_name": "Ring ID",
        "site_id": "Site ID",
        "site_name": "Site Name" if "site_name" in points_boq.columns else "N/A",
        "long": "Long",
        "lat": "Lat",
        "region": "Region",
        "vendor": "Vendor" if "vendor" in points_boq.columns else "N/A",
        "program": "Program" if "program" in points_boq.columns else "N/A",
        "geometry": "geometry",
        }

    available_col = [col for col in used_columns.keys() if col in points_boq.columns]

    # DESIGN
    # -- Topology --
    try:
        print(f"ℹ️ Total Point {len(points_boq)}")
        point_conn, connection = identify_connection(ring=folder, target_fiber=lines_boq, target_point=points_boq)
        points_boq = point_conn.copy()
    except:
        print(f"Failed debug.")

    ring_topology = create_topology(points_boq)
    ring_topology = ring_topology.to_crs(epsg=4326)
    ring_topology["connection"] = "Connection"

    # -- Route --
    ring_route = lines_boq.copy()
    route_columns = [ "near_end", "far_end", "geometry", "ring_name", "length"]
    ring_route = ring_route[route_columns].copy()
    ring_route["name"] = ring_route["near_end"] + "-" + ring_route["far_end"]
    
    sorted_route = []
    for num, ne in enumerate(connection, start=1):
        near_end = ring_route[ring_route['near_end'].astype(str).str.strip() == str(ne).strip()].copy()
        if not near_end.empty:
            sorted_route.append(near_end)
        else:
            far_end = ring_route[ring_route['far_end'].astype(str).str.strip() == str(ne).strip()].copy()
            if not far_end.empty:
                print(f"🟢 {ne} not found as NE, but found as FE")
                sorted_route.append(far_end)
            else:
                print(f"🔴 {ne} not found in ring route.")
                print(ring_route[['near_end', 'far_end']])
    
    sorted_route = pd.concat(sorted_route)
    sorted_route = sorted_route.drop_duplicates('geometry').reset_index(drop=True)
    ring_route = sorted_route.copy()

    # -- Sitelist & Hub --
    ring_sites = points_boq[~points_boq["site_type"].str.lower().str.contains('hub')].copy()
    ring_hub = points_boq[points_boq["site_type"].str.lower().str.contains('hub')].copy()

    ring_sites = ring_sites[available_col].rename(columns=used_columns)
    ring_hub = ring_hub[available_col].rename(columns=used_columns)

    ring_sites = ring_sites.drop_duplicates('geometry')
    ring_hub = ring_hub.drop_duplicates('geometry')

    # -- DESIGN --
    ring_topology = ring_topology.to_crs(epsg=4326)
    ring_route = ring_route.to_crs(epsg=4326)
    ring_sites = ring_sites.to_crs(epsg=4326)
    ring_hub = ring_hub.to_crs(epsg=4326)

    kml_updated = export_kml(ring_topology, main_kml, filename, subfolder=folder, name_col="connection", color="#FF00FF", size=2, popup=False)
    kml_updated = export_kml(ring_route, kml_updated, filename, subfolder=f"{folder}/Route", name_col="name", color="#0000FF", size=3, popup=False)
    kml_updated = export_kml(ring_sites, kml_updated, filename, subfolder=f"{folder}/Site List", name_col="Site ID", color="#FFFF00", size=0.8, popup=True)
    kml_updated = export_kml(ring_hub, kml_updated, filename, subfolder=f"{folder}/FO Hub", name_col="Site ID", icon="http://maps.google.com/mapfiles/kml/paddle/A.png", size=0.8, popup=True)
    
    # -- BOQ --
    result_boq = compile_boq(points_boq, lines_boq)
    odp, otb, closure, backbone, access_ne, access_fe, fo_exist, pole_exist, obstacle_railway, obstacle_toll = result_boq

    backbone = backbone.to_crs(epsg=4326)
    access_fe = access_fe.to_crs(epsg=4326)
    fo_exist = fo_exist.to_crs(epsg=4326)
    pole_exist = pole_exist.to_crs(epsg=4326)
    odp = odp.to_crs(epsg=4326)
    otb = otb.to_crs(epsg=4326)
    closure = closure.to_crs(epsg=4326)
    obstacle_railway = obstacle_railway.to_crs(epsg=4326)
    obstacle_toll = obstacle_toll.to_crs(epsg=4326)

    kml_updated = export_kml(backbone, kml_updated, filename, subfolder=f"{folder}/Route Backbone", name_col="name", color="#0000FF", size=3, popup=False)
    kml_updated = export_kml(access_fe, kml_updated, filename, subfolder=f"{folder}/Route Akses", name_col="name", color="#FF0000", size=3, popup=False)
    kml_updated = export_kml(odp, kml_updated, filename, subfolder=f"{folder}/ODP", name_col="name", icon="http://maps.google.com/mapfiles/kml/shapes/triangle.png", color="#00FF00", size=0.8, popup=False)
    kml_updated = export_kml(otb, kml_updated, filename, subfolder=f"{folder}/OTB", name_col="name", icon="http://maps.google.com/mapfiles/kml/shapes/triangle.png", color="#00FF00", size=0.8, popup=False)
    kml_updated = export_kml(closure, kml_updated, filename, subfolder=f"{folder}/Closure", name_col="name", icon="http://maps.google.com/mapfiles/kml/shapes/triangle.png", color="#00FF00", size=0.8, popup=False)
    kml_updated = export_kml(fo_exist, kml_updated, filename, subfolder=f"{folder}/FO Existing", name_col="name", color="#00FF00", size=6, popup=False)
    kml_updated = export_kml(pole_exist, kml_updated, filename, subfolder=f"{folder}/Pole Existing", name_col="name", color="#FFFFFF", size=6, popup=False)
    kml_updated = export_kml(obstacle_railway, kml_updated, filename, subfolder=f"{folder}/Obstacle", name_col="name", icon="http://maps.google.com/mapfiles/kml/shapes/rail.png", color="#FFFFFF", size=0.8, popup=False)
    kml_updated = export_kml(obstacle_toll, kml_updated, filename, subfolder=f"{folder}/Obstacle", name_col="name", icon="http://maps.google.com/mapfiles/kml/shapes/cabs.png", color="#FFFFFF", size=0.8, popup=False)
    
    return kml_updated

def save_boq(points_boq:gpd.GeoDataFrame, lines_boq:gpd.GeoDataFrame, export_dir:str):
    result_boq = compile_boq(points_boq, lines_boq)
    odp, otb, closure, backbone, access_ne, access_fe, fo_exist, pole_exist, obstacle_railway, obstacle_toll = result_boq

    # CLEAN GEOMETRY
    clean_col = ['otb', 'odp','backbone','fo_exist', 'pole_exist', 'closure', 'obstacle_railway', 'obstacle_toll']
    for col in clean_col:
        if col in points_boq.columns:
            points_boq = points_boq.drop(columns=col)
        if col in lines_boq.columns:
            lines_boq = lines_boq.drop(columns=col)

    # EXPORT
    points_boq.to_parquet(os.path.join(export_dir, "Points_BOQ.parquet"))
    lines_boq.to_parquet(os.path.join(export_dir, "Lines_BOQ.parquet"))
    if not odp.empty:
        odp.to_parquet(os.path.join(export_dir, "ODP_BOQ.parquet"))
    if not otb.empty:
        otb.to_parquet(os.path.join(export_dir, "OTB_BOQ.parquet"))
    if not backbone.empty:
        backbone.to_parquet(os.path.join(export_dir, "Backbone_BOQ.parquet"))
    if not access_ne.empty:
        access_ne.to_parquet(os.path.join(export_dir, "Access_NE_BOQ.parquet"))
    if not access_fe.empty:
        access_fe.to_parquet(os.path.join(export_dir, "Access_FE_BOQ.parquet"))
    if not closure.empty:
        closure.to_parquet(os.path.join(export_dir, "Closure_BOQ.parquet"))
    if not fo_exist.empty:
        fo_exist.to_parquet(os.path.join(export_dir, "FO_Exist_BOQ.parquet"))
    if not pole_exist.empty:
        pole_exist.to_parquet(os.path.join(export_dir, "Pole_Exist_BOQ.parquet"))
    if not obstacle_railway.empty:
        obstacle_railway.to_parquet(os.path.join(export_dir, "Obstacle_Railway_BOQ.parquet"))
    if not obstacle_toll.empty:
        obstacle_toll.to_parquet(os.path.join(export_dir, "Obstacle_Toll_BOQ.parquet"))
    print(f"✅ Save BOQ Done.")

def main_boq(points:gpd.GeoDataFrame, lines:gpd.GeoDataFrame, export_dir:str, **kwargs):
    vendor = kwargs.get("vendor", "TBG")
    program = kwargs.get("program", "Not Defined")

    start_time = time.time()
    points_boq, lines_boq = parallel_boq(points, lines)

    # EXPORT
    save_boq(points_boq, lines_boq, export_dir)
    end_time = time.time()
    boq_time = round((end_time-start_time)/60, 2)
    
    # EXCEL FILE
    start_time = time.time()
    excel_boq(points_boq, lines_boq, export_dir)
    end_time = time.time()
    excel_time = round((end_time-start_time)/60, 2)

    # KMZ
    start_time = time.time()
    ring_names = points_boq['ring_name'].dropna().unique().tolist()
    output_kmz = os.path.join(export_dir, "BOQ KMZ Design.kmz")
    main_kmz = simplekml.Kml()
    for ring in tqdm(ring_names, total=len(ring_names), desc='Process KMZ BOQ'):
        ring_points = points_boq[points_boq['ring_name'] == ring].copy()  
        ring_lines = lines_boq[lines_boq['ring_name'] == ring].copy()
        main_kmz = kmz_boq(main_kmz, lines_boq=ring_lines, points_boq=ring_points, folder=ring, vendor=vendor, program=program)
        print(f"🟢 {ring} BOQ KMZ inserted.")
    sanitize_kml(main_kmz)
    main_kmz.savekmz(output_kmz)
    end_time = time.time()
    kmz_time = round((end_time-start_time)/60,2)

    print(f"✅ All BOQ Process Done.")
    print(f"ℹ️ Time Consumed:")
    print(f"BOQ Parallel Time   : {boq_time:,} minutes")
    print(f"Excel Result Time   : {excel_time:,} minutes")
    print(f"KMZ Result Time     : {kmz_time:,} minutes")

if __name__ == "__main__":
    all_points = gpd.read_parquet(r"D:\Data Analytical\PROJECT\TASK\NOVEMBER\Week 2\BoQ Intersite\Export\20251107\Intersite Design\W45_20251107\Supervised\Checkpoint\All_Points.parquet")
    all_lines = gpd.read_parquet(r"D:\Data Analytical\PROJECT\TASK\NOVEMBER\Week 2\BoQ Intersite\Export\20251107\Intersite Design\W45_20251107\Supervised\Checkpoint\All_Paths.parquet")
    
    # list_ring = [
    #     "TBG-MRS-H2B2NewSiteCoverage-DF066",
    #     "TBG-MRS-H2B2NewSiteCoverage-DF065",
    #     "TBG-MRS-H2B2NewSiteCoverage-DF064",
    #     "TBG-MRS-H2B2NewSiteCoverage-DF063",
    # ]
    
    # all_points = all_points[all_points['ring_name'].isin(list_ring)]
    # all_lines = all_lines[all_lines['ring_name'].isin(list_ring)]
    
    export_dir = r"D:\Data Analytical\PROJECT\TASK\NOVEMBER\Week 2\BoQ Intersite\Export\Trial BOQ\BOQ"
    os.makedirs(export_dir, exist_ok=True)
    
    start_time = time.time()
    points_boq, lines_boq = parallel_boq(all_points, all_lines)

    # EXPORT
    save_boq(points_boq, lines_boq, export_dir)
    end_time = time.time()
    boq_time = round((end_time-start_time)/60, 2)
    
    # EXCEL FILE
    start_time = time.time()
    excel_boq(points_boq, lines_boq, export_dir)
    end_time = time.time()
    excel_time = round((end_time-start_time)/60, 2)


    # KMZ
    start_time = time.time()
    ring_names = points_boq['ring_name'].dropna().unique().tolist()
    output_kmz = os.path.join(export_dir, "BOQ KMZ Design.kmz")
    main_kmz = simplekml.Kml()
    for ring in tqdm(ring_names, total=len(ring_names), desc='Process KMZ BOQ'):
        ring_points = points_boq[points_boq['ring_name'] == ring].copy()  
        ring_lines = lines_boq[lines_boq['ring_name'] == ring].copy()
        main_kmz = kmz_boq(main_kmz, lines_boq=ring_lines, points_boq=ring_points, folder=ring, vendor='TBG', program="BOQ Method")
        print(f"🟢 {ring} BOQ KMZ inserted.")
    sanitize_kml(main_kmz)
    main_kmz.savekmz(output_kmz)
    end_time = time.time()
    kmz_time = round((end_time-start_time)/60,2)

    print(f"✅ All BOQ Process Done.")
    print(f"ℹ️ Time Consumed:")
    print(f"BOQ Parallel Time   : {boq_time:,} minutes")
    print(f"Excel Result Time   : {excel_time:,} minutes")
    print(f"KMZ Result Time     : {kmz_time:,} minutes")