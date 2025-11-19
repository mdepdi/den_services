import pandas as pd
import geopandas as gpd
import networkx as nx
import os
from shapely.ops import linemerge
from concurrent.futures import ThreadPoolExecutor, as_completed
from modules.table import find_best_match, sanitize_header
from modules.h3_route import identify_hexagon, retrieve_roads, build_graph

# UTILS
def remove_z(geom):
    from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon, GeometryCollection
    if geom.has_z:
        if geom.geom_type == 'Point':
            return Point(geom.x, geom.y)
        elif geom.geom_type == 'LineString':
            return LineString([(x, y) for x, y, z in geom.coords])
        elif geom.geom_type == 'Polygon':
            exterior = [(x, y) for x, y, z in geom.exterior.coords]
            interiors = [[(x, y) for x, y, z in interior.coords] for interior in geom.interiors]
            return Polygon(exterior, interiors)
        elif geom.geom_type.startswith('Multi') or geom.geom_type == 'GeometryCollection':
            geoms = [remove_z(part) for part in geom.geoms]
            return type(geom)(geoms)
    return geom

def safe_stringify(value):
    """Safely convert any value to string, handling geometric objects."""
    if pd.isna(value):
        return ''
    elif hasattr(value, 'wkt'):  # Shapely geometry object
        return str(value.wkt)  # Convert to WKT string
    elif hasattr(value, 'geom_type'):  # Another type of geometry
        return str(value)
    else:
        return str(value).strip()

# MODULES
# NEW RING
def input_newring(excel_file: str | pd.DataFrame | gpd.GeoDataFrame, method=None) -> gpd.GeoDataFrame:
    # READ EXCEL
    print(f"ℹ️ Checking {method} input data...")
    if isinstance(excel_file, pd.DataFrame):
        df = excel_file
    elif isinstance(excel_file, str):
        df = pd.read_excel(excel_file)
        print(f"📥 Reading Excel file: {excel_file}")
    elif isinstance(excel_file, gpd.GeoDataFrame):
        if 'lat' not in excel_file.columns or 'long' not in excel_file.columns:
            excel_file['lat'] = excel_file.geometry.to_crs(epsg=4326).y
            excel_file['long'] = excel_file.geometry.to_crs(epsg=4326).x
        df = pd.DataFrame(excel_file.drop(columns='geometry'))
        print(f"📥 Using provided GeoDataFrame.")

    # CHECKING USED COLUMNS
    match method:
        case 'supervised':
            required_columns = ["site_id", "site_name", "site_type", "lat", "long", "region", "ring_name", "flag"]
        case 'unsupervised':
            required_columns = ["site_id", "site_name", "site_type", "lat", "long", "region"]
        case _:
            raise ValueError("Method must be either 'supervised' or 'unsupervised'.")
        
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in sheets: {missing_cols}.\n Please ensure the input data contains {', '.join(required_columns)} columns.")
    
    # SANITIZE SITE ID AND NAME
    df["site_id"] = df["site_id"].apply(safe_stringify)
    df["site_name"] = df["site_name"].apply(safe_stringify)
    
    if "region" in df.columns:
        df["region"] = df["region"].apply(safe_stringify)

    # GEOMETRY
    df_geom = gpd.points_from_xy(df["long"], df["lat"], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(df, geometry=df_geom)

    # VALIDITY
    for idx, row in gdf.iterrows():
        if not row["geometry"].is_valid:
            raise ValueError(f"Invalid geometry at index {idx} with site_id '{row['site_id']}'.")
        
    gdf["geometry"] = gdf["geometry"].apply(remove_z)
    # SUMMARY
    print(f"✅ Input Data Validity Check Passed.")
    print(f"ℹ️ Total Records: {len(gdf):,}")
    return gdf

def input_unsupervised(sites, hubs) -> tuple:
    used_cols = ['site_id', 'site_name', 'site_type', 'long', 'lat']
    optional_cols = ['geometry', 'region', 'nearest_node']
    for col in used_cols:
        if col not in sites.columns:
            raise ValueError(f"Column '{col}' is missing from sites DataFrame.")
    for col in used_cols:
        if col not in hubs.columns:
            raise ValueError(f"Column '{col}' is missing from hubs DataFrame.")
    
    used_sites = used_cols.copy()
    used_hubs = used_cols.copy()
    for col in optional_cols:
        if col in sites.columns:
            used_sites.append(col)
        if col in hubs.columns:
            used_hubs.append(col)
    hubs = hubs[used_hubs].copy()
    sites = sites[used_sites].copy()

    for col in used_cols:
        if col in hubs.columns:
            hubs[col] = hubs[col].astype(str)
        if col in sites.columns:
            sites[col] = sites[col].astype(str)

    hubs_id = hubs['site_id'].dropna().unique().tolist()
    sites = sites[~sites['site_id'].isin(hubs_id)]

    # SUMMARY
    print(f"✅ Input Data Unsupervised Validity Check Passed.")
    print(f"ℹ️ Total Sitelist       : {len(sites):,} points")
    print(f"ℹ️ Total Hubs           : {len(hubs):,} points")
    
    return sites, hubs
# INSERT RING
def input_insertring(excel_file: str) -> tuple:
    used_sheets = ["New Sites", "Existing Sites", "Hub Sites"]
    with pd.ExcelFile(excel_file) as xls:
        sheet_names = xls.sheet_names
        for sheet in used_sheets:
            if sheet not in sheet_names:
                raise ValueError(f"Sheet '{sheet}' not found in the Excel file.")
        df_list = [pd.read_excel(xls, sheet_name=sheet) for sheet in used_sheets]
        new_sites_df, existing_sites_df, hub_sites_df = df_list

        # SANITIZE HEADERS
        new_sites_df = sanitize_header(new_sites_df)
        existing_sites_df = sanitize_header(existing_sites_df)
        hub_sites_df = sanitize_header(hub_sites_df)

        # CHECKING USED COLUMNS
        required_columns = {
            "New Sites": ["site_id", "site_name", "region", "lat", "long"],
            "Existing Sites": ["site_id", "site_name", "region", "lat", "long"],
            "Hub Sites": ["site_id", "site_name", "region", "lat", "long"],
        }

        for df, sheet in zip(df_list, used_sheets):
            missing_cols = [
                col for col in required_columns[sheet] if col not in df.columns
            ]
            if missing_cols:
                raise ValueError(f"Missing columns in sheet '{sheet}': {missing_cols}")
        
        # SANITIZE SITE ID AND NAME
        for df in [new_sites_df, existing_sites_df, hub_sites_df]:
            df["site_id"] = df["site_id"].apply(safe_stringify)
            df["site_name"] = df["site_name"].apply(safe_stringify)
            if "region" in df.columns:
                df["region"] = df["region"].apply(safe_stringify)

        # GEOMETRY
        newsites_geom = gpd.points_from_xy(
            new_sites_df["long"], new_sites_df["lat"], crs="EPSG:4326"
        )
        new_sites_gdf = gpd.GeoDataFrame(new_sites_df, geometry=newsites_geom)
        existingsites_geom = gpd.points_from_xy(existing_sites_df["long"], existing_sites_df["lat"], crs="EPSG:4326")
        existing_sites_gdf = gpd.GeoDataFrame(existing_sites_df, geometry=existingsites_geom)
        hubs_geom = gpd.points_from_xy(hub_sites_df["long"], hub_sites_df["lat"], crs="EPSG:4326")
        hub_sites_gdf = gpd.GeoDataFrame(hub_sites_df, geometry=hubs_geom)

        # VALIDITY
        for gdf, name in zip([new_sites_gdf, existing_sites_gdf, hub_sites_gdf], used_sheets):
            for idx, row in gdf.iterrows():
                if not row["geometry"].is_valid:
                    raise ValueError(f"Invalid geometry at index {idx} in sheet '{name}' with site_id '{row['site_id']}'.")

        # SITE TYPE
        new_sites_gdf["site_type"] = "New Site"
        existing_sites_gdf["site_type"] = "Existing Site"
        hub_sites_gdf["site_type"] = "FO Hub"

        # SUMMARY
        print(f"✅ Input Data Validity Check Passed.")
        print(f"ℹ️ Total New Sites       : {len(new_sites_gdf):,} points")
        print(f"ℹ️ Total Existing Sites  : {len(existing_sites_gdf):,} points")
        print(f"ℹ️ Total Hubs            : {len(hub_sites_gdf):,} points")
    return new_sites_gdf, existing_sites_gdf, hub_sites_gdf

def identify_fiberzone(
    new_sites: gpd.GeoDataFrame,
    prev_fiber: gpd.GeoDataFrame,
    search_radius: float = 2000,
) -> tuple:
    
    # CRS CHECK
    if new_sites.crs is None:
        new_sites.set_crs(epsg=4326, inplace=True)
    if prev_fiber.crs is None:
        prev_fiber.set_crs(epsg=4326, inplace=True)
    
    new_sites = new_sites.to_crs(epsg=3857)
    prev_fiber = prev_fiber.to_crs(epsg=3857)

    fiber_buffer = prev_fiber.copy()
    fiber_buffer['geometry'] = fiber_buffer.geometry.buffer(search_radius)  # 2 KM BUFFER
    newsites_within = gpd.sjoin(new_sites, fiber_buffer[['geometry']], how='inner', predicate='within').drop(columns='index_right')
    newsites_outside = new_sites[~new_sites.index.isin(newsites_within.index)]

    newsites_within = newsites_within.drop_duplicates(subset=['site_id', 'geometry']).reset_index(drop=True)
    newsites_outside = newsites_outside.drop_duplicates(subset=['site_id', 'geometry']).reset_index(drop=True)

    print(f"ℹ️ Total New Sites: {len(new_sites):,}")
    print(f"ℹ️ Within 2 KM of existing fiber : {len(newsites_within):,}")
    print(f"ℹ️ Outside 2 KM of existing fiber: {len(newsites_outside):,}")

    newsites_within = newsites_within.to_crs(epsg=4326)
    newsites_outside = newsites_outside.to_crs(epsg=4326)
    return newsites_within, newsites_outside

def define_ringdata(hubs:gpd.GeoDataFrame, sites_existing:gpd.GeoDataFrame, newsites_outside:gpd.GeoDataFrame)-> gpd.geodataframe:
    ring_data = pd.concat([hubs, sites_existing, newsites_outside], ignore_index=True)
    print(f"ℹ️ Total Combined Sites for Clustering: {len(ring_data):,}")

    duplicated = ring_data[ring_data.duplicated(subset=['geometry'])]
    print(f"⚠️ Found {len(duplicated):,} duplicated geometries.")

    ring_data = ring_data.drop_duplicates(subset=['geometry']).reset_index(drop=True)
    ring_data['site_id'] = ring_data['site_id'].astype(str)
    ring_data['site_name'] = ring_data['site_name'].astype(str)
    print(f"ℹ️ Total Combined Sites after dropping duplicates: {len(ring_data):,}")
    return ring_data

def folders_identify(gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf['version'] = gdf['folders'].str.split(';').str[0].str.strip()
    gdf['region'] = gdf['folders'].str.split(';').str[1].str.strip()
    gdf['project'] = gdf['folders'].str.split(';').str[2].str.strip()
    gdf['ring_name'] = gdf['folders'].str.split(';').str[3].str.strip()
    gdf['type'] = gdf['folders'].str.split(';').str[-1].str.strip()
    return gdf

def prepare_prevdata(prev_fiber:gpd.GeoDataFrame, prev_point:gpd.GeoDataFrame) -> tuple:
    if prev_fiber.crs is None:
        prev_fiber.set_crs(epsg=4326, inplace=True)
    if prev_point.crs is None:
        prev_point.set_crs(epsg=4326, inplace=True)
    
    prev_fiber = prev_fiber[['name', 'folders', 'geometry']].copy()
    prev_point = prev_point[['name', 'folders', 'geometry']].copy()

    prev_fiber = prev_fiber[prev_fiber['name'].str.lower() != 'connection'].reset_index(drop=True)
    prev_point = folders_identify(prev_point)
    prev_fiber = folders_identify(prev_fiber)
    prev_fiber = prev_fiber.rename(columns={'type': 'route_type'})
    prev_fiber['geometry'] = prev_fiber.geometry.apply(lambda x: linemerge(x) if x.geom_type == 'MultiLineString' else x)
    prev_point['geometry'] = prev_point.geometry.apply(lambda x: remove_z(x) if x.has_z else x)

    # MAPPING NEAR END AND FAR END
    sitetype_candidate = ['Site List', 'FO Hub']
    prev_point['type'] = prev_point['type'].map(lambda name: find_best_match(name, sitetype_candidate, 0.7)[0] if pd.notna(name) else name)
    prev_point['type'] = prev_point['type'].str.lower().str.contains('hub').map({True: 'FO Hub', False: 'Site List'})
    
    prev_fiber['near_end'] = prev_fiber['name'].str.split('-').str[0].str.strip()
    prev_fiber['far_end'] = prev_fiber['name'].str.split('-').str[-1].str.strip()

    return prev_fiber, prev_point

def nearest_ring(insert_gdf:gpd.GeoDataFrame, node_fiber:gpd.GeoDataFrame, G:nx.Graph, max_distance:int=2000) -> gpd.GeoDataFrame:
    fiber_nodes = set(node_fiber['node_id'])
    with ThreadPoolExecutor(max_workers=4) as executor:
        def calculate_nearest(row):
            source = row.nearest_node
            length_dict = nx.single_source_dijkstra_path_length(G, source, cutoff=max_distance, weight='length')
            if length_dict:
                nearest = {node: dist for node, dist in length_dict.items() if node in fiber_nodes}
                if nearest:
                    nearest_node = min(nearest, key=nearest.get)
                    distance = nearest[nearest_node]
                    return pd.Series({'nearest_fiber_node': nearest_node, 'distance_to_fiber': distance})
            return pd.Series({'nearest_fiber_node': None, 'distance_to_fiber': None})
        
        futures = {executor.submit(calculate_nearest, row): idx for idx, row in insert_gdf.iterrows()}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                if result is not None:
                    node_near = result['nearest_fiber_node']
                    dist_near = result['distance_to_fiber']
                    insert_gdf.at[idx, 'nearest_fiber_node'] = node_near
                    insert_gdf.at[idx, 'distance_to_fiber'] = dist_near
            except Exception as e:
                print(f"Error processing row {idx}: {e}")

        insert_gdf['nearest_ring'] = insert_gdf['nearest_fiber_node'].map(node_fiber.set_index('node_id')['ring_name'])
        insert_gdf['nearest_segment'] = insert_gdf['nearest_fiber_node'].map(node_fiber.set_index('node_id')['name'])
        insert_gdf['distance_to_fiber'] = insert_gdf['distance_to_fiber'].round(2)
        insert_gdf['nearest_fiber_node'] = insert_gdf['nearest_fiber_node']
    return insert_gdf

def identify_insertdata(newsites, prev_fiber, prev_points, search_radius=2000, max_member=None) -> gpd.GeoDataFrame:
    # CRS CHECK
    if newsites.crs is None:
        newsites.set_crs(epsg=4326, inplace=True)
    if prev_fiber.crs is None:
        prev_fiber.set_crs(epsg=4326, inplace=True)
    if prev_points.crs is None:
        prev_points.set_crs(epsg=4326, inplace=True)

    newsites = newsites.to_crs(epsg=3857)
    prev_fiber = prev_fiber.to_crs(epsg=3857)
    prev_points = prev_points.to_crs(epsg=3857)

    # LIST EXCEEDS MEMBER
    if max_member is not None:
        print(f" Checking for rings exceeding {max_member} members...")
        grouped_exceeds = prev_points.groupby('ring_name').filter(lambda x: len(x) > max_member)
        if not grouped_exceeds.empty:
            exceeds_rings = grouped_exceeds['ring_name'].unique().tolist()
            prev_points = prev_points[~prev_points['ring_name'].isin(exceeds_rings)].reset_index(drop=True)
            prev_fiber = prev_fiber[~prev_fiber['ring_name'].isin(exceeds_rings)].reset_index(drop=True)
            print(f"ℹ️ Rings with acceptable members retained for processing.")
        else:
            print(f"ℹ️ All rings are within the member limit.")

    fiber_buffer = prev_fiber.copy()
    fiber_buffer['geometry'] = fiber_buffer.geometry.buffer(10)

    region_list = newsites['region'].unique().tolist()
    mapped_insert = []
    for region in region_list:
        region_data = newsites[newsites['region'] == region].reset_index(drop=True)
        print(f"🔄 Processing region: {region} | Total Insert Site: {len(region_data):,}")

        # BUILD ROADS AND NODES
        hex_list = identify_hexagon(region_data, type='convex')
        print(f"ℹ️ Hexagons identified: {len(hex_list):,}")
        roads = retrieve_roads(hex_list, type='roads')
        nodes = retrieve_roads(hex_list, type='nodes')
        G = build_graph(roads, graph_type='fiber')

        roads = roads.to_crs(epsg=3857)
        nodes = nodes.to_crs(epsg=3857)
        print(f"ℹ️ Roads, nodes, graph ready.")

        node_fiber = gpd.sjoin(nodes, fiber_buffer[['geometry', 'ring_name', 'name']], how='inner', predicate='within').drop(columns=['index_right'])
        node_fiber = node_fiber.drop_duplicates(subset=['node_id']).reset_index(drop=True)
        print(f"ℹ️ Nodes within existing fiber: {len(node_fiber):,}")

        # NEAREST NODE
        region_data = gpd.sjoin_nearest(region_data, nodes[['node_id', 'geometry']], how='left', distance_col='dist_to_node')
        region_data = region_data.rename(columns={'node_id': 'nearest_node'})
        print(f"ℹ️ Nearest node assigned.")

        region_data = nearest_ring(region_data, node_fiber, G, max_distance=search_radius)
        print(f"ℹ️ Nearest fiber node and distance calculated.")

        # STORE
        mapped_insert.append(region_data)
        print(f"   ✅ Completed processing for region: {region}")
        
    mapped_insert = pd.concat(mapped_insert, ignore_index=True)
    print(f"✅ All regions processed. Total mapped insert sites: {len(mapped_insert):,}")
    dropped_insert = mapped_insert[(mapped_insert['distance_to_fiber'].isna()) & (mapped_insert['distance_to_fiber'] > 2000)].reset_index(drop=True)
    mapped_insert = mapped_insert[(~mapped_insert['distance_to_fiber'].isna()) & (mapped_insert['distance_to_fiber'] <= 2000)].reset_index(drop=True)
    print(f"ℹ️ Sites within {search_radius} meters to fiber: {len(mapped_insert):,}")
    print(f"ℹ️ Sites outside {search_radius} meters to fiber: {len(dropped_insert):,}")

    if 'index_right' in mapped_insert.columns:
        mapped_insert = mapped_insert.drop(columns=['index_right'])

    # MAPPING NEAR END AND FAR END
    mapped_insert['near_end'] = mapped_insert['nearest_segment'].str.split('-').str[0].str.strip()
    mapped_insert['far_end'] = mapped_insert['nearest_segment'].str.split('-').str[-1].str.strip()

    mapped_insert['nearest_fiber_node'] = mapped_insert['nearest_fiber_node'].astype(str).str.replace('.0', '', regex=False)
    mapped_insert = mapped_insert.reset_index(drop=True)
    mapped_insert = mapped_insert.to_crs(epsg=4326)
    
    # DRPPED INSERT
    if not dropped_insert.empty:
        dropped_insert = dropped_insert.reset_index(drop=True)
        dropped_insert = dropped_insert.to_crs(epsg=4326)
    else:
        dropped_insert = pd.DataFrame(columns=mapped_insert.columns)
        
    return mapped_insert, dropped_insert

# FIX ROUTE
def validate_fixroute(df: pd.DataFrame):
    df = sanitize_header(df, lowercase=True)
    col_ne = df.columns[df.columns.str.contains("_a")].tolist() + ["ring_name", "region"]
    col_fe = df.columns[df.columns.str.contains("_b")].tolist() + ["ring_name", "region"]

    df_ne = df[col_ne].copy()
    df_fe = df[col_fe].copy()
    df_ne.columns = [col.replace("_a", "") for col in df_ne.columns]
    df_fe.columns = [col.replace("_b", "") for col in df_fe.columns]
    print(f"ℹ️ Validating NE and FE data...")
    geom_ne = gpd.points_from_xy(df_ne["longitude"], df_ne["latitude"])
    geom_fe = gpd.points_from_xy(df_fe["longitude"], df_fe["latitude"])
    print(f"ℹ️ Converting to GeoDataFrame...")
    gdf_ne = gpd.GeoDataFrame(df_ne, geometry=geom_ne, crs="EPSG:4326").to_crs(epsg=3857)
    gdf_fe = gpd.GeoDataFrame(df_fe, geometry=geom_fe, crs="EPSG:4326").to_crs(epsg=3857)
    gdf_ne["point_type"] = "NE"
    gdf_fe["point_type"] = "FE"
    print(f"ℹ️ Validation completed.\n")
    return gdf_ne, gdf_fe