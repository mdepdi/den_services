import os
import pandas as pd
import geopandas as gpd
import networkx as nx
import shapely
from shapely.geometry import box
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from core.config import settings
from modules.geometry import explode_lines

MAINDATA_DIR = settings.MAINDATA_DIR
DATA_DIR = settings.DATA_DIR
EXPORT_DIR = settings.EXPORT_DIR

def identify_hexagon(data_gdf, resolution=5, buffer=10000, type="bound"):
    """
    Identify hexagon identifiers in a GeoDataFrame.

    Args:
        data_gdf (GeoDataFrame): GeoDataFrame containing geometries with 'hex_id' column.
        resolution (int): Resolution of the hexagon grid.
        buffer (int): Buffer distance in meters to expand the bounding box.
        type (str): Type of hexagon to identify. Options are ['bound', 'convex'].

    Returns:
        list: List of unique hexagon identifiers.
    """
    from shapely.geometry import box

    hex_path = f"{MAINDATA_DIR}/22. H3 Hex/Hex_{resolution}.parquet"
    if not os.path.exists(hex_path):
        raise FileNotFoundError(f"Hexagon file not found at {hex_path}")
    hex_gdf = gpd.read_parquet(hex_path)

    # CONVERT TO 3857
    if hex_gdf.crs != "EPSG:3857":
        hex_gdf = hex_gdf.to_crs("EPSG:3857")
    if data_gdf.crs != "EPSG:3857":
        data_gdf = data_gdf.to_crs("EPSG:3857")

    match type:
        case "bound":
            bounding_box = data_gdf.total_bounds
            bbox_polygon = box(*bounding_box).buffer(buffer)
            hex_gdf = hex_gdf[hex_gdf.intersects(bbox_polygon)]
        case "convex":
            convex_hull = data_gdf.geometry.union_all().convex_hull.buffer(buffer)
            hex_gdf = hex_gdf[hex_gdf.intersects(convex_hull)]
        case _:
            raise ValueError("Invalid type specified. Use 'bound' or 'convex'.")
    if hex_gdf.empty:
        raise ValueError("No hexagons found for the given bounding box or convex hull.")

    hex_list = hex_gdf[f"hex_{resolution}"].unique().tolist()
    if not hex_list:
        raise ValueError("No hexagons found for the given bounding box.")
    return hex_list

def hex_island(data_gdf, resolution=5, island_name=None):
    """
    Identify hexagon identifiers for each island in a GeoDataFrame.

    Args:
        data_gdf (GeoDataFrame): GeoDataFrame containing geometries with 'Island' column.
        resolution (int): Resolution of the hexagon grid.

    Returns:
        dict: Dictionary with island names as keys and lists of unique hexagon identifiers as values.
    """
    hex_path = f"{MAINDATA_DIR}/22. H3 Hex/Hex_{resolution}.parquet"
    hex_df = gpd.read_parquet(hex_path)

    if island_name:
        hex_list = hex_df[hex_df['Island'] == island_name][f'hex_{resolution}'].unique().tolist()
        return hex_list
    else:
        if 'Island' not in data_gdf.columns:
            admin_2024 = gpd.read_parquet(f"{MAINDATA_DIR}/01. Admin/Admin_2024_Kabkot.parquet")
            data_gdf = gpd.sjoin(data_gdf, admin_2024[['Island', 'geometry']], how='left', predicate='within')
        islands = data_gdf.sort_values('Island', ascending=True)['Island'].unique().tolist()
        island = islands[0]

        hex_list = hex_df[hex_df['Island'] == island][f'hex_{resolution}'].unique().tolist()
        return hex_list

def retrieve_roads(hex_list, hex_dir=None, type="roads"):
    """
    Retrieve road data from hex files.
    Args:
        hex_list (list): List of hex identifiers.
        hex_dir (str, optional): Directory containing hex files. Defaults to None.
        type (str, optional): Type of geometries to retrieve. Options ['roads', 'nodes']
    Returns:
        list: List of GeoDataFrames containing road data.
    """

    if hex_dir is None:
        hex_dir = f"{MAINDATA_DIR}/03. Road Network/Adm 2024/Hexed Roads V2"

    all_data = []
    def load_hex_file(hex_id):
        match type:
            case "roads":
                hex_path = os.path.join(hex_dir, f"{hex_id}_roads.parquet")
            case "nodes":
                hex_path = os.path.join(hex_dir, f"{hex_id}_nodes.parquet")
            case _:
                raise ValueError("Invalid type specified. Use 'roads' or 'nodes'.")

        if os.path.exists(hex_path):
            data = pd.read_parquet(hex_path)
            if 'Island' in data:
                if 'node_id' in data.columns:
                    data['node_id'] = data['Island'] + '_' + data['node_id'].astype(str)
                if 'node_end' in data.columns:
                    data['node_end'] = data['Island'] + '_' + data['node_end'].astype(str)
                if 'node_start' in data.columns:
                    data['node_start'] = data['Island'] + '_' + data['node_start'].astype(str)
            if not data.empty and "geometry" in data.columns:
                return data
        else:
            return None

    all_data = []
    with ThreadPoolExecutor() as executor:
        future_to_hex = {executor.submit(load_hex_file, hex_id): hex_id for hex_id in hex_list}
        
        for future in as_completed(future_to_hex):
            id = future_to_hex[future]
            result = future.result()
            if result is not None:
                all_data.append(result)

    if not all_data:
        print(f"⚠️ No data found for hex list: {hex_list}.")
        return []
    all_data = pd.concat(all_data, ignore_index=True)
    all_data = all_data.drop_duplicates(subset="geometry").reset_index(drop=True)
    all_data['geometry'] = all_data["geometry"].apply(shapely.wkb.loads)
    all_data = gpd.GeoDataFrame(all_data, geometry='geometry', crs="EPSG:4326")
    return all_data


def build_graph(roads_gdf:gpd.GeoDataFrame, graph_type="route", cable_cost=35000, ref_fo=None, avoid_railway=True):
    """
    Build a graph from road geometries.

    Args:
        roads_gdf (GeoDataFrame): GeoDataFrame containing road geometries.
        graph_type (str): Type of graph to build. Options are ['route', 'fiber', 'full_fiber', 'full_weighted'].
        cable_cost(int, optional): Cost of cable to build. Defaults to 35000.

    Returns:
        networkx.Graph: Graph representation of the road network.
    """
    road_weight = {
    "trunk": 1,
    "primary": 1,
    "secondary": 1,
    "tertiary": 1,
    "unclassified": 1,
    "service": 1,
    # "living_street": 1,
    "residential": 2,
    "path": 99,
    "footway": 99,
    "cycleway": 99,
    "pedestrian": 99,
    "track": 99,
    "motorway": 99,
    }

    if ref_fo is None:
        ref_fo = set()
        if graph_type != "route":
            if "ref_fo" not in roads_gdf:
                raise ValueError("Reference FO column ('ref_fo') is required.")
            # REF FO
            ref_fo = set(roads_gdf[roads_gdf["ref_fo"] == 1]['node_start']) | set(roads_gdf[roads_gdf["ref_fo"] == 1]['node_end'])
            
    # BASE WEIGHT
    if graph_type == "full_weighted":
        roads_gdf = roads_gdf.drop(columns='road_weight')
        roads_gdf['road_weight'] = roads_gdf['highway'].map(road_weight).fillna(1)

    edges = []
    for row in roads_gdf.itertuples(index=True):
        node_start = row.node_start
        node_end = row.node_end
        identified_fo = node_start in ref_fo and node_end in ref_fo
        
        match graph_type:
            case "fiber":
                weight = row.length * (10000 if identified_fo else cable_cost)

            case "full_fiber":
                weight = row.length * (100 if identified_fo else cable_cost)

            case "full_weighted":
                base_weight = row.road_weight
                build_charge = row.build_charge if avoid_railway else 0
                weight = row.length * (10000 if identified_fo else cable_cost) * base_weight + build_charge

            case _:
                weight = row.length

        # Collect edge
        edges.append(
            (row.node_start, row.node_end, {"weight": weight, "length": row.length})
        )

    # Build graph
    G = nx.Graph()
    G.add_edges_from(edges)

    return G



def fiber_utilization(data: gpd.GeoDataFrame, ref_fo: list, roads: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame):
    # PREPARE DATA
    print("Preparing data...")
    if 'existing_cable_length' in data.columns:
        data = data.drop(columns='existing_cable_length')
    if 'new_cable_length' in data.columns:
        data = data.drop(columns='new_cable_length')

    data = explode_lines(data)
    data = data.drop_duplicates(subset='geometry')
    data = data.reset_index(drop=True)

    # CRS
    print("Setting CRS to EPSG:3857...")
    data = data.to_crs(epsg=3857)
    roads = roads.to_crs(epsg=3857)
    nodes = nodes.to_crs(epsg=3857)

    nodes_sindex = nodes.sindex
    data['nearest_node'] = data['geometry'].apply(lambda geom: nodes.loc[nodes_sindex.nearest(geom)[1][0], 'node_id'])

    #  REF FO
    data['ref_fo'] = data['nearest_node'].isin(ref_fo).astype(int)
    data['fo_note'] = data.apply(lambda x: 'Existing' if x['ref_fo'] == 1 else 'New', axis=1)
    data['length'] = data['geometry'].length

    return data

# ISLAND BASED
def retrieve_island(island, roads_dir=None, type='roads'):
    if roads_dir is None:
        roads_dir = f"{MAINDATA_DIR}/03. Road Network/Adm 2024/Preprocessed Island"

    match type:
        case "roads":
            data = pd.read_parquet(os.path.join(roads_dir, f"Roads_{island}.parquet"))
        case "nodes":
            data = pd.read_parquet(os.path.join(roads_dir, f"Nodes_{island}.parquet"))
        case _:
            raise ValueError("Invalid type specified. Use 'roads' or 'nodes'.")
    data['geometry'] = data["geometry"].apply(shapely.wkb.loads)
    data = gpd.GeoDataFrame(data, geometry='geometry', crs="EPSG:4326")
    return data

def reference_fo(road_gdf: gpd.GeoDataFrame, node_gdf: gpd.GeoDataFrame, ref_gdf: gpd.GeoDataFrame = None):
    """
    Identify reference fiber optic nodes and roads within a buffered area around the reference geometries.
    Args:
        ref_gdf (GeoDataFrame): GeoDataFrame containing reference geometries (e.g., existing fiber optic lines).
        road_gdf (GeoDataFrame): GeoDataFrame containing road geometries.   
        node_gdf (GeoDataFrame): GeoDataFrame containing node geometries.
    Returns:
        tuple: A set of reference fiber optic node IDs and a GeoDataFrame of roads within the buffered area.
    """
    if ref_gdf is None or ref_gdf.empty:
        ref_gdf = gpd.read_parquet(f"{MAINDATA_DIR}/06. FO TBG/Compile FO Route Only June 2025/FO TBG Only_01062025.parquet")

    if ref_gdf.crs != 'EPSG:3857':
        ref_gdf = ref_gdf.to_crs('EPSG:3857')
    if road_gdf.crs != 'EPSG:3857':
        road_gdf = road_gdf.to_crs('EPSG:3857')
    
    ref_buffer = ref_gdf.copy()
    ref_buffer['geometry'] = ref_buffer['geometry'].buffer(30)
    ref_fo = gpd.sjoin(node_gdf, ref_buffer[['geometry']], how='inner', predicate='intersects').drop(columns='index_right')
    ref_fo = set(ref_fo['node_id'])

    return ref_fo

def graph_island(roads_gdf:gpd.GeoDataFrame, graph_type="route", cable_cost=35000, ref_fo=None):
    """
    Build a graph from road geometries.

    Args:
        roads_gdf (GeoDataFrame): GeoDataFrame containing road geometries.
        graph_type (str): Type of graph to build. Options are ['route', 'fiber', 'full_fiber', 'full_weighted'].
        cable_cost(int, optional): Cost of cable to build. Defaults to 35000.

    Returns:
        networkx.Graph: Graph representation of the road network.
    """
    if graph_type != 'route' and ref_fo is None:
        raise ValueError('Ref FO is needed in fiber based graph.')

    edges = []
    for row in roads_gdf.itertuples(index=True):
        node_start = row.node_start
        node_end = row.node_end
        identified_fo = node_start in ref_fo and node_end in ref_fo
        
        match graph_type:
            case "fiber":
                weight = row.length * (10000 if identified_fo else cable_cost)

            case "full_fiber":
                weight = row.length * (100 if identified_fo else cable_cost)

            case _:
                weight = row.length

        # Collect edge
        edges.append((row.node_start, row.node_end, {"weight": weight, "length": row.length}))

    # Build graph
    G = nx.Graph()
    G.add_edges_from(edges)

    return G

def retrieve_building(hex_list, centroid=True, hex_dir=None, **kwargs):
    """
    Retrieve building data from hex files.
    Args:
        hex_list (list): List of hex identifiers.
        hex_dir (str, optional): Directory containing hex files. Defaults to None.
    Returns:
        list: List of GeoDataFrames containing building data.
    """
    import shutil
    from concurrent.futures import ThreadPoolExecutor, as_completed

    one_unit = kwargs.get("one_unit", False)
    area_building = kwargs.get("area_building", True)
    aspect_ratio = kwargs.get("aspect_ratio", True)
    parameters = kwargs.get("parameters",
        {
            "aspect_ratio_value": 0.25,
            "area_building_value": {
                "min": 25,
                "max": 500,
            },
        },
    )

    # print("ℹ️ Retrieve Building:")
    # print(f"Centroid        : {centroid}")
    # print(f"One unit        : {one_unit}")
    # print(f"Area building   : {area_building}")
    # print(f"Aspect ratio    : {aspect_ratio}\n")

    if hex_dir is None:
        hex_dir = f"{MAINDATA_DIR}/02. Building/Hexed Building 2024"

    all_data = []
    def load_hex_file(hex_id):
        try:
            hex_path = os.path.join(hex_dir, f"{hex_id}_buildings.parquet")
            if os.path.exists(hex_path):
                data = gpd.read_parquet(hex_path)
                data = data.to_crs(epsg=3857)
                if centroid:
                    data["geometry"] = data.geometry.centroid
                if aspect_ratio:
                    data = data[data["asp_ratio"] > parameters["aspect_ratio_value"]]
                if area_building:
                    data = data[(data["area_in_meters"] > parameters["area_building_value"]["min"]) & (data["area_in_meters"] < parameters["area_building_value"]["max"])]
                if one_unit:
                    data = data[data["one_unit"] == 1]

                if not data.empty and "geometry" in data.columns:
                    return data
            else:
                return None
        except Exception as e:
            print(f"Error Hex {hex_id}: {e}")
            source_dir = r"Z:\01. DATABASE\02. Building\Adm 2024\Hexed Building 2024"
            target_dir = (
                f"{MAINDATA_DIR}/02. Building/Adm 2024/Hexed Building 2024"
            )
            source_file = os.path.join(source_dir, f"{hex_id}_buildings.parquet")
            target_file = os.path.join(target_dir, f"{hex_id}_buildings.parquet")
            shutil.copy(source_file, target_file)
            print(f"ℹ️ Copied {hex_id} from Z.")
            return load_hex_file(hex_id)

    all_data = []
    with ThreadPoolExecutor() as executor:
        future_to_hex = {
            executor.submit(load_hex_file, hex_id): hex_id for hex_id in hex_list
        }

        for future in as_completed(future_to_hex):
            id = future_to_hex[future]
            result = future.result()
            if result is not None:
                all_data.append(result)

    if not all_data:
        print(f"⚠️ No data found for hex list: {hex_list}.")
        return []

    # print(f"ℹ️ Concating building data.")
    all_data = pd.concat(all_data, ignore_index=True)
    all_data = all_data.drop_duplicates(subset="geometry").reset_index(drop=True)
    all_data = gpd.GeoDataFrame(all_data, geometry="geometry", crs="EPSG:3857")

    if "geom_point" in all_data.columns:
        all_data = all_data.drop(columns="geom_point")
    if centroid:
        all_data["geometry"] = all_data.geometry.centroid
    return all_data