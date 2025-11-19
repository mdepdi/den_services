import os
import geopandas as gpd
import pandas as pd
import networkx as nx
from datetime import datetime
from time import time
from shapely.ops import linemerge
from tqdm import tqdm
import sys

sys.path.append(r"D:\JACOBS\SERVICE\API")

from modules.data import fiber_utilization, read_gdf
from modules.h3_route import identify_hexagon, retrieve_roads, build_graph
from modules.table import excel_styler, detect_week, sanitize_header
from modules.utils import spof_detection, create_topology, dropwire_connection
from modules.kml import export_kml, sanitize_kml
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# =============================
# UTILITIES
# ============================
def remove_z(geom):
    from shapely.geometry import Point, LineString, Polygon
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

def folders_identify(gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf['version'] = gdf['folders'].str.split(';').str[0].str.strip()
    gdf['region'] = gdf['folders'].str.split(';').str[1].str.strip()
    gdf['program'] = gdf['folders'].str.split(';').str[2].str.strip()
    gdf['ring_name'] = gdf['folders'].str.split(';').str[3].str.strip()
    gdf['type'] = gdf['folders'].str.split(';').str[-1].str.strip()
    return gdf

def route_path(start_node, end_node, G, roads, merged=False):
    try:
        cost, path = nx.bidirectional_dijkstra(G, start_node, end_node, weight='weight')
        path_geom = roads[roads['node_start'].isin(path) & roads['node_end'].isin(path)].reset_index(drop=True)
        path_length = path_geom['length'].sum()
        if merged and not path_geom.empty:
            merged_line = linemerge(path_geom.geometry.union_all())
            return path, merged_line, path_length
        return path, path_geom, path_length
    except nx.NetworkXNoPath:
        return None, gpd.GeoSeries(), 0

# =============================
# MAIN FUNCTION
# =============================
def fixroute_process(
    gdf_ne_ring: gpd.GeoDataFrame,
    gdf_fe_ring: gpd.GeoDataFrame,
    region: str,
    ring: str,
    export_loc: str = None,
) -> tuple:
    
    if export_loc is None:
        raise ValueError(f"Export location is not defined.")
    
    if not os.path.exists(export_loc):
        os.makedirs(export_loc, exist_ok=True)

    print(f"🔄 Processing Ring: {ring}")

    result_route = os.path.join(export_loc, f"Route Ring_{region}_{ring}.parquet")
    result_point = os.path.join(export_loc, f"Points Ring_{region}_{ring}.parquet")
    
    if os.path.exists(result_route):
        print(f"ℹ️ Ring {ring} already processed. Loading existing data...")
        paths = gpd.read_parquet(result_route)
        points = gpd.GeoDataFrame(columns=gdf_ne_ring.columns, geometry='geometry', crs=gdf_ne_ring.crs)
        return points, paths
    
    start_time = time()
    concated = pd.concat([gdf_ne_ring, gdf_fe_ring], ignore_index=True)
    concated = concated.drop_duplicates(subset=['geometry']).reset_index(drop=True)
    total_range = len(gdf_ne_ring)

    print(f"ℹ️ Total NE points: {len(gdf_ne_ring)}, Total FE points: {len(gdf_fe_ring)}")
    if gdf_ne_ring.empty or gdf_fe_ring.empty:
        print(f"⚠️ No NE or FE data for ring: {ring}. Skipping...")
    
    hex_list = identify_hexagon(concated, type="convex")
    print(f"ℹ️ Total {len(hex_list)} hexagons for ring: {ring}")
    roads = retrieve_roads(hex_list, type="roads")
    nodes = retrieve_roads(hex_list, type="nodes")

    concated = concated.to_crs(epsg=3857).reset_index(drop=True)
    roads = roads.to_crs(epsg=3857).reset_index(drop=True)
    nodes = nodes.to_crs(epsg=3857).reset_index(drop=True)
    G = build_graph(roads, graph_type='full_weighted')

    concated = gpd.sjoin_nearest(concated, nodes[['node_id', 'geometry']], how='left', distance_col='dist_to_node')
    gdf_ne_ring = gpd.sjoin_nearest(gdf_ne_ring, nodes[['node_id', 'geometry']], how='left', distance_col='dist_to_node')
    gdf_fe_ring = gpd.sjoin_nearest(gdf_fe_ring, nodes[['node_id', 'geometry']], how='left', distance_col='dist_to_node')
    concated = concated.rename(columns={'node_id': 'nearest_node'})
    gdf_ne_ring = gdf_ne_ring.rename(columns={'node_id': 'nearest_node'})
    gdf_fe_ring = gdf_fe_ring.rename(columns={'node_id': 'nearest_node'})

    segments = []
    for i in range(total_range):
        ne_point = gdf_ne_ring.iloc[i]
        fe_point = gdf_fe_ring.iloc[i]
        start_id = ne_point['site_id']
        end_id = fe_point['site_id']
        node_start = ne_point['nearest_node']
        node_end = fe_point['nearest_node']
        project = ne_point['project'] if 'project' in ne_point else None
        program = ne_point['program'] if 'program' in ne_point else None

        print(f"🔄 Routing from NE {start_id} to FE {end_id}...")
        try:
            path, path_geom, path_length = route_path(node_start, node_end, G, roads, merged=True)
            path_geom, path_length = dropwire_connection(path_geom, ne_point, fe_point, nodes, node_start, node_end)

            if not path_geom.is_empty:
                segment_name = f"{start_id}-{end_id}"

                segment_record = {
                    'name': segment_name,
                    'near_end': start_id,
                    'far_end': end_id,
                    'start': node_start,
                    'end': node_end,
                    'length': path_length,
                    'fo_note': 'merged',
                    'ring_name': ring,
                    'region': region,
                    'geometry': path_geom
                }

                if project:
                    segment_record['project'] = project
                if program:
                    segment_record['program'] = program

                segments.append(segment_record)
                print(f"🟢 Length: {path_length:10,.2f} m  | Routed segment    : {segment_name}")

                ## UPDATE GRAPH
                # # PENALTY EXISTING FIBER
                # for j in range(len(path) - 1):
                #     if G.has_edge(path[j], path[j + 1]):
                #         G[path[j]][path[j + 1]]['weight'] = 1e10
                #     if G.has_edge(path[j + 1], path[j]):
                #         G[path[j + 1]][path[j]]['weight'] = 1e10

                ## MAXIMIZE EXISTING FIBER
                for j in range(len(path) - 1):
                    if G.has_edge(path[j], path[j + 1]):
                        G[path[j]][path[j + 1]]['weight'] = G[path[j]][path[j + 1]]['weight'] / 3
                    if G.has_edge(path[j + 1], path[j]):
                        G[path[j + 1]][path[j]]['weight'] = G[path[j + 1]][path[j]]['weight'] / 3
            else:
                print(f"⚠️ No geometry found for segment {start_id} to {end_id}, skipping.")
        except nx.NetworkXNoPath:
            print(f"⚠️ No path found between {start_id} and {end_id}, skipping segment.")
    ring_paths = gpd.GeoDataFrame(segments, geometry='geometry', crs="EPSG:3857")

    # SPOF CHECKING
    ring_paths = spof_detection(ring_paths, concated, G, roads, nodes, threshold_spof=500, threshold_alt=25)

    # EXPORT
    if not concated.empty:
        result_point = os.path.join(export_loc, f"Points Ring_{region}_{ring}.parquet")
        concated.to_parquet(result_point)
    if not ring_paths.empty:
        result_route = os.path.join(export_loc, f"Route Ring_{region}_{ring}.parquet")
        ring_paths.to_parquet(result_route)
    print(f"✅ Ring {ring} processing completed.\n")

    end_time = time()
    elapsed_time = end_time - start_time
    print(f"⏱️ {ring} processed in {elapsed_time:,.2f} seconds")
    return concated, ring_paths

def parallel_fixroute(
    ne_data: gpd.GeoDataFrame,
    fe_data: gpd.GeoDataFrame,
    export_dir: str,
    MAX_WORKERS: int = 4,
    ROUTE_TYPE: str = 'merged',
    ) -> tuple:

    print(f"🚀 START PARALLEL FIX ROUTE")
    ring_list = ne_data['ring_name'].dropna().unique().tolist()
    print(f"🔄 Total Rings to Process: {len(ring_list):,}")

    all_new_points = []
    all_new_segments = []
    checkpoint_dir = os.path.join(export_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for ring in ring_list:
            ne_ring = ne_data[ne_data["ring_name"] == ring].reset_index(drop=True)
            fe_ring = fe_data[fe_data["ring_name"] == ring].reset_index(drop=True)
            region = ne_ring['region'].mode()[0]

            if ne_ring.empty:
                print(f"⚠️ No NE data for ring {ring}, skipping.")
                continue
            if fe_ring.empty:
                print(f"⚠️ No FE data for ring {ring}, skipping.")
                continue

            future = executor.submit(
                fixroute_process,
                ne_ring,
                fe_ring,
                region,
                ring,
                checkpoint_dir,
            )
            futures[future] = ring

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Rings"):
            ring = futures[future]
            try:
                new_points, new_segments = future.result()
                if new_points is not None:
                    all_new_points.append(new_points)
                if new_segments is not None:
                    all_new_segments.append(new_segments)
            except Exception as e:
                print(f"❌ Error processing ring {ring}: {e}")

    if all_new_points:
        all_new_points = pd.concat(all_new_points, ignore_index=True)
        print(f"ℹ️ Total new points collected: {len(all_new_points):,}")
    else:
        all_new_points = gpd.GeoDataFrame(columns=ne_data.columns, geometry="geometry", crs="EPSG:3857")
        print(f"ℹ️ No new points collected.")

    if all_new_segments:
        all_new_segments = pd.concat(all_new_segments, ignore_index=True)
        print(f"ℹ️ Total new segments collected: {len(all_new_segments):,}")
    else:
        raise ValueError(f"No new segments collected.")
    return all_new_points, all_new_segments

def export_kml_fixroute(
    points: gpd.GeoDataFrame,
    paths: gpd.GeoDataFrame,
    topology: gpd.GeoDataFrame,
    export_dir: str,
):
    import simplekml
    
    date_today = datetime.now().strftime("%Y%m%d")
    kmz_path = os.path.join(export_dir, f"Fixed Route_Fiberization_{date_today}.kmz")

    points = points.to_crs(epsg=4326).reset_index(drop=True)
    paths = paths.to_crs(epsg=4326).reset_index(drop=True)
    topology = topology.to_crs(epsg=4326).reset_index(drop=True)
    
    main_kml = simplekml.Kml()
    region_list = points['region'].dropna().unique().tolist()
    for region in region_list:
        print(f"🔄 Exporting KML for Region: {region}")
        region_points = points[points['region'] == region].copy()
        region_paths = paths[paths['region'] == region].copy()
        topology_region = topology[topology['region'] == region].copy()

        if 'long' not in region_points.columns or 'lat' not in region_points.columns:
            region_points['long'] = region_points.geometry.to_crs(epsg=4326).x
            region_points['lat'] = region_points.geometry.to_crs(epsg=4326).y
        if 'vendor' not in region_points.columns:
            region_points['vendor'] = 'TBG'
        if 'program' not in region_points.columns:
            region_points['program'] = 'N/A'

        used_columns = {
        "ring_name": "Ring ID",
        "site_id": "Site ID",
        "site_name": "Site Name" if "site_name" in region_points.columns else "N/A",
        "long": "Long",
        "lat": "Lat",
        "region": "Region",
        "vendor": "Vendor" if "vendor" in region_points.columns else "N/A",
        "program": "Program" if "program" in region_points.columns else "N/A",
        "geometry": "geometry",
        }
        available_col = [col for col in used_columns.keys() if col in region_points.columns]
        ring_list = region_points['ring_name'].dropna().unique().tolist()
        for ring in tqdm(ring_list, desc=f"Processing Rings in {region}"):
            ring_points = region_points[region_points['ring_name'] == ring].copy()
            ring_paths = region_paths[region_paths['ring_name'] == ring].copy()
            topology_ring = topology_region[topology_region['ring_name'] == ring].copy()
            topology_ring = topology_ring.dissolve(by='ring_name').reset_index()
            topology_ring['geometry'] = topology_ring.geometry.apply(lambda geom: linemerge(geom) if geom.geom_type == 'MultiLineString' else geom)
            topology_ring = topology_ring[['name','ring_name', 'region', 'geometry']]

            # FO HUB & SITELIST
            fo_hub = ring_points[ring_points['site_type'] == 'FO Hub'].copy().reset_index(drop=True)
            site_list = ring_points[ring_points['site_type'] != 'FO Hub'].copy().reset_index(drop=True)
            fo_hub = fo_hub[available_col]
            site_list = site_list[available_col]
            fo_hub = fo_hub.rename(columns=used_columns)
            site_list = site_list.rename(columns=used_columns)

            # print(f"ℹ️ Ring {ring}: {len(fo_hub)} FO Hubs, {len(site_list)} Site Lists, {len(topology_ring)} Topology segments, {len(ring_paths)} Route segments")
            main_kml = export_kml(topology_ring, main_kml, folder_name=f"{region}_{ring}_Topology", subfolder=f"{region}/{ring}", name_col='name', color="#FF00FF", size=3, popup=False)
            main_kml = export_kml(ring_paths, main_kml, folder_name=f"{region}_{ring}_Route", subfolder=f"{region}/{ring}/Route", name_col='name', color="#000FFF", size=3, popup=False)
            main_kml = export_kml(site_list, main_kml, folder_name=f"{region}_{ring}_Site List", subfolder=f"{region}/{ring}/Site List", name_col='Site ID', color="#FFFF00", size=0.8)
            main_kml = export_kml(fo_hub, main_kml, folder_name=f"{region}_{ring}_FO_Hub", subfolder=f"{region}/{ring}/FO Hub", name_col='Site ID', icon='http://maps.google.com/mapfiles/kml/paddle/A.png', size=0.8)
    
    sanitize_kml(main_kml)
    main_kml.savekmz(kmz_path)
    print(f"✅ Exported KML/KMZ file at {kmz_path}")

def export_fixroute(
    points: gpd.GeoDataFrame,
    paths: gpd.GeoDataFrame,
    topology: gpd.GeoDataFrame,
    paths_utilization: gpd.GeoDataFrame,
    export_dir: str,
    route_type: str = 'merged',
):
    date_today = datetime.now().strftime("%Y%m%d")
    # EXPORT PARQUET
    if not points.empty:
        points.to_crs(epsg=4326).to_parquet(os.path.join(export_dir, f"Points Fixed_Fiberization_{date_today}.parquet"), index=False)
        print(f"✅ Exported Fixed with {len(points):,} records.")
    if not paths.empty:
        paths.to_crs(epsg=4326).to_parquet(os.path.join(export_dir, f"Route Fixed_Fiberization_{date_today}.parquet"), index=False)
        print(f"✅ Exported Route Fixed with {len(paths):,} records.")
    if not paths_utilization.empty:
        paths_utilization = paths_utilization.sort_values(by=['ring_name']).reset_index(drop=True)
        paths_utilization.to_crs(epsg=4326).to_parquet(os.path.join(export_dir, f"Fiber Utilization_Fiberization_{date_today}.parquet"), index=False)
        print(f"✅ Exported Fiber Utilization with {len(paths_utilization):,} records.")
    if not topology.empty:
        topology = topology.sort_values(by=['ring_name']).reset_index(drop=True)
        topology.to_crs(epsg=4326).to_parquet(os.path.join(export_dir, f"Topology Route_Fiberization_{date_today}.parquet"), index=False)
        print(f"✅ Exported Topology Route with {len(topology):,} records.")

    # EXPORT KML
    if not points.empty and not paths.empty and not topology.empty:
        export_kml_fixroute(points, paths, topology, export_dir)

    # EXPORT EXCEL
    excel_path = os.path.join(export_dir, f"Summary Report_Fiberization_{date_today}.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        if not points.empty:
            sheet_name = "Site Information"
            points_report = points.drop(columns="geometry")
            excel_styler(points_report).to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"📊 Excel sheet '{sheet_name}' with {len(points_report):,} records written.")
        if not paths.empty:
            sheet_name = "Route Information"
            paths_report = paths.drop(columns="geometry")
            paths_report = paths_report.pivot_table(
                index=['name', 'near_end', 'far_end', 'ring_name'],
                columns='fo_note',
                values='length',
                aggfunc='sum',
                fill_value=0
            ).reset_index()

            match route_type:
                case 'merged':
                    paths_report = paths_report.rename(columns={'merged': 'Length'})
                case 'classified':
                    paths_report['total_length'] = paths_report.get('Existing', 0) + paths_report.get('New', 0)
            
            paths_report = paths_report.merge(
                paths[['ring_name', 'region', 'program']].drop_duplicates(),
                on='ring_name',
                how='left'
            )
            paths_report = paths_report.sort_values(by=['ring_name', 'near_end']).reset_index(drop=True)
            paths_report.columns = paths_report.columns.str.replace(' ', '_').str.lower()
            excel_styler(paths_report).to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"📊 Excel sheet '{sheet_name}' with {len(paths_report):,} records written.")
        if not paths_utilization.empty:
            sheet_name = "Fiber Utils (No Overlap)"
            paths_utilization = paths_utilization.drop(columns="geometry")
            grouped_utilization = paths_utilization.pivot_table(
                index=['ring_name'],
                columns='fo_note',
                values='length',
                aggfunc='sum',
                fill_value=0
            ).reset_index()

            grouped_utilization = grouped_utilization.merge(
                paths_utilization[['ring_name', 'region', 'program']].drop_duplicates(),
                on='ring_name',
                how='left'
            )
            grouped_utilization['total_Length'] = grouped_utilization.get('Existing', 0) + grouped_utilization.get('New', 0)
            grouped_utilization = grouped_utilization.sort_values(by='ring_name').reset_index(drop=True)
            grouped_utilization.columns = grouped_utilization.columns.str.replace(' ', '_').str.lower()
            excel_styler(grouped_utilization).to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"📊 Excel sheet '{sheet_name}' with {len(grouped_utilization):,} records written.")

def main_fixroute(
    ne_data: gpd.GeoDataFrame,
    fe_data: gpd.GeoDataFrame,
    export_dir: str,
    max_workers: int = 4,
    route_type: str = 'merged',
    program_name: str = 'N/A',
):
    # CRS
    if ne_data.crs != "EPSG:3857":
        ne_data = ne_data.to_crs(epsg=3857)

    if fe_data.crs != "EPSG:3857":
        fe_data = fe_data.to_crs(epsg=3857)

    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    # PROCESS FIXED ROUTING
    points_path = os.path.join(export_dir, f"Fixed Points_Fiberization_{date_today}.parquet")
    paths_path = os.path.join(export_dir, f"Fixed Route_Fiberization_{date_today}.parquet")
    if os.path.exists(points_path) and os.path.exists(paths_path):
        updated_points = gpd.read_parquet(points_path)
        updated_paths = gpd.read_parquet(paths_path)
        print(f"✅ Loaded existing processed data from {export_dir}.")
    else:
        updated_points, updated_paths = parallel_fixroute(
            ne_data, fe_data, export_dir, max_workers, route_type
        )

    if not updated_points.empty:
        print(f"ℹ️ Total updated points: {len(updated_points):,}")
    if not updated_paths.empty:
        print(f"ℹ️ Total updated paths: {len(updated_paths):,}")

    if route_type == 'classified' and not updated_paths.empty:
        updated_paths = fiber_utilization(updated_paths, overlap=True)

    if 'program' not in updated_points.columns:
        updated_points['program'] = program_name

    if 'program' not in updated_paths.columns:
        updated_paths['program'] = program_name

    # TOPOLOGY CHECK
    topology_paths = create_topology(updated_points)

    # IDENTIFY ROUTE UTILIZATION
    path_utils_loc = os.path.join(export_dir, f"Fiber Utilization.parquet")
    if os.path.exists(path_utils_loc):
        paths_utilization = gpd.read_parquet(path_utils_loc)
        print(f"✅ Loaded existing fiber utilization data from {path_utils_loc}.")
    else:
        paths_utilization = fiber_utilization(updated_paths, overlap=False)
        paths_utilization.to_parquet(os.path.join(export_dir, f"Fiber Utilization.parquet"), index=False)
        paths_utilization = paths_utilization[['ring_name','fo_note', 'length', 'geometry']]
        paths_utilization = paths_utilization.merge(
            updated_paths[["ring_name", "region", "program"]].drop_duplicates(),
            on="ring_name",
            how="left",
        )
        print(f"ℹ️ Total paths for utilization analysis: {len(paths_utilization):,}")

    # EXPORT
    export_fixroute(updated_points, updated_paths, topology_paths, paths_utilization, export_dir, ROUTE_TYPE)
    print(f"✅ Export completed.")

def validate_fixroute(df: pd.DataFrame):
    df = sanitize_header(df, lowercase=True)
    col_ne = df.columns[df.columns.str.contains("_a")].tolist() + ["ring_name", "region"]
    col_fe = df.columns[df.columns.str.contains("_b")].tolist() + ["ring_name", "region"]

    df_ne = df[col_ne].copy()
    df_fe = df[col_fe].copy()
    df_ne.columns = [col.replace("_a", "") for col in df_ne.columns]
    df_fe.columns = [col.replace("_b", "") for col in df_fe.columns]
    print(f"ℹ️ Validating NE and FE data...")
    try:
        geom_ne = gpd.points_from_xy(df_ne["longitude"], df_ne["latitude"])
        geom_fe = gpd.points_from_xy(df_fe["longitude"], df_fe["latitude"])
    except:
        try:
            geom_ne = gpd.points_from_xy(df_ne["long"], df_ne["lat"])
            geom_fe = gpd.points_from_xy(df_fe["long"], df_fe["lat"])
        except Exception as e:
            raise ValueError(e)

    print(f"ℹ️ Converting to GeoDataFrame...")
    gdf_ne = gpd.GeoDataFrame(df_ne, geometry=geom_ne, crs="EPSG:4326").to_crs(epsg=3857)
    gdf_fe = gpd.GeoDataFrame(df_fe, geometry=geom_fe, crs="EPSG:4326").to_crs(epsg=3857)
    gdf_ne["point_type"] = "NE"
    gdf_fe["point_type"] = "FE"
    print(f"ℹ️ Validation completed.\n")
    return gdf_ne, gdf_fe

if __name__ == "__main__":
    # CONFIGURATION
    ROUTE_TYPE = "merged"       # Options: 'merged' or 'classified'
    MAX_WORKERS = 8             # Number of parallel workers

    # DATE
    date_today = datetime.now().strftime("%Y%m%d")
    week = detect_week(date_today)

    # INPUT & OUTPUT PATH
    INPUT_DIR = fr"D:\JACOBS\TASK\SEPTEMBER\WEEK 5\FIX ROUTE"
    EXCEL_FILE = os.path.join(INPUT_DIR, f"Template_Fixed_Route.xlsx")
    EXPORT_DIR = fr"D:\JACOBS\TASK\SEPTEMBER\WEEK 5\FIX ROUTE\export\Fixed Ring_{date_today}_W{week}_v1.0"

    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)

    # PREPARE DATA
    print(f"🧩 DATA PREPARATION")
    if not os.path.exists(os.path.join(EXPORT_DIR, "prepared_data", f"Fixed Route Data.parquet")):
        print(f"ℹ️ Preparing data from input files...")
        fixroute_input = pd.read_excel(EXCEL_FILE)
        gdf_ne, gdf_fe = validate_fixroute(fixroute_input)

        # EXPORT PREPARED DATA
        data_dir = os.path.join(EXPORT_DIR, "prepared_data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        gdf_ne.to_parquet(os.path.join(data_dir, f"NE_Data.parquet"), index=False)
        gdf_fe.to_parquet(os.path.join(data_dir, f"FE_Data.parquet"), index=False)
        print(f"ℹ️ Data preparation completed.\n")
    else:
        print(f"ℹ️ Loading prepared data from {os.path.join(EXPORT_DIR, 'prepared_data')}")
        data_dir = os.path.join(EXPORT_DIR, "prepared_data")
        gdf_ne = gpd.read_parquet(os.path.join(data_dir, f"NE_Data.parquet"))
        gdf_fe = gpd.read_parquet(os.path.join(data_dir, f"FE_Data.parquet"))

        print(f"ℹ️ Prepared data loaded.\n")


    # DETAIL INPUT
    print(f"==============================")
    print(f"📂 Near End Data    : {len(gdf_ne):,} features")
    print(f"📂 Near End Data    : {len(gdf_fe):,} features")
    print(f"==============================")

    # TRIAL RUNNING FIX ROUTE
    # =============================
    start_time = time()
    main_fixroute(
        ne_data = gdf_ne,
        fe_data = gdf_fe,
        export_dir=EXPORT_DIR,
        max_workers=MAX_WORKERS,
        route_type=ROUTE_TYPE,
        program_name="Q1NewSite2026"
    )
    end_time = time()
    elapsed_time = end_time - start_time
    print(f"⏱️ Elapsed time: {elapsed_time/60:.2f} minutes")
    print(f"==============================")
    # =============================