import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import MultiLineString
from shapely.ops import linemerge

def spof_detection(
    paths_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    G: nx.Graph,
    roads: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    threshold_spof: float = 500.0,
    threshold_alt: float = 25,
) -> tuple:
    """
    Detects Single Point of Failure (SPOF) in the paths.
    """
    from itertools import islice
    from shapely.ops import linemerge

    checked_route = {}
    spof_route = {}
    iter_gdf = paths_gdf.copy()

    paths_gdf['route_type'] = 'initial'
    for idx, row in iter_gdf.iterrows():
        ne = row["near_end"]
        fe = row["far_end"]
        node_ne = points_gdf[points_gdf["site_id"] == ne]["nearest_node"].values[0]
        node_fe = points_gdf[points_gdf["site_id"] == fe]["nearest_node"].values[0]

        path_geom = row["geometry"]
        length = row["length"]
        nodes_path = (
            nodes[nodes.geometry.intersects(path_geom)]["node_id"].unique().tolist()
        )
        total_nodes = len(nodes_path)
        if total_nodes == 0:
            continue

        path_roads = roads[
            roads["node_start"].isin(nodes_path) & roads["node_end"].isin(nodes_path)
        ]

        new_cable = path_roads[path_roads["ref_fo"] == 0]["geometry"]
        existing_cable = path_roads[path_roads["ref_fo"] == 1]["geometry"]
        new_length = float(new_cable.length.sum()) if not new_cable.empty else 0.0
        existing_length = (
            float(existing_cable.length.sum()) if not existing_cable.empty else 0.0
        )

        threshold_alt_calc = new_length * (1 + (threshold_alt / 100))

        if (ne, fe) in checked_route:
            checked_route[(ne, fe)] = list(set(checked_route[(ne, fe)] + nodes_path))
            continue

        if len(checked_route) > 0:
            total_checked = len(checked_route)
            print(f"ℹ️ Checking SPOF for {ne} to {fe}. Total Checked Routes: {total_checked}")

            for key, node_exist in list(checked_route.items()):
                # CHECKING SPOF
                protect_near = 10
                near_ne = set(
                    nx.single_source_shortest_path_length(
                        G, node_ne, cutoff=protect_near
                    ).keys()
                )
                near_fe = set(
                    nx.single_source_shortest_path_length(
                        G, node_fe, cutoff=protect_near
                    ).keys()
                )
                protected = near_ne | near_fe

                duplicated_nodes = set(node_exist) & set(nodes_path)
                duplicated_nodes = duplicated_nodes - protected

                duplicated_geom = roads[
                    roads["node_start"].isin(duplicated_nodes)
                    & roads["node_end"].isin(duplicated_nodes)
                ]["geometry"]
                duplicated_length = (
                    float(duplicated_geom.length.sum())
                    if not duplicated_geom.empty
                    else 0.0
                )

                if duplicated_length > threshold_spof:
                    print(f"{ne} to {fe} | ⚠️ SPOF issues. SPOF Length: {duplicated_length:.2f} / {length:.2f} ({duplicated_length / length:.2%})")
                    spof_route[(ne, fe)] = list(duplicated_nodes)
                    paths_gdf.loc[
                        (paths_gdf["near_end"] == ne) & (paths_gdf["far_end"] == fe),
                        "route_type",
                    ] = "spof"

                    # REDEFINE ALT G
                    G_ALT = G.copy()
                    PENALTY = 1e12
                    for u, v, d in G_ALT.edges(data=True):
                        if (u in duplicated_nodes) or (v in duplicated_nodes):
                            d["weight"] = d.get("weight", 1) + PENALTY

                    try:
                        print(f"{ne} to {fe} | 🔄 Finding alternative paths...")
                        alternative_paths = list(
                            islice(
                                nx.all_shortest_paths(
                                    G_ALT,
                                    source=node_ne,
                                    target=node_fe,
                                    weight="weight",
                                ),
                                2,
                            )
                        )

                        selected_alt = False
                        for alt_path in alternative_paths:
                            alt_path_roads = roads[
                                roads["node_start"].isin(alt_path)
                                & roads["node_end"].isin(alt_path)
                            ]

                            if alt_path_roads.empty:
                                continue

                            existing_alt_roads = alt_path_roads[
                                alt_path_roads["ref_fo"] == 1
                            ]
                            new_alt_roads = alt_path_roads[
                                alt_path_roads["ref_fo"] == 0
                            ]

                            existing_alt_length = (
                                existing_alt_roads["geometry"].length.sum()
                                if not existing_alt_roads.empty
                                else 0.0
                            )
                            new_alt_length = (
                                new_alt_roads["geometry"].length.sum()
                                if not new_alt_roads.empty
                                else 0.0
                            )

                            # CHECK ALTERNATIVE ROUTE
                            if new_alt_length > threshold_alt_calc:
                                print(f"{ne} to {fe} | 🔴 Alternative route new cable length exceeds threshold ({new_alt_length:.2f} > {threshold_alt_calc:.2f})")
                                continue

                            for connection, node_exist_inner in checked_route.items():
                                altspof = set(node_exist_inner) & set(alt_path)

                                if not altspof:
                                    selected_alt = True
                                    break

                                altspof_geom = roads[
                                    roads["node_start"].isin(altspof)
                                    & roads["node_end"].isin(altspof)
                                ]["geometry"]
                                altspof_length = (
                                    float(altspof_geom.length.sum())
                                    if not altspof_geom.empty
                                    else 0.0
                                )

                                if altspof_length < duplicated_length:
                                    print(
                                        f"{ne} to {fe} | 🟢 Alternative route found with better SPOF ({altspof_length:.2f}/{duplicated_length:.2f})"
                                    )
                                    selected_alt = True
                                    break
                                else:
                                    print(
                                        f"{ne} to {fe} | 🔴 Alternative got worse SPOF ({altspof_length:.2f}/{duplicated_length:.2f})"
                                    )

                            if selected_alt:
                                alt_route_geom = alt_path_roads["geometry"]
                                if not alt_route_geom.empty:
                                    alt_geom = alt_route_geom.union_all()
                                    alt_geom = (
                                        linemerge(alt_geom)
                                        if isinstance(alt_geom, MultiLineString)
                                        else alt_geom
                                    )
                                    alt_length = alt_geom.length
                                    length_diff = abs(length - alt_length)

                                    if length_diff < 2 * length:
                                        print(
                                            f"{ne} to {fe} | ✅ Alternative route found. Length Difference: {length_diff:.2f} meters"
                                        )

                                        # UPDATE CHECKED ROUTE
                                        print(
                                            f"ℹ️ Updating checked route for {ne} to {fe}."
                                        )
                                        paths_gdf.loc[
                                            (paths_gdf["near_end"] == ne)
                                            & (paths_gdf["far_end"] == fe),
                                            "geometry",
                                        ] = alt_geom
                                        paths_gdf.loc[
                                            (paths_gdf["near_end"] == ne)
                                            & (paths_gdf["far_end"] == fe),
                                            "length",
                                        ] = alt_length
                                        paths_gdf.loc[
                                            (paths_gdf["near_end"] == ne)
                                            & (paths_gdf["far_end"] == fe),
                                            "route_type",
                                        ] = "alternative"

                                        checked_route[(ne, fe)] = list(alt_path)
                                        break
                                    else:
                                        print(
                                            f"{ne} to {fe} | 🔴 Alternative route length {alt_length:.2f}. Exceeds double previous length {2 * length:.2f}."
                                        )
                                        continue

                            if selected_alt:
                                break

                        if not selected_alt:
                            print(
                                f"{ne} to {fe} | 🔴 No valid alternative paths found."
                            )
                            checked_route[(ne, fe)] = nodes_path

                    except nx.NetworkXNoPath:
                        print(f"{ne} to {fe} | 🔴 No alternative paths.")
                        checked_route[(ne, fe)] = nodes_path
                        continue
                    except Exception as e:
                        print(f"❌ Error finding alternative paths: {e}")
                        checked_route[(ne, fe)] = nodes_path
                        continue
                else:
                    checked_route[(ne, fe)] = nodes_path
        else:
            checked_route[(ne, fe)] = nodes_path

    return paths_gdf

def create_topology(points_gdf: gpd.GeoDataFrame, merge: bool=True) -> gpd.GeoDataFrame:
    from tqdm import tqdm
    from shapely.geometry import LineString
    from shapely.ops import linemerge

    if points_gdf.crs != 'EPSG:3857':
        points_gdf = points_gdf.to_crs(epsg=3857)

    ring_list = points_gdf['ring_name'].unique().tolist()
    topology_records = []
    for ring in ring_list:
        ring_points = points_gdf[points_gdf['ring_name'] == ring].reset_index(drop=True)
        region = ring_points['region'].mode()[0] if 'region' in ring_points.columns else 'Unknown Region'
        project = ring_points['project'].mode()[0] if 'project' in ring_points.columns else 'Unknown Project'
        fo_hub_count = len(ring_points[ring_points['site_type'].str.lower().str.contains('hub')])

        # if len(ring_points) < 3:
        #     print(f"⚠️ Ring {ring} has less than 3 points, skipping.")
        #     continue
        
        for i in range(len(ring_points)):
            start_point = ring_points.iloc[i]
            end_point = ring_points.iloc[(i + 1) % len(ring_points)]
            match fo_hub_count:
                case 1:
                    pass
                case 2:
                    if (i + 1) % len(ring_points) == 0:
                        continue
                case _:
                    raise ValueError(f"Ring {ring} has {fo_hub_count} FO Hubs, which is not supported.")

            line_geom = LineString([start_point.geometry, end_point.geometry])
            segment_name = f"{start_point['site_id']}-{end_point['site_id']}"
            record = {
                'name': segment_name,
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
    topology_gdf = gpd.GeoDataFrame(topology_records, columns=topology_records[0].keys(), geometry='geometry', crs='EPSG:3857')
    if merge:
        topology_gdf = topology_gdf.dissolve(by='ring_name')
        topology_gdf = topology_gdf[['geometry', 'region', 'project']].reset_index()
        topology_gdf['name'] = 'Connection'
        topology_gdf['geometry'] = topology_gdf['geometry'].apply(lambda geom: linemerge(geom) if geom.geom_type == 'MultiLineString' else geom)
    return topology_gdf

def route_path(start_node, end_node, G, roads, merged=False):
    try:
        cost, path = nx.bidirectional_dijkstra(G, start_node, end_node, weight='weight')
        path_geom = roads[roads['node_start'].isin(path) & roads['node_end'].isin(path)].drop_duplicates(subset='geometry').reset_index(drop=True)
        path_length = path_geom['length'].sum()
        if merged and not path_geom.empty:
            union_line = path_geom.geometry.union_all()
            merged_line = linemerge(union_line) if union_line.geom_type == 'MultiLineString' else union_line
            return path, merged_line, path_length
        return path, path_geom, path_length
    except nx.NetworkXNoPath:
        return None, gpd.GeoSeries(), 0

def dropwire_connection(path_geom, ne_point, fe_point, nodes, node_start, node_end):
    from shapely.geometry import LineString
    from shapely.ops import unary_union, linemerge

    ne_geom = ne_point['geometry']
    fe_geom = fe_point['geometry']

    nenode_geom = nodes.loc[nodes["node_id"] == node_start, "geometry"].values[0]
    fenode_geom = nodes.loc[nodes["node_id"] == node_end, "geometry"].values[0]

    line_ne = LineString([nenode_geom, ne_geom])
    line_fe = LineString([fenode_geom, fe_geom])

    connected_dropwire = linemerge(unary_union([line_ne, path_geom, line_fe]))
    length = connected_dropwire.length
    return connected_dropwire, length

def auto_group(data_gdf:gpd.GeoDataFrame, distance=25000):
    if data_gdf.crs != "EPSG:3857":
        data_gdf = data_gdf.to_crs(epsg=3857)

    groups = data_gdf.copy()
    groups["geometry"] = groups.geometry.buffer(distance)
    groups = groups.dissolve().explode(ignore_index=True)
    groups['region'] = groups.index + 1
    print(f"ℹ️ Total Group generated: {len(groups)}")

    return groups