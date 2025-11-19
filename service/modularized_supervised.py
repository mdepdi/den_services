import geopandas as gpd
import networkx as nx
import pandas as pd
import os
import simplekml
import numpy as np
import math
import random
import zipfile
import sys

sys.path.append(r"D:\Data Analytical\SERVICE\API")

from time import time
from datetime import datetime
from tqdm import tqdm
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import linemerge
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from itertools import permutations
from modules.data import fiber_utilization
from modules.h3_route import identify_hexagon, retrieve_roads, build_graph
from modules.utils import spof_detection, create_topology, route_path, dropwire_connection
from modules.table import sanitize_header, detect_week, excel_styler
from modules.validation import input_newring
from modules.kml import export_kml, sanitize_kml

# DISTANCE MAPPING
def distance_mapping(cluster_sites, G=None, weight="weight"):
    mapping = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for i, site_i in cluster_sites.iterrows():
            id_i = site_i["site_id"]
            node_i = site_i["nearest_node"]
            for j, site_j in cluster_sites.iterrows():
                if i == j or (j, i) in futures:
                    continue

                id_j = site_j["site_id"]
                node_j = site_j["nearest_node"]
                futures[
                    executor.submit(
                        nx.shortest_path_length,
                        G,
                        source=node_i,
                        target=node_j,
                        weight=weight,
                    )
                ] = (id_i, id_j)

    for future in tqdm(
        as_completed(futures.keys()),
        total=len(futures),
        desc=f"🧩 Calculating shortest paths",
    ):
        try:
            id_i, id_j = futures[future]
            path_length = future.result() if future.result() is not None else 999999
            mapping[(id_i, id_j)] = path_length
            mapping[(id_j, id_i)] = path_length
        except Exception as e:
            print(f"Error calculating path between {i} and {j}: {e}")
            mapping[(id_i, id_j)] = 999999
            mapping[(id_j, id_i)] = 999999
    return mapping


# ===============
# ROUTING METHODS
# ===============

# MULTI HUB METHOD
def optimal_routes(sitelist, distance_map, start_hub, end_hub, max_samples=1000):
    sites = sitelist["site_id"].unique().tolist()

    total_perms = math.factorial(len(sites))
    print(f"ℹ️ Total permutations to check: {total_perms:,}")

    if total_perms > max_samples:
        print(
            f"⚠️ Too many permutations ({total_perms:,}). Sampling {max_samples} random permutations."
        )
        all_perms = list(permutations(sites))
        sampled_perms = random.sample(all_perms, max_samples)
        print(f"ℹ️ Sampled {len(sampled_perms):,} permutations.")
    else:
        sampled_perms = list(permutations(sites))
        print(f"ℹ️ Using all {len(sampled_perms):,} permutations.\n")

    best_length = float("inf")
    best_route = None

    def parallel_perm(perm, distance_map, start_hub, end_hub, current_best):
        current_length = 0
        current_route = [start_hub] + list(perm) + [end_hub]
        for i in range(len(current_route) - 1):
            id_i = current_route[i]
            id_j = current_route[i + 1]
            if (id_i, id_j) in distance_map:
                current_length += distance_map[(id_i, id_j)]
            else:
                current_length += 999999
            if current_length >= current_best:
                return None, None

        return current_route, current_length

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                parallel_perm, perm, distance_map, start_hub, end_hub, best_length
            ): perm
            for perm in sampled_perms
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="🧩 Calculating optimal routes",
        ):
            try:
                result = future.result()
                if result[0] is not None and result[1] < best_length:
                    best_route, best_length = result

            except Exception as e:
                print(f"Error processing permutation: {e}")
                continue

    if best_route is None:
        print(f"❌ No valid route found")
        return None, float("inf")
    return best_route, best_length


def christofides_tsp(
    sitelist,
    distance_map,
    start_hub,
    end_hub,
    weight="weight",
    two_opt=True,
    max_2opt_iter=2000,
):
    import networkx as nx
    from networkx.algorithms import approximation

    sites = sitelist["site_id"].tolist()
    all_sites = [start_hub] + sites + [end_hub]

    # Graph
    G = nx.Graph()
    G.add_nodes_from(all_sites)

    # Add edges with weights
    for (site1, site2), distance in distance_map.items():
        if site1 in all_sites and site2 in all_sites:
            G.add_edge(site1, site2, weight=float(distance))

    for u in list(G.nodes):
        for v in list(G.nodes):
            if u != v and not G.has_edge(u, v):
                w = G[u][v]["weight"] if G.has_edge(u, v) else float("inf")
                G.add_edge(u, v, weight=w)

    DUMMY = "_DUMMY_"
    G.add_node(DUMMY)
    for node in all_sites:
        if node == start_hub or node == end_hub:
            G.add_edge(DUMMY, node, weight=0.0)
        else:
            G.add_edge(DUMMY, node, weight=float("inf"))

    try:
        tsp_path = approximation.christofides(G, weight=weight)
    except nx.NetworkXError as e:
        print(f"❌ Error in Christofides TSP: {e}")
        return None, None

    if len(tsp_path) > 1 and tsp_path[0] == tsp_path[-1]:
        tsp_path = tsp_path[:-1]

    if DUMMY in tsp_path:
        i = tsp_path.index(DUMMY)
        tsp_path = tsp_path[i:] + tsp_path[:i]
    else:
        print("❌ DUMMY node not found in TSP path. Cannot enforce endpoints.")
        return None, None

    if len(tsp_path) < 3:
        print("❌ TSP path is too short. Cannot enforce endpoints.")
        return None, None

    a = tsp_path[0]
    b = tsp_path[-1]
    path = tsp_path[1:]

    if path[0] != start_hub or path[-1] != end_hub:
        if path[0] == end_hub and path[-1] == start_hub:
            path = list(reversed(path))
    else:
        if a == end_hub and b == start_hub:
            path = list(reversed(path))
        else:
            if start_hub in path:
                i = path.index(start_hub)
                path = path[i:] + path[:i]

            if path[-1] != end_hub and end_hub in path:
                j = path.index(end_hub)
                path = list(reversed(path[j:] + path[:j]))

    def path_length(seq):
        total = 0.0
        for u, v in zip(seq[:-1], seq[1:]):
            if (u, v) in distance_map:
                total += float(distance_map[(u, v)])
            elif (v, u) in distance_map:
                total += float(distance_map[(v, u)])
            elif G.has_edge(u, v):
                total += float(G[u][v]["weight"])
            else:
                total += 1e12
        return total

    if two_opt and len(path) > 4:
        best = path[:]
        best_len = path_length(best)
        improved = True
        it = 0
        while improved and it < max_2opt_iter:
            improved = False
            it += 1
            for i in range(1, len(best) - 2):
                for k in range(i + 1, len(best) - 1):
                    new_path = best[:i] + best[i : k + 1][::-1] + best[k + 1 :]
                    new_len = path_length(new_path)
                    if new_len + 1e-9 < best_len:
                        best, best_len = new_path, new_len
                        improved = True
                        break
                if improved:
                    break
        path, total_len = best, best_len
    else:
        total_len = path_length(path)
    return path, total_len


# ==============
# ROUTING HUBS
# ==============
def single_fohub(
    cluster_site,
    cluster,
    fo_hub,
    roads,
    nodes,
    G,
    area="area",
    area_col="region",
):
    from modules.scoring import duplicate_scores
    from modules.graph import build_cluster_graph
    from shapely.ops import linemerge

    # BUILD CLUSTER GRAPH
    cluster_site = cluster_site.drop_duplicates(subset="nearest_node")
    cluster_graph = build_cluster_graph(cluster_site, G, node_col="nearest_node")
    if cluster_graph is None:
        print(f"🔴 Skipping ring {cluster}. No route available.")
        return None, None
    
    list_nodes = list(cluster_graph.nodes)
    hub_node = fo_hub["nearest_node"].iloc[0] if not fo_hub.empty else None
    if hub_node and hub_node in list_nodes:
        reordered = [hub_node] + [node for node in list_nodes if node != hub_node]
        if reordered[0] != reordered[-1]:
            init_cycle = reordered + [reordered[0]]
        else:
            init_cycle = reordered
    else:
        if list_nodes[0] != list_nodes[-1]:
            init_cycle = list_nodes + [list_nodes[0]]
        else:
            init_cycle = list_nodes

    # AUTOMATE SITE PATH ALGHORITHM
    algorithms = {
        "Threshold TSP": lambda g: nx.approximation.threshold_accepting_tsp(
            g,
            init_cycle=init_cycle,
            weight="weight",
            max_iterations=10,
            seed=42,
            source=hub_node,
        ),
        "Simulated Annealing": lambda g: nx.approximation.simulated_annealing_tsp(
            g,
            init_cycle=init_cycle,
            weight="weight",
            max_iterations=10,
            seed=42,
            source=hub_node,
        ),
        "Christofides": lambda g: nx.approximation.christofides(g, weight="weight"),
    }

    best_algo = None
    for algo_name, algo_func in algorithms.items():
        try:
            print(f"\n🔃 Running {algo_name} algorithm | Cluster {cluster}")
            sitepath = algo_func(cluster_graph)

            if not sitepath:
                continue

            if sitepath and len(sitepath) > 1:
                paths_data = []
                points_data = []
                for idx, node in enumerate(sitepath[:-1]):
                    start_site = cluster_site[cluster_site["nearest_node"] == node].iloc[0]
                    end_site = cluster_site[cluster_site["nearest_node"] == sitepath[idx + 1]].iloc[0]
                    start_node = start_site["nearest_node"]
                    end_node = end_site["nearest_node"]

                    try:
                        path, path_geom, path_length = route_path(start_node, end_node, G, roads, merged=True)
                        path_geom, path_length = dropwire_connection(path_geom, start_site, end_site, nodes, start_node, end_node)
                        print(f"Cluster {cluster:20} | From {start_site['site_id']:>15} to {end_site['site_id']:>15} | Length: {path_length:8.2f} meters")
                        
                        # STORE PATHS
                        paths_data.append(
                            {
                                "near_end": start_site["site_id"],
                                "far_end": end_site["site_id"],
                                "algo": algo_name,
                                area_col: area,
                                "ring_name": cluster,
                                "route_type": "merged",
                                "length": round(path_length, 2),
                                "geometry": path_geom,
                            }
                        )
                    except nx.NetworkXNoPath:
                        print(f"No path found for {algo_name} | Cluster {cluster} | Start: {start_site['SiteId TBG']} | End: {end_site['SiteId TBG']}")
                        continue
                    except Exception as e:
                        print(f"❌ Error processing path for {algo_name} | Cluster {cluster}: {e}")
                        continue

                # STORE POINTS
                for node in sitepath:
                    site = cluster_site[cluster_site["nearest_node"] == node].iloc[0]
                    if site["site_id"] not in [point["site_id"] for point in points_data]:
                        points_data.append(
                            {
                                "site_id": site["site_id"],
                                "site_name": (site["site_name"] if "site_name" in site else None),
                                "site_type": (site["site_type"] if "site_type" in site else None),
                                area_col: area,
                                "algo": algo_name,
                                "ring_name": cluster,
                                "nearest_node": node,
                                "geometry": site["geometry"],
                            }
                        )
            else:
                print(f"⚠️ No valid path found for {algo_name} in cluster {cluster}. Skipping...")
                continue
        except Exception as e:
            print(f"❌ Error in {algo_name} for cluster {cluster}: {e}")
            continue

        if not paths_data or not points_data:
            print(f"⚠️ Empty data for {algo_name} in cluster {cluster}. Skipping...")
            continue

        temp_paths_gdf = gpd.GeoDataFrame(paths_data, geometry="geometry", crs="EPSG:3857")
        temp_points_gdf = gpd.GeoDataFrame(points_data, geometry="geometry", crs="EPSG:3857")
        total_length = temp_paths_gdf["length"].sum()

        # SCORING
        duplicate_points = duplicate_scores(temp_paths_gdf)
        print(f"ℹ️ Summary {algo_name:20} | Cluster {cluster:2}")
        print(f"Total Length        : {total_length:8.2f} meters")
        print(f"Duplicate Points    : {duplicate_points}")

        total_cost = total_length + duplicate_points
        print(f"💰 {algo_name} | Total Cost: {total_cost:,.2f}")

        if best_algo is None or total_cost < best_algo["cost"]:
            best_algo = {
                "name": algo_name,
                "paths_gdf": temp_paths_gdf,
                "points_gdf": temp_points_gdf,
                "cost": total_cost,
                "length": total_length,
                "duplicate_points": duplicate_points,
            }

    if best_algo is None:
        print(f"⚠️ No valid paths found for cluster {cluster}. Skipping...")
        return None, None

    final_paths = best_algo["paths_gdf"].reset_index(drop=True)
    final_points = best_algo["points_gdf"].reset_index(drop=True)
    print(
        f"🏆 Best Algorithm | {cluster:2} | {best_algo['name']} | Cost {best_algo['cost']:2f}"
    )
    return final_paths, final_points


def multi_fohub(
    cluster_site,
    cluster,
    roads,
    nodes,
    G,
    weight="weight",
    area="area",
    area_col="region",
):
    # DISTANCE MAPPING
    total_sites = len(cluster_site)
    cluster_site = cluster_site.drop_duplicates(subset="nearest_node")
    cluster_site = cluster_site.to_crs(epsg=3857)
    cluster_site = cluster_site.reset_index(drop=True)

    sitelist = cluster_site[~cluster_site["site_type"].str.lower().str.contains("hub")].copy()
    fo_hub = cluster_site[cluster_site["site_type"].str.lower().str.contains("hub")].copy()
    start_hub = fo_hub["site_id"].iloc[0] if not fo_hub.empty else None
    end_hub = fo_hub["site_id"].iloc[-1] if not fo_hub.empty else None

    mapping = distance_mapping(cluster_site, G, weight=weight)
    print(f"ℹ️ Distance mapping complete | Total pairs: {len(mapping)}")

    # OPTIMAL ROUTE
    if total_sites <= 8:
        print(f"🧩 Multi Hub | Brute Force")
        best_route, best_length = optimal_routes(sitelist, mapping, start_hub, end_hub)
    else:
        print(f"🧩 Multi Hub | Christofides")
        best_route, best_length = christofides_tsp(sitelist, mapping, start_hub, end_hub)

    if best_route is None:
        print(f"❌ No valid route found for cluster {cluster}.")
        return None, None

    print(f"✅ Optimal route cluster {cluster:20}:")
    print(f"    - Route {(' > '.join(str(x) for x in best_route))}")
    print(f"    - Total Cost: {best_length / 1000:,.2f} km")

    # BUILD PATHS
    paths_data = []
    points_data = []
    for idx, site in enumerate(best_route[:-1]):
        start_site = cluster_site[cluster_site["site_id"] == site].iloc[0]
        end_site = cluster_site[cluster_site["site_id"] == best_route[idx + 1]].iloc[0]
        start_node = cluster_site.loc[cluster_site["site_id"] == site, "nearest_node"].values[0]
        end_node = cluster_site.loc[cluster_site["site_id"] == best_route[idx + 1], "nearest_node"].values[0]
        try:
            path, path_geom, path_length = route_path(start_node, end_node, G, roads, merged=True)
            path_geom, path_length = dropwire_connection(path_geom, start_site, end_site, nodes, start_node, end_node)
            print(f"Cluster {cluster:20} | From {start_site['site_id']:>15} to {end_site['site_id']:>15} | Length: {path_length:8.2f} meters")
            
            # STORE PATHS
            paths_data.append(
                {
                    "near_end": start_site["site_id"],
                    "far_end": end_site["site_id"],
                    "algo": "Multi Hub",
                    area_col: area,
                    "ring_name": cluster,
                    "route_type": "merged",
                    "length": round(path_length, 2),
                    "geometry": path_geom,
                }
            )

        except nx.NetworkXNoPath:
            print(
                f"No path found for Cluster {cluster} | Start: {site} | End: {best_route[idx + 1]}"
            )
            continue
        except Exception as e:
            print(f"❌ Error processing path for Cluster {cluster}: {e}")
            continue

    # STORE POINTS
    for site in best_route:
        site_data = cluster_site[cluster_site["site_id"] == site].iloc[0]
        if site_data["site_id"] not in [point["site_id"] for point in points_data]:
            points_data.append(
                {
                    "site_id": site_data["site_id"],
                    "site_name": (site_data["site_name"] if "site_name" in site_data else None),
                    "site_type": (site_data["site_type"] if "site_type" in site_data else None),
                    "algo": "Multi Hub",
                    area_col: area,
                    "ring_name": cluster,
                    "nearest_node": site_data["nearest_node"],
                    "geometry": site_data["geometry"],
                }
            )
    if not paths_data or not points_data:
        print(f"⚠️ Empty data for cluster {cluster}. Skipping...")
        return None, None

    final_paths = gpd.GeoDataFrame(paths_data, geometry="geometry", crs="EPSG:3857")
    final_points = gpd.GeoDataFrame(points_data, geometry="geometry", crs="EPSG:3857")
    total_length = final_paths["length"].sum()
    print(f"ℹ️ Total Length for cluster {cluster}: {total_length:8,.2f} meters")

    return final_paths, final_points


# ================
# RING MAIN METHOD
# ================
def cluster_ring(cluster_args):
    method = cluster_args.get("method", "unsupervised")
    cluster_site = cluster_args["cluster_site"]
    cluster = cluster_args["ring_name"]
    area = cluster_args.get("area", "area")
    area_col = cluster_args.get("area_col", "region")
    ref_fo = cluster_args.get("ref_fo", None)
    cable_cost = cluster_args.get("cable_cost", 35000)
    path_type = cluster_args.get("path_type", "merged")
    export_dir = cluster_args.get("export_dir", None)

    final_paths = []
    final_points = []

    print(f"🛟 Processing Cluster {cluster} | Area: {area} | Total Site: {len(cluster_site):,}")
    cluster_site = cluster_site.copy()
    cluster_site = cluster_site.to_crs(epsg=3857)
    cluster_site = cluster_site.reset_index(drop=True)

    if cluster_site.empty:
        print(f"⚠️ No data found for cluster {cluster}. Skipping...")
        return None, None

    route_path = os.path.join(export_dir, f"Paths_{area}_{cluster}.parquet")
    point_path = os.path.join(export_dir, f"Points_{area}_{cluster}.parquet")

    if os.path.exists(point_path):
        paths = gpd.read_parquet(route_path)
        points = gpd.read_parquet(point_path)
        print(f"ℹ️ Paths for cluster {cluster} already exist. Skipping...")
        return paths, points

    # IDENTIFY HEXAGON AND ROUTE
    hex_list = identify_hexagon(cluster_site, type="convex")
    roads = retrieve_roads(hex_list, type="roads").to_crs(epsg=3857)
    nodes = retrieve_roads(hex_list, type="nodes").to_crs(epsg=3857)
    G = build_graph(roads, graph_type="full_weighted", ref_fo=ref_fo, cable_cost=cable_cost, avoid_railway=False)

    node_sindex = nodes.sindex
    cluster_site["nearest_node"] = cluster_site.geometry.apply(lambda geom: nodes.at[node_sindex.nearest(geom)[1][0], "node_id"])

    # CYCLE
    fo_hub = cluster_site[cluster_site["site_type"].str.lower().str.contains("hub")].copy()
    sitelist = cluster_site[~cluster_site["site_type"].str.lower().str.contains("hub")].copy()
    total_sites = len(cluster_site)
    num_hub = len(fo_hub)
    num_sitelist = len(sitelist)

    if "flag" in cluster_site.columns:
        start_hub = (
            fo_hub[fo_hub["flag"] == "start"]["site_id"].iloc[0]
            if not fo_hub[fo_hub["flag"] == "start"].empty
            else (fo_hub["site_id"].iloc[0] if not fo_hub.empty else None)
        )
        end_hub = (
            fo_hub[fo_hub["flag"] == "end"]["site_id"].iloc[0]
            if not fo_hub[fo_hub["flag"] == "end"].empty
            else (fo_hub["site_id"].iloc[-1] if not fo_hub.empty else None)
        )
    else:
        start_hub = fo_hub["site_id"].iloc[0] if not fo_hub.empty else None
        end_hub = fo_hub["site_id"].iloc[-1] if not fo_hub.empty else None

    print(f"ℹ️ Cluster       :{cluster}")
    print(f"ℹ️ Total Sites   :{total_sites}")
    print(f"ℹ️ FO Hubs       :{num_hub:<2}| Start Hub: {start_hub} > End Hub: {end_hub}")
    print(f"ℹ️ Sitelist      :{num_sitelist:<2}| Total Sites: {num_sitelist}\n")

    # SINGLE FO HUB
    match num_hub:
        case 1:
            final_paths, final_points = single_fohub(
                cluster_site,
                cluster,
                fo_hub,
                roads,
                nodes,
                G,
                area=area,
                area_col=area_col,
            )

            if final_paths is None or final_points is None:
                print(f"❌ No valid paths found for cluster {cluster}.")
                return None, None
        case 2:
            # MULTIPLE FO HUBS
            final_paths, final_points = multi_fohub(
                cluster_site,
                cluster,
                roads,
                nodes,
                G,
                area=area,
                area_col=area_col,
            )
        case _:
            print(
                f"⚠️ No FO Hub found in cluster {cluster}. Using all nodes for pathfinding."
            )

    # CHECK SPOF
    if final_paths is None or final_points is None:
        print(
            f"⚠️ No valid paths or points found for cluster {cluster}. Skipping SPOF detection."
        )
        return None, None

    print(f"ℹ️ Running SPOF Detection for cluster {cluster}...")
    final_paths = spof_detection(
        final_paths,
        final_points,
        G,
        roads,
        nodes,
        threshold_spof=3000,
        threshold_alt=250,
    )

    if path_type == "classified":
        print(f"ℹ️ Paths classified by type.")
        final_paths = fiber_utilization(final_paths, roads=roads, nodes=nodes, overlap=True)

    # Debug: Show path structure
    print(f"\n📊 Summary for Cluster {cluster} | Area: {area}")
    print(f"Total Paths         | {area:15} | {cluster:2}: {len(final_paths):,}")
    print(f"Total Points        | {area:15} | {cluster:2}: {len(final_points):,}")
    print(f"FO Hub Nodes        | {area:15} | {cluster:2}: {len(fo_hub):,}")
    print()

    # CONNECTED ROUTES FO HUB
    print(f"📍 Running Connected Routes")
    connected_routes = []
    connection_list = []
    visited_sites = set()

    if len(fo_hub) == 1 and start_hub == end_hub:
        print(f"ℹ️ Single FO Hub: {start_hub}.")
        current_site = start_hub
        max_iterations = len(final_paths) + 1
        iteration_count = 0

        while iteration_count < max_iterations:
            next_row = final_paths[final_paths["near_end"] == current_site]
            if next_row.empty:
                print(f"⚠️ No outgoing path found from {current_site}. Breaking loop.")
                break

            for _, row in next_row.iterrows():
                connected_routes.append(row)

            if current_site not in connection_list:
                connection_list.append(current_site)

            next_site = next_row["far_end"].values[0]

            if next_site == start_hub and len(connected_routes) > 1:
                connection_list.append(next_site)
                print(f"✅ Ring completed back to hub {start_hub}")
                break

            # Check for infinite loop
            if next_site in visited_sites and next_site != start_hub:
                print(f"⚠️ Detected loop at site {next_site}. Breaking to prevent infinite loop.")
                break

            visited_sites.add(current_site)
            current_site = next_site
            iteration_count += 1
        print(
            f"ℹ️ Connected {len(connected_routes)} routes in {iteration_count} iterations"
        )
    else:
        print(f"ℹ️ Multiple FO Hubs: {start_hub} and {end_hub}.")
        current_site = start_hub
        max_iterations = len(final_paths) + 1
        iteration_count = 0

        while iteration_count < max_iterations:
            next_row = final_paths[final_paths["near_end"] == current_site]
            if next_row.empty:
                break

            for _, row in next_row.iterrows():
                connected_routes.append(row)

            if current_site not in connection_list:
                connection_list.append(current_site)

            next_site = next_row["far_end"].values[0]

            # Check end hub
            if next_site == end_hub:
                connection_list.append(end_hub)
                print(f"✅ Path completed from {start_hub} to {end_hub}")
                break

            # Check infinite loop
            if next_site in visited_sites and next_site != end_hub:
                print(
                    f"⚠️ Detected loop at site {next_site}. Breaking to prevent infinite loop."
                )
                visited_sites.add(current_site)
                break

            visited_sites.add(current_site)
            current_site = next_site
            iteration_count += 1
        print(
            f"ℹ️ Connected {len(connected_routes)} routes in {iteration_count} iterations"
        )
    if not connected_routes:
        print(
            f"⚠️ No connected routes found for FO Hub in cluster {cluster}. Skipping..."
        )
        return final_paths, final_points

    connected_routes_df = pd.DataFrame(connected_routes)
    connected_routes_gdf = gpd.GeoDataFrame(
        connected_routes_df, geometry="geometry", crs="EPSG:3857"
    )
    connected_routes_gdf = connected_routes_gdf.reset_index(drop=True)

    # SORT POINTS BY CONNECTION LIST
    connected_sites = []
    for site_id in connection_list:
        site_row = final_points[final_points["site_id"] == site_id]
        if not site_row.empty:
            connected_sites.append(site_row.iloc[0])

    connected_sites_gdf = gpd.GeoDataFrame(
        connected_sites, geometry="geometry", crs="EPSG:3857"
    )
    connected_sites_gdf = connected_sites_gdf.reset_index(drop=True)
    connected_sites_gdf = connected_sites_gdf.drop_duplicates(
        subset=["site_id", "geometry"]
    )

    if connected_sites_gdf.empty:
        return None, None

    total_length = connected_routes_gdf["length"].sum()
    print(f"Total Length | {area} | {cluster}: {total_length / 1000:,.2f} km")

    connected_routes_gdf = connected_routes_gdf.to_crs(epsg=4326)
    connected_sites_gdf = connected_sites_gdf.to_crs(epsg=4326)

    connected_routes_gdf.to_parquet(
        os.path.join(export_dir, f"Paths_{area}_{cluster}.parquet")
    )
    connected_sites_gdf.to_parquet(
        os.path.join(export_dir, f"Points_{area}_{cluster}.parquet")
    )
    print(f"🌏 Cluster {cluster} parquet saved. \n")
    return connected_routes_gdf, connected_sites_gdf

# ===============
# RING MAIN METHOD
# ===============
def parallel_ring(
    clustered_sites,
    area,
    method="unsupervised",
    area_col="region",
    cluster_col="ring_name",
    export_dir=None,
    ref_fo=None,
    cable_cost=35000,
    path_type="merged",
    task_celery=False,
):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print(f"🧩 Parallel Clustering {area}")
    if clustered_sites.empty:
        return None, None

    if export_dir is None:
        raise ValueError(
            "Export directory is not provided. Please specify a valid export directory."
        )

    os.makedirs(export_dir, exist_ok=True)
    clusters = sorted(clustered_sites[cluster_col].unique().tolist())
    print(f"ℹ️ Total Clusters: {len(clusters):,} | Area: {area}")

    cluster_args = []
    for cluster in clusters:
        if cluster == -1:
            continue

        cluster_sites = clustered_sites[clustered_sites[cluster_col] == cluster]
        if cluster_sites.empty:
            continue

        cluster_args.append(
            {
                "cluster_site": cluster_sites,
                "ring_name": cluster,
                "area": area,
                "area_col": area_col,
                "ref_fo": ref_fo,
                "export_dir": export_dir,
                "cable_cost": cable_cost,
                "path_type": path_type,
                "method": method,
            }
        )

    if not cluster_args:
        print("⚠️ No valid clusters found for parallel processing. Skipping...")
        return None, None

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(cluster_ring, args): args["ring_name"]
            for args in cluster_args
        }
        final_paths = []
        final_points = []
        total_futures = len(futures)

        if task_celery and total_futures > 1:
            task_celery.update_state(
                state="PROGRESS",
                meta={"status": f"Processing {total_futures} clusters in parallel"},
            )

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing Clusters"
        ):
            cluster_id = futures[future]
            try:
                paths, points = future.result()
                if paths is not None:
                    paths = paths.to_crs(epsg=4326)
                    final_paths.append(paths)
                if points is not None:
                    points = points.to_crs(epsg=4326)
                    final_points.append(points)
                
                if task_celery:
                    task_celery.update_state(
                        state="PROGRESS",
                        meta={
                            "status": f"Completed {len(final_paths)}/{total_futures} clusters",
                        },
                    )
            except Exception as e:
                print(f"❌ Error processing cluster {cluster_id}: {e}")
                if task_celery:
                    task_celery.update_state(
                        state="PROGRESS",
                        meta={
                            "status": f"Error in cluster {cluster_id}: {e}. Completed {len(final_paths)}/{total_futures} clusters",
                        },
                    )
                continue

        if final_paths:
            final_paths = pd.concat(final_paths, ignore_index=True)
            final_paths = final_paths.drop_duplicates(subset=["geometry"])
            print(f"ℹ️ Total Paths after parallel processing: {len(final_paths):,}")
        if final_points:
            final_points = pd.concat(final_points, ignore_index=True)
            final_points = final_points.drop_duplicates(subset=["geometry"])
            print(f"ℹ️ Total Points after parallel processing: {len(final_points):,}")
    return final_paths, final_points


# ===============
# SUPERVISED
# ===============
def supervised_ring(
    sites_input,
    area,
    area_col="region",
    cluster_col="ring_name",
    export_dir=None,
    fo_expand: gpd.GeoDataFrame = None,
    **kwargs,
):

    print(f"🧩 Ring Network Supervised | {area}")
    cable_cost = kwargs.get("cable_cost", 35000)
    path_type = kwargs.get("path_type", "merged")
    task_celery = kwargs.get("task_celery", False)

    os.makedirs(export_dir, exist_ok=True)
    try:
        if isinstance(sites_input, str):
            sites_input = gpd.read_parquet(sites_input)
            sites_input[area_col] = sites_input[area_col].str.strip().str.upper()
        else:
            sites_input = sites_input.copy()
            sites_input = sites_input.reset_index(drop=True)

        if sites_input.crs is None:
            sites_input.set_crs(epsg=4326, inplace=True)

        sites_input = sites_input.to_crs(epsg=3857)

        # SITE AREA
        site_area = sites_input[sites_input[area_col] == area].copy()
        unique_clusters = site_area[area_col].dropna().unique().tolist()
        if not unique_clusters:
            print(f"No clusters found for area {area}. Skipping...")
            return None

        print(f"✨ Processing {area_col}: {area}")
        print(f"ℹ️ Total Sites       | {area_col} {area}: {len(sites_input):,} data")

        ref_fo = None
        if fo_expand is not None:
            print(f"ℹ️ Fiber Expand defined")
            hex_list = identify_hexagon(sites_input, type="convex")
            print(f"ℹ️ Total Hexagons    | {area_col} {area}: {len(hex_list):,} hexagons")
            roads = retrieve_roads(hex_list, type="roads")
            nodes = retrieve_roads(hex_list, type="nodes")
            print(f"ℹ️ Total Roads       | {area_col} {area}: {len(roads):,} roads")
            print(f"ℹ️ Total Nodes       | {area_col} {area}: {len(nodes):,} nodes")

            if roads.empty or nodes.empty:
                print(f"⚠️ No roads or nodes found for area {area}. Skipping...")
                return None

            roads = roads.to_crs(epsg=3857)
            nodes = nodes.to_crs(epsg=3857)

            # Ref FO
            print("🧩 Preparing FO Ref Nodes...")
            ref_fo = set(nodes[nodes["ref_fo"] == 1]["node_id"])
            fo_expand = fo_expand.to_crs(3857)
            fo_expand["geometry"] = fo_expand.geometry.buffer(20)
            nodes_within = gpd.sjoin(
                nodes, fo_expand, how="inner", predicate="intersects"
            )
            nodes_within = set(nodes_within["node_id"])

            if isinstance(nodes_within, (list, tuple, set)):
                ref_fo.update(nodes_within)
                print(f"ℹ️ Added {len(nodes_within)} expanded FO nodes")
            else:
                print("⚠️ expand_fo should be a list, tuple, or set")
            print(f"ℹ️ Reference FO retrieved.")

        # PROCESS RING
        if task_celery:
            task_celery.update_state(state="PROGRESS", meta={"status": "Processing ring data"})

        paths, points = parallel_ring(
            clustered_sites=site_area,
            area=area,
            method="supervised",
            area_col=area_col,
            cluster_col=cluster_col,
            export_dir=export_dir,
            ref_fo=ref_fo,
            cable_cost=cable_cost,
            path_type=path_type,
            task_celery=task_celery,
        )
        return paths, points
    except Exception as e:
        print(f"❌❌ Error processing area {area}: {e} \n")
        return None

# ===============
# DATA EXPORT
# ===============
def merge_kmz(main_kml, ring_data, point_data, folder):
    ring_data = ring_data.copy()
    cluster_points = point_data.copy()
    ring_data["start"] = ring_data["near_end"].apply(lambda x: cluster_points[cluster_points["site_id"] == x]["geometry"].values[0])
    ring_data["end"] = ring_data["far_end"].apply(lambda x: cluster_points[cluster_points["site_id"] == x]["geometry"].values[0])

    ring_data = ring_data.reset_index(drop=True)
    filename = folder.replace("/", "-")
    if 'long' not in cluster_points.columns or 'lat' not in cluster_points.columns:
        cluster_points['long'] = cluster_points.geometry.to_crs(epsg=4326).x
        cluster_points['lat'] = cluster_points.geometry.to_crs(epsg=4326).y
    if 'vendor' not in cluster_points.columns:
        cluster_points['vendor'] = 'TBG'
    if 'program' not in cluster_points.columns:
        cluster_points['program'] = 'N/A'

    used_columns = {
        "ring_name": "Ring ID",
        "site_id": "Site ID",
        "site_name": "Site Name" if "site_name" in cluster_points.columns else "N/A",
        "long": "Long",
        "lat": "Lat",
        "region": "Region",
        "vendor": "Vendor" if "vendor" in cluster_points.columns else "N/A",
        "program": "Program" if "program" in cluster_points.columns else "N/A",
        "geometry": "geometry",
        }
    available_col = [col for col in used_columns.keys() if col in cluster_points.columns]

    # TOPOLOGY
    ring_topology = create_topology(point_data)
    ring_topology = ring_topology.to_crs(epsg=4326)
    ring_topology["connection"] = "Connection"

    # ROUTE
    route_columns = [ "near_end", "far_end", "geometry", "ring_name", "route_type", "length"]
    if "existing_cable_length" in ring_data.columns:
        route_columns.append("existing_cable_length")
    if "new_cable_length" in ring_data.columns:
        route_columns.append("new_cable_length")
    ring_route = ring_data[route_columns].copy()
    ring_route["route_name"] = ring_route["near_end"] + "-" + ring_route["far_end"]

    # SITE LIST & HUB
    ring_sites = cluster_points[~cluster_points["site_type"].str.lower().str.contains("hub")].copy()
    ring_hub = cluster_points[cluster_points["site_type"].str.lower().str.contains("hub")].copy()

    ring_sites = ring_sites[available_col].rename(columns=used_columns)
    ring_hub = ring_hub[available_col].rename(columns=used_columns)
    kml_updated = export_kml( ring_topology, main_kml, filename, subfolder=folder, name_col="connection", color="#FF00FF", size=2, popup=False)
    kml_updated = export_kml( ring_hub, kml_updated, filename, subfolder=f"{folder}/FO Hub", name_col="Site ID", icon="http://maps.google.com/mapfiles/kml/paddle/A.png", size=0.8, popup=True)
    kml_updated = export_kml(ring_sites, kml_updated, filename, subfolder=f"{folder}/Site List", name_col="Site ID", color="#FFFF00", size=0.8, popup=True)
    kml_updated = export_kml( ring_route, kml_updated, filename, subfolder=f"{folder}/Route", name_col="route_name", color="#0000FF", size=3, popup=False)

    return kml_updated


def merge_gpkg(gpkg_path, ring_data, point_data, cluster):
    ring_data = ring_data.copy()
    cluster_points = point_data.copy()

    ring_data["start"] = ring_data["near_end"].apply(
        lambda x: cluster_points[cluster_points["site_id"] == x]["geometry"].values[0]
    )
    ring_data["end"] = ring_data["far_end"].apply(
        lambda x: cluster_points[cluster_points["site_id"] == x]["geometry"].values[0]
    )

    ring_data = ring_data.reset_index(drop=True)

    # TOPOLOGY
    topology = create_topology(point_data)
    topology.to_file(gpkg_path, layer=f"{cluster}_Topology", driver="GPKG")

    # ROUTE
    ring_route = ring_data[["near_end", "far_end", "ring_name", "route_type", "length", "geometry"]].copy()
    ring_route["route_name"] = ring_route["near_end"] + "-" + ring_route["far_end"]
    ring_route.to_file(gpkg_path, layer=f"{cluster}_Route", driver="GPKG")

    # SITE LIST
    ring_sites = cluster_points[["site_id", "site_name", "site_type", "geometry"]].copy()
    ring_sites = ring_sites[~ring_sites["site_type"].str.lower().str.contains("hub")].copy()
    if ring_sites.empty:
        print("⚠️ No sites found. Skipping site export.")
        return gpkg_path
    ring_sites.to_file(gpkg_path, layer=f"{cluster}_Site List", driver="GPKG")


    # FO HUB
    ring_hub = cluster_points[cluster_points["site_type"].str.lower().str.contains("hub")].copy()
    if ring_hub.empty:
        print("⚠️ No FO Hub found. Skipping FO Hub export.")
        return gpkg_path

    ring_hub.to_file(gpkg_path, layer=f"{cluster}_FO Hub", driver="GPKG")
    return gpkg_path


def ring_compile(parquet_dir):
    import os

    list_files = os.listdir(parquet_dir)
    point_files = [
        f for f in list_files if f.endswith(".parquet") and f.startswith("Points")
    ]
    path_files = [
        f for f in list_files if f.endswith(".parquet") and f.startswith("Paths")
    ]

    points_data = []
    for file in tqdm(point_files, desc="Loading Point Files"):
        file_path = os.path.join(parquet_dir, file)
        gdf = gpd.read_parquet(file_path)
        if not gdf.empty:
            points_data.append(gdf)

    paths_data = []
    for file in tqdm(path_files, desc="Loading Path Files"):
        file_path = os.path.join(parquet_dir, file)
        gdf = gpd.read_parquet(file_path)
        if not gdf.empty:
            paths_data.append(gdf)

    print(f"ℹ️ Total Point Files : {len(point_files):,}")
    print(f"ℹ️ Total Path Files  : {len(path_files):,}")

    points_data = (
        pd.concat(points_data, ignore_index=True) if points_data else pd.DataFrame()
    )
    paths_data = (
        pd.concat(paths_data, ignore_index=True) if paths_data else pd.DataFrame()
    )

    print(f"ℹ️ Total Points DataFrames : {len(points_data):,}")
    print(f"ℹ️ Total Paths DataFrames  : {len(paths_data):,}")

    # Check Cluster Data
    result_cluster = points_data["ring_name"].unique().tolist()
    print(f"ℹ️ Total Cluster Data: {len(result_cluster):,}")

    # Save Points and Paths
    points_data.to_crs(epsg=4326).to_parquet(
        os.path.join(parquet_dir, "All_Points_Cluster.parquet"), index=False
    )
    paths_data.to_crs(epsg=4326).to_parquet(
        os.path.join(parquet_dir, "All_Paths_Cluster.parquet"), index=False
    )
    points_data.to_excel(
        os.path.join(parquet_dir, "All_Points_Cluster.xlsx"), index=False
    )

    print(f"✅ Compiled Parquet files saved to {parquet_dir}")
    # REMOVE INDIVIDUAL FILES
    for file in point_files + path_files:
        try:
            os.remove(os.path.join(parquet_dir, file))
        except Exception as e:
            print(f"❌ Error removing file {file}: {e}")
    return points_data, paths_data


def nearfar_connection(points_gdf: gpd.GeoDataFrame, paths_gdf: gpd.GeoDataFrame):
    miss_point = [c for c in ["site_id", "ring_name"] if c not in points_gdf.columns]
    miss_paths = [c for c in ["near_end", "far_end", "ring_name"] if c not in paths_gdf.columns]

    if miss_point:
        raise ValueError(f"Missing column in points: {miss_point}")
    if miss_paths:
        raise ValueError(f"Missing column in paths: {miss_paths}")

    print("🧩 Connecting NE and FE.")
    for ring, g in points_gdf.groupby("ring_name"):
        p = paths_gdf[paths_gdf["ring_name"] == ring]

        g_ids = g["site_id"].astype(str)
        ne_to_fe = p.drop_duplicates(subset="near_end").set_index("near_end")["far_end"]
        fe_to_ne = p.drop_duplicates(subset="far_end").set_index("far_end")["near_end"]

        points_gdf.loc[g.index, "far_end"] = g_ids.map(ne_to_fe)
        points_gdf.loc[g.index, "near_end"] = g_ids.map(fe_to_ne)

    points_gdf["far_end"] = points_gdf["far_end"].fillna("N/A")
    points_gdf["near_end"] = points_gdf["near_end"].fillna("N/A")
    return points_gdf


# ================
# MAIN FIBERIZATION FUNCTION
# ================
def main_supervised(
    site_data: gpd.GeoDataFrame,
    export_loc: str="./exports",
    method: str = "supervised",
    area_col: str = "region",
    cluster_col="ring_name",
    fo_expand: gpd.GeoDataFrame = None,
    **kwargs,
):
    """
    Main function to process unsupervised ring network clustering.
    """
    # DEFAULT PARAMETERS
    cable_cost = kwargs.get("cable_cost", 35000)
    path_type = kwargs.get("path_type", "merged")
    version = kwargs.get("version", 1)
    vendor = kwargs.get("vendor", "TBG")
    program = kwargs.get("program", "N/A")
    task_celery = kwargs.get("task_celery", None)

    if "site_id" in site_data.columns:
        site_data["site_id"] = site_data["site_id"].astype(str)
    site_data = sanitize_header(site_data)

    # VALIDATION
    site_data = input_newring(site_data, method=method)

    # EXPORT DIRECTORY
    date_today = datetime.now().strftime("%Y%m%d")
    week = detect_week(date_today)
    export_loc = f"{export_loc}/New Ring_{method.title()}_{date_today}_W{str(week)}_{str(version)}"
    gpkg_dir = os.path.join(export_loc, "GPKG")
    kmz_dir = os.path.join(export_loc, "KMZ")
    parquet_dir = os.path.join(export_loc, "PARQUET")

    if not os.path.exists(export_loc):
        os.makedirs(export_loc)
    if not os.path.exists(gpkg_dir):
        os.makedirs(gpkg_dir)
    if not os.path.exists(kmz_dir):
        os.makedirs(kmz_dir)
    if not os.path.exists(parquet_dir):
        os.makedirs(parquet_dir)

    # MAIN PROCESS RING NETWORK
    if "index_right" in site_data.columns:
        site_data = site_data.drop(columns=["index_right"])
    area_list = sorted(site_data[area_col].unique().tolist())

    output_paths = os.path.join(export_loc, "PARQUET", "All_Paths_Cluster.parquet")
    output_point = os.path.join(export_loc, "PARQUET", "All_Points_Cluster.parquet")
    if os.path.exists(output_paths) and os.path.exists(output_point):
        print(f"ℹ️ Output file already exists. Loading existing data...")
        all_paths = gpd.read_parquet(output_paths)
        all_points = gpd.read_parquet(output_point)
        print(f"✅ Loaded existing data successfully.")
    else:
        print(f"🔥 Starting Fiberization Process | Method: {method.title()}")
        all_paths = []
        all_points = []
        for area in tqdm(area_list, desc=f"Processing {area_col}"):
            site_area = site_data[site_data[area_col] == area].copy()
            if site_area.empty:
                print(f"⚠️ No data found for {area_col} {area}. Skipping...")
                continue

            paths, points = supervised_ring(
                site_area,
                area=area,
                area_col=area_col,
                cluster_col=cluster_col,
                export_dir=parquet_dir,
                fo_expand=fo_expand,
                cable_cost=cable_cost,
                path_type=path_type,
                task_celery=task_celery
            )

            if paths is None or points is None:
                print(f"⚠️ No paths or points found for {area_col} {area}. Skipping...")
                continue
            else:
                all_paths.append(paths)
                all_points.append(points)

        # all_paths = pd.concat(all_paths, ignore_index=True) if all_paths else pd.DataFrame()
        # all_points = pd.concat(all_points, ignore_index=True) if all_points else pd.DataFrame()

        print("🔗 Merging Parquet files...")
        all_points, all_paths = ring_compile(parquet_dir)
        all_points = all_points.reset_index(drop=True)
        all_paths = all_paths.reset_index(drop=True)

    # ADD EXTRA INFO
    if all_points.empty or all_paths.empty:
        print("❌ No data available for KMZ/GPKG export. Exiting...")
        return

    if "vendor" not in all_points.columns:
        all_points["vendor"] = vendor
    if "program" not in all_points.columns:
        all_points["program"] = program
    if "long" not in all_points.columns or "lat" not in all_points.columns:
        all_points["long"] = round(all_points.geometry.to_crs(epsg=4326).x, 6)
        all_points["lat"] = round(all_points.geometry.to_crs(epsg=4326).y, 6)

    # EXPORT EXCEL
    with pd.ExcelWriter(os.path.join(export_loc, f"Ring_Network_{method.title()}.xlsx"),engine="openpyxl") as writer:
        # NEAR END FAR END CONNECTION
        exported_points = nearfar_connection(all_points, all_paths)
        exported_paths = all_paths.copy()
        dropped_cols = ["geometry", "nearest_node"]
        for col in dropped_cols:
            if col in exported_points.columns:
                exported_points = exported_points.drop(columns=[col])
            if col in exported_paths.columns:
                exported_paths = exported_paths.drop(columns=[col])

        excel_styler(exported_points).to_excel(writer, sheet_name="Points", index=False)
        excel_styler(all_paths).to_excel(writer, sheet_name="Paths", index=False)

    # MERGE KMZ FILES
    print("🌏 Merging KMZ and GPKG files...")
    site_data = all_points.copy()
    if site_data.empty:
        print("❌ No site data available for KMZ/GPKG export. Exiting...")
        return

    site_data = site_data[site_data[area_col].notna()]

    kmz_main = simplekml.Kml()
    gpkg_main = os.path.join(gpkg_dir, f"Ring_Network_{method.title()}.gpkg")
    for area in area_list:
        site_area = site_data[site_data[area_col] == area].copy()
        if site_area.empty:
            print(f"⚠️ No data found for {area_col} {area}. Skipping...")
            continue
        try:
            cluster_list = sorted(site_area["ring_name"].unique().tolist())
            print(f"{area_col} {area} | Total Cluster {len(cluster_list):,}")
            for cluster in tqdm(cluster_list, desc=f"Merge KMZ | {area_col} {area}"):
                try:
                    ring_data = all_paths[all_paths["ring_name"] == cluster].copy()
                    point_data = all_points[all_points["ring_name"] == cluster].copy()

                    if ring_data.empty or point_data.empty:
                        print(f"Skipping empty cluster {cluster} in {area_col} {area}.")
                        continue

                    folder = f"{area}/{cluster}"
                    kmz_main = merge_kmz(kmz_main, ring_data, point_data, folder=folder)
                    gpkg_main = merge_gpkg(gpkg_main, ring_data, point_data, cluster=cluster)
                    print(f"✅ Merge {cluster} success.")
                except Exception as e:
                    print(
                        f"Error merging KMZ or GPKG for cluster {cluster} in {area_col} {area}: {e}"
                    )
                    continue
        except Exception as e:
            print(f"❌ Error processing {area_col} {area}: {e}")
            continue
    kmz_path = os.path.join(kmz_dir, f"Ring_Network_{method.title()}.kmz")
    sanitize_kml(kmz_main)
    kmz_main.savekmz(kmz_path)
    print(f"🌏 KMZ for {area_col} {area} saved.")
    print(f"🔥 Fiberization process completed. All files saved to \n{export_loc}.")

if __name__ == "__main__":
    
    excel_file = r"D:\JACOBS\TASK\OKTOBER\Week 2\SURGE BACKHAUL UNSUPERVISED\5800_Protect Sitelist\Surge_5800_3000m_Clustered_Data.xlsx"
    export_dir = r"D:\JACOBS\TASK\OKTOBER\Week 2\SURGE BACKHAUL UNSUPERVISED\5800_Protect Sitelist"
    area_col = 'region'
    cluster_col = 'ring_name'
    path_type =  'merged'

    # LOAD DATA
    site_data = pd.read_excel(excel_file)
    site_data = sanitize_header(site_data)
    site_data = input_newring(site_data, method='supervised')
    fo_expand = None

    date_today = datetime.now().strftime("%Y%m%d")
    export_loc = f"{export_dir}/{date_today}"
    os.makedirs(export_loc, exist_ok=True)

    result = main_supervised(
        site_data=site_data,
        fo_expand=fo_expand,
        export_loc=export_loc,
        area_col=area_col,
        cluster_col=cluster_col,
    )

    # ZIPFILE
    zip_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_Supervised_Task.zip"
    zip_filepath = os.path.join(export_loc, zip_filename)
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(export_loc):
            for file in files:
                if file != zip_filename:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_loc)
                    zipf.write(file_path, arcname)
    print(f"📦 Result files zipped.")