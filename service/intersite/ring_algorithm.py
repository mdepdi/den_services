# ======================================================
# MODULE : SUPERVISED RING NETWORK
# AUTHOR : Yakub Hariana
# ORG    : Tower Bersama Group
# DESC   : Routing automation (Supervised mode)
# VERSION: 2.0
# ======================================================

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
from itertools import permutations

from modules.data import fiber_utilization
from modules.h3_route import identify_hexagon, retrieve_roads, build_graph
from modules.utils import auto_group, spof_detection, create_topology, route_path, dropwire_connection
from modules.table import sanitize_header, detect_week, excel_styler
from modules.validation import input_newring
from modules.kml import export_kml, sanitize_kml
from boq_algorithm import main_boq



# ------------------------------------------------------
# 1) CORE DISTANCE & ROUTING
# ------------------------------------------------------
def map_distance(cluster_sites, G=None, weight="weight"):
    """Compute pairwise shortest-path distances for all site pairs."""
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
                futures[executor.submit(
                    nx.shortest_path_length, G, source=node_i, target=node_j, weight=weight
                )] = (id_i, id_j)

    for future in tqdm(as_completed(futures.keys()), total=len(futures), desc="🧩 Calculating shortest paths"):
        try:
            id_i, id_j = futures[future]
            path_length = future.result() if future.result() is not None else 999999
            mapping[(id_i, id_j)] = path_length
            mapping[(id_j, id_i)] = path_length
        except Exception as e:
            print(f"Error calculating path between {id_i} and {id_j}: {e}")
            mapping[(id_i, id_j)] = 999999
            mapping[(id_j, id_i)] = 999999
    return mapping


def route_bruteforce(sitelist, distance_map, start_hub, end_hub, max_samples=1000):
    """Brute-force (sampled) route evaluation for small multi-hub cases."""
    sites = sitelist["site_id"].unique().tolist()
    total_perms = math.factorial(len(sites))
    print(f"ℹ️ Total permutations to check: {total_perms:,}")

    if total_perms > max_samples:
        print(f"⚠️ Too many permutations ({total_perms:,}). Sampling {max_samples} random permutations.")
        all_perms = list(permutations(sites))
        sampled_perms = random.sample(all_perms, max_samples)
        print(f"ℹ️ Sampled {len(sampled_perms):,} permutations.")
    else:
        sampled_perms = list(permutations(sites))
        print(f"ℹ️ Using all {len(sampled_perms):,} permutations.\n")

    best_length = float("inf")
    best_route = None

    def _eval_perm(perm, distance_map, start_hub, end_hub, current_best):
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
            executor.submit(_eval_perm, perm, distance_map, start_hub, end_hub, best_length): perm
            for perm in sampled_perms
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="🧩 Calculating optimal routes"):
            try:
                result = future.result()
                if result[0] is not None and result[1] < best_length:
                    best_route, best_length = result
            except Exception as e:
                print(f"Error processing permutation: {e}")
                continue

    if best_route is None:
        print("❌ No valid route found")
        return None, float("inf")
    return best_route, best_length


def route_christofides(
    sitelist,
    distance_map,
    start_hub,
    end_hub,
    weight="weight",
    two_opt=True,
    max_2opt_iter=2000,
):
    """Christofides-based open TSP between start and end hubs (with optional 2-opt)."""
    import networkx as nx
    from networkx.algorithms import approximation

    sites = sitelist["site_id"].tolist()
    all_sites = [start_hub] + sites + [end_hub]

    G = nx.Graph()
    G.add_nodes_from(all_sites)

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

    def _plen(seq):
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
        best_len = _plen(best)
        improved = True
        it = 0
        while improved and it < max_2opt_iter:
            improved = False
            it += 1
            for i in range(1, len(best) - 2):
                for k in range(i + 1, len(best) - 1):
                    new_path = best[:i] + best[i : k + 1][::-1] + best[k + 1 :]
                    new_len = _plen(new_path)
                    if new_len + 1e-9 < best_len:
                        best, best_len = new_path, new_len
                        improved = True
                        break
                if improved:
                    break
        path, total_len = best, best_len
    else:
        total_len = _plen(path)
    return path, total_len


# ------------------------------------------------------
# 2) CLUSTER HUB ROUTING
# ------------------------------------------------------
def route_single(
    cluster_site,
    cluster,
    fo_hub,
    roads,
    nodes,
    G,
    area="area",
    area_col="region",
):
    """Route generation for clusters with a single FO hub."""
    from modules.scoring import duplicate_scores
    from modules.graph import build_cluster_graph

    cluster_site = cluster_site.drop_duplicates(subset="nearest_node")
    cluster_graph = build_cluster_graph(cluster_site, G, node_col="nearest_node")
    if cluster_graph is None:
        print(f"🔴 Skipping ring {cluster}. No route available.")
        return None, None

    list_nodes = list(cluster_graph.nodes)
    hub_node = fo_hub["nearest_node"].iloc[0] if not fo_hub.empty else None
    if hub_node and hub_node in list_nodes:
        reordered = [hub_node] + [node for node in list_nodes if node != hub_node]
        init_cycle = reordered + [reordered[0]] if reordered[0] != reordered[-1] else reordered
    else:
        init_cycle = list_nodes + [list_nodes[0]] if list_nodes and list_nodes[0] != list_nodes[-1] else list_nodes

    algorithms = {
        "Threshold TSP": lambda g: nx.approximation.threshold_accepting_tsp(
            g, init_cycle=init_cycle, weight="weight", max_iterations=10, seed=42, source=hub_node
        ),
        "Simulated Annealing": lambda g: nx.approximation.simulated_annealing_tsp(
            g, init_cycle=init_cycle, weight="weight", max_iterations=10, seed=42, source=hub_node
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

            paths_data, points_data = [], []
            if sitepath and len(sitepath) > 1:
                for idx, node in enumerate(sitepath[:-1]):
                    start_site = cluster_site[cluster_site["nearest_node"] == node].iloc[0]
                    end_site = cluster_site[cluster_site["nearest_node"] == sitepath[idx + 1]].iloc[0]
                    start_node = start_site["nearest_node"]
                    end_node = end_site["nearest_node"]

                    try:
                        path, path_geom, path_length = route_path(start_node, end_node, G, roads, merged=True)
                        path_geom, path_length = dropwire_connection(path_geom, start_site, end_site, nodes, start_node, end_node)
                        
                        if not path_geom.is_empty:
                            segment_name = f"{start_site['site_id']}-{end_site['site_id']}"
                            paths_data.append({
                                "name": segment_name,
                                "near_end": start_site["site_id"],
                                "far_end": end_site["site_id"],
                                "algo": algo_name,
                                "ring_name": cluster,
                                area_col: area,
                                "fo_note": "merged",
                                "length": round(path_length, 3),
                                "geometry": path_geom,
                            })
                    except nx.NetworkXNoPath:
                        print(f"No path found for {algo_name} | Cluster {cluster} | Start: {start_site['site_id']} | End: {end_site['site_id']}")
                        continue
                    except Exception as e:
                        print(f"❌ Error processing path for {algo_name} | Cluster {cluster}: {e}")
                        continue

                for node in sitepath:
                    site = cluster_site[cluster_site["nearest_node"] == node].iloc[0]
                    if site["site_id"] not in [point["site_id"] for point in points_data]:
                        points_data.append({
                            "site_id": site["site_id"],
                            "site_name": (site["site_name"] if "site_name" in site else None),
                            "site_type": (site["site_type"] if "site_type" in site else None),
                            area_col: area,
                            "algo": algo_name,
                            "ring_name": cluster,
                            "nearest_node": node,
                            "geometry": site["geometry"],
                        })
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
    print(f"🏆 Best Algorithm | {cluster:2} | {best_algo['name']} | Cost {best_algo['cost']:2f}")
    return final_paths, final_points


def route_multi(
    cluster_site,
    cluster,
    roads,
    nodes,
    G,
    weight="weight",
    area="area",
    area_col="region",
):
    """Route generation for clusters with multiple FO hubs."""
    total_sites = len(cluster_site)
    cluster_site = cluster_site.drop_duplicates(subset="nearest_node").to_crs(epsg=3857).reset_index(drop=True)

    sitelist = cluster_site[~cluster_site["site_type"].str.lower().str.contains("hub")].copy()
    fo_hub = cluster_site[cluster_site["site_type"].str.lower().str.contains("hub")].copy()
    start_hub = fo_hub["site_id"].iloc[0] if not fo_hub.empty else None
    end_hub   = fo_hub["site_id"].iloc[-1] if not fo_hub.empty else None

    mapping = map_distance(cluster_site, G, weight=weight)
    print(f"ℹ️ Distance mapping complete | Total pairs: {len(mapping)}")

    if total_sites <= 8:
        print("🧩 Multi Hub | Brute Force")
        best_route, best_length = route_bruteforce(sitelist, mapping, start_hub, end_hub)
    else:
        print("🧩 Multi Hub | Christofides")
        best_route, best_length = route_christofides(sitelist, mapping, start_hub, end_hub)

    if best_route is None:
        print(f"❌ No valid route found for cluster {cluster}.")
        return None, None

    print(f"✅ Optimal route cluster {cluster:20}:")
    print(f"    - Route {(' > '.join(str(x) for x in best_route))}")
    print(f"    - Total Cost: {best_length / 1000:,.2f} km")

    paths_data, points_data = [], []

    for idx, site in enumerate(best_route[:-1]):
        start_site = cluster_site[cluster_site["site_id"] == site].iloc[0]
        end_site = cluster_site[cluster_site["site_id"] == best_route[idx + 1]].iloc[0]
        start_node = cluster_site.loc[cluster_site["site_id"] == site, "nearest_node"].values[0]
        end_node = cluster_site.loc[cluster_site["site_id"] == best_route[idx + 1], "nearest_node"].values[0]
        try:
            path, path_geom, path_length = route_path(start_node, end_node, G, roads, merged=True)
            path_geom, path_length = dropwire_connection(path_geom, start_site, end_site, nodes, start_node, end_node)

            if not path_geom.is_empty:
                segment_name = f"{start_site['site_id']}-{end_site['site_id']}"
                paths_data.append({
                    "name": segment_name,
                    "near_end": start_site["site_id"],
                    "far_end": end_site["site_id"],
                    "algo": "Multi Hub",
                    "ring_name": cluster,
                    area_col: area,
                    "fo_note": "merged",
                    "length": round(path_length, 3),
                    "geometry": path_geom,
                })
        except nx.NetworkXNoPath:
            print(f"No path found for Cluster {cluster} | Start: {site} | End: {best_route[idx + 1]}")
            continue
        except Exception as e:
            print(f"❌ Error processing path for Cluster {cluster}: {e}")
            continue

    for site in best_route:
        site_data = cluster_site[cluster_site["site_id"] == site].iloc[0]
        if site_data["site_id"] not in [point["site_id"] for point in points_data]:
            points_data.append({
                "site_id": site_data["site_id"],
                "site_name": (site_data["site_name"] if "site_name" in site_data else None),
                "site_type": (site_data["site_type"] if "site_type" in site_data else None),
                "algo": "Multi Hub",
                area_col: area,
                "ring_name": cluster,
                "nearest_node": site_data["nearest_node"],
                "geometry": site_data["geometry"],
            })

    if not paths_data or not points_data:
        print(f"⚠️ Empty data for cluster {cluster}. Skipping...")
        return None, None

    final_paths = gpd.GeoDataFrame(paths_data, geometry="geometry", crs="EPSG:3857")
    final_points = gpd.GeoDataFrame(points_data, geometry="geometry", crs="EPSG:3857")
    total_length = final_paths["length"].sum()
    print(f"ℹ️ Total Length for cluster {cluster}: {total_length:8,.2f} meters")
    return final_paths, final_points


# ------------------------------------------------------
# 3) CLUSTER-LEVEL ORCHESTRATION
# ------------------------------------------------------
def ring_cluster(cluster_args):
    """Process a single cluster: build graph, route, SPOF, save parquet."""
    cluster_site = cluster_args["cluster_site"]
    cluster = cluster_args["ring_name"]
    area = cluster_args.get("area", "area")
    area_col = cluster_args.get("area_col", "region")
    ref_fo = cluster_args.get("ref_fo", None)
    cable_cost = cluster_args.get("cable_cost", 35000)
    export_dir = cluster_args.get("export_dir", None)

    final_paths = []
    final_points = []

    print(f"🛟 Processing Cluster {cluster} | Area: {area} | Total Site: {len(cluster_site):,}")
    cluster_site = cluster_site.copy().to_crs(epsg=3857).reset_index(drop=True)

    if cluster_site.empty:
        print(f"⚠️ No data found for cluster {cluster}. Skipping...")
        return None, None

    route_path_fp = os.path.join(export_dir, f"Paths_{area}_{cluster}.parquet")
    point_path_fp = os.path.join(export_dir, f"Points_{area}_{cluster}.parquet")

    if os.path.exists(point_path_fp):
        paths = gpd.read_parquet(route_path_fp)
        points = gpd.read_parquet(point_path_fp)
        print(f"ℹ️ Paths for cluster {cluster} already exist. Skipping...")
        return paths, points

    hex_list = identify_hexagon(cluster_site, type="convex")
    roads = retrieve_roads(hex_list, type="roads").to_crs(epsg=3857)
    nodes = retrieve_roads(hex_list, type="nodes").to_crs(epsg=3857)
    G = build_graph(roads, graph_type="full_weighted", ref_fo=ref_fo, cable_cost=cable_cost, avoid_railway=False)

    node_sindex = nodes.sindex
    cluster_site["nearest_node"] = cluster_site.geometry.apply(
        lambda geom: nodes.at[node_sindex.nearest(geom)[1][0], "node_id"]
    )

    fo_hub          = cluster_site[cluster_site["site_type"].str.lower().str.contains("hub")].copy()
    sitelist        = cluster_site[~cluster_site["site_type"].str.lower().str.contains("hub")].copy()
    total_sites     = len(cluster_site)
    num_hub         = len(fo_hub)
    num_sitelist    = len(sitelist)

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

    match num_hub:
        case 1:
            final_paths, final_points = route_single(cluster_site, cluster, fo_hub, roads, nodes, G, area=area, area_col=area_col)
            if final_paths is None or final_points is None:
                print(f"❌ No valid paths found for cluster {cluster}.")
                return None, None
        case 2:
            final_paths, final_points = route_multi(cluster_site, cluster, roads, nodes, G, area=area, area_col=area_col)
        case _:
            print(f"⚠️ No FO Hub found in cluster {cluster}. Using all nodes for pathfinding.")
            print(cluster_site['site_type'].value_counts())
    if final_paths is None or final_points is None:
        print(f"⚠️ No valid paths or points found for cluster {cluster}. Skipping SPOF detection.")
        return None, None

    print(f"ℹ️ Running SPOF Detection for cluster {cluster}...")
    final_paths = spof_detection(final_paths, final_points, G, roads, nodes, threshold_spof=3000, threshold_alt=25)

    print(f"\n📊 Summary for Cluster {cluster} | Area: {area}")
    print(f"Total Paths         | {area:15} | {cluster:2}: {len(final_paths):,}")
    print(f"Total Points        | {area:15} | {cluster:2}: {len(final_points):,}")
    print(f"FO Hub Nodes        | {area:15} | {cluster:2}: {len(fo_hub):,}")
    print()

    print("📍 Running Connected Routes")
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

            if next_site in visited_sites and next_site != start_hub:
                print(f"⚠️ Detected loop at site {next_site}. Breaking to prevent infinite loop.")
                break

            visited_sites.add(current_site)
            current_site = next_site
            iteration_count += 1
        print(f"ℹ️ Connected {len(connected_routes)} routes in {iteration_count} iterations")
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

            if next_site == end_hub:
                connection_list.append(end_hub)
                print(f"✅ Path completed from {start_hub} to {end_hub}")
                break

            if next_site in visited_sites and next_site != end_hub:
                print(f"⚠️ Detected loop at site {next_site}. Breaking to prevent infinite loop.")
                visited_sites.add(current_site)
                break

            visited_sites.add(current_site)
            current_site = next_site
            iteration_count += 1
        print(f"ℹ️ Connected {len(connected_routes)} routes in {iteration_count} iterations")

    if not connected_routes:
        print(f"⚠️ No connected routes found for FO Hub in cluster {cluster}. Skipping...")
        return final_paths, final_points

    connected_routes_df = pd.DataFrame(connected_routes)
    connected_routes_gdf = gpd.GeoDataFrame(connected_routes_df, geometry="geometry", crs="EPSG:3857").reset_index(drop=True)

    connected_sites = []
    for site_id in connection_list:
        site_row = final_points[final_points["site_id"] == site_id]
        if not site_row.empty:
            connected_sites.append(site_row.iloc[0])

    connected_sites_gdf = gpd.GeoDataFrame(connected_sites, geometry="geometry", crs="EPSG:3857").reset_index(drop=True)
    connected_sites_gdf = connected_sites_gdf.drop_duplicates(subset=["site_id", "geometry"])

    if connected_sites_gdf.empty:
        return None, None

    total_length = connected_routes_gdf["length"].sum()
    print(f"Total Length | {area} | {cluster}: {total_length / 1000:,.2f} km")

    connected_routes_gdf = connected_routes_gdf.to_crs(epsg=4326)
    connected_sites_gdf = connected_sites_gdf.to_crs(epsg=4326)

    connected_routes_gdf.to_parquet(os.path.join(export_dir, f"Paths_{area}_{cluster}.parquet"))
    connected_sites_gdf.to_parquet(os.path.join(export_dir, f"Points_{area}_{cluster}.parquet"))
    print(f"🌏 Cluster {cluster} parquet saved. \n")
    return connected_routes_gdf, connected_sites_gdf


def ring_parallel(
    clustered_sites: gpd.GeoDataFrame,
    area:str,
    area_col="region",
    cluster_col="ring_name",
    export_dir=None,
    ref_fo=None,
    cable_cost=35000,
    task_celery=False,
):
    """Process multiple clusters in parallel and merge results."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print(f"🧩 Parallel Clustering {area}")
    if clustered_sites.empty:
        return None, None

    if export_dir is None:
        raise ValueError("Export directory is not provided. Please specify a valid export directory.")

    os.makedirs(export_dir, exist_ok=True)
    clusters = sorted(clustered_sites[cluster_col].unique().tolist())
    print(f"ℹ️ Area: {area} | Total Clusters: {len(clusters):,}")

    cluster_args = []
    for cluster in clusters:
        if cluster == -1:
            continue
        cluster_sites = clustered_sites[clustered_sites[cluster_col] == cluster]
        if cluster_sites.empty:
            continue
        cluster_args.append({
            "cluster_site": cluster_sites,
            "ring_name": cluster,
            "area": area,
            "area_col": area_col,
            "ref_fo": ref_fo,
            "export_dir": export_dir,
            "cable_cost": cable_cost,
        })

    if not cluster_args:
        print("⚠️ No valid clusters found for parallel processing. Skipping...")
        return None, None

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(ring_cluster, args): args["ring_name"] for args in cluster_args}
        final_paths = []
        final_points = []
        total_futures = len(futures)

        if task_celery and total_futures > 1:
            task_celery.update_state(state="PROGRESS", meta={"status": f"Processing {total_futures} clusters in parallel"})

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Clusters"):
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
                        meta={"status": f"Completed {len(final_paths)}/{total_futures} clusters"},
                    )
            except Exception as e:
                print(f"❌ Error processing cluster {cluster_id}: {e}")
                if task_celery:
                    task_celery.update_state(
                        state="PROGRESS",
                        meta={"status": f"Error in cluster {cluster_id}: {e}. Completed {len(final_paths)}/{total_futures} clusters"},
                    )
                continue

        if final_paths:
            final_paths = pd.concat(final_paths, ignore_index=True).drop_duplicates(subset=["geometry"])
            print(f"ℹ️ Total Paths after parallel processing: {len(final_paths):,}")
        if final_points:
            final_points = pd.concat(final_points, ignore_index=True).drop_duplicates(subset=["geometry"])
            print(f"ℹ️ Total Points after parallel processing: {len(final_points):,}")
    return final_paths, final_points


# ------------------------------------------------------
# 4) MODE: SUPERVISED RING NETWORK
# ------------------------------------------------------
def supervised_validation(excel_file: str | pd.DataFrame | gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    print(f"ℹ️ Checking Supervised.")
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
    required_columns = ["site_id", "site_name", "site_type", "lat", "long", "ring_name", "flag"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in sheets: {missing_cols}.\n Please ensure the input data contains {', '.join(required_columns)} columns.")

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
    
    # SANITIZE SITE ID AND NAME
    df["site_id"] = df["site_id"].apply(safe_stringify)
    df["site_name"] = df["site_name"].apply(safe_stringify)
    
    # GEOMETRY
    df_geom = gpd.points_from_xy(df["long"], df["lat"], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(df, geometry=df_geom)

    if "region" in gdf.columns:
        gdf["region"] = gdf["region"].apply(safe_stringify)
    else:
        group = auto_group(gdf, distance=10000)
        grouped_ring = gpd.sjoin(gdf[['geometry', 'ring_name']], group[['geometry', 'region']]).drop(columns='index_right')
        grouped_ring = grouped_ring.groupby('ring_name')['region'].first().to_dict()
        gdf['region'] = gdf['ring_name'].map(grouped_ring)
        

    # VALIDITY
    for idx, row in gdf.iterrows():
        if not row["geometry"].is_valid:
            raise ValueError(f"Invalid geometry at index {idx} with site_id '{row['site_id']}'.")
    gdf["geometry"] = gdf["geometry"].apply(remove_z)

    # SUMMARY
    print(f"ℹ️ Total Records: {len(gdf):,}")
    print(f"✅ Input Data Validity Check Passed.")
    return gdf

def ring_supervised(
    sites_input,
    area,
    export_dir: str,
    area_col: str = "region",
    cluster_col: str = "ring_name",
    fo_expand: gpd.GeoDataFrame = None,
    **kwargs,
):
    """Run supervised routing for a single area, saving parquet results."""
    print(f"🧩 Ring Network | {area}")
    cable_cost = kwargs.get("cable_cost", 35000)
    task_celery = kwargs.get("task_celery", False)

    os.makedirs(export_dir, exist_ok=True)
    try:
        if isinstance(sites_input, str):
            sites_input = gpd.read_parquet(sites_input)
            sites_input[area_col] = sites_input[area_col].str.strip().str.upper()
        else:
            sites_input = sites_input.copy().reset_index(drop=True)

        if sites_input.crs is None:
            sites_input.set_crs(epsg=4326, inplace=True)
        sites_input = sites_input.to_crs(epsg=3857)

        site_area = sites_input[sites_input[area_col] == area].copy()
        unique_clusters = site_area[area_col].dropna().unique().tolist()
        if not unique_clusters:
            print(f"No clusters found for area {area}. Skipping...")
            return None

        print(f"✨ Processing {area_col}: {area}")
        print(f"ℹ️ Total Sites       | {area_col} {area}: {len(sites_input):,} data")

        ref_fo = None
        if fo_expand is not None:
            print("ℹ️ Fiber Expand defined")
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

            print("🧩 Preparing FO Ref Nodes...")
            ref_fo = set(nodes[nodes["ref_fo"] == 1]["node_id"])
            fo_expand = fo_expand.to_crs(3857)
            fo_expand["geometry"] = fo_expand.geometry.buffer(20)
            nodes_within = gpd.sjoin(nodes, fo_expand, how="inner", predicate="intersects")
            nodes_within = set(nodes_within["node_id"])

            if isinstance(nodes_within, (list, tuple, set)):
                ref_fo.update(nodes_within)
                print(f"ℹ️ Added {len(nodes_within)} expanded FO nodes")
            else:
                print("⚠️ expand_fo should be a list, tuple, or set")
            print("ℹ️ Reference FO retrieved.")

        if task_celery:
            task_celery.update_state(state="PROGRESS", meta={"status": "Processing ring data"})

        paths, points = ring_parallel(
            clustered_sites=site_area,
            area=area,
            area_col=area_col,
            cluster_col=cluster_col,
            export_dir=export_dir,
            ref_fo=ref_fo,
            cable_cost=cable_cost,
            task_celery=task_celery,
        )
        return paths, points
    except Exception as e:
        print(f"❌❌ Error processing area {area}: {e} \n")
        return None


# ------------------------------------------------------
# 5) EXPORT & COMPILATION
# ------------------------------------------------------
def export_kmz(main_kml, ring_data, point_data, folder):
    """Write topology, sites, hubs, and routes into KMZ structure."""
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

    ring_topology = create_topology(point_data).to_crs(epsg=4326)
    ring_topology["connection"] = "Connection"

    route_columns = ["near_end", "far_end", "geometry", "ring_name", "route_type", "length"]
    if "existing_cable_length" in ring_data.columns:
        route_columns.append("existing_cable_length")
    if "new_cable_length" in ring_data.columns:
        route_columns.append("new_cable_length")
    ring_route = ring_data[route_columns].copy()
    ring_route["route_name"] = ring_route["near_end"] + "-" + ring_route["far_end"]

    ring_sites = cluster_points[~cluster_points["site_type"].str.lower().str.contains("hub")].copy()
    ring_hub = cluster_points[cluster_points["site_type"].str.lower().str.contains("hub")].copy()

    ring_sites = ring_sites[available_col].rename(columns=used_columns)
    ring_hub = ring_hub[available_col].rename(columns=used_columns)

    kml_updated = export_kml(ring_topology, main_kml, filename, subfolder=folder, name_col="connection", color="#FF00FF", size=2, popup=False)
    kml_updated = export_kml(ring_hub, kml_updated, filename, subfolder=f"{folder}/FO Hub", name_col="Site ID", icon="http://maps.google.com/mapfiles/kml/paddle/A.png", size=0.8, popup=True)
    kml_updated = export_kml(ring_sites, kml_updated, filename, subfolder=f"{folder}/Site List", name_col="Site ID", color="#FFFF00", size=0.8, popup=True)
    kml_updated = export_kml(ring_route, kml_updated, filename, subfolder=f"{folder}/Route", name_col="route_name", color="#0000FF", size=3, popup=False)
    return kml_updated


def export_gpkg(gpkg_path, ring_data, point_data, cluster):
    """Write cluster topology, routes, sites, and hubs to GeoPackage."""
    ring_data = ring_data.copy()
    cluster_points = point_data.copy()

    ring_data["start"] = ring_data["near_end"].apply(lambda x: cluster_points[cluster_points["site_id"] == x]["geometry"].values[0])
    ring_data["end"] = ring_data["far_end"].apply(lambda x: cluster_points[cluster_points["site_id"] == x]["geometry"].values[0])
    ring_data = ring_data.reset_index(drop=True)

    topology = create_topology(point_data)
    topology.to_file(gpkg_path, layer=f"{cluster}_Topology", driver="GPKG")

    ring_route = ring_data[["near_end", "far_end", "ring_name", "route_type", "length", "geometry"]].copy()
    ring_route["route_name"] = ring_route["near_end"] + "-" + ring_route["far_end"]
    ring_route.to_file(gpkg_path, layer=f"{cluster}_Route", driver="GPKG")

    ring_sites = cluster_points[["site_id", "site_name", "site_type", "geometry"]].copy()
    ring_sites = ring_sites[~ring_sites["site_type"].str.lower().str.contains("hub")].copy()
    if ring_sites.empty:
        print("⚠️ No sites found. Skipping site export.")
        return gpkg_path
    ring_sites.to_file(gpkg_path, layer=f"{cluster}_Site List", driver="GPKG")

    ring_hub = cluster_points[cluster_points["site_type"].str.lower().str.contains("hub")].copy()
    if ring_hub.empty:
        print("⚠️ No FO Hub found. Skipping FO Hub export.")
        return gpkg_path

    ring_hub.to_file(gpkg_path, layer=f"{cluster}_FO Hub", driver="GPKG")
    return gpkg_path


def compile_ring(parquet_dir):
    """Compile all per-cluster parquet files into combined paths/points."""
    list_files = os.listdir(parquet_dir)
    point_files = [f for f in list_files if f.endswith(".parquet") and f.startswith("Points")]
    path_files  = [f for f in list_files if f.endswith(".parquet") and f.startswith("Paths")]

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

    points_data = pd.concat(points_data, ignore_index=True) if points_data else pd.DataFrame()
    paths_data  = pd.concat(paths_data,  ignore_index=True) if paths_data  else pd.DataFrame()

    print(f"ℹ️ Total Points DataFrames : {len(points_data):,}")
    print(f"ℹ️ Total Paths DataFrames  : {len(paths_data):,}")

    result_cluster = points_data["ring_name"].unique().tolist() if not points_data.empty else []
    print(f"ℹ️ Total Cluster Data: {len(result_cluster):,}")

    if not points_data.empty:
        points_data.to_crs(epsg=4326).to_parquet(os.path.join(parquet_dir, "All_Points.parquet"), index=False)
    if not paths_data.empty:
        paths_data.to_crs(epsg=4326).to_parquet(os.path.join(parquet_dir, "All_Paths.parquet"), index=False)
    print(f"✅ Compiled Parquet files saved to {parquet_dir}")
    for file in point_files + path_files:
        try:
            os.remove(os.path.join(parquet_dir, file))
        except Exception as e:
            print(f"❌ Error removing file {file}: {e}")
    return points_data, paths_data


def connect_nearfar(points_gdf: gpd.GeoDataFrame, paths_gdf: gpd.GeoDataFrame):
    """Attach near_end and far_end site ids to points based on per-ring routes."""
    miss_point = [c for c in ["site_id", "ring_name"] if c not in points_gdf.columns]
    miss_paths = [c for c in ["near_end", "far_end", "ring_name"] if c not in paths_gdf.columns]

    if miss_point:
        raise ValueError(f"Missing column in points: {miss_point}")
    if miss_paths:
        raise ValueError(f"Missing column in paths: {miss_paths}")

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

def save_kml(
    points: gpd.GeoDataFrame,
    paths: gpd.GeoDataFrame,
    topology: gpd.GeoDataFrame,
    export_dir: str,
    method='Supervised'
):
    import simplekml
    
    date_today = datetime.now().strftime("%Y%m%d")
    kmz_path = os.path.join(export_dir, f"Intersite Design_{method}_{date_today}.kmz")

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
            topology_ring['geometry'] = topology_ring.geometry.apply(lambda geom: linemerge(geom) if type(geom) == MultiLineString else geom)
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

def save_supervised(
    points: gpd.GeoDataFrame,
    paths: gpd.GeoDataFrame,
    topology: gpd.GeoDataFrame,
    export_dir: str,
    method:str = "Supervised"
):
    # EXPORT PARQUET
    if not points.empty:
        points.to_crs(epsg=4326).to_parquet(os.path.join(export_dir, f"Points.parquet"), index=False)
        print(f"✅ Exported Fixed with {len(points):,} records.")
    if not paths.empty:
        paths.to_crs(epsg=4326).to_parquet(os.path.join(export_dir, f"Route.parquet"), index=False)
        print(f"✅ Exported Route Fixed with {len(paths):,} records.")
    if not topology.empty:
        topology = topology.sort_values(by=['ring_name']).reset_index(drop=True)
        topology.to_crs(epsg=4326).to_parquet(os.path.join(export_dir, f"Topology.parquet"), index=False)
        print(f"✅ Exported Topology Route with {len(topology):,} records.")

    # EXPORT KML
    if not points.empty and not paths.empty and not topology.empty:
        save_kml(points, paths, topology, export_dir, method)

    # EXPORT EXCEL
    excel_path = os.path.join(export_dir, f"Summary Report_Intersite_{method}.xlsx")
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
            paths_report = paths_report.rename(columns={'merged': 'Length'})
            paths_report = paths_report.merge(
                paths[['ring_name', 'region', 'program']].drop_duplicates(),
                on='ring_name',
                how='left'
            )
            paths_report = paths_report.sort_values(by=['ring_name', 'near_end']).reset_index(drop=True)
            paths_report.columns = paths_report.columns.str.replace(' ', '_').str.lower()
            excel_styler(paths_report).to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"📊 Excel sheet '{sheet_name}' with {len(paths_report):,} records written.")


# ------------------------------------------------------
# 6) MAIN ENTRYPOINT
# ------------------------------------------------------
def main_supervised(
    site_data: gpd.GeoDataFrame,
    export_loc: str = "./exports",
    area_col: str = "region",
    cluster_col = "ring_name",
    fo_expand: gpd.GeoDataFrame = None,
    boq: bool = False,
    **kwargs,
):
    """Main supervised pipeline: per-area processing, compile, export KMZ/Excel."""
    cable_cost = kwargs.get("cable_cost", 35000)
    vendor = kwargs.get("vendor", "TBG")
    program = kwargs.get("program", "Fiberization")
    method = kwargs.get("method", "Supervised")
    task_celery = kwargs.get("task_celery", None)
    design_type = 'Bill of Quantity' if boq else 'Design'

    if "site_id" in site_data.columns:
        site_data["site_id"] = site_data["site_id"].astype(str)

    site_data = sanitize_header(site_data)
    site_data = supervised_validation(site_data)

    date_today = datetime.now().strftime("%Y%m%d")
    week = detect_week(date_today)
    export_dir = f"{export_loc}/Intersite Design/W{str(week)}_{date_today}/{method}"
    checkpoint_dir = os.path.join(export_dir, "Checkpoint")

    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if "index_right" in site_data.columns:
        site_data = site_data.drop(columns=["index_right"])
    area_list = sorted(site_data[area_col].unique().tolist())

    output_paths = os.path.join(checkpoint_dir, "All_Paths.parquet")
    output_point = os.path.join(checkpoint_dir, "All_Points.parquet")
    if os.path.exists(output_paths) and os.path.exists(output_point):
        print("ℹ️ Output file already exists. Loading existing data...")
        all_paths = gpd.read_parquet(output_paths)
        all_points = gpd.read_parquet(output_point)
        print("✅ Loaded existing data successfully.")
    else:
        print(f"🔥 Starting {method} Intersite")
        print(f"ℹ️ Vendor Name       : {vendor}")
        print(f"ℹ️ Program Name      : {program}")
        print(f"ℹ️ Design Type       : {design_type}")
        print(f"ℹ️ Total Sites       : {len(site_data):,}")

        for area in tqdm(area_list, desc=f"Processing {area_col}"):
            site_area = site_data[site_data[area_col] == area].copy()
            if site_area.empty:
                print(f"⚠️ No data found for {area_col} {area}. Skipping...")
                continue

            paths, points = ring_supervised(
                site_area,
                area=area,
                area_col=area_col,
                cluster_col=cluster_col,
                export_dir=checkpoint_dir,
                fo_expand=fo_expand,
                cable_cost=cable_cost,
                task_celery=task_celery
            )

            if paths is None or points is None:
                print(f"⚠️ No paths or points found for {area_col} {area}. Skipping...")
                continue

        print("🔗 Merging Parquet files...")
        all_points, all_paths = compile_ring(checkpoint_dir)
        all_points = all_points.reset_index(drop=True)
        all_paths = all_paths.reset_index(drop=True)

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
    if "program" not in all_paths.columns:
        all_paths["program"] = program


    # EXPORT
    if boq:
        main_boq(all_points, all_paths, export_dir=export_dir)
    else:
        # TOPOLOGY CHECK
        topology_paths = create_topology(all_points)
        save_supervised(all_points, all_paths, topology_paths, export_dir, method)
    print(f"✅ Export completed.")
    print(f"🔥 Fiberization process completed. All files saved to \n{export_dir}.")


if __name__ == "__main__":
    # excel_file = r"D:\Data Analytical\PROJECT\TASK\NOVEMBER\Week 3\BoQ Intersite\Trial BoQ Intersite.xlsx"
    excel_file = r"D:\JACOBSPACE\TBIG Impact 2025\QCC Fiberisasi\Asessment\Q1AOP2025 All_Smart Routing Automatic Template.xlsx"
    export_dir = r"D:\Data Analytical\PROJECT\TASK\NOVEMBER\Week 3\BoQ Intersite\Export\TaskForce"
    area_col = 'region'
    cluster_col = 'ring_name'
    path_type = 'classified'
    program = "Trial BOQ"
    vendor = "TBG"
    boq = True

    site_data = pd.read_excel(excel_file)
    site_data = sanitize_header(site_data)
    site_data = supervised_validation(site_data)
    fo_expand = None

    date_today = datetime.now().strftime("%Y%m%d")
    export_loc = f"{export_dir}/{date_today}"
    os.makedirs(export_loc, exist_ok=True)

    result = main_supervised(
        site_data=site_data,
        export_loc=export_loc,
        area_col=area_col,
        cluster_col=cluster_col,
        fo_expand=fo_expand,
        program=program,
        vendor=vendor,
        boq=boq
    )

    zip_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_Supervised_Task.zip"
    zip_filepath = os.path.join(export_loc, zip_filename)
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(export_loc):
            for file in files:
                if file != zip_filename:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_loc)
                    zipf.write(file_path, arcname)
    print("📦 Result files zipped.")
