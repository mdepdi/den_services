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

sys.path.append(r"D:\JACOBS\SERVICE\API")

from time import time
from datetime import datetime
from tqdm import tqdm
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import linemerge
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import permutations

from core.logger import create_logger
from modules.data import fiber_utilization
from modules.h3_route import identify_hexagon, retrieve_roads, build_graph
from modules.utils import auto_group, spof_detection, create_topology, route_path, dropwire_connection
from modules.table import sanitize_header, detect_week, excel_styler
from modules.validation import input_newring
from modules.kml import export_kml, sanitize_kml
from boq_algorithm import main_boq


# ------------------------------------------------------
# LOGGING CONFIGURATION
# ------------------------------------------------------
logger = create_logger(__file__)

# ------------------------------------------------------
# 1) CORE DISTANCE & ROUTING
# ------------------------------------------------------
def map_distance(cluster_sites, G=None, weight="weight"):
    """Compute pairwise shortest-path distances for all site pairs."""
    logger.info(
        f"🧩 Calculating pairwise shortest-path distances for {len(cluster_sites):,} sites "
        f"(weight='{weight}')"
    )
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
            logger.error(f"❌ Error calculating distance between {id_i} and {id_j}: {e}")
            mapping[(id_i, id_j)] = 999999
            mapping[(id_j, id_i)] = 999999
    logger.info(f"🏆 Distance mapping completed with {len(mapping):,} pairs")
    return mapping


def route_bruteforce(sitelist, distance_map, start_hub, end_hub, max_samples=1000):
    """Brute-force (sampled) route evaluation for small multi-hub cases."""
    sites = sitelist["site_id"].unique().tolist()
    total_perms = math.factorial(len(sites))

    if total_perms > max_samples:
        logger.warning(
            f"⚠️ Permutations too many ({total_perms:,}). Sampling {max_samples:,} permutations."
        )
        all_perms = list(permutations(sites))
        sampled_perms = random.sample(all_perms, max_samples)
    else:
        logger.info(f"🧩 Evaluating all {total_perms:,} permutations (brute-force).")
        sampled_perms = list(permutations(sites))

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
        for future in tqdm(as_completed(futures), total=len(futures), desc="🧩 Searching best route (brute-force)"):
            try:
                result = future.result()
                if result[0] is not None and result[1] < best_length:
                    best_route, best_length = result
            except Exception as e:
                logger.error(f"❌ Error processing permutation: {e}")
                continue

    if best_route is None:
        logger.error("❌ No valid route found with brute-force.")
        return None, float("inf")

    logger.info(
        f"🏆 Best brute-force route from {start_hub} to {end_hub} "
        f"with total cost {best_length:,.2f}"
    )
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
    """Christofides-based open TSP between start and end hubs."""
    import networkx as nx
    from networkx.algorithms import approximation

    logger.info(
        f"🧩 Running Christofides TSP from {start_hub} to {end_hub} "
        f"(two_opt={two_opt}, max_2opt_iter={max_2opt_iter})"
    )

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
        logger.error(f"❌ Christofides TSP error: {e}")
        return None, None

    if len(tsp_path) > 1 and tsp_path[0] == tsp_path[-1]:
        tsp_path = tsp_path[:-1]

    if DUMMY in tsp_path:
        i = tsp_path.index(DUMMY)
        tsp_path = tsp_path[i:] + tsp_path[:i]
    else:
        logger.error("❌ DUMMY node not found in Christofides path. Cannot enforce endpoints.")
        return None, None

    if len(tsp_path) < 3:
        logger.error("❌ Christofides TSP path too short. Cannot enforce endpoints.")
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
        logger.info("🧩 Running 2-opt refinement on Christofides path...")
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
        logger.info(
            f"🏆 2-opt refinement completed in {it} iterations "
            f"(cost {total_len:,.2f})"
        )
    else:
        total_len = _plen(path)
        logger.info(f"🏆 Christofides route cost (no 2-opt): {total_len:,.2f}")

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

    logger.info(
        f"ℹ️ Single-hub routing | Cluster={cluster} | Area={area} | "
        f"Sites={len(cluster_site):,} | Hubs={len(fo_hub):,}"
    )

    cluster_site = cluster_site.drop_duplicates(subset="nearest_node")
    cluster_graph = build_cluster_graph(cluster_site, G, node_col="nearest_node")
    if cluster_graph is None:
        logger.warning(f"⚠️ No route available for cluster {cluster} (graph is None).")
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
        # "Christofides": lambda g: nx.approximation.christofides(g, weight="weight"),
    }

    best_algo = None
    for algo_name, algo_func in algorithms.items():
        logger.info(f"🧩 Trying algorithm '{algo_name}' for cluster {cluster}")
        try:
            sitepath = algo_func(cluster_graph)
            if not sitepath:
                logger.warning(
                    f"⚠️ Algorithm '{algo_name}' returned empty path for cluster {cluster}. Skipping."
                )
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
                        path_geom, path_length = dropwire_connection(
                            path_geom, start_site, end_site, nodes, start_node, end_node
                        )

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
                        logger.warning(
                            f"⚠️ No path found ({algo_name}) | Cluster={cluster} | "
                            f"Start={start_site['site_id']} | End={end_site['site_id']}"
                        )
                        continue
                    except Exception as e:
                        logger.error(
                            f"❌ Error processing path ({algo_name}) in cluster {cluster}: {e}"
                        )
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
                logger.warning(
                    f"⚠️ No valid path returned by '{algo_name}' in cluster {cluster}. Skipping."
                )
                continue
        except Exception as e:
            logger.error(f"❌ Error running '{algo_name}' for cluster {cluster}: {e}")
            continue

        if not paths_data or not points_data:
            logger.warning(
                f"⚠️ Algorithm '{algo_name}' produced empty data in cluster {cluster}. Skipping."
            )
            continue

        temp_paths_gdf = gpd.GeoDataFrame(paths_data, geometry="geometry", crs="EPSG:3857")
        temp_points_gdf = gpd.GeoDataFrame(points_data, geometry="geometry", crs="EPSG:3857")
        total_length = temp_paths_gdf["length"].sum()
        duplicate_points = duplicate_scores(temp_paths_gdf)
        total_cost = total_length + duplicate_points

        logger.info(
            f"ℹ️ Algorithm '{algo_name}' | Cluster={cluster} | "
            f"Cost={total_cost:,.2f} (Length={total_length:,.2f}, DuplicateScore={duplicate_points:,.2f})"
        )

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
        logger.warning(f"⚠️ No valid routing result for cluster {cluster}.")
        return None, None

    final_paths = best_algo["paths_gdf"].reset_index(drop=True)
    final_points = best_algo["points_gdf"].reset_index(drop=True)
    logger.info(
        f"🏆 Best algorithm for cluster {cluster}: {best_algo['name']} "
        f"(Cost={best_algo['cost']:,.2f}, Length={best_algo['length']:,.2f})"
    )
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
    logger.info(
        f"ℹ️ Multi-hub routing | Cluster={cluster} | Area={area} | "
        f"Sites={len(cluster_site):,}"
    )
    total_sites = len(cluster_site)
    cluster_site = cluster_site.drop_duplicates(subset="nearest_node").to_crs(epsg=3857).reset_index(drop=True)

    sitelist = cluster_site[~cluster_site["site_type"].str.lower().str.contains("hub")].copy()
    fo_hub = cluster_site[cluster_site["site_type"].str.lower().str.contains("hub")].copy()
    start_hub = fo_hub["site_id"].iloc[0] if not fo_hub.empty else None
    end_hub = fo_hub["site_id"].iloc[-1] if not fo_hub.empty else None

    mapping = map_distance(cluster_site, G, weight=weight)
    logger.info(
        f"ℹ️ Distance map built for cluster {cluster} with {len(mapping):,} pairs."
    )

    if total_sites <= 8:
        logger.info(f"🧩 Cluster {cluster}: using brute-force multi-hub routing.")
        best_route, best_length = route_bruteforce(sitelist, mapping, start_hub, end_hub)
    else:
        logger.info(f"🧩 Cluster {cluster}: using Christofides multi-hub routing.")
        best_route, best_length = route_christofides(sitelist, mapping, start_hub, end_hub)

    if best_route is None:
        logger.error(f"❌ No valid route found for cluster {cluster}.")
        return None, None

    logger.info(f"🏆 Optimal route for cluster {cluster}:")
    logger.info(f"ℹ️ Route sequence: {' > '.join(str(x) for x in best_route)}")
    logger.info(f"ℹ️ Total path cost: {best_length / 1000:,.2f} km")

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
            logger.warning(
                f"⚠️ No path found in cluster {cluster} | Start={site} | End={best_route[idx + 1]}"
            )
            continue
        except Exception as e:
            logger.error(f"❌ Error processing segment in cluster {cluster}: {e}")
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
        logger.warning(f"⚠️ Empty path/point data for cluster {cluster}.")
        return None, None

    final_paths = gpd.GeoDataFrame(paths_data, geometry="geometry", crs="EPSG:3857")
    final_points = gpd.GeoDataFrame(points_data, geometry="geometry", crs="EPSG:3857")
    total_length = final_paths["length"].sum()
    logger.info(
        f"ℹ️ Total routed length for cluster {cluster}: {total_length:8,.2f} meters"
    )
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

    logger.info(
        f"ℹ️ Processing cluster {cluster} | Area={area} | Total sites={len(cluster_site):,}"
    )
    cluster_site = cluster_site.copy().to_crs(epsg=3857).reset_index(drop=True)

    if cluster_site.empty:
        logger.warning(f"⚠️ Cluster {cluster} has no data. Skipping.")
        return None, None

    route_path_fp = os.path.join(export_dir, f"Paths_{area}_{cluster}.parquet")
    point_path_fp = os.path.join(export_dir, f"Points_{area}_{cluster}.parquet")

    if os.path.exists(point_path_fp):
        paths = gpd.read_parquet(route_path_fp)
        points = gpd.read_parquet(point_path_fp)
        logger.info(
            f"ℹ️ Cluster {cluster} already processed. Loaded paths/points from disk."
        )
        return paths, points

    hex_list = identify_hexagon(cluster_site, type="convex")
    roads = retrieve_roads(hex_list, type="roads").to_crs(epsg=3857)
    nodes = retrieve_roads(hex_list, type="nodes").to_crs(epsg=3857)
    G = build_graph(roads, graph_type="full_weighted", ref_fo=ref_fo, cable_cost=cable_cost, avoid_railway=False)

    node_sindex = nodes.sindex
    cluster_site["nearest_node"] = cluster_site.geometry.apply(
        lambda geom: nodes.at[node_sindex.nearest(geom)[1][0], "node_id"]
    )

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

    logger.info(f"ℹ️ Cluster: {cluster}")
    logger.info(f"ℹ️ Total sites: {total_sites}")
    logger.info(
        f"ℹ️ FO hubs: {num_hub:<2} | Start hub={start_hub} | End hub={end_hub}"
    )
    logger.info(
        f"ℹ️ Sitelist (non-hub): {num_sitelist:<2} | Total sitelist={num_sitelist}"
    )

    match num_hub:
        case 1:
            final_paths, final_points = route_single(
                cluster_site, cluster, fo_hub, roads, nodes, G, area=area, area_col=area_col
            )
            if final_paths is None or final_points is None:
                logger.error(f"❌ No valid path result for cluster {cluster}.")
                return None, None
        case 2:
            final_paths, final_points = route_multi(
                cluster_site, cluster, roads, nodes, G, area=area, area_col=area_col
            )
        case _:
            logger.warning(
                f"⚠️ Cluster {cluster} has unsupported hub count ({num_hub}). "
                f"Using all nodes for pathfinding."
            )
            logger.info(cluster_site['site_type'].value_counts())

    if final_paths is None or final_points is None:
        logger.warning(
            f"⚠️ No route result for cluster {cluster}. Skipping SPOF detection."
        )
        return None, None

    logger.info(f"🧩 Running SPOF detection for cluster {cluster}...")
    final_paths = spof_detection(final_paths, final_points, G, roads, nodes, threshold_spof=3000, threshold_alt=25)

    logger.info("🧩 Reconstructing connected ring routes...")
    connected_routes = []
    connection_list = []
    visited_sites = set()

    if len(fo_hub) == 1 and start_hub == end_hub:
        logger.info(f"ℹ️ Single FO hub {start_hub} (closed ring expected).")
        current_site = start_hub
        max_iterations = len(final_paths) + 1
        iteration_count = 0

        while iteration_count < max_iterations:
            next_row = final_paths[final_paths["near_end"] == current_site]
            if next_row.empty:
                logger.warning(
                    f"⚠️ No outgoing path from {current_site}. Stopping traversal."
                )
                break

            for _, row in next_row.iterrows():
                connected_routes.append(row)

            if current_site not in connection_list:
                connection_list.append(current_site)

            next_site = next_row["far_end"].values[0]
            if next_site == start_hub and len(connected_routes) > 1:
                connection_list.append(next_site)
                break

            if next_site in visited_sites and next_site != start_hub:
                logger.warning(
                    f"⚠️ Detected loop at site {next_site}. Stopping traversal."
                )
                break

            visited_sites.add(current_site)
            current_site = next_site
            iteration_count += 1

        logger.info(
            f"ℹ️ Connected {len(connected_routes)} segments "
            f"in {iteration_count} iterations (single-hub ring)."
        )
    else:
        logger.info(
            f"ℹ️ Multiple FO hubs: start={start_hub}, end={end_hub} (open ring)."
        )
        current_site = start_hub
        max_iterations = len(final_paths) + 1
        iteration_count = 0

        while iteration_count < max_iterations:
            next_row = final_paths[final_paths["near_end"] == current_site]
            if next_row.empty:
                logger.warning(
                    f"⚠️ No outgoing path from {current_site}. Stopping traversal."
                )
                break

            for _, row in next_row.iterrows():
                connected_routes.append(row)

            if current_site not in connection_list:
                connection_list.append(current_site)

            next_site = next_row["far_end"].values[0]

            if next_site == end_hub:
                connection_list.append(end_hub)
                logger.info(
                    f"🏆 Completed path from {start_hub} to {end_hub} "
                    f"in {iteration_count + 1} iterations."
                )
                break

            if next_site in visited_sites and next_site != end_hub:
                logger.warning(
                    f"⚠️ Detected loop at site {next_site}. Stopping traversal."
                )
                visited_sites.add(current_site)
                break

            visited_sites.add(current_site)
            current_site = next_site
            iteration_count += 1

        logger.info(
            f"ℹ️ Connected {len(connected_routes)} segments "
            f"in {iteration_count} iterations (multi-hub path)."
        )

    if not connected_routes:
        logger.warning(
            f"⚠️ No connected routes for FO hub in cluster {cluster}. "
            f"Returning original paths/points."
        )
        return final_paths, final_points

    connected_routes_df = pd.DataFrame(connected_routes)
    connected_routes_gdf = gpd.GeoDataFrame(
        connected_routes_df, geometry="geometry", crs="EPSG:3857"
    ).reset_index(drop=True)

    connected_sites = []
    for site_id in connection_list:
        site_row = final_points[final_points["site_id"] == site_id]
        if not site_row.empty:
            connected_sites.append(site_row.iloc[0])

    connected_sites_gdf = gpd.GeoDataFrame(
        connected_sites, geometry="geometry", crs="EPSG:3857"
    ).reset_index(drop=True)
    connected_sites_gdf = connected_sites_gdf.drop_duplicates(subset=["site_id", "geometry"])

    if connected_sites_gdf.empty:
        logger.warning(
            f"⚠️ Connected site list is empty for cluster {cluster}. "
            f"Skipping parquet export."
        )
        return None, None

    total_length = connected_routes_gdf["length"].sum()
    logger.info(
        f"ℹ️ Total final ring length | Area={area} | Cluster={cluster} "
        f"| {total_length / 1000:,.2f} km"
    )

    connected_routes_gdf = connected_routes_gdf.to_crs(epsg=4326)
    connected_sites_gdf = connected_sites_gdf.to_crs(epsg=4326)

    connected_routes_gdf.to_parquet(os.path.join(export_dir, f"Paths_{area}_{cluster}.parquet"))
    connected_sites_gdf.to_parquet(os.path.join(export_dir, f"Points_{area}_{cluster}.parquet"))
    logger.info(
        f"🏆 Cluster {cluster} paths/points saved to {export_dir}."
    )
    return connected_routes_gdf, connected_sites_gdf


def ring_parallel(
    clustered_sites: gpd.GeoDataFrame,
    area: str,
    area_col="region",
    cluster_col="ring_name",
    export_dir=None,
    ref_fo=None,
    cable_cost=35000,
    task_celery=False,
):
    """Process multiple clusters in parallel and merge results."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if clustered_sites.empty:
        logger.warning("⚠️ clustered_sites is empty. Nothing to process.")
        return None, None

    if export_dir is None:
        raise ValueError("Export directory is not provided. Please specify a valid export directory.")

    os.makedirs(export_dir, exist_ok=True)
    clusters = sorted(clustered_sites[cluster_col].unique().tolist())
    logger.info(
        f"ℹ️ Area={area} | Total clusters={len(clusters):,} to process in parallel."
    )

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
        logger.warning("⚠️ No valid clusters for parallel processing. Skipping.")
        return None, None

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(ring_cluster, args): args["ring_name"] for args in cluster_args}
        final_paths = []
        final_points = []
        total_futures = len(futures)

        if task_celery and total_futures > 1:
            task_celery.update_state(
                state="PROGRESS",
                meta={"status": f"Processing {total_futures} clusters in parallel"},
            )

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing clusters"):
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
                            "status": (
                                f"Completed {len(final_paths)}/{total_futures} clusters "
                                f"in area {area}"
                            )
                        },
                    )
            except Exception as e:
                logger.error(f"❌ Error processing cluster {cluster_id}: {e}")
                if task_celery:
                    task_celery.update_state(
                        state="PROGRESS",
                        meta={
                            "status": (
                                f"Error in cluster {cluster_id}: {e}. "
                                f"Completed {len(final_paths)}/{total_futures} clusters"
                            )
                        },
                    )
                continue

        if final_paths:
            final_paths = pd.concat(final_paths, ignore_index=True).drop_duplicates(subset=["geometry"])
            logger.info(
                f"ℹ️ Total merged paths after parallel processing: {len(final_paths):,}"
            )
        if final_points:
            final_points = pd.concat(final_points, ignore_index=True).drop_duplicates(subset=["geometry"])
            logger.info(
                f"ℹ️ Total merged points after parallel processing: {len(final_points):,}"
            )
    return final_paths, final_points


# ------------------------------------------------------
# 4) MODE: SUPERVISED RING NETWORK
# ------------------------------------------------------
def supervised_validation(excel_file: str | pd.DataFrame | gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger.info("🧩 Validating supervised input data.")
    if isinstance(excel_file, pd.DataFrame):
        df = excel_file
        logger.info("ℹ️ Input type: DataFrame.")
    elif isinstance(excel_file, str):
        logger.info(f"ℹ️ Input type: Excel path. Reading file: {excel_file}")
        df = pd.read_excel(excel_file)
    elif isinstance(excel_file, gpd.GeoDataFrame):
        logger.info("ℹ️ Input type: GeoDataFrame.")
        if 'lat' not in excel_file.columns or 'long' not in excel_file.columns:
            excel_file['lat'] = excel_file.geometry.to_crs(epsg=4326).y
            excel_file['long'] = excel_file.geometry.to_crs(epsg=4326).x
        df = pd.DataFrame(excel_file.drop(columns='geometry'))
    else:
        raise TypeError(
            "Unsupported supervised input type. "
            "Use str (Excel path), DataFrame, or GeoDataFrame."
        )

    # CHECKING USED COLUMNS
    required_columns = ["site_id", "site_name", "site_type", "lat", "long", "ring_name", "flag"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns in sheets: {missing_cols}.\n"
            f"Required columns: {', '.join(required_columns)}."
        )

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
        logger.info("ℹ️ 'region' column not found. Auto-grouping regions by distance.")
        group = auto_group(gdf, distance=10000)
        grouped_ring = gpd.sjoin(
            gdf[['geometry', 'ring_name']],
            group[['geometry', 'region']]
        ).drop(columns='index_right')
        grouped_ring = grouped_ring.groupby('ring_name')['region'].first().to_dict()
        gdf['region'] = gdf['ring_name'].map(grouped_ring)

    # VALIDITY
    for idx, row in gdf.iterrows():
        if not row["geometry"].is_valid:
            raise ValueError(f"Invalid geometry at index {idx} with site_id '{row['site_id']}'.")

    gdf["geometry"] = gdf["geometry"].apply(remove_z)

    # SUMMARY
    logger.info(f"ℹ️ Total supervised records: {len(gdf):,}")
    logger.info("🏆 Supervised input validation passed.")
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
    logger.info(f"🧩 Running supervised ring design for area '{area}'.")
    cable_cost = kwargs.get("cable_cost", 35000)
    task_celery = kwargs.get("task_celery", False)

    os.makedirs(export_dir, exist_ok=True)
    try:
        if isinstance(sites_input, str):
            logger.info(f"ℹ️ Input is parquet path. Reading: {sites_input}")
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
            logger.warning(f"⚠️ No cluster found for area '{area}'. Skipping.")
            return None

        logger.info(f"ℹ️ Area={area} | Total records (all areas)={len(sites_input):,}")

        ref_fo = None
        if fo_expand is not None:
            logger.info("🧩 Preparing FO reference nodes (expand mode enabled).")
            hex_list = identify_hexagon(sites_input, type="convex")
            logger.info(
                f"ℹ️ Hex count for FO reference | Area={area}: {len(hex_list):,}"
            )
            roads = retrieve_roads(hex_list, type="roads")
            nodes = retrieve_roads(hex_list, type="nodes")
            logger.info(
                f"ℹ️ Roads for FO reference | Area={area}: {len(roads):,}"
            )
            logger.info(
                f"ℹ️ Nodes for FO reference | Area={area}: {len(nodes):,}"
            )

            if roads.empty or nodes.empty:
                logger.warning(
                    f"⚠️ No roads or nodes for area '{area}'. Skipping."
                )
                return None

            roads = roads.to_crs(epsg=3857)
            nodes = nodes.to_crs(epsg=3857)

            logger.info("🧩 Building FO reference node set...")
            ref_fo = set(nodes[nodes["ref_fo"] == 1]["node_id"])
            fo_expand = fo_expand.to_crs(3857)
            fo_expand["geometry"] = fo_expand.geometry.buffer(20)
            nodes_within = gpd.sjoin(nodes, fo_expand, how="inner", predicate="intersects")
            nodes_within = set(nodes_within["node_id"])

            if isinstance(nodes_within, (list, tuple, set, set.__class__)):
                ref_fo.update(nodes_within)
                logger.info(
                    f"ℹ️ Added {len(nodes_within)} expanded FO nodes "
                    f"(total reference nodes={len(ref_fo)})"
                )
            else:
                logger.warning("⚠️ expand_fo should be a list, tuple, or set.")
            logger.info("🏆 FO reference node set prepared.")

        if task_celery:
            task_celery.update_state(
                state="PROGRESS",
                meta={"status": f"Processing ring data for area {area}"},
            )

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
        logger.critical(f"❌ Error processing supervised ring for area '{area}': {e}")
        return None


# ------------------------------------------------------
# 5) EXPORT & COMPILATION
# ------------------------------------------------------
def export_kmz(main_kml, ring_data, point_data, folder):
    """Write topology, sites, hubs, and routes into KMZ structure."""
    ring_data = ring_data.copy()
    cluster_points = point_data.copy()
    logger.info(f"🧩 Exporting KMZ content for folder '{folder}'.")

    ring_data["start"] = ring_data["near_end"].apply(
        lambda x: cluster_points[cluster_points["site_id"] == x]["geometry"].values[0]
    )
    ring_data["end"] = ring_data["far_end"].apply(
        lambda x: cluster_points[cluster_points["site_id"] == x]["geometry"].values[0]
    )

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

    kml_updated = export_kml(
        ring_topology,
        main_kml,
        filename,
        subfolder=folder,
        name_col="connection",
        color="#FF00FF",
        size=2,
        popup=False,
    )
    kml_updated = export_kml(
        ring_hub,
        kml_updated,
        filename,
        subfolder=f"{folder}/FO Hub",
        name_col="Site ID",
        icon="http://maps.google.com/mapfiles/kml/paddle/A.png",
        size=0.8,
        popup=True,
    )
    kml_updated = export_kml(
        ring_sites,
        kml_updated,
        filename,
        subfolder=f"{folder}/Site List",
        name_col="Site ID",
        color="#FFFF00",
        size=0.8,
        popup=True,
    )
    kml_updated = export_kml(
        ring_route,
        kml_updated,
        filename,
        subfolder=f"{folder}/Route",
        name_col="route_name",
        color="#0000FF",
        size=3,
        popup=False,
    )
    logger.info(f"🏆 KMZ content for '{folder}' added to main KML object.")
    return kml_updated


def export_gpkg(gpkg_path, ring_data, point_data, cluster):
    """Write cluster topology, routes, sites, and hubs to GeoPackage."""
    logger.info(
        f"🧩 Exporting cluster '{cluster}' to GeoPackage: {gpkg_path}"
    )

    ring_data = ring_data.copy()
    cluster_points = point_data.copy()

    ring_data["start"] = ring_data["near_end"].apply(
        lambda x: cluster_points[cluster_points["site_id"] == x]["geometry"].values[0]
    )
    ring_data["end"] = ring_data["far_end"].apply(
        lambda x: cluster_points[cluster_points["site_id"] == x]["geometry"].values[0]
    )
    ring_data = ring_data.reset_index(drop=True)

    topology = create_topology(point_data)
    topology.to_file(gpkg_path, layer=f"{cluster}_Topology", driver="GPKG")

    ring_route = ring_data[["near_end", "far_end", "ring_name", "route_type", "length", "geometry"]].copy()
    ring_route["route_name"] = ring_route["near_end"] + "-" + ring_route["far_end"]
    ring_route.to_file(gpkg_path, layer=f"{cluster}_Route", driver="GPKG")

    ring_sites = cluster_points[["site_id", "site_name", "site_type", "geometry"]].copy()
    ring_sites = ring_sites[~ring_sites["site_type"].str.lower().str.contains("hub")].copy()
    if ring_sites.empty:
        logger.warning("⚠️ No sites found. Skipping 'Site List' export.")
        return gpkg_path
    ring_sites.to_file(gpkg_path, layer=f"{cluster}_Site List", driver="GPKG")

    ring_hub = cluster_points[cluster_points["site_type"].str.lower().str.contains("hub")].copy()
    if ring_hub.empty:
        logger.warning("⚠️ No FO hub found. Skipping FO hub export.")
        return gpkg_path

    ring_hub.to_file(gpkg_path, layer=f"{cluster}_FO Hub", driver="GPKG")
    logger.info(f"🏆 GeoPackage export for cluster '{cluster}' completed.")
    return gpkg_path


def compile_ring(parquet_dir):
    """Compile all per-cluster parquet files into combined paths/points."""
    logger.info(f"🧩 Compiling per-cluster parquet files from {parquet_dir}")
    list_files = os.listdir(parquet_dir)
    point_files = [f for f in list_files if f.endswith(".parquet") and f.startswith("Points")]
    path_files = [f for f in list_files if f.endswith(".parquet") and f.startswith("Paths")]

    points_data = []
    for file in tqdm(point_files, desc="Loading point files"):
        file_path = os.path.join(parquet_dir, file)
        gdf = gpd.read_parquet(file_path)
        if not gdf.empty:
            points_data.append(gdf)

    paths_data = []
    for file in tqdm(path_files, desc="Loading path files"):
        file_path = os.path.join(parquet_dir, file)
        gdf = gpd.read_parquet(file_path)
        if not gdf.empty:
            paths_data.append(gdf)

    logger.info(f"ℹ️ Point parquet files: {len(point_files):,}")
    logger.info(f"ℹ️ Path parquet files : {len(path_files):,}")

    points_data = pd.concat(points_data, ignore_index=True) if points_data else pd.DataFrame()
    paths_data = pd.concat(paths_data, ignore_index=True) if paths_data else pd.DataFrame()

    logger.info(f"ℹ️ Total point rows: {len(points_data):,}")
    logger.info(f"ℹ️ Total path rows : {len(paths_data):,}")

    result_cluster = points_data["ring_name"].unique().tolist() if not points_data.empty else []
    logger.info(f"ℹ️ Total cluster IDs: {len(result_cluster):,}")

    if not points_data.empty:
        points_data.to_crs(epsg=4326).to_parquet(
            os.path.join(parquet_dir, "All_Points.parquet"),
            index=False,
        )
    if not paths_data.empty:
        paths_data.to_crs(epsg=4326).to_parquet(
            os.path.join(parquet_dir, "All_Paths.parquet"),
            index=False,
        )
    logger.info(f"🏆 Compiled parquet files saved to {parquet_dir}")

    for file in point_files + path_files:
        try:
            os.remove(os.path.join(parquet_dir, file))
        except Exception as e:
            logger.error(f"❌ Error removing file {file}: {e}")
    return points_data, paths_data


def connect_nearfar(points_gdf: gpd.GeoDataFrame, paths_gdf: gpd.GeoDataFrame):
    """Attach near_end and far_end site ids to points based on per-ring routes."""
    logger.info("🧩 Attaching near_end / far_end relationships to points.")
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
    logger.info("🏆 Near/far relationships attached.")
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
    logger.info(f"🧩 Exporting KML/KMZ to {kmz_path}")

    points = points.to_crs(epsg=4326).reset_index(drop=True)
    paths = paths.to_crs(epsg=4326).reset_index(drop=True)
    topology = topology.to_crs(epsg=4326).reset_index(drop=True)

    main_kml = simplekml.Kml()
    region_list = points['region'].dropna().unique().tolist()
    for region in region_list:
        logger.info(f"ℹ️ Exporting KML for region {region}")
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
        for ring in tqdm(ring_list, desc=f"Processing rings in {region}"):
            ring_points = region_points[region_points['ring_name'] == ring].copy()
            ring_paths = region_paths[region_paths['ring_name'] == ring].copy()
            topology_ring = topology_region[topology_region['ring_name'] == ring].copy()
            topology_ring = topology_ring.dissolve(by='ring_name').reset_index()
            topology_ring['geometry'] = topology_ring.geometry.apply(
                lambda geom: linemerge(geom) if type(geom) == MultiLineString else geom
            )
            topology_ring = topology_ring[['name', 'ring_name', 'region', 'geometry']]

            # FO HUB & SITELIST
            fo_hub = ring_points[ring_points['site_type'] == 'FO Hub'].copy().reset_index(drop=True)
            site_list = ring_points[ring_points['site_type'] != 'FO Hub'].copy().reset_index(drop=True)
            fo_hub = fo_hub[available_col]
            site_list = site_list[available_col]
            fo_hub = fo_hub.rename(columns=used_columns)
            site_list = site_list.rename(columns=used_columns)

            main_kml = export_kml(
                topology_ring,
                main_kml,
                folder_name=f"{region}_{ring}_Topology",
                subfolder=f"{region}/{ring}",
                name_col='name',
                color="#FF00FF",
                size=3,
                popup=False,
            )
            main_kml = export_kml(
                ring_paths,
                main_kml,
                folder_name=f"{region}_{ring}_Route",
                subfolder=f"{region}/{ring}/Route",
                name_col='name',
                color="#000FFF",
                size=3,
                popup=False,
            )
            main_kml = export_kml(
                site_list,
                main_kml,
                folder_name=f"{region}_{ring}_Site List",
                subfolder=f"{region}/{ring}/Site List",
                name_col='Site ID',
                color="#FFFF00",
                size=0.8,
            )
            main_kml = export_kml(
                fo_hub,
                main_kml,
                folder_name=f"{region}_{ring}_FO_Hub",
                subfolder=f"{region}/{ring}/FO Hub",
                name_col='Site ID',
                icon='http://maps.google.com/mapfiles/kml/paddle/A.png',
                size=0.8,
            )

    sanitize_kml(main_kml)
    main_kml.savekmz(kmz_path)
    logger.info(f"🏆 KML/KMZ export completed at {kmz_path}")


def save_supervised(
    points: gpd.GeoDataFrame,
    paths: gpd.GeoDataFrame,
    topology: gpd.GeoDataFrame,
    export_dir: str,
    method: str = "Supervised"
):
    logger.info("🧩 Exporting supervised outputs (parquet, KML, Excel).")

    # EXPORT PARQUET
    if not points.empty:
        points.to_crs(epsg=4326).to_parquet(
            os.path.join(export_dir, f"Points.parquet"),
            index=False,
        )
        logger.info(f"🏆 Points parquet exported with {len(points):,} records.")
    if not paths.empty:
        paths.to_crs(epsg=4326).to_parquet(
            os.path.join(export_dir, f"Route.parquet"),
            index=False,
        )
        logger.info(f"🏆 Route parquet exported with {len(paths):,} records.")
    if not topology.empty:
        topology = topology.sort_values(by=['ring_name']).reset_index(drop=True)
        topology.to_crs(epsg=4326).to_parquet(
            os.path.join(export_dir, f"Topology.parquet"),
            index=False,
        )
        logger.info(f"🏆 Topology parquet exported with {len(topology):,} records.")

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
            logger.info(
                f"ℹ️ Excel sheet '{sheet_name}' written with {len(points_report):,} records."
            )
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
            logger.info(
                f"ℹ️ Excel sheet '{sheet_name}' written with {len(paths_report):,} records."
            )


# ------------------------------------------------------
# 6) MAIN ENTRYPOINT
# ------------------------------------------------------
def main_supervised(
    site_data: gpd.GeoDataFrame,
    export_loc: str = "./exports",
    area_col: str = "region",
    cluster_col="ring_name",
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

    logger.info(
        f"🧩 Starting supervised pipeline | Method={method} | Design={design_type}"
    )

    if "site_id" in site_data.columns:
        site_data["site_id"] = site_data["site_id"].astype(str)

    site_data = sanitize_header(site_data)
    site_data = supervised_validation(site_data)

    date_today = datetime.now().strftime("%Y%m%d")
    week = detect_week(date_today)
    export_dir = f"{export_loc}/Intersite Design/{method}"
    checkpoint_dir = os.path.join(export_dir, "Checkpoint")

    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if "index_right" in site_data.columns:
        site_data = site_data.drop(columns=["index_right"])
    area_list = sorted(site_data[area_col].unique().tolist())

    output_paths = os.path.join(checkpoint_dir, "All_Paths.parquet")
    output_point = os.path.join(checkpoint_dir, "All_Points.parquet")
    if os.path.exists(output_paths) and os.path.exists(output_point):
        logger.info("ℹ️ Checkpoint outputs already exist. Loading from disk.")
        all_paths = gpd.read_parquet(output_paths)
        all_points = gpd.read_parquet(output_point)
        logger.info("🏆 Checkpoint data loaded.")
    else:
        logger.info(f"🧩 Starting {method} intersite pipeline from scratch.")
        logger.info(f"ℹ️ Vendor  : {vendor}")
        logger.info(f"ℹ️ Program : {program}")
        logger.info(f"ℹ️ Design  : {design_type}")
        logger.info(f"ℹ️ Total sites: {len(site_data):,}")

        for area in tqdm(area_list, desc=f"Processing {area_col}"):
            site_area = site_data[site_data[area_col] == area].copy()
            if site_area.empty:
                logger.warning(
                    f"⚠️ No data for {area_col}='{area}'. Skipping."
                )
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
                logger.warning(
                    f"⚠️ No paths/points result for {area_col}='{area}'. Skipping."
                )
                continue

        logger.info("🧩 Merging per-area parquet files from checkpoint...")
        all_points, all_paths = compile_ring(checkpoint_dir)
        all_points = all_points.reset_index(drop=True)
        all_paths = all_paths.reset_index(drop=True)

    if all_points.empty or all_paths.empty:
        logger.critical("❌ No final paths/points found. Aborting export.")
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
        logger.info("🧩 Running BOQ calculation...")
        main_boq(all_points, all_paths, export_dir=export_dir)
    else:
        # TOPOLOGY CHECK
        logger.info("🧩 Creating topology for final export...")
        topology_paths = create_topology(all_points)
        save_supervised(all_points, all_paths, topology_paths, export_dir, method)

    logger.info("🏆 Supervised export completed.")
    logger.info(
        f"ℹ️ All files saved to: {export_dir}"
    )


if __name__ == "__main__":
    # excel_file = r"D:\JACOBS\PROJECT\TASK\NOVEMBER\Week 3\BoQ Intersite\Trial BoQ Intersite.xlsx"
    excel_file = r"D:\JACOBSPACE\TBIG Impact 2025\QCC Fiberisasi\Asessment\Q1AOP2025 All_Smart Routing Automatic Template.xlsx"
    export_dir = r"D:\JACOBS\PROJECT\TASK\NOVEMBER\Week 3\BoQ Intersite\Export\TaskForce"
    area_col = 'region'
    cluster_col = 'ring_name'
    path_type = 'classified'
    program = "Trial BOQ"
    vendor = "TBG"
    boq = True

    logger.info("🧩 Running supervised ring network as standalone script.")
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
    logger.info(f"🏆 Result files zipped at {zip_filepath}.")
