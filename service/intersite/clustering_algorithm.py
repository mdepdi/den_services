# ======================================================
# MODULE : UNSUPERVISED RING NETWORK
# AUTHOR : Yakub Hariana
# ORG    : Tower Bersama Group
# DESC   : Routing automation (Unsupervised mode)
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
from ring_algorithm import main_supervised

# ------------------------------------------------------
# 1) CORE DISTANCE & ROUTING
# ------------------------------------------------------
def map_distance(cluster_sites, G=None, weight="weight"):
    """Compute pairwise shortest-path distances for all site pairs."""
    mapping = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
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
# 1) CLUSTERING
# ------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering, DBSCAN

def dm_cluster(G, gdf_point, node_col="nearest_node", cutoff=None, weight='length', workers=4):
    """
    Builds a Dijkstra-based distance matrix between site nodes.

    Parameters:
    - G: NetworkX graph (with edge weights)
    - gdf_point: GeoDataFrame with node references
    - node_col: Column name of nearest node ID
    - cutoff: Max routing distance (e.g. 25000 meters)
    - weight: Edge attribute to use (default = 'length')
    - workers: Thread pool size

    Returns:
    - distance_df: Pandas DataFrame of shape [n x n]
    """
    print("🚧 Building Dijkstra Distance Matrix...")

    nodes = gdf_point[node_col].tolist()
    num = len(nodes)
    node_index = {node: idx for idx, node in enumerate(nodes)}

    def calculate_distance(node):
        try:
            return nx.single_source_dijkstra_path_length(G, node, weight=weight, cutoff=cutoff)
        except Exception as e:
            print(f"⚠️ No path from node {node}: {e}")
            return {}

    mapping = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(calculate_distance, node): node for node in nodes}
        for future in tqdm(futures, desc="⏱️ Calculating Dijkstra paths"):
            node = futures[future]
            try:
                mapping[node] = future.result()
            except Exception as e:
                print(f"❌ Error for node {node}: {e}")
                mapping[node] = {}

    distance_matrix = np.full((num, num), 1e10, dtype=np.float64)
    for i, src in enumerate(nodes):
        distance_matrix[i, i] = 0.0
        for dst, dist in mapping.get(src, {}).items():
            if dst in node_index:
                j = node_index[dst]
                distance_matrix[i, j] = dist

    print("✅ Distance matrix built successfully.")
    return distance_matrix

def metrics_calculation(distance_matrix, labels):
    print("📊 Clustering Metrics Calculation")
    n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"Total Cluster: {n_cluster}")

    if n_noise:
        print(f"Total Noise: {n_noise}")
    
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Label {label:2}: {count} members")

    silhouette = metrics.silhouette_score(distance_matrix, labels, metric="precomputed")
    calinski_harabasz = metrics.calinski_harabasz_score(distance_matrix, labels)
    davies_bouldin = metrics.davies_bouldin_score(distance_matrix, labels)

    print(f"Silhouette Score        : {silhouette}")
    print(f"Calinski-Harabasz Index : {calinski_harabasz}")
    print(f"Davies-Bouldin Index    : {davies_bouldin}")
    print()

    return silhouette, calinski_harabasz, davies_bouldin

def dbscan_clustering(distance_matrix, max_distance, min_sample):
    print("🧩 DBSCAN Clustering")
    db = DBSCAN(eps=max_distance, min_samples=min_sample, metric="precomputed").fit(distance_matrix)
    labels = db.labels_
    # metrics_calculation(distance_matrix, labels)
    return labels

def agglomerative_clustering(
    distance_matrix,
    member_expectation=10,
    linkage="average",
    metric="precomputed",
    distance_threshold=None,
):
    print("🧩 Agglomerative Clustering")
    if distance_threshold is None:
        n_cluster = int(np.ceil(len(distance_matrix) / member_expectation))
        ac = AgglomerativeClustering(
            n_clusters=n_cluster,
            metric=metric,
            linkage=linkage,
            distance_threshold=None,
        )
        labels = ac.fit_predict(distance_matrix)
    else:
        ac = AgglomerativeClustering(
            n_clusters=None,
            metric=metric,
            linkage=linkage,
            distance_threshold=distance_threshold,
        )
        labels = ac.fit_predict(distance_matrix)

    return labels


def initial_clustering(distance_matrix, sites_gdf, max_distance=10000):
    print("🧩 Initial Clustering")
    
    # DBSCAN
    sites = sites_gdf.copy()
    sites = sites.to_crs(epsg=3857)

    db = DBSCAN(eps=max_distance, min_samples=2, metric="precomputed")
    db_labels = db.fit_predict(distance_matrix)

    sites.loc[:, "__db"] = db_labels
    clean_sites = sites.loc[sites['__db'] != -1].copy()
    scatter_sites = sites.loc[sites['__db'] == -1].copy()
    print(f"✅ Clean sites: {len(clean_sites):,} | ❌ Scatter sites: {len(scatter_sites):,}")

    # AGGLOMERATIVE
    clean_idx = clean_sites.index
    clean_dist = distance_matrix[np.ix_(clean_idx, clean_idx)]

    agglo = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='complete',
        distance_threshold=max_distance
    )
    ag_labels = agglo.fit_predict(clean_dist)
    clean_sites['ring_name'] = ag_labels
    
    return clean_sites


def spatial_reclustering(site_cluster, sub_dm = None, member_expectation=10, method='agglomerative'):
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from tqdm import tqdm
    import numpy as np

    # print(f"Reclustering {len(site_cluster):,} sites")
    if len(site_cluster) < 2:
        return site_cluster

    coords = np.column_stack((site_cluster.geometry.x, site_cluster.geometry.y))
    n_clusters = int(np.ceil(len(site_cluster) / member_expectation))

    if method == 'kmeans':
        print("Using KMeans for reclustering")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(coords)
    elif method == 'agglomerative':
        print("Using Agglomerative Clustering for reclustering")
        if isinstance(sub_dm, np.ndarray) and sub_dm.size > 0:
            ac = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
            labels = ac.fit_predict(sub_dm)
        else:
            ac = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='average')
            labels = ac.fit_predict(coords)

    return labels

def identify_nearest(sites, cluster_column='ring_name', member_expectation=10, tolerance=2):
    print("🧩 Identifying Nearest Sites")
    sites = sites.copy().reset_index(drop=True)
    sites_sindex = sites.sindex
    for idx, row in sites.iterrows():
        group = row[cluster_column]
        geom = row.geometry
        source_id = row['site_id']
        nearest = sites_sindex.nearest(geom, return_all=True, return_distance=True, exclusive=True)

        if not nearest:
            print(f"ℹ️ No nearest found for {source_id}")
            continue

        distances = nearest[1]
        nums = nearest[0]

        if len(nearest[0][1]) == 0:
            print(f"ℹ️ No nearest found for {source_id} after filtering")
            continue
        for near_num, distance in zip(nums, distances):
            nearest_idx = nearest[0][1][near_num]
            if nearest_idx == idx:
                print(f"ℹ️ Skipping self {geom}")
                continue
            
            nearest_row = sites.loc[nearest_idx]
            if distance > 10000:
                print(f"ℹ️ Nearest {nearest_row.geometry.iloc[0]} | Distance {distance:.2f}m is too far, skipping.")
                continue

            group_nearest = nearest_row[cluster_column].iloc[0]
            member_nearest = sites[sites[cluster_column] == group_nearest].shape[0]

            if group_nearest == -1 or member_nearest < 3:
                print(f"ℹ️ Nearest is noise, skipping.")
                continue

            if group_nearest != group and member_nearest < member_expectation + tolerance:
                print(f"ℹ️ Nearest {nearest_row['site_id'].iloc[0]} | Group {group_nearest} | Members {member_nearest} | Distance {distance:.2f}m")
                print(f"ℹ️ Moving {source_id} | From Cluster {group} to Cluster {group_nearest}")
                sites.at[idx, cluster_column] = group_nearest
                break
    print("🧩 Nearest Sites Identification Completed")
    return sites

def filter_distance(sites, cluster_column='ring_name', min_sample=3, max_distance=50000):
    print("🧩 Filtering Sites by Distance")
    sites = sites.copy().reset_index(drop=True)
    crs = sites.crs

    sites = sites.to_crs(epsg=3857)
    unique_clusters = sites[cluster_column].unique().tolist()
    for label in unique_clusters:
        site_cluster = sites[sites[cluster_column] == label].copy()
        if site_cluster.empty:
            continue
        
        total_site = len(site_cluster)
        for idx, row in site_cluster.iterrows():
            geom = row.geometry
            buffer = geom.buffer(max_distance)
            intersected = site_cluster[site_cluster.geometry.intersects(buffer)]
            
            if len(intersected) < min_sample:
                print(f"ℹ️ {row['site_id']} | Cluster {label} exceeds max distance ({max_distance}m), marking as noise.")
                sites.at[idx, cluster_column] = -1
        sites[cluster_column] = sites[cluster_column].astype(int)
    print("🧩 Filtering Sites by Distance Completed")
    return sites.to_crs(crs)

def clean_noise(sites, cluster_column='ring_name', min_sample=3):
    print("🧩 Cleaning Noise Sites")
    sites = sites.copy().reset_index(drop=True)
    unique_clusters = sites[cluster_column].unique().tolist()
    
    for label in unique_clusters:
        site_cluster = sites[sites[cluster_column] == label].copy()
        if site_cluster.empty:
            continue
        
        counts = site_cluster[cluster_column].value_counts()
        if counts[label] < min_sample:
            print(f"⚠️ Cluster {label} has {counts[label]} members less than {min_sample} minimum, marking as noise.")
            sites.loc[sites[cluster_column] == label, cluster_column] = -1
    sites = sites[sites[cluster_column] != -1].reset_index(drop=True)
    print("🧩 Cleaning Noise Sites Completed")
    return sites

def clean_existings(sites:gpd.GeoDataFrame, cluster_col='ring_name'):
    ring_list = sorted(sites[cluster_col].unique().tolist())
    dropped_ring = []
    for ring in ring_list:
        if ring == -1:
            print(f"⚠️ Dropping noise (-1)")
            dropped_ring.append(ring)
            continue
        
        data_ring = sites[sites[cluster_col] == ring].copy()
        is_contain_new =  data_ring["site_type"].str.lower().str.contains('new', case=False, na=False).any()
        
        if is_contain_new:
            print(f"🟢 Ring {ring} contain new site, accept.")
            continue
        else:
            # print(f"🔴 Ring {ring} all existing, drop.")
            dropped_ring.append(ring)
    
    sites_cleaned = sites[~sites[cluster_col].isin(dropped_ring)]
    sites_cleaned = sites_cleaned.reset_index(drop=True)
    return sites_cleaned

def sequential_cluster(sites:gpd.GeoDataFrame, cluster_col='ring_name'):
    print(f"🧩 Sequential Cluster")
    sites = sites[sites[cluster_col] != -1]
    sites = sites.sort_values(by=cluster_col)
    cluster_list = sites[cluster_col].dropna().unique().tolist()
    numbered_list = {cluster : num for num, cluster in enumerate(cluster_list, start=1)}
    sites[cluster_col] = sites[cluster_col].map(numbered_list)
    sites = sites.reset_index(drop=True)
    return sites

def connect_hub(sites:gpd.GeoDataFrame, G, member_expectation=10, member_tolerance=0, cutoff=25000):
    fo_hub = sites[sites['site_type'].str.lower().str.contains("hub")].copy()
    sitelist = sites[~sites['site_type'].str.lower().str.contains("hub")].copy()
    hub_node = set(fo_hub['nearest_node'])
    site_node = set(sitelist['nearest_node'])
    total_hub = len(hub_node)
    total_member = member_expectation + member_tolerance - total_hub
    dijkstra = nx.multi_source_dijkstra_path_length(G, hub_node, weight='length', cutoff=cutoff)

    if total_hub == 2:
        start_hub = fo_hub[fo_hub['flag'].str.lower().str.contains("start")].copy()
        start_node = start_hub['nearest_node'].values[0]
        end_hub = fo_hub[fo_hub['flag'].str.lower().str.contains("end")].copy()
        end_node = end_hub['nearest_node'].values[0]
        hub_distance = nx.shortest_path_length(G, source=start_node, target=end_node, weight='length')
        dijkstra_filtered = {node: distance for node, distance in dijkstra.items() if node in site_node and distance <= (hub_distance + 1000)}
    else:
        dijkstra_filtered = {node: distance for node, distance in dijkstra.items() if node in site_node}

    sorted_distance = dict(sorted(dijkstra_filtered.items(), key=lambda kv: kv[1]))
    connected_node = list(sorted_distance.keys())
    if len(connected_node) > total_member:
        print(f"ℹ️ Total member exceeds {len(connected_node)}, trim nearest {member_expectation} member.")
        connected_node = connected_node[:total_member]
    accepted_sitelist = sitelist[sitelist['nearest_node'].isin(connected_node)]
    cleaned_ring = pd.concat([fo_hub, accepted_sitelist])
    cleaned_ring = cleaned_ring.reset_index(drop=True)

    return cleaned_ring

def finding_hubs(ring, ring_data, hubs_region, G, member_expectation=10, member_tolerance=0, max_distance=10000, drop_existings=False, swap_centroid=False):
    print(f"🛟 Ring {ring:2}| Sites {len(ring_data):,} | Hubs Candidate Region: {len(hubs_region):,}")
    
    # CLEAN BY NEAREST
    ring_data = ring_data.to_crs(epsg=3857)
    hubs_region = hubs_region.to_crs(epsg=3857)
    hubs_region = hubs_region.drop_duplicates('geometry')
    hubs_region = gpd.sjoin_nearest(hubs_region, ring_data[['geometry']], max_distance=max_distance, distance_col='dist_near_site').drop(columns='index_right')
    hubs_region = hubs_region[hubs_region['dist_near_site'] <= max_distance]

    nodes_hubs = set(hubs_region['nearest_node'])
    nodes_sites = set(ring_data['nearest_node'])
    nearest_nodes = nx.multi_source_dijkstra_path_length(G, nodes_sites, weight='length', cutoff=max_distance*2)
    nearest_hubs = {node: dist for node, dist in nearest_nodes.items() if node in nodes_hubs}

    connected_ring = pd.DataFrame()
    if nearest_hubs:
        nearest_hubs = dict(sorted(nearest_hubs.items(), key=lambda item: item[1]))
        print(f"🟢 Ring {ring:2}: Found {len(nearest_hubs):,} nearest hub")
        accepted_hub = {hub: dist for hub, dist in nearest_hubs.items() if dist <= max_distance}
        match len(accepted_hub):
            case 0:
                print(f"🔴 Ring {ring:2}: No hubs within {max_distance / 1000} km")
            case 1:
                print(f"🟡 Ring {ring:2}: Only one hub within {max_distance / 1000} km")
                accepted_hub = dict(list(accepted_hub.items())[:1])
                # if len(ring_data) == 1:
                #     print(f"🔴 Dropped only have one hub and {len(ring_data)} ring data.")
                #     return connected_ring
            case 2:
                print(f"🟢 Ring {ring:2}: Two hubs within {max_distance / 1000} km")
                accepted_hub = dict(list(accepted_hub.items()))
            case n if n > 2:
                print(f"🟢 Ring {ring:2}: {n} hubs within {max_distance / 1000} km. Select 2 nearest hubs.")
                accepted_hub = dict(list(accepted_hub.items())[:2])

        if len(accepted_hub) > 0:
            hubs_info = hubs_region[hubs_region['nearest_node'].isin(accepted_hub.keys())][['site_id', 'site_name', 'site_type','long', 'lat', 'region', 'nearest_node', 'geometry']].copy()
            hubs_info = hubs_info.reset_index(drop=True)
            hubs_info['ring_name'] = ring
            hubs_info['site_type'] = 'FO Hub'
            hubs_info['distance'] = hubs_info['nearest_node'].map(accepted_hub).round(3)
            hubs_info = hubs_info.sort_values(by='distance').reset_index(drop=True)
            hubs_info['flag'] = ['start' if i == 0 else 'end' for i in range(len(hubs_info))]

            start_hub = hubs_info.iloc[[0]]
            end_hub = hubs_info.iloc[[-1]] if len(hubs_info) > 1 else hubs_info.iloc[[0]]

            concated_rings = pd.concat([start_hub, ring_data, end_hub], ignore_index=True)
            concated_rings = gpd.GeoDataFrame(concated_rings, geometry='geometry', crs="EPSG:3857")
            # print(f"🟢 Ring {ring:2}: Hubs identified and concatenated with sites")
            # connected_ring = connect_hub(concated_rings, G, member_expectation, member_tolerance, cutoff=50000)
            
            if drop_existings:
                is_contain_new =  connected_ring["site_type"].str.lower().str.contains('new', case=False, na=False).any()
                if is_contain_new:
                    print(f"🟢 New sites valid")
                else:
                    print(f"🔴 New sites trimmed, drop ring.")
                    connected_ring = pd.DataFrame()
                    return connected_ring

            print(f"🟢 Ring {ring:2}: Hubs connected with sites\n")
        else:
            print(f"🔴 Ring {ring:2}: No hubs identified\n")
    else:
        print(f"⚠️ Ring {ring:2}: No nearest hubs found\n")
        if swap_centroid:
            if len(ring_data)  > 1:
                print(f"🟠 Try centroid point")
                sites_union = ring_data['geometry'].union_all()
                centroid_point = sites_union.centroid

                sites_nearest = ring_data.copy()
                sites_nearest['dist_to_rep'] = sites_nearest['geometry'].distance(centroid_point)
                sites_nearest = sites_nearest.sort_values('dist_to_rep')
                hubs_candidate = sites_nearest.iloc[[0]].copy()

                # # STATION NAME
                # hub_station = ring_data['station_name'].mode()[0]
                # print(f"🟠 Try station_name {hub_station}")
                # hubs_candidate = ring_data[ring_data['site_id'] == hub_station].copy()

                hubs_candidate['site_type'] = 'FO Hub'
                hub_id = hubs_candidate['site_id'].iloc[0]
                ring_data = ring_data[~ring_data.index.isin(hubs_candidate.index)]
                connected_ring = pd.concat([hubs_candidate, ring_data])

    return connected_ring

def main_clustering(G, sites_gdf, member_expectation=10, tolerance=0, max_distance=10000, drop_existings=False):
    distance_matrix = dm_cluster(G, sites_gdf, cutoff=max_distance)
    clean_sites = initial_clustering(distance_matrix, sites_gdf, max_distance=max_distance)
    
    clean_idx = clean_sites.index
    distance_matrix = distance_matrix[np.ix_(clean_idx, clean_idx)]
    clean_sites = clean_sites.reset_index(drop=True)

    # CLUSTER EXCEEDED
    sites_iter = clean_sites.copy()
    col = "ring_name"

    iteration = 0
    while iteration < 5:
        iteration += 1

        unique, counts = np.unique(sites_iter[col], return_counts=True)
        print(f"\nIteration {iteration} | {col}")

        if not any(counts > member_expectation + tolerance):
            print(f"✅ All clusters within expectation ({member_expectation} ± {tolerance}).\n")
            break

        for label, count in zip(unique, counts):
            try:
                if label == -1:
                    continue

                if count > member_expectation + tolerance:
                    print(f"🔃 Reclustering. Label {label:2}: {count} members")
                    # print(f"⚠️ Cluster {label} exceeds expectation ({member_expectation} ± {tolerance}).")
                    site_cluster = sites_iter[sites_iter[col] == label].copy()
                    
                    sub_idx = site_cluster.index
                    sub_dm = distance_matrix[np.ix_(sub_idx, sub_idx)]
                    new_labels = spatial_reclustering(site_cluster, sub_dm=sub_dm, member_expectation=member_expectation, method='agglomerative')

                    sub_label, sub_counts = np.unique(new_labels, return_counts=True)
                    for sub_label, sub_count in zip(sub_label, sub_counts):
                        print(f"Sub-label {sub_label:2}: {sub_count} members")

                    max_labels = max(sites_iter[col]) if not sites_iter.empty else 0
                    sites_iter.loc[sites_iter[col] == label, col] = new_labels + max_labels + 10
                    sites_iter[col] = sites_iter[col].astype(int)
                    # print(f"ℹ️ Reclustering completed for cluster {label}.")
            except Exception as e:
                print(f"Error processing cluster {label}: {e}")
                continue
        
    # sites_iter = identify_nearest(sites_iter, cluster_column=col, member_expectation=member_expectation, tolerance=tolerance)
    sites_iter = filter_distance(sites_iter, cluster_column=col, min_sample=1, max_distance=max_distance)
    sites_iter = clean_noise(sites_iter, cluster_column=col, min_sample=1)

    if drop_existings:
        sites_iter = clean_existings(sites_iter, cluster_col=col)
    sites_clustered = sequential_cluster(sites_iter, cluster_col=col)
    sites_clustered['ring_name'] = sites_clustered['region'] + "_" + sites_clustered['ring_name'].astype(str)
    sites_clustered = sites_clustered.reset_index(drop=True)
    print("🧩 Main Clustering Process Completed.")

    return sites_clustered

def hubs_finder(sitelist:gpd.GeoDataFrame, hubs:gpd.GeoDataFrame, max_distance=10000, member_expectation=10, ring="NA"):
    hex_list = identify_hexagon(sitelist, buffer=10000)
    nodes = retrieve_roads(hex_list, type='nodes')
    roads = retrieve_roads(hex_list, type='roads')
    nodes = nodes.to_crs(epsg=3857)
    roads = roads.to_crs(epsg=3857)
    sitelist = sitelist.to_crs(epsg=3857)
    hubs = hubs.to_crs(epsg=3857)
    G = build_graph(roads, graph_type='route')

    # JOIN NODES
    if 'nearest_node' not in sitelist.columns:
        sitelist = gpd.sjoin_nearest(sitelist, nodes[['node_id', 'geometry']]).drop(columns='index_right')
        sitelist = sitelist.rename(columns={'node_id':'nearest_node'})
    if 'nearest_node' not in hubs.columns:
        hubs = gpd.sjoin_nearest(hubs, nodes[['node_id', 'geometry']]).drop(columns='index_right')
        hubs = hubs.rename(columns={'node_id':'nearest_node'})
    
    try:
        hubs['site_type'] = 'FO Hub'
        connected_gdf = finding_hubs(ring, sitelist, hubs, G, max_distance=max_distance, member_expectation=member_expectation, drop_existings=False, swap_centroid=False)
        connected_gdf = connected_gdf.drop_duplicates(subset="geometry").reset_index(drop=True)
        print(f"ℹ️ {ring} total connected sites {len(connected_gdf):,}")
    except Exception as e:
        raise ValueError(f"Error finding hubs: {e}")
    
    return connected_gdf

def parallel_finding_hubs(sites_gdf: gpd.GeoDataFrame, hubs_gdf: gpd.GeoDataFrame, roads:gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, max_distance: int = 10000, use_centroid:bool=False, workers: int = 4, **kwargs):    
    print(f"🧩 Clustering and Finding Nearest Hubs")
    member_expectation = kwargs.get('member_expectation', 10)
    member_tolerance = kwargs.get('member_tolerance', 0)
    drop_existings = kwargs.get('drop_existings', False)
    export_dir = kwargs.get('export_dir', 'Standalones')
    member_sitelist = member_expectation - 2

    nodes = nodes.to_crs(epsg=3857)
    roads = roads.to_crs(epsg=3857)

    hubs_gdf = hubs_gdf.to_crs(epsg=3857)
    sites_gdf = sites_gdf.to_crs(epsg=3857)

    # Create nodes and edges
    print(f"🕸️ Create Graph for Clustering")
    G = build_graph(roads, graph_type='route')

    # Node Sindex
    print(f"ℹ️ Assigning Sindex.")
    node_sindex = nodes.sindex
    hubs_gdf['nearest_node'] = hubs_gdf.geometry.apply(lambda geom: nodes.at[node_sindex.nearest(geom)[1][0], 'node_id'])
    sites_gdf['nearest_node'] = sites_gdf.geometry.apply(lambda geom: nodes.at[node_sindex.nearest(geom)[1][0], 'node_id'])

    hubs_gdf = hubs_gdf.drop_duplicates(subset=['geometry']).reset_index(drop=True)
    sites_gdf = sites_gdf.drop_duplicates(subset=['geometry']).reset_index(drop=True)
    sites_clustered = main_clustering(G, sites_gdf, member_expectation=member_sitelist, tolerance=member_tolerance, max_distance=max_distance, drop_existings=drop_existings)

    # PARALLEL RING NEAREST HUBS
    standalones = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        ringlist = sorted(sites_clustered['ring_name'].unique().tolist())
        print(f"ℹ️ Total Rings   : {len(ringlist):,} rings")

        futures = {}
        for ring in tqdm(ringlist, total=len(ringlist), desc=f'Assign to {workers} finder'):
            sitelist_ring = sites_clustered[sites_clustered['ring_name'] == ring].copy()
            hubs_candidate = gpd.sjoin_nearest(hubs_gdf, sitelist_ring[['geometry']], how='inner', max_distance=max_distance).drop(columns='index_right')

            if hubs_candidate.empty and len(sitelist_ring) < 2:
                print(f"🔴 {ring:15} No hubs and standalone cluster")
                standalones.append(sitelist_ring)
                continue

            if hubs_candidate.empty and use_centroid:
                try:
                    # CENTROID POINT
                    print(f"🔴 {ring:15} No hubs around cluster. Try centroid point")
                    sites_union = sitelist_ring['geometry'].union_all()
                    centroid_point = sites_union.centroid

                    sites_nearest = sitelist_ring.copy()
                    sites_nearest['dist_to_rep'] = sites_nearest['geometry'].distance(centroid_point)
                    sites_nearest = sites_nearest.sort_values('dist_to_rep')
                    hubs_candidate = sites_nearest.iloc[[0]].copy()

                    hubs_candidate['site_type'] = 'FO Hub'
                    hub_id = hubs_candidate['site_id'].iloc[0]
                    sitelist_ring = sitelist_ring[~sitelist_ring.index.isin(hubs_candidate.index)]
                    print(f"🟢 {ring:15} using {hub_id} as centroid point")
                except Exception as e:
                    print(f"🔴 {ring:15} Error in find hubs candidate. {e}")
            elif hubs_candidate.empty:
                print(f"🔴 {ring:15} No hubs around cluster. ")
                continue
            else:
                print(f"🟢 {ring:15} Hubs candidate: {len(hubs_candidate):,}")

            sitelist_ring = sitelist_ring.reset_index(drop=True)
            hubs_candidate = hubs_candidate.reset_index(drop=True)
            
            future = executor.submit(hubs_finder, sitelist_ring, hubs_candidate, max_distance=max_distance, member_expectation=member_expectation, ring=ring)
            futures[future] = ring

        if len(standalones) > 0:
            standalones = pd.concat(standalones)
            standalones.to_parquet(os.path.join(export_dir, f"Sites_Standalones.parquet"))
            standalones.drop(columns='geometry').to_excel(os.path.join(export_dir, f"Sites_Standalones.xlsx"), index=False)

        clustered_data = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Process Hubs Finder"):
            ring = futures[future]
            try:
                result = future.result()
                if not result.empty:
                    clustered_data.append(result)
            except Exception as e:
                print(f"🔴 Error in ring {ring}: {e}")

        if len(clustered_data) > 0 :
            print(f"ℹ️ Clustered Data Defined")
            clustered_data = pd.concat(clustered_data)

    clustered_data = gpd.GeoDataFrame(clustered_data, geometry='geometry', crs="EPSG:3857")
    clustered_data = clustered_data.reset_index(drop=True)
    print(f"✅ Hubs finding done.")

    return clustered_data

def unsupervised_clustering(
    sites_input,
    hubs_input,
    area,
    export_dir: str,
    member_expectation: int = 10,
    area_col: str = "region",
    max_distance: int = 10000,
    **kwargs,
):
    """Run unsupervised routing for a single area, saving parquet results."""
    print(f"🧩 Ring Network Unsupervised | {area}")
    drop_existings = kwargs.get("drop_existings", False)
    use_centroid = kwargs.get("use_centroid", False)
    task_celery = kwargs.get("task_celery", False)
    os.makedirs(export_dir, exist_ok=True)
    try:
        if isinstance(sites_input, str):
            sites_input = gpd.read_parquet(sites_input)
        else:
            sites_input = sites_input.copy().reset_index(drop=True)
        
        if sites_input.crs is None:
            sites_input.set_crs(epsg=4326, inplace=True)
        sites_input = sites_input.to_crs(epsg=3857)

        if isinstance(hubs_input, str):
            hubs_input = gpd.read_parquet(hubs_input)
        else:
            hubs_input = hubs_input.copy().reset_index(drop=True)

        if hubs_input.crs is None:
            hubs_input.set_crs(epsg=4326, inplace=True)
        hubs_input = hubs_input.to_crs(epsg=3857)

        # CLUSTERING
        hex_list = identify_hexagon(sites_input, buffer=max_distance, type="convex")
        print(f"ℹ️ Total Hexagons    | {area_col} {area}: {len(hex_list):,} hexagons")
        roads = retrieve_roads(hex_list, type="roads")
        nodes = retrieve_roads(hex_list, type="nodes")
        print(f"ℹ️ Total Roads       | {area_col} {area}: {len(roads):,} roads")
        print(f"ℹ️ Total Nodes       | {area_col} {area}: {len(nodes):,} nodes")
        roads = roads.to_crs(epsg=3857)
        nodes = nodes.to_crs(epsg=3857)

        if task_celery:
            task_celery.update_state(state="PROGRESS", meta={"status": "Process Clustering Algorithm"})

        sites_clustered = parallel_finding_hubs(
            sites_gdf=sites_input,
            hubs_gdf=hubs_input,
            roads=roads,
            nodes=nodes,
            max_distance=max_distance,
            member_expectation=member_expectation,
            drop_existings=drop_existings,
            use_centroid=use_centroid,
            export_dir = export_dir
            )
        return sites_clustered
    except Exception as e:
        print(f"❌❌ Error processing area {area}: {e} \n")
        return pd.DataFrame()


def unsupervised_validation(sitelist: str | pd.DataFrame | gpd.GeoDataFrame, hubs: str | pd.DataFrame | gpd.GeoDataFrame) -> tuple:
    print(f"ℹ️ Checking Unsupervised Input")
    
    # SITELIST
    if isinstance(sitelist, pd.DataFrame):
        sitelist_df = sitelist
    elif isinstance(sitelist, str):
        sitelist_df = pd.read_excel(sitelist)
        print(f"📥 Reading Excel file: {sitelist}")
    elif isinstance(sitelist, gpd.GeoDataFrame):
        if 'lat' not in sitelist.columns or 'long' not in sitelist.columns:
            sitelist['lat'] = sitelist.geometry.to_crs(epsg=4326).y
            sitelist['long'] = sitelist.geometry.to_crs(epsg=4326).x
        sitelist_df = pd.DataFrame(sitelist.drop(columns='geometry'))
        print(f"📥 Using provided GeoDataFrame.")

    # CHECKING USED COLUMNS
    required_columns = ["site_id", "site_name", "lat", "long"]
    missing_cols = [col for col in required_columns if col not in sitelist_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in sheets: {missing_cols}.\n Please ensure the input data contains {', '.join(required_columns)} columns.")
    
    # HUBS
    if isinstance(hubs, pd.DataFrame):
        hubs_df = hubs
    elif isinstance(hubs, str):
        hubs_df = pd.read_excel(hubs)
        print(f"📥 Reading Excel file: {hubs}")
    elif isinstance(hubs, gpd.GeoDataFrame):
        if 'lat' not in hubs.columns or 'long' not in hubs.columns:
            hubs['lat'] = hubs.geometry.to_crs(epsg=4326).y
            hubs['long'] = hubs.geometry.to_crs(epsg=4326).x
        hubs_df = pd.DataFrame(hubs.drop(columns='geometry'))
        print(f"📥 Using provided GeoDataFrame.")

    # CHECKING USED COLUMNS
    required_columns = ["site_id", "site_name", "lat", "long"]
    missing_cols = [col for col in required_columns if col not in hubs_df.columns]
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
    sitelist_df["site_id"] = sitelist_df["site_id"].apply(safe_stringify)
    sitelist_df["site_name"] = sitelist_df["site_name"].apply(safe_stringify)
    hubs_df["site_id"] = hubs_df["site_id"].apply(safe_stringify)
    hubs_df["site_name"] = hubs_df["site_name"].apply(safe_stringify)
    
    # GEOMETRY
    sitelist_df_geom = gpd.points_from_xy(sitelist_df["long"], sitelist_df["lat"], crs="EPSG:4326")
    sitelist_gdf = gpd.GeoDataFrame(sitelist_df, geometry=sitelist_df_geom)
    hubs_df_geom = gpd.points_from_xy(hubs_df["long"], hubs_df["lat"], crs="EPSG:4326")
    hubs_gdf = gpd.GeoDataFrame(hubs_df, geometry=hubs_df_geom)

    if "region" in sitelist_gdf.columns:
        sitelist_gdf["region"] = sitelist_gdf["region"].apply(safe_stringify)
    else:
        group = auto_group(sitelist_gdf, distance=10000)
        sitelist_gdf = gpd.sjoin(sitelist_gdf, group[['geometry', 'region']]).drop(columns='index_right')
        hubs_gdf = gpd.sjoin(hubs_gdf, group[['geometry', 'region']]).drop(columns='index_right')

    # VALIDITY
    for idx, row in sitelist_gdf.iterrows():
        if not row["geometry"].is_valid:
            raise ValueError(f"Invalid geometry at sitelist index {idx} with site_id '{row['site_id']}'.")
    sitelist_gdf["geometry"] = sitelist_gdf["geometry"].apply(remove_z)

    for idx, row in hubs_gdf.iterrows():
        if not row["geometry"].is_valid:
            raise ValueError(f"Invalid geometry at hubs index {idx} with site_id '{row['site_id']}'.")
    hubs_gdf["geometry"] = hubs_gdf["geometry"].apply(remove_z)

    # SUMMARY
    print(f"ℹ️ Total Sitelist Records: {len(sitelist_gdf):,}")
    print(f"ℹ️ Total Hubs Records    : {len(hubs_gdf):,}")
    print(f"✅ Input Data Validity Check Passed.")
    return sitelist_gdf, hubs_gdf

# ------------------------------------------------------
# 6) MAIN ENTRYPOINT
# ------------------------------------------------------
def main_unsupervised(
    site_data: gpd.GeoDataFrame,
    hubs_data: gpd.GeoDataFrame,
    member_expectation:int = 10,
    export_loc: str = "./exports",
    area_col: str = "region",
    cluster_col = "ring_name",
    max_distance: int = 10000,
    fo_expand: gpd.GeoDataFrame = None,
    boq: bool = False,
    **kwargs,
):
    """Main unsupervised pipeline: per-area processing, compile, export KMZ/Excel."""
    vendor = kwargs.get("vendor", "TBG")
    program = kwargs.get("program", "Fiberization")
    method = 'Unsupervised'

    if "site_id" in site_data.columns:
        site_data["site_id"] = site_data["site_id"].astype(str)

    site_data = sanitize_header(site_data)
    hubs_data = sanitize_header(hubs_data)
    site_data, hubs_data = unsupervised_validation(site_data, hubs_data)

    date_today = datetime.now().strftime("%Y%m%d")
    week = detect_week(date_today)
    export_dir = f"{export_loc}/Intersite Design/W{str(week)}_{date_today}/{method}"
    checkpoint_dir = os.path.join(export_dir, "Checkpoint")

    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if "index_right" in site_data.columns:
        site_data = site_data.drop(columns=["index_right"])
    area_list = sorted(site_data[area_col].unique().tolist())

    output_point = os.path.join(checkpoint_dir, "Clustered_Points.parquet")
    if os.path.exists(output_point):
        print("ℹ️ Output file already exists. Loading existing data...")
        clustered_points = gpd.read_parquet(output_point)
        print("✅ Loaded existing data successfully.")
    else:
        print(f"🔥 Starting Unsupervised Clustering")
        print(f"ℹ️ Total Hubs        : {len(hubs_data):,}")
        print(f"ℹ️ Total Sites        : {len(site_data):,}")

        clustered_points = []
        for area in tqdm(area_list, desc=f"Processing {area_col}"):
            site_area = site_data[site_data[area_col] == area].copy()
            if site_area.empty:
                print(f"⚠️ No sitelist data found for {area_col} {area}. Skipping...")
                continue
            
            area_path = os.path.join(checkpoint_dir, f"Clustered_Points_{area}.parquet")
            if os.path.exists(area_path):
                clustered_sites = gpd.read_parquet(area_path)
            else:
                clustered_sites = unsupervised_clustering(
                    site_area,
                    hubs_data,
                    area=area,
                    member_expectation=member_expectation,
                    area_col=area_col,
                    cluster_col=cluster_col,
                    max_distance=max_distance,
                    export_dir=checkpoint_dir,
                    fo_expand=fo_expand,
                    **kwargs,
                )
            if not clustered_sites.empty:
                clustered_points.append(clustered_sites)
                clustered_sites.to_parquet(area_path)
            else:
                print(f"🔴 There is no result clustered sites for area {area}")

        if len(clustered_points) > 0:
            clustered_points = pd.concat(clustered_points)
            clustered_points.to_parquet(output_point)
        else:
            print(f"🔴 There is no result clustered sites. Check input data")

    # ROUTING
    main_supervised(
        site_data = clustered_points,
        export_loc= export_loc,
        fo_expand= fo_expand,
        boq= boq,
        vendor=vendor,
        program = program,
        method = method,
    )


if __name__ == "__main__":
    excel_file = r"D:\Data Analytical\PROJECT\TASK\NOVEMBER\Week 3\Unsupervised Ring\Template_Unsupervised_FTTT TSEL 2.xlsx"
    export_dir = r"D:\Data Analytical\PROJECT\TASK\NOVEMBER\Week 3\Unsupervised Ring"
    area_col = 'region'
    cluster_col = 'ring_name'
    member_expectation = 10
    max_distance = 10000
    use_centroid = False
    drop_existings = False
    boq = False

    site_data = pd.read_excel(excel_file, sheet_name='sitelist')
    hubs_data = pd.read_excel(excel_file, sheet_name='hubs')

    site_data = sanitize_header(site_data)
    hubs_data = sanitize_header(hubs_data)
    site_data, hubs_data = unsupervised_validation(site_data, hubs_data)
    fo_expand = None

    date_today = datetime.now().strftime("%Y%m%d")
    export_loc = f"{export_dir}/{date_today}"
    os.makedirs(export_loc, exist_ok=True)

    result = main_unsupervised(
        site_data=site_data,
        hubs_data=hubs_data,
        member_expectation=member_expectation,
        export_loc=export_loc,
        area_col=area_col,
        cluster_col=cluster_col,
        max_distance=max_distance,
        fo_expand=fo_expand,
        use_centroid=use_centroid,
        drop_existings=drop_existings,
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
