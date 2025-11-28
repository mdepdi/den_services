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
from time import time
from datetime import datetime
from tqdm import tqdm
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import linemerge
from sklearn import metrics
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from itertools import permutations

sys.path.append(r"D:\JACOBS\SERVICE\API")

# ------------------------------------------------------
# LOGGER
# ------------------------------------------------------
from core.logger import create_logger
logger = create_logger(__file__)

# ------------------------------------------------------
# MODULE IMPORTS
# ------------------------------------------------------
from modules.h3_route import identify_hexagon, retrieve_roads, build_graph
from modules.utils import auto_group
from modules.table import sanitize_header, detect_week
from service.intersite.ring_algorithm import main_supervised

# ------------------------------------------------------
# 1) CLUSTERING CORE
# ------------------------------------------------------

def dm_cluster(G, gdf_point, node_col="nearest_node", cutoff=None, weight='length', workers=4):
    """Builds a Dijkstra-based distance matrix."""
    logger.info("🚧 Building Dijkstra Distance Matrix...")

    nodes = gdf_point[node_col].tolist()
    num = len(nodes)
    node_index = {node: idx for idx, node in enumerate(nodes)}

    def calculate_distance(node):
        try:
            return nx.single_source_dijkstra_path_length(
                G, node, weight=weight, cutoff=cutoff
            )
        except Exception as e:
            logger.warning(f"⚠️ No path from node {node}: {e}")
            return {}

    mapping = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(calculate_distance, node): node for node in nodes}
        for future in tqdm(futures, desc="⏱️ Calculating Dijkstra paths"):
            node = futures[future]
            try:
                mapping[node] = future.result()
            except Exception as e:
                logger.error(f"❌ Error for node {node}: {e}")
                mapping[node] = {}

    distance_matrix = np.full((num, num), 1e10, dtype=np.float64)
    for i, src in enumerate(nodes):
        distance_matrix[i, i] = 0.0
        for dst, dist in mapping.get(src, {}).items():
            if dst in node_index:
                j = node_index[dst]
                distance_matrix[i, j] = dist

    logger.info("🏆 Distance matrix built successfully.")
    return distance_matrix


def metrics_calculation(distance_matrix, labels):
    logger.info("📊 Calculating clustering metrics")

    n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    logger.info(f"ℹ️ Total clusters: {n_cluster}")

    if n_noise:
        logger.info(f"ℹ️ Noise points: {n_noise}")

    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        logger.info(f"ℹ️ Label {label}: {count} members")

    silhouette = metrics.silhouette_score(distance_matrix, labels, metric="precomputed")
    ch = metrics.calinski_harabasz_score(distance_matrix, labels)
    db = metrics.davies_bouldin_score(distance_matrix, labels)

    logger.info(f"📈 Silhouette Score        : {silhouette}")
    logger.info(f"📈 Calinski-Harabasz Index : {ch}")
    logger.info(f"📈 Davies-Bouldin Index    : {db}")
    return silhouette, ch, db


def dbscan_clustering(distance_matrix, max_distance, min_sample):
    logger.info("🧩 Running DBSCAN clustering")
    db = DBSCAN(eps=max_distance, min_samples=min_sample, metric="precomputed").fit(distance_matrix)
    return db.labels_


def agglomerative_clustering(
    distance_matrix,
    member_expectation=10,
    linkage="average",
    metric="precomputed",
    distance_threshold=None,
):
    logger.info("🧩 Running Agglomerative clustering")

    if distance_threshold is None:
        n_cluster = int(np.ceil(len(distance_matrix) / member_expectation))
        ac = AgglomerativeClustering(
            n_clusters=n_cluster,
            metric=metric,
            linkage=linkage,
        )
    else:
        ac = AgglomerativeClustering(
            n_clusters=None,
            metric=metric,
            linkage=linkage,
            distance_threshold=distance_threshold,
        )

    return ac.fit_predict(distance_matrix)


def initial_clustering(distance_matrix, sites_gdf, max_distance=10000):
    logger.info("🧩 Initial Clustering started.")

    sites = sites_gdf.copy().to_crs(epsg=3857)

    # DBSCAN
    db = DBSCAN(eps=max_distance, min_samples=2, metric="precomputed")
    db_labels = db.fit_predict(distance_matrix)

    sites["__db"] = db_labels
    clean_sites = sites[sites["__db"] != -1].copy()
    scatter = sites[sites["__db"] == -1].copy()

    logger.info(f"🏆 Clean sites: {len(clean_sites):,} | Scatter: {len(scatter):,}")

    # AGGLOMERATIVE ONLY ON CLEAN SITES
    clean_idx = clean_sites.index
    sub_dm = distance_matrix[np.ix_(clean_idx, clean_idx)]

    agg = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="complete",
        distance_threshold=max_distance,
    )
    clean_sites["ring_name"] = agg.fit_predict(sub_dm)

    return clean_sites

# ------------------------------------------------------
# 2) RECLUSTERING & CLEANING
# ------------------------------------------------------

def spatial_reclustering(site_cluster, sub_dm=None, member_expectation=10, method='agglomerative'):
    from sklearn.cluster import KMeans, AgglomerativeClustering
    import numpy as np

    if len(site_cluster) < 2:
        return site_cluster

    logger.info("🧩 Reclustering oversized cluster")

    coords = np.column_stack((site_cluster.geometry.x, site_cluster.geometry.y))
    n_clusters = int(np.ceil(len(site_cluster) / member_expectation))

    if method == 'kmeans':
        logger.info("ℹ️ Using KMeans")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(coords)

    else:
        logger.info("ℹ️ Using Agglomerative Clustering")
        if isinstance(sub_dm, np.ndarray) and sub_dm.size > 0:
            ac = AgglomerativeClustering(
                n_clusters=n_clusters, metric='precomputed', linkage='complete'
            )
            labels = ac.fit_predict(sub_dm)
        else:
            ac = AgglomerativeClustering(
                n_clusters=n_clusters, metric='euclidean', linkage='average'
            )
            labels = ac.fit_predict(coords)

    return labels


def identify_nearest(sites, cluster_column='ring_name', member_expectation=10, tolerance=2):
    logger.info("🧩 Adjusting cluster using nearest neighbor logic")

    sites = sites.copy().reset_index(drop=True)
    sindex = sites.sindex

    for idx, row in sites.iterrows():
        geom = row.geometry
        source_id = row["site_id"]
        base_cluster = row[cluster_column]

        nearest = sindex.nearest(
            geom, return_all=True, return_distance=True, exclusive=True
        )
        if not nearest:
            logger.info(f"ℹ️ No nearest site for {source_id}")
            continue

        nums, dists = nearest
        for num, dist in zip(nums, dists):
            nearest_idx = nearest[0][1][num]
            if nearest_idx == idx:
                continue

            nearest_row = sites.loc[nearest_idx]
            near_cluster = nearest_row[cluster_column]

            # too far
            if dist > 10000:
                continue

            # skip noise
            cluster_count = sites[sites[cluster_column] == near_cluster].shape[0]
            if near_cluster == -1 or cluster_count < 3:
                continue

            if near_cluster != base_cluster and cluster_count < member_expectation + tolerance:
                logger.info(
                    f"ℹ️ Move {source_id} → cluster {near_cluster} | dist={dist:.2f}m"
                )
                sites.at[idx, cluster_column] = near_cluster
                break

    logger.info("🏆 Nearest site adjustment completed")
    return sites


def filter_distance(sites, cluster_column='ring_name', min_sample=3, max_distance=50000):
    logger.info("🧩 Filtering clusters by internal distance")

    sites = sites.copy().reset_index(drop=True)
    crs_original = sites.crs
    sites = sites.to_crs(epsg=3857)

    for lbl in sites[cluster_column].unique().tolist():
        cset = sites[sites[cluster_column] == lbl]
        if cset.empty:
            continue

        for idx, row in cset.iterrows():
            buff = row.geometry.buffer(max_distance)
            intr = cset[cset.geometry.intersects(buff)]

            if len(intr) < min_sample:
                logger.warning(
                    f"⚠️ Mark site {row['site_id']} as noise (distance > {max_distance}m)"
                )
                sites.at[idx, cluster_column] = -1

    sites[cluster_column] = sites[cluster_column].astype(int)
    logger.info("🏆 Distance filtering complete")
    return sites.to_crs(crs_original)


def clean_noise(sites, cluster_column='ring_name', min_sample=3):
    logger.info("🧩 Cleaning noise clusters")

    sites = sites.copy().reset_index(drop=True)
    for lbl in sites[cluster_column].unique().tolist():
        subset = sites[sites[cluster_column] == lbl]
        if subset.empty:
            continue

        if len(subset) < min_sample:
            logger.warning(f"⚠️ Cluster {lbl} too small → marking as noise")
            sites.loc[sites[cluster_column] == lbl, cluster_column] = -1

    cleaned = sites[sites[cluster_column] != -1].reset_index(drop=True)
    logger.info(f"🏆 Noise cleaned | remaining={len(cleaned):,}")
    return cleaned


def clean_existings(sites: gpd.GeoDataFrame, cluster_col='ring_name'):
    logger.info("🧩 Dropping purely existing-site rings")

    drop_list = []
    for ring in sorted(sites[cluster_col].unique().tolist()):
        if ring == -1:
            drop_list.append(ring)
            continue

        rset = sites[sites[cluster_col] == ring]
        has_new = rset["site_type"].str.contains("new", case=False, na=False).any()

        if not has_new:
            drop_list.append(ring)

    cleaned = sites[~sites[cluster_col].isin(drop_list)].reset_index(drop=True)
    logger.info(f"🏆 Dropped rings={len(drop_list)} | remaining rings={cleaned[cluster_col].nunique()}")
    return cleaned


def sequential_cluster(sites: gpd.GeoDataFrame, cluster_col='ring_name'):
    logger.info("🧩 Re-sequencing cluster numbering")

    sites = sites[sites[cluster_col] != -1].copy()
    sites = sites.sort_values(cluster_col)

    ordered = sites[cluster_col].unique().tolist()
    mapping = {old: i for i, old in enumerate(ordered, 1)}

    sites[cluster_col] = sites[cluster_col].map(mapping)
    sites = sites.reset_index(drop=True)

    logger.info("🏆 Cluster renumbering complete")
    return sites


# ------------------------------------------------------
# 3) CONNECT HUB
# ------------------------------------------------------

def connect_hub(sites: gpd.GeoDataFrame, G, member_expectation=10, member_tolerance=0, cutoff=25000):
    logger.info("🧩 Connecting ring sites to hubs")

    fo_hub = sites[sites['site_type'].str.lower().str.contains("hub")].copy()
    sitelist = sites[~sites['site_type'].str.lower().str.contains("hub")].copy()

    hub_node = set(fo_hub['nearest_node'])
    site_node = set(sitelist['nearest_node'])

    total_hub = len(hub_node)
    total_member = member_expectation + member_tolerance - total_hub

    dijkstra = nx.multi_source_dijkstra_path_length(
        G, hub_node, weight='length', cutoff=cutoff
    )

    if total_hub == 2:
        start_hub = fo_hub[fo_hub['flag'].str.lower().str.contains("start")].copy()
        end_hub = fo_hub[fo_hub['flag'].str.lower().str.contains("end")].copy()

        start_node = start_hub['nearest_node'].values[0]
        end_node = end_hub['nearest_node'].values[0]

        hub_distance = nx.shortest_path_length(
            G, source=start_node, target=end_node, weight='length'
        )

        dijkstra_filtered = {
            node: dist
            for node, dist in dijkstra.items()
            if node in site_node and dist <= (hub_distance + 1000)
        }
    else:
        dijkstra_filtered = {node: dist for node, dist in dijkstra.items() if node in site_node}

    sorted_dist = dict(sorted(dijkstra_filtered.items(), key=lambda kv: kv[1]))
    connected_node = list(sorted_dist.keys())

    if len(connected_node) > total_member:
        logger.info(f"ℹ️ Trimming sitelist to nearest {total_member} sites")
        connected_node = connected_node[:total_member]

    accepted = sitelist[sitelist['nearest_node'].isin(connected_node)]

    if accepted.empty:
        logger.warning("🟠 No members accepted for this hub connection.")
        return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=sites.crs)

    final_ring = pd.concat([fo_hub, accepted]).reset_index(drop=True)
    return final_ring

# ------------------------------------------------------
# 4) FINDING HUBS
# ------------------------------------------------------

def finding_hubs(ring, ring_data, hubs_region, G, member_expectation=10, member_tolerance=0,
                 max_distance=10000, drop_existings=False, swap_centroid=False):

    logger.info(f"🛟 Ring {ring} | Sites {len(ring_data):,} | Hub candidates: {len(hubs_region):,}")

    ring_data = ring_data.to_crs(epsg=3857)
    hubs_region = hubs_region.to_crs(epsg=3857)

    hubs_region = hubs_region.drop_duplicates('geometry')
    hubs_region = gpd.sjoin_nearest(
        hubs_region, ring_data[['geometry']], max_distance=max_distance,
        distance_col='dist_near_site'
    ).drop(columns='index_right')

    hubs_region = hubs_region[hubs_region['dist_near_site'] <= max_distance]

    nodes_hubs = set(hubs_region['nearest_node'])
    nodes_sites = set(ring_data['nearest_node'])

    nearest_nodes = nx.multi_source_dijkstra_path_length(
        G, nodes_sites, weight='length', cutoff=max_distance * 2
    )
    nearest_hubs = {node: dist for node, dist in nearest_nodes.items() if node in nodes_hubs}

    connected_ring = pd.DataFrame()

    if nearest_hubs:
        nearest_hubs = dict(sorted(nearest_hubs.items(), key=lambda x: x[1]))
        logger.info(f"🟢 Found {len(nearest_hubs):,} hub candidates")

        accepted_hub = {hub: dist for hub, dist in nearest_hubs.items() if dist <= max_distance}

        match len(accepted_hub):
            case 0:
                logger.warning(f"🔴 No hubs within range")
            case 1:
                logger.info(f"🟡 Only one hub nearby")
            case 2:
                logger.info(f"🟢 Two hubs within range")
            case n if n > 2:
                logger.info(f"🟢 {n} hubs within range, selecting 2 nearest")
                accepted_hub = dict(list(accepted_hub.items())[:2])

        if len(accepted_hub) > 0:
            hubs_info = hubs_region[
                hubs_region['nearest_node'].isin(accepted_hub.keys())
            ][['site_id','site_name','site_type','long','lat','region','nearest_node','geometry']].copy()

            hubs_info = hubs_info.reset_index(drop=True)
            hubs_info['ring_name'] = ring
            hubs_info['site_type'] = 'FO Hub'
            hubs_info['distance'] = hubs_info['nearest_node'].map(accepted_hub).round(3)
            hubs_info = hubs_info.sort_values(by='distance').reset_index(drop=True)
            hubs_info['flag'] = ['start' if i == 0 else 'end' for i in range(len(hubs_info))]

            start_hub = hubs_info.iloc[[0]]
            end_hub = hubs_info.iloc[[-1]] if len(hubs_info) > 1 else hubs_info.iloc[[0]]

            concated = pd.concat([start_hub, ring_data, end_hub], ignore_index=True)
            connected_ring = gpd.GeoDataFrame(concated, geometry='geometry', crs="EPSG:3857")

            if drop_existings:
                if not connected_ring["site_type"].str.contains('new', case=False).any():
                    logger.warning("🔴 Only existing sites found → dropping ring")
                    return pd.DataFrame()

            logger.info("🟢 Hubs assigned successfully\n")

        else:
            logger.warning("🔴 No accepted hubs\n")

    else:
        logger.warning("⚠️ No hub reachable within distance\n")
        if swap_centroid and len(ring_data) > 1:
            logger.info("🟠 Using centroid fallback")
            sites_union = ring_data['geometry'].union_all()
            centroid_point = sites_union.centroid

            sites_nearest = ring_data.copy()
            sites_nearest['dist_to_rep'] = sites_nearest['geometry'].distance(centroid_point)
            sites_nearest = sites_nearest.sort_values('dist_to_rep')

            hubs_candidate = sites_nearest.iloc[[0]].copy()
            hubs_candidate['site_type'] = 'FO Hub'

            ring_data = ring_data[~ring_data.index.isin(hubs_candidate.index)]
            connected_ring = pd.concat([hubs_candidate, ring_data])

    return connected_ring


# ------------------------------------------------------
# 5) HUBS FINDER WRAPPER
# ------------------------------------------------------
def main_clustering(
    G,
    sites_gdf,
    member_expectation: int = 10,
    tolerance: int = 0,
    max_distance: int = 10000,
    drop_existings: bool = False,
):
    """
    Main clustering routine:
    - Build Dijkstra-based distance matrix
    - Initial DBSCAN + Agglomerative clustering
    - Reclustering oversized clusters (max 5 iterations)
    - Distance filter + noise cleaning
    - Optional drop rings without NEW sites
    - Sequential renumbering + prefix with region
    """
    logger.info("🧩 Start main clustering pipeline")

    # 1) Distance matrix & initial clustering (DBSCAN + Agglo)
    distance_matrix = dm_cluster(G, sites_gdf, cutoff=max_distance)
    clean_sites = initial_clustering(distance_matrix, sites_gdf, max_distance=max_distance)

    # Restrict DM to clean sites only
    clean_idx = clean_sites.index
    distance_matrix = distance_matrix[np.ix_(clean_idx, clean_idx)]
    clean_sites = clean_sites.reset_index(drop=True)

    # 2) Handle oversized clusters iteratively
    sites_iter = clean_sites.copy()
    col = "ring_name"

    iteration = 0
    while iteration < 5:
        iteration += 1

        unique, counts = np.unique(sites_iter[col], return_counts=True)
        logger.info(f"\n🔁 Iteration {iteration} | column={col}")

        # Stop if all clusters within expectation
        if not any(counts > member_expectation + tolerance):
            logger.info(
                f"✅ All clusters within expectation "
                f"({member_expectation} ± {tolerance})."
            )
            break

        # Reclustering oversized clusters
        for label, count in zip(unique, counts):
            try:
                if label == -1:
                    continue

                if count > member_expectation + tolerance:
                    logger.info(
                        f"🔃 Reclustering label={label:2} "
                        f"| members={count}"
                    )

                    site_cluster = sites_iter[sites_iter[col] == label].copy()

                    sub_idx = site_cluster.index
                    sub_dm = distance_matrix[np.ix_(sub_idx, sub_idx)]

                    new_labels = spatial_reclustering(
                        site_cluster,
                        sub_dm=sub_dm,
                        member_expectation=member_expectation,
                        method="agglomerative",
                    )

                    sub_label, sub_counts = np.unique(new_labels, return_counts=True)
                    for s_lbl, s_cnt in zip(sub_label, sub_counts):
                        logger.info(
                            f"    ℹ️ Sub-label {s_lbl:2}: {s_cnt} members"
                        )

                    max_labels = max(sites_iter[col]) if not sites_iter.empty else 0
                    sites_iter.loc[sites_iter[col] == label, col] = (
                        new_labels + max_labels + 10
                    )
                    sites_iter[col] = sites_iter[col].astype(int)

            except Exception as e:
                logger.error(f"❌ Error processing cluster {label}: {e}")
                continue

    # 3) Distance-based noise + small-cluster cleaning
    # sites_iter = identify_nearest(sites_iter, cluster_column=col,
    #                               member_expectation=member_expectation,
    #                               tolerance=tolerance)
    sites_iter = filter_distance(sites_iter, cluster_column=col, min_sample=1)
    sites_iter = clean_noise(sites_iter, cluster_column=col, min_sample=1)

    # 4) Optionally drop rings without NEW sites
    if drop_existings:
        sites_iter = clean_existings(sites_iter, cluster_col=col)

    # 5) Sequential reindex + add region prefix
    sites_clustered = sequential_cluster(sites_iter, cluster_col=col)
    sites_clustered["ring_name"] = (
        sites_clustered["region"] + "_" + sites_clustered["ring_name"].astype(str)
    )
    sites_clustered = sites_clustered.reset_index(drop=True)

    logger.info("🧩 Main clustering process completed.")
    return sites_clustered

def hubs_finder(sitelist: gpd.GeoDataFrame, hubs: gpd.GeoDataFrame,
                max_distance=10000, member_expectation=10, ring="NA"):

    hex_list = identify_hexagon(sitelist, buffer=10000)
    nodes = retrieve_roads(hex_list, type='nodes').to_crs(epsg=3857)
    roads = retrieve_roads(hex_list, type='roads').to_crs(epsg=3857)

    sitelist = sitelist.to_crs(epsg=3857)
    hubs = hubs.to_crs(epsg=3857)

    G = build_graph(roads, graph_type='route')

    # join nodes
    if 'nearest_node' not in sitelist.columns:
        sidx = nodes.sindex
        sitelist['nearest_node'] = sitelist.geometry.apply(
            lambda geom: nodes.at[sidx.nearest(geom)[1][0], 'node_id']
        )

    if 'nearest_node' not in hubs.columns:
        sidx = nodes.sindex
        hubs['nearest_node'] = hubs.geometry.apply(
            lambda geom: nodes.at[sidx.nearest(geom)[1][0], 'node_id']
        )

    try:
        hubs['site_type'] = 'FO Hub'
        result = finding_hubs(
            ring, sitelist, hubs, G,
            max_distance=max_distance,
            member_expectation=member_expectation,
            drop_existings=False
        )
        result = result.drop_duplicates(subset="geometry").reset_index(drop=True)
        logger.info(f"ℹ️ {ring} total connected sites {len(result):,}")
        return result

    except Exception as e:
        raise ValueError(f"Error in hubs_finder: {e}")



# ------------------------------------------------------
# 6) PARALLEL HUB FINDING
# ------------------------------------------------------

def parallel_finding_hubs(
    sites_gdf: gpd.GeoDataFrame,
    hubs_gdf: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    max_distance: int = 10000,
    use_centroid: bool = False,
    workers: int = 4,
    **kwargs,
):

    logger.info("🧩 Clustering and Hub Assignment Pipeline")

    member_expectation = kwargs.get('member_expectation', 10)
    member_tolerance = kwargs.get('member_tolerance', 0)
    drop_existings = kwargs.get('drop_existings', False)
    export_dir = kwargs.get('export_dir', 'Standalones')

    member_sitelist = member_expectation - 2

    nodes = nodes.to_crs(epsg=3857)
    roads = roads.to_crs(epsg=3857)
    hubs_gdf = hubs_gdf.to_crs(epsg=3857)
    sites_gdf = sites_gdf.to_crs(epsg=3857)

    logger.info("🕸️ Building graph for clustering")
    G = build_graph(roads, graph_type='route')

    logger.info("ℹ️ Assigning nearest nodes using spatial index")
    sidx = nodes.sindex
    hubs_gdf['nearest_node'] = hubs_gdf.geometry.apply(
        lambda geom: nodes.at[sidx.nearest(geom)[1][0], 'node_id']
    )
    sites_gdf['nearest_node'] = sites_gdf.geometry.apply(
        lambda geom: nodes.at[sidx.nearest(geom)[1][0], 'node_id']
    )

    hubs_gdf = hubs_gdf.drop_duplicates('geometry').reset_index(drop=True)
    sites_gdf = sites_gdf.drop_duplicates('geometry').reset_index(drop=True)

    sites_clustered = main_clustering(
        G, sites_gdf,
        member_expectation=member_sitelist,
        tolerance=member_tolerance,
        max_distance=max_distance,
        drop_existings=drop_existings
    )

    # PARALLEL PROCESSING
    standalones = []
    clustered_results = []

    ringlist = sorted(sites_clustered['ring_name'].unique().tolist())
    logger.info(f"ℹ️ Total rings: {len(ringlist):,}")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}

        for ring in tqdm(ringlist, desc=f'Assign to {workers} workers'):
            sitelist_ring = sites_clustered[sites_clustered['ring_name'] == ring].copy()
            hubs_candidate = gpd.sjoin_nearest(
                hubs_gdf, sitelist_ring[['geometry']], how='inner',
                max_distance=max_distance
            ).drop(columns='index_right')

            if hubs_candidate.empty and len(sitelist_ring) < 2:
                logger.warning(f"🔴 {ring} has no hub and only 1 site → standalone")
                standalones.append(sitelist_ring)
                continue

            # centroid fallback
            if hubs_candidate.empty and use_centroid:
                logger.warning(f"🔴 {ring} no hubs found, using centroid fallback")
                sites_union = sitelist_ring['geometry'].union_all()
                centroid_point = sites_union.centroid

                nearest = sitelist_ring.copy()
                nearest['dist'] = nearest['geometry'].distance(centroid_point)
                nearest = nearest.sort_values('dist')
                hubs_candidate = nearest.iloc[[0]].copy()
                hubs_candidate['site_type'] = 'FO Hub'
                sitelist_ring = sitelist_ring[~sitelist_ring.index.isin(hubs_candidate.index)]

            elif hubs_candidate.empty:
                logger.warning(f"🔴 {ring} no hubs found")
                continue

            future = executor.submit(
                hubs_finder,
                sitelist_ring, hubs_candidate,
                max_distance=max_distance,
                member_expectation=member_expectation,
                ring=ring
            )
            futures[future] = ring

        if len(standalones) > 0:
            standalones_df = pd.concat(standalones)
            standalones_df.to_parquet(os.path.join(export_dir, "Sites_Standalones.parquet"))
            standalones_df.drop(columns='geometry').to_excel(
                os.path.join(export_dir, "Sites_Standalones.xlsx"),
                index=False
            )

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing hub finder"):
            ring = futures[future]
            try:
                result = future.result()
                if not result.empty:
                    clustered_results.append(result)
            except Exception as e:
                logger.error(f"🔴 Error processing ring {ring}: {e}")

    if len(clustered_results) > 0:
        clustered_results = pd.concat(clustered_results)

    clustered_results = gpd.GeoDataFrame(clustered_results, geometry='geometry', crs="EPSG:3857")
    clustered_results = clustered_results.reset_index(drop=True)

    logger.info("🏆 Hub assignment completed")
    return clustered_results


# ------------------------------------------------------
# 7) UNSUPERVISED CLUSTERING EXECUTION
# ------------------------------------------------------

def unsupervised_clustering(
    sites_input,
    hubs_input,
    area,
    export_dir,
    member_expectation=10,
    area_col="region",
    max_distance=10000,
    **kwargs,
):

    logger.info(f"🧩 Unsupervised Ring Network | Area {area}")
    drop_existings = kwargs.get("drop_existings", False)
    use_centroid = kwargs.get("use_centroid", False)
    task_celery = kwargs.get("task_celery", False)

    os.makedirs(export_dir, exist_ok=True)

    try:
        # prepare input
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

        # clustering
        hex_list = identify_hexagon(sites_input, buffer=max_distance, type="convex")
        logger.info(f"ℹ️ Hexagons: {len(hex_list):,}")

        roads = retrieve_roads(hex_list, type="roads").to_crs(epsg=3857)
        nodes = retrieve_roads(hex_list, type="nodes").to_crs(epsg=3857)

        logger.info(f"ℹ️ Roads: {len(roads):,} | Nodes: {len(nodes):,}")

        if task_celery:
            task_celery.update_state(state="PROGRESS", meta={"status": "Clustering"})

        clustered = parallel_finding_hubs(
            sites_gdf=sites_input,
            hubs_gdf=hubs_input,
            roads=roads,
            nodes=nodes,
            max_distance=max_distance,
            member_expectation=member_expectation,
            drop_existings=drop_existings,
            use_centroid=use_centroid,
            export_dir=export_dir
        )

        return clustered

    except Exception as e:
        logger.error(f"❌ Error processing area {area}: {e}")
        return pd.DataFrame()


# ------------------------------------------------------
# 8) VALIDATION
# ------------------------------------------------------

def unsupervised_validation(sitelist, hubs):
    logger.info("ℹ️ Validating unsupervised input")

    # SITELIST
    if isinstance(sitelist, pd.DataFrame):
        sitelist_df = sitelist
    elif isinstance(sitelist, str):
        sitelist_df = pd.read_excel(sitelist)
        logger.info(f"📥 Read sitelist: {sitelist}")
    elif isinstance(sitelist, gpd.GeoDataFrame):
        if 'lat' not in sitelist.columns or 'long' not in sitelist.columns:
            sitelist['lat'] = sitelist.geometry.to_crs(epsg=4326).y
            sitelist['long'] = sitelist.geometry.to_crs(epsg=4326).x
        sitelist_df = pd.DataFrame(sitelist.drop(columns='geometry'))

    required = ["site_id", "site_name", "lat", "long"]
    missing = [c for c in required if c not in sitelist_df.columns]
    if missing:
        raise ValueError(f"Missing sitelist columns: {missing}")

    # HUBS
    if isinstance(hubs, pd.DataFrame):
        hubs_df = hubs
    elif isinstance(hubs, str):
        hubs_df = pd.read_excel(hubs)
        logger.info(f"📥 Read hubs: {hubs}")
    elif isinstance(hubs, gpd.GeoDataFrame):
        if 'lat' not in hubs.columns or 'long' not in hubs.columns:
            hubs['lat'] = hubs.geometry.to_crs(epsg=4326).y
            hubs['long'] = hubs.geometry.to_crs(epsg=4326).x
        hubs_df = pd.DataFrame(hubs.drop(columns='geometry'))

    missing = [c for c in required if c not in hubs_df.columns]
    if missing:
        raise ValueError(f"Missing hubs columns: {missing}")

    def safe_string(value):
        if pd.isna(value):
            return ''
        if hasattr(value, 'wkt'):
            return str(value.wkt)
        return str(value).strip()

    sitelist_df["site_id"] = sitelist_df["site_id"].apply(safe_string)
    sitelist_df["site_name"] = sitelist_df["site_name"].apply(safe_string)
    hubs_df["site_id"] = hubs_df["site_id"].apply(safe_string)
    hubs_df["site_name"] = hubs_df["site_name"].apply(safe_string)

    # GEOMETRY
    sitelist_geom = gpd.points_from_xy(sitelist_df["long"], sitelist_df["lat"], crs="EPSG:4326")
    sitelist_gdf = gpd.GeoDataFrame(sitelist_df, geometry=sitelist_geom)

    hubs_geom = gpd.points_from_xy(hubs_df["long"], hubs_df["lat"], crs="EPSG:4326")
    hubs_gdf = gpd.GeoDataFrame(hubs_df, geometry=hubs_geom)

    if "region" not in sitelist_gdf.columns:
        group = auto_group(sitelist_gdf, distance=10000)
        sitelist_gdf = gpd.sjoin(sitelist_gdf, group[['geometry','region']]).drop(columns='index_right')
        hubs_gdf = gpd.sjoin(hubs_gdf, group[['geometry','region']]).drop(columns='index_right')

    logger.info(f"ℹ️ Sitelist: {len(sitelist_gdf):,} | Hubs: {len(hubs_gdf):,}")
    logger.info("🏆 Validation passed")

    return sitelist_gdf, hubs_gdf


# ------------------------------------------------------
# 9) MAIN ENTRYPOINT
# ------------------------------------------------------

def main_unsupervised(
    site_data: gpd.GeoDataFrame,
    hubs_data: gpd.GeoDataFrame,
    member_expectation=10,
    export_loc="./exports",
    area_col="region",
    cluster_col="ring_name",
    max_distance=10000,
    fo_expand=None,
    boq=False,
    **kwargs
):

    vendor = kwargs.get("vendor", "TBG")
    program = kwargs.get("program", "Fiberization")
    method = "Unsupervised"

    logger.info("🌏 Starting Unsupervised Intersite")

    site_data = sanitize_header(site_data)
    hubs_data = sanitize_header(hubs_data)

    site_data, hubs_data = unsupervised_validation(site_data, hubs_data)

    date_today = datetime.now().strftime("%Y%m%d")
    week = detect_week(date_today)

    export_dir = f"{export_loc}/Intersite Design/{method}"
    checkpoint_dir = os.path.join(export_dir, "Checkpoint")

    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if "index_right" in site_data.columns:
        site_data = site_data.drop(columns=["index_right"])

    area_list = sorted(site_data[area_col].unique().tolist())

    output_point = os.path.join(checkpoint_dir, "Clustered_Points.parquet")

    if os.path.exists(output_point):
        logger.info("ℹ️ Loading cached clustered points...")
        clustered_points = gpd.read_parquet(output_point)
        logger.info("🏆 Loaded cached data")

    else:
        logger.info("🔥 Executing clustering for all areas")

        clustered_points = []

        for area in tqdm(area_list, desc=f"Processing {area_col}"):
            site_area = site_data[site_data[area_col] == area].copy()

            if site_area.empty:
                logger.warning(f"⚠️ No sitelist for {area}")
                continue

            area_path = os.path.join(checkpoint_dir, f"Clustered_Points_{area}.parquet")

            if os.path.exists(area_path):
                clustered_sites = gpd.read_parquet(area_path)
            else:
                clustered_sites = unsupervised_clustering(
                    site_area, hubs_data,
                    area=area,
                    member_expectation=member_expectation,
                    area_col=area_col,
                    cluster_col=cluster_col,
                    max_distance=max_distance,
                    export_dir=checkpoint_dir,
                    fo_expand=fo_expand,
                    **kwargs
                )

            if not clustered_sites.empty:
                clustered_points.append(clustered_sites)
                clustered_sites.to_parquet(area_path)
            else:
                logger.warning(f"🔴 No result for {area}")

        if len(clustered_points) > 0:
            clustered_points = pd.concat(clustered_points)
            clustered_points.to_parquet(output_point)
        else:
            logger.error("🔴 No clustering results found.")
            return

    # ROUTING ENGINE
    main_supervised(
        site_data=clustered_points,
        export_loc=export_loc,
        fo_expand=fo_expand,
        boq=boq,
        vendor=vendor,
        program=program,
        method=method,
        **kwargs
    )


# ------------------------------------------------------
# 10) DIRECT EXECUTION
# ------------------------------------------------------
if __name__ == "__main__":
    excel_file = r"D:\JACOBS\PROJECT\TASK\NOVEMBER\Week 4\Surge Sitelist\Surge_Sitelist & Hub 343_27112025.xlsx"
    export_dir = r"D:\JACOBS\PROJECT\TASK\NOVEMBER\Week 4\Surge Sitelist"
    area_col = 'region'
    cluster_col = 'ring_name'
    member_expectation = 8
    max_distance = 3000
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

    zip_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_Unsupervised_Task.zip"
    zip_filepath = os.path.join(export_loc, zip_filename)

    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(export_loc):
            for file in files:
                if file != zip_filename and not file.endswith(".zip"):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_loc)
                    zipf.write(file_path, arcname)

    logger.info("📦 Result files zipped.")

