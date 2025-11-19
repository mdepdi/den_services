import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from core.config import settings

MAINDATA_DIR = settings.MAINDATA_DIR

# DISTANCE MAPPING
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

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

    # distance_df = pd.DataFrame(distance_matrix, index=nodes, columns=nodes)
    # distance_df[np.isinf(distance_df)] = 1e10
    # distance_df[np.isnan(distance_df)] = 1e10
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
    # metrics_calculation(distance_matrix, labels)

    return labels

def initial_clustering(distance_matrix, sites_gdf, member_expectation=10, max_distance=10000):
    print("🧩 Initial Clustering")
    
    # DBSCAN
    sites = sites_gdf.copy()
    sites = sites.to_crs(epsg=3857)
    # coords = sites.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()

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

def finding_hubs(ring, ring_data, hubs_region, G, member_expectation=10, member_tolerance=0, max_distance=10000, drop_existings=False):
    print(f"🛟 Ring {ring:2}: {len(ring_data):,} sites")
    # CLEAN BY NEAREST
    ring_data = ring_data.to_crs(epsg=3857)
    hubs_region = hubs_region.to_crs(epsg=3857)
    hubs_region = hubs_region.drop_duplicates('geometry')
    hubs_region = gpd.sjoin_nearest(hubs_region, ring_data[['geometry']], max_distance=max_distance, distance_col='dist_near_site').drop(columns='index_right')
    hubs_region = hubs_region[hubs_region['dist_near_site'] <= max_distance]

    nodes_hubs = set(hubs_region['nearest_node'])
    nodes_sites = set(ring_data['nearest_node'])
    nearest_nodes = nx.multi_source_dijkstra_path_length(G, nodes_sites, weight='length', cutoff=max_distance)
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
                if len(ring_data) == 1:
                    print(f"🔴 Dropped only have one hub and {len(ring_data)} ring data.")
                    return connected_ring
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
            connected_ring = connect_hub(concated_rings, G, member_expectation, member_tolerance)
            
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
    return connected_ring


def main_clustering(G, sites_gdf, member_expectation=10, tolerance=0, max_distance=10000, hubs_gdf=None, drop_existings=False):
    distance_matrix = dm_cluster(G, sites_gdf, cutoff=max_distance)
    clean_sites = initial_clustering(distance_matrix, sites_gdf, member_expectation, max_distance=max_distance)
    
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
    print("🧩 Main Clustering Process Completed.")

    if hubs_gdf is not None:
        sites_clustered = sites_clustered.reset_index(drop=True)
        sites_clustered['ring_name'] = sites_clustered['region'] + "_" + sites_clustered['ring_name'].astype(str)
        ringlist = sorted(sites_clustered['ring_name'].unique().tolist())
        print(f"ℹ️ Total Rings   : {len(ringlist):,} rings")

        # PARALLEL RING NEAREST HUBS
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {}
            for ring in ringlist:
                if "_-1" in ring:
                    print(f"⚠️ Ring {ring:2}: Noise, skipping...")
                    continue
                
                ring_data = sites_clustered[sites_clustered['ring_name'] == ring].copy()
                ring_data = ring_data.reset_index(drop=True)
                hubs_data = hubs_gdf.copy()

                ring_data = ring_data.to_crs(epsg=3857)
                hubs_data = hubs_data.to_crs(epsg=3857)

                future = executor.submit(finding_hubs, ring, ring_data, hubs_data, G, member_expectation, tolerance, max_distance=max_distance, drop_existings=drop_existings)
                futures[future] = ring

            identified_hubs = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Finding Hubs"):
                ring = futures[future]
                try:
                    concated_rings = future.result()
                    if not concated_rings.empty:
                        identified_hubs.append(concated_rings)
                except Exception as e:
                    print(f"❌ Error processing ring {ring}: {e}")
        identified_hubs = pd.concat(identified_hubs, ignore_index=True) if identified_hubs else gpd.GeoDataFrame()
        identified_hubs = identified_hubs.to_crs(epsg=4326).reset_index(drop=True) if not identified_hubs.empty else identified_hubs
        return identified_hubs
    else:
        return sites_clustered