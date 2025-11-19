import networkx as nx
import pandas as pd

def build_cluster_graph(sites_gdf, G, node_col='nearest_node'):
    cluster_graph = nx.Graph()
    for idx, row in sites_gdf.iterrows():
        for idx2, row2 in sites_gdf.iterrows():
            if row[node_col] != row2[node_col]:
                try:
                    shortest_path_length = nx.shortest_path_length(
                        G,
                        source=row[node_col],
                        target=row2[node_col],
                        weight="weight",
                    )
                    cluster_graph.add_edge(
                        row[node_col],
                        row2[node_col],
                        weight=shortest_path_length,
                    )
                except nx.NetworkXNoPath:
                    continue
            else:
                continue

    if not cluster_graph:
        print("⚠️ No connections found in the cluster graph. Please check the input data.")
        return None
    
    if cluster_graph.is_directed():
        cluster_graph = cluster_graph.to_undirected()

    return cluster_graph