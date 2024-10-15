import csv
import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

def load_data(filepath):
    with open(filepath, "r") as file:
        info = csv.reader(file)
        header = next(info)
        data = []
        for country in info:
            country_info = {}
            for i in range(len(header)):
                country_info[header[i]] = country[i]
            data.append(country_info)
    return data

def calc_features(row):
    x1 = float(row["Population"])
    x2 = float(row["Net migration"])
    x3 = float(row["GDP ($ per capita)"])
    x4 = float(row["Literacy (%)"])
    x5 = float(row["Phones (per 1000)"])
    x6 = float(row["Infant mortality (per 1000 births)"])
    features = np.array([x1 , x2, x3, x4, x5, x6], dtype=np.float64)
    return features


def hac(features):
    num_features = len(features)
    cluster_tree = [] 
    clusters = [(i, [i]) for i in range(num_features)]
    distances = distance_matrix(features)

    for n in range(len(distances) - 1):
        idx_i, idx_j, dist, cluster_i, cluster_j = calculate_cluster(clusters, distances)
        cluster_tree, clusters = merge(cluster_tree, features, clusters, n, idx_i, idx_j, dist, cluster_i, cluster_j)

    return np.array(cluster_tree).astype("float")


def calculate_cluster(clusters, distances):
    min_idx = -1
    max_idx = -1
    min_distance = np.inf
    cluster_i = []
    cluster_j = []

    for i in clusters:
        idx_i, cli_i = i
        for j in clusters:
            idx_j, cli_j = j
            if idx_i == idx_j:
                continue
            max_distance = -1
            for k in cli_i:
                for l in cli_j:
                    max_distance = max(max_distance, distances[k, l])
            if max_distance >= 0 and min_distance > max_distance:
                min_idx = idx_i
                max_idx = idx_j
                cluster_i = cli_i
                cluster_j = cli_j
                min_distance = max_distance

    return min_idx, max_idx, min_distance, cluster_i, cluster_j



def merge(Z, dataset, clusters, n, idx_i, idx_j, dist, clust_i, clust_j):
    num_samples = len(dataset)
    merged_cluster = list(clust_i) + list(clust_j)
    min_idx = min(idx_i, idx_j)
    max_idx = max(idx_i, idx_j)

    Z.append([min_idx, max_idx, dist, len(merged_cluster)])

    new_cluster = [num_samples + n, tuple(merged_cluster)]
    clusters = [cluster for cluster in clusters if cluster[0] not in (idx_i, idx_j)]
    clusters.append(tuple(new_cluster))

    return Z, clusters



def distance_matrix(dataset):
    num_samples = len(dataset)
    distances = np.zeros((num_samples, num_samples))
    
    for i in range(num_samples):
        for j in range(num_samples):
            distances[i][j] = np.linalg.norm(dataset[i] - dataset[j])
    
    return distances


def fig_hac(Z, names):
    fig = plt.figure(figsize=(6, 8))
    dn = hierarchy.dendrogram(Z, labels=names, leaf_rotation=90)
    plt.tight_layout()
    return fig


def normalize_features(features):
    input_data_array = np.array(features)
    data_means = np.mean(input_data_array, axis=0)
    data_stddevs = np.std(input_data_array, axis=0)
    standardized_data = (input_data_array - data_means) / data_stddevs
    standardized_data_list = [np.array(row) for row in standardized_data]
    return standardized_data_list


if __name__ == "__main__":
    data = load_data("countries.csv")
    country_names = [row["Country"] for row in data]
    features = [calc_features(row) for row in data]
    features_normalized = normalize_features(features)
    n = 10
    Z_raw = hac(features[:n])
    Z_normalized = hac(features_normalized[:n])
    fig = fig_hac(Z_raw, country_names[:n])
    plt.show()


