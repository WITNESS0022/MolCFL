import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

# Define similarity calculation function
def flatten(weight):
    return torch.cat([value.flatten() for value in weight.values()])

def cosine_similarity(weight1, weight2):
    flat_weight1 = flatten(weight1)
    flat_weight2 = flatten(weight2)

    dot_product = torch.dot(flat_weight1, flat_weight2)
    norm_weight1 = torch.norm(flat_weight1)
    norm_weight2 = torch.norm(flat_weight2)

    similarity = dot_product / (norm_weight1 * norm_weight2 + 1e-12)
    return similarity.item()

def compute_pairwise_similarities(weights_list):
    num_clients = len(weights_list)
    similarities = np.zeros((num_clients, num_clients))

    for i, weight1 in enumerate(weights_list):
        for j, weight2 in enumerate(weights_list):
            similarities[i, j] = cosine_similarity(weight1, weight2)

    return similarities


# Calculate the maximum norm and average norm
def compute_max_update_norm(weights_list):
    return np.max([torch.norm(flatten(weights)).item() for weights in weights_list])

def compute_mean_update_norm(weights_list):
    return torch.norm(torch.mean(torch.stack([flatten(weights) for weights in weights_list]), dim=0)).item()


# Cluster clients
def cluster_clients(S):
    clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(-S)

    c1 = np.argwhere(clustering.labels_ == 0).flatten()
    print("c1:", c1)
    c2 = np.argwhere(clustering.labels_ == 1).flatten()
    print("c2:", c2)

    return c1, c2
