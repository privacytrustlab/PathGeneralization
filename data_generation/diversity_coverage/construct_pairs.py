import random
from collections import defaultdict
import os
import pickle
import numpy as np
import sys
import logging
import networkx as nx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from utils import save_pickle, save_numpy, load_config, save_shard
from utils import get_indices, get_nodes, get_valid_neighbors

def generate_pairs(X, k, seed=None):
    """
    Efficient version to generate pairs with:
    - Each node appears k times as x_i
    - Each node appears k times as x_j
    - No duplicates, x_i ≠ x_j
    Returns a pair list and a dict mapping each node to its pairs.
    """
    if seed is not None:
        random.seed(seed)
    
    n = len(X)
    if k > n - 1:
        raise ValueError("k must be ≤ len(X) - 1 to avoid self-pairing")

    X = list(X)  # ensure it's indexable
    pair_set = set()
    node_pairs_start = defaultdict(set)
    node_pairs_end = defaultdict(set)
    
    x_i_counts = defaultdict(int)
    x_j_counts = defaultdict(int)

    # For each x as x_i, sample k distinct valid x_j
    for a in X:
        candidates = set(X) - {a}
        valid_bs = [b for b in candidates if x_j_counts[b] < k]
        # select k number of b from valid_bs
        k_b_list = random.sample(valid_bs, min(k, len(valid_bs)))
        for b in k_b_list:
            if (a, b) not in pair_set:
                pair_set.add((a, b))
                node_pairs_start[a].add((a, b))
                node_pairs_end[b].add((a, b))
                x_i_counts[a] += 1
                x_j_counts[b] += 1
                # remove b from valid_bs to ensure no duplicates
                valid_bs.remove(b)

        # Fallback: attempt to swap if quota is not satisfied
        if x_i_counts[a] < k:
            while len(valid_bs) > 0 and x_i_counts[a] < k:
                for b in valid_bs:
                    if x_i_counts[a] < k:
                        if (a, b) not in pair_set:
                            pair_set.add((a, b))
                            node_pairs_start[a].add((a, b))
                            node_pairs_end[b].add((a, b))
                            x_i_counts[a] += 1
                            x_j_counts[b] += 1
                            # remove b from valid_bs to ensure no duplicates
                        valid_bs.remove(b)
                    else:
                        break
            
            if x_i_counts[a] < k:
                gap = k - x_i_counts[a]
                remaining_bs = [b for b in set(X) if x_j_counts[b] < k]
                if len(remaining_bs) == 0:
                    raise ValueError(f"Not enough valid pairs for node {a} to reach k={k} pairs.")
                pair_list = list(pair_set)
                random.shuffle(pair_list)
                for c, d in pair_list:
                    if gap <= 0:
                        break
                    for b in remaining_bs:
                        if (
                            c != a and d != a and
                            (a, d) not in pair_set and (c, b) not in pair_set
                        ):
                            # Perform the swap
                            pair_set.remove((c, d))
                            node_pairs_start[c].remove((c, d))
                            node_pairs_end[d].remove((c, d))
                            x_i_counts[c] -= 1
                            x_j_counts[d] -= 1

                            # Add new pairs
                            pair_set.add((a, d))
                            node_pairs_start[a].add((a, d))
                            node_pairs_end[d].add((a, d))
                            x_i_counts[a] += 1
                            x_j_counts[d] += 1

                            pair_set.add((c, b))
                            node_pairs_start[c].add((c, b))
                            node_pairs_end[b].add((c, b))
                            x_i_counts[c] += 1
                            x_j_counts[b] += 1
                            
                            remaining_bs = [b for b in remaining_bs if x_j_counts[b] < k]
                            gap -= 1
                            break
                        else:
                            continue
                
                if x_i_counts[a] < k:
                    raise ValueError(f"Node {a} has not reached k={k} pairs after all attempts -current count: {x_i_counts[a]}")
                
    # Final check for completeness
    for node in X:
        if x_i_counts[node] != k:
            print(f"{node} has x_i count {x_i_counts[node]} - expected {k}")
        if x_j_counts[node] != k:
            print(f"{node} has x_j count {x_j_counts[node]} - expected {k}")

    return list(pair_set), node_pairs_start, node_pairs_end

# def function_coverage(nodes_index_G1, k, coverage_steps, sampling_num = 5, seed=None):
#     """
#     Vary coverage, fix diversity (k).
#     Args:
#         nodes_index_G1: list of all nodes in G1
#         k: fixed diversity
#         coverage_steps: list of floats (e.g., [0.1, 0.2, ..., 1.0])
#         seed: optional random seed
    
#     Returns:
#         Dict mapping coverage ratio to (pairs, node_to_pairs)
#     """
#     results = defaultdict(list)
#     for i, ratio in enumerate(coverage_steps):
#         subset_size = int(ratio * len(nodes_index_G1))
#         assert subset_size > 2, "Coverage ratio must be > 2"
#         for t in range(sampling_num):
#             if seed is not None:
#                 random.seed(seed + i + t)
#             X_sub = random.sample(nodes_index_G1, subset_size)
#             try:
#                 pairs, node_pairs_start, node_pairs_end = generate_pairs(X_sub, k, seed=(seed + i if seed else None))
#                 results[ratio].append((pairs, node_pairs_start, node_pairs_end))
#             except ValueError as e:
#                 print(f"Coverage ratio {ratio:.2f}-{t} skipped: {e}")
#     return results
            

# def function_diversity(X, diversity_range, sampling_num = 5, seed=None):
#     """
#     Vary diversity (k), fix coverage (X).
#     Args:
#         X: list of fixed nodes
#         diversity_range: list of ints (e.g., range(1, len(X)))
#         seed: optional random seed
    
#     Returns:
#         Dict mapping k to (pairs, node_to_pairs)
#     """
#     results = defaultdict(list)
#     for i, k in enumerate(diversity_range):
#         for t in range(sampling_num):
#             try:
#                 pairs, node_pairs_start, node_pairs_end = generate_pairs(X, k, seed=(seed + i if seed else None))
#                 results[k].append((pairs, node_pairs_start, node_pairs_end))
#             except ValueError as e:
#                 print(f"Diversity k={k}-{t} skipped: {e}")
#     return results

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
    config_path = os.path.join(parent_dir, "config.yaml")

    config = load_config(config_path)
    save_dir = config["dataset"]["dir"]
    save_dir = os.path.join(parent_dir, save_dir)
    single_size_m, single_size_n = config["dataset"]["grid"]["size_m"], config["dataset"]["grid"]["size_n"]
    size_m = single_size_m
    size_n = single_size_n * config["dataset"]["grid"]["num_grids"]
    max_len = config["dataset"]["shortest_path"]["max_len"]
    assert isinstance(max_len, int), "max_len must be an integer"

    # load the grid and adjacency matrix
    with open(os.path.join(save_dir, "map_stats", 'nodes_to_indices.pkl'), 'rb') as f:
        nodes_to_indices = pickle.load(f)
    with open(os.path.join(save_dir, "map_stats", 'indices_to_nodes.pkl'), 'rb') as f:
        indices_to_nodes = pickle.load(f)
    adj_matrix = np.load(os.path.join(save_dir, "map_stats", 'adj_matrix.npy'))
    nodes_coords = list(nodes_to_indices.keys())

    stats = {'interval': config["dataset"]["shortest_path"]["interval"], 
            'size_m': size_m, 'size_n': size_n,}

    log_dir = config["logdir"]
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'split_G1.log')
    # Set up logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,               # Log level
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    os.makedirs(os.path.join(save_dir, '_diversity_coverage'), exist_ok=True)
    component_id = 0
    with open(os.path.join(save_dir, 'map_stats', 'G1_indices.pkl'), 'rb') as f:
        G1_indices = pickle.load(f)
    # split indices into train (90%) and test (10%)
    from sklearn.model_selection import train_test_split
    train_indices, test_indices = train_test_split(G1_indices, test_size=0.1, random_state=42)
    logging.info(f"Split G1 indices into {len(train_indices)} train and {len(test_indices)} test indices.")

    # save train and test indices
    save_pickle(train_indices, os.path.join(save_dir, '_diversity_coverage', 'train_indices.pkl'))
    save_pickle(test_indices, os.path.join(save_dir, '_diversity_coverage', 'test_indices.pkl'))

    # generate pairs for coverage
    coverage_steps = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
    diversity_range = [1, 2, 4, 8, 16, 32, 64, 128]
    sampling_num = 5

    for t in range(sampling_num):
        logging.info(f"====================== Sampling {t+1}/{sampling_num} ======================")
        for k in diversity_range:
            for ratio in coverage_steps:
                logging.info(f"====================== Generating pairs for coverage ratio={ratio:.2f} with diversity k={k} ======================")
                X_sub = random.sample(train_indices, int(ratio * len(train_indices)))
                if k > len(X_sub) - 1:
                    logging.warning(f"Skipping coverage ratio={ratio:.2f} with diversity k={k} - not enough nodes in subset")
                    continue
                pairs, node_pairs_start, node_pairs_end = generate_pairs(X_sub, k, seed=None)
                folder_dir = os.path.join(save_dir, '_diversity_coverage', f'diversity_{k}', f'coverage_ratio_{ratio:.2f}', f'pairs_{t}')
                os.makedirs(folder_dir, exist_ok=True)
                save_pickle(pairs, os.path.join(folder_dir, 'pairs.pkl'))
                save_pickle(node_pairs_start, os.path.join(folder_dir, 'node_pairs_start.pkl'))
                save_pickle(node_pairs_end, os.path.join(folder_dir, 'node_pairs_end.pkl'))
                logging.info(f"Saved pairs for coverage ratio={ratio:.2f} with diversity k={k} and sampling {t+1}/{sampling_num} to {folder_dir}")

