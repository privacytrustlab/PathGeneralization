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
    log_file = os.path.join(log_dir, 'prepare_long_short.log')
    # Set up logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,               # Log level
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    component_id = 0
    with open(os.path.join(save_dir, 'map_stats', 'G1_indices.pkl'), 'rb') as f:
        G1_indices = pickle.load(f)
    # load train and test indices
    train_indices = pickle.load(open(os.path.join(save_dir, '_spatial_length', 'train_indices.pkl'), 'rb'))

    with open(os.path.join(save_dir, 'map_stats', f'distance_to_pair_component_{component_id}.pkl'), 'rb') as f:
        distance_to_pair = pickle.load(f)

    distance_groups = [(0,10), (10,20), (20,30), (30,40), (40,50), (50,60), (60,70), (70,80), (80,90)]
    
    # Group pairs by distance ranges
    distance_group_to_pairs = {}
    for a, b in distance_groups:
        pairs_in_group = []
        for distance in range(a+1, b+1):
            if distance in distance_to_pair:
                pairs_in_group.extend(distance_to_pair[distance])
        distance_group_to_pairs[(a, b)] = pairs_in_group

    # Filter train pairs for each group
    train_set = set(train_indices)
    distance_group_to_train_pairs = {}
    for group, pairs in distance_group_to_pairs.items():
        train_pairs = [(s, e) for s, e in pairs if s in train_set and e in train_set]
        distance_group_to_train_pairs[group] = train_pairs
        print(f"Distance group {group}: {len(pairs)} total pairs, {len(train_pairs)} train pairs")
    
    # Save results
    save_pickle(distance_group_to_pairs, os.path.join(save_dir, '_spatial_length', 'distance_group_to_pairs.pkl'))
    save_pickle(distance_group_to_train_pairs, os.path.join(save_dir, '_spatial_length', 'distance_group_to_train_pairs.pkl'))

    # Distance group (0, 10): 147723 total pairs, 94188 train pairs
    # Distance group (10, 20): 357908 total pairs, 228160 train pairs
    # Distance group (20, 30): 450811 total pairs, 288215 train pairs
    # Distance group (30, 40): 430623 total pairs, 275486 train pairs
    # Distance group (40, 50): 320100 total pairs, 203798 train pairs
    # Distance group (50, 60): 193198 total pairs, 124435 train pairs
    # Distance group (60, 70): 80949 total pairs, 53020 train pairs
    # Distance group (70, 80): 16838 total pairs, 11304 train pairs
    # Distance group (80, 90): 850 total pairs, 594 train pairs
    