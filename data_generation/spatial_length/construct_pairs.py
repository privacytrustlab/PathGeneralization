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
    # max_len = config["dataset"]["shortest_path"]["max_len"]
    max_len = 20
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
    log_file = os.path.join(log_dir, 'split_G1_with_len.log')
    # Set up logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,               # Log level
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    os.makedirs(os.path.join(save_dir, '_spatial_length'), exist_ok=True)
    component_id = 0
    with open(os.path.join(save_dir, 'map_stats', 'G1_indices.pkl'), 'rb') as f:
        G1_indices = pickle.load(f)
    # split indices into train (80%) and test (20%)
    from sklearn.model_selection import train_test_split
    train_indices, test_indices = train_test_split(G1_indices, test_size=0.2, random_state=42)
    logging.info(f"Split G1 indices into {len(train_indices)} train and {len(test_indices)} test indices.")

    # save train and test indices
    save_pickle(train_indices, os.path.join(save_dir, '_spatial_length', 'train_indices.pkl'))
    save_pickle(test_indices, os.path.join(save_dir, '_spatial_length', 'test_indices.pkl'))

    if os.path.exists(os.path.join(save_dir, '_spatial_length', 'train_pairs_pool.pkl')):
        train_pairs_pool = pickle.load(open(os.path.join(save_dir, '_spatial_length', 'train_pairs_pool.pkl'), 'rb'))
        test_pairs_pool = pickle.load(open(os.path.join(save_dir, '_spatial_length', 'test_pairs_pool.pkl'), 'rb'))
        train_indice_to_pairs = pickle.load(open(os.path.join(save_dir, '_spatial_length', 'train_indice_to_pairs.pkl'), 'rb'))
        test_indice_to_pairs = pickle.load(open(os.path.join(save_dir, '_spatial_length', 'test_indice_to_pairs.pkl'), 'rb'))
        logging.info(f"Loaded existing train and test pairs pool from disk.")
    else:
        logging.info(f"Generating train and test pairs pool based on max_len={max_len}...")
        # load dataset/map_stats/distance_to_pair_component_0.pkl
        with open(os.path.join(save_dir, 'map_stats', f'distance_to_pair_component_{component_id}.pkl'), 'rb') as f:
            distance_to_pair = pickle.load(f)

        # filter train_pairs_pool and test_pairs_pool based on max_len <= 10
        filtered_distance_to_pair = {dist: pairs for dist, pairs in distance_to_pair.items() if dist <= max_len}
        # Convert to sets for efficient lookup
        train_set = set(train_indices)
        test_set = set(test_indices)
        train_pairs_pool = []
        test_pairs_pool = []
        train_indice_to_pairs = defaultdict(list)
        test_indice_to_pairs = defaultdict(list)
        
        for pairs in filtered_distance_to_pair.values():
            for s, e in pairs:
                if s in train_set and e in train_set:
                    train_pairs_pool.append((s, e))
                    train_pairs_pool.append((e, s))
                    train_indice_to_pairs[s].append((s, e))
                    train_indice_to_pairs[e].append((e, s))
                elif s in test_set and e in test_set:
                    test_pairs_pool.append((s, e))
                    test_pairs_pool.append((e, s))
                    test_indice_to_pairs[s].append((s, e))
                    test_indice_to_pairs[e].append((e, s))
        logging.info(f"Filtered train pairs pool to {len(train_pairs_pool)} pairs and test pairs pool to {len(test_pairs_pool)} pairs based on max_len={max_len}.")
        save_pickle(train_pairs_pool, os.path.join(save_dir, '_spatial_length', 'train_pairs_pool.pkl'))
        save_pickle(test_pairs_pool, os.path.join(save_dir, '_spatial_length', 'test_pairs_pool.pkl'))
        save_pickle(train_indice_to_pairs, os.path.join(save_dir, '_spatial_length', 'train_indice_to_pairs.pkl'))
        save_pickle(test_indice_to_pairs, os.path.join(save_dir, '_spatial_length', 'test_indice_to_pairs.pkl'))

    # generate pairs for coverage
    # coverage_steps = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.8, 1.0] # len(train_pairs_pool) = 644696
    coverage_steps = [0.01, 0.05, 0.1, 0.2]
    sampling_num = 1

    for t in range(sampling_num):
        logging.info(f"====================== Sampling {t+1}/{sampling_num} ======================")
        
        # Shuffle once at the beginning
        shuffled_pairs = train_pairs_pool.copy()
        random.shuffle(shuffled_pairs)
        
        for i, ratio in enumerate(coverage_steps):
            logging.info(f"====================== Generating pairs for coverage ratio={ratio:.2f} ======================")
            num_pairs = int(ratio * len(train_pairs_pool))
            
            # Take the first num_pairs from shuffled list
            sample_pairs = shuffled_pairs[:num_pairs]
            
            # Calculate added pairs (new pairs since last step)
            if i == 0:
                added_pairs = sample_pairs
            else:
                prev_num_pairs = int(coverage_steps[i-1] * len(train_pairs_pool))
                added_pairs = shuffled_pairs[prev_num_pairs:num_pairs]
            
            folder_dir = os.path.join(save_dir, '_spatial_length', f'coverage_ratio_{ratio:.2f}', f'pairs_{t}')
            os.makedirs(folder_dir, exist_ok=True)
            save_pickle(sample_pairs, os.path.join(folder_dir, 'sample_pairs.pkl'))
            save_pickle(added_pairs, os.path.join(folder_dir, 'added_pairs.pkl'))
            logging.info(f"Saved {len(sample_pairs)} sample pairs and {len(added_pairs)} added pairs for coverage ratio={ratio:.2f} and sampling {t+1}/{sampling_num} to {folder_dir}")

