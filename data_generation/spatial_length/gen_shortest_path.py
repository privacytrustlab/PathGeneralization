"""
Generate shortest paths for spatial transfer experiments.

Three modes corresponding to different data needs:
  - coverage:    Paths for a single coverage ratio (e.g., 1.0) with m=1 path per pair
  - longshort:   Paths per distance group for length scaling rescue (Section 5)
  - incremental: Paths for a target coverage, building on an existing base coverage (Section 4.1)

Usage:
  python gen_shortest_path.py --mode coverage --coverage 1.0
  python gen_shortest_path.py --mode longshort
  python gen_shortest_path.py --mode incremental --target_coverage 0.6 --base_coverage 0.2 --answer_num 128
"""

import random
import sys
import pickle
import numpy as np
import os
import logging
import time
import heapq
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils import save_pickle, save_numpy, load_config, save_shard


def idx_path_to_directions(path, indices_to_nodes, interval=5):
    directions = [path[0]]
    direction_intervals = [path[0]]
    for move in range(1, len(path)):
        u, v = indices_to_nodes[path[move-1]], indices_to_nodes[path[move]]
        x1, y1 = u
        x2, y2 = v
        if x1 == x2:
            value = "N" if y2 > y1 else "S"
        else:
            value = "E" if x2 > x1 else "W"
        directions.append(value)
        direction_intervals.append(value)
        if move % interval == 0 and move != (len(path)-1):
            direction_intervals.append(path[move])
    directions.append(path[-1])
    direction_intervals.append(path[-1])
    return directions, direction_intervals


def dijkstra_sample_shortest_paths(adj_matrix, start, end, num_nodes, m=1, warn_insufficient=True):
    INF = np.iinfo(np.int32).max
    dist = np.full(num_nodes, INF, dtype=np.int32)
    prev_list = [[] for _ in range(num_nodes)]

    dist[start] = 0
    heap = [(0, start)]

    while heap:
        cost, node = heapq.heappop(heap)
        if cost > dist[node]:
            continue
        neighbors = np.nonzero(adj_matrix[node])[0]
        for neighbor in neighbors:
            new_cost = dist[node] + 1
            if new_cost < dist[neighbor]:
                dist[neighbor] = new_cost
                prev_list[neighbor] = [node]
                heapq.heappush(heap, (new_cost, neighbor))
            elif new_cost == dist[neighbor]:
                prev_list[neighbor].append(node)

    if dist[end] == INF:
        return [], INF

    def random_backtrack():
        path = [end]
        current = end
        while current != start:
            current = random.choice(prev_list[current])
            path.append(current)
        return path[::-1]

    sampled_paths = set()
    attempts = 0
    no_new_paths_count = 0
    max_attempts = 5 * m
    max_no_new = m

    while len(sampled_paths) < m and attempts < max_attempts and no_new_paths_count < max_no_new:
        path = tuple(random_backtrack())
        if path in sampled_paths:
            no_new_paths_count += 1
        else:
            sampled_paths.add(path)
            no_new_paths_count = 0
        attempts += 1

    if warn_insufficient and len(sampled_paths) < m:
        logging.info(f"Warning: Only found {len(sampled_paths)} unique paths for {start} -> {end}.")

    return [list(p) for p in sampled_paths], dist[end]


def run_dijkstra_for_pairs(adj_matrix, nodes_to_indices, indices_to_nodes, node_pairs,
                           paths_count_matrix, pairs_count_matrix, total_shortest_paths_matrix,
                           m=1, interval=5):
    num_nodes = adj_matrix.shape[0]
    all_paths = {}

    for start_idx, end_idx in node_pairs:
        start = indices_to_nodes[start_idx]
        end = indices_to_nodes[end_idx]
        select_paths, distance = dijkstra_sample_shortest_paths(adj_matrix, start_idx, end_idx, num_nodes, m)
        total_shortest_paths_matrix[start_idx, end_idx] = m
        logging.info(f"Select {len(select_paths)} unique paths for {start} -> {end} for distance {distance}.")

        formatted_paths = []
        for path in select_paths:
            direction_path, direction_with_mid = idx_path_to_directions(path, indices_to_nodes, interval=interval)
            formatted_paths.append({
                "index_paths": path,
                "directions": direction_path,
                "directions_with_mid": direction_with_mid,
                "sp_distance": distance,
                "coord_distance": np.linalg.norm(np.array(start) - np.array(end)),
                "start_node": start, "end_node": end,
                "start_idx": start_idx, "end_idx": end_idx,
                "direction_seq": f"<s> {start_idx} {end_idx} : {' '.join([str(node) for node in direction_path])} </s>",
                "direction_with_mid_seq": f"<s> {start_idx} {end_idx} : {' '.join([str(node) for node in direction_with_mid])} </s>"
            })
            for i in range(len(path) - 1):
                paths_count_matrix[path[i], path[i+1]] += 1

        all_paths[(start, end)] = formatted_paths
        pairs_count_matrix[start_idx, end_idx] = len(formatted_paths)

    return all_paths, paths_count_matrix, pairs_count_matrix, total_shortest_paths_matrix


def generate_and_save(nodes_to_indices, indices_to_nodes, train_pairs, adj_matrix,
                      dataset_dir, m=1, interval=5, existing_folder=None):
    n = len(nodes_to_indices)
    paths_count_matrix = np.zeros((n, n), dtype=int)
    pairs_count_matrix = np.zeros((n, n), dtype=int)
    total_shortest_paths_matrix = np.zeros((n, n), dtype=int)

    s = time.time()
    all_paths, paths_count_matrix, pairs_count_matrix, total_shortest_paths_matrix = run_dijkstra_for_pairs(
        adj_matrix, nodes_to_indices, indices_to_nodes, train_pairs,
        paths_count_matrix, pairs_count_matrix, total_shortest_paths_matrix, m, interval=interval)

    # Merge existing paths from base coverage (incremental mode)
    if existing_folder is not None:
        existing_paths_file = os.path.join(existing_folder, 'paths.pkl')
        if os.path.exists(existing_paths_file):
            with open(existing_paths_file, 'rb') as f:
                existing_paths = pickle.load(f)
            logging.info(f"Added {len(existing_paths)} existing paths from {existing_paths_file}.")
            all_paths.update(existing_paths)
            paths_count_matrix += np.load(os.path.join(existing_folder, 'paths_count_matrix.npy'))
            pairs_count_matrix += np.load(os.path.join(existing_folder, 'pairs_count_matrix.npy'))
            total_shortest_paths_matrix += np.load(os.path.join(existing_folder, 'total_shortest_paths_matrix.npy'))

    save_shard(all_paths, os.path.join(dataset_dir, 'paths.pkl'))
    logging.critical(f"Generated {len(all_paths)} paths in {time.time()-s:.1f}s.")
    save_numpy(paths_count_matrix, os.path.join(dataset_dir, 'paths_count_matrix.npy'))
    save_numpy(pairs_count_matrix, os.path.join(dataset_dir, 'pairs_count_matrix.npy'))
    save_numpy(total_shortest_paths_matrix, os.path.join(dataset_dir, 'total_shortest_paths_matrix.npy'))


def setup_logging(log_file):
    if os.path.exists(log_file):
        os.remove(log_file)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


# ============================================================
# Mode: coverage — single coverage ratio, m=1 path per pair
# ============================================================
def run_coverage(save_dir, interval, coverage):
    folder_dir = os.path.join(save_dir, '_spatial_length', f'coverage_ratio_{coverage:.2f}', 'pairs_0/shortest_path')
    pairs_file = os.path.abspath(os.path.join(folder_dir, '..', 'sample_pairs.pkl'))
    if not os.path.exists(pairs_file):
        print(f"Pairs file not found: {pairs_file}")
        return
    os.makedirs(folder_dir, exist_ok=True)
    setup_logging(os.path.join(folder_dir, 'shortest_path.log'))

    with open(pairs_file, 'rb') as f:
        train_pairs = pickle.load(f)

    logging.info(f"Generating paths for coverage={coverage:.2f}, {len(train_pairs)} pairs")
    generate_and_save(nodes_to_indices, indices_to_nodes, train_pairs, adj_matrix,
                      folder_dir, m=1, interval=interval)


# ============================================================
# Mode: longshort — paths per distance group
# ============================================================
def run_longshort(save_dir, interval):
    folder_dir = os.path.join(save_dir, '_spatial_length', 'longshort_pairs')
    os.makedirs(folder_dir, exist_ok=True)

    with open(os.path.join(save_dir, '_spatial_length', 'distance_group_to_train_pairs.pkl'), 'rb') as f:
        all_pairs = pickle.load(f)

    for group, train_pairs in all_pairs.items():
        save_folder = os.path.join(folder_dir, f'group_{group}')
        if len(train_pairs) > 10000:
            train_pairs = random.sample(train_pairs, 10000)
        print(f"Group {group}: {len(train_pairs)} pairs")
        os.makedirs(save_folder, exist_ok=True)
        with open(os.path.join(save_folder, 'train_pairs.pkl'), 'wb') as f:
            pickle.dump(train_pairs, f)

        setup_logging(os.path.join(save_folder, 'shortest_path.log'))
        generate_and_save(nodes_to_indices, indices_to_nodes, train_pairs, adj_matrix,
                          save_folder, m=1, interval=interval)


# ============================================================
# Mode: incremental — build on existing base coverage
# ============================================================
def run_incremental(save_dir, interval, target_coverage, base_coverage, answer_num):
    folder_dir = os.path.join(save_dir, '_spatial_length', f'coverage_ratio_{target_coverage:.2f}', 'pairs_0/shortest_path')
    os.makedirs(folder_dir, exist_ok=True)

    with open(os.path.abspath(os.path.join(folder_dir, '..', 'sample_pairs.pkl')), 'rb') as f:
        all_pairs = pickle.load(f)
    existing_folder = os.path.join(save_dir, '_spatial_length', f'coverage_ratio_{base_coverage:.2f}', 'pairs_0')
    with open(os.path.join(existing_folder, 'sample_pairs.pkl'), 'rb') as f:
        existing_pairs = pickle.load(f)
    existing_sp_folder = os.path.join(existing_folder, 'shortest_path')
    train_pairs = list(set(all_pairs) - set(existing_pairs))

    logging.info(f"Target coverage: {target_coverage}, base: {base_coverage}")
    logging.info(f"New pairs: {len(train_pairs)}, existing: {len(existing_pairs)}")

    setup_logging(os.path.join(folder_dir, 'shortest_path.log'))
    generate_and_save(nodes_to_indices, indices_to_nodes, train_pairs, adj_matrix,
                      folder_dir, m=answer_num, interval=interval, existing_folder=existing_sp_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate shortest paths for spatial transfer experiments.")
    parser.add_argument('--mode', type=str, required=True, choices=['coverage', 'longshort', 'incremental'])
    parser.add_argument('--coverage', type=float, default=1.0, help='Coverage ratio (for coverage mode)')
    parser.add_argument('--target_coverage', type=float, default=0.6, help='Target coverage (for incremental mode)')
    parser.add_argument('--base_coverage', type=float, default=0.2, help='Base coverage (for incremental mode)')
    parser.add_argument('--answer_num', type=int, default=128, help='Paths per pair (for incremental mode)')
    cli_args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, "../.."))
    config_path = os.path.join(parent_dir, "config.yaml")

    config = load_config(config_path)
    save_dir = os.path.join(parent_dir, config["dataset"]["dir"])
    interval = config["dataset"]["shortest_path"]["interval"]

    with open(os.path.join(save_dir, "map_stats", 'nodes_to_indices.pkl'), 'rb') as f:
        nodes_to_indices = pickle.load(f)
    with open(os.path.join(save_dir, "map_stats", 'indices_to_nodes.pkl'), 'rb') as f:
        indices_to_nodes = pickle.load(f)
    adj_matrix = np.load(os.path.join(save_dir, "map_stats", 'adj_matrix.npy'))

    if cli_args.mode == 'coverage':
        run_coverage(save_dir, interval, cli_args.coverage)
    elif cli_args.mode == 'longshort':
        run_longshort(save_dir, interval)
    elif cli_args.mode == 'incremental':
        run_incremental(save_dir, interval, cli_args.target_coverage, cli_args.base_coverage, cli_args.answer_num)
