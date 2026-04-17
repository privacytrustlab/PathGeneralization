import random
import sys
import pickle
import numpy as np
from collections import defaultdict
import os
import logging
import time
import numba
import networkx as nx
import heapq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from utils import save_pickle, save_numpy, load_config, save_shard
from utils import get_indices, get_nodes, get_valid_neighbors

def idx_path_to_directions(path, indices_to_nodes, interval=5):
    """
    Converts a path of node indices to a list of directions.
    """
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
    """
    Dijkstra's algorithm that samples up to m random shortest paths (without full enumeration).
    Returns a list of paths and the distance.
    """
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
        return [], INF  # No path exists

    def random_backtrack():
        path = [end]
        current = end
        while current != start:
            current = random.choice(prev_list[current])
            path.append(current)
        return path[::-1]

    # Sample paths with early stopping when no new paths found
    sampled_paths = set()
    attempts = 0
    no_new_paths_count = 0
    max_attempts = 5 * m
    max_no_new = m  # Stop if we don't find new paths for m consecutive attempts

    while len(sampled_paths) < m and attempts < max_attempts and no_new_paths_count < max_no_new:
        path = tuple(random_backtrack())
        if path in sampled_paths:
            no_new_paths_count += 1
        else:
            sampled_paths.add(path)
            no_new_paths_count = 0
        attempts += 1

    actual_count = len(sampled_paths)
    
    if warn_insufficient and actual_count < m:
        logging.info(f"Warning: Only found {actual_count} unique paths for {start} → {end}.")

    return [list(p) for p in sampled_paths], dist[end]


def run_fast_dijkstra_multiple_times(adj_matrix, nodes_to_indices, indices_to_nodes, node_pairs, paths_count_matrix, pairs_count_matrix, total_shortest_paths_matrix, m=1, interval=5):
    """
    Runs Dijkstra `m` times efficiently using JIT-optimized Dijkstra.
    - nodes_to_indices: Dict (Mapping node coord → index)
    - indices_to_nodes: Dict (Mapping index → node coord)
    """
    num_nodes = adj_matrix.shape[0]
    all_paths = {}

    for start_idx, end_idx in node_pairs:
        start = indices_to_nodes[start_idx]
        end = indices_to_nodes[end_idx]
        # path, distance = dijkstra_randomized_jit(adj_matrix, start_idx, end_idx, num_nodes, pseudo_seed)
        # prev_list, distance = dijkstra_all_shortest_paths_jit(adj_matrix, start_idx, end_idx, num_nodes)
        # paths = reconstruct_paths(start_idx, end_idx, prev_list)
        select_paths, distance = dijkstra_sample_shortest_paths(adj_matrix, start_idx, end_idx, num_nodes, m)
        total_shortest_paths_matrix[start_idx, end_idx] = m
        logging.info(f"Select {len(select_paths)} unique paths for {start} → {end} for distance {distance}.")

        formatted_paths = []
        for path in select_paths:
            direction_path, direction_with_mid = idx_path_to_directions(path, indices_to_nodes, interval=interval)
            formatted_paths.append({
                "index_paths": path,
                "directions": direction_path,
                "directions_with_mid": direction_with_mid,
                "sp_distance": distance,
                "coord_distance": np.linalg.norm(np.array(start) - np.array(end)),
                "start_node": start,
                "end_node": end,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "direction_seq": f"<s> {start_idx} {end_idx} : {' '.join([str(node) for node in direction_path])} </s>",
                "direction_with_mid_seq": f"<s> {start_idx} {end_idx} : {' '.join([str(node) for node in direction_with_mid])} </s>"
            })
            # add edges to the paths_count_matrix
            for i in range(len(path) - 1):
                paths_count_matrix[path[i], path[i+1]] += 1
            

        # Store paths, warn if fewer than m were found
        all_paths[(start, end)] = formatted_paths
        pairs_count_matrix[start_idx, end_idx] = len(formatted_paths)

    return all_paths, paths_count_matrix, pairs_count_matrix, total_shortest_paths_matrix

def generate_paths(nodes_to_indices, indices_to_nodes, train_pairs, adj_matrix, dataset_dir, m=1, interval=5):
    paths_count_matrix = np.zeros((len(nodes_to_indices), len(nodes_to_indices)), dtype=int)
    pairs_count_matrix = np.zeros((len(nodes_to_indices), len(nodes_to_indices)), dtype=int)
    total_shortest_paths_matrix = np.zeros((len(nodes_to_indices), len(nodes_to_indices)), dtype=int)

    s = time.time()
    all_paths, paths_count_matrix, pairs_count_matrix, total_shortest_paths_matrix = run_fast_dijkstra_multiple_times(adj_matrix, nodes_to_indices, indices_to_nodes, train_pairs, paths_count_matrix, pairs_count_matrix, total_shortest_paths_matrix, m, interval=interval)

    # Save paths
    save_shard(all_paths, os.path.join(dataset_dir, f'paths.pkl'))
    logging.critical(f"Generated {len(all_paths)} paths takes {time.time()-s} seconds.")

    save_numpy(paths_count_matrix, os.path.join(dataset_dir, 'paths_count_matrix.npy'))
    save_numpy(pairs_count_matrix, os.path.join(dataset_dir, 'pairs_count_matrix.npy'))
    save_numpy(total_shortest_paths_matrix, os.path.join(dataset_dir, 'total_shortest_paths_matrix.npy'))
    del paths_count_matrix, pairs_count_matrix, total_shortest_paths_matrix


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

    coverage_steps = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
    diversity_range = [1, 2, 4, 8, 16, 32, 64, 128]
    sampling_num = 3
    # for t in range(sampling_num):
    for t in range(sampling_num):
        for k in diversity_range[::-1]: # reverse the diversity range to generate more diverse pairs first
            for ratio in coverage_steps[::-1]: # reverse the coverage steps to generate more diverse pairs first
                folder_dir = os.path.join(save_dir, '_diversity_coverage', f'diversity_{k}', f'coverage_ratio_{ratio:.2f}', f'pairs_{t}/shortest_path')
                if not os.path.exists(os.path.abspath(os.path.join(folder_dir, '..', 'pairs.pkl'))):
                    continue # does not exist because the diversity k is too large for the coverage ratio
                else:
                    os.makedirs(folder_dir, exist_ok=True)

                # add logging file
                log_file = os.path.join(folder_dir, 'shortest_path.log')
                # delete existing log file if it exists
                if os.path.exists(log_file):
                    os.remove(log_file)
                for handler in logging.root.handlers[:]:
                    logging.root.removeHandler(handler)
                logging.basicConfig(
                    filename=log_file,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s'
                )

                logging.info(f"====================== Generating {t+1}/{sampling_num} pairs for coverage ratio={ratio:.2f} with diversity k={k} ======================")
                with open(os.path.abspath(os.path.join(folder_dir, '..', 'pairs.pkl')), 'rb') as f:
                    train_pairs = pickle.load(f)
    
                # G = nx.from_numpy_array(adj_matrix)
                # adjacency_list = prepare_graph(G)
                m = 128*1.0/(k*ratio)  # Adjust m based on diversity and coverage ratio, upper round m
                m = int(np.ceil(m))
                generate_paths(nodes_to_indices, indices_to_nodes, train_pairs, adj_matrix, folder_dir, m=m, interval=stats['interval'])