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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils import save_pickle, save_numpy, load_config, save_shard
from utils import get_indices, get_nodes, get_valid_neighbors

def get_all_node_pairs_by_distance(component):
    """
    Computes a dictionary mapping Manhattan distance d to all node pairs (i, j) (for i<= j) in an m x n grid 
    where the distance between them is exactly d.

    Parameters:
    - m_size (int): Number of rows in the grid.
    - n_size (int): Number of columns in the grid.

    Returns:
    - dict[int, list[tuple]]: Dictionary where keys are distances d, and values are lists of node index pairs (i, j).
    """
    distance_dict = defaultdict(list)

    for u in component:
        for v, dist in component[u].items():
            if u < v:  # avoid duplicate pairs
                # coord_distance = np.linalg.norm(np.array(indices_to_nodes[u]) - np.array(indices_to_nodes[v]))
                distance_dict[dist].append((u, v))

    return distance_dict

def split_pairs_by_distance(distance_dict, train_len_list, train_pair_matrix, node_to_indices):
    """
    Splits node pairs by distance into training and test sets.

    Parameters:
    - distance_dict (dict[int, list[tuple]]): Dictionary mapping distances to node pairs.
    - train_len_list (list[int]): List of kept distance in training dataset.
    - train_pair_matrix (np.ndarray): Matrix to record node pairs to keep in training dataset.

    Returns:
    - dict[int, list[tuple]], dict[int, list[tuple]]: Dictionaries of training and test node pairs by distance.
    """
    train_pairs = defaultdict(list)
    test_pairs = defaultdict(list)

    for distance, pairs in distance_dict.items():
        if distance in train_len_list:
            train_pairs[distance] = pairs.copy()
        else:
            # remove the pair from the training matrix
            test_pairs[distance] = pairs + [(b, a) for (a, b) in pairs]

            indices_1, indices_2 = zip(*[(node_to_indices[a], node_to_indices[b]) for a, b in pairs])
            indices_1 = np.array(indices_1, dtype=int)
            indices_2 = np.array(indices_2, dtype=int)

            train_pair_matrix[indices_1, indices_2] = 0
            train_pair_matrix[indices_2, indices_1] = 0
                
    return train_pairs, test_pairs, train_pair_matrix

@numba.njit
def should_remove(pair):
    (x1, y1), (x2, y2) = pair
    return (y1 == y2 and x2 - x1 >= 12) or (x1 == x2 and y2 - y1 >= 18) #pattern = ['E' * 12, 'N' * 18]

@numba.njit
def update_matrix(matrix, i1, i2):
    """ Updates the adjacency matrix by setting specified positions to 1. """
    matrix[i1, i2] = 1
    return matrix

@numba.njit
def update_both_direction_matrix(matrix, i1, i2):
    """ Updates the adjacency matrix by setting specified positions to 1. """
    matrix[i1, i2] = 1
    matrix[i2, i1] = 1
    return matrix

def filter_pairs_by_relative_position(train_pairs, test_pairs, train_len_list, train_pair_matrix, node_to_indices):
    """
    Filter training pairs by relative position patterns: 'EEEEEEEEEEEEEEE', 'NNNNNNNNNNNNNNNNNN'.

    Parameters:
    - train_pairs (dict[int, list[tuple]]): Dictionary of training node pairs by distance.
    - test_pairs (dict[int, list[tuple]]): Dictionary of test node pairs by distance.
    - train_len_list (list[int]): List of kept distance in training dataset.
    - train_pair_matrix (np.ndarray): Matrix to record node pairs to keep in training dataset.
    - patterns (list[str]): List of relative position patterns to filter.

    Returns:
    - dict[int, list[tuple]], dict[int, list[tuple]]: Filtered training and test node pairs by distance.
    """
    selected_train_pairs = defaultdict(list)
    for distance in train_len_list:
        if distance < 12: # skip the distance less than 12 since it is not possible to form the pattern
            continue
        pairs = train_pairs[distance]
        new_pairs = []
        for pair in pairs:
            i1, i2 = node_to_indices[pair[0]], node_to_indices[pair[1]]
            if should_remove(pair):
                train_pair_matrix = update_matrix(train_pair_matrix, i1, i2)  # Capture returned updated matrix
                test_pairs[distance].append(pair)
                selected_train_pairs[distance].append(pair[::-1])
            elif should_remove(pair[::-1]):
                train_pair_matrix = update_matrix(train_pair_matrix, i2, i1)
                test_pairs[distance].append(pair[::-1])
                selected_train_pairs[distance].append(pair)
            else:
                new_pairs.append(pair)
        
        train_pairs[distance] = new_pairs

    return train_pairs, selected_train_pairs, test_pairs, train_pair_matrix

def split_pairs(distance_dict, node_to_indices):
    """
    Split node pairs in train_pairs into 60% bidirection and 40% unidirection. For bidirection, 80% are kept in training dataset and 20% are kept in test dataset.

    Parameters:
    - distance_dict (dict[int, list[tuple]]): Dictionary mapping distances to node pairs.
    - node_to_indices (dict[str, int]): Dictionary mapping node to index.

    Returns:
    - dict[int, list[tuple]], dict[int, list[tuple]]: Updated training and test node pairs by distance.
    """
    selected_train_pairs = defaultdict(list)
    test_pairs = defaultdict(list)
    train_pair_matrix = np.zeros((len(node_to_indices), len(node_to_indices)), dtype=bool)
    for distance, pairs in distance_dict.items():
        random.shuffle(pairs)
        split_idx = int(len(pairs) * 0.6)
        bidirection, unidirection = pairs[:split_idx], pairs[split_idx:]

        for pair in bidirection:
            # 50% in training dataset and 50% in test dataset
            if random.random() < 0.5:
                selected_train_pairs[distance].extend([pair, pair[::-1]])
                train_pair_matrix = update_both_direction_matrix(train_pair_matrix, pair[0], pair[1])
            else:
                test_pairs[distance].extend([pair, pair[::-1]])
        for pair in unidirection:
            pair = pair[::-1] if random.random() < 0.5 else pair
            train_pair_matrix = update_matrix(train_pair_matrix, pair[0], pair[1])
            selected_train_pairs[distance].append(pair)
            test_pairs[distance].append(pair[::-1])
    
    return selected_train_pairs, test_pairs, train_pair_matrix

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

@numba.njit
def lcg_random(seed):
    """
    Linear Congruential Generator (LCG) for generating pseudo-random numbers.
    Uses the standard LCG formula:
        X_{n+1} = (a * X_n + c) % m
    """
    a = 1664525
    c = 1013904223
    m = 2**32
    return (a * seed + c) % m

@numba.njit
def shuffle_inplace(arr, seed):
    """
    Fisher-Yates (Knuth) Shuffle Algorithm implemented for JIT.
    Shuffles `arr` in place using `seed` for reproducibility.
    """
    n = len(arr)
    for i in range(n - 1, 0, -1):  # Iterate backwards
        seed = lcg_random(seed)  # Update seed
        j = seed % (i + 1)  # Get random index

        # Swap arr[i] and arr[j]
        arr[i], arr[j] = arr[j], arr[i]
    
    return arr

# @numba.njit
# def dijkstra_randomized_jit(adj_matrix, start, end, num_nodes, pseudo_seed):
#     """
#     JIT-optimized Dijkstra's algorithm with randomness using an adjacency list.
#     """
#     INF = np.iinfo(np.int32).max
#     dist = np.full(num_nodes, INF, dtype=np.int32)
#     prev = np.full(num_nodes, -1, dtype=np.int32)

#     dist[start] = 0
#     heap = [(0, start)]

#     while heap:
#         cost, node = heapq.heappop(heap)

#         if node == end:  # Early exit if we reach the destination
#             break

#         if cost > dist[node]:  # Ignore outdated distances
#             continue

#         # Randomize the neighbors
#         pseudo_seed = lcg_random(pseudo_seed)
#         neighbors = np.nonzero(adj_matrix[node])[0] # [(node_idx, weight), ...]
#         neighbors = [(neighbor, 1) for neighbor in neighbors]
#         # manually shuffle the neighbors
#         neighbors = shuffle_inplace(neighbors, pseudo_seed)
#         # shuffle the neighbors to get random path
#         for neighbor, weight in neighbors:
#             new_cost = dist[node] + weight
#             if new_cost < dist[neighbor]:
#                 dist[neighbor] = new_cost
#                 prev[neighbor] = node
#                 heapq.heappush(heap, (new_cost, neighbor))

#     # Reconstruct path
#     path = []
#     current = end
#     while current != -1:
#         path.append(current)
#         current = prev[current]

#     # return the shortest path and the distance
#     final_path = path[::-1] if path[-1] == start else [-1]
#     return final_path, dist[end] if final_path else INF

# @numba.njit
# def dijkstra_all_shortest_paths_jit(adj_matrix, start, end, num_nodes):
#     """
#     JIT-optimized Dijkstra's algorithm to find all shortest paths.
#     """
#     INF = np.iinfo(np.int32).max
#     dist = np.full(num_nodes, INF, dtype=np.int32)
#     prev_matrix = np.full((num_nodes, max_parents), -1, dtype=np.int32)  # Store up to `max_parents` predecessors per node
#     prev_count = np.zeros(num_nodes, dtype=np.int32)  # Track number of predecessors stored

#     dist[start] = 0
#     heap = [(0, start)]

#     while heap:
#         cost, node = heapq.heappop(heap)

#         if cost > dist[node]:  # Ignore outdated distances
#             continue

#         neighbors = np.nonzero(adj_matrix[node])[0]
#         neighbors = [(neighbor, 1) for neighbor in neighbors]  # [(node_idx, weight), ...] weight=1
#         for neighbor, weight in neighbors:
#             new_cost = dist[node] + weight
#             if new_cost < dist[neighbor]:  # Found a new shorter path
#                 dist[neighbor] = new_cost
#                 prev_list[neighbor] = np.array([node], dtype=np.int32)  # Reset with only this node
#                 heapq.heappush(heap, (new_cost, neighbor))
#             elif new_cost == dist[neighbor]:  # Found another shortest path
#                 prev_list[neighbor] = np.append(prev_list[neighbor], node)  # Add to predecessors

#     return prev_list, dist[end] if dist[end] != INF else INF

# @numba.njit
# def reconstruct_paths(start, end, prev_list):
#     """
#     JIT-optimized function to reconstruct all shortest paths using backtracking.
#     """
#     paths = []
#     queue = [[end]]  # Start backtracking from the end

#     while queue:
#         path = queue.pop(0)  # Get current path
#         node = path[-1]

#         if node == start:
#             paths.append(np.array(path[::-1], dtype=np.int32))  # Reverse and store the path
#             continue

#         for parent in prev_list[node]:
#             queue.append(path + [parent])

#     return paths

def dijkstra_all_shortest_paths_jit(adj_matrix, start, end, num_nodes):
    """
    JIT-optimized Dijkstra's algorithm modified to find all shortest paths.
    """
    INF = np.iinfo(np.int32).max
    dist = np.full(num_nodes, INF, dtype=np.int32)
    prev_list = [list() for _ in range(num_nodes)]  # Store multiple predecessors

    dist[start] = 0
    heap = [(0, start)]

    while heap:
        cost, node = heapq.heappop(heap)

        if cost > dist[node]:  # Ignore outdated distances
            continue

        neighbors = np.nonzero(adj_matrix[node])[0]  # Get neighbors
        neighbors = [(neighbor, 1) for neighbor in neighbors]  # [(node_idx, weight), ...] weight=1
        for neighbor, weight in neighbors:
            new_cost = dist[node] + weight
            if new_cost < dist[neighbor]:  # Found a new shorter path
                dist[neighbor] = new_cost
                prev_list[neighbor] = [node]  # Reset with only this node
                heapq.heappush(heap, (new_cost, neighbor))
            elif new_cost == dist[neighbor]:  # Found another shortest path
                prev_list[neighbor].append(node)  # Add to predecessors

    def backtrack(path, current):
        if current == start:
            paths.append(path[::-1])  # Reverse to start → end
            return
        for parent in prev_list[current]:
            backtrack(path + [parent], parent)

    paths = []
    if dist[end] != INF:
        backtrack([end], end)

    return paths, dist[end] if paths else INF

def run_fast_dijkstra_multiple_times(adj_matrix, nodes_to_indices, indices_to_nodes, node_pairs, paths_count_matrix, pairs_count_matrix, total_shortest_paths_matrix, interval=5):
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
        paths, distance = dijkstra_all_shortest_paths_jit(adj_matrix, start_idx, end_idx, num_nodes)
        total_shortest_paths_matrix[start_idx, end_idx] = len(paths)
        m = random.randint(1, len(paths)) if len(paths) > 1 else 1
        # shuffle the indices to get random paths
        indices = np.arange(len(paths))
        np.random.shuffle(indices)
        select_paths = [paths[i] for i in indices[:m]]
        logging.info(f"Select {len(select_paths)} unique paths from in total {len(paths)} paths for {start} → {end} for distance {distance}.")

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

def generate_paths(nodes_to_indices, indices_to_nodes, selected_train_pairs, adj_matrix, component_id, interval=5):
    total_count = 0
    for distance, node_pairs in selected_train_pairs.items():
        paths_count_matrix = np.zeros((len(nodes_to_indices), len(nodes_to_indices)), dtype=int)
        pairs_count_matrix = np.zeros((len(nodes_to_indices), len(nodes_to_indices)), dtype=int)
        total_shortest_paths_matrix = np.zeros((len(nodes_to_indices), len(nodes_to_indices)), dtype=int)

        s = time.time()
        all_paths, paths_count_matrix, pairs_count_matrix, total_shortest_paths_matrix = run_fast_dijkstra_multiple_times(adj_matrix, nodes_to_indices, indices_to_nodes, node_pairs, paths_count_matrix, pairs_count_matrix, total_shortest_paths_matrix, interval=interval)

        # Save paths
        save_shard(all_paths, os.path.join(save_dir, f'shortest_path/component_{component_id}/paths_{distance}', f'paths.pkl'))
        logging.critical(f"Generated {len(all_paths)} paths for shortest path distance {distance} takes {time.time()-s} seconds.")

        save_numpy(paths_count_matrix, os.path.join(save_dir, f'shortest_path/component_{component_id}/paths_{distance}', 'paths_count_matrix.npy'))
        save_numpy(pairs_count_matrix, os.path.join(save_dir, f'shortest_path/component_{component_id}/paths_{distance}', 'pairs_count_matrix.npy'))
        save_numpy(total_shortest_paths_matrix, os.path.join(save_dir, f'shortest_path/component_{component_id}/paths_{distance}', 'total_shortest_paths_matrix.npy'))
        total_count += np.sum(paths_count_matrix)
        del paths_count_matrix, pairs_count_matrix, total_shortest_paths_matrix
    logging.critical(f"Saved paths count matrix. Generated {total_count} paths in total.")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, "../../"))
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
    log_file = os.path.join(log_dir, 'gen_shortest_path.log')
    # Set up logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,               # Log level
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    os.makedirs(os.path.join(save_dir, 'shortest_path'), exist_ok=True)
    for component_id in range(2):
        if not os.path.exists(os.path.join(save_dir, 'map_stats', f'distance_to_pair_component_{component_id}.pkl')):
            component = pickle.load(open(os.path.join(save_dir, 'map_stats', f'component_{component_id}_shortest_paths.pkl'), 'rb'))
            distance_dict = get_all_node_pairs_by_distance(component)
            # Save the distance dictionary
            save_pickle(distance_dict, os.path.join(save_dir, 'map_stats', f'distance_to_pair_component_{component_id}.pkl'))
        else:
            with open(os.path.join(save_dir, 'map_stats', f'distance_to_pair_component_{component_id}.pkl'), 'rb') as f:
                distance_dict = pickle.load(f)
    
        if not os.path.exists(os.path.join(save_dir, 'shortest_path', f'component_{component_id}')):
            selected_train_pairs, test_pairs, train_pair_matrix = split_pairs(distance_dict, nodes_to_indices)
            os.makedirs(os.path.join(save_dir, 'shortest_path', f'component_{component_id}'), exist_ok=True)
            save_pickle(selected_train_pairs, os.path.join(save_dir, 'shortest_path', f'component_{component_id}', 'train_pairs.pkl'))
            save_pickle(test_pairs, os.path.join(save_dir, 'shortest_path', f'component_{component_id}', 'test_pairs.pkl'))
            save_pickle(train_pair_matrix, os.path.join(save_dir, 'shortest_path', f'component_{component_id}', 'train_pair_matrix.pkl'))
        else:
            with open(os.path.join(save_dir, 'shortest_path', f'component_{component_id}', 'train_pairs.pkl'), 'rb') as f:
                selected_train_pairs = pickle.load(f)
        # G = nx.from_numpy_array(adj_matrix)
        # adjacency_list = prepare_graph(G)
        generate_paths(nodes_to_indices, indices_to_nodes, selected_train_pairs, adj_matrix, component_id, interval=stats['interval'])