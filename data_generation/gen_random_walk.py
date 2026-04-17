import random
import sys
import pickle
import numpy as np
from collections import defaultdict
import os
import logging
import networkx as nx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils import save_pickle, save_numpy, load_config, save_shard
from utils import get_indices, get_nodes, get_valid_neighbors

def generate_single_path(start_coord, nodes_to_indices, G, adj_matrix, min_len, max_len, interval, size_m, size_n):
    """Generates a random walk starting from a coordinate"""
    if min_len == max_len:
        length = min_len
    else:
        length = random.randint(min_len, max_len)

    start_index = nodes_to_indices[start_coord]
    index_paths = [start_index]
    directions = [start_index]
    directions_with_mid = [start_index]
    coord_paths = [start_coord]
    
    # generate a random path
    current = start_coord
    prev_coord = None
    for move in range(1,length+1):
        neighbors = get_valid_neighbors(current, adj_matrix, nodes_to_indices, size_m, size_n)
        
        # Select a random valid neighbor
        next_coord, direction = random.choice(neighbors)
        index_paths.append(nodes_to_indices[next_coord])
        directions.append(direction)
        directions_with_mid.append(direction)
        coord_paths.append(next_coord)
        if move % interval == 0 and move != length:
            directions_with_mid.append(nodes_to_indices[next_coord])
        prev_coord = current
        current = next_coord

    end_coord = next_coord
    end_index = nodes_to_indices[end_coord]

    directions.append(end_index)
    directions_with_mid.append(end_index)
    return {'start_coord': start_coord, 
            'end_coord': end_coord, 
            'start_index': start_index,
            'end_index': end_index,
            'index_paths': index_paths, 
            'coord_paths': coord_paths,
            'direction_paths': directions, 
            'directions_with_mid': directions_with_mid,
            'coord_distance': np.linalg.norm(np.array(start_coord) - np.array(end_coord)),
            'shortest_path_length': nx.shortest_path_length(G, start_index, end_index),
            'length': length}

def generate_random_walk_shard(shard_num, nodes_to_indices, indices_to_nodes, nodes_coords, adj_matrix, stats):
    if stats['fix_len']:
        max_len = stats['max_len']
        min_len = max_len
    else:
        max_len = stats['max_len']
        assert max_len > 50, "max_len must be greater than 50"
        min_len = random.randint(50, stats['max_len'])
    interval = stats['interval']
    size_m, size_n = stats['size_m'], stats['size_n']

    paths = []
    coord_distance_ratio = []
    shortest_path_length_ratio = []
    # initialize the empty matrix
    emp_matrix = np.zeros(adj_matrix.shape, dtype=int)
    G = nx.from_numpy_array(adj_matrix)

    for path_idx in range(shard_num):
        start_coord = random.choice(nodes_coords)
        path = generate_single_path(start_coord, nodes_to_indices, G, adj_matrix, min_len, max_len, interval, size_m, size_n)
        
        start_idx, end_idx = path['start_index'], path['end_index']
        direction_seq = ' '.join([str(node) for node in path['direction_paths']])
        direction_with_mid_seq = ' '.join([str(node) for node in path['directions_with_mid']])

        path['direction_seq'] = f"<s> {start_idx} {end_idx} : {direction_seq} </s>"
        path['direction_with_mid_seq'] = f"<s> {start_idx} {end_idx} : {direction_with_mid_seq} </s>"
        paths.append(path)

        emp_matrix[start_idx, end_idx] += 1
        coord_distance_ratio.append(float(path['coord_distance']/path['length']))
        shortest_path_length_ratio.append(float(path['shortest_path_length']/path['length']))
        if path_idx % 1000 == 0:
            logging.info(f"Generated {path_idx+1} paths.")
    
    return paths, emp_matrix, coord_distance_ratio, shortest_path_length_ratio


def generate_paths(n_paths, nodes_to_indices, indices_to_nodes, nodes_coords, adj_matrix, stats, dataset_save_dir, shard_num=50000):
    """Generates paths until n_paths is reached"""
    # Generate shard_num random walks at a time
    if stats['fix_len']:
        name = f"fix_len_{stats['max_len']}"
    else:
        name = f"random_len_50-to-{stats['max_len']}"
    for i in range(n_paths // shard_num):
        indices_dir = os.path.join(dataset_save_dir, name, f'shard_{i}')
        paths, emp_matrix, coord_distance_ratio, shortest_path_length_ratio = generate_random_walk_shard(shard_num, nodes_to_indices, indices_to_nodes, nodes_coords, adj_matrix, stats)
        
        save_shard(paths, os.path.join(indices_dir, 'paths.bin'))
        np.savez(os.path.join(indices_dir, 'stats.npz'), 
                    coord_distance_ratio=coord_distance_ratio, 
                    shortest_path_length_ratio=shortest_path_length_ratio,
                    emp_matrix=emp_matrix)

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, "../../"))
    config_path = os.path.join(parent_dir, "config.yaml")

    config = load_config(config_path)
    save_dir = config["dataset"]["dir"]
    save_dir = os.path.join(parent_dir, save_dir)
    n_paths = config["dataset"]["random_walk"]["n_paths"]
    shard_num = config["dataset"]["random_walk"]["shard_num"]
    single_size_m, single_size_n = config["dataset"]["grid"]["size_m"], config["dataset"]["grid"]["size_n"]
    size_m = single_size_m
    size_n = single_size_n * config["dataset"]["grid"]["num_grids"]
    max_len = config["dataset"]["random_walk"]["max_len"]
    assert isinstance(max_len, int) or isinstance(max_len, list), "max_len must be an integer or a list of integers"
    if isinstance(max_len, int):
        max_len = [max_len]
    
    # load the grid and adjacency matrix
    with open(os.path.join(save_dir, "map_stats", 'nodes_to_indices.pkl'), 'rb') as f:
        nodes_to_indices = pickle.load(f)
    with open(os.path.join(save_dir, "map_stats", 'indices_to_nodes.pkl'), 'rb') as f:
        indices_to_nodes = pickle.load(f)
    adj_matrix = np.load(os.path.join(save_dir, "map_stats", 'adj_matrix.npy'))
    nodes_coords = list(nodes_to_indices.keys())

    for leng in max_len:
        stats = {'fix_len': config["dataset"]["random_walk"]["fix_len"], 'max_len': leng, 'interval': config["dataset"]["random_walk"]["interval"], 'size_m': size_m, 'size_n': size_n}

        log_dir = config["logdir"]
        data_log_dir = os.path.join(log_dir, "data_gen")
        os.makedirs(data_log_dir, exist_ok=True)
        log_file = os.path.join(data_log_dir, f'gen_random_walk_max_len-{leng}.log')
        # Set up logging
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,               # Log level
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
        # Generate the paths
        assert n_paths % shard_num == 0, "n_paths must be divisible by shard_num"
        logging.info(f"Generating {n_paths} random walk paths with max length {leng} and interval {stats['interval']}")
        generate_paths(n_paths, nodes_to_indices, indices_to_nodes, nodes_coords, adj_matrix, stats, dataset_save_dir=os.path.join(save_dir, 'random_walk'), shard_num=shard_num)
