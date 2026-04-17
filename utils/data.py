import random
import os
from datasets import Dataset
import torch
import ast
import numpy as np
import heapq

# Split paths into training and test sets
def split_dataset(paths, train_ratio=0.9):
    path_items = list(paths.items())
    random.shuffle(path_items)
    split_idx = int(len(path_items) * train_ratio)
    
    train_paths = dict(path_items[:split_idx])
    test_paths = dict(path_items[split_idx:])
    
    return train_paths, test_paths

def get_indices(G):
    """Returns the indices of the nodes"""
    return {idx: node for idx, node in enumerate(G.nodes())}

def get_nodes(G):
    """Returns the nodes of the graph"""
    return {node: idx for idx, node in enumerate(G.nodes())}

def get_neighbors(node, m_size, n_size):
    """Returns the neighboring nodes and corresponding directions for a given node
    Returns the neighboring nodes and corresponding directions for a given node in an m x n grid.

    Parameters:
    - node (tuple): (i, j) coordinates of the node.
    - m_size (int): Number of rows in the grid.
    - n_size (int): Number of columns in the grid.

    Returns:
    - List of tuples [(neighbor_node, direction), ...]
    """
    i, j = node
    neighbors = []

    if i > 0:         # West (Left)
        neighbors.append(((i - 1, j), 'W'))
    if i < m_size - 1: # East (Right)
        neighbors.append(((i + 1, j), 'E'))
    if j > 0:         # South (Down)
        neighbors.append(((i, j - 1), 'S'))
    if j < n_size - 1: # North (Up)
        neighbors.append(((i, j + 1), 'N'))
    return neighbors

def get_valid_neighbors(node, adj_matrix, nodes_to_indices, m_size, n_size):
    """Returns the valid neighboring nodes for a given node"""
    # get the type of key in the dictionary
    key_type = type(list(nodes_to_indices.keys())[0])
    neighbors = get_neighbors(node, m_size, n_size)
    valid_neighbors = []
    for neighbor, direction in neighbors:
        if key_type == str:
            node, neighbor = str(node), str(neighbor)
        elif key_type == tuple:
            if not isinstance(node, tuple):
                node = ast.literal(node)
            if not isinstance(neighbor, tuple):
                neighbor = ast.literal(neighbor)
        if adj_matrix[nodes_to_indices[node], nodes_to_indices[neighbor]] == 1:
            valid_neighbors.append((neighbor, direction))
    return valid_neighbors

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

def get_idx_path(i, text, tokenizer, nodes, idx_to_node, m_size, n_size):
    text_list = text.split(" ")
    start_node_idx, end_node_idx = int(text_list[1]), int(text_list[2])
    directions = text_list[5:]
    current_state = start_node_idx
    state_seq = [current_state]
    for i, direction in enumerate(directions):
        if direction != tokenizer.eos_token and not direction.isdigit():
            current_node = idx_to_node[current_state]
            neighbors = get_neighbors(current_node, m_size, n_size)
            valid_turns = [neighbor[1] for neighbor in neighbors]
            if direction in valid_turns:
                next_node_at = valid_turns.index(direction)
                current_state_node = neighbors[next_node_at][0]
                current_state = nodes[current_state_node]
                state_seq.append(current_state)
            else:
                print(f"Invalid sequence {i}")
        else:
            if direction == tokenizer.eos_token and current_state == end_node_idx:
                return state_seq
            elif direction == str(current_state):
                continue
            else:
                print(f"Invalid sequence {i}")

def dijkstra_all_shortest_paths(adj_matrix, start, end, num_nodes):
    """
    Optimized Dijkstra's algorithm modified to find all shortest paths.
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

# Construct data records
def construct_explicit_poisition_records(paths):
    """
    Format: start_idx end_idx directions
    Example: 262 397 : 262 N E N N S E 397
    """
    records = []
    for (start_idx, end_idx), path in paths.items():
        directions = ' '.join(path['directions'])
        record = f"<s> {start_idx}  {end_idx} : {start_idx} {directions} {end_idx} </s>"
        records.append(record)
    return records

def construct_implicit_poisition_records(paths):
    """
    Format: start_idx end_idx mid_indices
    Example: 262 397 512 463 997 123
    """
    records = []
    for (start_idx, end_idx), path in paths.items():
        mid_indices = ' '.join(map(str, path['path_nodes']))
        record = f"{start_idx} {end_idx} {mid_indices} end"
        records.append(record)
    return records

def trans_explicit_records_to_nodes(paths, idx_to_node, get_neighbors):
    node_paths = []
    for sample in paths:
        sample = sample.split(" ")
        start_node_idx, end_node_idx = int(sample[0]), int(sample[1])
        node = idx_to_node[start_node_idx]
        new_sample = [node]
        for direction in sample[2:]:
            neighbors = get_neighbors(node)
            valid_turns = [neighbor[1] for neighbor in neighbors]
            if direction in valid_turns:
                next_node_at = valid_turns.index(direction)
                node = neighbors[next_node_at][0]
                new_sample.append(node)
        node_paths.append(new_sample)
    return node_paths

def trans_implicit_records_to_nodes(paths, idx_to_node, get_neighbors):
    node_paths = []
    for sample in paths:
        start_node_idx, end_node_idx = int(sample[0]), int(sample[1])
        node = idx_to_node[start_node_idx]
        node_path = [node]
        for node_idx in sample[2:]:
            node = idx_to_node[node_idx]
            node_path.append(node)
        node_paths.append(node_path)
    return node_paths


# Create a dictionary dataset from the records
class TextDataset(Dataset):
    def __init__(self, sequences, tokenizer):
        assert isinstance(sequences, list)
        self.sequences = sequences
        self.tokenizer = tokenizer
           
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            text = self.sequences[idx]
            input_ids = self.tokenizer.encode(text)  #tokenized input
            attention_mask = [1] * len(input_ids) 
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': input_ids 
            }
        elif isinstance(idx, (list, tuple)) and all(isinstance(i, int) for i in idx):
            texts = [self.sequences[i] for i in idx]
            org_input_ids = [self.tokenizer.encode(text) for text in texts]
            max_len = max(len(ids) for ids in org_input_ids)
            attention_mask = []
            input_ids = []
            for ids in org_input_ids:
                mask = [1] * len(ids)
                padding_length = max_len - len(ids)
                ids = ids + [self.tokenizer.pad_token_id] * padding_length
                mask = mask + [0] * padding_length
                attention_mask.append(mask)
                input_ids.append(ids)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': input_ids
            }
        else:
            raise ValueError("idx must be an integer or a list of integers")