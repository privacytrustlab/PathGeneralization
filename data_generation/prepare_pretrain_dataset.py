import os
import torch
import pickle
import re
import argparse
from datasets import Dataset, DatasetDict
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import utils
from utils import save_pickle, save_numpy, load_config, save_shard
from utils import get_indices, get_nodes, get_valid_neighbors

def main(args):
    """ ======================== Prepare Tokenizer ======================== """
    if not os.path.exists(args.tokenizer_path):
        os.makedirs(os.path.dirname(args.tokenizer_path), exist_ok=True)
        node_tokens = list(map(str, args.nodes_to_indices.values()))
        direction_tokens = ['N', 'S', 'W', 'E', 'STAY']
        special_tokens = ['<s>', '<pad>', '</s>', ':']
        vocab = node_tokens + direction_tokens + special_tokens
        vocab_dict = {token: idx for idx, token in enumerate(vocab)}
        tokenizer = utils.DirectionTokenizer(vocab_dict)
        torch.save(tokenizer, args.tokenizer_path)
    else:
        tokenizer = torch.load(args.tokenizer_path)

    """ ======================== Construct Dataset ======================== """
    def get_data(selected_shards, data_dir):
        data = []
        emp_matrix = np.zeros((len(args.nodes_to_indices), len(args.nodes_to_indices)), dtype=int)
        coord_distance_ratio = []
        shortest_path_length_ratio = []
        for shard_idx in selected_shards:
            file_path = os.path.join(data_dir, f'shard_{shard_idx}/paths.bin')
            
            with open(file_path, 'rb') as f:
                file_data = pickle.load(f)
                data.extend(file_data)
            
            stats = np.load(os.path.join(data_dir, f'shard_{shard_idx}/stats.npz'))
            emp_matrix += stats['emp_matrix']
            coord_distance_ratio.extend(stats['coord_distance_ratio'].tolist())
            shortest_path_length_ratio.extend(stats['shortest_path_length_ratio'].tolist())
        
        return data, emp_matrix, coord_distance_ratio, shortest_path_length_ratio

    if args.path_len == 'mixed':
        assert args.num_train_samples % args.num_per_shard == 0, "Number of training samples must be divisible by num_per_shard."
        num_shards = args.num_train_samples // args.num_per_shard
        assert num_shards % 2 == 0, "Number of shards must be even for mixed lengths."
        shard_train_count = num_shards // 2  # half for 99 and half for 100
        shard_test_count = 2 # in total 4 shards for 99 and 100
        selected_shards_train = list(range(shard_train_count))
        selected_shards_test = list(range(shard_train_count, shard_train_count + shard_test_count))  # Assuming test shards are 10 and 11
        dataset_dir = os.path.join(args.data_dir,  args.dataset, f"fix_len_99")
        train_paths_99, train_emp_matrix_99, train_coord_distance_ratio_99, train_shortest_path_length_ratio_99 = get_data(selected_shards_train, dataset_dir)
        test_paths_99, test_emp_matrix_99, test_coord_distance_ratio_99, test_shortest_path_length_ratio_99 = get_data(selected_shards_test, dataset_dir)

        dataset_dir = os.path.join(args.data_dir, args.dataset, f"fix_len_100")
        train_paths_100, train_emp_matrix_100, train_coord_distance_ratio_100, train_shortest_path_length_ratio_100 = get_data(selected_shards_train, dataset_dir)
        test_paths_100, test_emp_matrix_100, test_coord_distance_ratio_100, test_shortest_path_length_ratio_100 = get_data(selected_shards_test, dataset_dir)

        # Combine the two datasets (dict)
        train_paths = train_paths_99 + train_paths_100
        test_paths = test_paths_99 + test_paths_100
        
        train_matrix = train_emp_matrix_99 + train_emp_matrix_100
        test_matrix = test_emp_matrix_99 + test_emp_matrix_100
        train_coord_distance_ratio = train_coord_distance_ratio_99 + train_coord_distance_ratio_100
        train_shortest_path_length_ratio = train_shortest_path_length_ratio_99 + train_shortest_path_length_ratio_100
        test_coord_distance_ratio = test_coord_distance_ratio_99 + test_coord_distance_ratio_100
        test_shortest_path_length_ratio = test_shortest_path_length_ratio_99 + test_shortest_path_length_ratio_100
        stats = {
            'train_matrix': train_matrix,
            'test_matrix': test_matrix,
            'train_coord_distance_ratio': train_coord_distance_ratio,
            'train_shortest_path_length_ratio': train_shortest_path_length_ratio,
            'test_coord_distance_ratio': test_coord_distance_ratio,
            'test_shortest_path_length_ratio': test_shortest_path_length_ratio
        }

    else:
        raise ValueError("Invalid path length specified. Use 'mixed' for mixed lengths of 99 and 100.")

    """ ======================== Preprocess Dataset ======================== """
    def data_generator(data, tokenizer):
        """
        A single generator function that processes input data using a tokenizer.
        """
        for record in data:
            input_ids = tokenizer.encode(record["direction_seq"])
            attention_mask = [1] * len(input_ids)

            input_ids_reveal = tokenizer.encode(record["direction_with_mid_seq"])
            attention_mask_reveal = [1] * len(input_ids_reveal)
            record = dict(record)  # make a shallow copy to avoid mutating original data
            record.update({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': input_ids,  # or a different label field
                'input_ids_reveal': input_ids_reveal,
                'attention_mask_reveal': attention_mask_reveal,
                'labels_reveal': input_ids_reveal
            })
            record['start_coord'] = str(record['start_coord'])
            record['end_coord'] = str(record['end_coord'])
            record['direction_paths'] = [str(item) for item in record['direction_paths']]
            record['directions_with_mid'] = [str(item) for item in record['directions_with_mid']]
            record['coord_paths'] = [str(item) for item in record['coord_paths']]

            yield record

    def convert_to_hf_dataset(train_data, test_data, tokenizer, save_path, stats=None):
        # Create Dataset objects from the single generator function
        train_dataset = Dataset.from_generator(lambda: data_generator(train_data, tokenizer))
        test_dataset = Dataset.from_generator(lambda: data_generator(test_data, tokenizer))

        # Combine train and test datasets into a DatasetDict
        combined_dataset = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })

        # Save the combined DatasetDict to disk
        os.makedirs(save_path, exist_ok=True)
        combined_dataset.save_to_disk(save_path)

        # Save the stats
        if stats is not None:
            np.savez(os.path.join(save_path, 'stats.npz'), 
                     train_matrix=stats['train_matrix'], 
                     test_matrix=stats['test_matrix'],
                     train_coord_distance_ratio=stats['train_coord_distance_ratio'], 
                     train_shortest_path_length_ratio=stats['train_shortest_path_length_ratio'],
                     test_coord_distance_ratio=stats['test_coord_distance_ratio'], 
                     test_shortest_path_length_ratio=stats['test_shortest_path_length_ratio'])
        
        return combined_dataset

    if args.path_len == 'mixed':
        if args.num_train_samples == 10000000:
            dataset_name = 'random_walk_10M'
        elif args.num_train_samples == 5000000:
            dataset_name = 'random_walk_5M'
            
    save_dir = os.path.join(args.data_dir, dataset_name)
    hf_datasets = convert_to_hf_dataset(train_paths, test_paths, tokenizer, save_dir, stats)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, "../../"))
    config_path = os.path.join(parent_dir, "config.yaml")

    config = load_config(config_path)
    save_dir = config["dataset"]["dir"]
    save_dir = os.path.join(parent_dir, save_dir)

    # load the grid and adjacency matrix
    with open(os.path.join(save_dir, "map_stats", 'nodes_to_indices.pkl'), 'rb') as f:
        nodes_to_indices = pickle.load(f)
    with open(os.path.join(save_dir, "map_stats", 'indices_to_nodes.pkl'), 'rb') as f:
        indices_to_nodes = pickle.load(f)
    adj_matrix = np.load(os.path.join(save_dir, "map_stats", 'adj_matrix.npy'))
    nodes_coords = list(nodes_to_indices.keys())

    parser = argparse.ArgumentParser(description="Prepare data and upload to Hugging Face Hub")

    # Add arguments
    parser.add_argument('--dataset', type=str, default='random_walk', help='Name of the dataset: [random_walk, shortest_path]', choices=['random_walk', 'shortest_path'])
    parser.add_argument('--tokenizer_path', type=str, default='models/tokenizer.pth', help='Path to save/load the tokenizer')
    parser.add_argument('--num_train_samples', type=int, default=10000000, help='Number of training samples')
    parser.add_argument('--num_test_samples', type=int, default=500000, help='Number of testing samples')
    parser.add_argument('--path_len', type=str, default='mixed', help='Path length for random walk dataset', choices=['99', '100', 'mixed'])

    args = parser.parse_args()

    args.data_dir = save_dir
    args.nodes_to_indices = nodes_to_indices
    args.indices_to_nodes = indices_to_nodes
    args.adj_matrix = adj_matrix
    args.nodes_coords = nodes_coords
    args.num_per_shard = config["dataset"][args.dataset]["shard_num"]

    main(args)
