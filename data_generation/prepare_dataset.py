"""
Unified dataset preparation: tokenize shortest-path pickle files into HuggingFace datasets.

Supports all experiment modes via --mode:
  - base:        Base shortest path dataset from component_0 (pretraining baseline)
  - cov_div:     Coverage-diversity experiment datasets (Section 4.2)
  - qa:          Question-answer tradeoff datasets (Section 4.1)
  - longshort:   Length scaling rescue datasets (Section 5)
  - spatial:     Spatial transfer pair test datasets

Usage:
  python prepare_dataset.py --mode base --max_len 27
  python prepare_dataset.py --mode cov_div --diversity 64 --coverage 0.6
  python prepare_dataset.py --mode qa --coverage 0.8
  python prepare_dataset.py --mode longshort
  python prepare_dataset.py --mode spatial --coverage 1.0
"""

import os
import sys
import torch
import pickle
import re
import time
import argparse
import fnmatch
import numpy as np
from datasets import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils
from utils import save_pickle, save_numpy, load_config


def get_or_create_tokenizer(tokenizer_path, nodes_to_indices):
    if not os.path.exists(tokenizer_path):
        node_tokens = list(map(str, nodes_to_indices.values()))
        direction_tokens = ['N', 'S', 'W', 'E', 'STAY']
        special_tokens = ['<s>', '<pad>', '</s>', ':']
        vocab = node_tokens + direction_tokens + special_tokens
        vocab_dict = {token: idx for idx, token in enumerate(vocab)}
        tokenizer = utils.DirectionTokenizer(vocab_dict)
        torch.save(tokenizer, tokenizer_path)
    else:
        tokenizer = torch.load(tokenizer_path)
    return tokenizer


def tokenize_paths_dict(formatted_paths_dict, tokenizer, save_path):
    """Tokenize a dict of {(start,end): [records]} and save as HF dataset."""
    os.makedirs(save_path, exist_ok=True)
    all_tokenized_records = []

    for (start, end), records in formatted_paths_dict.items():
        for record in records:
            input_ids_reveal = tokenizer.encode(record["direction_with_mid_seq"])
            input_ids = tokenizer.encode(record["direction_seq"])

            all_tokenized_records.append({
                'input_ids_reveal': input_ids_reveal,
                'attention_mask_reveal': [1] * len(input_ids_reveal),
                'labels_reveal': input_ids_reveal,
                'input_ids': input_ids,
                'attention_mask': [1] * len(input_ids),
                'labels': input_ids,
                'start_idx': record['start_idx'],
                'end_idx': record['end_idx'],
                'index_path': ' '.join([str(node) for node in record["index_paths"]]),
                'start_coord': str(record['start_node']),
                'end_coord': str(record['end_node']),
                'direction_seq': record['direction_seq'],
                'direction_seq_reveal': record['direction_with_mid_seq'],
                'sp_distance': record['sp_distance'],
                'coord_distance': record['coord_distance'],
            })

    s = time.time()
    hf_dataset = Dataset.from_list(all_tokenized_records)
    hf_dataset.save_to_disk(os.path.join(save_path, "hf_dataset"))
    print(f"Saved {len(all_tokenized_records)} records to {save_path}. Took {time.time() - s:.1f}s")
    return all_tokenized_records


def tokenize_flat_records(data, tokenizer, nodes_to_indices, save_path, adj_matrix=None, test_pairs=None, train_pair_matrix=None, train_pairs=None):
    """Tokenize a flat list of records (used by base mode) and save as HF dataset."""
    os.makedirs(save_path, exist_ok=True)

    def data_generator():
        for record in data:
            input_ids_reveal = tokenizer.encode(record["direction_with_mid_seq"])
            input_ids = tokenizer.encode(record["direction_seq"])
            yield {
                'input_ids_reveal': input_ids_reveal,
                'attention_mask_reveal': [1] * len(input_ids_reveal),
                'labels_reveal': input_ids_reveal,
                'input_ids': input_ids,
                'attention_mask': [1] * len(input_ids),
                'labels': input_ids,
                'start_idx': record['start_idx'],
                'end_idx': record['end_idx'],
                'index_path': ' '.join([str(node) for node in record["index_paths"]]),
                'start_coord': str(record['start_node']),
                'end_coord': str(record['end_node']),
                'direction_seq': record['direction_seq'],
                'direction_seq_reveal': record['direction_with_mid_seq'],
                'sp_distance': record['sp_distance'],
                'coord_distance': record['coord_distance'],
            }

    train_dataset = Dataset.from_generator(data_generator)
    train_dataset.save_to_disk(save_path)

    if test_pairs is not None:
        save_pickle(test_pairs, os.path.join(save_path, 'test_pairs.pkl'))
    if train_pair_matrix is not None:
        save_pickle(train_pair_matrix, os.path.join(save_path, 'train_pair_matrix.pkl'))
    if train_pairs is not None:
        save_pickle(train_pairs, os.path.join(save_path, 'train_pairs.pkl'))
    if adj_matrix is not None:
        np.save(os.path.join(save_path, 'adj_matrix.npy'), adj_matrix)

    print(f"Saved {len(train_dataset)} records to {save_path}")


# ============================================================
# Mode-specific logic
# ============================================================

def prepare_base(args, tokenizer):
    """Base shortest path dataset from component_0."""
    dataset_dir = os.path.join(args.data_root, args.dataset, f"component_{args.component}")

    data = []
    for file_name in os.listdir(dataset_dir):
        if fnmatch.fnmatch(file_name, "paths_*"):
            extract_dist = int(re.search(r'\d+', file_name).group())
            if extract_dist <= args.max_len:
                file_path = os.path.join(dataset_dir, file_name, 'paths.pkl')
                with open(file_path, "rb") as f:
                    records = list(pickle.load(f).values())
                    data.extend([record[0] for record in records])
                print(f"Loaded {file_name} - {len(records)} records")
    print(f"Loaded total {len(data)} records")

    with open(os.path.join(dataset_dir, 'test_pairs.pkl'), 'rb') as f:
        test_pairs = pickle.load(f)
    with open(os.path.join(dataset_dir, 'train_pair_matrix.pkl'), 'rb') as f:
        train_pair_matrix = pickle.load(f)
    with open(os.path.join(dataset_dir, 'train_pairs.pkl'), 'rb') as f:
        train_pairs = pickle.load(f)

    save_dir = os.path.join(args.data_root, f'shortest_path_max-{args.max_len}')
    tokenize_flat_records(data, tokenizer, args.nodes_to_indices, save_dir,
                          adj_matrix=args.adj_matrix, test_pairs=test_pairs,
                          train_pair_matrix=train_pair_matrix, train_pairs=train_pairs)


def prepare_cov_div(args, tokenizer):
    """Coverage-diversity experiment datasets."""
    data_dir = os.path.join(
        args.data_root, '_diversity_coverage',
        f'diversity_{args.diversity}',
        f'coverage_ratio_{args.coverage:.2f}',
        f'pairs_{args.pairs_idx}/{args.dataset}'
    )
    with open(os.path.join(data_dir, 'paths.pkl'), 'rb') as f:
        all_paths = pickle.load(f)
    tokenize_paths_dict(all_paths, tokenizer, data_dir)


def prepare_qa(args, tokenizer):
    """Question-answer tradeoff datasets."""
    data_dir = os.path.join(
        args.data_root, '_spatial_length',
        f'coverage_ratio_{args.coverage:.2f}',
        f'pairs_{args.pairs_idx}/{args.dataset}/tradeoff_datasets'
    )
    pattern = re.compile(r'paths_.*\.pkl')
    for pkl_file in os.listdir(data_dir):
        if not pattern.match(pkl_file):
            continue
        if "64" not in pkl_file:
            continue
        print(f"Processing {pkl_file}...")
        with open(os.path.join(data_dir, pkl_file), 'rb') as f:
            all_paths = pickle.load(f)
        save_name = pkl_file.split('_B')[0]
        tokenize_paths_dict(all_paths, tokenizer, os.path.join(data_dir, save_name))


def prepare_longshort(args, tokenizer):
    """Length scaling rescue datasets."""
    spatial_dir = os.path.join(args.data_root, '_spatial_length')
    dataset_dict = pickle.load(open(os.path.join(spatial_dir, "distance_group_to_train_pairs.pkl"), 'rb'))
    for group in dataset_dict:
        pkl_file = os.path.join(spatial_dir, f"longshort_pairs/group_{group}/paths.pkl")
        paths = pickle.load(open(pkl_file, 'rb'))
        output_dir = os.path.join(spatial_dir, f"longshort_pairs/group_{group}/")
        tokenize_paths_dict(paths, tokenizer, output_dir)


def prepare_spatial(args, tokenizer):
    """Spatial transfer pair test datasets."""
    if args.dataset == 'core_set':
        data_dir = os.path.join(args.data_root, '_spatial_length', 'core_pairs/shortest_path')
    else:
        data_dir = os.path.join(
            args.data_root, '_spatial_length',
            f'coverage_ratio_{args.coverage:.2f}',
            f'pairs_{args.pairs_idx}/{args.dataset}'
        )
    with open(os.path.join(data_dir, 'paths.pkl'), 'rb') as f:
        all_paths = pickle.load(f)
    tokenize_paths_dict(all_paths, tokenizer, data_dir)


def main():
    parser = argparse.ArgumentParser(description="Prepare HuggingFace datasets from shortest-path pickle files.")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['base', 'cov_div', 'qa', 'longshort', 'spatial'])
    parser.add_argument('--dataset', type=str, default='shortest_path')
    parser.add_argument('--tokenizer_path', type=str, default='models/tokenizer.pth')

    # base mode
    parser.add_argument('--component', type=int, default=0)
    parser.add_argument('--max_len', type=int, default=27)

    # cov_div mode
    parser.add_argument('--diversity', type=int, default=64)

    # shared
    parser.add_argument('--coverage', type=float, default=0.6)
    parser.add_argument('--pairs_idx', type=int, default=0)

    args = parser.parse_args()

    # Load config and map data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
    config_path = os.path.join(parent_dir, "config.yaml")

    config = load_config(config_path)
    data_root = os.path.join(parent_dir, config["dataset"]["dir"])

    with open(os.path.join(data_root, "map_stats", 'nodes_to_indices.pkl'), 'rb') as f:
        nodes_to_indices = pickle.load(f)
    with open(os.path.join(data_root, "map_stats", 'indices_to_nodes.pkl'), 'rb') as f:
        indices_to_nodes = pickle.load(f)
    adj_matrix = np.load(os.path.join(data_root, "map_stats", 'adj_matrix.npy'))

    args.data_root = data_root
    args.nodes_to_indices = nodes_to_indices
    args.indices_to_nodes = indices_to_nodes
    args.adj_matrix = adj_matrix

    tokenizer = get_or_create_tokenizer(args.tokenizer_path, nodes_to_indices)

    mode_fn = {
        'base': prepare_base,
        'cov_div': prepare_cov_div,
        'qa': prepare_qa,
        'longshort': prepare_longshort,
        'spatial': prepare_spatial,
    }
    mode_fn[args.mode](args, tokenizer)


if __name__ == "__main__":
    main()
