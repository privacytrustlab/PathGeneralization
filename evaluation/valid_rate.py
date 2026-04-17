#!/usr/bin/env python3
"""
Valid Rate Analysis Script


This script evaluates the valid rate and performance of different models on graph navigation tasks.
"""

import logging
import argparse
import numpy as np
import sys
import torch
from tqdm import tqdm
import os
import pickle
import networkx as nx
from collections import defaultdict
from datasets import load_dataset, Dataset, DatasetDict
import ast
from copy import deepcopy
from transformers import PreTrainedTokenizerFast, PreTrainedModel, LlamaForCausalLM
from typing import Optional
from itertools import product
import json

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import utils
from utils import get_neighbors, get_valid_neighbors
from utils import save_pickle, save_numpy, load_config, save_shard

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(args, tokenizer, model_name, n_head, head_to_embd, idx_to_node, nodes, adj_matrix):
    """Load a model from checkpoint or path."""
    if '.ckpt' in model_name:
        model_path = model_name
    else:
        model_path = os.path.join(args.model_dir, f'{model_name}')

    model = utils.Model(tokenizer, n_embd=head_to_embd[n_head], n_layer=8, n_head=n_head, 
                       idx_to_node=idx_to_node, nodes=nodes, connectivity_matrix=adj_matrix, 
                       size_m=args.size_m, size_n=args.size_n*2)

    model = model.to(device)
    if model_path and os.path.exists(model_path):
        if '.ckpt' in model_path:
            model_ckpt_state_dict = torch.load(model_path)['state_dict']
            model.load_state_dict(model_ckpt_state_dict)
        elif '.pth' in model_path:
            model.load_state_dict(torch.load(model_path))
    elif os.path.exists(args.checkpoint_dir):
        model_ckpt_state_dict = torch.load(args.checkpoint_dir)['state_dict']
        model.load_state_dict(model_ckpt_state_dict)
    else:
        raise ValueError("No model found.")
    model.eval()
    return model

def get_prompts(test_pairs, adj_matrix, indices_to_nodes, hf_tokenizer):
    """
    Identify test pairs where train_pair_matrix[I, J] == 0.
    
    Args:
        test_pairs (list): A list of (I, J) node pairs.
        adj_matrix (np.ndarray): Adjacency matrix for the graph.
        indices_to_nodes (dict): A dictionary mapping indices to node coordinates.
        hf_tokenizer: HuggingFace tokenizer.
    
    Returns:
        list: A list of dictionaries containing test data.
    """
    test_data = []
    G = nx.from_numpy_array(adj_matrix)
    for start_index, end_index in test_pairs:
        record = {
            'start_index': start_index,
            'end_index': end_index,
            'prompt': f"<s> {start_index} {end_index} :",
            'coord_distance': np.linalg.norm(np.array(ast.literal_eval(indices_to_nodes[start_index])) - 
                                           np.array(ast.literal_eval(indices_to_nodes[end_index]))),
            'shortest_path_length': nx.shortest_path_length(G, source=start_index, target=end_index),
            'input_ids': hf_tokenizer.encode(f'<s> {start_index} {end_index} :', add_special_tokens=False),
        }
        test_data.append(record)
    
    return test_data

def check_valid(responses, tokenizer, nodes_to_indices, idx_to_node, connectivity_matrix, m_size, n_size):
    """Check if a response is valid path navigation."""
    generated_list = responses.split(" ")
    start_node_idx, end_node_idx = int(generated_list[0]), int(generated_list[-1])
    directions = generated_list[1:]
    current_state = start_node_idx
    state_seq = [current_state]
    for i, direction in enumerate(directions):
        if direction != tokenizer.eos_token and not direction.isdigit():
            current_node = ast.literal_eval(idx_to_node[current_state])
            neighbors = get_valid_neighbors(current_node, connectivity_matrix, nodes_to_indices, m_size, n_size)
            valid_turns = [neighbor[1] for neighbor in neighbors]
            if direction in valid_turns:
                next_node_at = valid_turns.index(direction)
                current_state_node = neighbors[next_node_at][0]
                current_state = nodes_to_indices[current_state_node]
                state_seq.append(current_state)
            else:
                return False, state_seq
        else:
            if current_state == end_node_idx:
                return True, state_seq
            elif direction == str(current_state):  # print valid current node
                continue
            else:
                return False, state_seq
    return False, state_seq

def decode_without_pad(hf_tokenizer, input_ids):
    """Decode tokens without padding tokens."""
    tokens = hf_tokenizer.convert_ids_to_tokens(input_ids)
    tokens = [t for t in tokens if t != hf_tokenizer.pad_token]
    return hf_tokenizer.convert_tokens_to_string(tokens)

def eval_sp_rate(args, G_length_buckets, model, tokenizer, nodes_to_indices, indices_to_nodes, adj_matrix):
    """Evaluate the validation rate for different length groups."""
    valid_rates, sp_rates, groups = [], [], []
    for length_group, test_pairs in G_length_buckets.items():
        logging.info(f"Testing length group {length_group} with {len(test_pairs)} pairs")
        valid_count = 0
        shortest_count = 0
        total_count = 0
        num_batches = int(np.ceil(len(test_pairs) / args.batch_size))
        with tqdm(total=len(test_pairs)) as pbar:
            for i in range(num_batches):
                if i == num_batches - 1:
                    batch = test_pairs[i * args.batch_size:]
                else:
                    batch = test_pairs[i * args.batch_size:(i + 1) * args.batch_size]
                input_ids = [item['input_ids'] for item in batch]
                input_ids = torch.tensor(input_ids).to(device)

                with torch.no_grad():
                    outputs = model.model.generate(
                        input_ids=input_ids,
                        max_length=512,
                        num_return_sequences=1, 
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=False
                    )
                
                # Decode the outputs
                if isinstance(tokenizer, PreTrainedTokenizerFast):
                    outputs = [decode_without_pad(tokenizer, ids) for ids in outputs]
                else:
                    outputs = list(map(tokenizer.decode, outputs))

                for j, response in enumerate(outputs):
                    total_count += 1
                    if response.split(" ")[-1] == tokenizer.eos_token:
                        response = response[:-len(tokenizer.eos_token)].strip()
                    if response.split(" ")[0] == tokenizer.bos_token:
                        response = response.split(" ")
                        response = response[4:]
                        if len(response) < 2:
                            continue
                        if not response[0] == str(batch[j]['start_index']) or not response[-1] == str(batch[j]['end_index']):
                            continue
                        response = " ".join(response)

                    is_valid, state_seq = check_valid(response, tokenizer, nodes_to_indices, indices_to_nodes, 
                                                    adj_matrix, args.size_m, args.size_n * 2)
                    is_shortest = (len(state_seq)-1 == batch[j]['shortest_path_length'])

                    if is_valid:
                        valid_count += 1
                        if is_shortest:
                            shortest_count += 1
                
                pbar.update(len(batch))
        valid_rates.append(valid_count / total_count if total_count > 0 else 0)
        sp_rates.append(shortest_count / total_count if total_count > 0 else 0)
        groups.append(length_group)
        logging.info(f"Length group {length_group} - Valid: {valid_count}/{total_count}, Shortest: {shortest_count}/{total_count}")

    return valid_rates, sp_rates, groups

class GRPOModelWrapper(PreTrainedModel):
    """Wrapper for GRPO model."""
    def __init__(self, model: LlamaForCausalLM, tokenizer=None):
        super().__init__(model.config)
        self.model = model
        self.tokenizer = tokenizer

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None,
                **kwargs):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_path, *args, **kwargs):
        # This will load the LlamaForCausalLM and its config
        llama_model = LlamaForCausalLM.from_pretrained(pretrained_path, *args, **kwargs)
        
        tokenizer = None
        try:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_path)
        except Exception as e:
            print(f"Warning: Could not load tokenizer from {pretrained_path}: {e}")
            pass  # silently skip if tokenizer is not needed or not found
            
        return cls(llama_model, tokenizer)

    def save_pretrained(self, save_directory: str, **kwargs):
        # Save the internal LlamaForCausalLM model directly
        self.model.save_pretrained(save_directory, **kwargs)
        
        # Save tokenizer if available
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_directory)

def main(args):
    """Main function to run the evaluation."""
    # Load graph data
    with open(os.path.join(args.dataset_dir, "map_stats", 'nodes_to_indices.pkl'), 'rb') as f:
        nodes_to_indices = pickle.load(f)
    with open(os.path.join(args.dataset_dir, "map_stats", 'indices_to_nodes.pkl'), 'rb') as f:
        indices_to_nodes = pickle.load(f)
    adj_matrix = np.load(os.path.join(args.dataset_dir, "map_stats", 'adj_matrix.npy'))

    indices_to_nodes = {k: str(v) for k, v in indices_to_nodes.items()}
    nodes_to_indices = {str(k): v for k, v in nodes_to_indices.items()}

    head_to_embd = {
        8: 512,
        12: 768,
        16: 1024,
        20: 1280,
        25: 1600
    }
    n_head = args.n_head

    # Define the tokenizer paths
    args.tokenizer_path = os.path.join(args.model_dir, 'tokenizer.pth')
    if args.mode == 'sft':
        ft_model_path = os.path.join(args.model_dir, f'pretrain_random_walk_10M_reveal/{args.mode}-len20-shortest_path_reveal_coverage_{args.coverage:.2f}_pairs_{args.pairs_idx}_ans{args.num_ans}')
    else:
        ft_model_path = os.path.join(args.model_dir, f'{args.mode}-len20-shortest_path_reveal_coverage_{args.coverage:.2f}_num_ans_{args.num_ans}_pairs_{args.pairs_idx}_num_generation_{args.num_generation}_from_ckpt_{args.ckpt_idx}')
    # add .pth if args.mode == 'sft'
    if args.mode == 'sft' and not ft_model_path.endswith('.pth'):
        ft_model_path += '.pth'
    log_file = os.path.join(args.log_dir, f'spatial_length/{args.dataset_name}/num_ans_{args.num_ans}/coverage_{args.coverage:.2f}/pairs_{args.pairs_idx}/', f'{args.mode}_rate.log')
    if args.mode == 'grpo':
        log_file = log_file.replace('.log', f'_from_ckpt_{args.ckpt_idx}_num_generation_{args.num_generation}.log')

    if not os.path.exists(ft_model_path):
        raise FileNotFoundError(f"Model not found at {ft_model_path}. Please check the path.")
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Loading tokenizer...")
    tokenizer = torch.load(args.tokenizer_path)
    hf_tokenizer_path = os.path.join(args.model_dir, 'hf_tokenizer')
    hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(hf_tokenizer_path)

    """ Prepare test data """
    logging.info("Preparing test data...")
    # load test nodes in G1
    data_dir = os.path.join(args.dataset_dir, '_spatial_length', f'test_indices.pkl')
    with open(data_dir, 'rb') as f:
        test_nodes = pickle.load(f)

    if os.path.exists(os.path.join(args.dataset_dir, '_spatial_length/shortest_path', 'G2_length_buckets.pkl')):
        with open(os.path.join(args.dataset_dir, '_spatial_length/shortest_path', 'G1_length_buckets.pkl'), 'rb') as f:
            G1_length_buckets = pickle.load(f)

        with open(os.path.join(args.dataset_dir, '_spatial_length/shortest_path', 'G2_length_buckets.pkl'), 'rb') as f:
            G2_length_buckets = pickle.load(f)
    else:
        print("G1_length_buckets and G2_length_buckets not found. Please run the path generation script first.")
        # Form pairs from test nodes
        test_pairs = []
        for i in range(len(test_nodes)):
            for j in range(i + 1, len(test_nodes)):
                if i != j:
                    test_pairs.append((test_nodes[i], test_nodes[j]))

        # Get G2 indices and create graph
        G2_indices = pickle.load(open(os.path.join(args.dataset_dir, 'map_stats', 'G2_indices.pkl'), 'rb'))
        G = nx.from_numpy_array(adj_matrix)

        # Get the test node pairs in G2
        if args.dataset_name == 'shortest_path':
            if os.path.exists(os.path.join(args.dataset_dir, '_spatial_length', 'G2_testset')):
                G2_test_dataset = Dataset.load_from_disk(os.path.join(args.dataset_dir, '_spatial_length', 'G2_testset'))
            else:
                G2_test_pairs = []
                for i, j in product(G2_indices, G2_indices):
                    if i < j:  # Ensure unique pairs (i, j) where i < j
                        G2_test_pairs.append({
                            'start_index': i,
                            'end_index': j,
                            'coord_distance': np.linalg.norm(np.array(ast.literal_eval(indices_to_nodes[i])) - 
                                                        np.array(ast.literal_eval(indices_to_nodes[j]))),
                            'shortest_path_length': nx.shortest_path_length(G, i, j),
                            'prompt': f'<s> {i} {j} :',
                            'input_ids': hf_tokenizer.encode(f'<s> {i} {j} :', add_special_tokens=False),
                        })
                # Save G2 test pairs
                G2_test_dataset = Dataset.from_list(G2_test_pairs)
                G2_test_dataset.save_to_disk(os.path.join(args.dataset_dir, '_spatial_length' 'G2_testset'))

        # Get the test node pairs in G1
        if os.path.exists(os.path.join(args.dataset_dir, '_spatial_length', 'G1_testset')):
            G1_test_dataset = Dataset.load_from_disk(os.path.join(args.dataset_dir, '_spatial_length', 'G1_testset'))
        else:
            G1_test_dataset = get_prompts(test_pairs, adj_matrix, indices_to_nodes, hf_tokenizer)
            G1_test_dataset = Dataset.from_list(G1_test_dataset)
            G1_test_dataset.save_to_disk(os.path.join(args.dataset_dir, '_spatial_length', 'G1_testset'))

        # Turn dataset into lists
        G1_test_pairs = G1_test_dataset.to_list()
        G2_test_pairs = G2_test_dataset.to_list()

        # Split into different sp_lengths
        length_intervals = [(1, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 100)]
        G1_length_buckets = defaultdict(list)
        for record in G1_test_pairs:
            sp_length = record['shortest_path_length']
            for start, end in length_intervals:
                if start <= sp_length < end:
                    G1_length_buckets[(start, end)].append(record)
                    break

        G2_length_buckets = defaultdict(list)
        for record in G2_test_pairs:
            sp_length = record['shortest_path_length']
            for start, end in length_intervals:
                if start <= sp_length < end:
                    G2_length_buckets[(start, end)].append(record)
                    break

        # Print bucket sizes
        for key, value in G1_length_buckets.items():
            logging.info(f"G1 length bucket {key}: {len(value)} pairs")
        logging.info("\n")
        for key, value in G2_length_buckets.items():
            logging.info(f"G2 length bucket {key}: {len(value)} pairs")

        # Slice all buckets to randomly selected sequences
        for key, value in G2_length_buckets.items():
            np.random.shuffle(value)
            select_max = min(len(value), args.max_test_pairs)
            G1_length_buckets[key] = value[:select_max]
            G2_length_buckets[key] = value[:select_max]

        # Order buckets
        desired_order = length_intervals
        G1_length_buckets = {key: G1_length_buckets[key] for key in desired_order if key in G1_length_buckets}
        G2_length_buckets = {key: G2_length_buckets[key] for key in desired_order if key in G2_length_buckets}

        # Save length buckets
        os.makedirs(os.path.join(args.dataset_dir, '_spatial_length/shortest_path'), exist_ok=True)
        with open(os.path.join(args.dataset_dir, '_spatial_length/shortest_path', 'G1_length_buckets.pkl'), 'wb') as f:
            pickle.dump(G1_length_buckets, f)
        with open(os.path.join(args.dataset_dir, '_spatial_length/shortest_path', 'G2_length_buckets.pkl'), 'wb') as f:
            pickle.dump(G2_length_buckets, f)

    # evaluation
    logging.info("="*50)
    logging.info("EVALUATION")
    logging.info("="*50)

    logging.info("Loading model...")
    if args.mode == 'grpo':
        ft_model = GRPOModelWrapper.from_pretrained(ft_model_path)
        ft_model = ft_model.to(device)
        ft_model.eval()
    else:
        # Load fine-tuned model
        ft_model = utils.PathGenModel(tokenizer, n_embd=head_to_embd[n_head], n_layer=8, n_head=n_head, 
                                        indices_to_nodes=indices_to_nodes, nodes_to_indices=nodes_to_indices, 
                                        connectivity_matrix=adj_matrix, size_m=args.size_m, size_n=args.size_n*2)
        
        # Load pretrained model
        logging.info(f"Loading model from {ft_model_path}")
        ft_model.load_state_dict(torch.load(ft_model_path))
        ft_model = ft_model.to(device)
        ft_model.eval()

    logging.info("Evaluating G1 test pairs...")
    G1_valid_rates, G1_sp_rates, G1_groups = eval_sp_rate(args, G1_length_buckets, ft_model, tokenizer, nodes_to_indices, indices_to_nodes, adj_matrix)
    logging.info("Evaluating G2 test pairs...")
    G2_valid_rates, G2_sp_rates, G2_groups = eval_sp_rate(args, G2_length_buckets, ft_model, tokenizer, nodes_to_indices, indices_to_nodes, adj_matrix)

    # Save results
    results = {
        "G1_valid_rates": G1_valid_rates,
        "G1_sp_rates": G1_sp_rates,
        "G1_groups": G1_groups,
        "G2_valid_rates": G2_valid_rates,
        "G2_sp_rates": G2_sp_rates,
        "G2_groups": G2_groups,
    }
    if args.mode == 'sft':
        results_save_dir = os.path.join(args.result_dir, "spatial_length", f"{args.dataset_name}/pretrain_random_walk_10M_reveal/{args.mode}-len20-shortest_path_reveal_coverage_{args.coverage:.2f}_pairs_{args.pairs_idx}_ans{args.num_ans}")
    else:
        results_save_dir = os.path.join(args.result_dir, "spatial_length", f"{args.dataset_name}/pretrain_random_walk_10M_reveal/{args.mode}-len20-shortest_path_reveal_coverage_{args.coverage:.2f}_pairs_{args.pairs_idx}_ans{args.num_ans}/num_generation_{args.num_generation}_from_ckpt_{args.ckpt_idx}")
    os.makedirs(results_save_dir, exist_ok=True)
    results_path = os.path.join(results_save_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default='dataset/', help="Directory containing dataset files")
    parser.add_argument("--dataset_name", type=str, default='shortest_path', help="Name of the dataset to load.", choices=['shortest_path_vary_answers', 'shortest_path'])
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and validation")
    parser.add_argument("--model_dir", type=str, default='models/', help="Directory to save the trained model")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads in the GPT-2 model")
    parser.add_argument("--checkpoint_dir", type=str, default='ckpt/', help="Directory to save checkpoints")
    parser.add_argument("--result_dir", type=str, default='results/', help="Directory to save the results")
    parser.add_argument("--log_dir", type=str, default='logs/', help="Directory to save the logs")
    parser.add_argument("--size_m", type=int, default=50, help="Number of rows in the grid")
    parser.add_argument("--size_n", type=int, default=40, help="Number of columns in the grid")
    parser.add_argument("--max_test_pairs", type=int, default=3000, help="Maximum number of test pairs per bucket")

    parser.add_argument('--coverage', type=float, default=0.6, help='Coverage ratio of the training data', choices=[0.01, 0.05, 0.1, 0.2, 0.6, 0.8])
    parser.add_argument('--num_ans', type=int, default=64, help='num_ans of the training data', choices=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--pairs_idx', type=int, default=0, help='Index of the pairs to use for training')
    parser.add_argument("--mode", type=str, default='sft', help="Mode: grpo or sft", choices=['grpo', 'sft'])
    parser.add_argument("--ckpt_idx", type=int, default=200, help="Checkpoint index to load for grpo mode")
    parser.add_argument("--num_generation", type=int, default=8, help="Number of generations used during RL fine-tuning for grpo mode", choices=[4, 8, 16])
    return parser.parse_args(argv)

if __name__ == "__main__":
    """Parse command-line arguments."""
    args = parse_arguments(sys.argv[1:])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, "../../"))
    config_path = os.path.join(parent_dir, "config.yaml")
    config = load_config(config_path)

    main(args)