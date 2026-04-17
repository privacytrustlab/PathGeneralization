import logging
import numpy as np
import torch
from tqdm import tqdm
import os
import pickle
import networkx as nx
from collections import defaultdict
import ast
from transformers import PreTrainedTokenizerFast, PreTrainedModel, LlamaForCausalLM
from typing import Optional
import matplotlib.pyplot as plt
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils
from utils import get_valid_neighbors

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def check_valid_detailed(responses, start_node_idx, end_node_idx, tokenizer, nodes_to_indices, idx_to_node, connectivity_matrix, m_size, n_size):
    """Check if a response is valid path navigation and return detailed failure info."""
    generated_list = responses.split(" ")
    if len(generated_list) < 2:
        return False, [], "Too short response"
    
    # check if the first token is the start node index, if it is N, S, W, E, then report formatting issue
    if generated_list[0] in ['N', 'S', 'W', 'E', ':']:
        print(f"Formatting issue: first token is a direction or colon: {generated_list}")
        return False, [], "Formatting issue: first token is a direction"
    
    if start_node_idx != int(generated_list[0]):
        return False, [], "Start node mismatch"
    
    directions = generated_list[1:]
    current_state = start_node_idx
    state_seq = [current_state]
    failure_reason = "Unknown"
    
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
                failure_reason = f"Invalid move"
                return False, state_seq, failure_reason
        else:
            if current_state == end_node_idx:
                return True, state_seq, "Success"
            elif direction == str(current_state):  # print valid current node
                continue
            else:
                failure_reason = f"Didn't reach target"
                return False, state_seq, failure_reason
    
    if current_state != end_node_idx:
        failure_reason = f"Didn't reach target"
    else:
        failure_reason = f"Not predict the end token but stopped at correct target"
    return False, state_seq, failure_reason

def decode_without_pad(hf_tokenizer, input_ids):
    """Decode tokens without padding tokens."""
    tokens = hf_tokenizer.convert_ids_to_tokens(input_ids)
    tokens = [t for t in tokens if t != hf_tokenizer.pad_token]
    return hf_tokenizer.convert_tokens_to_string(tokens)

def load_model_for_inference(mode, coverage, diversity, pairs_idx, model_dir, indices_to_nodes, nodes_to_indices, adj_matrix):
    """Load model for inference"""
    head_to_embd = {8: 512, 12: 768, 16: 1024, 20: 1280, 25: 1600}
    n_head = 8
    
    # Load tokenizer
    tokenizer_path = os.path.join(model_dir, 'tokenizer.pth')
    tokenizer = torch.load(tokenizer_path)
    hf_tokenizer_path = os.path.join(model_dir, 'hf_tokenizer')
    hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(hf_tokenizer_path)
    
    if mode == 'sft':
        model_path = os.path.join(
            model_dir, 'pretrain_random_walk_10M_reveal',
            f"{mode}-len20-shortest_path_reveal_coverage_{coverage:.2f}_pairs_{pairs_idx}_ans{diversity}.pth"
        )
        model = utils.PathGenModel(tokenizer, n_embd=head_to_embd[n_head], n_layer=8, n_head=n_head, 
                                 indices_to_nodes=indices_to_nodes, nodes_to_indices=nodes_to_indices, 
                                 connectivity_matrix=adj_matrix, size_m=50, size_n=80)
        model.load_state_dict(torch.load(model_path))
        
    elif mode == 'grpo':
        model_path = os.path.join(
            model_dir,
            f"{mode}-len20-shortest_path_reveal_coverage_{coverage:.2f}_num_ans_4_pairs_{pairs_idx}_num_generation_16_from_ckpt_1800"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GRPO model path does not exist: {model_path}")
        # Pass attn_implementation to the GRPO model if specified
        model = GRPOModelWrapper.from_pretrained(model_path, attn_implementation="eager",local_files_only=True)
    
    model = model.to(device)
    model.eval()
    return model, tokenizer, hf_tokenizer

def annotate_failure_cases(model, tokenizer, hf_tokenizer, model_type, test_pairs,
                           nodes_to_indices, indices_to_nodes, adj_matrix, batch_size=32):
    """Evaluate ALL test_pairs in order and return a binary success list.

    Returns:
        success_list: list of bool, length == len(test_pairs), True if model generates a valid shortest path.
    """
    success_list = [False] * len(test_pairs)
    num_batches = int(np.ceil(len(test_pairs) / batch_size))
    G = nx.from_numpy_array(adj_matrix)

    with tqdm(total=len(test_pairs), desc="Annotating pairs") as pbar:
        for i in range(num_batches):
            if i == num_batches - 1:
                batch = test_pairs[i * batch_size:]
            else:
                batch = test_pairs[i * batch_size:(i + 1) * batch_size]

            input_ids = [item['input_ids'] for item in batch]
            input_ids = torch.tensor(input_ids).to(device)

            with torch.no_grad():
                if model_type == 'grpo':
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_length=512,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 50256,
                        do_sample=False
                    )
                    responses = [decode_without_pad(tokenizer, ids) for ids in outputs]
                else:
                    outputs = model.model.generate(
                        input_ids=input_ids,
                        max_length=512,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=False
                    )
                    responses = list(map(tokenizer.decode, outputs))

            for j, response in enumerate(responses):
                global_idx = i * batch_size + j
                start_idx = batch[j]['start_index']
                end_idx = batch[j]['end_index']

                # Clean up response
                if response.split(" ")[-1] == tokenizer.eos_token:
                    response = response[:-len(tokenizer.eos_token)].strip()
                if response.split(" ")[0] == tokenizer.bos_token:
                    response_parts = response.split(" ")
                    response_parts = response_parts[4:]
                    if len(response_parts) < 2:
                        continue
                    response = " ".join(response_parts)

                is_valid, predicted_path, failure_reason = check_valid_detailed(
                    response, start_idx, end_idx, tokenizer, nodes_to_indices, indices_to_nodes,
                    adj_matrix, 50, 80
                )

                shortest_path = nx.shortest_path(G, start_idx, end_idx)
                is_shortest_path = (len(predicted_path) - 1) == (len(shortest_path) - 1)

                success_list[global_idx] = is_valid and is_shortest_path

            pbar.update(len(batch))

    n_success = sum(success_list)
    print(f"Annotated {len(test_pairs)} pairs: {n_success} success, {len(test_pairs) - n_success} failure "
          f"({n_success / len(test_pairs) * 100:.1f}% success rate)")
    return success_list


def collect_failure_cases_from_model(model, tokenizer, hf_tokenizer, model_type, test_pairs,
                                    nodes_to_indices, indices_to_nodes, adj_matrix,
                                    batch_size=32, max_failures=20):
    """Collect failure cases from a loaded model"""
    failure_cases = []
    total_count = 0
    max_attempts = min(len(test_pairs), max_failures * 3)
    
    # Shuffle test pairs
    shuffled_pairs = np.random.choice(test_pairs, max_attempts, replace=False)
    num_batches = int(np.ceil(len(shuffled_pairs) / batch_size))
    
    with tqdm(total=len(shuffled_pairs), desc="Collecting failures") as pbar:
        for i in range(num_batches):
            if len(failure_cases) >= max_failures:
                break
                
            if i == num_batches - 1:
                batch = shuffled_pairs[i * batch_size:]
            else:
                batch = shuffled_pairs[i * batch_size:(i + 1) * batch_size]
            
            input_ids = [item['input_ids'] for item in batch]
            input_ids = torch.tensor(input_ids).to(device)

            with torch.no_grad():
                if model_type == 'grpo':
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_length=512,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 50256,
                        do_sample=False
                    )
                    responses = [decode_without_pad(tokenizer, ids) for ids in outputs]
                else:
                    outputs = model.model.generate(
                        input_ids=input_ids,
                        max_length=512,
                        num_return_sequences=1, 
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=False
                    )
                    responses = list(map(tokenizer.decode, outputs))

            for j, response in enumerate(responses):
                start_idx = batch[j]['start_index']
                end_idx = batch[j]['end_index']
                if len(failure_cases) >= max_failures:
                    break
                    
                total_count += 1
                
                # Clean up response
                if response.split(" ")[-1] == tokenizer.eos_token:
                    response = response[:-len(tokenizer.eos_token)].strip()
                if response.split(" ")[0] == tokenizer.bos_token:
                    response_parts = response.split(" ")
                    response_parts = response_parts[4:]
                    if len(response_parts) < 2:
                        continue
                    response = " ".join(response_parts)

                # Check validity with detailed failure info
                is_valid, predicted_path, failure_reason = check_valid_detailed(
                    response, start_idx, end_idx, tokenizer, nodes_to_indices, indices_to_nodes, 
                    adj_matrix, 50, 80
                )

                # Get shortest path for comparison
                G = nx.from_numpy_array(adj_matrix)
                start_idx = batch[j]['start_index']
                end_idx = batch[j]['end_index']
                shortest_path = nx.shortest_path(G, start_idx, end_idx)

                # check whether it is shortest path
                is_shortest_path = (len(predicted_path) - 1) == (len(shortest_path) - 1)

                if not is_valid or not is_shortest_path:
                    if is_valid and not is_shortest_path:
                        failure_reason = "Not shortest path"
                        
                    failure_case = {
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'start_coord': ast.literal_eval(indices_to_nodes[start_idx]),
                        'end_coord': ast.literal_eval(indices_to_nodes[end_idx]),
                        'shortest_path': shortest_path,
                        'predicted_path': predicted_path,
                        'failure_reason': failure_reason,
                        'shortest_length': len(shortest_path) - 1,
                        'predicted_length': len(predicted_path) - 1,
                        'model_response': response,
                        'model_type': model_type
                    }
                    
                    failure_cases.append(failure_case)
 
            
            pbar.update(len(batch))
    
    print(f"Found {len(failure_cases)} failures out of {total_count} attempts")
    return failure_cases

def collect_success_cases_from_model(model, tokenizer, hf_tokenizer, model_type, test_pairs, 
                                    nodes_to_indices, indices_to_nodes, adj_matrix, 
                                    batch_size=32):
    """Collect success cases from a loaded model"""
    success_cases = []
    total_count = 0
    
    # Shuffle test pairs
    shuffled_pairs = np.random.choice(test_pairs, len(test_pairs), replace=False)
    num_batches = int(np.ceil(len(shuffled_pairs) / batch_size))
    
    with tqdm(total=len(shuffled_pairs), desc="Collecting successes") as pbar:
        for i in range(num_batches):
            if i == num_batches - 1:
                batch = shuffled_pairs[i * batch_size:]
            else:
                batch = shuffled_pairs[i * batch_size:(i + 1) * batch_size]
            
            input_ids = [item['input_ids'] for item in batch]
            input_ids = torch.tensor(input_ids).to(device)

            with torch.no_grad():
                if model_type == 'grpo':
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_length=512,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 50256,
                        do_sample=False
                    )
                    responses = [decode_without_pad(tokenizer, ids) for ids in outputs]
                else:
                    outputs = model.model.generate(
                        input_ids=input_ids,
                        max_length=512,
                        num_return_sequences=1, 
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=False
                    )
                    responses = list(map(tokenizer.decode, outputs))

            for j, response in enumerate(responses):
                start_idx = batch[j]['start_index']
                end_idx = batch[j]['end_index']
                
                total_count += 1
                
                # Clean up response
                if response.split(" ")[-1] == tokenizer.eos_token:
                    response = response[:-len(tokenizer.eos_token)].strip()
                if response.split(" ")[0] == tokenizer.bos_token:
                    response_parts = response.split(" ")
                    response_parts = response_parts[4:]
                    if len(response_parts) < 2:
                        continue
                    response = " ".join(response_parts)

                # Check validity with detailed failure info
                is_valid, predicted_path, failure_reason = check_valid_detailed(
                    response, start_idx, end_idx, tokenizer, nodes_to_indices, indices_to_nodes, 
                    adj_matrix, 50, 80
                )

                # Get shortest path for comparison
                G = nx.from_numpy_array(adj_matrix)
                shortest_path = nx.shortest_path(G, start_idx, end_idx)

                # check whether it is shortest path
                is_shortest_path = (len(predicted_path) - 1) == (len(shortest_path) - 1)

                if is_valid and is_shortest_path:
                    success_case = {
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'start_coord': ast.literal_eval(indices_to_nodes[start_idx]),
                        'end_coord': ast.literal_eval(indices_to_nodes[end_idx]),
                        'shortest_path': shortest_path,
                        'predicted_path': predicted_path,
                        'path_length': len(shortest_path) - 1,
                        'model_response': response,
                        'model_type': model_type
                    }
                    
                    success_cases.append(success_case)
            
            pbar.update(len(batch))
    
    print(f"Found {len(success_cases)} successes out of {total_count} attempts")
    return success_cases

def visualize_failure_case(failure_case, adj_matrix, indices_to_nodes, title="", figsize=(10, 8)):
    """Visualize a single failure case on the grid map"""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get coordinates for visualization
    start_coord = failure_case['start_coord']
    end_coord = failure_case['end_coord']
    
    # Calculate zoom area around the paths
    path_coords = [start_coord, end_coord]
    for node_idx in failure_case['shortest_path']:
        coord = ast.literal_eval(indices_to_nodes[node_idx])
        path_coords.append(coord)
    for node_idx in failure_case['predicted_path']:
        if node_idx in indices_to_nodes:
            coord = ast.literal_eval(indices_to_nodes[node_idx])
            path_coords.append(coord)
    
    x_coords = [c[0] for c in path_coords]
    y_coords = [c[1] for c in path_coords]
    x_min, x_max = min(x_coords) - 3, max(x_coords) + 3
    y_min, y_max = min(y_coords) - 3, max(y_coords) + 3
    
    # Draw grid edges in the zoom area
    G = nx.from_numpy_array(adj_matrix)
    for node1 in range(len(adj_matrix)):
        if node1 not in indices_to_nodes:
            continue
        coord1 = ast.literal_eval(indices_to_nodes[node1])
        if not (x_min <= coord1[0] <= x_max and y_min <= coord1[1] <= y_max):
            continue
            
        for node2 in G.neighbors(node1):
            if node2 not in indices_to_nodes:
                continue
            coord2 = ast.literal_eval(indices_to_nodes[node2])
            if not (x_min <= coord2[0] <= x_max and y_min <= coord2[1] <= y_max):
                continue
            
            ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], 
                   color='lightgray', linewidth=0.5, alpha=0.7)
    
    # Draw nodes as small dots
    for node_idx, coord_str in indices_to_nodes.items():
        coord = ast.literal_eval(coord_str)
        if x_min <= coord[0] <= x_max and y_min <= coord[1] <= y_max:
            ax.plot(coord[0], coord[1], 'o', color='lightgray', markersize=3, alpha=0.6)
    
    # Draw shortest path (ground truth) in green
    shortest_coords = []
    for node_idx in failure_case['shortest_path']:
        if node_idx in indices_to_nodes:
            coord = ast.literal_eval(indices_to_nodes[node_idx])
            shortest_coords.append(coord)
    
    if len(shortest_coords) > 1:
        x_path = [c[0] for c in shortest_coords]
        y_path = [c[1] for c in shortest_coords]
        ax.plot(x_path, y_path, 'g-', linewidth=3, label='Shortest Path (One Sample)', alpha=0.8)
    
    # Draw predicted path (failure) in red dashed line
    predicted_coords = []
    for node_idx in failure_case['predicted_path']:
        if node_idx in indices_to_nodes:
            coord = ast.literal_eval(indices_to_nodes[node_idx])
            predicted_coords.append(coord)
    
    if len(predicted_coords) > 1:
        x_path = [c[0] for c in predicted_coords]
        y_path = [c[1] for c in predicted_coords]
        ax.plot(x_path, y_path, 'r--', linewidth=3, label='Predicted Path (Failed)', alpha=0.8)
    
    # Mark start and end points
    ax.plot(start_coord[0], start_coord[1], '*', color='blue', markersize=15, 
           label=f'Start ({start_coord[0]}, {start_coord[1]})')
    ax.plot(end_coord[0], end_coord[1], '*', color='orange', markersize=15, 
           label=f'End ({end_coord[0]}, {end_coord[1]})')
    
    # Set axis properties
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Title with failure information
    full_title = f"{title}\n{failure_case['failure_reason']}"
    full_title += f"\nShortest: {failure_case['shortest_length']}, Predicted: {failure_case['predicted_length']}"
    ax.set_title(full_title, fontsize=12)
    
    plt.tight_layout()
    return fig, ax

def plot_failure_cases_comparison(sft_failures, grpo_failures, adj_matrix, indices_to_nodes, 
                                 length_range, max_cases=5):
    """Compare failure cases between SFT and GRPO models side by side"""
    n_cases = min(len(sft_failures), len(grpo_failures), max_cases)
    if n_cases == 0:
        print("No failure cases found for comparison")
        return
    
    fig, axes = plt.subplots(n_cases, 2, figsize=(16, 6*n_cases))
    if n_cases == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'SFT vs GRPO Failure Cases Comparison - Length Range {length_range[0]}-{length_range[1]}', 
                fontsize=16, y=0.98)
    
    for i in range(n_cases):
        # SFT failure (left column)
        ax_sft = axes[i, 0]
        sft_case = sft_failures[i]
        plot_single_failure_on_axis(ax_sft, sft_case, adj_matrix, indices_to_nodes, f'SFT - Case {i+1}')
        
        # GRPO failure (right column)
        ax_grpo = axes[i, 1]
        grpo_case = grpo_failures[i]
        plot_single_failure_on_axis(ax_grpo, grpo_case, adj_matrix, indices_to_nodes, f'GRPO - Case {i+1}')
    
    plt.tight_layout()
    plt.show()

def plot_single_failure_on_axis(ax, failure_case, adj_matrix, indices_to_nodes, title):
    """Helper function to plot a failure case on a given axis"""
    start_coord = failure_case['start_coord']
    end_coord = failure_case['end_coord']
    
    # Calculate zoom area
    path_coords = [start_coord, end_coord]
    for node_idx in failure_case['shortest_path']:
        coord = ast.literal_eval(indices_to_nodes[node_idx])
        path_coords.append(coord)
    for node_idx in failure_case['predicted_path']:
        if node_idx in indices_to_nodes:
            coord = ast.literal_eval(indices_to_nodes[node_idx])
            path_coords.append(coord)
    
    x_coords = [c[0] for c in path_coords]
    y_coords = [c[1] for c in path_coords]
    x_min, x_max = min(x_coords) - 2, max(x_coords) + 2
    y_min, y_max = min(y_coords) - 2, max(y_coords) + 2
    
    # Draw grid edges
    G = nx.from_numpy_array(adj_matrix)
    for node1 in range(len(adj_matrix)):
        if node1 not in indices_to_nodes:
            continue
        coord1 = ast.literal_eval(indices_to_nodes[node1])
        if not (x_min <= coord1[0] <= x_max and y_min <= coord1[1] <= y_max):
            continue
            
        for node2 in G.neighbors(node1):
            if node2 not in indices_to_nodes:
                continue
            coord2 = ast.literal_eval(indices_to_nodes[node2])
            if not (x_min <= coord2[0] <= x_max and y_min <= coord2[1] <= y_max):
                continue
            
            ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], 
                   color='lightgray', linewidth=0.8, alpha=0.7)
    
    # Draw nodes
    for node_idx, coord_str in indices_to_nodes.items():
        coord = ast.literal_eval(coord_str)
        if x_min <= coord[0] <= x_max and y_min <= coord[1] <= y_max:
            ax.plot(coord[0], coord[1], 'o', color='lightgray', markersize=2, alpha=0.6)
    
    # Draw shortest path
    shortest_coords = []
    for node_idx in failure_case['shortest_path']:
        if node_idx in indices_to_nodes:
            coord = ast.literal_eval(indices_to_nodes[node_idx])
            shortest_coords.append(coord)
    
    if len(shortest_coords) > 1:
        x_path = [c[0] for c in shortest_coords]
        y_path = [c[1] for c in shortest_coords]
        ax.plot(x_path, y_path, 'g-', linewidth=2, alpha=0.8, label='Shortest Path')
    
    # Draw predicted path
    predicted_coords = []
    for node_idx in failure_case['predicted_path']:
        if node_idx in indices_to_nodes:
            coord = ast.literal_eval(indices_to_nodes[node_idx])
            predicted_coords.append(coord)
    
    if len(predicted_coords) > 1:
        x_path = [c[0] for c in predicted_coords]
        y_path = [c[1] for c in predicted_coords]
        ax.plot(x_path, y_path, 'r--', linewidth=2, alpha=0.8, label='Predicted Path')
    
    # Mark start and end points
    ax.plot(start_coord[0], start_coord[1], '*', color='blue', markersize=10)
    ax.plot(end_coord[0], end_coord[1], '*', color='orange', markersize=10)
    
    # Set properties
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Title with failure reason
    short_reason = failure_case["failure_reason"]
    if len(short_reason) > 40:
        short_reason = short_reason[:40] + "..."
    ax.set_title(f'{title}\n{short_reason}', fontsize=10)
    
    if ax.get_position().x0 < 0.1:  # Only add legend to left column
        ax.legend()

print("Failure case plotting utilities loaded successfully!")
print("Available functions:")
print("- load_model_for_inference()")
print("- collect_failure_cases_from_model()")  
print("- visualize_failure_case()")
print("- plot_failure_cases_comparison()")