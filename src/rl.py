"""
Reinforcement Learning (GRPO) training for shortest-path models.

Supports two experiment modes:
  - base:    GRPO on filtered dataset (Section 6, original GRPO)
  - spatial: GRPO from SFT checkpoints with spatial coverage (Section 6-7, Figures 5-7)

Usage examples:
  python src/rl.py --experiment base --pretrain_model_name gpt2_model --max_distance 20
  python src/rl.py --experiment spatial --coverage 0.2 --num_ans 4 --num_generation 8
"""

import pickle
import random
import os
import ast
import re
from math import ceil
from typing import Optional

import torch
import numpy as np
import wandb
from datasets import load_dataset, Dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import (
    PreTrainedModel, PreTrainedTokenizerFast, LlamaForCausalLM,
)
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
import argparse

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils
from utils import get_neighbors, get_valid_neighbors

# Set NCCL environment variables
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_TIMEOUT'] = str(7200000)
os.environ['NCCL_BLOCKING_WAIT'] = '0'


class GRPOModelWrapper(PreTrainedModel):
    def __init__(self, model: LlamaForCausalLM, tokenizer=None):
        super().__init__(model.config)
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_path, *args, **kwargs):
        llama_model = LlamaForCausalLM.from_pretrained(pretrained_path, *args, **kwargs)
        tokenizer = None
        try:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_path)
        except Exception:
            pass
        return cls(llama_model, tokenizer)

    def save_pretrained(self, save_directory: str, **kwargs):
        self.model.save_pretrained(save_directory, **kwargs)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_directory)


def setup_hf_tokenizer(tokenizer_path, indices_to_nodes):
    """Create or load HuggingFace-compatible tokenizer for GRPO."""
    if not os.path.exists(tokenizer_path):
        node_tokens = list(map(str, indices_to_nodes.keys()))
        direction_tokens = ['N', 'S', 'W', 'E', 'STAY']
        special_tokens = ['<s>', '<pad>', '</s>', ':']
        vocab = node_tokens + direction_tokens + special_tokens
        vocab_dict = {token: idx for idx, token in enumerate(vocab)}

        tokenizer = Tokenizer(models.WordLevel(vocab=vocab_dict, unk_token="<pad>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Whitespace(),
            pre_tokenizers.Split(":", behavior="isolated")
        ])
        tokenizer.decoder = decoders.WordPiece(prefix="##", cleanup=True)

        os.makedirs(tokenizer_path, exist_ok=True)
        tmp_path = os.path.join(tokenizer_path, 'tokenizer_tmp.json')
        tokenizer.save(tmp_path)

        hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tmp_path)
        hf_tokenizer.bos_token = '<s>'
        hf_tokenizer.eos_token = '</s>'
        hf_tokenizer.pad_token = '<pad>'
        hf_tokenizer.bos_token_id = vocab_dict['<s>']
        hf_tokenizer.eos_token_id = vocab_dict['</s>']
        hf_tokenizer.pad_token_id = vocab_dict['<pad>']
        hf_tokenizer.save_pretrained(tokenizer_path)
        os.remove(tmp_path)
        return hf_tokenizer
    else:
        return PreTrainedTokenizerFast.from_pretrained(tokenizer_path)


def check_valid(responses, tokenizer, nodes_to_indices, idx_to_node, connectivity_matrix, m_size, n_size):
    generated_list = responses.split(" ")
    start_node_idx, end_node_idx = int(generated_list[0]), int(generated_list[-1])
    directions = generated_list[1:]
    current_state = start_node_idx
    state_seq = [current_state]
    for direction in directions:
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
            elif direction == str(current_state):
                continue
            else:
                return False, state_seq
    return False, state_seq


def compute_reward(completions, sp_lengths, coord_distances, starts, ends,
                   tokenizer, nodes_to_indices, indices_to_nodes, connectivity_matrix, m_size, n_size):
    rewards = []
    for content, sp_length, coord_distance, start, end in zip(completions, sp_lengths, coord_distances, starts, ends):
        if content.endswith(tokenizer.eos_token):
            content = content[:-len(tokenizer.eos_token)].strip()
        if content.startswith(tokenizer.bos_token):
            content = content.split(":")[1].strip()
        if not content.startswith(str(start)) or not content.endswith(str(end)):
            rewards.append(0.0)
        else:
            valid_flag, state_seq = check_valid(content, tokenizer, nodes_to_indices, indices_to_nodes, connectivity_matrix, m_size, n_size)
            distance_gap = abs(len(state_seq) - 1 - sp_length) / sp_length
            score = 1.0 * valid_flag - distance_gap
            rewards.append(score)
    return rewards


def load_or_build_model(args, tokenizer, indices_to_nodes, nodes_to_indices, adj_matrix):
    """Load model: from HF directory or from .pth/.ckpt file."""
    head_to_embd = {8: 512, 12: 768, 16: 1024, 20: 1280, 25: 1600}
    n_head = args.n_head

    if os.path.exists(os.path.join(args.model_dir, args.pretrain_model_name)):
        model = GRPOModelWrapper.from_pretrained(os.path.join(args.model_dir, args.pretrain_model_name))
        model.config._name_or_path = os.path.join(args.model_dir, args.pretrain_model_name)
    else:
        base_model = utils.PathGenModel(
            tokenizer, n_embd=head_to_embd[n_head], n_layer=8, n_head=n_head,
            indices_to_nodes=indices_to_nodes, nodes_to_indices=nodes_to_indices,
            connectivity_matrix=adj_matrix, size_m=args.size_m, size_n=args.size_n * 2
        )
        if '.ckpt' in args.pretrain_model_name:
            base_model.load_state_dict(torch.load(args.pretrain_model_name, map_location='cpu')['state_dict'])
        else:
            model_path = os.path.join(args.model_dir, f'{args.pretrain_model_name}.pth')
            base_model.load_state_dict(torch.load(model_path, map_location='cpu'))

        model = GRPOModelWrapper(base_model.model, tokenizer)
        save_name = args.pretrain_model_name.split('.ckpt')[0] if '.ckpt' in args.pretrain_model_name else args.pretrain_model_name
        model.save_pretrained(os.path.join(args.model_dir, save_name))
        model.config._name_or_path = os.path.join(args.model_dir, save_name)

    return model


def main(args):
    os.environ["WANDB_PROJECT"] = args.project_name
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.model_dir, exist_ok=True)

    # Parse devices
    if isinstance(args.devices, str) and "," in args.devices:
        num_devices = len(args.devices.split(","))
    elif args.devices.isdigit():
        num_devices = int(args.devices)
    else:
        num_devices = 1

    # Load map data
    with open(os.path.join(args.dataset_dir, "map_stats", 'nodes_to_indices.pkl'), 'rb') as f:
        nodes_to_indices = pickle.load(f)
    with open(os.path.join(args.dataset_dir, "map_stats", 'indices_to_nodes.pkl'), 'rb') as f:
        indices_to_nodes = pickle.load(f)
    adj_matrix = np.load(os.path.join(args.dataset_dir, "map_stats", 'adj_matrix.npy'))
    indices_to_nodes = {k: str(v) for k, v in indices_to_nodes.items()}
    nodes_to_indices = {str(k): v for k, v in nodes_to_indices.items()}

    # Setup tokenizer
    tokenizer_path = os.path.join(args.model_dir, 'hf_tokenizer')
    tokenizer = setup_hf_tokenizer(tokenizer_path, indices_to_nodes)
    tokenizer.skip_special_tokens = False

    # Extract checkpoint index from model name
    ckpt_match = re.search(r'gpt2-\d+-(\d+)(?:\.ckpt)?$', args.pretrain_model_name)
    ckpt_idx = int(ckpt_match.group(1)) if ckpt_match else 0

    # Build saved model name
    if args.experiment == 'base':
        args.saved_model_name = f'grpo-{args.dataset_name}_{args.path_type}'
    else:
        args.saved_model_name = f"grpo-len20-{args.dataset_name}_{args.path_type}_coverage_{args.coverage:.2f}_num_ans_{args.num_ans}_pairs_{args.pairs_idx}_num_generation_{args.num_generation}"
        if ckpt_idx > 0:
            args.saved_model_name += f"_from_ckpt_{ckpt_idx}"

    checkpoint_dir = os.path.join(args.checkpoint_dir, args.saved_model_name)

    # Load dataset
    if args.experiment == 'base':
        dataset_path = os.path.join(args.dataset_dir, f"grpo_{args.dataset_name}_filtered_{args.max_distance}")
        if os.path.exists(dataset_path):
            dataset = Dataset.load_from_disk(dataset_path)
        else:
            sft_path = os.path.join(args.dataset_dir, f"{args.dataset_name}_filtered_{args.max_distance}")
            train_dataset = Dataset.load_from_disk(sft_path)

            def preprocess(sample):
                return {"prompt": tokenizer.decode(sample["input_ids"][:4]),
                        "start_idx": sample["start_idx"], "end_idx": sample["end_idx"],
                        'sp_distance': sample['sp_distance'], 'coord_distance': sample['coord_distance']}

            dataset = train_dataset.map(preprocess)
            dataset.save_to_disk(dataset_path)
            del train_dataset
    else:
        data_dir = os.path.join(
            args.dataset_dir, '_spatial_length',
            f'coverage_ratio_{args.coverage:.2f}',
            f'pairs_{args.pairs_idx}/{args.dataset_name}/tradeoff_datasets/paths_ans{args.num_ans}'
        )
        grpo_path = os.path.join(data_dir, "grpo_dataset")
        if os.path.exists(grpo_path):
            dataset = Dataset.load_from_disk(grpo_path)
        else:
            train_dataset = Dataset.load_from_disk(os.path.join(data_dir, "hf_dataset"))

            def preprocess(sample):
                return {"prompt": tokenizer.decode(sample["input_ids"][:4]),
                        "start_idx": sample["start_idx"], "end_idx": sample["end_idx"],
                        'sp_distance': sample['sp_distance'], 'coord_distance': sample['coord_distance']}

            dataset = train_dataset.map(preprocess)
            dataset.save_to_disk(grpo_path)
            del train_dataset

    # Load model
    model = load_or_build_model(args, tokenizer, indices_to_nodes, nodes_to_indices, adj_matrix)

    # Filter dataset for checkpoint resume
    if ckpt_idx > 0:
        steps_per_epoch = ceil(len(dataset) / args.batch_size)
        if ckpt_idx < steps_per_epoch:
            start_idx = ckpt_idx * args.batch_size
            dataset = dataset.select(range(start_idx, len(dataset)))
            print(f"Resuming from checkpoint {ckpt_idx}, dataset size: {len(dataset)}")

    # Calculate training steps
    per_device_batch_size = 4
    gradient_accumulation_steps = args.batch_size // (per_device_batch_size * num_devices)
    steps_per_epoch = ceil(len(dataset) / args.batch_size)

    grpo_config = GRPOConfig(
        output_dir=os.path.join(checkpoint_dir, args.saved_model_name),
        learning_rate=1e-5,
        remove_unused_columns=False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=steps_per_epoch,
        per_device_train_batch_size=per_device_batch_size,
        bf16=False,
        shuffle_dataset=False,
        max_completion_length=256,
        num_generations=args.num_generation,
        max_prompt_length=32,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        push_to_hub=False,
        save_strategy="steps",
        save_steps=min(300, steps_per_epoch // 2),
    )

    trainer = GRPOTrainer(
        args=grpo_config,
        model=model,
        reward_funcs=lambda completions, **kwargs: compute_reward(
            completions,
            sp_lengths=kwargs['sp_distance'],
            coord_distances=kwargs['coord_distance'],
            starts=kwargs['start_idx'],
            ends=kwargs['end_idx'],
            tokenizer=tokenizer,
            nodes_to_indices=nodes_to_indices,
            indices_to_nodes=indices_to_nodes,
            connectivity_matrix=adj_matrix,
            m_size=args.size_m,
            n_size=args.size_n * 2
        ),
        train_dataset=dataset,
    )

    trainer.train()

    # Save final model
    model_path = os.path.join(args.model_dir, args.saved_model_name)
    model.save_pretrained(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO training for shortest-path models.")

    parser.add_argument("--experiment", type=str, required=True, choices=['base', 'spatial'])

    # Common
    parser.add_argument("--dataset_dir", type=str, default='dataset/')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--model_dir", type=str, default='models/')
    parser.add_argument("--pretrain_model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default='shortest_path')
    parser.add_argument("--checkpoint_dir", type=str, default='ckpt')
    parser.add_argument("--devices", type=str, default="0,1")
    parser.add_argument("--project_name", type=str, default='grpo-map-llm')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size_m", type=int, default=50)
    parser.add_argument("--size_n", type=int, default=40)
    parser.add_argument("--path_type", type=str, default='reveal', choices=['reveal', 'standard'])
    parser.add_argument("--num_generation", type=int, default=8)

    # Base experiment
    parser.add_argument("--max_distance", type=int, default=20)
    parser.add_argument("--max_epochs", type=int, default=3)

    # Spatial experiment
    parser.add_argument("--coverage", type=float, default=0.2)
    parser.add_argument("--num_ans", type=int, default=4)
    parser.add_argument("--pairs_idx", type=int, default=0)

    args = parser.parse_args()
    main(args)
