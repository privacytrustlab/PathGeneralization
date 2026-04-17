"""
Supervised Fine-Tuning (SFT) for shortest-path models.

Supports multiple experiment modes corresponding to different paper sections:
  - cov_div:   Coverage vs. Diversity experiments (Section 4.2, Figure 3)
  - qa:        More Questions vs. More Answers (Section 4.1, Figure 2)
  - longshort: Rescue with slightly longer paths (Section 5, Figure 4)
  - spatial:   Spatial transfer with data fractions (Section 4.1 variant)

Usage examples:
  python src/sft.py --experiment cov_div --coverage 0.6 --diversity 64 --pairs_idx 0
  python src/sft.py --experiment qa --coverage 0.2 --num_ans 4
  python src/sft.py --experiment longshort --group "(30,40)" --add_num 1000
  python src/sft.py --experiment spatial --coverage 0.6 --data_frac "[0.05,0.1,0.2,0.6,1.0]"
"""

import pickle
import random
import os
import ast
from math import ceil

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from datasets import Dataset
import wandb
import argparse

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils

# Set NCCL environment variables
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_TIMEOUT'] = str(7200000)
os.environ['NCCL_BLOCKING_WAIT'] = '0'


def collate_fn(batch, path_type, pad_token_id):
    if path_type == 'standard':
        max_len = max(len(item['input_ids']) for item in batch)
    else:
        max_len = max(len(item['input_ids_reveal']) for item in batch)
    input_ids = []
    attention_mask = []

    for item in batch:
        if path_type == 'standard':
            ids = item['input_ids']
            mask = item['attention_mask']
        else:
            ids = item['input_ids_reveal']
            mask = item['attention_mask_reveal']
        padding_length = max_len - len(ids)
        ids = ids + [pad_token_id] * padding_length
        mask = mask + [0] * padding_length
        input_ids.append(ids)
        attention_mask.append(mask)

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'labels': torch.tensor(input_ids, dtype=torch.long)
    }


def load_map_data(dataset_dir):
    with open(os.path.join(dataset_dir, "map_stats", 'nodes_to_indices.pkl'), 'rb') as f:
        nodes_to_indices = pickle.load(f)
    with open(os.path.join(dataset_dir, "map_stats", 'indices_to_nodes.pkl'), 'rb') as f:
        indices_to_nodes = pickle.load(f)
    adj_matrix = np.load(os.path.join(dataset_dir, "map_stats", 'adj_matrix.npy'))
    indices_to_nodes = {k: str(v) for k, v in indices_to_nodes.items()}
    nodes_to_indices = {str(k): v for k, v in nodes_to_indices.items()}
    return nodes_to_indices, indices_to_nodes, adj_matrix


def load_tokenizer(tokenizer_path):
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run pretraining first.")
    return torch.load(tokenizer_path)


def build_model(tokenizer, n_head, indices_to_nodes, nodes_to_indices, adj_matrix, size_m, size_n):
    head_to_embd = {8: 512, 12: 768, 16: 1024, 20: 1280, 25: 1600}
    return utils.PathGenModel(
        tokenizer, n_embd=head_to_embd[n_head], n_layer=8, n_head=n_head,
        indices_to_nodes=indices_to_nodes, nodes_to_indices=nodes_to_indices,
        connectivity_matrix=adj_matrix, size_m=size_m, size_n=size_n * 2
    )


def parse_devices(devices_str):
    if isinstance(devices_str, str) and "," in devices_str:
        return [int(d) for d in devices_str.split(",")]
    elif devices_str.isdigit():
        return int(devices_str)
    return devices_str


class StepCheckpoint(ModelCheckpoint):
    def __init__(self, save_per_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_per_steps = save_per_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        global_step = trainer.global_step
        epoch = trainer.current_epoch
        if global_step % self.save_per_steps == 0 and global_step > 0:
            filename = os.path.join(self.dirpath, f'gpt2-{epoch:02d}-{global_step:06d}.ckpt')
            self._save_checkpoint(trainer, filename)
            print(f"Saved checkpoint at step {global_step}: {filename}")


# ============================================================
# Dataset loading per experiment mode
# ============================================================

def load_dataset_cov_div(args):
    """Coverage-diversity grid experiment (Section 4.2)."""
    data_dir = os.path.join(
        args.dataset_dir, '_diversity_coverage',
        f'diversity_{args.diversity}',
        f'coverage_ratio_{args.coverage:.2f}',
        f'pairs_{args.pairs_idx}/{args.dataset_name}'
    )
    return Dataset.load_from_disk(os.path.join(data_dir, "hf_dataset"))


def load_dataset_qa(args):
    """More Questions vs. More Answers experiment (Section 4.1)."""
    data_dir = os.path.join(
        args.dataset_dir, '_spatial_length',
        f'coverage_ratio_{args.coverage:.2f}',
        f'pairs_{args.pairs_idx}/{args.dataset_name}/tradeoff_datasets/paths_ans{args.num_ans}'
    )
    return Dataset.load_from_disk(os.path.join(data_dir, "hf_dataset"))


def load_dataset_longshort(args):
    """Length scaling rescue experiment (Section 5)."""
    data_dir = os.path.join(
        args.dataset_dir, '_spatial_length/longshort_pairs',
        f'group_{args.group}'
    )
    full_dataset = Dataset.load_from_disk(os.path.join(data_dir, "hf_dataset"))
    if args.add_num < len(full_dataset):
        return full_dataset.select(range(args.add_num))
    raise ValueError(f"add_num {args.add_num} exceeds dataset size {len(full_dataset)}")


def load_dataset_spatial(args):
    """Spatial transfer with data fractions."""
    data_dir = os.path.join(
        args.dataset_dir, '_spatial_length',
        f'coverage_ratio_{args.coverage:.2f}',
        f'pairs_{args.pairs_idx}/{args.dataset_name}'
    )
    return Dataset.load_from_disk(os.path.join(data_dir, "hf_dataset"))


# ============================================================
# Model name construction per experiment mode
# ============================================================

def get_model_name(args):
    if args.experiment == 'cov_div':
        return f'sft-{args.dataset_name}_{args.path_type}-diversity_{args.diversity}_coverage_{args.coverage:.2f}_pairs_{args.pairs_idx}'
    elif args.experiment == 'qa':
        return f'sft-len20-{args.dataset_name}_{args.path_type}_coverage_{args.coverage:.2f}_pairs_{args.pairs_idx}_ans{args.num_ans}'
    elif args.experiment == 'longshort':
        return f'sft-baselen20-{args.dataset_name}_{args.path_type}_add_group_{args.group}_num_{args.add_num}'
    elif args.experiment == 'spatial':
        return f'sft-len20-{args.dataset_name}_{args.path_type}_coverage_{args.coverage:.2f}_pairs_{args.pairs_idx}'
    return f'sft-{args.dataset_name}_{args.path_type}'


# ============================================================
# Training loops
# ============================================================

def train_single(args, model, tokenizer, train_dataset, checkpoint_dir, wandb_logger):
    """Standard single-pass training (cov_div, qa, longshort)."""
    devices = parse_devices(args.devices)
    if isinstance(devices, list):
        num_devices = len(devices)
    elif isinstance(devices, int):
        num_devices = devices
    else:
        num_devices = 1

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda x: collate_fn(x, args.path_type, tokenizer.pad_token_id),
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    steps_per_epoch = len(train_loader) // num_devices
    total_steps = steps_per_epoch * 1  # 1 epoch

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = StepCheckpoint(
        save_per_steps=args.save_per_steps,
        dirpath=checkpoint_dir,
        save_top_k=0, save_last=False, verbose=True,
        enable_version_counter=False
    )

    trainer = pl.Trainer(
        devices=devices, strategy='ddp', accelerator='gpu',
        callbacks=[checkpoint_callback],
        max_steps=total_steps, max_epochs=-1,
        logger=wandb_logger, log_every_n_steps=100, precision=16,
    )
    trainer.fit(model, train_loader)
    return model


def train_fractions(args, model, tokenizer, full_dataset, base_checkpoint_dir, wandb_logger):
    """Progressive fraction training for spatial experiment."""
    data_fractions = ast.literal_eval(args.data_frac)
    devices = parse_devices(args.devices)
    if isinstance(devices, list):
        num_devices = len(devices)
    elif isinstance(devices, int):
        num_devices = devices
    else:
        num_devices = 1

    for i, frac in enumerate(data_fractions):
        print(f"\n=== Training on fraction {frac} ({i+1}/{len(data_fractions)}) ===")
        total_size = len(full_dataset)
        prev_size = int(total_size * data_fractions[i-1]) if i > 0 else 0
        current_size = int(total_size * frac)
        train_dataset = full_dataset.select(range(prev_size, current_size))

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda x: collate_fn(x, args.path_type, tokenizer.pad_token_id),
            num_workers=4, pin_memory=True, persistent_workers=True
        )

        steps_per_epoch = len(train_loader) // num_devices
        total_steps = steps_per_epoch * 1

        current_model_name = f'sft-len20-{args.dataset_name}_{args.path_type}_coverage_{args.coverage:.2f}_pairs_{args.pairs_idx}_frac_{frac:.2f}'
        current_checkpoint_dir = os.path.join(base_checkpoint_dir, current_model_name)

        os.makedirs(current_checkpoint_dir, exist_ok=True)
        checkpoint_callback = StepCheckpoint(
            save_per_steps=args.save_per_steps,
            dirpath=current_checkpoint_dir,
            save_top_k=0, save_last=False, verbose=True,
            enable_version_counter=False
        )

        wandb_logger.experiment.name = f"{args.project_name}_frac_{frac:.2f}"
        trainer = pl.Trainer(
            devices=devices, strategy='ddp', accelerator='gpu',
            callbacks=[checkpoint_callback],
            max_steps=total_steps, max_epochs=-1,
            logger=wandb_logger, log_every_n_steps=100, precision=16,
        )
        trainer.fit(model, train_loader)

        # Save model for this fraction
        model_save_dir = os.path.join(args.model_dir, f'pretrain_{args.pretrain_model_name}')
        os.makedirs(model_save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_save_dir, f'{current_model_name}.pth'))

        del train_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model


def main(args):
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    seed_everything(args.seed)

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    wandb_logger = WandbLogger(project=args.project_name)

    os.makedirs(args.model_dir, exist_ok=True)

    # Load map data and tokenizer
    nodes_to_indices, indices_to_nodes, adj_matrix = load_map_data(args.dataset_dir)
    tokenizer_path = os.path.join(args.model_dir, 'tokenizer.pth')
    tokenizer = load_tokenizer(tokenizer_path)

    # Build model and load pretrained weights
    model = build_model(tokenizer, args.n_head, indices_to_nodes, nodes_to_indices, adj_matrix, args.size_m, args.size_n)

    if args.experiment == 'longshort':
        pretrain_path = os.path.join(args.model_dir, 'pretrain_random_walk_10M_reveal', f'{args.pretrain_model_name}.pth')
    else:
        pretrain_path = os.path.join(args.model_dir, f'{args.pretrain_model_name}.pth')
    model.load_state_dict(torch.load(pretrain_path))

    # Determine model name and checkpoint dir
    saved_model_name = get_model_name(args)
    checkpoint_dir = os.path.join(args.checkpoint_dir, f"pretrain_{args.pretrain_model_name}", saved_model_name)

    # Load dataset based on experiment mode
    if args.experiment == 'cov_div':
        train_dataset = load_dataset_cov_div(args)
    elif args.experiment == 'qa':
        train_dataset = load_dataset_qa(args)
    elif args.experiment == 'longshort':
        train_dataset = load_dataset_longshort(args)
    elif args.experiment == 'spatial':
        train_dataset = load_dataset_spatial(args)
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")

    # Train
    if args.experiment == 'spatial':
        model = train_fractions(args, model, tokenizer, train_dataset, os.path.join(args.checkpoint_dir, f"pretrain_{args.pretrain_model_name}"), wandb_logger)
    else:
        model = train_single(args, model, tokenizer, train_dataset, checkpoint_dir, wandb_logger)

        # Save final model
        model_save_dir = os.path.join(args.model_dir, f'pretrain_{args.pretrain_model_name}')
        os.makedirs(model_save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_save_dir, f'{saved_model_name}.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for shortest-path models.")

    # Experiment mode
    parser.add_argument("--experiment", type=str, required=True,
                        choices=['cov_div', 'qa', 'longshort', 'spatial'],
                        help="Experiment type: cov_div (Sec 4.2), qa (Sec 4.1), longshort (Sec 5), spatial (Sec 4.1 variant)")

    # Common arguments
    parser.add_argument("--dataset_dir", type=str, default='dataset/')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--model_dir", type=str, default='models/')
    parser.add_argument("--pretrain_model_name", type=str, default='random_walk_10M_reveal')
    parser.add_argument("--dataset_name", type=str, default='shortest_path')
    parser.add_argument("--checkpoint_dir", type=str, default='ckpt')
    parser.add_argument("--devices", type=str, default="0,1")
    parser.add_argument("--project_name", type=str, default='gpt2-directions')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size_m", type=int, default=50)
    parser.add_argument("--size_n", type=int, default=40)
    parser.add_argument("--path_type", type=str, default='reveal', choices=['reveal', 'standard'])
    parser.add_argument("--save_per_steps", type=int, default=200)

    # Coverage/diversity arguments
    parser.add_argument("--coverage", type=float, default=0.6)
    parser.add_argument("--diversity", type=int, default=64)
    parser.add_argument("--pairs_idx", type=int, default=0)

    # QA arguments
    parser.add_argument("--num_ans", type=int, default=1)

    # Longshort arguments
    parser.add_argument("--group", type=str, default='(30,40)',
                        choices=['(20,30)', '(30,40)', '(50,60)', '(60,70)', '(70,80)'])
    parser.add_argument("--add_num", type=int, default=100)

    # Spatial arguments
    parser.add_argument("--data_frac", type=str, default='[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.8, 1.0]')

    args = parser.parse_args()
    main(args)
