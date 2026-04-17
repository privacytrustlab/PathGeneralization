"""
Pretraining on random-walk trajectories.

Pretrain a LLaMA-style model on random-walk paths to learn map structure
(node adjacency) without leaking shortest-path information.

Usage:
  python src/pretrain.py --dataset_name random_walk_10M --n_head 8 --max_epochs 3
"""

import pickle
import random
import os

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
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
    args.tokenizer_path = os.path.join(args.model_dir, 'tokenizer.pth')
    args.saved_model_name = f'{args.saved_model_name}_{args.path_type}'
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.saved_model_name)

    # Load map data
    with open(os.path.join(args.dataset_dir, "map_stats", 'nodes_to_indices.pkl'), 'rb') as f:
        nodes_to_indices = pickle.load(f)
    with open(os.path.join(args.dataset_dir, "map_stats", 'indices_to_nodes.pkl'), 'rb') as f:
        indices_to_nodes = pickle.load(f)
    adj_matrix = np.load(os.path.join(args.dataset_dir, "map_stats", 'adj_matrix.npy'))
    indices_to_nodes = {k: str(v) for k, v in indices_to_nodes.items()}
    nodes_to_indices = {str(k): v for k, v in nodes_to_indices.items()}

    # Prepare tokenizer
    if not os.path.exists(args.tokenizer_path):
        node_tokens = list(map(str, nodes_to_indices.values()))
        direction_tokens = ['N', 'S', 'W', 'E', 'STAY']
        special_tokens = ['<s>', '<pad>', '</s>', ':']
        vocab = node_tokens + direction_tokens + special_tokens
        vocab_dict = {token: idx for idx, token in enumerate(vocab)}
        tokenizer = utils.DirectionTokenizer(vocab_dict)
        torch.save(tokenizer, args.tokenizer_path)
    else:
        tokenizer = torch.load(args.tokenizer_path)

    # Load dataset
    if os.path.exists(os.path.join(args.dataset_dir, args.dataset_name)):
        dataset = DatasetDict.load_from_disk(os.path.join(args.dataset_dir, args.dataset_name))
    else:
        dataset_name = f'YnezT/{args.dataset_name}'
        dataset = load_dataset(dataset_name)

    train_dataset = dataset['train']
    test_dataset = dataset['test'].select(range(5000))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, args.path_type, tokenizer.pad_token_id),
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        collate_fn=lambda batch: collate_fn(batch, args.path_type, tokenizer.pad_token_id),
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    del dataset, train_dataset, test_dataset

    # Build model
    head_to_embd = {8: 512, 12: 768, 16: 1024, 20: 1280, 25: 1600}
    n_head = args.n_head
    model = utils.Model(
        tokenizer, n_embd=head_to_embd[n_head], n_layer=8, n_head=n_head,
        indices_to_nodes=indices_to_nodes, nodes_to_indices=nodes_to_indices,
        connectivity_matrix=adj_matrix, size_m=args.size_m, size_n=args.size_n * 2
    )

    # Checkpoint callback
    class CustomModelCheckpoint(ModelCheckpoint):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
            global_step = trainer.global_step
            epoch = trainer.current_epoch
            filename = os.path.join(self.dirpath, f'gpt2-{epoch:02d}-{global_step:06d}')
            if epoch == 0:
                if (global_step < 10000 and global_step % 1000 == 0) or global_step in {13000, 17000, 21000, 25000, 29000, 33000, 37000, 41000}:
                    self._save_checkpoint(trainer, filename)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_callback = CustomModelCheckpoint(
        dirpath=args.checkpoint_dir, save_top_k=-1, save_last=True, verbose=True
    )

    # Resume check
    last_checkpoint = None
    if args.resume_training and os.path.exists(args.checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(args.checkpoint_dir) if f.endswith('.ckpt')]
        if checkpoint_files:
            last_checkpoint = os.path.join(args.checkpoint_dir, sorted(checkpoint_files)[-1])
            print(f"Resuming from checkpoint: {last_checkpoint}")

    # Parse devices
    if isinstance(args.devices, str) and "," in args.devices:
        devices = [int(d) for d in args.devices.split(",")]
    elif args.devices.isdigit():
        devices = int(args.devices)
    else:
        devices = args.devices

    trainer = pl.Trainer(
        devices=devices, strategy='ddp', accelerator='gpu',
        callbacks=[checkpoint_callback],
        max_epochs=args.max_epochs,
        logger=wandb_logger, log_every_n_steps=100, precision=16,
        val_check_interval=0.25,
    )

    if args.resume_training and last_checkpoint:
        trainer.fit(model, train_loader, val_loader, ckpt_path=last_checkpoint)
    else:
        trainer.fit(model, train_loader, val_loader)

    # Save model
    model_path = os.path.join(args.model_dir, f'{args.saved_model_name}.pth')
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain model on random walks.")

    parser.add_argument("--dataset_dir", type=str, default='dataset/')
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--model_dir", type=str, default='models/')
    parser.add_argument("--dataset_name", type=str, default='random_walk_10M')
    parser.add_argument("--checkpoint_dir", type=str, default='ckpt/')
    parser.add_argument("--resume_training", action='store_true')
    parser.add_argument("--devices", type=str, default="0,1,2,3")
    parser.add_argument("--saved_model_name", type=str, default='pretrain_model')
    parser.add_argument("--project_name", type=str, default='gpt2-directions')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size_m", type=int, default=50)
    parser.add_argument("--size_n", type=int, default=40)
    parser.add_argument("--path_type", type=str, default='reveal', choices=['reveal', 'standard'])

    args = parser.parse_args()
    main(args)
