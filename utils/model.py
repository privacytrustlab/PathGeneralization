import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, GPTJForCausalLM, GPTJConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)
from transformers import LlamaForCausalLM, LlamaConfig
from pytorch_lightning import LightningModule
import numpy as np
from torch import nn
import torch.nn.functional as F
import math
import ast
from collections import namedtuple
import os
import json

from typing import Callable, List, Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from typing_extensions import Unpack
from transformers.utils import logging

logger = logging.get_logger(__name__)


from .data import get_neighbors, get_valid_neighbors

class DirectionTokenizer:
    def __init__(self, vocab_dict):
        self.vocab_dict = vocab_dict
        self.id_to_token = {idx: token for token, idx in vocab_dict.items()}
        self.pad_token = '<pad>'
        self.pad_token_id = vocab_dict[self.pad_token]
        self.bos_token  = '<s>'
        self.bos_token_id = vocab_dict[self.bos_token]
        self.eos_token = '</s>'
        self.eos_token_id = vocab_dict[self.eos_token]

        self.vocab_size = len(vocab_dict)

    def encode(self, text, add_special_tokens=False):
        token_ids = [self.vocab_dict.get(token, self.pad_token_id) for token in text.split()]
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        return token_ids

    # def decode(self, token_ids):
    #     if isinstance(token_ids, torch.Tensor):
    #       token_ids = token_ids.cpu().numpy()
    #     if token_ids.ndim == 0:
    #       token_ids = np.array([token_ids])
    #     if token_ids.ndim == 1:
    #         tokens = [self.id_to_token.get(idx, self.pad_token) for idx in token_ids if idx != self.pad_token_id]
    #         return ' '.join(tokens)
    #     elif token_ids.ndim == 2:
    #         decoded_sequences = []
    #         for sequence in token_ids:
    #             tokens = [self.id_to_token.get(idx, self.pad_token) for idx in sequence if idx != self.pad_token_id]
    #             decoded_sequences.append(' '.join(tokens))
    #         return decoded_sequences
    #     # Raise an error for unsupported dimensions
    #     else:
    #         raise ValueError("token_ids must be a scalar, 1D array, or 2D array.")

    def decode(self, token_ids):
        # Ensure token_ids is a NumPy array
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()
        if isinstance(token_ids, list):
            token_ids = np.array(token_ids)

        # Handle scalar or 1D inputs by converting them to batch-like 2D arrays
        if np.isscalar(token_ids):
            token_ids = np.array([[token_ids]])  # Shape: (1, 1)
        elif token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]  # Shape: (1, sequence_length)
        elif token_ids.ndim == 0:
            token_ids = np.array([[token_ids.item()]])  # Shape: (1, 1)

        # Batch-wise decoding
        decoded_sequences = [
            ' '.join(
                [self.id_to_token.get(idx, self.pad_token) for idx in sequence if idx != self.pad_token_id]
            )
            for sequence in token_ids
        ]

        # Return a single string if the input was not a batch
        if len(decoded_sequences) == 1:
            return decoded_sequences[0]
        return decoded_sequences



# class GPT2Model(LightningModule):
#     def __init__(self, tokenizer, n_embd=768, n_layer=12, n_head=12, **kwargs):
#         super().__init__()
#         self.save_hyperparameters()
#         config = GPT2Config(vocab_size=tokenizer.vocab_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head, pad_token_id=tokenizer.pad_token_id)
#         self.model = GPT2LMHeadModel(config)
#         self.tokenizer = tokenizer
        
#         self.validation_step_outputs = []
#         self.train_step_outputs = []

#         # Set nodes and idx_to_node if they are provided in kwargs
#         self.nodes_to_indices = kwargs.get('nodes')
#         self.indices_to_nodes = kwargs.get('idx_to_node')
#         self.connectivity_matrix = kwargs.get('connectivity_matrix')
#         self.size_m = kwargs.get('size_m')
#         self.size_n = kwargs.get('size_n')
        
#         # Check unexpected kwargs
#         unexpected_kwargs = {k: v for k, v in kwargs.items() if k not in ['nodes', 'idx_to_node','connectivity_matrix', 'size_m', 'size_n']}
#         if unexpected_kwargs:
#             print(f"Warning: Unexpected kwargs: {unexpected_kwargs}")

#     def forward(self, input_ids, attention_mask=None, labels=None, return_logits=True):
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         return outputs.loss

#     def training_step(self, batch, batch_idx):
#         loss = self(batch['input_ids'], batch['attention_mask'], batch['labels'])
#         self.log('train_loss', loss, prog_bar=True, logger=True)
#         self.train_step_outputs.append({'train_loss': loss})
#         return loss
    
#     def on_train_epoch_end(self):
#         avg_loss = torch.stack([x['train_loss'] for x in self.train_step_outputs]).mean()
#         self.log('train_loss', avg_loss, prog_bar=True, sync_dist=True)
#         self.train_step_outputs = []

#     def validation_step(self, batch, batch_idx):
#         # Calculate loss
#         loss = self(batch['input_ids'], batch['attention_mask'], batch['labels'])
#         self.log('val_loss', loss, prog_bar=True, sync_dist=True)

#         success_nodes = 0
#         total_nodes = 1e-6  # Avoid division by zero
#         bsz, _ = batch['input_ids'].shape

#         with torch.no_grad():
#             input_ids = batch['input_ids'].to(self.model.device)
#             mask = batch['attention_mask'].to(self.model.device)
#             outputs = self.model(input_ids, attention_mask=mask, labels=input_ids)
#             logits = outputs.logits
#             top_preds = torch.argmax(logits, dim=-1)

#         # Compute success nodes and total nodes
#         for i in range(bsz):
#             sequence_str = self.tokenizer.decode(batch['input_ids'][i])
#             sequence_list = sequence_str.split(" ")
#             # format: <s> start_node_idx end_node_idx : start_node_idx node1 node2 ... end_node_idx </s>
#             start_node_idx, end_node_idx = int(sequence_list[1]), int(sequence_list[2])
#             current_state = start_node_idx

#             for length_of_partial_sequence in range(5, len(sequence_list)):
#                 top_pred = top_preds[i, length_of_partial_sequence - 1]
#                 top_pred_str = self.tokenizer.decode(top_pred)
#                 total_nodes += 1
#                 next_str = sequence_list[length_of_partial_sequence]
#                 current_node = self.indices_to_nodes[current_state]
                
#                 # Ensure current_node is processed correctly
#                 if type(current_node) == str:
#                     current_node = ast.literal_eval(current_node)
                
#                 neighbors = get_valid_neighbors(current_node, self.connectivity_matrix, self.nodes_to_indices, self.size_m, self.size_n)
#                 valid_turns = [neighbor[1] for neighbor in neighbors]
                
#                 # Evaluate success criteria
#                 if top_pred_str in valid_turns or (top_pred_str == self.tokenizer.eos_token and current_state == end_node_idx) or (top_pred_str == str(current_state) and current_state == next_str):
#                     success_nodes += 1

#                 # Update current state
#                 if next_str != self.tokenizer.eos_token and not next_str.isdigit():
#                     next_node_at = valid_turns.index(next_str)
#                     current_state_node = neighbors[next_node_at][0]
#                     current_state = self.nodes_to_indices[str(current_state_node)]

#         self.validation_step_outputs.append({'val_loss': loss, 'total_nodes': total_nodes, 'success_nodes': success_nodes})
#         return loss

#     def on_validation_epoch_end(self):
#         # Compute average loss and success rate
#         avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
#         total_nodes = sum([x['total_nodes'] for x in self.validation_step_outputs])
#         success_nodes = sum([x['success_nodes'] for x in self.validation_step_outputs])
#         success_rate = success_nodes / total_nodes

#         # Log metrics
#         self.log('val_loss', avg_loss, prog_bar=True, sync_dist=True)
#         self.log('success_rate', success_rate, prog_bar=True, sync_dist=True)

#         # Clear outputs to avoid memory buildup
#         self.validation_step_outputs.clear()

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
#         return optimizer

class LlamaAttentionWithOutputs(LlamaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_outputs, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        self.attn_outputs = attn_outputs

        attn_output = attn_outputs.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    
class Model(LightningModule):
    def __init__(self, tokenizer, n_embd=768, n_layer=12, n_head=12, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # config = GPT2Config(vocab_size=tokenizer.vocab_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head, pad_token_id=tokenizer.pad_token_id)
        # self.model = GPT2LMHeadModel(config)
        config = LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=n_embd,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_position_embeddings=512,
            
        )
        self.model = LlamaForCausalLM(config)
        self.tokenizer = tokenizer
        
        self.validation_step_outputs = []
        self.train_step_outputs = []

        # Set nodes and idx_to_node if they are provided in kwargs
        self.nodes_to_indices = kwargs.get('nodes_to_indices')
        self.indices_to_nodes = kwargs.get('indices_to_nodes')
        self.connectivity_matrix = kwargs.get('connectivity_matrix')
        self.size_m = kwargs.get('size_m')
        self.size_n = kwargs.get('size_n')
        
        # Check unexpected kwargs
        unexpected_kwargs = {k: v for k, v in kwargs.items() if k not in ['indices_to_nodes', 'nodes_to_indices','connectivity_matrix', 'size_m', 'size_n']}
        if unexpected_kwargs:
            print(f"Warning: Unexpected kwargs: {unexpected_kwargs}")

    def forward(self, input_ids, attention_mask=None, labels=None, return_logits=True):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs if return_logits else outputs.loss

    def training_step(self, batch, batch_idx):
        loss = self(batch['input_ids'], batch['attention_mask'], batch['labels'], return_logits=False)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.train_step_outputs.append({'train_loss': loss})
        return loss
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack([x['train_loss'] for x in self.train_step_outputs]).mean()
        self.log('train_loss', avg_loss, prog_bar=True, sync_dist=True)
        self.train_step_outputs = []

    def validation_step(self, batch, batch_idx):
        # Calculate loss
        loss = self(batch['input_ids'], batch['attention_mask'], batch['labels'], return_logits=False)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

        success_nodes = 0
        total_nodes = 1e-6  # Avoid division by zero
        bsz, _ = batch['input_ids'].shape

        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.model.device)
            mask = batch['attention_mask'].to(self.model.device)
            outputs = self.model(input_ids, attention_mask=mask, labels=input_ids)
            logits = outputs.logits
            top_preds = torch.argmax(logits, dim=-1)

        # Compute success nodes and total nodes
        for i in range(bsz):
            sequence_str = self.tokenizer.decode(batch['input_ids'][i])
            sequence_list = sequence_str.split(" ")
            # format: <s> start_node_idx end_node_idx : start_node_idx node1 node2 ... end_node_idx </s>
            start_node_idx, end_node_idx = int(sequence_list[1]), int(sequence_list[2])
            current_state = start_node_idx

            for length_of_partial_sequence in range(5, len(sequence_list)):
                top_pred = top_preds[i, length_of_partial_sequence - 1]
                top_pred_str = self.tokenizer.decode(top_pred)
                total_nodes += 1
                next_str = sequence_list[length_of_partial_sequence]
                current_node = self.indices_to_nodes[current_state]
                
                # Ensure current_node is processed correctly
                if type(current_node) == str:
                    current_node = ast.literal_eval(current_node)
                
                neighbors = get_valid_neighbors(current_node, self.connectivity_matrix, self.nodes_to_indices, self.size_m, self.size_n)
                valid_turns = [neighbor[1] for neighbor in neighbors]
                
                # Evaluate success criteria
                if top_pred_str in valid_turns or (top_pred_str == self.tokenizer.eos_token and current_state == end_node_idx) or (top_pred_str == str(current_state) and current_state == next_str):
                    success_nodes += 1

                # Update current state
                if next_str != self.tokenizer.eos_token and not next_str.isdigit():
                    next_node_at = valid_turns.index(next_str)
                    current_state_node = neighbors[next_node_at][0]
                    current_state = self.nodes_to_indices[str(current_state_node)]

        self.validation_step_outputs.append({'val_loss': loss, 'total_nodes': total_nodes, 'success_nodes': success_nodes})
        return loss

    def on_validation_epoch_end(self):
        # Compute average loss and success rate
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        total_nodes = sum([x['total_nodes'] for x in self.validation_step_outputs])
        success_nodes = sum([x['success_nodes'] for x in self.validation_step_outputs])
        success_rate = success_nodes / total_nodes

        # Log metrics
        self.log('val_loss', avg_loss, prog_bar=True, sync_dist=True)
        self.log('success_rate', success_rate, prog_bar=True, sync_dist=True)

        # Clear outputs to avoid memory buildup
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=0.01)
        return optimizer


class PathGenModel(Model):
    def __init__(self, tokenizer, n_embd=768, n_layer=12, n_head=12, **kwargs):
        super().__init__(tokenizer, n_embd, n_layer, n_head, **kwargs)

    # def forward(self, input_ids, attention_mask=None, labels=None, return_logits=True):
    #     outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #     return outputs.loss

    def forward(self, input_ids, attention_mask=None, labels=None, return_logits=True):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs if return_logits else outputs.loss

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]

        # Mask tokens before the colon ':' dynamically
        for i in range(labels.size(0)):
            colon_idx = (input_ids[i] == self.tokenizer.encode(":")[0]).nonzero(as_tuple=True)[0].item()
            labels[i, :colon_idx + 1] = -100  # Mask all tokens before and including the colon

        # Compute loss
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Log loss
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.train_step_outputs.append({"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        # Calculate loss
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]

        # Mask tokens before the colon ':' dynamically
        for i in range(labels.size(0)):
            colon_idx = (input_ids[i] == self.tokenizer.encode(":")[0]).nonzero(as_tuple=True)[0].item()
            labels[i, :colon_idx + 1] = -100  # Mask all tokens before and including the colon

        loss = self(input_ids, attention_mask, labels, return_logits=False)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

        success_nodes = 0
        total_nodes = 1e-6  # Avoid division by zero
        bsz, _ = batch['input_ids'].shape

        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.model.device)
            mask = batch['attention_mask'].to(self.model.device)
            outputs = self.model(input_ids, attention_mask=mask, labels=input_ids)
            logits = outputs.logits
            top_preds = torch.argmax(logits, dim=-1)

        # Compute success nodes and total nodes
        for i in range(bsz):
            sequence_str = self.tokenizer.decode(batch['input_ids'][i])
            sequence_list = sequence_str.split(" ")
            # # format: <s> start_node_idx mid_idx1 mid_idx2 mid_idx3 end_node_idx : start_node_idx node1 node2 ... end_node_idx </s>
            # if len(sequence_list) == 137:
            #     start_node_idx, end_node_idx = int(sequence_list[1]), int(sequence_list[5])
            #     current_state = start_node_idx
            #     for length_of_partial_sequence in range(8, len(sequence_list)):
            #         top_pred = top_preds[i, length_of_partial_sequence - 1]
            #         top_pred_str = self.tokenizer.decode(top_pred)
            #         total_nodes += 1
            #         next_str = sequence_list[length_of_partial_sequence]
            #         current_node = self.indices_to_nodes[current_state]
                    
            #         # Ensure current_node is processed correctly
            #         if type(current_node) == str:
            #             current_node = ast.literal_eval(current_node)
                    
            #         neighbors = get_valid_neighbors(current_node, self.connectivity_matrix, self.nodes_to_indices)
            #         valid_turns = [neighbor[1] for neighbor in neighbors]
                    
            #         # Evaluate success criteria
            #         if top_pred_str in valid_turns or (top_pred_str == self.tokenizer.eos_token and current_state == end_node_idx) or (top_pred_str == str(current_state) and current_state == next_str):
            #             success_nodes += 1

            #         # Update current state
            #         if next_str != self.tokenizer.eos_token and not next_str.isdigit():
            #             next_node_at = valid_turns.index(next_str)
            #             current_state_node = neighbors[next_node_at][0]
            #             current_state = self.nodes_to_indices[str(current_state_node)]
            # else: # format: <s> start_node_idx end_node_idx : start_node_idx node1 node2 ... end_node_idx </s>
            start_node_idx, end_node_idx = int(sequence_list[1]), int(sequence_list[2])
            current_state = start_node_idx

            for length_of_partial_sequence in range(5, len(sequence_list)):
                top_pred = top_preds[i, length_of_partial_sequence - 1]
                top_pred_str = self.tokenizer.decode(top_pred)
                total_nodes += 1
                next_str = sequence_list[length_of_partial_sequence]
                current_node = self.indices_to_nodes[current_state]
                
                # Ensure current_node is processed correctly
                if type(current_node) == str:
                    current_node = ast.literal_eval(current_node)
                
                neighbors = get_neighbors(current_node)
                valid_turns = [neighbor[1] for neighbor in neighbors]
                
                # Evaluate success criteria
                if top_pred_str in valid_turns or (top_pred_str == self.tokenizer.eos_token and current_state == end_node_idx) or (top_pred_str == str(current_state) and current_state == next_str):
                    success_nodes += 1

                # Update current state
                if next_str != self.tokenizer.eos_token and not next_str.isdigit():
                    next_node_at = valid_turns.index(next_str)
                    current_state_node = neighbors[next_node_at][0]
                    current_state = self.nodes_to_indices[str(current_state_node)]

        self.validation_step_outputs.append({'val_loss': loss, 'total_nodes': total_nodes, 'success_nodes': success_nodes})
        return loss