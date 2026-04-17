import numpy as np

from .data import get_neighbors, get_valid_neighbors

# format: <s> start_node_idx end_node_idx : start_node_idx node1 node2 ... end_node_idx </s>
def is_valid_sequence(sample, tokenizer, nodes, idx_to_node):
  generated_list = sample.split(" ")
  start_node_idx, end_node_idx = int(generated_list[1]), int(generated_list[2])
  directions = generated_list[5:]
  current_state = start_node_idx
  state_seq = [current_state]
  for i, direction in enumerate(directions):
    if direction != tokenizer.eos_token and not direction.isdigit():
      current_node = idx_to_node[current_state]
      neighbors = get_neighbors(current_node)
      valid_turns = [neighbor[1] for neighbor in neighbors]
      if direction in valid_turns:
        next_node_at = valid_turns.index(direction)
        current_state_node = neighbors[next_node_at][0]
        current_state = nodes[str(current_state_node)]
        state_seq.append(current_state)
      else:
        return False, state_seq
    else:
        if direction == tokenizer.eos_token and current_state == end_node_idx:
            return True, state_seq
        elif direction == str(current_state):
            continue
        else:
            return False, state_seq
  return False, state_seq

def is_valid_sequence_non_fully(sample, tokenizer, nodes, idx_to_node, connectivity_matrix, m_size, n_size):
  generated_list = sample.split(" ")
  start_node_idx, end_node_idx = int(generated_list[1]), int(generated_list[2])
  directions = generated_list[5:]
  current_state = start_node_idx
  state_seq = [current_state]
  for i, direction in enumerate(directions):
    if direction != tokenizer.eos_token and not direction.isdigit():
      current_node = idx_to_node[current_state]
      # print(f"current_state: {current_state}; current_node: {current_node}; direction: {direction}")
      neighbors = get_valid_neighbors(current_node, connectivity_matrix, nodes, m_size, n_size)
      valid_turns = [neighbor[1] for neighbor in neighbors]
      if direction in valid_turns:
        next_node_at = valid_turns.index(direction)
        current_state_node = neighbors[next_node_at][0]
        current_state = nodes[current_state_node]
        state_seq.append(current_state)
        # print(f"Move to current_state: {current_state}; current_state_node: {current_state_node}")
      else:
        return False, state_seq
    else:
        if direction == tokenizer.eos_token and current_state == end_node_idx:
            return True, state_seq
        elif direction == str(current_state): # print valid current node
            continue
        else:
            return False, state_seq
            # print(f"direction: {direction}; current_state: {current_state}")
  return False, state_seq

# format: <s> start_node_idx mid_idx1 mid_idx2 mid_idx3 end_node_idx : start_node_idx node1 node2 ... end_node_idx </s>
def is_valid_sequence_with_middle(sample, tokenizer, nodes, idx_to_node, connectivity_matrix):
  generated_list = sample.split(" ")
  start_node_idx, end_node_idx = int(generated_list[1]), int(generated_list[5])
  directions = generated_list[8:]
  current_state = start_node_idx
  state_seq = [current_state]
  for i, direction in enumerate(directions):
    if direction != tokenizer.eos_token and not direction.isdigit():
      current_node = idx_to_node[current_state]
      # print(f"current_state: {current_state}; current_node: {current_node}; direction: {direction}")
      neighbors = get_valid_neighbors(current_node, connectivity_matrix, nodes)
      valid_turns = [neighbor[1] for neighbor in neighbors]
      if direction in valid_turns:
        next_node_at = valid_turns.index(direction)
        current_state_node = neighbors[next_node_at][0]
        current_state = nodes[str(current_state_node)]
        state_seq.append(current_state)
        # print(f"Move to current_state: {current_state}; current_state_node: {current_state_node}")
      else:
        return False
    else:
        if direction == tokenizer.eos_token and current_state == end_node_idx:
            return True
        elif direction == str(current_state):
            continue
        else:
            return False
            # print(f"direction: {direction}; current_state: {current_state}")
  return False

def is_valid_prompt(sample, tokenizer, nodes, idx_to_node):
  generated_list = sample.split(" ")
  start_node_idx, end_node_idx = int(generated_list[1]), int(generated_list[2])
  directions = generated_list[5:]
  current_state = start_node_idx
  state_seq = [current_state]
  for i, direction in enumerate(directions):
    if direction != tokenizer.eos_token and not direction.isdigit():
      current_node = idx_to_node[current_state]
      neighbors = get_neighbors(current_node)
      valid_turns = [neighbor[1] for neighbor in neighbors]
      if direction in valid_turns:
        next_node_at = valid_turns.index(direction)
        current_state_node = neighbors[next_node_at][0]
        current_state = nodes[str(current_state_node)]
        state_seq.append(current_state)
      else:
        return False
    else:
        if direction == str(current_state):
          continue
  return True