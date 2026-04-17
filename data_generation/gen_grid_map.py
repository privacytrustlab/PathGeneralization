import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils import save_pickle, save_numpy, load_config
from utils import get_indices, get_nodes

def generate_sparsified_grid(rows, cols, drop_percentage=0.25):
    """
    Generates a fully connected grid graph, removes alpha percent of adjacency edges while maintaining connectivity.
    Uses Prim's algorithm for MST to ensure all nodes remain connected.
    """
    # Step 1: Create a full grid graph
    G = nx.grid_2d_graph(rows, cols)

    # Step 2: Compute MST using Prim’s algorithm (ensures full connectivity)
    mst = nx.minimum_spanning_tree(G, algorithm='prim')

    # Step 3: Identify removable edges (edges not in MST)
    all_edges = set(G.edges())
    mst_edges = set(mst.edges())
    removable_edges = list(all_edges - mst_edges)  # Only consider non-MST edges

    # Step 4: Randomly remove alpha% of the removable edges
    num_edges_to_remove = int(len(all_edges) * drop_percentage)
    random.shuffle(removable_edges)

    for edge in removable_edges:
        G.remove_edge(*edge)  # Try removing
        # if not nx.is_connected(G):  # Ensure connectivity
        #     G.add_edge(*edge)  # Revert if disconnected
        
        num_edges_to_remove -= 1
        if num_edges_to_remove <= 0:
            break  # Stop when enough edges are removed

    return G

def offset_grid(G, col_offset=0, row_offset=0):
    """Shift all nodes in the grid by (row_offset, col_offset)."""
    H = nx.Graph()
    for u, v in G.edges():
        u_new = (u[0] + row_offset, u[1] + col_offset)
        v_new = (v[0] + row_offset, v[1] + col_offset)
        H.add_edge(u_new, v_new)
    return H

def create_disjoint_combined_graph(rows, cols, drop_percentage=0.25):
    region_size = rows * cols

    G1_grid = generate_sparsified_grid(rows, cols, drop_percentage)
    G2_grid = generate_sparsified_grid(rows, cols, drop_percentage)
    G2_offset = offset_grid(G2_grid, col_offset=cols)
    G_combined = nx.compose(G1_grid, G2_offset)
    return G_combined, G1_grid, G2_offset

def visualize_graph(G):
    """
    Visualizes the input graph G.
    """
    pos = {node: (node[1], -node[0]) for node in G.nodes()}  # Layout for visualization
    nx.draw(G, pos, with_labels=False, node_size=1, edge_color="gray")
    plt.savefig("grid_graph.png")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, "../../"))
    config_path = os.path.join(parent_dir, "config.yaml")

    config = load_config(config_path)
    rows = config["dataset"]["grid"]["size_m"]
    cols = config["dataset"]["grid"]["size_n"]
    drop_percentage = config["dataset"]["grid"]["drop_edge_percent"]
    save_dir = config["dataset"]["dir"]
    save_dir = os.path.join(parent_dir, save_dir, "map_stats")
    os.makedirs(save_dir, exist_ok=True)

    G_combined, G1_grid, G2_offset =  create_disjoint_combined_graph(rows, cols, drop_percentage)
    visualize_graph(G_combined)
    print("Number of nodes: ", G_combined.number_of_nodes())
    print("Number of edges: ", G_combined.number_of_edges())
    adj_matrix = nx.to_numpy_array(G_combined, weight="weight")
    nodes_to_indices = get_nodes(G_combined)
    indices_to_nodes = get_indices(G_combined)
    nodes_G1 = set(G1_grid.nodes())
    nodes_G2 = set(G2_offset.nodes())
    G1_indices = [nodes_to_indices[n] for n in nodes_G1]
    G2_indices = [nodes_to_indices[n] for n in nodes_G2]

    # check the boudaries of the two grids
    boundary_G1 = [(i, cols - 1) for i in range(rows) if (i, cols - 1) in nodes_G1]
    boundary_G2 = [(i, cols) for i in range(rows) if (i, cols) in nodes_G2]
    for n1, n2 in zip(boundary_G1, boundary_G2):
        i1 = nodes_to_indices[n1]
        i2 = nodes_to_indices[n2]
        assert adj_matrix[i1, i2] == 0 and adj_matrix[i2, i1] == 0, \
            f"Boundary nodes {n1} and {n2} should not be connected."

    # save the grid
    save_pickle(nodes_to_indices, os.path.join(save_dir, "nodes_to_indices.pkl"))
    save_pickle(indices_to_nodes, os.path.join(save_dir, "indices_to_nodes.pkl"))
    save_numpy(adj_matrix, os.path.join(save_dir, "adj_matrix.npy"))
    save_pickle(G1_indices, os.path.join(save_dir, "G1_indices.pkl"))
    save_pickle(G2_indices, os.path.join(save_dir, "G2_indices.pkl"))
    save_pickle(nodes_G1, os.path.join(save_dir, "nodes_G1.pkl"))
    save_pickle(nodes_G2, os.path.join(save_dir, "nodes_G2.pkl"))





   