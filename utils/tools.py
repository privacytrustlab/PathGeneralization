import yaml
import pickle
import os
import numpy as np

# Load configuration from config.yaml
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def save_pickle(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

def save_numpy(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, data)

def save_shard(samples, path):
    """ Save to bin file """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # if len(samples) <= 50000:
    #     with open(path, 'wb') as f:
    #         pickle.dump(samples, f)
    # else:
    #     raise ValueError("Number of samples exceeds 50000")
    with open(path, 'wb') as f:
        pickle.dump(samples, f)
        