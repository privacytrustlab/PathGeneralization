# Generalization in LLM Problem Solving: The Case of the Shortest Path

[![Paper](https://img.shields.io/badge/arXiv-2604.15306-b31b1b.svg)](https://arxiv.org/abs/2604.15306)
[![Conference](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://openreview.net/)

Yao Tong, Jiayuan Ye, Anastasia Borovykh, Reza Shokri

Official implementation of the ICLR 2026 paper *"Generalization in LLM Problem Solving: The Case of the Shortest Path"*.

> Whether language models can systematically generalize remains actively debated. We introduce a controlled synthetic environment based on shortest-path planning and study two orthogonal axes of generalization: **spatial transfer** to unseen maps and **length scaling** to longer-horizon problems.

## Key Findings

**1. Models can spatially transfer to entirely unseen maps, providing evidence of systematic structural generalization, but fail under length scaling, primarily due to recursive instability.**

<p align="left">
  <img src="figures/Fig1.png" width="45%" alt="Spatial transfer vs. length scaling"/>
  &nbsp;
  <img src="figures/Tab1.png" width="40%" alt="Composition analysis of length scaling failure"/>
</p>

**2. Training data primarily determines capability limits**: allocating training budget to more distinct questions rather than more solutions yields better transfer generalization. Selecting questions with broader primitive coverage is more beneficial than more diverse primitive combinations. Length scaling requires exposure to slightly longer training examples.

**3. RL (GRPO) stabilizes training but does not surpass the performance ceiling of SFT**, and exhibits similar error patterns. RL remains below the best SFT performance even when stronger inference strategies are used to better unlock intrinsic capability, and appears to restrict the effective solution space.

<p align="left">
  <img src="figures/Fig6.png" width="85%" alt="SFT vs GRPO length scaling across training steps"/>
</p>

**4. Advanced inference-time strategies** can improve performance for both SFT and RL models but **cannot rescue length scaling failures**.

<p align="left">
  <img src="figures/Fig7.png" width="55%" alt="Length scaling under different decoding strategies"/>
</p>

## Repository Structure

```
PathGeneralization/
├── src/
│   ├── pretrain.py              # Pretraining on random-walk trajectories
│   ├── sft.py                   # Supervised fine-tuning (all experiment modes)
│   └── rl.py                    # GRPO reinforcement learning
├── utils/                       # Model, data, evaluation utilities
├── data_generation/             # Scripts to generate maps, paths, and datasets
│   ├── spatial_length/          # Spatial transfer & length scaling data
│   └── diversity_coverage/      # Coverage-diversity grid data
├── evaluation/                  # Evaluation scripts
│   ├── valid_rate.py            # Success rate computation
│   └── failure_plotting_utils.py# Failure case analysis & visualization
├── notebooks/
│   ├── plot_figures.ipynb       # Reproduce Fig 2-5, 8, 10-11 (no GPU needed)
│   └── inference_figures.ipynb  # Reproduce Fig 1, Table 1, failure cases (GPU)
├── results/                     # Pre-computed eval_results.json for all experiments
├── requirements.txt
└── .gitignore
```

## Setup

```bash
# Create a conda environment (recommended)
conda create -n pathgen python=3.10
conda activate pathgen

# Install PyTorch (adjust for your CUDA version, see https://pytorch.org)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install remaining dependencies
pip install -r requirements.txt
```

Optionally, set your W&B API key for training logging:
```bash
export WANDB_API_KEY=your_key_here
```

## Training Pipeline

### 1. Pretraining (random walks)
```bash
python src/pretrain.py --dataset_name random_walk_10M --n_head 8 --max_epochs 3
```

### 2. Supervised Fine-Tuning

Coverage vs. Diversity (Section 4.2):
```bash
python src/sft.py --experiment cov_div --coverage 0.6 --diversity 64
```

More Questions vs. More Answers (Section 4.1):
```bash
python src/sft.py --experiment qa --coverage 0.2 --num_ans 4
```

Length scaling rescue (Section 5):
```bash
python src/sft.py --experiment longshort --group "(30,40)" --add_num 1000
```

### 3. Reinforcement Learning (Section 6-7)
```bash
python src/rl.py --experiment spatial --pretrain_model_name <sft_model> --coverage 0.2 --num_generation 8
```

### 4. Evaluation
```bash
python evaluation/valid_rate.py --mode sft --coverage 0.6 --num_ans 1
```

## Data & Models

We release representative datasets (on 50×50 maps) and the selected model checkpoints on HuggingFace. The full sweep contains several hundred model variants across different (coverage, diversity, num_ans, ...) combinations — to keep the release focused, we upload only the artifacts needed to reproduce the paper's core figures. **If you need additional checkpoints or datasets, please email [tongyao@u.nus.edu](mailto:tongyao@u.nus.edu).**

| Resource | HuggingFace Link | Description |
|---|---|---|
| Pretrain data | [`YnezT/PathGen-random-walk-10M`](https://huggingface.co/datasets/YnezT/PathGen-random-walk-10M) | 10M random-walk trajectories (pretraining) |
| Shortest-path data | [`YnezT/PathGen-shortest-path`](https://huggingface.co/datasets/YnezT/PathGen-shortest-path) | Consolidated shortest-path datasets; each `(coverage, diversity)` / `(coverage, num_ans)` / `longshort group` variant is a separate HF config |
| Map data | [`YnezT/PathGen-maps`](https://huggingface.co/datasets/YnezT/PathGen-maps) | Grid maps (adjacency matrix, node/index mappings, G1/G2 indices) |
| Pretrained model | [`YnezT/PathGen-pretrained`](https://huggingface.co/YnezT/PathGen-pretrained) | Pretrained model (random walk) |
| Best SFT model | [`YnezT/PathGen-sft`](https://huggingface.co/YnezT/PathGen-sft) | Best SFT checkpoint (coverage=0.60, ans=64) |
| Best GRPO model | [`YnezT/PathGen-grpo`](https://huggingface.co/YnezT/PathGen-grpo) | Best GRPO checkpoint (coverage=0.60, ans=4, rollout=16, from step 1800) |

**Example usage** (loading a specific shortest-path variant):
```python
from datasets import load_dataset

# Coverage x Diversity experiment (Section 4.2)
ds = load_dataset("YnezT/PathGen-shortest-path", "cov_div_d64_c060", split="train")

# More Questions vs More Answers (Section 4.1)
ds = load_dataset("YnezT/PathGen-shortest-path", "qa_c060_ans64", split="train")

# Length scaling rescue (Section 5)
ds = load_dataset("YnezT/PathGen-shortest-path", "longshort_30_40", split="train")
```

To regenerate data from scratch, see [data_generation/README.md](data_generation/README.md).

## Citation

```bibtex
@inproceedings{tonggeneralization,
  title={Generalization in LLM Problem Solving: The Case of the Shortest Path},
  author={Tong, Yao and Ye, Jiayuan and Borovykh, Anastasia and Shokri, Reza},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```
