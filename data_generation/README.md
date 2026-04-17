# Data Generation Pipeline

This directory contains all scripts for generating training/evaluation data.
Run scripts from the **project root** directory.

## Pipeline Overview

```
gen_grid_map.py                        Step 1: Generate sparse grid maps
       │
       ├── gen_random_walk.py          Step 2a: Random walks (pretraining)
       │       └── prepare_pretrain_dataset.py      → HF dataset
       │
       ├── gen_shortest_path.py        Step 2b: Base shortest paths
       │       └── prepare_dataset.py --mode base   → HF dataset
       │
       ├── spatial_length/             Step 2c: Spatial transfer & length scaling
       │       ├── construct_pairs.py               → coverage ratio pairs
       │       ├── construct_longshort_pairs.py      → distance-grouped pairs
       │       ├── gen_shortest_path.py              → paths (3 modes)
       │       └── split_shortest_path_qa.py         → Q vs A budget splits (2 modes)
       │
       └── diversity_coverage/         Step 2d: Coverage x diversity grid
               ├── construct_pairs.py               → (coverage, diversity) pair combos
               └── gen_shortest_path.py              → shortest paths per combo
```

All HF dataset conversion is handled by the unified `prepare_dataset.py`.

## Step-by-Step Instructions

### 1. Generate Grid Maps
```bash
python data_generation/gen_grid_map.py
```
Creates two disjoint sparse grid regions (G1 for training, G2 for spatial transfer testing)
with 25% edge removal while maintaining connectivity.

Output: `dataset/map_stats/` (adjacency matrix, node mappings)

### 2a. Pretraining Data (Random Walks)
```bash
python data_generation/gen_random_walk.py
python data_generation/prepare_pretrain_dataset.py
```
Generates 10M random-walk trajectories for pretraining.

### 2b. Base Shortest Paths
```bash
python data_generation/gen_shortest_path.py
python data_generation/prepare_dataset.py --mode base --max_len 27
```

### 2c. Spatial Transfer & Length Scaling (Sections 4.1, 5)

**Construct node pairs for coverage ratios:**
```bash
python data_generation/spatial_length/construct_pairs.py
```

**Generate shortest paths (3 modes in one script):**
```bash
# Single coverage (e.g., coverage=1.0, m=1 path per pair)
python data_generation/spatial_length/gen_shortest_path.py --mode coverage --coverage 1.0

# Incremental: build on base coverage (e.g., 0.6 on top of 0.2, m=128)
python data_generation/spatial_length/gen_shortest_path.py --mode incremental \
    --target_coverage 0.6 --base_coverage 0.2 --answer_num 128

# Length scaling: paths per distance group (Section 5)
python data_generation/spatial_length/gen_shortest_path.py --mode longshort
```

**Create question-answer budget splits (2 modes in one script):**
```bash
# Single coverage split (e.g., coverage=0.8, target 64 answers per question)
python data_generation/spatial_length/split_shortest_path_qa.py \
    --mode single --coverage 0.8 --avg_num_answers 64

# Sub-coverage splits (extract from base coverage=0.2)
python data_generation/spatial_length/split_shortest_path_qa.py \
    --mode sub --base_coverage 0.2 --sub_coverages 0.01,0.05,0.1 \
    --avg_num_answers 1,2,4,8,16,32,64
```

**For length scaling rescue (Section 5):**
```bash
python data_generation/spatial_length/construct_longshort_pairs.py
python data_generation/spatial_length/gen_shortest_path.py --mode longshort
```

**Convert all to HF datasets:**
```bash
python data_generation/prepare_dataset.py --mode qa --coverage 0.8
python data_generation/prepare_dataset.py --mode longshort
python data_generation/prepare_dataset.py --mode spatial --coverage 1.0
```

### 2d. Coverage x Diversity Grid (Section 4.2)
```bash
python data_generation/diversity_coverage/construct_pairs.py
python data_generation/diversity_coverage/gen_shortest_path.py
python data_generation/prepare_dataset.py --mode cov_div --diversity 64 --coverage 0.6
```

## Paper Section -> Script Mapping

| Paper Section | Scripts |
|---|---|
| Pretraining (Sec 2) | `gen_grid_map.py` -> `gen_random_walk.py` -> `prepare_pretrain_dataset.py` |
| Sec 4.1 (Q vs A) | `spatial_length/construct_pairs.py` -> `gen_shortest_path.py --mode incremental` -> `split_shortest_path_qa.py` -> `prepare_dataset.py --mode qa` |
| Sec 4.2 (Cov x Div) | `diversity_coverage/construct_pairs.py` -> `diversity_coverage/gen_shortest_path.py` -> `prepare_dataset.py --mode cov_div` |
| Sec 5 (Length rescue) | `spatial_length/construct_longshort_pairs.py` -> `gen_shortest_path.py --mode longshort` -> `prepare_dataset.py --mode longshort` |
