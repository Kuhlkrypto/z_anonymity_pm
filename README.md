# Filtering at the Edge: Exploring the Privacy-Utility Trade-Off

This repository contains the implementation and experimental code for the paper **"Filtering at the Edge: Exploring the Privacy-Utility Trade-Off"** by Maximilian Weisenseel, Fabian Sandkuhl, Henrik Kirchmann, Matthias Weidlich and Florian Tschorsch, submitted to COMINDS2025: 4th Workshop on Collaboration Mining for Distributed Systems at ICPM 2025.

## Overview

This work presents an approach for **distributed filtering of event data directly at event sources**.

The project implements **z-anonymity** and **explicit z-anonymity** techniques that filter event streams based on different behavioral predicates, ensuring that only behavioral information observed by at least `z` distinct cases within a time window is released. This approach realizes fundamental privacy design strategies of *minimization* and *separation* by design.


## Project Structure

```
z_anonymity_pm/
├── data/
│   ├── input/                    # Event log datasets (.xes.gz files)
│   │   ├── env_permit.xes.gz          # Environmental Permit dataset
│   │   ├── Sepsis.xes.gz              # Sepsis Cases dataset  
│   │   ├── BPI_Challenge_2012_O.xes.gz # BPI Challenge 2012 (Offers)
│   │   └── BPIC20_PrepaidTravelCost.xes.gz # BPI Challenge 2020
│   └── output/                   # Experimental results for each event log and generated figures
│       ├── env_permit/           
│       ├── Sepsis/               
│       ├── BPIC_2012_O/          
│       ├── BPIC20_PTC/           
│       └── figures/              # Generated figures
├── src/
│   ├── anonymization/            # Anonymization algorithm implementations
│   │   ├── z_anonymity.py            # Standard z-anonymity (this also implemented as ngram_z_anonymity.py with n =1)
│   │   ├── ngram_z_anonymity.py      # N-gram based z-anonymity (what we use in the paper)
│   │   └── baseline.py               # Baseline anonymization (also used in the paper)
│   ├── evaluation/               # Privacy and utility metrics
│   │   ├── metrics.py                # Ratio of remaining directly follows relations and fitness valus
│   │   ├── reidentification_risk.py  # Reidentification protection
│   │   ├── follows_relations.py      # F1 scores of follow relations (not used for the paper)
│   │   └── event_log_stats.py        # Ratio of remaining events and traces
│   ├── utils/                    # Helper utilities
│   │   └── log_utils.py              # Event log processing utilities
│   ├── test_z_anonymity.py       # Main experimental script
│   ├── visualise.py              # Results visualization and plotting
│   └── definitions.py            # Project configuration and paths
├── requirements.txt              # Python dependencies
└── README.md                    # This file
```

## Setup

Make sure you are using **Python 3.11** to run the scripts.

Install all required packages using the [`requirements.txt`](requirements.txt) file.

## Reproducing Paper Experiments

### Running the Complete Experiment Suite

To reproduce all results from the paper:

```bash
python src/test_z_anonymity.py
```

This script will:
1. Process all four datasets in the `data/input/` directory
2. Run experiments with z-values from 1 to 30
3. Test n-gram sizes of 1, 2, and 3
4. Compare explicit vs. non-explicit modes
5. Generate results in JSON format stored in `data/output/{dataset_name}/`

### Experimental Configuration

**Paper Methodology**: The experiments simulate distributed event streams by treating each distinct `org:resource` (or `org:group` for Sepsis) as an originating stream, modeling the event logs as collections of distributed, concurrent data streams.

The evaluation workflow:
1. **Data Loading**: Loads event logs from `data/input/`
2. **Source Simulation**: Partitions logs by `org:resource`/`org:group` 
3. **Filtering**: Applies z-anonymity algorithms to each source independently
4. **Reassembly**: Combines filtered streams preserving trace order
5. **Metric Calculation**: Computes privacy and utility metrics
6. **Results Storage**: Saves detailed results as JSON files in `data/output/`

The experiments use the following parameters:
- **Z-values**: 1 to 30 (sweeping anonymity parameter)
- **Time windows**: 259,200 seconds (72 hours/3 days)
- **Behavioral Predicates**: Activity sequences of length 1, 2, and 3
- **Algorithms**: Standard z-anonymity, explicit z-anonymity, and centralized baseline
- **Parallelization**: Uses 80% of available CPU cores for efficiency
  - **Global configuration**: Modify `cores_to_use = int(total_cores * 0.8)` in [`src/test_z_anonymity.py`](src/test_z_anonymity.py)
    - Change multiplier: `0.5` for 50% of cores, `1.0` for all cores, etc.

### Runtime Flags

Two top-level flags at the top of `main()` in [`src/test_z_anonymity.py`](src/test_z_anonymity.py) control the conformance-checking strategy and the parallelism mode:

| Flag | Default | Effect |
|---|---|---|
| `ALIGNMENT_BASED` | `True` | When `True`, fitness and precision are computed via **alignment-based** conformance checking (more accurate, slower). When `False`, the faster **token-based replay** is used instead. |
| `TASK_INTERNAL_MULTI_PROCESSING` | `True` | When `True`, tasks run **sequentially** and pm4py's own internal thread pool is free to use all CPU cores (good for single-machine, alignment-heavy runs); the **re-identification risk assessment** also runs in parallel (`n_jobs=cpu_count()`). When `False`, an outer `multiprocessing.Pool` spawns one worker process per task in parallel, with pm4py internal parallelism and risk assessment both capped at a single thread per worker (`n_jobs=1`) to avoid nested process spawning. |


### Running the Evaluation

To execute the complete evaluation pipeline that reproduces the paper results:

#### 1. Run Main Experiments
```bash
# Run all experiments (this may take several hours)
python src/test_z_anonymity.py
```

#### 2. Generate Visualizations  
```bash
# Create comparison plots and figures
python src/visualise.py
```


**Customizing Visualizations**

You can modify the visualization configuration in `src/visualise.py`:

```python
# Current configuration includes all four datasets
LOG_NAMES = ["env_permit", "Sepsis", "BPIC_2012_O", "BPIC20_PTC"]
SAVE_PNG = True  # Enable PNG export in addition to PDF
```

## Evaluation Metrics

Following the paper's methodology, the implementation evaluates both privacy and utility using specific metrics:

### Privacy Metrics

#### Re-identification Protection (A* Projection)
- **Implementation**: [`src/evaluation/reidentification_risk.py`](src/evaluation/reidentification_risk.py)
  - Function: `calculate_reidentification_risk()` (lines 132-396)
  - Specific A* projection: `detailed_counts_projection_A()` (lines 70-129)
- **Methodology**: Samples k = ⌈0.1 × max_trace_length⌉ activity-timestamp pairs with day-granular timestamps
- **Assessment**: Measures fraction of traces that remain uniquely identifiable 
- **Protection Score**: Protection = 1 - (fraction of unique traces) 
- **Interpretation**: Higher values indicate stronger protection against re-identification attacks

### Utility Metrics

#### Data Preservation Metrics
- **Implementation**: [`src/evaluation/event_log_stats.py`](src/evaluation/event_log_stats.py)
  - Function: `get_event_log_stats()` (lines 1-14)
- **Ratio of Remaining Events (RRE)**: `event_removal_rate` - Fraction of original events retained after filtering
- **Ratio of Remaining Traces (RRT)**: `trace_removal_rate` - Fraction of original traces retained after filtering

#### Behavioral Preservation Metrics  
- **Implementation**: [`src/evaluation/metrics.py`](src/evaluation/metrics.py)
- **Preservation of Directly-Follows Relations (RDF)**: Fraction of original directly-follows relations preserved.

#### Process-Model Quality Metrics
A Petri net is discovered from the **anonymized** log using the Inductive Miner. All four classic quality dimensions are then computed against the **original** log:

| Metric | Description | Range |
|---|---|---|
| **Fitness** | How well the reference log can be replayed on the discovered model (token-based or alignment-based). | [0, 1] — higher is better |
| **Precision** | How much of the model's allowed behaviour is actually observed in the reference log (penalises overly general models). | [0, 1] — higher is better |
| **Generality** | How well the model generalises beyond the training log (token-based generalisation score). | [0, 1] — higher is better |
| **Simplicity** | Structural simplicity of the Petri net (inverse arc degree). | [0, 1] — higher is better |


## Results Structure

Experimental results are organized as follows:

```
data/output/
├── {dataset_name}/
│   ├── results_ngram_ngram1.json         # 1-gram, non-explicit
│   ├── results_ngram_ngram1_explicit.json # 1-gram, explicit
│   ├── results_ngram_ngram2.json         # 2-gram, non-explicit
│   ├── results_ngram_ngram2_explicit.json # 2-gram, explicit
│   ├── results_ngram_ngram3.json         # 3-gram, non-explicit
│   └── results_ngram_ngram3_explicit.json # 3-gram, explicit
└── figures/
    └── {combined_dataset_comparison}/     # Multi-dataset comparison plots
```

Each JSON file contains detailed metrics for all tested z-values and algorithms.









