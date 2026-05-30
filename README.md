# DQN Tool Selection Agent

A lightweight RL-based AutoML-style project that treats machine learning
pipeline construction as a sequential decision-making problem.

The MVP implements the full loop:

- Dataset loading
- Dataset profiling
- State construction
- Action-based pipeline building
- Reward calculation
- DQN training
- Baseline comparison
- Result logging and plotting

The agent learns to select preprocessing tools, feature tools, classifiers, and
an evaluation action based on dataset meta-features and the current pipeline
state.

## Environment Setup

This project is intended to run with Conda. GPU PyTorch is installed through
Conda channels, so `torch` is intentionally not listed in `requirements.txt`.

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate dqn-tool-agent
```

Verify that PyTorch can see the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Expected output should include:

```text
True
12.1
```

If `torch.cuda.is_available()` returns `False`, check that the NVIDIA driver is
installed correctly and that the installed CUDA runtime version is compatible
with your GPU driver.

## Project Structure

```text
.
|-- main.py
|-- train_dqn.py
|-- run_baselines.py
|-- config.py
|-- requirements.txt
|-- environment.yml
|
|-- data/
|   |-- dataset_manager.py
|   `-- data_profiler.py
|
|-- env/
|   `-- tool_selection_env.py
|
|-- tools/
|   |-- tool_executor.py
|   `-- pipeline_cache.py
|
|-- agents/
|   |-- q_network.py
|   |-- replay_buffer.py
|   `-- dqn_agent.py
|
|-- baselines/
|   |-- random_agent.py
|   |-- fixed_pipeline.py
|   `-- grid_search_baseline.py
|
|-- utils/
|   |-- seed.py
|   |-- metrics.py
|   `-- plot.py
|
`-- results/
    |-- logs/
    |-- figures/
    `-- tables/
```

## Run Order

Run the project from the repository root.

### 1. Test the Environment

```bash
python main.py
```

This checks that the environment can reset, sample actions, build a pipeline,
and return rewards.

### 2. Run Baselines

```bash
python run_baselines.py
```

This generates:

```text
results/tables/random_agent_results.csv
results/tables/fixed_pipeline_results.csv
results/tables/grid_search_results.csv
```

### 3. Train DQN

```bash
python train_dqn.py
```

This generates:

```text
results/logs/dqn_training_results.csv
results/logs/dqn_agent.pth
results/figures/dqn_reward_curve.png
results/figures/dqn_f1_curve.png
results/figures/dqn_invalid_curve.png
results/figures/dqn_pipeline_length_curve.png
```

## Datasets

The MVP uses built-in scikit-learn classification datasets:

- Iris
- Wine
- Breast cancer
- Digits

Each dataset is split into train, validation, and test sets. The current
environment evaluates pipelines on the validation split.

### OpenML-CC18 Dataset-Level Split

The default configuration uses the OpenML-CC18 benchmark suite:

```text
suite_id = 99
```

The split is done at the dataset/task level. The following 12 OpenML tasks are
held out for final testing only and are never used during DQN training:

```text
37, 53, 43, 9952, 9957, 146817, 3917, 3903, 28, 32, 9976, 9910
```

All remaining tasks from suite 99 are used as training datasets. Each dataset
still receives its own internal train/validation/test sample split; training
rewards use the validation split, and the held-out report uses the 12 reserved
datasets.

Run the split check:

```bash
python scripts/check_openml_split.py
```

Preload OpenML datasets into the local cache:

```bash
python scripts/load_openml_datasets.py --split train
python scripts/load_openml_datasets.py --split test
```

Train the DQN on training tasks only:

```bash
python train_dqn.py
```

Evaluate the trained model on the 12 held-out tasks:

```bash
python scripts/evaluate_heldout_openml.py
```

Use a specific DQN checkpoint for held-out evaluation:

```bash
python scripts/evaluate_heldout_openml.py --model-path results/logs/dqn_agent_ep_3000.pth
```

Held-out evaluation is written to:

```text
results/tables/openml_cc18_heldout_test_results.csv
```

The preload report is written to:

```text
results/tables/openml_cc18_dataset_load_report.csv
```

### Progress Bars

OpenML dataset loading, DQN training, split checking, and held-out evaluation
show progress bars by default. Training progress includes the current task,
dataset name, reward, validation F1, invalid action count, pipeline length, and
epsilon. Held-out evaluation progress includes task id, dataset name, F1,
accuracy, invalid action count, and pipeline length.

To disable progress bars, set this in `config.py`:

```python
SHOW_PROGRESS = False
```

## Action Space

The current action space is defined in `config.py`:

```text
standard_scaler
minmax_scaler
pca
feature_selection
random_forest
svm
knn
evaluate
```

Invalid actions are allowed but penalized, so the DQN learns action constraints
from rewards rather than using an action mask.

## Current Limitations

- OpenML-CC18 is now the default dataset source; scikit-learn built-in datasets
  remain available as a fallback in `DatasetManager`.
- Feature selection currently uses `k="all"` as a placeholder.
- Reward does not yet include a detailed computational cost term.
- DQN does not use action masking.
- Grid search uses a small candidate space for MVP comparison.

## Version Control Notes

Commit source code, configuration files, and `.gitkeep` files under `results/`.
Avoid committing local runtime caches, trained model files, and generated result
tables unless the experiment output is intentionally part of the report.
