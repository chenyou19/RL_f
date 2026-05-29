import os

import pandas as pd

from baselines.fixed_pipeline import run_fixed_baselines
from baselines.grid_search_baseline import run_grid_search_baseline
from baselines.random_agent import run_random_agent
from config import (
    DATASET_SPLIT_MODE,
    SEED,
    TABLE_DIR,
    TEST_OPENML_TASKS,
    USE_OPENML_CC18,
)
from data.dataset_manager import DatasetManager, OpenMLCC18DatasetManager
from env.tool_selection_env import ToolSelectionEnv
from utils.seed import set_seed


def load_baseline_training_datasets(seed: int):
    if not USE_OPENML_CC18:
        dataset_manager = DatasetManager(seed=seed)
        return dataset_manager.load_all()

    if DATASET_SPLIT_MODE != "openml_cc18_holdout":
        raise ValueError(f"Unsupported DATASET_SPLIT_MODE: {DATASET_SPLIT_MODE}")

    dataset_manager = OpenMLCC18DatasetManager(seed=seed)
    train_task_ids = dataset_manager.get_train_task_ids()
    leaked_tasks = sorted(set(train_task_ids) & set(TEST_OPENML_TASKS))
    if leaked_tasks:
        raise RuntimeError(f"Held-out OpenML test tasks leaked into baselines: {leaked_tasks}")

    print(
        "OpenML-CC18 baseline split: "
        f"{len(train_task_ids)} train tasks, "
        f"{len(dataset_manager.get_test_task_ids())} held-out test tasks."
    )
    return [
        dataset_manager.load_openml_task(task_id, split_type="train")
        for task_id in train_task_ids
    ]


def run_all_baselines():
    set_seed(SEED)

    os.makedirs(TABLE_DIR, exist_ok=True)

    datasets = load_baseline_training_datasets(SEED)

    env = ToolSelectionEnv(datasets=datasets, seed=SEED)

    random_results = run_random_agent(env, episodes=100)
    pd.DataFrame(random_results).to_csv(
        os.path.join(TABLE_DIR, "random_agent_results.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    fixed_results = run_fixed_baselines(datasets, seed=SEED)

    fixed_rows = []
    for method_name, rows in fixed_results.items():
        for r in rows:
            r["method"] = method_name
            fixed_rows.append(r)

    pd.DataFrame(fixed_rows).to_csv(
        os.path.join(TABLE_DIR, "fixed_pipeline_results.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    grid_results = run_grid_search_baseline(datasets, seed=SEED)
    pd.DataFrame(grid_results).to_csv(
        os.path.join(TABLE_DIR, "grid_search_results.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("Baseline experiments finished.")
    print(f"Tables saved to {TABLE_DIR}")


if __name__ == "__main__":
    run_all_baselines()
