import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.dqn_agent import DQNAgent

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

from config import (
    ACTION_DIM,
    BATCH_SIZE,
    GAMMA,
    LOG_DIR,
    LR,
    REPLAY_BUFFER_SIZE,
    SEED,
    SHOW_PROGRESS,
    STATE_DIM,
    TABLE_DIR,
    TEST_OPENML_TASKS,
)
from data.dataset_manager import OpenMLCC18DatasetManager
from env.tool_selection_env import ToolSelectionEnv
from tools.tool_executor import ToolExecutor
from utils.seed import set_seed


def evaluate_agent_on_dataset(agent, dataset, seed):
    env = ToolSelectionEnv(datasets=[dataset], seed=seed)
    state = env.reset()
    done = False
    final_info = None

    while not done:
        action_id = agent.select_action(state, epsilon=0.0)
        state, _, done, info = env.step(action_id)
        final_info = info

    actions = list(env.pipeline_actions)
    validation_score = final_info.get("f1") if final_info else None
    test_f1 = None
    test_accuracy = None

    if actions:
        try:
            executor = ToolExecutor(seed=seed)
            pipeline = executor.build_pipeline(actions)
            X_fit = np.vstack([dataset.X_train, dataset.X_val])
            y_fit = np.concatenate([dataset.y_train, dataset.y_val])
            pipeline.fit(X_fit, y_fit)
            y_pred = pipeline.predict(dataset.X_test)
            test_f1 = float(f1_score(dataset.y_test, y_pred, average="macro"))
            test_accuracy = float(accuracy_score(dataset.y_test, y_pred))
        except Exception as exc:
            tqdm.write(f"Could not score task {dataset.task_id} on test split: {exc}")

    return {
        "task_id": dataset.task_id,
        "dataset_name": dataset.name,
        "selected_pipeline": actions,
        "actions": actions,
        "validation_score": validation_score,
        "test_score": test_f1,
        "f1": test_f1,
        "accuracy": test_accuracy,
        "invalid_action_count": env.invalid_count,
        "pipeline_length": len(actions),
    }


def main():
    set_seed(SEED)
    model_path = os.path.join(LOG_DIR, "dqn_agent.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"DQN model not found at {model_path}. Run python train_dqn.py first."
        )

    manager = OpenMLCC18DatasetManager(seed=SEED)
    test_task_ids = manager.get_test_task_ids()
    if set(test_task_ids) != set(TEST_OPENML_TASKS):
        raise RuntimeError("Held-out test tasks do not match TEST_OPENML_TASKS.")

    agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        lr=LR,
        gamma=GAMMA,
        buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
    )
    agent.load(model_path)

    rows = []
    progress = tqdm(
        test_task_ids,
        desc="Evaluating held-out OpenML tasks",
        unit="dataset",
        disable=not SHOW_PROGRESS,
        file=sys.stdout,
    )
    for task_id in progress:
        progress.set_postfix(task_id=task_id, dataset_name="")
        dataset = manager.load_openml_task(task_id, split_type="test")
        progress.set_postfix(task_id=task_id, dataset_name=dataset.name)
        result = evaluate_agent_on_dataset(agent, dataset, SEED)
        rows.append(result)
        f1 = result["f1"]
        accuracy = result["accuracy"]
        progress.set_postfix(
            task_id=task_id,
            dataset_name=dataset.name,
            f1="None" if f1 is None else f"{f1:.4f}",
            accuracy="None" if accuracy is None else f"{accuracy:.4f}",
            invalid=result["invalid_action_count"],
            length=result["pipeline_length"],
        )

    os.makedirs(TABLE_DIR, exist_ok=True)
    output_path = os.path.join(TABLE_DIR, "openml_cc18_heldout_test_results.csv")
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    tqdm.write(f"Saved held-out OpenML results to {output_path}")


if __name__ == "__main__":
    main()
