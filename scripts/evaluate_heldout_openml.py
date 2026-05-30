import argparse
import os
from pathlib import Path
import sys
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    ACTION_DIM,
    ACTIONS,
    BATCH_SIZE,
    DEBUG_ACTION_TRACE,
    GAMMA,
    LOG_DIR,
    LR,
    MAX_STEPS,
    REPLAY_BUFFER_SIZE,
    SEED,
    SHOW_PROGRESS,
    STATE_DIM,
    TABLE_DIR,
    TEST_OPENML_TASKS,
)


DEFAULT_MODEL_PATH = os.path.join(LOG_DIR, "dqn_agent.pth")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN model on held-out OpenML-CC18 tasks."
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained DQN checkpoint.",
    )
    return parser.parse_args()


def load_runtime_dependencies():
    global DQNAgent
    global OpenMLCC18DatasetManager
    global ToolExecutor
    global ToolSelectionEnv
    global accuracy_score
    global f1_score
    global np
    global pd
    global set_seed
    global tqdm

    from agents.dqn_agent import DQNAgent
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score
    from tqdm.auto import tqdm

    from data.dataset_manager import OpenMLCC18DatasetManager
    from env.tool_selection_env import ToolSelectionEnv
    from tools.tool_executor import ToolExecutor
    from utils.seed import set_seed


def format_state_summary(state):
    return {
        "has_scaler": float(state[6]),
        "has_pca": float(state[7]),
        "has_feature_selection": float(state[8]),
        "has_model": float(state[9]),
        "pipeline_ratio": float(state[10]),
        "invalid_ratio": float(state[11]),
    }


def debug_action_trace(dataset_name, step, state, action_id, env, reward=None, done=None):
    if not DEBUG_ACTION_TRACE:
        return

    tqdm.write(
        " | ".join(
            [
                f"dataset={dataset_name}",
                f"step={step}",
                f"state={format_state_summary(state)}",
                f"selected_action={ACTIONS[action_id]}",
                f"current_pipeline={list(env.pipeline_actions)}",
                f"reward={reward}",
                f"done={done}",
                f"valid_actions={[ACTIONS[i] for i in env.get_valid_actions()]}",
                f"action_mask={env.get_action_mask().tolist()}",
            ]
        )
    )


def evaluate_agent_on_dataset(agent, dataset, seed):
    env = ToolSelectionEnv(datasets=[dataset], seed=seed)
    state = env.reset(dataset)
    done = False
    final_info = None
    step = 0

    while not done and step < MAX_STEPS:
        action_mask = env.get_action_mask()
        action_id = agent.select_action(state, epsilon=0.0, action_mask=action_mask)
        debug_action_trace(dataset.name, step, state, action_id, env)
        next_state, reward, done, info = env.step(action_id)
        debug_action_trace(dataset.name, step, next_state, action_id, env, reward, done)
        state = next_state
        final_info = info
        step += 1

    selected_pipeline = list(env.pipeline_actions)
    actions = list(env.action_history)
    validation_score = final_info.get("f1") if final_info else None
    test_f1 = None
    test_accuracy = None

    if selected_pipeline:
        try:
            executor = ToolExecutor(seed=seed)
            pipeline = executor.build_pipeline(selected_pipeline)
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
        "selected_pipeline": selected_pipeline,
        "actions": actions,
        "validation_score": validation_score,
        "test_score": test_f1,
        "f1": test_f1,
        "accuracy": test_accuracy,
        "invalid_action_count": env.invalid_count,
        "pipeline_length": len(selected_pipeline),
    }


def main():
    args = parse_args()
    model_path = args.model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"DQN model not found at {model_path}. Run python train_dqn.py first."
        )

    load_runtime_dependencies()
    set_seed(SEED)

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
    try:
        pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            TABLE_DIR,
            f"openml_cc18_heldout_test_results_{timestamp}.csv",
        )
        pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
        tqdm.write(
            "Could not overwrite openml_cc18_heldout_test_results.csv; "
            f"saved to {output_path} instead."
        )
    tqdm.write(f"Saved held-out OpenML results to {output_path}")


if __name__ == "__main__":
    main()
