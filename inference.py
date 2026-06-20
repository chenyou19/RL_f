import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from agents.dqn_agent import DQNAgent
from config import (
    ACTION_DIM,
    BATCH_SIZE,
    GAMMA,
    LOG_DIR,
    LR,
    REPLAY_BUFFER_SIZE,
    SEED,
    STATE_DIM,
    MAX_STEPS,
)
from data.dataset_manager import DatasetManager
from env.tool_selection_env import ToolSelectionEnv
from tools.tool_executor import ToolExecutor
from utils.seed import set_seed


def reset_env_with_dataset(env, dataset):
    env.current_dataset = dataset
    env.dataset_profile = env.profiler.profile(dataset)

    env.pipeline_actions = []
    env.done = False

    env.has_scaler = False
    env.has_pca = False
    env.has_feature_selection = False
    env.has_model = False

    env.invalid_count = 0
    env.step_count = 0

    return env._get_state()


def run_agent_on_dataset(agent, env, dataset):
    state = reset_env_with_dataset(env, dataset)
    done = False
    final_info = None

    while not done:
        action_id = agent.select_action(state, epsilon=0.0)
        next_state, reward, done, info = env.step(action_id)

        state = next_state
        final_info = info

        if env.step_count >= MAX_STEPS:
            break

    return env.pipeline_actions, final_info


def evaluate_pipeline_on_test(dataset, actions):
    executor = ToolExecutor(seed=SEED)
    pipeline = executor.build_pipeline(actions)

    X_train_full = np.concatenate([dataset.X_train, dataset.X_val], axis=0)
    y_train_full = np.concatenate([dataset.y_train, dataset.y_val], axis=0)

    pipeline.fit(X_train_full, y_train_full)
    y_pred = pipeline.predict(dataset.X_test)

    return {
        "accuracy": accuracy_score(dataset.y_test, y_pred),
        "macro_f1": f1_score(dataset.y_test, y_pred, average="macro"),
    }


def main():
    set_seed(SEED)

    dataset_manager = DatasetManager(seed=SEED)
    datasets = dataset_manager.load_all()

    env = ToolSelectionEnv(datasets=datasets, seed=SEED)

    agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        lr=LR,
        gamma=GAMMA,
        buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
    )

    model_path = os.path.join(LOG_DIR, "dqn_agent.pth")
    agent.load(model_path)

    print(f"Loaded DQN model from: {model_path}")
    print()

    for dataset in datasets:
        actions, info = run_agent_on_dataset(agent, env, dataset)

        print("=" * 60)
        print(f"Dataset: {dataset.name}")
        print(f"Selected pipeline: {actions}")
        print(f"Invalid actions: {env.invalid_count}")

        has_model = any(a in ["random_forest", "svm", "knn"] for a in actions)

        if not has_model:
            print("No valid model selected. Cannot evaluate on test set.")
            continue

        test_result = evaluate_pipeline_on_test(dataset, actions)

        print(f"Test Accuracy: {test_result['accuracy']:.4f}")
        print(f"Test Macro F1: {test_result['macro_f1']:.4f}")


if __name__ == "__main__":
    main()