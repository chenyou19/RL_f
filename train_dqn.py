import os
import sys

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from agents.dqn_agent import DQNAgent
from config import (
    ACTION_DIM,
    BATCH_SIZE,
    DATASET_SPLIT_MODE,
    EPS_DECAY,
    EPS_END,
    EPS_START,
    EPISODES,
    FIGURE_DIR,
    GAMMA,
    LOG_DIR,
    LR,
    MIN_REPLAY_SIZE,
    MODEL_SAVE_FREQ,
    REPLAY_BUFFER_SIZE,
    RESUME_MODEL_PATH,
    RESUME_TRAINING,
    SEED,
    SHOW_PROGRESS,
    STATE_DIM,
    TARGET_UPDATE_FREQ,
    TEST_OPENML_TASKS,
    USE_OPENML_CC18,
)
from data.dataset_manager import DatasetManager, OpenMLCC18DatasetManager
from env.tool_selection_env import ToolSelectionEnv
from utils.metrics import save_results_csv
from utils.plot import moving_average, plot_curve
from utils.seed import set_seed


def load_existing_training_log(log_path, start_episode):
    if not os.path.exists(log_path):
        return [], [], [], [], [], []

    df = pd.read_csv(log_path)
    if "episode" in df.columns:
        df = df[df["episode"] < start_episode]

    results = df.to_dict("records")
    rewards = df.get("reward", pd.Series(dtype=float)).fillna(0.0).astype(float).tolist()
    losses = df.get("loss", pd.Series(dtype=float)).fillna(0.0).astype(float).tolist()
    f1s = df.get("f1", pd.Series(dtype=float)).fillna(0.0).astype(float).tolist()
    invalids = (
        df.get("invalid_count", pd.Series(dtype=float))
        .fillna(0.0)
        .astype(float)
        .tolist()
    )
    lengths = (
        df.get("pipeline_length", pd.Series(dtype=float))
        .fillna(0.0)
        .astype(float)
        .tolist()
    )

    return results, rewards, losses, f1s, invalids, lengths


def load_training_datasets(seed: int):
    if not USE_OPENML_CC18:
        dataset_manager = DatasetManager(seed=seed)
        return dataset_manager.load_all()

    if DATASET_SPLIT_MODE != "openml_cc18_holdout":
        raise ValueError(f"Unsupported DATASET_SPLIT_MODE: {DATASET_SPLIT_MODE}")

    dataset_manager = OpenMLCC18DatasetManager(seed=seed)
    train_task_ids = dataset_manager.get_train_task_ids()
    leaked_tasks = sorted(set(train_task_ids) & set(TEST_OPENML_TASKS))
    if leaked_tasks:
        raise RuntimeError(f"Held-out OpenML test tasks leaked into training: {leaked_tasks}")

    tqdm.write(
        "OpenML-CC18 dataset-level split: "
        f"{len(train_task_ids)} train tasks, "
        f"{len(dataset_manager.get_test_task_ids())} held-out test tasks."
    )

    datasets = []
    progress = tqdm(
        train_task_ids,
        desc="Loading OpenML-CC18 train datasets",
        unit="task",
        disable=not SHOW_PROGRESS,
        file=sys.stdout,
    )
    for task_id in progress:
        progress.set_postfix(task_id=task_id, dataset_name="")
        try:
            dataset = dataset_manager.load_openml_task(task_id, split_type="train")
        except Exception as exc:
            tqdm.write(f"Failed to load OpenML task {task_id}: {exc}")
            raise
        progress.set_postfix(task_id=task_id, dataset_name=dataset.name)
        datasets.append(dataset)

    return datasets


def train_dqn():
    set_seed(SEED)

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    datasets = load_training_datasets(SEED)

    env = ToolSelectionEnv(datasets=datasets, seed=SEED)

    agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        lr=LR,
        gamma=GAMMA,
        buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
    )

    epsilon = EPS_START
    start_episode = 0

    if RESUME_TRAINING:
        if not os.path.exists(RESUME_MODEL_PATH):
            raise FileNotFoundError(f"Resume checkpoint not found: {RESUME_MODEL_PATH}")
        checkpoint = agent.load(RESUME_MODEL_PATH)
        start_episode = int(checkpoint.get("completed_episodes", 0) or 0)
        epsilon = float(
            checkpoint.get(
                "epsilon",
                max(EPS_END, EPS_START * (EPS_DECAY ** start_episode)),
            )
        )
        tqdm.write(
            "Resumed training from "
            f"{RESUME_MODEL_PATH} at episode {start_episode}, epsilon={epsilon:.4f}."
        )

    episode_rewards = []
    episode_losses = []
    episode_f1s = []
    episode_invalids = []
    episode_lengths = []

    results = []
    log_path = os.path.join(LOG_DIR, "dqn_training_results.csv")
    if RESUME_TRAINING:
        (
            results,
            episode_rewards,
            episode_losses,
            episode_f1s,
            episode_invalids,
            episode_lengths,
        ) = load_existing_training_log(log_path, start_episode)
        if results:
            tqdm.write(f"Loaded {len(results)} previous training log rows.")

    if start_episode >= EPISODES:
        tqdm.write(
            f"Start episode {start_episode} is already >= EPISODES {EPISODES}; "
            "no additional episodes will run."
        )

    progress = tqdm(
        range(start_episode, EPISODES),
        desc="Training DQN",
        unit="episode",
        disable=not SHOW_PROGRESS,
        file=sys.stdout,
    )
    last_completed_episode = start_episode
    for ep in progress:
        state = env.reset()
        done = False

        total_reward = 0.0
        losses = []
        final_info = None

        while not done:
            action_mask = env.get_action_mask()
            action_id = agent.select_action(state, epsilon, action_mask=action_mask)
            next_state, reward, done, info = env.step(action_id)

            agent.replay_buffer.push(state, action_id, reward, next_state, done)

            state = next_state
            total_reward += reward
            final_info = info

            if len(agent.replay_buffer) >= MIN_REPLAY_SIZE:
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)

        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if ep % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        avg_loss = np.mean(losses) if len(losses) > 0 else None

        episode_rewards.append(total_reward)
        episode_losses.append(avg_loss if avg_loss is not None else 0)
        episode_f1s.append(final_info.get("f1") if final_info.get("f1") is not None else 0)
        episode_invalids.append(env.invalid_count)
        episode_lengths.append(len(env.pipeline_actions))

        results.append(
            {
                "episode": ep,
                "dataset": final_info.get("dataset"),
                "dataset_name": final_info.get("dataset_name"),
                "task_id": final_info.get("task_id"),
                "dataset_id": final_info.get("dataset_id"),
                "split_type": final_info.get("split_type"),
                "reward": total_reward,
                "loss": avg_loss,
                "epsilon": epsilon,
                "f1": final_info.get("f1"),
                "pipeline": final_info.get("pipeline"),
                "invalid_count": env.invalid_count,
                "pipeline_length": len(env.pipeline_actions),
                "cache_hit": final_info.get("cache_hit"),
            }
        )

        f1 = final_info.get("f1")
        progress.set_postfix(
            task_id=final_info.get("task_id"),
            dataset=final_info.get("dataset_name"),
            reward=f"{total_reward:.4f}",
            f1="None" if f1 is None else f"{f1:.4f}",
            invalid=env.invalid_count,
            length=len(env.pipeline_actions),
            epsilon=f"{epsilon:.4f}",
        )

        if MODEL_SAVE_FREQ > 0 and (ep + 1) % MODEL_SAVE_FREQ == 0:
            checkpoint_path = os.path.join(LOG_DIR, f"dqn_agent_ep_{ep + 1}.pth")
            agent.save(
                checkpoint_path,
                completed_episodes=ep + 1,
                epsilon=epsilon,
            )
            tqdm.write(f"Saved checkpoint to {checkpoint_path}")
        last_completed_episode = ep + 1

    save_results_csv(results, log_path)

    plot_curve(
        moving_average(episode_rewards),
        "DQN Reward Curve",
        "Reward",
        os.path.join(FIGURE_DIR, "dqn_reward_curve.png"),
    )

    plot_curve(
        moving_average(episode_f1s),
        "DQN F1 Curve",
        "F1 Score",
        os.path.join(FIGURE_DIR, "dqn_f1_curve.png"),
    )

    plot_curve(
        moving_average(episode_invalids),
        "DQN Invalid Action Curve",
        "Invalid Actions",
        os.path.join(FIGURE_DIR, "dqn_invalid_curve.png"),
    )

    plot_curve(
        moving_average(episode_lengths),
        "DQN Pipeline Length Curve",
        "Pipeline Length",
        os.path.join(FIGURE_DIR, "dqn_pipeline_length_curve.png"),
    )

    agent.save(
        os.path.join(LOG_DIR, "dqn_agent.pth"),
        completed_episodes=last_completed_episode,
        epsilon=epsilon,
    )

    tqdm.write("Training finished.")
    tqdm.write(f"Results saved to {LOG_DIR}")
    tqdm.write(f"Figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    train_dqn()
