import random

import numpy as np

from config import (
    ACTIONS,
    INVALID_ACTION_PENALTY,
    INVALID_COUNT_PENALTY,
    MAX_STEPS,
    PIPELINE_LENGTH_PENALTY,
    STEP_PENALTY,
)
from data.data_profiler import DataProfiler
from tools.pipeline_cache import PipelineCache
from tools.tool_executor import ToolExecutor


class ToolSelectionEnv:
    def __init__(self, datasets, seed: int = 42):
        self.datasets = datasets
        self.seed = seed

        self.profiler = DataProfiler()
        self.executor = ToolExecutor(seed=seed)
        self.cache = PipelineCache()

        self.current_dataset = None
        self.dataset_profile = None

        self.pipeline_actions = []
        self.done = False

        self.has_scaler = False
        self.has_pca = False
        self.has_feature_selection = False
        self.has_model = False

        self.invalid_count = 0
        self.step_count = 0

    def reset(self):
        self.current_dataset = random.choice(self.datasets)
        self.dataset_profile = self.profiler.profile(self.current_dataset)

        self.pipeline_actions = []
        self.done = False

        self.has_scaler = False
        self.has_pca = False
        self.has_feature_selection = False
        self.has_model = False

        self.invalid_count = 0
        self.step_count = 0

        return self._get_state()

    def step(self, action_id: int):
        if self.done:
            raise RuntimeError("Episode is already done. Please call reset().")

        action = ACTIONS[action_id]
        self.step_count += 1

        info = {
            "dataset": self.current_dataset.name,
            "dataset_name": self.current_dataset.name,
            "task_id": getattr(self.current_dataset, "task_id", None),
            "dataset_id": getattr(self.current_dataset, "dataset_id", None),
            "split_type": getattr(self.current_dataset, "split_type", "train"),
            "action": action,
            "pipeline": list(self.pipeline_actions),
            "invalid": False,
            "f1": None,
            "cache_hit": False,
        }

        if self._is_invalid_action(action):
            self.invalid_count += 1
            reward = INVALID_ACTION_PENALTY
            done = self.step_count >= MAX_STEPS

            self.done = done
            next_state = self._get_state()

            info["invalid"] = True
            info["pipeline"] = list(self.pipeline_actions)

            return next_state, reward, done, info

        if action == "evaluate":
            reward, eval_info = self._evaluate_pipeline()
            self.done = True

            info.update(eval_info)
            info["pipeline"] = list(self.pipeline_actions)

            return self._get_state(), reward, True, info

        self._apply_action(action)

        reward = STEP_PENALTY

        if self.step_count >= MAX_STEPS:
            self.done = True
            reward = -0.5

        next_state = self._get_state()
        info["pipeline"] = list(self.pipeline_actions)

        return next_state, reward, self.done, info

    def _apply_action(self, action: str):
        self.pipeline_actions.append(action)

        if action in ["standard_scaler", "minmax_scaler"]:
            self.has_scaler = True
        elif action == "pca":
            self.has_pca = True
        elif action == "feature_selection":
            self.has_feature_selection = True
        elif action in ["random_forest", "svm", "knn"]:
            self.has_model = True

    def _is_invalid_action(self, action: str) -> bool:
        if action == "evaluate":
            return not self.has_model

        if action in ["standard_scaler", "minmax_scaler"]:
            return self.has_scaler or self.has_model

        if action == "pca":
            return self.has_pca or self.has_model

        if action == "feature_selection":
            return self.has_feature_selection or self.has_model

        if action in ["random_forest", "svm", "knn"]:
            return self.has_model

        return False

    def _evaluate_pipeline(self):
        dataset_name = self.current_dataset.name
        actions = list(self.pipeline_actions)

        if self.cache.has(dataset_name, actions):
            result = self.cache.get(dataset_name, actions)
            cache_hit = True
        else:
            result = self.executor.evaluate(self.current_dataset, actions)
            self.cache.set(dataset_name, actions, result)
            cache_hit = False

        f1 = result["f1"]

        reward = (
            f1
            - PIPELINE_LENGTH_PENALTY * len(self.pipeline_actions)
            - INVALID_COUNT_PENALTY * self.invalid_count
        )

        info = {
            "f1": f1,
            "eval_time": result["time"],
            "cache_hit": cache_hit,
        }

        return reward, info

    def _get_state(self):
        p = self.dataset_profile

        state = [
            p["n_samples_norm"],
            p["n_features_norm"],
            p["n_classes_norm"],
            p["missing_ratio"],
            p["imbalance_ratio"],
            p["numeric_ratio"],
            float(self.has_scaler),
            float(self.has_pca),
            float(self.has_feature_selection),
            float(self.has_model),
            min(len(self.pipeline_actions) / MAX_STEPS, 1.0),
            min(self.invalid_count / MAX_STEPS, 1.0),
        ]

        return np.array(state, dtype=np.float32)
