"""
Tool Executor module.

This module will eventually apply selected preprocessing and model tools to a
pipeline state.
"""

from __future__ import annotations

from typing import Any

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from src.constants import ACTIONS, MODEL_ACTIONS, PREPROCESS_ACTIONS, TERMINAL_ACTION


class ToolExecutor:
    """Create sklearn objects for supported tool-selection actions."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self._preprocess_names = {ACTIONS[index] for index in PREPROCESS_ACTIONS}
        self._model_names = {ACTIONS[index] for index in MODEL_ACTIONS}
        self._terminal_name = ACTIONS[TERMINAL_ACTION]
        self._supported_names = set(ACTIONS.values())

    def create_tool(self, action_name: str, n_features: int | None = None) -> Any | None:
        """
        Create the sklearn transformer or estimator for an action.

        ``do_nothing`` and ``evaluate`` are control actions, so they return
        ``None`` and should not be inserted into a sklearn Pipeline.
        """
        action_name = action_name.lower()
        if action_name not in self._supported_names:
            raise ValueError(f"Unsupported action '{action_name}'.")

        if action_name in {"do_nothing", "evaluate"}:
            return None
        if action_name == "standard_scaler":
            return StandardScaler()
        if action_name == "minmax_scaler":
            return MinMaxScaler()
        if action_name == "pca":
            return PCA(n_components=0.95)
        if action_name == "feature_selection":
            k = self._feature_selection_k(n_features)
            return Pipeline(
                [
                    ("variance_threshold", VarianceThreshold(threshold=0.0)),
                    ("select_k_best", SelectKBest(score_func=f_classif, k=k)),
                ]
            )
        if action_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
            )
        if action_name == "svm":
            return SVC(kernel="rbf", gamma="scale", random_state=self.random_state)
        if action_name == "knn":
            return KNeighborsClassifier(n_neighbors=5)

        raise ValueError(f"No tool mapping is defined for action '{action_name}'.")

    def is_preprocess_action(self, action_name: str) -> bool:
        """Return whether an action maps to a preprocessing transformer."""
        return action_name.lower() in self._preprocess_names

    def is_model_action(self, action_name: str) -> bool:
        """Return whether an action maps to a model estimator."""
        return action_name.lower() in self._model_names

    def is_terminal_action(self, action_name: str) -> bool:
        """Return whether an action is the terminal evaluation action."""
        return action_name.lower() == self._terminal_name

    @staticmethod
    def _feature_selection_k(n_features: int | None) -> int | str:
        """Choose a safe SelectKBest feature count."""
        if n_features is None:
            return "all"
        return max(1, min(10, int(n_features)))
