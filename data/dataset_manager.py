import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from config import OPENML_CACHE_DIR, OPENML_SUITE_ID, TEST_OPENML_TASKS


@dataclass
class DatasetBundle:
    name: str
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    task_id: Optional[int] = None
    dataset_id: Optional[int] = None
    split_type: str = "train"
    metadata: Dict = field(default_factory=dict)


class DatasetManager:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def load_all(self) -> List[DatasetBundle]:
        datasets = []

        dataset_loaders = [
            ("iris", load_iris),
            ("wine", load_wine),
            ("breast_cancer", load_breast_cancer),
            ("digits", load_digits),
        ]

        for name, loader in dataset_loaders:
            data = loader()
            X = data.data.astype(np.float32)
            y = data.target

            bundle = self._split_dataset(name, X, y)
            datasets.append(bundle)

        return datasets

    def _split_dataset(self, name: str, X, y) -> DatasetBundle:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.seed,
            stratify=y,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=0.25,
            random_state=self.seed,
            stratify=y_train_val,
        )

        return DatasetBundle(
            name=name,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )


class OpenMLCC18DatasetManager:
    def __init__(
        self,
        seed: int = 42,
        suite_id: int = OPENML_SUITE_ID,
        test_task_ids: Sequence[int] = TEST_OPENML_TASKS,
        cache_dir: str = OPENML_CACHE_DIR,
    ):
        self.seed = seed
        self.suite_id = suite_id
        self.test_task_ids = list(test_task_ids)
        self.cache_dir = cache_dir
        self._suite_task_ids: Optional[List[int]] = None
        self._train_task_ids: Optional[List[int]] = None
        self._test_task_ids: Optional[List[int]] = None

    def get_train_task_ids(self) -> List[int]:
        self._ensure_split()
        return list(self._train_task_ids)

    def get_suite_task_ids(self) -> List[int]:
        self._ensure_split()
        return list(self._suite_task_ids)

    def get_test_task_ids(self) -> List[int]:
        self._ensure_split()
        return list(self._test_task_ids)

    def get_random_train_dataset(self) -> DatasetBundle:
        task_id = random.choice(self.get_train_task_ids())
        return self.load_openml_task(task_id, split_type="train")

    def get_train_datasets(self) -> List[DatasetBundle]:
        return [
            self.load_openml_task(task_id, split_type="train")
            for task_id in self.get_train_task_ids()
        ]

    def get_test_datasets(self) -> List[DatasetBundle]:
        return [
            self.load_openml_task(task_id, split_type="test")
            for task_id in self.get_test_task_ids()
        ]

    def load_openml_task(self, task_id: int, split_type: str = "train") -> DatasetBundle:
        openml = self._import_openml()
        self._configure_openml(openml)

        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        X, y = task.get_X_and_y(dataset_format="dataframe")
        X = self._ensure_dataframe(X)
        y = pd.Series(y, name=getattr(task, "target_name", None))

        valid_target_mask = ~pd.isna(y)
        X = X.loc[valid_target_mask].reset_index(drop=True)
        y = y.loc[valid_target_mask].reset_index(drop=True)

        y_encoded = LabelEncoder().fit_transform(y.astype(str))
        train_idx, val_idx, test_idx, split_source = self._build_indices(
            task,
            X,
            y_encoded,
        )

        X_train_raw = X.iloc[train_idx].reset_index(drop=True)
        X_val_raw = X.iloc[val_idx].reset_index(drop=True)
        X_test_raw = X.iloc[test_idx].reset_index(drop=True)

        preprocessor = self._build_preprocessor(X_train_raw)
        X_train = preprocessor.fit_transform(X_train_raw)
        X_val = preprocessor.transform(X_val_raw)
        X_test = preprocessor.transform(X_test_raw)

        y_train = y_encoded[train_idx]
        y_val = y_encoded[val_idx]
        y_test = y_encoded[test_idx]

        metadata = self.get_task_metadata(task_id, task=task, X=X, y=y_encoded)
        metadata["split_source"] = split_source
        metadata["preprocessed_n_features"] = int(X_train.shape[1])

        return DatasetBundle(
            name=metadata["dataset_name"],
            X_train=self._to_float32_array(X_train),
            X_val=self._to_float32_array(X_val),
            X_test=self._to_float32_array(X_test),
            y_train=np.asarray(y_train, dtype=np.int64),
            y_val=np.asarray(y_val, dtype=np.int64),
            y_test=np.asarray(y_test, dtype=np.int64),
            task_id=int(task_id),
            dataset_id=metadata["dataset_id"],
            split_type=split_type,
            metadata=metadata,
        )

    def get_task_metadata(self, task_id: int, task=None, X=None, y=None) -> Dict:
        openml = self._import_openml()
        self._configure_openml(openml)

        if task is None:
            task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        if X is None or y is None:
            X, raw_y = task.get_X_and_y(dataset_format="dataframe")
            X = self._ensure_dataframe(X)
            y = LabelEncoder().fit_transform(pd.Series(raw_y).astype(str))

        numeric_columns = [
            col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])
        ]
        symbolic_columns = [col for col in X.columns if col not in numeric_columns]
        missing_values = int(X.isna().sum().sum())

        return {
            "task_id": int(task_id),
            "dataset_id": int(dataset.dataset_id),
            "dataset_name": dataset.name,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(len(np.unique(y))),
            "missing_values": missing_values,
            "numeric_features": int(len(numeric_columns)),
            "symbolic_features": int(len(symbolic_columns)),
        }

    def _ensure_split(self) -> None:
        if self._suite_task_ids is not None:
            return

        openml = self._import_openml()
        self._configure_openml(openml)

        suite = openml.study.get_suite(self.suite_id)
        suite_task_ids = sorted(int(task_id) for task_id in suite.tasks)
        test_task_ids = sorted(int(task_id) for task_id in self.test_task_ids)

        missing_test_tasks = sorted(set(test_task_ids) - set(suite_task_ids))
        if missing_test_tasks:
            raise ValueError(
                "Configured TEST_OPENML_TASKS are not in OpenML suite "
                f"{self.suite_id}: {missing_test_tasks}"
            )

        self._suite_task_ids = suite_task_ids
        self._test_task_ids = test_task_ids
        self._train_task_ids = [
            task_id for task_id in suite_task_ids if task_id not in set(test_task_ids)
        ]

    def _build_indices(self, task, X: pd.DataFrame, y: np.ndarray):
        try:
            train_idx, test_idx = task.get_train_test_split_indices(
                repeat=0,
                fold=0,
                sample=0,
            )
            train_idx = np.asarray(train_idx, dtype=int)
            test_idx = np.asarray(test_idx, dtype=int)
            train_idx = train_idx[train_idx < len(X)]
            test_idx = test_idx[test_idx < len(X)]
            train_idx, val_idx = self._split_indices(
                train_idx,
                y[train_idx],
                test_size=0.25,
            )
            return train_idx, val_idx, test_idx, "openml_task_fold_0"
        except Exception:
            all_idx = np.arange(len(X))
            train_val_idx, test_idx = self._split_indices(all_idx, y, test_size=0.2)
            train_idx, val_idx = self._split_indices(
                train_val_idx,
                y[train_val_idx],
                test_size=0.25,
            )
            return train_idx, val_idx, test_idx, "stratified_random"

    def _split_indices(self, indices: np.ndarray, y_subset: np.ndarray, test_size: float):
        stratify = y_subset if self._can_stratify(y_subset) else None
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=self.seed,
            stratify=stratify,
        )
        return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)

    def _can_stratify(self, y: np.ndarray) -> bool:
        _, counts = np.unique(y, return_counts=True)
        return len(counts) > 1 and counts.min() >= 2

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        numeric_columns = [
            col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])
        ]
        categorical_columns = [col for col in X.columns if col not in numeric_columns]

        transformers = []
        if numeric_columns:
            transformers.append(
                (
                    "numeric",
                    Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                    numeric_columns,
                )
            )
        if categorical_columns:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", self._make_one_hot_encoder()),
                        ]
                    ),
                    categorical_columns,
                )
            )

        return ColumnTransformer(transformers, sparse_threshold=0.0)

    def _make_one_hot_encoder(self):
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    def _ensure_dataframe(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

    def _to_float32_array(self, X) -> np.ndarray:
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32)

    def _configure_openml(self, openml) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        openml.config.cache_directory = self.cache_dir

    def _import_openml(self):
        try:
            import openml
        except ImportError as exc:
            raise ImportError(
                "The openml package is required for OpenML-CC18. Install it with "
                "'pip install openml' or 'conda install -c conda-forge openml'."
            ) from exc
        return openml
