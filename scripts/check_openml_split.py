from pathlib import Path
import sys

from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import OPENML_SUITE_ID, SEED, SHOW_PROGRESS, TEST_OPENML_TASKS
from data.dataset_manager import OpenMLCC18DatasetManager


def main():
    try:
        manager = OpenMLCC18DatasetManager(seed=SEED)
        suite_task_ids = manager.get_suite_task_ids()
        train_task_ids = manager.get_train_task_ids()
        test_task_ids = manager.get_test_task_ids()
    except ImportError as exc:
        print(exc)
        print("pip:   pip install openml")
        print("conda: conda install -c conda-forge openml")
        raise SystemExit(1)

    train_set = set(train_task_ids)
    test_set = set(test_task_ids)
    expected_test_set = set(TEST_OPENML_TASKS)

    if train_set & test_set:
        raise RuntimeError(f"Train/test task overlap: {sorted(train_set & test_set)}")
    if test_set != expected_test_set:
        raise RuntimeError(
            "OpenML test tasks do not match TEST_OPENML_TASKS. "
            f"Expected {sorted(expected_test_set)}, got {sorted(test_set)}"
        )

    print(f"OpenML-CC18 suite_id: {OPENML_SUITE_ID}")
    print(f"Total task count: {len(suite_task_ids)}")
    print(f"Train task count: {len(train_task_ids)}")
    print(f"Test task count: {len(test_task_ids)}")
    print(f"Train/test overlap: {len(train_set & test_set)}")
    print()
    metadata_rows = []
    progress = tqdm(
        test_task_ids,
        desc="Checking held-out OpenML task metadata",
        unit="task",
        disable=not SHOW_PROGRESS,
        file=sys.stdout,
    )
    for task_id in progress:
        progress.set_postfix(task_id=task_id)
        meta = manager.get_task_metadata(task_id)
        metadata_rows.append(meta)

    print(
        "task_id,dataset_id,dataset_name,n_samples,n_features,n_classes,"
        "missing_values,numeric_features,symbolic_features"
    )

    for meta in metadata_rows:
        print(
            f"{meta['task_id']},{meta['dataset_id']},{meta['dataset_name']},"
            f"{meta['n_samples']},{meta['n_features']},{meta['n_classes']},"
            f"{meta['missing_values']},{meta['numeric_features']},"
            f"{meta['symbolic_features']}"
        )


if __name__ == "__main__":
    main()
