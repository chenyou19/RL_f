import argparse
import os
from pathlib import Path
import sys

import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import OPENML_CACHE_DIR, SEED, SHOW_PROGRESS, TABLE_DIR
from data.dataset_manager import OpenMLCC18DatasetManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download/cache and validate-load OpenML-CC18 datasets."
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="all",
        help="Which OpenML-CC18 split to load.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        action="append",
        default=None,
        help="Load one task id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Load only the first N selected tasks for a quick smoke test.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep loading remaining tasks if one task fails.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(TABLE_DIR, "openml_cc18_dataset_load_report.csv"),
        help="CSV report path.",
    )
    return parser.parse_args()


def select_task_ids(manager, split, explicit_task_ids):
    if explicit_task_ids:
        test_task_ids = set(manager.get_test_task_ids())
        return [
            (task_id, "test" if task_id in test_task_ids else "train")
            for task_id in explicit_task_ids
        ]

    tasks = []
    if split in ["train", "all"]:
        tasks.extend((task_id, "train") for task_id in manager.get_train_task_ids())
    if split in ["test", "all"]:
        tasks.extend((task_id, "test") for task_id in manager.get_test_task_ids())
    return tasks


def bundle_to_report_row(bundle, status, error=""):
    meta = bundle.metadata
    return {
        "status": status,
        "error": error,
        "task_id": bundle.task_id,
        "dataset_id": bundle.dataset_id,
        "dataset_name": bundle.name,
        "split_type": bundle.split_type,
        "n_samples": meta.get("n_samples"),
        "n_features": meta.get("n_features"),
        "n_classes": meta.get("n_classes"),
        "missing_values": meta.get("missing_values"),
        "numeric_features": meta.get("numeric_features"),
        "symbolic_features": meta.get("symbolic_features"),
        "preprocessed_n_features": meta.get("preprocessed_n_features"),
        "split_source": meta.get("split_source"),
        "train_samples": int(bundle.X_train.shape[0]),
        "val_samples": int(bundle.X_val.shape[0]),
        "test_samples": int(bundle.X_test.shape[0]),
    }


def main():
    args = parse_args()
    manager = OpenMLCC18DatasetManager(seed=SEED)
    selected_tasks = select_task_ids(manager, args.split, args.task_id)

    if args.limit is not None:
        selected_tasks = selected_tasks[: args.limit]

    tqdm.write(
        "Loading OpenML datasets into cache: "
        f"{len(selected_tasks)} task(s), cache_dir={OPENML_CACHE_DIR}"
    )

    rows = []
    progress = tqdm(
        selected_tasks,
        desc="Loading OpenML datasets",
        unit="task",
        disable=not SHOW_PROGRESS,
        file=sys.stdout,
    )

    for task_id, split_type in progress:
        progress.set_postfix(task_id=task_id, split=split_type, dataset_name="")
        try:
            bundle = manager.load_openml_task(task_id, split_type=split_type)
        except Exception as exc:
            message = str(exc)
            tqdm.write(f"Failed to load task {task_id} ({split_type}): {message}")
            rows.append(
                {
                    "status": "failed",
                    "error": message,
                    "task_id": task_id,
                    "split_type": split_type,
                }
            )
            if not args.continue_on_error:
                raise
            continue

        progress.set_postfix(
            task_id=task_id,
            split=split_type,
            dataset_name=bundle.name,
        )
        rows.append(bundle_to_report_row(bundle, status="loaded"))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
    loaded_count = sum(row["status"] == "loaded" for row in rows)
    failed_count = sum(row["status"] == "failed" for row in rows)
    tqdm.write(
        f"Loaded {loaded_count} dataset(s), failed {failed_count}. "
        f"Report saved to {args.output}"
    )


if __name__ == "__main__":
    main()
