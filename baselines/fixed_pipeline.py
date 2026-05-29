from tools.tool_executor import ToolExecutor


def evaluate_fixed_pipeline(datasets, actions, seed=42):
    executor = ToolExecutor(seed=seed)
    results = []

    for dataset in datasets:
        result = executor.evaluate(dataset, actions)

        results.append(
            {
                "dataset": dataset.name,
                "dataset_name": dataset.name,
                "task_id": getattr(dataset, "task_id", None),
                "dataset_id": getattr(dataset, "dataset_id", None),
                "split_type": getattr(dataset, "split_type", "train"),
                "pipeline": actions,
                "f1": result["f1"],
                "time": result["time"],
                "pipeline_length": len(actions),
            }
        )

    return results


def run_fixed_baselines(datasets, seed=42):
    fixed_1 = ["standard_scaler", "svm"]
    fixed_2 = ["feature_selection", "random_forest"]
    fixed_3 = ["standard_scaler", "knn"]

    results = {}
    results["StandardScaler_SVM"] = evaluate_fixed_pipeline(datasets, fixed_1, seed)
    results["FeatureSelection_RF"] = evaluate_fixed_pipeline(datasets, fixed_2, seed)
    results["StandardScaler_KNN"] = evaluate_fixed_pipeline(datasets, fixed_3, seed)

    return results
