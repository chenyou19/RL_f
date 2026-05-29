import itertools

from tools.tool_executor import ToolExecutor


def generate_candidate_pipelines():
    scalers = [
        [],
        ["standard_scaler"],
        ["minmax_scaler"],
    ]

    feature_steps = [
        [],
        ["pca"],
        ["feature_selection"],
    ]

    models = [
        ["random_forest"],
        ["svm"],
        ["knn"],
    ]

    pipelines = []

    for scaler, feature, model in itertools.product(scalers, feature_steps, models):
        pipeline = scaler + feature + model
        pipelines.append(pipeline)

    return pipelines


def run_grid_search_baseline(datasets, seed=42):
    executor = ToolExecutor(seed=seed)
    candidate_pipelines = generate_candidate_pipelines()

    all_results = []

    for dataset in datasets:
        best_f1 = -1
        best_pipeline = None
        best_time = None

        for actions in candidate_pipelines:
            result = executor.evaluate(dataset, actions)

            if result["f1"] > best_f1:
                best_f1 = result["f1"]
                best_pipeline = actions
                best_time = result["time"]

        all_results.append(
            {
                "dataset": dataset.name,
                "dataset_name": dataset.name,
                "task_id": getattr(dataset, "task_id", None),
                "dataset_id": getattr(dataset, "dataset_id", None),
                "split_type": getattr(dataset, "split_type", "train"),
                "best_f1": best_f1,
                "best_pipeline": best_pipeline,
                "time": best_time,
                "num_candidates": len(candidate_pipelines),
            }
        )

    return all_results
