import time
from typing import Any, Dict, List

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC


class ToolExecutor:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def build_pipeline(self, actions: List[str]) -> Pipeline:
        steps = []

        for action in actions:
            if action == "standard_scaler":
                steps.append(("scaler", StandardScaler()))
            elif action == "minmax_scaler":
                steps.append(("scaler", MinMaxScaler()))
            elif action == "pca":
                steps.append(("pca", PCA(n_components=0.95, random_state=self.seed)))
            elif action == "feature_selection":
                steps.append(
                    (
                        "select",
                        Pipeline(
                            [
                                ("variance_threshold", VarianceThreshold(threshold=0.0)),
                                ("select_k_best", SelectKBest(score_func=f_classif, k="all")),
                            ]
                        ),
                    )
                )
            elif action == "random_forest":
                steps.append(
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=100,
                            random_state=self.seed,
                            n_jobs=-1,
                        ),
                    )
                )
            elif action == "svm":
                steps.append(
                    (
                        "model",
                        SVC(
                            kernel="rbf",
                            C=1.0,
                            gamma="scale",
                            random_state=self.seed,
                        ),
                    )
                )
            elif action == "knn":
                steps.append(("model", KNeighborsClassifier(n_neighbors=5)))

        return Pipeline(steps)

    def evaluate(self, dataset, actions: List[str]) -> Dict[str, Any]:
        pipeline = self.build_pipeline(actions)

        start_time = time.time()

        try:
            pipeline.fit(dataset.X_train, dataset.y_train)
            y_pred = pipeline.predict(dataset.X_val)
            f1 = f1_score(dataset.y_val, y_pred, average="macro")
            status = "success"
            error = None
        except ValueError as exc:
            f1 = 0.0
            status = "failed"
            error = str(exc)

        elapsed_time = time.time() - start_time

        return {
            "f1": float(f1),
            "time": float(elapsed_time),
            "pipeline": pipeline,
            "status": status,
            "error": error,
        }
