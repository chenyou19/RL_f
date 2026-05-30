import json
import os
from typing import Any

from config import (
    ENABLE_PIPELINE_CACHE,
    PIPELINE_CACHE_PATH,
    PIPELINE_CACHE_VERSION,
)


class PipelineCache:
    def __init__(
        self,
        cache_path: str = PIPELINE_CACHE_PATH,
        enabled: bool = ENABLE_PIPELINE_CACHE,
        version: str = PIPELINE_CACHE_VERSION,
        seed: int = 42,
    ):
        self.cache_path = cache_path
        self.enabled = enabled
        self.version = version
        self.seed = seed
        self.cache = self._load_cache()

    def make_key(self, dataset, actions):
        identity = {
            "version": self.version,
            "seed": self.seed,
            "dataset_name": getattr(dataset, "name", str(dataset)),
            "task_id": getattr(dataset, "task_id", None),
            "dataset_id": getattr(dataset, "dataset_id", None),
            "split_type": getattr(dataset, "split_type", None),
            "actions": list(actions),
        }
        return json.dumps(identity, sort_keys=True, separators=(",", ":"))

    def has(self, dataset, actions):
        if not self.enabled:
            return False
        key = self.make_key(dataset, actions)
        return key in self.cache

    def get(self, dataset, actions):
        key = self.make_key(dataset, actions)
        return dict(self.cache[key])

    def set(self, dataset, actions, result):
        if not self.enabled:
            return

        key = self.make_key(dataset, actions)
        self.cache[key] = self._serializable_result(result)
        self._save_cache()

    def size(self):
        return len(self.cache)

    def _load_cache(self):
        if not self.enabled or not os.path.exists(self.cache_path):
            return {}

        try:
            with open(self.cache_path, "r", encoding="utf-8") as file:
                cache = json.load(file)
        except (json.JSONDecodeError, OSError):
            return {}

        if isinstance(cache, dict):
            return cache
        return {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        tmp_path = self.cache_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as file:
            json.dump(self.cache, file, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_path, self.cache_path)

    @staticmethod
    def _serializable_result(result: dict[str, Any]):
        return {
            "f1": float(result["f1"]),
            "time": float(result.get("time", 0.0)),
            "status": result.get("status", "success"),
            "error": result.get("error"),
        }
