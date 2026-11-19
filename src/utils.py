"""Shared utilities: seeding, metrics, cost measurement. Used by train.py / evaluate.py."""
import os
import random
import time
import json
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
)


def set_seed(seed):
    """Full determinism control (Core Req. #6: run each config with >=3 seeds)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model):
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 ** 2)


class Timer:
    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._t0


def compute_metrics(y_true, y_pred, label_names=None):
    """Task-appropriate metrics (Core Req. #9): not accuracy-only, macro + per-class."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    per_class = {}
    names = label_names or [str(i) for i in range(len(per_class_f1))]
    for i, name in enumerate(names):
        per_class[name] = {
            "f1": float(per_class_f1[i]),
            "precision": float(per_class_prec[i]),
            "recall": float(per_class_rec[i]),
        }

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def aggregate_seed_runs(metric_dicts):
    """Mean +/- SD across seeds for each scalar metric (Core Req. #6 & #9)."""
    keys = ["accuracy", "macro_f1", "weighted_f1"]
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in metric_dicts])
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return out


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
