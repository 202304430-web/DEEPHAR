"""
Shared evaluation utilities for all models (PDF Section 4.1).
Four modes:
  --mode aggregate      : mean +/- SD across the 3 seeds for one run_name (Core Req. #6, #9)
  --mode compare         : builds the cross-model comparison table (Deliverable #5)
  --mode error_analysis  : reloads one checkpoint, dumps misclassified test examples (Core Req. #10)
  --mode train_test_gap  : train-set vs. test-set accuracy for one checkpoint -- the direct,
                           concrete overfitting check (train >> test accuracy is the smoking
                           gun; close values mean the model generalizes to held-out data)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from datasets.uci_har import get_loaders, UCIHARRawDataset, UCIHARFeatureDataset, LABEL_NAMES  # noqa: E402
from models import MODEL_REGISTRY  # noqa: E402
from utils import aggregate_seed_runs, compute_metrics, save_json  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(ROOT, "experiments", "results")


def aggregate(run_name):
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{run_name}_seed*.json")))
    if not files:
        raise FileNotFoundError(f"No results found for run_name={run_name}")
    runs = [json.load(open(f)) for f in files]
    agg = aggregate_seed_runs([r["test_metrics"] for r in runs])
    out = {
        "run_name": run_name,
        "n_seeds": len(runs),
        "seeds": [r["seed"] for r in runs],
        "aggregated_test_metrics": agg,
        "num_params": runs[0]["num_params"],
        "model_size_mb": runs[0]["model_size_mb"],
        "train_time_sec_mean": float(np.mean([r["train_time_sec"] for r in runs])),
        "inference_time_ms_per_sample_mean": float(np.mean([r["inference_time_ms_per_sample"] for r in runs])),
    }
    save_json(out, os.path.join(RESULTS_DIR, f"{run_name}_aggregated.json"))
    print(json.dumps(out, indent=2))
    return out


def compare(run_names):
    """Fair comparison table (Deliverable #5): same protocol, mean+/-SD per model."""
    rows = []
    for name in run_names:
        try:
            agg = aggregate(name)
        except FileNotFoundError as e:
            print(f"[skip] {e}")
            continue
        m = agg["aggregated_test_metrics"]
        rows.append({
            "run_name": name,
            "accuracy_mean": m["accuracy"]["mean"], "accuracy_std": m["accuracy"]["std"],
            "macro_f1_mean": m["macro_f1"]["mean"], "macro_f1_std": m["macro_f1"]["std"],
            "params": agg["num_params"], "size_mb": agg["model_size_mb"],
            "train_sec": agg["train_time_sec_mean"],
            "infer_ms_per_sample": agg["inference_time_ms_per_sample_mean"],
        })

    csv_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    if rows:
        header = list(rows[0].keys())
        with open(csv_path, "w") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join(str(r[h]) for h in header) + "\n")
        print(f"Comparison table -> {csv_path}")
    return rows


def error_analysis(checkpoint_path, n_examples=20):
    """Dumps misclassified test examples with true/pred labels (Core Req. #10)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["cfg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kind = "feat" if cfg["model"] == "baseline_mlp" else "raw"
    _, _, test_loader = get_loaders(kind=kind, modality=cfg.get("modality", "all"), batch_size=64)

    model_kwargs = cfg.get("model_params", {})
    if cfg["model"] != "baseline_mlp":
        model_kwargs.setdefault("in_channels", 6 if cfg.get("modality") == "accel_only" else 9)
    else:
        model_kwargs.setdefault("in_dim", 561)
    model_kwargs.setdefault("num_classes", len(LABEL_NAMES))
    model = MODEL_REGISTRY[cfg["model"]](**model_kwargs)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    criterion = nn.CrossEntropyLoss()
    all_true, all_pred, all_idx = [], [], []
    idx_counter = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            all_true.extend(y.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())
            all_idx.extend(range(idx_counter, idx_counter + len(y)))
            idx_counter += len(y)

    metrics = compute_metrics(all_true, all_pred, LABEL_NAMES)
    mis = [(i, t, p) for i, t, p in zip(all_idx, all_true, all_pred) if t != p]
    print(f"Total test={len(all_true)}  Misclassified={len(mis)}  "
          f"Accuracy={metrics['accuracy']:.4f}  MacroF1={metrics['macro_f1']:.4f}")

    examples = []
    for i, t, p in mis[:n_examples]:
        examples.append({
            "test_index": i,
            "true_label": LABEL_NAMES[t],
            "pred_label": LABEL_NAMES[p],
        })
        print(f"  idx={i:5d}  true={LABEL_NAMES[t]:<20s} pred={LABEL_NAMES[p]}")

    out = {
        "checkpoint": checkpoint_path,
        "metrics": metrics,
        "n_misclassified": len(mis),
        "n_total": len(all_true),
        "examples": examples,
    }
    run_name = cfg.get("run_name", "model")
    save_json(out, os.path.join(RESULTS_DIR, f"{run_name}_error_analysis.json"))
    return out


def _eval_loader(model, loader, device):
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            all_true.extend(y.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())
    return compute_metrics(all_true, all_pred, LABEL_NAMES)


def train_test_gap(checkpoint_path):
    """
    The direct overfitting check: evaluate the SAME trained checkpoint on its own
    (unaugmented, unshuffled) training split and on the held-out test split, and
    report the gap. A large gap (train >> test) is the actual signature of
    overfitting; a small gap means the model generalizes to data it never saw
    in training. Note this is a DIFFERENT question from data leakage (which the
    subject-disjoint-split assertion in data/preprocess.py already rules out at
    the data-construction level, before any model is ever trained).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["cfg"]

    model_kwargs = dict(cfg.get("model_params", {}))
    if cfg["model"] != "baseline_mlp":
        model_kwargs.setdefault("in_channels", 6 if cfg.get("modality") == "accel_only" else 9)
    else:
        model_kwargs.setdefault("in_dim", 561)
    model_kwargs.setdefault("num_classes", len(LABEL_NAMES))
    model = MODEL_REGISTRY[cfg["model"]](**model_kwargs)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    if cfg["model"] == "baseline_mlp":
        train_ds = UCIHARFeatureDataset("train")
        test_ds = UCIHARFeatureDataset("test")
    else:
        modality = cfg.get("modality", "all")
        # augment=False: we want the model's fit on the REAL unaugmented training
        # examples, not the randomly-jittered versions it trained on each epoch.
        train_ds = UCIHARRawDataset("train", modality=modality, augment=False)
        test_ds = UCIHARRawDataset("test", modality=modality, augment=False)

    common = dict(batch_size=128, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_ds, **common)
    test_loader = DataLoader(test_ds, **common)

    train_metrics = _eval_loader(model, train_loader, device)
    test_metrics = _eval_loader(model, test_loader, device)
    gap_acc = train_metrics["accuracy"] - test_metrics["accuracy"]
    gap_f1 = train_metrics["macro_f1"] - test_metrics["macro_f1"]

    print(f"Checkpoint: {checkpoint_path}")
    print(f"  Train accuracy: {train_metrics['accuracy']:.4f}   Train macro-F1: {train_metrics['macro_f1']:.4f}")
    print(f"  Test  accuracy: {test_metrics['accuracy']:.4f}   Test  macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Gap (train - test): accuracy {gap_acc:+.4f}   macro-F1 {gap_f1:+.4f}")
    if gap_acc > 0.10:
        print("  -> Gap > 10pp: meaningful overfitting signal, worth investigating "
              "(more regularization, more data, fewer epochs).")
    else:
        print("  -> Gap is small: no strong evidence of overfitting; the model "
              "generalizes to held-out data at roughly the same rate it fits "
              "training data.")

    out = {
        "checkpoint": checkpoint_path,
        "train_metrics": {"accuracy": train_metrics["accuracy"], "macro_f1": train_metrics["macro_f1"]},
        "test_metrics": {"accuracy": test_metrics["accuracy"], "macro_f1": test_metrics["macro_f1"]},
        "gap_accuracy": gap_acc,
        "gap_macro_f1": gap_f1,
    }
    run_name = cfg.get("run_name", "model")
    save_json(out, os.path.join(RESULTS_DIR, f"{run_name}_train_test_gap.json"))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["aggregate", "compare", "error_analysis", "train_test_gap"],
        required=True,
    )
    parser.add_argument("--run_name")
    parser.add_argument("--run_names", nargs="+")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    if args.mode == "aggregate":
        aggregate(args.run_name)
    elif args.mode == "compare":
        compare(args.run_names)
    elif args.mode == "error_analysis":
        error_analysis(args.checkpoint)
    elif args.mode == "train_test_gap":
        train_test_gap(args.checkpoint)


if __name__ == "__main__":
    main()