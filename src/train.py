"""
Training entrypoint for one (model, seed, config) run.
Called directly or via run_all.py (3-seed sweep, Core Req. #6).

Usage:
  python src/train.py --config configs/lstm.yaml --seed 0
  python src/train.py --config configs/ablation_no_gyro.yaml --seed 0
"""
import argparse
import os
import sys
import time
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(__file__))
from datasets.uci_har import get_loaders, LABEL_NAMES, NUM_CLASSES  # noqa: E402
from models import MODEL_REGISTRY  # noqa: E402
from utils import (  # noqa: E402
    set_seed, get_device, count_params, model_size_mb, Timer,
    compute_metrics, save_json,
)

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg):
    name = cfg["model"]
    cls = MODEL_REGISTRY[name]
    kwargs = cfg.get("model_params", {})
    if name == "baseline_mlp":
        kwargs.setdefault("in_dim", 561)
    else:
        in_channels = 6 if cfg.get("modality", "all") == "accel_only" else 9
        kwargs.setdefault("in_channels", in_channels)
    kwargs.setdefault("num_classes", NUM_CLASSES)
    return cls(**kwargs)


@torch.no_grad()
def run_eval(model, loader, device, criterion):
    model.eval()
    total_loss, y_true, y_pred = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        y_true.extend(y.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(y_true, y_pred, LABEL_NAMES)
    metrics["loss"] = avg_loss
    return metrics, y_true, y_pred


def train_one_run(cfg, seed, out_dir):
    set_seed(seed)
    device = get_device()

    kind = "feat" if cfg["model"] == "baseline_mlp" else "raw"
    train_loader, val_loader, test_loader = get_loaders(
        kind=kind,
        modality=cfg.get("modality", "all"),
        augment=cfg.get("augment", False),
        batch_size=cfg.get("batch_size", 64),   # RTX 3060 / 16GB RAM safe default
        num_workers=cfg.get("num_workers", 2),
    )

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg.get("lr", 1e-3),
                      weight_decay=cfg.get("weight_decay", 1e-4))

    n_params = count_params(model)
    size_mb = model_size_mb(model)
    epochs = cfg.get("epochs", 50)
    patience = cfg.get("early_stop_patience", 8)

    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    with Timer() as train_timer:
        for epoch in range(1, epochs + 1):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

            val_metrics, _, _ = run_eval(model, val_loader, device, criterion)
            print(f"[{cfg['run_name']} seed={seed}] epoch {epoch}/{epochs} "
                  f"val_macro_f1={val_metrics['macro_f1']:.4f} val_acc={val_metrics['accuracy']:.4f}")

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    model.load_state_dict(best_state)

    with Timer() as test_timer:
        test_metrics, y_true, y_pred = run_eval(model, test_loader, device, criterion)
    inference_time_per_sample_ms = (test_timer.elapsed / len(test_loader.dataset)) * 1000

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"{cfg['run_name']}_seed{seed}.pt")
    torch.save({"state_dict": best_state, "cfg": cfg, "seed": seed}, ckpt_path)

    result = {
        "run_name": cfg["run_name"],
        "model": cfg["model"],
        "seed": seed,
        "modality": cfg.get("modality", "all"),
        "augment": cfg.get("augment", False),
        "lr": cfg.get("lr", 1e-3),
        "test_metrics": test_metrics,
        "val_best_macro_f1": best_val_f1,
        "num_params": n_params,
        "model_size_mb": size_mb,
        "train_time_sec": train_timer.elapsed,
        "inference_time_ms_per_sample": inference_time_per_sample_ms,
        "hardware": "RTX 3060" if device.type == "cuda" else "CPU",
        "checkpoint": ckpt_path,
    }
    result_path = os.path.join(ROOT, "experiments", "results", f"{cfg['run_name']}_seed{seed}.json")
    save_json(result, result_path)
    print(f"Saved result -> {result_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default=os.path.join(ROOT, "checkpoints"))
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs (e.g. smoke tests)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("run_name", os.path.splitext(os.path.basename(args.config))[0])
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
        cfg["early_stop_patience"] = min(cfg.get("early_stop_patience", 8), args.epochs)
    train_one_run(cfg, args.seed, args.out_dir)


if __name__ == "__main__":
    main()
