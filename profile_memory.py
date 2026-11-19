"""
Peak GPU memory profiling for Table 7 (Core Req. #11 / Deliverable #5 gap fix).
Loads each of the 4 main seed0 checkpoints, runs one inference batch on the
GPU, and reports torch.cuda.max_memory_allocated() -- the actual peak VRAM
used by that model's forward pass, not an estimate.

Run from the project root (same place you run streamlit/train.py):
    python profile_memory.py

Prints a table you can paste back to fill Table 7's "Peak GPU Mem (MB)" column.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from models import MODEL_REGISTRY  # noqa: E402
from datasets.uci_har import LABEL_NAMES  # noqa: E402

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

RUNS = ["baseline_mlp", "lstm", "transformer", "cnn_lstm"]


def profile_one(run_name, device):
    ckpt_path = os.path.join(CKPT_DIR, f"{run_name}_seed0.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]

    kwargs = dict(cfg.get("model_params", {}))
    if cfg["model"] == "baseline_mlp":
        kwargs.setdefault("in_dim", 561)
        x = torch.randn(64, 561)
    else:
        in_ch = 6 if cfg.get("modality") == "accel_only" else 9
        kwargs.setdefault("in_channels", in_ch)
        x = torch.randn(64, 128, in_ch)
    kwargs.setdefault("num_classes", len(LABEL_NAMES))

    model = MODEL_REGISTRY[cfg["model"]](**kwargs)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    x = x.to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(10):  # warmup + stabilize peak
            _ = model(x)
        torch.cuda.synchronize()

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return peak_mb


def main():
    if not torch.cuda.is_available():
        print("No CUDA GPU available -- this script must run on the RTX 3060 machine.")
        return
    device = torch.device("cuda")
    print(f"{'run_name':<15} {'Peak GPU Mem (MB)':>18}")
    for run_name in RUNS:
        try:
            mb = profile_one(run_name, device)
            print(f"{run_name:<15} {mb:>18.2f}")
        except FileNotFoundError:
            print(f"{run_name:<15} {'checkpoint not found':>18}")


if __name__ == "__main__":
    main()
