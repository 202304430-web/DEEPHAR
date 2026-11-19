"""
Pulls ONE real representative test-set index for each of Table 6's 9
true->pred confusion-type rows (not just the first 20 misclassified
overall, which are all the same SITTING->STANDING run).

Run from the project root:
    python table6_examples.py --checkpoint checkpoints/cnn_lstm_seed0.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from datasets.uci_har import get_loaders, LABEL_NAMES  # noqa: E402
from models import MODEL_REGISTRY  # noqa: E402

# Must match Table 6's rows exactly, in order.
TARGET_PAIRS = [
    ("SITTING", "STANDING"),
    ("STANDING", "SITTING"),
    ("WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"),
    ("WALKING", "WALKING_DOWNSTAIRS"),
    ("WALKING_DOWNSTAIRS", "WALKING"),
    ("SITTING", "WALKING_UPSTAIRS"),
    ("WALKING_UPSTAIRS", "WALKING"),
    ("WALKING_DOWNSTAIRS", "WALKING_UPSTAIRS"),
    ("LAYING", "WALKING_UPSTAIRS"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["cfg"]

    kwargs = dict(cfg.get("model_params", {}))
    if cfg["model"] != "baseline_mlp":
        kwargs.setdefault("in_channels", 6 if cfg.get("modality") == "accel_only" else 9)
    else:
        kwargs.setdefault("in_dim", 561)
    kwargs.setdefault("num_classes", len(LABEL_NAMES))
    model = MODEL_REGISTRY[cfg["model"]](**kwargs)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    kind = "feat" if cfg["model"] == "baseline_mlp" else "raw"
    _, _, test_loader = get_loaders(kind=kind, modality=cfg.get("modality", "all"), batch_size=64)

    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x).argmax(dim=1)
            all_true.extend(y.tolist())
            all_pred.extend(pred.cpu().tolist())

    found = {pair: None for pair in TARGET_PAIRS}
    for idx, (t, p) in enumerate(zip(all_true, all_pred)):
        pair = (LABEL_NAMES[t], LABEL_NAMES[p])
        if pair in found and found[pair] is None:
            found[pair] = idx

    print(f"{'True':<20} {'Pred':<20} {'Example test idx'}")
    for (t, p), idx in found.items():
        print(f"{t:<20} {p:<20} {idx if idx is not None else 'NOT FOUND'}")


if __name__ == "__main__":
    main()
