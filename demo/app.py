"""
Live demo (Core Req. #12, Deliverable #10): Streamlit app.
Takes a real input (uploaded .csv window, or a sampled test-set window) and
returns the model's predicted activity + class probabilities.

Run: streamlit run demo/app.py
"""
import os
import sys

import numpy as np
import streamlit as st
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from models import MODEL_REGISTRY  # noqa: E402
from datasets.uci_har import LABEL_NAMES  # noqa: E402

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

st.set_page_config(page_title="Wearable HAR Demo", layout="centered")
st.title("Wearable Human Activity Recognition — Live Demo")
st.caption("UCI-HAR · 9-channel IMU window (128 timesteps @ 50Hz) → activity class")


@st.cache_resource
def list_checkpoints():
    if not os.path.isdir(CKPT_DIR):
        return []
    return sorted(f for f in os.listdir(CKPT_DIR) if f.endswith(".pt"))


@st.cache_resource
def load_model(ckpt_name):
    ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name), map_location="cpu")
    cfg = ckpt["cfg"]
    kwargs = dict(cfg.get("model_params", {}))
    if cfg["model"] == "baseline_mlp":
        kwargs.setdefault("in_dim", 561)
    else:
        kwargs.setdefault("in_channels", 6 if cfg.get("modality") == "accel_only" else 9)
    kwargs.setdefault("num_classes", len(LABEL_NAMES))
    model = MODEL_REGISTRY[cfg["model"]](**kwargs)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


# The 4 architectures compared in the report (Table 2); seed0 is the canonical
# checkpoint used for the reported error analysis. The other 23 checkpoints
# (LR sweep, augmentation on/off, modality ablation x3 seeds each) are real,
# valid trained models too -- see README "Checkpoint naming" and report
# Sections 11-12 -- just tucked behind "Advanced" so the demo isn't a 27-item
# dropdown by default.
MAIN_MODELS = {
    "Baseline MLP": "baseline_mlp_seed0.pt",
    "LSTM": "lstm_seed0.pt",
    "Transformer (from scratch)": "transformer_seed0.pt",
    "CNN-LSTM (advanced)": "cnn_lstm_seed0.pt",
}

ckpts = list_checkpoints()
if not ckpts:
    st.warning(
        "No checkpoints found in `checkpoints/`. Train a model first:\n\n"
        "`python src/train.py --config configs/cnn_lstm.yaml --seed 0`"
    )
    st.stop()

available_ckpts = set(ckpts)
main_options = {label: fname for label, fname in MAIN_MODELS.items() if fname in available_ckpts}

st.subheader("Model")
if main_options:
    choice_label = st.selectbox("Choose a model", list(main_options.keys()))
    ckpt_name = main_options[choice_label]
else:
    st.info("None of the 4 default seed0 checkpoints were found; pick one below instead.")
    ckpt_name = ckpts[0]

with st.expander(f"Advanced — all {len(ckpts)} checkpoints (seeds, LR sweep, augmentation study, modality ablation)"):
    st.caption(
        "Every config in `run_all.py` was trained with 3 random seeds. "
        "Filenames follow `<run_name>_seed<N>.pt` — see README “Checkpoint naming” "
        "or report §§11–12 for what each run_name is."
    )
    advanced_choice = st.selectbox("Pick a specific checkpoint", ["(use selection above)"] + ckpts)
    if advanced_choice != "(use selection above)":
        ckpt_name = advanced_choice

model, cfg = load_model(ckpt_name)
st.write(f"Loaded **{cfg['model']}** (modality={cfg.get('modality','all')}, "
         f"augment={cfg.get('augment', False)})")

# The baseline MLP takes the dataset's precomputed 561-dim handcrafted feature
# vector, not the raw 128x9 window -- every other model takes the raw window.
# Mixing these up is exactly the "mat1 and mat2 shapes cannot be multiplied"
# error: it must branch on this, not always load the raw array.
is_baseline = cfg["model"] == "baseline_mlp"

st.divider()
st.subheader("Input")
upload_label = "Upload CSV (561 values, single row)" if is_baseline else "Upload CSV (128 x 9)"
mode = st.radio("Input source", ["Sample from test set", upload_label])

window = None
if mode == "Sample from test set":
    if not os.path.isdir(PROC_DIR):
        st.error("No processed data found. Run `python data/preprocess.py` first.")
        st.stop()
    y_test = np.load(os.path.join(PROC_DIR, "y_test.npy"))
    if is_baseline:
        X_test = np.load(os.path.join(PROC_DIR, "X_test_feat.npy"))
        mean = np.load(os.path.join(PROC_DIR, "feat_mean.npy"))
        std = np.load(os.path.join(PROC_DIR, "feat_std.npy"))
    else:
        X_test = np.load(os.path.join(PROC_DIR, "X_test_raw.npy"))
        mean = np.load(os.path.join(PROC_DIR, "channel_mean.npy"))
        std = np.load(os.path.join(PROC_DIR, "channel_std.npy"))
    # X_test_raw.npy and X_test_feat.npy are built from the same test split in
    # the same row order (data/preprocess.py), so the same index names the
    # same underlying window/example in either array.
    idx = st.slider("Test sample index", 0, len(X_test) - 1, 0)
    window = (X_test[idx] - mean) / std
    st.caption(f"Ground truth: **{LABEL_NAMES[y_test[idx]]}**")
else:
    up = st.file_uploader(upload_label, type="csv")
    if up is not None:
        arr = np.loadtxt(up, delimiter=",")
        expected_shape = (561,) if is_baseline else (128, 9)
        if is_baseline and arr.shape not in [(561,), (1, 561)]:
            st.error(f"Expected 561 values, got shape {arr.shape}")
        elif not is_baseline and arr.shape != (128, 9):
            st.error(f"Expected shape (128, 9), got {arr.shape}")
        else:
            if is_baseline:
                arr = arr.reshape(561)
                mean = np.load(os.path.join(PROC_DIR, "feat_mean.npy"))
                std = np.load(os.path.join(PROC_DIR, "feat_std.npy"))
            else:
                mean = np.load(os.path.join(PROC_DIR, "channel_mean.npy"))
                std = np.load(os.path.join(PROC_DIR, "channel_std.npy"))
            window = (arr - mean) / std

if window is not None and st.button("Run inference"):
    x = window.astype(np.float32)
    if is_baseline:
        x_t = torch.from_numpy(x).unsqueeze(0)  # (1, 561)
    else:
        if cfg.get("modality") == "accel_only":
            x = np.delete(x, [3, 4, 5], axis=-1)  # drop gyro channels
        x_t = torch.from_numpy(x).unsqueeze(0)  # (1, 128, C)
    with torch.no_grad():
        logits = model(x_t)
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
    pred_idx = int(probs.argmax())

    st.success(f"Predicted activity: **{LABEL_NAMES[pred_idx]}**")
    st.bar_chart({LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))})