# Wearable Human Activity Recognition — Deep Learning Final Project (AI403)

## 1. Project, Team, Idea

- **Course:** Deep Learning – AI403, Fall 2026, Pharos University in Alexandria
- **Team:** [DEEPHAR] — [Eyad Wael], [Youssef Mahmoud], [Youssef Mostafa], [Dalia Elsayed], [Yasmin Mohamed]
- **Idea chosen:** #9 — Wearable Human Activity Recognition (Inertial sensors → activity class)
- **Dataset:** UCI-HAR (Human Activity Recognition Using Smartphones), raw 9-channel inertial
  signals, subject-wise split (leakage-free)

## 2. Task & Best Result

Input: a 128-timestep window (2.56s @ 50Hz) of 9-channel IMU data (body accelerometer,
body gyroscope, total accelerometer, each x/y/z). Output: one of 6 activities (WALKING,
WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING).

Four approaches are compared under an identical protocol (same split, preprocessing,
tuning budget, 3 seeds): a baseline MLP on handcrafted features, a classic BiLSTM, a
from-scratch Transformer encoder, and an advanced CNN-LSTM hybrid with attention pooling.
**Best result:** [94.0% / Baseline MLP] — see
`experiments/results/comparison_table.csv`.

## 3. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested on: Python 3.10+, PyTorch 2.1+, an RTX 3060 (12GB VRAM) + 16GB system RAM.
All batch sizes/worker counts in `configs/*.yaml` are set for this budget — see Section 8.

## 4. Dataset Download & Preprocessing

```bash
python data/download.py     # fetches + extracts UCI-HAR into data/raw/
python data/preprocess.py   # builds leakage-free subject-wise train/val/test -> data/processed/
```

This writes `data/processed/{X,y,subj}_{train,val,test}.npy` plus channel/feature
normalization stats (computed from TRAIN only). Raw data is **not** committed to git
(see `data/` — download scripts only, per Section 4.1).

**Split strategy:** UCI-HAR ships 21 subjects as "train" and 9 as "test", already
subject-disjoint. We keep the 9 UCI-test subjects as our final **test** set (touched
only for final evaluation, per Core Req. #2). The 21 UCI-train subjects are further
split by subject (17 train / 4 val) so no subject ever appears in two splits —
asserted in code (`data/preprocess.py`).

## 5. Training

Train one (model, seed) run:

```bash
python src/train.py --config configs/baseline_mlp.yaml --seed 0
python src/train.py --config configs/lstm.yaml          --seed 0
python src/train.py --config configs/transformer.yaml   --seed 0
python src/train.py --config configs/cnn_lstm.yaml       --seed 0
```

Run the **full protocol** (4 main models + LR sweep + augmentation study + modality
ablation, each x3 seeds, per Core Req. #6–#8) in one go:

```bash
python run_all.py            # full run
python run_all.py --quick    # 2-epoch smoke test, validates the pipeline only
```
Trained Checkpoints -> https://drive.google.com/drive/folders/1WseTer5ws8dRdT7Jx2WKdrGOxveb5toV?usp=sharing

Checkpoints -> `checkpoints/<run_name>_seed<N>.pt`. Per-run metrics ->
`experiments/results/<run_name>_seed<N>.json`.

### Checkpoint naming

`run_all.py` trains 9 configs x 3 seeds = 27 checkpoints total. `<run_name>` is:

| `run_name` | What it is |
| --- | --- |
| `baseline_mlp` | Baseline MLP on handcrafted features (Table 2) |
| `lstm` | Classic BiLSTM (Table 2) |
| `transformer` | From-scratch Transformer encoder (Table 2) |
| `cnn_lstm` | Advanced CNN-LSTM hybrid, default config (Table 2, main results) |
| `cnn_lstm_lr1e-2` / `cnn_lstm_lr1e-3` / `cnn_lstm_lr1e-4` | CNN-LSTM, learning-rate sweep (hyperparameter study) |
| `cnn_lstm_no_augment` | CNN-LSTM with augmentation off (data-strategy study; `cnn_lstm` is its augment-on counterpart) |
| `cnn_lstm_ablation_no_gyro` | CNN-LSTM, gyroscope channels removed (modality ablation study, Table 5) |

Each has a `_seed0`/`_seed1`/`_seed2` suffix. `demo/app.py` shows only the 4 main
models' `_seed0` checkpoint by default; the other 23 are available under its
"Advanced" expander.

## 6. Evaluation

```bash
# mean +/- SD across the 3 seeds for one run
python src/evaluate.py --mode aggregate --run_name cnn_lstm

# cross-model fair-comparison table (Deliverable #5) -> experiments/results/comparison_table.csv
python src/evaluate.py --mode compare --run_names baseline_mlp lstm transformer cnn_lstm

# misclassified test examples for a given checkpoint (Core Req. #10)
python src/evaluate.py --mode error_analysis --checkpoint checkpoints/cnn_lstm_seed0.pt
```

## 7. Demo

```bash
streamlit run demo/app.py
```

Pick a trained checkpoint, either sample a window from the held-out test set or upload
a `(128, 9)` CSV window, and get the predicted activity + class probabilities.

## 8. Hardware & Expected Runtimes

- **Target hardware:** NVIDIA RTX 3060 (12GB VRAM), 16GB system RAM.
- Batch size 64, `num_workers=2`, `pin_memory=True` — UCI-HAR windows are tiny
  (128x9 floats/sample) so this configuration comfortably avoids OOM on this card;
  all 4 models are <150K parameters (<1MB checkpoints).
- Expected per-model training time on the target GPU: **under 5 minutes** for 60
  epochs with early stopping (patience 8–10); the full `run_all.py` protocol
  (9 configs x 3 seeds = 27 runs) completes in well under an hour. CPU-only
  training is also feasible given the dataset size, just slower.
- Exact per-run `train_time_sec` / `inference_time_ms_per_sample` / parameter
  count / model size are logged automatically for every run (Core Req. #11,
  Table 7 in the report) — see `experiments/results/*.json`.

## 9. Credits

- **Dataset:** Anguita, D. et al., "A Public Domain Dataset for Human Activity
  Recognition Using Smartphones," UCI Machine Learning Repository.
- **Libraries:** PyTorch, scikit-learn (metrics only), Streamlit (demo UI).
- **Architectures:** standard BiLSTM and CNN-LSTM designs adapted for this task;
  Transformer encoder follows Vaswani et al. (2017), "Attention Is All You Need" —
  multi-head self-attention is **implemented from scratch** in
  `src/models/transformer.py` (`ManualMultiHeadSelfAttention`), not called from a
  library, per Core Req. #5.
- No external AI APIs are used for predictions, per Core Req. #13.

---

## Notes on Scoping Decisions

- **Pretrained-vs-scratch comparison (Core Req. #7):** marked **N/A** for this
  project — per team's architecture choice, the Transformer encoder is trained
  from scratch rather than fine-tuned from a pretrained IMU/self-supervised
  encoder, so there is no scratch-vs-pretrained pair to compare. This is a
  conscious scoping decision, documented here and in the technical report rather
  than silently omitted.
- **Modality ablation:** removes the gyroscope channels (9ch -> 6ch, accelerometer-only),
  since UCI-HAR's three sensor streams (body-acc, body-gyro, total-acc) make the
  gyroscope the natural single modality to ablate.
- **Hyperparameter study axis:** learning rate (1e-2 / 1e-3 / 1e-4), run on the
  advanced CNN-LSTM model (`configs/hp_lr_*.yaml`).
- **Data-strategy study:** augmentation on/off (`configs/cnn_lstm.yaml` vs
  `configs/data_strategy_no_augment.yaml`) — jitter, scaling, and time-warp,
  applied on TRAIN only (`src/datasets/augmentation.py`).