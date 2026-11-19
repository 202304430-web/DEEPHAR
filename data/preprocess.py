"""
Preprocesses UCI-HAR raw inertial signals + precomputed 561-dim feature vectors.
Produces leakage-free, SUBJECT-WISE train/val/test splits (Core Req. #2).

UCI-HAR already separates 30 subjects into 21 "train" (UCI) / 9 "test" (UCI) subjects
(subject-disjoint by design). We keep the UCI test subjects untouched as our FINAL
TEST set (touched only for final evaluation). We further split the 21 UCI-train
subjects into TRAIN (17 subjects) / VAL (4 subjects) by subject id, so no subject
ever appears in more than one split.

Outputs (data/processed/):
  X_{split}_raw.npy   float32 (N, 128, 9)   9 channels: body_acc(xyz), body_gyro(xyz), total_acc(xyz)
  X_{split}_feat.npy  float32 (N, 561)      precomputed handcrafted features (baseline MLP only)
  y_{split}.npy       int64   (N,)          0-indexed labels, 6 classes
  subj_{split}.npy    int64   (N,)          subject id per window
"""
import os
import numpy as np

from download import download_and_extract

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw", "UCI HAR Dataset")
OUT_DIR = os.path.join(os.path.dirname(__file__), "processed")

SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]
VAL_SUBJECT_FRACTION = 4 / 21  # held out from the 21 UCI-train subjects, subject-wise
SEED_FOR_SPLIT = 42  # fixed (not a model seed) so the split itself is reproducible


def _load_split_raw(split):  # split: "train" | "test" (UCI naming)
    base = os.path.join(RAW_DIR, split, "Inertial Signals")
    channels = []
    for sig in SIGNALS:
        path = os.path.join(base, f"{sig}_{split}.txt")
        channels.append(np.loadtxt(path, dtype=np.float32))  # (N, 128)
    X = np.stack(channels, axis=-1)  # (N, 128, 9)
    y = np.loadtxt(os.path.join(RAW_DIR, split, f"y_{split}.txt"), dtype=np.int64) - 1  # 0-indexed
    subj = np.loadtxt(os.path.join(RAW_DIR, split, f"subject_{split}.txt"), dtype=np.int64)
    feat = np.loadtxt(os.path.join(RAW_DIR, split, f"X_{split}.txt"), dtype=np.float32)  # (N, 561)
    return X, y, subj, feat


def _save(split, X, y, subj, feat):
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, f"X_{split}_raw.npy"), X)
    np.save(os.path.join(OUT_DIR, f"X_{split}_feat.npy"), feat)
    np.save(os.path.join(OUT_DIR, f"y_{split}.npy"), y)
    np.save(os.path.join(OUT_DIR, f"subj_{split}.npy"), subj)
    print(f"[{split}] X_raw={X.shape} X_feat={feat.shape} y={y.shape} "
          f"subjects={sorted(set(subj.tolist()))}")


def main():
    download_and_extract()

    X_tr_all, y_tr_all, subj_tr_all, feat_tr_all = _load_split_raw("train")
    X_test, y_test, subj_test, feat_test = _load_split_raw("test")

    # Subject-wise train/val split within the 21 UCI-train subjects.
    rng = np.random.RandomState(SEED_FOR_SPLIT)
    uniq_subjects = np.unique(subj_tr_all)
    rng.shuffle(uniq_subjects)
    n_val = max(1, round(len(uniq_subjects) * VAL_SUBJECT_FRACTION))
    val_subjects = set(uniq_subjects[:n_val].tolist())
    train_subjects = set(uniq_subjects[n_val:].tolist())
    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(set(subj_test.tolist()))
    assert val_subjects.isdisjoint(set(subj_test.tolist()))

    train_mask = np.isin(subj_tr_all, list(train_subjects))
    val_mask = np.isin(subj_tr_all, list(val_subjects))

    _save("train", X_tr_all[train_mask], y_tr_all[train_mask], subj_tr_all[train_mask], feat_tr_all[train_mask])
    _save("val", X_tr_all[val_mask], y_tr_all[val_mask], subj_tr_all[val_mask], feat_tr_all[val_mask])
    _save("test", X_test, y_test, subj_test, feat_test)

    # Normalization stats from TRAIN ONLY (no leakage).
    X_tr = X_tr_all[train_mask]
    mean = X_tr.reshape(-1, X_tr.shape[-1]).mean(axis=0)
    std = X_tr.reshape(-1, X_tr.shape[-1]).std(axis=0) + 1e-8
    np.save(os.path.join(OUT_DIR, "channel_mean.npy"), mean.astype(np.float32))
    np.save(os.path.join(OUT_DIR, "channel_std.npy"), std.astype(np.float32))

    fmean = feat_tr_all[train_mask].mean(axis=0)
    fstd = feat_tr_all[train_mask].std(axis=0) + 1e-8
    np.save(os.path.join(OUT_DIR, "feat_mean.npy"), fmean.astype(np.float32))
    np.save(os.path.join(OUT_DIR, "feat_std.npy"), fstd.astype(np.float32))

    print("Preprocessing complete. Splits are subject-disjoint (assert-checked above).")


if __name__ == "__main__":
    main()
