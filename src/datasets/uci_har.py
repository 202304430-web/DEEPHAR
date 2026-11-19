"""
PyTorch Dataset/loaders for UCI-HAR. Loads preprocessed .npy from data/processed/.

Channel layout (raw, 9ch): [body_acc_x,y,z, body_gyro_x,y,z, total_acc_x,y,z]
Modality ablation (Core Req. #8): `modality="accel_only"` drops the 3 gyro
channels (indices 3,4,5) -> 6 channels. `modality="all"` keeps all 9.

Classes (0-indexed): 0 WALKING, 1 WALKING_UPSTAIRS, 2 WALKING_DOWNSTAIRS,
                      3 SITTING, 4 STANDING, 5 LAYING
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from . import augmentation as aug

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
GYRO_IDX = [3, 4, 5]
NUM_CLASSES = 6

LABEL_NAMES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]


def _drop_modality(X, modality):
    if modality == "all":
        return X
    if modality == "accel_only":
        keep = [i for i in range(X.shape[-1]) if i not in GYRO_IDX]
        return X[..., keep]
    raise ValueError(f"Unknown modality: {modality}")


class UCIHARRawDataset(Dataset):
    """Raw 9(or 6)-channel windows for LSTM / Transformer / CNN-LSTM."""

    def __init__(self, split, modality="all", augment=False):
        assert split in ("train", "val", "test")
        self.split = split
        self.modality = modality
        self.augment = augment and split == "train"  # never augment val/test

        X = np.load(os.path.join(PROC_DIR, f"X_{split}_raw.npy"))
        mean = np.load(os.path.join(PROC_DIR, "channel_mean.npy"))
        std = np.load(os.path.join(PROC_DIR, "channel_std.npy"))
        X = (X - mean) / std  # normalize with TRAIN-only stats (no leakage)

        self.X = _drop_modality(X, modality).astype(np.float32)
        self.y = np.load(os.path.join(PROC_DIR, f"y_{split}.npy")).astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]  # (128, C)
        if self.augment:
            x = aug.augment_window(x)
        return torch.from_numpy(x), torch.tensor(self.y[idx])

    @property
    def num_channels(self):
        return self.X.shape[-1]


class UCIHARFeatureDataset(Dataset):
    """Precomputed 561-dim handcrafted features, for the baseline MLP only."""

    def __init__(self, split):
        assert split in ("train", "val", "test")
        X = np.load(os.path.join(PROC_DIR, f"X_{split}_feat.npy"))
        mean = np.load(os.path.join(PROC_DIR, "feat_mean.npy"))
        std = np.load(os.path.join(PROC_DIR, "feat_std.npy"))
        self.X = ((X - mean) / std).astype(np.float32)
        self.y = np.load(os.path.join(PROC_DIR, f"y_{split}.npy")).astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


def get_loaders(kind, modality="all", augment=False, batch_size=64, num_workers=2):
    """
    kind: "raw" (for LSTM/Transformer/CNN-LSTM) or "feat" (for baseline MLP).
    Tuned for RTX 3060 / 16GB RAM: small batch default, pin_memory, modest workers.
    """
    if kind == "raw":
        train_ds = UCIHARRawDataset("train", modality=modality, augment=augment)
        val_ds = UCIHARRawDataset("val", modality=modality, augment=False)
        test_ds = UCIHARRawDataset("test", modality=modality, augment=False)
    elif kind == "feat":
        train_ds = UCIHARFeatureDataset("train")
        val_ds = UCIHARFeatureDataset("val")
        test_ds = UCIHARFeatureDataset("test")
    else:
        raise ValueError(f"Unknown kind: {kind}")

    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                  persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader
