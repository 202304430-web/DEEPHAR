"""
Time-domain augmentation for 9-channel IMU windows (N, 128, C).
Used only on TRAIN split; controlled by configs/*.yaml `augment: true|false`
for the data-strategy ablation study (Core Req. #7).
"""
import numpy as np


def jitter(x, sigma=0.03):
    return x + np.random.normal(0.0, sigma, size=x.shape).astype(np.float32)


def scaling(x, sigma=0.1):
    # per-channel random scale factor, broadcast over time axis
    factor = np.random.normal(1.0, sigma, size=(1, x.shape[-1])).astype(np.float32)
    return x * factor


def time_warp(x, sigma=0.2, knots=4):
    """Smooth random time warp via a monotonic cubic-ish piecewise-linear resample."""
    T = x.shape[0]
    orig_steps = np.arange(T)
    random_warp = np.random.normal(1.0, sigma, size=knots + 2)
    warp_steps = np.linspace(0, T - 1, num=knots + 2)
    cum = np.cumsum(random_warp)
    cum = (cum - cum[0]) / (cum[-1] - cum[0]) * (T - 1)
    new_index = np.interp(orig_steps, warp_steps, cum)
    out = np.zeros_like(x)
    for c in range(x.shape[-1]):
        out[:, c] = np.interp(orig_steps, new_index, x[:, c])
    return out.astype(np.float32)


def augment_window(x, p_jitter=0.5, p_scale=0.5, p_warp=0.3):
    """x: (128, C) float32. Applies each transform independently at random."""
    if np.random.rand() < p_jitter:
        x = jitter(x)
    if np.random.rand() < p_scale:
        x = scaling(x)
    if np.random.rand() < p_warp:
        x = time_warp(x)
    return x
