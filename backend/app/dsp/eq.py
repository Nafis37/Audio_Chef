from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

def peaking_coefficients(fs: int, f0: float, q: float, gain_db: float):

    f0 = float(np.clip(f0, 1.0, 0.49 * fs))
    q = max(float(q), 0.05)

    A = 10.0 ** (gain_db / 40.0)          # linear amplitude of the boost/cut
    w0 = 2.0 * np.pi * f0 / fs            # normalized centre frequency
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / (2.0 * q)        # bandwidth term. alpha controls the width of the filter.

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A

    # lfilter expects the coefficients already divided by a0.
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a

def apply_peaking(x: np.ndarray, fs: int, f0: float, q: float, gain_db: float) -> np.ndarray:
    if abs(gain_db) < 1e-9:      # 0 dB band -> nothing to do, skip the filtering cost
        return x
    b, a = peaking_coefficients(fs, f0, q, gain_db)
    return lfilter(b, a, x)

def equalizer(
    x: np.ndarray,
    fs: int,
    low_gain: float = 0.0,
    mid_gain: float = 0.0,
    high_gain: float = 0.0,
    low_freq: float = 120.0,
    mid_freq: float = 1000.0,
    high_freq: float = 6000.0,
    q: float = 1.0,
) -> np.ndarray:
    """Three-band peaking equalizer: bands chained in series (low -> mid -> high)."""
    x = np.asarray(x, dtype=np.float64)
    y = x
    for f0, gain in ((low_freq, low_gain), (mid_freq, mid_gain), (high_freq, high_gain)):
        y = apply_peaking(y, fs, f0, q, gain)
    return y