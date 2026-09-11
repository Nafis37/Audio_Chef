"""
Simple Audio Compressor -- envelope follower + dB-domain gain computer
=====================================================================
(full derivation in the sections above)
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12     # keeps log10(0) from blowing up


def envelope_follower(x: np.ndarray, fs: int, attack_ms: float, release_ms: float) -> np.ndarray:
    """|x| smoothed by a one-pole IIR with separate attack and release coefficients."""
    # a = exp(-1/(tau*fs)), tau in seconds.  The clamp guards against a 0 ms setting.
    a_att = np.exp(-1.0 / (max(attack_ms, 0.01) * 0.001 * fs))
    a_rel = np.exp(-1.0 / (max(release_ms, 0.01) * 0.001 * fs))

    rectified = np.abs(x)
    env = np.zeros_like(rectified)
    prev = 0.0
    for n in range(rectified.size):
        # Rising signal -> fast attack coefficient; falling -> slow release coefficient.
        a = a_att if rectified[n] > prev else a_rel
        prev = a * prev + (1.0 - a) * rectified[n]
        env[n] = prev
    return env


def compressor(
    x: np.ndarray,
    fs: int,
    threshold: float = -20.0,
    ratio: float = 4.0,
    attack: float = 10.0,
    release: float = 100.0,
    knee: float = 6.0,
    makeup: float = 0.0,
) -> np.ndarray:
    """threshold/knee/makeup in dB, ratio as N:1, attack/release in milliseconds."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x

    ratio = max(float(ratio), 1.0)
    knee = max(float(knee), 0.0)

    env = envelope_follower(x, fs, attack, release)
    env_db = 20.0 * np.log10(np.maximum(env, EPS))       # envelope in dBFS
    over = env_db - threshold                            # how far above the threshold

    slope = 1.0 - 1.0 / ratio                            # dB of reduction per dB over
    gain_db = np.zeros_like(over)

    if knee > 0.0:
        knee_zone = (over > -knee / 2.0) & (over < knee / 2.0)
        above = over >= knee / 2.0
        # Quadratic blend inside the knee (derivation in the module docstring)
        gain_db[knee_zone] = -slope * (over[knee_zone] + knee / 2.0) ** 2 / (2.0 * knee)
        gain_db[above] = -slope * over[above]
    else:
        above = over > 0.0
        gain_db[above] = -slope * over[above]

    gain_db += float(makeup)                             # fixed makeup boost
    gain = 10.0 ** (gain_db / 20.0)                      # back to a linear multiplier
    return x * gain                                      # applied sample by sample