"""
Echo and Reverb Studio -- feedback delay line + Schroeder reverberator
======================================================================
(derivations above; source: M. R. Schroeder, "Natural Sounding Artificial
Reverberation", JAES 1962)
"""

from __future__ import annotations

import numpy as np


def _comb_feedback_scalar(x: np.ndarray, delay: int, g: float) -> np.ndarray:
    """Reference implementation: y[n] = x[n] + g*y[n-D], literally one sample at a time."""
    y = np.zeros_like(x)
    for n in range(x.size):
        # For n < D the delayed output does not exist yet, so it counts as 0.
        y[n] = x[n] + (g * y[n - delay] if n >= delay else 0.0)
    return y


def _comb_feedback(x: np.ndarray, delay: int, g: float) -> np.ndarray:
    """Same recursion, evaluated one block of D samples at a time (see module docstring)."""
    y = x.astype(np.float64, copy=True)
    n = y.size
    for start in range(delay, n, delay):
        stop = min(start + delay, n)
        # y[start:stop] += g * y[start-D : stop-D]   <- the previous block, already final
        y[start:stop] += g * y[start - delay:stop - delay]
    return y


def _allpass(x: np.ndarray, delay: int, g: float) -> np.ndarray:
    """y[n] = -g*x[n] + x[n-D] + g*y[n-D], same block trick as the comb."""
    n = x.size
    # The feed-forward part -g*x[n] + x[n-D] has no recursion, so build it in one shot.
    y = -g * x
    y[delay:] += x[:n - delay]
    # Then add the recursive term block by block.
    for start in range(delay, n, delay):
        stop = min(start + delay, n)
        y[start:stop] += g * y[start - delay:stop - delay]
    return y


def echo(
    x: np.ndarray,
    fs: int,
    delay: float = 0.3,
    feedback: float = 0.4,
    mix: float = 0.5,
) -> np.ndarray:
    """Feedback delay line.  delay in seconds, feedback = g, mix = wet amount 0..1."""
    x = np.asarray(x, dtype=np.float64)
    d = max(1, int(round(delay * fs)))
    if d >= x.size:                 # delay longer than the file -> nothing would repeat
        return x
    g = float(np.clip(feedback, 0.0, 0.95))   # < 1 keeps the loop stable (see docstring)

    wet = _comb_feedback(x, d, g)
    # _comb_feedback already contains the dry signal (the x[n] term), so this crossfade
    # runs between "input" and "input + repeats".
    m = float(np.clip(mix, 0.0, 1.0))
    return (1.0 - m) * x + m * wet


# Schroeder's comb delays in milliseconds -- mutually prime on purpose.
_COMB_MS = (29.7, 37.1, 41.1, 43.7)
# Allpass delays are much shorter: they diffuse rather than repeat.
_ALLPASS_MS = (5.0, 1.7)
_ALLPASS_G = 0.7            # the classic Schroeder allpass coefficient


def reverb(
    x: np.ndarray,
    fs: int,
    room_size: float = 0.5,
    decay: float = 2.0,
    mix: float = 0.3,
) -> np.ndarray:
    """Schroeder reverberator: 4 parallel combs -> 2 series allpasses -> dry/wet mix."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x

    scale = 0.5 + float(np.clip(room_size, 0.0, 1.0))   # 0.5x .. 1.5x the nominal delays
    rt60 = max(float(decay), 0.05)

    # --- 4 parallel comb filters ---------------------------------------------------
    wet = np.zeros_like(x)
    for ms in _COMB_MS:
        d = max(1, int(round(ms * scale * fs / 1000.0)))
        # g from the RT60 relation derived in the docstring
        g = float(np.clip(10.0 ** (-3.0 * d / (rt60 * fs)), 0.0, 0.95))
        wet += _comb_feedback(x, d, g)
    wet /= len(_COMB_MS)        # average, so 4 parallel paths do not quadruple the level

    # --- 2 allpass filters in series -----------------------------------------------
    for ms in _ALLPASS_MS:
        d = max(1, int(round(ms * scale * fs / 1000.0)))
        wet = _allpass(wet, d, _ALLPASS_G)

    m = float(np.clip(mix, 0.0, 1.0))
    return (1.0 - m) * x + m * wet