"""
Level & DC -- remove a DC offset, then set the level
====================================================
In plain words: re-centre a waveform that sits above or below the zero line, then turn
the whole file up or down so its loudest peak (or its average loudness) hits a target.

DC removal -- mean subtraction, then a one-pole DC blocker
----------------------------------------------------------
A constant offset d is removed exactly by subtracting the mean:  x[n] <- x[n] - mean(x).
An offset that DRIFTS (a warming-up preamp) is not constant, so a DC blocker follows:

        y[n] = x[n] - x[n-1] + R y[n-1]

        H(z) = (1 - z^-1) / (1 - R z^-1)

  * zero at z = 1 (w = 0): H(e^{j0}) = 0 -- DC is removed completely
  * pole at z = R, just inside the unit circle: away from DC, |e^{jw} - 1| ~ |e^{jw} - R|,
    so |H| ~ 1 -- everything audible passes
  * the corner: near w = 0, |1 - e^{-jw}| ~ w and |1 - R e^{-jw}| ~ sqrt((1-R)^2 + w^2),
    so |H| = 1/sqrt 2 at  w_c = 1 - R, i.e.

        R = 1 - 2 pi fc / fs              (fc = 10 Hz: below any musical content)

  Mean subtraction first also means the blocker starts from a zero-mean input, so its
  start-up transient (a decaying d * R^n) does not appear at the beginning of the file.

Normalisation -- one scalar gain
--------------------------------
        peak target:  g = 10^(T/20) / max |x[n]|
        RMS target:   g = 10^(T/20) / sqrt( mean x[n]^2 )
        y = g x                              (linear: the sound is unchanged, only louder
                                             or quieter)

An RMS target can push peaks past full scale; the recipe's final fit_to_full_scale
turns the whole file down if so, rather than clipping it.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

DC_CUTOFF_HZ = 10.0
EPS = 1e-12


def remove_dc(x: np.ndarray, fs: int, fc: float = DC_CUTOFF_HZ) -> np.ndarray:
    """Mean subtraction followed by the one-pole DC blocker (module docstring)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    centred = x - np.mean(x)
    r = 1.0 - 2.0 * np.pi * fc / fs     # pole radius: the -3 dB corner lands at fc
    b = [1.0, -1.0]                     # numerator  1 - z^-1   (zero at DC)
    a = [1.0, -r]                       # denominator 1 - R z^-1 (pole just inside z = 1)
    return lfilter(b, a, centred)


def normalise(x: np.ndarray, target: str = "peak", level_db: float = -1.0) -> np.ndarray:
    """Scale x so its peak (or RMS) sits at `level_db` dBFS.  Silence is left alone."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or target == "none":
        return x
    if target == "rms":
        measured = float(np.sqrt(np.mean(x * x)))
    else:
        measured = float(np.max(np.abs(x)))
    if measured <= EPS:
        return x
    return x * (10.0 ** (level_db / 20.0) / measured)


def level(
    x: np.ndarray,
    fs: int,
    dc: bool = True,
    target: str = "peak",
    peak_db: float = -1.0,
    rms_db: float = -18.0,
) -> np.ndarray:
    """Recipe-facing wrapper: optional DC removal, then normalisation."""
    y = remove_dc(x, fs) if dc else np.asarray(x, dtype=np.float64)
    return normalise(y, target, rms_db if target == "rms" else peak_db)
