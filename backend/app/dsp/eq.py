"""
Audio Equalizer -- bass shelf, mid bell, treble shelf
=====================================================
In plain words
--------------
Three tone controls, like the Bass / Mid / Treble knobs on a stereo.  Bass lifts or cuts
everything BELOW a corner frequency, Treble everything ABOVE one, and Mid a bell-shaped
band around a centre.  Each is a tiny recursive filter (a "biquad") that only looks at the
last two input and output samples, so the whole EQ costs a few multiplies per sample.

Source: R. Bristow-Johnson, "Cookbook formulae for audio EQ biquad filter coefficients"
(the RBJ Audio EQ Cookbook).  scipy.signal.lfilter only RUNS the difference equation;
every coefficient is derived here.

The biquad
----------
A second-order IIR section:

                 b0 + b1 z^-1 + b2 z^-2
        H(z) = --------------------------
                 a0 + a1 z^-1 + a2 z^-2

        y[n] = (b0 x[n] + b1 x[n-1] + b2 x[n-2] - a1 y[n-1] - a2 y[n-2]) / a0

Shared design quantities (all three filters)
--------------------------------------------
        A     = 10^(gain_dB / 40)          (sqrt of the linear gain)
        w0    = 2 pi f0 / fs               (corner / centre, rad/sample)
        alpha = sin(w0) / (2 Q)            (bandwidth term)

Each analog prototype H(s) is mapped to z with the bilinear transform, pre-warped so that
f0 lands exactly where it was asked for.

1. Peaking bell (Mid)
---------------------
Prototype  H(s) = (s^2 + s (A/Q) + 1) / (s^2 + s/(A Q) + 1)  gives

        b0 = 1 + alpha*A        a0 = 1 + alpha/A
        b1 = -2 cos(w0)         a1 = -2 cos(w0)
        b2 = 1 - alpha*A        a2 = 1 - alpha/A

At z = e^{j w0} numerator and denominator reduce to 2j sin(w0)(alpha*A) and
2j sin(w0)(alpha/A), so |H| = A^2 = 10^(gain_dB/20): exactly gain_dB at the centre.  At DC
and Nyquist (z = +-1) the two are equal and |H| = 1 -- a bell, not a shelf.

2. Shelves (Bass, Treble)
-------------------------
A shelf has gain A^2 on one side of f0 and 1 on the other, crossing A (half the dB) at f0.
Prototype for the low shelf:

        H(s) = A (s^2 + (sqrt(A)/Q) s + A) / (A s^2 + (sqrt(A)/Q) s + 1)

With shelf slope S = 1 (the steepest slope with no bump) the cookbook's Q is 1/sqrt(2),
so alpha = sin(w0) / sqrt(2).  Writing  c = cos(w0),  r = 2 sqrt(A) alpha:

    low shelf                                   high shelf
    b0 =    A ((A+1) - (A-1) c + r)             b0 =    A ((A+1) + (A-1) c + r)
    b1 =  2 A ((A-1) - (A+1) c)                 b1 = -2 A ((A-1) + (A+1) c)
    b2 =    A ((A+1) - (A-1) c - r)             b2 =    A ((A+1) + (A-1) c - r)
    a0 =       (A+1) + (A-1) c + r              a0 =       (A+1) - (A-1) c + r
    a1 =   -2 ((A-1) + (A+1) c)                 a1 =    2 ((A-1) - (A+1) c)
    a2 =       (A+1) + (A-1) c - r              a2 =       (A+1) - (A-1) c - r

The high shelf is the low shelf with z -> -z (c -> -c, and the sign of b1/a1 flipped):
mirroring the frequency axis about fs/4 swaps "below f0" with "above f0".

Why shelves for Bass/Treble instead of three bells: a bell at 120 Hz only moves a narrow
band, which on speech is barely audible.  A shelf moves everything below (or above) the
corner, which is what people mean by "more bass" -- and it shows up as a whole band of the
spectrogram getting brighter.

Stability: for A > 0, Q > 0 and 0 < w0 < pi every pole pair is inside the unit circle,
which is why f0 is clamped below 0.49 fs and Q above 0.05.  A negative gain gives A < 1 and
the cut is the exact inverse of the boost:  H_cut = 1 / H_boost.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

# The cookbook's shelf slope S = 1 corresponds to this Q: the steepest shelf with no
# overshoot bump next to the corner.
_SHELF_Q = 1.0 / np.sqrt(2.0)


def _design(fs: int, f0: float, q: float, gain_db: float):
    """The three shared quantities A, w0 and alpha, with f0 and Q kept in the stable range."""
    f0 = float(np.clip(f0, 1.0, 0.49 * fs))
    q = max(float(q), 0.05)
    A = 10.0 ** (gain_db / 40.0)          # sqrt of the linear gain
    w0 = 2.0 * np.pi * f0 / fs            # normalised corner / centre frequency
    alpha = np.sin(w0) / (2.0 * q)        # bandwidth term: bigger alpha = wider
    return A, w0, alpha


def _normalise(b0, b1, b2, a0, a1, a2):
    """lfilter wants the coefficients already divided by a0."""
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def peaking_coefficients(fs: int, f0: float, q: float, gain_db: float):
    """RBJ peaking bell: +gain_db at f0, 0 dB far away on both sides."""
    A, w0, alpha = _design(fs, f0, q, gain_db)
    c = np.cos(w0)
    return _normalise(
        1.0 + alpha * A,                  # b0: zero pair widened by A -> the bell's peak
        -2.0 * c,                         # b1: places the zeros at angle w0
        1.0 - alpha * A,                  # b2
        1.0 + alpha / A,                  # a0: pole pair narrowed by A (same angle w0)
        -2.0 * c,                         # a1: poles at angle w0 too -> DC / Nyquist at 0 dB
        1.0 - alpha / A,                  # a2
    )


def low_shelf_coefficients(fs: int, f0: float, gain_db: float):
    """RBJ low shelf: +gain_db below f0, 0 dB above, half the dB at f0."""
    A, w0, alpha = _design(fs, f0, _SHELF_Q, gain_db)
    c = np.cos(w0)
    r = 2.0 * np.sqrt(A) * alpha          # the shelf's bandwidth term
    return _normalise(
        A * ((A + 1) - (A - 1) * c + r),      # b0
        2 * A * ((A - 1) - (A + 1) * c),      # b1
        A * ((A + 1) - (A - 1) * c - r),      # b2
        (A + 1) + (A - 1) * c + r,            # a0
        -2 * ((A - 1) + (A + 1) * c),         # a1
        (A + 1) + (A - 1) * c - r,            # a2
    )


def high_shelf_coefficients(fs: int, f0: float, gain_db: float):
    """RBJ high shelf: +gain_db above f0, 0 dB below -- the low shelf mirrored (c -> -c)."""
    A, w0, alpha = _design(fs, f0, _SHELF_Q, gain_db)
    c = np.cos(w0)
    r = 2.0 * np.sqrt(A) * alpha
    return _normalise(
        A * ((A + 1) + (A - 1) * c + r),      # b0
        -2 * A * ((A - 1) + (A + 1) * c),     # b1
        A * ((A + 1) + (A - 1) * c - r),      # b2
        (A + 1) - (A - 1) * c + r,            # a0
        2 * ((A - 1) - (A + 1) * c),          # a1
        (A + 1) - (A - 1) * c - r,            # a2
    )


def _run(x: np.ndarray, gain_db: float, coefficients) -> np.ndarray:
    """Run one biquad -- or skip it entirely at 0 dB, where it is exactly the identity."""
    if abs(gain_db) < 1e-9:
        return x
    b, a = coefficients()
    return lfilter(b, a, x)


def equalizer(
    x: np.ndarray,
    fs: int,
    bass_gain: float = 0.0,
    mid_gain: float = 0.0,
    treble_gain: float = 0.0,
    bass_freq: float = 200.0,
    mid_freq: float = 1000.0,
    treble_freq: float = 4000.0,
    q: float = 1.0,
) -> np.ndarray:
    """Bass shelf -> mid bell -> treble shelf, in series.  Gains in dB, frequencies in Hz.

    `q` is the width of the mid bell only; the shelves use the fixed slope S = 1.
    """
    y = np.asarray(x, dtype=np.float64)
    y = _run(y, bass_gain, lambda: low_shelf_coefficients(fs, bass_freq, bass_gain))
    y = _run(y, mid_gain, lambda: peaking_coefficients(fs, mid_freq, q, mid_gain))
    y = _run(y, treble_gain, lambda: high_shelf_coefficients(fs, treble_freq, treble_gain))
    return y
