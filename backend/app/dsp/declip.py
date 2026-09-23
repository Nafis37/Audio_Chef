"""
De-clip -- redraw the tops that clipping flattened
==================================================
In plain words: when a recording was too loud, its peaks were sliced flat at the
converter's limit.  This finds each flat top and redraws it as a smooth curve that
continues the slopes going into and coming out of it.

Finding the flat tops
---------------------
        level  = thresh * max |x[n]|         (relative, so a file that was clipped and
                                             then turned down is still found)
        a run  = consecutive samples with |x[n]| >= level, all of one sign
        repair it only if it is >= MIN_RUN samples long -- a real waveform touches its
        own maximum for one sample; a clipped one sits there.

Redrawing a run -- cubic Hermite interpolation
----------------------------------------------
For a run occupying samples a .. b-1, use the good samples on either side:

        p0 = x[a-1]                 m0 = x[a-1] - x[a-2]      (slope going in, per sample)
        p1 = x[b]                   m1 = x[b+1] - x[b]        (slope coming out)
        L  = b - (a - 1)            (samples from p0 to p1)
        t  = (n - (a-1)) / L        for n = a .. b-1          (0 < t < 1)

The cubic p(t) with p(0) = p0, p(1) = p1, p'(0) = L m0, p'(1) = L m1 is

        p(t) = h00 p0 + h10 L m0 + h01 p1 + h11 L m1

        h00 = 2t^3 - 3t^2 + 1        h10 = t^3 - 2t^2 + t
        h01 = -2t^3 + 3t^2           h11 = t^3 - t^2

(the Hermite basis: each h is 1 in exactly one of the four conditions and 0 in the
other three).  The slope rising into the flat top carries the curve ABOVE the clip
level, which is the peak that was lost.  A run is never redrawn below the level it was
clipped at: |y| = max(|p(t)|, level).

The rebuilt peaks can exceed full scale; the recipe's final fit_to_full_scale turns the
whole file down to fit, so the repair is never clipped again.
"""

from __future__ import annotations

import numpy as np

MIN_RUN = 3          # samples: shorter "runs" are just a waveform touching its maximum


def clipped_runs(x: np.ndarray, thresh: float = 0.99) -> list[tuple[int, int]]:
    """[start, end) of every flat top: >= MIN_RUN samples of one sign at the clip level."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return []
    peak = float(np.max(np.abs(x)))
    if peak <= 0.0:
        return []
    level = thresh * peak
    # +1 at the positive ceiling, -1 at the negative floor, 0 in between: a run is a
    # stretch of one constant nonzero value, so a sign flip ends it too.
    state = np.where(x >= level, 1, np.where(x <= -level, -1, 0)).astype(np.int8)
    change = np.flatnonzero(np.diff(np.concatenate([[0], state, [0]])))
    runs = []
    for a, b in zip(change[:-1], change[1:]):
        if state[a] != 0 and b - a >= MIN_RUN:
            runs.append((int(a), int(b)))
    return runs


def declip(x: np.ndarray, fs: int, thresh: float = 0.99) -> np.ndarray:
    """Redraw every clipped run with a cubic Hermite curve (module docstring)."""
    x = np.asarray(x, dtype=np.float64)
    runs = clipped_runs(x, thresh)
    if not runs:
        return x
    y = x.copy()
    level = thresh * float(np.max(np.abs(x)))
    for a, b in runs:
        # Needs two good samples on each side to read a slope; runs at the very edge of
        # the file have no "going in" or "coming out" and are left as they are.
        if a < 2 or b + 1 >= x.size:
            continue
        p0, p1 = x[a - 1], x[b]
        m0 = x[a - 1] - x[a - 2]                 # slope going in (per sample)
        m1 = x[b + 1] - x[b]                     # slope coming out
        span = b - (a - 1)                       # L: samples from p0 to p1
        t = (np.arange(a, b) - (a - 1)) / span
        t2, t3 = t * t, t * t * t
        curve = ((2 * t3 - 3 * t2 + 1) * p0      # h00: 1 at t = 0
                 + (t3 - 2 * t2 + t) * span * m0 # h10: slope at t = 0
                 + (-2 * t3 + 3 * t2) * p1       # h01: 1 at t = 1
                 + (t3 - t2) * span * m1)        # h11: slope at t = 1
        sign = np.sign(x[a])
        y[a:b] = sign * np.maximum(np.abs(curve), level)
    return y
