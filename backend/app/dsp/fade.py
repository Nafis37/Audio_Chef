"""
Fade -- bring the sound in from silence, and take it back out
=============================================================
In plain words: the start rises from nothing to full volume, and the end sinks from
full volume to nothing, so the clip neither pops on nor stops dead.

For a buffer of N samples, fade lengths M_in and M_out (samples) and the position
f in 0 .. 1 across a fade:

        y[n] = x[n] * g_in(n) * g_out(n)

        smooth  g(f) = (1 - cos(pi f)) / 2      an S-curve: it leaves silence gently and
                                                lands on full volume gently, so neither
                                                end of the fade has a corner you can hear
        linear  g(f) = f                        a straight ramp in amplitude

g_in runs f from 0 to 1 over the first M_in samples; g_out runs it from 1 to 0 over the
last M_out.  Both are 1 everywhere else.

When the two fades together are longer than the clip, both are shortened in proportion
so they meet in the middle rather than overlap.
"""

from __future__ import annotations

import numpy as np

CURVES = ("smooth", "linear")


def fade_gain(n: int, rising: bool, curve: str = "smooth") -> np.ndarray:
    """n samples of fade gain, 0 -> 1 when rising, 1 -> 0 when falling."""
    f = (np.arange(n) + 0.5) / max(n, 1)
    if not rising:
        f = f[::-1]
    return 0.5 - 0.5 * np.cos(np.pi * f) if curve == "smooth" else f


def fade(x: np.ndarray, fs: int, fade_in: float = 1.0, fade_out: float = 2.0,
         curve: str = "smooth") -> np.ndarray:
    """Fade in over the first `fade_in` seconds and out over the last `fade_out`."""
    x = np.asarray(x, dtype=np.float64)
    n_in = int(round(max(0.0, fade_in) * fs))
    n_out = int(round(max(0.0, fade_out) * fs))
    if n_in + n_out > x.size and n_in + n_out > 0:     # shorten both, in proportion
        n_in = int(x.size * n_in / (n_in + n_out))
        n_out = x.size - n_in
    y = x.copy()
    if n_in:
        y[:n_in] *= fade_gain(n_in, rising=True, curve=curve)
    if n_out:
        y[x.size - n_out:] *= fade_gain(n_out, rising=False, curve=curve)
    return y
