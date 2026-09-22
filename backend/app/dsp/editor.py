"""
Mini Audio Editor -- trim and splice
====================================
Pure time-domain array operation: numpy slicing and concatenation, nothing more.

    trim(start, end)   -> keep   x[start_sample : end_sample]
    splice(start, end) -> remove that region:  concatenate(x[:start], x[end:])
A few milliseconds of linear fade at each new boundary removes the step.
"""

from __future__ import annotations

import numpy as np

FADE_MS = 5.0   # length of the anti-click fade applied at cut boundaries


def fade_edges(x: np.ndarray, fs: int, fade_in: bool = True, fade_out: bool = True) -> np.ndarray:
    """Apply a short linear fade at the start and/or end of a buffer.

    Public because the mixer creates joins of its own and has to close them with the
    same ramp this module uses -- one fade policy for the whole app.
    """
    n = int(FADE_MS * fs / 1000.0) # fade length
    if n <= 0 or x.size < 2 * n:
        return x
    y = x.copy()
    ramp = np.linspace(0.0, 1.0, n) #gradually make them die out
    if fade_in:
        y[:n] *= ramp
    if fade_out:
        y[-n:] *= ramp[::-1]
    return y


def to_samples(seconds: float, fs: int, length: int) -> int:
    """Seconds -> a sample index clamped into [0, length]."""
    return int(np.clip(round(seconds * fs), 0, length))


def trim(x: np.ndarray, fs: int, start: float = 0.0, end: float = 0.0) -> np.ndarray:
    """Keep only the region between `start` and `end` seconds (end <= 0 means 'to the end')."""
    x = np.asarray(x, dtype=np.float64)
    a = to_samples(start, fs, x.size)
    b = x.size if end <= 0 else to_samples(end, fs, x.size)
    if b <= a:
        return np.zeros(0, dtype=np.float64)
    return fade_edges(x[a:b], fs)


def splice(x: np.ndarray, fs: int, start: float = 0.0, end: float = 0.0) -> np.ndarray:
    """Cut the region between `start` and `end` seconds OUT and join what remains."""
    x = np.asarray(x, dtype=np.float64)
    a = to_samples(start, fs, x.size)
    b = to_samples(end, fs, x.size)
    if b <= a:
        return x
    head = fade_edges(x[:a], fs, fade_in=False, fade_out=True)   # fade down into the cut
    tail = fade_edges(x[b:], fs, fade_in=True, fade_out=False)   # fade up out of the cut
    return np.concatenate([head, tail])