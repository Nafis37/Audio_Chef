"""
Silence Remover -- cut the pauses out of a recording
====================================================
In plain words: find every stretch quieter than a threshold that lasts long enough to be
a real pause (not the gap between two syllables), and shorten it to a short breath.

Level detector -- a short-time RMS computed with one cumulative sum
-------------------------------------------------------------------
The buffer is judged in hops of H = 10 ms.  Hop k covers samples [kH, (k+1)H) and is
judged by the RMS of a W = 20 ms window centred on it:

        c = kH + H/2                          (centre of hop k)
        P[k] = (1/W) sum_{n=c-W/2}^{c+W/2-1} x[n]^2          (mean power)
        L[k] = 10 log10 P[k]                  (dBFS; power -> 10 log, amplitude -> 20 log)

Every window sum comes from one prefix sum S[m] = sum_{n<m} x[n]^2:

        sum_{n=a}^{b-1} x[n]^2 = S[b] - S[a]          (O(1) per hop, O(N) in total)

A window is W > H wide, so a single click in the middle of a pause lifts two hops at
most; a pause is not mistaken for sound because of one sample.

Deciding what to cut
--------------------
        silent[k]  = L[k] < threshold_db
        a run      = a maximal stretch of consecutive silent hops, [kH, jH)
        cut it     if  (j - k) H >= min_silence
                   (shorter runs are the natural gaps inside and between words; cutting
                    them makes speech sound machine-gunned)

A cut run keeps `keep` seconds of its own silence, split half before and half after, so
the edit reads as a quick breath rather than a splice:

        removed = [kH + keep/2 , jH - keep/2)
        (a run at the very start or end of the file keeps only its inner half)

The kept pieces are joined with editor.fade_edges on both sides of every join -- the same
5 ms ramp as Cut & Trim -- so even a join inside a not-quite-silent pause is click-free.
"""

from __future__ import annotations

import numpy as np

from .analysis import FLOOR_DB
from .editor import fade_edges

HOP_MS = 10.0      # H: the resolution of every decision
WINDOW_MS = 20.0   # W: the RMS window judged for each hop


def hop_levels_db(x: np.ndarray, fs: int) -> tuple[np.ndarray, int]:
    """L[k] in dBFS for every hop (module docstring), and the hop length in samples."""
    x = np.asarray(x, dtype=np.float64)
    hop = max(1, int(round(HOP_MS * fs / 1000.0)))
    half = max(1, int(round(WINDOW_MS * fs / 1000.0)) // 2)
    n_hops = int(np.ceil(x.size / hop))
    if n_hops == 0:
        return np.zeros(0), hop

    prefix = np.concatenate([[0.0], np.cumsum(x * x)])     # S[m] = sum_{n<m} x[n]^2
    centres = np.arange(n_hops) * hop + hop // 2
    a = np.clip(centres - half, 0, x.size)                 # window clipped at the edges
    b = np.clip(centres + half, 0, x.size)
    power = (prefix[b] - prefix[a]) / np.maximum(b - a, 1)
    with np.errstate(divide="ignore"):
        levels = 10.0 * np.log10(power)                    # power -> dB: 10 log10
    return np.maximum(levels, FLOOR_DB), hop


def silent_runs(
    x: np.ndarray, fs: int, threshold_db: float = -40.0, min_silence: float = 0.3,
) -> list[tuple[int, int]]:
    """Every pause long enough to cut, as [start, end) sample ranges."""
    x = np.asarray(x, dtype=np.float64)
    levels, hop = hop_levels_db(x, fs)
    if levels.size == 0:
        return []

    silent = levels < threshold_db
    # Run boundaries = where the mask changes.  Padding with False on both sides makes
    # every run have exactly one rising and one falling edge.
    edges = np.diff(np.concatenate([[False], silent, [False]]).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    min_hops = max(1, int(np.ceil(min_silence * 1000.0 / HOP_MS)))
    return [(int(k * hop), int(min(j * hop, x.size)))
            for k, j in zip(starts, ends) if j - k >= min_hops]


def remove_silence(
    x: np.ndarray,
    fs: int,
    threshold_db: float = -40.0,
    min_silence: float = 0.3,
    keep: float = 0.1,
) -> np.ndarray:
    """Shorten every pause longer than `min_silence` to `keep` seconds."""
    x = np.asarray(x, dtype=np.float64)
    runs = silent_runs(x, fs, threshold_db, min_silence)
    if not runs:
        return x

    half_keep = int(round(keep * fs / 2.0))
    pieces = []
    cursor = 0                        # first sample not yet copied
    for a, b in runs:
        # A pause touching the file edge has nothing on its outer side to breathe into.
        cut_a = a if a == 0 else a + half_keep
        cut_b = b if b == x.size else b - half_keep
        if cut_b <= cut_a:
            continue
        pieces.append(x[cursor:cut_a])
        cursor = cut_b
    pieces.append(x[cursor:])

    pieces = [p for p in pieces if p.size]
    if not pieces:                    # the whole file was silence
        return np.zeros(0, dtype=np.float64)
    # Ramp both sides of every join (not the file's own first and last sample).
    last = len(pieces) - 1
    joined = [fade_edges(p, fs, fade_in=i > 0, fade_out=i < last) for i, p in enumerate(pieces)]
    return np.concatenate(joined)
