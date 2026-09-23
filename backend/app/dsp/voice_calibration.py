"""
Voice calibration -- aligning two readings of one script, and the paired codebook
=================================================================================
In plain words
--------------
You and the target speaker each read the SAME script once.  This module lines the two
readings up in time (same words at the same moment), then remembers, for every voiced
frame of yours, how the target's spectral shape differed at that point.  New speech is
converted by looking up the calibration frames that sound most like it and applying their
remembered difference.  It is an experimental paired-codebook design in the spirit of
classical codebook voice conversion (L. M. Arslan and D. Talkin, "Voice conversion by
codebook mapping of line spectral frequencies and excitation spectrum", Eurospeech 1997)
-- not a reproduction of that paper or its results.

Input: two SpeechFrames (speech_analysis.py): per 10 ms frame, a voicing class and a
32-point mean-removed log envelope e[t] (dB).

1. Features for alignment
-------------------------
Each speaker's envelopes are z-normalised with THEIR OWN statistics over active frames

        z[t] = (e[t] - mean_active(e)) / std_active(e)          (per dimension)

which removes the fixed part of the timbre difference (a brighter voice, a different
microphone) so the alignment compares WHAT was said rather than WHO said it.  Frame
distance is the RMS difference, plus a penalty for a voicing-class mismatch
(silence / unvoiced / voiced):

        c(i, j) = sqrt( mean_d (z_a[i,d] - z_b[j,d])^2 )  +  LAMBDA * [class_a(i) != class_b(j)]
        c(i, j) = 0                                             if both frames are silence

Before aligning, leading/trailing silence is trimmed and every internal pause is capped
at MAX_PAUSE frames, since pause lengths are the least repeatable part of a reading.

2. Dynamic time warping -- banded, slope-constrained, row-vectorised
--------------------------------------------------------------------
Find the monotone path from (0,0) to (N-1,M-1) through the cost grid with the least
total cost.  The step pattern (symmetric, slope limited to [1/2, 2]):

        D(i,j) = c(i,j) + min(  D(i-1, j-1),                       (1,1)
                                D(i-1, j-2) + c(i, j-1),            (1,2)
                                D(i-2, j-1) + c(i-1, j) )           (2,1)

Every term refers to rows i-1 and i-2 only -- never to D(i, j-1) -- so one whole row is
a handful of vectorised NumPy operations rather than M Python iterations.  (The classic
(0,1) step would need D(i, j-1), a sequential dependency inside the row.)  The slope
limit also stops the degenerate "stay on one frame for a second" paths.

Sakoe-Chiba band: only cells with |j - i (M-1)/(N-1)| <= W are evaluated, with
W = max(2 s, 15 % of the longer take) in frames.  Backpointers are int8 (0/1/2 above),
stored only inside the band: N x (2W+1) bytes -- a few MB even for two 2-minute takes.

Alignment quality: the mean c over the expanded path, divided by the mean c of random
frame pairs.  ~0.5 is a good alignment; near 1 the path is no better than chance, so the
takes are not the same script (or are too noisy) and are rejected.

3. The codebook
---------------
Each voiced frame i of YOUR take that the path pairs with a voiced frame of the target
becomes one entry:

        key[i]   = z_a[i]                                          (your descriptor)
        delta[i] = mean over paired j of ( e_b[j] - e_a[i] )       (dB, 32 points)

The key is normalised with YOUR statistics, so new speech from you is normalised the
same way at lookup.  delta is not normalised: it is the actual dB difference to apply.

4. Lookup -- inverse-distance k-NN, with a reliability weight
-------------------------------------------------------------
For a new frame q:  d_n = RMS distance to its K = 8 nearest keys, and

        delta^(q) = sum_n w_n delta_n / sum_n w_n,      w_n = 1 / (d_n + eps)

The distances are computed in batches of BATCH query rows (so memory stays bounded) and
the 8 nearest found with np.argpartition (O(C) rather than a full sort).

A frame unlike anything in the calibration gets an unreliable average.  Its mean
neighbour distance dbar is compared with d_ref, the typical within-calibration distance
(median, over a subsample of keys, of the mean 8-NN distance EXCLUDING the key's own
+-EXCLUDE neighbouring frames, which are near-copies of it):

        rel = 1                                   dbar <= 1.5 d_ref
        rel = (3 d_ref - dbar) / (1.5 d_ref)      in between  (linear)
        rel = 0                                   dbar >= 3 d_ref

and the timbre change applied to that frame is scaled by rel (voice_match.py).

5. Pitch statistics
-------------------
Mean and standard deviation of log f0 over voiced frames, per speaker.  voice_match.py
maps one Gaussian onto the other.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .speech_analysis import SpeechFrames, analyse

MAX_TAKE_S = 120.0
MIN_VOICED_S = 5.0
MIN_ENTRIES = 200
MAX_PAUSE = 15                # frames kept of any internal pause (150 ms)
LAMBDA = 1.0                  # voicing-class mismatch penalty
BAND_MIN_S = 2.0
BAND_FRACTION = 0.15
MAX_ALIGN_COST = 0.6          # path cost / random-pair cost above this -> reject
K = 8
BATCH = 512
EXCLUDE = 10                  # +-frames ignored when measuring d_ref
D_REF_SAMPLE = 2000
STD_FLOOR_DB = 0.5

SILENT, UNVOICED, VOICED = 0, 1, 2


class CalibrationError(ValueError):
    """An unusable calibration pair.  A ValueError, so the router answers 400 with it."""


@dataclass
class VoiceProfile:
    keys: np.ndarray              # (C, 32) your z-normalised envelopes
    deltas: np.ndarray            # (C, 32) target - you, dB
    norm_mean: np.ndarray         # (32,) your active-frame mean
    norm_std: np.ndarray          # (32,)
    d_ref: float
    mu_cal: float                 # mean log f0 (natural log), you
    sd_cal: float
    mu_ref: float                 # ... target
    sd_ref: float
    mel_hz: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------- features
def _classes(frames: SpeechFrames) -> np.ndarray:
    return np.where(frames.voiced, VOICED, np.where(frames.active, UNVOICED, SILENT))


def _norm_stats(frames: SpeechFrames) -> tuple[np.ndarray, np.ndarray]:
    env = frames.env[frames.active] if np.any(frames.active) else frames.env
    return env.mean(axis=0), np.maximum(env.std(axis=0), STD_FLOOR_DB)


def _condense(cls: np.ndarray) -> np.ndarray:
    """Indices kept after trimming edge silence and capping internal pauses (section 1)."""
    active = np.flatnonzero(cls != SILENT)
    if active.size == 0:
        return active
    idx = np.arange(active[0], active[-1] + 1)
    keep = np.ones(idx.size, dtype=bool)
    silent = cls[idx] == SILENT
    edges = np.flatnonzero(np.diff(np.concatenate([[0], silent.astype(np.int8), [0]])))
    for a, b in zip(edges[0::2], edges[1::2]):
        if b - a > MAX_PAUSE:
            keep[a + MAX_PAUSE:b] = False
    return idx[keep]


def _pair_cost(za: np.ndarray, zb: np.ndarray, ca: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """c(i, j) for aligned arrays of pairs (section 1)."""
    d = np.sqrt(np.mean((za - zb) ** 2, axis=-1))
    d = d + LAMBDA * (ca != cb)
    return np.where((ca == SILENT) & (cb == SILENT), 0.0, d)


# ----------------------------------------------------------------------- DTW
def dtw(za: np.ndarray, zb: np.ndarray, ca: np.ndarray, cb: np.ndarray,
        band: int) -> tuple[np.ndarray, float]:
    """Banded, slope-constrained DTW (section 2).

    Returns (path as an (P, 2) array of (i, j), mean local cost along it).
    Raises CalibrationError when no path fits inside the band and slope limits.
    """
    n, m = za.shape[0], zb.shape[0]
    if n < 2 or m < 2:
        raise CalibrationError("A calibration take is too short to align.")
    centre = np.arange(n) * (m - 1) / (n - 1)
    lo = np.clip(np.floor(centre - band).astype(int), 0, m - 1)
    hi = np.clip(np.ceil(centre + band).astype(int), 0, m - 1)
    width = int(np.max(hi - lo)) + 1

    INF = np.inf
    back = np.full((n, width), -1, dtype=np.int8)
    d_prev2 = np.full(m + 2, INF)          # D(i-2, .), shifted by 2 so j-2 never wraps
    d_prev = np.full(m + 2, INF)           # D(i-1, .)
    c_prev = np.full(m + 2, INF)           # c(i-1, .)

    for i in range(n):
        a, b = lo[i], hi[i] + 1
        cols = np.arange(a, b)
        c_row = np.full(m + 2, INF)
        c_row[a + 2:b + 2] = _pair_cost(za[i][None, :], zb[a:b], ca[i], cb[a:b])
        d_row = np.full(m + 2, INF)
        if i == 0:
            if a == 0:
                d_row[2] = c_row[2]        # the path starts at (0, 0)
        else:
            j2 = cols + 2                  # position of column j in the shifted arrays
            diag = d_prev[j2 - 1]
            wide = d_prev[j2 - 2] + c_row[j2 - 1]
            tall = d_prev2[j2 - 1] + c_prev[j2]
            options = np.stack([diag, wide, tall])
            choice = np.argmin(options, axis=0)
            best = options[choice, np.arange(cols.size)]
            d_row[j2] = best + c_row[j2]
            back[i, :cols.size] = np.where(np.isfinite(best), choice, -1)
        d_prev2, d_prev, c_prev = d_prev, d_row, c_row

    if not np.isfinite(d_prev[m - 1 + 2]):
        raise CalibrationError(
            "The two calibration takes could not be lined up -- they should be the same "
            "script, read at a similar pace."
        )

    # Backtrack, expanding the (1,2) and (2,1) steps into the cells they pass through.
    path = []
    i, j = n - 1, m - 1
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        step = back[i, j - lo[i]]
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            path.append((i, j - 1))
            i, j = i - 1, j - 2
        elif step == 2:
            path.append((i - 1, j))
            i, j = i - 2, j - 1
        else:                               # unreachable for a finite end cell
            raise CalibrationError("Calibration alignment failed.")
    path = np.array(path[::-1])
    cost = _pair_cost(za[path[:, 0]], zb[path[:, 1]], ca[path[:, 0]], cb[path[:, 1]])
    return path, float(np.mean(cost))


# ----------------------------------------------------------------------- k-NN
def _knn(keys: np.ndarray, queries: np.ndarray, k: int,
         exclude: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(indices, RMS distances) of the k nearest keys to each query, batched.

    `exclude`: for query row r, a (lo, hi) key-index range that may not be chosen.
    """
    k = min(k, keys.shape[0])
    key_sq = np.sum(keys * keys, axis=1)
    dim = keys.shape[1]
    idx_out = np.zeros((queries.shape[0], k), dtype=int)
    dist_out = np.zeros((queries.shape[0], k))
    for start in range(0, queries.shape[0], BATCH):
        q = queries[start:start + BATCH]
        # ||q - key||^2 = ||q||^2 - 2 q.key + ||key||^2
        # errstate: numpy's Accelerate backend on macOS raises spurious FP warnings from
        # matmul on perfectly finite inputs; the result is checked finite below.
        with np.errstate(all="ignore"):
            d2 = np.sum(q * q, axis=1)[:, None] - 2.0 * q @ keys.T + key_sq[None, :]
        d2 = np.where(np.isfinite(d2), d2, np.inf)
        d2 = np.maximum(d2, 0.0) / dim
        if exclude is not None:
            ex = exclude[start:start + BATCH]
            cols = np.arange(keys.shape[0])[None, :]
            d2 = np.where((cols >= ex[:, :1]) & (cols < ex[:, 1:]), np.inf, d2)
        part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        idx_out[start:start + q.shape[0]] = part
        dist_out[start:start + q.shape[0]] = np.sqrt(np.take_along_axis(d2, part, axis=1))
    return idx_out, dist_out


def predict(profile: VoiceProfile, env: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Envelope change (dB, (Q, 32)) and reliability (Q,) for new frames (section 4)."""
    queries = (env - profile.norm_mean) / profile.norm_std
    idx, dist = _knn(profile.keys, queries, K)
    weights = 1.0 / (dist + 0.05 * profile.d_ref)
    weights /= weights.sum(axis=1, keepdims=True)
    delta = np.einsum("qk,qkd->qd", weights, profile.deltas[idx])
    dbar = dist.mean(axis=1)
    rel = np.clip((3.0 * profile.d_ref - dbar) / (1.5 * profile.d_ref), 0.0, 1.0)
    return delta, rel


# ----------------------------------------------------------------------- the profile
def _log_f0_stats(frames: SpeechFrames) -> tuple[float, float]:
    lf = np.log(frames.f0[frames.voiced])
    return float(np.mean(lf)), float(np.std(lf))


def build_profile(cal: np.ndarray, ref: np.ndarray, fs: int) -> VoiceProfile:
    """Analyse, align and pair the two calibration takes (sections 1-5)."""
    for name, take in (("Your calibration take", cal), ("The reference take", ref)):
        if take.size / fs > MAX_TAKE_S + 1e-9:
            raise CalibrationError(
                f"{name} is {take.size / fs:.0f} s long -- calibration takes are limited to "
                f"{MAX_TAKE_S:.0f} s. Trim it with a Cut & Trim card on that source."
            )

    fa, fb = analyse(cal, fs), analyse(ref, fs)
    for name, frames in (("Your calibration take", fa), ("The reference take", fb)):
        if frames.voiced_seconds < MIN_VOICED_S:
            raise CalibrationError(
                f"{name} has only {frames.voiced_seconds:.1f} s of voiced speech -- at least "
                f"{MIN_VOICED_S:.0f} s is needed. Record the full script (30-60 s)."
            )

    ca_all, cb_all = _classes(fa), _classes(fb)
    ia, ib = _condense(ca_all), _condense(cb_all)
    ratio = ib.size / max(ia.size, 1)
    if not 0.5 <= ratio <= 2.0:
        raise CalibrationError(
            f"One calibration take is {max(ratio, 1 / ratio):.1f}x as long as the other once "
            "pauses are ignored -- are they readings of the same script?"
        )

    mean_a, std_a = _norm_stats(fa)
    mean_b, std_b = _norm_stats(fb)
    za_all = (fa.env - mean_a) / std_a
    zb_all = (fb.env - mean_b) / std_b
    za, zb, ca, cb = za_all[ia], zb_all[ib], ca_all[ia], cb_all[ib]

    band = max(int(round(BAND_MIN_S / (fa.hop / fs))),
               int(round(BAND_FRACTION * max(ia.size, ib.size))))
    path, path_cost = dtw(za, zb, ca, cb, band)

    # Chance level: random pairs of active frames.
    rng = np.random.default_rng(0)
    ra = rng.choice(np.flatnonzero(ca != SILENT), 4000)
    rb = rng.choice(np.flatnonzero(cb != SILENT), 4000)
    chance = float(np.mean(_pair_cost(za[ra], zb[rb], ca[ra], cb[rb])))
    align_cost = path_cost / max(chance, 1e-9)
    if align_cost > MAX_ALIGN_COST:
        raise CalibrationError(
            f"The two calibration takes do not line up (alignment cost {align_cost:.2f}, "
            f"limit {MAX_ALIGN_COST:.2f}) -- they should be the same script, recorded "
            "cleanly with similar microphone placement."
        )

    # Codebook: voiced-voiced pairs, averaged per frame of yours (section 3).
    pa, pb = ia[path[:, 0]], ib[path[:, 1]]
    both = fa.voiced[pa] & fb.voiced[pb]
    pa, pb = pa[both], pb[both]
    diff = fb.env[pb] - fa.env[pa]
    frames_a, inverse = np.unique(pa, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    deltas = np.zeros((frames_a.size, diff.shape[1]))
    np.add.at(deltas, inverse, diff)
    deltas /= counts[:, None]
    if frames_a.size < MIN_ENTRIES:
        raise CalibrationError(
            "Too few voiced moments line up between the two takes -- record both readings "
            "of the full script, clearly, in a quiet room."
        )
    keys = za_all[frames_a]

    # d_ref: within-calibration neighbour distance, ignoring each key's own neighbours.
    sample = np.sort(rng.choice(frames_a.size, min(D_REF_SAMPLE, frames_a.size), replace=False))
    lo = np.searchsorted(frames_a, frames_a[sample] - EXCLUDE)
    hi = np.searchsorted(frames_a, frames_a[sample] + EXCLUDE, side="right")
    _, sdist = _knn(keys, keys[sample], K, exclude=np.stack([lo, hi], axis=1))
    finite = np.where(np.isfinite(sdist), sdist, np.nan)
    d_ref = float(np.nanmedian(np.nanmean(finite, axis=1)))
    d_ref = max(d_ref, 1e-3)

    mu_a, sd_a = _log_f0_stats(fa)
    mu_b, sd_b = _log_f0_stats(fb)
    semis = 12.0 / np.log(2.0)
    diagnostics = {
        "cal_voiced_s": round(fa.voiced_seconds, 1),
        "ref_voiced_s": round(fb.voiced_seconds, 1),
        "cal_f0_hz": round(float(np.exp(mu_a)), 1),
        "ref_f0_hz": round(float(np.exp(mu_b)), 1),
        "cal_f0_spread_st": round(float(sd_a * semis), 2),
        "ref_f0_spread_st": round(float(sd_b * semis), 2),
        "align_cost": round(float(align_cost), 3),
        "codebook": int(frames_a.size),
    }
    return VoiceProfile(keys=keys, deltas=deltas, norm_mean=mean_a, norm_std=std_a,
                        d_ref=d_ref, mu_cal=mu_a, sd_cal=sd_a, mu_ref=mu_b, sd_ref=sd_b,
                        mel_hz=fa.mel_hz, diagnostics=diagnostics)


# ----------------------------------------------------------------------- cache
# Auto-Bake re-renders on every slider move, and each render builds a fresh graph
# Evaluator.  Keying on the CONTENT of the two processed takes (not their source ids)
# means a strength change reuses the alignment, while any edit to either calibration
# chain produces different samples, a different key, and a fresh profile.
_CACHE_SIZE = 2
_cache: "OrderedDict[tuple[str, str, int], VoiceProfile]" = OrderedDict()
_lock = threading.Lock()


def _digest(x: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()


def get_profile(cal: np.ndarray, ref: np.ndarray, fs: int) -> VoiceProfile:
    """build_profile() through a small content-keyed LRU."""
    key = (_digest(cal), _digest(ref), int(fs))
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit
    profile = build_profile(cal, ref, fs)
    with _lock:
        _cache[key] = profile
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return profile


def clear_cache() -> None:
    with _lock:
        _cache.clear()
