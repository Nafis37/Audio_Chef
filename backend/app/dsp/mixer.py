"""
Assembly -- placing one buffer into another
===========================================
Three ways to combine a clip with the buffer flowing through a recipe:

    mix_at(base, clip, position)     -> overlay:  out[p:p+n] += clip   (both play together)
    insert_at(base, clip, position)  -> splice in: base[:p] + clip + base[p:]
    append_to(base, clip)            -> base + clip

All three are length-GROWING rather than length-preserving.  Echo and reverb truncate
their tails back to the input length because they are effects applied to a fixed take;
assembly is the opposite -- the clip is content, and silently cutting it off at the base's
length would eat audio the user explicitly placed.

Fades: this module closes only the joins IT creates in the base.  The clip arrives already
faded at both edges (graph.ClipContext runs it through editor.trim), so fading it again
here would double-ramp every boundary.
"""

from __future__ import annotations

import numpy as np

from .editor import fade_edges, to_samples


def _gain(db: float) -> float:
    """dB -> linear amplitude multiplier."""
    return 10.0 ** (float(db) / 20.0)


def _prepare(clip: np.ndarray, gain_db: float) -> np.ndarray:
    """The clip as a fresh, gain-scaled float64 array.

    Always a new array: mix_at accumulates in place, and the caller's clip may be a view
    into a memoised source buffer that other steps still have to read.
    """
    scaled = np.asarray(clip, dtype=np.float64) * _gain(gain_db)
    return np.ascontiguousarray(scaled)


def _offset(position: float, fs: int) -> int:
    """Seconds -> a sample offset, clamped at zero but NOT at any upper bound.

    Unlike editor.to_samples this has no `length` ceiling on purpose: an overlay placed
    past the end of the base is a real edit (a clip that starts after a gap of silence),
    and clamping it would quietly turn "at 30 s" into "at the end".
    """
    return max(0, int(round(float(position) * fs)))


def mix_at(
    base: np.ndarray,
    clip: np.ndarray,
    fs: int,
    position: float = 0.0,
    gain_db: float = 0.0,
) -> np.ndarray:
    """Overlay `clip` on top of `base`, starting `position` seconds in.

    The output is sized to the UNION of the two, never truncated to the base: silence is
    the identity for addition, so a long music bed under a short voice line keeps its
    tail instead of being cut at the voice's last sample.
    """
    base = np.asarray(base, dtype=np.float64)
    if np.size(clip) == 0:
        return base

    scaled = _prepare(clip, gain_db)
    a = _offset(position, fs)

    out = np.zeros(max(base.size, a + scaled.size), dtype=np.float64)
    out[: base.size] = base
    out[a : a + scaled.size] += scaled      # a gap before `a` stays the zeros above
    return out


def insert_at(
    base: np.ndarray,
    clip: np.ndarray,
    fs: int,
    position: float = 0.0,
    gain_db: float = 0.0,
) -> np.ndarray:
    """Cut `base` open at `position` seconds and drop `clip` into the gap.

    A position past the end of the base degenerates to an append, which is what the
    clamp in to_samples() gives us for free.
    """
    base = np.asarray(base, dtype=np.float64)
    if np.size(clip) == 0:
        return base

    scaled = _prepare(clip, gain_db)
    a = to_samples(position, fs, base.size)

    # The same head/tail treatment editor.splice uses: ramp down into the new boundary
    # and back up out of it, so the two discontinuities do not click.
    head = fade_edges(base[:a], fs, fade_in=False, fade_out=True)
    tail = fade_edges(base[a:], fs, fade_in=True, fade_out=False)
    return np.concatenate([head, scaled, tail])


def append_to(
    base: np.ndarray,
    clip: np.ndarray,
    fs: int,
    gain_db: float = 0.0,
) -> np.ndarray:
    """Put `clip` after `base`.  Position is meaningless here and is ignored."""
    base = np.asarray(base, dtype=np.float64)
    if np.size(clip) == 0:
        return base

    scaled = _prepare(clip, gain_db)
    # Only the base's trailing edge is a new join -- its start is untouched.
    return np.concatenate([fade_edges(base, fs, fade_in=False, fade_out=True), scaled])
