"""
Arrange -- a timeline of clips, summed
======================================
In plain words: an Arrange tab is a multitrack timeline.  Each block on it is a piece of
another tab's PROCESSED audio, placed at a time, with its own volume.  The tab's audio
is simply all the blocks added together; its recipe then runs on that mix like on any
file.

For blocks b = 1..B, each with source s_b, timeline position t_b (s), the span
[u_b, v_b) of that source (s; v_b = 0 means "to its end") and gain g_b (dB):

        c_b[m]   = 10^(g_b/20) * y_{s_b}[ round(u_b fs) + m ],  0 <= m < round((v_b - u_b) fs)
                   (y_s = source s AFTER its own recipe; the span is cut with editor.trim,
                    so both of its edges get the usual 5 ms fade)

        out[n]   = sum_b  c_b[ n - round(t_b fs) ]            (c_b[m] = 0 outside its span)

        length   = max_b ( round(t_b fs) + |c_b| )

Addition is the whole mixer: two blocks that overlap in time play together, a gap
between blocks is silence, and the order of the blocks does not matter (the sum
commutes).  Lanes exist only in the UI, to keep overlapping blocks visible.  Blocks that
overlap near full scale can sum past it; like any bake, the final buffer is then turned
down as a whole by fit_to_full_scale -- never clipped.

Each source is resolved through graph.Evaluator.processed(), so it is memoised (ten
blocks cut from one file run that file's recipe once) and cycle-checked: a block that
points back at its own arrangement is a GraphError, not a hang.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from . import editor, mixer


class ArrangeError(ValueError):
    """An arrangement that cannot be rendered (nothing on it)."""


def render_arrangement(processed, clips: Iterable[Any], fs: int) -> np.ndarray:
    """Sum every clip onto one buffer (module docstring).

    `processed(source_id)` resolves a source's processed output -- in the app that is
    graph.Evaluator.processed, injected so this module never sees the graph.
    """
    clips = list(clips)
    if not clips:
        raise ArrangeError("The arrangement is empty -- drag a file onto a lane.")

    out = np.zeros(0, dtype=np.float64)
    for clip in clips:
        whole = processed(clip.source)
        segment = editor.trim(whole, fs, start=clip.clip_start, end=clip.clip_end)
        # mix_at scales into a fresh array, so the memoised source is never written to.
        out = mixer.mix_at(out, segment, fs, position=clip.start, gain_db=clip.gain_db)
    return out
