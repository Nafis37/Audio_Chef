"""
Arrange -- a multitrack timeline: clips on tracks, faded, mixed down to stereo
==============================================================================
In plain words: an Arrange tab is a multitrack timeline.  Each block on it is a piece of
another tab's PROCESSED audio, placed at a time on a track, with its own volume and
fades.  Each track has a volume, a pan, mute and solo, and optional automation curves
for volume and pan.  The tab's audio is every track, panned into a stereo mix; its
recipe then runs on that mix (on each channel) like on any file.

1. A clip
---------
For a clip b with source s_b, timeline position t_b (s), span [u_b, v_b) of that source
(v_b = 0 means "to its end"), gain g_b (dB) and fade lengths F_in, F_out (s):

        c_b[m] = 10^(g_b/20) * w_b[m] * y_{s_b}[ round(u_b fs) + m ]
                 (y_s = source s AFTER its own recipe; the span is cut with editor.trim,
                  so both of its edges also get the usual 5 ms click guard)

The fade gain w_b is equal-power: over the first F_in and last F_out seconds

        fade in   w = sin( pi/2 * m / M_in )           fade out   w = cos( pi/2 * k / M_out )

(k counts from the start of the fade-out).  Equal-power, not linear, because a
crossfade between two unrelated sounds should keep the LOUDNESS steady, and loudness
follows power: sin^2 + cos^2 = 1 at every instant of the overlap.

2. Crossfades
-------------
Two clips on the SAME track that overlap crossfade automatically: over the overlap the
earlier clip fades out and the later one fades in (each fade is the longer of the
drawn one and the overlap).  On different tracks, overlapping clips simply play together.

3. A track
----------
        bus_k[n]  = sum of its clips c_b[n - round(t_b fs)]
        gain_k(t) = volume automation (dB, linear between points, held flat before the
                    first and after the last), or the volume knob if the curve is empty
        pan_k(t)  = the same for pan, -1 (left) .. +1 (right)

Constant-power pan, scaled so that the CENTRE is unity on both channels (a centred
track plays exactly as the old mono mix did):

        theta = (pan + 1) * pi / 4          L = sqrt(2) cos(theta)    R = sqrt(2) sin(theta)

Hard left gives L = sqrt(2) (+3 dB), R = 0: the track's total power, L^2 + R^2 = 2,
is the same at every pan position, so moving a sound across the stereo field does not
change how loud it seems.

Mute and solo: if any track is soloed, only the soloed tracks play; a muted track never
plays (mute wins over solo, as in most DAWs).

4. The mix
----------
        out[n, 0] = sum_k L_k(n) gain_k(n) bus_k[n]        out[n, 1] = ... R_k(n) ...
        length    = max_b ( round(t_b fs) + |c_b| )          (muted clips included, so
                                                              muting never moves the end)

Blocks that overlap near full scale can sum past it; like any bake, the final buffer is
then turned down as a whole by fit_to_full_scale -- never clipped.

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


def _attr(obj: Any, name: str, default: Any) -> Any:
    value = getattr(obj, name, default)
    return default if value is None else value


def fade_curve(n: int, rising: bool) -> np.ndarray:
    """Equal-power ramp of n samples: sin 0 -> 1 when rising, cos 1 -> 0 when falling."""
    phase = (np.arange(n) + 0.5) / max(n, 1) * (np.pi / 2.0)
    return np.sin(phase) if rising else np.cos(phase)


def automation(points: Iterable[Any], fallback: float, n: int, fs: int) -> np.ndarray | float:
    """A control curve sampled at every sample, or the constant knob value if empty.

    `points` are (time s, value) pairs; between points the value moves in a straight
    line, and it is held flat before the first and after the last.
    """
    pts = sorted((float(t), float(v)) for t, v in points or [])
    if not pts:
        return float(fallback)
    times = np.array([t for t, _ in pts])
    values = np.array([v for _, v in pts])
    return np.interp(np.arange(n) / fs, times, values)


def pan_gains(pan: np.ndarray | float) -> tuple[np.ndarray | float, np.ndarray | float]:
    """(left, right) gains for pan -1..+1: constant power, unity at the centre."""
    theta = (np.clip(pan, -1.0, 1.0) + 1.0) * (np.pi / 4.0)
    return np.sqrt(2.0) * np.cos(theta), np.sqrt(2.0) * np.sin(theta)


def _clip_fades(clips: list[Any], lengths: list[int], fs: int) -> list[tuple[int, int]]:
    """(fade-in, fade-out) in samples per clip: drawn fades, lengthened to cover overlaps
    with the previous clip on the same track (module docstring, section 2)."""
    fades = []
    for clip, length in zip(clips, lengths):
        fin = int(round(max(0.0, _attr(clip, "fade_in", 0.0)) * fs))
        fout = int(round(max(0.0, _attr(clip, "fade_out", 0.0)) * fs))
        fades.append([fin, fout])

    by_track: dict[int, list[int]] = {}
    for i, clip in enumerate(clips):
        by_track.setdefault(int(_attr(clip, "track", 0)), []).append(i)
    for members in by_track.values():
        members.sort(key=lambda i: mixer._offset(clips[i].start, fs))
        latest = None                  # the clip that ends last so far on this track
        for i in members:
            start = mixer._offset(clips[i].start, fs)
            if latest is not None:
                end = mixer._offset(clips[latest].start, fs) + lengths[latest]
                overlap = min(end, start + lengths[i]) - start
                if overlap > 0:
                    fades[i][0] = max(fades[i][0], overlap)
                    # The earlier clip fades out over the overlap, which ends at ITS end
                    # only when the later clip outlasts it.
                    if end <= start + lengths[i]:
                        fades[latest][1] = max(fades[latest][1], overlap)
            if latest is None or start + lengths[i] > (
                mixer._offset(clips[latest].start, fs) + lengths[latest]
            ):
                latest = i

    # A fade can never be longer than its clip, and the two may not cross.
    out = []
    for (fin, fout), length in zip(fades, lengths):
        fin = min(fin, length)
        fout = min(fout, length - fin) if fin + fout > length else fout
        out.append((fin, fout))
    return out


def render_arrangement(
    processed, clips: Iterable[Any], fs: int, tracks: Iterable[Any] | None = None
) -> np.ndarray:
    """Mix every clip, track by track, into a (frames, 2) stereo buffer (docstring).

    `processed(source_id)` resolves a source's processed output -- in the app that is
    graph.Evaluator.processed, injected so this module never sees the graph.
    """
    clips = list(clips)
    tracks = list(tracks or [])
    if not clips:
        raise ArrangeError("The arrangement is empty -- drag a file onto a lane.")

    segments = []
    for clip in clips:
        whole = processed(clip.source)
        segment = editor.trim(whole, fs, start=clip.clip_start, end=clip.clip_end)
        segments.append(mixer._prepare(segment, _attr(clip, "gain_db", 0.0)))
    fades = _clip_fades(clips, [s.size for s in segments], fs)

    total = max(mixer._offset(c.start, fs) + s.size for c, s in zip(clips, segments))
    track_ids = sorted({int(_attr(c, "track", 0)) for c in clips})

    def setting(k: int, name: str, default: Any) -> Any:
        return _attr(tracks[k], name, default) if 0 <= k < len(tracks) else default

    soloing = any(setting(k, "solo", False) for k in track_ids)
    out = np.zeros((total, 2), dtype=np.float64)
    for k in track_ids:
        if setting(k, "mute", False) or (soloing and not setting(k, "solo", False)):
            continue
        bus = np.zeros(total, dtype=np.float64)
        for clip, segment, (fin, fout) in zip(clips, segments, fades):
            if int(_attr(clip, "track", 0)) != k:
                continue
            if fin:
                segment[:fin] *= fade_curve(fin, rising=True)
            if fout:
                segment[segment.size - fout:] *= fade_curve(fout, rising=False)
            p = mixer._offset(clip.start, fs)
            bus[p:p + segment.size] += segment

        gain_db = automation(setting(k, "volume_env", []), setting(k, "volume_db", 0.0), total, fs)
        pan = automation(setting(k, "pan_env", []), setting(k, "pan", 0.0), total, fs)
        bus *= 10.0 ** (np.asarray(gain_db) / 20.0)
        left, right = pan_gains(pan)
        out[:, 0] += left * bus
        out[:, 1] += right * bus
    return out
