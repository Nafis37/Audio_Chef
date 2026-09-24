"""
Arrange tabs: the timeline maths (dsp/arrange.py), the graph integration (a clip source
resolves other tabs' PROCESSED output, memoised and cycle-checked), and the request
validation in routers/process.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import HTTPException

from app.dsp import graph
from app.dsp.arrange import ArrangeError, fade_curve, render_arrangement
from app.dsp.editor import FADE_MS
from app.routers.process import (
    ClipSpec, ProcessRequest, RecipeStep, SourceSpec, TrackSpec, _validate,
)

FS = 8_000
RAMP = int(FADE_MS * FS / 1000)


def ramp_buffer(seconds: float, value: float = 0.25) -> np.ndarray:
    return np.full(int(seconds * FS), value)


def run(sources, output, files):
    out, _, report = graph.evaluate(sources, [], output, lambda f: (files[f], FS), FS,
                                    apply_master=False)
    return out, report


def test_blocks_land_sample_exact_with_their_gain():
    a, b = ramp_buffer(1.0, 0.2), ramp_buffer(0.5, 0.3)
    clips = [ClipSpec(source="a", start=0.0), ClipSpec(source="b", start=2.0, gain_db=-6.0)]
    out = render_arrangement({"a": a, "b": b}.__getitem__, clips, FS)
    assert out.shape == (int(2.5 * FS), 2)            # stereo; a centred track is unity on both
    # Middle of each block (clear of the 5 ms edge fades): its own level times its gain.
    assert out[FS // 2] == pytest.approx([0.2, 0.2])
    assert out[int(2.25 * FS)] == pytest.approx([0.3 * 10 ** (-6 / 20)] * 2)
    # The gap between them is exact silence.
    assert np.all(out[FS + RAMP: 2 * FS - RAMP] == 0.0)


def test_overlapping_blocks_on_two_tracks_sum_and_trim_takes_a_span():
    a = np.arange(2 * FS) / FS * 0.1              # a ramp: every sample says where it is from
    clips = [ClipSpec(source="a", start=0.0, clip_start=0.5, clip_end=1.0),
             ClipSpec(source="a", start=0.25, clip_start=0.5, clip_end=1.0, track=1)]
    out = render_arrangement({"a": a}.__getitem__, clips, FS)
    assert out.shape[0] == int(0.75 * FS)
    n = int(0.3 * FS)                               # inside both blocks
    expected = a[int(0.5 * FS) + n] + a[int(0.5 * FS) + n - int(0.25 * FS)]
    assert out[n] == pytest.approx([expected, expected])


def test_empty_arrangement_is_refused():
    with pytest.raises(ArrangeError):
        render_arrangement({}.__getitem__, [], FS)


def test_an_arrangement_plays_each_tabs_processed_audio_once():
    files = {"f1": ramp_buffer(1.0, 0.5)}
    sources = [
        SourceSpec(id="s1", file_id="f1",
                   recipe=[RecipeStep(op="level", params={"dc": False, "peak_db": -12})]),
        SourceSpec(id="mix", clips=[ClipSpec(source="s1", start=0.0),
                                    ClipSpec(source="s1", start=1.0)]),
    ]
    out, report = run(sources, "mix", files)
    assert out.shape == (2 * FS, 2)
    assert out[FS // 2] == pytest.approx([10 ** (-12 / 20)] * 2)   # s1's recipe ran
    assert [row["id"] for row in report["sources"]].count("s1") == 1


def test_a_block_pointing_at_its_own_arrangement_is_a_cycle():
    sources = [SourceSpec(id="mix", clips=[ClipSpec(source="mix")])]
    with pytest.raises(graph.GraphError, match="Circular"):
        run(sources, "mix", {})


def test_a_block_from_a_closed_tab_is_a_readable_error():
    sources = [SourceSpec(id="mix", clips=[ClipSpec(source="gone")])]
    with pytest.raises(graph.GraphError, match="Unknown source"):
        run(sources, "mix", {})


@pytest.mark.parametrize("spec", [
    {"id": "x"},                                              # neither
    {"id": "x", "file_id": "abc", "clips": []},               # both
])
def test_a_source_needs_exactly_one_of_file_or_clips(spec):
    request = ProcessRequest(sources=[SourceSpec(**spec)], output_source="x")
    with pytest.raises(HTTPException) as caught:
        _validate(request)
    assert caught.value.status_code == 400


# ---- multitrack: pan, volume, mute / solo ---------------------------------------------
def render(clips, tracks, files=None):
    files = files or {"a": ramp_buffer(1.0, 0.5)}
    return render_arrangement(files.__getitem__, clips, FS, tracks)


def mid(out, t=0.5):
    return out[int(t * FS)]


@pytest.mark.parametrize("pan, left, right", [
    (-1.0, np.sqrt(2.0), 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, np.sqrt(2.0)),
])
def test_pan_is_constant_power_and_unity_at_the_centre(pan, left, right):
    out = render([ClipSpec(source="a")], [TrackSpec(pan=pan)])
    assert mid(out) == pytest.approx([0.5 * left, 0.5 * right])
    assert np.sum(mid(out) ** 2) == pytest.approx(2 * 0.25)      # same power anywhere


def test_track_volume():
    out = render([ClipSpec(source="a")], [TrackSpec(volume_db=-6.0)])
    assert mid(out) == pytest.approx([0.5 * 10 ** (-6 / 20)] * 2)


def test_mute_and_solo():
    clips = [ClipSpec(source="a", track=0), ClipSpec(source="b", track=1)]
    files = {"a": ramp_buffer(1.0, 0.1), "b": ramp_buffer(1.0, 0.2)}
    both = render(clips, [], files)
    assert mid(both) == pytest.approx([0.3, 0.3])
    assert mid(render(clips, [TrackSpec(mute=True)], files)) == pytest.approx([0.2, 0.2])
    assert mid(render(clips, [TrackSpec(), TrackSpec(solo=True)], files)) == pytest.approx([0.2, 0.2])
    # Mute wins over solo; a muted track still counts toward the length.
    silent = render(clips, [TrackSpec(mute=True, solo=True), TrackSpec(mute=True)], files)
    assert silent.shape == both.shape and np.all(silent == 0.0)


# ---- fades and crossfades -------------------------------------------------------------
def test_fades_are_equal_power_ramps():
    out = render([ClipSpec(source="a", fade_in=0.25, fade_out=0.25)], [])[:, 0]
    n = int(0.25 * FS)
    assert out[:n] == pytest.approx(0.5 * fade_curve(n, rising=True) * _guard(n), abs=1e-9)
    assert out[-n:] == pytest.approx(0.5 * fade_curve(n, rising=False) * _guard(n)[::-1], abs=1e-9)
    assert mid(out) == pytest.approx(0.5)


def _guard(n):
    """editor.trim's 5 ms click guard, which every block edge also carries."""
    g = np.ones(n)
    g[:RAMP] = np.linspace(0.0, 1.0, RAMP)
    return g


def test_overlapping_blocks_on_one_track_crossfade_at_constant_power():
    rng = np.random.default_rng(0)
    files = {"a": rng.standard_normal(FS) * 0.1, "b": rng.standard_normal(FS) * 0.1}
    clips = [ClipSpec(source="a", start=0.0), ClipSpec(source="b", start=0.5)]
    out = render(clips, [], files)[:, 0]
    a, b = files["a"], files["b"]
    n = int(0.5 * FS)                                  # the overlap: 0.5 .. 1.0 s
    fin, fout = fade_curve(n, rising=True), fade_curve(n, rising=False)
    k = np.arange(RAMP, n - RAMP)                      # clear of the click guards
    assert out[n + k] == pytest.approx(a[n + k] * fout[k] + b[k] * fin[k])
    assert fin ** 2 + fout ** 2 == pytest.approx(np.ones(n))
    # Outside the overlap both clips play untouched.
    assert out[RAMP:n - 1] == pytest.approx(a[RAMP:n - 1])
    assert out[FS + 1: int(1.5 * FS) - RAMP] == pytest.approx(b[n + 1: FS - RAMP])


# ---- automation -----------------------------------------------------------------------
def test_volume_automation_overrides_the_knob_and_interpolates():
    track = TrackSpec(volume_db=-40.0, volume_env=[(0.25, 0.0), (0.75, -12.0)])
    out = render([ClipSpec(source="a")], [track])[:, 0]
    assert mid(out, 0.1) == pytest.approx(0.5)                      # held before the first
    assert mid(out, 0.5) == pytest.approx(0.5 * 10 ** (-6 / 20))    # halfway: -6 dB
    assert mid(out, 0.9) == pytest.approx(0.5 * 10 ** (-12 / 20))  # held after the last


def test_pan_automation_sweeps_left_to_right():
    track = TrackSpec(pan_env=[(0.0, -1.0), (1.0, 1.0)])
    out = render([ClipSpec(source="a")], [track])
    early, late = mid(out, 0.1), mid(out, 0.9)
    assert early[0] > early[1] and late[1] > late[0]
    assert mid(out, 0.5) == pytest.approx([0.5, 0.5], abs=1e-3)


# ---- stereo through the graph -----------------------------------------------------------
def test_an_arrangement_renders_stereo_and_its_recipe_runs_per_channel():
    files = {"f1": ramp_buffer(1.0, 0.5)}
    sources = [
        SourceSpec(id="s1", file_id="f1"),
        SourceSpec(id="mix", clips=[ClipSpec(source="s1")], tracks=[TrackSpec(pan=-1.0)],
                   recipe=[RecipeStep(op="level", params={"dc": False, "target": "none"})]),
    ]
    out, _ = run(sources, "mix", files)
    assert out.shape == (FS, 2)
    assert mid(out)[1] == pytest.approx(0.0) and mid(out)[0] > 0.5


def test_another_tab_reads_an_arrangement_as_mono():
    files = {"f1": ramp_buffer(1.0, 0.5)}
    sources = [
        SourceSpec(id="s1", file_id="f1"),
        SourceSpec(id="mix", clips=[ClipSpec(source="s1")], tracks=[TrackSpec(pan=1.0)]),
        SourceSpec(id="outer", clips=[ClipSpec(source="mix")]),
    ]
    out, _ = run(sources, "outer", files)
    # The fold of (0, 0.5 sqrt 2) is 0.5 / sqrt 2 on each side of the centred outer track.
    assert mid(out) == pytest.approx([0.5 / np.sqrt(2.0)] * 2)


def test_track_settings_are_validated():
    with pytest.raises(ValueError):
        TrackSpec(pan=2.0)
    with pytest.raises(ValueError):
        ClipSpec(source="a", track=-1)
