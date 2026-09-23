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
from app.dsp.arrange import ArrangeError, render_arrangement
from app.dsp.editor import FADE_MS
from app.routers.process import ClipSpec, ProcessRequest, RecipeStep, SourceSpec, _validate

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
    assert out.size == int(2.5 * FS)
    # Middle of each block (clear of the 5 ms edge fades): its own level times its gain.
    assert out[FS // 2] == pytest.approx(0.2)
    assert out[int(2.25 * FS)] == pytest.approx(0.3 * 10 ** (-6 / 20))
    # The gap between them is exact silence.
    assert np.all(out[FS + RAMP: 2 * FS - RAMP] == 0.0)


def test_overlapping_blocks_sum_and_trim_takes_a_span():
    a = np.arange(2 * FS) / FS * 0.1              # a ramp: every sample says where it is from
    clips = [ClipSpec(source="a", start=0.0, clip_start=0.5, clip_end=1.0),
             ClipSpec(source="a", start=0.25, clip_start=0.5, clip_end=1.0)]
    out = render_arrangement({"a": a}.__getitem__, clips, FS)
    assert out.size == int(0.75 * FS)
    n = int(0.3 * FS)                               # inside both blocks
    expected = a[int(0.5 * FS) + n] + a[int(0.5 * FS) + n - int(0.25 * FS)]
    assert out[n] == pytest.approx(expected)


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
    assert out.size == 2 * FS
    assert out[FS // 2] == pytest.approx(10 ** (-12 / 20))       # s1's recipe ran
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
