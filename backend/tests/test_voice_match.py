"""
Voice Match: speech analysis, calibration, conversion and graph integration.

The "voices" are synthetic (tests/speech_synth.py): a harmonic comb at a known f0 through
resonators at known formants, so every check has a known right answer.  Speaker A is a
low voice (120 Hz, table formants); speaker B is higher (210 Hz), with formants 18 %
higher and a darker tilt.

Run from backend/:   python -m pytest tests -q
"""

from __future__ import annotations

import pathlib
import time

import numpy as np
import pytest

from app.dsp import graph, voice_calibration
from app.dsp.dsp_engine import run_recipe_measured
from app.dsp.speech_analysis import _fix_octaves, analyse
from app.dsp.stft import istft, stft
from app.dsp.voice_calibration import CalibrationError, build_profile, dtw, get_profile
from app.dsp.voice_changer import shift_pitch_bins, shift_spectrum
from app.dsp.voice_match import voice_match
from app.routers.process import RecipeStep, SourceSpec

from speech_synth import Speaker, harmonic_source, random_script, resonator, speak

FS = 16_000
A = Speaker(f0=120.0, formant_scale=1.0)
B = Speaker(f0=210.0, formant_scale=1.18, tilt=1.3, rate=1.1)
SCRIPT = random_script(150, seed=1)          # ~35 s: the shared calibration script
HELD_OUT = random_script(40, seed=9)         # a sentence NOT in the calibration


def median_f0(x: np.ndarray, fs: int = FS) -> float:
    s = analyse(x, fs)
    return float(np.median(s.f0[s.voiced]))


def mean_env(x: np.ndarray, fs: int = FS) -> np.ndarray:
    s = analyse(x, fs)
    return s.env[s.voiced].mean(axis=0)


@pytest.fixture(scope="module")
def takes():
    return speak(SCRIPT, A, FS, seed=1), speak(SCRIPT, B, FS, seed=2)


@pytest.fixture(scope="module")
def profile(takes):
    return build_profile(*takes, FS)


# --------------------------------------------------------------------------- analysis
def test_constant_pitch_is_found_within_one_percent():
    x = 0.3 * harmonic_source(np.full(FS, 150.0), FS, 1.0)
    s = analyse(x, FS)
    assert s.voiced.mean() > 0.95
    assert np.median(s.f0[s.voiced]) == pytest.approx(150.0, rel=0.01)


def test_pitch_sweep_is_tracked_within_two_percent():
    truth = np.geomspace(90.0, 400.0, 2 * FS)
    s = analyse(0.3 * harmonic_source(truth, FS, 1.0), FS)
    expected = truth[np.clip((s.times * FS).astype(int), 0, truth.size - 1)]
    assert s.voiced.mean() > 0.9
    assert np.max(np.abs(s.f0[s.voiced] / expected[s.voiced] - 1.0)) < 0.02


def test_silence_and_noise_are_not_voiced():
    silent = analyse(np.zeros(FS), FS)
    assert not silent.voiced.any() and not silent.active.any()
    noise = analyse(0.1 * np.random.default_rng(0).standard_normal(FS), FS)
    assert noise.voiced.mean() < 0.02


def test_an_isolated_octave_error_is_folded_back():
    f0 = np.full(40, 200.0)
    f0[17] = 100.0                       # a sub-octave slip
    f0[25] = 400.0                       # and an octave-up one
    fixed, voiced = _fix_octaves(f0, np.ones(40, dtype=bool))
    assert fixed[17] == pytest.approx(200.0) and fixed[25] == pytest.approx(200.0)
    assert voiced.all()


def test_voiced_runs_shorter_than_three_frames_are_dropped():
    voiced = np.zeros(20, dtype=bool)
    voiced[5:7] = True
    _, kept = _fix_octaves(np.where(voiced, 150.0, 0.0), voiced)
    assert not kept.any()


def test_envelope_peaks_at_the_resonance():
    # A flat source (tilt 0), so the resonance -- not the fundamental -- is the peak.
    x = resonator(harmonic_source(np.full(FS, 120.0), FS, 0.0), 1000.0, 90.0, FS)
    s = analyse(0.3 * x / np.max(np.abs(x)), FS)
    peak_hz = s.mel_hz[np.argmax(s.env[s.voiced].mean(axis=0))]
    assert peak_hz == pytest.approx(1000.0, rel=0.15)


# --------------------------------------------------------------------------- calibration
def test_dtw_recovers_a_known_time_warp():
    rng = np.random.default_rng(4)
    n = 300
    # Smooth random features, so neighbouring frames are similar (like real envelopes).
    za = np.cumsum(rng.standard_normal((n, 32)), axis=0)
    za = (za - za.mean(0)) / za.std(0)
    # b[j] = a[w(j)] with w piecewise linear: slow (x0.7), then fast (x1.4).
    m = 320
    knots_j, knots_i = [0, 160, m - 1], [0, 112, n - 1]
    warp = np.interp(np.arange(m), knots_j, knots_i)
    zb = np.array([za[int(round(w))] for w in warp])
    classes = np.full(n, 2), np.full(m, 2)
    path, cost = dtw(za, zb, classes[0], classes[1], band=80)
    assert path[0].tolist() == [0, 0] and path[-1].tolist() == [n - 1, m - 1]
    errors = [abs(i - warp[j]) for i, j in path]
    assert np.median(errors) <= 1.0 and np.max(errors) <= 3.0
    assert cost < 0.2


def test_calibration_diagnostics(profile):
    d = profile.diagnostics
    assert d["cal_f0_hz"] == pytest.approx(120.0, rel=0.03)
    assert d["ref_f0_hz"] == pytest.approx(210.0, rel=0.03)
    assert d["cal_voiced_s"] > 20 and d["ref_voiced_s"] > 20
    assert d["align_cost"] < 0.5
    assert profile.keys.shape == profile.deltas.shape and profile.keys.shape[1] == 32


def test_a_take_over_two_minutes_is_rejected():
    long = np.zeros(int(121 * FS))
    with pytest.raises(CalibrationError, match="limited to 120 s"):
        build_profile(long, long, FS)


def test_too_little_voiced_material_is_rejected(takes):
    short = speak(SCRIPT[:12], A, FS, seed=1)                  # ~2 s
    with pytest.raises(CalibrationError, match="voiced speech"):
        build_profile(short, takes[1], FS)


def test_a_much_longer_take_is_rejected(takes):
    tripled = speak(SCRIPT * 3, B, FS, seed=2)[: int(119 * FS)]
    with pytest.raises(CalibrationError, match="same script"):
        build_profile(takes[0], tripled, FS)


def test_an_unrelated_take_does_not_align(takes):
    other = speak(random_script(150, seed=77), B, FS, seed=3)
    with pytest.raises(CalibrationError, match="do not line up"):
        build_profile(takes[0], other, FS)


def test_profile_cache_is_keyed_on_content(takes):
    voice_calibration.clear_cache()
    first = get_profile(*takes, FS)
    assert get_profile(takes[0].copy(), takes[1].copy(), FS) is first
    edited = takes[0] * 0.5
    assert get_profile(edited, takes[1], FS) is not first


# --------------------------------------------------------------------------- conversion
def test_constant_ratio_shift_equals_the_legacy_shifter():
    fixture = np.load(pathlib.Path(__file__).parent / "fixtures" / "shift_pitch_legacy.npz")
    assert np.array_equal(shift_pitch_bins(fixture["x"], 5.0), fixture["up"])
    assert np.array_equal(shift_pitch_bins(fixture["x"], -7.0), fixture["down"])
    spec = stft(fixture["x"])
    ratios = np.full(spec.shape[0], 2.0 ** (5.0 / 12.0))
    y = istft(shift_spectrum(spec, ratios), length=fixture["x"].size)
    assert np.array_equal(y, fixture["up"])


def test_full_conversion_moves_pitch_and_envelope_toward_the_reference(profile):
    new = speak(HELD_OUT, A, FS, seed=9)
    target = speak(HELD_OUT, B, FS, seed=9)
    pitch_only, _ = voice_match(new, FS, profile, 1.0, 0.0)
    full, diag = voice_match(new, FS, profile, 1.0, 0.7)

    for y in (pitch_only, full):
        assert y.size == new.size and np.all(np.isfinite(y))
        assert median_f0(y) == pytest.approx(median_f0(target), rel=0.05)

    dist = lambda y: float(np.sqrt(np.mean((mean_env(y) - mean_env(target)) ** 2)))  # noqa: E731
    assert dist(full) < dist(pitch_only) < dist(new)
    assert diag["shift_st"] == pytest.approx(12 * np.log2(210 / 120), abs=0.6)
    assert 0.0 <= diag["reduced_fraction"] <= 1.0


def test_pitch_only_keeps_the_formants_where_they_were(profile):
    # The E_in - E_sh term: a big upward shift must not become a chipmunk.  Compare with
    # speaker A's own formants spoken at B's pitch.
    new = speak(HELD_OUT, A, FS, seed=9)
    ideal = speak(HELD_OUT, Speaker(210.0, 1.0), FS, seed=9)
    naive = shift_pitch_bins(new, 12 * np.log2(210 / 120))
    y, _ = voice_match(new, FS, profile, 1.0, 0.0)
    err = lambda z: float(np.sqrt(np.mean((mean_env(z) - mean_env(ideal)) ** 2)))  # noqa: E731
    assert err(y) < 0.6 * err(naive)


def test_timbre_only_leaves_the_pitch_alone(profile):
    new = speak(HELD_OUT, A, FS, seed=9)
    y, diag = voice_match(new, FS, profile, 0.0, 0.7)
    assert median_f0(y) == pytest.approx(median_f0(new), rel=0.01)
    assert diag["shift_st"] == 0.0
    assert not np.allclose(y, new)


def test_zero_strengths_are_an_exact_bypass(profile):
    new = speak(HELD_OUT, A, FS, seed=9)
    y, _ = voice_match(new, FS, profile, 0.0, 0.0)
    assert np.array_equal(y, new)


def test_transitions_stay_smooth(profile):
    new = speak(HELD_OUT, A, FS, seed=9)
    y, _ = voice_match(new, FS, profile, 1.0, 0.7)
    # A click is a jump far larger than the signal's own steepest slope (which a pitch
    # shift up by 1.75x may raise by about that factor).
    assert np.max(np.abs(np.diff(y))) < 3.0 * np.max(np.abs(np.diff(new)))
    assert np.max(np.abs(y)) < 3.0 * np.max(np.abs(new))


def test_silence_stays_silent(profile):
    new = np.concatenate([np.zeros(FS // 2), speak(HELD_OUT, A, FS, seed=9), np.zeros(FS // 2)])
    y, _ = voice_match(new, FS, profile, 1.0, 1.0)
    assert np.max(np.abs(y[: FS // 4])) < 1e-3 and np.max(np.abs(y[-FS // 4:])) < 1e-3


# --------------------------------------------------------------------------- the graph
def _step(op, bypass=False, **params):
    return RecipeStep(op=op, bypass=bypass, params=params)


def _vm(cal="mine", ref="theirs", **params):
    return _step("voice_match", calibration_source=cal, reference_source=ref, **params)


def _project(takes, new_recipe, cal_recipe=(), extra=()):
    files = {"f_mine": takes[0], "f_theirs": takes[1], "f_new": speak(HELD_OUT, A, FS, seed=9)}
    sources = [
        SourceSpec(id="mine", file_id="f_mine", recipe=list(cal_recipe)),
        SourceSpec(id="theirs", file_id="f_theirs"),
        SourceSpec(id="new", file_id="f_new", recipe=list(new_recipe)),
        *extra,
    ]
    return sources, (lambda file_id: (files[file_id], FS)), files


def _render(sources, loader, output="new", master=(), fs=FS):
    return graph.evaluate(sources, list(master), output, loader, fs)


def test_three_source_recipe_converts_and_reports(takes):
    sources, loader, files = _project(takes, [_vm()])
    out, raw, report = _render(sources, loader)
    assert out.size == files["f_new"].size and np.all(np.isfinite(out))
    assert median_f0(out) == pytest.approx(210.0, rel=0.06)
    (row,) = report["voice_match"]
    assert row["id"] == "new"
    for key in ("cal_voiced_s", "ref_voiced_s", "cal_f0_hz", "ref_f0_hz", "align_cost",
                "reduced_fraction", "shift_st"):
        assert key in row
    assert np.array_equal(raw, files["f_new"])


def test_voice_match_default_is_audible(takes):
    # test_every_default_is_audible skips ops that need other sources; this is its
    # voice_match half, with the references supplied.
    sources, loader, files = _project(takes, [_vm()])
    out, _, _ = _render(sources, loader)
    x = files["f_new"]
    assert np.sqrt(np.mean((out - x) ** 2)) / np.sqrt(np.mean(x ** 2)) > 0.1


def test_a_deleted_reference_is_reported(takes):
    sources, loader, _ = _project(takes, [_vm(ref="gone")])
    with pytest.raises(graph.GraphError, match="Unknown source 'gone'"):
        _render(sources, loader)


@pytest.mark.parametrize("cal, ref, message", [
    ("", "theirs", "no calibration take"),
    ("mine", "", "no reference take"),
    ("mine", "mine", "two different"),
    ("new", "theirs", "its own calibration"),
])
def test_bad_reference_choices_are_rejected(takes, cal, ref, message):
    sources, loader, _ = _project(takes, [_vm(cal=cal, ref=ref)])
    with pytest.raises(graph.GraphError, match=message):
        _render(sources, loader)


def test_a_cycle_is_named(takes):
    # "mine" converts itself using "new", which uses "mine": new -> mine -> new.
    cal_recipe = [_step("voice_match", calibration_source="new", reference_source="theirs")]
    sources, loader, _ = _project(takes, [_vm()], cal_recipe=cal_recipe)
    with pytest.raises(graph.GraphError, match="Circular reference: new -> mine -> new"):
        _render(sources, loader)


def test_a_bypassed_card_is_free(takes):
    sources, loader, files = _project(takes, [_vm(ref="gone", bypass=True)])
    out, _, report = _render(sources, loader)
    assert np.array_equal(out, files["f_new"])
    assert "voice_match" not in report


def test_zero_strengths_bake_to_the_input(takes):
    sources, loader, files = _project(takes, [_vm(pitch_strength=0, timbre_strength=0)])
    out, _, _ = _render(sources, loader)
    assert np.array_equal(out, files["f_new"])


def test_master_chain_use_gives_a_readable_error(takes):
    y = speak(HELD_OUT, A, FS, seed=9)
    with pytest.raises(ValueError, match="only works inside a source's recipe"):
        run_recipe_measured(y, FS, [{"op": "voice_match"}])


def test_editing_a_calibration_chain_changes_the_output(takes):
    sources, loader, _ = _project(takes, [_vm()])
    before, _, _ = _render(sources, loader)
    eq = _step("equalizer", bass_gain=-12.0, treble_gain=12.0)
    sources, loader, _ = _project(takes, [_vm()], cal_recipe=[eq])
    after, _, _ = _render(sources, loader)
    assert before.size == after.size and not np.allclose(before, after)


def test_same_pair_twice_is_calibrated_once(takes, monkeypatch):
    voice_calibration.clear_cache()
    calls = []
    real = voice_calibration.build_profile
    monkeypatch.setattr(voice_calibration, "build_profile",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    two = [_vm(), _vm(pitch_strength=50)]
    sources, loader, _ = _project(takes, two)
    _, _, report = _render(sources, loader)
    assert len(calls) == 1 and len(report["voice_match"]) == 2


def test_another_project_rate_works(takes):
    fs = 22_050
    files = {"f_mine": speak(SCRIPT, A, fs, seed=1), "f_theirs": speak(SCRIPT, B, fs, seed=2),
             "f_new": speak(HELD_OUT, A, fs, seed=9)}
    sources = [SourceSpec(id="mine", file_id="f_mine"),
               SourceSpec(id="theirs", file_id="f_theirs"),
               SourceSpec(id="new", file_id="f_new", recipe=[_vm()])]
    out, _, _ = graph.evaluate(sources, [], "new", lambda f: (files[f], fs), fs)
    assert out.size == files["f_new"].size
    assert median_f0(out, fs) == pytest.approx(210.0, rel=0.06)


def test_cached_bakes_are_fast(takes):
    voice_calibration.clear_cache()
    sources, loader, _ = _project(takes, [_vm()])
    t0 = time.perf_counter()
    _render(sources, loader)
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    _render(sources, loader)
    second = time.perf_counter() - t0
    assert second < first and second < 5.0
