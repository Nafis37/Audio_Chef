"""
Sanity checks for every DSP block.

Each test feeds a synthetic signal whose correct answer is known in closed form (a sine
of known frequency, a recursion evaluated one sample at a time, a length we can count)
and checks the hand-written implementation against it.

Run from backend/:   python -m pytest tests -q
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from app.dsp import analysis, mixer
from app.dsp.compressor import block_envelope, compressor, envelope_follower
from app.dsp.dsp_engine import (
    HEADROOM, OPERATIONS, _coerce, operations_schema, run_recipe, run_recipe_measured,
)
from app.dsp.echo_reverb import echo, reverb
from app.dsp.declip import clipped_runs, declip
from app.dsp.editor import reverse, splice, trim
from app.dsp.level import level, remove_dc
from app.dsp.silence import remove_silence, silent_runs
from app.dsp.eq import equalizer, high_shelf_coefficients, low_shelf_coefficients
from app.dsp.filters import butterworth_qs, cutoff_filter, filter_response, sections
from app.dsp.noise import noise_reduce
from app.dsp.spectrogram import DB_FLOOR, ROWS, spectrogram_image
from app.dsp.speed_pitch import speed_pitch
from app.dsp.stft import istft, principal_argument, stft
from app.dsp.voice_changer import voice_changer

FS = 16_000


def sine(freq: float, seconds: float = 1.0, amp: float = 0.5, fs: int = FS) -> np.ndarray:
    n = np.arange(int(seconds * fs))
    return amp * np.sin(2.0 * np.pi * freq * n / fs)


def dominant_hz(x: np.ndarray, fs: int = FS) -> float:
    """Frequency of the biggest rfft bin of the middle half (edges skipped)."""
    mid = x[x.size // 4: 3 * x.size // 4]
    spectrum = np.abs(np.fft.rfft(mid * np.hanning(mid.size)))
    return float(np.fft.rfftfreq(mid.size, 1.0 / fs)[np.argmax(spectrum)])


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


# --------------------------------------------------------------------------- stft.py
def test_stft_round_trip_is_identity_in_the_interior():
    x = np.random.default_rng(0).standard_normal(FS)
    y = istft(stft(x), length=x.size)
    # The edges are deliberately faded by the divisor floor; the interior is exact.
    np.testing.assert_allclose(y[2048:-2048], x[2048:-2048], atol=1e-10)


def test_principal_argument_wraps_into_minus_pi_pi():
    p = np.linspace(-20.0, 20.0, 10_001)
    w = principal_argument(p)
    assert np.all(w >= -np.pi - 1e-12) and np.all(w <= np.pi + 1e-12)
    # Same angle, only whole turns removed.
    np.testing.assert_allclose(np.cos(w), np.cos(p), atol=1e-12)


# --------------------------------------------------------------------------- noise.py
def test_noise_remover_lowers_the_noise_and_keeps_the_length():
    rng = np.random.default_rng(1)
    noise = 0.05 * rng.standard_normal(2 * FS)
    tone = sine(440.0, 2.0)
    # Noise only until 0.7 s; the profile is 0-0.5 s.  The gap matters: noise.py picks
    # profile frames by START time, so a frame starting at 0.49 s reaches 0.49 s + N/fs
    # and would sweep a tone that begins right at 0.5 s into the "noise" estimate.
    tone[: int(0.7 * FS)] = 0.0
    x = tone + noise

    y = noise_reduce(x, FS, amount=2.0, floor=0.02, profile="region",
                     noise_start=0.0, noise_end=0.5)

    assert y.size == x.size
    quiet = slice(int(0.1 * FS), int(0.4 * FS))
    assert rms(y[quiet]) < 0.5 * rms(x[quiet])           # at least 6 dB less noise
    loud = slice(FS, int(1.5 * FS))
    assert rms(y[loud]) == pytest.approx(rms(tone[loud]), rel=0.15)   # the tone survives


def test_auto_noise_profile_needs_no_marked_region():
    # Speech-like: a tone that is ON from t = 0 (so "the first half second" is not noise)
    # but pauses for 30 % of the time.  The auto profile finds the noise in the pauses.
    rng = np.random.default_rng(4)
    seconds = 3.0
    tone = sine(440.0, seconds)
    t = np.arange(tone.size) / FS
    gate = (t % 1.0) < 0.7                               # on 0.0-0.7, 1.0-1.7, 2.0-2.7 s
    x = tone * gate + 0.05 * rng.standard_normal(tone.size)

    y = noise_reduce(x, FS)                              # defaults: profile="auto"

    pause = slice(int(1.75 * FS), int(1.95 * FS))
    assert rms(y[pause]) < 0.5 * rms(x[pause])           # >= 6 dB less noise in the gap
    on = slice(int(1.1 * FS), int(1.6 * FS))
    assert rms(y[on]) == pytest.approx(rms(tone[on]), rel=0.15)   # the tone survives


# --------------------------------------------------------------------------- eq.py
def _response_db(b: np.ndarray, a: np.ndarray, f: float, fs: int = FS) -> float:
    """|H(e^{jw})| in dB, evaluated straight from the coefficients."""
    z = np.exp(-1j * 2.0 * np.pi * f / fs * np.arange(3))
    return float(20.0 * np.log10(abs(np.dot(b, z) / np.dot(a, z))))


@pytest.mark.parametrize("gain_db", [9.0, -12.0])
def test_shelves_reach_their_gain_on_their_side_only(gain_db):
    b, a = low_shelf_coefficients(FS, 500.0, gain_db)
    assert _response_db(b, a, 20.0) == pytest.approx(gain_db, abs=0.2)     # far below
    assert _response_db(b, a, 7000.0) == pytest.approx(0.0, abs=0.2)       # far above
    assert _response_db(b, a, 500.0) == pytest.approx(gain_db / 2, abs=0.2)  # half at f0

    b, a = high_shelf_coefficients(FS, 2000.0, gain_db)
    assert _response_db(b, a, 7900.0) == pytest.approx(gain_db, abs=0.3)
    assert _response_db(b, a, 30.0) == pytest.approx(0.0, abs=0.2)
    assert _response_db(b, a, 2000.0) == pytest.approx(gain_db / 2, abs=0.2)


@pytest.mark.parametrize("gain_db", [6.0, -9.0])
def test_peaking_band_hits_its_gain_at_the_centre(gain_db):
    x = sine(1000.0, 1.0)
    y = equalizer(x, FS, mid_gain=gain_db, mid_freq=1000.0, q=1.0)
    settled = slice(FS // 2, None)                       # past the filter's transient
    measured = 20.0 * np.log10(rms(y[settled]) / rms(x[settled]))
    assert measured == pytest.approx(gain_db, abs=0.1)


def test_flat_equalizer_is_the_identity():
    x = sine(300.0)
    np.testing.assert_array_equal(equalizer(x, FS), x)


# --------------------------------------------------------------------------- echo_reverb.py
# --------------------------------------------------------------------------- filters.py
def _cascade_db(mode: str, fc: float, order: str, f: float, q: float = 2.0) -> float:
    """Response of the whole cascade: the sections' dB simply add."""
    return sum(_response_db(b, a, f) for b, a in sections(FS, mode, fc, order, q))


def test_butterworth_qs_are_the_pole_pairs():
    assert butterworth_qs(2) == pytest.approx([0.7071], abs=1e-4)
    assert butterworth_qs(4) == pytest.approx([0.5412, 1.3066], abs=1e-4)


@pytest.mark.parametrize("order", ["2", "4", "6", "8"])
def test_lowpass_is_3db_down_at_the_cutoff_and_falls_6n_db_per_octave(order):
    n = int(order)
    assert _cascade_db("lowpass", 1000.0, order, 1000.0) == pytest.approx(-3.01, abs=0.1)
    assert _cascade_db("lowpass", 1000.0, order, 100.0) == pytest.approx(0.0, abs=0.1)
    assert _cascade_db("lowpass", 1000.0, order, 2000.0) <= -6.0 * n


@pytest.mark.parametrize("order", ["2", "4", "6", "8"])
def test_highpass_mirrors_the_lowpass(order):
    n = int(order)
    assert _cascade_db("highpass", 1000.0, order, 1000.0) == pytest.approx(-3.01, abs=0.1)
    assert _cascade_db("highpass", 1000.0, order, 10000.0) == pytest.approx(0.0, abs=0.1)
    assert _cascade_db("highpass", 1000.0, order, 500.0) <= -6.0 * n


def test_bandpass_peaks_at_0db_and_notch_removes_its_centre():
    assert _cascade_db("bandpass", 1500.0, "4", 1500.0, q=1.0) == pytest.approx(0.0, abs=0.01)
    assert _cascade_db("bandpass", 1500.0, "4", 150.0, q=1.0) < -15.0
    assert _cascade_db("notch", 1000.0, "4", 1000.0, q=10.0) < -60.0
    assert _cascade_db("notch", 1000.0, "4", 4000.0, q=10.0) == pytest.approx(0.0, abs=0.1)


def test_lowpass_removes_the_high_tone_and_keeps_the_low_one():
    x = sine(300.0, amp=0.3) + sine(4000.0, amp=0.3)
    y = cutoff_filter(x, FS, mode="lowpass", cutoff=500.0, order="4")
    assert dominant_hz(y) == pytest.approx(300.0, abs=2.0)
    spectrum = lambda s: np.abs(np.fft.rfft(s[FS // 4:]))      # past the start-up transient
    k = int(round(4000.0 * (FS - FS // 4) / FS))
    drop_db = 20.0 * np.log10(spectrum(y)[k] / spectrum(x)[k])
    assert drop_db < -40.0


def test_response_curve_is_the_filter_that_runs():
    freqs, db = filter_response(FS, "lowpass", 800.0, "6", 2.0)
    assert np.all(np.isfinite(db)) and freqs[0] == pytest.approx(20.0)
    at = np.argmin(np.abs(freqs - 800.0))
    assert db[at] == pytest.approx(_cascade_db("lowpass", 800.0, "6", freqs[at]), abs=1e-6)
    _, notch = filter_response(FS, "notch", 1000.0, "4", 10.0)
    assert np.all(np.isfinite(notch))                    # the exact zero is floored, not -inf


def test_filter_response_route_clamps_and_returns_json():
    from app.routers.process import filter_response as route

    body = route(mode="nonsense", cutoff=1e9, order="3", q=-5.0, fs=FS)
    assert body["cutoff"] == 20000.0                     # clamped to the slider's max
    assert len(body["freqs"]) == len(body["db"]) and all(np.isfinite(body["db"]))


def _comb_feedback_reference(x: np.ndarray, d: int, g: float) -> np.ndarray:
    """y[n] = x[n] + g*y[n-D], one sample at a time: the readable definition."""
    y = np.zeros_like(x)
    for n in range(x.size):
        y[n] = x[n] + (g * y[n - d] if n >= d else 0.0)
    return y


def test_echo_convolution_equals_the_feedback_recursion():
    x = np.random.default_rng(2).standard_normal(3000)
    d, g = 137, 0.6
    y = echo(x, FS, delay=d / FS, feedback=g, mix=1.0)
    assert y.size == x.size
    np.testing.assert_allclose(y, _comb_feedback_reference(x, d, g), atol=1e-9)


def test_reverb_is_deterministic_and_length_preserving():
    x = sine(220.0, 0.5)
    a = reverb(x, FS, room_size=0.8, decay=1.5, mix=0.5)
    b = reverb(x, FS, room_size=0.8, decay=1.5, mix=0.5)
    assert a.size == x.size and np.all(np.isfinite(a))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(reverb(x, FS, mix=0.0), x)


# --------------------------------------------------------------------------- editor.py
def test_trim_and_splice_lengths():
    x = sine(440.0, 2.0)
    assert trim(x, FS, start=0.5, end=1.5).size == FS
    assert trim(x, FS, start=0.5, end=0.0).size == int(1.5 * FS)     # end=0 -> to the end
    assert splice(x, FS, start=0.5, end=1.0).size == int(1.5 * FS)
    assert trim(x, FS, start=1.5, end=0.5).size == 0                  # empty selection


# --------------------------------------------------------------------------- speed_pitch.py
def test_speed_changes_duration():
    x = sine(440.0, 1.0)
    # The phase vocoder's output is (frames-1)*H_s + N: it keeps a full frame of tail
    # instead of scaling it, so it overshoots N/2 by up to one frame (N = 2048).
    stretched = speed_pitch(x, FS, speed=2.0).size
    assert x.size // 2 <= stretched <= x.size // 2 + 2048
    assert speed_pitch(x, FS, speed=2.0, preserve_pitch=False).size == x.size // 2


def test_octave_up_keeps_length_and_doubles_frequency():
    x = sine(440.0, 1.0)
    y = speed_pitch(x, FS, semitones=12.0)
    assert y.size == x.size
    assert dominant_hz(y) == pytest.approx(880.0, rel=0.03)


# --------------------------------------------------------------------------- compressor.py
def test_compressor_reduces_by_the_static_curve():
    x = sine(440.0, 1.0, amp=1.0)
    threshold, ratio = -20.0, 4.0
    y = compressor(x, FS, threshold=threshold, ratio=ratio, knee=0.0, auto_makeup=False)

    settled = slice(FS // 2, None)
    env = block_envelope(x, FS, 5.0, 150.0)[settled]
    over = 20.0 * np.log10(np.mean(env)) - threshold
    expected = -(1.0 - 1.0 / ratio) * over
    measured = 20.0 * np.log10(rms(y[settled]) / rms(x[settled]))
    assert measured == pytest.approx(expected, abs=1.5)   # envelope ripple -> loose


@pytest.mark.parametrize("freq", [100.0, 440.0, 3000.0])
def test_block_envelope_agrees_with_the_per_sample_follower(freq):
    # The block version reads the true peak; the per-sample one sags ~1 dB below it on a
    # sine because it partly releases inside every cycle (compressor.py docstring).
    x = sine(freq, 1.0, amp=1.0)
    settled = slice(FS // 2, None)
    per_sample = 20.0 * np.log10(np.mean(envelope_follower(x, FS, 10.0, 100.0)[settled]))
    per_block = 20.0 * np.log10(np.mean(block_envelope(x, FS, 10.0, 100.0)[settled]))
    assert per_block == pytest.approx(per_sample, abs=1.5)


def test_auto_makeup_restores_the_input_peak_and_lifts_the_quiet_part():
    loud = sine(440.0, 0.5, amp=0.9)
    quiet = sine(440.0, 0.5, amp=0.05)
    x = np.concatenate([loud, quiet])
    y = compressor(x, FS, threshold=-30.0, ratio=6.0)
    # Once the attack has caught up, the loud half sits back at its original peak ...
    settled_loud = slice(int(0.25 * FS), int(0.5 * FS))
    peak_ratio_db = 20 * np.log10(np.max(np.abs(y[settled_loud])) / 0.9)
    assert peak_ratio_db == pytest.approx(0.0, abs=1.0)
    # ... and the quiet half came UP.
    tail = slice(int(0.75 * FS), None)
    assert rms(y[tail]) > 1.5 * rms(x[tail])             # the quiet half came UP


def test_compressor_leaves_quiet_signal_alone():
    x = sine(440.0, amp=0.01)                              # -40 dBFS, far below -20
    np.testing.assert_allclose(compressor(x, FS, threshold=-20.0, knee=6.0), x)


# --------------------------------------------------------------------------- voice_changer.py
@pytest.mark.parametrize("semitones", [12.0, -12.0, 7.0])
def test_voice_pitch_mode_moves_the_frequency_not_the_length(semitones):
    x = sine(440.0, 1.0)
    y = voice_changer(x, FS, mode="pitch", semitones=semitones)
    assert y.size == x.size
    assert dominant_hz(y) == pytest.approx(440.0 * 2 ** (semitones / 12), rel=0.03)


@pytest.mark.parametrize("mode", ["robot", "whisper"])
def test_voice_robot_and_whisper_are_deterministic(mode):
    x = sine(300.0, 0.5)
    a = voice_changer(x, FS, mode=mode)
    b = voice_changer(x, FS, mode=mode)
    assert a.size == x.size and np.all(np.isfinite(a))
    np.testing.assert_array_equal(a, b)


def test_robot_buzzes_at_robot_freq():
    # White noise has no pitch of its own; the robot must impose one at fs / H.
    x = 0.3 * np.random.default_rng(3).standard_normal(FS)
    y = voice_changer(x, FS, mode="robot", robot_freq=100.0)
    spectrum = np.abs(np.fft.rfft(y[FS // 4: 3 * FS // 4]))
    freqs = np.fft.rfftfreq(FS // 2, 1.0 / FS)
    # Energy piles up on the harmonics of 100 Hz: the 100 Hz bin beats its neighbours.
    at = spectrum[np.argmin(abs(freqs - 100.0))]
    around = spectrum[(freqs > 60) & (freqs < 90)].mean()
    assert at > 5 * around


def test_voice_mix_zero_is_dry():
    x = sine(440.0, 0.5)
    np.testing.assert_allclose(voice_changer(x, FS, mode="whisper", mix=0.0), x)


# --------------------------------------------------------------------------- mixer.py
def test_mixer_lengths():
    base, clip = np.ones(FS), np.ones(FS // 2)
    assert mixer.mix_at(base, clip, FS, position=0.25).size == FS
    assert mixer.mix_at(base, clip, FS, position=0.75).size == FS + FS // 4   # grows
    assert mixer.insert_at(base, clip, FS, position=0.5).size == FS + FS // 2
    assert mixer.append_to(base, clip, FS).size == FS + FS // 2


# --------------------------------------------------------------------------- analysis.py
def test_measure_a_sine():
    m = analysis.measure(sine(1000.0, 1.0, amp=0.5), FS)
    assert m["rms"] == pytest.approx(0.5 / np.sqrt(2), rel=1e-3)
    assert m["crest_db"] == pytest.approx(3.01, abs=0.02)
    assert m["zcr"] == pytest.approx(2000.0, rel=0.01)
    assert m["centroid_hz"] == pytest.approx(1000.0, rel=0.05)


# --------------------------------------------------------------------------- dsp_engine.py
def test_bypass_skips_the_step():
    x = sine(440.0)
    recipe = [{"op": "equalizer", "bypass": True, "params": {"mid_gain": 12.0}}]
    y, report = run_recipe_measured(x, FS, recipe)
    np.testing.assert_array_equal(y, x)
    assert report["steps_bypassed"] == 1 and report["steps_applied"] == 0


def test_unknown_op_is_rejected():
    with pytest.raises(ValueError):
        run_recipe(sine(440.0), FS, [{"op": "does_not_exist"}])


def test_params_are_clamped_not_rejected():
    ratio = next(p for p in next(o for o in OPERATIONS if o["id"] == "compressor")["params"]
                 if p["name"] == "ratio")
    assert _coerce(ratio, 999) == ratio["max"]
    assert _coerce(ratio, "nonsense") == ratio["default"]
    assert _coerce(ratio, float("nan")) == ratio["default"]


def test_output_is_turned_down_not_clipped_and_the_overshoot_reported():
    x = sine(1000.0, amp=0.9)
    recipe = [{"op": "equalizer", "params": {"bass_gain": 0, "treble_gain": 0, "mid_gain": 24.0}}]
    y, report = run_recipe_measured(x, FS, recipe)
    assert np.max(np.abs(y)) == pytest.approx(HEADROOM)
    assert report["pre_clip_peak"] > 1.0 and report["clipped"] > 0
    assert report["normalised_db"] < 0.0
    # One constant gain, not a clamp: the shape of the boosted signal is untouched.
    raw, _ = run_recipe_measured(x, FS, recipe, clip=False)
    np.testing.assert_allclose(y, raw * HEADROOM / np.max(np.abs(raw)))


def test_percent_params_are_scaled_for_the_handler():
    mix = next(p for p in next(o for o in OPERATIONS if o["id"] == "echo")["params"]
               if p["name"] == "mix")
    assert _coerce(mix, 60) == pytest.approx(0.6)
    assert _coerce(mix, 500) == pytest.approx(1.0)       # clamped THEN scaled


def test_every_default_is_audible():
    # Dropping a card in with no tweaks must change the sound.  (The editor, assemble and
    # voice_match need a selection or other files first, so they are exempt -- voice_match
    # is covered with its references in test_voice_match.py.  declip and silence_remover
    # repair something this signal does not have -- flat tops, long pauses -- and are
    # tested on signals that do, below.)
    # Speech-like on purpose: a loud half and a quiet half (a compressor with auto-makeup
    # is rightly a no-op on a constant level) over a little hiss (for the noise remover).
    x = np.concatenate([sine(300.0, 0.5, amp=0.4), sine(300.0, 0.5, amp=0.04)])
    x = x + 0.02 * np.random.default_rng(5).standard_normal(x.size)
    for op in OPERATIONS:
        if op["id"] in {"editor", "assemble", "voice_match", "declip", "silence_remover"} or op.get("hidden"):
            continue
        y = run_recipe(x, FS, [{"op": op["id"]}])
        n = min(x.size, y.size)
        changed = rms(y[:n] - x[:n]) / rms(x[:n]) if y.size == x.size else 1.0
        assert changed > 0.1, op["id"]


def test_schema_extras_point_at_real_params():
    for op in OPERATIONS:
        params = {p["name"]: p for p in op["params"]}
        for p in op["params"]:
            for other, values in (p.get("show_when") or {}).items():
                assert other in params, (op["id"], p["name"])
                assert set(values) <= set(params[other]["options"]), (op["id"], p["name"])
        for name, settings in (op.get("quick") or {}).items():
            for key, value in settings.items():
                assert key in params, (op["id"], name, key)
                spec = params[key]
                if spec["type"] == "float":
                    assert spec["min"] <= value <= spec["max"], (op["id"], name, key)
                elif spec["type"] == "enum":
                    assert value in spec["options"], (op["id"], name, key)


def test_old_echo_reverb_recipes_still_bake():
    x = sine(440.0, 0.5)
    y = run_recipe(x, FS, [{"op": "echo_reverb", "params": {"mode": "reverb", "mix": 0.3}}])
    assert y.size == x.size and not np.allclose(y, x)


# --------------------------------------------------------------------------- spectrogram.py
def test_spectrogram_puts_a_full_scale_sine_at_0_db_in_the_right_row():
    image, meta = spectrogram_image(sine(1000.0, 1.0, amp=1.0), FS)
    assert image.shape == (ROWS, meta["cols"]) and image.dtype == np.uint8
    column = image[:, image.shape[1] // 2].astype(float)
    row = int(np.argmax(column))
    # Row 0 is the TOP (highest frequency): undo that to find the row's centre frequency.
    r = ROWS - 1 - row
    centre = meta["f_min"] * (meta["f_max"] / meta["f_min"]) ** (r / (ROWS - 1))
    assert centre == pytest.approx(1000.0, rel=0.06)
    level_db = DB_FLOOR + column[row] / 255.0 * (0.0 - DB_FLOOR)
    # Band power averages the sine's main lobe with its neighbours, so it reads a few dB
    # under the peak bin -- but on an ABSOLUTE scale, not normalised per picture.
    assert -8.0 < level_db <= 0.5


def test_spectrogram_of_silence_is_black():
    image, _ = spectrogram_image(np.zeros(FS), FS)
    assert image.max() == 0


def test_schema_is_json_safe():
    import json
    for op in operations_schema():
        assert "handler" not in op
    json.dumps(operations_schema())
    assert any(op["id"] == "voice_changer" for op in operations_schema())


# --------------------------------------------------------------------------- the hard rule
BANNED_MODULES = {"librosa", "noisereduce", "pedalboard"}
BANNED_SCIPY = {"iirpeak", "iirfilter", "resample", "stft"}


def test_no_banned_dsp_library_is_used():
    """Every effect is written by hand: no DSP library imports, no scipy shortcuts."""
    app = pathlib.Path(__file__).resolve().parents[1] / "app"
    for path in app.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
                assert not roots & BANNED_MODULES, path
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in BANNED_MODULES, path
                if node.module.startswith("scipy"):
                    assert not {a.name for a in node.names} & BANNED_SCIPY, path
            elif isinstance(node, ast.Attribute) and node.attr in BANNED_SCIPY:
                # signal.stft(...) / scipy.signal.resample(...) -- our own stft is a
                # bare name, so an attribute access with this name is a scipy call.
                base = node.value
                name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
                assert name != "signal", path


# ---- Reverse, Silence Remover, Level & DC, De-clip ----------------------------------

def test_reverse_twice_is_the_identity():
    x = np.random.default_rng(1).standard_normal(FS)
    np.testing.assert_array_equal(reverse(reverse(x, FS), FS), x)


def test_reverse_keeps_the_magnitude_spectrum():
    # x[-n] <-> conj X for real x: only the phase changes.
    x = sine(440.0, 0.5) + sine(1250.0, 0.5, amp=0.2)
    np.testing.assert_allclose(np.abs(np.fft.rfft(reverse(x, FS))),
                               np.abs(np.fft.rfft(x)), atol=1e-8)


def test_reverse_selection_leaves_the_rest_alone():
    x = np.linspace(-0.5, 0.5, FS)
    y = reverse(x, FS, start=0.25, end=0.5)
    a, b = int(0.25 * FS), int(0.5 * FS)
    ramp = int(0.005 * FS)
    np.testing.assert_array_equal(y[: a - ramp], x[: a - ramp])
    np.testing.assert_array_equal(y[b + ramp:], x[b + ramp:])
    np.testing.assert_array_equal(y[a + ramp: b - ramp], x[a:b][::-1][ramp:-ramp])


def test_silence_remover_shortens_each_pause_to_keep():
    tone = sine(300.0, 0.5)
    gap = np.zeros(FS)                           # 1 s of digital silence
    x = np.concatenate([tone, gap, tone, gap, tone])
    assert len(silent_runs(x, FS)) == 2
    y = remove_silence(x, FS, threshold_db=-40.0, min_silence=0.3, keep=0.1)
    # Each 1 s gap shrinks to ~0.1 s (the RMS window blurs each edge by ~one hop).
    assert abs(y.size / FS - (1.5 + 2 * 0.1)) < 0.05
    # Every join is faded: no sample-to-sample step larger than the tone's own.
    assert np.max(np.abs(np.diff(y))) <= np.max(np.abs(np.diff(tone))) + 1e-9


def test_silence_remover_leaves_short_gaps():
    tone = sine(300.0, 0.3)
    x = np.concatenate([tone, np.zeros(int(0.1 * FS)), tone])
    np.testing.assert_array_equal(remove_silence(x, FS, min_silence=0.3), x)


def test_dc_blocker_removes_an_offset():
    x = sine(200.0, 1.0) + 0.2
    y = remove_dc(x, FS)
    assert abs(np.mean(y)) < 1e-3
    # Away from DC the blocker passes: the 200 Hz tone keeps its level.
    assert abs(rms(y) - rms(sine(200.0, 1.0))) < 0.01


def test_level_hits_its_targets():
    x = 0.1 * sine(440.0, 0.5, amp=1.0)
    y = level(x, FS, dc=False, target="peak", peak_db=-6.0)
    assert np.max(np.abs(y)) == pytest.approx(10 ** (-6 / 20), rel=1e-9)
    y = level(x, FS, dc=False, target="rms", rms_db=-20.0)
    assert rms(y) == pytest.approx(0.1, rel=1e-9)


def test_declip_restores_flattened_peaks():
    clean = sine(200.0, 0.5, amp=1.4)            # 1.4 peak, sliced at 1.0
    clipped = np.clip(clean, -1.0, 1.0)
    assert clipped_runs(clipped)
    y = declip(clipped, FS)
    assert np.max(np.abs(y)) > 1.2                # the lost tops are redrawn
    assert rms(y - clean) < 0.5 * rms(clipped - clean)
