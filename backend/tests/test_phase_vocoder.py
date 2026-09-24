"""
Phase Vocoder card: easy / accurate pitch shift and the gender swaps (voice_shift.py).

Checked on the source-filter "speakers" of speech_synth.py, whose f0 and formants are
known: pitch must move by the asked amount (or land on the preset's target), and the
formants must move -- or stay -- independently of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.dsp.voice_shift import GENDER_TARGETS, pitch_formant_shift
from app.dsp.dsp_engine import run_recipe
from app.dsp.speech_analysis import analyse

from speech_synth import Speaker, random_script, speak

FS = 16_000
VOICE = Speaker(f0=140.0, formant_scale=1.0)


@pytest.fixture(scope="module")
def speech():
    return speak(random_script(30, seed=3), VOICE, FS, seed=3)


def median_f0(x):
    s = analyse(x, FS)
    return float(np.median(s.f0[s.voiced]))


def envelope_centroid(x):
    """Magnitude-weighted mean frequency of the average voiced envelope, 300-4000 Hz."""
    s = analyse(x, FS)
    env = 10.0 ** (s.env[s.voiced].mean(axis=0) / 20.0)
    from app.dsp.speech_analysis import mel_points
    hz = mel_points(FS, env.size)
    band = (hz > 300) & (hz < 4000)
    return float(np.sum(hz[band] * env[band]) / np.sum(env[band]))


def test_pitch_moves_formants_stay(speech):
    y = pitch_formant_shift(speech, FS, semitones=4.0, formant=1.0)
    ratio = median_f0(y) / median_f0(speech)
    assert ratio == pytest.approx(2 ** (4 / 12), rel=0.04)
    assert envelope_centroid(y) == pytest.approx(envelope_centroid(speech), rel=0.05)


def test_formants_move_pitch_stays(speech):
    y = pitch_formant_shift(speech, FS, semitones=0.0, formant=1.15)
    assert median_f0(y) == pytest.approx(median_f0(speech), rel=0.03)
    assert envelope_centroid(y) > 1.04 * envelope_centroid(speech)


def bake(x, mode, **params):
    return run_recipe(x, FS, [{"op": "voice_changer", "params": {"mode": mode, **params}}])


def test_easy_pitch_moves_pitch_and_formants(speech):
    y = bake(speech, "easy_pitch", shift=4)
    assert y.size == speech.size
    assert median_f0(y) / median_f0(speech) == pytest.approx(2 ** (4 / 12), rel=0.04)
    assert envelope_centroid(y) > 1.08 * envelope_centroid(speech)


def test_accurate_pitch_moves_pitch_keeps_formants(speech):
    y = bake(speech, "accurate_pitch", shift=4)
    assert y.size == speech.size
    assert median_f0(y) / median_f0(speech) == pytest.approx(2 ** (4 / 12), rel=0.04)
    assert envelope_centroid(y) == pytest.approx(envelope_centroid(speech), rel=0.05)


@pytest.mark.parametrize("mode", list(GENDER_TARGETS))
def test_low_and_high_voices_land_on_the_same_target(mode):
    # The point of a target: whatever voice goes in, it ends up near the preset's pitch.
    target = GENDER_TARGETS[mode]
    for f0 in (120.0, 200.0):          # (the shift is clamped to an octave either way)
        x = speak(random_script(30, seed=3), Speaker(f0, 1.0), FS, seed=3)
        y = bake(x, mode)
        assert np.all(np.isfinite(y)) and y.size == x.size
        assert median_f0(y) == pytest.approx(target.f0_hz, rel=0.05), f0


def test_male_to_female_brightens_a_low_voice():
    x = speak(random_script(30, seed=3), Speaker(120.0, 1.0), FS, seed=3)
    assert envelope_centroid(bake(x, "male_to_female")) > 1.06 * envelope_centroid(x)


def test_female_to_male_darkens_a_high_voice():
    x = speak(random_script(30, seed=3), Speaker(210.0, 1.0), FS, seed=3)
    assert envelope_centroid(bake(x, "female_to_male")) < 0.94 * envelope_centroid(x)


@pytest.mark.parametrize("mode", ["easy_pitch", "accurate_pitch", *GENDER_TARGETS])
def test_input_without_a_voice_still_bakes(mode):
    noise = 0.05 * np.random.default_rng(0).standard_normal(3 * FS)
    y = bake(noise, mode)
    assert np.all(np.isfinite(y)) and y.size == noise.size


def test_removed_modes_fall_back_to_the_default(speech):
    # A recipe saved with a mode that no longer exists still bakes (as the default).
    old = bake(speech, "ronaldo")
    np.testing.assert_allclose(old, bake(speech, "chipmunk"))


def test_old_voice_changer_recipes_still_bake(speech):
    y = run_recipe(speech, FS, [{"op": "voice_changer", "params": {"mode": "chipmunk"}}])
    assert y.size == speech.size and not np.allclose(y, speech)
