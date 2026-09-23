"""
Phase Vocoder card: the celebrity caricatures (caricature.py).

Checked on the source-filter "speakers" of speech_synth.py, whose f0 and formants are
known: the pitch must move by the preset's semitones, the formants by its ratio -- and
independently of each other.  Whether a preset sounds like anyone is a listening
question, not something a test can claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.dsp.caricature import CARICATURES, pitch_formant_shift
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


@pytest.mark.parametrize("mode", list(CARICATURES))
def test_every_caricature_bakes_with_its_pitch_and_tempo(speech, mode):
    c = CARICATURES[mode]
    y = run_recipe(speech, FS, [{"op": "voice_changer", "params": {"mode": mode}}])
    assert np.all(np.isfinite(y))
    assert y.size == pytest.approx(speech.size / c.tempo, rel=0.02)
    ratio = median_f0(y) / median_f0(speech)
    # Grit jitters the pitch, so allow a little more than for the clean shift.
    assert ratio == pytest.approx(2 ** (c.semitones / 12), rel=0.06)


def test_old_voice_changer_recipes_still_bake(speech):
    y = run_recipe(speech, FS, [{"op": "voice_changer", "params": {"mode": "chipmunk"}}])
    assert y.size == speech.size and not np.allclose(y, speech)
