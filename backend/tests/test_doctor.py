"""
Signal Doctor (dsp/doctor.py): each injected defect is found, a clean take is all-ok,
and the prescription it returns -- run through the ordinary recipe fold -- cures it.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.dsp.doctor import diagnose
from app.dsp.dsp_engine import run_recipe

from speech_synth import Speaker, random_script, speak

FS = 16_000


@pytest.fixture(scope="module")
def voice():
    x = speak(random_script(30, seed=4), Speaker(f0=140.0, formant_scale=1.0), FS, seed=4)
    return 0.5 * x / np.max(np.abs(x))


def problems(x, channels=None):
    return {f["id"]: f["status"] for f in diagnose(x, FS, channels)["findings"]
            if f["status"] != "ok"}


def cured(x, finding):
    """Apply the whole prescription, then re-diagnose the output."""
    fix = diagnose(x, FS)["fix"]
    assert fix, "a problem with no prescription"
    y = run_recipe(x, FS, fix)
    return finding not in problems(y)


def tone(freq, x, amp):
    return amp * np.sin(2.0 * np.pi * freq * np.arange(x.size) / FS)


def test_clean_speech_is_all_ok(voice):
    assert problems(voice) == {}
    assert diagnose(voice, FS)["fix"] == []


def test_clipping(voice):
    x = np.clip(3.0 * voice, -1.0, 1.0)
    assert problems(x).get("clipping") == "bad"
    assert cured(x, "clipping")


def test_clipping_in_one_channel_is_not_hidden_by_the_mono_fold(voice):
    left = np.clip(3.0 * voice, -1.0, 1.0)
    right = 0.4 * left                          # fold peaks at 0.7: under the gate
    fold = 0.5 * (left + right)
    assert "clipping" not in problems(fold)
    assert problems(fold, np.stack([left, right], axis=1)).get("clipping") == "bad"


def test_dc_offset(voice):
    x = voice + 0.08
    assert problems(x) == {"dc": "bad"}
    assert cured(x, "dc")


@pytest.mark.parametrize("mains", [50.0, 60.0])
def test_hum(voice, mains):
    x = voice + tone(mains, voice, 0.02) + tone(2 * mains, voice, 0.01)
    assert problems(x) == {"hum": "bad"}
    notches = [s["params"]["cutoff"] for s in diagnose(x, FS)["fix"]]
    # The fundamental always; a harmonic only where it stands out of the voice around it.
    assert notches[0] == mains and all(n % mains == 0 for n in notches)
    assert cured(x, "hum")


def test_hiss(voice):
    x = voice + 0.01 * np.random.default_rng(0).standard_normal(voice.size)
    assert "hiss" in problems(x)
    y = run_recipe(x, FS, diagnose(x, FS)["fix"])
    # Spectral subtraction lowers the floor; it need not reach the -60 dB "clean" line.
    before = diagnose(x, FS)["findings"][1]["value"]
    after = diagnose(y, FS)["findings"][1]["value"]
    floor = lambda v: float(v.split("floor ")[1].split(" dBFS")[0])     # noqa: E731
    assert floor(after) < floor(before) - 6.0


def test_low_level(voice):
    x = 0.01 * voice
    assert problems(x) == {"level": "bad"}
    assert cured(x, "level")


def test_long_silences(voice):
    x = np.concatenate([voice, np.zeros(4 * FS), voice])
    assert problems(x) == {"silence": "warn"}
    assert cured(x, "silence")


def test_stereo(voice):
    assert problems(voice, np.stack([voice, 0.5 * voice], axis=1)) == {"stereo": "warn"}
    assert problems(voice, np.stack([voice, -voice], axis=1)) == {"stereo": "bad"}
    assert problems(voice, np.stack([voice, voice], axis=1)) == {}


def test_prescription_order(voice):
    # Checks run clipping, hiss, hum, DC, level ... but the prescription is always
    # DC -> hum notch -> level, gain last.
    x = 0.01 * voice + 0.03 + tone(60.0, voice, 0.005)
    fix = diagnose(x, FS)["fix"]
    assert [(s["op"], s["params"].get("mode") or s["params"].get("target")) for s in fix] == [
        ("level", "none"), ("filter", "notch"), ("level", "peak"),
    ]
