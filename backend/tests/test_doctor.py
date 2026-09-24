"""
Signal Doctor (dsp/doctor.py): each injected defect is found -- and named correctly --
clean takes are all-ok, and the prescription it returns, run through the ordinary recipe
fold, cures it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from app.dsp.doctor import diagnose
from app.dsp.dsp_engine import run_recipe

from speech_synth import Speaker, random_script, speak

FS = 16_000
CHECKS = {"clipping", "fan", "noise", "rumble", "uneven", "silence"}


def say(f0=140.0, formant_scale=1.0, tilt=1.0, seed=4, fs=FS):
    x = speak(random_script(30, seed=seed), Speaker(f0, formant_scale, tilt), fs, seed=seed)
    return 0.5 * x / np.max(np.abs(x))


@pytest.fixture(scope="module")
def voice():
    return say()


def problems(x, channels=None, fs=FS):
    return {f["id"]: f["status"] for f in diagnose(x, fs, channels)["findings"]
            if f["status"] != "ok"}


def cured(x, finding, fs=FS):
    """Apply the whole prescription, then re-diagnose the output."""
    fix = diagnose(x, fs)["fix"]
    assert fix, "a problem with no prescription"
    y = run_recipe(x, fs, fix)
    return finding not in problems(y, fs=fs)


def noise(n, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


def band_noise(n, lo, hi, order=2, seed=0):
    x = sosfilt(butter(order, [lo, hi], "bp", fs=FS, output="sos"), noise(n, seed))
    return x / np.std(x)


def test_exactly_the_six_checks(voice):
    assert {f["id"] for f in diagnose(voice, FS)["findings"]} == CHECKS


@pytest.mark.parametrize("f0, formant_scale, tilt, seed", [
    (100.0, 1.0, 1.0, 1), (100.0, 1.0, 1.0, 4), (120.0, 1.0, 1.0, 1), (120.0, 1.0, 1.0, 4),
    (140.0, 1.0, 1.0, 1), (150.0, 1.0, 1.0, 4), (180.0, 1.1, 0.8, 1), (200.0, 1.1, 1.0, 4),
    (220.0, 1.15, 1.0, 4), (240.0, 1.15, 1.0, 1), (110.0, 0.95, 1.3, 4),
])
def test_clean_speech_is_all_ok(f0, formant_scale, tilt, seed):
    # Deep, high, dark and bright voices; with pauses (seed 4) and without (seed 1).
    x = say(f0, formant_scale, tilt, seed)
    assert problems(x) == {}
    assert diagnose(x, FS)["fix"] == []


@pytest.mark.parametrize("fs", [44_100, 48_000])
@pytest.mark.parametrize("f0", [100.0, 150.0])
def test_clean_speech_at_high_rates_is_all_ok(fs, f0):
    # A rounded peak spends more samples near its top at a high rate: not a flat top.
    # (A take with pauses: the no-pause guard of the noise checks is covered at 16 kHz.)
    assert problems(say(f0, seed=4, fs=fs), fs=fs) == {}


# ---- clipping ------------------------------------------------------------------------
def test_clipping_is_reported_without_a_card(voice):
    # Too audible to leave out, but no recipe card repairs it: advice, not a fix.
    x = np.clip(3.0 * voice, -1.0, 1.0)
    assert problems(x).get("clipping") == "bad"
    report = diagnose(x, FS)
    assert next(f for f in report["findings"] if f["id"] == "clipping")["fix"] == []
    assert report["fix"] == []


def test_clipping_turned_down_afterwards_is_still_clipping(voice):
    assert problems(0.5 * np.clip(3.0 * voice, -1.0, 1.0)) == {"clipping": "bad"}


def test_clipping_on_one_side_only(voice):
    assert problems(np.clip(3.0 * voice, -0.6, 1.0)) == {"clipping": "bad"}


def test_a_loud_clean_take_is_not_clipping(voice):
    assert problems(0.99 * voice / np.max(np.abs(voice))) == {}


def test_clipping_in_one_channel_is_not_hidden_by_the_mono_fold(voice):
    left = np.clip(3.0 * voice, -1.0, 1.0)
    right = 0.4 * voice
    fold = 0.5 * (left + right)
    assert problems(fold, np.stack([left, right], axis=1)).get("clipping") == "bad"


# ---- fan / AC vs other background noise ----------------------------------------------
def test_fan_noise(voice):
    x = voice + 0.02 * band_noise(voice.size, 80.0, 1200.0)
    assert problems(x) == {"fan": "bad"}
    # Speech < 20 dB over the fan: denoising dents the quiet words, the leveler restores.
    assert [s["op"] for s in diagnose(x, FS)["fix"]] == ["noise_remover", "leveler"]
    assert cured(x, "fan")


def test_fan_noise_between_a_recorders_silent_edges(voice):
    # A browser recorder starts and stops on pure zeros.  They are not the room: the fan
    # must still be found (they once made the take look like it had no pauses).
    x = voice + 0.01 * band_noise(voice.size, 80.0, 1200.0)
    x = np.concatenate([np.zeros(int(0.3 * FS)), x, np.zeros(int(0.2 * FS))])
    assert problems(x) == {"fan": "bad"}
    assert cured(x, "fan")


def test_a_fan_25_db_under_the_voice_gets_a_card(voice):
    # Plain to hear in every pause -- not a "minor" note.
    x = voice + 0.005 * band_noise(voice.size, 80.0, 1200.0)
    assert problems(x) == {"fan": "bad"}
    assert cured(x, "fan")


def test_air_conditioning(voice):
    # A low whoosh with the compressor's 123 Hz whine riding on it.
    t = np.arange(voice.size) / FS
    low = sosfilt(butter(1, 500.0, "lp", fs=FS, output="sos"), noise(voice.size))
    x = voice + 0.02 * low / np.std(low) + 0.003 * np.sin(2.0 * np.pi * 123.0 * t)
    assert problems(x) == {"fan": "bad"}
    assert cured(x, "fan")


def test_hiss_is_other_noise_not_fan(voice):
    x = voice + 0.01 * noise(voice.size)
    assert problems(x) == {"noise": "bad"}
    finding = next(f for f in diagnose(x, FS)["findings"] if f["id"] == "noise")
    assert finding["title"] == "Background hiss"
    assert cured(x, "noise")


def test_heavy_broadband_noise(voice):
    x = voice + 0.03 * noise(voice.size)
    assert problems(x) == {"noise": "bad"}
    assert cured(x, "noise")


def test_faint_noise_is_not_worth_fixing(voice):
    assert problems(voice + 0.0018 * noise(voice.size)) == {}


# ---- rumble --------------------------------------------------------------------------
def test_rumble(voice):
    low = sosfilt(butter(4, 50.0, "lp", fs=FS, output="sos"), noise(voice.size))
    x = voice + 0.05 * low / np.std(low)
    assert problems(x) == {"rumble": "bad"}
    assert diagnose(x, FS)["fix"][0]["params"] == {"mode": "highpass", "cutoff": 80.0, "order": "4"}
    assert cured(x, "rumble")


def test_mic_pops(voice):
    x = voice.copy()
    n = int(0.04 * FS)
    for k in range(5):
        i = int((k + 0.5) * x.size / 5)
        x[i:i + n] += 0.4 * np.hanning(n) * np.sin(2.0 * np.pi * 30.0 * np.arange(n) / FS)
    assert problems(x) == {"rumble": "bad"}
    assert cured(x, "rumble")


def test_dc_offset_is_not_rumble(voice):
    assert problems(voice + 0.08) == {}


# ---- uneven volume -------------------------------------------------------------------
def test_uneven_volume(voice):
    half = voice.size // 2
    x = np.concatenate([voice[:half], 0.18 * voice[half:]])        # -15 dB
    assert problems(x) == {"uneven": "bad"}
    assert [s["op"] for s in diagnose(x, FS)["fix"]] == ["leveler"]
    assert cured(x, "uneven")
    assert problems(np.concatenate([voice[:half], 0.6 * voice[half:]])) == {}   # -4.4 dB


def test_uneven_volume_in_a_noisy_room(voice):
    # A fan in the pauses must not read as quiet speech -- nor hide the quiet half.
    half = voice.size // 2
    x = np.concatenate([voice[:half], 0.18 * voice[half:]])
    x = x + 0.003 * band_noise(x.size, 80.0, 1200.0)
    report = diagnose(x, FS)
    assert {f["id"]: f["status"] for f in report["findings"]}["uneven"] == "bad"
    assert "leveler" in [s["op"] for s in report["fix"]]
    assert cured(x, "uneven")


# ---- long silences -------------------------------------------------------------------
def test_a_silence_in_a_noisy_room(voice):
    # The fan keeps going through the gap: still nobody speaking for 4 s.
    x = with_gap(voice, 4.0)
    x = x + 0.01 * band_noise(x.size, 80.0, 1200.0)
    assert problems(x).get("silence") == "bad"
    assert cured(x, "silence")


def test_a_take_that_is_mostly_silence(voice):
    # 10 s of room, then a little speech: the "loud" part of the file is the room.
    x = np.concatenate([np.zeros(10 * FS), voice[: 2 * FS]])
    x = x + 0.002 * band_noise(x.size, 80.0, 1200.0)
    assert problems(x).get("silence") == "bad"
    assert cured(x, "silence")


def with_gap(voice, seconds, floor=0.0):
    half = voice.size // 2
    x = np.concatenate([voice[:half], np.zeros(int(seconds * FS)), voice[half:]])
    return x + floor * noise(x.size, seed=3)


def test_a_three_second_silence(voice):
    x = with_gap(voice, 3.0)
    assert problems(x) == {"silence": "bad"}
    step = diagnose(x, FS)["fix"][0]
    assert step["op"] == "silence_remover" and step["params"]["min_silence"] < 3.0
    assert cured(x, "silence")


def test_a_silence_over_a_noise_floor(voice):
    assert problems(with_gap(voice, 4.0, floor=0.001)) == {"silence": "bad"}


def test_a_shorter_pause_is_fine(voice):
    assert problems(with_gap(voice, 2.0)) == {}


@pytest.mark.parametrize("where", ["start", "end"])
def test_silence_at_the_ends(voice, where):
    gap = np.zeros(4 * FS)
    x = np.concatenate([gap, voice] if where == "start" else [voice, gap])
    assert problems(x) == {"silence": "bad"}
    y = run_recipe(x, FS, diagnose(x, FS)["fix"])
    assert y.size < voice.size + FS                 # the edge pause is cut away


# ---- the combined prescription -------------------------------------------------------
def test_warn_only_adds_no_cards(voice):
    # Borderline findings are reported, but not worth changing the sound for.
    x = voice + 0.003 * noise(voice.size)
    assert problems(x) == {"noise": "warn"}
    assert diagnose(x, FS)["fix"] == []


def test_prescription_order(voice):
    # Checks run clipping, noise, rumble, ... -- but the prescription is always
    # high-pass -> noise remover -> silence remover -> voice leveler.
    # The take (a level drop, then a 4 s gap), with a fan running under all of it.
    half = voice.size // 2
    x = np.concatenate([voice, np.zeros(4 * FS), voice[:half], 0.15 * voice[half:]])
    n = int(0.04 * FS)
    for k in range(5):                                   # mic pops in the first take
        i = int((k + 0.5) * voice.size / 5)
        x[i:i + n] += 0.4 * np.hanning(n) * np.sin(2.0 * np.pi * 30.0 * np.arange(n) / FS)
    fan = sosfilt(butter(2, [80.0, 1200.0], "bp", fs=FS, output="sos"), noise(x.size, seed=6))
    x = x + 0.01 * fan / np.std(fan)
    report = diagnose(x, FS)
    assert {f["id"] for f in report["findings"] if f["status"] == "bad"} == {
        "rumble", "fan", "silence", "uneven"}
    ops = [s["params"].get("mode", s["op"]) for s in report["fix"]]
    assert ops == ["highpass", "noise_remover", "silence_remover", "leveler"]
