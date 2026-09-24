"""
Voice Leveler (dsp/leveler.py): quiet and loud stretches of speech end up at one level,
a steady take is left alone, and the room's noise does not fool it.
"""

from __future__ import annotations

import numpy as np

from app.dsp.dsp_engine import run_recipe
from app.dsp.leveler import level_voice
from app.dsp.speech_analysis import analyse, speech_mask

from speech_synth import Speaker, random_script, speak

FS = 16_000


def say(seed=4, f0=140.0):
    x = speak(random_script(30, seed=seed), Speaker(f0, 1.0), FS, seed=seed)
    return 0.5 * x / np.max(np.abs(x))


def half_levels_db(x, split):
    """Speech level of each side of `split` samples, dB (speech frames only)."""
    frames = analyse(x, FS)
    speech = speech_mask(frames, 0.1)
    power = 10.0 ** (frames.energy_db / 10.0)
    first = frames.times * FS < split
    return [10.0 * np.log10(np.mean(power[speech & side])) for side in (first, ~first)]


def test_a_quiet_half_comes_up_to_the_loud_one():
    v = say()
    split = v.size // 2
    x = np.concatenate([v[:split], 0.15 * v[split:]])            # -16.5 dB
    before = np.subtract(*half_levels_db(x, split))
    after = np.subtract(*half_levels_db(level_voice(x, FS), split))
    assert before > 14.0
    assert abs(after) < 4.0


def test_a_shouted_stretch_is_brought_in_line():
    # Shouted middle, normal ends: the ends set the target, the middle comes down.
    v = say(seed=1)
    third = v.size // 3
    x = np.concatenate([0.2 * v[:third], v[third:2 * third], 0.2 * v[2 * third:]])
    y = level_voice(x, FS)
    before = np.subtract(*half_levels_db(x[third:], third))
    after = np.subtract(*half_levels_db(y[third:], third))
    assert before > 12.0 and abs(after) < 4.0


def test_a_steady_take_is_barely_touched():
    x = say()
    y = level_voice(x, FS)
    assert np.sqrt(np.mean((y - x) ** 2)) < 0.25 * np.sqrt(np.mean(x ** 2))


def test_no_speech_is_left_alone():
    noise = 0.05 * np.random.default_rng(0).standard_normal(3 * FS)
    np.testing.assert_array_equal(level_voice(noise, FS), noise)


def test_a_fan_does_not_count_as_quiet_speech():
    # Between the phrases there is only the room; the gain must not be driven by it.
    v = say()
    split = v.size // 2
    x = np.concatenate([v[:split], 0.15 * v[split:]])
    x = x + 0.01 * np.random.default_rng(1).standard_normal(x.size)
    after = np.subtract(*half_levels_db(level_voice(x, FS), split))
    assert abs(after) < 5.0


def test_card_bakes_and_zero_strength_is_a_bypass():
    x = say()
    np.testing.assert_array_equal(run_recipe(x, FS, [{"op": "leveler", "params": {"amount": 0}}]), x)
    y = run_recipe(x, FS, [{"op": "leveler"}])
    assert y.size == x.size and np.all(np.isfinite(y))
