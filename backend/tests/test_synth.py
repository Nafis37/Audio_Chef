"""
Synthesis building blocks (dsp/synth.py), each checked against its theory -- note
numbers from the pitch track, a Fourier series with nothing above Nyquist, the
Karplus-Strong pitch fs / (N + 1/2 + d), FM sidebands at f_c +- k f_m.

Run from backend/:   python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

from app.dsp.synth import (
    extract_notes, fm_tone, fourier_tone, hz_to_midi, karplus_strong, midi_to_hz,
)

from speech_synth import harmonic_source

FS = 16_000
MELODY_HZ = [220.0, 246.94, 261.63, 293.66]          # A3 B3 C4 D4
MELODY_MIDI = [57, 59, 60, 62]
NOTE_S, PAD_S = 0.4, 0.25


@pytest.fixture(scope="module")
def hummed():
    """A 'voice' humming four notes, with silence either side."""
    f0 = np.repeat(MELODY_HZ, int(NOTE_S * FS))
    x = 0.3 * harmonic_source(f0, FS, 1.0)
    pad = np.zeros(int(PAD_S * FS))
    return np.concatenate([pad, x, pad])


def spectrum(y, fs=FS, zoom=4):
    sp = np.abs(np.fft.rfft(y * np.blackman(y.size), zoom * y.size)) ** 2
    return np.fft.rfftfreq(zoom * y.size, 1.0 / fs), sp


# --------------------------------------------------------------------------- notes
def test_note_numbers_are_logarithmic_in_frequency():
    assert hz_to_midi(440.0) == pytest.approx(69.0)
    assert hz_to_midi(880.0) == pytest.approx(81.0)                # an octave = 12
    assert midi_to_hz(60.0) == pytest.approx(261.63, abs=0.01)


def test_the_hummed_melody_becomes_its_notes(hummed):
    notes = extract_notes(hummed, FS)
    assert [n.midi for n in notes] == MELODY_MIDI
    for i, n in enumerate(notes):
        assert n.start / FS == pytest.approx(PAD_S + NOTE_S * i, abs=0.03)
        assert n.end / FS == pytest.approx(PAD_S + NOTE_S * (i + 1), abs=0.03)


def test_a_wobble_is_not_a_new_note():
    f0 = np.full(FS, 220.0)
    f0[FS // 2: FS // 2 + int(0.03 * FS)] = 233.0                  # 30 ms off by a semitone
    notes = extract_notes(0.3 * harmonic_source(f0, FS, 1.0), FS)
    assert [n.midi for n in notes] == [57]


def test_silence_and_noise_give_no_notes():
    assert extract_notes(np.zeros(FS), FS) == []
    assert extract_notes(0.05 * np.random.default_rng(0).standard_normal(FS), FS) == []


# --------------------------------------------------------------------------- Fourier series
def test_band_limited_saw_has_nothing_above_nyquist_to_alias():
    f = 3100.0                                  # K = 2: a naive saw would alias badly
    y = fourier_tone(np.full(FS, f), FS, brightness=1.0)
    naive = 2.0 * ((f * np.arange(FS) / FS) % 1.0) - 1.0

    def alias_db(z):
        fr, sp = spectrum(z)
        harmonic = np.zeros(fr.size, dtype=bool)
        for k in range(1, 10):
            harmonic |= np.abs(fr - k * f) < 15.0
        return 10.0 * np.log10(sp[~harmonic].sum() / sp[harmonic].sum())

    assert alias_db(naive) > -20.0              # the naive wave really does alias
    assert alias_db(y) < -80.0                  # the band-limited one does not


def test_square_has_only_odd_harmonics():
    f = 200.0
    fr, sp = spectrum(fourier_tone(np.full(FS, f), FS, odd_only=True, brightness=1.0))
    level = lambda k: sp[np.argmin(np.abs(fr - k * f))]                # noqa: E731
    for k in (2, 4, 6):
        assert level(k) < 1e-6 * level(1)
    assert level(3) > 1e-2 * level(1)


def test_brightness_is_a_low_pass_on_the_harmonics():
    f = 200.0
    def tenth_harmonic(b):
        fr, sp = spectrum(fourier_tone(np.full(FS, f), FS, brightness=b))
        return sp[np.argmin(np.abs(fr - 10 * f))] / sp[np.argmin(np.abs(fr - f))]
    assert tenth_harmonic(0.0) < 1e-2 * tenth_harmonic(1.0)


# --------------------------------------------------------------------------- Karplus-Strong
@pytest.mark.parametrize("fs", [16_000, 44_100])
@pytest.mark.parametrize("f", [110.0, 220.0, 440.0, 880.0])
def test_plucked_string_is_in_tune(fs, f):
    # The fractional delay makes the loop exactly fs / f samples long.
    y = karplus_strong(f, fs, fs)[: fs // 2]
    fr, sp = spectrum(y, fs, zoom=8)
    band = (fr > 0.75 * f) & (fr < 1.3 * f)
    assert fr[band][np.argmax(sp[band])] == pytest.approx(f, rel=0.003)   # ~5 cents


def test_plucked_string_decays_at_its_t60():
    f, t60 = 220.0, 1.0
    y = karplus_strong(f, 2 * FS, FS, t60=t60)
    rms = lambda a, b: np.sqrt(np.mean(y[int(a * FS):int(b * FS)] ** 2))  # noqa: E731
    drop_db = 20.0 * np.log10(rms(0.9, 1.1) / rms(0.0, 0.2))
    assert -70.0 < drop_db < -45.0              # ~60 dB down one T60 later


# --------------------------------------------------------------------------- FM
def test_fm_sidebands_sit_at_fc_plus_minus_k_fm():
    fc, ratio = 400.0, 3.5
    y = fm_tone(np.full(FS, fc), FS, ratio, 3.0, 1e3, 1e3)       # no decay: steady index
    fr, sp = spectrum(y)
    peaks = fr[np.argsort(sp)[-400:]]
    expected = {abs(fc + k * ratio * fc) for k in range(-3, 4)}
    for line in expected:
        assert np.min(np.abs(peaks - line)) < 5.0, line


def test_electric_piano_sidebands_are_harmonic():
    fc = 220.0
    y = fm_tone(np.full(FS, fc), FS, 1.0, 3.0, 1e3, 1e3)
    fr, sp = spectrum(y)
    top = fr[np.argsort(sp)[-20:]]
    assert np.all(np.abs(top / fc - np.round(top / fc)) < 0.02)
