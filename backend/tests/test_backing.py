"""
Background Music (dsp/backing.py): the key from a circular cross-correlation, the chord
progression on a tempo grid, the synthesized drums, the sidechain duck -- and the voice
left untouched.

Run from backend/:   python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

from app.dsp.backing import (
    STYLES, Chord, Key, background_music, backing_track, choose_chord, detect_key,
    diatonic_triads, kick, pitch_class_histogram, seventh, voicing,
)
from app.dsp.dsp_engine import run_recipe
from app.dsp.synth import midi_to_hz

from speech_synth import harmonic_source

FS = 16_000
NOTE_S, PAD_S = 0.35, 0.25
TWINKLE = [60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60]      # C major
A_MINOR_TUNE = [57, 60, 64, 60, 57, 59, 60, 62, 64, 62, 60, 59, 57, 64, 57]


def hum(midis, gap_after=None):
    parts = []
    for i, m in enumerate(midis):
        parts.append(0.3 * harmonic_source(np.full(int(NOTE_S * FS), float(midi_to_hz(m))), FS, 1.0))
        if gap_after is not None and i == gap_after:
            parts.append(np.zeros(int(0.8 * FS)))
    pad = np.zeros(int(PAD_S * FS))
    return np.concatenate([pad, *parts, pad])


def level_db(y, a, b):
    seg = y[int(a * FS):int(b * FS)]
    return 20.0 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)


# --------------------------------------------------------------------------- key
@pytest.mark.parametrize("tune, key", [(TWINKLE, "C major"), (A_MINOR_TUNE, "A minor")])
def test_the_key_of_a_sung_tune(tune, key):
    _, info = backing_track(hum(tune), FS)
    assert info["key"] == key


@pytest.mark.parametrize("shift", range(-6, 6))
def test_transposing_the_melody_rotates_the_key(shift):
    # The circular cross-correlation's shift property: rotate h, the peak moves with it.
    h = pitch_class_histogram([m + shift for m in TWINKLE], np.ones(len(TWINKLE)))
    assert detect_key(h) == Key(shift % 12, "major")


def test_no_notes_defaults_to_c_major():
    assert detect_key(np.zeros(12)) == Key(0, "major")


# --------------------------------------------------------------------------- chords
def test_diatonic_triads_leave_out_the_diminished_one():
    triads = dict(diatonic_triads(Key(0, "major")))
    assert triads[0] == (0, 4, 7)                          # C E G
    assert triads[4] == (7, 11, 2)                         # G B D
    assert triads[5] == (9, 0, 4)                          # A C E
    assert 6 not in triads                                  # B diminished


@pytest.mark.parametrize("sung, chord", [((0, 4, 7), (0, 4, 7)), ((9, 0, 4), (9, 0, 4)),
                                         ((5, 9, 0), (5, 9, 0)), ((7, 11, 2), (7, 11, 2))])
def test_the_chord_that_covers_the_notes_wins(sung, chord):
    w = np.zeros(12)
    w[list(sung)] = 1.0
    assert choose_chord(w, Key(0, "major"), None) == chord


@pytest.mark.parametrize("key, pcs, sev", [
    (Key(0, "major"), (0, 4, 7), 11),                       # Cmaj7
    (Key(0, "major"), (7, 11, 2), 5),                       # G7
    (Key(0, "major"), (9, 0, 4), 7),                        # Am7
    (Key(9, "minor"), (9, 0, 4), 7),                        # Am7 in A minor
    (Key(9, "minor"), (5, 9, 0), 4),                        # Fmaj7
])
def test_lofi_sevenths_are_diatonic(key, pcs, sev):
    assert seventh(key, Chord(0, 1, pcs[0], pcs)) == sev


def test_lofi_tape_rolls_off_the_top():
    x = hum(TWINKLE)
    music, _ = backing_track(x, FS, style="lofi", duck_db=0.0)
    plain, _ = backing_track(x, FS, style="epiano", duck_db=0.0)
    sp = lambda y: np.abs(np.fft.rfft(y)) ** 2          # noqa: E731
    f = np.fft.rfftfreq(x.size, 1.0 / FS)
    top = lambda y: sp(y)[f > 4500.0].sum() / sp(y).sum()   # noqa: E731
    assert top(music) < 0.5 * top(plain)                     # the tape rolls the top off


def test_voicing_sits_the_triad_mid_range_and_the_bass_below():
    triad, bass = voicing(Chord(0, 1, 9, (9, 0, 4)))       # A minor
    assert all(55 <= m <= 67 + 12 for m in triad)
    assert sorted(m % 12 for m in triad) == [0, 4, 9]
    assert 36 <= bass <= 47 and bass % 12 == 9


def test_chords_follow_the_melody():
    _, info = backing_track(hum(TWINKLE), FS, chords="follow", tempo_bpm=120)
    names = [name for _, name in info["chords"]]
    assert names[0] == "C" and set(names) <= {"C", "F", "G", "Am", "Dm", "Em"}


@pytest.mark.parametrize("tune, loop", [(TWINKLE, ["C", "G", "Am", "F"]),
                                        (A_MINOR_TUNE, ["Am", "F", "C", "G"])])
def test_the_loop_is_the_pop_progression_in_the_key(tune, loop):
    x = np.concatenate([hum(tune), np.zeros(12 * FS)])      # long enough for 2 loops
    _, info = backing_track(x, FS, tempo_bpm=120)
    names = [name for _, name in info["chords"]]
    assert names[:8] == loop + loop


@pytest.mark.parametrize("bpm", [60, 90, 150])
def test_one_chord_per_bar_on_the_tempo_grid(bpm):
    _, info = backing_track(hum(TWINKLE), FS, tempo_bpm=bpm)
    starts = [t for t, _ in info["chords"]]
    bar = 4 * 60.0 / bpm
    assert starts[0] == 0.0
    assert np.allclose(np.diff(starts), bar, atol=0.01)


def test_kick_is_a_falling_chirp():
    k = kick(FS)
    early, late = k[: int(0.03 * FS)], k[int(0.15 * FS): int(0.3 * FS)]
    zero_rate = lambda z: np.count_nonzero(np.diff(np.sign(z))) / (z.size / FS) / 2   # noqa: E731
    assert zero_rate(early) > 1.5 * zero_rate(late)            # the pitch falls
    assert 40.0 < zero_rate(late) < 70.0                       # toward 50 Hz
    assert np.max(np.abs(k[-FS // 100:])) < 0.3                # and dies away


def test_drums_add_a_beat():
    x = hum(TWINKLE)
    with_drums, _ = backing_track(x, FS, drums=True, style="pad")
    without, _ = backing_track(x, FS, drums=False, style="pad")
    assert not np.allclose(with_drums, without)


# --------------------------------------------------------------------------- mix
@pytest.mark.parametrize("style", STYLES)
def test_the_voice_is_untouched(style):
    x = hum(TWINKLE)
    backing, _ = backing_track(x, FS, style=style)
    y = background_music(x, FS, style=style)
    assert y.size == x.size and np.all(np.isfinite(y))
    np.testing.assert_allclose(y - backing, x, atol=1e-12)
    assert np.max(np.abs(backing)) > 0.0


def test_backing_sits_under_the_voice_and_ducks_while_singing():
    x = hum(TWINKLE, gap_after=6)
    backing, _ = backing_track(x, FS, volume=0.2, duck_db=8.0)           # 20 % ~ -14 dB
    singing = (PAD_S + 0.3, PAD_S + 6 * NOTE_S)
    gap = PAD_S + 7 * NOTE_S
    under = level_db(x, *singing) - level_db(backing, *singing)
    assert 17.0 < under < 28.0                               # ~14 dB + the 8 dB duck
    assert level_db(backing, gap + 0.4, gap + 0.75) > level_db(backing, *singing) + 3.0


def test_no_duck_means_a_steadier_level():
    x = hum(TWINKLE, gap_after=6)
    ducked, _ = backing_track(x, FS, duck_db=12.0)
    flat, _ = backing_track(x, FS, duck_db=0.0)
    gap = PAD_S + 7 * NOTE_S
    swell = lambda b: level_db(b, gap + 0.4, gap + 0.75) - level_db(b, PAD_S + 0.3, PAD_S + 2.0)  # noqa: E731
    assert swell(ducked) > swell(flat) + 6.0


def test_the_music_covers_the_whole_recording():
    x = np.concatenate([np.zeros(3 * FS), hum(TWINKLE), np.zeros(3 * FS)])
    music, _ = backing_track(x, FS)
    assert level_db(music, 1.5, 2.5) > -60.0                   # before anyone sings
    assert level_db(music, x.size / FS - 2.5, x.size / FS - 2.0) > -60.0   # after
    assert music[0] == 0.0 and abs(music[-1]) < 1e-9           # faded in and out


def test_music_plays_even_without_a_voice():
    silence = np.zeros(4 * FS)
    music, info = backing_track(silence, FS)
    assert info["key"] == "C major"
    assert level_db(music, 1.0, 3.0) > -35.0


def test_the_card_bakes_through_the_recipe():
    x = hum(A_MINOR_TUNE)
    for params in ({}, {"style": "guitar", "bass": False}, {"style": "organ", "chords": "follow"},
                   {"style": "epiano", "drums": False, "tempo_bpm": 140},
                   {"style": "lofi", "tempo_bpm": 75}):
        y = run_recipe(x, FS, [{"op": "backing", "params": params}])
        assert y.size == x.size and np.all(np.isfinite(y)) and not np.allclose(y, x)


# --------------------------------------------------------------------------- volume
@pytest.mark.parametrize("volume", [0.25, 1.0, 2.0])
def test_music_volume_is_a_plain_ratio(volume):
    x = hum(TWINKLE)
    half, _ = backing_track(x, FS, volume=0.5)
    moved, _ = backing_track(x, FS, volume=volume)
    np.testing.assert_allclose(moved, half * volume / 0.5, atol=1e-12)


def test_zero_volume_is_silence():
    x = hum(TWINKLE)
    music, _ = backing_track(x, FS, volume=0.0)
    assert np.all(music == 0.0)
    np.testing.assert_array_equal(background_music(x, FS, volume=0.0), x)


def test_full_volume_is_about_as_loud_as_the_voice_on_small_speakers():
    # Judged above 150 Hz, where laptop and phone speakers actually play, and with no dip.
    from scipy.signal import lfilter
    from app.dsp.filters import highpass_coefficients
    b, a = highpass_coefficients(FS, 150.0, 0.707)
    heard = lambda y: 20.0 * np.log10(np.sqrt(np.mean(lfilter(b, a, y) ** 2)))   # noqa: E731
    x = hum(TWINKLE)
    for style in STYLES:
        music, _ = backing_track(x, FS, style=style, volume=1.0, duck_db=0.0)
        voice = x[int(PAD_S * FS): -int(PAD_S * FS)]
        assert heard(music) == pytest.approx(heard(voice), abs=3.0), style


def test_most_of_the_music_is_where_small_speakers_play():
    x = hum(TWINKLE)
    for style in ("pad", "epiano", "organ", "lofi"):
        music, _ = backing_track(x, FS, style=style)
        sp = np.abs(np.fft.rfft(music)) ** 2
        f = np.fft.rfftfreq(music.size, 1.0 / FS)
        assert sp[f < 150.0].sum() < 0.4 * sp.sum(), style


def test_the_voice_is_untouched_at_any_volume():
    x = hum(TWINKLE)
    for volume in (0.1, 1.0, 2.0):
        music, _ = backing_track(x, FS, volume=volume)
        np.testing.assert_allclose(background_music(x, FS, volume=volume) - music, x,
                                   atol=1e-12)


def test_volume_is_the_first_control_in_percent():
    from app.dsp.dsp_engine import OPERATIONS
    card = next(op for op in OPERATIONS if op["id"] == "backing")
    volume = card["params"][0]
    assert volume["name"] == "volume" and volume["unit"] == "%"
    assert volume["min"] == 0.0 and volume["max"] == 200.0 and volume["default"] == 50
