"""
Background Music -- a music bed under your recording: chords, bass and drums
============================================================================
In plain words: listen to your recording, pick a key that suits it, and play a short
song underneath it -- a chord progression on a pad, piano, organ or guitar, a bass line
and a drum beat, at a tempo you choose, from the first second to the last.  Whenever you
speak or sing, the music dips so your voice stays in front; in the pauses it comes back
up.  Your voice itself is not changed at all:

        y = x + b                (x your recording, b the music -- superposition)

1. Notes
--------
synth.extract_notes: the autocorrelation pitch track as note numbers
m = 69 + 12 log2(f0 / 440), with start, end and velocity.  Speech gives notes too -- just
not a melody -- which is enough to pick a key the voice sits comfortably in.

2. Key -- circular cross-correlation with the key profiles
----------------------------------------------------------
Fold every note onto its pitch class p = m mod 12 (C = 0 ... B = 11), weighted by
duration x velocity: a 12-bin histogram h[p].  The Krumhansl-Kessler probe-tone profiles
K_major, K_minor (listening experiments, 1982) say how strongly each pitch class belongs
to C major / C minor; the key with tonic t is the profile ROTATED by t.  So the key is
the peak of a circular cross-correlation (a Pearson correlation for every rotation):

        r_mode(t) = corr( h[p],  K_mode[(p - t) mod 12] ),     t = 0..11
        key       = argmax over (t, mode) of r_mode(t)

Transposing the voice rotates h and the peak moves with it -- the shift property a DFT
has.  With no notes at all the key is C major.

3. Time -- tempo, beats, bars
-----------------------------
        beat = 60 / BPM seconds,       bar = 4 beats,       bar k starts at k * bar

The music is laid out on this grid for the whole length of the recording.

4. Chords
---------
The key's scale (major: 0 2 4 5 7 9 11; natural minor: 0 2 3 5 7 8 10 semitones above
the tonic) gives a triad on each degree d: scale[d], scale[d+2], scale[d+4].  One chord
per bar, chosen one of two ways:

  loop    the most used progression in pop music, repeated:
          major  I - V - vi - IV          minor  i - VI - III - VII
  follow  for singing: in each bar, with w[p] the pitch-class weights of the notes sung
          there, the diatonic triad (the diminished one left out) maximising

              score = sum_{p in chord} w[p]  +  W ( PRIOR[degree] + STAY [same as before] )

          W = sum_p w[p], so the bonuses scale with how much was sung.  A bar with no
          singing keeps the chord before it.

Voicing: the triad around C4 (midi 55..67), the bass on the root an octave or two below
(midi 36..47), moving to the fifth on beat 3.

5. The instruments (synth.py's oscillators)
-------------------------------------------
        pad      two band-limited saws per note, detuned +-DETUNE_CENTS -- the slow beating
                 between them is a string pad's shimmer -- dark low-pass, slow ADSR, held
                 for the bar
        epiano   FM, ratio 1, struck on beats 1 and 3
        organ    additive sines at harmonics 1..4 (drawbars), held for the bar
        guitar   Karplus-Strong, strummed on every beat, down on the beat and up between
        bass     a band-limited saw, low-passed but keeping harmonics 2-4: on a speaker
                 that cannot play the fundamental, the ear still hears the bass note
                 from them (the "missing fundamental" -- pitch is heard from the
                 spacing of the harmonics)

6. Drums -- three textbook signals
----------------------------------
        kick    a CHIRP: a sine whose frequency falls  f(t) = 50 + 100 e^{-t/0.04} Hz,
                phase accumulated, under an e^{-t/0.25} decay, plus a 4 ms burst of
                high-passed noise (the beater click -- what small speakers
                actually play of a kick)                             beats 1 and 3
        snare   white noise through an RBJ band-pass at 1.8 kHz (the rattle) plus a
                185 Hz sine (the drum head), e^{-t/0.12}              beats 2 and 4
        hat     white noise through an RBJ high-pass at 7 kHz, e^{-t/0.03}   every 8th,
                the off-beats softer

7. Volume, ducking and the edges
--------------------------------
Volume is a plain ratio V: 0 = off, 1 (100 %) = as loud as your voice, 2 = twice as
loud.  "As loud" is judged the way small speakers hear it: laptop and phone speakers
play almost nothing below ~150 Hz, and most of a bass line and a kick drum lives there.
Comparing plain RMS let the bass use up the volume while the audible part stayed faint.
So both the music and the voice are first passed through a 2nd-order high-pass at
AUDIBLE_HZ (a crude loudness weighting), and

        b <- b * V * rms(HP x_active) / rms(HP b)        (x_active: the voice's speech)

(with no voice, rms(HP x_active) is taken as NO_VOICE_DBFS).  Then the voice drives the
music's gain -- a sidechain.  Per 10 ms frame, how much voice there is:

        u[t] = clip( (E[t] - (E_90 - 25 dB)) / 15 dB, 0, 1 )      (E = frame energy, dB)

smoothed by a one-pole follower that is fast when the voice comes in and slow when it
stops, so the music ducks at once but does not pump between syllables:

        a[t] = a[t-1] + alpha ( u[t] - a[t-1] ),   alpha = 1 - exp(-hop / tau)
        tau  = DUCK_ATTACK_S if u[t] > a[t-1] else DUCK_RELEASE_S

(an exponential moving average: the difference equation of a first-order low-pass with
its pole at exp(-hop / tau)).  The gain is g[t] = 10^(-DUCK a[t] / 20).  Finally the bed
fades in over FADE_IN_S and out over FADE_OUT_S at the ends of the file (each at most a
quarter of a short recording, so the two fades never meet).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .filters import bandpass_coefficients, highpass_coefficients
from .speech_analysis import analyse
from .synth import adsr, extract_notes, fm_tone, fourier_tone, karplus_strong, midi_to_hz

STYLES = ("pad", "epiano", "organ", "guitar")
CHORD_MODES = ("loop", "follow")
PITCH_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")

# Krumhansl & Kessler (1982) probe-tone ratings, tonic first.
K_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
K_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
SCALES = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10)}
DIMINISHED = {"major": 6, "minor": 1}          # the degree left out
LOOP = {"major": (0, 4, 5, 3), "minor": (0, 5, 2, 6)}      # I-V-vi-IV,  i-VI-III-VII
PRIOR = {"major": (0.30, 0.05, 0.02, 0.20, 0.20, 0.15, 0.0),
         "minor": (0.30, 0.0, 0.15, 0.20, 0.15, 0.15, 0.10)}
STAY = 0.10
DETUNE_CENTS = 7.0
STRUM_S = 0.025
DUCK_ATTACK_S = 0.03
DUCK_RELEASE_S = 0.40
FADE_IN_S = 1.0
FADE_OUT_S = 2.0
NO_VOICE_DBFS = -20.0
AUDIBLE_HZ = 150.0             # below this, small speakers play almost nothing
MIX = {"chords": 1.6, "bass": 0.35, "drums": 0.6}


@dataclass(frozen=True)
class Key:
    tonic: int                 # pitch class 0..11
    mode: str                  # "major" | "minor"

    @property
    def name(self) -> str:
        return f"{PITCH_NAMES[self.tonic]} {self.mode}"


@dataclass(frozen=True)
class Chord:
    start: int                 # sample
    end: int
    root: int                  # pitch class
    pcs: tuple[int, int, int]  # the triad's pitch classes, root first

    @property
    def name(self) -> str:
        minor = (self.pcs[1] - self.pcs[0]) % 12 == 3
        return PITCH_NAMES[self.root] + ("m" if minor else "")


# ----------------------------------------------------------------------- 2. key
def pitch_class_histogram(midis, weights) -> np.ndarray:
    h = np.zeros(12)
    for m, w in zip(midis, weights):
        h[int(round(m)) % 12] += w
    return h


def detect_key(h: np.ndarray) -> Key:
    """The peak of the circular cross-correlation with both key profiles (section 2)."""
    best, best_r = Key(0, "major"), -np.inf
    if not np.any(h > 0):
        return best
    for mode, profile in (("major", K_MAJOR), ("minor", K_MINOR)):
        for t in range(12):
            r = float(np.corrcoef(h, np.roll(profile, t))[0, 1])
            if r > best_r:
                best, best_r = Key(t, mode), r
    return best


# ----------------------------------------------------------------------- 4. chords
def triad(key: Key, degree: int) -> tuple[int, int, int]:
    scale = SCALES[key.mode]
    return tuple((key.tonic + scale[(degree + i) % 7]) % 12 for i in (0, 2, 4))


def diatonic_triads(key: Key) -> list[tuple[int, tuple[int, int, int]]]:
    """(degree, pitch classes root-first) of every usable triad in the key."""
    return [(d, triad(key, d)) for d in range(7) if d != DIMINISHED[key.mode]]


def choose_chord(w: np.ndarray, key: Key, previous: tuple[int, int, int] | None):
    """The best-scoring triad for one bar's pitch-class weights (section 4, follow)."""
    total = float(w.sum())
    best, best_score = None, -np.inf
    for d, pcs in diatonic_triads(key):
        score = sum(w[p] for p in pcs) + total * (PRIOR[key.mode][d] + STAY * (pcs == previous))
        if score > best_score:
            best, best_score = pcs, score
    return best


def plan_chords(notes, key: Key, fs: int, bar: int, total: int, mode: str) -> list[Chord]:
    """One chord per bar over the whole recording (sections 3-4)."""
    chords: list[Chord] = []
    previous = None
    for k, a in enumerate(range(0, total, bar)):
        b = min(a + bar, total)
        if mode == "follow":
            w = np.zeros(12)
            for n in notes:
                overlap = min(n.end, b) - max(n.start, a)
                if overlap > 0:
                    w[int(round(n.midi)) % 12] += overlap / fs * n.velocity
            if w.sum() > 0:
                pcs = choose_chord(w, key, previous)
            else:
                pcs = previous or triad(key, 0)               # nothing sung: hold
        else:
            pcs = triad(key, LOOP[key.mode][k % 4])
        chords.append(Chord(a, b, pcs[0], pcs))
        previous = pcs
    return chords


def voicing(chord: Chord) -> tuple[list[int], int]:
    """(triad note numbers around C4, bass note number) -- section 4."""
    root = 55 + (chord.root - 55) % 12                  # 55..66
    notes = sorted(root + (p - chord.root) % 12 for p in chord.pcs)
    bass = 36 + (chord.root - 36) % 12                  # 36..47
    return notes, bass


# ----------------------------------------------------------------------- 5. instruments
def _add(out: np.ndarray, start: int, samples: np.ndarray) -> None:
    """Mix `samples` into `out` at `start`, cut at the end."""
    if start >= out.size or samples.size == 0:
        return
    n = min(samples.size, out.size - start)
    out[start:start + n] += samples[:n]


def _env(gate: int, length: int, fs: int, *adsr_args) -> np.ndarray:
    e = adsr(gate, fs, *adsr_args)
    return np.pad(e, (0, max(0, length - e.size)))[:length]


def _held(style: str, midis: list[int], gate: int, fs: int) -> np.ndarray:
    """pad / organ: the whole chord held for `gate` samples, plus its release."""
    length = gate + int(0.5 * fs)
    out = np.zeros(length)
    for m in midis:
        f = float(midi_to_hz(m))
        if style == "pad":
            for cents in (-DETUNE_CENTS, DETUNE_CENTS):
                out += 0.5 * fourier_tone(np.full(length, f * 2.0 ** (cents / 1200.0)), fs,
                                          brightness=0.35)
        else:
            phi = 2.0 * np.pi * f * np.arange(length) / fs
            out += sum(a * np.sin(k * phi) for k, a in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.2))
                       if k * f < 0.45 * fs)
    shape = (0.35, 0.6, 0.8, 0.5) if style == "pad" else (0.03, 0.1, 1.0, 0.15)
    return out * _env(gate, length, fs, *shape) / len(midis)


def _struck(midis: list[int], length: int, fs: int) -> np.ndarray:
    """epiano: every note struck at once, left to ring."""
    out = sum(fm_tone(np.full(length, float(midi_to_hz(m))), fs, 1.0, 2.5, 0.3, 1.5)
              for m in midis)
    fade = np.ones(length)
    tail = min(length, int(0.05 * fs))
    fade[length - tail:] = np.linspace(1.0, 0.0, tail)
    return out * fade / len(midis)


def _strum(midis: list[int], length: int, fs: int, down: bool, rng) -> np.ndarray:
    """guitar: the notes one after another, STRUM_S apart."""
    out = np.zeros(length)
    order = midis if down else midis[::-1]
    for i, m in enumerate(order):
        offset = int(round(i * STRUM_S * fs))
        if offset < length:
            out[offset:] += karplus_strong(float(midi_to_hz(m)), length - offset, fs, 0.45,
                                           t60=1.5, rng=rng)
    fade = np.ones(length)
    tail = min(length, int(0.04 * fs))
    fade[length - tail:] = np.linspace(1.0, 0.0, tail)
    return out * fade / len(midis) * (3.0 if down else 2.0)   # plucks are quiet: bring up


def _bass_note(midi: int, gate: int, fs: int) -> np.ndarray:
    length = gate + int(0.2 * fs)
    tone = fourier_tone(np.full(length, float(midi_to_hz(midi))), fs, brightness=0.25)
    return tone * _env(gate, length, fs, 0.01, 0.3, 0.7, 0.15)


# ----------------------------------------------------------------------- 6. drums
def kick(fs: int, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    t = np.arange(int(0.35 * fs)) / fs
    f = 50.0 + 100.0 * np.exp(-t / 0.04)                       # a falling chirp
    body = np.sin(2.0 * np.pi * np.cumsum(f) / fs) * np.exp(-t / 0.25)
    b, a = highpass_coefficients(fs, min(2000.0, 0.4 * fs), 0.7)
    click = lfilter(b, a, rng.standard_normal(t.size)) * np.exp(-t / 0.004)
    return body + 0.5 * click


def snare(fs: int, rng) -> np.ndarray:
    t = np.arange(int(0.25 * fs)) / fs
    b, a = bandpass_coefficients(fs, min(1800.0, 0.4 * fs), 0.8)
    rattle = lfilter(b, a, rng.standard_normal(t.size))
    head = 0.5 * np.sin(2.0 * np.pi * 185.0 * t) * np.exp(-t / 0.06)
    return (rattle * np.exp(-t / 0.12) + head) * 0.8


def hat(fs: int, rng) -> np.ndarray:
    t = np.arange(int(0.08 * fs)) / fs
    b, a = highpass_coefficients(fs, min(7000.0, 0.4 * fs), 0.7)
    return lfilter(b, a, rng.standard_normal(t.size)) * np.exp(-t / 0.03) * 0.5


# ----------------------------------------------------------------------- 7. volume, ducking
def _audible_rms(y: np.ndarray, fs: int, where: np.ndarray | None = None) -> float:
    """RMS after a 2nd-order high-pass at AUDIBLE_HZ: loudness as small speakers play it."""
    if y.size == 0:
        return 0.0
    b, a = highpass_coefficients(fs, AUDIBLE_HZ, 0.707)
    heard = lfilter(b, a, y)
    if where is not None:
        heard = heard[where]
    return float(np.sqrt(np.mean(heard ** 2))) if heard.size else 0.0


def duck_gain(frames, duck_db: float, n: int, fs: int) -> np.ndarray:
    """The sidechain gain for every sample (section 7)."""
    e = frames.energy_db
    if e.size == 0:
        return np.ones(n)
    loud = float(np.percentile(e, 90))
    u = np.clip((e - (loud - 25.0)) / 15.0, 0.0, 1.0)
    u[~frames.active] = 0.0
    hop_s = frames.hop / frames.fs
    up = 1.0 - np.exp(-hop_s / DUCK_ATTACK_S)
    down = 1.0 - np.exp(-hop_s / DUCK_RELEASE_S)
    a = np.zeros(u.size)
    level = 0.0
    for t, target in enumerate(u):
        level += (up if target > level else down) * (target - level)
        a[t] = level
    return np.interp(np.arange(n) / fs, frames.times, 10.0 ** (-duck_db * a / 20.0))


# ----------------------------------------------------------------------- the card
def backing_track(x: np.ndarray, fs: int, style: str = "pad", chords: str = "loop",
                  tempo_bpm: float = 90.0, drums: bool = True, bass: bool = True,
                  volume: float = 0.5, duck_db: float = 4.0):
    """(the music alone, info) -- everything but the final sum with the voice."""
    x = np.asarray(x, dtype=np.float64)
    style = style if style in STYLES else "pad"
    chords = chords if chords in CHORD_MODES else "loop"
    total = x.size
    frames = analyse(x, fs)
    notes = extract_notes(x, fs, frames)
    key = detect_key(pitch_class_histogram(
        [n.midi for n in notes], [(n.end - n.start) / fs * n.velocity for n in notes]))

    beat = max(1, int(round(60.0 / float(np.clip(tempo_bpm, 40.0, 220.0)) * fs)))
    bar = 4 * beat
    plan = plan_chords(notes, key, fs, bar, total, chords)
    rng = np.random.default_rng(0xBAC4)
    parts = {"chords": np.zeros(total), "bass": np.zeros(total), "drums": np.zeros(total)}
    kick_s, snare_s, hat_s = kick(fs, rng), snare(fs, rng), hat(fs, rng)
    held: dict[tuple, np.ndarray] = {}         # a loop repeats its chords: render each once

    for chord in plan:
        triad_notes, root = voicing(chord)
        gate = chord.end - chord.start
        if style in ("pad", "organ"):
            key_ = (tuple(triad_notes), gate)
            if key_ not in held:
                held[key_] = _held(style, triad_notes, gate, fs)
            _add(parts["chords"], chord.start, held[key_])
        elif style == "epiano":
            for b in (0, 2):
                _add(parts["chords"], chord.start + b * beat, _struck(triad_notes, 2 * beat, fs))
        else:
            for half in range(8):                         # down on the beat, up between
                _add(parts["chords"], chord.start + half * beat // 2,
                     _strum(triad_notes, beat // 2, fs, half % 2 == 0, rng))
        if bass:
            fifth = root + 7 if root + 7 <= 47 else root - 5
            _add(parts["bass"], chord.start, _bass_note(root, 2 * beat - beat // 8, fs))
            _add(parts["bass"], chord.start + 2 * beat, _bass_note(fifth, 2 * beat - beat // 8, fs))
        if drums:
            for b in range(4):
                at = chord.start + b * beat
                _add(parts["drums"], at, kick_s if b % 2 == 0 else snare_s)
                _add(parts["drums"], at, hat_s)
                _add(parts["drums"], at + beat // 2, 0.5 * hat_s)

    music = sum(MIX[name] * part for name, part in parts.items())

    # Volume: a ratio to the voice, both measured as small speakers hear them (section 7).
    active = np.repeat(frames.active, frames.hop)[:total]
    active = np.pad(active, (0, total - active.size))
    music_rms = _audible_rms(music, fs)
    if music_rms > 0.0:
        voice_rms = (_audible_rms(x, fs, active) if np.any(active)
                     else 10.0 ** (NO_VOICE_DBFS / 20.0))
        music *= voice_rms / music_rms * max(0.0, float(volume))
    music *= duck_gain(frames, duck_db, total, fs)

    # Edges: fade the bed in and out -- each fade at most a quarter of a short recording.
    n_in = min(total // 4, int(FADE_IN_S * fs))
    n_out = min(total // 4, int(FADE_OUT_S * fs))
    music[:n_in] *= np.linspace(0.0, 1.0, n_in)
    music[total - n_out:] *= np.linspace(1.0, 0.0, n_out)

    info = {"key": key.name, "chords": [(round(c.start / fs, 2), c.name) for c in plan]}
    return music, info


def background_music(x: np.ndarray, fs: int, style: str = "pad", chords: str = "loop",
                     tempo_bpm: float = 90.0, drums: bool = True, bass: bool = True,
                     volume: float = 0.5, duck_db: float = 4.0) -> np.ndarray:
    """Your recording, untouched, over a music bed (module docstring)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    music, _ = backing_track(x, fs, style, chords, tempo_bpm, drums, bass, volume, duck_db)
    return x + music
