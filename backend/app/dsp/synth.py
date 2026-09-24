"""
Synthesis building blocks -- notes from a voice, and instruments built from maths
=================================================================================
In plain words: the pieces Background Music (backing.py) is made of.  A voice's pitch
cut into notes, and instruments built from scratch -- a band-limited synth, a plucked
string, an electric piano.  Nothing is sampled: each instrument is a few lines of
signals-and-systems maths, written out below.

Notation: fs the sample rate, n the sample index, f the note frequency (Hz),
phi[n] = 2 pi sum_{m<=n} f[m] / fs the running phase (a phase ACCUMULATOR, so a gliding
pitch stays continuous -- 2 pi f n / fs would jump whenever f changes).

1. From pitch to notes
----------------------
speech_analysis.analyse gives f0 every 10 ms (normalised autocorrelation, voiced frames
only).  A frequency becomes a note number on the equal-tempered scale, which is
logarithmic in frequency:

        m = 69 + 12 log2( f0 / 440 )            (69 = A4 = 440 Hz; one semitone = 1)

Rounded, and median-filtered over 5 frames (a wobble on one frame is not a new note),
m is cut into notes wherever it changes or the voicing stops.  Pieces shorter than
MIN_NOTE_S are merged into the note before them.  Each note keeps its start, end, its
note number, the voice's pitch curve inside it (for "follow my pitch" mode), and a
velocity: its mean amplitude relative to the loudest note, square-rooted (hearing is
closer to logarithmic than linear).  Back from a note number to a frequency:
f = 440 * 2^((m - 69)/12).

2. Synth, square, bass -- a band-limited Fourier series, filtered
-----------------------------------------------------------------
A sawtooth is the Fourier series

        saw(phi) = sum_{k>=1} sin(k phi) / k            (every harmonic, falling as 1/k)

and a square wave keeps only the odd k.  Generating the ideal waveform directly (a
jump once per period) is WRONG in discrete time: its harmonics go on forever, and every
one above fs/2 folds back below it (aliasing) as a harsh, out-of-tune tone.  So the sum
is cut off at the Nyquist frequency -- only k with k f_max < ALIAS_GUARD * fs/2:

        x[n] = sum_{k=1}^{K} c_k H(k f) sin(k phi[n]) / k,     K = floor(ALIAS_GUARD fs / (2 f_max))

(c_k = 1 for the saw; 1 for odd k and 0 for even k for the square).  Nothing above
Nyquist is ever generated, so there is nothing to alias.

The tone control is SUBTRACTIVE synthesis: the waveform passes through a 2nd-order
Butterworth low-pass at f_c.  The filter is linear and time-invariant, so each harmonic
(a sinusoid) comes out as the same sinusoid scaled by the magnitude response at its
frequency -- no convolution needed, just the response evaluated at k f:

        |H(f)| = 1 / sqrt(1 + (f / f_c)^4),          f_c = f (1 + 15 * brightness)

(the cutoff tracks the note, so every note has the same timbre).

Amplitude: an ADSR envelope -- attack (linear rise), decay (exponential fall to the
sustain level), sustain, release (exponential fall after the note ends).  Multiplying by
the envelope is amplitude modulation: in frequency it convolves each harmonic with the
envelope's (narrow) spectrum, which is why a fast attack sounds clicky and a slow one soft.

3. Plucked string -- Karplus-Strong
-----------------------------------
A delay line of N samples fed back through a two-point average -- a difference equation:

        y[n] = x[n] + g * ( y[n - N] + y[n - N - 1] ) / 2

x is a burst of N samples of noise (the pluck).  Around the loop the signal repeats
every N + 1/2 samples (the averager adds half a sample of delay), so the pitch is
f = fs / (N + 1/2).  N is a whole number, so on its own the string can only be tuned to
those steps -- at 16 kHz an 880 Hz note would come out ~30 cents flat.  The fix is a
FRACTIONAL delay d in [0, 1): linear interpolation between y[n - N - 1/2] and one sample
further back, which folds into the averager as three taps:

        y[n] = x[n] + g ( (1-d)/2 y[n-N] + 1/2 y[n-N-1] + d/2 y[n-N-2] )

        loop delay = N + 1/2 + d = fs / f      ->     N = floor(fs/f - 1/2),  d = the rest

(linear interpolation is itself a gentle low-pass, so high notes decay a little faster).
Each trip round the loop the averager, a low-pass H(w) = cos(w/2) e^{-jw/2}, takes
more off the high harmonics than the low ones: the string starts bright and mellows, as a
real one does.  g < 1 sets the decay: the loop gain per period is g, so after t seconds
the level is g^(f t); for a T60 (60 dB down) of PLUCK_T60 seconds,

        g^(f T60) = 10^-3     ->     g = 10^( -3 / (f T60) )

The poles of this system sit just inside the unit circle at the harmonics of f; g is how
far inside.  Brightness low-passes the noise burst (a moving average) before it enters.
The recursion only reaches N, N+1 and N+2 samples back, so it is computed a block of N
samples at a time: every block depends only on blocks already finished.

4. Electric piano and bell -- FM synthesis
------------------------------------------
        y[n] = A(t) sin( phi_c[n] + I(t) sin(phi_m[n]) ),          f_m = ratio * f_c

A sine whose phase is itself swung by another sine.  Its spectrum is a line at f_c and
sidelobes at f_c +- k f_m, with amplitudes J_k(I) (Bessel functions): the index I
decides how many sidebands are strong.  ratio = 1 puts every sideband on a harmonic
(electric piano); ratio = 3.5 puts them between harmonics -- inharmonic, like a bell.
I(t) decays faster than A(t), so the note starts bright and turns pure, which is how
struck instruments behave.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import medfilt

from .speech_analysis import SpeechFrames, analyse

ALIAS_GUARD = 0.95              # stop the Fourier series at 95 % of Nyquist
MAX_HARMONICS = 80
MIN_NOTE_S = 0.06
PLUCK_T60 = 2.0
_SEED = 0x1A57                  # fixed: the same input always renders the same way


@dataclass
class Note:
    start: int                  # first sample
    end: int                    # one past the last sample
    midi: float                 # note number (rounded)
    velocity: float             # 0..1
    times: np.ndarray           # the voice's frame times inside the note, s ...
    f0: np.ndarray              # ... and its pitch there, Hz


def midi_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 440.0 * 2.0 ** ((np.asarray(m) - 69.0) / 12.0)


def hz_to_midi(f: np.ndarray | float) -> np.ndarray | float:
    return 69.0 + 12.0 * np.log2(np.asarray(f) / 440.0)


# ----------------------------------------------------------------------- 1. notes
def extract_notes(x: np.ndarray, fs: int, frames: SpeechFrames | None = None) -> list[Note]:
    """Cut the voice's pitch track into notes (docstring section 1).  `frames`: an
    analysis of x already made, to reuse."""
    frames = frames if frames is not None else analyse(x, fs)
    voiced, f0, times = frames.voiced, frames.f0, frames.times
    if not np.any(voiced):
        return []
    midi = np.where(voiced, np.round(hz_to_midi(np.where(voiced, f0, 440.0))), -1.0)
    midi = medfilt(midi, 5)
    midi[~voiced] = -1.0

    # Runs of one note number; unvoiced frames (-1) separate notes.
    edges = np.flatnonzero(np.diff(np.concatenate([[-2.0], midi, [-2.0]])))
    runs = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if midi[a] >= 0]
    min_frames = max(1, int(round(MIN_NOTE_S * fs / frames.hop)))
    merged: list[list[float]] = []            # [start frame, end frame, note number]
    for a, b in runs:
        touching = bool(merged) and a == merged[-1][1]
        # A short piece touching the previous note is a wobble, not a note -- and the same
        # note resuming right after that wobble is still the one note.
        if touching and (b - a < min_frames or midi[a] == merged[-1][2]):
            merged[-1][1] = b
        else:
            merged.append([a, b, midi[a]])
    merged = [(int(a), int(b)) for a, b, _ in merged if b - a >= min_frames]
    if not merged:
        return []

    amp = 10.0 ** (frames.energy_db / 20.0)
    loud = [float(np.mean(amp[a:b])) for a, b in merged]
    top = max(loud) or 1.0
    half_hop = frames.hop / 2.0
    notes = []
    for (a, b), level in zip(merged, loud):
        ok = voiced[a:b]
        seg_f0 = f0[a:b][ok] if np.any(ok) else np.full(1, midi_to_hz(midi[a]))
        seg_t = times[a:b][ok] if np.any(ok) else times[a:a + 1]
        notes.append(Note(
            start=max(0, int(round(times[a] * fs - half_hop))),
            end=min(x.size, int(round(times[b - 1] * fs + half_hop))),
            midi=float(np.median(midi[a:b][midi[a:b] >= 0])),
            velocity=float(np.sqrt(level / top)),
            times=seg_t, f0=seg_f0,
        ))
    return notes


# ----------------------------------------------------------------------- envelopes
def adsr(length: int, fs: int, attack: float, decay: float, sustain: float,
         release: float) -> np.ndarray:
    """Gate open for `length` samples, then the release tail (docstring section 2)."""
    n_rel = int(round(release * fs))
    t = np.arange(length) / fs
    env = np.where(t < attack, t / max(attack, 1e-9),
                   sustain + (1.0 - sustain) * np.exp(-(t - attack) / max(decay, 1e-9)))
    end_level = float(env[-1]) if length else 0.0
    tail = end_level * np.exp(-np.arange(n_rel) / fs / max(release / 4.0, 1e-9))
    return np.concatenate([env, tail])


def _phase(freq: np.ndarray, fs: int) -> np.ndarray:
    """The phase accumulator: 2 pi sum f / fs."""
    return 2.0 * np.pi * np.cumsum(freq) / fs


# ----------------------------------------------------------------------- 2. synth, square
def fourier_tone(freq: np.ndarray, fs: int, odd_only: bool = False,
                 brightness: float = 0.6) -> np.ndarray:
    """Band-limited saw (or square), through the low-pass evaluated per harmonic."""
    phi = _phase(freq, fs)
    f_max = float(np.max(freq))
    k_max = int(min(MAX_HARMONICS, np.floor(ALIAS_GUARD * fs / 2.0 / f_max)))
    f_ref = float(np.median(freq))
    f_c = f_ref * (1.0 + 15.0 * brightness)
    out = np.zeros(freq.size)
    for k in range(1, max(k_max, 1) + 1):
        if odd_only and k % 2 == 0:
            continue
        h = 1.0 / np.sqrt(1.0 + (k * f_ref / f_c) ** 4)
        if h / k < 1e-4:            # -80 dB and falling: the rest is inaudible
            break
        out += h * np.sin(k * phi) / k
    return out


# ----------------------------------------------------------------------- 3. pluck
def karplus_strong(f: float, length: int, fs: int, brightness: float = 0.6,
                   t60: float = PLUCK_T60, rng=None) -> np.ndarray:
    """A plucked string at f Hz, `length` samples long (docstring section 4)."""
    rng = rng or np.random.default_rng(_SEED)
    period = fs / f - 0.5                      # samples of delay the loop still needs
    n = max(2, int(np.floor(period)))
    d = float(np.clip(period - n, 0.0, 1.0))   # the fraction: linear interpolation
    g = 10.0 ** (-3.0 / (f * t60))
    burst = rng.uniform(-1.0, 1.0, n)
    smooth = max(1, int(round((1.0 - brightness) * 6)) + 1)   # darker = longer average
    burst = np.convolve(burst, np.ones(smooth) / smooth, mode="same")
    burst -= burst.mean()
    # Loop filter = two-point average * linear-interpolation fractional delay:
    # taps (1-d)/2, 1/2, d/2 at delays n, n+1, n+2  ->  total loop delay n + 1/2 + d.
    taps = (g * (1.0 - d) / 2.0, g * 0.5, g * d / 2.0)
    y = np.zeros(length + n + 2)
    y[:n] = burst[: min(n, y.size)]
    # One block of n samples at a time: y[i:i+n] reads only y[i-n-2 : i] (section 4).
    i = n
    while i < y.size:
        j = min(i + n, y.size)
        for tap, lag in zip(taps, (n, n + 1, n + 2)):
            src = np.arange(i, j) - lag
            y[i:j] += tap * np.where(src >= 0, y[np.maximum(src, 0)], 0.0)
        i = j
    return y[:length]


# ----------------------------------------------------------------------- 4. FM
def fm_tone(f: np.ndarray, fs: int, ratio: float, index0: float, index_decay: float,
            amp_decay: float) -> np.ndarray:
    t = np.arange(f.size) / fs
    phi_c = _phase(f, fs)
    phi_m = _phase(ratio * f, fs)
    index = index0 * np.exp(-t / index_decay)
    return np.exp(-t / amp_decay) * np.sin(phi_c + index * np.sin(phi_m))
