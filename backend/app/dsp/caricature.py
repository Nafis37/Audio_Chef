"""
Voice caricatures -- pitch, formants, tempo and rasp, moved separately
======================================================================
In plain words: an impression of a famous voice is a handful of broad strokes -- how
high the voice sits, how big the speaker's throat sounds, how fast they talk, and how
rough it is.  Each stroke is a separate DSP step below, and each preset is just four
numbers (plus an optional EQ bell).  These are CARICATURES: the speaker's own words,
rhythm and accent are untouched, so the result is at best a loose impression, never a
clone.  Real resemblance needs a recording of the target -- that is Voice Match.

Why chipmunk sounds like a chipmunk, and how to avoid it
--------------------------------------------------------
voice_changer.shift_spectrum() moves every spectral lobe by the ratio r.  That moves the
harmonics (pitch) AND the envelope that shapes them (the formants -- the resonances of the
vocal tract), which is the cartoon sound.  A person with a higher voice has higher
harmonics but formants set by their throat size.  So the two are separated:

    r = 2^(semitones / 12)         pitch ratio (harmonics)
    a = formant ratio              envelope ratio (a > 1: smaller throat, brighter)

  1. shift:     S = STFT( ISTFT( shift_spectrum(X, r) ) )
                (re-analysed: the shifted STFT is not a consistent one, so its envelope
                 is not what the output will contain -- the trick voice_match.py uses)
  2. envelopes: E_in[t,f] = cepstral_envelope_db(|X|)    peak-picked at f0
                E_sh[t,f] = cepstral_envelope_db(|S|)    peak-picked at r * f0
  3. target:    E_tg[t,f] = E_in[t, f / a]
                (the input's own envelope, stretched along frequency by a: a formant
                 at F moves to aF)
  4. correct:   Y[t,f] = S[t,f] * 10^( (E_tg[t,f] - E_sh[t,f]) / 20 )

With a = r the correction is ~0 dB (the shift already moved the formants by r): plain
chipmunk.  With a = 1 it undoes the formant movement: a pure pitch change.

  5. level:     every frame is rescaled to the input frame's energy,
                sum_f |Y[t,f]|^2 = sum_f |X[t,f]|^2
                (the envelope gain must change the colour, not the loudness)

Rasp (grit g in 0..1)
---------------------
A rough voice is an unsteady one: its pitch jitters from cycle to cycle and there is
breath noise under it.  Both are added:

    r_t  = r * 2^( g * JITTER_ST * n_t / 12 )      n_t = seeded noise, smoothed over 3 frames
    y   <- y + g * BREATH * whisper(y)             (voice_changer.whisperise: same
                                                    envelope, random phase = breath)

Tempo
-----
    y <- time_stretch(y, rate = 1 / tempo)         (phase vocoder, pitch unchanged;
                                                   tempo 1.1 = 10 % faster)
Colour
------
One optional RBJ peaking bell (eq.peaking_coefficients) for a nasal or chesty tint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .eq import peaking_coefficients
from .speech_analysis import analyse, cepstral_envelope_db, sample_at
from .speed_pitch import time_stretch
from .stft import istft, stft
from .voice_changer import shift_spectrum, whisperise

JITTER_ST = 0.6           # semitones of pitch wobble at full grit
BREATH = 0.12             # breath-noise level at full grit, relative to the voice
CORRECTION_DB = 24.0      # clamp on the envelope correction (step 4) in either direction
_SEED = 0xCA41CA7         # fixed: the same input always bakes to the same output


@dataclass(frozen=True)
class Caricature:
    label: str
    semitones: float        # pitch
    formant: float          # a: envelope ratio
    tempo: float            # speaking rate (1 = unchanged)
    grit: float             # 0..1 rasp
    bell_hz: float = 0.0    # optional peaking bell (0 = none)
    bell_db: float = 0.0
    bell_q: float = 1.2


# Loose impressions, tuned by ear.  Each is a direction to move ANY voice in, from the
# public's broad image of the speaker -- none is measured from, or claims to reproduce,
# the real person's voice.
CARICATURES: dict[str, Caricature] = {
    # Bright, energetic, quick: a little higher, a slightly smaller-sounding tract.
    "ronaldo": Caricature("Ronaldo (caricature)", semitones=2.0, formant=1.05,
                          tempo=1.1, grit=0.15, bell_hz=3000.0, bell_db=3.0),
    # Soft and light: higher, smoothest, no push.
    "messi": Caricature("Messi (caricature)", semitones=3.0, formant=1.08,
                        tempo=1.0, grit=0.0, bell_hz=5000.0, bell_db=-3.0),
    # Low, slow, emphatic and nasal: lower pitch, bigger tract, some rasp, a nasal bell.
    "trump": Caricature("Trump (caricature)", semitones=-3.0, formant=0.93,
                        tempo=0.9, grit=0.45, bell_hz=1200.0, bell_db=5.0, bell_q=1.5),
}


def _stft_size(fs: int) -> tuple[int, int]:
    """~40 ms frames (a power of two), 75 % overlap -- the voice_match.py grid."""
    n_fft = 256
    while n_fft < 0.040 * fs:
        n_fft *= 2
    return n_fft, n_fft // 4


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    return np.convolve(x, np.ones(width) / width, mode="same")


def pitch_formant_shift(
    x: np.ndarray, fs: int, semitones: float, formant: float, grit: float = 0.0,
) -> np.ndarray:
    """Steps 1-5 of the module docstring: pitch by r, formants by a, independently."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    n_fft, hop = _stft_size(fs)
    spec = stft(x, n_fft, hop)
    n_frames = spec.shape[0]

    # Per-frame pitch ratio, with the rasp's jitter.
    r = 2.0 ** (semitones / 12.0)
    noise = _smooth(np.random.default_rng(_SEED).standard_normal(n_frames), 3)
    ratios = r * 2.0 ** (grit * JITTER_ST * noise / 12.0)

    # f0 on the STFT grid, for the envelope's peak picking (0 = unvoiced -> default spacing).
    frames = analyse(x, fs)
    t_stft = (np.arange(n_frames) * hop + n_fft / 2.0) / fs
    voiced_u = np.interp(t_stft, frames.times, frames.voiced.astype(float)) >= 0.5
    if np.any(frames.voiced):
        f0_u = np.where(voiced_u, np.interp(t_stft, frames.times[frames.voiced],
                                            frames.f0[frames.voiced]), 0.0)
    else:
        f0_u = np.zeros(n_frames)

    # 1. shift, then re-analyse.
    moved = istft(shift_spectrum(spec, ratios, n_fft, hop), hop=hop, n_fft=n_fft,
                  length=x.size)
    shifted = stft(moved, n_fft, hop)

    # 2-4. envelope correction toward the input's envelope stretched by a.
    e_in = cepstral_envelope_db(np.abs(spec), fs, n_fft, f0_u)
    e_sh = cepstral_envelope_db(np.abs(shifted), fs, n_fft, f0_u * ratios)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    e_tg = sample_at(e_in, freqs, freqs / formant)            # E_in[t, f / a]
    gain_db = np.clip(e_tg - e_sh, -CORRECTION_DB, CORRECTION_DB)
    gain_db[np.max(np.abs(shifted), axis=1) <= 1e-9] = 0.0     # empty frame: no envelope
    out = shifted * 10.0 ** (gain_db / 20.0)

    # 5. every frame keeps the input frame's energy.
    e_x = np.sum(np.abs(spec) ** 2, axis=1)
    e_y = np.sum(np.abs(out) ** 2, axis=1)
    level = np.where(e_y > 1e-20, np.sqrt((e_x + 1e-20) / (e_y + 1e-20)), 1.0)
    out *= np.clip(_smooth(level, 3), 0.25, 4.0)[:, None]      # at most +-12 dB
    return istft(out, hop=hop, n_fft=n_fft, length=x.size)


def caricature(x: np.ndarray, fs: int, preset: str) -> np.ndarray:
    """Apply one CARICATURES preset, same length as the input (tempo is applied by the
    caller AFTER the dry/wet mix, since it changes the length)."""
    c = CARICATURES[preset]
    y = pitch_formant_shift(x, fs, c.semitones, c.formant, c.grit)
    if c.grit > 0.0:
        y = y + c.grit * BREATH * whisperise(y)
    if c.bell_hz > 0.0 and abs(c.bell_db) > 1e-9 and c.bell_hz < fs / 2.0:
        b, a = peaking_coefficients(fs, c.bell_hz, c.bell_q, c.bell_db)
        y = lfilter(b, a, y)
    return y


def tempo_of(preset: str) -> float:
    return CARICATURES[preset].tempo if preset in CARICATURES else 1.0


def apply_tempo(x: np.ndarray, tempo: float) -> np.ndarray:
    """Speaking rate without a pitch change (tempo 1.1 = 10 % faster, so shorter)."""
    if abs(tempo - 1.0) < 1e-9 or x.size == 0:
        return x
    return time_stretch(x, rate=1.0 / tempo)
