"""
Voice shift -- pitch and formants, moved separately
===================================================
In plain words: a higher voice is not just the same voice sped up.  Two things set how a
voice sounds: how high it sits (pitch) and how big the speaker's throat is (formants).
The plain phase-vocoder shift (voice_changer.shift_pitch_bins) moves both together, which
is the cartoon sound.  This module moves them independently, for

  * Accurate pitch shift -- pitch by a fixed number of semitones, formants kept
  * Male -> Female / Female -> Male -- pitch moved TO a target, formants follow a part of
    that move

Why chipmunk sounds like a chipmunk, and how to avoid it
--------------------------------------------------------
voice_changer.shift_spectrum() moves every spectral lobe by the ratio r.  That moves the
harmonics (pitch) AND the envelope that shapes them (the formants -- the resonances of the
vocal tract).  A person with a higher voice has higher harmonics but formants set by their
throat size.  So the two are separated:

    r[t]  pitch ratio per frame (harmonics)
    a     formant ratio          envelope ratio (a > 1: smaller throat, brighter)

  1. shift:     S = STFT( ISTFT( shift_spectrum(X, r) ) )
                (re-analysed: the shifted STFT is not a consistent one, so its envelope
                 is not what the output will contain)
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

A target, not an offset (gender swap)
-------------------------------------
A gender preset names WHERE the voice should end up: an average speaking pitch f0_hz and
an intonation spread range_st.  The input is measured first (speech_analysis.analyse:
mean mu_a and std sd_a of ln f0 over voiced frames), and its pitch is mapped onto the
target by matching z-scores:

        l~ = ln(f0_hz) + (sd_b / sd_a) (ln f0 - mu_a),      sd_b = range_st * ln 2 / 12

so a 100 Hz voice and a 200 Hz voice both land around f0_hz.  Only voiced frames move
(short unvoiced gaps are bridged, the rest fades in and out over RAMP_S); the ratio is
clamped to an octave either way.

Formants follow a part of the pitch move:

        a = (f0_hz / f0_a)^FORMANT_FOLLOW          clipped to [A_MIN, A_MAX]

120 Hz -> 210 Hz gives a ~ 1.18 and 210 Hz -> 115 Hz gives a ~ 0.83 -- roughly the
difference between typical adult vocal tracts -- while a voice already near the target
keeps its own throat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .speech_analysis import SpeechFrames, analyse, cepstral_envelope_db, sample_at
from .stft import istft, stft
from .voice_changer import shift_spectrum

CORRECTION_DB = 24.0      # clamp on the envelope correction (step 4) in either direction
FORMANT_FOLLOW = 0.3      # formants follow this power of the pitch move (docstring)
A_MIN, A_MAX = 0.8, 1.25
GAP_S = 0.060             # unvoiced gaps shorter than this are bridged
RAMP_S = 0.030            # voicing fade in / out
MIN_SD = 0.02             # ~0.35 semitone: below this, mean shift only
RATIO_MIN, RATIO_MAX = 0.5, 2.0


@dataclass(frozen=True)
class GenderTarget:
    label: str
    f0_hz: float            # target average speaking pitch
    range_st: float         # target intonation spread (std of pitch, semitones)


GENDER_TARGETS: dict[str, GenderTarget] = {
    "male_to_female": GenderTarget("Male → Female", f0_hz=210.0, range_st=3.0),
    "female_to_male": GenderTarget("Female → Male", f0_hz=115.0, range_st=2.2),
}


def _stft_size(fs: int) -> tuple[int, int]:
    """~40 ms frames (a power of two), 75 % overlap."""
    n_fft = 256
    while n_fft < 0.040 * fs:
        n_fft *= 2
    return n_fft, n_fft // 4


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    return np.convolve(x, np.ones(width) / width, mode="same")


def _moving_average(x: np.ndarray, width: int) -> np.ndarray:
    """Centred moving average with edge padding (so the ends are not pulled to 0)."""
    if width <= 1:
        return x
    padded = np.pad(x, (width // 2, width - 1 - width // 2), mode="edge")
    csum = np.cumsum(np.concatenate([[0.0], padded]))
    return (csum[width:] - csum[:-width]) / width


def _fill_short_gaps(known: np.ndarray, max_gap: int) -> np.ndarray:
    """Mark unknown runs of at most max_gap frames BETWEEN two known frames as known."""
    filled = known.copy()
    edges = np.flatnonzero(np.diff(np.concatenate([[0], (~known).astype(np.int8), [0]])))
    for a, b in zip(edges[0::2], edges[1::2]):
        if a > 0 and b < known.size and b - a <= max_gap:
            filled[a:b] = True
    return filled


def shift_voice(
    x: np.ndarray, fs: int, frames: SpeechFrames, rho: np.ndarray, formant: float,
) -> np.ndarray:
    """Steps 1-5 of the module docstring: pitch by e^rho[t] (one value per analysis frame
    of `frames`), formants by `formant`, independently."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    n_fft, hop = _stft_size(fs)
    spec = stft(x, n_fft, hop)
    n_frames = spec.shape[0]
    t_stft = (np.arange(n_frames) * hop + n_fft / 2.0) / fs

    # Per-frame pitch ratio on the STFT grid.
    ratios = np.exp(np.interp(t_stft, frames.times, rho))

    # f0 on the STFT grid, for the envelope's peak picking (0 = unvoiced -> default spacing).
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


def pitch_formant_shift(
    x: np.ndarray, fs: int, semitones: float, formant: float,
) -> np.ndarray:
    """A fixed pitch move of `semitones` everywhere, formants by `formant`."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    frames = analyse(x, fs)
    rho = np.full(frames.times.size, semitones * np.log(2.0) / 12.0)
    return shift_voice(x, fs, frames, rho, formant)


def accurate_pitch(x: np.ndarray, fs: int, semitones: float) -> np.ndarray:
    """Pitch by `semitones`, formants where they were: no chipmunk."""
    return pitch_formant_shift(x, fs, semitones, 1.0)


def target_moves(frames: SpeechFrames, target: GenderTarget) -> tuple[np.ndarray, float]:
    """(log pitch ratio per analysis frame, formant ratio a) that take the voice in
    `frames` to `target` -- the "target, not an offset" section of the docstring."""
    n = frames.times.size
    voiced = frames.voiced
    if not np.any(voiced):                        # no voiced speech: nothing to move
        return np.zeros(n), 1.0
    lf_voiced = np.log(frames.f0[voiced])
    mu_a, sd_a = float(np.mean(lf_voiced)), float(np.std(lf_voiced))
    mu_b, sd_b = float(np.log(target.f0_hz)), target.range_st * np.log(2.0) / 12.0

    lf = np.log(np.where(voiced, frames.f0, 1.0))
    goal = lf + (mu_b - mu_a) if sd_a < MIN_SD else mu_b + (sd_b / sd_a) * (lf - mu_a)
    rho = goal - lf

    # Bridge short unvoiced gaps; hold the nearest voiced value elsewhere (it is faded
    # out by the voicing weight, so the held value only matters inside the ramps).
    hop_s = frames.hop / frames.fs
    idx = np.arange(n)
    rho_filled = np.interp(idx, idx[voiced], rho[voiced])
    bridged = _fill_short_gaps(voiced, int(round(GAP_S / hop_s)))
    ramp = max(1, int(round(RAMP_S / hop_s)))
    weight = np.clip(_moving_average(bridged.astype(np.float64), 2 * ramp + 1), 0.0, 1.0)
    rho = np.clip(_moving_average(weight * rho_filled, 5), np.log(RATIO_MIN), np.log(RATIO_MAX))

    follow = (target.f0_hz / float(np.exp(mu_a))) ** FORMANT_FOLLOW
    return rho, float(np.clip(follow, A_MIN, A_MAX))


def gender_swap(x: np.ndarray, fs: int, mode: str) -> np.ndarray:
    """Apply one GENDER_TARGETS preset; same length as the input."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    frames = analyse(x, fs)
    rho, formant = target_moves(frames, GENDER_TARGETS[mode])
    return shift_voice(x, fs, frames, rho, formant)
