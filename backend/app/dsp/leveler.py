"""
Voice Leveler -- ride the gain so every stretch of speech sits at one level
==========================================================================
In plain words: a sound engineer "rides the fader" -- turns a quiet speaker up and a
shouted phrase down, slowly, so the listener never reaches for the volume.  This does the
same, phrase by phrase.  A compressor cannot: it reacts to peaks within milliseconds and
only above a threshold, so a whole sentence spoken 15 dB too quietly stays 15 dB too quiet.

Notation: frames t every 10 ms (speech_analysis.analyse), frame power p[t] (from its
energy in dB), speech mask s[t] (speech_analysis.speech_mask: voiced frames, widened a
little).

1. Noise floor.  The room's level is the median power of the non-speech frames,
        n = median{ p[t] : s[t] = 0 }            (0 when every frame is speech)
   and speech power is what is left over it:  q[t] = max(p[t] - n, 1e-3 p[t]).
   Without this a quiet phrase in a noisy room reads louder than it is.

2. Local speech level, over a window of W frames (WINDOW seconds), speech frames only:
        L[t] = 10 log10( sum_{|u-t|<W/2} s[u] q[u]  /  sum_{|u-t|<W/2} s[u] )
   defined where the window holds at least a quarter of speech.

3. Target and gain.  The target is the 75th percentile of L over the speech frames -- the
   file's own "normal loud" -- so quiet stretches come up and shouted ones come down to it:
        g[t] = amount * clip(target - L[t], -max_gain, +max_gain)          (dB)
   Where L is not defined (pauses), g is interpolated between the neighbouring speech,
   so the gain changes inside the pauses rather than on the words.  g is then smoothed
   over SMOOTH seconds and interpolated to every sample:  y[n] = x[n] 10^(g(n)/20).

A file with no speech at all comes back unchanged: there is nothing to level.  Loudness
over full scale is left to the recipe's final fit (dsp_engine), which turns the whole
buffer down rather than clipping.
"""

from __future__ import annotations

import numpy as np

from .speech_analysis import SpeechFrames, analyse, speech_mask

WINDOW = 1.0        # seconds of the local speech level (step 2): about a phrase
SMOOTH = 0.3        # seconds of gain smoothing
TARGET_PERCENTILE = 75.0


def _moving_sum(x: np.ndarray, width: int) -> np.ndarray:
    """Centred running sum over `width` frames, the same length as x (zeros past the
    ends), even when the window is longer than the clip."""
    left = width // 2
    csum = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(x.size)
    return csum[np.clip(idx + width - left, 0, x.size)] - csum[np.clip(idx - left, 0, x.size)]


def speech_levels(frames: SpeechFrames) -> tuple[np.ndarray, np.ndarray]:
    """(L[t] in dB -- NaN where undefined, speech mask s[t]): steps 1-2 of the module
    docstring.  The Signal Doctor measures unevenness with this same function, so what
    it reports is exactly what this card corrects."""
    speech = speech_mask(frames, 0.1)
    level = np.full(frames.times.size, np.nan)
    if np.count_nonzero(speech) < 10:
        return level, speech
    power = 10.0 ** (frames.energy_db / 10.0)
    noise = float(np.median(power[~speech])) if np.any(~speech) else 0.0
    clean = np.maximum(power - noise, 1e-3 * power)

    width = max(1, int(round(WINDOW * frames.fs / frames.hop)))
    weight = _moving_sum(speech.astype(np.float64), width)
    total = _moving_sum(np.where(speech, clean, 0.0), width)
    defined = weight >= 0.25 * width
    level[defined] = 10.0 * np.log10(total[defined] / weight[defined] + 1e-30)
    return level, speech


def voice_gain_db(x: np.ndarray, fs: int, amount: float = 1.0,
                  max_gain: float = 18.0) -> tuple[np.ndarray, np.ndarray] | None:
    """(frame times, gain in dB per frame), or None when there is no speech to level."""
    frames = analyse(x, fs)
    level, speech = speech_levels(frames)
    defined = np.isfinite(level)
    if not np.any(defined & speech):
        return None
    hop_s = frames.hop / frames.fs

    target = float(np.percentile(level[defined & speech], TARGET_PERCENTILE))
    gain = np.full(level.size, np.nan)
    gain[defined] = amount * np.clip(target - level[defined], -max_gain, max_gain)
    idx = np.arange(gain.size)
    gain = np.interp(idx, idx[defined], gain[defined])        # bridge the pauses
    smooth = max(1, int(round(SMOOTH / hop_s)))
    padded = np.pad(gain, (smooth // 2, smooth - 1 - smooth // 2), mode="edge")
    gain = np.convolve(padded, np.ones(smooth) / smooth, mode="valid")
    return frames.times, gain


def level_voice(x: np.ndarray, fs: int, amount: float = 1.0,
                max_gain: float = 18.0) -> np.ndarray:
    """Every stretch of speech brought to the file's own loud level (module docstring)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or amount <= 0.0:
        return x
    result = voice_gain_db(x, fs, amount, max_gain)
    if result is None:
        return x
    times, gain = result
    per_sample = np.interp(np.arange(x.size) / fs, times, gain)
    return x * 10.0 ** (per_sample / 20.0)
