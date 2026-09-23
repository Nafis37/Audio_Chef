"""
Spectrogram -- the picture under each waveform
==============================================
In plain words
--------------
A waveform shows HOW LOUD the audio is over time, but not WHICH frequencies are loud.
Most effects here (EQ, noise removal, pitch, whisper, robot) change the second and barely
the first.  A spectrogram is time left-to-right, pitch bottom-to-top, and brightness =
loudness: a voice shows stacked horizontal stripes (its harmonics), hiss shows a grey
haze, and a bass boost lights up the bottom band.

It is the same STFT every frequency-domain effect already uses (stft.py), displayed
instead of modified.  No inverse is taken, so the COLA condition does not apply and the
hop can be as coarse as the picture needs.

1. Magnitudes in dBFS -- where "0 dB = a full-scale sine"
---------------------------------------------------------
A sine of amplitude A exactly on bin k0 under a periodic Hann window of length N gives

        |X[k0]| = A * (1/2) * sum_n w[n] = A * (1/2) * (N/2) = A N / 4

(the 1/2 is the sine splitting between +k0 and -k0; a Hann window averages to 1/2).  So

        L[t,k] = 20 log10( |X[t,k]| / (N/4) )           dBFS

puts a full-scale sine at 0 dB, whatever N is.  The picture maps a FIXED range,
-90 .. 0 dBFS, to 0 .. 255 -- never normalised per image -- so an output that got louder
really is brighter than its input, and a denoised output really is darker in the pauses.

2. A log-frequency axis
-----------------------
The ear hears in ratios (an octave is x2 at any height), but FFT bins are spaced linearly:
at 44.1 kHz and N = 2048, 40-80 Hz spans 2 bins while 8-16 kHz spans 372.  Drawn
linearly, everything below 1 kHz -- all of the bass and most of a voice -- would be a thin
strip at the bottom.  So the picture has R rows spaced geometrically:

        f_r = f_min * (f_max / f_min)^(r / (R - 1)),       r = 0 .. R-1

Each row is the MEAN POWER of the bins between the geometric midpoints of its
neighbours (a band), computed with one cumulative sum per frame:

        C[t,k]    = sum_{i<k} |X[t,i]|^2
        P[t,r]    = (C[t,hi_r] - C[t,lo_r]) / (hi_r - lo_r)

Averaging power, not picking one bin, is what keeps a 400 Hz-wide top row from missing
the energy between two sampled bins.  Low rows narrower than one bin just repeat it.

3. Resolution
-------------
The hop is chosen so a clip of any length gives at most MAX_COLUMNS frames; the canvas
stretches the columns to the waveform's width.
"""

from __future__ import annotations

import numpy as np

from .stft import stft

N_FFT = 2048
MIN_HOP = N_FFT // 4
MAX_COLUMNS = 600
ROWS = 160
F_MIN = 40.0
F_MAX = 16_000.0
DB_FLOOR = -90.0     # black
DB_CEIL = 0.0        # white: a full-scale sine
EPS = 1e-20


def spectrogram_image(x: np.ndarray, fs: int) -> tuple[np.ndarray, dict]:
    """uint8 image of shape (ROWS, cols); row 0 is the HIGHEST frequency (top of the canvas)."""
    x = np.asarray(x, dtype=np.float64)
    f_max = min(F_MAX, fs / 2.0)
    meta = {"rows": ROWS, "f_min": F_MIN, "f_max": f_max,
            "db_floor": DB_FLOOR, "db_ceil": DB_CEIL}
    if x.size == 0:
        return np.zeros((ROWS, 1), dtype=np.uint8), {**meta, "cols": 1}

    hop = max(MIN_HOP, int(np.ceil(x.size / MAX_COLUMNS)))
    power = np.abs(stft(x, N_FFT, hop)) ** 2 / (N_FFT / 4.0) ** 2   # (frames, bins), section 1

    # Section 2: band edges at the geometric midpoints between row centres, in bins.
    centres = F_MIN * (f_max / F_MIN) ** (np.arange(ROWS) / (ROWS - 1))
    edges_hz = np.concatenate((
        [centres[0] / np.sqrt(centres[1] / centres[0])],
        np.sqrt(centres[:-1] * centres[1:]),
        [centres[-1] * np.sqrt(centres[-1] / centres[-2])],
    ))
    n_bins = power.shape[1]
    edges = edges_hz * N_FFT / fs                                   # Hz -> fractional bin
    lo = np.clip(np.floor(edges[:-1]).astype(int), 0, n_bins - 1)
    hi = np.clip(np.maximum(lo + 1, np.ceil(edges[1:]).astype(int)), 1, n_bins)

    cumulative = np.concatenate((np.zeros((power.shape[0], 1)), np.cumsum(power, axis=1)), axis=1)
    band = (cumulative[:, hi] - cumulative[:, lo]) / (hi - lo)[None, :]   # (frames, rows)

    level_db = 10.0 * np.log10(band + EPS)                          # power -> dB
    scaled = (level_db - DB_FLOOR) / (DB_CEIL - DB_FLOOR)           # fixed range -> 0..1
    image = np.round(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)

    # (frames, rows) -> (rows, frames), highest frequency first so row 0 draws at the top.
    image = np.ascontiguousarray(image.T[::-1])
    return image, {**meta, "cols": int(image.shape[1])}
