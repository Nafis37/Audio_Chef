"""
Signal measurements -- the numbers shown next to the two waveforms
==================================================================
measure() runs on the input and on the output of every bake; ListenStats shows the two
side by side so each effect's influence is a number, not just a picture.

For a buffer x[0 .. N-1] at sample rate fs:

        peak      = max |x[n]|
        RMS       = sqrt( (1/N) sum x[n]^2 )             (a sine of amplitude A: A / sqrt 2)
        dBFS(v)   = 20 log10(v)                          (0 dB = full scale, floored -120)
        crest     = peak_dB - RMS_dB = 20 log10(peak / RMS)
                    (sine: 3.01 dB; a compressor pushes it down, a transient pushes it up)
        DC        = (1/N) sum x[n]                       (the 0 Hz component)
        ZCR       = #{ n : sign x[n] != sign x[n-1] } * fs / N
                    (sign changes per second; a pure tone of f Hz gives 2f)
        centroid  = sum_k f_k |X[k]|  /  sum_k |X[k]|    (X = rfft of the whole buffer)
                    ("centre of mass" of the spectrum -- where the brightness sits;
                    an EQ boost at f0 pulls it toward f0)
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12         # keeps log10(0) finite
FLOOR_DB = -120.0   # anything quieter than this is reported as "silence"


def to_db(value: float) -> float:
    """Amplitude ratio -> dBFS, floored so digital silence does not return -inf."""
    return float(max(FLOOR_DB, 20.0 * np.log10(max(float(value), EPS))))


def spectral_centroid(x: np.ndarray, fs: int) -> float:
    """Magnitude-weighted mean frequency of the whole buffer, in Hz (0 for silence)."""
    spectrum = np.abs(np.fft.rfft(x))            # one transform, no windowing/framing
    total = float(np.sum(spectrum))
    if total <= EPS:                             # silent input -> the mean is undefined
        return 0.0
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs)    # bin index -> Hz, so the result is in Hz
    return float(np.sum(freqs * spectrum) / total)


def zero_crossing_rate(x: np.ndarray, fs: int) -> float:
    """Sign changes per second.  A pure tone of f Hz gives ~2f."""
    if x.size < 2:
        return 0.0
    signs = np.signbit(x)                        # bool array; True where the sample is < 0
    crossings = int(np.count_nonzero(signs[1:] != signs[:-1]))
    return float(crossings * fs / x.size)


def measure(x: np.ndarray, fs: int) -> dict[str, float]:
    """Every scalar the Listen panel shows for one buffer, rounded for a JSON header."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:                              # a recipe can legitimately empty the buffer
        return {"duration": 0.0, "frames": 0, "peak": 0.0, "peak_db": FLOOR_DB,
                "rms": 0.0, "rms_db": FLOOR_DB, "crest_db": 0.0, "dc": 0.0,
                "zcr": 0.0, "centroid_hz": 0.0}

    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x * x)))         # sqrt of mean power, not mean amplitude
    peak_db, rms_db = to_db(peak), to_db(rms)

    return {
        "duration": round(x.size / fs, 4),
        "frames": int(x.size),
        "peak": round(peak, 6),
        "peak_db": round(peak_db, 2),
        "rms": round(rms, 6),
        "rms_db": round(rms_db, 2),
        # Difference of logs = log of the ratio peak/rms; 0 when both are at the floor.
        "crest_db": round(max(0.0, peak_db - rms_db), 2),
        "dc": round(float(np.mean(x)), 6),
        "zcr": round(zero_crossing_rate(x, fs), 1),
        "centroid_hz": round(spectral_centroid(x, fs), 1),
    }