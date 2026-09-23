"""
Noise Remover -- spectral subtraction (implemented by hand)
===========================================================
In plain words
--------------
Steady background noise (hiss, fan, hum) sits at roughly the same level in every
frequency band all the time; speech comes and goes.  So we measure how loud the noise is
in each band, and in every short slice of audio we turn each band down by that much.
Bands where speech is far louder than the noise barely change; bands that were only
noise drop toward silence.

Source: S. F. Boll, "Suppression of acoustic noise in speech using spectral
subtraction", IEEE Trans. ASSP 27(2), 1979.

Model
-----
The recording is the wanted signal plus additive, stationary noise:

        x[n] = s[n] + d[n]    ->    X[t,k] = S[t,k] + D[t,k]            (STFT, stft.py)

If s and d are uncorrelated their magnitudes roughly add, so the clean magnitude is
estimated by subtracting an estimate of the noise magnitude.

1. Noise profile  N^[k]  -- two ways
------------------------------------
(a) "region": the user marks a stretch that is noise only (noise_start .. noise_end s;
    frame t starts at t*H/fs) and we average it:

        N^[k] = mean_{t in ref} |X[t,k]|

(b) "auto" (the default): no marking needed.  Speech has pauses, so in every band the
    QUIETEST frames are the noise-only ones.  Take a low percentile over time:

        P10[k] = 10th percentile_t |X[t,k]|

    This is a simplified form of "minimum statistics" (R. Martin, "Noise power spectral
    density estimation based on optimal smoothing and minimum statistics", IEEE Trans.
    SAP 9(5), 2001).  The 10th percentile of the noise is smaller than its mean, so it
    is bias-corrected.  The magnitude of a complex Gaussian noise bin is Rayleigh(sigma):

        CDF(m) = 1 - exp(-m^2 / (2 sigma^2))
        P10    = sigma * sqrt(-2 ln 0.9)  = 0.459 sigma
        mean   = sigma * sqrt(pi / 2)     = 1.253 sigma
        N^[k]  = P10[k] * (1.253 / 0.459) = 2.73 * P10[k]

    It needs the noise to be exposed in at least ~10 % of the frames of each band --
    true for speech with any pauses at all; false for a tone that never stops.

2. Over-subtraction with a spectral floor, written as a GAIN
------------------------------------------------------------
        |S^[t,k]| = max( |X[t,k]| - alpha N^[k],   beta |X[t,k]| )
        G[t,k]    = |S^[t,k]| / |X[t,k]|                  (0 < beta <= G <= 1)

alpha (`amount`, "Strength") > 1 subtracts more than the average, because the noise in any
one frame fluctuates ABOVE its mean about half the time.  beta (`floor`) stops a bin going
to zero.

3. Temporal smoothing of the gain -- against "musical noise"
------------------------------------------------------------
Noise bins that randomly survive subtraction next to bins that were zeroed are heard as
isolated warbling tones.  They are random from frame to frame, while speech gains are not,
so averaging each bin's gain over three neighbouring frames shrinks the random spikes
(a 3-tap mean cuts the variance of independent values by 3) and leaves speech almost alone:

        G~[t,k] = ( G[t-1,k] + G[t,k] + G[t+1,k] ) / 3,      then  G~ >= beta

4. Apply to the complex STFT -- the noisy phase is kept
-------------------------------------------------------
        S^[t,k] = G~[t,k] * X[t,k]

A real gain leaves the angle untouched.  The ear is far less sensitive to phase than to
magnitude, and where speech dominates the noisy phase is already close to the clean one.
Back to the time domain with istft() (weighted overlap-add), cut to len(x).
"""

from __future__ import annotations

import numpy as np

from .stft import DEFAULT_HOP, DEFAULT_N_FFT, istft, stft

EPS = 1e-12

# Section 1(b): the percentile and the Rayleigh mean / P10 ratio that de-biases it.
_AUTO_PERCENTILE = 10.0
_RAYLEIGH_MEAN_OVER_P10 = np.sqrt(np.pi / 2.0) / np.sqrt(-2.0 * np.log(0.9))   # ~ 2.73


def noise_profile(
    mag: np.ndarray,
    frame_times: np.ndarray,
    profile: str,
    noise_start: float,
    noise_end: float,
) -> np.ndarray:
    """N^[k], one value per frequency bin (docstring section 1)."""
    if profile == "region":
        ref = (frame_times >= noise_start) & (frame_times < noise_end)
        if not ref.any():           # window fell outside the file -> fall back to frame 0
            ref[0] = True
        return mag[ref].mean(axis=0)

    p10 = np.percentile(mag, _AUTO_PERCENTILE, axis=0)
    return p10 * _RAYLEIGH_MEAN_OVER_P10


def smooth_over_time(gain: np.ndarray) -> np.ndarray:
    """3-frame moving average along the time axis, edges padded by repetition."""
    padded = np.pad(gain, ((1, 1), (0, 0)), mode="edge")
    return (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0


def noise_reduce(
    x: np.ndarray,
    fs: int,
    amount: float = 2.0,
    floor: float = 0.05,
    profile: str = "auto",
    noise_start: float = 0.0,
    noise_end: float = 0.5,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
) -> np.ndarray:
    """Spectral-subtraction denoiser.

    amount   -- over-subtraction factor alpha (1.0 = subtract exactly the estimate)
    floor    -- spectral floor beta, as a fraction of the original magnitude
    profile  -- "auto" (quietest frames) or "region" (noise_start .. noise_end seconds)
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x

    spec = stft(x, n_fft, hop)              # (frames, bins) complex
    mag = np.abs(spec)                      # |X[t,k]|

    # Frame t covers samples [t*hop, t*hop + n_fft), so its start time is t*hop/fs.
    frame_times = np.arange(spec.shape[0]) * hop / fs
    noise = noise_profile(mag, frame_times, profile, noise_start, noise_end)

    # Section 2: subtraction with a floor, expressed as a gain in [floor, 1].
    clean_mag = np.maximum(mag - amount * noise[None, :], floor * mag)
    gain = clean_mag / np.maximum(mag, EPS)

    # Section 3: smooth the gain over time, then re-apply the floor.
    gain = np.maximum(smooth_over_time(gain), floor)

    # Section 4: a real gain on the complex spectrum keeps the original phase.
    return istft(gain * spec, hop=hop, n_fft=n_fft, length=x.size)
