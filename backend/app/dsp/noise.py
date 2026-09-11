"""
Noise Remover -- spectral subtraction (implemented by hand)
===========================================================
"""

from __future__ import annotations

import numpy as np

from .stft import DEFAULT_HOP, DEFAULT_N_FFT, istft, stft


def noise_reduce(
    x: np.ndarray,
    fs: int,
    amount: float = 1.5,
    floor: float = 0.05,
    noise_start: float = 0.0,
    noise_end: float = 0.5,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
) -> np.ndarray:
    """Spectral-subtraction denoiser.

    amount      -- over-subtraction factor alpha (1.0 = subtract exactly the estimate)
    floor       -- spectral floor as a fraction of the original magnitude
    noise_start / noise_end -- seconds; the reference segment assumed to be noise only
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x

    spec = stft(x, n_fft, hop)              # (frames, bins) complex
    mag = np.abs(spec)                      # |X[t,k]|
    phase = np.angle(spec)                  # angle(X[t,k]) -- preserved untouched

    # --- Which frames form the noise reference? -----------------------------------
    # Frame t covers samples [t*hop, t*hop + n_fft), so its start time is t*hop/fs.
    frame_times = np.arange(spec.shape[0]) * hop / fs
    ref = (frame_times >= noise_start) & (frame_times < noise_end)
    if not ref.any():           # window fell outside the file -> fall back to frame 0
        ref[0] = True

    noise_profile = mag[ref].mean(axis=0)   # N[k], one value per frequency bin

    # --- Subtract, with the floor ---------------------------------------------------
    clean_mag = mag - amount * noise_profile[None, :]
    clean_mag = np.maximum(clean_mag, floor * mag)

    # --- Recombine with the ORIGINAL phase and invert -------------------------------
    clean_spec = clean_mag * np.exp(1j * phase)
    return istft(clean_spec, hop=hop, n_fft=n_fft, length=x.size)