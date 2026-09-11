"""
Phase Vocoder Voice Changer -- pitch shifting by moving frequency bins
"""

from __future__ import annotations

import numpy as np

from .stft import DEFAULT_HOP, DEFAULT_N_FFT, istft, principal_argument, stft

ENVELOPE_SMOOTHING_BINS = 31    # width of the moving average used as the formant envelope


def _spectral_envelope(mag: np.ndarray, width: int = ENVELOPE_SMOOTHING_BINS) -> np.ndarray:
    """Smooth |X| along the frequency axis with a moving average -> the formant envelope."""
    kernel = np.ones(width) / width
    # 'same' keeps the bin count; edges are slightly darker, which is harmless here.
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, mag)


def _shift_bins(values: np.ndarray, ratio: float) -> np.ndarray:
    """Resample an array along the bin axis: out[j] = values[j / ratio] (linear interp)."""
    n_bins = values.shape[-1]
    src = np.arange(n_bins) / ratio           # where each output bin reads from
    bins = np.arange(n_bins)
    out = np.empty_like(values)
    for t in range(values.shape[0]):
        out[t] = np.interp(src, bins, values[t], left=0.0, right=0.0)
    return out


def voice_changer(
    x: np.ndarray,
    fs: int,
    semitones: float = 4.0,
    formant: float = 0.0,
    mode: str = "pitch",
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
) -> np.ndarray:
    """Duration-preserving pitch/formant shifter.

    semitones -- pitch shift (12 = one octave up)
    formant   -- independent shift of the spectral envelope, in semitones
    mode      -- "pitch" (normal), "robot" (zeroed phase), "whisper" (random phase)
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x

    spec = stft(x, n_fft, hop)
    mag = np.abs(spec)
    phi = np.angle(spec)
    n_frames, n_bins = spec.shape

    # ---- Simple phase-only modes ---------------------------------------------------
    if mode == "robot":
        return istft(mag.astype(complex), hop=hop, n_fft=n_fft, length=x.size)
    if mode == "whisper":
        rng = np.random.default_rng(0)          # fixed seed -> repeatable bakes
        noise_phase = rng.uniform(-np.pi, np.pi, size=mag.shape)
        return istft(mag * np.exp(1j * noise_phase), hop=hop, n_fft=n_fft, length=x.size)

    ratio = 2.0 ** (float(semitones) / 12.0)            # pitch multiplier r
    formant_ratio = 2.0 ** (float(formant) / 12.0)      # extra envelope multiplier f

    # ---- Analysis: true frequency of every bin (phase vocoder core) ------------------
    k = np.arange(n_bins)
    dphi_expected = 2.0 * np.pi * k * hop / n_fft
    omega = np.zeros_like(mag)                          # rad/sample per bin per frame
    omega[0] = dphi_expected / hop                      # frame 0 has no predecessor
    for t in range(1, n_frames):
        dphi_dev = principal_argument((phi[t] - phi[t - 1]) - dphi_expected)
        omega[t] = (dphi_expected + dphi_dev) / hop

    # ---- Synthesis: move each bin's energy and frequency to its shifted position -----
    syn_mag = np.zeros_like(mag)
    syn_freq = np.zeros_like(omega)
    target = np.round(k * ratio).astype(int)            # j = round(k*r)
    valid = target < n_bins                             # anything past Nyquist is dropped
    src_idx = k[valid]
    dst_idx = target[valid]
    for t in range(n_frames):
        # np.add.at accumulates when several source bins round onto the same target bin.
        np.add.at(syn_mag[t], dst_idx, mag[t, src_idx])
        syn_freq[t, dst_idx] = omega[t, src_idx] * ratio    # the frequency scales too

    # ---- Optional formant correction (see the module docstring) ---------------------
    if abs(formant) > 1e-9:
        envelope = _spectral_envelope(mag)
        env_current = _shift_bins(envelope, ratio)                  # where formants landed
        env_target = _shift_bins(envelope, ratio * formant_ratio)   # where we want them
        # Floor the divisor at a fraction of each frame's loudest envelope value: in bins
        # holding nothing but numerical noise the ratio would otherwise be meaningless and
        # could amplify junk enormously.
        floor = np.maximum(1e-3 * env_current.max(axis=1, keepdims=True), 1e-12)
        correction = env_target / np.maximum(env_current, floor)
        syn_mag = syn_mag * np.clip(correction, 0.0, 8.0)   # keep the boost sane

    # ---- Accumulate the synthesis phase at the NEW frequencies ----------------------
    out_phase = np.zeros_like(phi)
    running = np.zeros(n_bins)
    for t in range(n_frames):
        running = running + syn_freq[t] * hop           # phase gained over one hop
        out_phase[t] = running

    out_spec = syn_mag * np.exp(1j * out_phase)
    return istft(out_spec, hop=hop, n_fft=n_fft, length=x.size)