"""
Audio Speed and Pitch Lab -- phase vocoder time stretch + manual resampling
==========================================================================
"""

from __future__ import annotations

import numpy as np

from .stft import DEFAULT_HOP, DEFAULT_N_FFT, istft, principal_argument, stft


def resample(x: np.ndarray, speed: float) -> np.ndarray:
    """Read the signal at `speed` input samples per output sample (linear interpolation).

    speed > 1 -> shorter and higher pitched;  speed < 1 -> longer and lower pitched.
    """
    x = np.asarray(x, dtype=np.float64)
    speed = max(float(speed), 1e-6)
    if x.size == 0 or abs(speed - 1.0) < 1e-9:
        return x
    n_out = int(np.floor(x.size / speed))
    if n_out < 2:
        return x[:1].copy()
    positions = np.arange(n_out) * speed          # fractional read positions p_m
    return np.interp(positions, np.arange(x.size), x)


def time_stretch(
    x: np.ndarray,
    rate: float = 1.0,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
) -> np.ndarray:
    """Phase-vocoder time stretch.  rate > 1 makes the audio LONGER (slower), pitch intact."""
    x = np.asarray(x, dtype=np.float64)
    rate = float(rate)
    if x.size == 0 or abs(rate - 1.0) < 1e-6:
        return x

    hop_a = hop                                   # analysis hop H_a
    hop_s = max(1, int(round(hop * rate)))        # synthesis hop H_s

    spec = stft(x, n_fft, hop_a)                  # (frames, bins)
    mag = np.abs(spec)
    phi = np.angle(spec)
    n_frames, n_bins = spec.shape

    k = np.arange(n_bins)
    # Phase each bin is expected to gain over one analysis hop, from its centre frequency.
    dphi_expected = 2.0 * np.pi * k * hop_a / n_fft

    out_phase = np.zeros((n_frames, n_bins))
    out_phase[0] = phi[0]                          # first frame keeps its original phase
    running = phi[0].copy()

    for t in range(1, n_frames):
        dphi_measured = phi[t] - phi[t - 1]
        # Deviation from the expected advance, wrapped into (-pi, pi] -- this is the step
        # that recovers the true frequency instead of an aliased multiple of it.
        dphi_dev = principal_argument(dphi_measured - dphi_expected)
        omega = (dphi_expected + dphi_dev) / hop_a      # true frequency, rad/sample
        running = running + omega * hop_s               # advance over the NEW hop
        out_phase[t] = running

    out_spec = mag * np.exp(1j * out_phase)
    return istft(out_spec, hop=hop_s, n_fft=n_fft)


def pitch_shift(
    x: np.ndarray,
    semitones: float = 0.0,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
) -> np.ndarray:
    """Shift pitch by `semitones` while keeping the original duration (stretch + resample)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or abs(semitones) < 1e-9:
        return x

    ratio = 2.0 ** (float(semitones) / 12.0)       # equal temperament: 12 semitones = 2x
    stretched = time_stretch(x, rate=ratio, n_fft=n_fft, hop=hop)
    shifted = resample(stretched, speed=ratio)

    # Round-off in the framing can leave us a few samples off; force the original length.
    if shifted.size < x.size:
        shifted = np.pad(shifted, (0, x.size - shifted.size))
    return shifted[:x.size]


def speed_pitch(
    x: np.ndarray,
    fs: int,
    speed: float = 1.0,
    semitones: float = 0.0,
    preserve_pitch: bool = True,
) -> np.ndarray:
    """Recipe-facing wrapper.

    preserve_pitch=True  -> speed change via the phase vocoder (duration only)
    preserve_pitch=False -> speed change via resampling (the tape effect: pitch follows)
    `semitones` is then applied on top, always duration-preserving.
    """
    y = np.asarray(x, dtype=np.float64)
    if abs(speed - 1.0) > 1e-9:
        # `speed` is a playback rate, so the duration multiplier for the stretcher is 1/speed.
        y = time_stretch(y, rate=1.0 / speed) if preserve_pitch else resample(y, speed)
    if abs(semitones) > 1e-9:
        y = pitch_shift(y, semitones)
    return y