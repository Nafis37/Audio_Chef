"""
Audio Speed and Pitch Lab -- phase vocoder time stretch + manual resampling
==========================================================================
In plain words
--------------
Speed makes the clip shorter or longer.  With "Keep pitch" on, the voice stays at its own
pitch (like a podcast app's 1.5x button); with it off, the clip is played like a sped-up
tape and the pitch rises with the speed.  Pitch moves every note up or down by a number of
semitones (12 = one octave) without changing the length.

Speed and pitch are coupled on tape: play it faster and it also sounds higher.  Two
building blocks decouple them.  (scipy.signal.resample is banned; both are hand-made.)

1. Resampling by linear interpolation -- the tape effect
--------------------------------------------------------
Read the input at fractional positions  p_m = m * s  (s = speed, input samples per output
sample) and interpolate between neighbours:

        y[m] = (1 - frac) x[i] + frac x[i+1],     i = floor(p_m), frac = p_m - i

(np.interp does exactly this.)  Output length is N / s, and every frequency f becomes
s * f: duration and pitch move together.  Linear interpolation is a crude low-pass
(sinc^2 response), so speeding up far enough aliases a little -- audible only on bright
material at s > 2.

2. Phase-vocoder time stretch -- duration without pitch
-------------------------------------------------------
Analyse with hop H_a, resynthesise with hop H_s = rate * H_a.  Frames placed further
apart make the sound longer; the problem is the phase, which must advance by
omega * H_s (not omega * H_a) between output frames or neighbouring frames cancel.

  (a) Expected phase advance of bin k over one analysis hop:
        dphi_exp[k] = 2 pi k H_a / N
  (b) Measured advance, and its deviation wrapped into (-pi, pi] (princarg, stft.py):
        dev[t,k] = princarg( phi[t,k] - phi[t-1,k] - dphi_exp[k] )
  (c) True instantaneous frequency (rad/sample):
        omega[t,k] = (dphi_exp[k] + dev[t,k]) / H_a
  (d) Accumulate synthesis phase over the NEW hop:
        psi[t,k] = psi[t-1,k] + omega[t,k] * H_s,      psi[0] = phi[0]
  (e) Y[t,k] = |X[t,k]| e^{j psi[t,k]}  ->  istft with hop H_s.

  Steps (b)-(d) run only for the spectral PEAKS of each frame; every other bin is locked to
  its peak (identity phase locking, derived in stft.py).  Without it the bins of each
  partial drift apart and partly cancel: the stretch sounds metallic and comes out a few
  dB quieter than the input.

Magnitudes are untouched, so every partial keeps its frequency; only its duration scales
by H_s / H_a.

3. Pitch shift with constant duration = 2 then 1
------------------------------------------------
For a ratio  r = 2^(semitones / 12):

        stretch by r   (longer, same pitch)      ->  N * r samples
        resample by r  (shorter, r x higher)     ->  N samples, pitch * r

The recipe wrapper: `speed` with preserve_pitch uses (2) with rate = 1/speed; without it,
(1) -- the tape.  `semitones` is then applied with (3).
"""

from __future__ import annotations

import numpy as np

from .stft import (
    DEFAULT_HOP, DEFAULT_N_FFT, istft, peak_regions, principal_argument, spectral_peaks, stft,
)


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

    for t in range(1, n_frames):
        dphi_measured = phi[t] - phi[t - 1]
        # Deviation from the expected advance, wrapped into (-pi, pi] -- this is the step
        # that recovers the true frequency instead of an aliased multiple of it.
        dphi_dev = principal_argument(dphi_measured - dphi_expected)
        omega = (dphi_expected + dphi_dev) / hop_a      # true frequency, rad/sample
        advanced = out_phase[t - 1] + omega * hop_s     # (d): advance over the NEW hop

        peaks = spectral_peaks(mag[t])
        if peaks.size == 0:                             # silent frame: nothing to lock to
            out_phase[t] = advanced
            continue
        # Identity phase locking: only the peaks keep the advanced phase; each other bin
        # takes its peak's new phase plus the offset it had from that peak in the input.
        owner = peaks[peak_regions(n_bins, peaks)]
        out_phase[t] = advanced[owner] + (phi[t] - phi[t, owner])

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