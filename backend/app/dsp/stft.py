"""
Short-Time Fourier Transform core
=================================
"""

from __future__ import annotations

import numpy as np

# Defaults used across every STFT-based effect.  2048 samples is ~46 ms at 44.1 kHz:
# long enough for decent frequency resolution (~21 Hz per bin), short enough that
# transients are not smeared badly.
DEFAULT_N_FFT = 2048
DEFAULT_HOP = 512  # N/4 -> 75% overlap


def make_window(n_fft: int) -> np.ndarray:
    """Periodic-ish Hann window used for BOTH analysis and synthesis.

    np.hanning(N) is the symmetric window (first and last sample are 0).  For overlap-add
    the periodic version is the textbook choice, so we build it as 0.5*(1 - cos(2*pi*n/N)).
    """
    n = np.arange(n_fft)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / n_fft))


def frame_signal(x: np.ndarray, n_fft: int = DEFAULT_N_FFT, hop: int = DEFAULT_HOP) -> np.ndarray:
    """Cut `x` into overlapping windowed frames -> shape (n_frames, n_fft).

    The tail is zero-padded so the last partial frame survives.  We build the frame index
    matrix explicitly (outer sum of frame starts and in-frame offsets) instead of using a
    stride trick, because it is easier to read and the copy cost is irrelevant here.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < n_fft:  # pad short inputs up to one full frame
        x = np.pad(x, (0, n_fft - x.size))

    # number of frames such that the last frame start is still inside the signal
    n_frames = 1 + int(np.ceil((x.size - n_fft) / hop))
    padded_len = (n_frames - 1) * hop + n_fft
    x = np.pad(x, (0, padded_len - x.size))

    starts = np.arange(n_frames) * hop          # (n_frames,)  where each frame begins
    offsets = np.arange(n_fft)                  # (n_fft,)     position inside a frame
    idx = starts[:, None] + offsets[None, :]    # (n_frames, n_fft) absolute sample index

    return x[idx] * make_window(n_fft)[None, :]  # apply the analysis window


def stft(x: np.ndarray, n_fft: int = DEFAULT_N_FFT, hop: int = DEFAULT_HOP) -> np.ndarray:
    """Forward STFT -> complex array of shape (n_frames, n_fft//2 + 1)."""
    frames = frame_signal(x, n_fft, hop)
    # rfft along the frame axis: one DFT per frame.
    return np.fft.rfft(frames, n=n_fft, axis=1)


def istft(
    spec: np.ndarray,
    hop: int = DEFAULT_HOP,
    n_fft: int | None = None,
    length: int | None = None,
) -> np.ndarray:
    """Inverse STFT by weighted overlap-add (see the module docstring for the derivation)."""
    if n_fft is None:
        # rfft of an N-point frame has N/2+1 bins  ->  N = 2*(bins-1)
        n_fft = 2 * (spec.shape[1] - 1)

    frames = np.fft.irfft(spec, n=n_fft, axis=1)   # back to the time domain, real output
    window = make_window(n_fft)
    frames = frames * window[None, :]              # synthesis window (the second one)

    n_frames = frames.shape[0]
    out_len = (n_frames - 1) * hop + n_fft
    out = np.zeros(out_len, dtype=np.float64)
    wsum = np.zeros(out_len, dtype=np.float64)     # accumulates sum_t w^2[n - t*H]

    for t in range(n_frames):
        start = t * hop
        out[start:start + n_fft] += frames[t]
        wsum[start:start + n_fft] += window ** 2

    # Divide out the window-square envelope.
    #
    # In the steady middle of the signal wsum settles at a constant (1.5 for a Hann window
    # at 75% overlap), but in the first and last half-frame only one tapering window
    # contributes, so wsum falls towards 0.  Dividing by that tiny number is fine for an
    # UNMODIFIED spectrum (numerator and denominator vanish together) but explodes as soon
    # as an effect has changed the magnitudes -- you get a loud click at the file edges.
    # So the divisor is floored at half the steady-state value: the edges come out gently
    # faded instead of amplified, and the interior is untouched.
    floor = 0.5 * np.median(wsum[wsum > 0]) if np.any(wsum > 0) else 1.0
    out = out / np.maximum(wsum, floor)

    if length is not None:
        if out.size < length:
            out = np.pad(out, (0, length - out.size))
        out = out[:length]
    return out


def principal_argument(phase: np.ndarray) -> np.ndarray:
    """Wrap a phase (in radians) into (-pi, +pi].

    Phase differences between frames are only known modulo 2*pi.  The phase vocoder needs
    the *smallest* consistent difference, which is what this "princarg" mapping gives:

        princarg(p) = p - 2*pi*round(p / (2*pi))
    """
    return phase - 2.0 * np.pi * np.round(phase / (2.0 * np.pi))