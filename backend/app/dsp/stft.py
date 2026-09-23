"""
Short-Time Fourier Transform core
=================================
The shared engine behind every frequency-domain effect (noise.py, speed_pitch.py,
voice_changer.py).  Hand-written with np.fft.rfft / irfft -- scipy.signal.stft is banned.

Analysis
--------
Cut x into frames of N samples that start every H samples (the hop) and weight each with
a window w before transforming:

        X[t,k] = sum_{n=0}^{N-1}  x[n + tH] * w[n] * e^{-j 2 pi k n / N},   k = 0 .. N/2

(rfft keeps only k = 0..N/2: for a real signal the other half is the complex conjugate.)
Defaults: N = 2048, H = N/4 = 512 -> 75 % overlap, ~21.5 Hz per bin at 44.1 kHz.

The window is the PERIODIC Hann

        w[n] = 0.5 * (1 - cos(2 pi n / N)),     n = 0 .. N-1

(not np.hanning, which is the symmetric N-1 version) because the periodic one is the one
whose shifted copies sum to a constant -- see COLA below.

Synthesis -- weighted overlap-add (WOLA)
----------------------------------------
Invert each frame, window it AGAIN (the synthesis window), and add the frames back at
their original offsets:

        y_t[n] = irfft(Y[t,:])[n] * w[n]
        out[m] = sum_t y_t[m - tH]
        x^[m]  = out[m] / sum_t w^2[m - tH]

If Y = X (nothing modified) then y_t[n] = x[n + tH] * w^2[n], so the numerator is
x[m] * sum_t w^2[m - tH] and the division returns x exactly.  Dividing by the window-
square sum is also the least-squares optimal reconstruction when Y HAS been modified
(Griffin & Lim, 1984) -- which is the case every effect relies on.

COLA and the constant 1.5
-------------------------
For a Hann window, sum_t w^2[m - tH] is a CONSTANT (the "constant overlap-add" condition)
when H = N/4:

        w^2 = 0.25 (1 - cos)^2 = 0.375 - 0.5 cos(theta) + 0.125 cos(2 theta)

The cosine terms cancel when four copies are spaced a quarter period apart, leaving
4 * 0.375 = 1.5.  So in the steady middle of the file the divisor is 1.5 everywhere.

The edge floor
--------------
In the first and last N - H samples fewer frames overlap, and sum w^2 tapers to 0.  For
an unmodified spectrum numerator and denominator vanish together, but once an effect has
changed the magnitudes the numerator no longer tracks the denominator and the ratio
explodes into a click.  istft() floors the divisor at half its steady-state value
(0.75 for Hann at N/4): the interior is untouched and the edges fade gently instead.

princarg
--------
Phase is only known modulo 2 pi.  The phase vocoder needs the SMALLEST consistent phase
difference between frames, which is

        princarg(p) = p - 2 pi * round(p / 2 pi)        in (-pi, pi]

Identity phase locking -- shared by speed_pitch.py and voice_changer.py
-----------------------------------------------------------------------
Source: J. Laroche and M. Dolson, "Improved phase vocoder time-scale modification of
audio", IEEE Trans. SAP 7(3), 1999.

A windowed sinusoid does not occupy one bin: it is a LOBE several bins wide (4 for Hann),
and within a lobe the bins' phases have a fixed relationship -- that is what makes them
add up to one clean partial.  A plain phase vocoder advances every bin's phase
independently, so after a few frames the bins of a lobe have drifted apart; they then
partly cancel in the inverse DFT.  You hear it as "phasiness" (a metallic, distant sound)
and see it as a quieter, spikier output.

The fix: treat each spectral PEAK as the sinusoid and let only the peaks run the phase
vocoder recursion.  Every other bin inherits its peak's new phase plus the phase offset
it had from that peak in the ANALYSIS frame:

        peaks        P_t    = { k : |X[t,k]| > |X[t,k-1]|  and  |X[t,k]| >= |X[t,k+1]| }
        region of p  R(p)   = the bins nearer to p than to any other peak
        psi[t,p]            = (the usual recursion, only for p in P_t)
        psi[t,k]            = psi[t,p] + ( phi[t,k] - phi[t,p] ),      k in R(p)

so the shape of each lobe -- its phase profile -- is copied unchanged from the input.
spectral_peaks() and peak_regions() below compute P_t and R.
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

def spectral_peaks(mag_frame: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    """Indices of the local maxima of one frame's magnitude spectrum (docstring: P_t).

    Bins below `floor` times the frame's maximum are ignored, so digital silence and the
    numerical dust around it do not become "partials".
    """
    m = mag_frame
    if m.size < 3 or m.max() <= 0.0:
        return np.empty(0, dtype=int)
    inner = (m[1:-1] > m[:-2]) & (m[1:-1] >= m[2:]) & (m[1:-1] > floor * m.max())
    return np.flatnonzero(inner) + 1


def peak_regions(n_bins: int, peaks: np.ndarray) -> np.ndarray:
    """For every bin, the index INTO `peaks` of the nearest peak (docstring: R(p)).

    Region boundaries sit halfway between neighbouring peaks, so searchsorted over those
    midpoints hands each bin the peak whose region it falls in.
    """
    midpoints = (peaks[:-1] + peaks[1:]) / 2.0
    return np.searchsorted(midpoints, np.arange(n_bins), side="right")
