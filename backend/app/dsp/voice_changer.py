"""
Voice Changer -- phase-vocoder bin shifting (+ robot / whisper modes)
=====================================================================
In plain words
--------------
Four characters, all made the same way: slice the voice into short overlapping frames,
keep HOW LOUD each frequency is in each frame (that is what carries the words), and
rebuild the frames with a different PHASE (that is what carries the pitch):

  chipmunk -- every frequency moved UP (default +7 semitones), same length
  monster  -- every frequency moved DOWN (default -7 semitones), same length
  robot    -- every frame restarted on one fixed buzz, so the melody is gone
  whisper  -- random phase, so nothing is periodic any more: breath instead of voice

Chipmunk and monster are mode 1 below with a positive / negative shift; they differ from
the Speed & Pitch Lab (which stretches then resamples) in that they move the spectrum
directly, formants and all -- which is what makes them sound like cartoon characters.

Three ways of rebuilding an STFT with a DIFFERENT phase than the one it was measured with.
The magnitude |X[t,k]| (what frequencies are present, and how loud) is always kept; what
changes is the phase, which is where pitch and periodicity live.

Notation:  N = frame length, H = hop, k = bin index, t = frame index,
           X[t,k] = |X[t,k]| e^{j phi[t,k]}   (see stft.py)


1. pitch -- shifting bins in frequency, duration untouched
----------------------------------------------------------
speed_pitch.py changes pitch by stretching in TIME and then resampling.  This mode moves
the spectrum along the FREQUENCY axis directly, frame by frame, with the hop unchanged --
so the duration can never drift.

(a) The true frequency of each bin.  Bin k's centre frequency is

        w_k = 2*pi*k / N                     rad/sample

    so over one hop its phase is expected to advance by  w_k * H.  The measured advance
    phi[t,k] - phi[t-1,k] is only known modulo 2*pi; the part that differs from the
    expectation, wrapped into (-pi, pi], is the offset of the real sinusoid from the bin
    centre:

        dev[t,k]   = princarg( phi[t,k] - phi[t-1,k] - w_k*H )
        omega[t,k] = w_k + dev[t,k] / H               (true frequency, rad/sample)

    This is the same estimator time_stretch() uses.

(b) Moving whole lobes, not single bins.  A partial is a lobe several bins wide (stft.py,
    "Identity phase locking").  Scaling every bin to round(k*r) would stretch each lobe
    to r times its width and scatter its phases; instead each lobe is moved RIGIDLY to
    where its peak p belongs (J. Laroche and M. Dolson, "New phase-vocoder techniques for
    pitch-shifting, harmonizing and other exotic effects", IEEE WASPAA 1999):

        r               = 2^(semitones/12)            (12 semitones = 2x)
        shift(p)        = round(p * r) - p            (bins; one per lobe)
        |Y[t, k+shift]| += |X[t,k]|                   for every k in the region of p
        omega'[t,p]     = omega[t,p] * r              (the sinusoid really is r x higher)

    Bins that land outside 0 .. N/2 are dropped -- past Nyquist they would alias.

(c) Coherent phase.  A sinusoid of frequency omega' must gain omega'*H radians per hop,
    otherwise neighbouring frames cancel each other in the overlap-add.  Only the peaks
    run that recursion, at their destination p' = p + shift(p); every other bin of the
    lobe keeps its offset from the peak (identity phase locking):

        psi[t,p']        = psi[t-1,p'] + omega'[t,p'] * H        (psi[0,p'] = phi[0,p])
        psi[t,k+shift]   = psi[t,p'] + ( phi[t,k] - phi[t,p] )

    Y[t,k'] = |Y[t,k']| e^{j psi[t,k']}  goes through the ISTFT at the ORIGINAL hop.

    The starting phase matters.  Starting from a phase linear in k instead lines all the
    partials up at one instant -- the IDFT of a linear-phase spectrum is a single pulse --
    and the output comes out as a click followed by a spiky, over-aligned waveform.  With
    measured phases and locked lobes the output keeps the input's crest factor and level.

    Formants move with the harmonics (nothing here separates the envelope from the
    source), which is why +7 st sounds like a chipmunk and not a soprano.

(d) A ratio that changes from frame to frame.  Nothing in (a)-(c) needs r to be constant:
    frame t is moved by its own r_t, and the recursion simply uses the ratio of the frame
    it is advancing INTO:

        psi[t,p'_t] = psi[t-1,p'_t] + r_t * omega[t,p] * H

    The phase accumulator is continuous across the change, so a smoothly varying r_t (a
    pitch contour) glides instead of clicking.  shift_spectrum() is this general form;
    shift_pitch_bins() calls it with r_t = r for every t, and voice_shift.py with a
    per-frame contour.


2. robot -- one phase for every frame
-------------------------------------
Throw the measured phase away and give every bin of every frame the phase of a pulse at
the frame centre:

        Y[t,k] = |X[t,k]| * e^{-j*pi*k}  =  |X[t,k]| * (-1)^k

The inverse DFT of a real, non-negative spectrum is a zero-phase pulse at n = 0; the
(-1)^k factor is a circular shift by N/2 that moves it to the middle of the frame, where
the Hann synthesis window is 1 instead of 0.  Every frame now contributes one pulse, and
the frames are H samples apart, so the output is a pulse train with period H:

        f_robot = fs / H        ->        H = round(fs / robot_freq)

The voice keeps its spectral envelope (you can still hear the words) but loses its own
pitch: every syllable is sung on one monotone buzz.  The frame length is grown to keep
H <= N/2, so consecutive frames always overlap and the ISTFT's window-square divisor
(stft.py) stays well defined.


3. whisper -- random phase
--------------------------
Replace the phase with independent noise, uniform on (-pi, pi]:

        Y[t,k] = |X[t,k]| * e^{j*theta[t,k]},     theta ~ U(-pi, pi]

A voiced sound is a harmonic series whose phases advance coherently from frame to frame;
with random phase there is nothing periodic left, but the short-time energy distribution
is the same -- which is exactly what a whisper is (breath noise shaped by the vocal
tract).  A short frame (N = 512, ~32 ms at 16 kHz) keeps the noise from smearing the
consonants.  The generator is seeded with a fixed constant so the same input always bakes
to the same output (Auto-Bake re-runs this on every slider move).


Dry / wet
---------
        y[n] = (1 - m) * x[n] + m * wet[n],       0 <= m <= 1
"""

from __future__ import annotations

import numpy as np

from .stft import (
    DEFAULT_HOP, DEFAULT_N_FFT, istft, peak_regions, principal_argument, spectral_peaks, stft,
)

# Whisper's short frame: breath noise needs time resolution more than frequency resolution.
_WHISPER_N_FFT = 512
_WHISPER_HOP = 128
# Any constant does; what matters is that it never changes between bakes.
_WHISPER_SEED = 0x5EED1E55


def shift_spectrum(
    spec: np.ndarray,
    ratios: np.ndarray,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
) -> np.ndarray:
    """Move every lobe of frame t to ratios[t] times its frequency (docstring section 1).

    Takes and returns an STFT so a caller can reshape the magnitudes afterwards (voice_shift
    corrects the envelope) before the one ISTFT.  With a constant `ratios` this is exactly
    the fixed shift; with a varying one it is section 1(d).
    """
    mag = np.abs(spec)
    phi = np.angle(spec)
    n_frames, n_bins = spec.shape
    ratios = np.broadcast_to(np.asarray(ratios, dtype=np.float64), (n_frames,))

    k = np.arange(n_bins)
    w_k = 2.0 * np.pi * k / n_fft                 # bin centre frequency, rad/sample

    out = np.zeros_like(spec)
    psi_prev = np.zeros(n_bins)                   # last frame's OUTPUT phase, per bin
    for t in range(n_frames):
        r = ratios[t]
        peaks = spectral_peaks(mag[t])
        if peaks.size == 0:                       # silent frame: nothing to move
            psi_prev = np.zeros(n_bins)
            continue

        # (b) one rigid shift per lobe, applied to every bin of the lobe.
        region = peak_regions(n_bins, peaks)                   # bin -> index into peaks
        owner = peaks[region]                                  # bin -> its peak p
        shift = np.round(peaks * r).astype(int) - peaks        # per peak, in bins
        dest = k + shift[region]                               # k' for every bin
        keep = (dest >= 0) & (dest < n_bins)

        # (c) the peaks run the phase recursion at their destination ...
        dest_peaks = peaks + shift
        if t == 0:
            psi_peaks = phi[0, peaks]                          # start from the measured phase
        else:
            # (a) true frequency of each peak, then r x that over one hop.
            dev = principal_argument(phi[t, peaks] - phi[t - 1, peaks] - w_k[peaks] * hop)
            omega = w_k[peaks] + dev / hop
            safe = np.clip(dest_peaks, 0, n_bins - 1)
            psi_peaks = psi_prev[safe] + r * omega * hop

        # ... and every other bin keeps its offset from its peak (phase locking).
        psi_bins = psi_peaks[region] + (phi[t] - phi[t, owner])

        frame_mag = np.zeros(n_bins)
        np.add.at(frame_mag, dest[keep], mag[t, keep])         # colliding bins ADD
        frame_phase = np.zeros(n_bins)
        frame_phase[dest[keep]] = psi_bins[keep]
        out[t] = frame_mag * np.exp(1j * frame_phase)
        psi_prev = frame_phase

    return out


def shift_pitch_bins(
    x: np.ndarray,
    semitones: float,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
) -> np.ndarray:
    """Mode 1: move every spectral lobe to r times its frequency (docstring section 1)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or abs(semitones) < 1e-9:
        return x

    r = 2.0 ** (float(semitones) / 12.0)          # pitch ratio, equal temperament
    spec = stft(x, n_fft, hop)                    # (frames, bins)
    out = shift_spectrum(spec, np.full(spec.shape[0], r), n_fft, hop)
    return istft(out, hop=hop, n_fft=n_fft, length=x.size)


def robotise(x: np.ndarray, fs: int, robot_freq: float = 100.0) -> np.ndarray:
    """Mode 2: centre-pulse phase in every frame, hop = one period of robot_freq."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x

    hop = max(1, int(round(fs / max(float(robot_freq), 1.0))))   # H = fs / f_robot
    # Grow N (powers of two) until H <= N/2 so consecutive frames always overlap.
    n_fft = DEFAULT_N_FFT
    while hop > n_fft // 2:
        n_fft *= 2

    mag = np.abs(stft(x, n_fft, hop))
    k = np.arange(mag.shape[1])
    centre = np.where(k % 2 == 0, 1.0, -1.0)      # e^{-j*pi*k} = (-1)^k: pulse at n = N/2
    return istft(mag * centre[None, :], hop=hop, n_fft=n_fft, length=x.size)


def whisperise(x: np.ndarray) -> np.ndarray:
    """Mode 3: keep |X|, replace the phase with seeded uniform noise."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x

    mag = np.abs(stft(x, _WHISPER_N_FFT, _WHISPER_HOP))
    theta = np.random.default_rng(_WHISPER_SEED).uniform(-np.pi, np.pi, mag.shape)
    return istft(mag * np.exp(1j * theta), hop=_WHISPER_HOP, n_fft=_WHISPER_N_FFT,
                 length=x.size)


def voice_changer(
    x: np.ndarray,
    fs: int,
    mode: str = "chipmunk",
    semitones: float = 7.0,
    robot_freq: float = 100.0,
    mix: float = 1.0,
) -> np.ndarray:
    """Recipe-facing wrapper.

    chipmunk / monster shift by +|semitones| / -|semitones|; any other mode ("pitch", used
    by direct callers and the tests) shifts by the signed `semitones` exactly as given.  robot_freq is used only
    by robot.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x

    if mode == "robot":
        wet = robotise(x, fs, robot_freq)
    elif mode == "whisper":
        wet = whisperise(x)
    elif mode == "chipmunk":
        wet = shift_pitch_bins(x, abs(semitones))
    elif mode == "monster":
        wet = shift_pitch_bins(x, -abs(semitones))
    else:
        wet = shift_pitch_bins(x, semitones)

    m = float(np.clip(mix, 0.0, 1.0))
    return (1.0 - m) * x + m * wet
