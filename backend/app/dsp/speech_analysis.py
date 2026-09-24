"""
Speech analysis -- pitch, voicing and spectral envelope, frame by frame
=======================================================================
In plain words
--------------
The voice tools and the Signal Doctor need three facts about every 10 ms of a recording: is anyone speaking, is the
sound voiced (a buzzing vocal fold, with a pitch) or unvoiced (a hiss like "s"), and what
shape does the spectrum have once the individual harmonics are smoothed away (the vocal
tract's resonances -- the formants).  This module measures those three and nothing else.
It is kept apart from analysis.py, whose numbers describe any waveform; these only make
sense for speech.

Framing
-------
        L = round(0.050 fs)     frame length (50 ms: at least 3 periods of a 60 Hz voice)
        H = round(0.010 fs)     hop (10 ms)
        frame t covers x[tH : tH + L],   its centre is at  (tH + L/2) / fs  seconds

Frames are processed in chunks of at most CHUNK_FRAMES, so a two-minute take at 96 kHz
never builds a (12000 x 4800) matrix in one go.

1. Pitch -- normalised autocorrelation
--------------------------------------
For one frame s[n] (DC removed), the normalised cross-correlation at lag tau compares the
frame with itself shifted by tau, over the part where the two overlap:

                          sum_{n=0}^{L-tau-1} s[n] s[n+tau]
        r(tau) = -----------------------------------------------------
                  sqrt( sum_{n<L-tau} s[n]^2  *  sum_{n>=tau} s[n]^2 )

r = 1 for a perfectly periodic frame at tau = one period, and ~0 for noise.  (Dividing by
the energy of the two OVERLAPPING segments, rather than by r(0), removes the linear taper
that the plain autocorrelation has, so long lags are not penalised.)

  * The numerator for every tau at once:  irfft( |rfft(s, 2L)|^2 ).  Zero-padding to 2L
    makes the circular correlation equal the linear one.
  * The two energies come from one cumulative sum c[m] = sum_{n<m} s[n]^2:
        e0(tau) = c[L - tau],        e1(tau) = c[L] - c[tau]
  * Search tau in [fs/F0_MAX, fs/F0_MIN] = 60 .. 500 Hz.
  * A periodic frame also correlates at 2T, 3T, ... (sub-octaves).  Among the local maxima,
    take the SHORTEST lag whose value is within 10 % of the best one -- that is the period,
    not a multiple of it.
  * Parabolic interpolation through (tau-1, tau, tau+1) refines the lag to a fraction of a
    sample:
                          y[-1] - y[+1]
        delta = 0.5 * ----------------------- ,        f0 = fs / (tau + delta)
                      y[-1] - 2 y[0] + y[+1]

Voicing:  active = frame energy above an adaptive floor (below, it is silence);
          voiced = active  and  r(tau) >= VOICING_THRESHOLD.
The floor is  max( E_95 - 40 dB,  min(E_10 + 6 dB, E_95 - 15 dB) )  in terms of the
file's frame-energy percentiles -- 6 dB above the noise floor, but never so high that the
quieter half of a speech-only file counts as silence.

Octave errors: inside each voiced run, a frame whose log2 f0 is more than 0.75 octave
from the median of its 7 neighbours is folded back by x2 or x1/2.  Runs shorter than 3
frames (30 ms) are not real syllables and are marked unvoiced.

2. Spectral envelope -- real cepstrum with a low-quefrency lifter
-----------------------------------------------------------------
Speech is a source (the harmonic comb at f0) filtered by the vocal tract.  In the log
spectrum the two ADD:

        log|X(k)| = log|Source(k)| + log|Tract(k)|

The tract term varies slowly with k, the comb quickly (period f0 in Hz).  The real
cepstrum is the inverse DFT of the log magnitude:

        c[q] = IDFT( log|X(k)| )[q]         q = "quefrency", in samples

so the slow tract lands at small q and the comb at q = fs/f0 (one pitch period).

Peak interpolation first.  Between two harmonics the spectrum dips 20-40 dB, and after a
pitch shift that moves lobes apart there are bins with NOTHING in them at all.  A log
average over those valleys drags the envelope down by an amount that depends on the
harmonic spacing -- i.e. on f0, the very thing the envelope is meant to be free of (and
after a 1.75x shift the "envelope" came out 30 dB low).  So the log magnitude is first
replaced by the straight lines joining its spectral peaks (stft.spectral_peaks):

        L(k) = interp( k ;  peaks p,  log|X(p)| )      held flat outside the first/last peak

which passes through the harmonics and bridges the gaps; the lifter then smooths L.

Not every local maximum is a harmonic: sidelobes and resynthesis ripple leave small peaks
in the valleys, and joining THOSE drags L back down (after a shift the envelope read
~13 dB low, and the correction boosted the whole file to match).  With the frame's f0
known, a peak counts only if it is the largest bin within +-0.6 f0 of itself -- which
keeps one peak per harmonic (the next harmonic is a full f0 away) and drops the ripple
between them:

        keep p   iff   |X(p)| = max{ |X(k)| : |k - p| <= 0.6 f0 N / fs }

Unvoiced or unknown rows use f0 = DEFAULT_SPACING_HZ, an upper envelope of the noise.
Keeping only q < q_c and transforming back gives the smooth envelope:

        env(k) = Re DFT( c[q] * l[q] ),     l[0] = 1,  l[1..q_c-1] = 2,  l[q >= q_c] = 0

(the factor 2 folds the mirrored half of the real, even cepstrum onto the kept half).
q_c = round(LIFTER_S * fs), with LIFTER_S = 1.6 ms, below the shortest period we search
(2 ms at 500 Hz).  It is FIXED, not pitch-adaptive, so two speakers are measured with the
same smoothing and their envelopes can be compared.  env is in natural log; times
20/ln 10 gives dB.

3. The 32-point descriptor
--------------------------
The envelope is sampled at N_MEL = 32 points equally spaced on the mel scale

        mel(f) = 2595 log10(1 + f / 700)

from 80 Hz to min(8 kHz, 0.45 fs) -- dense where hearing resolves formants, sparse above.
Each frame's mean dB is then subtracted, so the descriptor holds the SHAPE of the
spectrum and not how loudly it was spoken.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stft import make_window, spectral_peaks

FRAME_S = 0.050
HOP_S = 0.010
F0_MIN = 60.0
F0_MAX = 500.0
VOICING_THRESHOLD = 0.5
# speech_mask: what a syllable looks like (the denoiser's residue does not)
SPEECH_RANGE_DB = 25.0
SPEECH_MIN_RUN_S = 0.08
SPEECH_PERIODICITY = 0.75
LIFTER_S = 0.0016
N_MEL = 32
MEL_LO_HZ = 80.0
MEL_HI_HZ = 8000.0
CHUNK_FRAMES = 1000
OCTAVE_WINDOW = 7
MIN_VOICED_RUN = 3
DEFAULT_SPACING_HZ = 200.0     # peak spacing assumed for unvoiced frames

_DB_PER_NEPER = 20.0 / np.log(10.0)     # ln|X| -> 20 log10|X|


@dataclass
class SpeechFrames:
    """Per-frame measurements of one recording (docstring sections 1-3)."""

    fs: int
    frame_len: int
    hop: int
    times: np.ndarray        # frame centres, seconds
    f0: np.ndarray           # Hz, 0 where not voiced
    voiced: np.ndarray       # bool
    active: np.ndarray       # bool: not silence
    periodicity: np.ndarray  # r(tau) at the chosen lag, 0..1
    energy_db: np.ndarray    # frame power in dB
    env: np.ndarray          # (frames, N_MEL) mean-removed log envelope, dB
    mel_hz: np.ndarray       # (N_MEL,) where the descriptor is sampled

    @property
    def voiced_seconds(self) -> float:
        return float(np.count_nonzero(self.voiced)) * self.hop / self.fs


def frame_params(fs: int) -> tuple[int, int]:
    """(L, H) in samples for this sample rate."""
    return max(8, int(round(FRAME_S * fs))), max(1, int(round(HOP_S * fs)))


def mel_points(fs: int, n: int = N_MEL) -> np.ndarray:
    """n frequencies equally spaced in mel between MEL_LO_HZ and min(8 kHz, 0.45 fs)."""
    hi = min(MEL_HI_HZ, 0.45 * fs)
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)          # noqa: E731
    inv = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)          # noqa: E731
    return inv(np.linspace(mel(MEL_LO_HZ), mel(hi), n))


def hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def cepstral_envelope_db(mag: np.ndarray, fs: int, n_fft: int,
                         f0: np.ndarray | None = None) -> np.ndarray:
    """Smooth log envelope (dB) of each row of an rfft magnitude array (docstring 2).

    mag has shape (frames, n_fft//2 + 1); f0 (Hz per row, 0 = unvoiced) sets how far
    apart harmonic peaks must be.  Peaks are picked and joined first, then liftered.  An
    epsilon relative to each frame's maximum keeps log() finite in empty bins.
    """
    mag = np.atleast_2d(np.asarray(mag, dtype=np.float64))
    eps = 1e-5 * np.max(mag, axis=1, keepdims=True) + 1e-12
    log_mag = np.log(mag + eps)
    n_rows, n_bins = mag.shape
    bins = np.arange(n_bins)
    if f0 is None:
        f0 = np.zeros(n_rows)
    spacing = np.where(np.asarray(f0) > 0, f0, DEFAULT_SPACING_HZ)
    half = np.maximum(1, np.round(0.6 * spacing * n_fft / fs)).astype(int)
    for row in range(n_rows):                                      # pick and join peaks
        peaks = spectral_peaks(mag[row], floor=1e-4)
        if peaks.size < 2:
            continue
        w = int(half[row])
        padded = np.pad(mag[row], w, mode="constant")
        local_max = np.lib.stride_tricks.sliding_window_view(padded, 2 * w + 1).max(axis=1)
        peaks = peaks[mag[row, peaks] >= local_max[peaks]]
        if peaks.size >= 2:
            log_mag[row] = np.interp(bins, peaks, log_mag[row, peaks])
    ceps = np.fft.irfft(log_mag, n=n_fft, axis=1)                  # real cepstrum c[q]

    q_c = int(np.clip(round(LIFTER_S * fs), 2, n_fft // 2 - 1))
    lifter = np.zeros(n_fft)
    lifter[0] = 1.0
    lifter[1:q_c] = 2.0                                            # fold the mirror half
    env = np.fft.rfft(ceps * lifter[None, :], n=n_fft, axis=1).real
    return env * _DB_PER_NEPER


def sample_at(values: np.ndarray, freqs: np.ndarray, at_hz: np.ndarray) -> np.ndarray:
    """Linear interpolation of every row of `values` (sampled at `freqs`) at `at_hz`.

    One np.interp per row would be a Python loop over frames; since every row shares the
    same abscissa, the fractional indices are computed once and applied to all rows.
    """
    at_hz = np.clip(at_hz, freqs[0], freqs[-1])
    pos = np.interp(at_hz, freqs, np.arange(freqs.size))
    i0 = np.clip(np.floor(pos).astype(int), 0, freqs.size - 2)
    a = pos - i0
    return values[:, i0] * (1.0 - a) + values[:, i0 + 1] * a


def _frames(x: np.ndarray, frame_len: int, hop: int, first: int, count: int) -> np.ndarray:
    starts = (first + np.arange(count)) * hop
    return x[starts[:, None] + np.arange(frame_len)[None, :]]


def _pitch_chunk(frames: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """f0 (Hz) and periodicity r(tau) for each row (docstring 1)."""
    n_rows, L = frames.shape
    s = frames - frames.mean(axis=1, keepdims=True)

    n_pad = 1
    while n_pad < 2 * L:
        n_pad *= 2
    spectrum = np.fft.rfft(s, n=n_pad, axis=1)
    acf = np.fft.irfft(spectrum.real ** 2 + spectrum.imag ** 2, n=n_pad, axis=1)[:, :L]

    csum = np.concatenate([np.zeros((n_rows, 1)), np.cumsum(s * s, axis=1)], axis=1)
    tau_lo = max(2, int(np.floor(fs / F0_MAX)))
    tau_hi = min(L - 2, int(np.ceil(fs / F0_MIN)))
    taus = np.arange(tau_lo - 1, tau_hi + 2)                       # one extra each side
    e0 = csum[:, L - taus]                                         # sum_{n < L-tau} s^2
    e1 = csum[:, L:L + 1] - csum[:, taus]                          # sum_{n >= tau} s^2
    r = acf[:, taus] / np.sqrt(np.maximum(e0 * e1, 1e-20))

    inner = r[:, 1:-1]
    is_peak = (inner > r[:, :-2]) & (inner >= r[:, 2:]) & (inner > 0.0)
    best = np.max(np.where(is_peak, inner, -np.inf), axis=1)
    has_peak = np.isfinite(best)
    ok = is_peak & (inner >= 0.9 * best[:, None])
    choice = np.argmax(ok, axis=1)                                 # FIRST (shortest) lag

    rows = np.arange(n_rows)
    ym, y0, yp = r[rows, choice], r[rows, choice + 1], r[rows, choice + 2]
    denom = ym - 2.0 * y0 + yp
    delta = np.where(np.abs(denom) > 1e-12, 0.5 * (ym - yp) / np.where(denom == 0, 1, denom), 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    lag = taus[choice + 1] + delta
    peak = y0 - 0.25 * (ym - yp) * delta

    f0 = np.where(has_peak, fs / lag, 0.0)
    periodicity = np.where(has_peak, np.clip(peak, 0.0, 1.0), 0.0)
    return f0, periodicity


def _fix_octaves(f0: np.ndarray, voiced: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fold isolated octave jumps back and drop runs shorter than MIN_VOICED_RUN."""
    f0 = f0.copy()
    voiced = voiced.copy()
    edges = np.flatnonzero(np.diff(np.concatenate([[0], voiced.astype(np.int8), [0]])))
    half = OCTAVE_WINDOW // 2
    for a, b in zip(edges[0::2], edges[1::2]):
        if b - a < MIN_VOICED_RUN:
            voiced[a:b] = False
            f0[a:b] = 0.0
            continue
        run = np.log2(f0[a:b])
        padded = np.pad(run, half, mode="edge")
        med = np.median(np.lib.stride_tricks.sliding_window_view(padded, OCTAVE_WINDOW), axis=1)
        dev = run - med
        run = np.where(dev > 0.75, run - 1.0, np.where(dev < -0.75, run + 1.0, run))
        f0[a:b] = 2.0 ** run
    return f0, voiced


def analyse(x: np.ndarray, fs: int) -> SpeechFrames:
    """Pitch, voicing and 32-point envelope for every 10 ms frame of x."""
    x = np.asarray(x, dtype=np.float64)
    frame_len, hop = frame_params(fs)
    if x.size < frame_len:
        x = np.pad(x, (0, frame_len - x.size))
    n_frames = 1 + int(np.ceil((x.size - frame_len) / hop))
    x = np.pad(x, (0, (n_frames - 1) * hop + frame_len - x.size))

    n_fft = 1
    while n_fft < frame_len:
        n_fft *= 2
    window = make_window(frame_len)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    mel_hz = mel_points(fs)

    f0 = np.zeros(n_frames)
    periodicity = np.zeros(n_frames)
    energy_db = np.zeros(n_frames)
    env = np.zeros((n_frames, N_MEL))
    for first in range(0, n_frames, CHUNK_FRAMES):
        count = min(CHUNK_FRAMES, n_frames - first)
        frames = _frames(x, frame_len, hop, first, count)
        sl = slice(first, first + count)
        f0[sl], periodicity[sl] = _pitch_chunk(frames, fs)
        energy_db[sl] = 10.0 * np.log10(np.mean(frames * frames, axis=1) + 1e-12)
        mag = np.abs(np.fft.rfft(frames * window[None, :], n=n_fft, axis=1))
        spacing = np.where(periodicity[sl] >= VOICING_THRESHOLD, f0[sl], 0.0)
        env_db = sample_at(cepstral_envelope_db(mag, fs, n_fft, spacing), freqs, mel_hz)
        env[sl] = env_db - env_db.mean(axis=1, keepdims=True)      # shape, not loudness

    e95, e10 = np.percentile(energy_db, 95), np.percentile(energy_db, 10)
    floor = max(e95 - 40.0, min(e10 + 6.0, e95 - 15.0))
    active = (energy_db > floor) & (energy_db > -100.0)
    voiced = active & (periodicity >= VOICING_THRESHOLD) & (f0 > 0)
    f0, voiced = _fix_octaves(np.where(voiced, f0, 0.0), voiced)

    times = (np.arange(n_frames) * hop + frame_len / 2.0) / fs
    return SpeechFrames(fs=fs, frame_len=frame_len, hop=hop, times=times, f0=f0,
                        voiced=voiced, active=active, periodicity=periodicity,
                        energy_db=energy_db, env=env, mel_hz=mel_hz)


def speech_mask(frames: SpeechFrames, pad_s: float = 0.2) -> np.ndarray:
    """Frames where someone is speaking: the voiced frames, widened by `pad_s` on each
    side so the unvoiced consonants around them ("s", "t", breaths into a word) count too.

    Voicing, not level, is what tells speech from a room: a fan can be as loud as quiet
    talking, but it has no pitch.  Only voiced runs that sound like syllables count:
    at least SPEECH_MIN_RUN_S long, clearly periodic (mean r >= SPEECH_PERIODICITY) and
    within SPEECH_RANGE_DB of the loud speech (P90 of those runs).  The tonal residue a
    denoiser leaves in the pauses is voiced too, but in 30-90 ms flickers at r ~ 0.65,
    20-30 dB under the talking; real syllables run 100 ms and more at r ~ 0.9.
    """
    voiced = frames.voiced.copy()
    edges = np.flatnonzero(np.diff(np.concatenate([[0], voiced.astype(np.int8), [0]])))
    min_run = max(1, int(round(SPEECH_MIN_RUN_S * frames.fs / frames.hop)))
    for a, b in zip(edges[0::2], edges[1::2]):
        if b - a < min_run or np.mean(frames.periodicity[a:b]) < SPEECH_PERIODICITY:
            voiced[a:b] = False
    if np.any(voiced):
        loud = float(np.percentile(frames.energy_db[voiced], 90))
        voiced &= frames.energy_db >= loud - SPEECH_RANGE_DB
    pad = max(0, int(round(pad_s * frames.fs / frames.hop)))
    if pad == 0 or voiced.size == 0:
        return voiced
    return np.convolve(voiced.astype(np.float64), np.ones(2 * pad + 1), mode="same") > 0.0
