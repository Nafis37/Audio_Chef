"""
Voice Match -- pitch mapping, time-varying pitch shift and envelope correction
=============================================================================
In plain words
--------------
Move a recording of YOUR voice toward a reference speaker's.  Two things are changed:

  pitch   -- how high you speak, mapped from your calibration pitch range onto theirs
  timbre  -- the shape of the spectrum (the formants), nudged by what the calibration
             pair says the reference sounded like on similar sounds

Silence and unvoiced consonants are left alone.  With both strengths at 0 the input comes
back untouched.  The calibration (voice_calibration.py) supplies the pitch statistics
and the codebook; this module applies them.  Perfect imitation is not a promise: this
is a classical signal-processing mapping, not a trained model.

Notation: analysis frames t every 10 ms (speech_analysis.py), STFT frames u, bins k.

1. Pitch mapping -- one Gaussian onto another, in log frequency
---------------------------------------------------------------
Pitch perception is roughly logarithmic, so the two speakers' voiced log f0 are modelled
as Gaussians N(mu_a, sd_a^2) (you) and N(mu_b, sd_b^2) (reference).  Matching the z-score
maps your f0 to

        l~ = mu_b + (sd_b / sd_a) (ln f0 - mu_a)

which moves the average AND scales the intonation range.  If sd_a is tiny (a monotone
calibration take) the ratio sd_b/sd_a is meaningless, so a plain mean shift is used:
l~ = ln f0 + (mu_b - mu_a).  The per-frame log pitch ratio, scaled by the strength s_p:

        rho[t] = s_p (l~[t] - ln f0[t])           r[t] = e^rho[t],   clamped to [1/2, 2]

2. Only where there is a voice
------------------------------
rho exists only on voiced frames.  Unvoiced gaps shorter than GAP_S are bridged by linear
interpolation (a consonant between two vowels should not snap the pitch back to 1 and up
again).  Everywhere else a voicing weight v[t] -- 1 on voiced frames, 0 elsewhere, then a
RAMP_S moving average so it ramps rather than steps -- fades the change out:

        rho[t] <- v[t] * rho_filled[t]          then a 5-frame moving average

3. Time-varying pitch shift
---------------------------
The STFT uses N = the next power of two >= 40 ms and H = N/4 (the COLA condition of
stft.py).  rho is interpolated onto the STFT frame centres, and voice_changer.
shift_spectrum() moves frame u's lobes by r[u] with a phase recursion that stays
continuous as r changes (voice_changer.py, section 1d).  The sample count is kept by
istft(..., length = len(x)).

4. Envelope correction -- separating the formant shift from the timbre change
-----------------------------------------------------------------------------
Moving every lobe by r also moves the formants by r (a chipmunk).  Let E(k) be the
cepstral envelope in dB (speech_analysis.cepstral_envelope_db):

        E_in[u,k]   envelope of the input frame
        E_sh[u,k]   envelope of the SHIFTED frame, measured on stft(istft(Y_sh))
        D[u,k]      the codebook's predicted change (32 mel points -> every bin)

The gain applied to the shifted frame, per bin:

        g[u,k] = clip(E_in - E_sh, -36, +18)  +  clip(s_t * rel[u] * v[u] * D[u,k], +-12)  dB

  * E_in - E_sh undoes what the shift did to the envelope (formant preservation), so pitch
    changes WITHOUT the chipmunk.  With r = 1 it is zero.
  * s_t * rel * v * D is the intentional timbre change: timbre strength, times the
    codebook's confidence for this frame (voice_calibration section 4), times voicing.

Why re-analyse: a modified STFT is generally not the STFT of any signal, and the overlap-
add of neighbouring shifted frames partly cancels and re-mixes their lobes.  Measured
directly on Y_sh, E_sh was off by 10-25 dB from what the output actually contained, so the
"correction" corrected the wrong thing.  One extra ISTFT/STFT pair makes Y_sh consistent.

The two terms are clipped separately.  The timbre term gets +-12 dB: an intentional
change larger than that is more likely a codebook error than a voice.  The formant term
needs more: undoing a 1.75x shift takes ~-29 dB where a moved formant now sits over what
used to be a valley.  Cuts go to -36 dB; boosts stop at +18 dB, because a boost lands on
bins the shift may have emptied, and would lift only noise.

g is smoothed over 3 frames in time.  A positive real gain scales magnitudes and leaves
phases alone, so it can be applied after the phase recursion:  Y = Y_sh * 10^(g/20).
Above the descriptor's top point (8 kHz) D fades linearly to 0 over one octave;
nothing was measured there.

5. Loudness, then output
------------------------
The descriptors are mean-removed: they say nothing about loudness, so the conversion
should not change it.  But a pitch shift and a dB-domain envelope gain both move frame
energy (a +6 dB bump where the formants sit counts for more than a -6 dB one where
they do not).  So each frame is rescaled to its input energy:

        a[u] = 10 log10( sum_k |X[u,k]|^2  /  sum_k |Y[u,k]|^2 )   dB,  clipped to +-12,
        Y[u,:] <- Y[u,:] * 10^(a~[u]/20)            a~ = a smoothed over 3 frames

and then weighted overlap-add (stft.istft).  The fold's fit_to_full_scale (dsp_engine.py) then
handles any level over full scale, as for every other operation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .speech_analysis import analyse, cepstral_envelope_db, hz_to_mel, sample_at
from .stft import istft, stft
from .voice_calibration import VoiceProfile, predict
from .voice_changer import shift_spectrum

GAP_S = 0.060
RAMP_S = 0.030
MIN_SD = 0.02                   # ~0.35 semitone: below this, mean shift only
RATIO_MIN, RATIO_MAX = 0.5, 2.0
MAX_GAIN_DB = 12.0              # the intentional timbre change
FORMANT_CUT_DB = 36.0           # undoing the shift's own formant movement: cuts ...
FORMANT_BOOST_DB = 18.0         # ... and boosts (a boost lifts bins that may be empty)
SEMITONES_PER_NEPER = 12.0 / np.log(2.0)


def _moving_average(x: np.ndarray, width: int, axis: int = 0) -> np.ndarray:
    """Centred moving average with edge padding (so the ends are not pulled to 0)."""
    if width <= 1:
        return x
    pad = [(0, 0)] * x.ndim
    pad[axis] = (width // 2, width - 1 - width // 2)
    padded = np.moveaxis(np.pad(x, pad, mode="edge"), axis, 0)
    # A running sum: out[i] = (S[i + width] - S[i]) / width, for every column at once.
    csum = np.cumsum(np.concatenate([np.zeros((1,) + padded.shape[1:]), padded]), axis=0)
    return np.moveaxis((csum[width:] - csum[:-width]) / width, 0, axis)


def _fill_short_gaps(values: np.ndarray, known: np.ndarray, max_gap: int) -> np.ndarray:
    """Mark unknown runs of at most max_gap frames BETWEEN two known frames as known."""
    filled = known.copy()
    edges = np.flatnonzero(np.diff(np.concatenate([[0], (~known).astype(np.int8), [0]])))
    for a, b in zip(edges[0::2], edges[1::2]):
        if a > 0 and b < known.size and b - a <= max_gap:
            filled[a:b] = True
    return filled


def pitch_contour(f0: np.ndarray, voiced: np.ndarray, profile: VoiceProfile,
                  strength: float, hop_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(rho per analysis frame, voicing weight v) -- docstring sections 1 and 2."""
    n = f0.size
    if not np.any(voiced):
        return np.zeros(n), np.zeros(n)

    lf = np.log(np.where(voiced, f0, 1.0))
    if profile.sd_cal < MIN_SD:
        target = lf + (profile.mu_ref - profile.mu_cal)
    else:
        target = profile.mu_ref + (profile.sd_ref / profile.sd_cal) * (lf - profile.mu_cal)
    rho = strength * (target - lf)

    # Bridge short unvoiced gaps; hold the nearest voiced value elsewhere (it is faded
    # out by v below, so the held value only matters inside the ramps).
    idx = np.arange(n)
    rho_filled = np.interp(idx, idx[voiced], rho[voiced])
    bridged = _fill_short_gaps(rho_filled, voiced, int(round(GAP_S / hop_s)))
    ramp = max(1, int(round(RAMP_S / hop_s)))
    v = np.clip(_moving_average(bridged.astype(np.float64), 2 * ramp + 1), 0.0, 1.0)
    rho = _moving_average(v * rho_filled, 5)
    return np.clip(rho, np.log(RATIO_MIN), np.log(RATIO_MAX)), v


def _stft_size(fs: int) -> tuple[int, int]:
    n_fft = 256
    while n_fft < 0.040 * fs:
        n_fft *= 2
    return n_fft, n_fft // 4


def voice_match(
    x: np.ndarray,
    fs: int,
    profile: VoiceProfile,
    pitch_strength: float = 1.0,
    timbre_strength: float = 0.7,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert x toward the reference speaker.  Returns (audio, diagnostics)."""
    x = np.asarray(x, dtype=np.float64)
    s_p = float(np.clip(pitch_strength, 0.0, 1.0))
    s_t = float(np.clip(timbre_strength, 0.0, 1.0))
    if x.size == 0 or (s_p <= 0.0 and s_t <= 0.0):
        return x, {"reduced_fraction": 0.0, "shift_st": 0.0, "voiced_s": 0.0}

    frames = analyse(x, fs)
    hop_s = frames.hop / fs
    rho, v = pitch_contour(frames.f0, frames.voiced, profile, s_p, hop_s)

    delta, rel = predict(profile, frames.env)
    delta = _moving_average(delta, 5)

    # Onto the STFT grid (section 3).
    n_fft, hop = _stft_size(fs)
    spec = stft(x, n_fft, hop)
    n_frames, n_bins = spec.shape
    t_stft = (np.arange(n_frames) * hop + n_fft / 2.0) / fs
    rho_u = np.interp(t_stft, frames.times, rho)
    # f0 on the STFT grid for the envelope's peak picking (0 = unvoiced).
    f0_u = np.where(np.interp(t_stft, frames.times, frames.voiced.astype(float)) >= 0.5,
                    np.interp(t_stft, frames.times[frames.voiced], frames.f0[frames.voiced])
                    if np.any(frames.voiced) else 0.0, 0.0)
    weight_u = np.interp(t_stft, frames.times, s_t * rel * v)
    delta_u = np.stack([np.interp(t_stft, frames.times, delta[:, d])
                        for d in range(delta.shape[1])], axis=1)

    if np.max(np.abs(rho_u)) > 1e-6:
        # Resynthesise and re-analyse: the shifted STFT is not a CONSISTENT one (no signal
        # has exactly that STFT), so its envelope is not what the output will contain.
        # The envelope of the re-analysed shift is, and the gain goes onto that.
        moved = istft(shift_spectrum(spec, np.exp(rho_u), n_fft, hop),
                      hop=hop, n_fft=n_fft, length=x.size)
        shifted = stft(moved, n_fft, hop)
        e_in = cepstral_envelope_db(np.abs(spec), fs, n_fft, f0_u)
        e_sh = cepstral_envelope_db(np.abs(shifted), fs, n_fft, f0_u * np.exp(rho_u))
        formant = e_in - e_sh
        # Where the shifted frame is empty (a lobe moved out, or silence), its envelope
        # is the epsilon floor and the difference means nothing.
        silent = np.max(np.abs(shifted), axis=1) <= 1e-9
        formant[silent] = 0.0
    else:
        shifted = spec
        formant = np.zeros((n_frames, n_bins))

    # Section 4: the 32 mel points onto every bin, in the mel domain.
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    mel_pts = hz_to_mel(profile.mel_hz)
    d_bins = sample_at(delta_u, mel_pts, hz_to_mel(freqs))
    top = profile.mel_hz[-1]
    d_bins *= np.clip((2.0 * top - freqs) / top, 0.0, 1.0)[None, :]

    gain_db = (np.clip(formant, -FORMANT_CUT_DB, FORMANT_BOOST_DB)
               + np.clip(weight_u[:, None] * d_bins, -MAX_GAIN_DB, MAX_GAIN_DB))
    gain_db = _moving_average(gain_db, 3, axis=0)
    out = shifted * 10.0 ** (gain_db / 20.0)

    # Section 5: every frame keeps the input frame's energy.
    e_x = np.sum(np.abs(spec) ** 2, axis=1)
    e_y = np.sum(np.abs(out) ** 2, axis=1)
    level_db = 10.0 * np.log10((e_x + 1e-20) / (e_y + 1e-20))
    level_db = np.where(e_y > 1e-20, np.clip(level_db, -MAX_GAIN_DB, MAX_GAIN_DB), 0.0)
    out *= 10.0 ** (_moving_average(level_db, 3) / 20.0)[:, None]
    y = istft(out, hop=hop, n_fft=n_fft, length=x.size)

    voiced = frames.voiced
    diagnostics = {
        "voiced_s": round(frames.voiced_seconds, 1),
        "reduced_fraction": round(float(np.mean(rel[voiced] < 1.0)) if np.any(voiced) else 0.0, 3),
        "shift_st": round(float(np.median(rho[voiced]) * SEMITONES_PER_NEPER)
                          if np.any(voiced) else 0.0, 2),
    }
    return y, diagnostics
