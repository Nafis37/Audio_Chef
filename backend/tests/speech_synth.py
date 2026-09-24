"""
A tiny source-filter speech synthesiser for the voice and Signal Doctor tests.

Real voices cannot live in a unit test, but the two things those tests measure can be
synthesised with known values:

  source  -- a harmonic comb at a KNOWN f0 contour (sum of harmonics, 1/k roll-off, so it
             is band-limited by construction: harmonics above 0.45 fs are left out)
  filter  -- a cascade of two-pole resonators at KNOWN formant frequencies

A two-pole resonator at centre F with bandwidth B:

        R      = exp(-pi B / fs)              pole radius   (bandwidth -> radius)
        theta  = 2 pi F / fs                  pole angle    (frequency -> angle)
        H(z)   = (1 - R) / (1 - 2R cos(theta) z^-1 + R^2 z^-2)

so a = [1, -2R cos(theta), R^2] and b = [1 - R] (a rough unit-ish gain at F), run with
scipy.signal.lfilter.  Unvoiced "fricatives" are white noise through one resonance.

A Speaker scales the formants (vocal-tract length), sets its own pitch and tilt; a script
is a list of phoneme names, and each speaker speaks it with jittered durations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

VOWELS = {                                # (F1, F2, F3) Hz, a typical adult male
    "a": (730, 1090, 2440),
    "i": (270, 2290, 3010),
    "u": (300, 870, 2240),
    "e": (530, 1840, 2480),
    "o": (570, 840, 2410),
}
FRICATIVES = {"s": 5000, "sh": 2500, "f": 3500}
PHONES = list(VOWELS) + list(FRICATIVES) + ["_"]      # "_" = a pause


@dataclass
class Speaker:
    f0: float                 # mean pitch, Hz
    formant_scale: float      # 1.0 = the table above
    tilt: float = 1.0         # harmonic amplitude ~ 1/k^tilt
    f0_spread: float = 0.08   # +- fraction of intonation
    rate: float = 1.0         # >1 speaks slower (longer phones)


def resonator(x: np.ndarray, freq: float, bw: float, fs: int) -> np.ndarray:
    R = np.exp(-np.pi * bw / fs)
    theta = 2.0 * np.pi * freq / fs
    return lfilter([1.0 - R], [1.0, -2.0 * R * np.cos(theta), R * R], x)


def harmonic_source(f0: np.ndarray, fs: int, tilt: float) -> np.ndarray:
    phase = 2.0 * np.pi * np.cumsum(f0) / fs
    out = np.zeros(f0.size)
    for k in range(1, 80):
        alive = k * f0 < 0.45 * fs
        if not np.any(alive):
            break
        out += np.where(alive, np.sin(k * phase) / k ** tilt, 0.0)
    return out


def random_script(n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    script = []
    for _ in range(n):
        script.append(str(rng.choice(list(VOWELS))))
        roll = rng.random()
        if roll < 0.35:
            script.append(str(rng.choice(list(FRICATIVES))))
        elif roll < 0.45:
            script.append("_")
    return script


def speak(script: list[str], speaker: Speaker, fs: int, seed: int = 0,
          amp: float = 0.3) -> np.ndarray:
    """Render `script` in `speaker`'s voice, with per-phone duration jitter."""
    rng = np.random.default_rng(seed)
    durations = []
    for phone in script:
        base = 0.16 if phone in VOWELS else (0.10 if phone in FRICATIVES else 0.25)
        durations.append(base * speaker.rate * rng.uniform(0.75, 1.3))
    lengths = [int(d * fs) for d in durations]
    total = sum(lengths)

    # One continuous intonation contour: slow random wander around the speaker's mean.
    knots = rng.uniform(-1.0, 1.0, max(2, total // (fs // 3) + 2))
    contour = np.interp(np.arange(total), np.linspace(0, total, knots.size), knots)
    f0 = speaker.f0 * (1.0 + speaker.f0_spread * contour)
    source = harmonic_source(f0, fs, speaker.tilt)

    fade = int(0.008 * fs)
    out = np.zeros(total + fade)
    pos = 0
    for phone, n in zip(script, lengths):
        seg_len = n + fade
        if phone in VOWELS:
            seg = source[pos:pos + seg_len]
            seg = np.pad(seg, (0, seg_len - seg.size))
            for k, formant in enumerate(VOWELS[phone]):
                seg = resonator(seg, formant * speaker.formant_scale, 80.0 + 40.0 * k, fs)
            seg = seg / (np.max(np.abs(seg)) + 1e-12)
        elif phone in FRICATIVES:
            noise = rng.standard_normal(seg_len)
            seg = resonator(noise, FRICATIVES[phone] * speaker.formant_scale, 1500.0, fs)
            seg = 0.25 * seg / (np.max(np.abs(seg)) + 1e-12)
        else:
            seg = np.zeros(seg_len)
        ramp = np.ones(seg_len)
        ramp[:fade] = np.linspace(0.0, 1.0, fade)
        ramp[-fade:] = np.linspace(1.0, 0.0, fade)
        out[pos:pos + seg_len] += seg * ramp
        pos += n
    out = out + 1e-4 * rng.standard_normal(out.size)      # a realistic noise floor
    return amp * out / (np.max(np.abs(out)) + 1e-12)
