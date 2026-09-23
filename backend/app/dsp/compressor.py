"""
Simple Audio Compressor -- envelope follower + dB-domain gain computer
=====================================================================
In plain words
--------------
A compressor is an automatic volume knob.  It watches how loud the audio has recently
been; whenever that goes above the THRESHOLD it turns the audio down, by more the further
over it is (the RATIO).  Afterwards everything is lifted back up so the loudest peak is
where it started (AUTO-MAKEUP) -- so the loud parts stay put and the quiet parts come up.
The result sounds denser and more even, and the waveform looks like a solid block.

  threshold -> WHEN does it start turning down?           (dBFS)
  ratio     -> HOW MUCH?  R:1 lets 1 dB out per R dB in  (4:1 -> 8 dB over becomes 2)
  attack    -> how fast it reacts when things get louder  (ms)
  release   -> how fast it lets go when they get quieter  (ms)
  knee      -> how gently it eases in around the threshold (dB)
  makeup    -> extra fixed boost at the end               (dB)

1. Envelope follower -- a one-pole low-pass on the level
--------------------------------------------------------
        env[n] = a * env[n-1] + (1 - a) * |x[n]|

For a step input the error decays as a^n.  Choosing the time constant tau as the time for
the error to fall to 1/e:

        a^(tau * fs) = e^-1     ->     a = exp( -1 / (tau * fs) )

Two coefficients: a_att (tau = attack) while the level is rising, a_rel (tau = release)
while it falls.  A short attack catches transients; a longer release avoids "pumping" as
the gain recovers between syllables.

The switch between a_att and a_rel makes the recursion NONLINEAR, so it cannot be handed to
lfilter and must be a loop.  envelope_follower() is that loop, per sample: the readable
reference.  block_envelope() is what compressor() runs -- the same recursion evaluated on
1 ms blocks, ~44x fewer Python iterations at 44.1 kHz:

        p[b]    = max_{n in block b} |x[n]|             (block peak, B = fs/1000 samples)
        e[b]    = a_B * e[b-1] + (1 - a_B) * p[b],      a_B = exp( -B / (tau * fs) ) = a^B
        env[n]  = linear interpolation of e[b] between block centres

a_B = a^B is the per-sample coefficient applied B times, so time constants are unchanged.
Feeding the block PEAK instead of every |x[n]| means the follower reads the true peak of a
steady tone; the per-sample version partly releases between the peaks of every cycle and
sits ~1 dB lower on a sine (tests/test_dsp.py pins that agreement).

Lookahead -- no overshoot on attacks
------------------------------------
A live compressor only knows the past, so every sudden onset slips through for about one
attack time before the envelope catches up.  With auto-makeup (section 4) lifting
everything by ~20 dB, those few milliseconds become the loudest thing in the file -- the
waveform turns spikier, the opposite of compression.  Here the whole file is in memory,
so the detector is fed the peak of the NEXT L blocks instead of the current one:

        q[b] = max( p[b], p[b+1], ..., p[b+L] ),      L = 3 attack times, in blocks

A one-pole follower covers 1 - e^-3 = 95 % of a step in 3 time constants, so by the time
an onset actually arrives the gain is already (almost) all the way down.  Releases are not
delayed: q falls as soon as the loud block is behind it.  For the same reason the
follower starts from the signal's own level, e[-1] = q[0], rather than from silence --
otherwise the file's first moment would be one big unhandled attack.

2. Gain computer -- the static curve in dB
------------------------------------------
With L = 20 log10(env) (dBFS), T = threshold, R = ratio and over = L - T, a hard-knee
compressor lets the output rise only 1/R dB per dB above T:

        L_out = T + over / R      ->    gain_dB = L_out - L = -(1 - 1/R) * over,  over > 0
                                        gain_dB = 0,                              over <= 0

3. Soft knee of width W
-----------------------
The hard curve has a corner at over = 0.  Inside |over| < W/2 it is replaced by a
quadratic that matches both neighbours in value AND slope:

        gain_dB = -(1 - 1/R) * (over + W/2)^2 / (2 W)

  at over = -W/2 :  value 0,                    slope 0          (meets the flat part)
  at over = +W/2 :  value -(1 - 1/R) * W/2,     slope -(1 - 1/R) (meets the hard line)

4. Makeup and application
-------------------------
        g[n] = 10^( (gain_dB[n] + makeup) / 20 ),        y[n] = g[n] * x[n]

Auto-makeup undoes exactly the reduction the static curve applies at the input's own
peak level P = peak_dB(x):

        makeup_auto = -gain_dB(P)          (sections 2-3 evaluated at L = P)

so a sustained sound as loud as the loudest moment comes out at its original level, and
everything quieter has come UP -- the crest factor (peak / RMS) shrinks, which is the
whole point.  Any manual `makeup` is added on top.

Why not simply match the output peak to the input peak?  Because the envelope starts at
0, the very first transient slips through before the attack catches it (section 1): the
output's peak would be that uncompressed overshoot, and the "makeup" would come out near
0 dB.  The static curve does not care about transients.  Any brief overshoot above full
scale is handled by the recipe's final fit to full scale (dsp_engine.py).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12     # keeps log10(0) from blowing up
BLOCK_MS = 1.0  # block length for block_envelope()


def _coefficient(tau_ms: float, fs: float) -> float:
    """a = exp(-1 / (tau * fs)), tau in ms.  The clamp guards against a 0 ms setting."""
    return float(np.exp(-1.0 / (max(tau_ms, 0.01) * 0.001 * fs)))


def envelope_follower(x: np.ndarray, fs: int, attack_ms: float, release_ms: float) -> np.ndarray:
    """|x| smoothed by a one-pole IIR, one sample at a time.  The readable reference."""
    a_att = _coefficient(attack_ms, fs)
    a_rel = _coefficient(release_ms, fs)

    rectified = np.abs(x)
    env = np.zeros_like(rectified)
    prev = 0.0
    for n in range(rectified.size):
        # Rising signal -> fast attack coefficient; falling -> slow release coefficient.
        a = a_att if rectified[n] > prev else a_rel
        prev = a * prev + (1.0 - a) * rectified[n]
        env[n] = prev
    return env


def block_envelope(
    x: np.ndarray, fs: int, attack_ms: float, release_ms: float, lookahead: bool = False,
) -> np.ndarray:
    """The same follower run on 1 ms block peaks, interpolated back to every sample.

    lookahead=True feeds it the peak of the next 3 attack times and starts it at the
    signal's own level (docstring, "Lookahead"); False is the plain causal follower.
    """
    block = max(1, int(round(fs * BLOCK_MS / 1000.0)))       # B samples per block
    n_blocks = -(-x.size // block)                           # ceil(N / B)
    padded = np.pad(np.abs(x), (0, n_blocks * block - x.size))
    peaks = padded.reshape(n_blocks, block).max(axis=1)      # p[b]

    prev = 0.0
    if lookahead:
        span = int(np.ceil(3.0 * max(attack_ms, 0.01) / BLOCK_MS))       # L, in blocks
        ahead = np.pad(peaks, (0, span))                                 # past the end: 0
        # q[b] = max(p[b .. b+L]): the max over each length-(L+1) window starting at b.
        peaks = np.lib.stride_tricks.sliding_window_view(ahead, span + 1).max(axis=1)
        prev = float(peaks[0])                                           # e[-1] = q[0]

    # The per-block rate is fs / B, so a_B = a^B with the per-sample a.
    a_att = _coefficient(attack_ms, fs / block)
    a_rel = _coefficient(release_ms, fs / block)

    env_blocks = np.empty(n_blocks)
    for b in range(n_blocks):
        a = a_att if peaks[b] > prev else a_rel
        prev = a * prev + (1.0 - a) * peaks[b]
        env_blocks[b] = prev

    centres = np.arange(n_blocks) * block + (block - 1) / 2.0
    return np.interp(np.arange(x.size), centres, env_blocks)


def gain_computer(level_db: np.ndarray, threshold: float, ratio: float, knee: float) -> np.ndarray:
    """Static curve (docstring sections 2-3): dB of gain change for each input level."""
    over = level_db - threshold                          # how far above the threshold
    slope = 1.0 - 1.0 / ratio                            # dB of reduction per dB over
    gain_db = np.zeros_like(over)

    if knee > 0.0:
        in_knee = (over > -knee / 2.0) & (over < knee / 2.0)
        above = over >= knee / 2.0
        gain_db[in_knee] = -slope * (over[in_knee] + knee / 2.0) ** 2 / (2.0 * knee)
        gain_db[above] = -slope * over[above]
    else:
        above = over > 0.0
        gain_db[above] = -slope * over[above]
    return gain_db


def _peak_db(x: np.ndarray) -> np.ndarray:
    """max |x| in dBFS, as a one-element array -- the shape gain_computer() takes."""
    return np.array([20.0 * np.log10(max(float(np.max(np.abs(x))), EPS))])


def compressor(
    x: np.ndarray,
    fs: int,
    threshold: float = -30.0,
    ratio: float = 6.0,
    attack: float = 5.0,
    release: float = 150.0,
    knee: float = 6.0,
    makeup: float = 0.0,
    auto_makeup: bool = True,
) -> np.ndarray:
    """threshold/knee/makeup in dB, ratio as R:1, attack/release in milliseconds."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x

    ratio = max(float(ratio), 1.0)
    knee = max(float(knee), 0.0)

    env = block_envelope(x, fs, attack, release, lookahead=True)
    env_db = 20.0 * np.log10(np.maximum(env, EPS))       # envelope in dBFS
    gain_db = gain_computer(env_db, threshold, ratio, knee)

    total_makeup = float(makeup)
    if auto_makeup:
        # Section 4: undo the static curve's reduction at the input's peak level.
        total_makeup -= float(gain_computer(_peak_db(x), threshold, ratio, knee)[0])

    return x * 10.0 ** ((gain_db + total_makeup) / 20.0)  # applied sample by sample
