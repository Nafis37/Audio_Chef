"""
Echo and Reverb Studio -- impulse response convolution
======================================================
In plain words
--------------
Echo: distinct copies of the sound come back after a fixed delay, each one quieter than
the last -- shouting across a canyon.  Delay is the gap between copies, Repeats how much
of each copy survives into the next.

Reverb: thousands of reflections arrive so close together that they blur into one smooth
tail -- singing in a hall.  Room size moves the first reflections further apart, Tail
length is how long the room keeps ringing (RT60), and Wet is how much room you hear
against the original.

Both effects are LTI systems, so each is completely described by its impulse response h
and applied as one convolution  y = x * h.

Echo -- the feedback delay as an IR
-----------------------------------
The textbook echo is the feedback comb filter

        y[n] = x[n] + g * y[n - D],        D = round(delay * fs),  0 <= g < 1

Unrolling the recursion (substitute y[n - D] into itself):

        y[n] = x[n] + g x[n-D] + g^2 x[n-2D] + ...    =   sum_k g^k x[n - kD]

which is a convolution with the sparse impulse response

        h[kD] = g^k,     h = 0 elsewhere        (H(z) = 1 / (1 - g z^-D))

The IR is truncated once the taps have decayed _TAIL_DB below the first:

        g^k < 10^(-_TAIL_DB/20)    ->    k > ln(10^(-_TAIL_DB/20)) / ln(g)

At 240 dB that is far below float64 round-off relative to the signal, so the convolution
is not an approximation of the recursion but equal to it (tests/test_dsp.py checks this
against a per-sample loop).  g is capped at 0.95: at g = 1 the taps never decay.

Reverb -- a synthesised room
----------------------------
A room's IR has three parts, all built in _room_ir():

  1. Pre-delay: silence until the first wall reflection returns (_PREDELAY_MS * scale).
  2. Early reflections: a handful of discrete taps at mutually prime delays (so they never
     line up into a pitched flutter), the nth one attenuated _EARLY_GAIN^n.  Their
     spacing is what the ear reads as room SIZE, so room_size scales them.
  3. The diffuse tail: Gaussian noise under an exponential envelope.  RT60 is by
     definition the time to fall 60 dB, i.e. amplitude x 10^-3:

        env[n] = 10^(-3 n / (RT60 * fs))              (env = 10^-3 at n = RT60*fs)

     The tail fades in over _BUILDUP_MS rather than starting on a step.

Absorption: real surfaces eat highs faster than lows.  A static linear tilt on the IR's
spectrum,  |H(f)| *= 1 - _HF_DAMPING * f / f_Nyquist,  approximates it.

Energy normalisation: the IR is scaled so  sum h^2 = 1.  For a white input the output
power then equals the input power (Parseval), so decay and room size change the
CHARACTER of the tail but not its loudness.  The noise is drawn from a fixed seed so
every bake of the same settings is identical.

Linear vs circular convolution
------------------------------
Multiplying two length-L FFTs computes a CIRCULAR convolution: the tail past L wraps
around onto the start.  Zero-padding both to  L >= len(x) + len(h) - 1  makes the
circular result equal to the linear one; _fft_convolve rounds that up to a power of two
and then cuts the output back to len(x) (effects preserve length).

Dry / wet
---------
        y = (1 - m) x + m (x * h)

For the echo, h[0] = 1 already carries the dry signal, so mix fades in the repeats; the
room IR has no dry tap, so mix = 1 is fully wet.
"""

from __future__ import annotations

import numpy as np


# An echo IR is finished once it has decayed this far below its own first sample.  The
# number is deliberately far past audibility (16-bit quantisation is -96 dB): the point
# is for the truncation to disappear into float64 arithmetic, so that the convolution is
# not merely a good approximation of the old recursion but indistinguishable from it.
# It costs nothing to be this careful -- the IR is capped at the file length regardless.
_TAIL_DB = 240.0


def _fft_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Linear convolution of x with h, cut back to len(x).

    The zero padding to len(x) + len(h) - 1 is what turns the FFT's inherently CIRCULAR
    product into a linear convolution.  Without it the tail wraps around and the end of
    the reverb is heard at the start of the file.
    """
    n = x.size + h.size - 1
    fft_len = 1 << (n - 1).bit_length()      # rfft is fastest on a power of two
    spectrum = np.fft.rfft(x, fft_len) * np.fft.rfft(h, fft_len)
    # Everything past x.size is the effect ringing on past the end of the input.  Every
    # operation except the editor preserves length, so that part is dropped.
    return np.fft.irfft(spectrum, fft_len)[:x.size]


def _echo_ir(delay: int, g: float, max_len: int) -> np.ndarray:
    """The tapped delay response h[k*D] = g^k, ending once the taps have died away."""
    if g <= 0.0:
        taps = 1                    # no feedback: the input and nothing after it
    else:
        # Solve g^k < 10^(-_TAIL_DB/20) for k.  Both logs are negative, so the division
        # flips the inequality and floor() lands on the last audible tap.
        taps = int(np.floor(np.log(10.0 ** (-_TAIL_DB / 20.0)) / np.log(g))) + 1

    # A long delay with heavy feedback can want more taps than the file has room for.
    length = min(max_len, (taps - 1) * delay + 1)
    h = np.zeros(length, dtype=np.float64)
    index = np.arange(0, length, delay)
    h[index] = g ** np.arange(index.size)
    return h


# Early reflections: how long the first bounces off the near surfaces take, in ms.
# Mutually prime, so the taps never line up into a pitched flutter.
_EARLY_MS = (11.3, 17.9, 23.1, 29.7, 37.1, 43.7)
# What one bounce costs -- the nth reflection has been absorbed n times over.
_EARLY_GAIN = 0.75
# Silence before the first reflection returns: the cue for how far away the walls are.
_PREDELAY_MS = 8.0
# The diffuse field fades in over this long instead of arriving on a step edge.
_BUILDUP_MS = 12.0
# How much of the top octave the surfaces absorb.  0 = a tiled bathroom, 1 = curtains.
_HF_DAMPING = 0.6
# Any constant does; what matters is that it never changes between bakes.
_IR_SEED = 0x9E3779B9


def _room_ir(fs: int, scale: float, rt60: float, max_len: int) -> np.ndarray:
    """A synthetic room response: early reflections over a decaying noise tail."""
    length = int(min(max_len, max(1, round(rt60 * fs))))
    n = np.arange(length, dtype=np.float64)

    # --- the diffuse tail: noise under the RT60 envelope ----------------------------
    envelope = 10.0 ** (-3.0 * n / (rt60 * fs))
    h = np.random.default_rng(_IR_SEED).standard_normal(length) * envelope

    # Nothing comes back before the first wall, and the diffuse field takes a moment to
    # fill in -- a room does not go from empty to fully dense in one sample.
    pre = min(length, int(round(_PREDELAY_MS * scale * fs / 1000.0)))
    build = min(length - pre, max(1, int(round(_BUILDUP_MS * scale * fs / 1000.0))))
    h[:pre] = 0.0
    h[pre:pre + build] *= np.linspace(0.0, 1.0, build)

    # --- the early reflections ------------------------------------------------------
    # These sit on top of the tail rather than replacing it: they are the part the ear
    # reads as room SIZE, which is why `scale` moves them.
    for order, ms in enumerate(_EARLY_MS):
        tap = int(round(ms * scale * fs / 1000.0))
        if tap < length:
            h[tap] += _EARLY_GAIN ** (order + 1) * envelope[tap]

    # --- absorption: surfaces eat the highs faster than the lows ---------------------
    # One static tilt over the whole IR rather than a genuinely time-varying damping.
    # It is a first-order approximation, but it is the difference between a room and a
    # burst of hiss.
    spectrum = np.fft.rfft(h)
    tilt = 1.0 - _HF_DAMPING * np.linspace(0.0, 1.0, spectrum.size)
    h = np.fft.irfft(spectrum * tilt, length)

    # --- energy normalise (see module docstring) -------------------------------------
    energy = float(np.sqrt(np.sum(h * h)))
    if energy > 0.0:
        h /= energy
    return h


def echo(
    x: np.ndarray,
    fs: int,
    delay: float = 0.3,
    feedback: float = 0.4,
    mix: float = 0.5,
) -> np.ndarray:
    """Tapped delay line.  delay in seconds, feedback = g, mix = wet amount 0..1."""
    x = np.asarray(x, dtype=np.float64)
    d = max(1, int(round(delay * fs)))
    if d >= x.size:                 # delay longer than the file -> nothing would repeat
        return x
    g = float(np.clip(feedback, 0.0, 0.95))   # < 1 keeps every tap smaller than the last

    wet = _fft_convolve(x, _echo_ir(d, g, x.size))
    # The IR's h[0] = 1 tap carries the input through, so this crossfade runs between
    # "input" and "input + repeats".
    m = float(np.clip(mix, 0.0, 1.0))
    return (1.0 - m) * x + m * wet


def reverb(
    x: np.ndarray,
    fs: int,
    room_size: float = 0.5,
    decay: float = 2.0,
    mix: float = 0.3,
) -> np.ndarray:
    """Convolution reverb against a synthesised room response."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x

    scale = 0.5 + float(np.clip(room_size, 0.0, 1.0))   # 0.5x .. 1.5x the nominal delays
    rt60 = max(float(decay), 0.05)

    # No point synthesising tail past the end of the buffer: _fft_convolve cuts the
    # output at x.size, so anything longer is convolved and then thrown away.
    wet = _fft_convolve(x, _room_ir(fs, scale, rt60, x.size))

    # Unlike the echo IR this one has no dry tap -- it is purely the room -- so mix = 1
    # really is 100% wet.
    m = float(np.clip(mix, 0.0, 1.0))
    return (1.0 - m) * x + m * wet
