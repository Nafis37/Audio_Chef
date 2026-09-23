"""
Filter -- low-pass / high-pass (Butterworth), band-pass, notch
==============================================================
In plain words
--------------
A low-pass lets everything BELOW a cutoff frequency through and removes what is above it:
speech turns muffled, like a voice through a wall.  A high-pass is the mirror image (thin,
tinny -- all the body gone).  A band-pass keeps only a band around a centre (the
"telephone" sound), and a notch removes one narrow band and leaves the rest (mains hum).

Unlike the equalizer's shelves, which lift or cut by a few dB, these REMOVE: on the
spectrogram the content past the cutoff goes dark, and the response curve on the card
shows exactly how fast it falls away.

Source: R. Bristow-Johnson, "Cookbook formulae for audio EQ biquad filter coefficients"
(the RBJ Audio EQ Cookbook) -- the same source as eq.py, whose _design() / _normalise()
are reused here.  S. Butterworth, "On the Theory of Filter Amplifiers" (1930) for the pole
positions.  scipy.signal.lfilter only RUNS the difference equations; every coefficient is
derived below.

From s to z -- the bilinear transform
-------------------------------------
Each filter starts as an analog prototype H(s) with its corner at s = j (1 rad/s) and is
mapped to the z-plane by

        s  =  K (1 - z^-1) / (1 + z^-1),          K = 1 / tan(w0 / 2),   w0 = 2 pi fc / fs

This maps the whole j-omega axis onto the unit circle once (so a stable analog filter
stays stable and nothing aliases), but it WARPS frequency:  Omega = K tan(w / 2).  Choosing
K = 1/tan(w0/2) -- the pre-warp -- makes the analog corner Omega = 1 land exactly on the
digital w0, so the cutoff is where it was asked for.  Substituting and collecting powers
of z^-1 gives, with

        c = cos(w0),      alpha = sin(w0) / (2 Q)

the four second-order sections (all share the denominator 1 + s/Q + s^2):

    prototype                  b0          b1          b2        a0        a1      a2
    low-pass   1/(s^2+s/Q+1)   (1-c)/2     1-c         (1-c)/2   1+alpha   -2c     1-alpha
    high-pass  s^2/(...)       (1+c)/2     -(1+c)      (1+c)/2   1+alpha   -2c     1-alpha
    band-pass  (s/Q)/(...)     alpha       0           -alpha    1+alpha   -2c     1-alpha
    notch      (s^2+1)/(...)   1           -2c         1         1+alpha   -2c     1-alpha

Quick checks straight from the table:
  * low-pass at DC (z = 1):      b-sum = 2(1-c),  a-sum = 2(1-c)          -> |H| = 1
    low-pass at Nyquist (z = -1): b0 - b1 + b2 = 0                        -> |H| = 0
  * high-pass is the mirror: 0 at DC, 1 at Nyquist.
  * band-pass at z = e^{j w0}: numerator and denominator both reduce to 2j alpha sin(w0)
    times the same factor, so |H(w0)| = 1 -- 0 dB at the centre, whatever Q is.
  * notch: b = [1, -2c, 1] = (1 - e^{jw0} z^-1)(1 - e^{-jw0} z^-1), zeros ON the unit
    circle at +-w0, so |H(w0)| = 0 exactly (-inf dB).  Q sets how wide the dip is: the
    -3 dB width is about fc / Q.

Poles: 1 + alpha > |1 - alpha| and |2c| < 1 + alpha for 0 < w0 < pi and Q > 0, so both
poles are inside the unit circle -- which is why _design() keeps fc below 0.49 fs.

Steeper slopes -- a Butterworth cascade
---------------------------------------
One section falls at 12 dB/octave.  An order-N Butterworth filter falls at 6N dB/octave
and is "maximally flat": its analog magnitude is

        |H(j Omega)|^2 = 1 / (1 + Omega^(2N))

Its poles sit evenly on the unit circle of the s-plane, in conjugate pairs at angles
theta_k from the negative real axis:

        theta_k = pi (2k + 1) / (2N),        k = 0 .. N/2 - 1

A pole pair at angle theta is the factor  s^2 + 2 cos(theta) s + 1,  i.e. a second-order
section with

        Q_k = 1 / (2 cos(theta_k))

        N = 2:  0.7071
        N = 4:  0.5412   1.3066
        N = 6:  0.5176   0.7071   1.9319
        N = 8:  0.5098   0.6013   0.9000   2.5629

So the order-N filter is N/2 of the biquads above, one per Q_k, run one after the other.
Every section is warped by the same K, so the cascade is exactly the bilinear transform of
the analog Butterworth, and at the cutoff (Omega = 1)

        |H|^2 = 1 / (1 + 1) = 1/2     ->   -3.01 dB   for EVERY order.

Past the cutoff the bilinear warp squeezes the whole analog axis into 0..fs/2, so the
digital filter falls a little FASTER than 6N dB/oct near Nyquist, never slower.

Why a cascade and not one polynomial: multiplying the N/2 sections out into a single
order-N [b], [a] pair puts all the poles into the roots of one polynomial, and for a low
cutoff those roots crowd together near z = 1 where a rounding error of 1e-16 in a
coefficient moves them by far more -- an 8th-order low-pass at 50 Hz can be pushed
outside the unit circle and blow up.  Each biquad on its own is well conditioned, so each
gets its own lfilter call (what eq.py does too).

The response curve on the card
------------------------------
filter_response() evaluates  H(e^{jw}) = prod_k B_k(e^{jw}) / A_k(e^{jw})  directly from
the same coefficients (plain numpy, no freqz), so the curve drawn is the filter that runs.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .eq import _design, _normalise

MODES = ("lowpass", "highpass", "bandpass", "notch")
ORDERS = ("2", "4", "6", "8")

# The response curve is drawn down to here; below it the card's plot is just the floor.
RESPONSE_FLOOR_DB = -90.0


def butterworth_qs(order: int) -> list[float]:
    """Q of each second-order section of an order-N Butterworth (module docstring)."""
    order = max(2, 2 * (int(order) // 2))                  # even orders only: N/2 biquads
    return [
        1.0 / (2.0 * np.cos(np.pi * (2 * k + 1) / (2 * order)))   # Q_k = 1 / (2 cos theta_k)
        for k in range(order // 2)
    ]


def lowpass_coefficients(fs: int, fc: float, q: float):
    """RBJ low-pass: 1 at DC, 0 at Nyquist, |H(fc)| = Q (so 1/sqrt(2) = -3 dB at Q = 0.7071)."""
    _, w0, alpha = _design(fs, fc, q, 0.0)
    c = np.cos(w0)
    return _normalise(
        (1.0 - c) / 2.0,                  # b0: \
        1.0 - c,                          # b1:  } (1 + z^-1)^2 * (1-c)/2 -> double zero at Nyquist
        (1.0 - c) / 2.0,                  # b2: /
        1.0 + alpha,                      # a0: the shared denominator 1 + s/Q + s^2, warped
        -2.0 * c,                         # a1: poles at angle ~w0
        1.0 - alpha,                      # a2: pole radius -- smaller alpha (higher Q) = closer to 1
    )


def highpass_coefficients(fs: int, fc: float, q: float):
    """RBJ high-pass: 0 at DC, 1 at Nyquist -- the low-pass with c -> -c on the zeros."""
    _, w0, alpha = _design(fs, fc, q, 0.0)
    c = np.cos(w0)
    return _normalise(
        (1.0 + c) / 2.0,                  # b0: \
        -(1.0 + c),                       # b1:  } (1 - z^-1)^2 * (1+c)/2 -> double zero at DC
        (1.0 + c) / 2.0,                  # b2: /
        1.0 + alpha,                      # a0
        -2.0 * c,                         # a1
        1.0 - alpha,                      # a2
    )


def bandpass_coefficients(fs: int, fc: float, q: float):
    """RBJ band-pass with 0 dB at the centre (constant peak gain)."""
    _, w0, alpha = _design(fs, fc, q, 0.0)
    c = np.cos(w0)
    return _normalise(
        alpha,                            # b0: \  alpha (1 - z^-2): zeros at DC and Nyquist,
        0.0,                              # b1:  } scaled so |H(w0)| = 1 exactly
        -alpha,                           # b2: /
        1.0 + alpha,                      # a0
        -2.0 * c,                         # a1
        1.0 - alpha,                      # a2
    )


def notch_coefficients(fs: int, fc: float, q: float):
    """RBJ notch: a zero pair ON the unit circle at +-w0, unity everywhere else."""
    _, w0, alpha = _design(fs, fc, q, 0.0)
    c = np.cos(w0)
    return _normalise(
        1.0,                              # b0: \
        -2.0 * c,                         # b1:  } 1 - 2c z^-1 + z^-2 -> zeros at e^{+-j w0}
        1.0,                              # b2: /
        1.0 + alpha,                      # a0: poles at the same angle, just inside the circle,
        -2.0 * c,                         # a1: so the dip is narrow and the rest stays at 0 dB
        1.0 - alpha,                      # a2
    )


def sections(fs: int, mode: str, cutoff: float, order: int | str, q: float):
    """The (b, a) biquads that make up this filter, in the order they run."""
    if mode in ("lowpass", "highpass"):
        design = lowpass_coefficients if mode == "lowpass" else highpass_coefficients
        return [design(fs, cutoff, qk) for qk in butterworth_qs(int(order))]
    design = bandpass_coefficients if mode == "bandpass" else notch_coefficients
    return [design(fs, cutoff, q)]


def cutoff_filter(
    x: np.ndarray,
    fs: int,
    mode: str = "lowpass",
    cutoff: float = 500.0,
    order: str = "4",
    q: float = 2.0,
) -> np.ndarray:
    """Run the filter: one lfilter call per second-order section."""
    y = np.asarray(x, dtype=np.float64)
    for b, a in sections(fs, mode, cutoff, order, q):
        y = lfilter(b, a, y)
    return y


def filter_response(
    fs: int,
    mode: str = "lowpass",
    cutoff: float = 500.0,
    order: str = "4",
    q: float = 2.0,
    n: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """|H(e^{jw})| in dB on n log-spaced frequencies, 20 Hz .. min(20 kHz, 0.49 fs).

    Evaluated straight from the coefficients sections() returns -- the same ones
    cutoff_filter() runs.  Clamped at RESPONSE_FLOOR_DB (the notch's exact zero is -inf).
    """
    freqs = np.geomspace(20.0, min(20_000.0, 0.49 * fs), n)
    w = 2.0 * np.pi * freqs / fs
    z1 = np.exp(-1j * w)                                 # z^-1 on the unit circle
    z2 = z1 * z1                                         # z^-2
    h = np.ones(n, dtype=np.complex128)
    for b, a in sections(fs, mode, cutoff, order, q):
        h *= (b[0] + b[1] * z1 + b[2] * z2) / (a[0] + a[1] * z1 + a[2] * z2)   # B/A per section
    db = 20.0 * np.log10(np.maximum(np.abs(h), 10.0 ** (RESPONSE_FLOOR_DB / 20.0)))
    return freqs, db
