# Guide: add Filter, Paulstretch and Chorus/Flanger to Audio Chef

## Context
You asked which of these three are missing: (1) low-pass / high-pass / band-pass / notch filtering,
(2) Paulstretch, (3) chorus / flanger. **All three are missing.**
- `eq.py` has only a low shelf, a peaking bell and a high shelf. There is no cutoff filter, so
  no op makes content above a frequency "disappear".
- `speed_pitch.py` is a phase-locked vocoder that preserves phase. Nothing randomises phase or
  stretches by 20–50× (speed goes down to 0.25×, which is only a 4× stretch).
- `echo_reverb.py` uses fixed delays only. There is no LFO-modulated fractional delay anywhere.

**What happens once you approve:**
1. `git checkout -b feature/filter-paulstretch-chorus` from `main` (the tree is clean). Nothing goes on `main`, and nothing is pushed.
2. This guide is saved on that branch as `docs/NEW_EFFECTS_GUIDE.md`.
3. No effect code gets written until you say which of the three to build.

All three follow the project's hard rule: numpy (`rfft/irfft`, `np.interp`) plus `scipy.signal.lfilter`
running coefficients derived here. Nothing is banned.

---

## The pattern every new op follows (applies to all three)

| # | File | What to add |
|---|------|-------------|
| a | `backend/app/dsp/<module>.py` | Opens with a docstring deriving the maths in ASCII and naming its source, with every coefficient line annotated (CLAUDE.md comment convention). The handler signature is `fn(x: np.ndarray, fs: int, **params) -> np.ndarray`. |
| b | `backend/app/dsp/dsp_engine.py` | Import the handler and add one dict to `OPERATIONS` with `id, label, icon, category, summary, how, listen_for, handler, quick, params`. Build params with the existing `_num / _pct / _enum / _bool` helpers. Use `_pct` for anything 0..1 (it sets `scale=0.01`, so the handler still gets 0..1). |
| c | `backend/tests/test_dsp.py` | Maths tests. `test_every_default_is_audible`, `test_schema_extras_point_at_real_params` and `test_no_banned_dsp_library_is_used` then cover the new op automatically. |
| d | `frontend/src/explain.ts` | One entry in `BUILDERS` (`(get, fs) => ({equation, effect})`). `%` params arrive as 0..100, so read them with `fraction(get, name)`. |
| e | `frontend/src/icons.ts` | Import the lucide icon and add it to `ICONS`. An unknown name silently falls back to `Wand2`. |
| f | optional: `frontend/src/presets.ts` | A starter recipe (values in display units). |
| g | `CLAUDE.md` | A file-map line for the new module, plus a gotcha if one comes up. |

Sliders, the Advanced section, `show_when`, quick chips and bypass all render from the schema. **Do not write any React controls.**

---

## 1. Filter: low-pass / high-pass / band-pass / notch

**New module:** `backend/app/dsp/filters.py`. **Op id:** `filter`. **Category:** `"Tone & level"`. **Icon:** `Filter`.

### Maths (goes in the docstring)
Everything is the RBJ cookbook again, the same source and bilinear transform as `eq.py`.
With `w0 = 2π fc/fs`, `c = cos w0` and `α = sin w0 / (2Q)`:

```
low-pass   b = [(1-c)/2, 1-c, (1-c)/2]      a = [1+α, -2c, 1-α]   prototype 1/(s²+s/Q+1)
high-pass  b = [(1+c)/2, -(1+c), (1+c)/2]   a = [1+α, -2c, 1-α]   prototype s²/(s²+s/Q+1)
band-pass  b = [α, 0, -α]                   a = [1+α, -2c, 1-α]   0 dB at fc  (s/Q)/(...)
notch      b = [1, -2c, 1]                  a = [1+α, -2c, 1-α]   zeros ON the unit circle at ±w0
```

**Butterworth of order N (steeper cutoff).** Cascade N/2 biquads. Section k gets

```
θ_k = π(2k+1)/(2N),   Q_k = 1 / (2 cos θ_k)        k = 0 .. N/2-1
N=2: Q = 0.7071     N=4: Q = 0.5412, 1.3066     N=6: 0.5176, 0.7071, 1.9319
```

These Q values are the Butterworth pole pairs, which sit evenly on a circle in the s-plane. So the cascade
is maximally flat, is exactly −3.01 dB at fc for every order, and falls at 6N dB/octave (a little
steeper near Nyquist because of the bilinear warp). The z-transform story for the demo:
poles at `z = e^{s T}` under the bilinear map `s = (2/T)(1 − z⁻¹)/(1 + z⁻¹)`, pre-warped so that fc
lands exactly where it is asked for.

**Why cascade instead of one big polynomial:** multiplying the sections out into an 8th-order
`[b], [a]` pair loses precision, and the poles can drift outside the unit circle at low fc. Run each
section with its own `lfilter` call. That is also what `eq.py` does.

### Code sketch
```python
from scipy.signal import lfilter
from .eq import _design, _normalise          # reuse: clamps f0 < 0.49 fs, Q > 0.05; divides by a0

def butterworth_qs(order: int) -> list[float]:
    return [1.0 / (2.0 * np.cos(np.pi * (2*k + 1) / (2*order))) for k in range(order // 2)]

def lowpass_coefficients(fs, fc, q):  A, w0, alpha = _design(fs, fc, q, 0.0); c = np.cos(w0); ...
def highpass_coefficients(fs, fc, q): ...
def bandpass_coefficients(fs, fc, q): ...
def notch_coefficients(fs, fc, q): ...

def sections(fs, mode, cutoff, order, q):      # -> list of (b, a); also used by the response plot
    if mode in ("lowpass", "highpass"):
        design = lowpass_coefficients if mode == "lowpass" else highpass_coefficients
        return [design(fs, cutoff, qk) for qk in butterworth_qs(order)]
    design = bandpass_coefficients if mode == "bandpass" else notch_coefficients
    return [design(fs, cutoff, q)]

def cutoff_filter(x, fs, mode="lowpass", cutoff=800.0, order=4, q=0.707):
    y = x
    for b, a in sections(fs, mode, cutoff, int(order), q):
        y = lfilter(b, a, y)
    return y
```

### Schema
```python
_enum("mode", "Type", "lowpass", ["lowpass", "highpass", "bandpass", "notch"],
      option_labels={"lowpass": "Low-pass", "highpass": "High-pass",
                     "bandpass": "Band-pass", "notch": "Notch"}),
_num("cutoff", "Frequency", 800.0, 20.0, 20000.0, 1.0, "Hz", help="..."),
_enum("order", "Steepness", "4", ["2", "4", "6", "8"],
      option_labels={"2": "12 dB/oct", "4": "24 dB/oct", "6": "36 dB/oct", "8": "48 dB/oct"},
      show_when={"mode": ["lowpass", "highpass"]}),
_num("q", "Width (Q)", 2.0, 0.3, 30.0, 0.1, "", show_when={"mode": ["bandpass", "notch"]}),
```
The order is an enum of strings because `_enum` only handles strings; the handler does `int(order)`.
`_design` already clamps the cutoff below 0.49·fs, which covers a 20 kHz slider on a 22.05 kHz file.

**quick:** `"Telephone": {"mode": "bandpass", "cutoff": 1500, "q": 1.0}`,
`"Muffled": {"mode": "lowpass", "cutoff": 500, "order": "8"}`,
`"Rumble cut": {"mode": "highpass", "cutoff": 80}`,
`"Hum notch": {"mode": "notch", "cutoff": 50, "q": 10}`.

### Tests (`test_dsp.py`, reusing the `_response_db` helper at line 108)
- For each order, a low-pass `sum(_response_db(b, a, fc) for sections)` is −3.01 ± 0.1 dB. It is about 0 dB at fc/10 and ≤ −6·order dB at 2·fc. Do the mirror image for high-pass.
- The band-pass is 0 dB at fc. The notch is ≤ −60 dB at fc and about 0 dB an octave away.
- An end-to-end check: a low-pass at 500 Hz on `sine(300) + sine(4000)` leaves `dominant_hz == 300`, and the 4 kHz energy drops more than 40 dB (FFT bin check).

### Frontend
The `explain.ts` equation is `H(z) = Π_k (b₀+b₁z⁻¹+b₂z⁻²)/(1+a₁z⁻¹+a₂z⁻²)`. The effect text is:
"N/2 Butterworth sections: −3 dB at {fc} Hz, then −{6N} dB per octave. The spectrogram goes dark above that line."

### Optional: response curve next to the spectrogram (the demo tip)
- Add `filter_response(fs, mode, cutoff, order, q, n=512)` in `filters.py`. It evaluates
  `H(e^{jω}) = Π B(e^{jω})/A(e^{jω})` on log-spaced frequencies with plain numpy (`np.exp(-1j*w)`,
  no `freqz`) and returns `(freqs, dB)`. It reuses `sections()`.
- Add `GET /operations/filter/response?fs=&mode=&cutoff=&order=&q=` in `routers/process.py`. It returns JSON.
- Add a small SVG line plot inside the `filter` card (`RecipeCard.tsx`, rendered only when `step.op === "filter"`). Put the dB axis on the same −90…0 scale as `Spectrogram.tsx` so the two line up.
- Test: the curve at fc is −3 dB, the same value `_response_db` gives.

### Optional: swept notch
Run the notch in blocks of 256 samples and recompute its coefficients each block from
`fc[n] = f_lo · (f_hi/f_lo)^(½(1+sin 2π r t))`. Pass lfilter's state between blocks with
`lfilter(b, a, block, zi=z)` so the output is continuous. That gives a phaser-like sweep and is still
our own coefficients. Only do this after the static filter is done.

---

## 2. Paulstretch: extreme FFT time-stretch

**New module:** `backend/app/dsp/paulstretch.py`. **Op id:** `paulstretch`. **Category:** `"Time & pitch"`. **Icon:** `Hourglass`.

### Maths (docstring)
Source: Paul Nasca, "Paulstretch" (2006). The method:

```
W   = window length (samples, rounded to a power of two)       w = periodic Hann (stft.make_window)
H_s = W/4                          output hop  -> Hann² COLA, Σ w² = 1.5 (same constant as stft.py)
H_a = H_s / stretch                input hop   (fractional; the read position is rounded)

for output frame t:
    X_t     = rfft( w · x[ round(t·H_a) : round(t·H_a) + W ] )
    Y_t[k]  = |X_t[k]| · e^{j φ_t[k]},     φ_t[k] ~ Uniform[0, 2π)      <- throw the phase away
    out    += w · irfft(Y_t)   at offset t·H_s                            (weighted overlap-add)
out /= max(Σ w², floor)                                                   (istft's edge floor)
```

- **Why it works:** the magnitude spectrum says *which* frequencies are present. The phase says *when*
  inside the frame they happen. Randomising the phase keeps the timbre but erases the timing, so every
  transient smears into a steady texture. A plain phase vocoder (our `time_stretch`) keeps the phase
  coherent instead, which at 20× sounds like a stuttering, metallic loop.
- **The window sets the texture:** a big W (0.25–1 s) gives fine frequency resolution (fs/W Hz per bin) and a smooth drone. A small W gives grainy, noisy output.
- **Loudness:** overlapping frames with independent random phases add in *power*, not amplitude, so
  the raw output comes out quieter than the input by an amount that depends on the overlap. Finish with
  `y *= rms(x) / rms(y)` to match the input level. Document that, because it is the equivalent of the gotcha the phase-locked vocoder has (~4 dB quieter).
- **Deterministic:** use `rng = np.random.default_rng(0)`, as `reverb` does. Otherwise every Auto-Bake of the
  same settings sounds different and A/B means nothing. Set the phases of the DC and Nyquist bins to 0
  (irfft drops their imaginary part anyway).

### Code sketch
```python
from .stft import make_window
MAX_OUTPUT_SECONDS = 600.0

def paulstretch(x, fs, stretch=8.0, window=0.25):
    stretch = min(stretch, MAX_OUTPUT_SECONDS * fs / max(x.size, 1))   # cap memory/time
    W = int(2 ** np.round(np.log2(max(window * fs, 256))))
    w = make_window(W); Hs = W // 4; Ha = Hs / stretch
    xp = np.pad(x, (W // 2, W))                      # centre the first frame on sample 0
    out_len = int(round(x.size * stretch))
    n_frames = out_len // Hs + 1
    out = np.zeros(n_frames * Hs + W); wsum = np.zeros_like(out)
    rng = np.random.default_rng(0)
    for t in range(n_frames):                        # frame loop, NOT one big STFT matrix
        s = min(int(round(t * Ha)), xp.size - W)
        mag = np.abs(np.fft.rfft(xp[s:s + W] * w))
        phase = rng.uniform(0, 2*np.pi, mag.size); phase[[0, -1]] = 0
        out[t*Hs:t*Hs + W] += np.fft.irfft(mag * np.exp(1j*phase), n=W) * w
        wsum[t*Hs:t*Hs + W] += w**2
    y = out / np.maximum(wsum, 0.75)                 # 0.75 = half the Hann COLA constant, as istft
    y = y[W // 2 : W // 2 + out_len]
    return y * (rms(x) / max(rms(y), 1e-12))
```
**Why there is a loop and no `stft()` / `istft()` matrix:** a 5 s clip at 44.1 kHz stretched 50× gives
~2 700 frames × 8 193 bins of complex128, about 350 MB. Accumulating frame by frame keeps memory at
O(output length), and 2 700 rffts of 16 k points take well under a second.

### Schema
```python
_num("stretch", "Stretch", 8.0, 1.0, 50.0, 0.5, "x", help="8x = eight times longer."),
_num("window", "Smoothness", 0.25, 0.05, 1.0, 0.01, "s", advanced=True,
     help="Analysis window. Longer = smoother drone, shorter = grainier."),
```
**quick:** `"Slow drift": {"stretch": 4}`, `"Ambient": {"stretch": 12}`, `"Drone": {"stretch": 40, "window": 0.6}`.

### Tests
- The output length is `round(len(x) · stretch)`.
- The same input gives the same output twice (determinism).
- `dominant_hz` of a stretched 440 Hz sine is 440 ± fs/W. The pitch does not move.
- The output RMS is within 1 dB of the input RMS.
- A 50× stretch of a 30 s input is capped at `MAX_OUTPUT_SECONDS`.

### Frontend
- **`regions.ts` (important):** add `'paulstretch'` to `TIME_SHIFTING_OPS`. It changes the duration, so a region card below it must show the "clocks diverge" warning, and Arrange blocks must show their warning icon.
- `explain.ts`: the equation is `Y_t[k] = |X(t·H/s)[k]|·e^{jφ},  φ ~ U[0,2π)`. The effect text is:
  "Each frame keeps its spectrum and loses its timing — the output is {s}× longer and transients smear into a drone."

---

## 3. Chorus / Flanger: modulated fractional delay

**New module:** `backend/app/dsp/modulation.py`. **Op id:** `chorus_flanger`. **Category:** `"Space"`. **Icon:** `Orbit`.

### Maths (docstring)
```
Static comb (the building block):   y[n] = x[n] + g·x[n−D]
   = x * h,   h = δ[n] + g·δ[n−D]            <- a convolution with a two-impulse kernel
   H(z) = 1 + g z^{−D},   |H(e^{jω})|² = 1 + g² + 2g·cos(ωD)
   notches at f = (k + ½)·fs/D,  spacing fs/D   (D = 1 ms -> 500, 1500, 2500 Hz ...)

Feedback comb:  v[n] = x[n] + fb·v[n−D],   H(z) = 1/(1 − fb z^{−D})
   resonant peaks at k·fs/D  (fb > 0)   or  (k+½)·fs/D  (fb < 0),   |fb| ≤ 0.95 for stability

Modulate the delay with an LFO:   d[n] = D0 + A·sin(2π r n/fs + ψ)     (in samples, fractional)
Fractional read (sampling theory -- linear interpolation between neighbours):
   x[n − d] ≈ (1−f)·x[n−i] + f·x[n−i−1],    i = ⌊d⌋,  f = d − i
```
- **Flanger** (D0 ≈ 1–5 ms, feedback): the comb's notches sweep up and down the spectrum, which is the "jet" sound. On a spectrogram of noise or speech you can see moving diagonal stripes.
- **Chorus** (D0 ≈ 15–30 ms, 2–4 voices at LFO phases ψ = 2πv/V, no feedback): the notches are too
  dense to hear. What you hear is the **Doppler detune** of each voice. The instantaneous
  playback ratio is `1 − d'(t)`, whose peak deviation is `2π·r·A`. For A = 3 ms and r = 0.8 Hz that is ±1.5 %, about ±26 cents. Several slightly
  detuned copies sound like "many singers". Put that number in the docstring and in the `explain.ts` text.
- Mix: `y = (1−m)·x + m·wet`.

### Implementation
**Chorus (no feedback) is fully vectorised.** `np.interp` does the fractional read:
```python
n = np.arange(x.size)
wet = sum(np.interp(n - d_v, n, x, left=0.0) for d_v in voice_delays) / V
```

**Flanger feedback is a recursion with a time-varying delay.** Use the same block trick as the echo
(see the CLAUDE.md "Recursive filters" gotcha). Every read needs `v[n − i − 1]` with
`i = ⌊d[n]⌋ ≥ ⌊d_min⌋ = B`, so a block of B samples only reads values that are already finished.
That makes each block one vectorised numpy step:
```python
B = max(1, int(np.floor(d.min())))
for start in range(0, N, B):
    idx = np.arange(start, min(start + B, N)); di = d[idx]; i = np.floor(di).astype(int); f = di - i
    past = (1 - f) * v_at(idx - i) + f * v_at(idx - i - 1)      # v_at returns 0 for negative indices
    v[idx] = x[idx] + fb * past
```
**Keep `_flanger_scalar(x, d, fb)` as the readable per-sample reference, and add a test asserting the two
agree to 1e-12**, the same way `_comb_feedback_scalar` is kept for echo. The wet signal is the delayed tap of `v`.

### Schema
```python
_enum("mode", "Effect", "flanger", ["flanger", "chorus"]),
_num("rate", "Rate", 0.3, 0.05, 5.0, 0.05, "Hz", help="How fast the sweep moves."),
_num("depth", "Depth", 2.0, 0.1, 10.0, 0.1, "ms", help="How far the delay swings."),
_num("delay", "Base delay", 2.0, 0.5, 30.0, 0.1, "ms", advanced=True),
_pct("feedback", "Feedback", 60, lo=-95, hi=95, show_when={"mode": ["flanger"]}),
_enum("voices", "Voices", "3", ["1", "2", "3", "4"], show_when={"mode": ["chorus"]}),
_pct("mix", "Wet", 50),
```
Clamp `delay − depth/2 ≥ 0.5 ms` inside the handler so the delay never goes negative.
**quick:** `"Jet flanger": {"mode": "flanger", "rate": 0.2, "depth": 3, "delay": 2, "feedback": 70}`,
`"Subtle chorus": {"mode": "chorus", "rate": 0.8, "depth": 3, "delay": 20, "voices": "3", "mix": 40}`,
`"Wobble": {"mode": "chorus", "rate": 4, "depth": 6, "delay": 10, "mix": 60}`.

### Tests
- **Convolution identity:** with depth = 0, one voice, an integer D, 100 % wet and dry summed, the output equals `np.convolve(x, [1, 0, ..., g])[:N]`.
- **Comb notches:** white noise through the static comb (rate 0) has spectral minima at `(k+½)·fs/D` (check the first 3, ±1 bin).
- **Block vs scalar:** the block flanger equals `_flanger_scalar` for fb ∈ {−0.7, 0.5, 0.9}.
- **Length and determinism:** the length is preserved and there is no randomness.

### Frontend
`explain.ts`: flanger `y = x + g·v[n − d(n)],  d(n) = D₀ + A sin 2πrn` and chorus
`y = Σ_v x[n − d_v(n)]/V`. The effect text states the notch spacing `1/D₀` for the flanger and the cents of detune for the chorus.

---

## Suggested build order
1. **Filter.** It is smallest and reuses `eq.py`, and it gives the best theory-vs-result demo.
2. **Chorus / Flanger.** It reuses the echo's block-recursion idea.
3. **Paulstretch.** It is self-contained but needs the output-length cap and a `regions.ts` change.

Each one is its own commit on `feature/filter-paulstretch-chorus`.

## Verification (per effect)
```bash
cd backend && source .venv/bin/activate && python -m pytest tests/ -q     # all green, incl. audible-default
grep -rn "librosa\|noisereduce\|pedalboard\|iirpeak" backend/              # prints nothing
python3 run.py   &   cd ../frontend && npm run dev                          # http://localhost:5173
```
In the browser:
- The card appears in the palette under its category, and its sliders and quick chips work.
- **Filter:** at 500 Hz on speech, the Output spectrogram goes dark above the line.
- **Paulstretch:** at 20×, the output waveform is 20× longer, and a region card below it shows the warning.
- **Flanger:** on noise or speech, the spectrogram shows the moving comb stripes.
