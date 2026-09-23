"""
Signal Doctor -- measure what is wrong with a recording, and prescribe the recipe
=================================================================================
In plain words: run a handful of measurements on the file, say which ones look wrong,
and for each problem name the ordinary recipe cards that fix it.  "Fix automatically"
in the UI just appends those cards -- nothing here processes audio, and every fix is a
card you can see, tweak or bypass.

Notation: x[n] the mono fold (what the recipe processes), c_i[n] the channels, N samples.

1. Clipping -- flat tops
------------------------
A converter that ran out of range writes the SAME extreme value several samples in a
row; a real waveform only touches its maximum.  So

        runs   = declip.clipped_runs(x, 0.999)   (>= 3 samples at >= 99.9 % of the peak)
        counted only if peak >= -1 dBFS (0.891) -- flat tops far below full scale are a
                                            file that was clipped and then turned down,
                                            which declip still repairs, so the flag goes
                                            by the SHAPE when the level is near full scale
        share  = (samples in runs) / N
        warn if any run, bad if share > 0.1 %          fix: De-clip

Measured on every CHANNEL, not the mono fold: a converter clips each channel on its own,
and averaging a clipped left with a quieter right drags the fold's peak under the
-1 dBFS gate, hiding the damage.  The fix still runs on the fold, where the flat tops
survive (both channels are flat over the same samples, so their average is too).

2. Hiss -- broadband noise in the pauses
----------------------------------------
Only the pauses show the noise on its own.  With the 10 ms hop levels L[k] of
silence.py:

        quiet  = hops with L[k] <= 10th percentile       loud = 90th percentile
        judged only if  loud - quiet >= 15 dB   (otherwise there are no real pauses, and
                                               the "quiet" frames are programme material)
        P(f)   = mean over quiet hops of |rfft(hann * frame)|^2
        hf     = sum_{f >= 2 kHz} P(f) / sum_{f >= 100 Hz} P(f)
        floor  = 10th-percentile level L_q  (dBFS)

White noise puts (fs/2 - 2000) / (fs/2 - 100) of its power above 2 kHz (75 % at 16 kHz);
a room's rumble or hum puts almost none there.  So

        hiss if  floor > -60 dBFS  and  hf > 0.5         bad if floor > -45 dBFS
        fix: Noise Remover (+ a 4th-order low-pass at 8 kHz when fs/2 > 10 kHz, since
             hiss above the voice's useful band is pure loss)

3. Mains hum -- a line at 50 or 60 Hz and its harmonics
-------------------------------------------------------
One rfft of (at most HUM_SECONDS of) the file -- resolution fs / N, 0.03 Hz for 30 s;
hum is stationary, so a longer stretch adds memory, not information.  For each mains
frequency F and harmonic h = 1..4:

        peak(hF)  = max P(f)       over |f - hF| <= 1 Hz
        bed(hF)   = max P(f)       over 3 Hz <= |f - hF| <= 15 Hz
        prominence = 10 log10( peak / bed )            dB

Max against MAX, not against the median: the largest of a few dozen bins of any noisy
spectrum sits several dB over their median by chance alone (the max of M exponential
variables grows like ln M), so a median bed flagged ordinary speech as hum.  Two
maxima of neighbouring bands differ by chance by only a few dB, and a mains line is
far narrower than any musical note (it never wanders); speech pitch moves, so its
harmonics smear over many bins and are no sharper than their neighbours.

        hum if max_h prominence >= 12 dB                 bad if >= 20 dB
        fix: one Filter notch (Q = 30, ~2 Hz wide) per prominent harmonic

4. DC offset
------------
        dc = (1/N) sum x[n]                              (analysis.measure)
        warn if |dc| > 0.01, bad if |dc| > 0.05          fix: Level & DC (DC only)

5. Low level
------------
        RMS of the ACTIVE hops only (L[k] > -60 dBFS) -- pauses would drag a perfectly
        normal recording down
        warn if < -30 dBFS, bad if < -40 dBFS
        fix: Level & DC to a -1 dB peak -- plus a Compressor first if the crest factor
             (peak - RMS) is over 20 dB, since normalising the peak alone would then
             leave the average still quiet

6. Long silences
----------------
A fixed -40 dBFS would call all of a quiet (but otherwise fine) recording "silence",
so the threshold follows the file's own loud level L_90 (90th percentile of L[k]):

        T      = min(-40 dB, L_90 - 30 dB)
        pauses = silence.silent_runs(x, T, 0.3 s)
        warn if their total > 20 % of the file
        fix: Silence Remover with threshold T -- it runs BEFORE the level fix, so it
             must judge the file at the level it has then

7. Stereo balance (read from the channels BEFORE the mono fold)
---------------------------------------------------------------
        diff  = RMS_dB(L) - RMS_dB(R)
        rho   = sum L R / sqrt( sum L^2  sum R^2 )        (correlation, -1 .. 1)
        warn if |diff| > 3 dB
        bad  if rho < -0.3: the channels are largely out of phase, and the mono fold
             (L + R) / 2 that every recipe starts from will cancel them
        no fix card: processing is mono, so this is information only

Order of the combined fix
-------------------------
Order changes the sound, so the prescription is always assembled in one order:

        DC -> de-clip -> hum notches -> noise remover -> low-pass -> silence -> compressor
           -> level

DC first (every later measurement assumes a centred wave), de-clip before any filter
smears the flat tops, noise removal before silence detection (the pauses get quieter,
so they are found), and gain last so nothing after it can push the peak back over.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .analysis import measure, to_db
from .declip import clipped_runs
from .silence import hop_levels_db, silent_runs

MAINS = (50.0, 60.0)
HUM_HARMONICS = 4
HUM_SECONDS = 30.0

_STAGES = ["dc", "declip", "hum", "noise", "lowpass", "silence", "compressor", "level"]


def _finding(id_: str, status: str, title: str, value: str, detail: str = "",
             fix: list[dict[str, Any]] | None = None, stage: str | None = None) -> dict:
    return {"id": id_, "status": status, "title": title, "value": value, "detail": detail,
            "fix": fix or [], "stage": stage}


def _step(op: str, **params: Any) -> dict[str, Any]:
    """One recipe card, in the UI's display units (the same contract as presets)."""
    return {"op": op, "bypass": False, "params": params}


# ---- 1. clipping ---------------------------------------------------------------------
def check_clipping(x: np.ndarray, channels: np.ndarray | None = None) -> dict:
    signals = [x] if channels is None or channels.ndim < 2 else list(channels.T)
    peak, runs, share = 0.0, [], 0.0
    for signal in signals:                      # the worst channel decides
        p = float(np.max(np.abs(signal))) if signal.size else 0.0
        r = clipped_runs(signal, 0.999) if p >= 10 ** (-1 / 20) else []
        s = sum(b - a for a, b in r) / signal.size if signal.size else 0.0
        peak = max(peak, p)
        if s > share or (r and not runs):
            runs, share = r, s
    if not runs:
        return _finding("clipping", "ok", "No clipping", f"peak {to_db(peak):.1f} dBFS")
    status = "bad" if share > 0.001 else "warn"
    return _finding(
        "clipping", status, "Clipping detected",
        f"{len(runs)} flat tops · {100 * share:.2f} % of samples",
        "The loudest peaks were sliced flat. De-clip redraws them.",
        [_step("declip", thresh=99)], "declip",
    )


# ---- 2. hiss -------------------------------------------------------------------------
def check_hiss(x: np.ndarray, fs: int) -> dict:
    levels, hop = hop_levels_db(x, fs)
    if levels.size < 20:
        return _finding("hiss", "ok", "Too short to judge background noise", "")
    quiet_db = float(np.percentile(levels, 10))
    loud_db = float(np.percentile(levels, 90))
    if loud_db - quiet_db < 15.0:
        return _finding("hiss", "ok", "No pauses to measure noise in",
                        f"quietest 10 % at {quiet_db:.0f} dBFS")

    n_fft = 1 << int(np.ceil(np.log2(max(64, 2 * hop))))
    window = np.hanning(n_fft)
    quiet_hops = np.flatnonzero(levels <= quiet_db)
    starts = np.clip(quiet_hops * hop + hop // 2 - n_fft // 2, 0, max(0, x.size - n_fft))
    frames = np.stack([x[s:s + n_fft] for s in starts if s + n_fft <= x.size])
    power = np.mean(np.abs(np.fft.rfft(frames * window, axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    total = float(np.sum(power[freqs >= 100.0])) + 1e-30
    hf = float(np.sum(power[freqs >= 2000.0])) / total

    value = f"floor {quiet_db:.0f} dBFS · {100 * hf:.0f} % above 2 kHz"
    if quiet_db <= -60.0 or hf <= 0.5:
        return _finding("hiss", "ok", "No high-frequency noise", value)
    fix = [_step("noise_remover", amount=2.0, profile="auto")]
    if fs / 2.0 > 10_000.0:
        fix.append(_step("filter", mode="lowpass", cutoff=8000.0, order="4"))
    return _finding(
        "hiss", "bad" if quiet_db > -45.0 else "warn", "High-frequency noise detected",
        value, "Broadband hiss fills the pauses. Noise Remover learns it from them.",
        fix, "noise",
    )


# ---- 3. hum --------------------------------------------------------------------------
def check_hum(x: np.ndarray, fs: int) -> dict:
    if x.size < fs // 2:              # < 0.5 s: 2 Hz resolution cannot separate a line
        return _finding("hum", "ok", "Too short to judge hum", "")
    x = x[: int(HUM_SECONDS * fs)]
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs)

    def prominence(f: float) -> float:
        near = np.abs(freqs - f)
        line, bed = power[near <= 1.0], power[(near >= 3.0) & (near <= 15.0)]
        if line.size == 0 or bed.size == 0:
            return 0.0
        return float(10.0 * np.log10((np.max(line) + 1e-30) / (np.max(bed) + 1e-30)))

    best = None
    for mains in MAINS:
        scores = [(h * mains, prominence(h * mains)) for h in range(1, HUM_HARMONICS + 1)
                  if h * mains < fs / 2.0]
        top = max(p for _, p in scores)
        if best is None or top > best[1]:
            best = (mains, top, scores)
    mains, top, scores = best
    if top < 12.0:
        return _finding("hum", "ok", "No mains hum", f"strongest line {top:.0f} dB")
    lines = [f for f, p in scores if p >= 12.0]
    return _finding(
        "hum", "bad" if top >= 20.0 else "warn", f"{mains:.0f} Hz hum detected",
        f"{top:.0f} dB line · {len(lines)} harmonic{'s' if len(lines) > 1 else ''}",
        "A steady electrical buzz. Narrow notches remove the lines and nothing else.",
        [_step("filter", mode="notch", cutoff=round(f, 1), q=30.0) for f in lines], "hum",
    )


# ---- 4. DC ---------------------------------------------------------------------------
def check_dc(stats: dict) -> dict:
    dc = stats["dc"]
    value = f"{dc:+.4f}"
    if abs(dc) <= 0.01:
        return _finding("dc", "ok", "No DC offset", value)
    return _finding(
        "dc", "bad" if abs(dc) > 0.05 else "warn", "Large DC offset detected", value,
        "The waveform sits off the centre line, wasting headroom.",
        [_step("level", dc=True, target="none")], "dc",
    )


# ---- 5. level ------------------------------------------------------------------------
def check_level(x: np.ndarray, fs: int, stats: dict) -> tuple[dict, list[dict]]:
    """Returns the finding, plus a compressor step when the level fix needs one."""
    levels, hop = hop_levels_db(x, fs)
    active = np.repeat(levels > -60.0, hop)[: x.size]
    rms_db = to_db(float(np.sqrt(np.mean(x[active] ** 2)))) if np.any(active) else -120.0
    value = f"RMS {rms_db:.1f} dBFS (speaking parts)"
    if rms_db >= -30.0:
        return _finding("level", "ok", "Healthy level", value), []
    crest = stats["peak_db"] - rms_db
    extra = [_step("compressor")] if crest > 20.0 else []
    return _finding(
        "level", "bad" if rms_db < -40.0 else "warn", "Low RMS level", value,
        "Quiet recording. Level & DC brings the peak up to -1 dB"
        + (", after a Compressor evens out the peaks." if extra else "."),
        [_step("level", dc=False, target="peak", peak_db=-1.0)], "level",
    ), extra


# ---- 6. silences ---------------------------------------------------------------------
def check_silence(x: np.ndarray, fs: int) -> dict:
    levels, _ = hop_levels_db(x, fs)
    threshold = round(min(-40.0, float(np.percentile(levels, 90)) - 30.0)) if levels.size else -40
    runs = silent_runs(x, fs, threshold, 0.3)
    total = sum(b - a for a, b in runs) / fs
    share = total / (x.size / fs) if x.size else 0.0
    value = f"{total:.1f} s in {len(runs)} pause{'s' if len(runs) != 1 else ''}"
    if share <= 0.2:
        return _finding("silence", "ok", "No long silences", value)
    return _finding(
        "silence", "warn", "Long silences", f"{value} · {100 * share:.0f} %",
        "Much of the file is silent. Silence Remover shortens every long pause.",
        [_step("silence_remover", threshold_db=max(-80, threshold))], "silence",
    )


# ---- 7. stereo -----------------------------------------------------------------------
def check_stereo(channels: np.ndarray | None) -> dict:
    if channels is None or channels.ndim < 2 or channels.shape[1] < 2:
        return _finding("stereo", "ok", "Mono file", "1 channel")
    left, right = channels[:, 0], channels[:, 1]
    l_db = to_db(float(np.sqrt(np.mean(left ** 2))))
    r_db = to_db(float(np.sqrt(np.mean(right ** 2))))
    denom = float(np.sqrt(np.sum(left ** 2) * np.sum(right ** 2)))
    rho = float(np.sum(left * right) / denom) if denom > 0 else 1.0
    diff = l_db - r_db
    value = f"L-R {diff:+.1f} dB · correlation {rho:+.2f}"
    if rho < -0.3:
        return _finding("stereo", "bad", "Channels out of phase", value,
                        "Averaging to mono (which every recipe does) will cancel them.")
    if abs(diff) > 3.0:
        side = "right" if diff > 0 else "left"
        return _finding("stereo", "warn", "Stereo channels unbalanced", value,
                        f"The {side} channel is {abs(diff):.1f} dB quieter.")
    return _finding("stereo", "ok", "Stereo channels balanced", value)


def diagnose(x: np.ndarray, fs: int, channels: np.ndarray | None = None) -> dict[str, Any]:
    """Every check, plus the combined prescription in the fixed order (docstring)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"findings": [], "fix": []}
    stats = measure(x, fs)
    level_finding, level_extra = check_level(x, fs, stats)
    findings = [
        check_clipping(x, channels),
        check_hiss(x, fs),
        check_hum(x, fs),
        check_dc(stats),
        level_finding,
        check_silence(x, fs),
        check_stereo(channels),
    ]
    by_stage: dict[str, list[dict]] = {stage: [] for stage in _STAGES}
    for finding in findings:
        if finding["stage"] == "noise":           # the low-pass has its own later slot
            for step in finding["fix"]:
                slot = "lowpass" if step["params"].get("mode") == "lowpass" else "noise"
                by_stage[slot].append(step)
        elif finding["stage"]:
            by_stage[finding["stage"]].extend(finding["fix"])
    by_stage["compressor"].extend(level_extra)
    fix = [step for stage in _STAGES for step in by_stage[stage]]
    for finding in findings:
        finding.pop("stage")
    return {"findings": findings, "fix": fix}
