"""
Signal Doctor -- find the problems a listener would notice, and prescribe the recipe
====================================================================================
In plain words: run a handful of measurements on the file, say which ones sound wrong,
and for each problem name the ordinary recipe cards that fix it.  "Fix automatically"
in the UI just appends those cards -- nothing here processes audio, and every fix is a
card you can see, tweak or bypass.

Six checks: clipping, uneven volume, fan / AC noise, other background noise,
low-frequency rumble and long silences.  Clipping is reported with advice instead of a
card -- no card can rebuild sliced peaks.  Only "bad" findings get cards: "warn" is
reported as minor, since a borderline number is not worth changing the sound for.

Notation: x[n] the mono fold (what the recipe processes), N samples.  Three shared
analyses of the CENTRED signal x - mean(x) (a DC offset is not rumble):

  Spectra    Hann frames of ~64 ms (hop half of that), power P[t, f] = |rfft|^2, frame
             level L[t] in dBFS, loud level L_90 = 90th percentile of L.  "Voice frames"
             are those within 20 dB of L_90.
  LongTerm   Hann segments of ~1 s (hop a quarter), for ~1 Hz resolution in the rumble
             band.
  Speech     speech_analysis.analyse every 10 ms, and speech_analysis.speech_mask: the
             VOICED frames (they have a pitch), widened by 0.2 s for the consonants
             around them.  Voicing, not level, is what tells talking from a room -- a fan
             can be as loud as quiet speech, and a take that is mostly pauses has its
             "loud" percentiles in the room noise.

1. Clipping -- flat tops                                  no card: re-record lower
    Per sign s (+ and -, which may clip at different levels) the ceiling C_s = max(s x);
    a flat top is a run of >= 3 samples (at 16 kHz; scaled with the rate) with
    s x >= 0.999 C_s that is FLAT (varies by < 0.05 % of C_s -- a rounded peak also
    spends a few samples near its top, but curves).  A clean waveform reaches its ceiling once; a clipped one sits
    on it again and again, at any final gain -- so this also finds audio clipped first
    and turned down afterwards.
    Measured per CHANNEL (a clipped left averaged with a quieter right would hide in the
    fold).  Clipped if >= 3 flat tops; bad if they cover > 0.1 % of the samples or
    number >= 20.

2/3. Background noise: fan / AC vs other                          fix: Noise Remover
    Minimum statistics, as the Noise Remover's own "auto" profile: in every bin the
    quietest frames are noise only, and for Rayleigh-distributed noise magnitudes
        noise power N(f) = 2 / (-2 ln 0.9) * P10_t(|X[t, f]|)^2        (f >= 150 Hz)
        SNR = sum_f voice LTAS(f) / sum_f N(f)
    warn if SNR < 36 dB, bad if < 30 dB (a fan 25 dB under the voice is plain to hear
    in every pause).  Only judged when the take has pauses: in a pause every bin sits at
    its floor at once, so the quietest frames (P5 of L, digital silence under -90 dBFS
    left out) are within 3 dB of sum_f N(f) -- real takes land at -5 .. 0 dB.  Speech
    without pauses reaches each bin's minimum at a different moment, so its quietest
    frames sit 4-8 dB over that sum: its per-bin minima are voice, not noise.

    The noise is then classified by its shape:
        low = share of N in 150 Hz .. 1 kHz,   high = share of N above 2 kHz
    Fan / air conditioning: low >= 0.55 and high < 0.2 -- a steady low whoosh.  Anything
    else is "other background noise" (named hiss when high > 0.5).  Exactly one of the
    two findings reports the noise; the other says there is none of that kind.
    The Noise Remover runs at 1.5x with a 2 % floor: on real takes that clears a fan to
    ~40 dB under the voice, while a harder push eats quiet words that sit only a few dB
    over the noise.  Hiss gets a 4th-order low-pass at 8 kHz after the Noise Remover when
    fs/2 > 10 kHz.

4. Rumble and mic pops                                     fix: high-pass at 80 Hz
    No voice has a fundamental under ~65 Hz, so energy there is handling, traffic, air
    or plosive bursts:
        rumble = LongTerm power under 60 Hz / power in 100 Hz .. 4 kHz
        pop    = a run of loud frames (L > L_90 - 10) with > 50 % of their power < 60 Hz
    bad if rumble > -12 dB or >= 3 pops; warn if > -20 dB or any pop.

5. Uneven volume                                                  fix: Voice Leveler
    The speech level L[t] of leveler.speech_levels: the speech frames' power over the
    room's noise floor, averaged over ~1 s windows of speech -- so pauses and the room
    never count as "quiet speech".
        spread = P90 - P10 of L over the speech frames          (>= 3 s of speech needed)
    warn if > 8 dB, bad if > 11 dB.  The fix is the Voice Leveler, which rides the gain
    phrase by phrase toward the file's own loud level.  (A Compressor cannot: it acts
    on peaks within milliseconds, so a whole sentence spoken too quietly stays quiet.)

6. Long silences (>= 3 s, anywhere, the ends included)            fix: Silence Remover
    A silence is a stretch with no syllable (see Speech above) lasting >= 3 s from one
    syllable to the next, whose inside (0.2 s clear of them, for a trailing "s" or
    breath) is >= 10 dB under the speech -- a room's fan or hiss in the gap does not
    stop it being a silence, and a take that is mostly pauses is still judged against
    its voice.  bad if there is any.  The Silence Remover judges every 10 ms hop on its
    level alone, so its threshold goes 1 dB over the loudest hop inside those silences
    (and at least 6 dB under the speech) -- the room's level, so a quiet stretch of
    voice is never taken for a pause -- with "Longer than" 2.6 s, as the consonants at
    the edges may be over it.  It shortens each to 0.8 s, and removes it entirely at
    the start and end of the file.

Order of the combined fix
-------------------------
        high-pass -> noise remover -> low-pass -> silence remover -> voice leveler

Rumble goes before the noise remover so its profile is not dominated by it; silences
are cut before the leveler, whose gain would lift the pauses toward the silence
threshold.  When the Noise Remover is prescribed and the take is borderline uneven or its
speech sits < 20 dB over the room, the Voice Leveler joins it even without an "uneven"
finding: removing noise that close to the voice takes a few dB off the quietest words.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .analysis import to_db
from .leveler import speech_levels
from .silence import hop_levels_db
from .speech_analysis import SpeechFrames, analyse, speech_mask

FRAME_SECONDS = 0.064          # Spectra frame; hop is half of it
LONG_SECONDS = 1.0             # LongTerm segment; hop is a quarter of it
RAYLEIGH_P10_TO_POWER = 2.0 / (-2.0 * np.log(0.9))   # mean |X|^2 = this * P10(|X|)^2
NOISE_WARN_SNR = 36.0          # dB: noise closer to the voice than this is audible
NOISE_BAD_SNR = 30.0           # dB: ... and closer than this is worth a card
DIGITAL_SILENCE_DB = -90.0     # frames under this are a recorder's zeros, not a room
NOISE_FROM_HZ = 150.0          # below is check 5's (rumble), whose skirt reaches ~150 Hz
LONG_SILENCE_S = 3.0
# Speech frames are 50 ms long, so a pause measures a little short at each edge; this
# much slack keeps an exact 3 s pause in (and a 2.9 s one out).
SILENCE_SLACK_S = 0.05
SPEECH_PAD_S = 0.2             # consonants and breaths around the voiced frames
UNEVEN_WARN_DB, UNEVEN_BAD_DB = 8.0, 11.0
LEVEL_AFTER_DENOISE_SNR = 20.0  # dB: under this, denoising dents the quiet words

_STAGES = ["highpass", "noise", "lowpass", "silence", "leveler"]


def _finding(id_: str, status: str, title: str, value: str, detail: str = "",
             fix: list[dict[str, Any]] | None = None, stage: str | None = None) -> dict:
    return {"id": id_, "status": status, "title": title, "value": value, "detail": detail,
            "fix": fix or [], "stage": stage}


def _step(op: str, **params: Any) -> dict[str, Any]:
    """One recipe card, in the UI's display units (the same contract as presets)."""
    return {"op": op, "bypass": False, "params": params}


def _db(power: float) -> float:
    return float(10.0 * np.log10(max(power, 1e-30)))


def _frame_power(x: np.ndarray, n_fft: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """(|rfft|^2 of Hann frames, mean-square level of each frame)."""
    count = max(0, 1 + (x.size - n_fft) // hop)
    if count == 0:
        return np.zeros((0, n_fft // 2 + 1)), np.zeros(0)
    index = np.arange(n_fft)[None, :] + hop * np.arange(count)[:, None]
    frames = x[index]
    power = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)) ** 2
    return power, np.mean(frames ** 2, axis=1)


class Spectra:
    """Power spectra of every ~64 ms frame, shared by the noise and rumble checks."""

    def __init__(self, x: np.ndarray, fs: int):
        n_fft = 1 << int(np.ceil(np.log2(FRAME_SECONDS * fs)))
        self.freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
        self.power, mean_sq = _frame_power(x, n_fft, n_fft // 2)
        self.level_db = 10.0 * np.log10(np.maximum(mean_sq, 1e-12))
        self.loud_db = float(np.percentile(self.level_db, 90)) if mean_sq.size else -120.0

    def band(self, lo: float, hi: float) -> np.ndarray:
        return (self.freqs >= lo) & (self.freqs < hi)

    def active(self, below_loud_db: float) -> np.ndarray:
        """Frames within `below_loud_db` of the loud (90th-percentile) level: the voice."""
        return self.level_db > max(self.loud_db - below_loud_db, -70.0)

    def voice_power(self) -> float:
        """Mean power of the voice frames, per sample (the same scale as mean x^2)."""
        active = self.active(20.0)
        if not np.any(active):
            return 0.0
        return float(np.mean(10.0 ** (self.level_db[active] / 10.0)))


class LongTerm:
    """~1 s segments: ~1 Hz resolution in the rumble band."""

    def __init__(self, x: np.ndarray, fs: int):
        n_fft = 1 << int(np.ceil(np.log2(LONG_SECONDS * fs)))
        self.freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
        power, mean_sq = _frame_power(x, n_fft, n_fft // 4)
        self.power = power[mean_sq > 1e-9]           # digital silence says nothing
        self.mean = np.mean(self.power, axis=0) if self.power.shape[0] else np.zeros(
            self.freqs.size)


# ---- clipping ------------------------------------------------------------------------
FLAT = 0.0005        # a flat top varies by less than this share of the ceiling


def _min_run(fs: int) -> int:
    """Samples a flat top must last: 3 at 16 kHz, proportionally more at higher rates,
    where a rounded peak also spends more samples within FLAT of its top."""
    return max(3, int(round(3 * fs / 16_000)))


def clipped_runs(x: np.ndarray, fs: int, thresh: float = 0.999) -> list[tuple[int, int]]:
    """[start, end) of every flat top: >= _min_run(fs) samples at one sign's ceiling."""
    x = np.asarray(x, dtype=np.float64)
    min_run = _min_run(fs)
    runs: list[tuple[int, int]] = []
    for sign in (1.0, -1.0):
        s = sign * x
        ceiling = float(np.max(s)) if s.size else 0.0
        if ceiling <= 1e-6:
            continue
        on = (s >= thresh * ceiling).astype(np.int8)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], on, [0]])))
        runs += [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2])
                 if b - a >= min_run and np.ptp(s[a:b]) <= FLAT * ceiling]
    return sorted(runs)


def check_clipping(x: np.ndarray, fs: int, channels: np.ndarray | None = None) -> dict:
    signals = [x] if channels is None or channels.ndim < 2 else list(channels.T)
    peak, runs, share = 0.0, [], 0.0
    for signal in signals:                      # the worst channel decides
        peak = max(peak, float(np.max(np.abs(signal))) if signal.size else 0.0)
        r = clipped_runs(signal, fs)
        s = sum(b - a for a, b in r) / signal.size if signal.size else 0.0
        if len(r) > len(runs):
            runs, share = r, s
    if len(runs) < 3:
        return _finding("clipping", "ok", "No clipping", f"peak {to_db(peak):.1f} dBFS")
    status = "bad" if share > 0.001 or len(runs) >= 20 else "warn"
    return _finding(
        "clipping", status, "Clipping",
        f"{len(runs)} flat tops · {100 * share:.2f} % of samples",
        "The loudest peaks were sliced flat, which crackles. No recipe card can bring "
        "them back: re-record with the input level turned down if you can.",
    )


# ---- background noise ----------------------------------------------------------------
def measure_noise(spec: Spectra) -> tuple[float, float, float, float]:
    """(noise level dBFS, SNR dB, share of noise 150 Hz..1 kHz, share above 2 kHz)."""
    band = spec.band(NOISE_FROM_HZ, np.inf)
    freqs = spec.freqs[band]
    p10 = np.percentile(np.sqrt(spec.power[:, band]), 10, axis=0)
    noise = RAYLEIGH_P10_TO_POWER * p10 ** 2
    voice = np.mean(spec.power[spec.active(20.0)][:, band], axis=0)
    snr = _db(float(np.sum(voice))) - _db(float(np.sum(noise)))
    total = float(np.sum(noise)) + 1e-30
    low = float(np.sum(noise[freqs < 1000.0]) / total)
    high = float(np.sum(noise[freqs >= 2000.0]) / total)
    return _db(spec.voice_power()) - snr, snr, low, high


def has_pauses(spec: Spectra) -> bool:
    """Whether the per-bin minima come from real pauses (docstring, checks 2/3).

    In a pause every bin is at its floor AT ONCE, so the quietest frames are as loud as
    the sum of the per-bin floors.  Without pauses each bin reaches its minimum at a
    different moment (the formants move), and no frame is that quiet.  Digital silence
    -- a recorder starting or stopping -- is left out: it is not where the noise lives.
    """
    live = spec.level_db > DIGITAL_SILENCE_DB
    if np.count_nonzero(live) < 20:
        return False
    p10 = np.percentile(np.sqrt(spec.power[live]), 10, axis=0)
    floors = RAYLEIGH_P10_TO_POWER * p10 ** 2
    n_fft = 2 * (spec.freqs.size - 1)
    # sum |rfft(w x)|^2 -> mean x^2 of the frame: one-sided x2, Hann sum w^2 = 3N/8.
    floor_db = _db(float(np.sum(floors)) * 2.0 / (n_fft * n_fft * 0.375))
    quietest_db = float(np.percentile(spec.level_db[live], 5))
    return quietest_db - floor_db <= 3.0


def check_noise(spec: Spectra, fs: int) -> list[dict]:
    """[fan / AC finding, other-noise finding] -- at most one of them reports the noise."""
    def none(value: str = "") -> list[dict]:
        return [_finding("fan", "ok", "No fan or AC noise", value),
                _finding("noise", "ok", "No other background noise", value)]

    if spec.power.shape[0] < 20:
        return [_finding("fan", "ok", "Too short to judge fan noise", ""),
                _finding("noise", "ok", "Too short to judge background noise", "")]
    noise_db, snr, low, high = measure_noise(spec)
    value = f"noise {noise_db:.0f} dBFS · {snr:.0f} dB under the voice"
    if snr >= NOISE_WARN_SNR:
        return none(value)
    if not has_pauses(spec):
        return [_finding("fan", "ok", "No pauses to measure fan noise in", ""),
                _finding("noise", "ok", "No pauses to measure noise in", "")]
    status = "bad" if snr < NOISE_BAD_SNR else "warn"
    # 1.5x with a 2 % floor: clears a fan to ~40 dB under the voice on real takes while
    # keeping the quiet words -- a harder push (2-3x) ate speech only ~7 dB over the fan,
    # and the watery residue it left read as voice.
    fix = [_step("noise_remover", amount=1.5, profile="auto", floor=2)]
    if low >= 0.55 and high < 0.2:
        return [_finding(
            "fan", status, "Fan or AC noise", value,
            "A steady low whoosh under the voice (fan, air conditioning, computer). Noise "
            "Remover learns it from the pauses and takes it out.", fix, "noise",
        ), _finding("noise", "ok", "No other background noise", "")]
    kind = "hiss" if high > 0.5 else "noise"
    if kind == "hiss" and fs / 2.0 > 10_000.0:
        fix.append(_step("filter", mode="lowpass", cutoff=8000.0, order="4"))
    return [_finding("fan", "ok", "No fan or AC noise", ""), _finding(
        "noise", status, f"Background {kind}", value,
        f"Steady background {kind} under the voice. Noise Remover learns it and takes it out.",
        fix, "noise",
    )]


# ---- rumble and pops -----------------------------------------------------------------
def measure_rumble(spec: Spectra, lt: LongTerm) -> tuple[float, int]:
    """(power under 60 Hz relative to the voice band, dB; number of pops)."""
    low = (lt.freqs > 0.0) & (lt.freqs < 60.0)
    voice = (lt.freqs >= 100.0) & (lt.freqs < 4000.0)
    rumble = _db(float(np.sum(lt.mean[low]))) - _db(float(np.sum(lt.mean[voice])))
    total = np.sum(spec.power, axis=1) + 1e-30
    sub = np.sum(spec.power[:, spec.band(0.0, 60.0)], axis=1)
    popping = (sub / total > 0.5) & (spec.level_db > spec.loud_db - 10.0)
    pops = int(np.sum(popping[1:] & ~popping[:-1]) + (popping[0] if popping.size else 0))
    return rumble, pops


def check_rumble(spec: Spectra, lt: LongTerm) -> dict:
    if spec.power.shape[0] == 0 or lt.power.shape[0] == 0:
        return _finding("rumble", "ok", "No rumble", "")
    rumble, pops = measure_rumble(spec, lt)
    value = f"{rumble:+.0f} dB under 60 Hz · {pops} pop{'s' if pops != 1 else ''}"
    bad = rumble > -12.0 or pops >= 3
    warn = rumble > -20.0 or pops >= 1
    if not (bad or warn):
        return _finding("rumble", "ok", "No low-frequency rumble", value)
    title = "Mic pops" if pops >= 3 and rumble <= -12.0 else "Low-frequency rumble"
    return _finding(
        "rumble", "bad" if bad else "warn", title, value,
        "Thumps below the voice (handling, traffic, wind, 'p' bursts). A high-pass at "
        "80 Hz removes them without touching the voice.",
        [_step("filter", mode="highpass", cutoff=80.0, order="4")], "highpass",
    )


# ---- uneven volume -------------------------------------------------------------------
def measure_unevenness(frames: SpeechFrames) -> float | None:
    """P90 - P10 of the speech level over the speech frames, dB (None: < 3 s of speech)."""
    level, speech = speech_levels(frames)
    on = speech & np.isfinite(level)
    if np.count_nonzero(on) * frames.hop / frames.fs < 3.0:
        return None
    return float(np.percentile(level[on], 90) - np.percentile(level[on], 10))


def check_uneven(frames: SpeechFrames) -> dict:
    spread = measure_unevenness(frames)
    if spread is None:
        return _finding("uneven", "ok", "Too little speech to judge volume changes", "")
    value = f"{spread:.0f} dB between loud and quiet parts"
    if spread <= UNEVEN_WARN_DB:
        return _finding("uneven", "ok", "Steady volume", value)
    return _finding(
        "uneven", "bad" if spread > UNEVEN_BAD_DB else "warn", "Uneven volume", value,
        "Some stretches of speech are much louder than others. The Voice Leveler turns "
        "the quiet ones up and the loud ones down, phrase by phrase.",
        [_step("leveler", amount=100, max_gain=18.0)], "leveler",
    )


# ---- long silences -------------------------------------------------------------------
def silent_stretches(frames: SpeechFrames) -> list[tuple[int, int, int, int]]:
    """Every pause of >= 3 s (docstring, check 6), as (a, b, ia, ib) frame indices.

    [a, b) runs from one syllable to the next -- that is the pause's length.  [ia, ib)
    is its inside, SPEECH_PAD_S clear of the syllables on either side (not at the file's
    ends), where no trailing "s" or breath can be: the level is judged there.
    """
    syllables = speech_mask(frames, 0.0)
    if not np.any(syllables):
        return []
    voice_db = float(np.median(frames.energy_db[syllables]))
    pad = int(round(SPEECH_PAD_S * frames.fs / frames.hop))
    edges = np.flatnonzero(np.diff(np.concatenate([[0], (~syllables).astype(np.int8), [0]])))
    min_frames = int(np.ceil((LONG_SILENCE_S - SILENCE_SLACK_S) * frames.fs / frames.hop))
    runs = []
    for a, b in zip(edges[0::2], edges[1::2]):
        if b - a < min_frames:
            continue
        ia = a if a == 0 else a + pad
        ib = b if b == syllables.size else b - pad
        run_db = _db(float(np.mean(10.0 ** (frames.energy_db[ia:ib] / 10.0))))
        if run_db <= voice_db - 10.0:           # a music bed is not a silence
            runs.append((int(a), int(b), int(ia), int(ib)))
    return runs


def check_silence(x: np.ndarray, fs: int, frames: SpeechFrames) -> dict:
    syllables = speech_mask(frames, 0.0)
    if not np.any(syllables):
        return _finding("silence", "ok", "No speech to judge silences by", "")
    runs = silent_stretches(frames)
    if not runs:
        return _finding("silence", "ok", "No long silences", "none of 3 s or more")
    step_s = frames.hop / frames.fs
    lengths = [(b - a) * step_s for a, b, _, _ in runs]
    value = (f"{len(runs)} pause{'s' if len(runs) != 1 else ''} · longest {max(lengths):.1f} s"
             + (f" · {sum(lengths):.1f} s in all" if len(runs) > 1 else ""))
    # The Silence Remover judges every hop on its level alone: its threshold goes just
    # over the loudest hop inside the pauses -- the room, never a quiet stretch of voice.
    levels, _ = hop_levels_db(x, fs)
    # Both step 10 ms; a 50 ms frame's centre is 25 ms in, i.e. on hop a + 2.
    inside = np.concatenate([levels[min(ia + 2, levels.size - 1):max(ia + 3, ib + 2)]
                             for _, _, ia, ib in runs])
    voice_db = float(np.median(frames.energy_db[syllables]))
    threshold = min(float(np.max(inside)) + 1.0, voice_db - 6.0)
    # A pause measures from syllable to syllable, but the Remover sees it only where the
    # level is under the room's: a trailing "s" or breath can take SPEECH_PAD_S off each
    # side, so it is told to look for pauses that much shorter.
    min_silence = round(LONG_SILENCE_S - 2.0 * SPEECH_PAD_S, 2)
    return _finding(
        "silence", "bad", "Long silences", value,
        "Stretches of 3 seconds or more with nobody speaking. Silence Remover shortens "
        "each to a short pause, and cuts the ones at the start and end.",
        [_step("silence_remover", threshold_db=float(np.clip(np.ceil(threshold), -80, -10)),
               min_silence=min_silence, keep=0.8)], "silence",
    )


def diagnose(x: np.ndarray, fs: int, channels: np.ndarray | None = None) -> dict[str, Any]:
    """Every check, plus the combined prescription in the fixed order (docstring)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"findings": [], "fix": []}
    centred = x - np.mean(x)                   # a DC offset is not rumble
    spec = Spectra(centred, fs)
    lt = LongTerm(centred, fs)
    frames = analyse(centred, fs)
    findings = [
        check_clipping(x, fs, channels),
        *check_noise(spec, fs),
        check_rumble(spec, lt),
        check_uneven(frames),
        check_silence(x, fs, frames),
    ]
    by_stage: dict[str, list[dict]] = {stage: [] for stage in _STAGES}
    for finding in findings:
        if finding["status"] != "bad":            # only what you can hear (docstring)
            continue
        if finding["stage"] == "noise":           # the low-pass has its own later slot
            for step in finding["fix"]:
                slot = "lowpass" if step["params"].get("mode") == "lowpass" else "noise"
                by_stage[slot].append(step)
        elif finding["stage"]:
            by_stage[finding["stage"]].extend(finding["fix"])
    # Noise removal costs the quietest words a few dB when they sit near the noise, so a
    # take that is already borderline uneven -- or whose speech is < 20 dB over the room
    # -- gets the Voice Leveler with it, to put them back.
    if by_stage["noise"] and not by_stage["leveler"]:
        uneven = next(f for f in findings if f["id"] == "uneven")
        if uneven["status"] == "warn" or measure_noise(spec)[1] < LEVEL_AFTER_DENOISE_SNR:
            by_stage["leveler"].append(_step("leveler", amount=100, max_gain=18.0))
    fix = [step for stage in _STAGES for step in by_stage[stage]]
    for finding in findings:
        finding.pop("stage")
    return {"findings": findings, "fix": fix}
