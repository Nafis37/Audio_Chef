"""
The recipe router
This is the single entry point the API calls.  It holds
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import mixer
from .compressor import compressor
from .echo_reverb import echo, reverb
from .editor import splice, trim
from .eq import equalizer
from .noise import noise_reduce
from .speed_pitch import speed_pitch


# --------------------------------------------------------------------------------------
# Parameter schema helpers.  A param is a plain dict so it can be sent to the browser
# as-is; these constructors just keep the definitions below readable.
# --------------------------------------------------------------------------------------
def _num(name, label, default, lo, hi, step=0.1, unit="", control="slider"):
    return {
        "name": name, "label": label, "type": "float", "default": default,
        "min": lo, "max": hi, "step": step, "unit": unit, "control": control,
    }


def _enum(name, label, default, options):
    return {"name": name, "label": label, "type": "enum", "default": default,
            "options": options, "unit": "", "control": "select"}


def _bool(name, label, default):
    return {"name": name, "label": label, "type": "bool", "default": default,
            "unit": "", "control": "toggle"}


def _source(name, label):
    """A reference to another loaded source.

    The only parameter whose OPTIONS this catalogue cannot supply: which files are loaded
    is a fact about the browser session, not about the DSP.  So the schema declares the
    kind of control and the client fills the dropdown -- and the value that comes back is
    a source id, validated for existence by the graph walk rather than by _coerce().
    """
    return {"name": name, "label": label, "type": "source", "default": "",
            "options": None, "unit": "", "control": "source"}


# --------------------------------------------------------------------------------------
# Adapters: some ops need a bit of dispatch before reaching the DSP function.
# --------------------------------------------------------------------------------------
def _echo_reverb_op(x, fs, mode="echo", delay=0.3, feedback=0.4, room_size=0.5,
                    decay=2.0, mix=0.5):
    """One card covering both room effects; add it twice to chain echo INTO reverb."""
    if mode == "reverb":
        return reverb(x, fs, room_size=room_size, decay=decay, mix=mix)
    return echo(x, fs, delay=delay, feedback=feedback, mix=mix)


def _editor_op(x, fs, mode="trim", start=0.0, end=0.0):
    """Trim keeps the selected region; splice removes it."""
    if mode == "splice":
        return splice(x, fs, start=start, end=end)
    return trim(x, fs, start=start, end=end)


def _assemble_op(x, fs, ctx, source="", mode="mix", position=0.0,
                 clip_start=0.0, clip_end=0.0, gain=0.0):
    """Pull a clip out of another source and place it into this one.

    The only operation that needs `ctx` -- everything else in the catalogue is a pure
    function of the buffer in front of it.  ctx.clip() resolves the other source's whole
    chain (memoised) and hands back an independent copy of the requested span.
    """
    clip = ctx.clip(source, start=clip_start, end=clip_end)
    if mode == "insert":
        return mixer.insert_at(x, clip, fs, position=position, gain_db=gain)
    if mode == "append":
        return mixer.append_to(x, clip, fs, gain_db=gain)
    return mixer.mix_at(x, clip, fs, position=position, gain_db=gain)


# --------------------------------------------------------------------------------------
# The catalogue.  `icon` values are lucide-react icon names used by the palette.
# --------------------------------------------------------------------------------------
OPERATIONS: list[dict[str, Any]] = [
    {
        "id": "noise_remover",
        "label": "Noise Remover",
        "icon": "Waves",
        "description": "Spectral subtraction: learns a noise profile from a quiet region "
                       "and subtracts it from every STFT frame.",
        "handler": noise_reduce,
        "params": [
            _num("amount", "Reduction", 1.5, 0.0, 4.0, 0.1, "x"),
            _num("floor", "Spectral floor", 0.05, 0.0, 0.5, 0.01, ""),
            _num("noise_start", "Profile start", 0.0, 0.0, 30.0, 0.1, "s"),
            _num("noise_end", "Profile end", 0.5, 0.05, 30.0, 0.05, "s"),
        ],
    },
    {
        "id": "equalizer",
        "label": "Audio Equalizer",
        "icon": "SlidersHorizontal",
        "description": "Three peaking biquad bands (RBJ cookbook coefficients) in series.",
        "handler": equalizer,
        "params": [
            _num("low_gain", "Low gain", 0.0, -24.0, 24.0, 0.5, "dB"),
            _num("low_freq", "Low freq", 120.0, 20.0, 500.0, 1.0, "Hz"),
            _num("mid_gain", "Mid gain", 0.0, -24.0, 24.0, 0.5, "dB"),
            _num("mid_freq", "Mid freq", 1000.0, 200.0, 5000.0, 10.0, "Hz"),
            _num("high_gain", "High gain", 0.0, -24.0, 24.0, 0.5, "dB"),
            _num("high_freq", "High freq", 6000.0, 2000.0, 16000.0, 50.0, "Hz"),
            _num("q", "Q", 1.0, 0.1, 10.0, 0.1, ""),
        ],
    },
    {
        "id": "echo_reverb",
        "label": "Echo & Reverb Studio",
        "icon": "AudioLines",
        "description": "Convolution with a synthesised impulse response: a tapped delay line "
                       "for echo, a decaying-noise room for reverb.",
        "handler": _echo_reverb_op,
        "params": [
            _enum("mode", "Mode", "echo", ["echo", "reverb"]),
            _num("delay", "Delay", 0.3, 0.01, 2.0, 0.01, "s"),
            _num("feedback", "Feedback", 0.4, 0.0, 0.95, 0.01, ""),
            _num("room_size", "Room size", 0.5, 0.0, 1.0, 0.01, ""),
            _num("decay", "Decay (RT60)", 2.0, 0.1, 10.0, 0.1, "s"),
            _num("mix", "Dry / wet", 0.5, 0.0, 1.0, 0.01, ""),
        ],
    },
    {
        "id": "editor",
        "label": "Mini Audio Editor",
        "icon": "Scissors",
        "description": "Trim to a region or splice one out, by numpy slicing.",
        "handler": _editor_op,
        "params": [
            _enum("mode", "Mode", "trim", ["trim", "splice"]),
            _num("start", "Start", 0.0, 0.0, 600.0, 0.01, "s", "number"),
            _num("end", "End", 0.0, 0.0, 600.0, 0.01, "s", "number"),
        ],
    },
    {
        "id": "assemble",
        "label": "Assemble Clip",
        "icon": "Layers",
        "description": "Take a span of another source -- after ITS recipe has run -- and "
                       "overlay, insert or append it here.",
        "handler": _assemble_op,
        # The flag run_recipe_measured looks for before handing a handler the graph.
        "needs_ctx": True,
        "params": [
            _source("source", "Clip from"),
            _enum("mode", "Placement", "mix", ["mix", "insert", "append"]),
            _num("position", "Place at", 0.0, 0.0, 600.0, 0.01, "s", "number"),
            _num("clip_start", "Clip start", 0.0, 0.0, 600.0, 0.01, "s", "number"),
            _num("clip_end", "Clip end", 0.0, 0.0, 600.0, 0.01, "s", "number"),
            _num("gain", "Clip gain", 0.0, -60.0, 12.0, 0.5, "dB"),
        ],
    },
    {
        "id": "speed_pitch",
        "label": "Speed & Pitch Lab",
        "icon": "Gauge",
        "description": "Phase-vocoder time stretch plus manual linear-interpolation resampling.",
        "handler": speed_pitch,
        "params": [
            _num("speed", "Speed", 1.0, 0.25, 4.0, 0.01, "x"),
            _num("semitones", "Pitch", 0.0, -24.0, 24.0, 0.5, "st"),
            _bool("preserve_pitch", "Keep pitch when changing speed", True),
        ],
    },
    {
        "id": "compressor",
        "label": "Simple Compressor",
        "icon": "Minimize2",
        "description": "One-pole envelope follower feeding a dB-domain gain computer.",
        "handler": compressor,
        "params": [
            _num("threshold", "Threshold", -20.0, -60.0, 0.0, 0.5, "dB"),
            _num("ratio", "Ratio", 4.0, 1.0, 20.0, 0.1, ":1"),
            _num("attack", "Attack", 10.0, 0.1, 200.0, 0.1, "ms"),
            _num("release", "Release", 100.0, 5.0, 1000.0, 1.0, "ms"),
            _num("knee", "Knee", 6.0, 0.0, 24.0, 0.5, "dB"),
            _num("makeup", "Makeup gain", 0.0, 0.0, 24.0, 0.5, "dB"),
        ],
    },
]

# id -> definition, for O(1) lookup while baking.
_BY_ID = {op["id"]: op for op in OPERATIONS}


def operations_schema() -> list[dict[str, Any]]:
    """The catalogue without the python callables, ready to be sent as JSON."""
    return [{k: v for k, v in op.items() if k != "handler"} for op in OPERATIONS]


def _coerce(param: dict[str, Any], value: Any) -> Any:
    """Validate one incoming parameter against its schema entry (clamp, never reject)."""
    if value is None:
        return param["default"]
    if param["type"] == "bool":
        return bool(value)
    if param["type"] == "enum":
        return value if value in param["options"] else param["default"]
    if param["type"] == "source":
        # Shape only.  Whether that id names a source that exists is a question about the
        # graph, which this function knows nothing about -- and the contract here is
        # "clamp, never reject".  graph.Evaluator does the existence check, once, with a
        # message good enough to show the user.
        return value if isinstance(value, str) else param["default"]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return param["default"]
    if not np.isfinite(number):
        return param["default"]
    return float(np.clip(number, param["min"], param["max"]))


def run_recipe_measured(
    x: np.ndarray, fs: int, recipe: list[dict[str, Any]], ctx: Any = None,
    clip: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply every non-bypassed operation in order, and report what the fold did.

    Same fold as run_recipe(), but it also returns the handful of facts only this loop
    can see -- how much headroom was exceeded before the final clamp, and whether the
    recipe ran to the end -- so the UI can explain the output instead of just drawing it.

    `ctx` is the graph handle for operations that reach outside their own buffer (see
    _assemble_op).  It is None for the master chain, which has no source identity for a
    clip position to be relative to.

    `clip` is False for an INTERMEDIATE chain -- one whose output feeds another chain
    rather than the WAV encoder.  The clamp below exists because a 16-bit file would wrap
    and click, which is a fact about the output format, not about the audio: clamping a
    buffer that a later step is about to turn down by 12 dB would throw away headroom
    that was still recoverable.  The measurement above it is taken either way, so nothing
    is lost from the report.

    Raises ValueError on an unknown op id, which the router turns into an HTTP 400.
    """
    y = np.asarray(x, dtype=np.float64)
    applied = 0
    bypassed = 0
    truncated = False

    for step in recipe or []:
        op_id = step.get("op")
        if op_id not in _BY_ID:
            raise ValueError(f"Unknown operation: {op_id!r}")
        if step.get("bypass"):
            bypassed += 1
            continue        # the card is switched off -- leave the buffer untouched

        definition = _BY_ID[op_id]
        supplied = step.get("params") or {}
        kwargs = {p["name"]: _coerce(p, supplied.get(p["name"])) for p in definition["params"]}

        if definition.get("needs_ctx"):
            if ctx is None:
                raise ValueError(
                    f"{op_id!r} only works inside a source's recipe -- the master chain "
                    "has no position for a clip to be placed relative to."
                )
            y = definition["handler"](y, fs, ctx=ctx, **kwargs)
        else:
            y = definition["handler"](y, fs, **kwargs)
        applied += 1
        if y.size == 0:     # e.g. a trim that selected nothing -- stop rather than crash
            truncated = True
            break

    # Measure the overshoot BEFORE clamping: once the samples are clipped the evidence
    # that they ever went past full scale is gone, and the flat tops in the output
    # waveform would have no explanation.
    pre_clip_peak = float(np.max(np.abs(y))) if y.size else 0.0
    clipped = int(np.count_nonzero(np.abs(y) > 1.0)) if y.size else 0

    report = {
        "pre_clip_peak": round(pre_clip_peak, 6),
        "clipped": clipped,
        "steps_applied": applied,
        "steps_bypassed": bypassed,
        "truncated": truncated,
    }

    # Effects (especially echo, reverb and makeup gain) can push samples past full scale,
    # which would wrap around and click when written to a 16-bit WAV.  Clamp to +-1 --
    # unless this buffer is on its way into another chain rather than into a file.
    return (np.clip(y, -1.0, 1.0) if clip else y), report


def run_recipe(x: np.ndarray, fs: int, recipe: list[dict[str, Any]]) -> np.ndarray:
    """Apply every non-bypassed operation in order and return the processed buffer.

    Raises ValueError on an unknown op id, which the router turns into an HTTP 400.
    """
    return run_recipe_measured(x, fs, recipe)[0]