"""
The recipe router
=================
This is the single entry point the API calls.  It holds

  * OPERATIONS -- the catalogue.  Every tool the UI offers is one dict here: its id, label,
    icon, plain-language texts, the Python handler, and the parameter schema.  GET /operations
    serves this list (minus the handlers) and the frontend renders every palette entry and
    every slider from it, so a parameter added here appears in the UI with no React change.

  * run_recipe_measured() -- the fold.  A recipe is an ordered list of steps; the output is

        y = f_n( ... f_2( f_1( x ) ) ... )          (bypassed f_i are the identity)

    Order matters because the f_i do not commute: EQ-then-compressor squashes the boosted
    band, compressor-then-EQ boosts a band that has already been levelled; reverb-then-trim
    keeps the tail inside the cut, trim-then-reverb lets it ring over the new edge.

Parameter contract -- clamp, never reject
-----------------------------------------
Every incoming value goes through _coerce():

        missing / None / NaN / wrong type   ->  the schema default
        number outside [min, max]           ->  clip(value, min, max)
        enum not in options                 ->  the schema default
        param with a `scale`                ->  the clamped value * scale

(min / max / default are in the units the user SEES -- e.g. 0..100 % -- and the scale turns
them into the units the DSP function wants -- 0..1.)  So a half-typed number in a text
box, or a stale recipe from an older schema, still bakes.
The only hard failure is an unknown op id (ValueError -> HTTP 400): there is no sensible
default for "a tool that does not exist".

Fit to full scale -- turn down, never clip
------------------------------------------
The WAV encoder needs |y| <= 1.  Hard-clamping to +-1 flattens every peak that went over,
which is a nonlinearity: it smears energy across the whole spectrum and sounds like
crackle -- the opposite of what an EQ boost or an echo was meant to show.  Instead, when
the peak is over full scale the WHOLE buffer is scaled down by one constant:

        peak = max |y[n]|
        g    = HEADROOM / peak    if peak > 1,   else 1         (HEADROOM = 0.98)
        y   <- g * y                      normalised_db = 20 log10(g)   (<= 0)

A constant gain is linear, so the sound is unchanged apart from being quieter.  The fold
records what it saw first

        pre_clip_peak = max |y[n]|            clipped = #{ n : |y[n]| > 1 }

(`clipped` = how many samples WOULD have clipped) -- that is how ListenStats can say "the
recipe peaked at +4.2 dB and was turned down 4.4 dB" instead of silently getting quieter.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import mixer
from .compressor import compressor
from .echo_reverb import echo, reverb
from .editor import splice, trim
from .eq import equalizer
from .filters import cutoff_filter
from .noise import noise_reduce
from .speed_pitch import speed_pitch
from .voice_changer import voice_changer
from .voice_match import voice_match


# --------------------------------------------------------------------------------------
# Parameter schema helpers.  A param is a plain dict so it can be sent to the browser
# as-is; these constructors just keep the definitions below readable.
#
# Besides the numbers the DSP needs (min / max / step / default), every param carries
# what a person needs to use it:
#
#   help       one plain sentence, shown as the control's tooltip
#   advanced   True -> tucked into the card's "Advanced" section, out of the way
#   show_when  {"mode": ["reverb"]} -> only shown while another param has one of those
#              values (a knob that does nothing in the current mode is hidden, not greyed)
#   scale      the DSP value is  shown_value * scale.  Lets a 0..1 mix be shown and typed
#              as 0..100 % while echo()/reverb() keep their natural units.  _coerce()
#              applies it, so no handler ever sees a percentage.
#   log        the slider moves in equal RATIOS, not equal steps (frequency: every octave
#              gets the same travel).  UI only -- the value sent is still plain Hz.
# --------------------------------------------------------------------------------------
def _num(name, label, default, lo, hi, step=0.1, unit="", control="slider", *,
         help="", advanced=False, show_when=None, scale=None, log=False):
    return {
        "name": name, "label": label, "type": "float", "default": default,
        "min": lo, "max": hi, "step": step, "unit": unit, "control": control,
        "help": help, "advanced": advanced, "show_when": show_when, "scale": scale,
        "log": log,
    }


def _pct(name, label, default, *, lo=0.0, hi=100.0, help="", advanced=False, show_when=None):
    """A 0..1 quantity shown as 0..100 %."""
    return _num(name, label, default, lo, hi, 1.0, "%", help=help, advanced=advanced,
                show_when=show_when, scale=0.01)


def _enum(name, label, default, options, *, option_labels=None, help="", advanced=False,
          show_when=None):
    return {"name": name, "label": label, "type": "enum", "default": default,
            "options": options, "option_labels": option_labels or {}, "unit": "",
            "control": "select", "help": help, "advanced": advanced,
            "show_when": show_when, "scale": None}


def _bool(name, label, default, *, help="", advanced=False, show_when=None):
    return {"name": name, "label": label, "type": "bool", "default": default,
            "unit": "", "control": "toggle", "help": help, "advanced": advanced,
            "show_when": show_when, "scale": None}


def _source(name, label, *, help=""):
    """A reference to another loaded source.

    The only parameter whose OPTIONS this catalogue cannot supply: which files are loaded
    is a fact about the browser session, not about the DSP.  So the schema declares the
    kind of control and the client fills the dropdown -- and the value that comes back is
    a source id, validated for existence by the graph walk rather than by _coerce().
    """
    return {"name": name, "label": label, "type": "source", "default": "",
            "options": None, "unit": "", "control": "source", "help": help,
            "advanced": False, "show_when": None, "scale": None}


# --------------------------------------------------------------------------------------
# Adapters: some ops need a bit of dispatch before reaching the DSP function.
# --------------------------------------------------------------------------------------
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


def _voice_match_op(x, fs, ctx, calibration_source="", reference_source="",
                    pitch_strength=1.0, timbre_strength=0.7):
    """Convert this source toward the reference speaker, calibrated on two other sources.

    Both strengths at 0 is an exact bypass -- the buffer comes back untouched, and the
    calibration takes are not even read.
    """
    if pitch_strength <= 0.0 and timbre_strength <= 0.0:
        return x
    profile = ctx.voice_profile(calibration_source, reference_source)
    y, diagnostics = voice_match(x, fs, profile, pitch_strength=pitch_strength,
                                 timbre_strength=timbre_strength)
    ctx.note({**profile.diagnostics, **diagnostics})
    return y


def _echo_reverb_legacy(x, fs, mode="echo", delay=0.3, feedback=0.4, room_size=0.5,
                        decay=2.0, mix=0.5):
    """The old combined card, before echo and reverb became two.  Hidden from the palette;
    kept so a recipe saved against the old catalogue still bakes."""
    if mode == "reverb":
        return reverb(x, fs, room_size=room_size, decay=decay, mix=mix)
    return echo(x, fs, delay=delay, feedback=feedback, mix=mix)


# --------------------------------------------------------------------------------------
# The catalogue.
#
# Op-level fields, besides id / label / handler / params:
#   icon        lucide-react icon name used by the palette and the card
#   category    palette group heading
#   summary     what it does, in one plain sentence -- the card's subtitle
#   how         the technique in one line -- shown next to the maths
#   listen_for  what a listener should hear and see change -- the card's footer
#   quick       one-click settings: {"name": {param: shown_value, ...}}.  Names only the
#               params it moves; everything else keeps its current value.
#   hidden      served (so old recipes still render and bake) but not offered in the palette
#
# The defaults are deliberately NOT neutral: dropping a card in should make an audible,
# visible difference straight away, and the quick settings go from there.
# --------------------------------------------------------------------------------------
OPERATIONS: list[dict[str, Any]] = [
    # ---- Clean up --------------------------------------------------------------------
    {
        "id": "noise_remover",
        "label": "Noise Remover",
        "icon": "Waves",
        "category": "Clean up",
        "summary": "Removes steady background hiss, hum and fan noise.",
        "how": "Spectral subtraction: a per-frequency noise level is subtracted from every "
               "STFT frame, with the gain smoothed over time.",
        "listen_for": "The hiss in the pauses drops away. On the spectrogram the grey haze "
                      "between words goes dark while the voice stripes stay.",
        "handler": noise_reduce,
        "quick": {
            "Subtle": {"amount": 1.2, "floor": 10},
            "Strong": {"amount": 2.0, "floor": 5},
            "Extreme": {"amount": 3.5, "floor": 2},
        },
        "params": [
            _num("amount", "Strength", 2.0, 0.0, 4.0, 0.1, "x",
                 help="How hard to push the noise down. 1x removes the average noise level; "
                      "more also catches its louder moments, but too much starts eating the voice."),
            _enum("profile", "Find the noise", "auto", ["auto", "region"],
                  option_labels={"auto": "Automatically", "region": "From a part I mark"},
                  help="Automatically: uses the quietest moments of every frequency band, so any "
                       "recording with pauses works. From a part I mark: drag the amber region on "
                       "the Input waveform over a stretch that is ONLY noise."),
            _num("noise_start", "Noise-only from", 0.0, 0.0, 600.0, 0.01, "s", "number",
                 help="Start of the noise-only stretch (drag the amber region instead).",
                 show_when={"profile": ["region"]}),
            _num("noise_end", "Noise-only to", 0.5, 0.05, 600.0, 0.01, "s", "number",
                 help="End of the noise-only stretch.",
                 show_when={"profile": ["region"]}),
            _pct("floor", "Leave a little behind", 5, hi=50,
                 help="Never cut any band below this much of its original level. A small "
                      "remainder masks the watery 'musical noise' that full removal leaves.",
                 advanced=True),
        ],
    },
    # ---- Tone & level ----------------------------------------------------------------
    {
        "id": "equalizer",
        "label": "Equalizer",
        "icon": "SlidersHorizontal",
        "category": "Tone & level",
        "summary": "Bass, mid and treble tone controls.",
        "how": "Low shelf, peaking bell and high shelf biquads in series (RBJ cookbook).",
        "listen_for": "Bass up: fuller and boomier; treble up: crisper and brighter. The "
                      "bottom or top band of the spectrogram lights up.",
        "handler": equalizer,
        "quick": {
            "Subtle": {"bass_gain": 3, "mid_gain": 0, "treble_gain": 3},
            "Strong": {"bass_gain": 6, "mid_gain": 0, "treble_gain": 6},
            "Extreme": {"bass_gain": 15, "mid_gain": -6, "treble_gain": 15},
            "Telephone": {"bass_gain": -24, "mid_gain": 8, "treble_gain": -24,
                          "bass_freq": 400, "treble_freq": 3000},
        },
        "params": [
            _num("bass_gain", "Bass", 6.0, -24.0, 24.0, 0.5, "dB",
                 help="Boost or cut everything below the bass corner (200 Hz by default)."),
            _num("mid_gain", "Mid", 0.0, -24.0, 24.0, 0.5, "dB",
                 help="Boost or cut a band around the mid frequency — where speech is clearest."),
            _num("treble_gain", "Treble", 6.0, -24.0, 24.0, 0.5, "dB",
                 help="Boost or cut everything above the treble corner (4 kHz by default)."),
            _num("bass_freq", "Bass corner", 200.0, 40.0, 1000.0, 5.0, "Hz",
                 help="Below this the bass control acts fully.", advanced=True),
            _num("mid_freq", "Mid frequency", 1000.0, 200.0, 6000.0, 10.0, "Hz",
                 help="Centre of the mid band.", advanced=True),
            _num("q", "Mid width (Q)", 1.0, 0.2, 10.0, 0.1, "",
                 help="Higher = a narrower mid band.", advanced=True),
            _num("treble_freq", "Treble corner", 4000.0, 1000.0, 16000.0, 50.0, "Hz",
                 help="Above this the treble control acts fully.", advanced=True),
        ],
    },
    {
        "id": "filter",
        "label": "Filter",
        "icon": "Filter",
        "category": "Tone & level",
        "summary": "Removes everything above (low-pass) or below (high-pass) a frequency, keeps "
                   "one band, or notches one out.",
        "how": "RBJ biquads via the bilinear transform; low/high-pass are Butterworth "
               "cascades of N/2 sections.",
        "listen_for": "Low-pass sounds muffled, high-pass thin, band-pass like a telephone. "
                      "Dashed lines mark the cutoff on both spectrograms — on the Output "
                      "everything past them goes dark.",
        "handler": cutoff_filter,
        "quick": {
            "Muffled": {"mode": "lowpass", "cutoff": 500, "order": "8"},
            "Telephone": {"mode": "bandpass", "cutoff": 1500, "q": 1.0},
            "Thin": {"mode": "highpass", "cutoff": 1000, "order": "4"},
            "Rumble cut": {"mode": "highpass", "cutoff": 80, "order": "4"},
            "Hum notch": {"mode": "notch", "cutoff": 50, "q": 10.0},
        },
        "params": [
            _enum("mode", "Type", "lowpass", ["lowpass", "highpass", "bandpass", "notch"],
                  option_labels={"lowpass": "Low-pass", "highpass": "High-pass",
                                 "bandpass": "Band-pass", "notch": "Notch"},
                  help="Low-pass keeps the lows, high-pass the highs, band-pass one band, "
                       "notch removes one band."),
            _num("cutoff", "Frequency", 500.0, 20.0, 20000.0, 1.0, "Hz", log=True,
                 help="The cutoff (low/high-pass, where it is 3 dB down) or the centre "
                      "(band-pass, notch)."),
            _enum("order", "Steepness", "4", ["2", "4", "6", "8"],
                  option_labels={"2": "12 dB/oct", "4": "24 dB/oct", "6": "36 dB/oct",
                                 "8": "48 dB/oct"},
                  help="How fast it falls past the cutoff. Each step adds one more biquad.",
                  show_when={"mode": ["lowpass", "highpass"]}),
            _num("q", "Width (Q)", 2.0, 0.3, 30.0, 0.1, "",
                 help="Higher = a narrower band. The band is about frequency / Q wide.",
                 show_when={"mode": ["bandpass", "notch"]}),
        ],
    },
    {
        "id": "compressor",
        "label": "Compressor",
        "icon": "Minimize2",
        "category": "Tone & level",
        "summary": "Evens out the volume: loud parts down, quiet parts up.",
        "how": "Peak envelope follower feeding a soft-knee dB gain computer, with auto-makeup.",
        "listen_for": "Quiet words become as loud as the loud ones. The waveform turns from "
                      "spiky to a solid block at the same height.",
        "handler": compressor,
        "quick": {
            "Subtle": {"threshold": -18, "ratio": 2.5},
            "Strong": {"threshold": -30, "ratio": 6},
            "Extreme": {"threshold": -45, "ratio": 20},
        },
        "params": [
            _num("threshold", "Start squashing at", -30.0, -60.0, 0.0, 0.5, "dB",
                 help="Anything louder than this gets turned down. Lower = more of the audio "
                      "is affected."),
            _num("ratio", "Squash ratio", 6.0, 1.0, 20.0, 0.1, ":1",
                 help="How hard. At 6:1, sound that is 6 dB over the threshold comes out only "
                      "1 dB over."),
            _bool("auto_makeup", "Auto-makeup", True,
                  help="Lift everything back up so the loudest peak is where it started. This "
                       "is what makes the quiet parts come UP."),
            _num("attack", "Attack", 5.0, 0.1, 200.0, 0.1, "ms",
                 help="How quickly it reacts to a sudden loud sound.", advanced=True),
            _num("release", "Release", 150.0, 5.0, 1000.0, 1.0, "ms",
                 help="How quickly it lets go once things get quiet again.", advanced=True),
            _num("knee", "Knee", 6.0, 0.0, 24.0, 0.5, "dB",
                 help="How gently it eases in around the threshold. 0 = a hard corner.",
                 advanced=True),
            _num("makeup", "Extra gain", 0.0, 0.0, 24.0, 0.5, "dB",
                 help="A fixed boost added on top, after auto-makeup.", advanced=True),
        ],
    },
    # ---- Space -----------------------------------------------------------------------
    {
        "id": "echo",
        "label": "Echo",
        "icon": "Repeat",
        "category": "Space",
        "summary": "Distinct repeats, like shouting across a canyon.",
        "how": "Convolution with a tapped delay line: h[kD] = g^k (the feedback comb, unrolled).",
        "listen_for": "Each word comes back again and again, quieter each time. The "
                      "waveform shows smaller copies trailing every burst.",
        "handler": echo,
        "quick": {
            "Slapback": {"delay": 0.12, "feedback": 15, "mix": 40},
            "Canyon": {"delay": 0.35, "feedback": 50, "mix": 60},
            "Endless": {"delay": 0.5, "feedback": 80, "mix": 75},
        },
        "params": [
            _num("delay", "Delay", 0.35, 0.02, 2.0, 0.01, "s",
                 help="The gap between one repeat and the next."),
            _pct("feedback", "Repeats", 50, hi=95,
                 help="How much of each repeat survives into the next. Higher = more, "
                      "longer-lasting repeats."),
            _pct("mix", "Wet", 60,
                 help="How loud the repeats are against the original."),
        ],
    },
    {
        "id": "reverb",
        "label": "Reverb",
        "icon": "AudioLines",
        "category": "Space",
        "summary": "Puts the sound in a room, from a small booth to a cathedral.",
        "how": "Convolution with a synthesised room: early reflections over exponentially "
               "decaying noise (RT60).",
        "listen_for": "A smooth tail rings on after each word. On the spectrogram the "
                      "stripes smear to the right instead of stopping sharply.",
        "handler": reverb,
        "quick": {
            "Small room": {"room_size": 30, "decay": 0.6, "mix": 25},
            "Hall": {"room_size": 70, "decay": 2.5, "mix": 45},
            "Cathedral": {"room_size": 100, "decay": 6.0, "mix": 65},
        },
        "params": [
            _pct("room_size", "Room size", 70,
                 help="Spreads the first reflections further apart — the sense of how big the "
                      "space is."),
            _num("decay", "Tail length", 2.5, 0.1, 10.0, 0.1, "s",
                 help="How long the room keeps ringing (RT60: the time to fall by 60 dB)."),
            _pct("mix", "Wet", 45,
                 help="How much room you hear against the original."),
        ],
    },
    # ---- Time & pitch ----------------------------------------------------------------
    {
        "id": "speed_pitch",
        "label": "Speed & Pitch",
        "icon": "Gauge",
        "category": "Time & pitch",
        "summary": "Change how fast it plays and how high it sounds — separately.",
        "how": "Phase-vocoder time stretch plus linear-interpolation resampling.",
        "listen_for": "Faster speech at the same voice (Keep pitch on), or a sped-up tape "
                      "(off). The output waveform is shorter or longer than the input.",
        "handler": speed_pitch,
        "quick": {
            "Slow-mo": {"speed": 0.6, "semitones": 0, "preserve_pitch": True},
            "Fast talk": {"speed": 1.5, "semitones": 0, "preserve_pitch": True},
            "Tape speed-up": {"speed": 1.3, "semitones": 0, "preserve_pitch": False},
            "Octave down": {"speed": 1.0, "semitones": -12},
        },
        "params": [
            _num("speed", "Speed", 1.5, 0.25, 4.0, 0.01, "x",
                 help="Playback rate. 2x = half the length."),
            _bool("preserve_pitch", "Keep pitch", True,
                  help="On: only the length changes. Off: like a tape played faster — the "
                       "pitch goes up with the speed."),
            _num("semitones", "Pitch", 0.0, -24.0, 24.0, 0.5, "st",
                 help="Move every note up or down without changing the length. 12 semitones "
                      "= one octave."),
        ],
    },
    # ---- Voice -----------------------------------------------------------------------
    {
        "id": "voice_changer",
        "label": "Voice Changer",
        "icon": "Mic",
        "category": "Voice",
        "summary": "Turn a voice into a chipmunk, a monster, a robot or a whisper.",
        "how": "The STFT rebuilt with a new phase: bins shifted in frequency, a fixed pulse "
               "per frame, or random phase.",
        "listen_for": "Chipmunk/monster: the stripes on the spectrogram move up/down. Robot: "
                      "they snap to evenly spaced lines. Whisper: they dissolve into haze.",
        "handler": voice_changer,
        "params": [
            _enum("mode", "Voice", "chipmunk", ["chipmunk", "monster", "robot", "whisper"],
                  option_labels={"chipmunk": "Chipmunk", "monster": "Monster",
                                 "robot": "Robot", "whisper": "Whisper"},
                  help="Which character to turn the voice into."),
            _num("semitones", "How far", 7.0, 1.0, 12.0, 0.5, "st",
                 help="How many semitones up (chipmunk) or down (monster). 12 = one octave.",
                 show_when={"mode": ["chipmunk", "monster"]}),
            _num("robot_freq", "Robot buzz", 100.0, 40.0, 300.0, 1.0, "Hz",
                 help="The one note the robot speaks on. Lower = deeper.",
                 show_when={"mode": ["robot"]}),
            _pct("mix", "Wet", 100,
                 help="Blend with the original voice.", advanced=True),
        ],
    },
    {
        "id": "voice_match",
        "label": "Voice Match",
        "icon": "UsersRound",
        "category": "Voice",
        "summary": "Move your voice toward another speaker's, learned from one shared "
                   "script you both read.",
        "how": "Autocorrelation pitch + cepstral envelopes, DTW-aligned into a paired "
               "codebook; a time-varying phase-locked pitch shift, then an envelope "
               "correction toward the looked-up target.",
        "listen_for": "The pitch lands in the other speaker's range without a chipmunk "
                      "sound, and the vowels take on their colour. Silences and 's' "
                      "sounds stay as they were.",
        "handler": _voice_match_op,
        "needs_ctx": True,
        "quick": {
            "Pitch only": {"pitch_strength": 100, "timbre_strength": 0},
            "Timbre only": {"pitch_strength": 0, "timbre_strength": 70},
            "Full": {"pitch_strength": 100, "timbre_strength": 70},
        },
        "params": [
            _source("calibration_source", "Your calibration",
                    help="YOUR reading of the shared script (30-60 s, at most 2 min). Record "
                         "it cleanly, with the same mic placement as the other speaker."),
            _source("reference_source", "Their calibration",
                    help="The OTHER speaker's reading of the same script."),
            _pct("pitch_strength", "Pitch", 100,
                 help="How far to move your pitch into their range. 0 keeps your own pitch."),
            _pct("timbre_strength", "Timbre", 70,
                 help="How far to move your vowel colour (formants) toward theirs. Frames "
                      "unlike anything in the calibration get less."),
        ],
    },
    # ---- Edit ------------------------------------------------------------------------
    {
        "id": "editor",
        "label": "Cut & Trim",
        "icon": "Scissors",
        "category": "Edit",
        "summary": "Keep only a part of the clip, or cut a part out.",
        "how": "numpy slicing and concatenation, with 5 ms fades at every cut.",
        "listen_for": "The output is shorter. Drag the green region on the Input waveform "
                      "to choose the part.",
        "handler": _editor_op,
        "params": [
            _enum("mode", "Action", "trim", ["trim", "splice"],
                  option_labels={"trim": "Keep the selection", "splice": "Cut it out"},
                  help="Keep only the selected part, or remove it and join the two sides."),
            _num("start", "From", 0.0, 0.0, 600.0, 0.01, "s", "number",
                 help="Start of the selection (or drag the green region)."),
            _num("end", "To", 0.0, 0.0, 600.0, 0.01, "s", "number",
                 help="End of the selection. 0 = the end of the file."),
        ],
    },
    {
        "id": "assemble",
        "label": "Assemble Clip",
        "icon": "Layers",
        "category": "Edit",
        "summary": "Bring in a piece of another loaded file.",
        "how": "Take a span of another source — after ITS recipe has run — and overlay, "
               "insert or append it here.",
        "listen_for": "The other clip plays at the chosen point.",
        "handler": _assemble_op,
        # The flag run_recipe_measured looks for before handing a handler the graph.
        "needs_ctx": True,
        "params": [
            _source("source", "Clip from", help="Which loaded file to take the clip from."),
            _enum("mode", "Placement", "mix", ["mix", "insert", "append"],
                  option_labels={"mix": "Layer on top", "insert": "Insert",
                                 "append": "Add at the end"},
                  help="Layer: both play at once. Insert: split this clip open and push the rest "
                       "later. Add at the end: join it on after."),
            _num("position", "Place at", 0.0, 0.0, 600.0, 0.01, "s", "number",
                 help="Where in THIS clip it goes.",
                 show_when={"mode": ["mix", "insert"]}),
            _num("clip_start", "Clip from", 0.0, 0.0, 600.0, 0.01, "s", "number",
                 help="Start of the piece to take from the other file."),
            _num("clip_end", "Clip to", 0.0, 0.0, 600.0, 0.01, "s", "number",
                 help="End of the piece. 0 = to its end."),
            _num("gain", "Clip volume", 0.0, -60.0, 12.0, 0.5, "dB",
                 help="Turn the clip up or down before placing it.", advanced=True),
        ],
    },
    # ---- Hidden ----------------------------------------------------------------------
    {
        "id": "echo_reverb",
        "label": "Echo & Reverb (old)",
        "icon": "AudioLines",
        "category": "Space",
        "summary": "The old combined card — use Echo or Reverb instead.",
        "how": "Either of the two convolutions, chosen by mode.",
        "listen_for": "",
        "hidden": True,
        "handler": _echo_reverb_legacy,
        "params": [
            _enum("mode", "Mode", "echo", ["echo", "reverb"]),
            _num("delay", "Delay", 0.3, 0.01, 2.0, 0.01, "s", show_when={"mode": ["echo"]}),
            _num("feedback", "Feedback", 0.4, 0.0, 0.95, 0.01, "", show_when={"mode": ["echo"]}),
            _num("room_size", "Room size", 0.5, 0.0, 1.0, 0.01, "", show_when={"mode": ["reverb"]}),
            _num("decay", "Decay (RT60)", 2.0, 0.1, 10.0, 0.1, "s", show_when={"mode": ["reverb"]}),
            _num("mix", "Dry / wet", 0.5, 0.0, 1.0, 0.01, ""),
        ],
    },
]

# id -> definition, for O(1) lookup while baking.
_BY_ID = {op["id"]: op for op in OPERATIONS}

# Target peak after fit_to_full_scale(): a hair under 1 so the 16-bit encoder's rounding
# can never land on the wrapping value.
HEADROOM = 0.98


def fit_to_full_scale(y: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale y down by one constant if it peaks over 1 (module docstring).

    Returns (buffer, gain applied in dB) -- 0.0 when nothing had to change.
    """
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 1.0:
        return y, 0.0
    g = HEADROOM / peak
    return y * g, round(20.0 * np.log10(g), 2)


def operations_schema() -> list[dict[str, Any]]:
    """The catalogue without the python callables, ready to be sent as JSON."""
    return [{k: v for k, v in op.items() if k != "handler"} for op in OPERATIONS]


def _coerce(param: dict[str, Any], value: Any) -> Any:
    """Validate one incoming parameter against its schema entry (clamp, never reject),
    then convert it from the units shown in the UI to the units the handler takes."""
    scale = param.get("scale") or 1.0
    if value is None:
        value = param["default"]
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
        number = float(param["default"])
    if not np.isfinite(number):
        number = float(param["default"])
    return float(np.clip(number, param["min"], param["max"])) * scale


def run_recipe_measured(
    x: np.ndarray, fs: int, recipe: list[dict[str, Any]], ctx: Any = None,
    clip: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply every non-bypassed operation in order, and report what the fold did.

    Same fold as run_recipe(), but it also returns the handful of facts only this loop
    can see -- how much headroom was exceeded before the final fit to full scale, and whether the
    recipe ran to the end -- so the UI can explain the output instead of just drawing it.

    `ctx` is the graph handle for operations that reach outside their own buffer (see
    _assemble_op and _voice_match_op).  It is None for the master chain, which has no source identity for a
    clip position to be relative to.

    `clip` is False for an INTERMEDIATE chain -- one whose output feeds another chain
    rather than the WAV encoder.  The fit to full scale below exists because a 16-bit file
    would wrap and click, which is a fact about the output format, not about the audio:
    turning down a buffer that a later step is about to mix with something else would
    change the balance of the mix.  The measurement is taken either way.

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
                    f"{definition['label']!r} only works inside a source's recipe -- it "
                    "reads other loaded sources, and the master chain has no source of "
                    "its own."
                )
            y = definition["handler"](y, fs, ctx=ctx, **kwargs)
        else:
            y = definition["handler"](y, fs, **kwargs)
        applied += 1
        if y.size == 0:     # e.g. a trim that selected nothing -- stop rather than crash
            truncated = True
            break

    # Measure the overshoot BEFORE fitting, so the report can say how far over it went.
    pre_clip_peak = float(np.max(np.abs(y))) if y.size else 0.0
    clipped = int(np.count_nonzero(np.abs(y) > 1.0)) if y.size else 0

    # Effects (especially EQ boosts, echo and makeup gain) can push samples past full
    # scale.  Turn the whole buffer down to fit -- unless it is on its way into another
    # chain rather than into a file.
    normalised_db = 0.0
    if clip:
        y, normalised_db = fit_to_full_scale(y)

    report = {
        "pre_clip_peak": round(pre_clip_peak, 6),
        "clipped": clipped,
        "normalised_db": normalised_db,
        "steps_applied": applied,
        "steps_bypassed": bypassed,
        "truncated": truncated,
    }
    return y, report


def run_recipe(x: np.ndarray, fs: int, recipe: list[dict[str, Any]]) -> np.ndarray:
    """Apply every non-bypassed operation in order and return the processed buffer.

    Raises ValueError on an unknown op id, which the router turns into an HTTP 400.
    """
    return run_recipe_measured(x, fs, recipe)[0]