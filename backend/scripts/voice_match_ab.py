"""
Render Voice Match listening-test material: original / pitch-only / full, level-matched.

    cd backend
    python -m scripts.voice_match_ab --mine my_script.wav --theirs friend_script.wav \
        --out ab_test held_out/*.wav

For every held-out sentence (one NOT in the calibration script) this writes

    <name>.original.wav     your recording, untouched
    <name>.pitch_only.wav   pitch 100 %, timbre 0 %
    <name>.full.wav         pitch 100 %, timbre 70 %   (the card's defaults)

plus reference.wav (a slice of the friend's calibration take, for listeners to anchor
on), all scaled to the same RMS (TARGET_RMS_DB, backed off if a peak would pass
-1 dBFS), because a louder version reliably wins a "which is closer" vote.  Play them in
shuffled order and ask similarity and intelligibility as two SEPARATE questions.
Acceptance: full beats pitch-only on similarity without losing intelligibility.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from app import config
from app.audio_io import load_audio_at, probe, write_wav_bytes
from app.dsp.graph import project_sample_rate
from app.dsp.voice_calibration import build_profile
from app.dsp.voice_match import voice_match

TARGET_RMS_DB = -23.0
PEAK_LIMIT = 10 ** (-1.0 / 20.0)
REFERENCE_S = 8.0


def level_match(x: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    if rms <= 0.0:
        return x
    gain = 10 ** (TARGET_RMS_DB / 20.0) / rms
    peak = float(np.max(np.abs(x))) * gain
    if peak > PEAK_LIMIT:
        gain *= PEAK_LIMIT / peak
    return x * gain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mine", required=True, type=pathlib.Path,
                        help="your reading of the calibration script")
    parser.add_argument("--theirs", required=True, type=pathlib.Path,
                        help="the other speaker's reading of the same script")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("sentences", nargs="+", type=pathlib.Path,
                        help="held-out recordings of YOU (not from the script)")
    args = parser.parse_args()

    paths = [args.mine, args.theirs, *args.sentences]
    fs = project_sample_rate([probe(p)["sample_rate"] for p in paths], None,
                             config.MAX_PROJECT_SAMPLE_RATE)
    mine, _ = load_audio_at(args.mine, fs)
    theirs, _ = load_audio_at(args.theirs, fs)

    print(f"Calibrating at {fs} Hz ...")
    profile = build_profile(mine, theirs, fs)
    for key, value in profile.diagnostics.items():
        print(f"  {key:>18}: {value}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "reference.wav").write_bytes(
        write_wav_bytes(level_match(theirs[: int(REFERENCE_S * fs)]), fs))

    for path in args.sentences:
        x, _ = load_audio_at(path, fs)
        versions = {
            "original": x,
            "pitch_only": voice_match(x, fs, profile, 1.0, 0.0)[0],
            "full": voice_match(x, fs, profile, 1.0, 0.7)[0],
        }
        for label, audio in versions.items():
            (args.out / f"{path.stem}.{label}.wav").write_bytes(
                write_wav_bytes(level_match(audio), fs))
        print(f"  {path.name}: written")


if __name__ == "__main__":
    main()
