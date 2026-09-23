"""GET /diagnose/... -- Signal Doctor's checklist and prescription (dsp/doctor.py).

Same two sources as the spectrogram router:

    /diagnose/file/{file_id}   an upload as decoded (with its channels, for the stereo check)
    /diagnose/bake/{bake_id}   a rendered result -- "did the fix work?"  Bakes are mono,
                               so the stereo row reads "Mono file" there.

The response is JSON: {"findings": [...], "fix": [recipe steps]}.  The fix steps are
ordinary recipe cards in display units; the UI appends them to the open tab's recipe.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..audio_io import load_audio_cached, load_channels, resolve_source
from ..dsp.doctor import diagnose
from .process import recall_bake

router = APIRouter()


@router.get("/diagnose/file/{file_id}")
def diagnose_file(file_id: str) -> dict:
    if not file_id.isalnum():            # same guard as /process: never escape storage
        raise HTTPException(status_code=400, detail="Malformed file_id.")
    path = resolve_source(file_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Unknown file_id -- re-upload the audio.")
    samples, fs = load_audio_cached(path)
    channels, _ = load_channels(path)
    return diagnose(samples, fs, channels)


@router.get("/diagnose/bake/{bake_id}")
def diagnose_bake(bake_id: str) -> dict:
    entry = recall_bake(bake_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="That bake has expired -- bake again.")
    samples, fs = entry
    return diagnose(samples, fs)
