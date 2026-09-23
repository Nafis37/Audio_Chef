"""GET /spectrogram/... -- the picture drawn under each waveform.

Two sources of audio, one picture format (dsp/spectrogram.py):

    /spectrogram/file/{file_id}   an upload as decoded -- the Input panel, before any bake
    /spectrogram/bake/{bake_id}   a rendered result -- the Output panel, after each bake

The body is the raw uint8 grid, row-major, highest frequency first; the shape and the
frequency range ride in headers so the browser can draw it straight onto a canvas.  Raw
bytes rather than a PNG: at 160 x 600 it is under 100 kB, and it keeps image encoding
(and any library for it) out of the backend.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, Response

from ..audio_io import load_audio_cached, resolve_source
from ..dsp.spectrogram import spectrogram_image
from .process import recall_bake

router = APIRouter()


def _respond(x: np.ndarray, fs: int) -> Response:
    image, meta = spectrogram_image(x, fs)
    return Response(
        content=image.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Spec-Rows": str(meta["rows"]),
            "X-Spec-Cols": str(meta["cols"]),
            "X-Spec-Fmin": f"{meta['f_min']:.1f}",
            "X-Spec-Fmax": f"{meta['f_max']:.1f}",
            "Cache-Control": "no-store",
        },
    )


@router.get("/spectrogram/file/{file_id}")
def file_spectrogram(file_id: str) -> Response:
    if not file_id.isalnum():            # same guard as /process: never escape storage
        raise HTTPException(status_code=400, detail="Malformed file_id.")
    path = resolve_source(file_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Unknown file_id -- re-upload the audio.")
    samples, fs = load_audio_cached(path)
    return _respond(samples, fs)


@router.get("/spectrogram/bake/{bake_id}")
def bake_spectrogram(bake_id: str) -> Response:
    entry = recall_bake(bake_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="That bake has expired -- bake again.")
    return _respond(*entry)
