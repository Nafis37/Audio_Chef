"""POST /upload -- accept an audio file and park it in the storage directory.

The response contains a `file_id` which the frontend then quotes on every /process call.
Uploading once and re-sending only the recipe JSON is what makes Auto-Bake feel instant:
a slider move costs a few hundred bytes, not a few megabytes.

There is deliberately NO format whitelist here.  libsndfile already knows exactly what it
can open, and probe() below raises on anything it cannot -- so the decoder is the gate and
the list of supported formats has exactly one definition (its own).  A whitelist would
only ever drift out of sync with the library actually doing the work.
"""

from __future__ import annotations

import re
import uuid

import soundfile as sf
from fastapi import APIRouter, File, HTTPException, UploadFile

from .. import config
from ..audio_io import probe

router = APIRouter()

# Suffixes are echoed onto a path, so they are rebuilt from scratch rather than trusted:
# letters and digits only, and short enough that "..\..\evil" cannot survive the filter.
_SUFFIX = re.compile(r"^[a-z0-9]{1,8}$")


def _extension(filename: str | None) -> str:
    """A safe on-disk suffix for an uploaded name, defaulting to .bin."""
    candidate = (filename or "").rsplit(".", 1)
    if len(candidate) == 2 and _SUFFIX.match(candidate[1].lower()):
        return f".{candidate[1].lower()}"
    return ".bin"


@router.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    contents = await file.read()
    if len(contents) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4().hex
    path = config.STORAGE_DIR / f"{file_id}{_extension(file.filename)}"
    path.write_bytes(contents)

    try:
        meta = probe(path)
    except (sf.LibsndfileError, RuntimeError) as exc:   # nothing libsndfile can decode
        # Nothing half-written is left behind for resolve_source() to find later.
        path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Could not decode this file as audio. Supported formats include WAV, "
                   "MP3, FLAC, OGG/Opus and AIFF (AAC/m4a is not supported).",
        ) from exc

    return {"file_id": file_id, "filename": file.filename, **meta}