"""POST /process -- run a recipe over a previously uploaded file and return a WAV.

GET /operations exposes the same catalogue the DSP engine validates against, so the
palette in the browser is generated from the backend's own schema.
"""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from ..audio_io import load_audio_cached, resolve_source, write_wav_bytes
from ..dsp import analysis, dsp_engine

router = APIRouter()


class RecipeStep(BaseModel):
    op: str
    bypass: bool = False
    params: dict[str, Any] = Field(default_factory=dict)


class ProcessRequest(BaseModel):
    file_id: str
    recipe: list[RecipeStep] = Field(default_factory=list)


@router.get("/operations")
def list_operations() -> list[dict]:
    """The 7 tools and their parameter schemas (min/max/step/default/unit)."""
    return dsp_engine.operations_schema()


@router.post("/process")
def process(request: ProcessRequest) -> Response:
    # file_id comes from the client, so refuse anything that could escape the storage dir.
    if not request.file_id.isalnum():
        raise HTTPException(status_code=400, detail="Malformed file_id.")

    # Uploads keep their own extension, so the id has to be resolved rather than formatted.
    path = resolve_source(request.file_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Unknown file_id -- re-upload the audio.")

    samples, samplerate = load_audio_cached(path)

    started = time.perf_counter()
    try:
        processed, report = dsp_engine.run_recipe_measured(
            samples, samplerate, [step.model_dump() for step in request.recipe]
        )
    except ValueError as exc:                    # unknown op id from run_recipe
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    # Both buffers are already in memory, so the Listen panel's numbers cost a few O(N)
    # passes here rather than a second decode in the browser -- and they describe the
    # exact mono float64 signal the DSP actually ran on.
    stats = {
        "sample_rate": samplerate,
        "input": analysis.measure(samples, samplerate),
        "output": analysis.measure(processed, samplerate),
        **report,
    }

    wav_bytes = write_wav_bytes(processed, samplerate)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            # Read by the UI to show "baked in 143 ms"; must be exposed for CORS to allow it.
            "X-Bake-Ms": f"{elapsed_ms:.1f}",
            "X-Output-Duration": f"{len(processed) / samplerate:.3f}",
            # Input/output measurements plus what the fold did, as one compact JSON
            # header, so a bake stays a single round trip.
            "X-Bake-Stats": json.dumps(stats, separators=(",", ":")),
            # No Content-Disposition: the browser fetches this, turns it into a blob URL
            # and plays it inline.  Marking it `attachment` made the browser offer to save
            # the file on every bake, which is not what a live preview should do.
            "Cache-Control": "no-store",
        },
    )