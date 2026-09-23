"""POST /process -- render a multi-source project and return a WAV.

A project is a set of named sources, each with its own recipe, plus a master recipe that
runs over whichever source is designated the output.  An `assemble` card inside one
source's recipe pulls in another source's PROCESSED audio, so the set is a graph; the
walk lives in dsp/graph.py and this module is only the HTTP shell around it.

GET /operations exposes the same catalogue the DSP engine validates against, so the
palette in the browser is generated from the backend's own schema.

Every bake is also remembered for a short while under a `bake_id` (X-Bake-Id header), so
the spectrogram router can draw the exact buffer that was just rendered without the
browser sending the recipe again.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections import OrderedDict
from typing import Any, Optional

import numpy as np

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from .. import config
from ..audio_io import load_audio_at, probe, resolve_source, write_wav_bytes
from ..dsp import analysis, dsp_engine, filters, graph

router = APIRouter()

# Source ids are minted by the browser and echoed back inside error messages, so they are
# held to something narrow rather than trusted.
_SOURCE_ID = re.compile(r"[A-Za-z0-9_-]{1,32}")

# The last few rendered buffers, by bake_id.  A handful is plenty: the browser only ever
# asks for the spectrogram of the bake it has just received.
_BAKE_LIMIT = 4
_bakes: "OrderedDict[str, tuple[np.ndarray, int]]" = OrderedDict()


def remember_bake(buffer: np.ndarray, fs: int) -> str:
    """Keep a rendered buffer for the spectrogram router; returns its id."""
    bake_id = uuid.uuid4().hex
    _bakes[bake_id] = (buffer, fs)
    while len(_bakes) > _BAKE_LIMIT:
        _bakes.popitem(last=False)          # evict the oldest
    return bake_id


def recall_bake(bake_id: str) -> tuple[np.ndarray, int] | None:
    return _bakes.get(bake_id)


class RecipeStep(BaseModel):
    op: str
    bypass: bool = False
    params: dict[str, Any] = Field(default_factory=dict)


class ClipSpec(BaseModel):
    """One block on an Arrange timeline (dsp/arrange.py).  Seconds throughout."""

    source: str
    start: float = 0.0          # where on the timeline the block begins
    clip_start: float = 0.0     # the span of the source it plays ...
    clip_end: float = 0.0       # ... 0 = to the source's end
    gain_db: float = 0.0


class SourceSpec(BaseModel):
    """One tab and the recipe that belongs to it: a loaded file, or an arrangement.

    Exactly one of `file_id` / `clips` is set.  A file tab's audio is the decoded upload;
    an Arrange tab's audio is the sum of its clips (dsp/arrange.py), each cut from another
    tab's processed output.

    `id` is separate from `file_id` on purpose: the same upload can be loaded twice under
    two ids with two different chains -- a dry copy and a reverb copy, mixed together --
    which keying on file_id alone would make impossible.
    """

    id: str
    file_id: Optional[str] = None
    clips: Optional[list[ClipSpec]] = None
    recipe: list[RecipeStep] = Field(default_factory=list)


class ProcessRequest(BaseModel):
    sources: list[SourceSpec] = Field(default_factory=list)
    master: list[RecipeStep] = Field(default_factory=list)
    # Whose processed buffer gets rendered.  Not necessarily the source being edited:
    # the UI auditions a chain by pointing this at it with apply_master off.
    output_source: str
    apply_master: bool = True
    # Overrides the automatic choice (the highest rate among the sources).
    sample_rate: Optional[int] = None


@router.get("/operations")
def list_operations() -> list[dict]:
    """The tool catalogue and its parameter schemas (min/max/step/default/unit)."""
    return dsp_engine.operations_schema()


@router.get("/filter/response")
def filter_response(
    mode: str = "lowpass",
    cutoff: float = 500.0,
    order: str = "4",
    q: float = 2.0,
    fs: int = config.DEFAULT_SAMPLE_RATE,
) -> dict[str, Any]:
    """The Filter card's frequency-response curve: |H(e^{jw})| in dB on a log axis.

    Values go through the same schema coercion as a bake (clamp, never reject), and the
    curve is evaluated from the very coefficients cutoff_filter() runs -- so the picture
    on the card is the filter that is applied, not a sketch of it.
    """
    schema = {p["name"]: p for p in dsp_engine._BY_ID["filter"]["params"]}
    supplied = {"mode": mode, "cutoff": cutoff, "order": order, "q": q}
    params = {name: dsp_engine._coerce(schema[name], value) for name, value in supplied.items()}
    fs = int(np.clip(fs, 8000, config.MAX_PROJECT_SAMPLE_RATE))
    freqs, db = filters.filter_response(fs, **params)
    return {"freqs": np.round(freqs, 2).tolist(), "db": np.round(db, 2).tolist(),
            "cutoff": params["cutoff"], "fs": fs}


def _validate(request: ProcessRequest) -> None:
    """Everything that can be rejected without touching the filesystem."""
    if not request.sources:
        raise HTTPException(status_code=400, detail="No sources loaded.")
    if len(request.sources) > config.MAX_SOURCES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many sources -- this project holds at most {config.MAX_SOURCES}.",
        )

    seen: set[str] = set()
    for spec in request.sources:
        if (spec.file_id is None) == (spec.clips is None):
            raise HTTPException(
                status_code=400,
                detail=f"Source {spec.id!r} needs exactly one of file_id or clips.",
            )
        if spec.clips is not None:
            if len(spec.clips) > config.MAX_CLIPS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Too many blocks -- an arrangement holds at most {config.MAX_CLIPS}.",
                )
        # file_id comes from the client, so refuse anything that could escape storage.
        elif not spec.file_id.isalnum():
            raise HTTPException(status_code=400, detail="Malformed file_id.")
        if not _SOURCE_ID.fullmatch(spec.id):
            raise HTTPException(status_code=400, detail=f"Malformed source id {spec.id!r}.")
        if spec.id in seen:
            raise HTTPException(status_code=400, detail=f"Duplicate source id {spec.id!r}.")
        seen.add(spec.id)

    if request.output_source not in seen:
        raise HTTPException(
            status_code=400,
            detail=f"Output source {request.output_source!r} is not one of the loaded sources.",
        )


@router.post("/process")
def process(request: ProcessRequest) -> Response:
    _validate(request)

    # Resolve every upload up front: a missing file should fail before any DSP runs, and
    # the project's sample rate is decided from the whole set.  probe() reads the header
    # only, so this costs no decoding.
    paths = {}
    rates = []
    for spec in request.sources:
        if spec.file_id is None:        # an arrangement: its rate is its sources' rates
            continue
        path = resolve_source(spec.file_id)
        if path is None:
            raise HTTPException(
                status_code=404, detail="Unknown file_id -- re-upload the audio."
            )
        paths[spec.file_id] = path
        rates.append(probe(path)["sample_rate"])

    project_fs = graph.project_sample_rate(
        rates, request.sample_rate, config.MAX_PROJECT_SAMPLE_RATE
    )

    def loader(file_id: str):
        """Decode one source at the project rate.  Injected so graph.py stays file-free."""
        return load_audio_at(paths[file_id], project_fs)

    started = time.perf_counter()
    try:
        processed, raw_input, report = graph.evaluate(
            sources=request.sources,
            master=request.master,
            output_source=request.output_source,
            loader=loader,
            fs=project_fs,
            apply_master=request.apply_master,
        )
    except ValueError as exc:      # unknown op id, or any graph.GraphError
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    # "input" stays the RAW buffer of the rendered source, which is what the Input
    # waveform shows and what the input column of the stats table has always meant.
    stats = {
        "sample_rate": project_fs,
        "input": analysis.measure(raw_input, project_fs),
        "output": analysis.measure(processed, project_fs),
        **report,
    }

    wav_bytes = write_wav_bytes(processed, project_fs)
    bake_id = remember_bake(processed, project_fs)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            # Read by the UI to show "baked in 143 ms"; must be exposed for CORS to allow it.
            "X-Bake-Ms": f"{elapsed_ms:.1f}",
            "X-Output-Duration": f"{len(processed) / project_fs:.3f}",
            # Input/output measurements plus what the walk did, as one compact JSON
            # header, so a bake stays a single round trip.
            "X-Bake-Stats": json.dumps(stats, separators=(",", ":")),
            # GET /spectrogram/bake/{id} draws this exact buffer.
            "X-Bake-Id": bake_id,
            # No Content-Disposition: the browser fetches this, turns it into a blob URL
            # and plays it inline.  Marking it `attachment` made the browser offer to save
            # the file on every bake, which is not what a live preview should do.
            "Cache-Control": "no-store",
        },
    )
