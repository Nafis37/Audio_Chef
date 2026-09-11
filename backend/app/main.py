"""Audio Chef FastAPI application.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .routers import process, upload


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Make sure the scratch directory exists before the first upload arrives.
    # starts before app, yeilds during the app, does after yeild part after the app stops
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Audio Chef", version="1.0.0", lifespan=lifespan)

# The Vite dev server runs on a different port, so the browser treats it as a
# cross-origin caller; origins come from config so there is one place to change them.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Custom response headers are hidden from JS unless they are explicitly exposed.
    expose_headers=["X-Bake-Ms", "X-Output-Duration", "X-Bake-Stats"],
)

app.include_router(upload.router)
app.include_router(process.router)


@app.get("/health")
def health() -> dict:
    """Used by the frontend to show whether the DSP backend is reachable."""
    return {"status": "ok", "storage": str(config.STORAGE_DIR)}