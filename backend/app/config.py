"""Runtime configuration for the Audio Chef backend."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# where will the audio files be stored
STORAGE_DIR = Path(
    os.environ.get("AUDIO_CHEF_STORAGE", Path(tempfile.gettempdir()) / "audio-chef")
)

#vite dev servers
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "AUDIO_CHEF_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

MAX_UPLOAD_BYTES = int(os.environ.get("AUDIO_CHEF_MAX_UPLOAD_BYTES", 100 * 1024 * 1024))

DEFAULT_SAMPLE_RATE = 44100

# How many sources one project may hold.  This bounds evaluation cost, and also keeps the
# per-source rows in the X-Bake-Stats response header well under the server's header size
# limit -- they ride back on every bake.
MAX_SOURCES = 16
# Blocks on one Arrange timeline.  Each is a slice + a sum, so this is a sanity bound.
MAX_CLIPS = 256

# The project runs at the highest rate among its sources, but not higher than this: past
# here the extra samples cost real time on every bake and buy nothing audible.
MAX_PROJECT_SAMPLE_RATE = 96000