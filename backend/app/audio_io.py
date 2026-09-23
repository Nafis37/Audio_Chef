"""Audio loading and writing.
"""
from __future__ import annotations

import io
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config
from .dsp.speed_pitch import resample


def resolve_source(file_id: str) -> Path | None:
    """The stored upload for `file_id`, whatever extension it landed under, or None."""
    return next(config.STORAGE_DIR.glob(f"{file_id}.*"), None)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Read any libsndfile-decodable file -> (mono float64 samples, sample rate)."""
    data, samplerate = sf.read(str(path), dtype="float64", always_2d=True)
    mono = data.mean(axis=1)          # average the channels down to one
    return np.ascontiguousarray(mono), int(samplerate)


def load_channels(path: Path) -> tuple[np.ndarray, int]:
    """Every channel, unfolded: (frames, channels).  Only Signal Doctor's stereo check
    needs this -- everything else processes the mono fold from load_audio()."""
    data, samplerate = sf.read(str(path), dtype="float64", always_2d=True)
    return data, int(samplerate)


def probe(path: Path) -> dict:
    """Metadata for the frontend without decoding the whole file into the response."""
    info = sf.info(str(path))
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "duration": float(info.duration),
        "frames": int(info.frames),
    }


def write_wav_bytes(samples: np.ndarray, samplerate: int) -> bytes:
    """Encode a float array as a 16-bit PCM WAV in memory -> bytes for the HTTP response."""
    buffer = io.BytesIO()
    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    sf.write(buffer, clipped, int(samplerate), format="WAV", subtype="PCM_16")
    return buffer.getvalue()

# --------------------------------------------------------------------------------------
# Decoded-source cache.
# --------------------------------------------------------------------------------------
_CACHE_LIMIT = 4
_cache: "OrderedDict[tuple[str, int, int], tuple[np.ndarray, int]]" = OrderedDict()


def load_audio_cached(path: Path) -> tuple[np.ndarray, int]:
    """load_audio(), but decoding at most once per (file, revision)."""
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)

    entry = _cache.get(key)
    if entry is None:
        entry = load_audio(path)
        _cache[key] = entry
        while len(_cache) > _CACHE_LIMIT:
            _cache.popitem(last=False)      # evict the least recently used
    else:
        _cache.move_to_end(key)             # a hit refreshes its position

    samples, samplerate = entry
    return samples.copy(), samplerate


# --------------------------------------------------------------------------------------
# Resampled-source cache.
#
# The cache above memoises the DECODE.  A multi-source project whose files disagree on
# sample rate converts every off-rate source on the way in, and without a second cache a
# slider drag would pay a whole-file np.interp per off-rate source, 3x a second.
# --------------------------------------------------------------------------------------
_RESAMPLE_CACHE_LIMIT = 8
_resampled: "OrderedDict[tuple[str, int, int, int], np.ndarray]" = OrderedDict()


def load_audio_at(path: Path, target_fs: int) -> tuple[np.ndarray, int]:
    """The file's samples converted to `target_fs`, plus its own original rate.

    Returning the original rate as well lets the caller report which sources were
    converted, so a pitch change never goes unexplained in the UI.
    """
    samples, samplerate = load_audio_cached(path)
    target_fs = int(target_fs)
    if samplerate == target_fs:
        return samples, samplerate          # already a fresh copy from load_audio_cached

    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size, target_fs)

    entry = _resampled.get(key)
    if entry is None:
        # resample(x, speed) returns floor(size / speed) samples, so this ratio is what
        # holds the duration in SECONDS constant across the rate change.
        entry = resample(samples, samplerate / target_fs)
        _resampled[key] = entry
        while len(_resampled) > _RESAMPLE_CACHE_LIMIT:
            _resampled.popitem(last=False)
    else:
        _resampled.move_to_end(key)

    # Same discipline as above: hand out a copy, never the cached array itself.
    return entry.copy(), samplerate