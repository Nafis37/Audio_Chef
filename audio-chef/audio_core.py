"""
Day 1 — Audio as a signal.

Core idea: an audio file is nothing more than a 1D (mono) or 2D (stereo)
NumPy array of amplitude values, captured `sample_rate` times per second.
This module only *reads* and *describes* that array — no processing yet,
that starts on later days (noise removal, EQ, effects, etc).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import soundfile as sf


@dataclass
class SignalInfo:
    """Everything Day 1 asks us to understand about a loaded signal."""

    sample_rate: int
    channels: int
    num_samples: int
    duration_sec: float
    subtype: str          # underlying bit-depth / encoding, e.g. PCM_16
    dtype: str             # numpy dtype the array was decoded into
    min_amplitude: float
    max_amplitude: float
    rms_amplitude: float
    preview: list = field(default_factory=list)  # first N raw sample values


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file into a NumPy array.

    Returns (data, sample_rate).
    data.shape is (num_samples,) for mono, or (num_samples, num_channels)
    for stereo/multichannel — soundfile always puts *time* on axis 0.
    """
    data, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    return data, sample_rate


def analyze(data: np.ndarray, sample_rate: int, path: str, preview_len: int = 12) -> SignalInfo:
    """Turn the raw array + sample rate into human-readable signal facts."""

    is_stereo = data.ndim > 1
    channels = data.shape[1] if is_stereo else 1
    num_samples = data.shape[0]
    duration = num_samples / sample_rate

    info = sf.info(path)

    # For preview + min/max/rms, collapse to mono view without losing the
    # per-channel picture if stereo (we show channel 0 in the preview list).
    preview_source = data[:, 0] if is_stereo else data

    return SignalInfo(
        sample_rate=sample_rate,
        channels=channels,
        num_samples=num_samples,
        duration_sec=duration,
        subtype=info.subtype,
        dtype=str(data.dtype),
        min_amplitude=float(np.min(data)),
        max_amplitude=float(np.max(data)),
        rms_amplitude=float(np.sqrt(np.mean(np.square(data)))),
        preview=[round(float(x), 5) for x in preview_source[:preview_len]],
    )


def downsample_for_plot(channel: np.ndarray, max_points: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Min/max envelope downsampling so long waveforms still render fast
    and still look like a waveform (not an aliased mess).

    Returns (x_indices, y_values) ready to hand to a plotting library.
    """
    n = len(channel)
    if n <= max_points:
        return np.arange(n), channel

    bucket_size = int(np.ceil(n / (max_points / 2)))
    num_buckets = int(np.ceil(n / bucket_size))

    xs = np.empty(num_buckets * 2)
    ys = np.empty(num_buckets * 2)

    for i in range(num_buckets):
        start = i * bucket_size
        end = min(start + bucket_size, n)
        bucket = channel[start:end]
        xs[2 * i] = start
        xs[2 * i + 1] = end - 1
        ys[2 * i] = bucket.max()
        ys[2 * i + 1] = bucket.min()

    return xs, ys
