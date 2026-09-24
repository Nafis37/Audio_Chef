"""
The source graph -- resolving one recipe per source, plus a master chain
=======================================================================
A project is no longer one buffer and one recipe.  It is a set of named sources, each
with its own chain, and an `assemble` card inside a chain can pull in the PROCESSED
output of another source.  That makes the set a DAG rather than a list, and this module
is what walks it.  An Arrange tab is a source whose raw audio is itself built from other
sources' processed output (arrange.py), which is just more edges in the same DAG.
It is also the one STEREO source (its tracks are panned): its chain runs per channel
(run_chain), and whatever reads it from another source gets its mono fold (processed()).

    evaluate(sources, master, output_source, ...) -> (samples, fs, report)

Three properties the walk is built around:

  * Memoised.  Two cards referencing the same source evaluate it once.
  * Lazy.  A source is resolved when a handler actually reaches for it, not by
    pre-scanning the chain.  That makes a BYPASSED assemble card free -- and, more
    usefully, it means bypassing a card is a working escape hatch out of an accidental
    A -> B -> A cycle, instead of having to delete it to make the app respond again.
  * Ordered cycle detection.  The in-progress stack is a list, not a set, so the error
    can name the actual path ("s1 -> s3 -> s1") rather than just asserting one exists.

File I/O is injected as `loader`, so this module can be exercised against a dict of fake
buffers with no storage directory in sight.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np

from . import arrange, dsp_engine, editor

# A chain that assembles a chain that assembles ... this deep is not a real edit; it is
# almost certainly a mistake, and the recursion has to stop somewhere regardless.
MAX_DEPTH = 32

# (file_id) -> (mono samples already at the project rate, that file's ORIGINAL rate).
# The original comes back so the walk can report which sources were converted.
Loader = Callable[[str], "tuple[np.ndarray, int]"]


class GraphError(ValueError):
    """A structural problem with the graph: unknown source, cycle, nothing to render.

    Deliberately a ValueError -- the process router already turns ValueError into a 400
    with the message as `detail`, so these reach the UI's error chip unaided.
    """


class ClipContext:
    """What a graph-aware handler (`assemble`) is handed to reach other sources.

    Deliberately tiny: a handler gets one verb -- `clip()` -- and no way to touch the
    evaluator's caches, the file store or the rest of the graph.
    """

    def __init__(self, evaluator: "Evaluator", owner: str) -> None:
        self._evaluator = evaluator
        self._owner = owner

    def clip(self, source_id: str, start: float = 0.0, end: float = 0.0) -> np.ndarray:
        """The processed output of `source_id`, trimmed to [start, end) seconds."""
        source_id = (source_id or "").strip()
        if not source_id:
            raise GraphError("An Assemble Clip card has no source selected.")
        if source_id == self._owner:
            raise GraphError(
                f"Source {source_id!r} cannot assemble itself -- pick a different source."
            )

        whole = self._evaluator.processed(source_id)
        # editor.trim gives us the app's existing conventions for free: `end <= 0` means
        # "to the end of the clip", and both cut edges come back already faded.
        segment = editor.trim(whole, self._evaluator.fs, start=start, end=end)

        # COPY, and not an optional one.  trim() returns a VIEW (x[a:b]), and fade_edges
        # hands back its input untouched for buffers shorter than two ramps -- so this
        # can easily be a window onto the memoised array in _done.  mix_at accumulates
        # with `+=`, so without this copy one assemble card would quietly rewrite the
        # source that the next card is about to read.
        return np.array(segment, dtype=np.float64, copy=True)


class Evaluator:
    """Resolves each source's chain at most once, detecting cycles as it goes."""

    def __init__(self, specs: Iterable[Any], loader: Loader, fs: int) -> None:
        self._specs = {spec.id: spec for spec in specs}
        self._loader = loader
        self.fs = fs

        self._raw: dict[str, np.ndarray] = {}      # decoded + resampled, pre-chain
        self._done: dict[str, np.ndarray] = {}     # memo: chain already folded
        self._mono: dict[str, np.ndarray] = {}     # memo: stereo outputs folded to mono
        self._stack: list[str] = []                # in-progress, IN ORDER, for the path
        self._reports: dict[str, dict[str, Any]] = {}
        self.resampled: list[str] = []             # ids whose file rate != project rate

    # -- raw input ---------------------------------------------------------------------
    def raw(self, source_id: str) -> np.ndarray:
        """The source's decoded audio at the PROJECT rate, before its chain runs.

        The loader delivers it already converted, so the whole evaluation has exactly one
        sample rate and no handler ever learns that mixed rates existed.  That matters
        because every DSP module derives its constants from fs -- the EQ's biquad
        coefficients, the compressor's attack/release poles, the STFT's bin-to-Hz mapping
        -- and converting once at the door keeps all of them correct for free.
        """
        cached = self._raw.get(source_id)
        if cached is not None:
            return cached

        spec = self._spec(source_id)
        if getattr(spec, "clips", None) is not None:
            # An Arrange tab: its "file" is the sum of its clips.  Built here, which
            # processed() only calls AFTER pushing this source on the stack -- so a block
            # that points back at its own arrangement is caught as a cycle.
            # Stereo (frames, 2): the only kind of source that is -- tracks are panned.
            buffer = arrange.render_arrangement(
                self.processed, spec.clips, self.fs, getattr(spec, "tracks", None)
            )
            self._raw[source_id] = buffer
            return buffer

        samples, file_fs = self._loader(spec.file_id)
        if int(file_fs) != int(self.fs):
            self.resampled.append(source_id)

        # The loader may be serving from a cache, so never let a chain mutate what it
        # handed over -- every handler is free to write in place.
        buffer = np.array(samples, dtype=np.float64, copy=True)
        self._raw[source_id] = buffer
        return buffer

    # -- processed output --------------------------------------------------------------
    def processed(self, source_id: str) -> np.ndarray:
        """`source_id`'s processed audio, MONO -- what another source reaches for.

        Everything that consumes a source (an assemble card, a timeline block, a voice
        reference) works in mono, so a stereo arrangement is folded here, once.
        """
        result = self.output(source_id)
        if result.ndim == 1:
            return result
        mono = self._mono.get(source_id)
        if mono is None:
            mono = self._mono[source_id] = fold_to_mono(result)
        return mono

    def output(self, source_id: str) -> np.ndarray:
        """Fold `source_id`'s chain and return the result, resolving what it reaches for.

        Mono for a file tab, (frames, 2) for an Arrange tab.
        """
        memo = self._done.get(source_id)
        if memo is not None:
            return memo

        if source_id in self._stack:
            path = self._stack[self._stack.index(source_id) :] + [source_id]
            raise GraphError("Circular reference: " + " -> ".join(path))
        if len(self._stack) >= MAX_DEPTH:
            raise GraphError(
                f"Assembly is nested more than {MAX_DEPTH} sources deep -- "
                "something is referencing itself in a long loop."
            )

        spec = self._spec(source_id)
        self._stack.append(source_id)
        try:
            result, report = run_chain(
                self.raw(source_id),
                self.fs,
                [step.model_dump() for step in spec.recipe],
                ctx=ClipContext(self, source_id),
                # A source chain is an intermediate: its output feeds an assemble card or
                # the master chain, not the encoder.  Turning it down here would change
                # its balance against whatever it is mixed with next.  The fit to full
                # scale happens once, in evaluate(), on the buffer that becomes the WAV.
                clip=False,
            )
        finally:
            # Pop even when the fold raised, so a failed branch does not poison the
            # stack and make an unrelated later source look like a cycle.
            self._stack.pop()

        self._done[source_id] = result
        self._reports[source_id] = report
        return result

    # -- reporting ---------------------------------------------------------------------
    def source_rows(self) -> list[dict[str, Any]]:
        """A compact row per source that was actually evaluated.

        Compact on purpose: this rides in the X-Bake-Stats response header, and a full
        analysis.measure() for every source would push a 16-source project close to the
        server's header size limit.
        """
        rows = []
        for source_id, report in self._reports.items():
            buffer = self._done[source_id]
            rows.append({
                "id": source_id,
                "frames": int(buffer.shape[0]),
                "duration": round(buffer.shape[0] / self.fs, 3) if self.fs else 0.0,
                "steps_applied": report["steps_applied"],
                "pre_clip_peak": report["pre_clip_peak"],
                "truncated": report["truncated"],
            })
        return rows

    def _spec(self, source_id: str):
        spec = self._specs.get(source_id)
        if spec is None:
            raise GraphError(
                f"Unknown source {source_id!r} -- it may have been removed from the project."
            )
        return spec


def fold_to_mono(x: np.ndarray) -> np.ndarray:
    """(frames, channels) -> the average of the channels; a mono buffer passes through."""
    return x.mean(axis=1) if x.ndim == 2 else x


def run_chain(
    x: np.ndarray, fs: int, recipe: list[dict[str, Any]], ctx: Any = None, clip: bool = True
) -> tuple[np.ndarray, dict[str, Any]]:
    """dsp_engine.run_recipe_measured, for a mono OR a stereo buffer.

    Every operation is written for one channel, so a stereo buffer runs the recipe on
    each channel on its own.  The fit to full scale is then made once over BOTH channels
    -- fitting them separately would shift the stereo balance.
    """
    if x.ndim == 1:
        return dsp_engine.run_recipe_measured(x, fs, recipe, ctx=ctx, clip=clip)

    runs = [dsp_engine.run_recipe_measured(x[:, c], fs, recipe, ctx=ctx, clip=False)
            for c in range(x.shape[1])]
    frames = max(y.size for y, _ in runs)       # every op is deterministic, but be safe
    y = np.zeros((frames, len(runs)), dtype=np.float64)
    for c, (channel, _) in enumerate(runs):
        y[: channel.size, c] = channel
    reports = [report for _, report in runs]
    report = {
        **reports[0],
        "pre_clip_peak": max(r["pre_clip_peak"] for r in reports),
        "clipped": sum(r["clipped"] for r in reports),
        "truncated": any(r["truncated"] for r in reports),
    }
    if clip:
        y, report["normalised_db"] = dsp_engine.fit_to_full_scale(y)
    return y, report


def project_sample_rate(
    rates: Iterable[int], requested: int | None, cap: int
) -> int:
    """Pick the one rate the whole evaluation runs at.

    The MAXIMUM of the loaded files, not the minimum: speed_pitch.resample is bare linear
    interpolation with no anti-alias filter, so downsampling through it folds everything
    above the new Nyquist back into the audible band.  Upsampling only adds imaging,
    which is far more forgiving, so we only ever go down when the cap forces it.
    """
    if requested:
        return int(min(max(int(requested), 8000), cap))
    known = [int(rate) for rate in rates if rate]
    if not known:
        return cap
    return min(max(known), cap)


def evaluate(
    sources: list[Any],
    master: list[Any],
    output_source: str,
    loader: Loader,
    fs: int,
    apply_master: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Render `output_source`, optionally through the master chain.

    Returns (final buffer, the output source's RAW buffer, report) -- the raw buffer
    comes back because only this walk knows which source was rendered and at what rate,
    and the router needs it for the "input" half of the stats.
    """
    if not sources:
        raise GraphError("No sources loaded.")

    evaluator = Evaluator(sources, loader, fs)
    rendered = evaluator.output(output_source)       # stereo for an Arrange tab

    # The Input panel and the input column of the stats table have always meant "the
    # audio as it arrived, before the recipe" -- keep that meaning by measuring the
    # output source's raw buffer rather than anything assembled.
    raw_input = evaluator.raw(output_source)

    if apply_master and master:
        # No ctx: the master chain has no source identity of its own, so an assemble
        # card here has nothing to be relative to.  run_recipe_measured raises on that, which
        # the router turns into a readable 400.
        final, master_report = run_chain(
            rendered, fs, [step.model_dump() for step in master]
        )
    else:
        final, normalised_db = dsp_engine.fit_to_full_scale(rendered)
        master_report = {
            "pre_clip_peak": round(float(np.max(np.abs(rendered))) if rendered.size else 0.0, 6),
            "clipped": int(np.count_nonzero(np.abs(rendered) > 1.0)) if rendered.size else 0,
            "normalised_db": normalised_db,
            "steps_applied": 0,
            "steps_bypassed": 0,
            "truncated": False,
        }

    report = {
        **master_report,
        "project_sample_rate": fs,
        "resampled": evaluator.resampled,
        "sources": evaluator.source_rows(),
    }
    return final, raw_input, report
