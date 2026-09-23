/**
 * The Arrange tab's timeline: file blocks on lanes, like a DAW or a video editor.
 *
 *   - drag a file chip from the strip above onto a lane      -> a new block
 *   - drag a block's body                                   -> move it (in time and lane)
 *   - drag a block's left / right edge                      -> trim it
 *   - Alt-drag a block                                      -> a copy
 *   - click empty space (or the ruler)                      -> put the cursor there
 *   - S, or the Split button                                -> cut the selected block at the cursor
 *   - Delete / Backspace                                    -> remove the selected block
 *
 * Moves and trims snap to the edges of other blocks (within SNAP_PX) and otherwise to a
 * 0.05 s grid.  A drag is previewed locally and committed on release, so one drag is one
 * change and one bake.
 *
 * The audio is the sum of the blocks (backend dsp/arrange.py): lanes only keep
 * overlapping blocks visible, and have no effect on the sound.  Each block plays the
 * file tab's PROCESSED output but draws the raw file's outline (peaks.ts); when that
 * tab's recipe changes the length, the block shows a warning, as recipe cards do.
 */

import { AlertTriangle, Minus, Plus, Scissors, Trash2 } from 'lucide-react'
import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'

import { usePeaks, type Peaks } from '../peaks'
import { TIME_SHIFTING_OPS } from '../regions'
import { formatTime, rulerSpacing } from '../time'
import type { ArrangeClip, Source } from '../types'

interface Props {
  clips: ArrangeClip[]
  /** The file tabs a block may play. */
  files: Source[]
  onChange: (clips: ArrangeClip[]) => void
}

const LANE_H = 56
const RULER_H = 24
const EDGE_PX = 7          // grab zone for trimming, each side
const SNAP_PX = 8
const GRID_S = 0.05
const MIN_LEN_S = 0.05
const DRAG_MIME = 'application/x-audio-chef-source'

type Mode = 'move' | 'left' | 'right'

interface Drag {
  uid: string
  mode: Mode
  x0: number
  y0: number
  original: ArrangeClip
  /** Alt-drag: the preview is a NEW block, the original stays put. */
  copy: boolean
  moved: boolean
}

function uid(): string {
  return `clip-${crypto.randomUUID()}`
}

/** Seconds of source the block plays. */
function spanOf(clip: ArrangeClip, duration: number): number {
  const end = clip.clipEnd > 0 ? clip.clipEnd : duration
  return Math.max(0, end - clip.clipStart)
}

function TimelineImpl({ clips, files, onChange }: Props) {
  const root = useRef<HTMLDivElement>(null)
  const scroller = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(600)
  const [zoom, setZoom] = useState(1)
  const [selected, setSelected] = useState<string | null>(null)
  const [cursor, setCursor] = useState<number | null>(null)
  // The block being dragged, previewed until release; `copy` = Alt-drag (a new block).
  const [preview, setPreview] = useState<{ clip: ArrangeClip; copy: boolean } | null>(null)
  const drag = useRef<Drag | null>(null)

  const byId = useMemo(() => new Map(files.map((file) => [file.id, file])), [files])
  const durationOf = (clip: ArrangeClip) => byId.get(clip.source)?.duration ?? 0

  useLayoutEffect(() => {
    const element = scroller.current
    if (!element) return
    const observer = new ResizeObserver(() => setWidth(element.clientWidth))
    observer.observe(element)
    setWidth(element.clientWidth)
    return () => observer.disconnect()
  }, [])

  // What is drawn: the committed blocks with the one being dragged swapped for its preview.
  const shown = useMemo(() => {
    if (!preview) return clips
    if (preview.copy) return [...clips, preview.clip]
    return clips.map((clip) => (clip.uid === preview.clip.uid ? preview.clip : clip))
  }, [clips, preview])

  const contentEnd = Math.max(
    0,
    ...shown.map((clip) => clip.start + spanOf(clip, durationOf(clip))),
  )
  const fitSeconds = Math.max(10, contentEnd + 2)
  const pps = (width / fitSeconds) * zoom                      // pixels per second
  const innerWidth = Math.max(width, (contentEnd + 5) * pps)
  const lanes = Math.max(3, ...shown.map((clip) => clip.lane + 2))
  const spacing = rulerSpacing(innerWidth / pps, innerWidth)

  /**
   * Snap a block starting at `t` (and `len` long) so its start OR its end meets another
   * block's edge when within SNAP_PX; otherwise round the start to the grid.
   */
  const snap = (t: number, except: string, len = 0): number => {
    const edges = [0]
    for (const clip of clips) {
      if (clip.uid === except) continue
      edges.push(clip.start, clip.start + spanOf(clip, durationOf(clip)))
    }
    let best: number | null = null
    let bestPx = SNAP_PX
    for (const edge of edges) {
      const startPx = Math.abs(edge - t) * pps
      if (startPx < bestPx) [best, bestPx] = [edge, startPx]
      const endPx = Math.abs(edge - (t + len)) * pps
      if (len > 0 && endPx < bestPx) [best, bestPx] = [edge - len, endPx]
    }
    return Math.max(0, best ?? Math.round(t / GRID_S) * GRID_S)
  }

  const timeAt = (clientX: number): number => {
    const box = scroller.current!.getBoundingClientRect()
    return Math.max(0, (clientX - box.left + scroller.current!.scrollLeft) / pps)
  }

  // --- blocks: move / trim / copy ----------------------------------------------------
  const onBlockDown = (event: React.PointerEvent<HTMLDivElement>, clip: ArrangeClip) => {
    event.stopPropagation()
    root.current?.focus()
    setSelected(clip.uid)
    const box = event.currentTarget.getBoundingClientRect()
    const x = event.clientX - box.left
    const mode: Mode = x < EDGE_PX ? 'left' : x > box.width - EDGE_PX ? 'right' : 'move'
    const copy = mode === 'move' && event.altKey
    drag.current = {
      uid: copy ? uid() : clip.uid,
      mode,
      x0: event.clientX,
      y0: event.clientY,
      original: clip,
      copy,
      moved: false,
    }
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  const onBlockMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const d = drag.current
    if (!d) return
    const dx = (event.clientX - d.x0) / pps
    if (!d.moved && Math.abs(event.clientX - d.x0) < 3 && Math.abs(event.clientY - d.y0) < 3) return
    d.moved = true
    const o = d.original
    const duration = durationOf(o)
    const len = spanOf(o, duration)
    let next: ArrangeClip
    if (d.mode === 'move') {
      const lane = Math.max(0, o.lane + Math.round((event.clientY - d.y0) / LANE_H))
      next = { ...o, uid: d.uid, lane, start: snap(o.start + dx, o.uid, len) }
    } else if (d.mode === 'left') {
      // The left edge moves in time AND in the source together: the audio under the
      // block stays where it was, only less of it plays.
      const start = snap(o.start + dx, o.uid)
      // Never before the source's first sample, never shorter than MIN_LEN_S.
      const shift = Math.min(Math.max(start - o.start, -o.clipStart), len - MIN_LEN_S)
      next = { ...o, start: o.start + shift, clipStart: o.clipStart + shift }
    } else {
      const end = snap(o.start + len + dx, o.uid) - o.start + o.clipStart
      const clamped = Math.min(Math.max(end, o.clipStart + MIN_LEN_S), duration)
      next = { ...o, clipEnd: clamped >= duration - 1e-3 ? 0 : clamped }
    }
    setPreview({ clip: next, copy: d.copy })
  }

  const onBlockUp = () => {
    const d = drag.current
    drag.current = null
    const result = preview?.clip
    setPreview(null)
    if (!d || !d.moved || !result) return
    if (d.copy) {
      onChange([...clips, result])
      setSelected(result.uid)
    } else {
      onChange(clips.map((clip) => (clip.uid === result.uid ? result : clip)))
    }
  }

  // --- new blocks from the file strip -----------------------------------------------
  const onDrop = (event: React.DragEvent<HTMLDivElement>) => {
    const source = event.dataTransfer.getData(DRAG_MIME)
    if (!source || !byId.has(source)) return
    event.preventDefault()
    const box = event.currentTarget.getBoundingClientRect()
    const lane = Math.max(0, Math.floor((event.clientY - box.top) / LANE_H))
    const clip: ArrangeClip = {
      uid: uid(),
      source,
      start: snap(timeAt(event.clientX), ''),
      clipStart: 0,
      clipEnd: 0,
      gainDb: 0,
      lane,
    }
    onChange([...clips, clip])
    setSelected(clip.uid)
  }

  // --- selection tools ---------------------------------------------------------------
  const selectedClip = clips.find((clip) => clip.uid === selected) ?? null

  const splitSelected = () => {
    if (!selectedClip || cursor === null) return
    const len = spanOf(selectedClip, durationOf(selectedClip))
    const at = cursor - selectedClip.start
    if (at <= MIN_LEN_S || at >= len - MIN_LEN_S) return
    const cut = selectedClip.clipStart + at
    const left = { ...selectedClip, clipEnd: cut }
    const right = { ...selectedClip, uid: uid(), start: cursor, clipStart: cut }
    onChange(clips.flatMap((clip) => (clip.uid === selectedClip.uid ? [left, right] : [clip])))
    setSelected(right.uid)
  }

  const deleteSelected = () => {
    if (!selectedClip) return
    onChange(clips.filter((clip) => clip.uid !== selectedClip.uid))
    setSelected(null)
  }

  const setGain = (gainDb: number) => {
    if (!selectedClip) return
    onChange(clips.map((clip) => (clip.uid === selectedClip.uid ? { ...clip, gainDb } : clip)))
  }

  const onKeyDown = (event: React.KeyboardEvent) => {
    if ((event.target as HTMLElement).closest('input, select, textarea')) return
    if (event.key === 's' || event.key === 'S') {
      event.preventDefault()
      splitSelected()
    } else if (event.key === 'Delete' || event.key === 'Backspace') {
      event.preventDefault()
      deleteSelected()
    }
  }

  const ticks = useMemo(() => {
    const out: number[] = []
    const total = innerWidth / pps
    for (let t = 0; t <= total + 1e-9; t += spacing.label) out.push(t)
    return out
  }, [innerWidth, pps, spacing.label])

  return (
    <section
      ref={root}
      tabIndex={-1}
      onKeyDown={onKeyDown}
      className="rounded-lg border border-[var(--chef-border)] bg-[var(--chef-panel)] outline-none"
    >
      <header className="flex flex-wrap items-center gap-2 border-b border-[var(--chef-border)] px-4 py-2.5">
        <h2 className="text-xs font-semibold uppercase tracking-[0.18em] text-[var(--chef-muted)]">
          Timeline
        </h2>
        <span className="font-mono text-xs text-[var(--chef-muted)]">{formatTime(contentEnd, 2)}</span>
        <div className="ml-auto flex items-center gap-1">
          <button
            type="button"
            onClick={() => setZoom((z) => Math.max(1, z / 1.5))}
            disabled={zoom <= 1}
            className="rounded p-1 text-[var(--chef-muted)] hover:bg-[var(--chef-hover)] disabled:opacity-30"
            title="Zoom out"
          >
            <Minus className="size-3.5" />
          </button>
          <button
            type="button"
            onClick={() => setZoom((z) => Math.min(40, z * 1.5))}
            className="rounded p-1 text-[var(--chef-muted)] hover:bg-[var(--chef-hover)]"
            title="Zoom in"
          >
            <Plus className="size-3.5" />
          </button>
        </div>
      </header>

      {/* The file strip: drag a chip onto a lane. */}
      <div className="flex flex-wrap items-center gap-1.5 border-b border-[var(--chef-border)] px-4 py-2">
        {files.length === 0 ? (
          <span className="text-[11px] text-[var(--chef-muted)]">
            Open some audio files first — they appear here to drag onto the lanes.
          </span>
        ) : (
          files.map((file) => (
            <span
              key={file.id}
              draggable
              onDragStart={(event) => {
                event.dataTransfer.setData(DRAG_MIME, file.id)
                event.dataTransfer.effectAllowed = 'copy'
              }}
              title={`Drag ${file.filename} onto a lane`}
              className="flex cursor-grab items-center gap-1.5 rounded-md border bg-[var(--chef-surface)] px-2 py-1 text-[11px] active:cursor-grabbing"
              style={{ borderColor: file.color }}
            >
              <span className="size-2 rounded-full" style={{ backgroundColor: file.color }} />
              <span className="max-w-[140px] truncate">{file.filename}</span>
              <span className="font-mono text-[10px] text-[var(--chef-muted)]">
                {formatTime(file.duration, 1)}
              </span>
            </span>
          ))
        )}
      </div>

      <div ref={scroller} className="overflow-x-auto">
        <div style={{ width: innerWidth }} className="relative select-none">
          {/* Ruler: click to place the cursor. */}
          <div
            className="relative border-b border-[var(--chef-border)] bg-[var(--chef-inset)]"
            style={{ height: RULER_H }}
            onPointerDown={(event) => {
              setCursor(timeAt(event.clientX))
              root.current?.focus()
            }}
          >
            {ticks.map((t) => (
              <span
                key={t}
                className="absolute top-0 h-full border-l border-[var(--chef-border)] pl-1 font-mono text-[11px] leading-[24px] text-[var(--chef-text)]"
                style={{ left: t * pps }}
              >
                {formatTime(t, spacing.decimals)}
              </span>
            ))}
          </div>

          {/* Lanes. */}
          <div
            className="relative"
            style={{ height: lanes * LANE_H }}
            onDragOver={(event) => {
              if (event.dataTransfer.types.includes(DRAG_MIME)) event.preventDefault()
            }}
            onDrop={onDrop}
            onPointerDown={(event) => {
              setCursor(timeAt(event.clientX))
              setSelected(null)
              root.current?.focus()
            }}
          >
            {Array.from({ length: lanes }, (_, lane) => (
              <div
                key={lane}
                className={`absolute inset-x-0 border-b border-[var(--chef-border)] ${
                  lane % 2 ? 'bg-[var(--chef-inset)]/40' : ''
                }`}
                style={{ top: lane * LANE_H, height: LANE_H }}
              />
            ))}

            {shown.map((clip) => {
              const file = byId.get(clip.source)
              if (!file) return null
              return (
                <Block
                  key={clip.uid}
                  clip={clip}
                  file={file}
                  pps={pps}
                  selected={clip.uid === selected || clip.uid === preview?.clip.uid}
                  onPointerDown={(event) => onBlockDown(event, clip)}
                  onPointerMove={onBlockMove}
                  onPointerUp={onBlockUp}
                />
              )
            })}

            {cursor !== null && (
              <div
                className="pointer-events-none absolute inset-y-0 w-px bg-[var(--chef-text)]"
                style={{ left: cursor * pps }}
              />
            )}
          </div>
        </div>
      </div>

      {/* The selected block's controls. */}
      <footer className="flex flex-wrap items-center gap-3 border-t border-[var(--chef-border)] px-4 py-2 text-xs">
        {selectedClip ? (
          <>
            <span className="max-w-[160px] truncate font-medium">
              {byId.get(selectedClip.source)?.filename}
            </span>
            <label className="flex items-center gap-2 text-[var(--chef-muted)]">
              Volume
              <input
                type="range"
                min={-24}
                max={12}
                step={0.5}
                value={selectedClip.gainDb}
                onChange={(event) => setGain(Number(event.target.value))}
              />
              <span className="w-14 font-mono text-[var(--chef-text)]">
                {selectedClip.gainDb > 0 ? '+' : ''}
                {selectedClip.gainDb.toFixed(1)} dB
              </span>
            </label>
            <button
              type="button"
              onClick={splitSelected}
              disabled={cursor === null}
              title="Split at the cursor (S)"
              className="flex items-center gap-1 rounded px-2 py-1 hover:bg-[var(--chef-hover)] disabled:opacity-30"
            >
              <Scissors className="size-3.5" /> Split
            </button>
            <button
              type="button"
              onClick={deleteSelected}
              title="Remove the block (Delete)"
              className="flex items-center gap-1 rounded px-2 py-1 text-rose-600 hover:bg-[var(--chef-hover)]"
            >
              <Trash2 className="size-3.5" /> Remove
            </button>
          </>
        ) : (
          <span className="text-[var(--chef-muted)]">
            Drag a file onto a lane. Drag blocks to move them, their edges to trim, Alt-drag
            to copy. Click to place the cursor, then S to split.
          </span>
        )}
      </footer>
    </section>
  )
}

interface BlockProps {
  clip: ArrangeClip
  file: Source
  pps: number
  selected: boolean
  onPointerDown: (event: React.PointerEvent<HTMLDivElement>) => void
  onPointerMove: (event: React.PointerEvent<HTMLDivElement>) => void
  onPointerUp: () => void
}

function Block({ clip, file, pps, selected, onPointerDown, onPointerMove, onPointerUp }: BlockProps) {
  const peaks = usePeaks(file.url)
  const len = spanOf(clip, file.duration)
  const width = Math.max(4, len * pps)
  const shifted = file.recipe.some((step) => !step.bypass && TIME_SHIFTING_OPS.has(step.op))
  return (
    <div
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      className={`absolute cursor-grab overflow-hidden rounded-md border-2 active:cursor-grabbing ${
        selected ? 'ring-2 ring-[var(--chef-text)]/40' : ''
      }`}
      style={{
        left: clip.start * pps,
        top: clip.lane * LANE_H + 3,
        width,
        height: LANE_H - 6,
        borderColor: file.color,
        backgroundColor: `${file.color}26`,
      }}
      title={`${file.filename} · ${formatTime(clip.clipStart, 2)}–${formatTime(clip.clipStart + len, 2)}`}
    >
      {peaks && (
        <Outline peaks={peaks} from={clip.clipStart} seconds={len} width={width} color={file.color} />
      )}
      <span className="pointer-events-none absolute left-1.5 top-0.5 flex max-w-[calc(100%-12px)] items-center gap-1 truncate text-[10px] font-medium">
        {shifted && (
          <AlertTriangle
            className="size-3 shrink-0 text-amber-500"
            aria-label="This tab's recipe changes its length; the outline is the original file"
          />
        )}
        {file.filename}
        {clip.gainDb !== 0 && (
          <span className="font-mono text-[var(--chef-muted)]">
            {clip.gainDb > 0 ? '+' : ''}
            {clip.gainDb.toFixed(1)} dB
          </span>
        )}
      </span>
      {/* Trim handles: visible grips on both edges. */}
      <span className="absolute inset-y-0 left-0 w-1.5 cursor-ew-resize" style={{ backgroundColor: file.color }} />
      <span className="absolute inset-y-0 right-0 w-1.5 cursor-ew-resize" style={{ backgroundColor: file.color }} />
    </div>
  )
}

function Outline({
  peaks,
  from,
  seconds,
  width,
  color,
}: {
  peaks: Peaks
  from: number
  seconds: number
  width: number
  color: string
}) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const height = LANE_H - 10
  useEffect(() => {
    const element = canvas.current
    if (!element) return
    const ratio = window.devicePixelRatio || 1
    const w = Math.max(1, Math.round(width))
    element.width = w * ratio
    element.height = height * ratio
    const g = element.getContext('2d')
    if (!g) return
    g.scale(ratio, ratio)
    g.clearRect(0, 0, w, height)
    g.fillStyle = `${color}b0`
    const mid = height / 2
    for (let x = 0; x < w; x++) {
      const a = Math.floor((from + (x / w) * seconds) * peaks.rate)
      const b = Math.max(a + 1, Math.floor((from + ((x + 1) / w) * seconds) * peaks.rate))
      let lo = 0
      let hi = 0
      for (let i = a; i < b && i < peaks.min.length; i++) {
        if (peaks.min[i] < lo) lo = peaks.min[i]
        if (peaks.max[i] > hi) hi = peaks.max[i]
      }
      // True amplitude, like the waveform panels: full height = full scale.
      g.fillRect(x, mid - hi * mid, 1, Math.max(1, (hi - lo) * mid))
    }
  }, [peaks, from, seconds, width, color, height])
  return (
    <canvas
      ref={canvas}
      className="pointer-events-none absolute left-0 top-[2px]"
      style={{ width, height }}
    />
  )
}

export const Timeline = memo(TimelineImpl)
