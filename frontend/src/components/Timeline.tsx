/**
 * The Arrange tab's timeline: a multitrack arranger, like a DAW.
 *
 *   - drag a file chip from the strip above onto a track     -> a new block
 *   - drag a block's body                                   -> move it (in time and track)
 *   - drag a block's left / right edge                      -> trim it
 *   - drag the round handle at a block's top corner          -> fade it in / out
 *   - Alt-drag a block                                      -> a copy
 *   - click empty space (or the ruler)                      -> put the cursor there
 *   - S, or the Split button                                -> cut the selected block at the cursor
 *   - Delete / Backspace                                    -> remove the selected block
 *
 * Each track has a header: mute (M), solo (S), volume and pan, and an automation lane
 * (the curve icon) where volume or pan is drawn over time -- click to add a point, drag
 * it, double-click it to remove it.  A curve with points overrides its knob.  Two blocks
 * that overlap on ONE track crossfade; on different tracks they play together.  The mix
 * is stereo (backend dsp/arrange.py).
 *
 * Moves and trims snap to the edges of other blocks (within SNAP_PX) and otherwise to a
 * 0.05 s grid.  A drag is previewed locally and committed on release, so one drag is one
 * change and one bake.  Each block plays the file tab's PROCESSED output but draws the
 * raw file's outline (peaks.ts); when that tab's recipe changes the length, the block
 * shows a warning, as recipe cards do.
 */

import { AlertTriangle, Minus, Plus, Scissors, Spline, Trash2 } from 'lucide-react'
import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'

import { usePeaks, type Peaks } from '../peaks'
import { TIME_SHIFTING_OPS } from '../regions'
import { formatTime, rulerSpacing } from '../time'
import type { ArrangeClip, ArrangeTrack, EnvPoint, Source } from '../types'

interface Props {
  clips: ArrangeClip[]
  tracks: ArrangeTrack[]
  /** The file tabs a block may play. */
  files: Source[]
  onChange: (clips: ArrangeClip[]) => void
  onTracksChange: (tracks: ArrangeTrack[]) => void
}

const LANE_H = 64
const ENV_H = 72
const HEADER_W = 176
const RULER_H = 24
const EDGE_PX = 7          // grab zone for trimming, each side
const SNAP_PX = 8
const GRID_S = 0.05
const MIN_LEN_S = 0.05
const DRAG_MIME = 'application/x-audio-chef-source'

const DEFAULT_TRACK: ArrangeTrack = {
  volumeDb: 0,
  pan: 0,
  mute: false,
  solo: false,
  volumeEnv: [],
  panEnv: [],
}

type EnvKind = 'volume' | 'pan'

/** Value range of each automation lane, top to bottom. */
const ENV_RANGE: Record<EnvKind, { top: number; bottom: number }> = {
  volume: { top: 12, bottom: -36 },     // dB
  pan: { top: -1, bottom: 1 },          // left at the top, right at the bottom
}

type Mode = 'move' | 'left' | 'right' | 'fadeIn' | 'fadeOut'

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

interface EnvDrag {
  track: number
  kind: EnvKind
  points: EnvPoint[]
  index: number
}

function uid(): string {
  return `clip-${crypto.randomUUID()}`
}

/** Seconds of source the block plays. */
function spanOf(clip: ArrangeClip, duration: number): number {
  const end = clip.clipEnd > 0 ? clip.clipEnd : duration
  return Math.max(0, end - clip.clipStart)
}

function envKey(kind: EnvKind): 'volumeEnv' | 'panEnv' {
  return kind === 'volume' ? 'volumeEnv' : 'panEnv'
}

function yOf(kind: EnvKind, value: number): number {
  const { top, bottom } = ENV_RANGE[kind]
  const f = (value - top) / (bottom - top)
  return 6 + Math.min(1, Math.max(0, f)) * (ENV_H - 12)
}

function valueAt(kind: EnvKind, y: number): number {
  const { top, bottom } = ENV_RANGE[kind]
  const f = Math.min(1, Math.max(0, (y - 6) / (ENV_H - 12)))
  const value = top + f * (bottom - top)
  return kind === 'volume' ? Math.round(value * 2) / 2 : Math.round(value * 100) / 100
}

function formatPan(pan: number): string {
  if (Math.abs(pan) < 0.005) return 'C'
  return `${pan < 0 ? 'L' : 'R'}${Math.round(Math.abs(pan) * 100)}`
}

function formatDb(db: number): string {
  return `${db > 0 ? '+' : ''}${db.toFixed(1)} dB`
}

function TimelineImpl({ clips, tracks, files, onChange, onTracksChange }: Props) {
  const root = useRef<HTMLDivElement>(null)
  const scroller = useRef<HTMLDivElement>(null)
  const lanesEl = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(600)
  const [zoom, setZoom] = useState(1)
  const [selected, setSelected] = useState<string | null>(null)
  const [cursor, setCursor] = useState<number | null>(null)
  // The block being dragged, previewed until release; `copy` = Alt-drag (a new block).
  const [preview, setPreview] = useState<{ clip: ArrangeClip; copy: boolean } | null>(null)
  const drag = useRef<Drag | null>(null)
  // Which tracks show an automation lane, and for what.
  const [envOpen, setEnvOpen] = useState<Record<number, EnvKind>>({})
  const [envDrag, setEnvDrag] = useState<EnvDrag | null>(null)

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
  // Always one empty track below the last used one, to drop onto.
  const trackCount = Math.max(3, tracks.length, ...shown.map((clip) => clip.lane + 2))
  const spacing = rulerSpacing(innerWidth / pps, innerWidth)

  // --- tracks --------------------------------------------------------------------------
  const trackAt = (k: number): ArrangeTrack => tracks[k] ?? DEFAULT_TRACK
  const soloing = Array.from({ length: trackCount }, (_, k) => trackAt(k).solo).some(Boolean)
  const audible = (k: number) => !trackAt(k).mute && (!soloing || trackAt(k).solo)

  const updateTrack = (k: number, patch: Partial<ArrangeTrack>) => {
    const next = Array.from({ length: Math.max(tracks.length, k + 1) }, (_, i) => trackAt(i))
    next[k] = { ...next[k], ...patch }
    onTracksChange(next)
  }

  // Row geometry: a track is its lane, plus its automation lane when that is open.
  const rowTops = useMemo(() => {
    const tops: number[] = []
    let y = 0
    for (let k = 0; k < trackCount; k++) {
      tops.push(y)
      y += LANE_H + (envOpen[k] ? ENV_H : 0)
    }
    tops.push(y)
    return tops
  }, [trackCount, envOpen])
  const lanesHeight = rowTops[trackCount]

  /** The track under a y offset inside the lanes area. */
  const trackAtY = (y: number): number => {
    for (let k = 0; k < trackCount; k++) if (y < rowTops[k + 1]) return k
    return trackCount - 1
  }
  const laneY = (clientY: number) => clientY - (lanesEl.current?.getBoundingClientRect().top ?? 0)

  // --- fades: the drawn ones, lengthened by same-track overlaps (as the backend does) ---
  const effectiveFades = useMemo(() => {
    const out = new Map<string, { fadeIn: number; fadeOut: number }>()
    for (const clip of shown) out.set(clip.uid, { fadeIn: clip.fadeIn, fadeOut: clip.fadeOut })
    const endOf = (clip: ArrangeClip) =>
      clip.start + spanOf(clip, byId.get(clip.source)?.duration ?? 0)
    const byTrack = new Map<number, ArrangeClip[]>()
    for (const clip of shown) byTrack.set(clip.lane, [...(byTrack.get(clip.lane) ?? []), clip])
    for (const members of byTrack.values()) {
      members.sort((a, b) => a.start - b.start)
      let latest: ArrangeClip | null = null
      for (const clip of members) {
        const end = endOf(clip)
        if (latest) {
          const latestEnd = endOf(latest)
          const overlap = Math.min(latestEnd, end) - clip.start
          if (overlap > 0) {
            const mine = out.get(clip.uid)!
            mine.fadeIn = Math.max(mine.fadeIn, overlap)
            if (latestEnd <= end) {
              const theirs = out.get(latest.uid)!
              theirs.fadeOut = Math.max(theirs.fadeOut, overlap)
            }
          }
        }
        if (!latest || end > endOf(latest)) latest = clip
      }
    }
    return out
  }, [shown, byId])

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

  // --- blocks: move / trim / fade / copy -------------------------------------------------
  const onBlockDown = (event: React.PointerEvent<HTMLDivElement>, clip: ArrangeClip) => {
    event.stopPropagation()
    root.current?.focus()
    setSelected(clip.uid)
    const box = event.currentTarget.getBoundingClientRect()
    const x = event.clientX - box.left
    const handle = (event.target as HTMLElement).dataset.handle as Mode | undefined
    const mode: Mode =
      handle ?? (x < EDGE_PX ? 'left' : x > box.width - EDGE_PX ? 'right' : 'move')
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
      const lane = trackAtY(laneY(event.clientY))
      next = { ...o, uid: d.uid, lane, start: snap(o.start + dx, o.uid, len) }
    } else if (d.mode === 'fadeIn') {
      const fadeIn = Math.min(Math.max(0, o.fadeIn + dx), len - o.fadeOut)
      next = { ...o, fadeIn: Math.round(fadeIn * 100) / 100 }
    } else if (d.mode === 'fadeOut') {
      const fadeOut = Math.min(Math.max(0, o.fadeOut - dx), len - o.fadeIn)
      next = { ...o, fadeOut: Math.round(fadeOut * 100) / 100 }
    } else if (d.mode === 'left') {
      // The left edge moves in time AND in the source together: the audio under the
      // block stays where it was, only less of it plays.
      const start = snap(o.start + dx, o.uid)
      // Never before the source's first sample, never shorter than MIN_LEN_S.
      const shift = Math.min(Math.max(start - o.start, -o.clipStart), len - MIN_LEN_S)
      next = {
        ...o,
        start: o.start + shift,
        clipStart: o.clipStart + shift,
        fadeIn: Math.min(o.fadeIn, len - shift),
      }
    } else {
      const end = snap(o.start + len + dx, o.uid) - o.start + o.clipStart
      const clamped = Math.min(Math.max(end, o.clipStart + MIN_LEN_S), duration)
      next = {
        ...o,
        clipEnd: clamped >= duration - 1e-3 ? 0 : clamped,
        fadeOut: Math.min(o.fadeOut, clamped - o.clipStart),
      }
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
    const clip: ArrangeClip = {
      uid: uid(),
      source,
      start: snap(timeAt(event.clientX), ''),
      clipStart: 0,
      clipEnd: 0,
      gainDb: 0,
      lane: trackAtY(laneY(event.clientY)),
      fadeIn: 0,
      fadeOut: 0,
    }
    onChange([...clips, clip])
    setSelected(clip.uid)
  }

  // --- automation lanes ------------------------------------------------------------------
  const envPoints = (k: number, kind: EnvKind): EnvPoint[] =>
    envDrag && envDrag.track === k && envDrag.kind === kind
      ? envDrag.points
      : trackAt(k)[envKey(kind)]

  const onEnvDown = (event: React.PointerEvent<SVGSVGElement>, k: number, kind: EnvKind) => {
    event.stopPropagation()
    root.current?.focus()
    const box = event.currentTarget.getBoundingClientRect()
    const t = Math.max(0, (event.clientX - box.left) / pps)
    const v = valueAt(kind, event.clientY - box.top)
    const points = [...trackAt(k)[envKey(kind)]]
    const hit = (event.target as SVGElement).dataset.index
    let index: number
    if (hit !== undefined) {
      index = Number(hit)
    } else {
      // A new point, slotted in by time so the curve stays in order.
      index = points.findIndex((p) => p.t > t)
      if (index < 0) index = points.length
      points.splice(index, 0, { t: Math.round(t * 100) / 100, v })
    }
    setEnvDrag({ track: k, kind, points, index })
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  const onEnvMove = (event: React.PointerEvent<SVGSVGElement>) => {
    if (!envDrag) return
    const box = event.currentTarget.getBoundingClientRect()
    const { points, index, kind } = envDrag
    // A point may not pass its neighbours, so the curve never folds back on itself.
    const lo = index > 0 ? points[index - 1].t : 0
    const hi = index < points.length - 1 ? points[index + 1].t : Infinity
    const t = Math.min(hi, Math.max(lo, (event.clientX - box.left) / pps))
    const next = [...points]
    next[index] = { t: Math.round(t * 100) / 100, v: valueAt(kind, event.clientY - box.top) }
    setEnvDrag({ ...envDrag, points: next })
  }

  const onEnvUp = () => {
    if (!envDrag) return
    updateTrack(envDrag.track, { [envKey(envDrag.kind)]: envDrag.points })
    setEnvDrag(null)
  }

  const removeEnvPoint = (k: number, kind: EnvKind, index: number) => {
    setEnvDrag(null)
    updateTrack(k, { [envKey(kind)]: trackAt(k)[envKey(kind)].filter((_, i) => i !== index) })
  }

  const toggleEnv = (k: number) =>
    setEnvOpen((open) => {
      const next = { ...open }
      if (next[k]) delete next[k]
      else next[k] = trackAt(k).panEnv.length && !trackAt(k).volumeEnv.length ? 'pan' : 'volume'
      return next
    })

  // --- selection tools ---------------------------------------------------------------
  const selectedClip = clips.find((clip) => clip.uid === selected) ?? null

  const splitSelected = () => {
    if (!selectedClip || cursor === null) return
    const len = spanOf(selectedClip, durationOf(selectedClip))
    const at = cursor - selectedClip.start
    if (at <= MIN_LEN_S || at >= len - MIN_LEN_S) return
    const cut = selectedClip.clipStart + at
    // The fade-in stays with the left half, the fade-out with the right.
    const left = { ...selectedClip, clipEnd: cut, fadeIn: Math.min(selectedClip.fadeIn, at), fadeOut: 0 }
    const right = {
      ...selectedClip,
      uid: uid(),
      start: cursor,
      clipStart: cut,
      fadeIn: 0,
      fadeOut: Math.min(selectedClip.fadeOut, len - at),
    }
    onChange(clips.flatMap((clip) => (clip.uid === selectedClip.uid ? [left, right] : [clip])))
    setSelected(right.uid)
  }

  const deleteSelected = () => {
    if (!selectedClip) return
    onChange(clips.filter((clip) => clip.uid !== selectedClip.uid))
    setSelected(null)
  }

  const patchSelected = (patch: Partial<ArrangeClip>) => {
    if (!selectedClip) return
    onChange(clips.map((clip) => (clip.uid === selectedClip.uid ? { ...clip, ...patch } : clip)))
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

  const selectedLen = selectedClip ? spanOf(selectedClip, durationOf(selectedClip)) : 0

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
        <span className="text-[11px] text-[var(--chef-muted)]">· stereo mix</span>
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

      {/* The file strip: drag a chip onto a track. */}
      <div className="flex flex-wrap items-center gap-1.5 border-b border-[var(--chef-border)] px-4 py-2">
        {files.length === 0 ? (
          <span className="text-[11px] text-[var(--chef-muted)]">
            Open some audio files first — they appear here to drag onto the tracks.
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
              title={`Drag ${file.filename} onto a track`}
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

      <div className="flex">
        {/* Track headers: fixed while the lanes scroll. */}
        <div className="shrink-0 border-r border-[var(--chef-border)]" style={{ width: HEADER_W }}>
          <div
            className="border-b border-[var(--chef-border)] bg-[var(--chef-inset)]"
            style={{ height: RULER_H }}
          />
          {Array.from({ length: trackCount }, (_, k) => (
            <TrackHeader
              key={k}
              index={k}
              track={trackAt(k)}
              audible={audible(k)}
              envKind={envOpen[k] ?? null}
              onPatch={(patch) => updateTrack(k, patch)}
              onToggleEnv={() => toggleEnv(k)}
              onEnvKind={(kind) => setEnvOpen((open) => ({ ...open, [k]: kind }))}
            />
          ))}
        </div>

        <div ref={scroller} className="min-w-0 flex-1 overflow-x-auto">
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

            {/* Tracks. */}
            <div
              ref={lanesEl}
              className="relative"
              style={{ height: lanesHeight }}
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
              {Array.from({ length: trackCount }, (_, k) => (
                <div
                  key={k}
                  className={`absolute inset-x-0 border-b border-[var(--chef-border)] ${
                    k % 2 ? 'bg-[var(--chef-inset)]/40' : ''
                  }`}
                  style={{ top: rowTops[k], height: LANE_H }}
                />
              ))}

              {shown.map((clip) => {
                const file = byId.get(clip.source)
                if (!file) return null
                const fades = effectiveFades.get(clip.uid) ?? clip
                return (
                  <Block
                    key={clip.uid}
                    clip={clip}
                    file={file}
                    pps={pps}
                    top={rowTops[clip.lane] ?? 0}
                    fadeIn={fades.fadeIn}
                    fadeOut={fades.fadeOut}
                    dimmed={!audible(clip.lane)}
                    selected={clip.uid === selected || clip.uid === preview?.clip.uid}
                    onPointerDown={(event) => onBlockDown(event, clip)}
                    onPointerMove={onBlockMove}
                    onPointerUp={onBlockUp}
                  />
                )
              })}

              {/* Automation lanes, under their tracks. */}
              {Object.entries(envOpen).map(([key, kind]) => {
                const k = Number(key)
                if (k >= trackCount) return null
                return (
                  <EnvelopeLane
                    key={k}
                    kind={kind}
                    top={rowTops[k] + LANE_H}
                    width={innerWidth}
                    pps={pps}
                    points={envPoints(k, kind)}
                    knob={kind === 'volume' ? trackAt(k).volumeDb : trackAt(k).pan}
                    onPointerDown={(event) => onEnvDown(event, k, kind)}
                    onPointerMove={onEnvMove}
                    onPointerUp={onEnvUp}
                    onRemove={(index) => removeEnvPoint(k, kind, index)}
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
                onChange={(event) => patchSelected({ gainDb: Number(event.target.value) })}
              />
              <span className="w-14 font-mono text-[var(--chef-text)]">
                {formatDb(selectedClip.gainDb)}
              </span>
            </label>
            {(['fadeIn', 'fadeOut'] as const).map((key) => (
              <label key={key} className="flex items-center gap-1.5 text-[var(--chef-muted)]">
                {key === 'fadeIn' ? 'Fade in' : 'Fade out'}
                <input
                  type="number"
                  min={0}
                  max={selectedLen}
                  step={0.05}
                  value={selectedClip[key]}
                  onChange={(event) => {
                    const other = key === 'fadeIn' ? selectedClip.fadeOut : selectedClip.fadeIn
                    const value = Math.min(Math.max(0, Number(event.target.value) || 0), selectedLen - other)
                    patchSelected({ [key]: Math.round(value * 100) / 100 })
                  }}
                  className="w-16 rounded border border-[var(--chef-border)] bg-[var(--chef-surface)] px-1.5 py-0.5 font-mono text-[var(--chef-text)]"
                />
                s
              </label>
            ))}
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
            Drag a file onto a track. Drag blocks to move them, their edges to trim, the round
            corner handles to fade; overlap two blocks on one track to crossfade. Alt-drag to
            copy. Click to place the cursor, then S to split.
          </span>
        )}
      </footer>
    </section>
  )
}

// --- track header ------------------------------------------------------------------------
interface TrackHeaderProps {
  index: number
  track: ArrangeTrack
  audible: boolean
  envKind: EnvKind | null
  onPatch: (patch: Partial<ArrangeTrack>) => void
  onToggleEnv: () => void
  onEnvKind: (kind: EnvKind) => void
}

function TrackHeader({ index, track, audible, envKind, onPatch, onToggleEnv, onEnvKind }: TrackHeaderProps) {
  const volumeAuto = track.volumeEnv.length > 0
  const panAuto = track.panEnv.length > 0
  const toggle = (on: boolean, color: string) =>
    `w-5 rounded text-[10px] font-bold leading-4 ${
      on ? `${color} text-white` : 'bg-[var(--chef-inset)] text-[var(--chef-muted)] hover:text-[var(--chef-text)]'
    }`
  return (
    <>
      <div
        className={`flex flex-col justify-center gap-1 border-b border-[var(--chef-border)] px-2 text-[11px] ${
          index % 2 ? 'bg-[var(--chef-inset)]/40' : ''
        } ${audible ? '' : 'opacity-60'}`}
        style={{ height: LANE_H }}
      >
        <div className="flex items-center gap-1">
          <span className="mr-auto truncate font-medium">Track {index + 1}</span>
          <button
            type="button"
            onClick={() => onPatch({ mute: !track.mute })}
            className={toggle(track.mute, 'bg-amber-500')}
            title="Mute"
          >
            M
          </button>
          <button
            type="button"
            onClick={() => onPatch({ solo: !track.solo })}
            className={toggle(track.solo, 'bg-[var(--chef-accent-strong)]')}
            title="Solo: only soloed tracks play"
          >
            S
          </button>
          <button
            type="button"
            onClick={onToggleEnv}
            className={`rounded p-0.5 ${
              envKind || volumeAuto || panAuto
                ? 'text-[var(--chef-accent-strong)]'
                : 'text-[var(--chef-muted)] hover:text-[var(--chef-text)]'
            }`}
            title="Automation: draw volume or pan over time"
          >
            <Spline className="size-3.5" />
          </button>
        </div>
        <label className="flex items-center gap-1 text-[var(--chef-muted)]" title="Volume (double-click: 0 dB)">
          <span className="w-6">Vol</span>
          <input
            type="range"
            min={-36}
            max={12}
            step={0.5}
            value={track.volumeDb}
            disabled={volumeAuto}
            onChange={(event) => onPatch({ volumeDb: Number(event.target.value) })}
            onDoubleClick={() => onPatch({ volumeDb: 0 })}
            className="h-3 min-w-0 flex-1 disabled:opacity-40"
          />
          <span className="w-11 text-right font-mono text-[10px] text-[var(--chef-text)]">
            {volumeAuto ? 'auto' : `${track.volumeDb > 0 ? '+' : ''}${track.volumeDb.toFixed(1)}`}
          </span>
        </label>
        <label className="flex items-center gap-1 text-[var(--chef-muted)]" title="Pan (double-click: centre)">
          <span className="w-6">Pan</span>
          <input
            type="range"
            min={-1}
            max={1}
            step={0.05}
            value={track.pan}
            disabled={panAuto}
            onChange={(event) => onPatch({ pan: Number(event.target.value) })}
            onDoubleClick={() => onPatch({ pan: 0 })}
            className="h-3 min-w-0 flex-1 disabled:opacity-40"
          />
          <span className="w-11 text-right font-mono text-[10px] text-[var(--chef-text)]">
            {panAuto ? 'auto' : formatPan(track.pan)}
          </span>
        </label>
      </div>
      {envKind && (
        <div
          className="flex flex-col justify-center gap-1.5 border-b border-[var(--chef-border)] bg-[var(--chef-inset)] px-2 text-[11px]"
          style={{ height: ENV_H }}
        >
          <div className="flex rounded-md border border-[var(--chef-border)] bg-[var(--chef-surface)] p-0.5">
            {(['volume', 'pan'] as const).map((kind) => (
              <button
                key={kind}
                type="button"
                onClick={() => onEnvKind(kind)}
                className={`flex-1 rounded px-1.5 py-0.5 ${
                  envKind === kind
                    ? 'bg-[var(--chef-accent-strong)] font-medium text-white'
                    : 'text-[var(--chef-muted)] hover:text-[var(--chef-text)]'
                }`}
              >
                {kind === 'volume' ? 'Volume' : 'Pan'}
                {(kind === 'volume' ? volumeAuto : panAuto) ? ' •' : ''}
              </button>
            ))}
          </div>
          <button
            type="button"
            disabled={!(envKind === 'volume' ? volumeAuto : panAuto)}
            onClick={() => onPatch({ [envKey(envKind)]: [] })}
            className="self-start rounded px-1.5 py-0.5 text-[var(--chef-muted)] hover:bg-[var(--chef-hover)] hover:text-[var(--chef-text)] disabled:opacity-30"
            title="Remove every point: the knob takes over again"
          >
            Clear curve
          </button>
        </div>
      )}
    </>
  )
}

// --- automation lane -----------------------------------------------------------------------
interface EnvelopeLaneProps {
  kind: EnvKind
  top: number
  width: number
  pps: number
  points: EnvPoint[]
  /** The knob's value: drawn dashed when the curve has no points. */
  knob: number
  onPointerDown: (event: React.PointerEvent<SVGSVGElement>) => void
  onPointerMove: (event: React.PointerEvent<SVGSVGElement>) => void
  onPointerUp: () => void
  onRemove: (index: number) => void
}

function EnvelopeLane({
  kind, top, width, pps, points, knob, onPointerDown, onPointerMove, onPointerUp, onRemove,
}: EnvelopeLaneProps) {
  const { top: topValue, bottom: bottomValue } = ENV_RANGE[kind]
  const label = (v: number) => (kind === 'volume' ? formatDb(v) : formatPan(v))
  const line = points.length
    ? [
        `M0,${yOf(kind, points[0].v)}`,
        ...points.map((p) => `L${p.t * pps},${yOf(kind, p.v)}`),
        `L${width},${yOf(kind, points[points.length - 1].v)}`,
      ].join(' ')
    : ''
  return (
    <svg
      className="absolute left-0 cursor-crosshair border-b border-[var(--chef-border)] bg-[var(--chef-inset)]"
      style={{ top, width, height: ENV_H }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    >
      <text x={4} y={14} className="fill-[var(--chef-muted)] text-[9px]">
        {kind === 'volume' ? label(topValue) : 'L'}
      </text>
      <text x={4} y={ENV_H - 5} className="fill-[var(--chef-muted)] text-[9px]">
        {kind === 'volume' ? label(bottomValue) : 'R'}
      </text>
      {points.length ? (
        <>
          <path d={line} fill="none" stroke="var(--chef-accent-strong)" strokeWidth={1.5} />
          {points.map((p, i) => (
            <circle
              key={i}
              data-index={i}
              cx={p.t * pps}
              cy={yOf(kind, p.v)}
              r={4.5}
              className="cursor-grab fill-[var(--chef-panel)] stroke-[var(--chef-accent-strong)]"
              strokeWidth={1.5}
              onDoubleClick={(event) => {
                event.stopPropagation()
                onRemove(i)
              }}
            >
              <title>{`${formatTime(p.t, 2)} · ${label(p.v)} — double-click to remove`}</title>
            </circle>
          ))}
        </>
      ) : (
        <>
          <line
            x1={0}
            x2={width}
            y1={yOf(kind, knob)}
            y2={yOf(kind, knob)}
            stroke="var(--chef-muted)"
            strokeDasharray="4 4"
          />
          <text x={56} y={14} className="fill-[var(--chef-muted)] text-[10px]">
            Click to add {kind} points — the curve overrides the knob
          </text>
        </>
      )}
    </svg>
  )
}

// --- blocks ------------------------------------------------------------------------------
interface BlockProps {
  clip: ArrangeClip
  file: Source
  pps: number
  top: number
  /** Effective fades, seconds: the drawn ones or the crossfade, whichever is longer. */
  fadeIn: number
  fadeOut: number
  dimmed: boolean
  selected: boolean
  onPointerDown: (event: React.PointerEvent<HTMLDivElement>) => void
  onPointerMove: (event: React.PointerEvent<HTMLDivElement>) => void
  onPointerUp: () => void
}

function Block({
  clip, file, pps, top, fadeIn, fadeOut, dimmed, selected, onPointerDown, onPointerMove, onPointerUp,
}: BlockProps) {
  const peaks = usePeaks(file.url)
  const len = spanOf(clip, file.duration)
  const width = Math.max(4, len * pps)
  const height = LANE_H - 6
  const shifted = file.recipe.some((step) => !step.bypass && TIME_SHIFTING_OPS.has(step.op))
  const inPx = Math.min(width, fadeIn * pps)
  const outPx = Math.min(width, fadeOut * pps)
  return (
    <div
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      className={`absolute cursor-grab overflow-hidden rounded-md border-2 active:cursor-grabbing ${
        selected ? 'ring-2 ring-[var(--chef-text)]/40' : ''
      } ${dimmed ? 'opacity-40' : ''}`}
      style={{
        left: clip.start * pps,
        top: top + 3,
        width,
        height,
        borderColor: file.color,
        backgroundColor: `${file.color}26`,
      }}
      title={`${file.filename} · ${formatTime(clip.clipStart, 2)}–${formatTime(clip.clipStart + len, 2)}`}
    >
      {peaks && (
        <Outline peaks={peaks} from={clip.clipStart} seconds={len} width={width} color={file.color} />
      )}
      <FadeShade width={width} height={height} inPx={inPx} outPx={outPx} />
      <span className="pointer-events-none absolute left-1.5 top-0.5 flex max-w-[calc(100%-24px)] items-center gap-1 truncate text-[10px] font-medium">
        {shifted && (
          <AlertTriangle
            className="size-3 shrink-0 text-amber-500"
            aria-label="This tab's recipe changes its length; the outline is the original file"
          />
        )}
        {file.filename}
        {clip.gainDb !== 0 && (
          <span className="font-mono text-[var(--chef-muted)]">{formatDb(clip.gainDb)}</span>
        )}
      </span>
      {/* Fade handles: drag sideways to lengthen the fade. */}
      <span
        data-handle="fadeIn"
        title={`Fade in ${fadeIn.toFixed(2)} s — drag`}
        className="absolute top-0.5 size-2.5 -translate-x-1/2 cursor-ew-resize rounded-full border border-white shadow"
        style={{ left: Math.max(8, inPx), backgroundColor: file.color }}
      />
      <span
        data-handle="fadeOut"
        title={`Fade out ${fadeOut.toFixed(2)} s — drag`}
        className="absolute top-0.5 size-2.5 translate-x-1/2 cursor-ew-resize rounded-full border border-white shadow"
        style={{ right: Math.max(8, outPx), backgroundColor: file.color }}
      />
      {/* Trim handles: visible grips on both edges. */}
      <span className="absolute inset-y-0 left-0 w-1.5 cursor-ew-resize" style={{ backgroundColor: file.color }} />
      <span className="absolute inset-y-0 right-0 w-1.5 cursor-ew-resize" style={{ backgroundColor: file.color }} />
    </div>
  )
}

/** The equal-power fade curves, shading what each fade takes away. */
function FadeShade({ width, height, inPx, outPx }: { width: number; height: number; inPx: number; outPx: number }) {
  if (inPx < 1 && outPx < 1) return null
  const steps = 24
  const curve = (px: number, rising: boolean) => {
    const pts: string[] = []
    for (let i = 0; i <= steps; i++) {
      const f = i / steps
      const gain = rising ? Math.sin((f * Math.PI) / 2) : Math.cos((f * Math.PI) / 2)
      const x = rising ? f * px : width - px + f * px
      pts.push(`${x},${height * (1 - gain)}`)
    }
    return pts
  }
  return (
    <svg className="pointer-events-none absolute inset-0" width={width} height={height}>
      {inPx >= 1 && (
        <>
          <polygon points={['0,0', ...curve(inPx, true), `${inPx},0`].join(' ')} fill="rgba(0,0,0,0.22)" />
          <polyline points={curve(inPx, true).join(' ')} fill="none" stroke="rgba(0,0,0,0.5)" strokeWidth={1} />
        </>
      )}
      {outPx >= 1 && (
        <>
          <polygon
            points={[`${width - outPx},0`, ...curve(outPx, false), `${width},0`].join(' ')}
            fill="rgba(0,0,0,0.22)"
          />
          <polyline points={curve(outPx, false).join(' ')} fill="none" stroke="rgba(0,0,0,0.5)" strokeWidth={1} />
        </>
      )}
    </svg>
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
