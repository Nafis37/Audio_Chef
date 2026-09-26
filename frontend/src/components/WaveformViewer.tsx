/**
 * A Wavesurfer.js instance wrapped in a React component.
 *
 * Used twice in column 3: once for the uploaded Input and once for the baked Output.
 *
 * The instance is created ONCE, on mount, and kept for the life of the component; only
 * the audio inside it is swapped when `url` changes.  The earlier version keyed the
 * whole `WaveSurfer.create()` on the url, which meant every bake tore down the canvas
 * and the AudioContext and built a new one -- a full re-decode plus a whole-buffer peak
 * pass, 300 ms after every slider move.  Reusing the instance is what makes Auto-Bake
 * feel live, and it also means playback does not jump back to the start on a re-bake.
 *
 * Three plugins ride along, all of which ship inside wavesurfer.js itself:
 *
 *   timeline -- the seconds axis under both waveforms.  Without it there is no way to
 *               tell WHEN anything happens, which made the Mini Audio Editor's start/end
 *               numbers pure guesswork.  Its label step is chosen from the clip length
 *               and the panel width (time.ts rulerSpacing) so labels always land on
 *               round times, never crowd, and read as m:ss like every other clock here.
 *               The plugin reads its intervals at creation, so a new length or width
 *               re-registers it.
 *   hover    -- a line and a m:ss.cc label that follow the pointer, so you can see
 *               exactly where a click will jump to BEFORE clicking.
 *   regions  -- the draggable span on the Input, bound to the linked card's two time
 *               parameters.  See src/regions.ts for which operations have one.
 *
 * TRUE amplitude, not normalised.  wavesurfer's `normalize` scales every drawing to fill
 * its box, which made the Input and Output look the same height whatever the recipe did
 * to the level: an EQ boost, a compressor or a denoise were invisible.  Both panels now
 * draw on the same absolute scale (full height = full scale), so "louder" looks taller.
 *
 * Under the waveform sits the spectrogram (Spectrogram.tsx) -- the view in which EQ, noise
 * removal and the voice effects are actually visible.
 *
 * The parent gets an imperative handle (play / pause / seek) so the A/B switch in App can
 * hand playback from one viewer to the other at the same moment.
 */

import { Pause, Play } from 'lucide-react'
import { memo, useEffect, useImperativeHandle, useRef, useState, type Ref } from 'react'
import WaveSurfer from 'wavesurfer.js'
import RegionsPlugin, { type Region } from 'wavesurfer.js/dist/plugins/regions.esm.js'
import HoverPlugin from 'wavesurfer.js/dist/plugins/hover.esm.js'
import TimelinePlugin from 'wavesurfer.js/dist/plugins/timeline.esm.js'
import { formatTime, rulerSpacing, type RulerSpacing } from '../time'
import type { SpectrogramData, SpectrogramMarker } from '../types'
import { FrequencyBars, type FrequencyBarsHandle } from './FrequencyBars'
import { Spectrogram } from './Spectrogram'

/** What App can do to a viewer from outside -- enough for an A/B switch. */
export interface WaveformHandle {
  play: () => void
  pause: () => void
  isPlaying: () => boolean
  getCurrentTime: () => number
  getDuration: () => number
  setTime: (seconds: number) => void
}

/** The span to draw, in seconds.  Null means this viewer shows no region at all. */
export interface RegionSpec {
  start: number
  end: number
  color: string
}

interface Props {
  title: string
  url: string | null
  accent: string
  emptyHint: string
  busy?: boolean
  region?: RegionSpec | null
  onRegionChange?: (start: number, end: number) => void
  /** The hand-made spectrogram of the same audio; null while it is loading. */
  spectrogram?: SpectrogramData | null
  /** Lines to draw across the spectrogram (a Filter step's cutoff). */
  markers?: SpectrogramMarker[]
  /** Highlights the panel -- used by the A/B switch to show which one you are hearing. */
  highlighted?: boolean
  /** Fires when this viewer starts playing, so the parent can pause the other one. */
  onPlay?: () => void
  children?: React.ReactNode
  ref?: Ref<WaveformHandle>
}

/** Matches RecipeCard's slider throttle: a drag reports at most this often. */
const THROTTLE_MS = 80

/** Region edges within this many seconds of the props count as "already there". */
const EPSILON = 0.001

/** A timeline with round label steps for this clip length and panel width. */
function makeTimeline(spacing: RulerSpacing) {
  return TimelinePlugin.create({
    height: 22,
    timeInterval: spacing.tick,
    primaryLabelInterval: spacing.label,
    secondaryLabelInterval: spacing.label,   // no dim in-between labels: one clear row
    secondaryLabelOpacity: 0.35,             // the small ticks
    formatTimeCallback: (seconds) => formatTime(seconds, spacing.decimals),
    style: { color: '#1f2421', fontSize: '11px', fontFamily: 'ui-monospace, monospace' },
  })
}

function WaveformViewerImpl({
  title,
  url,
  accent,
  emptyHint,
  busy,
  region,
  onRegionChange,
  spectrogram = null,
  markers,
  highlighted = false,
  onPlay,
  children,
  ref,
}: Props) {
  const container = useRef<HTMLDivElement>(null)
  const wavesurfer = useRef<WaveSurfer | null>(null)
  const regions = useRef<RegionsPlugin | null>(null)
  const active = useRef<Region | null>(null)
  // What this component last told the parent.  When those exact numbers come back as
  // props we know it is our own drag echoing, and must NOT re-position the region --
  // doing so mid-drag makes the handle fight the pointer.  Same shape as the `sent` ref
  // in RecipeCard's useLiveValue.
  const emitted = useRef<{ start: number; end: number } | null>(null)
  const lastEmit = useRef(0)
  // The callback is read through a ref so the subscription below can be made once.
  const notify = useRef(onRegionChange)
  notify.current = onRegionChange
  const notifyPlay = useRef(onPlay)
  notifyPlay.current = onPlay

  useImperativeHandle(
    ref,
    () => ({
      play: () => void wavesurfer.current?.play(),
      pause: () => wavesurfer.current?.pause(),
      isPlaying: () => wavesurfer.current?.isPlaying() ?? false,
      getCurrentTime: () => wavesurfer.current?.getCurrentTime() ?? 0,
      getDuration: () => wavesurfer.current?.getDuration() ?? 0,
      setTime: (seconds: number) => wavesurfer.current?.setTime(seconds),
    }),
    [],
  )

  const [playing, setPlaying] = useState(false)
  const [duration, setDuration] = useState(0)
  // The running clock is written straight into the DOM: a state update per
  // 'timeupdate' would re-render this panel ~60 times a second during playback.
  const clock = useRef<HTMLSpanElement>(null)
  // The live frequency bars ride on the same clock, for the same reason.
  const bars = useRef<FrequencyBarsHandle>(null)
  const [ready, setReady] = useState(false)
  // Set when the BROWSER cannot decode the file.  The backend decodes with libsndfile,
  // which reads more formats than any browser does (AIFF, W64, CAF, AU), so a file can
  // process perfectly while refusing to preview.  Saying so beats a blank panel.
  const [undecodable, setUndecodable] = useState(false)

  // --- Create the instance once -----------------------------------------------------
  useEffect(() => {
    if (!container.current) return

    const regionsPlugin = RegionsPlugin.create()
    regions.current = regionsPlugin
    let timeline: ReturnType<typeof makeTimeline> | null = null
    let spacingKey = ''

    const instance = WaveSurfer.create({
      container: container.current,
      height: 96,
      waveColor: `${accent}80`,      // the waveform body, semi-transparent
      progressColor: accent,          // the part already played
      cursorColor: '#1f2421',   // --chef-text, literal: wavesurfer does not read vars
      cursorWidth: 1,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      normalize: false,         // true amplitude: see the header comment
      plugins: [
        regionsPlugin,
        HoverPlugin.create({
          lineColor: accent,
          lineWidth: 1,
          labelBackground: '#1f2421',
          labelColor: '#ffffff',
          labelSize: 11,
          formatTimeCallback: (seconds) => formatTime(seconds, 2),
        }),
      ],
    })
    wavesurfer.current = instance

    /** (Re)build the ruler when the clip length or panel width asks for another step. */
    const syncTimeline = () => {
      const length = instance.getDuration()
      if (!(length > 0)) return
      const spacing = rulerSpacing(length, instance.getWidth())
      const key = `${spacing.label}/${spacing.tick}`
      if (key === spacingKey && timeline) return
      spacingKey = key
      timeline?.destroy()
      timeline = instance.registerPlugin(makeTimeline(spacing))
    }
    instance.on('redraw', syncTimeline)

    const writeClock = (seconds: number) => {
      if (clock.current) clock.current.textContent = formatTime(seconds, 2)
      bars.current?.setTime(seconds, instance.getDuration())
    }
    instance.on('timeupdate', writeClock)

    instance.on('play', () => {
      setPlaying(true)
      notifyPlay.current?.()
    })
    instance.on('pause', () => setPlaying(false))
    instance.on('finish', () => setPlaying(false))
    instance.on('ready', () => {
      setDuration(instance.getDuration())
      setReady(true)
      syncTimeline()
      writeClock(instance.getCurrentTime())
    })

    /** Publishes a dragged edge upward, throttled so a drag is not one render per pixel. */
    const publish = (dragged: Region, force: boolean) => {
      const now = performance.now()
      if (!force && now - lastEmit.current < THROTTLE_MS) return
      lastEmit.current = now
      emitted.current = { start: dragged.start, end: dragged.end }
      notify.current?.(dragged.start, dragged.end)
    }

    // 'region-update' fires continuously while dragging, which is what makes the card's
    // numbers track the handle; 'region-updated' fires once at the end and is never
    // throttled away, so the final position lands even if the last move fell inside the
    // throttle window.  Auto-Bake's own 300 ms debounce still collapses the whole drag
    // into a single /process call.
    regionsPlugin.on('region-update', (dragged) => publish(dragged, false))
    regionsPlugin.on('region-updated', (dragged) => publish(dragged, true))

    return () => {
      instance.destroy()
      wavesurfer.current = null
      regions.current = null
      active.current = null
      setPlaying(false)
    }
    // `accent` is a constant per call site, so the instance never needs rebuilding.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // --- Load new audio into the existing instance ------------------------------------
  useEffect(() => {
    const instance = wavesurfer.current
    if (!instance) return

    setReady(false)
    setUndecodable(false)
    setDuration(0)          // otherwise the header keeps showing the previous length
    // The old region belongs to the old audio; its seconds mean nothing against the new.
    regions.current?.clearRegions()
    active.current = null
    emitted.current = null

    if (!url) {
      instance.empty()
      return
    }

    let cancelled = false
    // load() rejects when a newer load supersedes it (and on StrictMode's double
    // mount in dev); that is expected, so only real failures are worth surfacing.
    instance.load(url).catch((error: unknown) => {
      if (cancelled) return
      if (error instanceof DOMException && error.name === 'AbortError') return
      setUndecodable(true)
    })

    return () => {
      cancelled = true
    }
  }, [url])

  // --- Keep the region in step with the linked card ---------------------------------
  useEffect(() => {
    const plugin = regions.current
    // A region needs a duration to clamp against, so nothing happens before `ready`.
    if (!plugin || !ready) return

    if (!region) {
      plugin.clearRegions()
      active.current = null
      emitted.current = null
      return
    }

    // Our own drag coming back around: the region is already exactly there.
    const mine = emitted.current
    if (
      mine &&
      Math.abs(mine.start - region.start) < EPSILON &&
      Math.abs(mine.end - region.end) < EPSILON
    ) {
      return
    }

    if (active.current) {
      active.current.setOptions({
        start: region.start,
        end: region.end,
        color: region.color,
      })
      return
    }

    active.current = plugin.addRegion({
      start: region.start,
      end: region.end,
      color: region.color,
      drag: true,
      resize: true,
    })
  }, [region, ready])

  return (
    <section
      className={`rounded-lg border bg-[var(--chef-panel)] transition ${
        highlighted
          ? 'border-[var(--chef-accent-strong)] ring-2 ring-[var(--chef-accent)]/60'
          : 'border-[var(--chef-border)]'
      }`}
    >
      <header className="flex items-center gap-3 border-b border-[var(--chef-border)] px-4 py-2.5">
        <button
          type="button"
          disabled={!url || !ready}
          onClick={() => wavesurfer.current?.playPause()}
          className="rounded-full border border-[var(--chef-border)] p-1.5 transition hover:border-[var(--chef-accent-strong)] disabled:opacity-30"
          title={playing ? 'Pause' : 'Play'}
        >
          {playing ? <Pause className="size-4" /> : <Play className="size-4" />}
        </button>
        <h2 className="text-xs font-semibold uppercase tracking-[0.18em] text-[var(--chef-muted)]">
          {title}
        </h2>
        {duration > 0 && (
          <span className="font-mono text-xs text-[var(--chef-muted)]">
            <span ref={clock} className="text-[var(--chef-text)]">
              {formatTime(0, 2)}
            </span>
            {' / '}
            {formatTime(duration, 2)}
          </span>
        )}
        <div className="ml-auto flex items-center gap-2">{children}</div>
      </header>

      <div className="px-4 pb-3 pt-3">
        {url && <GraphHeading title="Waveform" hint="Loudness over time" first />}
        {/* The overlays cover the waveform only; the spectrogram below is drawn by the
            backend, so it still shows for formats this browser cannot preview. */}
        <div className="relative">
          {/* The container is always mounted and always laid out at full width -- the
              instance lives in it across bakes, and hiding it would hand wavesurfer a
              zero-width canvas to draw the next waveform into. */}
          <div className="min-h-24">
            <div ref={container} />
          </div>
          {!url && (
            <p className="absolute inset-0 flex items-center justify-center text-center text-xs text-[var(--chef-muted)]">
              {emptyHint}
            </p>
          )}
          {undecodable && (
            <p className="absolute inset-0 flex items-center justify-center px-6 text-center text-xs leading-relaxed text-[var(--chef-muted)]">
              This browser can’t preview this format — processing still works, and the Output
              below is audio you can play and export.
            </p>
          )}
          {busy && (
            <div className="absolute inset-0 flex items-center justify-center bg-[var(--chef-panel)]/70 text-xs text-[var(--chef-accent-strong)]">
              baking…
            </div>
          )}
        </div>
        {url && (
          <>
            <GraphHeading title="Spectrogram" hint="Pitch over time, brighter is louder" />
            <Spectrogram data={spectrogram} hint="drawing spectrogram…" markers={markers} />
            <GraphHeading title="Frequency bars" hint="What's sounding now, low to high" />
            <FrequencyBars ref={bars} data={spectrogram ?? null} playing={playing} />
          </>
        )}
      </div>
    </section>
  )
}

/**
 * The name above each graph in the panel, styled like the palette's category headings
 * (one step below the panel's h2), with a one-line reading hint on the right.
 */
function GraphHeading({ title, hint, first = false }: { title: string; hint: string; first?: boolean }) {
  return (
    <div className={`mb-1.5 flex items-baseline justify-between gap-3 ${first ? '' : 'mt-3'}`}>
      <h3 className="shrink-0 text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--chef-muted)]">
        {title}
      </h3>
      <span className="min-w-0 truncate text-[10px] text-[var(--chef-muted)]">{hint}</span>
    </div>
  )
}

export const WaveformViewer = memo(WaveformViewerImpl)
