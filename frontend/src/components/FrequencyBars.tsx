/**
 * The live frequency bars under a spectrogram: a classic analyser, frequency left to
 * right (log, same span as the spectrogram), level bottom to top.
 *
 * There is no Web Audio AnalyserNode here.  Each bar is read from the SAME uint8 grid the
 * spectrogram paints (backend/app/dsp/spectrogram.py, fixed -90..0 dBFS, log-spaced rows):
 *
 *     column = floor(t / duration * cols)                    -- where playback is
 *     band b = rows b*R/B .. (b+1)*R/B counted from the BOTTOM  -- low to high frequency
 *     value  = max over the band's rows, averaged over columns col-1..col+1  (no flicker)
 *
 * so a bar's height and colour mean exactly what a pixel's brightness means above it.
 *
 * Meter ballistics, per frame of dt seconds:
 *
 *     level = max(value, level - RELEASE * 255 * dt)    -- instant attack, linear release
 *     peak  = level if level >= peak, else hold PEAK_HOLD s, then fall at PEAK_FALL
 *
 * The clock arrives through setTime() (called from WaveformViewer's writeClock), never
 * through React state, and the requestAnimationFrame loop stops itself once every bar
 * has fallen -- paused means silence, and an idle page costs nothing.
 */

import { memo, useEffect, useImperativeHandle, useRef, type Ref } from 'react'
import type { SpectrogramData } from '../types'
import { COLORMAP } from './Spectrogram'

const BANDS = 48          // bars across the strip
const RELEASE = 1.2       // full scales per second a bar falls
const PEAK_HOLD = 0.5     // s a peak cap hangs before falling
const PEAK_FALL = 0.8     // full scales per second, once released
const LABEL_ROOM = 12     // px at the bottom kept for the frequency labels
const TICKS = [100, 1000, 10000]

const rgb = (value: number) => {
  const c = Math.max(0, Math.min(255, Math.round(value))) * 3
  return `rgb(${COLORMAP[c]},${COLORMAP[c + 1]},${COLORMAP[c + 2]})`
}

export interface FrequencyBarsHandle {
  /** The playhead, in seconds of a clip `duration` long (the grid spans the whole clip). */
  setTime: (seconds: number, duration: number) => void
}

interface Props {
  data: SpectrogramData | null
  /** Whether this panel's audio is playing; paused, the bars fall to zero. */
  playing: boolean
  ref?: Ref<FrequencyBarsHandle>
}

function FrequencyBarsImpl({ data, playing, ref }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null)

  // Everything the frame loop reads lives in refs: it runs outside React.
  const clock = useRef({ seconds: 0, duration: 0, at: 0 })
  const isPlaying = useRef(playing)
  const levels = useRef(new Float32Array(BANDS))
  const peaks = useRef(new Float32Array(BANDS))
  const peakAge = useRef(new Float32Array(BANDS))
  const lastFrame = useRef(0)
  const frame = useRef(0)
  const reducedMotion = useRef(false)
  const dataRef = useRef(data)
  dataRef.current = data

  /** Draws one frame; returns whether any bar is still moving (i.e. loop again). */
  const draw = (now: number): boolean => {
    const element = canvas.current
    const context = element?.getContext('2d')
    const grid = dataRef.current
    if (!element || !context) return false
    const dt = Math.min(0.1, lastFrame.current ? (now - lastFrame.current) / 1000 : 0)
    lastFrame.current = now
    const W = element.width
    const H = element.height
    context.clearRect(0, 0, W, H)
    if (!grid || W === 0) return false

    const dpr = window.devicePixelRatio || 1
    const playingNow = isPlaying.current
    const still = reducedMotion.current

    // Where playback is: the last reported time, run forward while playing so the bars
    // keep moving between clock updates.
    const { seconds, duration, at } = clock.current
    const t = playingNow ? seconds + (now - at) / 1000 : seconds
    const fraction = duration > 0 ? Math.max(0, Math.min(1, t / duration)) : 0
    const col = Math.min(grid.cols - 1, Math.floor(fraction * grid.cols))

    const floor = H - LABEL_ROOM * dpr       // the bars' baseline
    const slot = W / BANDS
    const gap = Math.max(1, Math.round(slot * 0.25))
    let moving = playingNow
    for (let b = 0; b < BANDS; b++) {
      // Band b counted from the bottom row (lowest frequency); row 0 is the top.
      const k0 = Math.floor((b * grid.rows) / BANDS)
      const k1 = Math.max(k0 + 1, Math.floor(((b + 1) * grid.rows) / BANDS))
      let value = 0
      if (playingNow) {
        let sum = 0
        let count = 0
        for (let c = Math.max(0, col - 1); c <= Math.min(grid.cols - 1, col + 1); c++) {
          let best = 0
          for (let k = k0; k < k1; k++) {
            best = Math.max(best, grid.pixels[(grid.rows - 1 - k) * grid.cols + c])
          }
          sum += best
          count++
        }
        value = sum / count
      }
      const level = still ? value : Math.max(value, levels.current[b] - RELEASE * 255 * dt)
      levels.current[b] = level
      if (level >= peaks.current[b]) {
        peaks.current[b] = level
        peakAge.current[b] = 0
      } else {
        peakAge.current[b] += dt
        if (still) peaks.current[b] = level
        else if (peakAge.current[b] > PEAK_HOLD) {
          peaks.current[b] = Math.max(level, peaks.current[b] - PEAK_FALL * 255 * dt)
        }
      }
      if (level > 0.5 || peaks.current[b] > 0.5) moving = true

      const x = b * slot + gap / 2
      const width = slot - gap
      // A faint 1 px floor per bar, so the idle strip reads as an instrument, not a gap.
      context.fillStyle = 'rgba(255,255,255,0.08)'
      context.fillRect(x, floor - dpr, width, dpr)
      if (level > 0.5) {
        const height = (level / 255) * (floor - 2 * dpr)
        context.fillStyle = rgb(level)
        context.fillRect(x, floor - height, width, height)
      }
      if (peaks.current[b] > 0.5) {
        const y = floor - (peaks.current[b] / 255) * (floor - 2 * dpr)
        context.fillStyle = 'rgba(255,255,255,0.75)'
        context.fillRect(x, y - 1.5 * dpr, width, 1.5 * dpr)
      }
    }
    return moving
  }

  const tick = (now: number) => {
    frame.current = 0
    if (draw(now)) frame.current = requestAnimationFrame(tick)
    else lastFrame.current = 0
  }

  /** Asks for a frame if the loop is idle (it restarts itself for as long as bars move). */
  const kick = () => {
    if (!frame.current) frame.current = requestAnimationFrame(tick)
  }

  useImperativeHandle(
    ref,
    () => ({
      setTime: (seconds: number, duration: number) => {
        clock.current = { seconds, duration, at: performance.now() }
        if (isPlaying.current) kick()
      },
    }),
    [],
  )

  useEffect(() => {
    isPlaying.current = playing
    clock.current.at = performance.now()   // extrapolate from now, not from the last pause
    kick()
  }, [playing])

  useEffect(() => {
    levels.current.fill(0)
    peaks.current.fill(0)
    kick()
  }, [data])

  // Canvas sized in device pixels so the bars stay crisp; follows column resizes.
  useEffect(() => {
    const element = canvas.current
    if (!element) return
    const motion = window.matchMedia('(prefers-reduced-motion: reduce)')
    reducedMotion.current = motion.matches
    const onMotion = () => (reducedMotion.current = motion.matches)
    motion.addEventListener('change', onMotion)
    const observer = new ResizeObserver(() => {
      const dpr = window.devicePixelRatio || 1
      element.width = Math.round(element.clientWidth * dpr)
      element.height = Math.round(element.clientHeight * dpr)
      kick()
    })
    observer.observe(element)
    return () => {
      observer.disconnect()
      motion.removeEventListener('change', onMotion)
      cancelAnimationFrame(frame.current)
      frame.current = 0
    }
  }, [])

  // Same log axis as the spectrogram's rows, but running left to right.
  const position = (hz: number) =>
    data ? Math.log(hz / data.fMin) / Math.log(data.fMax / data.fMin) : -1

  return (
    <div className="relative h-16 overflow-hidden rounded bg-[#0a0e0c]">
      <canvas
        ref={canvas}
        className={`h-full w-full ${data ? '' : 'invisible'}`}
        aria-label="Live frequency bars: frequency left to right, loudness bottom to top"
      />
      {data &&
        TICKS.map((hz) => {
          const at = position(hz)
          if (at <= 0.03 || at >= 0.97) return null
          return (
            <span
              key={hz}
              className="pointer-events-none absolute bottom-0 -translate-x-1/2 font-mono text-[9px] leading-3 text-white/70"
              style={{ left: `${at * 100}%` }}
            >
              {hz >= 1000 ? `${hz / 1000}k` : hz}
            </span>
          )
        })}
    </div>
  )
}

export const FrequencyBars = memo(FrequencyBarsImpl)
