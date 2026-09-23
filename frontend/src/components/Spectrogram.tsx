/**
 * The spectrogram strip under a waveform.
 *
 * The backend does all the maths (backend/app/dsp/spectrogram.py: our own STFT, dBFS on a
 * FIXED -90..0 dB scale, log-spaced rows) and sends a raw uint8 grid.  This component only
 * paints it: one canvas pixel per (column, row), stretched by CSS to the waveform's width,
 * each byte looked up in a colour map.
 *
 * Because the scale is absolute rather than normalised per picture, Input and Output can
 * be compared by eye: a louder band IS brighter, a removed hiss IS darker.
 */

import { memo, useEffect, useRef } from 'react'
import type { SpectrogramData, SpectrogramMarker } from '../types'

/**
 * 256 RGB entries, interpolated between a few stops: near-black -> the app's greens ->
 * yellow -> white.  Built once.  Dark = quiet, bright = loud.
 */
const COLORMAP: Uint8Array = (() => {
  const stops: [number, [number, number, number]][] = [
    [0.0, [10, 14, 12]],
    [0.3, [20, 60, 40]],
    [0.55, [21, 128, 61]],
    [0.75, [74, 222, 128]],
    [0.9, [250, 204, 21]],
    [1.0, [254, 252, 232]],
  ]
  const table = new Uint8Array(256 * 3)
  for (let i = 0; i < 256; i++) {
    const t = i / 255
    const upper = stops.findIndex(([at]) => at >= t)
    const [t1, c1] = stops[Math.max(upper, 1)]
    const [t0, c0] = stops[Math.max(upper, 1) - 1]
    const f = (t - t0) / (t1 - t0 || 1)
    for (let k = 0; k < 3; k++) table[i * 3 + k] = Math.round(c0[k] + f * (c1[k] - c0[k]))
  }
  return table
})()

/** Frequencies worth labelling on the side, where they fall inside the picture. */
const TICKS = [100, 1000, 10000]

function tickLabel(hz: number): string {
  return hz >= 1000 ? `${hz / 1000}k` : `${hz}`
}

interface Props {
  data: SpectrogramData | null
  /** Shown when there is no picture (yet). */
  hint?: string
  /**
   * Labelled dashed lines across the picture -- a Filter step's cutoff.  Drawn on the
   * Input AND the Output so both are read against the same line (see filterMarkers.ts).
   */
  markers?: SpectrogramMarker[]
}

function SpectrogramImpl({ data, hint = '', markers = [] }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const element = canvas.current
    if (!element || !data) return
    element.width = data.cols
    element.height = data.rows
    const context = element.getContext('2d')
    if (!context) return
    const image = context.createImageData(data.cols, data.rows)
    // The grid is row-major with row 0 = the highest frequency, which is exactly the
    // canvas's top-to-bottom order.
    for (let i = 0; i < data.pixels.length; i++) {
      const c = data.pixels[i] * 3
      image.data[i * 4] = COLORMAP[c]
      image.data[i * 4 + 1] = COLORMAP[c + 1]
      image.data[i * 4 + 2] = COLORMAP[c + 2]
      image.data[i * 4 + 3] = 255
    }
    context.putImageData(image, 0, 0)
  }, [data])

  // Row r sits at log(f / f_min) / log(f_max / f_min) of the height, measured from the BOTTOM.
  const position = (hz: number) =>
    data ? Math.log(hz / data.fMin) / Math.log(data.fMax / data.fMin) : -1

  return (
    <div className="relative mt-2 h-28 overflow-hidden rounded bg-[#0a0e0c]">
      <canvas
        ref={canvas}
        className={`h-full w-full ${data ? '' : 'invisible'}`}
        aria-label="Spectrogram: time left to right, frequency bottom to top, brightness = loudness"
      />
      {data &&
        TICKS.map((hz) => {
          const at = position(hz)
          if (at <= 0.02 || at >= 0.98) return null
          return (
            <span
              key={hz}
              className="pointer-events-none absolute left-1 -translate-y-1/2 font-mono text-[9px] text-white/70"
              style={{ top: `${(1 - at) * 100}%` }}
            >
              {tickLabel(hz)}
            </span>
          )
        })}
      {data &&
        markers.map(({ hz, label }) => {
          const at = position(hz)
          if (at <= 0 || at >= 1) return null
          return (
            <div
              key={`${label}-${hz}`}
              className="pointer-events-none absolute inset-x-0 border-t border-dashed border-white/90"
              style={{ top: `${(1 - at) * 100}%` }}
            >
              <span className="absolute right-1 -translate-y-full rounded-sm bg-black/60 px-1 font-mono text-[9px] font-semibold text-white">
                {label}
              </span>
            </div>
          )
        })}
      {!data && (
        <p className="absolute inset-0 flex items-center justify-center text-[11px] text-white/50">
          {hint}
        </p>
      )}
    </div>
  )
}

export const Spectrogram = memo(SpectrogramImpl)
