/**
 * The Filter card's frequency-response curve.
 *
 * Plots |H(e^{jw})| in dB against a log frequency axis -- the textbook picture of a
 * filter -- with the cutoff and the -3 dB level marked.  The points come from
 * GET /filter/response, which evaluates the very biquads the bake runs
 * (backend/app/dsp/filters.py), so this is the filter that is applied, not a sketch of it.
 * Put it next to the Output spectrogram: where this curve falls away, the picture goes dark.
 *
 * Fetches are debounced and each new one aborts the last, the same pattern as Auto-Bake,
 * so dragging the frequency slider never lets a stale curve land after a newer one.
 */

import { memo, useEffect, useState } from 'react'
import { fetchFilterResponse } from '../api'
import { bandEdges, hzLabel } from '../filterMarkers'
import type { FilterResponse as ResponseData, ParamValue } from '../types'

const DEBOUNCE_MS = 150

// Plot geometry, in SVG user units (the SVG scales to the card's width).
const W = 300
const H = 96
const PAD_L = 26 // room for the dB labels
const PAD_B = 12 // room for the Hz labels
const F_MIN = 20
const F_MAX = 20000
const DB_TOP = 6
const DB_BOTTOM = -60

const X_TICKS = [100, 1000, 10000]
const DB_TICKS = [0, -24, -48]

const x = (hz: number) =>
  PAD_L + ((W - PAD_L) * Math.log(hz / F_MIN)) / Math.log(F_MAX / F_MIN)
const y = (db: number) =>
  ((H - PAD_B) * (DB_TOP - Math.max(DB_BOTTOM, Math.min(DB_TOP, db)))) / (DB_TOP - DB_BOTTOM)

interface Props {
  values: Record<string, ParamValue>
  bypassed: boolean
}

function FilterResponseImpl({ values, bypassed }: Props) {
  const mode = String(values.mode)
  const cutoff = Number(values.cutoff)
  const order = String(values.order)
  const q = Number(values.q)
  const [data, setData] = useState<ResponseData | null>(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      fetchFilterResponse({ mode, cutoff, order, q }, controller.signal)
        .then((next) => {
          setData(next)
          setFailed(false)
        })
        .catch((error) => {
          if (error?.name !== 'AbortError') setFailed(true)
        })
    }, DEBOUNCE_MS)
    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [mode, cutoff, order, q])

  if (failed && !data) return null

  const path = data
    ? data.freqs.map((f, i) => `${i ? 'L' : 'M'}${x(f).toFixed(1)},${y(data.db[i]).toFixed(1)}`).join('')
    : ''
  // Band-pass: mark both -3 dB edges; everything else: the one cutoff / centre.
  const marks = mode === 'bandpass' ? bandEdges(cutoff, q) : [cutoff]
  const slope =
    mode === 'lowpass' || mode === 'highpass' ? `${6 * Number(order)} dB/oct` : `Q ${q}`

  return (
    <figure
      className={`border-t border-[var(--chef-border)] px-3 pt-2 pb-1 transition ${
        bypassed ? 'opacity-40 grayscale' : ''
      }`}
    >
      <figcaption className="mb-0.5 flex justify-between text-[10px] text-[var(--chef-muted)]">
        <span>Frequency response |H(f)|</span>
        <span className="font-mono">{slope}</span>
      </figcaption>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="h-24 w-full overflow-visible"
        role="img"
        aria-label={`Filter response: ${mode} at ${hzLabel(cutoff)}`}
      >
        {/* Between the curve and 0 dB: what is removed -- the part of the spectrogram that
            goes dark. */}
        {data && (
          <path
            d={`${path}L${W},${y(0)}L${PAD_L},${y(0)}Z`}
            fill="var(--chef-muted)"
            opacity={0.08}
          />
        )}
        {DB_TICKS.map((db) => (
          <g key={db}>
            <line x1={PAD_L} x2={W} y1={y(db)} y2={y(db)} stroke="var(--chef-border)" strokeWidth={0.6} />
            <text x={PAD_L - 3} y={y(db) + 3} textAnchor="end" fontSize={8} fill="var(--chef-muted)">
              {db}
            </text>
          </g>
        ))}
        {/* -3 dB: where a Butterworth low/high-pass sits exactly at its cutoff. */}
        <line
          x1={PAD_L} x2={W} y1={y(-3)} y2={y(-3)}
          stroke="var(--chef-accent-strong)" strokeWidth={0.6} strokeDasharray="2 2" opacity={0.6}
        />
        <text x={W - 2} y={y(-3) - 2} textAnchor="end" fontSize={7} fill="var(--chef-accent-strong)">
          −3 dB
        </text>
        {X_TICKS.map((hz) => (
          <text key={hz} x={x(hz)} y={H - 1} textAnchor="middle" fontSize={8} fill="var(--chef-muted)">
            {hz >= 1000 ? `${hz / 1000}k` : hz}
          </text>
        ))}
        {marks.map((hz) => (
          <line
            key={hz}
            x1={x(hz)} x2={x(hz)} y1={0} y2={H - PAD_B}
            stroke="var(--chef-accent-strong)" strokeWidth={0.8} strokeDasharray="3 2"
          />
        ))}
        {data && (
          <path d={path} fill="none" stroke="var(--chef-accent-strong)" strokeWidth={1.8} />
        )}
        <text
          x={Math.min(Math.max(x(cutoff) + 3, PAD_L + 2), W - 44)}
          y={9}
          fontSize={8}
          fontWeight={600}
          fill="var(--chef-accent-strong)"
        >
          {hzLabel(cutoff)}
        </text>
      </svg>
    </figure>
  )
}

export const FilterResponse = memo(FilterResponseImpl)
