/**
 * The panel under the two waveforms: input vs output peak / RMS / crest / duration / DC /
 * centroid, measured in numpy by backend/app/dsp/analysis.py and delivered on the same
 * response as the audio (the X-Bake-Stats header), so there is no second decode and the
 * figures describe the exact buffer the DSP ran on.
 *
 * Presentational only: every number arrives as a prop.
 */

import { memo } from 'react'

import type { BakeStats, Measures } from '../types'

interface Props {
  /** null until the first bake lands, or if the backend did not send the header. */
  stats: BakeStats | null
  bakeMs: number
}

/** -120 dB is analysis.py's silence floor; showing "-120.00 dB" would be noise. */
function db(value: number): string {
  return value <= -119.99 ? '−∞ dB' : `${value.toFixed(2)} dB`
}

/** Signed difference, so the column reads as "what the recipe did". */
function deltaDb(before: number, after: number): string {
  if (before <= -119.99 || after <= -119.99) return '—'
  const d = after - before
  return `${d >= 0 ? '+' : '−'}${Math.abs(d).toFixed(2)} dB`
}

function hz(value: number): string {
  return value >= 1000 ? `${(value / 1000).toFixed(2)} kHz` : `${value.toFixed(0)} Hz`
}

interface Row {
  label: string
  hint: string
  input: string
  output: string
  delta: string
}

function buildRows(input: Measures, output: Measures): Row[] {
  return [
    {
      label: 'Peak',
      hint: 'max |x[n]| — the height of the drawing',
      input: db(input.peak_db),
      output: db(output.peak_db),
      delta: deltaDb(input.peak_db, output.peak_db),
    },
    {
      label: 'RMS',
      hint: '√(mean x²) — the loudness the ear follows',
      input: db(input.rms_db),
      output: db(output.rms_db),
      delta: deltaDb(input.rms_db, output.rms_db),
    },
    {
      label: 'Crest',
      hint: 'peak − RMS — spiky (high) vs. solid (low)',
      input: db(input.crest_db),
      output: db(output.crest_db),
      delta: deltaDb(input.crest_db, output.crest_db),
    },
    {
      label: 'Centroid',
      hint: 'Σf·|X(f)| / Σ|X(f)| — where the energy sits',
      input: hz(input.centroid_hz),
      output: hz(output.centroid_hz),
      delta:
        input.centroid_hz > 0
          ? `${output.centroid_hz >= input.centroid_hz ? '+' : '−'}${Math.abs(
              (output.centroid_hz / input.centroid_hz - 1) * 100,
            ).toFixed(0)}%`
          : '—',
    },
    {
      label: 'Duration',
      hint: 'N / fs',
      input: `${input.duration.toFixed(2)}s`,
      output: `${output.duration.toFixed(2)}s`,
      delta: `${output.duration >= input.duration ? '+' : '−'}${Math.abs(
        output.duration - input.duration,
      ).toFixed(2)}s`,
    },
    {
      label: 'Samples',
      hint: 'N',
      input: input.frames.toLocaleString(),
      output: output.frames.toLocaleString(),
      delta: `${output.frames >= input.frames ? '+' : '−'}${Math.abs(
        output.frames - input.frames,
      ).toLocaleString()}`,
    },
    {
      label: 'DC offset',
      hint: 'mean x[n] — should sit at 0',
      input: input.dc.toFixed(5),
      output: output.dc.toFixed(5),
      delta: (output.dc - input.dc).toFixed(5),
    },
  ]
}

function ListenStatsImpl({ stats, bakeMs }: Props) {
  return (
    <section className="rounded-lg border border-[var(--chef-border)] bg-[var(--chef-panel)]">
      <header className="flex items-center gap-3 border-b border-[var(--chef-border)] px-4 py-2.5">
        <h2 className="text-xs font-semibold uppercase tracking-[0.18em] text-[var(--chef-muted)]">
          What changed
        </h2>
        {stats && (
          <span className="ml-auto font-mono text-xs text-[var(--chef-muted)]">
            {stats.sample_rate} Hz · mono · baked in {bakeMs.toFixed(0)} ms
          </span>
        )}
      </header>

      {!stats ? (
        <p className="px-4 pb-4 pt-3.5 text-xs leading-relaxed text-[var(--chef-muted)]">
          Add an operation to the recipe and Audio Chef will measure the input against the
          output here.
        </p>
      ) : (
        <div className="space-y-4 px-4 pb-4 pt-3.5">
          <details open className="group">
            <summary className="cursor-pointer text-[11px] font-semibold uppercase tracking-wider text-[var(--chef-muted)] hover:text-[var(--chef-text)]">
              The numbers
            </summary>
          <table className="mt-2 w-full border-collapse text-xs">
            <thead>
              <tr className="text-[11px] uppercase tracking-wider text-[var(--chef-muted)]">
                <th className="pb-1.5 text-left font-medium">Measure</th>
                <th className="pb-1.5 text-right font-medium">Input</th>
                <th className="pb-1.5 text-right font-medium">Output</th>
                <th className="pb-1.5 text-right font-medium">Δ</th>
              </tr>
            </thead>
            <tbody>
              {buildRows(stats.input, stats.output).map((row) => (
                <tr key={row.label} className="border-t border-[var(--chef-border)]">
                  <td className="py-1.5 pr-3">
                    <span>{row.label}</span>{' '}
                    <span className="font-mono text-[10px] text-[var(--chef-muted)]">
                      {row.hint}
                    </span>
                  </td>
                  <td className="py-1.5 text-right font-mono text-[var(--chef-muted)]">
                    {row.input}
                  </td>
                  <td className="py-1.5 text-right font-mono">{row.output}</td>
                  <td className="py-1.5 pl-3 text-right font-mono text-[var(--chef-accent-strong)]">
                    {row.delta}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          </details>

        </div>
      )}
    </section>
  )
}

export const ListenStats = memo(ListenStatsImpl)
