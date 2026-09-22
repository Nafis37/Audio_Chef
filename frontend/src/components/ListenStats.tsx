/**
 * The panel under the two waveforms: what changed, by the numbers, and why.
 *
 * Two waveform pictures on their own do not say what is different about them -- the
 * parameters live in column 2 and the maths lived only in backend docstrings.  This fills
 * the empty bottom of the Listen column with the answer:
 *
 *   1. MEASUREMENTS -- input vs output peak / RMS / crest / duration / DC / centroid,
 *      measured in numpy by backend/app/dsp/analysis.py and delivered on the same
 *      response as the audio (the X-Bake-Stats header), so there is no second decode and
 *      the figures describe the exact buffer the DSP ran on.
 *   2. WHY IT LOOKS LIKE THIS -- one equation per active step (see explain.ts) plus the
 *      notices for things the engine otherwise does silently: clipping at +-1, a recipe
 *      that emptied the buffer, a duration or loudness that moved.
 *
 * Presentational only: every number arrives as a prop.
 */

import { memo } from 'react'
import { AlertTriangle, Info } from 'lucide-react'

import { explainStep } from '../explain'
import { TIME_SHIFTING_OPS } from '../regions'
import type { BakeStats, Measures, OperationDef, RecipeStep } from '../types'

interface Props {
  /** null until the first bake lands, or if the backend did not send the header. */
  stats: BakeStats | null
  recipe: RecipeStep[]
  operations: OperationDef[]
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

/**
 * The notices that turn a number into an explanation.  These are the things the engine
 * does silently -- clamping, an emptied buffer, a big level or spectrum move -- so each
 * one names the cause rather than leaving the user to infer it from the picture.
 */
function buildNotices(stats: BakeStats, recipe: RecipeStep[], operations: OperationDef[]) {
  const { input, output } = stats
  const notices: { tone: 'warn' | 'info'; text: string }[] = []

  if (stats.truncated) {
    notices.push({
      tone: 'warn',
      text:
        'A step left nothing to process, so the recipe stopped there. Usually a trim whose ' +
        'region selects an empty span.',
    })
  }

  if (stats.resampled?.length) {
    notices.push({
      tone: 'info',
      text:
        `${stats.resampled.length === 1 ? 'One source was' : `${stats.resampled.length} sources were`} ` +
        `recorded at a different sample rate and ${stats.resampled.length === 1 ? 'was' : 'were'} ` +
        `converted to ${stats.project_sample_rate} Hz before anything else ran. A project has one ` +
        `clock, and mixing buffers at two rates would have played one of them at the wrong speed ` +
        `and pitch.`,
    })
  }

  if (stats.clipped > 0) {
    const over = (20 * Math.log10(stats.pre_clip_peak)).toFixed(1)
    notices.push({
      tone: 'warn',
      text:
        `${stats.clipped.toLocaleString()} samples went past full scale (the recipe peaked at ` +
        `${stats.pre_clip_peak.toFixed(3)}, i.e. +${over} dB) and were clamped to ±1. The flat tops ` +
        `in the output waveform are that clamp, and clipping is a nonlinearity — it smears energy ` +
        `across the whole spectrum.`,
    })
  }

  if (Math.abs(output.duration - input.duration) > 0.005) {
    // Only two ops can do this, and regions.ts already knows which.
    const culprits = recipe
      .filter((step) => !step.bypass && TIME_SHIFTING_OPS.has(step.op))
      .map((step) => operations.find((op) => op.id === step.op)?.label ?? step.op)
    notices.push({
      tone: 'info',
      text:
        `The output is ${Math.abs(output.duration - input.duration).toFixed(2)}s ` +
        `${output.duration > input.duration ? 'longer' : 'shorter'} than the input` +
        `${culprits.length ? ` — ${[...new Set(culprits)].join(' and ')} changed the length` : ''}. ` +
        `The two waveforms are drawn to their own widths, so the same x position is not the same ` +
        `moment in both.`,
    })
  }

  const loudness = output.rms_db - input.rms_db
  if (Math.abs(loudness) >= 1 && input.rms_db > -119.99) {
    notices.push({
      tone: 'info',
      text:
        `The output is ${Math.abs(loudness).toFixed(1)} dB ` +
        `${loudness > 0 ? 'louder' : 'quieter'} on average, which is most of what makes the second ` +
        `waveform look ${loudness > 0 ? 'taller and fuller' : 'smaller'}.`,
    })
  }

  if (input.centroid_hz > 0 && output.centroid_hz > 0) {
    const move = output.centroid_hz / input.centroid_hz - 1
    if (Math.abs(move) >= 0.1) {
      notices.push({
        tone: 'info',
        text:
          `The spectral centroid moved ${move > 0 ? 'up' : 'down'} ` +
          `${Math.abs(move * 100).toFixed(0)}%, so energy shifted toward ` +
          `${move > 0 ? 'higher frequencies — the waveform looks busier and more jagged' : 'lower frequencies — the waveform looks smoother and rounder'}.`,
      })
    }
  }

  const crest = output.crest_db - input.crest_db
  if (crest <= -2 && !stats.clipped) {
    notices.push({
      tone: 'info',
      text:
        `The crest factor dropped ${Math.abs(crest).toFixed(1)} dB: the peaks came down relative to ` +
        `the average, which is the signature of compression.`,
    })
  }

  return notices
}

function ListenStatsImpl({ stats, recipe, operations, bakeMs }: Props) {
  const active = recipe.filter((step) => !step.bypass)

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
          output here, and show the equation each step applied.
        </p>
      ) : (
        <div className="space-y-4 px-4 pb-4 pt-3.5">
          {/* --- 1. The measurements ------------------------------------------------ */}
          <table className="w-full border-collapse text-xs">
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

          {/* --- 2. The notices: what the numbers above mean ------------------------- */}
          {buildNotices(stats, recipe, operations).map((notice) => (
            <p
              key={notice.text}
              className={`flex gap-2 text-[11px] leading-relaxed ${
                notice.tone === 'warn' ? 'text-rose-600' : 'text-[var(--chef-muted)]'
              }`}
            >
              {notice.tone === 'warn' ? (
                <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
              ) : (
                <Info className="mt-0.5 size-3.5 shrink-0" />
              )}
              <span>{notice.text}</span>
            </p>
          ))}

          {/* --- 3. The maths, one line per step that actually ran ------------------- */}
          {active.length > 0 && (
            <div className="space-y-2.5 border-t border-[var(--chef-border)] pt-3">
              <h3 className="text-[11px] font-semibold uppercase tracking-wider text-[var(--chef-muted)]">
                The maths, step by step
              </h3>
              {active.map((step, index) => {
                const op = operations.find((candidate) => candidate.id === step.op)
                if (!op) return null
                const explanation = explainStep(op, step, stats.sample_rate)
                return (
                  <div key={step.uid} className="space-y-1">
                    <div className="flex items-baseline gap-2">
                      <span className="font-mono text-[10px] text-[var(--chef-accent-strong)]">
                        {index + 1}
                      </span>
                      <span className="text-xs font-medium">{op.label}</span>
                    </div>
                    {explanation && (
                      <>
                        {/* Long equations scroll inside their own box rather than
                            widening the column. */}
                        <p className="overflow-x-auto rounded bg-[var(--chef-inset)] px-2 py-1.5 font-mono text-[11px] whitespace-nowrap">
                          {explanation.equation}
                        </p>
                        <p className="text-[11px] leading-relaxed text-[var(--chef-muted)]">
                          {explanation.effect}
                        </p>
                      </>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}
    </section>
  )
}

export const ListenStats = memo(ListenStatsImpl)
