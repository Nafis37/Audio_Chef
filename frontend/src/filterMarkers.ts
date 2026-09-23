/**
 * Where each Filter step cuts, as lines to draw across the spectrograms.
 *
 * The same lines go on BOTH the Input and the Output spectrogram on purpose: above a
 * low-pass line the Input is still bright and the Output has gone dark, so the before /
 * after is read against one fixed reference instead of two pictures compared by memory.
 *
 * Values arrive in the units the card shows (Hz, plain Q) -- the Filter op has no
 * `scale`d params, so they are exactly what backend/app/dsp/filters.py receives.
 */

import type { ParamValue, RecipeStep, SpectrogramMarker } from './types'

/** Filter defaults, for a stored step that predates a param (mirrors dsp_engine.py). */
const DEFAULTS: Record<string, ParamValue> = { mode: 'lowpass', cutoff: 500, order: '4', q: 2 }

export function hzLabel(hz: number): string {
  return hz >= 1000 ? `${Number((hz / 1000).toFixed(hz >= 10000 ? 0 : 1))} kHz` : `${Math.round(hz)} Hz`
}

/**
 * The band-pass's two -3 dB edges.  The analog prototype (s/Q)/(s^2 + s/Q + 1) is 3 dB
 * down where |1 - W^2| = W/Q, i.e.  W = sqrt(1 + 1/(4Q^2)) +- 1/(2Q)  (their product is 1,
 * so they sit symmetrically around fc on a log axis).
 */
export function bandEdges(fc: number, q: number): [number, number] {
  const half = 1 / (2 * q)
  const root = Math.sqrt(1 + half * half)
  return [fc * (root - half), fc * (root + half)]
}

export function filterMarkers(recipe: RecipeStep[]): SpectrogramMarker[] {
  const markers: SpectrogramMarker[] = []
  for (const step of recipe) {
    if (step.op !== 'filter' || step.bypass) continue
    const get = (name: string) => step.params[name] ?? DEFAULTS[name]
    const mode = String(get('mode'))
    const fc = Number(get('cutoff'))
    if (mode === 'lowpass') markers.push({ hz: fc, label: `LP ${hzLabel(fc)}` })
    else if (mode === 'highpass') markers.push({ hz: fc, label: `HP ${hzLabel(fc)}` })
    else if (mode === 'notch') markers.push({ hz: fc, label: `Notch ${hzLabel(fc)}` })
    else {
      const [lo, hi] = bandEdges(fc, Number(get('q')))
      markers.push({ hz: lo, label: `BP ${hzLabel(lo)}` }, { hz: hi, label: `BP ${hzLabel(hi)}` })
    }
  }
  return markers
}
