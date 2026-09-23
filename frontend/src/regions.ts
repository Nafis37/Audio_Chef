/**
 * Which operations have time parameters, and therefore a draggable region.
 *
 * Some of the ops select a span of the audio by seconds.  Typing those seconds blind
 * is the thing this table exists to remove: the linked card's numbers and the handles on
 * the Input waveform are two views of the same pair of parameters.
 *
 * Stated once here rather than spread through App / Recipe / RecipeCard, so adding a
 * third time-based operation is one entry and no component changes.  The param NAMES must
 * match the backend schema in dsp_engine.py exactly -- they are what /process receives.
 */

import type { ParamValue } from './types'

export interface RegionBinding {
  /** Parameter holding the start of the span, in seconds. */
  start: string
  /** Parameter holding the end of the span, in seconds. */
  end: string
  /** Region fill.  Translucent, because the waveform underneath is the point. */
  color: string
  /** Shown on the waveform header while this op is linked. */
  label: string
  /** When the region applies at all -- e.g. the noise window only in "region" mode. */
  when?: (params: Record<string, ParamValue>) => boolean
}

/** The binding for a step, or undefined when the op has none or it is switched off. */
export function regionFor(op: string, params: Record<string, ParamValue>): RegionBinding | undefined {
  const binding = REGION_OPS[op]
  if (!binding || (binding.when && !binding.when(params))) return undefined
  return binding
}

export const REGION_OPS: Record<string, RegionBinding> = {
  // The span trim keeps / splice removes.  Green: it is the app's primary accent.
  editor: {
    start: 'start',
    end: 'end',
    color: 'rgba(34, 197, 94, 0.22)',
    label: 'Trim / splice region',
  },
  // The quiet stretch spectral subtraction learns its noise profile from.  Amber, so a
  // glance tells the two apart -- they measure different things and are often both set.
  noise_remover: {
    start: 'noise_start',
    end: 'noise_end',
    color: 'rgba(245, 158, 11, 0.22)',
    label: 'Noise-only stretch',
    // "Automatically" needs no region; only "From a part I mark" does.
    when: (params) => params.profile === 'region',
  },
  // The span Reverse plays backwards.  Violet: not the green of a cut -- nothing is
  // removed, only turned around.
  reverse: {
    start: 'start',
    end: 'end',
    color: 'rgba(139, 92, 246, 0.22)',
    label: 'Reversed part',
    when: (params) => params.mode === 'selection',
  },
}

/**
 * Operations that change the DURATION of what flows through them.
 *
 * A region is drawn against the ORIGINAL input, but every step sees the audio as the
 * steps above it left it.  Once one of these runs first, the two clocks diverge and the
 * handles no longer mean what they appear to -- the card says so rather than lying.
 */
export const TIME_SHIFTING_OPS = new Set(['editor', 'speed_pitch', 'assemble', 'silence_remover'])

/**
 * Why `assemble` is in the set above but NOT in REGION_OPS.
 *
 * It definitely changes the duration -- every placement mode grows the buffer -- so the
 * warning above applies to it like any other.  But its clip_start / clip_end name times
 * inside a DIFFERENT file, and a region binding draws its handles on the waveform of the
 * CURRENT one.  Binding it would put handles on the input that look like they edit what
 * is under them and do not.  Typed seconds are honest until the Input viewer can follow
 * a linked assemble card to the source it references.
 */
