/**
 * One line of maths per recipe step.
 *
 * The Listen column shows two waveforms; this is the table that says WHY the second one
 * differs from the first.  Each entry renders the difference equation or transfer
 * function the backend module actually implements -- with the card's live parameter
 * values substituted in -- plus one sentence on what that does to the drawing.
 *
 * The equations are deliberately the same ones derived in the backend docstrings
 * (backend/app/dsp/*.py); this file is the user-facing end of that derivation trail, not
 * a second, independent explanation.  Sources are named where the backend names them.
 *
 * Prose about what an operation IS comes from `OperationDef.description`, served by
 * GET /operations -- it is not duplicated here.
 */

import type { OperationDef, ParamValue, RecipeStep } from './types'

export interface StepExplanation {
  /** The maths, as a single short line of Unicode. */
  equation: string
  /** What that does to the waveform you are looking at. */
  effect: string
}

/** Trim trailing zeros so "6" reads as 6, not 6.0, but 0.35 keeps its digits. */
function num(value: number, digits = 2): string {
  return Number(value.toFixed(digits)).toString()
}

/** Signed, for gains where the direction is the whole point. */
function signed(value: number, digits = 1): string {
  return `${value >= 0 ? '+' : ''}${num(value, digits)}`
}

/**
 * Reads one parameter, falling back to the schema default.  A step built by `newStep`
 * always carries every key, but a hand-made or older stored recipe may not.
 */
function reader(op: OperationDef, step: RecipeStep) {
  return (name: string): ParamValue => {
    if (name in step.params) return step.params[name]
    return op.params.find((p) => p.name === name)?.default ?? 0
  }
}

/** Which EQ bands are actually doing something, so a flat band is not "explained". */
function activeBands(get: (name: string) => ParamValue): string[] {
  return (['low', 'mid', 'high'] as const)
    .filter((band) => Math.abs(Number(get(`${band}_gain`))) >= 0.05)
    .map(
      (band) =>
        `${signed(Number(get(`${band}_gain`)))} dB @ ${num(Number(get(`${band}_freq`)), 0)} Hz`,
    )
}

type Builder = (get: (name: string) => ParamValue, fs: number) => StepExplanation

const BUILDERS: Record<string, Builder> = {
  // noise.py -- spectral subtraction with a spectral floor.
  noise_remover: (get) => {
    const amount = Number(get('amount'))
    const floor = Number(get('floor'))
    return {
      equation: `|Y(k)| = max(|X(k)| − ${num(amount)}·N̂(k), ${num(floor)}·|X(k)|)`,
      effect:
        `Every frame's magnitude spectrum has ${num(amount)}× the noise profile measured over ` +
        `${num(Number(get('noise_start')))}–${num(Number(get('noise_end')))}s subtracted from it, ` +
        `never falling below ${Math.round(floor * 100)}% of the original bin. Phase is kept, so the ` +
        `waveform holds its shape while its quiet stretches flatten toward the axis.`,
    }
  },

  // eq.py -- three RBJ peaking biquads in series.
  equalizer: (get) => {
    const bands = activeBands(get)
    return {
      equation: 'H(z) = (b₀ + b₁z⁻¹ + b₂z⁻²) / (1 + a₁z⁻¹ + a₂z⁻²)',
      effect: bands.length
        ? `Peaking biquads (RBJ cookbook coefficients) at ${bands.join(', ')} with Q = ` +
          `${num(Number(get('q')))}. A boosted band adds energy at those frequencies, so the output ` +
          `is taller and the spectral centroid below moves toward them.`
        : 'Every band sits at 0 dB, so all three biquads are exactly unity — the output is the ' +
          'input, sample for sample.',
    }
  },

  // echo_reverb.py -- convolution with a tapped-delay IR, or with a synthesised room.
  echo_reverb: (get, fs) => {
    const mix = Number(get('mix'))
    if (get('mode') === 'reverb') {
      const decay = Number(get('decay'))
      return {
        equation:
          `y[n] = Σₖ h[k]·x[n−k],  h[k] = w[k]·10^(−3k / (${num(decay)}s·${fs} Hz))` +
          `   ·   mix ${Math.round(mix * 100)}%`,
        effect:
          `The input is convolved with a synthesised room response: a few early reflections over ` +
          `noise w[k] that fades 60 dB in ${num(decay)}s. Every sample of the output is a sum over ` +
          `that whole response, which is why the result grows a smooth tail instead of discrete ` +
          `repeats.`,
      }
    }
    const delay = Number(get('delay'))
    const feedback = Number(get('feedback'))
    return {
      equation:
        `y = x ⊛ h,  h = Σₖ ${num(feedback)}^k·δ[n − k·D],  D = ${num(delay)}s × ${fs} Hz = ` +
        `${Math.round(delay * fs)} samples`,
      effect:
        `Convolution with a tapped delay line: the impulse response is one spike every ` +
        `${Math.round(delay * fs)} samples, each ${num(feedback)}× the last, so copies of the ` +
        `waveform appear every ${num(delay)}s and die away geometrically. Overlapping repeats add, ` +
        `which is how an echo can push the output past full scale.`,
    }
  },

  // editor.py -- pure numpy slicing.
  editor: (get, fs) => {
    const start = Number(get('start'))
    const end = Number(get('end'))
    const a = Math.round(start * fs)
    const b = Math.round(end * fs)
    if (get('mode') === 'splice') {
      return {
        equation: `y = [ x[0 : ${a}] , x[${b} : N] ]`,
        effect:
          `The ${num(end - start)}s between ${num(start)}s and ${num(end)}s is removed and the two ` +
          `sides are concatenated, so the output is shorter and everything after the cut slides left.`,
      }
    }
    return {
      equation: `y = x[⌊${num(start)}·${fs}⌋ : ${end > 0 ? `⌊${num(end)}·${fs}⌋` : 'N'}]`,
      effect:
        `Only samples ${a}${end > 0 ? `–${b}` : ' onward'} survive. Nothing is filtered — the output ` +
        `is that slice of the input, unchanged.`,
    }
  },

  // speed_pitch.py -- phase-vocoder stretch plus np.interp resampling.
  speed_pitch: (get) => {
    const speed = Number(get('speed'))
    const semitones = Number(get('semitones'))
    const preserve = Boolean(get('preserve_pitch'))
    const ratio = Math.pow(2, semitones / 12)
    const stretching = Math.abs(speed - 1) > 1e-3
    const shifting = Math.abs(semitones) > 1e-3
    const parts: string[] = []
    if (stretching) parts.push(`hop ratio r = 1/${num(speed)} = ${num(1 / speed, 3)}`)
    if (shifting) parts.push(`2^(${num(semitones, 0)}/12) = ${num(ratio, 3)}`)
    if (!parts.length) {
      return {
        equation: 'r = 1   (identity)',
        effect: 'Speed 1× and 0 semitones: the buffer passes through untouched.',
      }
    }
    return {
      equation: parts.join('   ·   '),
      effect:
        (stretching
          ? preserve
            ? `The phase vocoder resynthesises on a ${num(1 / speed, 3)}× hop, so the duration changes ` +
              `while accumulated phase holds the pitch — that is why the output waveform is ` +
              `${speed > 1 ? 'shorter' : 'longer'} than the input above.`
            : `Reading the samples ${num(speed)}× faster by linear interpolation moves duration and ` +
              `pitch together, like a tape sped up.`
          : '') +
        (shifting
          ? `${stretching ? ' ' : ''}Pitch is resampled by ${num(ratio, 3)}×, which shifts every ` +
            `harmonic without touching the length.`
          : ''),
    }
  },

  // compressor.py -- one-pole envelope follower + dB-domain gain computer.
  compressor: (get, fs) => {
    const attack = Number(get('attack'))
    const ratio = Number(get('ratio'))
    const threshold = Number(get('threshold'))
    const makeup = Number(get('makeup'))
    // Same alpha the backend derives: a^(tau*fs) = 1/e.
    const alpha = Math.exp(-1 / (Math.max(attack, 0.01) / 1000) / fs)
    return {
      equation:
        `env[n] = α·env[n−1] + (1−α)|x[n]|,  α = e^(−1/(${num(attack, 0)}ms·fs)) = ${num(alpha, 4)}` +
        `   ·   g_dB = −(1 − 1/${num(ratio)})·(L − ${num(threshold, 0)})`,
      effect:
        `Whenever the smoothed level L climbs above ${num(threshold, 0)} dBFS the signal is turned ` +
        `down by ${num(1 - 1 / ratio, 2)} dB per dB of excess` +
        `${Math.abs(makeup) > 0.05 ? `, then the whole thing is lifted ${signed(makeup)} dB` : ''}. ` +
        `Peaks come down while the average does not, which is exactly why the output looks like a ` +
        `solid block rather than spikes — watch the crest factor above.`,
    }
  },
}

/**
 * The maths for one card, with its current parameter values filled in.
 *
 * `fs` is the sample rate reported by the last bake -- several equations only become
 * concrete once seconds are turned into samples.
 */
export function explainStep(
  op: OperationDef,
  step: RecipeStep,
  fs: number,
): StepExplanation | null {
  const build = BUILDERS[op.id]
  if (!build) return null // a backend op with no entry yet: say nothing rather than guess
  return build(reader(op, step), fs)
}
