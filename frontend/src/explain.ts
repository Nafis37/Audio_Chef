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
 * Prose about what an operation IS comes from `OperationDef.summary` / `how`, served by
 * GET /operations -- it is not duplicated here.
 *
 * Values arrive in the units the card SHOWS: a `%` param is 0..100 here, and is divided
 * by 100 below wherever the maths wants the 0..1 the backend actually uses.
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
  const where = { bass: 'below', mid: 'around', treble: 'above' } as const
  return (['bass', 'mid', 'treble'] as const)
    .filter((band) => Math.abs(Number(get(`${band}_gain`))) >= 0.05)
    .map(
      (band) =>
        `${band} ${signed(Number(get(`${band}_gain`)))} dB ${where[band]} ` +
        `${num(Number(get(`${band}_freq`)), 0)} Hz`,
    )
}

/** A `%` param as the 0..1 fraction the backend uses. */
function fraction(get: (name: string) => ParamValue, name: string): number {
  return Number(get(name)) / 100
}

function echoExplanation(delay: number, feedback: number, mix: number, fs: number): StepExplanation {
  return {
    equation:
      `y = x ⊛ h,  h = Σₖ ${num(feedback)}^k·δ[n − k·D],  D = ${num(delay)}s × ${fs} Hz = ` +
      `${Math.round(delay * fs)} samples   ·   wet ${Math.round(mix * 100)}%`,
    effect:
      `Convolution with a tapped delay line: the impulse response is one spike every ` +
      `${Math.round(delay * fs)} samples, each ${num(feedback)}× the last, so copies of the ` +
      `waveform appear every ${num(delay)}s and die away geometrically. Overlapping repeats add, ` +
      `which is how an echo can push the output past full scale.`,
  }
}

function reverbExplanation(decay: number, mix: number, fs: number): StepExplanation {
  return {
    equation:
      `y[n] = Σₖ h[k]·x[n−k],  h[k] = w[k]·10^(−3k / (${num(decay)}s·${fs} Hz))` +
      `   ·   wet ${Math.round(mix * 100)}%`,
    effect:
      `The input is convolved with a synthesised room response: a few early reflections over ` +
      `noise w[k] that fades 60 dB in ${num(decay)}s. Every sample of the output is a sum over ` +
      `that whole response, which is why the result grows a smooth tail instead of discrete ` +
      `repeats.`,
  }
}

type Builder = (get: (name: string) => ParamValue, fs: number) => StepExplanation

const BUILDERS: Record<string, Builder> = {
  // mixer.py -- the only op that reads audio from outside its own chain.
  assemble: (get, fs) => {
    const mode = String(get('mode'))
    const gain = Number(get('gain'))
    const g = 10 ** (gain / 20)
    const position = Number(get('position'))
    const start = Number(get('clip_start'))
    const end = Number(get('clip_end'))
    const p = Math.round(position * fs)
    const span =
      end > start
        ? `${num(start)}–${num(end)}s (${num(end - start)}s)`
        : start > 0
          ? `${num(start)}s to its end`
          : 'the whole clip'

    const equation =
      mode === 'mix'
        ? `y[n] = x[n] + ${num(g, 3)}·c[n − ${p.toLocaleString()}],  length = max(N, ${p.toLocaleString()} + M)`
        : mode === 'insert'
          ? `y = [ x[0…${p.toLocaleString()}) , ${num(g, 3)}·c , x[${p.toLocaleString()}…N) ],  length = N + M`
          : `y = [ x , ${num(g, 3)}·c ],  length = N + M`

    const effect =
      (mode === 'mix'
        ? `The clip is ADDED on top, so both play at once and the two waveforms sum sample by ` +
          `sample — which is also how this step can push the result past full scale even when ` +
          `neither part was close to it on its own. `
        : mode === 'insert'
          ? `The buffer is cut open at ${num(position)}s and the clip is dropped into the gap, so ` +
            `everything after it moves later by the clip's length. `
          : `The clip is joined onto the end, so nothing already in the buffer moves. `) +
      `The clip is ${span} of the other source AFTER that source's own recipe has run, taken at ` +
      `${signed(gain)} dB (×${num(g, 3)}). Both cut edges get a 5 ms fade so the joins do not click.`

    return { equation, effect }
  },

  // noise.py -- spectral subtraction with a spectral floor and a smoothed gain.
  noise_remover: (get) => {
    const amount = Number(get('amount'))
    const floor = fraction(get, 'floor')
    const auto = get('profile') !== 'region'
    return {
      equation:
        `G = max(1 − ${num(amount)}·N̂(k)/|X(t,k)|, ${num(floor)}),  Y = mean₃ₜ(G)·X` +
        (auto ? `,  N̂(k) = 2.73·P₁₀ₜ|X(t,k)|` : ''),
      effect:
        (auto
          ? `The noise level of every frequency band is found automatically from its quietest ` +
            `10% of moments (the pauses between words), de-biased by 2.73 for Rayleigh-distributed ` +
            `noise. `
          : `The noise level of every band is averaged over the marked stretch ` +
            `${num(Number(get('noise_start')))}–${num(Number(get('noise_end')))}s. `) +
        `Each frame then has ${num(amount)}× that level taken off every band, never below ` +
        `${Math.round(floor * 100)}% of the original, with the gain averaged over 3 frames so no ` +
        `lone bins warble. Phase is kept, so speech holds its shape while the pauses flatten ` +
        `toward the axis.`,
    }
  },

  // eq.py -- low shelf, peaking bell, high shelf (RBJ cookbook) in series.
  equalizer: (get) => {
    const bands = activeBands(get)
    return {
      equation: 'H(z) = H_bass(z)·H_mid(z)·H_treble(z),  each (b₀ + b₁z⁻¹ + b₂z⁻²) / (1 + a₁z⁻¹ + a₂z⁻²)',
      effect: bands.length
        ? `Three biquads with RBJ cookbook coefficients: ${bands.join(', ')}. The bass and treble ` +
          `controls are SHELVES — they move everything past their corner, not one narrow band — ` +
          `so a boost lights up a whole strip of the spectrogram and pulls the spectral centroid ` +
          `toward it.`
        : 'Every band sits at 0 dB, so all three biquads are exactly unity — the output is the ' +
          'input, sample for sample.',
    }
  },

  // echo_reverb.py -- convolution with a tapped-delay IR.
  echo: (get, fs) =>
    echoExplanation(Number(get('delay')), fraction(get, 'feedback'), fraction(get, 'mix'), fs),

  // echo_reverb.py -- convolution with a synthesised room.
  reverb: (get, fs) => reverbExplanation(Number(get('decay')), fraction(get, 'mix'), fs),

  // The old combined card (hidden from the palette), whose params were still 0..1.
  echo_reverb: (get, fs) =>
    get('mode') === 'reverb'
      ? reverbExplanation(Number(get('decay')), Number(get('mix')), fs)
      : echoExplanation(Number(get('delay')), Number(get('feedback')), Number(get('mix')), fs),

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

  // compressor.py -- block-peak envelope follower + dB-domain gain computer.
  compressor: (get, fs) => {
    const attack = Number(get('attack'))
    const ratio = Number(get('ratio'))
    const threshold = Number(get('threshold'))
    const makeup = Number(get('makeup'))
    const auto = Boolean(get('auto_makeup'))
    // Same alpha the backend derives: a^(tau*fs) = 1/e.
    const alpha = Math.exp(-1 / (Math.max(attack, 0.01) / 1000) / fs)
    return {
      equation:
        `env[n] = α·env[n−1] + (1−α)|x|,  α = e^(−1/(${num(attack, 1)}ms·fs)) = ${num(alpha, 4)}` +
        `   ·   g_dB = −(1 − 1/${num(ratio)})·(L − ${num(threshold, 0)})` +
        (auto ? `   ·   makeup = −g_dB(peak)` : ''),
      effect:
        `Whenever the smoothed level L climbs above ${num(threshold, 0)} dBFS the signal is turned ` +
        `down by ${num(1 - 1 / ratio, 2)} dB per dB of excess` +
        (auto
          ? `, then everything is lifted by the amount the loudest peak lost, so the peaks end up ` +
            `where they were and the quiet parts come UP`
          : '') +
        `${Math.abs(makeup) > 0.05 ? `, plus a further ${signed(makeup)} dB` : ''}. ` +
        `That is why the output looks like a solid block rather than spikes — watch the crest ` +
        `factor above.`,
    }
  },

  // voice_changer.py -- the STFT rebuilt with a new phase; |X| is always kept.
  voice_changer: (get, fs) => {
    const mode = String(get('mode'))
    const mix = fraction(get, 'mix')
    const wet =
      mix < 0.995 ? ` Mixed ${Math.round(mix * 100)}% wet over the dry signal.` : ''

    if (mode === 'robot') {
      const f = Number(get('robot_freq'))
      const hop = Math.max(1, Math.round(fs / Math.max(f, 1)))
      return {
        equation: `Y[t,k] = |X[t,k]|·(−1)ᵏ,  H = fs / ${num(f, 0)} Hz = ${hop.toLocaleString()} samples`,
        effect:
          `Every frame keeps its magnitudes but gets the phase of one pulse at its centre, and ` +
          `frames sit ${hop.toLocaleString()} samples apart — so the output is a pulse train at ` +
          `${num(f, 0)} Hz shaped by the voice's spectrum. The words survive; the melody does not.` +
          wet,
      }
    }

    if (mode === 'whisper') {
      return {
        equation: 'Y[t,k] = |X[t,k]|·e^(jθ),  θ ~ U(−π, π]',
        effect:
          `The phase of every bin is replaced by noise, so no harmonic stays coherent from frame ` +
          `to frame. The energy envelope is unchanged — the waveform keeps its outline but loses ` +
          `its regular cycles, and the zero-crossing rate above jumps.` +
          wet,
      }
    }

    // chipmunk shifts up, monster down, by the same "How far".
    const amount = Math.abs(Number(get('semitones')))
    const st = mode === 'monster' ? -amount : amount
    const r = 2 ** (st / 12)
    return {
      equation: `|Y[t, round(${num(r, 3)}k)]| += |X[t,k]|,   ψ[t] = ψ[t−1] + ${num(r, 3)}·ω[t,k]·H`,
      effect:
        `Every bin moves to ${num(r, 3)}× its frequency (${signed(st)} st) and its phase is ` +
        `re-accumulated at the new rate, all at the original hop — so the pitch changes and the ` +
        `duration does not. Formants move with it, which is what gives the cartoon quality.` +
        wet,
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
