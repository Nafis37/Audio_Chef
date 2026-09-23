/**
 * One way of writing a time, used by every clock in the app: the ruler under each
 * waveform, the hover label, the header readout and the Arrange timeline.
 *
 * Always minutes:seconds ("0:07", "1:32") so a time reads the same whatever the clip's
 * length -- "12" on one ruler and "0:12" on another was the confusion this removes.
 * Fractions of a second only appear when they carry information.
 */

/** m:ss, with `decimals` digits after the seconds (0 = whole seconds). */
export function formatTime(seconds: number, decimals = 0): string {
  const safe = Number.isFinite(seconds) && seconds > 0 ? seconds : 0
  // Round ONCE at the requested precision, so 59.96 s with one decimal becomes "1:00.0"
  // rather than "0:60.0".
  const scale = 10 ** decimals
  const total = Math.round(safe * scale) / scale
  const minutes = Math.floor(total / 60)
  const rest = total - minutes * 60
  const whole = Math.floor(rest)
  const frac = decimals > 0 ? `.${Math.round((rest - whole) * scale).toString().padStart(decimals, '0')}` : ''
  return `${minutes}:${whole.toString().padStart(2, '0')}${frac}`
}

/**
 * Label steps a ruler may use, in seconds, each with the tick step that subdivides it
 * into round pieces (1 s -> 0.2 s ticks, 15 s -> 5 s ticks, 30 s -> 5 s ticks ...).
 */
const NICE_STEPS: Array<[label: number, tick: number]> = [
  [0.1, 0.02], [0.2, 0.05], [0.5, 0.1], [1, 0.2], [2, 0.5], [5, 1], [10, 2],
  [15, 5], [30, 5], [60, 10], [120, 30], [300, 60], [600, 120],
]

export interface RulerSpacing {
  /** Seconds between labelled notches. */
  label: number
  /** Seconds between small tick notches. */
  tick: number
  /** Digits after the seconds that the labels need. */
  decimals: number
}

/** The smallest round label step that keeps labels at least `minLabelPx` apart. */
export function rulerSpacing(duration: number, widthPx: number, minLabelPx = 72): RulerSpacing {
  const pxPerSec = duration > 0 && widthPx > 0 ? widthPx / duration : 100
  const [label, tick] =
    NICE_STEPS.find(([step]) => step * pxPerSec >= minLabelPx) ?? NICE_STEPS[NICE_STEPS.length - 1]
  return { label, tick, decimals: label < 1 ? 1 : 0 }
}
