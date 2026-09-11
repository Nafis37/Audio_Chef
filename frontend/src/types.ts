/**
 * Shared types.
 *
 * `OperationDef` / `ParamDef` mirror exactly what GET /operations returns, which is
 * generated from the OPERATIONS catalogue in backend/app/dsp/dsp_engine.py.  The palette
 * and every slider are rendered from that data, so adding a parameter on the backend
 * makes it appear in the UI without touching this file.
 */

export type ParamValue = number | string | boolean

export interface ParamDef {
  name: string
  label: string
  type: 'float' | 'bool' | 'enum'
  default: ParamValue
  unit: string
  control: 'slider' | 'number' | 'select' | 'toggle'
  min?: number
  max?: number
  step?: number
  options?: string[]
}

export interface OperationDef {
  id: string
  label: string
  icon: string
  description: string
  params: ParamDef[]
}

/** One card in the recipe column.  `uid` keeps React keys stable when the same
 *  operation is added twice (e.g. echo followed by reverb). */
export interface RecipeStep {
  uid: string
  op: string
  bypass: boolean
  params: Record<string, ParamValue>
}

/** Response of POST /upload. */
export interface UploadInfo {
  file_id: string
  filename: string
  sample_rate: number
  channels: number
  duration: number
  frames: number
}

/**
 * Measurements of one buffer, as computed by backend/app/dsp/analysis.py.
 *
 * Amplitudes are in [0, 1] and the `_db` fields are dBFS (0 dB = full scale), floored at
 * -120 dB so digital silence stays a finite number.
 */
export interface Measures {
  duration: number
  frames: number
  peak: number
  peak_db: number
  rms: number
  rms_db: number
  /** peak_db - rms_db: how spiky the waveform is relative to how loud it is. */
  crest_db: number
  dc: number
  /** Zero crossings per second -- a transform-free brightness proxy. */
  zcr: number
  /** Magnitude-weighted mean frequency, in Hz. */
  centroid_hz: number
}

/** The X-Bake-Stats header: both buffers measured, plus what the recipe fold did. */
export interface BakeStats {
  sample_rate: number
  input: Measures
  output: Measures
  /** max |y| BEFORE run_recipe clamps to +-1 -- >1 means the output was clipped. */
  pre_clip_peak: number
  clipped: number
  steps_applied: number
  steps_bypassed: number
  /** A step emptied the buffer and the fold stopped early. */
  truncated: boolean
}

/** Result of a bake: the rendered audio plus what the backend reported about it. */
export interface BakeResult {
  url: string
  bakeMs: number
  duration: number
  /** null if the backend did not send (or CORS hid) the X-Bake-Stats header. */
  stats: BakeStats | null
}
