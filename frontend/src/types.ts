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
  /** `source` holds the id of another loaded source -- see the `control` note below. */
  type: 'float' | 'bool' | 'enum' | 'source'
  default: ParamValue
  unit: string
  /**
   * `source` is the one control whose OPTIONS the backend cannot supply: which files are
   * loaded is a fact about this browser session, not about the DSP.  The schema declares
   * the kind of widget and the client fills the list -- see sources.tsx.
   */
  control: 'slider' | 'number' | 'select' | 'toggle' | 'source'
  min?: number
  max?: number
  step?: number
  options?: string[] | null
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
 * One loaded file and everything that belongs to it.
 *
 * `recipe` and `activeUid` live INSIDE the source rather than in parallel lookup maps
 * keyed by id: removing a source is then one filter() and cannot leave an orphaned chain
 * behind, and an activeUid is meaningless outside the recipe it points into, so keeping
 * the two together makes them impossible to desync.
 */
export interface Source {
  /** Client-minted handle ("s1").  What an `assemble` card's `source` param stores. */
  id: string
  /** Distinct from `id`: the same upload may be loaded twice with two different chains. */
  fileId: string
  filename: string
  /** Blob URL of the RAW file, drawn locally by the Input waveform. */
  url: string
  sampleRate: number
  channels: number
  /** Raw length in seconds -- what this source's region handles are measured against. */
  duration: number
  recipe: RecipeStep[]
  /** uid of the step whose region is drawn, if any.  Per source, not global. */
  activeUid: string | null
}

/** Which chain column 2 is editing: a source id, or the master chain. */
export type ChainId = string
export const MASTER: ChainId = 'master'

/** The body of POST /process. */
export interface ProcessGraphRequest {
  sources: { id: string; file_id: string; recipe: WireStep[] }[]
  master: WireStep[]
  output_source: string
  apply_master: boolean
}

/** A recipe step as the backend wants it -- `uid` is a UI concern and is stripped. */
export interface WireStep {
  op: string
  bypass: boolean
  params: Record<string, ParamValue>
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

/** One row of the per-source summary in X-Bake-Stats.  Deliberately compact: it rides
 *  back on every bake inside a response header. */
export interface SourceReport {
  id: string
  frames: number
  duration: number
  steps_applied: number
  pre_clip_peak: number
  truncated: boolean
}

/** The X-Bake-Stats header: both buffers measured, plus what the graph walk did. */
export interface BakeStats {
  sample_rate: number
  /** The RAW buffer of the rendered source, before its chain ran. */
  input: Measures
  output: Measures
  /** The one rate the whole evaluation ran at (the highest among the sources). */
  project_sample_rate: number
  /** Ids whose file rate differed and were converted -- so a pitch change is explained. */
  resampled: string[]
  /** Only the sources the walk actually reached; an unreferenced source is absent. */
  sources: SourceReport[]
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
