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
  /** Human names for enum values ("trim" -> "Keep the selection"). */
  option_labels?: Record<string, string>
  /** One plain sentence: the control's tooltip. */
  help?: string
  /** Tucked into the card's "Advanced" section. */
  advanced?: boolean
  /** Only shown while another param holds one of these values: {"mode": ["reverb"]}. */
  show_when?: Record<string, string[]> | null
  /**
   * The value shown and sent is in display units (e.g. 0..100 %); the backend multiplies
   * by this before the DSP sees it.  Informational here -- the UI never converts.
   */
  scale?: number | null
  /** Slider moves in equal ratios (e.g. every octave the same travel); the value is unchanged. */
  log?: boolean
}

export interface OperationDef {
  id: string
  label: string
  icon: string
  /** Palette group heading. */
  category: string
  /** What it does, in one plain sentence. */
  summary: string
  /** The technique, in one line -- shown beside the maths. */
  how: string
  /** What a listener should hear and see change. */
  listen_for: string
  /** One-click settings: name -> the params it moves (display units). */
  quick?: Record<string, Record<string, ParamValue>>
  /** Served so old recipes still render, but not offered in the palette. */
  hidden?: boolean
  params: ParamDef[]
}

/** Is this param relevant given the step's current values?  (See ParamDef.show_when.) */
export function isParamShown(param: ParamDef, params: Record<string, ParamValue>): boolean {
  if (!param.show_when) return true
  return Object.entries(param.show_when).every(([other, values]) =>
    values.includes(String(params[other])),
  )
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
  /**
   * 'file': a loaded upload.  'arrange': a timeline whose audio is the sum of its
   * `clips`, each a piece of another tab's processed output (backend dsp/arrange.py).
   */
  kind: 'file' | 'arrange'
  /** The tab's colour (colors.ts), fixed for the life of the source. */
  color: string
  /** Distinct from `id`: the same upload may be loaded twice with two different chains.
   *  null for an Arrange tab. */
  fileId: string | null
  /** The tab's title: the file's name, or "Arrangement 2". */
  filename: string
  /** Blob URL of the RAW file, drawn locally by the Input waveform; null for Arrange. */
  url: string | null
  /** An Arrange tab's blocks; empty for a file tab. */
  clips: ArrangeClip[]
  sampleRate: number
  channels: number
  /** Raw length in seconds -- what this source's region handles are measured against. */
  duration: number
  recipe: RecipeStep[]
  /** uid of the step whose region is drawn, if any.  Per source, not global. */
  activeUid: string | null
}

/**
 * One block on an Arrange timeline.  Seconds throughout; `clipEnd` 0 = the source's end.
 * `lane` is UI-only (the backend just sums the blocks).
 */
export interface ArrangeClip {
  uid: string
  /** The file tab it plays (its PROCESSED output). */
  source: string
  /** Where on the timeline it starts. */
  start: number
  clipStart: number
  clipEnd: number
  gainDb: number
  lane: number
}

/** An Arrange block as the backend wants it. */
export interface WireClip {
  source: string
  start: number
  clip_start: number
  clip_end: number
  gain_db: number
}

/** The body of POST /process. */
export interface ProcessGraphRequest {
  sources: { id: string; file_id?: string; clips?: WireClip[]; recipe: WireStep[] }[]
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

/** What one Voice Match card reported: calibration feedback, never a similarity score. */
export interface VoiceMatchReport {
  /** The source whose recipe holds the card. */
  id: string
  /** Seconds of voiced speech found in each calibration take. */
  cal_voiced_s: number
  ref_voiced_s: number
  /** Median pitch of each speaker's calibration take, Hz. */
  cal_f0_hz: number
  ref_f0_hz: number
  /** Spread (standard deviation) of each speaker's pitch, in semitones. */
  cal_f0_spread_st: number
  ref_f0_spread_st: number
  /** DTW path cost relative to random frame pairs: lower = the takes lined up better. */
  align_cost: number
  /** Paired calibration frames available to look up. */
  codebook: number
  /** Voiced seconds in the speech being converted. */
  voiced_s: number
  /** Share of voiced frames unlike the calibration, which got less timbre change. */
  reduced_fraction: number
  /** Median pitch shift applied, semitones. */
  shift_st: number
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
  /** max |y| BEFORE the final fit to full scale -- >1 means it had to be turned down. */
  pre_clip_peak: number
  /** How many samples WOULD have gone past full scale. */
  clipped: number
  /** <= 0: how far the whole output was turned down to fit instead of clipping. */
  normalised_db?: number
  steps_applied: number
  steps_bypassed: number
  /** A step emptied the buffer and the fold stopped early. */
  truncated: boolean
  /** One row per Voice Match card that ran; absent when there were none. */
  voice_match?: VoiceMatchReport[]
}

/** Result of a bake: the rendered audio plus what the backend reported about it. */
export interface BakeResult {
  url: string
  /** Handle for GET /spectrogram/bake/{id}; null from a backend that does not send it. */
  bakeId: string | null
  bakeMs: number
  duration: number
  /** null if the backend did not send (or CORS hid) the X-Bake-Stats header. */
  stats: BakeStats | null
}

/** A spectrogram as served by GET /spectrogram/...: uint8 grid, row 0 = highest frequency. */
export interface SpectrogramData {
  rows: number
  cols: number
  fMin: number
  fMax: number
  pixels: Uint8Array
}

/** A labelled horizontal line on a spectrogram, e.g. a filter's cutoff. */
export interface SpectrogramMarker {
  hz: number
  label: string
}

/** One row of Signal Doctor's checklist (backend/app/dsp/doctor.py). */
export interface DoctorFinding {
  id: string
  status: 'ok' | 'warn' | 'bad'
  title: string
  /** The measurement behind the verdict, e.g. "floor -40 dBFS · 77 % above 2 kHz". */
  value: string
  detail: string
  fix: WireStep[]
}

/** GET /diagnose/...: the checklist plus the combined prescription, in recipe order. */
export interface DoctorReport {
  findings: DoctorFinding[]
  fix: WireStep[]
}
