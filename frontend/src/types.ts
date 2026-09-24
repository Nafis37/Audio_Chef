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
  /** An Arrange tab's track settings, by track index; a track past the end is default. */
  tracks: ArrangeTrack[]
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
 * `lane` is the track it sits on (backend dsp/arrange.py).
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
  /** Equal-power fade lengths, seconds.  Overlaps on one track crossfade on their own. */
  fadeIn: number
  fadeOut: number
}

/** One point of an automation curve: time (s) and value (dB, or pan -1 .. +1). */
export interface EnvPoint {
  t: number
  v: number
}

/** One track of an Arrange timeline.  A curve with points overrides its knob. */
export interface ArrangeTrack {
  volumeDb: number
  /** -1 = left, +1 = right. */
  pan: number
  mute: boolean
  solo: boolean
  volumeEnv: EnvPoint[]
  panEnv: EnvPoint[]
}

/** An Arrange block as the backend wants it. */
export interface WireClip {
  source: string
  start: number
  clip_start: number
  clip_end: number
  gain_db: number
  track: number
  fade_in: number
  fade_out: number
}

/** A track as the backend wants it. */
export interface WireTrack {
  volume_db: number
  pan: number
  mute: boolean
  solo: boolean
  volume_env: [number, number][]
  pan_env: [number, number][]
}

/** The body of POST /process. */
export interface ProcessGraphRequest {
  sources: {
    id: string
    file_id?: string
    clips?: WireClip[]
    tracks?: WireTrack[]
    recipe: WireStep[]
  }[]
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

/** Result of a bake: the rendered audio plus what the backend reported about it. */
export interface BakeResult {
  url: string
  /** Handle for GET /spectrogram/bake/{id}; null from a backend that does not send it. */
  bakeId: string | null
  bakeMs: number
  duration: number
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
