/**
 * Thin wrapper around the FastAPI backend.
 *
 * Requests go to /api/... which the Vite dev server proxies to http://127.0.0.1:8000
 * (see vite.config.ts), so there is no hard-coded port in the app code.
 */

import type {
  ArrangeClip,
  ArrangeTrack,
  BakeResult,
  DoctorReport,
  EnvPoint,
  OperationDef,
  ProcessGraphRequest,
  RecipeStep,
  SpectrogramData,
  UploadInfo,
  WireClip,
  WireStep,
  WireTrack,
} from './types'

const BASE = '/api'

async function detail(response: Response): Promise<string> {
  try {
    const body = await response.json()
    return body.detail ?? response.statusText
  } catch {
    return response.statusText
  }
}

/** GET /operations -- the tool catalogue that drives the palette and every control. */
export async function fetchOperations(): Promise<OperationDef[]> {
  const response = await fetch(`${BASE}/operations`)
  if (!response.ok) throw new Error(await detail(response))
  return response.json()
}

/** POST /upload -- send the WAV once; every later bake only quotes the returned file_id. */
export async function uploadFile(file: File): Promise<UploadInfo> {
  const form = new FormData()
  form.append('file', file)
  const response = await fetch(`${BASE}/upload`, { method: 'POST', body: form })
  if (!response.ok) throw new Error(await detail(response))
  return response.json()
}

/** The backend only cares about op / bypass / params -- uid is a UI concern. */
export function toWire(recipe: RecipeStep[]): WireStep[] {
  return recipe.map(({ op, bypass, params }) => ({ op, bypass, params }))
}

/** Arrange blocks as the backend's ClipSpec -- lane and uid are UI concerns. */
export function toWireClips(clips: ArrangeClip[]): WireClip[] {
  return clips.map(({ source, start, clipStart, clipEnd, gainDb, lane, fadeIn, fadeOut }) => ({
    source,
    start,
    clip_start: clipStart,
    clip_end: clipEnd,
    gain_db: gainDb,
    track: lane,
    fade_in: fadeIn,
    fade_out: fadeOut,
  }))
}

export function toWireTracks(tracks: ArrangeTrack[]): WireTrack[] {
  const curve = (points: EnvPoint[]) => points.map((p): [number, number] => [p.t, p.v])
  return tracks.map((track) => ({
    volume_db: track.volumeDb,
    pan: track.pan,
    mute: track.mute,
    solo: track.solo,
    volume_env: curve(track.volumeEnv),
    pan_env: curve(track.panEnv),
  }))
}

/**
 * POST /process -- render the project and hand back an object URL for the rendered WAV.
 *
 * `signal` comes from an AbortController owned by the caller: while Auto-Bake is on, a
 * new slider move should cancel the request that is already in flight instead of racing
 * it, otherwise a slow older bake can land after a newer one and show a stale waveform.
 */
export async function processGraph(
  request: ProcessGraphRequest,
  signal?: AbortSignal,
): Promise<BakeResult> {
  const response = await fetch(`${BASE}/process`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal,
    body: JSON.stringify(request),
  })
  if (!response.ok) throw new Error(await detail(response))

  const blob = await response.blob()
  return {
    url: URL.createObjectURL(blob),
    bakeId: response.headers.get('X-Bake-Id'),
    bakeMs: Number(response.headers.get('X-Bake-Ms') ?? 0),
    duration: Number(response.headers.get('X-Output-Duration') ?? 0),
  }
}

/**
 * GET /spectrogram/file/{file_id} or /spectrogram/bake/{bake_id} -- the hand-made
 * spectrogram (backend/app/dsp/spectrogram.py) as a raw uint8 grid plus its shape.
 */
export async function fetchSpectrogram(
  kind: 'file' | 'bake',
  id: string,
  signal?: AbortSignal,
): Promise<SpectrogramData> {
  const response = await fetch(`${BASE}/spectrogram/${kind}/${id}`, { signal })
  if (!response.ok) throw new Error(await detail(response))
  return {
    rows: Number(response.headers.get('X-Spec-Rows')),
    cols: Number(response.headers.get('X-Spec-Cols')),
    fMin: Number(response.headers.get('X-Spec-Fmin')),
    fMax: Number(response.headers.get('X-Spec-Fmax')),
    pixels: new Uint8Array(await response.arrayBuffer()),
  }
}

/**
 * GET /diagnose/file/{file_id} or /diagnose/bake/{bake_id} -- Signal Doctor's checklist
 * (backend/app/dsp/doctor.py) and the recipe steps that fix what it found.
 */
export async function fetchDiagnosis(
  kind: 'file' | 'bake',
  id: string,
  signal?: AbortSignal,
): Promise<DoctorReport> {
  const response = await fetch(`${BASE}/diagnose/${kind}/${id}`, { signal })
  if (!response.ok) throw new Error(await detail(response))
  return response.json()
}
