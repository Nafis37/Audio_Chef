/**
 * Thin wrapper around the FastAPI backend.
 *
 * Requests go to /api/... which the Vite dev server proxies to http://127.0.0.1:8000
 * (see vite.config.ts), so there is no hard-coded port in the app code.
 */

import type { BakeResult, BakeStats, OperationDef, RecipeStep, UploadInfo } from './types'

const BASE = '/api'

async function detail(response: Response): Promise<string> {
  try {
    const body = await response.json()
    return body.detail ?? response.statusText
  } catch {
    return response.statusText
  }
}

/**
 * X-Bake-Stats is a convenience, not a contract: an older backend or a deployment that
 * forgot to expose the header should cost the numbers panel, not the bake.
 */
function readStats(response: Response): BakeStats | null {
  const raw = response.headers.get('X-Bake-Stats')
  if (!raw) return null
  try {
    return JSON.parse(raw) as BakeStats
  } catch {
    return null
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

/**
 * POST /process -- run the recipe and hand back an object URL for the rendered WAV.
 *
 * `signal` comes from an AbortController owned by the caller: while Auto-Bake is on, a
 * new slider move should cancel the request that is already in flight instead of racing
 * it, otherwise a slow older bake can land after a newer one and show a stale waveform.
 */
export async function processRecipe(
  fileId: string,
  recipe: RecipeStep[],
  signal?: AbortSignal,
): Promise<BakeResult> {
  const response = await fetch(`${BASE}/process`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal,
    body: JSON.stringify({
      file_id: fileId,
      // The backend only cares about op / bypass / params -- uid is a UI concern.
      recipe: recipe.map(({ op, bypass, params }) => ({ op, bypass, params })),
    }),
  })
  if (!response.ok) throw new Error(await detail(response))

  const blob = await response.blob()
  return {
    url: URL.createObjectURL(blob),
    bakeMs: Number(response.headers.get('X-Bake-Ms') ?? 0),
    duration: Number(response.headers.get('X-Output-Duration') ?? 0),
    stats: readStats(response),
  }
}
