/**
 * usePeaks(url): a file tab's waveform outline, for the blocks on the Arrange timeline.
 *
 * Decoded ONCE per file (a module-level cache of promises, keyed by the blob URL) and
 * reduced to a min/max pair per 10 ms bucket, so a block of any width can draw itself
 * from a slice of two small arrays instead of the whole signal.
 *
 * This is the RAW file.  A block plays the tab's PROCESSED output, which after a time
 * changing step (Cut & Trim, Speed, Silence Remover) is a different length -- the block
 * shows a warning in that case instead of pretending the outline is exact.
 */

import { useEffect, useState } from 'react'

export interface Peaks {
  /** Buckets per second. */
  rate: number
  min: Float32Array
  max: Float32Array
  duration: number
}

const BUCKETS_PER_SECOND = 100

const cache = new Map<string, Promise<Peaks>>()
let context: AudioContext | null = null

async function decode(url: string): Promise<Peaks> {
  context ??= new AudioContext()
  const bytes = await (await fetch(url)).arrayBuffer()
  const audio = await context.decodeAudioData(bytes)
  const per = Math.max(1, Math.round(audio.sampleRate / BUCKETS_PER_SECOND))
  const count = Math.ceil(audio.length / per)
  const min = new Float32Array(count)
  const max = new Float32Array(count)
  // Mono fold, the same average the backend processes.
  const channels = Array.from({ length: audio.numberOfChannels }, (_, c) => audio.getChannelData(c))
  const scale = 1 / channels.length
  for (let b = 0; b < count; b++) {
    let lo = 0
    let hi = 0
    const end = Math.min(audio.length, (b + 1) * per)
    for (let n = b * per; n < end; n++) {
      let v = 0
      for (const data of channels) v += data[n]
      v *= scale
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
    min[b] = lo
    max[b] = hi
  }
  return { rate: audio.sampleRate / per, min, max, duration: audio.duration }
}

/** The outline for `url`, or null while it decodes (or if the browser cannot). */
export function usePeaks(url: string | null): Peaks | null {
  const [result, setResult] = useState<{ url: string; peaks: Peaks } | null>(null)
  useEffect(() => {
    if (!url) return
    let alive = true
    let pending = cache.get(url)
    if (!pending) {
      pending = decode(url)
      cache.set(url, pending)
      // A format the browser cannot decode: forget it so nothing waits on it forever.
      pending.catch(() => cache.delete(url))
    }
    pending.then((peaks) => alive && setResult({ url, peaks })).catch(() => {})
    return () => {
      alive = false
    }
  }, [url])
  return result && result.url === url ? result.peaks : null
}

/** Drop a closed file's outline. */
export function forgetPeaks(url: string): void {
  cache.delete(url)
}
