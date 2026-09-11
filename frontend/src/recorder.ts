/**
 * useRecorder -- microphone capture, as a state machine in a hook.
 *
 * Why MediaRecorder plus a transcode, rather than an AudioWorklet capturing raw PCM:
 * MediaRecorder hands back WebM/Opus (Chrome, Firefox) or MP4/AAC (Safari), none of which
 * libsndfile can open, so a take has to be converted to WAV before /upload will take it.
 * decodeAudioData() does that with the decoder the browser already ships, which is far
 * less machinery than a worklet plus a hand-managed ring buffer -- and there is nothing to
 * gain from touching the samples live, since the whole point is to run the recipe on them
 * afterwards.
 *
 * This hook is used from the leaf Recorder component on purpose.  The meter and the clock
 * repaint several times a second; owning that state in App would re-render both
 * drag-and-drop columns at the same rate for the duration of the recording.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { encodeWav } from './wav'

export type RecorderState = 'idle' | 'requesting' | 'recording' | 'encoding' | 'preview'

export interface Take {
  /** Ready for App's handleFile -- named .wav, because /upload checks the extension. */
  file: File
  /** Object URL for the preview player.  Revoked by reset(). */
  url: string
  seconds: number
}

/** Five minutes -- about 28 MB of mono 48 kHz 16-bit, well under the backend's 100 MB. */
export const MAX_RECORDING_MS = 5 * 60 * 1000

/** The meter and clock repaint at this rate, not at rAF rate.  See the loop below. */
const UI_INTERVAL_MS = 50

/** Speech RMS lands around 0.05-0.2, so the raw value alone barely moves the bar. */
const METER_GAIN = 4

export function useRecorder() {
  const [state, setState] = useState<RecorderState>('idle')
  const [elapsedMs, setElapsed] = useState(0)
  const [level, setLevel] = useState(0)
  const [take, setTake] = useState<Take | null>(null)
  const [error, setError] = useState<string | null>(null)

  const stream = useRef<MediaStream | null>(null)
  const recorder = useRef<MediaRecorder | null>(null)
  const context = useRef<AudioContext | null>(null)
  const chunks = useRef<Blob[]>([])
  const startedAt = useRef(0)
  const frame = useRef<number | null>(null)
  const lastPaint = useRef(0)
  // The take's URL is also held in a ref so unmount can revoke it without the cleanup
  // effect depending on the state (which would re-run the cleanup on every take).
  const takeUrl = useRef<string | null>(null)

  /** Everything the browser holds on our behalf: the mic light, the loop. */
  const releaseHardware = useCallback(() => {
    if (frame.current !== null) cancelAnimationFrame(frame.current)
    frame.current = null
    // Until every track is stopped the tab keeps showing "recording" and the mic stays
    // held against other applications -- stopping the MediaRecorder alone does not do it.
    stream.current?.getTracks().forEach((track) => track.stop())
    stream.current = null
    recorder.current = null
  }, [])

  useEffect(
    () => () => {
      releaseHardware()
      void context.current?.close()
      context.current = null
      if (takeUrl.current) URL.revokeObjectURL(takeUrl.current)
    },
    [releaseHardware],
  )

  /** Drops the current take and returns to idle.  Used by Retake AND by Use this take:
   *  handleFile builds its own object URL from the File, so ours is finished either way. */
  const reset = useCallback(() => {
    if (takeUrl.current) URL.revokeObjectURL(takeUrl.current)
    takeUrl.current = null
    setTake(null)
    setElapsed(0)
    setLevel(0)
    setError(null)
    setState('idle')
  }, [])

  const stop = useCallback(() => {
    // onstop (wired in start) is what actually finalises; this only asks for it.
    if (recorder.current?.state === 'recording') recorder.current.stop()
  }, [])

  const start = useCallback(async () => {
    if (state !== 'idle') return
    setError(null)
    setState('requesting')

    let media: MediaStream
    try {
      media = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          // The browser's own AGC / noise suppression / echo cancellation is a second,
          // invisible DSP chain -- it would be doing this app's Noise Remover and
          // Compressor's job, badly, on a signal the user came here to process raw.
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      })
    } catch (err) {
      setState('idle')
      setError(describeMicError(err))
      return
    }

    stream.current = media
    chunks.current = []

    const audio = new AudioContext()
    context.current = audio
    const analyser = audio.createAnalyser()
    analyser.fftSize = 1024
    audio.createMediaStreamSource(media).connect(analyser)

    const capture = new MediaRecorder(media)
    recorder.current = capture
    capture.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.current.push(event.data)
    }
    capture.onstop = () => void finalise()
    // A timeslice keeps chunks arriving during the take rather than one blob at the end.
    capture.start(1000)

    startedAt.current = performance.now()
    lastPaint.current = 0
    setElapsed(0)
    setState('recording')

    const samples = new Uint8Array(analyser.fftSize)
    const tick = () => {
      frame.current = requestAnimationFrame(tick)
      const now = performance.now()
      const running = now - startedAt.current
      if (running >= MAX_RECORDING_MS) {
        stop()
        return
      }
      // One loop drives both readouts, but it repaints on an interval: at rAF rate this
      // would set state 60 times a second for a bar nobody can read that fast.
      if (now - lastPaint.current < UI_INTERVAL_MS) return
      lastPaint.current = now

      analyser.getByteTimeDomainData(samples)
      let sum = 0
      for (let i = 0; i < samples.length; i += 1) {
        const centred = (samples[i] - 128) / 128   // bytes are 0..255 around a 128 zero
        sum += centred * centred
      }
      setLevel(Math.min(1, Math.sqrt(sum / samples.length) * METER_GAIN))
      setElapsed(running)
    }

    /** Turns the recorded chunks into an uploadable WAV File. */
    async function finalise() {
      const seconds = (performance.now() - startedAt.current) / 1000
      setState('encoding')
      setLevel(0)
      // Free the mic the instant capture ends -- decoding takes a moment on a long take,
      // and leaving the indicator lit through it looks like we are still listening.
      releaseHardware()

      try {
        const recorded = new Blob(chunks.current, { type: chunks.current[0]?.type })
        const decoded = await audio.decodeAudioData(await recorded.arrayBuffer())
        const wav = encodeWav(decoded)
        const url = URL.createObjectURL(wav)
        takeUrl.current = url
        setTake({
          file: new File([wav], `recording-${stamp()}.wav`, { type: 'audio/wav' }),
          url,
          seconds: decoded.duration || seconds,
        })
        setState('preview')
      } catch (err) {
        const why = err instanceof Error ? err.message : 'unknown error'
        setError(`Could not encode the recording: ${why}`)
        setState('idle')
      } finally {
        await audio.close()
        context.current = null
      }
    }

    frame.current = requestAnimationFrame(tick)
  }, [state, releaseHardware, stop])

  return {
    state,
    elapsedMs,
    level,
    take,
    error,
    start,
    stop,
    reset,
    supported: isSupported(),
  }
}

/** getUserMedia is undefined outside a secure context -- plain http on a LAN IP, notably
 *  (localhost is always secure, so `npm run dev` is fine; `vite --host` is not). */
function isSupported() {
  return (
    typeof window !== 'undefined' &&
    typeof window.MediaRecorder !== 'undefined' &&
    typeof navigator.mediaDevices?.getUserMedia === 'function'
  )
}

function describeMicError(err: unknown): string {
  if (err instanceof DOMException) {
    if (err.name === 'NotAllowedError' || err.name === 'SecurityError') {
      return 'Microphone permission denied — allow it in the browser’s site settings.'
    }
    if (err.name === 'NotFoundError' || err.name === 'OverconstrainedError') {
      return 'No microphone found.'
    }
    if (err.name === 'NotReadableError') {
      return 'The microphone is in use by another application.'
    }
  }
  return err instanceof Error ? err.message : 'Could not start recording.'
}

/** Local-time yyyymmdd-hhmmss, so takes sort and the filename says when it was made. */
function stamp() {
  const now = new Date()
  const pad = (value: number) => String(value).padStart(2, '0')
  return (
    `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}` +
    `-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`
  )
}
