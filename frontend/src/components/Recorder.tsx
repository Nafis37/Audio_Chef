/**
 * Record from the microphone, as an alternative to dropping a WAV in.
 *
 * Mounted twice in column 3: once as the big button on the empty-state card, once as a
 * compact icon button in the Listen header so a take can replace the input at any time.
 * Only the idle affordance differs between the two -- once recording starts, both open the
 * same panel, because the meter and the transport need the room either way.
 *
 * The take never reaches the backend until "Use this take" is pressed.  A recording that
 * caught the wrong moment is the common case, and uploading every one of them would also
 * re-bake the whole recipe against audio the user is about to throw away.
 */

import { Check, Loader2, Mic, RotateCcw, Square } from 'lucide-react'
import { memo } from 'react'
import { MAX_RECORDING_MS, useRecorder } from '../recorder'

interface Props {
  /** Handed the finished take.  App passes handleFile, the same one the file input uses. */
  onTake: (file: File) => void
  variant: 'large' | 'compact'
}

/** m:ss, from milliseconds. */
function clock(ms: number) {
  const total = Math.floor(ms / 1000)
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}

function RecorderImpl({ onTake, variant }: Props) {
  const { state, elapsedMs, level, take, error, start, stop, reset, supported } =
    useRecorder()

  // No MediaRecorder, or an insecure origin: say nothing rather than offer a dead button.
  if (!supported) return null

  // --- Idle: just the affordance, in whichever shape the call site needs ---------------
  if (state === 'idle' || state === 'requesting') {
    const busy = state === 'requesting'
    return (
      <div className={variant === 'large' ? 'flex flex-col items-center gap-2' : 'contents'}>
        <button
          type="button"
          onClick={() => void start()}
          disabled={busy}
          title="Record from your microphone"
          className={
            variant === 'large'
              ? 'flex items-center gap-2 rounded-md border border-[var(--chef-accent-strong)] bg-[var(--chef-accent)]/25 px-4 py-2 text-sm font-medium text-[var(--chef-accent-strong)] transition hover:bg-[var(--chef-accent)]/40 disabled:opacity-50'
              : 'flex items-center gap-1.5 rounded-md border border-[var(--chef-border)] px-2.5 py-1 text-xs transition hover:border-[var(--chef-accent-strong)] disabled:opacity-50'
          }
        >
          {busy ? <Loader2 className="size-4 animate-spin" /> : <Mic className="size-4" />}
          {variant === 'large' ? (busy ? 'Waiting for the mic…' : 'Record with your mic') : 'Record'}
        </button>
        {error && (
          <p className="max-w-xs text-center text-xs text-rose-600">{error}</p>
        )}
      </div>
    )
  }

  // --- Encoding: decodeAudioData on a long take is not instant -------------------------
  if (state === 'encoding') {
    return (
      <Panel>
        <span className="flex items-center gap-2 text-xs text-[var(--chef-muted)]">
          <Loader2 className="size-4 animate-spin" /> Encoding the take…
        </span>
      </Panel>
    )
  }

  // --- Preview: hear it before it costs an upload and a bake ---------------------------
  if (state === 'preview' && take) {
    return (
      <Panel>
        <div className="flex items-center justify-between gap-3">
          <span className="text-xs font-medium">Take · {take.seconds.toFixed(1)}s</span>
          <button
            type="button"
            onClick={reset}
            className="flex items-center gap-1.5 rounded-md border border-[var(--chef-border)] px-2.5 py-1 text-xs transition hover:border-rose-600 hover:text-rose-600"
          >
            <RotateCcw className="size-3.5" /> Retake
          </button>
        </div>

        {/* A plain audio element, not a WaveformViewer: that component builds its
            wavesurfer instance once on mount and is meant for the two long-lived
            Input/Output panels, not for something this transient. */}
        <audio controls src={take.url} className="w-full" />

        <button
          type="button"
          onClick={() => {
            onTake(take.file)
            reset()
          }}
          className="flex items-center justify-center gap-2 rounded-md border border-[var(--chef-accent-strong)] bg-[var(--chef-accent)]/25 px-3 py-1.5 text-xs font-medium text-[var(--chef-accent-strong)] transition hover:bg-[var(--chef-accent)]/40"
        >
          <Check className="size-4" /> Use this take
        </button>
      </Panel>
    )
  }

  // --- Recording ----------------------------------------------------------------------
  return (
    <Panel>
      <div className="flex items-center justify-between gap-3">
        <span className="flex items-center gap-2 text-xs font-medium">
          <span className="size-2 animate-pulse rounded-full bg-rose-600" />
          REC
          <span className="font-mono text-[var(--chef-muted)]">
            {clock(elapsedMs)} / {clock(MAX_RECORDING_MS)}
          </span>
        </span>
        <button
          type="button"
          onClick={stop}
          className="flex items-center gap-1.5 rounded-md border border-[var(--chef-accent-strong)] px-2.5 py-1 text-xs font-medium text-[var(--chef-accent-strong)] transition hover:bg-[var(--chef-accent)]/25"
        >
          <Square className="size-3.5" /> Stop
        </button>
      </div>

      {/* The level bar earns its place: a clock alone cannot tell a muted or wrongly
          selected microphone from a genuinely silent room until after you stop. */}
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-[var(--chef-inset)]">
        <div
          className="h-full rounded-full bg-[var(--chef-accent-strong)] transition-[width] duration-75"
          style={{ width: `${Math.round(level * 100)}%` }}
        />
      </div>
      <p className="text-[11px] text-[var(--chef-muted)]">
        Input level — speak and watch it move.
      </p>
    </Panel>
  )
}

/** Shared shell for every non-idle state, so the two call sites look identical there. */
function Panel({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex w-full max-w-sm flex-col gap-2 rounded-lg border border-[var(--chef-border)] bg-[var(--chef-surface)] p-3 text-left">
      {children}
    </div>
  )
}

export const Recorder = memo(RecorderImpl)
