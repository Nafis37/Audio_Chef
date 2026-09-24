/**
 * The top bar: file name, Auto-Bake toggle, manual Bake, Export audio, and status.
 *
 * Auto-Bake is the CyberChef behaviour -- every recipe or slider change immediately
 * re-runs the pipeline.  It can be switched off for long files, where each bake costs
 * real DSP time and you would rather stage several changes and bake once.
 *
 * `fileInfo` (duration / rate / channels) is its own chip rather than sharing the status
 * span: it used to be written into `status` and then immediately overwritten by the next
 * bake message, so the thing you actually wanted to keep reading was the first to go.
 */

import { ChefHat, Download, Loader2, Play, Zap } from 'lucide-react'
import { memo } from 'react'

interface Props {
  filename: string | null
  fileInfo: string | null
  autoBake: boolean
  onToggleAutoBake: () => void
  onBake: () => void
  onExport: () => void
  canBake: boolean
  canExport: boolean
  busy: boolean
  status: string
  error: string | null
}

function ToolbarImpl({
  filename,
  fileInfo,
  autoBake,
  onToggleAutoBake,
  onBake,
  onExport,
  canBake,
  canExport,
  busy,
  status,
  error,
}: Props) {
  return (
    <header className="flex flex-wrap items-center gap-3 border-b border-[var(--chef-border)] bg-[var(--chef-panel)] px-4 py-2.5">
      <div className="flex items-center gap-2">
        <ChefHat className="size-5 text-[var(--chef-accent-strong)]" />
        <span className="text-sm font-semibold tracking-wide">Audio Chef</span>
      </div>

      {filename && (
        <span className="max-w-[220px] truncate rounded border border-[var(--chef-border)] px-2 py-1 font-mono text-xs text-[var(--chef-muted)]">
          {filename}
        </span>
      )}

      {fileInfo && (
        <span className="rounded border border-[var(--chef-border)] px-2 py-1 font-mono text-xs text-[var(--chef-muted)]">
          {fileInfo}
        </span>
      )}

      <div className="ml-auto flex items-center gap-2">
        <span
          className={`text-xs ${error ? 'text-rose-600' : 'text-[var(--chef-muted)]'}`}
        >
          {error ?? status}
        </span>

        <button
          type="button"
          onClick={onToggleAutoBake}
          className={`flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium transition ${
            autoBake
              ? 'border-[var(--chef-accent-strong)] bg-[var(--chef-accent)]/30 text-[var(--chef-accent-strong)]'
              : 'border-[var(--chef-border)] text-[var(--chef-muted)] hover:text-[var(--chef-text)]'
          }`}
          title="Re-bake automatically on every change"
        >
          <Zap className="size-3.5" /> Auto-Bake {autoBake ? 'on' : 'off'}
        </button>

        <button
          type="button"
          onClick={onBake}
          disabled={!canBake || busy}
          className="flex items-center gap-1.5 rounded-md border border-[var(--chef-border)] px-3 py-1.5 text-xs font-medium transition hover:border-[var(--chef-accent-strong)] disabled:opacity-30"
        >
          {busy ? <Loader2 className="size-3.5 animate-spin" /> : <Play className="size-3.5" />}
          Bake
        </button>

        {/* Deliberately styled like the other secondary buttons: this app is for
            listening, and downloading is something you reach for, not something it
            should be pushing at you after every bake. */}
        <button
          type="button"
          onClick={onExport}
          disabled={!canExport}
          title="Save the baked audio"
          className="flex items-center gap-1.5 rounded-md border border-[var(--chef-border)] px-3 py-1.5 text-xs font-medium transition hover:border-[var(--chef-accent-strong)] disabled:opacity-30"
        >
          <Download className="size-3.5" /> Export audio
        </button>
      </div>
    </header>
  )
}

export const Toolbar = memo(ToolbarImpl)
