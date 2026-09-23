/**
 * One browser-style tab per loaded file, between the toolbar and the three columns.
 *
 * It sits full width rather than inside a column because it is project-level, like the
 * toolbar: the open tab is what columns 2 and 3 are both scoped BY -- its recipe in the
 * middle, its input and output on the right.
 *
 * Each tab does three jobs, which are deliberately different clicks:
 *   - the body opens it (its recipe, its waveforms)
 *   - the download icon saves it: that file run through ITS recipe, as a WAV
 *   - the x closes it
 * Every file carries its own colour (colors.ts) so the tab, and the recipe header below
 * it, say at a glance which file you are working on.
 */

import { Download, LayoutPanelTop, Plus, X } from 'lucide-react'
import { memo } from 'react'
import { Recorder } from './Recorder'
import type { Source } from '../types'

interface Props {
  sources: Source[]
  /** The open tab. */
  activeId: string | null
  accept: string
  onSelect: (id: string) => void
  onSave: (id: string) => void
  onClose: (id: string) => void
  onPick: (event: React.ChangeEvent<HTMLInputElement>) => void
  onRecord: (file: File) => void
  /** Opens a new Arrange (timeline) tab. */
  onArrange: () => void
}

function SourceTabsImpl({
  sources,
  activeId,
  accept,
  onSelect,
  onSave,
  onClose,
  onPick,
  onRecord,
  onArrange,
}: Props) {
  if (sources.length === 0) return null

  return (
    // The strip's bottom border is drawn by the inactive tabs and the filler, not the
    // container, so the ACTIVE tab can leave it out and read as joined to the page.
    <div className="flex items-end gap-1 bg-[var(--chef-bg)] px-3 pt-2">
      <div className="flex min-w-0 flex-1 items-end gap-1 overflow-x-auto" role="tablist">
        {sources.map((source) => {
          const active = source.id === activeId
          const steps = source.recipe.length
          const arrange = source.kind === 'arrange'
          return (
            <div
              key={source.id}
              role="tab"
              aria-selected={active}
              className={`group flex min-w-[140px] max-w-[240px] shrink-0 items-center gap-1.5 rounded-t-lg border border-t-[3px] pl-2.5 pr-1 text-xs transition ${
                active
                  ? 'border-[var(--chef-border)] border-b-transparent bg-[var(--chef-panel)] py-2'
                  : 'border-transparent border-b-[var(--chef-border)] bg-[var(--chef-inset)] py-1.5 opacity-80 hover:opacity-100'
              }`}
              style={{ borderTopColor: source.color }}
            >
              <button
                type="button"
                onClick={() => onSelect(source.id)}
                title={
                  arrange
                    ? `${source.filename} — ${source.clips.length} blocks`
                    : `${source.filename} — ${source.duration.toFixed(2)}s · ${source.sampleRate} Hz`
                }
                className="flex min-w-0 flex-1 items-center gap-1.5 text-left"
              >
                {arrange ? (
                  <LayoutPanelTop className="size-3 shrink-0" style={{ color: source.color }} aria-hidden />
                ) : (
                  <span
                    className="size-2.5 shrink-0 rounded-full"
                    style={{ backgroundColor: source.color }}
                    aria-hidden
                  />
                )}
                <span className={`truncate ${active ? 'font-medium' : ''}`}>{source.filename}</span>
                {steps > 0 && (
                  <span className="shrink-0 font-mono text-[10px] text-[var(--chef-muted)]">
                    {steps}
                  </span>
                )}
              </button>

              <button
                type="button"
                onClick={() => onSave(source.id)}
                title={`Save ${source.filename} with its recipe applied (WAV)`}
                className="rounded p-1 text-[var(--chef-muted)] transition hover:bg-[var(--chef-hover)] hover:text-[var(--chef-accent-strong)]"
              >
                <Download className="size-3" />
              </button>
              <button
                type="button"
                onClick={() => onClose(source.id)}
                title={`Close ${source.filename}`}
                className="rounded p-1 text-[var(--chef-muted)] transition hover:bg-[var(--chef-hover)] hover:text-rose-600"
              >
                <X className="size-3" />
              </button>
            </div>
          )
        })}

        {/* A new tab, browser style. */}
        <label
          title="Open another audio file in a new tab"
          className="mb-1 flex shrink-0 cursor-pointer items-center rounded-md p-1.5 text-[var(--chef-muted)] transition hover:bg-[var(--chef-hover)] hover:text-[var(--chef-text)]"
        >
          <Plus className="size-4" />
          <input type="file" accept={accept} className="hidden" onChange={onPick} />
        </label>

        {/* A new timeline tab: arrange pieces of the open files on lanes. */}
        <button
          type="button"
          onClick={onArrange}
          title="New Arrange tab — place pieces of your files on a multitrack timeline"
          className="mb-1 flex shrink-0 items-center gap-1 rounded-md px-2 py-1.5 text-xs text-[var(--chef-muted)] transition hover:bg-[var(--chef-hover)] hover:text-[var(--chef-text)]"
        >
          <LayoutPanelTop className="size-4" />
          Arrange
        </button>

        {/* Carries the strip's bottom line across the rest of the width. */}
        <div className="flex-1 self-stretch border-b border-[var(--chef-border)]" />
      </div>

      <div className="flex items-center self-stretch border-b border-[var(--chef-border)] pb-1 pl-2">
        <Recorder onTake={onRecord} variant="compact" />
      </div>
    </div>
  )
}

export const SourceTabs = memo(SourceTabsImpl)
