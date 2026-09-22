/**
 * The strip of loaded files, between the toolbar and the three columns.
 *
 * It sits full width rather than inside column 3 because it is project-level, like the
 * toolbar: column 3 already carries two waveforms and the stats panel, and the set of
 * sources is what columns 2 and 3 are both scoped BY.
 *
 * Each chip does three jobs, which are deliberately different clicks:
 *   - the body selects the source, pointing column 2's recipe and column 3's input at it
 *   - the star designates it the OUTPUT: whose chain feeds master and what Export renders
 *   - the x removes it
 */

import { Star, Upload, X } from 'lucide-react'
import { memo } from 'react'
import { Recorder } from './Recorder'
import { MASTER, type ChainId, type Source } from '../types'

interface Props {
  sources: Source[]
  /** Which chain column 2 is editing -- a source id, or MASTER. */
  editing: ChainId
  /** Whose processed chain feeds the master chain and Export. */
  outputId: string | null
  accept: string
  onSelect: (id: ChainId) => void
  onSetOutput: (id: string) => void
  onRemove: (id: string) => void
  onPick: (event: React.ChangeEvent<HTMLInputElement>) => void
  onRecord: (file: File) => void
}

function SourcesRailImpl({
  sources,
  editing,
  outputId,
  accept,
  onSelect,
  onSetOutput,
  onRemove,
  onPick,
  onRecord,
}: Props) {
  if (sources.length === 0) return null

  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-[var(--chef-border)] bg-[var(--chef-panel)] px-4 py-2">
      <span className="text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--chef-muted)]">
        Sources
      </span>

      {sources.map((source) => {
        const isOutput = source.id === outputId
        return (
          <div
            key={source.id}
            className={`flex items-center gap-1.5 rounded-md border pl-1 pr-1 text-xs transition ${
              editing === source.id
                ? 'border-[var(--chef-accent-strong)] bg-[var(--chef-accent)]/25'
                : 'border-[var(--chef-border)]'
            }`}
          >
            <button
              type="button"
              onClick={() => onSetOutput(source.id)}
              title={
                isOutput
                  ? 'This source is the project output — its chain feeds MASTER and Export'
                  : 'Make this the project output'
              }
              className={`rounded p-1 transition hover:bg-[var(--chef-hover)] ${
                isOutput ? 'text-amber-500' : 'text-[var(--chef-muted)] opacity-40'
              }`}
            >
              <Star className={`size-3 ${isOutput ? 'fill-current' : ''}`} />
            </button>

            <button
              type="button"
              onClick={() => onSelect(source.id)}
              title={`Edit ${source.filename}`}
              className="flex items-baseline gap-1.5 py-1"
            >
              <span className="max-w-[160px] truncate font-medium">{source.filename}</span>
              <span className="font-mono text-[10px] text-[var(--chef-muted)]">
                {source.duration.toFixed(2)}s · {(source.sampleRate / 1000).toFixed(1)}k
              </span>
              {source.recipe.length > 0 && (
                <span className="font-mono text-[10px] text-[var(--chef-accent-strong)]">
                  {source.recipe.length} step{source.recipe.length === 1 ? '' : 's'}
                </span>
              )}
            </button>

            <button
              type="button"
              onClick={() => onRemove(source.id)}
              title="Remove this source from the project"
              className="rounded p-1 text-[var(--chef-muted)] transition hover:bg-[var(--chef-hover)] hover:text-rose-600"
            >
              <X className="size-3" />
            </button>
          </div>
        )
      })}

      {/* MASTER is a chain without a file, so it gets a chip here too rather than only
          appearing in column 2 -- otherwise there is nothing on screen saying the
          project HAS a final stage. */}
      <button
        type="button"
        onClick={() => onSelect(MASTER)}
        title="The chain that runs over the finished mix"
        className={`rounded-md border px-2 py-1.5 text-xs transition ${
          editing === MASTER
            ? 'border-[var(--chef-accent-strong)] bg-[var(--chef-accent)]/25 text-[var(--chef-accent-strong)]'
            : 'border-dashed border-[var(--chef-border)] text-[var(--chef-muted)] hover:text-[var(--chef-text)]'
        }`}
      >
        MASTER
      </button>

      <div className="ml-auto flex items-center gap-2">
        <Recorder onTake={onRecord} variant="compact" />
        <label className="flex cursor-pointer items-center gap-1.5 rounded-md border border-[var(--chef-border)] px-2.5 py-1.5 text-xs transition hover:border-[var(--chef-accent-strong)]">
          <Upload className="size-3.5" />
          Add a source
          <input type="file" accept={accept} className="hidden" onChange={onPick} />
        </label>
      </div>
    </div>
  )
}

export const SourcesRail = memo(SourcesRailImpl)
