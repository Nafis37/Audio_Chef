/**
 * Column 2 -- The Recipe.
 *
 * A droppable column that holds the ordered list of operations.  Order is meaningful:
 * the backend folds the audio through the steps top to bottom, so EQ -> compressor
 * sounds different from compressor -> EQ.  Cards can be reordered by dragging their
 * handle, and new ones arrive by dragging from the palette.
 *
 * The empty state does double duty as the app's signpost.  Before a file is loaded it
 * stays quiet; the moment there is audio and no recipe it becomes the loudest thing on
 * screen, because "you have a file, now pick an effect" is exactly the step people were
 * getting stuck on.  It renders as a SIBLING of the droppable, not inside it: dnd
 * measures the droppable once at drag start, and unmounting ~200px of preset chips from
 * within it the instant `isDraggingOver` flips left every cached dimension stale.
 */

import { Droppable, Draggable } from '@hello-pangea/dnd'
import { Trash2 } from 'lucide-react'
import { memo, useMemo } from 'react'
import { RecipeCard } from './RecipeCard'
import { iconFor } from '../icons'
import { TIME_SHIFTING_OPS } from '../regions'
import { PRESETS, type Preset } from '../presets'
import type { OperationDef, ParamValue, RecipeStep } from '../types'

interface Props {
  recipe: RecipeStep[]
  operations: OperationDef[]
  hasFile: boolean
  /** uid of the step whose region is drawn on the Input waveform, if any. */
  activeUid: string | null
  onSelect: (uid: string) => void
  onParamChange: (uid: string, name: string, value: ParamValue) => void
  onToggleBypass: (uid: string) => void
  onRemove: (uid: string) => void
  onClear: () => void
  onApplyPreset: (preset: Preset) => void
}

function RecipeImpl({
  recipe,
  operations,
  hasFile,
  activeUid,
  onSelect,
  onParamChange,
  onToggleBypass,
  onRemove,
  onClear,
  onApplyPreset,
}: Props) {
  const byId = useMemo(() => new Map(operations.map((op) => [op.id, op])), [operations])
  const empty = recipe.length === 0

  return (
    <section className="flex h-full min-h-0 flex-col border-r border-[var(--chef-border)] bg-[var(--chef-panel)]">
      <header className="flex items-center justify-between border-b border-[var(--chef-border)] px-4 py-3">
        <h2 className="text-xs font-semibold uppercase tracking-[0.18em] text-[var(--chef-muted)]">
          <span className="text-[var(--chef-accent-strong)]">2</span> · Recipe
        </h2>
        <button
          type="button"
          onClick={onClear}
          disabled={empty}
          className="flex items-center gap-1 rounded px-2 py-1 text-xs text-[var(--chef-muted)] transition hover:text-rose-600 disabled:opacity-30 disabled:hover:text-[var(--chef-muted)]"
        >
          <Trash2 className="size-3.5" /> Clear
        </button>
      </header>

      {/* The scroller holds the droppable AND the empty state, so column 2 still scrolls
          as one surface even though only the first of the two accepts drops. */}
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        <Droppable droppableId="recipe">
          {(provided, snapshot) => (
            <div
              ref={provided.innerRef}
              {...provided.droppableProps}
              /* Empty, this still has to be a real target -- min-h-24 gives the first
                 drop somewhere to land.  Per-item margins live on the cards. */
              className={`min-h-24 rounded-lg transition ${
                snapshot.isDraggingOver ? 'bg-[var(--chef-dragover)]' : ''
              }`}
            >
              {recipe.map((step, index) => {
                const definition = byId.get(step.op)
                if (!definition) return null
                // Anything above this step that is live and changes the duration puts the
                // region's seconds and this step's seconds on different clocks.
                const timesShifted = recipe
                  .slice(0, index)
                  .some((earlier) => !earlier.bypass && TIME_SHIFTING_OPS.has(earlier.op))
                return (
                  <Draggable key={step.uid} draggableId={step.uid} index={index}>
                    {(drag, dragSnapshot) => (
                      <RecipeCard
                        step={step}
                        definition={definition}
                        index={index}
                        drag={drag}
                        isDragging={dragSnapshot.isDragging}
                        active={step.uid === activeUid}
                        timesShifted={timesShifted}
                        onSelect={onSelect}
                        onParamChange={onParamChange}
                        onToggleBypass={onToggleBypass}
                        onRemove={onRemove}
                      />
                    )}
                  </Draggable>
                )
              })}
              {provided.placeholder}
            </div>
          )}
        </Droppable>

        {empty && (
          <div
            className={`mt-1 rounded-lg border border-dashed p-5 text-center transition ${
              hasFile
                ? 'border-[var(--chef-accent-strong)] bg-[var(--chef-accent)]/20'
                : 'border-[var(--chef-border)]'
            }`}
          >
            {hasFile ? (
              <>
                <p className="text-sm font-medium text-[var(--chef-accent-strong)]">
                  Pick an operation on the left
                </p>
                <p className="mt-1 text-xs text-[var(--chef-muted)]">
                  Click it or drag it here. Steps run top to bottom.
                </p>
              </>
            ) : (
              <>
                <p className="text-sm text-[var(--chef-muted)]">Drag operations here</p>
                <p className="mt-1 text-xs text-[var(--chef-muted)] opacity-70">
                  They run top to bottom, like a CyberChef recipe.
                </p>
              </>
            )}

            {/* Starter recipes -- the fastest way to hear the thing actually work. */}
            <p className="mt-5 mb-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--chef-muted)]">
              or start from a preset
            </p>
            <div className="flex flex-wrap justify-center gap-1.5">
              {PRESETS.map((preset) => {
                const Icon = iconFor(preset.icon)
                return (
                  <button
                    key={preset.id}
                    type="button"
                    title={preset.description}
                    onClick={() => onApplyPreset(preset)}
                    className="flex items-center gap-1.5 rounded-full border border-[var(--chef-border)] bg-[var(--chef-surface)] px-2.5 py-1 text-xs text-[var(--chef-text)] transition hover:border-[var(--chef-accent-strong)] hover:text-[var(--chef-accent-strong)]"
                  >
                    <Icon className="size-3.5 text-[var(--chef-accent-strong)]" />
                    {preset.label}
                  </button>
                )
              })}
            </div>
          </div>
        )}
      </div>
    </section>
  )
}

export const Recipe = memo(RecipeImpl)
