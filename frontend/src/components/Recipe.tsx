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
 * measures the droppable once at drag start, and unmounting the whole box from within it
 * the instant `isDraggingOver` flips would leave every cached dimension stale.
 */

import { Droppable, Draggable } from '@hello-pangea/dnd'
import { Trash2 } from 'lucide-react'
import { memo, useMemo } from 'react'
import { RecipeCard } from './RecipeCard'
import type { OperationDef, ParamValue, RecipeStep } from '../types'

interface Props {
  recipe: RecipeStep[]
  operations: OperationDef[]
  hasFile: boolean
  /** uid of the step whose region is drawn on the Input waveform, if any. */
  activeUid: string | null
  /** Which tab (source id) the `recipe` above belongs to. */
  selected: string
  /** That tab's file and colour, repeated in the header so you know what you are editing. */
  tabName: string
  tabColor: string
  onSelect: (uid: string) => void
  onParamChange: (uid: string, name: string, value: ParamValue) => void
  onToggleBypass: (uid: string) => void
  onRemove: (uid: string) => void
  onClear: () => void
  /** Replaces the open chain with a starter recipe. */
}

function RecipeImpl({
  recipe,
  operations,
  hasFile,
  activeUid,
  selected,
  tabName,
  tabColor,
  onSelect,
  onParamChange,
  onToggleBypass,
  onRemove,
  onClear,
}: Props) {
  const byId = useMemo(() => new Map(operations.map((op) => [op.id, op])), [operations])
  const empty = recipe.length === 0

  return (
    <section className="flex h-full min-h-0 flex-col border-r border-[var(--chef-border)] bg-[var(--chef-panel)]">
      <header className="flex items-center justify-between border-b border-[var(--chef-border)] px-4 py-3">
        <h2 className="flex min-w-0 items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-[var(--chef-muted)]">
          <span className="shrink-0">
            <span className="text-[var(--chef-accent-strong)]">2</span> · Recipe
          </span>
          {tabName && (
            <span
              className="flex min-w-0 items-center gap-1.5 normal-case tracking-normal text-[var(--chef-text)]"
              title={`Editing ${tabName}`}
            >
              <span
                className="size-2 shrink-0 rounded-full"
                style={{ backgroundColor: tabColor }}
                aria-hidden
              />
              <span className="truncate font-medium">{tabName}</span>
            </span>
          )}
        </h2>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={onClear}
            disabled={empty}
            className="flex items-center gap-1 rounded px-2 py-1 text-xs text-[var(--chef-muted)] transition hover:text-rose-600 disabled:opacity-30 disabled:hover:text-[var(--chef-muted)]"
          >
            <Trash2 className="size-3.5" /> Clear
          </button>
        </div>
      </header>

      {/* The scroller holds the droppable AND the empty state, so column 2 still scrolls
          as one surface even though only the first of the two accepts drops. */}
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {/* The droppable id carries the chain, so switching tabs is a NEW droppable
            rather than the same one with its items swapped underneath.  dnd measures a
            droppable once at drag start; reusing a fixed id across a tab switch is
            exactly how its cached dimensions go stale. */}
        <Droppable droppableId={`recipe:${selected}`}>
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
              </>
            ) : (
              <>
                <p className="text-sm text-[var(--chef-muted)]">Drag operations here</p>
                <p className="mt-1 text-xs text-[var(--chef-muted)] opacity-70">
                  They run top to bottom, like a CyberChef recipe.
                </p>
              </>
            )}
          </div>
        )}
      </div>
    </section>
  )
}

export const Recipe = memo(RecipeImpl)
