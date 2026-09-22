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
import { TIME_SHIFTING_OPS } from '../regions'
import { MASTER, type ChainId, type OperationDef, type ParamValue, type RecipeStep } from '../types'

export interface ChainTab {
  id: ChainId
  label: string
  /** How many live steps this chain holds -- shown as a count on the chip. */
  steps: number
  /** This chain's output is what Export renders.  Only ever one. */
  isOutput: boolean
}

interface Props {
  recipe: RecipeStep[]
  operations: OperationDef[]
  hasFile: boolean
  /** uid of the step whose region is drawn on the Input waveform, if any. */
  activeUid: string | null
  /** One chip per source plus MASTER. */
  tabs: ChainTab[]
  /** Which chain the `recipe` above belongs to. */
  selected: ChainId
  onSelectTab: (id: ChainId) => void
  onSelect: (uid: string) => void
  onParamChange: (uid: string, name: string, value: ParamValue) => void
  onToggleBypass: (uid: string) => void
  onRemove: (uid: string) => void
  onClear: () => void
}

function RecipeImpl({
  recipe,
  operations,
  hasFile,
  activeUid,
  tabs,
  selected,
  onSelectTab,
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

      {/* One chip per source plus MASTER.  Each source carries its own chain, so this is
          what makes "EQ the vocals, compress the drums" expressible at all. */}
      {tabs.length > 0 && (
        <div className="flex flex-wrap gap-1 border-b border-[var(--chef-border)] px-3 py-2">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => onSelectTab(tab.id)}
              title={tab.id === MASTER ? 'Runs over the finished mix' : `Recipe for ${tab.label}`}
              className={`flex max-w-full items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] transition ${
                tab.id === selected
                  ? 'border-[var(--chef-accent-strong)] bg-[var(--chef-accent)]/25 text-[var(--chef-accent-strong)]'
                  : 'border-[var(--chef-border)] text-[var(--chef-muted)] hover:text-[var(--chef-text)]'
              }`}
            >
              {tab.isOutput && (
                <span title="This source is what Export renders" className="text-amber-500">
                  ★
                </span>
              )}
              <span className="truncate">
                {tab.id === MASTER ? 'MASTER' : tab.label}
              </span>
              {tab.steps > 0 && (
                <span className="font-mono opacity-60">{tab.steps}</span>
              )}
            </button>
          ))}
        </div>
      )}

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
                  {selected === MASTER
                    ? 'Steps here run over the finished mix, after every source chain.'
                    : 'Click it or drag it here. Steps run top to bottom, on this source only.'}
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
