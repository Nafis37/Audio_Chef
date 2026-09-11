/**
 * Column 1 -- the Operations Palette.
 *
 * The 7 audio tools.  Each entry is a @hello-pangea/dnd Draggable living in a droppable
 * that never accepts drops (`isDropDisabled`), so the only thing you can do with a palette
 * item is drag it OUT, into the recipe column.  Clicking an item adds it too, which is
 * faster once you know the tools.
 *
 * There is no search box: seven entries all fit on screen at once, so filtering them was
 * a control that cost more attention than it saved.
 *
 * Two things here are load-bearing and easy to undo by accident:
 *
 *   1. The draggable root is a <div>, NOT a <button>.  dnd's mouse sensor runs
 *      isEventInInteractiveElement() on every mousedown and ABORTS the lift when the
 *      press lands inside a button / input / select / textarea / a[href].  When this card
 *      was a <button> carrying its own dragHandleProps, every press was its own veto and
 *      the item never lifted at all.  The click-to-add behaviour is kept with onClick +
 *      an explicit Enter handler instead (Space belongs to dnd, which uses it to lift).
 *
 *   2. `isDropDisabled` + `renderClone` are the two halves of dnd's documented "copy from
 *      a source list" pattern.  With only the first, the real node is torn out of this
 *      column's scroll container for the duration of the drag: the list collapses behind
 *      it and the preview is one `transform` on an ancestor away from being clipped at
 *      the column edge.  The clone renders the same PaletteItem, so the two cannot drift.
 */

import { Draggable, Droppable } from '@hello-pangea/dnd'
import type { DraggableProvided } from '@hello-pangea/dnd'
import { GripVertical } from 'lucide-react'
import { memo } from 'react'
import { iconFor } from '../icons'
import type { OperationDef } from '../types'

interface Props {
  operations: OperationDef[]
  onAdd: (op: OperationDef) => void
}

interface ItemProps {
  op: OperationDef
  drag: DraggableProvided
  isDragging: boolean
  onAdd: (op: OperationDef) => void
}

/** One palette card.  Rendered both in the list and as the drag clone. */
function PaletteItem({ op, drag, isDragging, onAdd }: ItemProps) {
  const Icon = iconFor(op.icon)
  return (
    <div
      ref={drag.innerRef}
      {...drag.draggableProps}
      {...drag.dragHandleProps}
      onClick={() => onAdd(op)}
      onKeyDown={(event) => {
        // dragHandleProps already owns Space (lift) and the arrow keys (move), so Enter
        // is the only key left for the click-equivalent.
        if (event.key === 'Enter') {
          event.preventDefault()
          onAdd(op)
        }
      }}
      className={`mb-2 w-full cursor-grab rounded-lg border bg-[var(--chef-surface)] p-3 text-left transition ${
        isDragging
          ? 'border-[var(--chef-accent-strong)] shadow-lg shadow-[var(--chef-accent)]/50'
          : 'border-[var(--chef-border)] hover:border-[var(--chef-accent-strong)]'
      }`}
    >
      <div className="flex items-center gap-2">
        {/* The card is no longer a <button>, so it needs a visible "grab me" affordance. */}
        <GripVertical className="size-3.5 shrink-0 text-[var(--chef-muted)]" />
        <Icon className="size-4 shrink-0 text-[var(--chef-accent-strong)]" />
        <span className="text-sm font-medium">{op.label}</span>
      </div>
      <p className="mt-1 text-xs leading-snug text-[var(--chef-muted)]">{op.description}</p>
    </div>
  )
}

function OperationsPaletteImpl({ operations, onAdd }: Props) {
  return (
    <section className="flex h-full min-h-0 flex-col border-r border-[var(--chef-border)] bg-[var(--chef-panel)]">
      <header className="border-b border-[var(--chef-border)] px-4 py-3">
        <h2 className="text-xs font-semibold uppercase tracking-[0.18em] text-[var(--chef-muted)]">
          <span className="text-[var(--chef-accent-strong)]">1</span> · Operations
        </h2>
        <p className="mt-1 text-[11px] text-[var(--chef-muted)]">
          Click or drag one into the recipe.
        </p>
      </header>

      <Droppable
        droppableId="palette"
        isDropDisabled
        renderClone={(provided, snapshot, rubric) => (
          <PaletteItem
            op={operations[rubric.source.index]}
            drag={provided}
            isDragging={snapshot.isDragging}
            onAdd={onAdd}
          />
        )}
      >
        {(provided) => (
          <div
            ref={provided.innerRef}
            {...provided.droppableProps}
            /* Per-item `mb-2` rather than `space-y-2`: the latter is a sibling-selector
               margin that also lands on provided.placeholder, which dnd measures. */
            className="min-h-0 flex-1 overflow-y-auto px-3 pb-4"
          >
            {operations.map((op, index) => (
              <Draggable key={op.id} draggableId={`palette-${op.id}`} index={index}>
                {(drag, snapshot) => (
                  <PaletteItem
                    op={op}
                    drag={drag}
                    isDragging={snapshot.isDragging}
                    onAdd={onAdd}
                  />
                )}
              </Draggable>
            ))}
            {/* The placeholder is required by dnd even though this list never receives drops. */}
            {provided.placeholder}
          </div>
        )}
      </Droppable>
    </section>
  )
}

export const OperationsPalette = memo(OperationsPaletteImpl)
