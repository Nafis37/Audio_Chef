/**
 * A draggable gutter between two workbench columns.
 *
 * Controlled, and deliberately absolute rather than incremental: the width at the moment
 * of the press is captured once, and every move reports `startWidth + (x - startX)`.  An
 * incremental splitter (emitting each frame's dx) drifts as soon as the parent clamps --
 * the pointer keeps travelling past the limit, and dragging back does nothing until the
 * accumulated overshoot is paid off.  Reporting an absolute target makes the clamp a pure
 * function of the pointer, so the edge is sticky and the return is immediate.
 *
 * Pointer capture (not window listeners) keeps the events coming while the cursor is over
 * the waveform canvases, and releases itself if the pointer is lost -- nothing to leak.
 */

import { memo, useCallback, useRef, useState } from 'react'

interface Props {
  /** Current width in px of the column to the LEFT of this splitter. */
  width: number
  /** Reports the desired new width.  The parent clamps; this component does not. */
  onResize: (width: number) => void
  /** Announced to screen readers, e.g. "Operations column width". */
  label: string
}

/** Arrow-key step, and the coarser step when Shift is held. */
const STEP = 16
const STEP_COARSE = 64

function ColumnSplitterImpl({ width, onResize, label }: Props) {
  const [dragging, setDragging] = useState(false)
  // The press origin.  Null whenever no drag is in progress.
  const origin = useRef<{ x: number; width: number } | null>(null)

  const handlePointerDown = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (event.button !== 0) return
      event.preventDefault()
      event.currentTarget.setPointerCapture(event.pointerId)
      origin.current = { x: event.clientX, width }
      setDragging(true)
      // Without these the drag paints a text selection across all three columns and the
      // cursor flickers back to the default over every element it crosses.
      document.body.style.userSelect = 'none'
      document.body.style.cursor = 'col-resize'
    },
    [width],
  )

  const handlePointerMove = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      const start = origin.current
      if (!start) return
      onResize(start.width + (event.clientX - start.x))
    },
    [onResize],
  )

  const endDrag = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (!origin.current) return
    origin.current = null
    setDragging(false)
    document.body.style.userSelect = ''
    document.body.style.cursor = ''
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
  }, [])

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      const step = event.shiftKey ? STEP_COARSE : STEP
      if (event.key === 'ArrowLeft') {
        event.preventDefault()
        onResize(width - step)
      } else if (event.key === 'ArrowRight') {
        event.preventDefault()
        onResize(width + step)
      }
    },
    [onResize, width],
  )

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      aria-valuenow={Math.round(width)}
      tabIndex={0}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onKeyDown={handleKeyDown}
      className="group relative w-1.5 shrink-0 cursor-col-resize touch-none outline-none"
      title="Drag to resize"
    >
      {/* The visible hairline sits inside a wider hit area, so the gutter is easy to
          grab without showing a 6px seam between the columns. */}
      <div
        className={`h-full w-full transition ${
          dragging
            ? 'bg-[var(--chef-accent-strong)]'
            : 'bg-transparent group-hover:bg-[var(--chef-accent)] group-focus-visible:bg-[var(--chef-accent-strong)]'
        }`}
      />
    </div>
  )
}

export const ColumnSplitter = memo(ColumnSplitterImpl)
