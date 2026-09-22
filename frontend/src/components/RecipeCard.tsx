/**
 * One card in the recipe column.
 *
 * The controls are generated from the operation's parameter schema (which the backend
 * sends), so a slider always has exactly the min/max/step the DSP function expects.
 * Each card also carries a Bypass toggle -- the step stays in the recipe but the backend
 * skips it, which is the quickest way to A/B an effect.
 *
 * Continuous controls keep their own live value while you drag them.  A `range` input
 * fires an event per pixel of travel, and every one of those used to replace the whole
 * recipe array in App -- re-rendering both drag-and-drop columns at pointer rate.  Now
 * the thumb and the readout track the pointer from local state, and the parent is told
 * at most once per THROTTLE_MS, with the final value always delivered.
 */

import type { DraggableProvided } from '@hello-pangea/dnd'
import { GripVertical, Power, TriangleAlert, X } from 'lucide-react'
import { memo, useCallback, useEffect, useRef, useState } from 'react'
import { iconFor } from '../icons'
import { REGION_OPS } from '../regions'
import { useSources } from '../sources'
import type { OperationDef, ParamDef, ParamValue, RecipeStep } from '../types'

/** How often a dragged slider is allowed to push a value up into App's state. */
const THROTTLE_MS = 80

interface Props {
  step: RecipeStep
  definition: OperationDef
  index: number
  drag: DraggableProvided
  isDragging: boolean
  /** This card owns the region currently drawn on the Input waveform. */
  active: boolean
  /** True when an earlier un-bypassed step changes the duration.  See below. */
  timesShifted: boolean
  onSelect: (uid: string) => void
  onParamChange: (uid: string, name: string, value: ParamValue) => void
  onToggleBypass: (uid: string) => void
  onRemove: (uid: string) => void
}

/**
 * Renders `value` immediately, forwards it upward on a leading+trailing throttle.
 *
 * `sent` remembers the last value this control pushed, so when the parent echoes it back
 * we know it is our own value and leave the local one alone; anything else (a preset, a
 * Clear, a reordered card) is an outside edit and does resync the control.
 */
function useLiveValue(
  value: ParamValue,
  name: string,
  onChange: (name: string, value: ParamValue) => void,
) {
  const [local, setLocal] = useState<ParamValue>(value)
  const sent = useRef<ParamValue>(value)
  const latest = useRef<ParamValue>(value)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (value === sent.current) return   // our own change coming back around
    sent.current = value
    latest.current = value
    setLocal(value)
  }, [value])

  const push = useCallback(
    (next: ParamValue) => {
      sent.current = next
      onChange(name, next)
    },
    [name, onChange],
  )

  const set = useCallback(
    (next: ParamValue) => {
      setLocal(next)                       // repaint this one control, nothing else
      latest.current = next
      if (timer.current !== null) return   // a trailing push is already scheduled
      push(next)                           // leading edge: react at once
      timer.current = setTimeout(() => {
        timer.current = null
        if (latest.current !== sent.current) push(latest.current)   // trailing edge
      }, THROTTLE_MS)
    },
    [push],
  )

  // A card removed mid-drag must not leave a stray timer behind.
  useEffect(
    () => () => {
      if (timer.current !== null) clearTimeout(timer.current)
    },
    [],
  )

  return [local, set] as const
}

/**
 * The one control whose options come from the session rather than the backend schema.
 *
 * Its own component so that useSources() is called at the leaf: only the dropdowns
 * re-render when a source is added or removed, not the card or the column around them.
 */
function SourceControl({
  param,
  value,
  onChange,
}: {
  param: ParamDef
  value: ParamValue
  onChange: (name: string, value: ParamValue) => void
}) {
  const sources = useSources()
  const current = String(value ?? '')
  // A source can be removed while a card still points at it.  Say so rather than
  // silently showing the first remaining source, which would misreport the recipe.
  const dangling = current !== '' && !sources.some((source) => source.id === current)

  return (
    <label className="flex items-center justify-between gap-3 py-1 text-xs">
      <span className="text-[var(--chef-muted)]">{param.label}</span>
      <select
        value={current}
        onChange={(event) => onChange(param.name, event.target.value)}
        className={`max-w-[60%] truncate rounded border bg-[var(--chef-inset)] px-2 py-1 text-xs outline-none focus:border-[var(--chef-accent-strong)] ${
          current === '' || dangling
            ? 'border-rose-500/60 text-rose-600'
            : 'border-[var(--chef-border)] text-[var(--chef-text)]'
        }`}
      >
        <option value="">
          {sources.length === 0 ? 'Load another file first…' : 'Pick a source…'}
        </option>
        {dangling && <option value={current}>{current} (removed)</option>}
        {sources.map((source) => (
          <option key={source.id} value={source.id}>
            {source.label}
          </option>
        ))}
      </select>
    </label>
  )
}

/** Renders the right widget for one parameter, based on its `control` field. */
function Control({
  param,
  value,
  onChange,
}: {
  param: ParamDef
  value: ParamValue
  onChange: (name: string, value: ParamValue) => void
}) {
  const [live, setLive] = useLiveValue(value, param.name, onChange)

  // Straight through, unthrottled: picking a source is one discrete click, not a drag,
  // and its options come from the session rather than from `param`.
  if (param.control === 'source') {
    return <SourceControl param={param} value={value} onChange={onChange} />
  }

  if (param.control === 'toggle') {
    return (
      <label className="flex items-center justify-between gap-3 py-1 text-xs">
        <span className="text-[var(--chef-muted)]">{param.label}</span>
        <input
          type="checkbox"
          checked={Boolean(live)}
          onChange={(event) => setLive(event.target.checked)}
          className="size-4 accent-[var(--chef-accent)]"
        />
      </label>
    )
  }

  if (param.control === 'select') {
    return (
      <label className="flex items-center justify-between gap-3 py-1 text-xs">
        <span className="text-[var(--chef-muted)]">{param.label}</span>
        <select
          value={String(live)}
          onChange={(event) => setLive(event.target.value)}
          className="rounded border border-[var(--chef-border)] bg-[var(--chef-inset)] px-2 py-1 text-xs text-[var(--chef-text)] outline-none focus:border-[var(--chef-accent-strong)]"
        >
          {param.options?.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </label>
    )
  }

  if (param.control === 'number') {
    return (
      <label className="flex items-center justify-between gap-3 py-1 text-xs">
        <span className="text-[var(--chef-muted)]">
          {param.label} {param.unit && <span className="opacity-60">({param.unit})</span>}
        </span>
        <input
          type="number"
          value={Number(live)}
          min={param.min}
          max={param.max}
          step={param.step}
          onChange={(event) => setLive(Number(event.target.value))}
          className="w-24 rounded border border-[var(--chef-border)] bg-[var(--chef-inset)] px-2 py-1 text-right text-xs outline-none focus:border-[var(--chef-accent-strong)]"
        />
      </label>
    )
  }

  // Default: a slider with a live numeric readout.
  return (
    <label className="block py-1 text-xs">
      <span className="flex items-center justify-between">
        <span className="text-[var(--chef-muted)]">{param.label}</span>
        <span className="font-mono text-[var(--chef-text)]">
          {Number(live).toFixed(2)}
          {param.unit && <span className="ml-0.5 opacity-60">{param.unit}</span>}
        </span>
      </span>
      <input
        type="range"
        className="mt-1.5 w-full"
        value={Number(live)}
        min={param.min}
        max={param.max}
        step={param.step}
        onChange={(event) => setLive(Number(event.target.value))}
      />
    </label>
  )
}

function RecipeCardImpl({
  step,
  definition,
  index,
  drag,
  isDragging,
  active,
  timesShifted,
  onSelect,
  onParamChange,
  onToggleBypass,
  onRemove,
}: Props) {
  const Icon = iconFor(definition.icon)
  // Only ops with time parameters can drive a region -- see src/regions.ts.
  const linkable = definition.id in REGION_OPS

  // One stable callback for the whole card instead of a fresh closure per parameter.
  const handleParam = useCallback(
    (name: string, value: ParamValue) => onParamChange(step.uid, name, value),
    [onParamChange, step.uid],
  )

  return (
    <div
      ref={drag.innerRef}
      {...drag.draggableProps}
      className={`mb-2 rounded-lg border bg-[var(--chef-surface)] transition ${
        step.bypass
          ? 'border-dashed border-[var(--chef-border)] opacity-50'
          : 'border-[var(--chef-border)]'
      } ${active && linkable && !step.bypass ? 'border-[var(--chef-accent-strong)]' : ''} ${
        isDragging ? 'border-[var(--chef-accent-strong)] shadow-lg shadow-[var(--chef-accent)]/50' : ''
      }`}
    >
      {/* Selecting on the HEADER, not the card: the body is full of sliders and selects,
          and a click there must not also re-point the waveform's region. */}
      <header
        onClick={() => linkable && onSelect(step.uid)}
        className={`flex items-center gap-2 border-b border-[var(--chef-border)] px-3 py-2 ${
          linkable ? 'cursor-pointer' : ''
        }`}
      >
        {/* Only this handle starts a drag, so sliders stay usable inside the card. */}
        <span {...drag.dragHandleProps} className="cursor-grab text-[var(--chef-muted)]">
          <GripVertical className="size-4" />
        </span>
        <span className="font-mono text-[10px] text-[var(--chef-muted)]">{index + 1}</span>
        <Icon className="size-4 text-[var(--chef-accent-strong)]" />
        <span className="flex-1 truncate text-sm font-medium">{definition.label}</span>

        {linkable && (
          <span
            title={
              active
                ? 'The region on the Input waveform is driving this card'
                : 'Click this header to drag this card’s times on the waveform'
            }
            className={`rounded-full px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider ${
              active
                ? 'bg-[var(--chef-accent)]/40 text-[var(--chef-accent-strong)]'
                : 'text-[var(--chef-muted)]'
            }`}
          >
            {active ? 'linked' : 'link'}
          </span>
        )}

        <button
          type="button"
          title={step.bypass ? 'Enable this step' : 'Bypass this step'}
          onClick={() => onToggleBypass(step.uid)}
          className={`rounded p-1 transition hover:bg-[var(--chef-hover)] ${
            step.bypass ? 'text-[var(--chef-muted)]' : 'text-[var(--chef-accent-strong)]'
          }`}
        >
          <Power className="size-4" />
        </button>
        <button
          type="button"
          title="Remove from recipe"
          onClick={() => onRemove(step.uid)}
          className="rounded p-1 text-[var(--chef-muted)] transition hover:bg-[var(--chef-hover)] hover:text-rose-600"
        >
          <X className="size-4" />
        </button>
      </header>

      {/* Every step sees the audio as the steps ABOVE it left it, but the region is drawn
          against the original input.  Once an earlier step has changed the duration the
          two clocks have diverged, and the handles no longer mean what they look like. */}
      {linkable && timesShifted && !step.bypass && (
        <p className="flex items-start gap-1.5 border-b border-[var(--chef-border)] px-3 py-1.5 text-[10px] leading-snug text-[var(--chef-muted)]">
          <TriangleAlert className="mt-px size-3 shrink-0" />
          <span>
            An earlier step changes the length, so these times are measured on the audio
            entering <em>this</em> step — not on the Input waveform.
          </span>
        </p>
      )}

      <div className="space-y-0.5 px-3 py-2">
        {definition.params.map((param) => (
          <Control
            key={param.name}
            param={param}
            value={step.params[param.name] ?? param.default}
            onChange={handleParam}
          />
        ))}
      </div>
    </div>
  )
}

export const RecipeCard = memo(RecipeCardImpl)
