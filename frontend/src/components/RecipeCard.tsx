/**
 * One card in the recipe column.
 *
 * The controls are generated from the operation's parameter schema (which the backend
 * sends), so a slider always has exactly the min/max/step the DSP function expects.
 * Each card also carries a Bypass toggle -- the step stays in the recipe but the backend
 * skips it, which is the quickest way to A/B an effect.
 *
 * What a card shows, top to bottom -- all of it generated from the backend schema:
 *
 *   header       name, one-line summary, link / On-Off / remove
 *   quick row    one-click settings ("Subtle", "Hall", "Fast talk"...) from `op.quick`
 *   main knobs   the params a first-time user needs
 *   Advanced     everything marked `advanced`, folded away
 *   footer       "Listen for": what should change when you A/B it
 *
 * A param whose `show_when` does not match the step's current values (the robot's buzz
 * while in chipmunk mode, say) is not rendered at all: a knob that does nothing is worse
 * than none.
 *
 * Continuous controls keep their own live value while you drag them.  A `range` input
 * fires an event per pixel of travel, and every one of those used to replace the whole
 * recipe array in App -- re-rendering both drag-and-drop columns at pointer rate.  Now
 * the thumb and the readout track the pointer from local state, and the parent is told
 * at most once per THROTTLE_MS, with the final value always delivered.
 */

import type { DraggableProvided } from '@hello-pangea/dnd'
import { ChevronRight, Ear, GripVertical, Info, TriangleAlert, X } from 'lucide-react'
import { memo, useCallback, useEffect, useRef, useState } from 'react'
import { iconFor } from '../icons'
import { regionFor } from '../regions'
import { useSources } from '../sources'
import {
  isParamShown,
  type OperationDef,
  type ParamDef,
  type ParamValue,
  type RecipeStep,
} from '../types'

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
  /** Several params in one edit -- a quick-setting chip. */
  onParamsChange: (uid: string, values: Record<string, ParamValue>) => void
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

/** A readout in the param's own units: "+6.0 dB", "45%", "1.50×", "−7.0 st", "4.0 kHz". */
function formatValue(param: ParamDef, value: number): string {
  const signed = (text: string) =>
    value > 0 ? `+${text}` : value < 0 ? `−${text.replace('-', '')}` : text
  switch (param.unit) {
    case '%':
      return `${Math.round(value)}%`
    case 'dB':
      return `${signed(value.toFixed(1))} dB`
    case 'st':
      return `${signed(value.toFixed(1))} st`
    case 'x':
      return `${value.toFixed(2)}×`
    case 'Hz':
      return value >= 1000 ? `${(value / 1000).toFixed(1)} kHz` : `${Math.round(value)} Hz`
    case 's':
      return `${value.toFixed(2)} s`
    case 'ms':
      return `${value.toFixed(value < 10 ? 1 : 0)} ms`
    default:
      return `${Number(value.toFixed(2))}${param.unit}`
  }
}

/** The label, with its plain-language help as a tooltip and a small hint that there is one. */
function Label({ param }: { param: ParamDef }) {
  return (
    <span
      title={param.help || undefined}
      className="flex items-center gap-1 text-[var(--chef-muted)]"
    >
      {param.label}
      {param.help && <Info className="size-3 opacity-50" aria-hidden />}
    </span>
  )
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
      <Label param={param} />
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
        <Label param={param} />
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
    const options = param.options ?? []
    const nameOf = (option: string) => param.option_labels?.[option] ?? option
    // A handful of choices reads better as buttons you can see all at once.
    if (options.length <= 4) {
      return (
        <div className="py-1 text-xs">
          <Label param={param} />
          <div className="mt-1 flex rounded-md border border-[var(--chef-border)] bg-[var(--chef-inset)] p-0.5">
            {options.map((option) => (
              <button
                key={option}
                type="button"
                onClick={() => setLive(option)}
                className={`flex-1 rounded px-1.5 py-1 text-[11px] transition ${
                  String(live) === option
                    ? 'bg-[var(--chef-accent-strong)] font-medium text-white'
                    : 'text-[var(--chef-muted)] hover:text-[var(--chef-text)]'
                }`}
              >
                {nameOf(option)}
              </button>
            ))}
          </div>
        </div>
      )
    }
    return (
      <label className="flex items-center justify-between gap-3 py-1 text-xs">
        <Label param={param} />
        <select
          value={String(live)}
          onChange={(event) => setLive(event.target.value)}
          className="rounded border border-[var(--chef-border)] bg-[var(--chef-inset)] px-2 py-1 text-xs text-[var(--chef-text)] outline-none focus:border-[var(--chef-accent-strong)]"
        >
          {options.map((option) => (
            <option key={option} value={option}>
              {nameOf(option)}
            </option>
          ))}
        </select>
      </label>
    )
  }

  if (param.control === 'number') {
    return (
      <label className="flex items-center justify-between gap-3 py-1 text-xs">
        <span className="flex items-center gap-1">
          <Label param={param} />
          {param.unit && (
            <span className="text-[var(--chef-muted)] opacity-60">({param.unit})</span>
          )}
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
        <Label param={param} />
        <span className="font-mono text-[var(--chef-text)]">
          {formatValue(param, Number(live))}
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
  onParamsChange,
  onToggleBypass,
  onRemove,
}: Props) {
  const Icon = iconFor(definition.icon)

  // Every param's current value, defaults filled in -- what show_when and the chips read.
  const values: Record<string, ParamValue> = {}
  for (const param of definition.params) values[param.name] = step.params[param.name] ?? param.default

  // Only ops with time parameters can drive a region -- see src/regions.ts -- and only
  // while the current mode uses one (the noise remover's "Automatically" does not).
  const linkable = regionFor(definition.id, values) !== undefined

  const shown = definition.params.filter((param) => isParamShown(param, values))
  const main = shown.filter((param) => !param.advanced)
  const advanced = shown.filter((param) => param.advanced)
  const quick = Object.entries(definition.quick ?? {})
  // A chip is lit when every value it sets is what the card currently holds.
  const isQuickActive = (settings: Record<string, ParamValue>) =>
    Object.entries(settings).every(([name, value]) =>
      typeof value === 'number'
        ? Math.abs(Number(values[name]) - value) < 1e-6
        : values[name] === value,
    )

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
        <Icon className="size-4 shrink-0 text-[var(--chef-accent-strong)]" />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-medium">{definition.label}</span>
          <span
            className="block truncate text-[11px] text-[var(--chef-muted)]"
            title={definition.summary}
          >
            {definition.summary}
          </span>
        </span>

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

        {/* stopPropagation: a click here must not also link the card via the header. */}
        <button
          type="button"
          title={
            step.bypass ? 'Switch this step back on' : 'Switch this step off to hear the difference'
          }
          onClick={(event) => {
            event.stopPropagation()
            onToggleBypass(step.uid)
          }}
          className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider transition ${
            step.bypass
              ? 'border-[var(--chef-border)] text-[var(--chef-muted)] hover:text-[var(--chef-text)]'
              : 'border-[var(--chef-accent-strong)] bg-[var(--chef-accent-strong)] text-white'
          }`}
        >
          {step.bypass ? 'Off' : 'On'}
        </button>
        <button
          type="button"
          title="Remove from recipe"
          onClick={(event) => {
            event.stopPropagation()
            onRemove(step.uid)
          }}
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

      {quick.length > 0 && (
        <div className="flex flex-wrap gap-1 px-3 pt-2">
          {quick.map(([name, settings]) => (
            <button
              key={name}
              type="button"
              onClick={() => onParamsChange(step.uid, settings)}
              className={`rounded-full border px-2 py-0.5 text-[11px] transition ${
                isQuickActive(settings)
                  ? 'border-[var(--chef-accent-strong)] bg-[var(--chef-accent)]/40 text-[var(--chef-accent-strong)]'
                  : 'border-[var(--chef-border)] text-[var(--chef-muted)] hover:border-[var(--chef-accent-strong)] hover:text-[var(--chef-text)]'
              }`}
            >
              {name}
            </button>
          ))}
        </div>
      )}

      <div className="space-y-0.5 px-3 py-2">
        {main.map((param) => (
          <Control
            key={param.name}
            param={param}
            value={values[param.name]}
            onChange={handleParam}
          />
        ))}

        {advanced.length > 0 && (
          <details className="group pt-1">
            <summary className="flex cursor-pointer list-none items-center gap-1 text-[11px] text-[var(--chef-muted)] hover:text-[var(--chef-text)]">
              <ChevronRight className="size-3 transition group-open:rotate-90" />
              Advanced ({advanced.length})
            </summary>
            <div className="mt-1 space-y-0.5 border-l border-[var(--chef-border)] pl-2">
              {advanced.map((param) => (
                <Control
                  key={param.name}
                  param={param}
                  value={values[param.name]}
                  onChange={handleParam}
                />
              ))}
            </div>
          </details>
        )}
      </div>

      {definition.listen_for && (
        <p className="flex items-start gap-1.5 border-t border-[var(--chef-border)] px-3 py-1.5 text-[11px] leading-snug text-[var(--chef-muted)]">
          <Ear className="mt-px size-3 shrink-0 text-[var(--chef-accent-strong)]" />
          <span>
            <span className="font-medium text-[var(--chef-text)]">Listen for: </span>
            {definition.listen_for}
          </span>
        </p>
      )}
    </div>
  )
}

export const RecipeCard = memo(RecipeCardImpl)
