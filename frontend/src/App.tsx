/**
 * Audio Chef -- CyberChef, but for audio.
 *
 * This component owns all of the global state and wires the three columns together:
 *
 *   Column 1  Operations Palette  (drag source)
 *   Column 2  The Recipe          (ordered list of operations + their parameters)
 *   Column 3  Input / Output      (two Wavesurfer views)
 *
 * Data flow:
 *   1. the tool catalogue is fetched once from GET /operations
 *   2. the WAV is POSTed once to /upload, which returns a file_id
 *   3. every bake POSTs {file_id, recipe} to /process and gets a rendered WAV back
 *
 * With Auto-Bake on, step 3 fires (debounced) whenever the recipe or any parameter
 * changes, which is what makes the app feel live.  An EMPTY recipe never bakes: there is
 * nothing to hear that the Input panel is not already showing, and re-encoding the file
 * the instant it was uploaded was pure noise.
 */

import { DragDropContext, type DropResult } from '@hello-pangea/dnd'
import { ChefHat, Upload } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchOperations, processRecipe, uploadFile } from './api'
import { ColumnSplitter } from './components/ColumnSplitter'
import { ListenStats } from './components/ListenStats'
import { OperationsPalette } from './components/OperationsPalette'
import { Recipe } from './components/Recipe'
import { Recorder } from './components/Recorder'
import { Toolbar } from './components/Toolbar'
import { WaveformViewer, type RegionSpec } from './components/WaveformViewer'
import type { Preset } from './presets'
import { REGION_OPS } from './regions'
import type { BakeResult, OperationDef, ParamValue, RecipeStep } from './types'

/** A bake faster than this never shows the "baking…" overlay -- it would only flicker. */
const BUSY_DELAY_MS = 150

/**
 * What the file picker advertises and what a desktop drop is checked against.
 *
 * This is a courtesy list, NOT the rule: the backend has no whitelist, because libsndfile
 * already knows exactly what it can open and probe() rejects the rest with a readable
 * message.  These are simply the ones worth naming in a browse dialog.
 */
const ACCEPTED_EXTENSIONS = [
  '.wav', '.mp3', '.flac', '.ogg', '.oga', '.opus', '.aiff', '.aif', '.aifc', '.w64',
  '.caf', '.au',
]
// No `audio/*` alongside these: it would let the picker offer m4a/AAC, which
// libsndfile cannot decode and the upload would reject after the wait.
const ACCEPT_ATTR = ACCEPTED_EXTENSIONS.join(',')

// --- Column layout ------------------------------------------------------------------
// Columns 1 and 2 are sized in px and dragged by the splitters between them; column 3 is
// whatever is left, so the waveforms always take the slack.
const PALETTE_DEFAULT = 280
const RECIPE_DEFAULT = 360
const PALETTE_MIN = 200
const PALETTE_MAX = 560
const RECIPE_MIN = 260
const RECIPE_MAX = 760
/** Column 3 never gets squeezed below this -- a narrower waveform is unreadable. */
const LISTEN_MIN = 360
const WIDTH_KEYS = { palette: 'chef.paletteWidth', recipe: 'chef.recipeWidth' } as const

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value))

/**
 * Reads a persisted column width.  localStorage throws outright in some privacy modes, so
 * every read is guarded and simply falls back to the default -- the same defensive shape
 * the blob URLs use below.
 */
function storedWidth(key: string, fallback: number, min: number, max: number) {
  try {
    const raw = window.localStorage.getItem(key)
    if (raw === null) return fallback
    const value = Number(raw)
    return Number.isFinite(value) ? clamp(value, min, max) : fallback
  } catch {
    return fallback
  }
}

/** Builds a fresh recipe step with every parameter set to its schema default. */
function newStep(op: OperationDef, overrides?: Record<string, ParamValue>): RecipeStep {
  const params: Record<string, ParamValue> = {}
  for (const param of op.params) params[param.name] = param.default
  // A preset only names the few parameters that make the effect; the rest keep the
  // backend's defaults, so nothing here can drift from the OPERATIONS schema.
  if (overrides) {
    for (const param of op.params) {
      if (param.name in overrides) params[param.name] = overrides[param.name]
    }
  }
  return {
    // crypto.randomUUID keeps React keys stable even when the same op is added twice.
    uid: `${op.id}-${crypto.randomUUID()}`,
    op: op.id,
    bypass: false,
    params,
  }
}

export default function App() {
  const [operations, setOperations] = useState<OperationDef[]>([])
  const [recipe, setRecipe] = useState<RecipeStep[]>([])

  const [fileId, setFileId] = useState<string | null>(null)
  const [filename, setFilename] = useState<string | null>(null)
  const [fileInfo, setFileInfo] = useState<string | null>(null)
  const [inputUrl, setInputUrl] = useState<string | null>(null)
  // The Input's length in seconds, kept as a number (fileInfo below is display text).
  // The region needs it: `end = 0` means "to the end" to the backend, and a handle still
  // has to be drawn somewhere concrete.
  const [inputDuration, setInputDuration] = useState(0)
  // uid of the step whose time parameters the waveform region is driving, if any.
  const [activeUid, setActiveUid] = useState<string | null>(null)
  const [output, setOutput] = useState<BakeResult | null>(null)

  // Column widths.  Lazily initialised so localStorage is touched once, not per render.
  const [paletteWidth, setPaletteWidth] = useState(() =>
    storedWidth(WIDTH_KEYS.palette, PALETTE_DEFAULT, PALETTE_MIN, PALETTE_MAX),
  )
  const [recipeWidth, setRecipeWidth] = useState(() =>
    storedWidth(WIDTH_KEYS.recipe, RECIPE_DEFAULT, RECIPE_MIN, RECIPE_MAX),
  )
  const [autoBake, setAutoBake] = useState(true)
  const [busy, setBusy] = useState(false)
  const [showBusy, setShowBusy] = useState(false)
  const [dropping, setDropping] = useState(false)
  const [status, setStatus] = useState('Load a WAV file to start')
  const [error, setError] = useState<string | null>(null)

  // Only one bake may be in flight; a newer one aborts the older so a slow response
  // can never overwrite a newer waveform.
  const inFlight = useRef<AbortController | null>(null)
  // Held so the object URLs can be revoked (they leak otherwise).  These are refs, not
  // state updaters: React 19 StrictMode double-invokes updaters, so creating/revoking a
  // URL inside one leaked a blob on every upload.
  const previousInputUrl = useRef<string | null>(null)
  const previousOutputUrl = useRef<string | null>(null)
  // Counts dragenter/dragleave so moving over a child element does not clear the state.
  const dragDepth = useRef(0)
  // Delays the "baking…" overlay; a fast bake should update silently, not flash.
  const busyTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  // Each resize clamp needs the OTHER column's current width.  Reading it from a ref
  // keeps both handlers stable, so dragging one splitter does not re-render the other.
  const paletteWidthRef = useRef(paletteWidth)
  const recipeWidthRef = useRef(recipeWidth)

  // --- Load the tool catalogue from the backend -------------------------------------
  useEffect(() => {
    fetchOperations()
      .then(setOperations)
      .catch(() => setError('Backend unreachable — is uvicorn running on port 8000?'))
  }, [])

  // Nothing else frees these, so release both blobs when the workbench goes away.
  useEffect(
    () => () => {
      if (previousInputUrl.current) URL.revokeObjectURL(previousInputUrl.current)
      if (previousOutputUrl.current) URL.revokeObjectURL(previousOutputUrl.current)
      if (busyTimer.current !== null) clearTimeout(busyTimer.current)
    },
    [],
  )

  const clearOutput = useCallback(() => {
    if (previousOutputUrl.current) URL.revokeObjectURL(previousOutputUrl.current)
    previousOutputUrl.current = null
    setOutput(null)
  }, [])

  // --- Upload -----------------------------------------------------------------------
  const handleFile = useCallback(async (file: File) => {
    setError(null)
    setStatus('Uploading…')
    try {
      const info = await uploadFile(file)
      setFileId(info.file_id)
      setFilename(info.filename)
      setFileInfo(
        `${info.duration.toFixed(2)}s · ${info.sample_rate} Hz · ${info.channels}ch`,
      )
      setInputDuration(info.duration)

      if (previousInputUrl.current) URL.revokeObjectURL(previousInputUrl.current)
      const url = URL.createObjectURL(file)   // the input waveform is drawn locally
      previousInputUrl.current = url
      setInputUrl(url)

      setStatus('Ready — add an operation to the recipe')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
      setStatus('')
    }
  }, [])

  // --- Bake -------------------------------------------------------------------------
  const bake = useCallback(async () => {
    if (!fileId) return
    if (recipe.length === 0) {
      // An empty recipe is a no-op pipeline: show the Input panel, do not round-trip.
      inFlight.current?.abort()
      clearOutput()
      setStatus('Ready — add an operation to the recipe')
      return
    }

    inFlight.current?.abort()
    const controller = new AbortController()
    inFlight.current = controller

    setBusy(true)
    setError(null)
    if (busyTimer.current === null) {
      busyTimer.current = setTimeout(() => setShowBusy(true), BUSY_DELAY_MS)
    }
    try {
      const result = await processRecipe(fileId, recipe, controller.signal)
      if (previousOutputUrl.current) URL.revokeObjectURL(previousOutputUrl.current)
      previousOutputUrl.current = result.url
      setOutput(result)
      setStatus(`Baked in ${result.bakeMs.toFixed(0)} ms · ${result.duration.toFixed(2)}s out`)
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return  // superseded
      setError(err instanceof Error ? err.message : 'Processing failed')
    } finally {
      if (inFlight.current === controller) {
        inFlight.current = null
        setBusy(false)
        if (busyTimer.current !== null) clearTimeout(busyTimer.current)
        busyTimer.current = null
        setShowBusy(false)
      }
    }
  }, [fileId, recipe, clearOutput])

  // --- Auto-Bake: debounce so dragging a slider does not fire a request per pixel ----
  useEffect(() => {
    if (!autoBake) return
    const timer = setTimeout(bake, 300)
    return () => clearTimeout(timer)
  }, [autoBake, bake])

  // --- Column resizing --------------------------------------------------------------
  // Both clamps live here rather than in the splitter: only App knows the other column's
  // width, and the pair has to leave LISTEN_MIN for column 3 between them.
  const resizePalette = useCallback((next: number) => {
    const room = window.innerWidth - recipeWidthRef.current - LISTEN_MIN
    // Math.max keeps the minimum winning over the room budget: on a very narrow window
    // the columns overflow rather than collapsing into unusable slivers.
    setPaletteWidth(clamp(next, PALETTE_MIN, Math.max(PALETTE_MIN, Math.min(PALETTE_MAX, room))))
  }, [])

  const resizeRecipe = useCallback((next: number) => {
    const room = window.innerWidth - paletteWidthRef.current - LISTEN_MIN
    setRecipeWidth(clamp(next, RECIPE_MIN, Math.max(RECIPE_MIN, Math.min(RECIPE_MAX, room))))
  }, [])

  useEffect(() => {
    paletteWidthRef.current = paletteWidth
    recipeWidthRef.current = recipeWidth
    try {
      window.localStorage.setItem(WIDTH_KEYS.palette, String(paletteWidth))
      window.localStorage.setItem(WIDTH_KEYS.recipe, String(recipeWidth))
    } catch {
      // Private mode / storage disabled: the widths just do not survive a reload.
    }
  }, [paletteWidth, recipeWidth])

  // --- Recipe mutations -------------------------------------------------------------
  const addOperation = useCallback((op: OperationDef, index?: number) => {
    const step = newStep(op)
    setRecipe((current) => {
      const next = [...current]
      next.splice(index ?? next.length, 0, step)
      return next
    })
    // Adding an op with time parameters links it straight away, so its region is on the
    // waveform without a second click.
    if (op.id in REGION_OPS) setActiveUid(step.uid)
  }, [])

  const handleAdd = useCallback((op: OperationDef) => addOperation(op), [addOperation])
  const handleClear = useCallback(() => {
    setRecipe([])
    setActiveUid(null)
  }, [])
  const toggleAutoBake = useCallback(() => setAutoBake((value) => !value), [])

  /** Drops a whole starter recipe in, skipping any op the backend does not serve. */
  const applyPreset = useCallback(
    (preset: Preset) => {
      const steps = preset.steps.flatMap((entry) => {
        const definition = operations.find((op) => op.id === entry.op)
        return definition ? [newStep(definition, entry.params)] : []
      })
      if (steps.length > 0) {
        setRecipe(steps)
        setActiveUid(steps.find((step) => step.op in REGION_OPS)?.uid ?? null)
      }
    },
    [operations],
  )

  const updateParam = useCallback((uid: string, name: string, value: ParamValue) => {
    setRecipe((current) =>
      current.map((step) =>
        step.uid === uid ? { ...step, params: { ...step.params, [name]: value } } : step,
      ),
    )
  }, [])

  /** Several parameters of one step in a SINGLE update.
   *
   *  A region drag moves start and end together; two updateParam calls would queue two
   *  renders and two Auto-Bake dependency changes for what is one edit. */
  const updateParams = useCallback((uid: string, values: Record<string, ParamValue>) => {
    setRecipe((current) =>
      current.map((step) =>
        step.uid === uid ? { ...step, params: { ...step.params, ...values } } : step,
      ),
    )
  }, [])

  const toggleBypass = useCallback((uid: string) => {
    setRecipe((current) =>
      current.map((step) => (step.uid === uid ? { ...step, bypass: !step.bypass } : step)),
    )
  }, [])

  const removeStep = useCallback((uid: string) => {
    setRecipe((current) => current.filter((step) => step.uid !== uid))
    // Otherwise the region would keep pointing at a card that no longer exists.
    setActiveUid((current) => (current === uid ? null : current))
  }, [])

  // --- The waveform region, and the card it is bound to ------------------------------
  const activeStep = recipe.find((step) => step.uid === activeUid) ?? null
  const binding = activeStep ? REGION_OPS[activeStep.op] : undefined

  const region: RegionSpec | null =
    activeStep && binding && inputDuration > 0
      ? {
          start: Number(activeStep.params[binding.start] ?? 0),
          // The editor reads `end <= 0` as "run to the end of the file", so a stored 0 is
          // not an empty region -- it is the whole tail.  Draw it that way.
          end:
            Number(activeStep.params[binding.end] ?? 0) > 0
              ? Number(activeStep.params[binding.end])
              : inputDuration,
          color: binding.color,
        }
      : null

  const onRegionChange = useCallback(
    (start: number, end: number) => {
      if (!activeUid || !binding) return
      // Milliseconds are as fine as these controls go, and rounding keeps the number
      // inputs from showing 1.2000000000000002 after a drag.
      updateParams(activeUid, {
        [binding.start]: Number(start.toFixed(3)),
        [binding.end]: Number(end.toFixed(3)),
      })
    },
    [activeUid, binding, updateParams],
  )

  // --- Drag and drop ----------------------------------------------------------------
  const onDragEnd = useCallback(
    (result: DropResult) => {
      const { source, destination } = result
      if (!destination || destination.droppableId !== 'recipe') return

      if (source.droppableId === 'palette') {
        // Dragged in from column 1: the palette id is "palette-<operation id>".
        const opId = result.draggableId.replace(/^palette-/, '')
        const definition = operations.find((op) => op.id === opId)
        if (definition) addOperation(definition, destination.index)
        return
      }

      // Reordering inside the recipe -- order changes the sound, so this is a real edit.
      setRecipe((current) => {
        const next = [...current]
        const [moved] = next.splice(source.index, 1)
        next.splice(destination.index, 0, moved)
        return next
      })
    },
    [operations, addOperation],
  )

  // --- Dropping a WAV from the desktop ----------------------------------------------
  // (HTML5 file DnD; @hello-pangea/dnd works off pointer events, so the two never meet.)
  const onFileDragEnter = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    dragDepth.current += 1
    setDropping(true)
  }, [])

  const onFileDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'copy'
  }, [])

  const onFileDragLeave = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    dragDepth.current = Math.max(0, dragDepth.current - 1)
    if (dragDepth.current === 0) setDropping(false)
  }, [])

  const onFileDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault()
      dragDepth.current = 0
      setDropping(false)
      const file = event.dataTransfer.files?.[0]
      if (!file) return
      const name = file.name.toLowerCase()
      if (!ACCEPTED_EXTENSIONS.some((ext) => name.endsWith(ext))) {
        setError('Unsupported file type — try WAV, MP3, FLAC, OGG/Opus or AIFF.')
        return
      }
      void handleFile(file)
    },
    [handleFile],
  )

  const onPickFile = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0]
      if (file) void handleFile(file)
      event.target.value = ''   // allow re-selecting the same file
    },
    [handleFile],
  )

  // --- Export -----------------------------------------------------------------------
  const exportWav = useCallback(() => {
    if (!output) return
    const link = document.createElement('a')
    link.href = output.url
    // Strip whatever extension came in, not just .wav -- the output is always a WAV, so
    // song.mp3 must not export as audio-chef-song.mp3.wav.
    link.download = `audio-chef-${filename?.replace(/\.[^.]+$/, '') ?? 'output'}.wav`
    link.click()
  }, [output, filename])

  return (
    <div className="flex h-screen flex-col bg-[var(--chef-bg)] text-[var(--chef-text)]">
      <Toolbar
        filename={filename}
        fileInfo={fileInfo}
        autoBake={autoBake}
        onToggleAutoBake={toggleAutoBake}
        onBake={bake}
        onExport={exportWav}
        canBake={Boolean(fileId) && recipe.length > 0}
        canExport={Boolean(output)}
        busy={busy}
        status={status}
        error={error}
      />

      <DragDropContext onDragEnd={onDragEnd}>
        {/* The strict 3-column workbench: palette | recipe | input+output, with a
            draggable gutter between each pair.  Columns 1 and 2 are explicit px so the
            splitters have something to write to; column 3 takes the remainder. */}
        <main
          className="grid min-h-0 flex-1"
          style={{
            gridTemplateColumns: `${paletteWidth}px auto ${recipeWidth}px auto 1fr`,
          }}
        >
          <OperationsPalette operations={operations} onAdd={handleAdd} />

          <ColumnSplitter
            width={paletteWidth}
            onResize={resizePalette}
            label="Operations column width"
          />

          <Recipe
            recipe={recipe}
            operations={operations}
            hasFile={Boolean(fileId)}
            activeUid={activeUid}
            onSelect={setActiveUid}
            onParamChange={updateParam}
            onToggleBypass={toggleBypass}
            onRemove={removeStep}
            onClear={handleClear}
            onApplyPreset={applyPreset}
          />

          <ColumnSplitter
            width={recipeWidth}
            onResize={resizeRecipe}
            label="Recipe column width"
          />

          <section
            onDragEnter={onFileDragEnter}
            onDragOver={onFileDragOver}
            onDragLeave={onFileDragLeave}
            onDrop={onFileDrop}
            className={`relative flex min-h-0 flex-col overflow-y-auto transition ${
              dropping ? 'bg-[var(--chef-accent)]/15' : ''
            }`}
          >
            <header className="flex items-center gap-3 border-b border-[var(--chef-border)] px-4 py-3">
              <h2 className="text-xs font-semibold uppercase tracking-[0.18em] text-[var(--chef-muted)]">
                <span className="text-[var(--chef-accent-strong)]">3</span> · Listen
              </h2>
              {inputUrl && (
                <div className="ml-auto flex items-center gap-2">
                  <Recorder onTake={handleFile} variant="compact" />
                  <label className="flex cursor-pointer items-center gap-1.5 rounded-md border border-[var(--chef-border)] px-2.5 py-1 text-xs transition hover:border-[var(--chef-accent-strong)]">
                    <Upload className="size-3.5" />
                    Load another WAV
                    <input
                      type="file"
                      accept={ACCEPT_ATTR}
                      className="hidden"
                      onChange={onPickFile}
                    />
                  </label>
                </div>
              )}
            </header>

            {inputUrl ? (
              <div className="space-y-4 p-4">
                <WaveformViewer
                  title="Input"
                  url={inputUrl}
                  accent="#22c55e"
                  emptyHint=""
                  region={region}
                  onRegionChange={onRegionChange}
                >
                  {binding && (
                    <span className="text-[11px] text-[var(--chef-muted)]">
                      {binding.label} — drag the edges
                    </span>
                  )}
                </WaveformViewer>
                <WaveformViewer
                  title="Output"
                  url={output?.url ?? null}
                  accent="#15803d"
                  busy={showBusy}
                  emptyHint={
                    recipe.length === 0
                      ? 'Add an operation to the recipe to hear a result'
                      : 'Baking…'
                  }
                />
                {/* The numbers behind the two pictures.  Always mounted: before the first
                    bake it explains itself rather than leaving the column half empty. */}
                <ListenStats
                  stats={output?.stats ?? null}
                  recipe={recipe}
                  operations={operations}
                  bakeMs={output?.bakeMs ?? 0}
                />
              </div>
            ) : (
              /* Before anything is loaded, column 3 is one big obvious target rather
                 than two empty panels with a file picker hidden in a header. */
              <div className="flex flex-1 flex-col items-center justify-center gap-4 p-8">
                <label
                  className={`flex w-full max-w-md cursor-pointer flex-col items-center gap-3 rounded-xl border-2 border-dashed px-8 py-14 text-center transition ${
                    dropping
                      ? 'border-[var(--chef-accent-strong)] bg-[var(--chef-accent)]/25'
                      : 'border-[var(--chef-border)] hover:border-[var(--chef-accent-strong)]'
                  }`}
                >
                  <ChefHat className="size-10 text-[var(--chef-accent-strong)]" />
                  <span className="text-base font-medium">Drop a WAV here, or browse</span>
                  <span className="text-xs leading-relaxed text-[var(--chef-muted)]">
                    Then drag operations from column 1 into your recipe — they run top to
                    bottom, and you hear the result as you tweak them.
                  </span>
                  <input
                    type="file"
                    accept={ACCEPT_ATTR}
                    className="hidden"
                    onChange={onPickFile}
                  />
                </label>

                {/* Outside the label above on purpose: a click anywhere inside a <label>
                    opens its file input, so nesting this would pop the file picker the
                    instant you pressed Record. */}
                <div className="flex w-full max-w-md flex-col items-center gap-3">
                  <span className="flex w-full items-center gap-3 text-[11px] uppercase tracking-[0.18em] text-[var(--chef-muted)]">
                    <span className="h-px flex-1 bg-[var(--chef-border)]" />
                    or
                    <span className="h-px flex-1 bg-[var(--chef-border)]" />
                  </span>
                  <Recorder onTake={handleFile} variant="large" />
                </div>
              </div>
            )}

            {dropping && inputUrl && (
              <div className="pointer-events-none absolute inset-3 flex items-center justify-center rounded-xl border-2 border-dashed border-[var(--chef-accent-strong)] bg-[var(--chef-bg)]/85 text-sm text-[var(--chef-accent-strong)]">
                Drop to replace the input
              </div>
            )}
          </section>
        </main>
      </DragDropContext>
    </div>
  )
}
