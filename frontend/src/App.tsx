/**
 * Audio Chef -- CyberChef, but for audio.
 *
 * This component owns all of the global state and wires the columns together:
 *
 *   Tabs                one colour-coded, browser-style tab per loaded file
 *   Column 1  Operations Palette  (drag source)
 *   Column 2  The Recipe          (the open tab's chain)
 *   Column 3  Input / Output      (two Wavesurfer views of the open tab)
 *
 * Every file is its own tab with its own recipe: open a tab, build its recipe, hear it,
 * save it, close it.  Tabs are independent -- an `assemble` card can still pull a piece of
 * another file's processed audio into this one, and the backend walks that in
 * dsp/graph.py.
 *
 * Data flow:
 *   1. the tool catalogue is fetched once from GET /operations
 *   2. each file is POSTed once to /upload, which returns a file_id
 *   3. every bake POSTs the whole project to /process and gets a rendered WAV back
 *
 * With Auto-Bake on, step 3 fires (debounced) whenever any chain or parameter changes.
 *
 * Seeing and hearing the difference:
 *   - each panel shows a spectrogram under its waveform, drawn by the backend from our own
 *     STFT -- GET /spectrogram/file/{file_id} for the input, /spectrogram/bake/{bake_id}
 *     for each fresh output
 *   - the A/B switch above the panels (or the B key) hands playback between Original and
 *     Processed at the same point in the clip, so a change is heard, not remembered
 * A project where EVERY chain is empty never bakes: there is nothing to hear that the
 * Input panel is not already showing.
 */

import { DragDropContext, type DropResult } from '@hello-pangea/dnd'
import { ChefHat, Pause, Play, Upload } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { fetchOperations, fetchSpectrogram, processGraph, toWire, uploadFile } from './api'
import { ColumnSplitter } from './components/ColumnSplitter'
import { ListenStats } from './components/ListenStats'
import { OperationsPalette } from './components/OperationsPalette'
import { Recipe } from './components/Recipe'
import { Recorder } from './components/Recorder'
import { SourceTabs } from './components/SourceTabs'
import { Toolbar } from './components/Toolbar'
import { WaveformViewer, type RegionSpec, type WaveformHandle } from './components/WaveformViewer'
import { sourceColor } from './colors'
import type { Preset } from './presets'
import { regionFor } from './regions'
import { SourcesProvider, type SourceOption } from './sources'
import {
  type BakeResult,
  type OperationDef,
  type ParamValue,
  type RecipeStep,
  type Source,
  type SpectrogramData,
} from './types'

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
function newStep(op: OperationDef): RecipeStep {
  const params: Record<string, ParamValue> = {}
  for (const param of op.params) params[param.name] = param.default
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

  // --- The project ------------------------------------------------------------------
  const [sources, setSources] = useState<Source[]>([])
  /** The open tab: whose recipe column 2 edits and whose audio column 3 plays. */
  const [editing, setEditing] = useState<string | null>(null)

  const [output, setOutput] = useState<BakeResult | null>(null)

  // Spectrograms: one per loaded source (its raw file), and one for the latest bake.
  const [inputSpectrograms, setInputSpectrograms] = useState<Map<string, SpectrogramData>>(
    () => new Map(),
  )
  const [outputSpectrogram, setOutputSpectrogram] = useState<SpectrogramData | null>(null)

  // A/B: which panel you are listening to, and handles to drive both.
  const [hearing, setHearing] = useState<'input' | 'output'>('input')
  const inputView = useRef<WaveformHandle>(null)
  const outputView = useRef<WaveformHandle>(null)

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
  const inputUrls = useRef(new Map<string, string>())
  const previousOutputUrl = useRef<string | null>(null)
  // Source ids must stay unique for the lifetime of the session, including across
  // removals -- an assemble card may still be pointing at an id that has gone.
  const nextSourceId = useRef(1)
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

  // Nothing else frees these, so release every blob when the workbench goes away.
  useEffect(
    () => () => {
      for (const url of inputUrls.current.values()) URL.revokeObjectURL(url)
      inputUrls.current.clear()
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
  /** Adds a file to the project.  Never replaces: that is what the rail's x is for. */
  const handleFile = useCallback(async (file: File) => {
    setError(null)
    setStatus('Uploading…')
    try {
      const info = await uploadFile(file)
      const serial = nextSourceId.current++
      const id = `s${serial}`

      const url = URL.createObjectURL(file)   // the input waveform is drawn locally
      inputUrls.current.set(id, url)

      setSources((current) => [
        ...current,
        {
          id,
          color: sourceColor(serial),
          fileId: info.file_id,
          filename: info.filename,
          url,
          sampleRate: info.sample_rate,
          channels: info.channels,
          duration: info.duration,
          recipe: [],
          activeUid: null,
        },
      ])
      // Opening a file opens its tab, as a browser does with a new page.
      setEditing(id)

      // The input spectrogram is a nicety: if it fails the waveform is still there.
      fetchSpectrogram('file', info.file_id)
        .then((data) => setInputSpectrograms((current) => new Map(current).set(id, data)))
        .catch(() => {})

      setStatus('Ready — add an operation to the recipe')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
      setStatus('')
    }
  }, [])

  const removeSource = useCallback(
    (id: string) => {
      // Revoke out here, not inside an updater: StrictMode double-invokes updaters, and
      // revoking twice would kill a URL that a re-render is still handing to wavesurfer.
      const url = inputUrls.current.get(id)
      if (url) {
        URL.revokeObjectURL(url)
        inputUrls.current.delete(id)
      }

      // Every setState below is a plain call on a value computed here.  Nesting them
      // inside the setSources updater would fire them twice under StrictMode for what
      // is one edit -- the same trap the blob URLs above are avoiding.
      const index = sources.findIndex((source) => source.id === id)
      const remaining = sources.filter((source) => source.id !== id)
      // Closing the open tab opens its right-hand neighbour, else the left one -- the
      // tab that slides under the pointer, as in a browser.
      const fallback = remaining[Math.min(index, remaining.length - 1)]?.id ?? null

      setSources(remaining)
      setInputSpectrograms((current) => {
        const next = new Map(current)
        next.delete(id)
        return next
      })
      // An assemble card still naming the closed tab keeps the id on purpose: the card
      // shows "(removed)" and the backend refuses the bake with a readable message, which
      // beats silently repointing it at a different file.
      if (editing === id) setEditing(fallback)
      if (remaining.length === 0) clearOutput()
    },
    [sources, editing, clearOutput],
  )

  // --- Chain helpers ----------------------------------------------------------------
  // Column 2 edits ONE tab's chain at a time; every mutation below goes through these.
  const editChain = useCallback(
    (id: string | null, update: (steps: RecipeStep[]) => RecipeStep[]) => {
      setSources((current) =>
        current.map((source) =>
          source.id === id ? { ...source, recipe: update(source.recipe) } : source,
        ),
      )
    },
    [],
  )

  const setActiveUid = useCallback((id: string | null, uid: string | null) => {
    setSources((current) =>
      current.map((source) =>
        source.id === id ? { ...source, activeUid: uid } : source,
      ),
    )
  }, [])

  const editingSource = sources.find((source) => source.id === editing) ?? null
  const recipe = useMemo(() => editingSource?.recipe ?? [], [editingSource])
  const activeUid = editingSource?.activeUid ?? null

  // --- Bake -------------------------------------------------------------------------
  /** The request that renders one tab: its file through its own recipe, nothing else. */
  const renderRequest = useCallback(
    (id: string) => ({
      sources: sources.map((source) => ({
        id: source.id,
        file_id: source.fileId,
        recipe: toWire(source.recipe),
      })),
      // No shared final chain: every tab stands on its own.
      master: [],
      output_source: id,
      apply_master: false,
    }),
    [sources],
  )

  const bake = useCallback(async () => {
    if (!editing) return

    // An empty recipe is a no-op pipeline: show the Input panel, do not round-trip.
    if (recipe.length === 0) {
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
      const result = await processGraph(renderRequest(editing), controller.signal)
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
  }, [editing, recipe, renderRequest, clearOutput])

  // --- The output spectrogram follows each bake ------------------------------------
  const bakeId = output?.bakeId ?? null
  useEffect(() => {
    // No bake, no picture: the Output panel is handed null while `output` is null.
    if (!bakeId) return
    // A newer bake supersedes this fetch; keep showing the previous picture meanwhile,
    // which reads as "updating" rather than flashing empty after every slider move.
    const controller = new AbortController()
    fetchSpectrogram('bake', bakeId, controller.signal)
      .then(setOutputSpectrogram)
      .catch(() => {})
    return () => controller.abort()
  }, [bakeId])

  // --- A/B: hand playback from one panel to the other at the same point ------------
  const listenTo = useCallback((side: 'input' | 'output') => {
    const from = (side === 'input' ? outputView : inputView).current
    const to = (side === 'input' ? inputView : outputView).current
    setHearing(side)
    if (!from || !to) return
    const wasPlaying = from.isPlaying() || to.isPlaying()
    // Proportional, not absolute: after a speed change the same moment of the clip sits
    // at a different number of seconds in the two files.
    const fromDuration = from.getDuration()
    const fraction = fromDuration > 0 ? from.getCurrentTime() / fromDuration : 0
    from.pause()
    if (!to.isPlaying()) to.setTime(Math.min(fraction * to.getDuration(), to.getDuration()))
    if (wasPlaying) to.play()
  }, [])

  const toggleListening = useCallback(() => {
    const side = hearing === 'input' ? 'output' : 'input'
    if (side === 'output' && !output) return
    listenTo(side)
  }, [hearing, output, listenTo])

  const playPauseCurrent = useCallback(() => {
    const view = (hearing === 'input' ? inputView : outputView).current
    if (!view) return
    if (view.isPlaying()) view.pause()
    else view.play()
  }, [hearing])

  // The B key flips A/B -- the quickest way to hear what a step does during a demo.
  // Ignored while typing in a control, where "b" is just a letter.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'b' && event.key !== 'B') return
      if (event.metaKey || event.ctrlKey || event.altKey) return
      const target = event.target as HTMLElement | null
      if (target?.closest('input, select, textarea, [contenteditable="true"]')) return
      event.preventDefault()
      toggleListening()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [toggleListening])

  // Pressing play on one panel silences the other, so you only ever hear one.
  const onInputPlay = useCallback(() => {
    outputView.current?.pause()
    setHearing('input')
  }, [])
  const onOutputPlay = useCallback(() => {
    inputView.current?.pause()
    setHearing('output')
  }, [])

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
  // All of these act on the chain currently open in column 2.
  const addOperation = useCallback(
    (op: OperationDef, index?: number, chain: string | null = editing) => {
      const step = newStep(op)
      editChain(chain, (current) => {
        const next = [...current]
        next.splice(index ?? next.length, 0, step)
        return next
      })
      // Adding an op with time parameters links it straight away, so its region is on the
      // waveform without a second click.
      if (regionFor(op.id, step.params)) setActiveUid(chain, step.uid)
    },
    [editing, editChain, setActiveUid],
  )

  const handleAdd = useCallback((op: OperationDef) => addOperation(op), [addOperation])

  const handleClear = useCallback(() => {
    editChain(editing, () => [])
    setActiveUid(editing, null)
  }, [editing, editChain, setActiveUid])

  // A preset REPLACES the open chain.  Each step starts as newStep() -- every param at its
  // schema default -- and the preset only overrides the ones it names, so a preset can
  // never hold a stale schema.  An op the catalogue does not have is skipped.
  const applyPreset = useCallback(
    (preset: Preset) => {
      const byId = new Map(operations.map((op) => [op.id, op]))
      const steps = preset.steps.flatMap(({ op: id, params }) => {
        const op = byId.get(id)
        if (!op) return []
        const step = newStep(op)
        return [{ ...step, params: { ...step.params, ...params } }]
      })
      editChain(editing, () => steps)
      // Same rule as addOperation: a step with a region is linked straight away.
      setActiveUid(editing, steps.find((step) => regionFor(step.op, step.params))?.uid ?? null)
    },
    [operations, editing, editChain, setActiveUid],
  )

  const toggleAutoBake = useCallback(() => setAutoBake((value) => !value), [])

  const updateParam = useCallback(
    (uid: string, name: string, value: ParamValue) => {
      editChain(editing, (current) =>
        current.map((step) =>
          step.uid === uid ? { ...step, params: { ...step.params, [name]: value } } : step,
        ),
      )
    },
    [editing, editChain],
  )

  /** Several parameters of one step in a SINGLE update.
   *
   *  A region drag moves start and end together; two updateParam calls would queue two
   *  renders and two Auto-Bake dependency changes for what is one edit. */
  const updateParams = useCallback(
    (uid: string, values: Record<string, ParamValue>) => {
      editChain(editing, (current) =>
        current.map((step) =>
          step.uid === uid ? { ...step, params: { ...step.params, ...values } } : step,
        ),
      )
    },
    [editing, editChain],
  )

  const toggleBypass = useCallback(
    (uid: string) => {
      editChain(editing, (current) =>
        current.map((step) => (step.uid === uid ? { ...step, bypass: !step.bypass } : step)),
      )
    },
    [editing, editChain],
  )

  const removeStep = useCallback(
    (uid: string) => {
      editChain(editing, (current) => current.filter((step) => step.uid !== uid))
      // Otherwise the region would keep pointing at a card that no longer exists.
      if (activeUid === uid) setActiveUid(editing, null)
    },
    [editing, editChain, activeUid, setActiveUid],
  )

  const selectStep = useCallback(
    (uid: string) => setActiveUid(editing, uid),
    [editing, setActiveUid],
  )

  // --- What the source dropdown sees ----------------------------------------------
  /** Everything an assemble / voice_match card in THIS chain may point at.
   *
   *  The chain's own source is filtered out: a source referencing itself is always a
   *  cycle, so offering it would only ever produce a backend error. */
  const sourceOptions: SourceOption[] = useMemo(
    () =>
      sources
        .filter((source) => source.id !== editing)
        .map((source) => ({ id: source.id, label: source.filename })),
    [sources, editing],
  )

  // --- The waveform region, and the card it is bound to ------------------------------
  const activeStep = recipe.find((step) => step.uid === activeUid) ?? null
  // regionFor() also asks whether the step's current mode uses a region at all.
  const binding = activeStep ? regionFor(activeStep.op, activeStep.params) : undefined
  const regionDuration = editingSource?.duration ?? 0

  const region: RegionSpec | null =
    activeStep && binding && regionDuration > 0
      ? {
          start: Number(activeStep.params[binding.start] ?? 0),
          // The editor reads `end <= 0` as "run to the end of the file", so a stored 0 is
          // not an empty region -- it is the whole tail.  Draw it that way.
          end:
            Number(activeStep.params[binding.end] ?? 0) > 0
              ? Number(activeStep.params[binding.end])
              : regionDuration,
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
      // The droppable id carries its chain ("recipe:s2"), so a stale id from a previous
      // tab can never be mistaken for the current one.
      if (!destination || !destination.droppableId.startsWith('recipe:')) return
      const chain = destination.droppableId.slice('recipe:'.length)

      if (source.droppableId === 'palette') {
        // Dragged in from column 1: the palette id is "palette-<operation id>".
        const opId = result.draggableId.replace(/^palette-/, '')
        const definition = operations.find((op) => op.id === opId)
        if (definition) addOperation(definition, destination.index, chain)
        return
      }

      // Reordering inside the recipe -- order changes the sound, so this is a real edit.
      editChain(chain, (current) => {
        const next = [...current]
        const [moved] = next.splice(source.index, 1)
        next.splice(destination.index, 0, moved)
        return next
      })
    },
    [operations, addOperation, editChain],
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
      // Several files at once is now a normal thing to do, so take all of them.
      const files = Array.from(event.dataTransfer.files ?? [])
      if (files.length === 0) return
      const accepted = files.filter((file) =>
        ACCEPTED_EXTENSIONS.some((ext) => file.name.toLowerCase().endsWith(ext)),
      )
      if (accepted.length === 0) {
        setError('Unsupported file type — try WAV, MP3, FLAC, OGG/Opus or AIFF.')
        return
      }
      // Sequentially: each upload mints the next source id off a ref.
      void accepted.reduce(
        (chain, file) => chain.then(() => handleFile(file)),
        Promise.resolve(),
      )
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

  // --- Save a tab -------------------------------------------------------------------
  /**
   * Downloads one tab's file run through its own recipe, as a WAV.
   *
   * Always rendered afresh rather than reusing the Output panel's blob: for a moment after
   * a tab switch (the Auto-Bake debounce) that blob still holds the PREVIOUS tab, and a
   * save must never hand over the wrong file.  The one-off blob is released right after.
   */
  const saveSource = useCallback(
    async (id: string) => {
      const source = sources.find((s) => s.id === id)
      if (!source) return
      let url: string
      try {
        url = (await processGraph(renderRequest(id))).url
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Save failed')
        return
      }
      const link = document.createElement('a')
      link.href = url
      // Strip whatever extension came in, not just .wav -- the output is always a WAV,
      // so song.mp3 must not save as audio-chef-song.mp3.wav.
      link.download = `audio-chef-${source.filename.replace(/\.[^.]+$/, '')}.wav`
      link.click()
      setTimeout(() => URL.revokeObjectURL(url), 1000)
    },
    [sources, renderRequest],
  )

  const exportWav = useCallback(() => {
    if (editing) void saveSource(editing)
  }, [editing, saveSource])

  // --- Toolbar summary ---------------------------------------------------------------
  const projectInfo = editingSource
    ? `${editingSource.duration.toFixed(2)}s · ${editingSource.sampleRate} Hz · ${editingSource.channels}ch` +
      (sources.length > 1 ? ` · ${sources.length} tabs` : '')
    : null

  const hasAnySteps = recipe.length > 0

  return (
    <div className="flex h-screen flex-col bg-[var(--chef-bg)] text-[var(--chef-text)]">
      <Toolbar
        filename={editingSource?.filename ?? null}
        fileInfo={projectInfo}
        autoBake={autoBake}
        onToggleAutoBake={toggleAutoBake}
        onBake={bake}
        onExport={exportWav}
        canBake={sources.length > 0 && hasAnySteps}
        canExport={Boolean(editingSource)}
        busy={busy}
        status={status}
        error={error}
      />

      <SourceTabs
        sources={sources}
        activeId={editing}
        accept={ACCEPT_ATTR}
        onSelect={setEditing}
        onSave={saveSource}
        onClose={removeSource}
        onPick={onPickFile}
        onRecord={handleFile}
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

          {/* The source dropdown inside an assemble card reads this, at the leaf, so
              adding or removing a file does not re-render the memoised columns. */}
          <SourcesProvider value={sourceOptions}>
            <Recipe
              recipe={recipe}
              operations={operations}
              hasFile={sources.length > 0}
              activeUid={activeUid}
              selected={editing ?? ''}
              tabName={editingSource?.filename ?? ''}
              tabColor={editingSource?.color ?? 'transparent'}
              onSelect={selectStep}
              onParamChange={updateParam}
              onParamsChange={updateParams}
              onToggleBypass={toggleBypass}
              onRemove={removeStep}
              onClear={handleClear}
              onPreset={applyPreset}
            />
          </SourcesProvider>

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
            </header>

            {sources.length > 0 ? (
              <div className="space-y-4 p-4">
                {/* A/B: one transport for both panels.  Picking a side mid-playback carries
                    on from the same point in the other file. */}
                <div className="flex flex-wrap items-center gap-2 rounded-lg border border-[var(--chef-border)] bg-[var(--chef-panel)] px-3 py-2">
                  <button
                    type="button"
                    onClick={playPauseCurrent}
                    className="rounded-full border border-[var(--chef-border)] p-1.5 transition hover:border-[var(--chef-accent-strong)]"
                    title="Play / pause what you are listening to"
                  >
                    <PlayPauseIcon viewRef={hearing === 'input' ? inputView : outputView} />
                  </button>
                  <span className="text-xs text-[var(--chef-muted)]">Listening to</span>
                  <div className="flex rounded-md border border-[var(--chef-border)] bg-[var(--chef-inset)] p-0.5">
                    {(['input', 'output'] as const).map((side) => (
                      <button
                        key={side}
                        type="button"
                        disabled={side === 'output' && !output}
                        onClick={() => listenTo(side)}
                        className={`rounded px-3 py-1 text-xs transition disabled:opacity-30 ${
                          hearing === side
                            ? 'bg-[var(--chef-accent-strong)] font-medium text-white'
                            : 'text-[var(--chef-muted)] hover:text-[var(--chef-text)]'
                        }`}
                      >
                        {side === 'input' ? 'A · Original' : 'B · Processed'}
                      </button>
                    ))}
                  </div>
                  <span className="ml-auto text-[11px] text-[var(--chef-muted)]">
                    press <kbd className="rounded border border-[var(--chef-border)] px-1 font-mono">B</kbd> to flip
                  </span>
                </div>

                <WaveformViewer
                  title={`Input — ${editingSource?.filename ?? ''}`}
                  url={editingSource?.url ?? null}
                  accent="#22c55e"
                  emptyHint=""
                  region={region}
                  onRegionChange={onRegionChange}
                  ref={inputView}
                  spectrogram={inputSpectrograms.get(editing ?? '') ?? null}
                  highlighted={hearing === 'input'}
                  onPlay={onInputPlay}
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
                  ref={outputView}
                  spectrogram={output ? outputSpectrogram : null}
                  highlighted={hearing === 'output'}
                  onPlay={onOutputPlay}
                  emptyHint={
                    hasAnySteps
                      ? 'Baking…'
                      : 'Add an operation to the recipe to hear a result'
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
                  <span className="text-base font-medium">Drop audio here, or browse</span>
                  <span className="text-xs leading-relaxed text-[var(--chef-muted)]">
                    Load as many files as you like — each opens in its own tab with its
                    own recipe, ready to hear, save and close.
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

            {dropping && sources.length > 0 && (
              <div className="pointer-events-none absolute inset-3 flex items-center justify-center rounded-xl border-2 border-dashed border-[var(--chef-accent-strong)] bg-[var(--chef-bg)]/85 text-sm text-[var(--chef-accent-strong)]">
                <span className="flex items-center gap-2">
                  <Upload className="size-4" /> Drop to open it in a new tab
                </span>
              </div>
            )}
          </section>
        </main>
      </DragDropContext>
    </div>
  )
}

/**
 * The A/B bar's play icon.  wavesurfer owns the playing state, so this polls the handle a
 * few times a second rather than threading play/pause events up through App's state.
 */
function PlayPauseIcon({ viewRef }: { viewRef: React.RefObject<WaveformHandle | null> }) {
  const [playing, setPlaying] = useState(false)
  useEffect(() => {
    const timer = setInterval(() => setPlaying(viewRef.current?.isPlaying() ?? false), 200)
    return () => clearInterval(timer)
  }, [viewRef])
  return playing ? <Pause className="size-4" /> : <Play className="size-4" />
}
