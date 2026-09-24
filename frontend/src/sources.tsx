/**
 * The list of sources an `assemble` card may point at.
 *
 * Every other control is generated entirely from the backend's parameter schema, but the
 * set of loaded files is a fact about this browser session that the DSP catalogue cannot
 * know.  So the schema declares `control: "source"` and this supplies the options.
 *
 * Delivered by context rather than as a prop, and consumed at the LEAF (the one <select>
 * that needs it) on purpose.  Recipe and RecipeCard are both wrapped in memo(), and that
 * memoisation is load-bearing -- see the header of RecipeCard.tsx, where a slider drag
 * used to re-render both drag-and-drop columns at pointer rate.  Threading a fresh array
 * prop through those two would bust both memos every time a source was added or removed.
 */

import { createContext, useContext } from 'react'

export interface SourceOption {
  /** What gets submitted: the source id. */
  id: string
  /** What gets shown: the filename. */
  label: string
}

const SourcesContext = createContext<SourceOption[]>([])

export function SourcesProvider({
  value,
  children,
}: {
  value: SourceOption[]
  children: React.ReactNode
}) {
  return <SourcesContext.Provider value={value}>{children}</SourcesContext.Provider>
}

/** The sources a source-picking card in the CURRENT chain is allowed to reference. */
export function useSources(): SourceOption[] {
  return useContext(SourcesContext)
}
