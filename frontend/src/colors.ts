/**
 * One colour per loaded file, shown on its tab and in the recipe header.
 *
 * Handed out in LOAD order from the same counter that mints source ids, so a tab keeps its
 * colour when another tab closes.  Picked to stay distinct from each other AND from the
 * greens the Input/Output waveforms use, in both themes.
 */
const SOURCE_COLORS = [
  '#3b82f6', // blue
  '#f97316', // orange
  '#a855f7', // violet
  '#ec4899', // pink
  '#14b8a6', // teal
  '#eab308', // amber
  '#ef4444', // red
  '#6366f1', // indigo
]

/** Colour for the n-th file loaded this session (1-based, like the source ids). */
export function sourceColor(serial: number): string {
  return SOURCE_COLORS[(serial - 1) % SOURCE_COLORS.length]
}
