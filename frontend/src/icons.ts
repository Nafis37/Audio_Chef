/**
 * Maps the `icon` field that the backend sends for each operation to an actual
 * lucide-react component.  Keeping it explicit (rather than a dynamic import of the whole
 * icon set) means only these few icons end up in the bundle.
 */

import {
  AudioLines,
  Gauge,
  Mic,
  Minimize2,
  Scissors,
  SlidersHorizontal,
  Wand2,
  Waves,
  type LucideIcon,
} from 'lucide-react'

const ICONS: Record<string, LucideIcon> = {
  Waves,
  SlidersHorizontal,
  AudioLines,
  Scissors,
  Gauge,
  Minimize2,
  Mic,
}

/** Falls back to a generic wand for any icon name we do not know about yet. */
export function iconFor(name: string): LucideIcon {
  return ICONS[name] ?? Wand2
}
