/**
 * Maps the `icon` field that the backend sends for each operation to an actual
 * lucide-react component.  Keeping it explicit (rather than a dynamic import of the whole
 * icon set) means only these few icons end up in the bundle.
 */

import {
  AudioLines,
  BarChart3,
  Blend,
  Filter,
  Gauge,
  Layers,
  Mic,
  Minimize2,
  Piano,
  Repeat,
  Scissors,
  SlidersHorizontal,
  Undo2,
  VolumeX,
  Wand2,
  Waves,
  type LucideIcon,
} from 'lucide-react'

const ICONS: Record<string, LucideIcon> = {
  Waves,
  SlidersHorizontal,
  Filter,
  AudioLines,
  Scissors,
  Gauge,
  Minimize2,
  Mic,
  Piano,
  Layers,
  Repeat,
  Blend,
  BarChart3,
  Undo2,
  VolumeX,
}

/** Falls back to a generic wand for any icon name we do not know about yet. */
export function iconFor(name: string): LucideIcon {
  return ICONS[name] ?? Wand2
}
