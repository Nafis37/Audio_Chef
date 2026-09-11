/**
 * Starter recipes.
 *
 * A blank recipe column is the hardest moment for a new user: seven tools, no idea which
 * one to reach for.  A preset is just a pre-filled recipe -- clicking one drops real
 * cards into column 2 that can then be dragged, retuned or bypassed like any others.
 *
 * A preset only ever names an operation id and the *few* parameters that make the effect;
 * everything else is filled in from the schema default by `newStep()` in App.tsx, and an
 * op id the backend does not serve is skipped.  So OPERATIONS in dsp_engine.py stays the
 * one source of truth for what a parameter is and what range it has -- nothing here can
 * drift away from it.
 */

import type { ParamValue } from './types'

export interface Preset {
  id: string
  label: string
  description: string
  icon: string                                    // resolved through icons.ts
  steps: { op: string; params: Record<string, ParamValue> }[]
}

export const PRESETS: Preset[] = [
  {
    id: 'clean-voice',
    label: 'Clean Up Voice',
    description: 'De-noise, cut rumble, lift presence, then even out the level.',
    icon: 'Mic',
    steps: [
      { op: 'noise_remover', params: { amount: 2.0, floor: 0.05 } },
      { op: 'equalizer', params: { low_gain: -6, low_freq: 100, mid_gain: 3, high_gain: 2 } },
      { op: 'compressor', params: { threshold: -18, ratio: 4, makeup: 6 } },
    ],
  },
  {
    id: 'cathedral',
    label: 'Cathedral',
    description: 'A big Schroeder reverb: large room, long RT60, plenty of wet.',
    icon: 'AudioLines',
    steps: [
      { op: 'echo_reverb', params: { mode: 'reverb', room_size: 0.9, decay: 5.0, mix: 0.6 } },
    ],
  },
  {
    id: 'chipmunk',
    label: 'Chipmunk',
    description: 'Phase-vocoder pitch shift up a fifth, same duration.',
    icon: 'Gauge',
    steps: [{ op: 'voice_changer', params: { mode: 'pitch', semitones: 7 } }],
  },
  {
    id: 'robot',
    label: 'Robot',
    description: 'Phase reset every frame -- a monotone, metallic voice.',
    icon: 'Waves',
    steps: [{ op: 'voice_changer', params: { mode: 'robot', semitones: 0 } }],
  },
  {
    id: 'telephone',
    label: 'Telephone',
    description: 'Kill everything outside the 300 Hz - 3 kHz band with the peaking EQ.',
    icon: 'SlidersHorizontal',
    steps: [
      {
        op: 'equalizer',
        params: {
          low_gain: -24, low_freq: 150,
          mid_gain: 8, mid_freq: 1500,
          high_gain: -24, high_freq: 5000,
          q: 1.2,
        },
      },
    ],
  },
]
