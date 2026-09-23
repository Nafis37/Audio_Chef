/**
 * Starter recipes.
 *
 * Each preset names op ids and ONLY the parameters it wants to move -- everything else
 * falls back to the schema default served by GET /operations, the same way a card dragged
 * in from the palette starts out.  So a preset can never go stale against the backend: a
 * renamed parameter is simply ignored and its default used, and an op id the catalogue no
 * longer has is skipped when the preset is applied (App.tsx `applyPreset`).
 *
 * No preset uses `assemble`: it needs a second source picked by hand, and a starter
 * recipe cannot know what else is loaded.
 */

import type { ParamValue } from './types'

export interface PresetStep {
  op: string
  params?: Record<string, ParamValue>
}

export interface Preset {
  id: string
  label: string
  /** One line for the dropdown's tooltip: what you will hear. */
  description: string
  steps: PresetStep[]
}

export const PRESETS: Preset[] = [
  {
    id: 'podcast',
    label: 'Podcast cleanup',
    description: 'Remove room hiss, add presence, then even out the level.',
    steps: [
      { op: 'noise_remover', params: { amount: 2, profile: 'auto' } },
      { op: 'equalizer', params: { bass_gain: -3, mid_gain: 3, mid_freq: 3000, treble_gain: 3 } },
      { op: 'compressor', params: { threshold: -24, ratio: 4 } },
    ],
  },
  {
    id: 'telephone',
    label: 'Telephone',
    description: 'Cut the lows and highs hard and push the midrange: a 300–3400 Hz line.',
    steps: [
      {
        op: 'equalizer',
        params: {
          bass_gain: -24,
          bass_freq: 400,
          mid_gain: 8,
          mid_freq: 1500,
          q: 0.7,
          treble_gain: -24,
          treble_freq: 3000,
        },
      },
      { op: 'compressor', params: { threshold: -30, ratio: 8 } },
    ],
  },
  {
    id: 'radio',
    label: 'Radio voice',
    description: 'Big bass, crisp top and heavy compression: the late-night DJ.',
    steps: [
      { op: 'equalizer', params: { bass_gain: 8, bass_freq: 150, treble_gain: 5 } },
      { op: 'compressor', params: { threshold: -36, ratio: 10 } },
    ],
  },
  {
    id: 'slapback',
    label: 'Slapback echo',
    description: 'One short, quiet repeat — the 1950s rockabilly vocal.',
    steps: [{ op: 'echo', params: { delay: 0.12, feedback: 15, mix: 40 } }],
  },
  {
    id: 'cathedral',
    label: 'Cathedral',
    description: 'A huge room with a six-second tail.',
    steps: [{ op: 'reverb', params: { room_size: 100, decay: 6, mix: 65 } }],
  },
  {
    id: 'nightcore',
    label: 'Nightcore',
    description: 'Played 30% faster like a tape, so the pitch rises with it.',
    steps: [{ op: 'speed_pitch', params: { speed: 1.3, preserve_pitch: false, semitones: 0 } }],
  },
  {
    id: 'chipmunk',
    label: 'Chipmunk',
    description: 'Voice up a fifth with the length unchanged.',
    steps: [{ op: 'voice_changer', params: { mode: 'chipmunk', semitones: 7 } }],
  },
  {
    id: 'monster',
    label: 'Monster',
    description: 'Voice down seven semitones with a dark room around it.',
    steps: [
      { op: 'voice_changer', params: { mode: 'monster', semitones: 7 } },
      { op: 'reverb', params: { room_size: 60, decay: 1.5, mix: 30 } },
    ],
  },
  {
    id: 'robot',
    label: 'Robot',
    description: 'Every syllable on one 100 Hz buzz, with a little room around it.',
    steps: [
      { op: 'voice_changer', params: { mode: 'robot', robot_freq: 100 } },
      { op: 'reverb', params: { room_size: 30, decay: 0.8, mix: 20 } },
    ],
  },
  {
    id: 'whisper',
    label: 'Whisper',
    description: 'Keep the words, lose the pitch: the voice becomes breath.',
    steps: [{ op: 'voice_changer', params: { mode: 'whisper' } }],
  },
]
