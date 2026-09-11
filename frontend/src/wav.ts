/**
 * AudioBuffer -> 16-bit PCM WAV, written by hand.
 *
 * The browser records into WebM/Opus (Chrome, Firefox) or MP4/AAC (Safari); the backend
 * decodes with libsndfile, which reads neither.  So a take has to be transcoded here.
 * decodeAudioData() does the hard half -- the browser already owns the Opus/AAC decoder --
 * and this file does the easy half: wrap raw samples in the 44-byte canonical header.
 *
 * The layout (RIFF little-endian throughout, byte offsets on the left):
 *
 *    0  "RIFF"                     4  chunk size = 36 + data bytes
 *    8  "WAVE"
 *   12  "fmt "                    16  subchunk size = 16 for PCM
 *   20  audioFormat = 1 (PCM)     22  channels
 *   24  sampleRate                28  byteRate   = rate * channels * bytesPerSample
 *   32  blockAlign = channels * bytesPerSample   34  bitsPerSample = 16
 *   36  "data"                    40  data bytes
 *   44  the samples
 *
 * `36 +` at offset 4 is everything after that field: 4 ("WAVE") + 24 (fmt) + 8 (data
 * header).  Nothing here is a size the encoder guesses -- they all follow from the buffer.
 */

/** Header size in bytes.  PCM has no extension field, so this is fixed. */
const HEADER_BYTES = 44
const BYTES_PER_SAMPLE = 2

/**
 * Encodes `buffer` as a mono 16-bit PCM WAV.
 *
 * Mono is deliberate, not a simplification: `audio_io.load_audio` averages the channels the
 * moment the file lands, so uploading stereo would double the transfer for samples the
 * backend discards on arrival.  Folding here means the Input waveform the user sees is
 * also exactly what the DSP chain will run on.
 */
export function encodeWav(buffer: AudioBuffer): Blob {
  const mono = toMono(buffer)
  const bytes = new ArrayBuffer(HEADER_BYTES + mono.length * BYTES_PER_SAMPLE)
  const view = new DataView(bytes)

  writeAscii(view, 0, 'RIFF')
  view.setUint32(4, 36 + mono.length * BYTES_PER_SAMPLE, true)
  writeAscii(view, 8, 'WAVE')

  writeAscii(view, 12, 'fmt ')
  view.setUint32(16, 16, true)                      // PCM subchunk is 16 bytes
  view.setUint16(20, 1, true)                       // 1 = uncompressed PCM
  view.setUint16(22, 1, true)                       // channels: mono, see toMono
  view.setUint32(24, buffer.sampleRate, true)       // usually 48000, whatever the device ran at
  view.setUint32(28, buffer.sampleRate * BYTES_PER_SAMPLE, true)   // byteRate, channels = 1
  view.setUint16(32, BYTES_PER_SAMPLE, true)        // blockAlign, channels = 1
  view.setUint16(34, 8 * BYTES_PER_SAMPLE, true)    // bitsPerSample

  writeAscii(view, 36, 'data')
  view.setUint32(40, mono.length * BYTES_PER_SAMPLE, true)

  let offset = HEADER_BYTES
  for (let i = 0; i < mono.length; i += 1) {
    // Clamp first: decodeAudioData can hand back values slightly outside +-1, and the
    // int16 range is asymmetric (-32768..32767), so the two directions scale differently.
    // Using 0x7fff for both would leave the negative half a hair quiet; using 0x8000 for
    // both would wrap the loudest positive sample around to full-scale negative.
    const sample = Math.max(-1, Math.min(1, mono[i]))
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true)
    offset += BYTES_PER_SAMPLE
  }

  return new Blob([bytes], { type: 'audio/wav' })
}

/** Averages every channel into one.  Returns channel 0 untouched when already mono. */
function toMono(buffer: AudioBuffer): Float32Array {
  if (buffer.numberOfChannels === 1) return buffer.getChannelData(0)

  const channels: Float32Array[] = []
  for (let c = 0; c < buffer.numberOfChannels; c += 1) channels.push(buffer.getChannelData(c))

  const mono = new Float32Array(buffer.length)
  for (let i = 0; i < buffer.length; i += 1) {
    let sum = 0
    for (const channel of channels) sum += channel[i]
    mono[i] = sum / channels.length
  }
  return mono
}

/** WAV chunk tags are four ASCII bytes, written raw rather than through a TextEncoder. */
function writeAscii(view: DataView, offset: number, text: string) {
  for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i))
}
