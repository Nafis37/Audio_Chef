# Audio Chef

**CyberChef, but for audio.** Drag effects from a palette into a recipe, and Audio Chef
bakes your recording through them in order. You can compare the result with the original
by ear (A/B switch) and by eye (waveform, spectrogram and before/after measurements).

The DSP is written by hand in NumPy: STFT, phase vocoder, biquad EQ, spectral subtraction,
convolution reverb, synthesis and more. Each module's docstring explains the maths behind
it in plain words.

## Features

- **Recipe-based processing.** Each loaded file gets its own tab and its own chain of
  effect cards. You can reorder, tweak or bypass any card. With Auto-Bake on, the output
  re-renders whenever something changes.
- **Effect palette**

  | Category     | Tools                                                     |
  |--------------|-----------------------------------------------------------|
  | Clean up     | Noise Remover (spectral subtraction), Silence Remover     |
  | Tone & level | Volume, Equalizer, Filter, Compressor, Voice Leveler      |
  | Space        | Echo, Reverb (impulse-response convolution)               |
  | Time & pitch | Speed & Pitch (phase vocoder + resampling)                |
  | Voice        | Voice changer (chipmunk, monster, robot, whisper, child, old person, gender shift) |
  | Music        | Background Music (chords, bass and drums that duck under your voice) |
  | Edit         | Cut & Trim, Fade, Reverse, Assemble Clip                  |

- **Signal Doctor.** Checks a file for clipping, uneven volume, hum, background noise,
  rumble and long silences, then suggests the recipe cards that fix each problem.
- **Arrange.** A multitrack timeline with clips, per-track volume, pan, mute and solo, and
  automation. It mixes other tabs' processed audio down to stereo.
- **Multi-source projects.** An `Assemble Clip` card can pull in another tab's processed
  output. The backend resolves the project as a DAG (`backend/app/dsp/graph.py`).
- **Recorder.** Record straight from the microphone in the browser.
- **Visual feedback.** Wavesurfer waveforms, spectrograms from the backend's own STFT,
  filter response curves, and peak, RMS, crest factor and similar stats for input vs output.

## Tech stack

| Part     | Stack                                                                 |
|----------|-----------------------------------------------------------------------|
| Backend  | Python, FastAPI, NumPy, SciPy (filters only), soundfile               |
| Frontend | React 19, TypeScript, Vite, Tailwind CSS, wavesurfer.js, @hello-pangea/dnd |

## Project structure

```
Audio_Chef/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app, CORS, router wiring
│   │   ├── config.py        # Storage path, limits, allowed origins
│   │   ├── audio_io.py      # Reading and writing audio files
│   │   ├── routers/         # upload, process, spectrogram, doctor
│   │   └── dsp/             # every effect and analysis, one module each
│   ├── tests/               # pytest suite
│   ├── requirements.txt
│   └── run.py               # starts uvicorn on 127.0.0.1:8000
└── frontend/
    ├── src/
    │   ├── App.tsx          # global state and the three-column layout
    │   ├── api.ts           # backend client
    │   └── components/      # palette, recipe, waveform, spectrogram, timeline, ...
    └── vite.config.ts       # proxies /api -> http://127.0.0.1:8000
```

## Getting started

### Prerequisites

- Python 3.10+
- Node.js 20.19+ (or 22.12+) and npm

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

The API runs at <http://127.0.0.1:8000>. Interactive docs are at
<http://127.0.0.1:8000/docs>.

### 2. Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The Vite dev server forwards `/api/*` to the backend, so
both have to be running.

## Configuration

The backend reads these optional environment variables:

| Variable                      | Default                                          | Purpose                          |
|-------------------------------|--------------------------------------------------|----------------------------------|
| `AUDIO_CHEF_STORAGE`          | `<system temp>/audio-chef`                       | Where uploaded files are stored  |
| `AUDIO_CHEF_ALLOWED_ORIGINS`  | `http://localhost:5173,http://127.0.0.1:5173`    | Comma-separated CORS origins     |
| `AUDIO_CHEF_MAX_UPLOAD_BYTES` | `104857600` (100 MB)                             | Upload size limit                |

## API overview

| Method | Endpoint                          | Description                                              |
|--------|-----------------------------------|----------------------------------------------------------|
| GET    | `/health`                         | Backend status                                           |
| POST   | `/upload`                         | Upload an audio file and get a `file_id` back            |
| GET    | `/operations`                     | Tool catalogue with parameter schemas (the UI is built from this) |
| POST   | `/process`                        | Bake a whole project and return a WAV                    |
| GET    | `/filter/response`                | Frequency response curve for a filter card               |
| GET    | `/spectrogram/file/{file_id}`     | Spectrogram of an uploaded file                          |
| GET    | `/spectrogram/bake/{bake_id}`     | Spectrogram of a baked output                            |
| GET    | `/diagnose/file/{file_id}`        | Signal Doctor report for an uploaded file                |
| GET    | `/diagnose/bake/{bake_id}`        | Signal Doctor report for a baked output                  |

Adding an effect only needs a backend change: add an entry to `OPERATIONS` in
`backend/app/dsp/dsp_engine.py`. The frontend renders its palette entry and sliders from
`GET /operations`.

## Development

Backend tests:

```bash
cd backend
pip install -r requirements-dev.txt
pytest
```

Frontend lint and production build:

```bash
cd frontend
npm run lint
npm run build
```
