# 🎤 Mandarin Tone Trainer

A browser-based tool for practicing Mandarin Chinese tones. Record your voice, see your pitch contour visualized in real time, and get graded against native-speaker reference models using AMDF pitch detection and dynamic time warping (DTW).

## How it works

1. **Select a word** from the dropdown — single syllables or tone pairs
2. **Click Record** and speak the word aloud
3. **Stop recording** — your pitch contour is extracted, Z-score normalized, and compared against the native reference via DTW
4. **Get a score** (0–100%) and diagnostic feedback ("Pitch Dropped", "Not Deep Enough", "Too Soft/Slow")
5. **Click Play** to hear the native speaker reference

The pitch visualization uses a real-time scrolling canvas — watch your voice move as you speak.

## Tech stack

| Layer | Technology |
|---|---|
| Pitch detection | AMDF (Average Magnitude Difference Function) |
| Smoothing | 3-point moving average filter |
| Normalization | Z-score (μ=0, σ=1) + [-3, +3] clamping |
| Comparison | Dynamic Time Warping → MAE (Mean Absolute Error) |
| Audio pipeline | Web Audio API (AnalyserNode → Float32Array → Web Worker) |
| Visualization | HTML Canvas (800×400) |
| Build | Vite |
| Tests | Vitest (unit) + Playwright (E2E) |
| Quality | Fallow (dead code, complexity, duplication) |

## Project structure

```
├── src/
│   ├── index.html          # Main app (user-facing)
│   ├── admin.html          # Admin dashboard (native speaker recording)
│   ├── app.js              # Main app: mic capture, Canvas, grading
│   ├── admin.js            # Admin: 24-word recording corpus, export
│   ├── pitchMath.js        # DSP: AMDF, smoothing, Z-score, DTW, scoring
│   ├── pitchWorker.js      # Web Worker wrapping pitchMath
│   └── presets.json        # Native reference pitch data (auto-generated)
├── scripts/
│   └── ingestPresets.js    # Build-time WAV → presets.json pipeline
├── tests/
│   ├── fixtures/
│   │   └── native_samples/ # Native speaker WAV recordings (7 files)
│   ├── helpers.js          # Shared test utilities
│   ├── pitchMath.test.js   # 27 DSP unit tests
│   ├── pitchWorker.test.js # 5 worker tests
│   ├── ingestPresets.test.js # 2 integration tests
│   └── toneTrainer.spec.js # 4 E2E tests (Playwright)
├── public/                 # Vite publicDir (presets.json copied here)
├── vite.config.js
└── package.json
```

## Getting started

```bash
# Install dependencies
npm install

# Run tests (34 unit + integration)
npm test

# Run E2E tests
npm run e2e

# Build for production (ingests WAVs → presets.json → bundles via Vite)
npm run build

# Dev server (serves src/ directory)
python3 -m http.server 3000 --directory src
# Then open http://localhost:3000
```

## Build pipeline

`npm run build` runs two steps in sequence:

1. **`npm run ingest`** — Reads native speaker WAV files from `tests/fixtures/native_samples/`, extracts pitch contours through the same DSP pipeline as the client (AMDF → 3-point smoothing → Z-score → clamp → resample to 100 points), and writes `presets.json` to the project root, `src/`, and `public/`.

2. **`vite build`** — Bundles the app into `dist/`, copying `public/presets.json` into the output.

## Admin dashboard

The admin page (`src/admin.html`) provides a recording interface for populating the native reference database:

- **24-word corpus** covering all Mandarin tones (single syllables + tone pairs)
- **Sidebar** with red/green recording status indicators
- **Live pitch visualization** on a mini-canvas during recording
- **Keyboard shortcuts**: ← → to navigate, Space to record
- **Auto-download** of each recording as `.mp3`
- **Export** button generates `presets.json` for production use

## How the grading works

1. **Pitch extraction** — AMDF detects fundamental frequency (60–400 Hz bandpass) from your microphone input, processed in a Web Worker
2. **Smoothing** — 3-point moving average filter removes plosive artifacts
3. **Z-score normalization** — Centers your pitch contour at μ=0, σ=1, then clamps to [-3, +3]
4. **DTW alignment** — Both your contour and the native reference are resampled to exactly 100 points via linear interpolation
5. **MAE scoring** — Mean Absolute Error computed, mapped to 0–100% (MAE of 2σ = 0%)
6. **Diagnostics** — Shape analysis detects specific tone errors (rising when should fall, flat when should dip, gradual when should drop sharply)

## Adding new presets

1. Record a native speaker WAV file and place it in `tests/fixtures/native_samples/`
2. Add an entry to the `PRESET_DEFS` array in `scripts/ingestPresets.js`
3. Run `npm run ingest` to regenerate `presets.json`
4. Place the corresponding `.mp3` delivery file in `src/assets/audio/`

## CI

PRs are gated on:
- **Fallow audit** — dead code, duplication, complexity (hard gate)
- **Vitest** — 34 unit + integration tests (hard gate)

Pre-commit hooks run both checks in advisory mode.
