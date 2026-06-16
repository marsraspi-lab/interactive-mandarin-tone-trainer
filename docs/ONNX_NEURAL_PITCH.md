# ONNX Neural Pitch Detection — Dual-Engine Architecture

**Branch:** `feat/onnx-neural-pitch`  
**Status:** Built, ready for A/B comparison  
**Strategy:** Export CREPE-Tiny to ONNX (auto-quantize only if fp32 > 10 MB)  

---

## Summary

The current AMDF pitch detector has a hard ceiling at **MAE 0.444** because difference-function detectors cannot track pitch through unvoiced fricative regions the way pYIN's HMM does. This feature adds a **second engine**: a CREPE-Tiny neural model that runs in the browser via WebGL/WebAssembly.

CREPE is a convolutional neural network trained on millions of voiced+unvoiced frames. Its filters learn to "see through" friction noise by recognizing harmonic structure — precisely the capability AMDF lacks for fricative-heavy words (shuiguo, xiexie, haizi, jiejie).

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      app.js                             │
│  currentEngine: 'AMDF' | 'NEURAL'                       │
│  Toggle in sidebar → switch backend at runtime          │
└────────────────┬────────────────────────────────────────┘
                 │  postMessage({ audioBuffer, engine })
                 ▼
┌─────────────────────────────────────────────────────────┐
│                   pitchWorker.js                        │
│                                                         │
│  if engine === 'NEURAL':                                │
│    ┌──────────────────────────────────────────┐         │
│    │  InferenceSession (onnxruntime-web)      │         │
│    │  ├─ WebGL backend (GPU accelerated)      │         │
│    │  ├─ Falls back to WASM (CPU)             │         │
│    │  └─ Model: crepe_tiny.onnx               │         │
│    │     (~5-7 MB fp32, or ~1.5 MB if quant)  │         │
│    └──────────────────────────────────────────┘         │
│  else:                                                  │
│    ┌──────────────────────────────────────────┐         │
│    │  detectPitchAMDF (pure JS, zero deps)    │         │
│    │  → 5-point median smoothing              │         │
│    └──────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────┘
```

### Zero overhead for AMDF users

The 404KB `onnxruntime-web` bundle (111KB gzipped) is **never loaded** unless the user switches to NEURAL mode. The AMDF path stays completely unchanged — no imports, no code path divergence, no performance impact.

### CREPE Hz decoding

CREPE outputs 360 probability bins spaced at ~10 cents (C1=32.7 Hz to B7=1975.5 Hz). Our decoder uses **weighted top-3 averaging** for sub-bin precision — the top three bins are averaged by their confidence scores, producing smoother pitch contours essential for tracking Mandarin tones where a few Hz difference matters.

---

## Repository Files

| File | Role |
|------|------|
| `scripts/convert_crepe_onnx.py` | PyTorch → fp32 ONNX. Auto-quantizes only if > 10 MB. Run once. |
| `src/assets/models/crepe_tiny.onnx` | The output model. **Git-ignored** (~5-7 MB). |
| `src/pitchWorker.js` | Dual-engine Web Worker. Lazy-loads ONNX runtime. |
| `src/app.js` | Engine toggle in sidebar (`currentEngine`). |
| `scripts/train.js` | `--neural` flag runs CREPE through the evaluation pipeline. |
| `package.json` | `onnxruntime-web` in dependencies. |

---

## How to Use

### Phase 1: Generate the ONNX model (Python)

You only do this **once**. The `.onnx` file is not committed to git.

#### 1a. Set up a Python virtual environment

```bash
cd /path/to/interactive-mandarin-tone-trainer

# Create the venv
python3 -m venv .venv

# Activate it
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows (PowerShell)

# Upgrade pip
pip install --upgrade pip
```

#### 1b. Install the Python dependencies

```bash
pip install torch onnx onnxruntime
```

**Package breakdown:**

| Package | Purpose |
|---------|---------|
| `torch` | Loads the PyTorch CREPE model and its pretrained weights |
| `onnx` | Exports the model from PyTorch into the ONNX format |
| `onnxruntime` | Validates the model and handles quantization if needed |

All three are pure-CPU installs. No GPU or CUDA required.

#### 1c. Run the conversion script

```bash
python3 scripts/convert_crepe_onnx.py
```

**What happens during the run:**

1. **Export Phase** — Downloads the CREPE-Tiny pretrained weights from torchcrepe (the `tiny.pth` is only 1.87 MB — already stored as fp16). Wraps the model, feeds a dummy 4096-sample frame through it, and writes `crepe_tiny_fp32.onnx` (~5-7 MB) to `/tmp/`.

2. **Conditional Quantization** — The script checks the fp32 size. If it's ≤ 10 MB (which it should be — torchcrepe's tiny model is compact), the fp32 file is copied directly to the output path with **no quantization**. If it somehow exceeds 10 MB, dynamic uint8 quantization is applied to `Conv`/`Gemm`/`MatMul` layers. The output lands at `src/assets/models/crepe_tiny.onnx`.

3. **Validation Phase** — Runs a single inference through the final model with random noise, verifies the output shape and range are sane.

**Expected output (no quantization needed):**

```
╔══════════════════════════════════════════╗
║   CREPE-Tiny → ONNX Export              ║
╚══════════════════════════════════════════╝
   Target: .../src/assets/models/crepe_tiny.onnx
   Input size: 4096 samples

📦 Loading CREPE-Tiny model...
   Using torchcrepe package
   Exporting to ONNX (input shape: [1, 4096])...
   ✅ FP32 ONNX exported: 5.8 MB

   fp32 ONNX size: 5.8 MB
   ✅ Below 10 MB threshold — shipping fp32 as-is
   📁 Final model: 5.8 MB — .../src/assets/models/crepe_tiny.onnx
🔍 Validating model with test inference...
   Input shape:  (1, 4096)
   Output shape: (1, 360)
   Output range: [0.0000, 0.0142]
   Peak bin:     187 (confidence: 0.0142)
   ✅ Validation passed

✅ Done!
```

#### 1d. Deactivate the venv when done

```bash
deactivate
```

The `.venv/` directory can be safely deleted after the model is generated — it's only needed for the conversion step.

---

### Phase 2: Run the A/B comparison (JavaScript)

```bash
# AMDF baseline (what the app uses today)
node scripts/train.js

# Neural comparison
node scripts/train.js --neural --voiced-only
```

**Flags:**

| Flag | Effect |
|------|--------|
| `--neural` | Switches evaluation engine to CREPE-Tiny ONNX |
| `--voiced-only` | Computes MAE only on frames where **both** JS and pYIN detect voice (ignores silence/fricative gaps) — isolates pure pitch-tracking accuracy |

**What to compare:**

1. **Aggregate MAE:** Does CREPE drop below the 0.44 AMDF ceiling?
2. **Fricative words:** shuiguo, xiexie, haizi, jiejie — these are where CREPE's CNN should dominate
3. **Voiced-only MAE (`--voiced-only`):** When both detectors agree a frame is voiced, how close is CREPE to pYIN? This isolates pure pitch accuracy from voiced/unvoiced decision errors.

---

### Phase 3: Use in the browser app

1. Make sure `crepe_tiny.onnx` is in `src/assets/models/` (from Phase 1)
2. Start the dev server: `npm run dev`
3. In the sidebar, click the engine row: **Engine: AMDF (click to switch)**
4. It toggles to **NEURAL**. The first time you record after switching, the browser downloads the model (~5-7 MB) and initializes WebGL inference. Subsequent recordings use the cached session.

**Browser requirements:**

- WebGL 2.0 or WebGPU for GPU acceleration
- ~50MB RAM for the ONNX runtime
- First load latency: ~2-5 seconds (model download + warmup)
- Per-frame inference: ~5-15ms (WebGL) or ~20-50ms (WASM fallback)

---

## How It Works: The Full Pipeline

### AMDF path (default)

```
Audio frame (4096 samples, 44.1kHz)
  → AMDF difference-function (60–400 Hz)
  → ZCR fricative gate (0.15)
  → Silence gate (RMS < 0.002)
  → Periodicity gate (0.4×RMS)
  → 5-frame median smoothing
→ Raw Hz value
```

### NEURAL path

```
Audio frame (4096 samples, 44.1kHz)
  → Resample to 16kHz (linear interpolation)
  → ONNX inference (CREPE-Tiny)
  → 360-bin probability vector
  → Weighted top-3 Hz decoding
  → Confidence gate (max prob < 0.3 → return 0)
→ Raw Hz value
```

### Shared post-processing (both engines)

```
Raw pitch curve (Hz)
  → Octave correction (anti-jump filter)
  → Spline interpolation (fill ≤8-frame gaps)
  → 5-frame median smoothing
  → Shared Z-score normalization (GT μ/σ)
  → Clamp [-3, +3]
  → Resample to 100 points
→ Final curve for DTW → MAE → Score
```

---

## Technical Notes

### Why 4096 samples?

93ms at 44.1kHz gives ~9.3 pitch periods at 100 Hz (typical male fundamental). At 2048 samples (~4.6 periods), low male voices can confuse the AMDF lag search — a period multiple aligns better with the buffer than the true period, causing octave errors. 4096 eliminates this. CREPE-Tiny natively accepts variable-length input, so the larger window is just more temporal context for the CNN.

### Model size

Torchcrepe's `tiny.pth` is already **fp16 at 1.87 MB**. When PyTorch exports to ONNX, weights expand to fp32 (~3.7 MB) and the ONNX container adds graph metadata (~1-2 MB), so the final file lands around **5-7 MB**. The conversion script automatically detects the actual size — if it's ≤ 10 MB (which it will be for this model), no quantization is applied. fp32 precision is preserved, and the 5-7 MB download is already reasonable for a progressive enhancement.

### Why CREPE over YIN/other lightweight detectors?

| Detector | MAE (tested) | Approach | Key limitation |
|----------|-------------|----------|----------------|
| AMDF (4096) | 0.444 | Time-domain difference | Can't track through fricatives |
| YIN | 0.481–0.609 | Time-domain autocorrelation | No HMM continuity |
| Butterworth + AMDF | 0.513 | Bandpass filter | Strips energy needed for detection |
| CREPE-Tiny | **TBD** | ConvNet on spectrogram | Requires ONNX runtime |

### Node.js evaluation (`--neural`)

The CLI evaluation uses `onnxruntime-node` (the Node.js binding) rather than `onnxruntime-web`. Both load the same `.onnx` file. The CLI path is synchronous-friendly and runs on CPU; the browser path uses WebGL for GPU acceleration.

---

## Roadmap

- [ ] Run A/B comparison: `node scripts/train.js --neural`
- [ ] If MAE drops below 0.30, keep the branch and merge
- [ ] If MAE is similar or worse, investigate:
  - Model calibration on Mandarin speech (the original CREPE was trained on multi-instrument datasets)
  - Different confidence thresholds
  - CREPE "full" model (32MB, higher accuracy)
  - Per-frame voiced/unvoiced decisions vs. argmax
- [ ] Measure time-to-first-note latency in real browser sessions
- [ ] Consider Worker-thread preloading of the ONNX model on app startup

---

## Deleting the Python environment

Once `crepe_tiny.onnx` is generated and working:

```bash
rm -rf .venv/
```

The model file is the only artifact needed at runtime. The Python conversion is a one-shot build step, not a dependency of the app.
