/**
 * Dual-engine pitch detection Web Worker.
 *
 * Supports two backends selectable via message.engine:
 *   'AMDF'   — lightweight time-domain detection (default)
 *   'NEURAL' — CREPE-Tiny ONNX inference (1024-sample @ 16kHz, GPU-accelerated)
 *
 * Message format (received):
 *   { audioBuffer: Float32Array, sampleRate: number, engine?: 'AMDF'|'NEURAL', useBandpass?: boolean }
 *
 * Message format (posted):
 *   { frequency: number, rawFrequency: number, timestamp: number, engine: string }
 */

import { detectPitchAMDF, applyThreePointSmoothing } from './pitchMath.js';

// ── CREPE frequency mapping constants ────────────────────────────
// CREPE outputs 360 probability bins from C1 (32.7 Hz) to B7 (1975.5 Hz),
// spaced at ~10 cents per bin (logarithmic).
const CREPE_FMIN = 32.7;
const CREPE_FMAX = 1975.5;
const CREPE_BINS = 360;
const CREPE_CENTS_PER_BIN = 1200 * Math.log2(CREPE_FMAX / CREPE_FMIN) / (CREPE_BINS - 1);

// Precompute Hz for each bin index [0..359]
const CREPE_BIN_HZ = new Float32Array(CREPE_BINS);
for (let i = 0; i < CREPE_BINS; i++) {
  CREPE_BIN_HZ[i] = CREPE_FMIN * Math.pow(2, (i * CREPE_CENTS_PER_BIN) / 1200);
}

// CREPE expects exactly 1024 samples at 16 kHz
const CREPE_INPUT_SIZE = 1024;
const CREPE_SAMPLE_RATE = 16000;

/**
 * Decode CREPE sigmoid output to Hz using weighted argmax.
 *
 * Matches torchcrepe's weighted_argmax decoder:
 *   1. Find argmax bin → window ±4 bins around it
 *   2. Sigmoid already applied by model → values in [0, 1]
 *   3. Weighted average by CENTS (log frequency), not Hz
 *
 * @param {Float32Array} probabilities — 360-element sigmoid probability array [0, 1]
 * @returns {number} frequency in Hz, or 0 if peak confidence < threshold
 */
function decodeCrepeFrequency(probabilities) {
  // Find argmax bin
  let maxIdx = 0, maxVal = probabilities[0];
  for (let i = 1; i < probabilities.length; i++) {
    if (probabilities[i] > maxVal) { maxIdx = i; maxVal = probabilities[i]; }
  }

  // Confidence gate: sigmoid near 0.5 = model uncertainty
  if (maxVal < 0.6) return 0;

  // Window ±4 bins around argmax
  const lo = Math.max(0, maxIdx - 4);
  const hi = Math.min(probabilities.length - 1, maxIdx + 4);

  // Weighted average by cents (log frequency)
  const fminCents = 1200 * Math.log2(CREPE_FMIN);
  let wSum = 0, wTot = 0;
  for (let i = lo; i <= hi; i++) {
    const cents = fminCents + i * CREPE_CENTS_PER_BIN;
    wSum += cents * probabilities[i];
    wTot += probabilities[i];
  }

  if (wTot <= 0) return 0;

  // Convert weighted mean cents back to Hz: 10 * 2^(cents/1200)
  const meanCents = wSum / wTot;
  return 10 * Math.pow(2, meanCents / 1200);
}

// ── Resampling (linear interpolation) ─────────────────────────────

/**
 * Linear resample a Float32Array to a target sample rate.
 * @param {Float32Array} buffer — input samples at sourceRate
 * @param {number} sourceRate — original sample rate (e.g. 44100)
 * @param {number} targetRate — target sample rate (e.g. 16000)
 * @returns {Float32Array} resampled buffer
 */
function resampleBuffer(buffer, sourceRate, targetRate) {
  const ratio = targetRate / sourceRate;
  const outLen = Math.floor(buffer.length * ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcIdx = i / ratio;
    const lo = Math.floor(srcIdx);
    const hi = Math.min(lo + 1, buffer.length - 1);
    const frac = srcIdx - lo;
    out[i] = buffer[lo] + (buffer[hi] - buffer[lo]) * frac;
  }
  return out;
}

// ── Butterworth bandpass filter (60–400 Hz, zero-phase) ───────────

/**
 * 2nd-order Butterworth bandpass (biquad, forward-then-backward).
 * Passband 60–400 Hz covers the full Mandarin vocal range.
 * Strips sub-bass rumble and high-frequency noise before CREPE inference.
 *
 * @param {Float32Array} buffer — input samples at sourceRate
 * @param {number} sampleRate — original sample rate
 * @returns {Float32Array} filtered buffer (same length)
 */
function applyBandpassFilter(buffer, sampleRate) {
  const lo = 60, hi = 400;
  const wLo = 2 * Math.PI * lo / sampleRate;
  const wHi = 2 * Math.PI * hi / sampleRate;
  const bw = wHi - wLo;
  const w0 = Math.sqrt(wLo * wHi);
  const alpha = Math.sin(bw / 2);
  const cosw0 = Math.cos(w0);

  // Biquad coefficients (RBJ cookbook, peak gain = 1)
  const b0 = alpha, b1 = 0, b2 = -alpha;
  const a0 = 1 + alpha, a1 = -2 * cosw0, a2 = 1 - alpha;
  const bn = [b0 / a0, b1 / a0, b2 / a0];
  const an = [a1 / a0, a2 / a0];

  // Forward pass
  const fwd = new Float32Array(buffer.length);
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < buffer.length; i++) {
    const y = bn[0] * buffer[i] + bn[1] * x1 + bn[2] * x2 - an[0] * y1 - an[1] * y2;
    fwd[i] = y;
    x2 = x1; x1 = buffer[i];
    y2 = y1; y1 = y;
  }

  // Backward pass (zero-phase)
  const out = new Float32Array(buffer.length);
  x1 = 0; x2 = 0; y1 = 0; y2 = 0;
  for (let i = buffer.length - 1; i >= 0; i--) {
    const y = bn[0] * fwd[i] + bn[1] * x1 + bn[2] * x2 - an[0] * y1 - an[1] * y2;
    out[i] = y;
    x2 = x1; x1 = fwd[i];
    y2 = y1; y1 = y;
  }
  return out;
}

// ── Neural engine state ───────────────────────────────────────────
let ortSession = null;
let neuralReady = false;
let neuralLoading = false;

/**
 * Initialize the ONNX Runtime inference session.
 * Loads the CREPE-Tiny ONNX model from the public assets folder.
 * Uses WebGL execution provider for GPU acceleration where available.
 */
async function initNeuralEngine() {
  if (ortSession) return;
  if (neuralLoading) {
    // Wait for concurrent init to complete
    while (neuralLoading) {
      await new Promise(r => setTimeout(r, 50));
    }
    return;
  }

  neuralLoading = true;
  try {
    // Dynamic import — only loads onnxruntime-web when NEURAL engine is used
    const { InferenceSession, Tensor } = await import('onnxruntime-web');

    ortSession = await InferenceSession.create(
      '/models/crepe_tiny.onnx',
      {
        executionProviders: ['webgl', 'wasm'],
        graphOptimizationLevel: 'all',
      }
    );

    neuralReady = true;
    console.log('[pitchWorker] Neural engine ready (CREPE-Tiny ONNX)');
  } catch (err) {
    console.error('[pitchWorker] Neural engine init failed:', err.message);
    neuralReady = false;
  } finally {
    neuralLoading = false;
  }
}

/**
 * Run CREPE-Tiny inference on an audio frame.
 *
 * Pipeline: optional bandpass → resample to 16kHz → pad/trim to 1024 → ONNX → decode Hz
 *
 * @param {Float32Array} audioBuffer — audio frame at the source sample rate
 * @param {number} sampleRate — original sample rate (e.g. 44100)
 * @param {boolean} [useBandpass=false] — apply 60–400 Hz bandpass before resampling
 * @returns {Promise<number>} detected frequency in Hz, or 0
 */
async function detectPitchNeural(audioBuffer, sampleRate, useBandpass = false) {
  if (!neuralReady || !ortSession) return 0;

  try {
    // Step 1: optional bandpass filter (at source sample rate)
    let filtered = audioBuffer;
    if (useBandpass) {
      filtered = applyBandpassFilter(audioBuffer, sampleRate);
    }

    // Step 2: resample to CREPE's native 16 kHz
    const resampled = resampleBuffer(filtered, sampleRate, CREPE_SAMPLE_RATE);

    // Step 3: pad or trim to exactly 1024 samples
    let frame16k;
    if (resampled.length >= CREPE_INPUT_SIZE) {
      // Take the first 1024 samples (streaming — most recent audio)
      frame16k = resampled.slice(0, CREPE_INPUT_SIZE);
    } else {
      // Zero-pad to 1024 (silence at end is fine for CREPE)
      frame16k = new Float32Array(CREPE_INPUT_SIZE);
      frame16k.set(resampled);
    }

    // Step 4: ONNX inference
    const { Tensor } = await import('onnxruntime-web');
    const audioTensor = new Tensor('float32', frame16k, [1, CREPE_INPUT_SIZE]);
    const results = await ortSession.run({ input_audio: audioTensor });

    // Step 5: decode 360-bin probability vector to Hz
    const probabilities = results.pitch_probabilities.data;
    return decodeCrepeFrequency(probabilities);
  } catch (err) {
    console.error('[pitchWorker] Neural inference error:', err.message);
    return 0;
  }
}

// ── AMDF pipeline (existing) ──────────────────────────────────────

/**
 * Process a single audio frame through AMDF detection + 5-point median smoothing.
 * Exported for unit testing.
 */
export function processAudioFrame(audioBuffer, sampleRate, pitchHistory) {
  const rawFrequency = detectPitchAMDF(audioBuffer, sampleRate);

  // Maintain rolling history (max 5 entries)
  pitchHistory.push(rawFrequency);
  if (pitchHistory.length > 5) {
    pitchHistory.shift();
  }

  // Apply 5-point median smoothing when we have enough history
  let frequency = rawFrequency;
  if (pitchHistory.length >= 3) {
    const smoothed = applyThreePointSmoothing(pitchHistory);
    frequency = smoothed[smoothed.length - 1];
  }

  return { frequency, rawFrequency, pitchHistory };
}

// ── Worker message handler ────────────────────────────────────────

if (typeof self !== 'undefined' && typeof self.onmessage !== 'undefined') {
  let pitchHistory = [];
  let currentEngine = 'AMDF';
  let useBandpass = false;

  self.onmessage = async function (e) {
    const { audioBuffer, sampleRate, engine, useBandpass: bandpassFlag } = e.data;

    // Handle engine switch
    if (engine && engine !== currentEngine) {
      currentEngine = engine;
      if (engine === 'NEURAL') {
        await initNeuralEngine();
      }
    }

    // Update bandpass preference
    if (typeof bandpassFlag === 'boolean') {
      useBandpass = bandpassFlag;
    }

    let frequency, rawFrequency;

    if (currentEngine === 'NEURAL' && neuralReady) {
      rawFrequency = await detectPitchNeural(audioBuffer, sampleRate, useBandpass);
      frequency = rawFrequency; // CREPE output is already clean
    } else {
      // Default: AMDF pipeline
      const result = processAudioFrame(audioBuffer, sampleRate, pitchHistory);
      frequency = result.frequency;
      rawFrequency = result.rawFrequency;
    }

    self.postMessage({
      frequency,
      rawFrequency,
      timestamp: Date.now(),
      engine: currentEngine,
    });
  };
}
