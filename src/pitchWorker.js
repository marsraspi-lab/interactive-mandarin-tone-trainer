/**
 * Dual-engine pitch detection Web Worker.
 *
 * Supports two backends selectable via message.engine:
 *   'AMDF'   — lightweight time-domain detection (default)
 *   'NEURAL' — CREPE-Tiny ONNX inference (~3MB model, GPU-accelerated)
 *
 * Message format (received):
 *   { audioBuffer: Float32Array, sampleRate: number, engine?: 'AMDF'|'NEURAL' }
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

/**
 * Convert CREPE probability output to Hz using weighted top-K averaging.
 * Averages the top 3 bins weighted by their confidence, giving sub-bin
 * precision for smoother pitch contours — critical for tone tracking.
 *
 * @param {Float32Array} probabilities — 360-element probability array
 * @param {number} [topK=3] — number of peak bins to average
 * @returns {number} frequency in Hz, or 0 if max confidence < threshold
 */
function decodeCrepeFrequency(probabilities, topK = 3) {
  // Find top-K bins by probability
  const indexed = [];
  for (let i = 0; i < probabilities.length; i++) {
    indexed.push({ idx: i, prob: probabilities[i] });
  }
  indexed.sort((a, b) => b.prob - a.prob);

  // Confidence gate: if the peak probability is too low, return 0
  if (indexed[0].prob < 0.3) return 0;

  // Weighted average of top K bins
  let weightedSum = 0;
  let weightTotal = 0;
  const k = Math.min(topK, indexed.length);

  for (let j = 0; j < k; j++) {
    if (indexed[j].prob <= 0) break;
    weightedSum += CREPE_BIN_HZ[indexed[j].idx] * indexed[j].prob;
    weightTotal += indexed[j].prob;
  }

  return weightedSum / weightTotal;
}

// ── Neural engine state ───────────────────────────────────────────
let ortSession = null;
let neuralReady = false;
let neuralLoading = false;

/**
 * Initialize the ONNX Runtime inference session.
 * Loads the quantized CREPE-Tiny model from the public assets folder.
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
      '/models/crepe_tiny_quantized.onnx',
      {
        executionProviders: ['webgl', 'wasm'],
        graphOptimizationLevel: 'all',
      }
    );

    neuralReady = true;
    console.log('[pitchWorker] Neural engine ready (CREPE-Tiny, quantized)');
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
 * @param {Float32Array} audioBuffer — 4096-sample audio frame
 * @param {number} sampleRate — samples per second (unused, CREPE expects 16kHz internally)
 * @returns {Promise<number>} detected frequency in Hz, or 0
 */
async function detectPitchNeural(audioBuffer) {
  if (!neuralReady || !ortSession) return 0;

  try {
    const { Tensor } = await import('onnxruntime-web');

    // Wrap audio as ONNX tensor [batch=1, samples=4096]
    const audioTensor = new Tensor('float32', audioBuffer, [1, audioBuffer.length]);

    // Run inference
    const results = await ortSession.run({ input_audio: audioTensor });

    // Extract probability vector
    const probabilities = results.pitch_probabilities.data;

    // Decode to Hz with weighted top-K averaging
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

  self.onmessage = async function (e) {
    const { audioBuffer, sampleRate, engine } = e.data;

    // Handle engine switch
    if (engine && engine !== currentEngine) {
      currentEngine = engine;
      if (engine === 'NEURAL') {
        await initNeuralEngine();
      }
    }

    let frequency, rawFrequency;

    if (currentEngine === 'NEURAL' && neuralReady) {
      rawFrequency = await detectPitchNeural(audioBuffer);
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
