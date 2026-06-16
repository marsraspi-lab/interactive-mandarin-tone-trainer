#!/usr/bin/env node
/**
 * AUTORESEARCH Training Loop — Evaluation Runner
 *
 * Loads pYIN ground truth from data/ground-truth.json, processes each audio
 * file through the CURRENT pitchMath.js pipeline, and computes aggregate
 * error metrics.
 *
 * Usage:  node scripts/train.js
 *
 * Output metrics:
 *   Path A — Native Reference MAE (vs pYIN ground truth, z-score space)
 *   Path B — Validation Generalization Gap (train vs holdout)
 *   Path C — Human Alignment (% directional agreement with rating.txt)
 *
 * All DSP is imported from src/pitchMath.js — zero duplication.
 */

import { readFileSync, existsSync, mkdirSync, writeFileSync } from 'fs';
import { resolve, join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { createHash } from 'crypto';

import {
  detectPitchAMDF,
  applyThreePointSmoothing,
  applyOctaveCorrection,
  applySplineInterpolation,
  normalizeZScore,
  normalizeWithSharedStats,
  clampValues,
  resampleArray,
  calculateMAEScore,
} from '../src/pitchMath.js';

// ── CREPE neural pitch detection (optional, --neural flag) ──────────

/**
 * CREPE-Tiny outputs 360 probability bins from C1 (32.7 Hz) to B7 (1975.5 Hz),
 * spaced at ~10 cents per bin. Precompute the Hz lookup table.
 */
const CREPE_FMIN = 32.7;
const CREPE_CENTS_PER_BIN = (1200 * Math.log2(1975.5 / 32.7)) / 359;
const CREPE_BIN_HZ = new Float32Array(360);
for (let i = 0; i < 360; i++) {
  CREPE_BIN_HZ[i] = CREPE_FMIN * Math.pow(2, (i * CREPE_CENTS_PER_BIN) / 1200);
}

/**
 * Decode CREPE probability vector to Hz using weighted top-3 averaging.
 * @param {Float32Array} probs — 360-element probability array
 * @returns {number} frequency in Hz, or 0 if peak confidence < 0.3
 */
function decodeCrepeHz(probs) {
  const indexed = [];
  for (let i = 0; i < probs.length; i++) indexed.push({ idx: i, p: probs[i] });
  indexed.sort((a, b) => b.p - a.p);
  if (indexed[0].p < 0.3) return 0;

  let wSum = 0, wTot = 0;
  for (let j = 0; j < 3 && indexed[j].p > 0; j++) {
    wSum += CREPE_BIN_HZ[indexed[j].idx] * indexed[j].p;
    wTot += indexed[j].p;
  }
  return wSum / wTot;
}

/**
 * Linear resample a Float32Array from sourceRate to 16000 Hz (CREPE native).
 * @param {Float32Array} buffer — input samples at sourceRate
 * @param {number} sourceRate — original sample rate (e.g. 44100)
 * @returns {Float32Array} resampled buffer at 16000 Hz
 */
function resample16k(buffer, sourceRate) {
  const ratio = 16000 / sourceRate;
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

let ortNeuralSession = null;

/**
 * Initialize the ONNX Runtime session for CREPE-Tiny inference.
 * Called once when --neural flag is first used.
 */
async function initNeuralSession() {
  if (ortNeuralSession) return;
  const ort = await import('onnxruntime-node');
  const modelPath = resolve(PROJECT_ROOT, 'src', 'assets', 'models', 'crepe_tiny.onnx');
  if (!existsSync(modelPath)) {
    throw new Error(`Neural model not found: ${modelPath}\n  Run: python3 scripts/convert_crepe_onnx.py`);
  }
  ortNeuralSession = await ort.InferenceSession.create(modelPath);
  console.log('   🧠 Neural engine ready (CREPE-Tiny, quantized)');
}

/**
 * Neural pitch detector — wraps CREPE-Tiny ONNX inference.
 * Resamples the 44.1kHz frame to 16kHz before inference.
 *
 * @param {Float32Array} buffer — 4096-sample audio frame at 44.1kHz
 * @param {number} sampleRate — original sample rate
 * @returns {Promise<number>} detected frequency in Hz, or 0
 */
async function detectPitchNeural(buffer, sampleRate) {
  if (!ortNeuralSession) return 0;

  try {
    // Resample to CREPE's native 16kHz
    const resampled = resample16k(buffer, sampleRate);

    // Run ONNX inference
    const feeds = {
      input_audio: new (await import('onnxruntime-node')).Tensor('float32', resampled, [1, resampled.length])
    };
    const results = await ortNeuralSession.run(feeds);
    const probs = results.pitch_probabilities.data;

    return decodeCrepeHz(probs);
  } catch (err) {
    console.error(`  ⚠️ Neural inference error: ${err.message}`);
    return 0;
  }
}

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(__dirname, '..');
const AUDIO_DIR = resolve(PROJECT_ROOT, 'src', 'assets', 'audio');
const DATA_DIR = resolve(PROJECT_ROOT, 'data');
const GROUND_TRUTH_PATH = resolve(DATA_DIR, 'ground-truth.json');
const VALIDATION_DIR = resolve(DATA_DIR, 'validation-male');
const RATING_PATH = resolve(VALIDATION_DIR, 'rating.txt');
const SPLIT_PATH = resolve(DATA_DIR, 'train-split.json');

const FRAME_SIZE = 4096;
const HOP_SIZE = 512;
const RESAMPLE_LEN = 100;

// ── WAV parser (same as ingestPresets.js) ──────────────────────────

function parseWAV(buffer) {
  const riff = buffer.toString('ascii', 0, 4);
  if (riff !== 'RIFF') throw new Error('Not a valid WAV file: missing RIFF header');
  const wave = buffer.toString('ascii', 8, 12);
  if (wave !== 'WAVE') throw new Error('Not a valid WAV file: missing WAVE identifier');

  let offset = 12;
  let fmtChunk = null;
  let dataChunk = null;

  while (offset < buffer.length - 8) {
    const chunkId = buffer.toString('ascii', offset, offset + 4);
    const chunkSize = buffer.readUInt32LE(offset + 4);

    if (chunkId === 'fmt ') {
      fmtChunk = {
        audioFormat: buffer.readUInt16LE(offset + 8),
        channels: buffer.readUInt16LE(offset + 10),
        sampleRate: buffer.readUInt32LE(offset + 12),
        byteRate: buffer.readUInt32LE(offset + 16),
        blockAlign: buffer.readUInt16LE(offset + 20),
        bitsPerSample: buffer.readUInt16LE(offset + 22),
      };
    } else if (chunkId === 'data') {
      dataChunk = { offset: offset + 8, size: chunkSize };
      if (fmtChunk) break;
    }
    offset += 8 + chunkSize;
  }

  if (!fmtChunk) throw new Error('WAV file missing fmt chunk');
  if (!dataChunk) throw new Error('WAV file missing data chunk');

  const { channels, sampleRate, bitsPerSample } = fmtChunk;
  const bytesPerSample = bitsPerSample / 8;
  const totalSamples = Math.floor(dataChunk.size / bytesPerSample);
  const frameCount = Math.floor(totalSamples / channels);
  const samples = new Float32Array(frameCount);

  for (let i = 0; i < frameCount; i++) {
    let sum = 0;
    for (let ch = 0; ch < channels; ch++) {
      const byteOffset = dataChunk.offset + (i * channels + ch) * bytesPerSample;
      let sample;
      if (bitsPerSample === 8) {
        sample = (buffer.readUInt8(byteOffset) - 128) / 128;
      } else if (bitsPerSample === 16) {
        sample = buffer.readInt16LE(byteOffset) / 32768;
      } else if (bitsPerSample === 32) {
        sample = buffer.readInt32LE(byteOffset) / 2147483648;
      } else {
        throw new Error(`Unsupported bits per sample: ${bitsPerSample}`);
      }
      sum += sample;
    }
    samples[i] = sum / channels;
  }
  return { samples, sampleRate, channels, bitsPerSample };
}

// ── Load audio (MP3 via ffmpeg → parse WAV) ────────────────────────

import { execSync } from 'child_process';
import { tmpdir } from 'os';

function loadAudio(stem) {
  // Prefer MP3 (real recordings) over WAV (test fixtures)
  const mp3Path = join(AUDIO_DIR, `${stem}.mp3`);
  const wavPath = join(AUDIO_DIR, `${stem}.wav`);

  let filepath;
  if (existsSync(mp3Path)) {
    filepath = mp3Path;
  } else if (existsSync(wavPath)) {
    filepath = wavPath;
  } else {
    throw new Error(`No audio file found for ${stem}`);
  }

  const ext = filepath.toLowerCase().split('.').pop();
  if (ext === 'wav') {
    const buffer = readFileSync(filepath);
    return parseWAV(buffer);
  }

  // MP3 → WAV via ffmpeg
  const tmpWav = join(tmpdir(), `tt_train_${stem}.wav`);
  try {
    execSync(`ffmpeg -y -i "${filepath}" -ac 1 -ar 44100 -sample_fmt s16 "${tmpWav}" 2>/dev/null`, {
      timeout: 30000,
    });
    const buffer = readFileSync(tmpWav);
    return parseWAV(buffer);
  } finally {
    try { execSync(`rm -f "${tmpWav}"`); } catch {}
  }
}

// ── Pitch curve extraction ─────────────────────────────────────────

function extractPitchCurve(samples, sampleRate) {
  const pitches = [];
  for (let start = 0; start + FRAME_SIZE <= samples.length; start += HOP_SIZE) {
    const frame = samples.slice(start, start + FRAME_SIZE);
    pitches.push(detectPitchAMDF(frame, sampleRate));
  }
  return pitches;
}

function fullPipeline(samples, sampleRate) {
  const raw = extractPitchCurve(samples, sampleRate);
  const octaveCorrected = applyOctaveCorrection(raw);
  const interpolated = applySplineInterpolation(octaveCorrected);
  const smoothed = applyThreePointSmoothing(interpolated);
  const zScored = normalizeZScore(smoothed);
  const clamped = clampValues(zScored, -3, 3);
  return resampleArray(clamped, RESAMPLE_LEN);
}

function fullPipelineRaw(samples, sampleRate) {
  const raw = extractPitchCurve(samples, sampleRate);
  const octaveCorrected = applyOctaveCorrection(raw);
  const interpolated = applySplineInterpolation(octaveCorrected);
  return applyThreePointSmoothing(interpolated);
}

// ── Train/Validation Split (deterministic by stem hash) ────────────

function getOrCreateSplit(allStems) {
  if (existsSync(SPLIT_PATH)) {
    return JSON.parse(readFileSync(SPLIT_PATH, 'utf-8'));
  }

  // Deterministic split: hash stem → last hex digit decides fold
  const train = [];
  const val = [];

  for (const stem of allStems.sort()) {
    const hash = createHash('md5').update(stem).digest('hex');
    const lastNibble = parseInt(hash[hash.length - 1], 16);
    // ~20% validation split
    if (lastNibble < 3) {
      val.push(stem);
    } else {
      train.push(stem);
    }
  }

  const split = { train, val };
  mkdirSync(DATA_DIR, { recursive: true });
  writeFileSync(SPLIT_PATH, JSON.stringify(split, null, 2), 'utf-8');
  return split;
}

// ── Human Alignment (rating.txt parser) ────────────────────────────

function loadRatings() {
  if (!existsSync(RATING_PATH)) return null;

  const text = readFileSync(RATING_PATH, 'utf-8');
  const ratings = {};
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const [stem, rating] = trimmed.split(/\s+/);
    if (stem && rating) {
      ratings[stem] = rating.toUpperCase();
    }
  }
  return ratings;
}

// ── Main ────────────────────────────────────────────────────────────

function main() {
  const args = process.argv.slice(2);
  const useNeural = args.includes('--neural');
  const voicedOnly = args.includes('--voiced-only');

  if (useNeural) {
    console.log('╔══════════════════════════════════════════╗');
    console.log('║   AUTORESEARCH — Training Evaluation    ║');
    console.log('╚══════════════════════════════════════════╝');
    console.log('   Detector: CREPE-Tiny (neural)');
    if (voicedOnly) console.log('   MAE mode: voiced-only (JS>0 ∩ pYIN>0)');
    console.log('');
    mainNeural(voicedOnly);
    return;
  }

  console.log('╔══════════════════════════════════════════╗');
  console.log('║   AUTORESEARCH — Training Evaluation    ║');
  console.log('╚══════════════════════════════════════════╝');
  console.log('   Detector: AMDF');
  if (voicedOnly) console.log('   MAE mode: voiced-only (JS>0 ∩ pYIN>0)');
  console.log('');

  // Load ground truth
  if (!existsSync(GROUND_TRUTH_PATH)) {
    console.error('❌ Ground truth not found. Run: python3 scripts/prepare_pyin.py');
    process.exit(1);
  }
  const groundTruth = JSON.parse(readFileSync(GROUND_TRUTH_PATH, 'utf-8'));
  const allStems = Object.keys(groundTruth);
  console.log(`📊 Ground truth: ${allStems.length} words loaded\n`);

  // Train/validation split
  const split = getOrCreateSplit(allStems);
  console.log(`📂 Split: ${split.train.length} train / ${split.val.length} validation\n`);

  // ── Evaluate all words ────────────────────────────────────────────
  const results = {};

  for (const stem of allStems) {
    const gt = groundTruth[stem];
    let jsRaw;

    try {
      const { samples, sampleRate } = loadAudio(stem);
      jsRaw = fullPipelineRaw(samples, sampleRate);
    } catch (err) {
      console.log(`  ✗ ${stem}: ${err.message}`);
      continue;
    }

    // ── Shared Z-score normalization (Option A) ──────────────────
    // Compute μ/σ from ground truth voiced frames, then normalize
    // BOTH curves with the same statistics. This prevents unstable
    // z-scores when the JS pipeline finds few voiced frames.
    //
    // Guard: only use shared stats when GT has ≥20 voiced frames.
    // Below that, the GT's own μ/σ is too unstable to share.
    const gtVoicedHz = gt.pitch.filter(v => v > 0);
    const useSharedStats = gtVoicedHz.length >= 20;
    const gtMean = gtVoicedHz.length > 0
      ? gtVoicedHz.reduce((a, b) => a + b, 0) / gtVoicedHz.length
      : 0;
    const gtVar = gtVoicedHz.length > 1
      ? gtVoicedHz.reduce((s, v) => s + (v - gtMean) ** 2, 0) / gtVoicedHz.length
      : 0;
    const gtStd = Math.sqrt(gtVar);

    let jsNormalized, gtNormalized;

    if (useSharedStats && gtStd > 1e-10) {
      // Option A: shared normalization
      jsNormalized = normalizeWithSharedStats(jsRaw, gtMean, gtStd);
      gtNormalized = normalizeWithSharedStats(gt.pitch, gtMean, gtStd);
    } else {
      // Fallback: independent z-score normalization
      jsNormalized = normalizeZScore(jsRaw);
      gtNormalized = normalizeZScore(gt.pitch);
    }

    // Clamp and resample both to 100 points
    const jsClamped = clampValues(jsNormalized, -3, 3);
    const gtClamped = clampValues(gtNormalized, -3, 3);
    const jsResampled = resampleArray(jsClamped, RESAMPLE_LEN);
    const gtResampled = resampleArray(gtClamped, RESAMPLE_LEN);

    // Compute MAE
    const rawMae = computeRawMAE(jsResampled, gtResampled);
    const maeScore = calculateMAEScore(jsResampled, gtResampled);

    // Voiced-only MAE: only frames where both JS and pYIN detect voice
    let voicedMae = null;
    let voicedMaeScore = null;
    if (voicedOnly) {
      let vSum = 0, vCount = 0;
      for (let i = 0; i < jsResampled.length; i++) {
        if (jsResampled[i] !== 0 && gtResampled[i] !== 0) {
          vSum += Math.abs(jsResampled[i] - gtResampled[i]);
          vCount++;
        }
      }
      if (vCount > 0) {
        voicedMae = vSum / vCount;
        voicedMaeScore = Math.round(Math.max(0, 100 * (1 - voicedMae / 2.0)));
      }
    }

    // Voiced frame rate
    const jsVoiced = jsResampled.filter(v => v !== 0).length;
    const gtVoiced = gtResampled.filter(v => v !== 0).length;
    const voicedRate = (RESAMPLE_LEN - Math.abs(jsVoiced - gtVoiced)) / RESAMPLE_LEN;

    results[stem] = {
      mae: rawMae,
      maeScore,
      voicedMae,
      voicedMaeScore,
      voicedRate,
      jsVoiced,
      gtVoiced,
    };

    const setName = split.train.includes(stem) ? 'TRAIN' : 'VAL';
    const bar = makeBar(maeScore);
    let line = `  ${stem.padEnd(10)} │ ${bar} ${String(maeScore).padStart(3)}% │ MAE=${rawMae.toFixed(3)}`;
    if (voicedOnly && voicedMae != null) {
      line += ` │ vMAE=${voicedMae.toFixed(3)} (${voicedMaeScore}%)`;
    }
    line += ` │ ${setName}`;
    console.log(line);
  }

  // ── Aggregate ──────────────────────────────────────────────────────
  printAggregate(results, split, voicedOnly);
}

function printAggregate(results, split, voicedOnly) {
  const trainStems = split.train.filter(s => results[s]);
  const valStems = split.val.filter(s => results[s]);

  const trainMAE = avg(trainStems.map(s => results[s].mae));
  const valMAE = avg(valStems.map(s => results[s].mae));
  const trainScore = avg(trainStems.map(s => results[s].maeScore));
  const valScore = avg(valStems.map(s => results[s].maeScore));
  const allMAE = avg(Object.values(results).map(r => r.mae));
  const allScore = avg(Object.values(results).map(r => r.maeScore));

  let allVoicedMae = null;
  let allVoicedScore = null;
  if (voicedOnly) {
    const voicedResults = Object.values(results).filter(r => r.voicedMae != null);
    if (voicedResults.length > 0) {
      allVoicedMae = avg(voicedResults.map(r => r.voicedMae));
      allVoicedScore = avg(voicedResults.map(r => r.voicedMaeScore));
    }
  }

  console.log('\n═══════════════════════════════════════════');
  console.log('  AGGREGATE RESULTS');
  console.log('═══════════════════════════════════════════');
  console.log(`  Path A: Reference MAE (train)   = ${trainMAE.toFixed(4)}`);
  console.log(`  Path A: MAE Score (train)       = ${trainScore.toFixed(1)}%`);
  console.log(`  Path A: MAE Score (all words)   = ${allScore.toFixed(1)}%`);
  console.log(`  Path A: Raw MAE  (all words)    = ${allMAE.toFixed(4)}`);
  if (voicedOnly && allVoicedMae != null) {
    console.log(`  Path V: Voiced-only MAE (all)   = ${allVoicedMae.toFixed(4)} (${allVoicedScore.toFixed(1)}%)`);
  }
  console.log('───────────────────────────────────────────');
  if (valStems.length > 0) {
    const gap = Math.abs(trainMAE - valMAE);
    const gapPct = trainMAE > 0 ? (gap / trainMAE * 100).toFixed(1) : 'N/A';
    console.log(`  Path B: Validation MAE          = ${valMAE.toFixed(4)}`);
    console.log(`  Path B: Generalization Gap      = ${gap.toFixed(4)} (${gapPct}%)`);
    console.log(`  Path B: Validation Score        = ${valScore.toFixed(1)}%`);
  }
  console.log('───────────────────────────────────────────');

  const ratings = loadRatings();
  let alignPass = 0;
  let alignTotal = 0;
  let alignPct = 'N/A';
  if (ratings) {
    let alignDetails = [];
    for (const [stem, rating] of Object.entries(ratings)) {
      if (!results[stem]) continue;
      alignTotal++;
      const score = results[stem].maeScore;
      const shouldPass = rating === 'PERFECT';
      const shouldFail = rating === 'TOTAL_FAIL';
      const didPass = score >= 70;
      const didFail = score < 30;
      let correct = false;
      if (shouldPass && didPass) correct = true;
      if (shouldFail && didFail) correct = true;
      if (correct) alignPass++;
      alignDetails.push(`${stem}: ${rating} → score=${score}% ${correct ? '✓' : '✗'}`);
    }
    alignPct = alignTotal > 0 ? (alignPass / alignTotal * 100).toFixed(1) : 'N/A';
    console.log(`  Path C: Human Alignment         = ${alignPass}/${alignTotal} (${alignPct}%)`);
    for (const d of alignDetails) console.log(`    ${d}`);
  } else {
    console.log(`  Path C: Human Alignment         = (no rating.txt yet)`);
  }
  console.log('═══════════════════════════════════════════\n');

  console.log('  ACCEPTANCE CRITERIA:');
  const maeOk = allMAE <= 0.15;
  const gapOk = valStems.length === 0 || (trainMAE > 0 && Math.abs(trainMAE - valMAE) / trainMAE <= 0.05);
  console.log(`    MAE ≤ 0.15:      ${maeOk ? '✅ PASS' : '❌ FAIL'} (${allMAE.toFixed(3)})`);
  console.log(`    Val Gap ≤ 5%:     ${gapOk ? '✅ PASS' : '❌ FAIL'} (${trainMAE > 0 ? (Math.abs(trainMAE - valMAE) / trainMAE * 100).toFixed(1) : 'N/A'}%)`);
  console.log(`    Human Align:      ${ratings ? (alignPct === '100.0' ? '✅ PASS' : '❌ FAIL') : '⏳ PENDING'}`);
}

function avg(arr) {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function computeRawMAE(a, b) {
  if (a.length !== b.length) return Infinity;
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  return sum / a.length;
}

function makeBar(score) {
  const filled = Math.round(score / 5);
  const empty = 20 - filled;
  return '█'.repeat(Math.max(0, filled)) + '░'.repeat(Math.max(0, empty));
}

// ── Neural evaluation loop (async, --neural flag) ────────────────────

async function mainNeural(voicedOnly) {
  // Initialize ONNX session
  try {
    await initNeuralSession();
  } catch (err) {
    console.error(`❌ ${err.message}`);
    process.exit(1);
  }

  const groundTruth = JSON.parse(readFileSync(GROUND_TRUTH_PATH, 'utf-8'));
  const allStems = Object.keys(groundTruth);
  const split = getOrCreateSplit(allStems);
  console.log(`📊 Ground truth: ${allStems.length} words loaded`);
  console.log(`📂 Split: ${split.train.length} train / ${split.val.length} validation\n`);

  const results = {};

  for (const stem of allStems) {
    const gt = groundTruth[stem];

    try {
      const { samples, sampleRate } = loadAudio(stem);

      // Neural pitch curve extraction (async per frame)
      const rawPitches = [];
      for (let start = 0; start + FRAME_SIZE <= samples.length; start += HOP_SIZE) {
        const frame = samples.slice(start, start + FRAME_SIZE);
        const pitch = await detectPitchNeural(frame, sampleRate);
        rawPitches.push(pitch);
      }

      // Same post-processing as AMDF pipeline
      const octaveCorrected = applyOctaveCorrection(rawPitches);
      const interpolated = applySplineInterpolation(octaveCorrected);
      const jsRaw = applyThreePointSmoothing(interpolated);

      // Shared Z-score normalization (same as AMDF path)
      const gtVoicedHz = gt.pitch.filter(v => v > 0);
      const useSharedStats = gtVoicedHz.length >= 20;
      const gtMean = gtVoicedHz.length > 0
        ? gtVoicedHz.reduce((a, b) => a + b, 0) / gtVoicedHz.length : 0;
      const gtVar = gtVoicedHz.length > 1
        ? gtVoicedHz.reduce((s, v) => s + (v - gtMean) ** 2, 0) / gtVoicedHz.length : 0;
      const gtStd = Math.sqrt(gtVar);

      let jsNormalized, gtNormalized;
      if (useSharedStats && gtStd > 1e-10) {
        jsNormalized = normalizeWithSharedStats(jsRaw, gtMean, gtStd);
        gtNormalized = normalizeWithSharedStats(gt.pitch, gtMean, gtStd);
      } else {
        jsNormalized = normalizeZScore(jsRaw);
        gtNormalized = normalizeZScore(gt.pitch);
      }

      const jsClamped = clampValues(jsNormalized, -3, 3);
      const gtClamped = clampValues(gtNormalized, -3, 3);
      const jsResampled = resampleArray(jsClamped, RESAMPLE_LEN);
      const gtResampled = resampleArray(gtClamped, RESAMPLE_LEN);

      const rawMae = computeRawMAE(jsResampled, gtResampled);
      const maeScore = calculateMAEScore(jsResampled, gtResampled);

      let voicedMae = null, voicedMaeScore = null;
      if (voicedOnly) {
        let vSum = 0, vCount = 0;
        for (let i = 0; i < jsResampled.length; i++) {
          if (jsResampled[i] !== 0 && gtResampled[i] !== 0) {
            vSum += Math.abs(jsResampled[i] - gtResampled[i]);
            vCount++;
          }
        }
        if (vCount > 0) {
          voicedMae = vSum / vCount;
          voicedMaeScore = Math.round(Math.max(0, 100 * (1 - voicedMae / 2.0)));
        }
      }

      const jsVoiced = jsResampled.filter(v => v !== 0).length;
      const gtVoiced = gtResampled.filter(v => v !== 0).length;
      const voicedRate = (RESAMPLE_LEN - Math.abs(jsVoiced - gtVoiced)) / RESAMPLE_LEN;

      results[stem] = { mae: rawMae, maeScore, voicedMae, voicedMaeScore, voicedRate, jsVoiced, gtVoiced };

      const setName = split.train.includes(stem) ? 'TRAIN' : 'VAL';
      const bar = makeBar(maeScore);
      let line = `  ${stem.padEnd(10)} │ ${bar} ${String(maeScore).padStart(3)}% │ MAE=${rawMae.toFixed(3)}`;
      if (voicedOnly && voicedMae != null) line += ` │ vMAE=${voicedMae.toFixed(3)} (${voicedMaeScore}%)`;
      line += ` │ ${setName}`;
      console.log(line);

    } catch (err) {
      console.log(`  ✗ ${stem}: ${err.message}`);
    }
  }

  // Aggregate (same as synchronous path)
  printAggregate(results, split, voicedOnly);
}

main();
