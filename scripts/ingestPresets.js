#!/usr/bin/env node
/**
 * Offline WAV-to-JSON preset ingestion pipeline.
 * Reads WAV files from the audio/ directory, extracts pitch curves using AMDF,
 * applies 3-point smoothing, normalizes to 0–1 range, resamples to exactly
 * 100 points, and outputs presets.json at the project root.
 *
 * AMDF implementation is self-contained (not imported from src/) so this
 * script works standalone as a Node ESM module.
 */

import { readFileSync, writeFileSync, readdirSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(__dirname, '..');
const AUDIO_DIR = resolve(PROJECT_ROOT, 'src', 'assets', 'audio');
const OUTPUT_PATH = resolve(PROJECT_ROOT, 'src', 'presets.json');

// ── WAV header parser ──────────────────────────────────────────────

/**
 * Parse a WAV file buffer and extract PCM audio samples as Float32Array.
 * Supports 8-bit, 16-bit, and 32-bit PCM; mono and stereo.
 *
 * @param {Buffer} buffer — raw WAV file contents
 * @returns {{ samples: Float32Array, sampleRate: number, channels: number, bitsPerSample: number }}
 */
function parseWAV(buffer) {
  // RIFF header
  const riff = buffer.toString('ascii', 0, 4);
  if (riff !== 'RIFF') throw new Error('Not a valid WAV file: missing RIFF header');

  // File size (bytes 4–7), skip
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
      dataChunk = {
        offset: offset + 8,
        size: chunkSize,
      };
      // We found both chunks we need, stop parsing
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
        // 8-bit PCM is unsigned
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

    // Average across channels (mono mixdown for stereo)
    samples[i] = sum / channels;
  }

  return { samples, sampleRate, channels, bitsPerSample };
}

// ── AMDF pitch detection (replicated from src/pitchMath.js) ───────

/**
 * Detect fundamental frequency (f0) using AMDF.
 * Same algorithm as src/pitchMath.js detectPitchAMDF.
 *
 * @param {Float32Array} buffer — time-domain audio samples
 * @param {number} sampleRate — samples per second
 * @returns {number} detected frequency in Hz, or 0 if no voice detected
 */
function detectPitchAMDF(buffer, sampleRate) {
  const MIN_FREQ = 60;
  const MAX_FREQ = 400;
  const MIN_PERIOD = Math.floor(sampleRate / MAX_FREQ);
  const MAX_PERIOD = Math.floor(sampleRate / MIN_FREQ);

  // Silence gate
  let sumSq = 0;
  for (let i = 0; i < buffer.length; i++) {
    sumSq += buffer[i] * buffer[i];
  }
  const rms = Math.sqrt(sumSq / buffer.length);
  if (rms < 0.005) return 0;

  let bestPeriod = -1;
  let bestDiff = Infinity;

  for (let period = MIN_PERIOD; period <= MAX_PERIOD; period++) {
    let diffSum = 0;
    let count = 0;
    for (let i = 0; i < buffer.length - period; i++) {
      diffSum += Math.abs(buffer[i] - buffer[i + period]);
      count++;
    }
    const avgDiff = diffSum / count;

    if (avgDiff < bestDiff - 1e-12) {
      bestDiff = avgDiff;
      bestPeriod = period;
    }
  }

  if (bestPeriod <= 0) return 0;

  // Periodicity gate
  if (bestDiff > rms * 0.4) return 0;

  const frequency = sampleRate / bestPeriod;

  // Bandpass gate
  if (frequency < MIN_FREQ || frequency > MAX_FREQ) return 0;

  return frequency;
}

// ── 3-point smoothing (replicated from src/pitchMath.js) ───────────

/**
 * Apply 3-point moving average filter.
 * Same algorithm as src/pitchMath.js applyThreePointSmoothing.
 *
 * @param {number[]} pitchArray — sequential pitch values in Hz
 * @returns {number[]} smoothed pitch array
 */
function applyThreePointSmoothing(pitchArray) {
  if (pitchArray.length < 2) return [...pitchArray];

  const result = new Array(pitchArray.length);
  result[0] = (pitchArray[0] + pitchArray[1]) / 2;
  for (let i = 1; i < pitchArray.length - 1; i++) {
    result[i] = (pitchArray[i - 1] + pitchArray[i] + pitchArray[i + 1]) / 3;
  }
  result[pitchArray.length - 1] =
    (pitchArray[pitchArray.length - 2] + pitchArray[pitchArray.length - 1]) / 2;
  return result;
}

// ── Pitch curve extraction ────────────────────────────────────────

/**
 * Extract a pitch curve from PCM samples by sliding a window across the
 * signal and running AMDF on each frame.
 *
 * @param {Float32Array} samples — full audio samples
 * @param {number} sampleRate — samples per second
 * @param {number} [frameSize=2048] — analysis window size
 * @param {number} [hopSize=512] — stride between windows
 * @returns {number[]} pitch values in Hz (0 = no voice)
 */
function extractPitchCurve(samples, sampleRate, frameSize = 2048, hopSize = 512) {
  const pitches = [];

  for (let start = 0; start + frameSize <= samples.length; start += hopSize) {
    const frame = samples.slice(start, start + frameSize);
    const pitch = detectPitchAMDF(frame, sampleRate);
    pitches.push(pitch);
  }

  return pitches;
}

// ── Resampling via linear interpolation ───────────────────────────

/**
 * Resample an array to exactly targetLength points via linear interpolation.
 *
 * @param {number[]} arr — input array
 * @param {number} targetLength — desired output length
 * @returns {number[]}
 */
function resample(arr, targetLength) {
  if (arr.length === 0) return new Array(targetLength).fill(0);
  if (arr.length === 1) return new Array(targetLength).fill(arr[0]);

  const result = new Array(targetLength);
  const step = (arr.length - 1) / (targetLength - 1);

  for (let i = 0; i < targetLength; i++) {
    const pos = i * step;
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, arr.length - 1);
    const frac = pos - lo;
    result[i] = arr[lo] + (arr[hi] - arr[lo]) * frac;
  }

  return result;
}

// ── Normalization ─────────────────────────────────────────────────

/**
 * Normalize pitch values to 0–1 range. Zeros are kept as 0.
 *
 * @param {number[]} pitches — array of pitch values in Hz
 * @returns {number[]} normalized values (0–1)
 */
function normalizePitches(pitches) {
  // Find min/max excluding zeros
  const nonZero = pitches.filter((v) => v > 0);
  if (nonZero.length === 0) return new Array(pitches.length).fill(0);

  const min = Math.min(...nonZero);
  const max = Math.max(...nonZero);
  const range = max - min;

  if (range === 0) return pitches.map((v) => (v > 0 ? 0.5 : 0));

  return pitches.map((v) => {
    if (v <= 0) return 0;
    return (v - min) / range;
  });
}

// ── Pipeline ───────────────────────────────────────────────────────

/**
 * Process a single WAV file and return a pitch reference curve (100 points, 0–1).
 *
 * @param {string} wavPath — absolute path to .wav file
 * @returns {number[]} 100-point normalized pitch reference
 */
function processWAV(wavPath) {
  const buffer = readFileSync(wavPath);
  const { samples, sampleRate } = parseWAV(buffer);

  // Extract raw pitch curve
  const rawPitches = extractPitchCurve(samples, sampleRate);

  // Apply 3-point smoothing
  const smoothed = applyThreePointSmoothing(rawPitches);

  // Filter out zeros for resampling (but we want to keep the contour shape)
  // Smoothing already helps. Let's interpolate over isolated zeros.
  const filled = fillZeros(smoothed);

  // Normalize to 0–1
  const normalized = normalizePitches(filled);

  // Resample to exactly 100 points
  const resampled = resample(normalized, 100);

  // Round to 4 decimal places
  return resampled.map((v) => Math.round(v * 10000) / 10000);
}

/**
 * Fill isolated zeros with linear interpolation between nearest non-zero neighbors.
 * Long runs of zeros (silence) are left as 0.
 *
 * @param {number[]} arr — pitch array with possible zeros
 * @returns {number[]} pitch array with isolated zeros filled
 */
function fillZeros(arr) {
  const result = [...arr];

  for (let i = 0; i < result.length; i++) {
    if (result[i] === 0) {
      // Find previous non-zero
      let prevIdx = i - 1;
      while (prevIdx >= 0 && result[prevIdx] === 0) prevIdx--;

      // Find next non-zero
      let nextIdx = i + 1;
      while (nextIdx < result.length && result[nextIdx] === 0) nextIdx++;

      // Only fill if both neighbors exist and the gap is small (≤ 3 frames)
      if (prevIdx >= 0 && nextIdx < result.length && nextIdx - prevIdx <= 4) {
        const frac = (i - prevIdx) / (nextIdx - prevIdx);
        result[i] = result[prevIdx] + (result[nextIdx] - result[prevIdx]) * frac;
      }
    }
  }

  return result;
}

// ── Preset definitions ────────────────────────────────────────────

const PRESET_DEFS = [
  { word: '公司', pinyin: 'gōngsī', tones: [1, 4], file: 'gongsi.wav', audioSrc: '/assets/audio/gongsi.mp3' },
  { word: '银行', pinyin: 'yínháng', tones: [2, 4], file: 'yinhang.wav', audioSrc: '/assets/audio/yinhang.mp3' },
  { word: '老师', pinyin: 'lǎoshī', tones: [3, 1], file: 'laoshi.wav', audioSrc: '/assets/audio/laoshi.mp3' },
];

// ── Main ───────────────────────────────────────────────────────────

function main() {
  const presets = [];

  for (const def of PRESET_DEFS) {
    const wavPath = resolve(AUDIO_DIR, def.file);

    let nativePitchReference;
    try {
      nativePitchReference = processWAV(wavPath);
    } catch (err) {
      console.error(`Error processing ${def.file}: ${err.message}`);
      // Fallback: produce a flat reference
      nativePitchReference = new Array(100).fill(0.5);
    }

    presets.push({
      word: def.word,
      pinyin: def.pinyin,
      tones: def.tones,
      audioSrc: def.audioSrc,
      nativePitchReference,
    });

    console.log(`✓ Processed ${def.word} (${def.pinyin}) — ${nativePitchReference.filter(v => v > 0).length}/100 non-zero points`);
  }

  const output = { presets };
  writeFileSync(OUTPUT_PATH, JSON.stringify(output, null, 2), 'utf-8');
  console.log(`\n✅ presets.json written with ${presets.length} presets`);
}

main();
