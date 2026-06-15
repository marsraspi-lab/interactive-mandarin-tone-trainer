#!/usr/bin/env node
/**
 * Offline WAV-to-JSON preset ingestion pipeline.
 *
 * Reads native speaker WAV recordings from tests/fixtures/native_samples/,
 * processes them through the EXACT same mathematical pipeline as the
 * client-side app (pitchMath.js), and outputs a production-ready
 * presets.json at the project root.
 *
 * Pipeline:  WAV parse → AMDF pitch detection → 3-point smoothing
 *           → Z-score normalization → threshold clamping → resample(100)
 *
 * All DSP functions are imported from src/pitchMath.js — no duplication.
 */

import { readFileSync, writeFileSync, readdirSync, existsSync, mkdirSync } from 'fs';
import { resolve, join, dirname, extname, basename } from 'path';
import { fileURLToPath } from 'url';

import {
  detectPitchAMDF,
  applyThreePointSmoothing,
  normalizeZScore,
  clampValues,
  resampleArray,
} from '../src/pitchMath.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(__dirname, '..');
const FIXTURES_DIR = resolve(PROJECT_ROOT, 'tests', 'fixtures', 'native_samples');
const OUTPUT_PATH = resolve(PROJECT_ROOT, 'presets.json');
const SRC_OUTPUT_PATH = resolve(PROJECT_ROOT, 'src', 'presets.json');
const PUBLIC_OUTPUT_PATH = resolve(PROJECT_ROOT, 'public', 'presets.json');

// ── WAV header parser ──────────────────────────────────────────────

/**
 * Parse a WAV file buffer and extract PCM audio samples as Float32Array.
 * Supports 8-bit, 16-bit, and 32-bit PCM; mono and stereo.
 *
 * @param {Buffer} buffer — raw WAV file contents
 * @returns {{ samples: Float32Array, sampleRate: number, channels: number, bitsPerSample: number }}
 */
// fallow-ignore-next-line complexity
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
      dataChunk = {
        offset: offset + 8,
        size: chunkSize,
      };
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
    samples[i] = sum / channels; // mono mixdown
  }

  return { samples, sampleRate, channels, bitsPerSample };
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
 * @returns {number[]} pitch values in Hz (0 = no voice detected)
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

// ── Pipeline ───────────────────────────────────────────────────────

/**
 * Process a single WAV file through the full DSP pipeline.
 *
 * Pipeline: parse → AMDF → 3-point smoothing → Z-score → clamp → resample(100)
 *
 * @param {string} wavPath — absolute path to .wav file
 * @returns {number[]} 100-point Z-score normalized, clamped pitch reference
 */
function processWAV(wavPath) {
  const buffer = readFileSync(wavPath);
  const { samples, sampleRate } = parseWAV(buffer);

  // Step 1: Extract raw pitch curve via AMDF
  const rawPitches = extractPitchCurve(samples, sampleRate);

  // Step 2: Apply 3-point moving average smoothing
  const smoothed = applyThreePointSmoothing(rawPitches);

  // Step 3: Z-score statistical normalization (μ=0, σ=1)
  const zScored = normalizeZScore(smoothed);

  // Step 4: Time-domain threshold clamping to [-3, +3]
  const clamped = clampValues(zScored, -3, 3);

  // Step 5: Resample to exactly 100 points via linear interpolation
  const resampled = resampleArray(clamped, 100);

  // Round to 4 decimal places for clean JSON
  return resampled.map(v => Math.round(v * 10000) / 10000);
}

// ── Preset definitions — single source of truth ───────────────────

/**
 * Each preset maps a word to its WAV fixture and delivery audio path.
 * The 'file' field is the WAV filename in tests/fixtures/native_samples/.
 * The 'audioSrc' field is the relative path for browser playback.
 */
const PRESET_DEFS = [
  { word: '妈',   pinyin: 'mā',       tones: [1],    file: 'ma1.wav',     audioSrc: '/assets/audio/ma1.mp3' },
  { word: '麻',   pinyin: 'má',       tones: [2],    file: 'ma2.wav',     audioSrc: '/assets/audio/ma2.mp3' },
  { word: '马',   pinyin: 'mǎ',       tones: [3],    file: 'ma3.wav',     audioSrc: '/assets/audio/ma3.mp3' },
  { word: '骂',   pinyin: 'mà',       tones: [4],    file: 'ma4.wav',     audioSrc: '/assets/audio/ma4.mp3' },
  { word: '公司', pinyin: 'gōngsī',   tones: [1, 1], file: 'gongsi.wav',  audioSrc: '/assets/audio/gongsi.mp3' },
  { word: '银行', pinyin: 'yínháng',  tones: [2, 2], file: 'yinhang.wav', audioSrc: '/assets/audio/yinhang.mp3' },
  { word: '老师', pinyin: 'lǎoshī',   tones: [3, 1], file: 'laoshi.wav',  audioSrc: '/assets/audio/laoshi.mp3' },
];

// ── Main ───────────────────────────────────────────────────────────

// fallow-ignore-next-line complexity
function main() {
  // Verify fixtures directory exists
  if (!existsSync(FIXTURES_DIR)) {
    console.error(`❌ Fixtures directory not found: ${FIXTURES_DIR}`);
    console.error('   Create tests/fixtures/native_samples/ with .wav files first.');
    process.exit(1);
  }

  const presets = [];

  for (const def of PRESET_DEFS) {
    const wavPath = join(FIXTURES_DIR, def.file);

    let nativePitchReference;
    try {
      nativePitchReference = processWAV(wavPath);
    } catch (err) {
      console.error(`✗ Error processing ${def.file}: ${err.message}`);
      console.error(`  Falling back to flat reference for ${def.word}`);
      nativePitchReference = new Array(100).fill(0);
    }

    // Schema validation: must be exactly 100 numeric values
    if (!Array.isArray(nativePitchReference) || nativePitchReference.length !== 100) {
      console.error(`✗ Schema violation for ${def.word}: expected 100 elements, got ${nativePitchReference?.length ?? 'non-array'}`);
      nativePitchReference = new Array(100).fill(0);
    }

    presets.push({
      word: def.word,
      pinyin: def.pinyin,
      tones: def.tones,
      audioSrc: def.audioSrc,
      nativePitchReference,
    });

    const voiced = nativePitchReference.filter(v => v !== 0).length;
    const range = nativePitchReference.filter(v => v !== 0);
    const min = range.length ? Math.min(...range).toFixed(2) : 'N/A';
    const max = range.length ? Math.max(...range).toFixed(2) : 'N/A';
    console.log(`✓ ${def.word.padEnd(4)} (${def.pinyin.padEnd(10)}) — ${voiced}/100 voiced, range [${min}, ${max}]`);
  }

  // Write to project root (canonical location)
  const output = { presets };
  const json = JSON.stringify(output, null, 2);
  writeFileSync(OUTPUT_PATH, json, 'utf-8');
  console.log(`\n✅ presets.json written → ${OUTPUT_PATH}`);

  // Also write to src/ for dev server compatibility (serve serves from src/)
  writeFileSync(SRC_OUTPUT_PATH, json, 'utf-8');
  console.log(`✅ presets.json synced  → ${SRC_OUTPUT_PATH}`);

  // Also write to public/ for Vite build (publicDir is copied to dist/ root)
  mkdirSync(resolve(PROJECT_ROOT, 'public'), { recursive: true });
  writeFileSync(PUBLIC_OUTPUT_PATH, json, 'utf-8');
  console.log(`✅ presets.json synced  → ${PUBLIC_OUTPUT_PATH}`);
  console.log(`   (${presets.length} presets, all Z-score normalized)`);
}

main();
