#!/usr/bin/env node
/**
 * Offline JSON-to-JSON preset ingestion pipeline using pYIN ground truth.
 *
 * Reads pYIN pitch curves from data/ground-truth.json, processes them
 * through the improved DSP pipeline (5-point median → Z-score → clamp
 * → resample 100), captures native μ/σ for shared normalization, and
 * outputs presets.json with all 24 words.
 *
 * Pipeline:  pYIN Hz → median smoothing → Z-score (capture μ/σ)
 *           → clamp → resample(100)
 *
 * For words with WAV fixtures, we use the AMDF pipeline instead
 * to maintain compatibility. Words without WAV fixtures fall back
 * to pYIN ground truth.
 *
 * All DSP functions are imported from src/pitchMath.js — no duplication.
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { resolve, join, dirname, extname } from 'path';
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
const GROUND_TRUTH_PATH = resolve(PROJECT_ROOT, 'data', 'ground-truth.json');
const OUTPUT_PATH = resolve(PROJECT_ROOT, 'presets.json');
const SRC_OUTPUT_PATH = resolve(PROJECT_ROOT, 'src', 'presets.json');
const PUBLIC_OUTPUT_PATH = resolve(PROJECT_ROOT, 'public', 'presets.json');

// ── WAV header parser ──────────────────────────────────────────────

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

// ── Pitch curve extraction from WAV ────────────────────────────────

function extractPitchCurve(samples, sampleRate, frameSize = 4096, hopSize = 512) {
  const pitches = [];
  for (let start = 0; start + frameSize <= samples.length; start += hopSize) {
    const frame = samples.slice(start, start + frameSize);
    const pitch = detectPitchAMDF(frame, sampleRate);
    pitches.push(pitch);
  }
  return pitches;
}

// ── Compute μ and σ before normalization ─────────────────────────

/**
 * Compute mean and std of voiced frames in a pitch array.
 * Returns { mean: number, std: number } or null if < 2 voiced frames.
 */
function computeNativeStats(pitchArray) {
  const voiced = pitchArray.filter(v => v > 0);
  if (voiced.length < 2) return null;

  let sum = 0;
  for (const v of voiced) sum += v;
  const mean = sum / voiced.length;

  let sumSqDiff = 0;
  for (const v of voiced) sumSqDiff += (v - mean) ** 2;
  const std = Math.sqrt(sumSqDiff / voiced.length);

  if (std < 1e-10) return null;

  return { mean: Math.round(mean * 100) / 100, std: Math.round(std * 100) / 100 };
}

// ── Pipeline ───────────────────────────────────────────────────────

/**
 * Process raw Hz pitch array through the DSP pipeline.
 * Returns { reference: number[], stats: {mean, std} | null }
 */
function processCurve(rawPitches) {
  // Step 1: 5-point median smoothing
  const smoothed = applyThreePointSmoothing(rawPitches);

  // Step 2: Capture native μ/σ before normalization
  const stats = computeNativeStats(smoothed);

  // Step 3: Z-score normalization
  const zScored = normalizeZScore(smoothed);

  // Step 4: Clamp to [-3, +3]
  const clamped = clampValues(zScored, -3, 3);

  // Step 5: Resample to exactly 100 points
  const resampled = resampleArray(clamped, 100);

  // Round to 4 decimal places for clean JSON
  const reference = resampled.map(v => Math.round(v * 10000) / 10000);

  return { reference, stats };
}

/**
 * Process a WAV file through AMDF → smoothing → Z-score pipeline.
 */
function processWAV(wavPath) {
  const buffer = readFileSync(wavPath);
  const { samples, sampleRate } = parseWAV(buffer);
  const rawPitches = extractPitchCurve(samples, sampleRate);
  return processCurve(rawPitches);
}

/**
 * Process pYIN ground truth Hz values through smoothing → Z-score pipeline.
 */
function processPYIN(pitchHzArray) {
  return processCurve(pitchHzArray);
}

// ── Full preset definitions (24 words) ───────────────────────────

const PRESET_DEFS = [
  // Single syllables
  { word: '妈',   pinyin: 'mā',       tones: [1],    wav: 'ma1.wav',     audioSrc: '/assets/audio/ma1.mp3' },
  { word: '麻',   pinyin: 'má',       tones: [2],    wav: 'ma2.wav',     audioSrc: '/assets/audio/ma2.mp3' },
  { word: '马',   pinyin: 'mǎ',       tones: [3],    wav: 'ma3.wav',     audioSrc: '/assets/audio/ma3.mp3' },
  { word: '骂',   pinyin: 'mà',       tones: [4],    wav: 'ma4.wav',     audioSrc: '/assets/audio/ma4.mp3' },
  // Tone pairs — 1+X
  { word: '今天', pinyin: 'jīntiān',  tones: [1,1],  wav: null,          audioSrc: '/assets/audio/jintian.mp3' },
  { word: '今年', pinyin: 'jīnnián',  tones: [1,2],  wav: null,          audioSrc: '/assets/audio/jinnian.mp3' },
  { word: '机场', pinyin: 'jīchǎng',  tones: [1,3],  wav: null,          audioSrc: '/assets/audio/jichang.mp3' },
  { word: '音乐', pinyin: 'yīnyuè',   tones: [1,4],  wav: 'yinyue.wav',  audioSrc: '/assets/audio/yinyue.mp3' },
  { word: '哥哥', pinyin: 'gēge',     tones: [1,0],  wav: null,          audioSrc: '/assets/audio/gege.mp3' },
  // 2+X
  { word: '明天', pinyin: 'míngtiān', tones: [2,1],  wav: 'mingtian.wav', audioSrc: '/assets/audio/mingtian.mp3' },
  { word: '明年', pinyin: 'míngnián', tones: [2,2],  wav: null,           audioSrc: '/assets/audio/mingnian.mp3' },
  { word: '苹果', pinyin: 'píngguǒ',  tones: [2,3],  wav: null,           audioSrc: '/assets/audio/pingguo.mp3' },
  { word: '决定', pinyin: 'juédìng',  tones: [2,4],  wav: null,           audioSrc: '/assets/audio/jueding.mp3' },
  { word: '孩子', pinyin: 'háizi',    tones: [2,0],  wav: null,           audioSrc: '/assets/audio/haizi.mp3' },
  // 3+X
  { word: '老师', pinyin: 'lǎoshī',   tones: [3,1],  wav: 'laoshi.wav',  audioSrc: '/assets/audio/laoshi.mp3' },
  { word: '旅行', pinyin: 'lǚxíng',   tones: [3,2],  wav: null,          audioSrc: '/assets/audio/luxing.mp3' },
  { word: '水果', pinyin: 'shuǐguǒ',  tones: [3,3],  wav: null,          audioSrc: '/assets/audio/shuiguo.mp3' },
  { word: '好看', pinyin: 'hǎokàn',   tones: [3,4],  wav: 'haokan.wav',  audioSrc: '/assets/audio/haokan.mp3' },
  { word: '姐姐', pinyin: 'jiějie',   tones: [3,0],  wav: null,          audioSrc: '/assets/audio/jiejie.mp3' },
  // 4+X
  { word: '唱歌', pinyin: 'chànggē',  tones: [4,1],  wav: 'changge.wav', audioSrc: '/assets/audio/changge.mp3' },
  { word: '问题', pinyin: 'wèntí',    tones: [4,2],  wav: null,          audioSrc: '/assets/audio/wenti.mp3' },
  { word: '电脑', pinyin: 'diànnǎo',  tones: [4,3],  wav: null,          audioSrc: '/assets/audio/diannao.mp3' },
  { word: '再见', pinyin: 'zàijiàn',  tones: [4,4],  wav: null,          audioSrc: '/assets/audio/zaijian.mp3' },
  { word: '谢谢', pinyin: 'xièxie',   tones: [4,0],  wav: null,          audioSrc: '/assets/audio/xiexie.mp3' },
];

// ── Main ───────────────────────────────────────────────────────────

// fallow-ignore-next-line complexity
function main() {
  // Load pYIN ground truth
  let groundTruth = {};
  if (existsSync(GROUND_TRUTH_PATH)) {
    groundTruth = JSON.parse(readFileSync(GROUND_TRUTH_PATH, 'utf-8'));
    const nWords = Object.keys(groundTruth).filter(k => typeof groundTruth[k] === 'object' && 'pitch' in groundTruth[k]).length;
    console.log(`📖 Loaded pYIN ground truth: ${nWords} words`);
  } else {
    console.warn('⚠️  data/ground-truth.json not found — words without WAV will have zero references');
  }

  const presets = [];
  let wavCount = 0;
  let pyinCount = 0;
  let zeroCount = 0;

  for (const def of PRESET_DEFS) {
    let reference, stats;
    const wavPath = def.wav ? join(FIXTURES_DIR, def.wav) : null;

    if (wavPath && existsSync(wavPath)) {
      // Use WAV → AMDF pipeline
      try {
        const result = processWAV(wavPath);
        reference = result.reference;
        stats = result.stats;
        wavCount++;
      } catch (err) {
        console.error(`✗ WAV error ${def.wav}: ${err.message}`);
        reference = new Array(100).fill(0);
        stats = null;
      }
    } else if (groundTruth[def.wav ? def.wav.replace('.wav', '') : def.audioSrc.split('/').pop().replace('.mp3', '')]) {
      // Use pYIN ground truth
      const key = def.wav ? def.wav.replace('.wav', '') : def.audioSrc.split('/').pop().replace('.mp3', '');
      const gtData = groundTruth[key];
      if (gtData && gtData.pitch && gtData.pitch.length > 0) {
        try {
          const result = processPYIN(gtData.pitch);
          reference = result.reference;
          stats = result.stats;
          pyinCount++;
        } catch (err) {
          console.error(`✗ pYIN error ${def.word}: ${err.message}`);
          reference = new Array(100).fill(0);
          stats = null;
        }
      } else {
        reference = new Array(100).fill(0);
        stats = null;
        zeroCount++;
      }
    } else {
      reference = new Array(100).fill(0);
      stats = null;
      zeroCount++;
    }

    // Validate schema: must be exactly 100 numeric values
    if (!Array.isArray(reference) || reference.length !== 100) {
      console.error(`✗ Schema violation for ${def.word}: got ${reference?.length ?? 'non-array'}`);
      reference = new Array(100).fill(0);
      stats = null;
    }

    const preset = {
      word: def.word,
      pinyin: def.pinyin,
      tones: def.tones,
      audioSrc: def.audioSrc,
      nativePitchReference: reference,
    };

    // Embed native μ/σ for shared z-score normalization (if available)
    if (stats && stats.std >= 1e-10) {
      preset.nativeMean = stats.mean;
      preset.nativeStd = stats.std;
    }

    presets.push(preset);

    const voiced = reference.filter(v => v !== 0).length;
    const nonZero = reference.filter(v => v !== 0);
    const min = nonZero.length ? Math.min(...nonZero).toFixed(2) : 'N/A';
    const max = nonZero.length ? Math.max(...nonZero).toFixed(2) : 'N/A';
    const source = wavPath && existsSync(wavPath) ? 'WAV' : (stats ? 'pYIN' : 'ZERO');
    const statsInfo = stats ? ` μ=${stats.mean} σ=${stats.std}` : '';
    console.log(`✓ ${def.word.padEnd(4)} (${def.pinyin.padEnd(10)}) [${source}] — ${voiced}/100 voiced, range [${min}, ${max}]${statsInfo}`);
  }

  // Write outputs
  const output = { presets };
  const json = JSON.stringify(output, null, 2);

  writeFileSync(OUTPUT_PATH, json, 'utf-8');
  console.log(`\n✅ presets.json written → ${OUTPUT_PATH}`);

  mkdirSync(resolve(PROJECT_ROOT, 'src'), { recursive: true });
  writeFileSync(SRC_OUTPUT_PATH, json, 'utf-8');
  console.log(`✅ presets.json synced  → ${SRC_OUTPUT_PATH}`);

  mkdirSync(resolve(PROJECT_ROOT, 'public'), { recursive: true });
  writeFileSync(PUBLIC_OUTPUT_PATH, json, 'utf-8');
  console.log(`✅ presets.json synced  → ${PUBLIC_OUTPUT_PATH}`);

  console.log(`\n📊 Summary: ${presets.length} presets (${wavCount} WAV, ${pyinCount} pYIN, ${zeroCount} zero)`);
}

main();
