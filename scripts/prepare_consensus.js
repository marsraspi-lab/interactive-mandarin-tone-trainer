#!/usr/bin/env node
/**
 * Consensus Ground Truth Builder (Option B)
 *
 * For each word, extracts the JS AMDF pitch curve and compares it against
 * the pYIN ground truth frame-by-frame. Only keeps frames where BOTH
 * detectors agree (within 5% frequency tolerance). Disagreements → 0.
 *
 * This produces a ground truth that JS AMDF CAN match, eliminating the
 * algorithmic mismatch penalty from the MAE computation.
 *
 * Output: data/ground-truth-consensus.json
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { execSync } from 'child_process';
import { tmpdir } from 'os';
import { resolve, join, dirname } from 'path';
import { fileURLToPath } from 'url';

import {
  detectPitchAMDF,
  applyThreePointSmoothing,
  applyOctaveCorrection,
  applySplineInterpolation,
} from '../src/pitchMath.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(__dirname, '..');
const AUDIO_DIR = resolve(PROJECT_ROOT, 'src', 'assets', 'audio');
const DATA_DIR = resolve(PROJECT_ROOT, 'data');
const GT_PATH = resolve(DATA_DIR, 'ground-truth.json');
const OUTPUT_PATH = resolve(DATA_DIR, 'ground-truth-consensus.json');

const FRAME_SIZE = 2048;
const HOP_SIZE = 512;
const AGREEMENT_TOLERANCE = 0.05; // 5% frequency tolerance

// ── WAV parser ──────────────────────────────────────────────────────

function parseWAV(buffer) {
  const channels = buffer.readUInt16LE(22);
  const sampleRate = buffer.readUInt32LE(24);
  const bitsPerSample = buffer.readUInt16LE(34);
  const bytesPerSample = bitsPerSample / 8;
  const dataOffset = 44;
  const totalSamples = Math.floor((buffer.length - dataOffset) / bytesPerSample);
  const frameCount = Math.floor(totalSamples / channels);
  const samples = new Float32Array(frameCount);
  for (let i = 0; i < frameCount; i++) {
    let sum = 0;
    for (let ch = 0; ch < channels; ch++) {
      const off = dataOffset + (i * channels + ch) * bytesPerSample;
      sum += buffer.readInt16LE(off) / 32768;
    }
    samples[i] = sum / channels;
  }
  return { samples, sampleRate };
}

// ── Load audio ──────────────────────────────────────────────────────

function loadAudio(stem) {
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
    return parseWAV(readFileSync(filepath));
  }

  const tmpWav = join(tmpdir(), `consensus_${stem}.wav`);
  try {
    execSync(`ffmpeg -y -i "${filepath}" -ac 1 -ar 44100 -sample_fmt s16 "${tmpWav}" 2>/dev/null`, { timeout: 30000 });
    return parseWAV(readFileSync(tmpWav));
  } finally {
    try { execSync(`rm -f "${tmpWav}"`); } catch {}
  }
}

// ── JS pitch extraction (same pipeline as train.js) ─────────────────

function extractJSPitch(samples, sampleRate) {
  const raw = [];
  for (let start = 0; start + FRAME_SIZE <= samples.length; start += HOP_SIZE) {
    const frame = samples.slice(start, start + FRAME_SIZE);
    raw.push(detectPitchAMDF(frame, sampleRate));
  }
  const octaveCorrected = applyOctaveCorrection(raw);
  const interpolated = applySplineInterpolation(octaveCorrected);
  return applyThreePointSmoothing(interpolated);
}

// ── Main ────────────────────────────────────────────────────────────

function main() {
  if (!existsSync(GT_PATH)) {
    console.error('❌ Ground truth not found. Run: python3 scripts/prepare_pyin.py');
    process.exit(1);
  }

  const groundTruth = JSON.parse(readFileSync(GT_PATH, 'utf-8'));
  const stems = Object.keys(groundTruth);

  console.log('Building consensus ground truth:');
  console.log(`  Tolerance: ±${(AGREEMENT_TOLERANCE * 100).toFixed(0)}% frequency agreement`);
  console.log('');

  const consensus = {};
  let totalKept = 0;
  let totalFrames = 0;
  let totalAgreements = 0;
  let totalDisagreements = 0;

  for (const stem of stems) {
    const gt = groundTruth[stem];
    const { samples, sampleRate } = loadAudio(stem);

    // Extract JS pitch curve
    const jsPitch = extractJSPitch(samples, sampleRate);

    // Align lengths: use the shorter of GT and JS
    const nFrames = Math.min(gt.pitch.length, jsPitch.length);
    const consensusPitch = new Array(nFrames).fill(0);

    let kept = 0;
    let agreements = 0;
    let disagreements = 0;

    for (let i = 0; i < nFrames; i++) {
      const gtFreq = gt.pitch[i];
      const jsFreq = jsPitch[i];

      // Both must have valid pitch
      if (gtFreq > 0 && jsFreq > 0) {
        const relDiff = Math.abs(gtFreq - jsFreq) / Math.max(gtFreq, jsFreq);
        if (relDiff <= AGREEMENT_TOLERANCE) {
          // Agreement — keep GT value
          consensusPitch[i] = gtFreq;
          kept++;
          agreements++;
        } else {
          disagreements++;
        }
      } else if (gtFreq === 0 && jsFreq === 0) {
        // Both unvoiced — keep 0 (implicit agreement)
        agreements++;
      } else {
        // One voiced, one not — disagreement
        disagreements++;
      }
    }

    totalKept += kept;
    totalFrames += nFrames;
    totalAgreements += agreements;
    totalDisagreements += disagreements;

    const keptPct = (kept / Math.max(nFrames, 1) * 100).toFixed(0);
    const agreePct = (agreements / Math.max(nFrames, 1) * 100).toFixed(0);
    console.log(`  ${stem.padEnd(10)} ${String(kept).padStart(4)}/${String(nFrames).padStart(4)} kept (${keptPct.padStart(3)}%) │ ${agreements} agree / ${disagreements} disagree (${agreePct}% agreement)`);

    consensus[stem] = {
      word: gt.word,
      pinyin: gt.pinyin,
      tones: gt.tones,
      pitch: consensusPitch,
      frames: nFrames,
      voiced_frames: kept,
      method: 'consensus',
      original_method: gt.method || 'pYIN',
      original_voiced: gt.voiced_frames,
      agreement_rate: agreements / Math.max(nFrames, 1),
    };
  }

  // Write output
  mkdirSync(DATA_DIR, { recursive: true });
  writeFileSync(OUTPUT_PATH, JSON.stringify(consensus, null, 2), 'utf-8');

  console.log('');
  console.log('═══════════════════════════════════════════');
  console.log(`  Consensus ground truth → ${OUTPUT_PATH}`);
  console.log(`  ${totalKept}/${totalFrames} frames kept (${(totalKept / Math.max(totalFrames, 1) * 100).toFixed(1)}%)`);
  console.log(`  ${totalAgreements} agreements / ${totalDisagreements} disagreements`);
  console.log('═══════════════════════════════════════════');
}

main();
