#!/usr/bin/env node
/**
 * Add consensus validity masks to presets.json.
 *
 * Reads standard presets.json and consensus ground truth, then
 * maps consensus-valid frames to 100-point positions. The resulting
 * presets carry a `consensusMask` boolean array that the app uses
 * to optionally grade only on reliable (pYIN ∩ AMDF) frames.
 *
 * Usage: node scripts/addConsensusMask.js
 */

import { readFileSync, writeFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(__dirname, '..');

const PRESETS_PATH = resolve(PROJECT_ROOT, 'presets.json');
const CONSENSUS_GT_PATH = resolve(PROJECT_ROOT, 'data', 'ground-truth-consensus.json');
const SRC_PATH = resolve(PROJECT_ROOT, 'src', 'presets.json');
const PUBLIC_PATH = resolve(PROJECT_ROOT, 'public', 'presets.json');

// ── Main ──────────────────────────────────────────────────────────

function main() {
  if (!existsSync(CONSENSUS_GT_PATH)) {
    console.error('❌ Consensus ground truth not found. Run scripts/prepare_consensus.js first.');
    process.exit(1);
  }
  if (!existsSync(PRESETS_PATH)) {
    console.error('❌ presets.json not found. Run scripts/ingestAllPresets.js first.');
    process.exit(1);
  }

  const consensusGT = JSON.parse(readFileSync(CONSENSUS_GT_PATH, 'utf-8'));
  const presetsData = JSON.parse(readFileSync(PRESETS_PATH, 'utf-8'));
  const presets = presetsData.presets;

  let totalConsensusFrames = 0;
  let wordsWithConsensus = 0;

  for (const preset of presets) {
    // Derive consensus GT key from audio filename (e.g., '/assets/audio/ma1.mp3' → 'ma1')
    const audioFile = preset.audioSrc.split('/').pop().replace('.mp3', '');
    const gtKey = consensusGT[audioFile] ? audioFile : null;

    if (!gtKey) {
      preset.consensusMask = new Array(100).fill(false);
      continue;
    }

    const gt = consensusGT[gtKey];
    const pitch = gt.pitch; // Hz array with zeros for non-consensus frames
    const totalFrames = pitch.length;

    // Map consensus frames to 100-point positions
    const mask = new Array(100).fill(false);
    for (let frameIdx = 0; frameIdx < totalFrames; frameIdx++) {
      if (pitch[frameIdx] > 0) {
        // Map frame index → 100-point position (round to nearest)
        const pos = Math.round((frameIdx / (totalFrames - 1)) * 99);
        mask[Math.min(pos, 99)] = true;
        totalConsensusFrames++;
      }
    }

    const consensusCount = mask.filter(Boolean).length;
    if (consensusCount > 0) wordsWithConsensus++;

    preset.consensusMask = mask;
    console.log(`✓ ${preset.word.padEnd(4)} ${preset.pinyin.padEnd(10)} consensus: ${consensusCount}/100 positions`);
  }

  // Write updated presets
  const json = JSON.stringify(presetsData, null, 2);
  writeFileSync(PRESETS_PATH, json, 'utf-8');
  writeFileSync(SRC_PATH, json, 'utf-8');
  writeFileSync(PUBLIC_PATH, json, 'utf-8');

  console.log(`\n📊 ${wordsWithConsensus}/${presets.length} words have consensus frames (${totalConsensusFrames} total)`);
  console.log('✅ presets.json updated with consensusMask');
}

main();
