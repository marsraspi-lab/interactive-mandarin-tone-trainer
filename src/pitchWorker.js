/**
 * Thin ESM Web Worker wrapper around pitchMath.js.
 * Receives audio buffer chunks from the main thread, runs AMDF pitch detection,
 * maintains a rolling pitch history, applies 3-point smoothing, and posts results back.
 *
 * Message format (received):
 *   { audioBuffer: Float32Array, sampleRate: number }
 *
 * Message format (posted):
 *   { frequency: number, rawFrequency: number, timestamp: number }
 */

import { detectPitchAMDF, applyThreePointSmoothing } from './pitchMath.js';

/**
 * Process a single audio frame through pitch detection + smoothing pipeline.
 * Exported for unit testing.
 *
 * @param {Float32Array} audioBuffer — time-domain audio samples
 * @param {number} sampleRate — samples per second
 * @param {number[]} pitchHistory — mutable rolling history array (max 5 entries)
 * @returns {{ frequency: number, rawFrequency: number, pitchHistory: number[] }}
 */
export function processAudioFrame(audioBuffer, sampleRate, pitchHistory) {
  const rawFrequency = detectPitchAMDF(audioBuffer, sampleRate);

  // Maintain rolling history (max 5 entries)
  pitchHistory.push(rawFrequency);
  if (pitchHistory.length > 5) {
    pitchHistory.shift();
  }

  // Apply 3-point smoothing when we have enough history
  let frequency = rawFrequency;
  if (pitchHistory.length >= 3) {
    const smoothed = applyThreePointSmoothing(pitchHistory);
    frequency = smoothed[smoothed.length - 1];
  }

  return { frequency, rawFrequency, pitchHistory };
}

// Worker message handler (only runs in Web Worker context)
if (typeof self !== 'undefined' && typeof self.onmessage !== 'undefined') {
  let pitchHistory = [];

  self.onmessage = function (e) {
    const { audioBuffer, sampleRate } = e.data;
    const result = processAudioFrame(audioBuffer, sampleRate, pitchHistory);

    self.postMessage({
      frequency: result.frequency,
      rawFrequency: result.rawFrequency,
      timestamp: Date.now(),
    });
  };
}
