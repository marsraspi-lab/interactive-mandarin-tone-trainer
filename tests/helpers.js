/**
 * Shared test helpers for generating synthetic audio buffers.
 */

/**
 * Generate a sine wave Float32Array at a given frequency.
 */
export function generateSineWave(frequency, sampleRate, durationSec = 0.1) {
  const samples = Math.floor(sampleRate * durationSec);
  const buffer = new Float32Array(samples);
  for (let i = 0; i < samples; i++) {
    buffer[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate);
  }
  return buffer;
}

/** White noise buffer — aperiodic, AMDF should find no clear period */
export function generateWhiteNoise(sampleRate, durationSec = 0.1) {
  const samples = Math.floor(sampleRate * durationSec);
  const buffer = new Float32Array(samples);
  for (let i = 0; i < samples; i++) {
    buffer[i] = (Math.random() - 0.5) * 2;
  }
  return buffer;
}
