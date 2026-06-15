import { describe, it, expect } from 'vitest';
import { detectPitchAMDF, applyThreePointSmoothing } from '../src/pitchMath.js';

/**
 * Generate a sine wave Float32Array at a given frequency.
 * Uses frequencies that divide evenly into 44100 Hz to avoid subharmonic
 * false positives (a known AMDF limitation with pure synthetic tones).
 */
function generateSineWave(frequency, sampleRate, durationSec = 0.1) {
  const samples = Math.floor(sampleRate * durationSec);
  const buffer = new Float32Array(samples);
  for (let i = 0; i < samples; i++) {
    buffer[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate);
  }
  return buffer;
}

/** White noise buffer — aperiodic, AMDF should find no clear period */
function generateWhiteNoise(sampleRate, durationSec = 0.1) {
  const samples = Math.floor(sampleRate * durationSec);
  const buffer = new Float32Array(samples);
  for (let i = 0; i < samples; i++) {
    buffer[i] = (Math.random() - 0.5) * 2;
  }
  return buffer;
}

describe('detectPitchAMDF', () => {
  it('detects a 220.5 Hz sine wave within 2% tolerance', () => {
    const sampleRate = 44100;
    const buffer = generateSineWave(220.5, sampleRate, 0.1);
    const pitch = detectPitchAMDF(buffer, sampleRate);
    expect(pitch).toBeGreaterThan(216);
    expect(pitch).toBeLessThan(225);
  });

  it('detects a 315 Hz sine wave within 2% tolerance', () => {
    const sampleRate = 44100;
    const buffer = generateSineWave(315, sampleRate, 0.1);
    const pitch = detectPitchAMDF(buffer, sampleRate);
    expect(pitch).toBeGreaterThan(308);
    expect(pitch).toBeLessThan(322);
  });

  it('rejects frequencies below 60 Hz (bandpass floor)', () => {
    const sampleRate = 44100;
    const buffer = generateSineWave(50, sampleRate, 0.1);
    const pitch = detectPitchAMDF(buffer, sampleRate);
    expect(pitch).toBe(0); // Filtered out — sub-vocal
  });

  it('rejects aperiodic high-frequency noise (bandpass ceiling)', () => {
    const sampleRate = 44100;
    const buffer = generateWhiteNoise(sampleRate, 0.1);
    const pitch = detectPitchAMDF(buffer, sampleRate);
    // White noise has no periodic structure — AMDF should find no valid pitch
    expect(pitch).toBe(0);
  });

  it('returns 0 for silence (near-zero amplitude)', () => {
    const buffer = new Float32Array(2048); // All zeros
    const pitch = detectPitchAMDF(buffer, 44100);
    expect(pitch).toBe(0);
  });
});

describe('applyThreePointSmoothing', () => {
  it('dampens a single-sample spike outlier', () => {
    const input = [150, 150, 900, 150, 150];
    const result = applyThreePointSmoothing(input);
    // f_smooth[2] = (150 + 900 + 150) / 3 = 400
    expect(result[2]).toBeCloseTo(400, 0);
    expect(result[2]).toBeLessThan(900);
  });

  it('preserves a flat signal unchanged', () => {
    const input = [200, 200, 200, 200, 200];
    const result = applyThreePointSmoothing(input);
    result.forEach((val, i) => {
      expect(val).toBeCloseTo(200, 1);
    });
  });

  it('handles array with fewer than 3 elements', () => {
    const input = [180];
    const result = applyThreePointSmoothing(input);
    expect(result).toEqual([180]);
  });

  it('handles endpoints correctly', () => {
    const input = [100, 200, 300];
    const result = applyThreePointSmoothing(input);
    // i=0: (100 + 200) / 2 = 150
    // i=1: (100 + 200 + 300) / 3 = 200
    // i=2: (200 + 300) / 2 = 250
    expect(result[0]).toBeCloseTo(150, 0);
    expect(result[1]).toBeCloseTo(200, 0);
    expect(result[2]).toBeCloseTo(250, 0);
  });
});
