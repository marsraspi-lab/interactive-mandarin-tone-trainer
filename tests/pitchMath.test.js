import { describe, it, expect } from 'vitest';
import { detectPitchAMDF, applyThreePointSmoothing, computeDynamicTimeWarping, calculateMAEScore, evaluateDiagnosticFeedback } from '../src/pitchMath.js';

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

describe('computeDynamicTimeWarping', () => {
  it('aligns two identical arrays to same length', () => {
    const user = [0.2, 0.5, 0.8];
    const native = [0.2, 0.5, 0.8];
    const result = computeDynamicTimeWarping(user, native, 100);
    expect(result.userAligned.length).toBe(100);
    expect(result.nativeAligned.length).toBe(100);
    expect(result.userAligned[0]).toBeCloseTo(0.2, 2);
    expect(result.userAligned[99]).toBeCloseTo(0.8, 2);
  });

  it('aligns arrays of different lengths to a common target length', () => {
    const user = new Array(50).fill(0).map((_, i) => i / 50);
    const native = new Array(200).fill(0).map((_, i) => i / 200);
    const result = computeDynamicTimeWarping(user, native, 100);
    expect(result.userAligned.length).toBe(100);
    expect(result.nativeAligned.length).toBe(100);
  });

  it('handles single-element arrays', () => {
    const user = [0.5];
    const native = [0.5];
    const result = computeDynamicTimeWarping(user, native, 100);
    result.userAligned.forEach(v => expect(v).toBeCloseTo(0.5, 2));
  });
});

describe('calculateMAEScore', () => {
  it('returns 100% for identical arrays', () => {
    const arr = [0.1, 0.3, 0.5, 0.7, 0.9];
    const score = calculateMAEScore(arr, arr);
    expect(score).toBe(100);
  });

  it('returns a low score for inverted arrays', () => {
    const user = [0.1, 0.3, 0.5, 0.7, 0.9];
    const native = [0.9, 0.7, 0.5, 0.3, 0.1];
    const score = calculateMAEScore(user, native);
    expect(score).toBeLessThan(50);
  });

  it('returns a score between 0 and 100', () => {
    const user = [0.2, 0.4, 0.6, 0.8];
    const native = [0.3, 0.5, 0.4, 0.9];
    const score = calculateMAEScore(user, native);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(100);
  });

  it('handles empty arrays', () => {
    const score = calculateMAEScore([], []);
    expect(score).toBe(0);
  });
});

describe('evaluateDiagnosticFeedback', () => {
  it('detects rising→falling error (Tone 2 violation)', () => {
    const native = [0.2, 0.35, 0.5, 0.65, 0.8];
    const user = [0.8, 0.65, 0.5, 0.35, 0.2];
    const feedback = evaluateDiagnosticFeedback(user, native, [2]);
    expect(feedback).toContain('Pitch Dropped');
    expect(feedback).toContain('rising tone');
  });

  it('detects flat-when-should-dip error (Tone 3 violation)', () => {
    const native = [0.5, 0.3, 0.1, 0.3, 0.5];
    const user = [0.5, 0.5, 0.5, 0.5, 0.5];
    const feedback = evaluateDiagnosticFeedback(user, native, [3]);
    expect(feedback).toContain('Not Deep Enough');
    expect(feedback).toContain('dipping tone');
  });

  it('detects gradual-fall error (Tone 4 violation)', () => {
    const native = [0.9, 0.8, 0.5, 0.25, 0.1];
    const user = [0.9, 0.88, 0.85, 0.82, 0.8];
    const feedback = evaluateDiagnosticFeedback(user, native, [4]);
    expect(feedback).toContain('Too Soft/Slow');
    expect(feedback).toContain('falling tone');
  });

  it('returns empty string when shape matches well', () => {
    const arr = [0.1, 0.3, 0.5, 0.7, 0.9];
    const feedback = evaluateDiagnosticFeedback(arr, arr, [1]);
    expect(feedback).toBe('');
  });
});
