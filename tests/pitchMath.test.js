import { describe, it, expect } from 'vitest';
import { detectPitchAMDF, applyThreePointSmoothing, computeDynamicTimeWarping, calculateMAEScore, evaluateDiagnosticFeedback, normalizeZScore, clampValues } from '../src/pitchMath.js';
import { generateSineWave, generateWhiteNoise } from './helpers.js';

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
  it('eliminates a single-sample spike (median is outlier-immune)', () => {
    const input = [150, 150, 900, 150, 150];
    const result = applyThreePointSmoothing(input);
    // 5-point median: window around spike is [150,150,900,150,150]
    // sorted → [150,150,150,150,900], median = 150 (spike erased)
    expect(result[2]).toBeCloseTo(150, 0);
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

  it('handles short arrays with full-window median', () => {
    const input = [100, 200, 300];
    const result = applyThreePointSmoothing(input);
    // 3-element array: 5-neighbor window at each position
    // clips to array bounds → uses all [100,200,300]
    // sorted → [100,200,300], median = 200 for all positions
    expect(result[0]).toBeCloseTo(200, 0);
    expect(result[1]).toBeCloseTo(200, 0);
    expect(result[2]).toBeCloseTo(200, 0);
  });
});

describe('normalizeZScore', () => {
  it('centers at μ=0 and scales to σ=1 for simple data', () => {
    // Values: 100, 200, 300 → μ=200, σ≈81.65
    const input = [100, 200, 300];
    const result = normalizeZScore(input);
    // z[0] = (100-200)/81.65 ≈ -1.22
    // z[1] = (200-200)/81.65 = 0
    // z[2] = (300-200)/81.65 ≈ 1.22
    expect(result[1]).toBeCloseTo(0, 1);
    expect(result[0]).toBeLessThan(0);
    expect(result[2]).toBeGreaterThan(0);
    // Symmetry: z[0] ≈ -z[2]
    expect(Math.abs(result[0] + result[2])).toBeLessThan(0.1);
  });

  it('returns all zeros for fewer than 2 voiced values', () => {
    expect(normalizeZScore([0, 0, 150, 0])).toEqual([0, 0, 0, 0]);
  });

  it('preserves zeros (silence) as zeros', () => {
    const input = [0, 150, 0, 250, 0];
    const result = normalizeZScore(input);
    expect(result[0]).toBe(0);
    expect(result[2]).toBe(0);
    expect(result[4]).toBe(0);
    // Voiced values should be non-zero
    expect(result[1]).not.toBe(0);
    expect(result[3]).not.toBe(0);
  });

  it('returns all zeros for flat (zero-variance) voiced signal', () => {
    const input = [200, 200, 200, 200];
    const result = normalizeZScore(input);
    result.forEach(v => expect(v).toBe(0));
  });
});

describe('clampValues', () => {
  it('clamps values exceeding the range', () => {
    const input = [-5, -1, 0, 0.5, 4];
    const result = clampValues(input, -2, 2);
    expect(result).toEqual([-2, -1, 0, 0.5, 2]);
  });

  it('leaves zeros untouched and clamps outliers', () => {
    const input = [0, -3.5, 0, 5, 0];
    const result = clampValues(input, -3, 3);
    // Zeros untouched
    expect(result.filter(v => v === 0).length).toBe(3);
    // Outliers clamped
    expect(result[1]).toBe(-3);
    expect(result[3]).toBe(3);
  });

  it('preserves values within range unchanged', () => {
    const input = [-1.5, -0.5, 0, 0.3, 1.8];
    const result = clampValues(input, -2, 2);
    expect(result).toEqual([-1.5, -0.5, 0, 0.3, 1.8]);
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
    const arr = [-1.0, -0.5, 0.0, 0.5, 1.0];
    const score = calculateMAEScore(arr, arr);
    expect(score).toBe(100);
  });

  it('returns a low score for inverted arrays (Z-score range)', () => {
    const user = [-1.5, -0.75, 0.0, 0.75, 1.5];
    const native = [1.5, 0.75, 0.0, -0.75, -1.5];
    const score = calculateMAEScore(user, native);
    expect(score).toBeLessThan(50);
  });

  it('returns a score between 0 and 100', () => {
    const user = [-0.5, 0.0, 0.5, 1.0];
    const native = [0.0, 0.5, 0.0, 1.5];
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
  it('detects rising→falling error (Tone 2 violation, Z-score range)', () => {
    // Native rises: -1 → 1, slope = 2 > 0.4
    const native = [-1.0, -0.5, 0.0, 0.5, 1.0];
    // User falls: 1 → -1, slope = -2 < -0.4
    const user = [1.0, 0.5, 0.0, -0.5, -1.0];
    const feedback = evaluateDiagnosticFeedback(user, native, [2]);
    expect(feedback).toContain('Pitch Dropped');
    expect(feedback).toContain('rising tone');
  });

  it('detects flat-when-should-dip error (Tone 3 violation, Z-score range)', () => {
    // Native dips: 1 → -0.5 → 1, dip depth = 1.5 > 0.6
    const native = [1.0, 0.5, -0.5, 0.5, 1.0];
    // User flat: all 0, dip depth = 0 < 0.2
    const user = [0.0, 0.0, 0.0, 0.0, 0.0];
    const feedback = evaluateDiagnosticFeedback(user, native, [3]);
    expect(feedback).toContain('Not Deep Enough');
    expect(feedback).toContain('dipping tone');
  });

  it('detects gradual-fall error (Tone 4 violation, Z-score range)', () => {
    // Native falls sharply: 1.5 → -1.5, drop rate = 3.0/5 = 0.6 > 0.02
    const native = [1.5, 1.0, 0.5, -0.5, -1.5];
    // User falls gradually: 1.5 → 1.3, drop rate = 0.2/5 = 0.04 < 0.6*0.3
    const user = [1.5, 1.45, 1.4, 1.35, 1.3];
    const feedback = evaluateDiagnosticFeedback(user, native, [4]);
    expect(feedback).toContain('Too Soft/Slow');
    expect(feedback).toContain('falling tone');
  });

  it('returns empty string when shape matches well', () => {
    const arr = [-1.0, -0.5, 0.0, 0.5, 1.0];
    const feedback = evaluateDiagnosticFeedback(arr, arr, [1]);
    expect(feedback).toBe('');
  });
});
