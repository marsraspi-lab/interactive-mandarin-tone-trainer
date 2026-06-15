import { describe, it, expect } from 'vitest';
import { processAudioFrame } from '../src/pitchWorker.js';
import { generateSineWave } from './helpers.js';

describe('pitchWorker processAudioFrame', () => {
  it('detects pitch from a sine wave and returns raw frequency', () => {
    const buffer = generateSineWave(220.5, 44100, 0.1);
    const history = [];
    const result = processAudioFrame(buffer, 44100, history);

    expect(result.rawFrequency).toBeGreaterThan(216);
    expect(result.rawFrequency).toBeLessThan(225);
    // No smoothing with < 3 entries: frequency === rawFrequency
    expect(result.frequency).toBe(result.rawFrequency);
    expect(result.pitchHistory.length).toBe(1);
  });

  it('maintains pitchHistory with max 5 entries', () => {
    const history = [100, 200, 300, 400, 500];
    const buffer = generateSineWave(220.5, 44100, 0.1);
    const result = processAudioFrame(buffer, 44100, history);

    expect(result.pitchHistory.length).toBe(5);
    // Oldest entry (100) should be shifted out
    expect(result.pitchHistory[0]).toBe(200);
  });

  it('applies 3-point smoothing when history has >= 3 entries', () => {
    const history = [200, 200, 900]; // anomalous spike at end
    const buffer = generateSineWave(150, 44100, 0.1);
    const result = processAudioFrame(buffer, 44100, history);

    // Smoothed frequency should differ from raw (spike is dampened)
    expect(result.frequency).not.toBe(result.rawFrequency);
    expect(result.frequency).toBeLessThan(900);
  });

  it('returns 0 for silence (no smoothing applied)', () => {
    const buffer = new Float32Array(2048);
    const history = [];
    const result = processAudioFrame(buffer, 44100, history);

    expect(result.rawFrequency).toBe(0);
    expect(result.frequency).toBe(0);
    expect(result.pitchHistory.length).toBe(1);
    expect(result.pitchHistory[0]).toBe(0);
  });

  it('does not apply smoothing with fewer than 3 history entries', () => {
    const history = [200];
    const buffer = generateSineWave(220.5, 44100, 0.1);
    const result = processAudioFrame(buffer, 44100, history);

    expect(result.frequency).toBe(result.rawFrequency);
    expect(result.pitchHistory.length).toBe(2);
  });
});
