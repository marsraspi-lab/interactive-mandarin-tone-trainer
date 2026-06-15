import { describe, it, expect } from 'vitest';
import { execSync } from 'child_process';
import { readFileSync, existsSync } from 'fs';

const OUTPUT = 'presets.json'; // relative to cwd

describe('ingestPresets pipeline', () => {
  it('generates presets.json with correct structure', () => {
    execSync('node scripts/ingestPresets.js', { stdio: 'pipe' });
    expect(existsSync(OUTPUT)).toBe(true);
    const data = JSON.parse(readFileSync(OUTPUT, 'utf-8'));
    expect(data).toHaveProperty('presets');
    expect(Array.isArray(data.presets)).toBe(true);
    expect(data.presets.length).toBeGreaterThan(0);
    for (const preset of data.presets) {
      expect(preset).toHaveProperty('word');
      expect(preset).toHaveProperty('pinyin');
      expect(preset).toHaveProperty('tones');
      expect(preset).toHaveProperty('audioSrc');
      expect(preset).toHaveProperty('nativePitchReference');
      expect(Array.isArray(preset.nativePitchReference)).toBe(true);
      expect(preset.nativePitchReference.length).toBe(100);
      for (const v of preset.nativePitchReference) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
      }
    }
  });

  it('preset words match expected vocabulary', () => {
    execSync('node scripts/ingestPresets.js', { stdio: 'pipe' });
    const data = JSON.parse(readFileSync(OUTPUT, 'utf-8'));
    const words = data.presets.map(p => p.word);
    expect(words).toContain('公司');
    expect(words).toContain('銀行');
    expect(words).toContain('老師');
  });
});
