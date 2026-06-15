import { describe, it, expect } from 'vitest';
import { execSync } from 'child_process';
import { readFileSync, existsSync } from 'fs';

const OUTPUT = 'presets.json'; // project root (canonical location)

describe('ingestPresets pipeline', () => {
  it('generates presets.json with correct Z-score normalized structure', () => {
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
        // Z-score normalized + clamped to [-3, +3]; zeros for silence
        expect(v).toBeGreaterThanOrEqual(-3);
        expect(v).toBeLessThanOrEqual(3);
      }
    }
  });

  it('preset words match expected vocabulary', () => {
    execSync('node scripts/ingestPresets.js', { stdio: 'pipe' });
    const data = JSON.parse(readFileSync(OUTPUT, 'utf-8'));
    const words = data.presets.map(p => p.word);
    expect(words).toContain('音乐');
    expect(words).toContain('明天');
    expect(words).toContain('老师');
    expect(words).toContain('妈');
    expect(words).toContain('麻');
    expect(words).toContain('马');
    expect(words).toContain('骂');
  });
});
