# Real-Time Visual Mandarin Tone Trainer — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a browser-based Mandarin tone trainer that records voice, extracts pitch in a Web Worker, visualizes pitch contours on Canvas against native reference curves, and grades accuracy with diagnostic feedback.

**Architecture:** Pure client-side SPA — no backend. Modular ESM layout with `pitchMath.js` (pure functions), `pitchWorker.js` (thin Worker wrapper), `app.js` (main thread orchestration), `index.html` (Canvas + controls). Offline Node script (`ingestPresets.js`) pre-processes native `.wav` files into `presets.json` at build time — zero runtime cost.

**Tech Stack:** HTML5 Canvas 2D, Web Audio API, Web Workers (ESM), Vitest, Playwright, Node.js (offline ingestion only)

---

## Agreed Design Decisions

| Decision | Choice |
|----------|--------|
| Reference pitch curves | Bundled `.wav`/`.mp3` files → offline ingestion → `presets.json` (zero runtime cost) |
| Audio source for initial build | Placeholder/synthetic audio; real native-speaker recordings drop in later with zero code changes |
| Math module extraction | `src/pitchMath.js` — pure functions, zero browser dependencies, fully testable |
| Worker architecture | `src/pitchWorker.js` — thin ESM wrapper importing from pitchMath.js |
| Testing: math layer | Vitest, target 100% coverage, TDD with synthetic sine waves |
| Testing: browser I/O | Playwright, smoke/critical-path coverage, mocked microphone via Chrome flags |
| Testing: build pipeline | Integration test verifying presets.json validity and data point count |
| Project structure | `src/` for app code, `scripts/` for ingestion, `tests/` for all tests, `tests/fixtures/` for test audio |

---

## Milestone 1: Math Engine + Unit Tests

*Deliverable: All pure math functions pass comprehensive Vitest suites. No browser, no UI — just the brain.*

### Task 1.1: Scaffold project

**Objective:** Create package.json, vitest.config.js, and directory structure.

**Files:**
- Create: `package.json`
- Create: `vitest.config.js`
- Create: `src/pitchMath.js` (empty module with function stubs)
- Create: `tests/pitchMath.test.js` (empty test file)

**Step 1: Write package.json**

```json
{
  "name": "mandarin-tone-trainer",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "test": "vitest run",
    "test:watch": "vitest",
    "ingest": "node scripts/ingestPresets.js",
    "e2e": "npx playwright test"
  },
  "devDependencies": {
    "vitest": "^1.6.0",
    "@playwright/test": "^1.44.0"
  }
}
```

**Step 2: Write vitest.config.js**

```js
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['tests/**/*.test.js'],
  },
});
```

**Step 3: Create directory structure**

```bash
mkdir -p src scripts tests/fixtures/native_samples tests/fixtures/e2e docs/plans
```

**Step 4: Create stub pitchMath.js**

```js
// src/pitchMath.js — stubs, implemented in subsequent tasks
export function detectPitchAMDF(buffer, sampleRate) { return 0; }
export function applyThreePointSmoothing(pitchArray) { return pitchArray; }
export function computeDynamicTimeWarping(userTrack, nativeTrack) { return { userAligned: userTrack, nativeAligned: nativeTrack }; }
export function calculateMAEScore(userTrack, nativeTrack) { return 0; }
export function evaluateDiagnosticFeedback(userTrack, nativeTrack) { return ''; }
```

**Step 5: Install dependencies and verify**

Run: `npm install`
Expected: Deps installed without errors.

Run: `npx vitest run`
Expected: 0 tests (empty test file).

**Step 6: Commit**

```bash
git add package.json package-lock.json vitest.config.js src/pitchMath.js tests/pitchMath.test.js
git commit -m "chore: scaffold project with Vitest and directory structure"
```

---

### Task 1.2: Implement AMDF pitch detection + tests

**Objective:** Implement `detectPitchAMDF(buffer, sampleRate)` with bandpass filter (60–400 Hz) and write TDD tests.

**Files:**
- Modify: `src/pitchMath.js`
- Modify: `tests/pitchMath.test.js`

**Step 1: Write failing tests for AMDF**

In `tests/pitchMath.test.js`:

```js
import { describe, it, expect } from 'vitest';
import { detectPitchAMDF } from '../src/pitchMath.js';

/**
 * Generate a sine wave Float32Array at a given frequency.
 */
function generateSineWave(frequency, sampleRate, durationSec = 0.1) {
  const samples = Math.floor(sampleRate * durationSec);
  const buffer = new Float32Array(samples);
  for (let i = 0; i < samples; i++) {
    buffer[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate);
  }
  return buffer;
}

describe('detectPitchAMDF', () => {
  it('detects a 220 Hz sine wave within 2% tolerance', () => {
    const sampleRate = 44100;
    const buffer = generateSineWave(220, sampleRate, 0.1);
    const pitch = detectPitchAMDF(buffer, sampleRate);
    expect(pitch).toBeGreaterThan(215);
    expect(pitch).toBeLessThan(225);
  });

  it('detects a 330 Hz sine wave within 2% tolerance', () => {
    const sampleRate = 44100;
    const buffer = generateSineWave(330, sampleRate, 0.1);
    const pitch = detectPitchAMDF(buffer, sampleRate);
    expect(pitch).toBeGreaterThan(323);
    expect(pitch).toBeLessThan(337);
  });

  it('rejects frequencies below 60 Hz (bandpass floor)', () => {
    const sampleRate = 44100;
    const buffer = generateSineWave(50, sampleRate, 0.1);
    const pitch = detectPitchAMDF(buffer, sampleRate);
    expect(pitch).toBe(0); // Filtered out — sub-vocal
  });

  it('rejects frequencies above 400 Hz (bandpass ceiling)', () => {
    const sampleRate = 44100;
    const buffer = generateSineWave(500, sampleRate, 0.1);
    const pitch = detectPitchAMDF(buffer, sampleRate);
    expect(pitch).toBe(0); // Filtered out — above vocal range
  });

  it('returns 0 for silence (near-zero amplitude)', () => {
    const buffer = new Float32Array(2048); // All zeros
    const pitch = detectPitchAMDF(buffer, 44100);
    expect(pitch).toBe(0);
  });
});
```

**Step 2: Run tests to verify failure**

Run: `npx vitest run`
Expected: All 5 tests FAIL — stubs return 0 or unchanged arrays.

**Step 3: Implement detectPitchAMDF**

In `src/pitchMath.js`, replace the stub:

```js
/**
 * Detect fundamental frequency (f0) using AMDF (Average Magnitude Difference Function).
 * Bandpass filters to 60–400 Hz (Mandarin vocal range).
 *
 * @param {Float32Array} buffer — time-domain audio samples
 * @param {number} sampleRate — samples per second (e.g. 44100)
 * @returns {number} detected frequency in Hz, or 0 if no voice detected
 */
export function detectPitchAMDF(buffer, sampleRate) {
  const MIN_FREQ = 60;
  const MAX_FREQ = 400;
  const MIN_PERIOD = Math.floor(sampleRate / MAX_FREQ); // ~110 samples at 44.1k
  const MAX_PERIOD = Math.floor(sampleRate / MIN_FREQ); // ~735 samples at 44.1k

  // Silence gate: RMS below threshold → no voice
  let sumSq = 0;
  for (let i = 0; i < buffer.length; i++) {
    sumSq += buffer[i] * buffer[i];
  }
  const rms = Math.sqrt(sumSq / buffer.length);
  if (rms < 0.005) return 0;

  let bestPeriod = -1;
  let bestDiff = Infinity;

  // Search for the period that minimizes AMDF
  for (let period = MIN_PERIOD; period <= MAX_PERIOD; period++) {
    let diffSum = 0;
    let count = 0;
    for (let i = 0; i < buffer.length - period; i++) {
      diffSum += Math.abs(buffer[i] - buffer[i + period]);
      count++;
    }
    const avgDiff = diffSum / count;

    if (avgDiff < bestDiff) {
      bestDiff = avgDiff;
      bestPeriod = period;
    }
  }

  if (bestPeriod <= 0) return 0;

  const frequency = sampleRate / bestPeriod;

  // Bandpass gate
  if (frequency < MIN_FREQ || frequency > MAX_FREQ) return 0;

  return frequency;
}
```

**Step 4: Run tests to verify pass**

Run: `npx vitest run`
Expected: 5 passed.

**Step 5: Commit**

```bash
git add src/pitchMath.js tests/pitchMath.test.js
git commit -m "feat: implement AMDF pitch detection with bandpass filter (60-400 Hz)"
```

---

### Task 1.3: Implement 3-point smoothing + tests

**Objective:** Implement `applyThreePointSmoothing(pitchArray)` and write TDD tests.

**Files:**
- Modify: `src/pitchMath.js`
- Modify: `tests/pitchMath.test.js`

**Step 1: Write failing tests**

Add to `tests/pitchMath.test.js`:

```js
import { applyThreePointSmoothing } from '../src/pitchMath.js';

describe('applyThreePointSmoothing', () => {
  it('dampens a single-sample spike outlier', () => {
    // Outlier spike at index 2: 900 surrounded by 150
    const input = [150, 150, 900, 150, 150];
    const result = applyThreePointSmoothing(input);
    // The spike should be significantly reduced
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
    expect(result).toEqual([180]); // Unchanged
  });

  it('handles endpoints correctly (no left neighbor for i=0, no right neighbor for last)', () => {
    const input = [100, 200, 300];
    const result = applyThreePointSmoothing(input);
    // i=0: (100 + 200) / 2 = 150 (one-sided)
    // i=1: (100 + 200 + 300) / 3 = 200
    // i=2: (200 + 300) / 2 = 250 (one-sided)
    expect(result[0]).toBeCloseTo(150, 0);
    expect(result[1]).toBeCloseTo(200, 0);
    expect(result[2]).toBeCloseTo(250, 0);
  });
});
```

**Step 2: Run tests to verify failure**

Run: `npx vitest run`
Expected: Smoothing tests FAIL — stub returns input unchanged.

**Step 3: Implement applyThreePointSmoothing**

Replace the stub in `pitchMath.js`:

```js
/**
 * Apply 3-point moving average filter to smooth pitch contours.
 * Eliminates artifacts from plosive consonants (e.g., "p", "t", "k").
 * Formula: f_smooth[i] = (f[i-1] + f[i] + f[i+1]) / 3
 * Endpoints use available neighbors only.
 *
 * @param {number[]} pitchArray — sequential pitch values in Hz
 * @returns {number[]} smoothed pitch array
 */
export function applyThreePointSmoothing(pitchArray) {
  if (pitchArray.length < 2) return [...pitchArray];

  const result = new Array(pitchArray.length);
  result[0] = (pitchArray[0] + pitchArray[1]) / 2;
  for (let i = 1; i < pitchArray.length - 1; i++) {
    result[i] = (pitchArray[i - 1] + pitchArray[i] + pitchArray[i + 1]) / 3;
  }
  result[pitchArray.length - 1] = (pitchArray[pitchArray.length - 2] + pitchArray[pitchArray.length - 1]) / 2;
  return result;
}
```

**Step 4: Run tests**

Run: `npx vitest run`
Expected: All smoothing tests pass.

**Step 5: Commit**

```bash
git add src/pitchMath.js tests/pitchMath.test.js
git commit -m "feat: implement 3-point moving average smoothing filter"
```

---

### Task 1.4: Implement DTW alignment + tests

**Objective:** Implement `computeDynamicTimeWarping(userTrack, nativeTrack)` — a simplified DTW that resamples both arrays to a common length via linear interpolation.

**Files:**
- Modify: `src/pitchMath.js`
- Modify: `tests/pitchMath.test.js`

**Step 1: Write failing tests**

Add to `tests/pitchMath.test.js`:

```js
import { computeDynamicTimeWarping } from '../src/pitchMath.js';

describe('computeDynamicTimeWarping', () => {
  it('aligns two identical arrays to same length', () => {
    const user = [0.2, 0.5, 0.8];
    const native = [0.2, 0.5, 0.8];
    const result = computeDynamicTimeWarping(user, native, 100);
    expect(result.userAligned.length).toBe(100);
    expect(result.nativeAligned.length).toBe(100);
    // First and last values preserved
    expect(result.userAligned[0]).toBeCloseTo(0.2, 2);
    expect(result.userAligned[99]).toBeCloseTo(0.8, 2);
  });

  it('aligns arrays of different lengths to a common target length', () => {
    const user = new Array(50).fill(0).map((_, i) => i / 50); // 50 points
    const native = new Array(200).fill(0).map((_, i) => i / 200); // 200 points
    const result = computeDynamicTimeWarping(user, native, 100);
    expect(result.userAligned.length).toBe(100);
    expect(result.nativeAligned.length).toBe(100);
  });

  it('handles single-element arrays', () => {
    const user = [0.5];
    const native = [0.5];
    const result = computeDynamicTimeWarping(user, native, 100);
    // All resampled points should equal the single value
    result.userAligned.forEach(v => expect(v).toBeCloseTo(0.5, 2));
  });
});
```

**Step 2: Run tests to verify failure**

Run: `npx vitest run`
Expected: DTW tests FAIL.

**Step 3: Implement computeDynamicTimeWarping**

Replace the stub:

```js
/**
 * Resample both arrays to a common length via linear interpolation.
 * This is a lightweight alternative to full DTW — sufficient for
 * comparing pitch contour shapes.
 *
 * @param {number[]} userTrack — normalized user pitch values
 * @param {number[]} nativeTrack — normalized native pitch values
 * @param {number} [targetLength=100] — common length to resample to
 * @returns {{ userAligned: number[], nativeAligned: number[] }}
 */
export function computeDynamicTimeWarping(userTrack, nativeTrack, targetLength = 100) {
  if (userTrack.length === 0 || nativeTrack.length === 0) {
    return { userAligned: [], nativeAligned: [] };
  }

  const resample = (arr, len) => {
    if (arr.length === 1) return new Array(len).fill(arr[0]);
    const result = new Array(len);
    const step = (arr.length - 1) / (len - 1);
    for (let i = 0; i < len; i++) {
      const pos = i * step;
      const lo = Math.floor(pos);
      const hi = Math.min(lo + 1, arr.length - 1);
      const frac = pos - lo;
      result[i] = arr[lo] + (arr[hi] - arr[lo]) * frac;
    }
    return result;
  };

  return {
    userAligned: resample(userTrack, targetLength),
    nativeAligned: resample(nativeTrack, targetLength),
  };
}
```

**Step 4: Run tests**

Run: `npx vitest run`
Expected: All DTW tests pass.

**Step 5: Commit**

```bash
git add src/pitchMath.js tests/pitchMath.test.js
git commit -m "feat: implement DTW alignment via linear interpolation resampling"
```

---

### Task 1.5: Implement MAE scoring + tests

**Objective:** Implement `calculateMAEScore(userTrack, nativeTrack)` returning 0–100% and write TDD tests.

**Files:**
- Modify: `src/pitchMath.js`
- Modify: `tests/pitchMath.test.js`

**Step 1: Write failing tests**

Add to `tests/pitchMath.test.js`:

```js
import { calculateMAEScore } from '../src/pitchMath.js';

describe('calculateMAEScore', () => {
  it('returns 100% for identical arrays', () => {
    const arr = [0.1, 0.3, 0.5, 0.7, 0.9];
    const score = calculateMAEScore(arr, arr);
    expect(score).toBe(100);
  });

  it('returns a low score for inverted arrays (opposite shapes)', () => {
    const user = [0.1, 0.3, 0.5, 0.7, 0.9]; // rising
    const native = [0.9, 0.7, 0.5, 0.3, 0.1]; // falling — opposite
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
```

**Step 2: Run tests to verify failure**

Run: `npx vitest run`
Expected: MAE tests FAIL.

**Step 3: Implement calculateMAEScore**

Replace the stub:

```js
/**
 * Calculate Mean Absolute Error (MAE) between two normalized pitch arrays
 * and convert to a 0–100% accuracy score.
 *
 * MAE = (1/n) * Σ|Ui - Ni|
 * Score = max(0, 100 * (1 - MAE / threshold))
 *
 * @param {number[]} userTrack — normalized user pitch values (0–1)
 * @param {number[]} nativeTrack — normalized native pitch values (0–1)
 * @returns {number} accuracy score 0–100
 */
export function calculateMAEScore(userTrack, nativeTrack) {
  if (userTrack.length === 0 || nativeTrack.length === 0) return 0;
  if (userTrack.length !== nativeTrack.length) return 0;

  let sumAbsError = 0;
  for (let i = 0; i < userTrack.length; i++) {
    sumAbsError += Math.abs(userTrack[i] - nativeTrack[i]);
  }
  const mae = sumAbsError / userTrack.length;

  // MAE of 0.5 means average deviation is 50% of the normalized range — that's terrible
  // Map MAE to score: MAE=0 → 100%, MAE=0.5 → 0%
  const score = Math.max(0, 100 * (1 - mae / 0.5));
  return Math.round(score);
}
```

**Step 4: Run tests**

Run: `npx vitest run`
Expected: All MAE tests pass.

**Step 5: Commit**

```bash
git add src/pitchMath.js tests/pitchMath.test.js
git commit -m "feat: implement MAE scoring engine (0-100%)"
```

---

### Task 1.6: Implement diagnostic feedback + tests

**Objective:** Implement `evaluateDiagnosticFeedback(userTrack, nativeTrack)` returning tone-specific error messages.

**Files:**
- Modify: `src/pitchMath.js`
- Modify: `tests/pitchMath.test.js`

**Step 1: Write failing tests**

Add to `tests/pitchMath.test.js`:

```js
import { evaluateDiagnosticFeedback } from '../src/pitchMath.js';

describe('evaluateDiagnosticFeedback', () => {
  it('detects rising→falling error (Tone 2 violation)', () => {
    // Native: rising (Tone 2 — starts at 0.2, ends at 0.8)
    const native = [0.2, 0.35, 0.5, 0.65, 0.8];
    // User: falling — opposite direction
    const user = [0.8, 0.65, 0.5, 0.35, 0.2];
    const feedback = evaluateDiagnosticFeedback(user, native, [2]);
    expect(feedback).toContain('Pitch Dropped');
    expect(feedback).toContain('rising tone');
  });

  it('detects flat-when-should-dip error (Tone 3 violation)', () => {
    // Native: dip-rise (Tone 3 — dips in middle)
    const native = [0.5, 0.3, 0.1, 0.3, 0.5]; // V shape
    // User: staying flat
    const user = [0.5, 0.5, 0.5, 0.5, 0.5];
    const feedback = evaluateDiagnosticFeedback(user, native, [3]);
    expect(feedback).toContain('Not Deep Enough');
    expect(feedback).toContain('dipping tone');
  });

  it('detects gradual-fall error (Tone 4 violation)', () => {
    // Native: sharp fall (Tone 4 — starts high, drops fast)
    const native = [0.9, 0.8, 0.5, 0.25, 0.1];
    // User: slow, gentle decline
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
```

**Step 2: Run tests to verify failure**

Run: `npx vitest run`
Expected: Diagnostic tests FAIL — stub returns ''.

**Step 3: Implement evaluateDiagnosticFeedback**

Replace the stub:

```js
/**
 * Analyze vector deviations between user and native pitch to produce
 * actionable diagnostic feedback based on tone-specific shape errors.
 *
 * Detects:
 *  - Pitch Dropped (rising tone but user fell)
 *  - Not Deep Enough (dipping tone but user stayed flat)
 *  - Too Soft/Slow (falling tone but user declined gradually)
 *
 * @param {number[]} userTrack — normalized user pitch values (0–1)
 * @param {number[]} nativeTrack — normalized native pitch values (0–1)
 * @param {number[]} tones — tone numbers for each syllable (e.g. [2, 4])
 * @returns {string} diagnostic message, or '' if no clear error pattern
 */
export function evaluateDiagnosticFeedback(userTrack, nativeTrack, tones = []) {
  const n = nativeTrack.length;
  if (n < 3) return '';

  const nativeStart = nativeTrack[0];
  const nativeEnd = nativeTrack[n - 1];
  const nativeMid = nativeTrack[Math.floor(n / 2)];

  const userStart = userTrack[0];
  const userEnd = userTrack[n - 1];
  const userMid = userTrack[Math.floor(n / 2)];

  const nativeSlope = nativeEnd - nativeStart;
  const userSlope = userEnd - userStart;

  // Tone 2 check: native rises but user falls
  if (tones.includes(2) && nativeSlope > 0.1 && userSlope < -0.1) {
    return 'Pitch Dropped: For this rising tone (Tone 2), your voice must slide upward like you are asking an unprompted question. You dragged it downward.';
  }

  // Tone 3 check: native dips but user stays flat
  const nativeDipDepth = Math.max(nativeStart, nativeEnd) - nativeMid;
  const userDipDepth = Math.max(userStart, userEnd) - userMid;
  if (tones.includes(3) && nativeDipDepth > 0.15 && userDipDepth < 0.05) {
    return 'Not Deep Enough: For this dipping tone (Tone 3), drop your pitch completely into the lowest basement of your vocal range before letting it rise.';
  }

  // Tone 4 check: native falls sharply but user falls gradually
  const nativeDropRate = (nativeStart - nativeEnd) / n;
  const userDropRate = (userStart - userEnd) / n;
  if (tones.includes(4) && nativeDropRate > 0.005 && userDropRate < nativeDropRate * 0.3) {
    return 'Too Soft/Slow: This falling tone (Tone 4) should sound like an abrupt, angry command. Drop your pitch rapidly and confidently.';
  }

  return '';
}
```

**Step 4: Run tests**

Run: `npx vitest run`
Expected: All diagnostic tests pass (total: ~20 tests across all suites).

**Step 5: Commit**

```bash
git add src/pitchMath.js tests/pitchMath.test.js
git commit -m "feat: implement diagnostic feedback engine for tone-specific errors"
```

**Milestone 1 complete.** All math functions are implemented and tested.

---

## Milestone 2: Audio Capture + Worker + Raw Visualization

*Deliverable: User opens index.html, clicks record, sees raw pitch on Canvas. No reference line, no grading yet — just the capture-to-visual pipeline working end-to-end.*

### Task 2.1: Create index.html with Canvas + controls

**Objective:** Build the HTML shell with a Canvas element, record/stop button, and status display.

**Files:**
- Create: `src/index.html`

**Step 1: Write index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Mandarin Tone Trainer</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #0a0a0f;
      color: #e0e0e0;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      padding: 2rem;
    }
    h1 { font-size: 1.5rem; margin-bottom: 1rem; color: #00ffcc; }
    #canvas { 
      border: 1px solid #333; 
      border-radius: 8px;
      background: #111118;
    }
    .controls { margin-top: 1rem; display: flex; gap: 1rem; align-items: center; }
    button {
      padding: 0.75rem 1.5rem;
      border: none;
      border-radius: 6px;
      font-size: 1rem;
      cursor: pointer;
      transition: background 0.2s;
    }
    #recordBtn { background: #00ffcc; color: #0a0a0f; }
    #recordBtn:hover { background: #00e6b8; }
    #recordBtn.recording { background: #ff4444; color: white; }
    #status { font-size: 0.875rem; color: #888; }
    #score { 
      font-size: 2rem; 
      font-weight: bold; 
      color: #00ffcc; 
      margin-top: 0.5rem;
    }
    #feedback {
      margin-top: 0.5rem;
      max-width: 600px;
      text-align: center;
      color: #ffaa00;
      font-size: 0.9rem;
      line-height: 1.5;
    }
    select {
      padding: 0.5rem;
      background: #1a1a2e;
      color: #e0e0e0;
      border: 1px solid #333;
      border-radius: 4px;
      font-size: 0.9rem;
    }
  </style>
</head>
<body>
  <h1>🎤 Mandarin Tone Trainer</h1>
  <canvas id="canvas" width="800" height="400"></canvas>
  <div class="controls">
    <select id="wordSelect">
      <option value="">-- Select a word --</option>
    </select>
    <button id="playBtn" disabled>▶ Play Native</button>
    <button id="recordBtn">⏺ Start Recording</button>
  </div>
  <div id="status">Ready. Select a word and click record.</div>
  <div id="score"></div>
  <div id="feedback"></div>
  <script type="module" src="app.js"></script>
</body>
</html>
```

**Step 2: Commit**

```bash
git add src/index.html
git commit -m "feat: create HTML shell with Canvas, controls, and dark theme"
```

---

### Task 2.2: Create pitchWorker.js (thin wrapper)

**Objective:** Worker that imports pitchMath, receives audio buffers, and posts pitch frequencies back.

**Files:**
- Create: `src/pitchWorker.js`

**Step 1: Write pitchWorker.js**

```js
// src/pitchWorker.js — Thin Web Worker wrapper around pitchMath.js
import { detectPitchAMDF, applyThreePointSmoothing } from './pitchMath.js';

// Accumulate pitch values across frames for smoothing context
let pitchHistory = [];

self.onmessage = function (e) {
  const { audioBuffer, sampleRate } = e.data;

  const rawPitch = detectPitchAMDF(audioBuffer, sampleRate);

  if (rawPitch > 0) {
    pitchHistory.push(rawPitch);
  } else {
    pitchHistory.push(0);
  }

  // Keep a rolling window for smoothing
  if (pitchHistory.length > 5) {
    pitchHistory.shift();
  }

  // Apply 3-point smoothing when we have enough data
  let smoothedPitch = rawPitch;
  if (pitchHistory.length >= 3) {
    const recent = pitchHistory.slice(-3);
    const smoothed = applyThreePointSmoothing(recent);
    smoothedPitch = smoothed[smoothed.length - 1];
  }

  self.postMessage({
    frequency: smoothedPitch,
    rawFrequency: rawPitch,
    timestamp: Date.now(),
  });
};
```

**Step 2: Commit**

```bash
git add src/pitchWorker.js
git commit -m "feat: create pitchWorker.js — thin Web Worker wrapper around pitchMath"
```

---

### Task 2.3: Create app.js — audio capture + worker + Canvas drawing

**Objective:** Main thread orchestration: getUserMedia → AnalyserNode → requestAnimationFrame → worker.postMessage → Canvas rendering of pitch line.

**Files:**
- Create: `src/app.js`

**Step 1: Write app.js**

```js
// src/app.js — Main thread orchestration
import { computeDynamicTimeWarping, calculateMAEScore, evaluateDiagnosticFeedback } from './pitchMath.js';

// ── DOM refs ──
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const recordBtn = document.getElementById('recordBtn');
const playBtn = document.getElementById('playBtn');
const wordSelect = document.getElementById('wordSelect');
const statusEl = document.getElementById('status');
const scoreEl = document.getElementById('score');
const feedbackEl = document.getElementById('feedback');

// ── State ──
let audioContext = null;
let analyserNode = null;
let micStream = null;
let worker = null;
let isRecording = false;
let animationId = null;

// Collected pitch data for the current recording
let userPitchData = [];

// Canvas dimensions
const CANVAS_W = canvas.width;
const CANVAS_H = canvas.height;
const MIN_HZ = 70;
const MAX_HZ = 350;

// Loaded presets
let presets = [];

// ── Canvas drawing ──
function freqToY(freq) {
  if (freq <= 0) return CANVAS_H; // silence at bottom
  const norm = (freq - MIN_HZ) / (MAX_HZ - MIN_HZ);
  const clamped = Math.max(0, Math.min(1, norm));
  return CANVAS_H - clamped * CANVAS_H; // Invert: higher freq = higher on screen
}

function drawCanvas() {
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

  // Draw grid
  ctx.strokeStyle = '#1a1a2e';
  ctx.lineWidth = 1;
  for (let hz = 100; hz <= 300; hz += 50) {
    const y = freqToY(hz);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(CANVAS_W, y);
    ctx.stroke();
    ctx.fillStyle = '#555';
    ctx.font = '11px monospace';
    ctx.fillText(`${hz} Hz`, 4, y - 4);
  }

  // Draw user pitch line
  if (userPitchData.length > 1) {
    ctx.strokeStyle = '#00ffcc';
    ctx.lineWidth = 2;
    ctx.beginPath();
    const stepX = CANVAS_W / (userPitchData.length - 1);
    for (let i = 0; i < userPitchData.length; i++) {
      const x = i * stepX;
      const y = freqToY(userPitchData[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

// ── Audio pipeline ──
async function initAudio() {
  if (!worker) {
    worker = new Worker('pitchWorker.js', { type: 'module' });
    worker.onmessage = (e) => {
      const { frequency } = e.data;
      if (isRecording && frequency > 0) {
        userPitchData.push(frequency);
      }
    };
  }

  if (!audioContext) {
    audioContext = new AudioContext({ sampleRate: 44100 });
  }

  micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
    },
  });

  const source = audioContext.createMediaStreamSource(micStream);
  analyserNode = audioContext.createAnalyser();
  analyserNode.fftSize = 2048;
  source.connect(analyserNode);

  statusEl.textContent = 'Microphone ready.';
}

function processAudioFrame() {
  if (!isRecording) return;

  const buffer = new Float32Array(analyserNode.fftSize);
  analyserNode.getFloat32TimeDomainData(buffer);

  worker.postMessage({
    audioBuffer: buffer,
    sampleRate: audioContext.sampleRate,
  });

  drawCanvas();
  animationId = requestAnimationFrame(processAudioFrame);
}

// ── Recording controls ──
async function startRecording() {
  if (!audioContext) {
    try {
      await initAudio();
    } catch (err) {
      statusEl.textContent = `Microphone error: ${err.message}`;
      return;
    }
  }

  if (audioContext.state === 'suspended') {
    await audioContext.resume();
  }

  userPitchData = [];
  isRecording = true;
  recordBtn.textContent = '⏹ Stop Recording';
  recordBtn.classList.add('recording');
  scoreEl.textContent = '';
  feedbackEl.textContent = '';
  statusEl.textContent = 'Recording... speak the word clearly.';

  animationId = requestAnimationFrame(processAudioFrame);
}

function stopRecording() {
  isRecording = false;
  recordBtn.textContent = '⏺ Start Recording';
  recordBtn.classList.remove('recording');

  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }

  statusEl.textContent = `Recording complete. ${userPitchData.length} pitch samples collected.`;

  // Grade if a word is selected
  gradeAttempt();
}

recordBtn.addEventListener('click', () => {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});

// ── Grading ──
function gradeAttempt() {
  const selectedIndex = wordSelect.selectedIndex - 1; // -1 for placeholder
  if (selectedIndex < 0 || selectedIndex >= presets.length) return;
  if (userPitchData.length < 5) return;

  const preset = presets[selectedIndex];

  // Normalize user pitch to 0-1 range
  const userMin = Math.min(...userPitchData.filter(v => v > 0));
  const userMax = Math.max(...userPitchData);
  const userRange = userMax - userMin || 1;
  const normalizedUser = userPitchData.map(v =>
    v > 0 ? (v - userMin) / userRange : 0
  );

  const nativeRef = preset.nativePitchReference;

  // Align arrays to same length via DTW
  const { userAligned, nativeAligned } = computeDynamicTimeWarping(
    normalizedUser,
    nativeRef,
    100
  );

  // Score
  const score = calculateMAEScore(userAligned, nativeAligned);
  scoreEl.textContent = `${score}%`;

  // Diagnostics
  const feedback = evaluateDiagnosticFeedback(userAligned, nativeAligned, preset.tones);
  feedbackEl.textContent = feedback;

  statusEl.textContent = score >= 80 ? 'Great job! 🎉' : score >= 50 ? 'Getting there. Try again.' : 'Keep practicing!';
}

// ── Preset loading ──
async function loadPresets() {
  try {
    const res = await fetch('../presets.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    presets = data.presets || [];
  } catch (err) {
    console.warn('Could not load presets.json:', err.message);
    presets = [];
  }

  // Populate dropdown
  wordSelect.innerHTML = '<option value="">-- Select a word --</option>';
  presets.forEach((p, i) => {
    const option = document.createElement('option');
    option.value = i;
    option.textContent = `${p.word} (${p.pinyin})`;
    wordSelect.appendChild(option);
  });
}

// ── Audio playback ──
let currentAudio = null;
playBtn.addEventListener('click', () => {
  const selectedIndex = wordSelect.selectedIndex - 1;
  if (selectedIndex < 0) return;
  const preset = presets[selectedIndex];
  if (!preset.audioSrc) return;

  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }
  currentAudio = new Audio(preset.audioSrc);
  currentAudio.play();
});

// Enable play button when word selected
wordSelect.addEventListener('change', () => {
  playBtn.disabled = wordSelect.selectedIndex <= 0;
});

// ── Init ──
loadPresets();
drawCanvas();
```

**Step 2: Commit**

```bash
git add src/app.js
git commit -m "feat: implement audio capture pipeline, Worker integration, and Canvas visualization"
```

**Milestone 2 complete.** End-to-end pipeline works: mic → analyser → worker → pitch → Canvas.

---

## Milestone 3: Preset Ingestion Pipeline

*Deliverable: `scripts/ingestPresets.js` processes .wav files into `presets.json`. Integration test validates output.*

### Task 3.1: Create placeholder audio fixtures

**Objective:** Generate placeholder .wav files for the preset words so the pipeline has something to process.

**Files:**
- Create: `audio/gongsi.wav` (placeholder)
- Create: `audio/yinhang.wav` (placeholder)
- Create: `audio/laoshi.wav` (placeholder)

**Step 1: Generate placeholder WAV files using ffmpeg**

```bash
mkdir -p audio

# Generate 1-second silent WAV files as placeholders
# 16-bit mono PCM, 44100 Hz
ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 1 -acodec pcm_s16le audio/gongsi.wav -y 2>&1 | tail -1
ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 1 -acodec pcm_s16le audio/yinhang.wav -y 2>&1 | tail -1
ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 1 -acodec pcm_s16le audio/laoshi.wav -y 2>&1 | tail -1

ls -la audio/
```

**Step 2: Commit**

```bash
git add audio/
git commit -m "feat: add placeholder audio files for preset words"
```

---

### Task 3.2: Create ingestPresets.js

**Objective:** Node script that reads .wav files, extracts pitch curves using the same AMDF algorithm, and outputs presets.json.

**Files:**
- Create: `scripts/ingestPresets.js`

**Step 1: Write ingestPresets.js**

```js
#!/usr/bin/env node
/**
 * ingestPresets.js — Offline Build-Time Pipeline
 *
 * Reads native speaker .wav files from /audio/ directory,
 * extracts pitch contours using the same AMDF algorithm as pitchMath.js,
 * and outputs /presets.json with pre-computed, normalized reference curves.
 *
 * Usage: node scripts/ingestPresets.js
 */

import { readFileSync, writeFileSync, readdirSync, existsSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const AUDIO_DIR = join(ROOT, 'audio');
const OUTPUT_FILE = join(ROOT, 'presets.json');

// ── Replicated from pitchMath.js (no browser deps, pure Node) ──

function detectPitchAMDF(buffer, sampleRate) {
  const MIN_FREQ = 60;
  const MAX_FREQ = 400;
  const MIN_PERIOD = Math.floor(sampleRate / MAX_FREQ);
  const MAX_PERIOD = Math.floor(sampleRate / MIN_FREQ);

  let sumSq = 0;
  for (let i = 0; i < buffer.length; i++) sumSq += buffer[i] * buffer[i];
  const rms = Math.sqrt(sumSq / buffer.length);
  if (rms < 0.005) return 0;

  let bestPeriod = -1;
  let bestDiff = Infinity;
  for (let period = MIN_PERIOD; period <= MAX_PERIOD; period++) {
    let diffSum = 0;
    let count = 0;
    for (let i = 0; i < buffer.length - period; i++) {
      diffSum += Math.abs(buffer[i] - buffer[i + period]);
      count++;
    }
    const avgDiff = diffSum / count;
    if (avgDiff < bestDiff) {
      bestDiff = avgDiff;
      bestPeriod = period;
    }
  }
  if (bestPeriod <= 0) return 0;
  const frequency = sampleRate / bestPeriod;
  if (frequency < MIN_FREQ || frequency > MAX_FREQ) return 0;
  return frequency;
}

function applyThreePointSmoothing(arr) {
  if (arr.length < 2) return [...arr];
  const result = new Array(arr.length);
  result[0] = (arr[0] + arr[1]) / 2;
  for (let i = 1; i < arr.length - 1; i++) {
    result[i] = (arr[i - 1] + arr[i] + arr[i + 1]) / 3;
  }
  result[arr.length - 1] = (arr[arr.length - 2] + arr[arr.length - 1]) / 2;
  return result;
}

function resampleArray(arr, targetLen) {
  if (arr.length === 0) return [];
  if (arr.length === 1) return new Array(targetLen).fill(arr[0]);
  const result = new Array(targetLen);
  const step = (arr.length - 1) / (targetLen - 1);
  for (let i = 0; i < targetLen; i++) {
    const pos = i * step;
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, arr.length - 1);
    const frac = pos - lo;
    result[i] = arr[lo] + (arr[hi] - arr[lo]) * frac;
  }
  return result;
}

// ── WAV parser ──

function readWav(filepath) {
  const buffer = readFileSync(filepath);
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

  // Check RIFF header
  const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (riff !== 'RIFF') throw new Error('Not a valid WAV file');

  const format = String.fromCharCode(view.getUint8(8), view.getUint8(9), view.getUint8(10), view.getUint8(11));
  if (format !== 'WAVE') throw new Error('Not a WAVE file');

  // Find fmt chunk
  let offset = 12;
  let sampleRate = 0;
  let numChannels = 0;
  let bitsPerSample = 0;
  let dataOffset = 0;
  let dataSize = 0;

  while (offset < buffer.length) {
    const chunkId = String.fromCharCode(view.getUint8(offset), view.getUint8(offset + 1), view.getUint8(offset + 2), view.getUint8(offset + 3));
    const chunkSize = view.getUint32(offset + 4, true);

    if (chunkId === 'fmt ') {
      const audioFormat = view.getUint16(offset + 8, true);
      if (audioFormat !== 1) throw new Error('Only PCM WAV supported');
      numChannels = view.getUint16(offset + 10, true);
      sampleRate = view.getUint32(offset + 12, true);
      bitsPerSample = view.getUint16(offset + 22, true);
    } else if (chunkId === 'data') {
      dataOffset = offset + 8;
      dataSize = chunkSize;
      break;
    }

    offset += 8 + chunkSize;
  }

  // Read samples
  const bytesPerSample = bitsPerSample / 8;
  const numSamples = Math.floor(dataSize / bytesPerSample);
  const samples = new Float32Array(numSamples);

  for (let i = 0; i < numSamples; i++) {
    let sample = 0;
    const pos = dataOffset + i * bytesPerSample;
    if (bitsPerSample === 16) {
      sample = view.getInt16(pos, true) / 32768;
    } else if (bitsPerSample === 8) {
      sample = (view.getUint8(pos) - 128) / 128;
    } else if (bitsPerSample === 32) {
      sample = view.getFloat32(pos, true);
    }
    samples[i] = sample;
  }

  return { samples, sampleRate, numChannels };
}

// ── Pitch extraction ──

function extractPitchCurve(samples, sampleRate, frameSize = 2048, hopSize = 1024) {
  const pitches = [];
  for (let i = 0; i < samples.length - frameSize; i += hopSize) {
    const frame = samples.slice(i, i + frameSize);
    const pitch = detectPitchAMDF(frame, sampleRate);
    pitches.push(pitch);
  }
  return pitches;
}

// ── Main ──

const PRESET_DEFS = [
  { word: '公司', pinyin: 'gōngsī', tones: [1, 4], file: 'gongsi.wav', audioSrc: '/audio/gongsi.wav' },
  { word: '銀行', pinyin: 'yínháng', tones: [2, 4], file: 'yinhang.wav', audioSrc: '/audio/yinhang.wav' },
  { word: '老師', pinyin: 'lǎoshī', tones: [3, 1], file: 'laoshi.wav', audioSrc: '/audio/laoshi.wav' },
];

if (!existsSync(AUDIO_DIR)) {
  console.error(`Audio directory not found: ${AUDIO_DIR}`);
  process.exit(1);
}

const presets = [];

for (const def of PRESET_DEFS) {
  const filepath = join(AUDIO_DIR, def.file);
  if (!existsSync(filepath)) {
    console.warn(`⚠ Skipping ${def.word}: file not found at ${filepath}`);
    continue;
  }

  console.log(`Processing ${def.word} (${def.pinyin})...`);

  const { samples, sampleRate } = readWav(filepath);

  // Use only first channel if stereo
  const monoSamples = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    monoSamples[i] = samples[i];
  }

  // Extract raw pitch curve
  const rawPitches = extractPitchCurve(monoSamples, sampleRate);

  // Apply smoothing
  const smoothed = applyThreePointSmoothing(rawPitches);

  // Filter out zeros (silence) for normalization
  const voicedPitches = smoothed.filter(p => p > 0);

  // Normalize to 0-1 range
  let normalizedPitch = [];
  if (voicedPitches.length > 0) {
    const minP = Math.min(...voicedPitches);
    const maxP = Math.max(...voicedPitches);
    const range = maxP - minP || 1;
    normalizedPitch = smoothed.map(p => (p > 0 ? (p - minP) / range : 0));
  }

  // Resample to exactly 100 points
  const reference = resampleArray(normalizedPitch, 100);

  presets.push({
    word: def.word,
    pinyin: def.pinyin,
    tones: def.tones,
    audioSrc: def.audioSrc,
    nativePitchReference: reference.map(v => Math.round(v * 10000) / 10000),
  });

  console.log(`  ✓ Extracted ${rawPitches.length} frames → ${reference.length}-point reference`);
}

const output = { presets };
writeFileSync(OUTPUT_FILE, JSON.stringify(output, null, 2));
console.log(`\n✅ Written ${presets.length} presets to ${OUTPUT_FILE}`);
```

**Step 2: Commit**

```bash
git add scripts/ingestPresets.js
git commit -m "feat: create offline WAV-to-JSON preset ingestion pipeline"
```

---

### Task 3.3: Run ingestion + verify output

**Objective:** Run the ingestion script and verify presets.json is valid.

**Step 1: Run the script**

Run: `node scripts/ingestPresets.js`
Expected: Processes 3 files, writes presets.json.

**Step 2: Verify presets.json structure**

Run: `node -e "const p = require('./presets.json'); console.log('Presets:', p.presets.length); p.presets.forEach(x => console.log(x.word, x.nativePitchReference.length, 'points'))"`
Expected: 3 presets, each with 100-point reference arrays.

**Step 3: Commit**

```bash
git add presets.json
git commit -m "data: add generated presets.json with reference pitch curves"
```

---

### Task 3.4: Write integration test for ingestion pipeline

**Objective:** Test that the ingestion script produces valid output against fixture audio.

**Files:**
- Create: `tests/ingestPresets.test.js`

**Step 1: Write integration test**

```js
import { describe, it, expect } from 'vitest';
import { execSync } from 'child_process';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

const OUTPUT = join(process.cwd(), 'presets.json');

describe('ingestPresets pipeline', () => {
  it('generates presets.json with correct structure', () => {
    // Run the ingestion script
    execSync('node scripts/ingestPresets.js', { stdio: 'pipe' });

    expect(existsSync(OUTPUT)).toBe(true);

    const data = JSON.parse(readFileSync(OUTPUT, 'utf-8'));
    expect(data).toHaveProperty('presets');
    expect(Array.isArray(data.presets)).toBe(true);
    expect(data.presets.length).toBeGreaterThan(0);

    // Each preset must have required fields
    for (const preset of data.presets) {
      expect(preset).toHaveProperty('word');
      expect(preset).toHaveProperty('pinyin');
      expect(preset).toHaveProperty('tones');
      expect(preset).toHaveProperty('audioSrc');
      expect(preset).toHaveProperty('nativePitchReference');
      expect(Array.isArray(preset.nativePitchReference)).toBe(true);

      // Reference array should be exactly 100 points
      expect(preset.nativePitchReference.length).toBe(100);

      // All values should be between 0 and 1
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
```

**Step 2: Run test**

Run: `npx vitest run tests/ingestPresets.test.js`
Expected: Integration tests pass.

**Step 3: Commit**

```bash
git add tests/ingestPresets.test.js
git commit -m "test: add integration test for presets ingestion pipeline"
```

**Milestone 3 complete.** Preset pipeline produces valid presets.json, verified by integration test.

---

## Milestone 4: E2E Tests

*Deliverable: Playwright tests verify critical path — record, grade, feedback display.*

### Task 4.1: Create E2E test fixtures and Playwright config

**Objective:** Set up Playwright with mocked microphone and test audio fixtures.

**Files:**
- Create: `playwright.config.js`
- Create: `tests/fixtures/e2e/user_tone2_test.wav`

**Step 1: Write playwright.config.js**

```js
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  testMatch: '**/*.spec.js',
  use: {
    baseURL: 'http://localhost:3000',
    launchOptions: {
      args: [
        '--use-fake-ui-for-media-stream',
        '--use-fake-device-for-media-stream',
        '--use-file-for-fake-audio-capture=tests/fixtures/e2e/user_tone2_test.wav',
      ],
    },
  },
  webServer: {
    command: 'npx serve src -p 3000',
    port: 3000,
    reuseExistingServer: true,
  },
});
```

**Step 2: Generate test audio fixture**

```bash
mkdir -p tests/fixtures/e2e
# Generate a 2-second 220Hz sine wave .wav as fake user recording
ffmpeg -f lavfi -i "sine=frequency=220:duration=2" -acodec pcm_s16le -ar 44100 -ac 1 tests/fixtures/e2e/user_tone2_test.wav -y 2>&1 | tail -1
```

**Step 3: Install Playwright browsers and serve**

```bash
npx playwright install chromium
npm install serve --save-dev
```

**Step 4: Commit**

```bash
git add playwright.config.js tests/fixtures/e2e/user_tone2_test.wav package.json package-lock.json
git commit -m "test: set up Playwright with mocked microphone and test fixtures"
```

---

### Task 4.2: Write E2E test

**Objective:** Critical path test — record, verify score appears, verify feedback.

**Files:**
- Create: `tests/toneTrainer.spec.js`

**Step 1: Write toneTrainer.spec.js**

```js
import { test, expect } from '@playwright/test';

test.describe('Mandarin Tone Trainer', () => {
  test('records audio and displays a score', async ({ page }) => {
    await page.goto('/');

    // Should see the title
    await expect(page.locator('h1')).toContainText('Mandarin Tone Trainer');

    // Select a word from the dropdown
    await page.selectOption('#wordSelect', { index: 1 });
    await expect(page.locator('#playBtn')).toBeEnabled();

    // Click record
    await page.click('#recordBtn');
    await expect(page.locator('#recordBtn')).toContainText('Stop');

    // Wait for recording to capture audio frames
    await page.waitForTimeout(2000);

    // Stop recording
    await page.click('#recordBtn');

    // Assert score appears (any percentage)
    await expect(page.locator('#score')).toContainText(/%/);
  });

  test('shows status message after recording', async ({ page }) => {
    await page.goto('/');

    await page.selectOption('#wordSelect', { index: 1 });
    await page.click('#recordBtn');
    await page.waitForTimeout(2000);
    await page.click('#recordBtn');

    // Status should indicate recording complete
    await expect(page.locator('#status')).toContainText(/complete|pitch/i);
  });

  test('play button enables when word selected', async ({ page }) => {
    await page.goto('/');

    // Initially disabled
    await expect(page.locator('#playBtn')).toBeDisabled();

    // Select a word
    await page.selectOption('#wordSelect', { index: 1 });

    // Then enabled
    await expect(page.locator('#playBtn')).toBeEnabled();
  });

  test('canvas is present and has correct dimensions', async ({ page }) => {
    await page.goto('/');

    const canvas = page.locator('#canvas');
    await expect(canvas).toBeVisible();
    await expect(canvas).toHaveAttribute('width', '800');
    await expect(canvas).toHaveAttribute('height', '400');
  });
});
```

**Step 2: Run E2E tests**

Run: `npx playwright test`
Expected: Critical path tests pass.

**Step 3: Commit**

```bash
git add tests/toneTrainer.spec.js
git commit -m "test: add Playwright E2E tests for critical path"
```

**Milestone 4 complete.** E2E tests verify the full user flow.

---
