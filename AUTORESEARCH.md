# AUTORESEARCH — Pitch Detection Optimisation Loop

A self-improving research harness that lets an AI agent experiment with the
pitch detection pipeline in `src/pitchMath.js` and automatically evaluate
results against a high-quality reference. Inspired by Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch).

## Architecture

```
src/pitchMath.js          ← AGENT EDITS THIS (AMDF, smoothing, scoring, ...)
scripts/prepare.js        ← ONE-TIME: extract ground-truth pitch curves
scripts/train.js          ← RUNNER: evaluates pitchMath against ground truth
data/ground-truth.json    ← REFERENCE: pYIN pitch curves for all 24 words
AUTORESEARCH.md           ← THIS FILE (human edits the instructions)
program.md                ← AGENT INSTRUCTIONS (human edits the prompt)
```

| Karpathy          | This project                                      |
|-------------------|---------------------------------------------------|
| `prepare.py`      | `scripts/prepare.js` — extract ground truth once  |
| `train.py`        | `src/pitchMath.js` — agent modifies this          |
| `program.md`      | `program.md` — agent prompt / research org        |
| `val_bpb`         | aggregate MAE vs ground truth across 24 files     |
| 5-minute budget   | ~12 minutes (24 files × ~30s each)                |

## The metric

**Aggregate pitch error** across all 24 native-speaker recordings. For each
word the runner:

1. Loads the WAV/MP3, extracts a pitch curve using the current `pitchMath.js`
2. Z-score-normalises both the detected curve and the ground-truth curve
3. Resamples both to 100 points
4. Computes MAE (Mean Absolute Error)
5. Computes voiced-frame detection rate (% of frames where pitch was found)

The single aggregate score rewards the agent for both *accurate frequency*
and *fewer missed voiced frames*.

## Ground truth — Path A: pYIN (recommended)

A one-time Python script (`scripts/prepare_pyin.py`) processes the 24 MP3
files through [pYIN](https://librosa.org/doc/main/generated/librosa.pyin.html)
(via librosa), the state-of-the-art probabilistic pitch tracker. Output is
`data/ground-truth.json`:

```json
{
  "妈":   { "pinyin": "mā",   "tones": [1],    "pitch": [200, 200, 200, ...] },
  "公司": { "pinyin": "gōngsī", "tones": [1, 1], "pitch": [200, 200, 190, ...] },
  ...
}
```

**Why pYIN:** It handles subharmonic errors, unvoiced regions, and pitch
halving/doubling far better than AMDF alone. Running it once gives us a
high-quality target that the JS pipeline tries to match.

**Dependencies:** Python 3, librosa, ffmpeg (for MP3→WAV conversion).

## Ground truth — Path B: consensus (pure Node, fallback)

Run 3 different pitch detectors (AMDF, autocorrelation, cepstrum) on each
file. Where ≥2 agree (within 5 Hz), use the median as ground truth. Flag
disagreements for human review.

Lower accuracy than pYIN but zero Python dependency.

## The loop

```
1. Agent reads program.md + AUTORESEARCH.md
2. Agent edits src/pitchMath.js
3. Agent runs:  node scripts/train.js
4. Agent sees score (MAE + voiced rate) — better or worse?
5. If better → keep. If worse → revert.
6. Repeat until budget exhausted or score plateaus
```

## What the agent can experiment with

| Category           | Knobs                                                           |
|--------------------|-----------------------------------------------------------------|
| Pitch detection    | AMDF window size, hop size, bandpass range (60–400 Hz), silence RMS gate, periodicity gate ratio, epsilon tiebreaker |
| Algorithm choice   | Swap AMDF for autocorrelation, cepstrum, or hybrid approaches   |
| Pre-processing     | Pre-emphasis filter, noise gate, DC offset removal              |
| Smoothing          | 3-point → 5-point, median filter, Kalman filter, spline interpolation |
| Post-processing    | Octave-jump correction (rolling median cache from BACKLOG), fricative eraser (ZCR from BACKLOG) |
| Normalisation      | Z-score vs min-max, clamping range, per-syllable normalisation   |
| Scoring            | MAE threshold, DTW vs linear resample, per-segment weighting     |

## Human's role

Edit **`program.md`** — the prompt the agent receives. Start simple, then
refine as you observe what works:

```markdown
# program.md (example start)

You are optimising the pitch detection pipeline in src/pitchMath.js for
Mandarin tone recognition.

## Setup
1. Run `node scripts/train.js` to see the current aggregate score
2. Read the BACKLOG.md for ideas that were deferred
3. Make ONE change at a time to src/pitchMath.js
4. Run `node scripts/train.js` again
5. If score improved → keep. If not → revert and try something else

## Rules
- Do NOT modify scripts/train.js or data/ground-truth.json
- Focus on detectPitchAMDF first — it's the bottleneck
- Each experiment must complete within ~15 minutes
- Log every attempt: what you changed, why, and the score delta
```

## Implementation order

1. **`scripts/prepare.js`** (or `scripts/prepare_pyin.py`) — extract ground truth
2. **`scripts/train.js`** — evaluation runner
3. **`program.md`** — agent instructions
4. **First run** — let the agent loose, review results, refine program.md
