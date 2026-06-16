# AUTORESEARCH LOG — Final Session Summary
# Project: interactive-mandarin-tone-trainer
# Target: MAE <= 0.15, Val Gap <= 5%, Human Align = 100%

## Iteration History (Full)

| # | ZCR | Silence | Periodicity | Smooth | Octave | Spline | Autocorr | Frame | Train MAE | All MAE | Val Gap | Human | Note |
|---|-----|---------|-------------|--------|--------|--------|----------|-------|-----------|---------|---------|-------|------|
| 0 | — | 0.005 | 0.4×RMS | 3pt MA | ✗ | ✗ | ✗ | 2048 | 0.3031 | 0.3112 | 9.1% | 100% | BASELINE |
| 1 | 0.18 | 0.005 | 0.4×RMS | 3pt MA | ✗ | ✗ | ✗ | 2048 | 0.3031 | 0.3112 | 9.1% | 100% | ZCR too high, no effect |
| 2 | 0.12 | 0.003 | 0.6×RMS | 3pt MA | ✓ | ✗ | ✗ | 2048 | 0.3307 | 0.3474 | 17.3% | 100% | REGRESSION: loose periodicity |
| 3 | 0.15 | 0.003 | 0.4×RMS | 3pt MA | ✓ | ✗ | ✗ | 2048 | 0.3036 | 0.3115 | 8.9% | 100% | Recovered, octave ON |
| 4 | 0.15 | 0.003 | 0.4×RMS | 3pt MA | ✓ | gap=8 | ✗ | 2048 | 0.2986 | 0.3091 | 12.1% | 100% | +spline, marginal improvement |
| 5 | 0.15 | 0.003 | 0.4×RMS | 3pt MA | ✓ | gap=8 | th=0.15 | 2048 | 0.6972 | 0.7062 | 4.4% | 27% | CATASTROPHE: blind autocorr |
| 6 | 0.15 | 0.003 | dip | 3pt MA | ✓ | gap=8 | ✗ | 2048 | 0.6472 | 0.6432 | 2.1% | 73% | REGRESSION: dip prominence |
| 7 | 0.15 | 0.003 | 0.4×RMS | 3pt MA | ✓ | gap=8 | ✗ | 4096 | 0.3017 | 0.3095 | 8.9% | 100% | Larger window, no effect |
| 8 | 0.15 | 0.003 | 0.4×RMS | 3pt MA | ✓ | gap=8 | ✗ | 2048 | 0.4533 | 0.4637 | 7.9% | 100% | REGRESSION: pre-emphasis α=0.97 |
| 9 | 0.15 | 0.003 | 0.4×RMS | **5pt MED** | ✓ | gap=8 | ✗ | 2048 | **0.2711** | **0.2829** | 14.8% | 100% | **BEST: 5pt median smoothing** |
|10 | 0.15 | 0.003 | 0.4×RMS | 5pt MED | ✓ | **gap=16** | ✗ | 2048 | 0.2658 | 0.2791 | 17.1% | 100% | Wider spline, jinnian→97% |
|11 | 0.15 | 0.003 | 0.4×RMS | 5pt MED | ✓ | gap=16 | ✗ | 2048 | 0.2686 | 0.2811 | 15.9% | 100% | Fricative trimmer, no help |
|12 | 0.15 | 0.003 | 0.4×RMS | 5pt MED | ✓ | gap=16 | th=0.4 | 2048 | 0.3827 | 0.4190 | 32.5% | 100% | REGRESSION: strict autocorr |
|13 | 0.15 | **0.002** | 0.4×RMS | 5pt MED | ✓ | gap=8 | ✗ | 2048 | 0.2711 | 0.2829 | 14.8% | 100% | Quieter silence, no change |

## Best Configuration (Iteration 9/10)

```
detectPitchAMDF:   ZCR=0.15, silence=0.003, periodicity=0.4×RMS
applyOctaveCorrection:  3-frame median, ε=5% or 5Hz
applySplineInterpolation:  linear gap fill ≤8 frames
applyThreePointSmoothing:  5-point median (NOT 3-point MA)
AMDF window:  2048 samples @ 44.1kHz (46ms)
```

### Aggregate: MAE 0.283 | Train 0.271 | Val 0.311 | Gap 14.8% | Human 100%

## Per-Word Breakdown (Best Config)

### Top Performers (83-97%)
| Word | Score | MAE | Tones |
|------|-------|-----|-------|
| jinnian | 97% | 0.058 | 1+2 |
| jinzhang | 96% | 0.081 | 3+1 |
| wenti | 96% | 0.084 | 4+2 |
| hanyu | 95% | 0.101 | 4+3 |
| gege | 95% | 0.099 | 1+5 |
| zaijian | 95% | 0.102 | 4+4 |
| diannao | 94% | 0.124 | 4+3 |
| gongsi | 93% | 0.146 | 1+1 |
| yinhang | 93% | 0.133 | 2+2 |
| ma4 | 92% | 0.152 | 4 |
| pingguo | 91% | 0.177 | 2+3 |
| jueding | 89% | 0.217 | 2+4 |
| laoshi | 86% | 0.288 | 3+1 |
| jintian | 84% | 0.313 | 1+1 |
| ma1 | 83% | 0.344 | 1 |
| mingnian | 83% | 0.335 | 2+2 |

### Fricative Wall (65-74%) — THE BOTTLENECK
| Word | Score | MAE | Initial | Note |
|------|-------|-----|---------|------|
| shuiguo | 74% | 0.518 | sh- | |
| xiexie | 72% | 0.552 | x- | |
| luxing | 71% | 0.586 | l- | (liquid, not fricative) |
| jiejie | 70% | 0.601 | j- | (affricate) |
| haizi | 65% | 0.695 | h- | (glottal fricative) |
| ma2 | 79% | 0.416 | m- | (nasal — unexpected) |
| ma3 | 79% | 0.426 | m- | (nasal — unexpected) |

## Root Cause: The Fricative Wall

6 words score <80% because AMDF finds too few voiced pitch frames.
The ground truth (pYIN/autocorr) finds dense pitch curves (up to 97% voiced),
but AMDF rejects most frames due to:

1. **Fricative noise** — unvoiced onset dominates early frames, periodicity gate rejects them
2. **Low SNR** — these recordings have lower signal-to-noise, so bestDiff/rms ratio is unfavorable
3. **Algorithmic mismatch** — the 6 autocorrelation-fallback ground truth words use a different
   algorithm than our AMDF, creating systematic bias

## What Worked
- **5-point median smoothing** — ONLY change with real impact (-9% MAE)
  - Eliminates spikes completely (not just dampens)
  - jintian +10%, laoshi +8%, jinnian +5%
- **Spline interpolation** — marginal improvement by filling short gaps
- **ZCR gate at 0.15** — safely rejects fricative noise without false positives

## What Didn't Work
- Autocorrelation fallback (any threshold) — catastrophic regression
- Pre-emphasis filter — hurts clean signals more than it helps fricatives
- Dip prominence gate — wrong threshold
- Larger AMDF window (4096) — no effect
- Fricative onset trimming — no effect
- Quieter silence gate (0.002) — no effect

## Path to MAE ≤ 0.15

The remaining 2× gap requires addressing the fricative wall. Options:

1. **Better ground truth** — Re-extract with tuned pYIN params per word to reduce AMDF/pYIN mismatch
2. **Hybrid detection with per-frame confidence** — Use both AMDF and autocorrelation, weight by confidence score
3. **Voiced-only comparison** — Only compute MAE on frames where BOTH curves have valid pitch (exclude zeros)
4. **Spectral-domain pitch detection** — FFT-based cepstrum or harmonic product spectrum, more robust to noise
5. **Recording quality** — Re-record fricative-initial words with better mic placement (pop filter, closer to mouth)
