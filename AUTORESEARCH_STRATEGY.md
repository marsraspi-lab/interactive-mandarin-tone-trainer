Here is a comprehensive, production-ready engineering guide for your repository. It defines exactly how the automated research loop will ingest, train, evaluate, and prevent regression across your three distinct data tiers using Python pYIN as the absolute truth engine.

You can save this directly as `docs/AUTORESEARCH_STRATEGY.md`.

---

# Machine Learning Lab: Autoresearch Optimization Strategy

This document details the test-driven digital signal processing (DSP) optimization loop for the Mandarin Tone Trainer. By combining **Python pYIN (the Teacher)** with a **Multi-Tiered Human-Validated Corpus (the Student)**, an automated agent can programmatically modify `src/pitchMath.js` to achieve factory-grade pitch tracking accuracy.

---

## 1. The Multi-Tiered Data Architecture

The project repository holds three distinct sound sample directories serving different roles in the optimization and validation lifecycle:

```
                      ┌───────────────────────────┐
                      │    /data/train/corpus     │
                      └─────────────┬─────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
 `/reference`                 `/validation`          `/validation-male`
 (Primary Target)             (Overfitting Guard)    (Context/Diagnostic Check)
   ├── 24 Native MP3s           ├── 24 Native MP3s     ├── 24 Learner MP3s
   └── Master Baseline          └── Same Speaker       └── Includes rating.txt

```

1. **`train/reference` (The Target Model - $Y_{train}$):** The master reference set recorded by the native speaker. The JavaScript engine must learn to match this with maximum geometric fidelity.
2. **`train/validation` (The Overfitting Guard - $Y_{val}$):** A completely separate, second recording pass of the same 24 words by the same native speaker. This ensures the math optimizes for the *phonetic tone shapes*, not the acoustic anomalies (background clicks, mic distance, sighs) of the first recording session.
3. **`train/validation-male` (The Context & Diagnostic Check - $X_{test}$):** The learner's recordings mapped to an explicit human grading matrix (`rating.txt`). This file classifies each recording into three categories: `PERFECT`, `FLAWED`, or `TOTAL_FAIL`.

---

## 2. Stage 1: Establishing the Ground Truth Vector Matrix (`prepare.py`)

Before the JavaScript agent begins optimization, a one-time Python script parses **all three directories** to extract the true pitch contours using the probabilistic pYIN algorithm.

* **The Math Engine:** Uses `librosa.pyin(fmin=60, fmax=400)`.
* **The Normalization Wrapper:** Applies 1D linear interpolation across unvoiced segments (fricative consonants and vocal fry drops) and converts absolute Hz metrics into relative $Z$-scores:

$$z = \frac{f_0 - \mu}{\sigma}$$

* **The Output:** Generates a unified `data/ground-truth-master.json` containing the pristine, 100-point normalized vector tracks for all three voice configurations.

---

## 3. Stage 2: The Multi-Tier Loss Function Matrix (`train.js`)

When the automated agent alters `src/pitchMath.js`, `train.js` compiles the changes and outputs a multi-dimensional **Fitness Score** based on three distinct paths:

### Path A: The Reference Convergence Score (Target: Minimize)

The JavaScript AMDF detector processes the `train/reference` audio files. The runner computes the **Mean Absolute Error (MAE)** against the pYIN reference ground truth:

$$\text{MAE}_{ref} = \frac{1}{24} \sum_{w=1}^{24} \left( \frac{1}{100} \sum_{i=1}^{100} |JS_{w}[i] - pYIN_{w}[i]| \right)$$

* *Goal:* Force the lightweight JavaScript engine to match the tracking capabilities of Python's heavy scientific libraries.

### Path B: The Validation Generalization Gap (Target: Minimize Delta)

The JavaScript engine tracks the second native set (`train/validation`).

* *Goal:* If $\text{MAE}_{ref}$ drops to $0.05$ but $\text{MAE}_{val}$ spikes to $0.40$, the agent has accidentally overfitted the filters to the first audio file's quirks. The loop must reject the math modifications unless both scores drop symmetrically.

### Path C: The Human-in-the-Loop Diagnostic Filter (Target: Maximum Alignment)

The engine processes the `train/validation-male` files and scores the learner's tracking accuracy against the native speaker targets.

* **The Verification Rule:** The agent reads `rating.txt`.
* For files tagged `PERFECT`, the computed similarity score *must* be high ($>90\%$).
* For files tagged `TOTAL_FAIL`, the score *must* be low ($<60\%$).


* **The Penalty Constraint:** If the JavaScript engine awards a $95\%$ accuracy grade to a file your wife manually flagged as a `TOTAL_FAIL`, the engine inflicts a **+500 point Blindness Penalty** to the global loss score, forcing the agent to reject the math changes.

---

## 4. Hyperparameter Search Paths for the Agent (`program.md`)

The `program.md` file explicitly dictates which parameters the optimization loop is allowed to alter inside `src/pitchMath.js` to find the winning configuration:

| Parameter Handle | Mutation Bounds | Optimization Vector Target |
| --- | --- | --- |
| `amdfValleyThreshold` | `0.05` to `0.25` | Lower values catch deep 3rd tone dips (vocal fry) but risk adding octave double-jumps. |
| `movingAverageSpan` | `3`, `5`, or `7` points | Higher values erase harsh plosive noise spikes but round off sharp, intentional 4th tone drops. |
| `zeroCrossingCutoff` | `1000Hz` to `2500Hz` | Dynamically silences unvoiced fricatives ("s", "sh", "ch") at the onset of words. |
| `slopeWeightRatio` | `0.1` to `0.6` | Alters how heavily the grading engine weights the *direction of the curve* (the first derivative) versus the vertical placement. |

---

## 5. Loop Termination Criteria

The automated agent will run sequential experiments, mutating code and reading the stdout from `train.js`. The optimization loop safely terminates only when:

1. **Convergence:** $\text{MAE}_{ref}$ falls below a target threshold of $\le 0.15$.
2. **Generalization:** The performance delta between `/reference` and `/validation` is $\le 5\%$.
3. **Classification Validity:** The algorithm achieves $100\%$ directional alignment with the human classifications provided in `rating.txt` (consistently flunking what she flunked and passing what she passed).

Once these three validation criteria match perfectly, the agent copies the optimal `pitchMath.js` structure and configuration coefficients directly into the production frontend build pipeline.
