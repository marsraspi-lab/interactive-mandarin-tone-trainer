Here is a comprehensive, production-ready **Advanced DSP & Math backlog**. It is formatted specifically as a technical appendix that you can drop directly into your repository as a `BACKLOG.md` file, or feed into your AI agent when you are ready to transition the app from an MVP into a commercial-grade, highly robust tone tracking engine.

---

# Repository Backlog: Advanced DSP & Pitch Grading Enhancements

This document tracks specialized mathematical, statistical, and digital signal processing (DSP) optimizations required to transition the Mandarin Tone Trainer from a basic happy-path implementation into a robust, noise-resilient production application.

---

## 🚀 Epic 1: Signal Conditioning & Noise Immunity (The Input Pipeline)

### Task 1.1: Implement Fricative Eraser via Zero-Crossing Rate (ZCR)

* **Reasoning:** Mandarin consonants like "s", "x", "sh", and "ch" are unvoiced fricatives. They do not involve vocal cord vibration ($f_0$) and present acoustically as chaotic white noise. If passed directly to the AMDF engine, they cause random high-frequency tracking spikes at the beginning of syllables.
* **Implementation Specs:** Before running AMDF on an audio frame buffer ($x$), calculate its Zero-Crossing Rate (ZCR), which measures how many times the signal changes sign per second:

$$\text{ZCR} = \frac{1}{2N} \sum_{n=1}^{N-1} |\text{sgn}(x[n]) - \text{sgn}(x[n-1])|$$



If the ZCR exceeds a determined threshold (indicating unvoiced high-frequency noise), explicitly flag the frame as `isUnvoiced = true`, skip the AMDF loop, and prevent the canvas from rendering junk lines during that window.

### Task 1.2: Build an Anti-Octave Jump Sub-Harmonic Filter

* **Reasoning:** Human speech contains strong harmonic overtones. Time-domain pitch detection algorithms frequently fall into "octave errors," where they accidentally track the second harmonic ($2 \times f_0$) or half-frequency sub-harmonics ($0.5 \times f_0$), causing the canvas line to glitch and teleport vertically.
* **Implementation Specs:**
Maintain a short rolling 3-frame historical cache of verified frequencies. When a new frequency $f_{new}$ is calculated, run a proximity check:

$$\text{If } |f_{new} - 2\cdot f_{median}| < \epsilon \implies f_{corrected} = \frac{f_{new}}{2}$$


$$\text{If } |f_{new} - 0.5\cdot f_{median}| < \epsilon \implies f_{corrected} = f_{new} \cdot 2$$



Reject frames that cannot be logically reconciled with recent vocal history to prevent vertical spike artifacts.

---

## 📈 Epic 2: Normalization & Acoustic Fidelity (The Comparison Engine)

### Task 2.1: Implement Z-Score Normalization over Global Min/Max Stretching

* **Reasoning:** Linear min/max scaling stretches pitch arrays to absolute bounds ($0.0$ to $1.0$). If a native speaker uses a narrow, calm pitch range for a word, but the user speaks with high emotional expressiveness, linear stretching distorts the geometric shape of the contour, leading to unfair grading penalties.
* **Implementation Specs:**
Replace min/max normalization in `pitchMath.js` with Z-Score normalization. This tracks how far a pitch data point deviates from the speaker's own mean calculation across that specific phrase, preserving the true relative contour structural DNA:

$$z_i = \frac{f_0[i] - \mu}{\sigma}$$



Where $\mu$ is the mathematical mean of the valid pitch track segments, and $\sigma$ is the standard deviation.

### Task 2.2: Syllable-Anchored Multi-Segment Alignment

* **Reasoning:** Mandarin tone pairs are heavily influenced by *coarticulation* (e.g., a 1st tone starts lower when recovering from a deep 3rd tone). Normalizing a multi-syllable word as a single uniform string masks these subtle, critical acoustic transitions.
* **Implementation Specs:**
Segment incoming pitch streams into separate arrays based on intensity dips (energy valleys that indicate the boundary between two syllables). Run Z-score normalization and Dynamic Time Warping (DTW) alignments *per syllable segment* rather than across the whole phrase globally. Grade the transition slope linking Syllable A to Syllable B as its own isolated vector checking point.

---

## 🎯 Epic 3: Robust Grading & Intelligent Diagnostic Rules

### Task 3.1: Pitch Trajectory Spline Interpolation for Vocal Fry Support

* **Reasoning:** When hitting the lowest sections of a dipping 3rd tone (e.g., in **老師**), human voices naturally slip into "vocal fry" (creaky voice). Vocal fry is non-periodic, meaning AMDF will return `null` or `0` for those frames, leaving empty gaps right in the most critical phase of the tone.
* **Implementation Specs:**
Implement a cubic spline or linear interpolation routine in the post-processing phase. If a user's tracking array drops frames due to vocal fry, but the root-mean-square (RMS) volume amplitude indicates they are still phonating heavily at low frequency, bridge the gap mathematically by smoothly connecting the valid tracking points across the drop window instead of penalizing them for a missing line.

### Task 3.2: First-Derivative Slope Vector Matching

* **Reasoning:** A user can get the "shape" of a tone correct but receive a terrible score from standard Mean Absolute Error (MAE) algorithms if their baseline voice shifts or cracks slightly mid-recording.
* **Implementation Specs:**
Calculate the first derivative (the local slope) at every point along both the user and native arrays:

$$\Delta f[i] = f[i] - f[i-1]$$



Base $40\%$ of the final grading matrix score on matching the *signs and magnitudes of the derivatives* (i.e., whether the pitch is rising, falling, or holding flat at the exact same relative timeline positions) rather than checking absolute vertical coordinates.

---

### 📋 How to feed this to your AI agent later:

When you are ready to implement any of these specific tasks, hand the agent your core codebase along with the following instructional prompt:

```text
Pull Task [X.X] from the BACKLOG.md. Refactor the existing functions inside `src/pitchMath.js` to implement the advanced mathematical formulas specified. Ensure you write corresponding unit tests in `tests/pitchMath.test.js` to verify the new algorithm changes against edge cases before modifying the live canvas wrapper.

```
