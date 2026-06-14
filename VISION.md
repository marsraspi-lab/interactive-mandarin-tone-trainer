Here is a comprehensive, production-ready **Product Requirement Document (PRD)** optimized specifically for an AI coding agent. It bridges user-experience requirements with explicit software architecture, math constraints, and algorithmic instructions so the AI can build the complete application without needing to guess the implementation details.

---

# Product Requirement Document (PRD)

## Project Name: Real-Time Visual Mandarin Tone Trainer

---

## 1. Product Overview & Core Objective

The **Mandarin Tone Trainer** is an interactive, browser-based web application designed to fix the "Intermediate Tone Trap." It allows users to practice Mandarin pronunciation by recording their voice and visualizing their **fundamental vocal frequency ($f_0$ pitch contour)** in real-time.

Unlike traditional passive audio quizzes, this app acts as a **vector shape-matching engine**. It compares the normalized geometric shape of the user's spoken pitch trajectory against a native speaker's reference curve, calculates an algorithmic accuracy grade ($0\% - 100\%$), and generates actionable, diagnostic feedback.

---

## 2. Target User Persona & Use Case

* **Persona:** Intermediate learners who have high passive vocabulary/comprehension but face communication breakdowns because of incorrect tones, flat speech, or unstable pitch.
* **Core Loop:** User selects a target tone/word $\rightarrow$ Listens to native audio $\rightarrow$ Records their own voice $\rightarrow$ Sees their pitch line overlaid on the native line $\rightarrow$ Receives a score and logical explanation of their mistakes $\rightarrow$ Retries to beat their high score.

---

## 3. Technical System Architecture

To ensure high-performance UI rendering ($60\text{ FPS}$) and seamless digital signal processing (DSP), the application must adhere to the following 3-tier architecture:

```
[ UI Layer: HTML5/CSS3 ] <───> [ Core Thread: App.js / Canvas 2D ]
                                         │
                                         ▼ (Transferable Float32Array)
                               [ Worker Thread: pitchWorker.js ]
                                 └── Runs Time-Domain Pitch Extraction (YIN/AMDF)

```

### 3.1 Architecture Requirements for the AI Agent:

1. **Zero Main-Thread Bottlenecks:** All heavy array traversals and mathematical calculations for pitch detection must reside in a dedicated Web Worker (`pitchWorker.js`).
2. **State Management:** Maintain a clean internal state object mapping the `nativePitchArray`, `userPitchArray`, `isRecording`, `currentWord`, and `gradingResults`.

---

## 4. Feature Specifications & Implementation Details

### Feature 1: Real-Time Audio Capture Pipeline

* **Description:** Capture clean, low-latency vocal data from the user's microphone.
* **Implementation Requirements:**
* Use `navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } })`.
* Initialize an `AudioContext` with an `AnalyserNode`. Set `analyserNode.fftSize = 2048`.
* Implement a `requestAnimationFrame` loop that calls `analyserNode.getFloat32TimeDomainData(buffer)` on every frame while recording is active.
* Pass the raw `Float32Array` buffer along with the current `sampleRate` to the Web Worker via `worker.postMessage()`.



### Feature 2: Time-Domain Pitch Detection Engine (Web Worker)

* **Description:** Extract the core human vocal frequency ($f_0$) continuously from raw audio streams.
* **Implementation Requirements:**
* Implement a time-domain pitch algorithm like **AMDF (Average Magnitude Difference Function)** or **Autocorrelation** within the worker file.
* **Human Speech Bandpass Filter:** Mandarin vocal pitch resides between $60\text{ Hz}$ and $400\text{ Hz}$. Hard-filter and discard any calculated $f_0$ values outside this range to eliminate background humming, line noise, or quiet breathing.
* **DSP Smoothing Filter:** Apply a 3-point moving average filter onto sequential pitch values to eliminate mathematical artifacts caused by plosive consonants (e.g., "p", "t", "k"). Formula: $f_{smooth}[i] = \frac{f[i-1] + f[i] + f[i+1]}{3}$.
* Post the clean, computed numerical frequency back to the main thread.



### Feature 3: Dynamic Pitch & Time Normalization Graph (HTML5 Canvas)

* **Description:** A viewport plotting the user’s vocal track dynamically alongside a reference track.
* **Implementation Requirements:**
* **The Y-Axis Correction:** Canvas coordinates map $(0,0)$ to the top-left corner. Invert the rendering math so higher frequencies sit visibly higher on the screen:

$$\text{Y}_{canvas} = \text{Height}_{canvas} - \left(\frac{f_0 - \text{Min}_{\text{Hz}}}{\text{Max}_{\text{Hz}} - \text{Min}_{\text{Hz}}}\right) \times \text{Height}_{canvas}$$



*(Set $\text{Min}_{\text{Hz}} = 70$ and $\text{Max}_{\text{Hz}} = 350$).*
* **Vocal Range Normalization (Relative Pitch):** To ensure a baritone male can match a soprano female native model, convert absolute Hz values into relative offsets from the speaker's median pitch, or normalize arrays relative to their respective min/max bounds before comparison.
* **Visual Elements:** Render the native speaker's tone path as a dashed, semi-transparent template line (e.g., `#888888`). Overlay the user's recorded pitch as a bold, solid neon line (e.g., `#00ffcc`).



### Feature 4: Mathematical Grading Engine (Vector Comparison)

* **Description:** Quantify the accuracy of the user's tone shape against the target.
* **Implementation Requirements:**
* **Time Warping Alignment:** Users speak at different speeds than native tracks. Implement a basic **Dynamic Time Warping (DTW)** sequence or a linear interpolation resampler to snap both arrays to an identical length (e.g., 100 data points).
* **Scoring Algorithm:** Compute the **Mean Absolute Error (MAE)** between the normalized user array ($U$) and native array ($N$):

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |U_i - N_i|$$


* Convert this error to a $0\% - 100\%$ score string. A threshold where average deviation is minimal outputs a score $\ge 90\%$.



### Feature 5: Diagnostic Feedback Engine (Expert Error Logging)

* **Description:** Analyze the mathematical vector deviations to output targeted instructions instead of just an arbitrary percentage.
* **Implementation Requirements:** Use conditional array evaluation to detect shape failures:
* *Condition:* Native slope is positive ($N_{end} > N_{start}$) but User slope is negative ($U_{end} < U_{start}$).
**Output Error:** `"Pitch Dropped: For this rising tone (Tone 2), your voice must slide upward like you are asking an unprompted question. You dragged it downward."`
* *Condition:* Native curve dips low ($N_{mid} < \text{Threshold}$) but User curve remains flat ($U_{mid} \approx U_{start}$).
**Output Error:** `"Not Deep Enough: For this dipping tone (Tone 3), drop your pitch completely into the lowest basement of your vocal range before letting it rise."`
* *Condition:* Native slope drops sharply ($N_{end} \ll N_{start}$) but User slope drops gradually.
**Output Error:** `"Too Soft/Slow: This falling tone (Tone 4) should sound like an abrupt, angry command. Drop your pitch rapidly and confidently."`



---

## 5. Scope of Content & Presets

To make the application instantly playable, preload the app with native data models for **Tone Pairs** (the ultimate bottleneck for intermediate speakers). Provide sample presets for:

1. **Tone 1 + Tone 4:** e.g., 公司 (*gōngsī* - company)
2. **Tone 2 + Tone 4:** e.g., 銀行 (*yínháng* - bank)
3. **Tone 3 + Tone 1:** e.g., 老師 (*lǎoshī* - teacher)

---

## 6. Prompt to Initiate the AI Agent

*Copy and hand the text box below directly to the AI coding agent along with this document:*

```text
Read the attached PRD for the "Real-Time Visual Mandarin Tone Trainer". Your task is to implement the code cleanly, securely, and completely. Build the application across three cohesive files: `index.html`, `app.js`, and `pitchWorker.js` as specified in the PRD. Ensure all normalization calculations, mathematical vector grading formulas, and worker thread communication functions are completely written out with no partial logic blocks or missing handlers. Focus on high rendering performance and clean canvas visual design.

```