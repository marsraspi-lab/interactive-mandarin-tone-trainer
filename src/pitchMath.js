/**
 * Detect fundamental frequency (f0) using AMDF (Average Magnitude Difference Function).
 * Bandpass filters to 60–400 Hz (Mandarin vocal range).
 *
 * NOTE: AMDF can lock onto subharmonics with pure-tone synthetic signals where
 * a harmonic aligns more perfectly with the buffer length than the fundamental.
 * This doesn't affect real speech (rich harmonic spectrum). Test frequencies are
 * chosen to divide evenly into 44100 Hz to avoid this synthetic-edge-case.
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

    // Use epsilon-greater to break near-ties in favor of smaller τ
    // (higher frequency = fundamental over subharmonic). AMDF on real
    // speech has values in the 0.01–1.0 range; 1e-12 is negligible there
    // but prevents floating-point subharmonic steals on pure tones.
    if (avgDiff < bestDiff - 1e-12) {
      bestDiff = avgDiff;
      bestPeriod = period;
    }
  }

  if (bestPeriod <= 0) return 0;

  // Periodicity gate: if the minimum AMDF is more than 40% of RMS,
  // the signal isn't periodic enough for a reliable pitch estimate
  if (bestDiff > rms * 0.4) return 0;

  const frequency = sampleRate / bestPeriod;

  // Bandpass gate
  if (frequency < MIN_FREQ || frequency > MAX_FREQ) return 0;

  return frequency;
}

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

/**
 * Resample an array to exactly targetLength points via linear interpolation.
 * Used by both the build pipeline and client-side grading for consistent
 * 100-point time normalization.
 *
 * @param {number[]} arr — input array
 * @param {number} targetLength — desired output length
 * @returns {number[]} resampled array
 */
export function resampleArray(arr, targetLength) {
  if (arr.length === 0) return new Array(targetLength).fill(0);
  if (arr.length === 1) return new Array(targetLength).fill(arr[0]);

  const result = new Array(targetLength);
  const step = (arr.length - 1) / (targetLength - 1);

  for (let i = 0; i < targetLength; i++) {
    const pos = i * step;
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, arr.length - 1);
    const frac = pos - lo;
    result[i] = arr[lo] + (arr[hi] - arr[lo]) * frac;
  }
  return result;
}

/**
 * Resample both arrays to a common length via linear interpolation.
 * This is a lightweight alternative to full DTW — sufficient for
 * comparing pitch contour shapes.
 *
 * @param {number[]} userTrack — Z-score normalized user pitch values
 * @param {number[]} nativeTrack — Z-score normalized native pitch values
 * @param {number} [targetLength=100] — common length to resample to
 * @returns {{ userAligned: number[], nativeAligned: number[] }}
 */
export function computeDynamicTimeWarping(userTrack, nativeTrack, targetLength = 100) {
  if (userTrack.length === 0 || nativeTrack.length === 0) {
    return { userAligned: [], nativeAligned: [] };
  }

  return {
    userAligned: resampleArray(userTrack, targetLength),
    nativeAligned: resampleArray(nativeTrack, targetLength),
  };
}

/**
 * Z-score statistical normalization: centers the pitch array at μ=0 with σ=1.
 * Zeros (silence/unvoiced frames) are excluded from μ/σ calculation and left as 0.
 * Produces values typically in the range [-2.0, +2.0] for Mandarin pitch contours.
 *
 * @param {number[]} pitchArray — sequential pitch values in Hz (0 = silence)
 * @returns {number[]} Z-score normalized values
 */
export function normalizeZScore(pitchArray) {
  const voiced = pitchArray.filter(v => v > 0);
  if (voiced.length < 2) return new Array(pitchArray.length).fill(0);

  // Mean
  let sum = 0;
  for (const v of voiced) sum += v;
  const mean = sum / voiced.length;

  // Standard deviation (population)
  let sumSqDiff = 0;
  for (const v of voiced) sumSqDiff += (v - mean) ** 2;
  const std = Math.sqrt(sumSqDiff / voiced.length);
  if (std < 1e-10) return pitchArray.map(v => (v > 0 ? 0 : 0));

  return pitchArray.map(v => (v > 0 ? (v - mean) / std : 0));
}

/**
 * Clamp Z-score values to a fixed range to prevent extreme outliers
 * from skewing DTW comparisons. Default range [-3, +3] captures 99.7% of
 * normally-distributed data.
 *
 * @param {number[]} values — Z-score normalized pitch values
 * @param {number} [min=-3] — floor
 * @param {number} [max=3] — ceiling
 * @returns {number[]} clamped values
 */
export function clampValues(values, min = -3, max = 3) {
  const result = new Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    result[i] = v === 0 ? 0 : (v < min ? min : (v > max ? max : v));
  }
  return result;
}

/**
 * Calculate Mean Absolute Error (MAE) between two Z-score normalized pitch arrays
 * and convert to a 0–100% accuracy score.
 *
 * MAE = (1/n) * Σ|Ui - Ni|
 * Score = max(0, 100 * (1 - MAE / threshold))
 *
 * For Z-score normalized data (μ=0, σ=1, typical range ±2):
 *   threshold=2.0 → MAE of 2σ = 0% score
 *   threshold=1.0 → MAE of 1σ = 50% score
 *
 * @param {number[]} userTrack — Z-score normalized user pitch values
 * @param {number[]} nativeTrack — Z-score normalized native pitch values
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

  // Z-score threshold: MAE of 2.0 means avg deviation is 2σ — that's failing
  const score = Math.max(0, 100 * (1 - mae / 2.0));
  return Math.round(score);
}

/**
 * Analyze vector deviations between user and native pitch to produce
 * actionable diagnostic feedback based on tone-specific shape errors.
 *
 * Detects:
 *  - Pitch Dropped (rising tone but user fell)
 *  - Not Deep Enough (dipping tone but user stayed flat)
 *  - Too Soft/Slow (falling tone but user declined gradually)
 *
 * Thresholds calibrated for Z-score normalized values (μ=0, σ=1, range ~±2).
 *
 * @param {number[]} userTrack — Z-score normalized user pitch values
 * @param {number[]} nativeTrack — Z-score normalized native pitch values
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
  if (tones.includes(2) && nativeSlope > 0.4 && userSlope < -0.4) {
    return 'Pitch Dropped: For this rising tone (Tone 2), your voice must slide upward like you are asking an unprompted question. You dragged it downward.';
  }

  // Tone 3 check: native dips but user stays flat
  const nativeDipDepth = Math.max(nativeStart, nativeEnd) - nativeMid;
  const userDipDepth = Math.max(userStart, userEnd) - userMid;
  if (tones.includes(3) && nativeDipDepth > 0.6 && userDipDepth < 0.2) {
    return 'Not Deep Enough: For this dipping tone (Tone 3), drop your pitch completely into the lowest basement of your vocal range before letting it rise.';
  }

  // Tone 4 check: native falls sharply but user falls gradually
  const nativeDropRate = (nativeStart - nativeEnd) / n;
  const userDropRate = (userStart - userEnd) / n;
  if (tones.includes(4) && nativeDropRate > 0.02 && userDropRate < nativeDropRate * 0.3) {
    return 'Too Soft/Slow: This falling tone (Tone 4) should sound like an abrupt, angry command. Drop your pitch rapidly and confidently.';
  }

  return '';
}
