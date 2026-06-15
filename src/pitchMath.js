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

// Stubs — implemented in subsequent tasks
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
export function computeDynamicTimeWarping(userTrack, nativeTrack) { return { userAligned: userTrack, nativeAligned: nativeTrack }; }
export function calculateMAEScore(userTrack, nativeTrack) { return 0; }
export function evaluateDiagnosticFeedback(userTrack, nativeTrack) { return ''; }
