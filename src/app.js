/**
 * Main thread orchestrator for the Mandarin Tone Trainer.
 * Manages microphone capture, Web Worker pitch detection, Canvas rendering,
 * and UI event handling.
 *
 * Imports pitchMath functions for future grading use (issue #4).
 */

import {
  computeDynamicTimeWarping,
  calculateMAEScore,
  evaluateDiagnosticFeedback,
} from './pitchMath.js';

// ── DOM references ──────────────────────────────────────────────
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const recordBtn = document.getElementById('recordBtn');
const playBtn = document.getElementById('playBtn');
const wordSelect = document.getElementById('wordSelect');
const statusEl = document.getElementById('status');
const scoreEl = document.getElementById('score');
const feedbackEl = document.getElementById('feedback');

// ── Application state ───────────────────────────────────────────
let audioContext = null;
let analyserNode = null;
let micStream = null;
let worker = null;
let isRecording = false;
let animationId = null;
let userPitchData = [];
let presets = [];
let currentAudio = null;

// ── Canvas constants ────────────────────────────────────────────
const MIN_HZ = 70;
const MAX_HZ = 350;

/**
 * Convert a frequency (Hz) to a Canvas Y coordinate.
 * Higher frequencies → higher on screen (smaller Y).
 * Also clamps values outside the display range.
 */
function freqToY(hz) {
  if (hz <= 0) return canvas.height; // silence → bottom
  const clamped = Math.min(MAX_HZ, Math.max(MIN_HZ, hz));
  const ratio = (clamped - MIN_HZ) / (MAX_HZ - MIN_HZ);
  return canvas.height - ratio * canvas.height;
}

// ── Canvas rendering ────────────────────────────────────────────

/** Full redraw: grid lines + user pitch contour */
function drawCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Grid lines at 100, 150, 200, 250, 300 Hz
  const gridHz = [100, 150, 200, 250, 300];
  ctx.strokeStyle = '#1a1a2e';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 8]);
  for (const hz of gridHz) {
    const y = freqToY(hz);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();

    // Frequency label
    ctx.fillStyle = '#444466';
    ctx.font = '11px monospace';
    ctx.fillText(`${hz} Hz`, 6, y - 4);
  }
  ctx.setLineDash([]);

  // Draw user pitch contour
  if (userPitchData.length < 2) return;

  const xStep = canvas.width / Math.max(userPitchData.length - 1, 1);
  ctx.strokeStyle = '#00ffcc';
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.beginPath();

  for (let i = 0; i < userPitchData.length; i++) {
    const x = i * xStep;
    const y = freqToY(userPitchData[i]);
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();

  // Draw dots at each data point
  ctx.fillStyle = '#00ffcc';
  for (let i = 0; i < userPitchData.length; i++) {
    const x = i * xStep;
    const y = freqToY(userPitchData[i]);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ── Audio initialization ─────────────────────────────────────────

async function initAudio() {
  // Create Web Worker
  worker = new Worker('pitchWorker.js', { type: 'module' });

  worker.onmessage = function (e) {
    const { frequency } = e.data;
    if (isRecording && frequency > 0) {
      userPitchData.push(frequency);
    }
  };

  // Create AudioContext
  audioContext = new AudioContext({ sampleRate: 44100 });

  // Request microphone
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
    },
  });

  // Create AnalyserNode
  analyserNode = audioContext.createAnalyser();
  analyserNode.fftSize = 2048;

  // Connect: microphone → analyser (not to speakers, so no feedback)
  const source = audioContext.createMediaStreamSource(micStream);
  source.connect(analyserNode);
}

// ── Audio processing loop ────────────────────────────────────────

function processAudioFrame() {
  if (!isRecording) return;

  const bufferLength = analyserNode.fftSize;
  const dataArray = new Float32Array(bufferLength);
  analyserNode.getFloatTimeDomainData(dataArray);

  // Send to worker for pitch detection
  if (worker) {
    worker.postMessage({
      audioBuffer: dataArray,
      sampleRate: audioContext.sampleRate,
    });
  }

  // Redraw canvas
  drawCanvas();

  // Schedule next frame
  animationId = requestAnimationFrame(processAudioFrame);
}

// ── Recording controls ───────────────────────────────────────────

async function startRecording() {
  // Initialize audio if not done yet
  if (!audioContext) {
    try {
      statusEl.textContent = 'Requesting microphone access…';
      await initAudio();
    } catch (err) {
      statusEl.textContent = `Mic error: ${err.message}`;
      console.error('Mic access denied:', err);
      return;
    }
  }

  // Resume context if suspended (autoplay policy)
  if (audioContext.state === 'suspended') {
    await audioContext.resume();
  }

  // Reset state
  userPitchData = [];
  isRecording = true;

  // Update UI
  recordBtn.textContent = '⏹ Stop';
  recordBtn.classList.add('recording');
  scoreEl.textContent = '';
  feedbackEl.textContent = '';
  statusEl.textContent = 'Recording…';

  // Start animation loop
  processAudioFrame();
}

function stopRecording() {
  isRecording = false;

  // Update UI
  recordBtn.textContent = '🎙 Record';
  recordBtn.classList.remove('recording');

  // Stop animation
  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }

  // Status
  const sampleCount = userPitchData.length;
  statusEl.textContent = `Stopped — ${sampleCount} pitch samples captured`;

  // One final draw
  drawCanvas();

  // Stub grading
  gradeAttempt();
}

// ── Record button toggle ─────────────────────────────────────────

recordBtn.addEventListener('click', () => {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});

// ── Grading ─────────────────────────────────────────────────────

function gradeAttempt() {
  // Guard: need a preset selected and enough pitch data
  const selectedIndex = wordSelect.selectedIndex - 1; // skip placeholder
  if (selectedIndex < 0 || selectedIndex >= presets.length) {
    statusEl.textContent += ' | Select a word before grading';
    return;
  }
  if (userPitchData.length < 5) {
    statusEl.textContent += ' | Not enough pitch data to grade';
    return;
  }

  const preset = presets[selectedIndex];

  // Normalize user pitch data to 0–1 range
  const nonZeroUser = userPitchData.filter(v => v > 0);
  if (nonZeroUser.length === 0) {
    statusEl.textContent += ' | No voice detected';
    return;
  }
  const userMin = Math.min(...nonZeroUser);
  const userMax = Math.max(...nonZeroUser);
  const userRange = userMax - userMin || 1;
  const normalizedUser = userPitchData.map(v =>
    v > 0 ? (v - userMin) / userRange : 0
  );

  // DTW alignment
  const { userAligned, nativeAligned } = computeDynamicTimeWarping(
    normalizedUser,
    preset.nativePitchReference,
    100
  );

  // MAE score
  const score = calculateMAEScore(userAligned, nativeAligned);
  scoreEl.textContent = `${score}%`;

  // Diagnostic feedback
  const feedback = evaluateDiagnosticFeedback(userAligned, nativeAligned, preset.tones);
  feedbackEl.textContent = feedback;

  // Status message based on score
  if (score >= 80) {
    statusEl.textContent = 'Great job! 🎉';
  } else if (score >= 50) {
    statusEl.textContent = 'Getting there. Try again.';
  } else {
    statusEl.textContent = 'Keep practicing!';
  }
}

async function loadPresets() {
  try {
    const resp = await fetch('/presets.json');
    if (resp.ok) {
      const data = await resp.json();
      presets = data.presets || [];
      // Populate wordSelect dropdown
      for (let i = 0; i < presets.length; i++) {
        const preset = presets[i];
        const option = document.createElement('option');
        option.value = String(i);
        option.textContent = `${preset.word} (${preset.pinyin})`;
        wordSelect.appendChild(option);
      }
    }
  } catch (err) {
    console.warn('Failed to load presets.json:', err);
    presets = [];
  }
}

// ── Play button stub ─────────────────────────────────────────────

playBtn.addEventListener('click', () => {
  const selectedIndex = wordSelect.selectedIndex - 1; // skip placeholder
  if (selectedIndex < 0) {
    statusEl.textContent = 'Select a word first';
    return;
  }

  const preset = presets[selectedIndex];
  if (!preset || !preset.audioSrc) {
    statusEl.textContent = 'No audio available for this word';
    return;
  }

  // Stop any currently playing audio
  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }

  currentAudio = new Audio(preset.audioSrc);
  currentAudio.play().catch(err => {
    statusEl.textContent = `Playback failed: ${err.message}`;
    console.error('Audio playback error:', err);
    currentAudio = null;
  });

  statusEl.textContent = `Playing: ${preset.word} (${preset.pinyin})`;
});

// ── Word select ──────────────────────────────────────────────────

wordSelect.addEventListener('change', () => {
  // Enable play button when a word is selected
  playBtn.disabled = !wordSelect.value;
});

// ── Initialization ───────────────────────────────────────────────

drawCanvas();
loadPresets();
