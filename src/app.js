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

// ── Stubs (for issues #4 and #5) ──────────────────────────────────

function gradeAttempt() {
  // STUB: In issue #4 this will load presets, run DTW+MAE+diagnostics
  // For now, just check if there's data
  if (userPitchData.length === 0) {
    return;
  }
  // Placeholder
  statusEl.textContent += ' | Grading coming in issue #4';
}

async function loadPresets() {
  // STUB: fetches presets.json (doesn't exist yet, will fail gracefully)
  try {
    const resp = await fetch('presets.json');
    if (resp.ok) {
      const presets = await resp.json();
      // Populate wordSelect dropdown
      for (const preset of presets) {
        const option = document.createElement('option');
        option.value = preset.word;
        option.textContent = preset.word;
        wordSelect.appendChild(option);
      }
    }
  } catch {
    // presets.json doesn't exist yet — that's fine for now
    console.log('No presets.json found — skipping preset load (issue #4)');
  }
}

// ── Play button stub ─────────────────────────────────────────────

playBtn.addEventListener('click', () => {
  // STUB: playback comes in issue #5
  statusEl.textContent = 'Playback coming in issue #5';
});

// ── Word select ──────────────────────────────────────────────────

wordSelect.addEventListener('change', () => {
  // Enable play button when a word is selected
  playBtn.disabled = !wordSelect.value;
});

// ── Initialization ───────────────────────────────────────────────

drawCanvas();
loadPresets();
