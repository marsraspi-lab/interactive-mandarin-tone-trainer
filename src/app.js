/**
 * Main thread orchestrator for the Mandarin Tone Trainer.
 * Manages microphone capture, Web Worker pitch detection, Canvas rendering,
 * and UI event handling with sidebar word selection.
 */
import {
  computeDynamicTimeWarping,
  calculateMAEScore,
  calculateConsensusMAEScore,
  evaluateDiagnosticFeedback,
  normalizeZScore,
  normalizeWithSharedStats,
  clampValues,
  applyOctaveCorrection,
  applySplineInterpolation,
} from './pitchMath.js';

// ── Preset corpus (24 entries, sorted by tone) ──────────────────

const PRESETS = [
  // Single syllables
  { word: '妈', pinyin: 'mā',   tones: [1],    audioSrc: '/assets/audio/ma1.mp3' },
  { word: '麻', pinyin: 'má',   tones: [2],    audioSrc: '/assets/audio/ma2.mp3' },
  { word: '马', pinyin: 'mǎ',   tones: [3],    audioSrc: '/assets/audio/ma3.mp3' },
  { word: '骂', pinyin: 'mà',   tones: [4],    audioSrc: '/assets/audio/ma4.mp3' },
  // Tone pairs — 1+X
  { word: '今天', pinyin: 'jīntiān', tones: [1,1], audioSrc: '/assets/audio/jintian.mp3' },
  { word: '今年', pinyin: 'jīnnián', tones: [1,2], audioSrc: '/assets/audio/jinnian.mp3' },
  { word: '机场', pinyin: 'jīchǎng', tones: [1,3], audioSrc: '/assets/audio/jichang.mp3' },
  { word: '音乐', pinyin: 'yīnyuè', tones: [1,4], audioSrc: '/assets/audio/yinyue.mp3' },
  { word: '哥哥', pinyin: 'gēge',    tones: [1,0], audioSrc: '/assets/audio/gege.mp3' },
  // 2+X
  { word: '明天', pinyin: 'míngtiān',tones: [2,1], audioSrc: '/assets/audio/mingtian.mp3' },
  { word: '明年', pinyin: 'míngnián',tones: [2,2], audioSrc: '/assets/audio/mingnian.mp3' },
  { word: '苹果', pinyin: 'píngguǒ', tones: [2,3], audioSrc: '/assets/audio/pingguo.mp3' },
  { word: '决定', pinyin: 'juédìng', tones: [2,4], audioSrc: '/assets/audio/jueding.mp3' },
  { word: '孩子', pinyin: 'háizi',   tones: [2,0], audioSrc: '/assets/audio/haizi.mp3' },
  // 3+X
  { word: '老师', pinyin: 'lǎoshī',  tones: [3,1], audioSrc: '/assets/audio/laoshi.mp3' },
  { word: '旅行', pinyin: 'lǚxíng',  tones: [3,2], audioSrc: '/assets/audio/luxing.mp3' },
  { word: '水果', pinyin: 'shuǐguǒ', tones: [3,3], audioSrc: '/assets/audio/shuiguo.mp3' },
  { word: '好看', pinyin: 'hǎokàn',  tones: [3,4], audioSrc: '/assets/audio/haokan.mp3' },
  { word: '姐姐', pinyin: 'jiějie',  tones: [3,0], audioSrc: '/assets/audio/jiejie.mp3' },
  // 4+X
  { word: '唱歌', pinyin: 'chànggē', tones: [4,1], audioSrc: '/assets/audio/changge.mp3' },
  { word: '问题', pinyin: 'wèntí',   tones: [4,2], audioSrc: '/assets/audio/wenti.mp3' },
  { word: '电脑', pinyin: 'diànnǎo', tones: [4,3], audioSrc: '/assets/audio/diannao.mp3' },
  { word: '再见', pinyin: 'zàijiàn', tones: [4,4], audioSrc: '/assets/audio/zaijian.mp3' },
  { word: '谢谢', pinyin: 'xièxie',  tones: [4,0], audioSrc: '/assets/audio/xiexie.mp3' },
];

// ── DOM references ──────────────────────────────────────────────
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const recordBtn = document.getElementById('recordBtn');
const playBtn = document.getElementById('playBtn');
const navList = document.getElementById('navList');
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
let presets = PRESETS;           // full word list
let pitchRefs = new Map();       // word → nativePitchReference (from presets.json)
let selectedIndex = -1;
let currentAudio = null;

// ── Canvas constants ────────────────────────────────────────────
const MIN_HZ = 70;
const MAX_HZ = 350;

/**
 * Convert a frequency (Hz) to a Canvas Y coordinate.
 * Higher frequencies → higher on screen (smaller Y).
 */
function freqToY(hz) {
  if (hz <= 0) return canvas.height;
  const clamped = Math.min(MAX_HZ, Math.max(MIN_HZ, hz));
  const ratio = (clamped - MIN_HZ) / (MAX_HZ - MIN_HZ);
  return canvas.height - ratio * canvas.height;
}

// ── Sidebar ─────────────────────────────────────────────────────

function renderSidebar() {
  navList.innerHTML = '';
  presets.forEach((p, i) => {
    const div = document.createElement('div');
    div.className = 'nav-item' + (i === selectedIndex ? ' active' : '');
    div.innerHTML = `
      <span class="hanzi">${p.word}</span>
      <span class="pinyin">${p.pinyin}</span>
    `;
    div.addEventListener('click', () => selectWord(i));
    navList.appendChild(div);
  });
}

function selectWord(index) {
  selectedIndex = index;
  playBtn.disabled = false;
  renderSidebar();
  statusEl.textContent = `Selected: ${presets[index].word} (${presets[index].pinyin})`;
}

// ── Canvas rendering ────────────────────────────────────────────

/** Full redraw: grid lines + user pitch contour */
function drawCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Grid lines
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
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Dots at each data point
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
  worker = new Worker('pitchWorker.js', { type: 'module' });

  worker.onmessage = function (e) {
    const { frequency } = e.data;
    if (isRecording && frequency > 0) {
      userPitchData.push(frequency);
    }
  };

  audioContext = new AudioContext({ sampleRate: 44100 });

  micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
    },
  });

  analyserNode = audioContext.createAnalyser();
  analyserNode.fftSize = 2048;

  const source = audioContext.createMediaStreamSource(micStream);
  source.connect(analyserNode);
}

// ── Audio processing loop ────────────────────────────────────────

function processAudioFrame() {
  if (!isRecording) return;

  const bufferLength = analyserNode.fftSize;
  const dataArray = new Float32Array(bufferLength);
  analyserNode.getFloatTimeDomainData(dataArray);

  if (worker) {
    worker.postMessage({
      audioBuffer: dataArray,
      sampleRate: audioContext.sampleRate,
    });
  }

  drawCanvas();
  animationId = requestAnimationFrame(processAudioFrame);
}

// ── Recording controls ───────────────────────────────────────────

async function startRecording() {
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

  if (audioContext.state === 'suspended') {
    await audioContext.resume();
  }

  userPitchData = [];
  isRecording = true;

  recordBtn.textContent = '⏹ Stop';
  recordBtn.classList.add('recording');
  scoreEl.textContent = '';
  feedbackEl.textContent = '';
  statusEl.textContent = 'Recording…';

  processAudioFrame();
}

function stopRecording() {
  isRecording = false;

  recordBtn.textContent = '🎙 Record';
  recordBtn.classList.remove('recording');

  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }

  const sampleCount = userPitchData.length;
  statusEl.textContent = `Stopped — ${sampleCount} pitch samples captured`;

  drawCanvas();
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
  if (selectedIndex < 0 || selectedIndex >= presets.length) {
    statusEl.textContent += ' | Select a word before grading';
    return;
  }
  if (userPitchData.length < 5) {
    statusEl.textContent += ' | Not enough pitch data to grade';
    return;
  }

  const preset = presets[selectedIndex];

  const refData = pitchRefs.get(preset.word);
  if (!refData) {
    statusEl.textContent += ' | No pitch reference for this word yet';
    return;
  }

  // Extract reference curve and optional native stats
  const nativeRef = refData.reference || refData;
  const nativeMean = refData.nativeMean ?? null;
  const nativeStd = refData.nativeStd ?? null;

  // ── Improved post-processing pipeline ──────────────────────────
  // Step 1: Octave-jump correction (rolling median, 3-frame window)
  const octaveCorrected = applyOctaveCorrection(userPitchData, 3);

  // Step 2: Fill short gaps via linear interpolation (max gap 8 frames)
  const gapFilled = applySplineInterpolation(octaveCorrected, 8);

  // Step 3: Z-score normalization — shared stats when available
  let normalizedUser;
  if (nativeMean != null && nativeStd != null && nativeStd > 1e-10) {
    normalizedUser = normalizeWithSharedStats(gapFilled, nativeMean, nativeStd);
  } else {
    normalizedUser = normalizeZScore(gapFilled);
  }

  // Step 4: Clamp to [-3, +3]
  const clampedUser = clampValues(normalizedUser, -3, 3);

  const { userAligned, nativeAligned } = computeDynamicTimeWarping(
    clampedUser,
    nativeRef,
    100
  );

  // Standard score (all 100 frames)
  const stdScore = calculateMAEScore(userAligned, nativeAligned);

  // Consensus score (only pYIN ∩ AMDF frames)
  const consensusMask = refData.consensusMask || null;
  const conScore = calculateConsensusMAEScore(userAligned, nativeAligned, consensusMask);

  // Display both scores side by side
  let scoreText = `Standard: ${stdScore}%`;
  if (conScore != null) {
    const maskCount = consensusMask.filter(Boolean).length;
    scoreText += `  |  Consensus: ${conScore}% (${maskCount}/100 frames)`;
  } else {
    scoreText += `  |  Consensus: N/A`;
  }
  scoreEl.textContent = scoreText;

  const feedback = evaluateDiagnosticFeedback(userAligned, nativeAligned, preset.tones);
  feedbackEl.textContent = feedback;

  if (stdScore >= 80) {
    statusEl.textContent = 'Great job! 🎉';
  } else if (stdScore >= 50) {
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
      for (const p of (data.presets || [])) {
        // Store full preset data: reference curve + optional native µ/σ + consensus mask
        pitchRefs.set(p.word, {
          reference: p.nativePitchReference || [],
          nativeMean: p.nativeMean,
          nativeStd: p.nativeStd,
          consensusMask: p.consensusMask || null,
        });
      }
    }
  } catch (err) {
    console.warn('Failed to load presets.json:', err);
  }
}

// ── Play button ─────────────────────────────────────────────────

playBtn.addEventListener('click', () => {
  if (selectedIndex < 0) {
    statusEl.textContent = 'Select a word first';
    return;
  }

  const preset = presets[selectedIndex];
  if (!preset || !preset.audioSrc) {
    statusEl.textContent = 'No audio available for this word';
    return;
  }

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

// ── Initialization ───────────────────────────────────────────────

drawCanvas();
renderSidebar();
loadPresets();
