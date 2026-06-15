/**
 * admin.js — Local Development Recording Dashboard
 *
 * Records native audio, extracts pitch contours via the existing
 * pitchWorker.js pipeline, and exports a production-ready presets.json.
 */
import { processAudioFrame } from './pitchWorker.js';
import { applyThreePointSmoothing } from './pitchMath.js';

// ── Preset Corpus (24 entries) ────────────────────────────────────────────

const PRESETS = [
  // Single syllables
  { word: '媽', pinyin: 'mā',   tones: [1],    file: 'ma1.wav',  audioSrc: '/audio/ma1.wav' },
  { word: '麻', pinyin: 'má',   tones: [2],    file: 'ma2.wav',  audioSrc: '/audio/ma2.wav' },
  { word: '馬', pinyin: 'mǎ',   tones: [3],    file: 'ma3.wav',  audioSrc: '/audio/ma3.wav' },
  { word: '罵', pinyin: 'mà',   tones: [4],    file: 'ma4.wav',  audioSrc: '/audio/ma4.wav' },
  // Tone pairs
  { word: '今天', pinyin: 'jīntiān', tones: [1,1], file: 'jintian.wav',    audioSrc: '/audio/jintian.wav' },
  { word: '今年', pinyin: 'jīnnián', tones: [1,2], file: 'jinnian.wav',    audioSrc: '/audio/jinnian.wav' },
  { word: '緊張', pinyin: 'jǐnzhāng',tones: [1,3], file: 'jinzhang.wav',   audioSrc: '/audio/jinzhang.wav' },
  { word: '公司', pinyin: 'gōngsī',  tones: [1,4], file: 'gongsi.wav',     audioSrc: '/audio/gongsi.wav' },
  { word: '哥哥', pinyin: 'gēge',    tones: [1,0], file: 'gege.wav',       audioSrc: '/audio/gege.wav' },
  { word: '銀行', pinyin: 'yínháng', tones: [2,1], file: 'yinhang.wav',    audioSrc: '/audio/yinhang.wav' },
  { word: '明年', pinyin: 'míngnián',tones: [2,2], file: 'mingnian.wav',   audioSrc: '/audio/mingnian.wav' },
  { word: '蘋果', pinyin: 'píngguǒ', tones: [2,3], file: 'pingguo.wav',    audioSrc: '/audio/pingguo.wav' },
  { word: '決定', pinyin: 'juédìng', tones: [2,4], file: 'jueding.wav',    audioSrc: '/audio/jueding.wav' },
  { word: '孩子', pinyin: 'háizi',   tones: [2,0], file: 'haizi.wav',      audioSrc: '/audio/haizi.wav' },
  { word: '老師', pinyin: 'lǎoshī',  tones: [3,1], file: 'laoshi.wav',     audioSrc: '/audio/laoshi.wav' },
  { word: '旅行', pinyin: 'lǚxíng',  tones: [3,2], file: 'luxing.wav',     audioSrc: '/audio/luxing.wav' },
  { word: '水果', pinyin: 'shuǐguǒ', tones: [3,3], file: 'shuiguo.wav',    audioSrc: '/audio/shuiguo.wav' },
  { word: '電腦', pinyin: 'diànnǎo', tones: [3,4], file: 'diannao.wav',    audioSrc: '/audio/diannao.wav' },
  { word: '姐姐', pinyin: 'jiějie',  tones: [3,0], file: 'jiejie.wav',     audioSrc: '/audio/jiejie.wav' },
  { word: '機場', pinyin: 'jīchǎng', tones: [4,1], file: 'jichang.wav',    audioSrc: '/audio/jichang.wav' },
  { word: '問題', pinyin: 'wèntí',   tones: [4,2], file: 'wenti.wav',      audioSrc: '/audio/wenti.wav' },
  { word: '漢語', pinyin: 'hànyǔ',   tones: [4,3], file: 'hanyu.wav',      audioSrc: '/audio/hanyu.wav' },
  { word: '再見', pinyin: 'zàijiàn', tones: [4,4], file: 'zaijian.wav',    audioSrc: '/audio/zaijian.wav' },
  { word: '謝謝', pinyin: 'xièxie',  tones: [4,0], file: 'xiexie.wav',     audioSrc: '/audio/xiexie.wav' },
];

// ── State ─────────────────────────────────────────────────────────────────

let currentIndex = 0;
let recordings = new Map(); // index → { nativePitchReference: number[] }
let audioContext = null;
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let micStream = null;

// ── DOM refs ──────────────────────────────────────────────────────────────

const navList       = document.getElementById('navList');
const hanziDisplay  = document.getElementById('hanziDisplay');
const pinyinDisplay = document.getElementById('pinyinDisplay');
const toneLabel     = document.getElementById('toneLabel');
const miniCanvas    = document.getElementById('miniCanvas');
const miniCtx       = miniCanvas.getContext('2d');
const recordBtn     = document.getElementById('recordBtn');
const prevBtn       = document.getElementById('prevBtn');
const nextBtn       = document.getElementById('nextBtn');
const exportBtn     = document.getElementById('exportBtn');
const recordingCount= document.getElementById('recordingCount');

// ── Navigation ────────────────────────────────────────────────────────────

function renderSidebar() {
  navList.innerHTML = '';
  PRESETS.forEach((p, i) => {
    const div = document.createElement('div');
    div.className = 'nav-item' + (i === currentIndex ? ' active' : '');
    div.innerHTML = `
      <span class="hanzi">${p.word}</span>
      <span class="pinyin">${p.pinyin}</span>
      <span class="status-dot ${recordings.has(i) ? 'recorded' : 'missing'}"></span>
    `;
    div.addEventListener('click', () => navigateTo(i));
    navList.appendChild(div);
  });
}

function navigateTo(index) {
  currentIndex = index;
  renderSidebar();
  renderCard();
  renderMiniCanvas();
  updateExportButton();
}

function renderCard() {
  const p = PRESETS[currentIndex];
  hanziDisplay.textContent = p.word;
  pinyinDisplay.textContent = p.pinyin;
  const toneStr = p.tones.map(t => t === 0 ? 'Neutral' : String(t)).join(' + ');
  toneLabel.textContent = p.tones.length === 1
    ? `Tone: ${toneStr}`
    : `Tone Pair: ${toneStr}`;
}

// ── Mini Canvas ───────────────────────────────────────────────────────────

function renderMiniCanvas() {
  const data = recordings.get(currentIndex);
  miniCtx.clearRect(0, 0, miniCanvas.width, miniCanvas.height);

  // Grid
  miniCtx.strokeStyle = '#1a1a2e';
  miniCtx.lineWidth = 0.5;
  for (let y = 0; y < miniCanvas.height; y += 28) {
    miniCtx.beginPath();
    miniCtx.moveTo(0, y);
    miniCtx.lineTo(miniCanvas.width, y);
    miniCtx.stroke();
  }

  if (!data || !data.nativePitchReference || data.nativePitchReference.length < 2) {
    miniCtx.fillStyle = '#333';
    miniCtx.font = '12px monospace';
    miniCtx.textAlign = 'center';
    miniCtx.fillText('No recording yet', miniCanvas.width / 2, miniCanvas.height / 2);
    return;
  }

  const ref = data.nativePitchReference;
  const w = miniCanvas.width;
  const h = miniCanvas.height;
  const pad = 10;

  miniCtx.strokeStyle = '#00ffcc';
  miniCtx.lineWidth = 1.5;
  miniCtx.beginPath();
  const stepX = (w - pad * 2) / (ref.length - 1);
  for (let i = 0; i < ref.length; i++) {
    const x = pad + i * stepX;
    const y = h - pad - ref[i] * (h - pad * 2);
    if (i === 0) miniCtx.moveTo(x, y);
    else miniCtx.lineTo(x, y);
  }
  miniCtx.stroke();

  // Dots at each point
  miniCtx.fillStyle = '#00ffcc';
  for (let i = 0; i < ref.length; i++) {
    const x = pad + i * stepX;
    const y = h - pad - ref[i] * (h - pad * 2);
    miniCtx.beginPath();
    miniCtx.arc(x, y, 1.5, 0, Math.PI * 2);
    miniCtx.fill();
  }
}

// ── Recording Pipeline ────────────────────────────────────────────────────

async function armMic() {
  if (!audioContext) {
    audioContext = new AudioContext({ sampleRate: 44100 });
  }
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });
  recordBtn.textContent = '⏺ Record';
  recordBtn.className = 'armed';
  recordBtn.disabled = false;
}

recordBtn.addEventListener('click', async () => {
  if (!micStream) {
    try { await armMic(); } catch (err) {
      alert('Microphone access denied: ' + err.message);
      return;
    }
  }

  if (!isRecording) {
    startRecording();
  } else {
    stopRecording();
  }
});

function startRecording() {
  recordedChunks = [];
  mediaRecorder = new MediaRecorder(micStream, { mimeType: 'audio/webm;codecs=opus' });
  mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };
  mediaRecorder.onstop = () => processRecording();

  mediaRecorder.start();
  isRecording = true;
  recordBtn.textContent = '⏹ Stop';
  recordBtn.className = 'recording';
  prevBtn.disabled = true;
  nextBtn.disabled = true;
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  isRecording = false;
  recordBtn.textContent = '⏺ Record';
  recordBtn.className = 'armed';
  prevBtn.disabled = false;
  nextBtn.disabled = false;

  // Release mic to save resources
  if (micStream) {
    micStream.getTracks().forEach(t => t.stop());
    micStream = null;
  }
}

async function processRecording() {
  const blob = new Blob(recordedChunks, { type: 'audio/webm' });
  recordedChunks = [];

  try {
    // Decode the recorded audio to raw PCM
    const arrayBuffer = await blob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const sampleRate = audioBuffer.sampleRate;
    const samples = audioBuffer.getChannelData(0); // Float32Array

    // Process frame-by-frame through the pitch pipeline
    const frameSize = 2048;
    const hopSize = 1024;
    const pitchHistory = [];
    const rawPitches = [];

    for (let i = 0; i < samples.length - frameSize; i += hopSize) {
      const frame = samples.slice(i, i + frameSize);
      const result = processAudioFrame(frame, sampleRate, pitchHistory);
      rawPitches.push(result.frequency);
    }

    // Smooth the entire curve
    const smoothed = applyThreePointSmoothing(rawPitches);

    // Filter out zeros (unvoiced/silence) for normalization
    const voiced = smoothed.filter(v => v > 0);

    // Normalize to 0–1 range
    let normalized = new Array(smoothed.length).fill(0);
    if (voiced.length > 0) {
      const minP = Math.min(...voiced);
      const maxP = Math.max(...voiced);
      const range = maxP - minP || 1;
      normalized = smoothed.map(v => v > 0 ? (v - minP) / range : 0);
    }

    // Resample to exactly 100 points via linear interpolation
    const resampled = resampleArray(normalized, 100);

    // Store
    recordings.set(currentIndex, {
      nativePitchReference: resampled.map(v => Math.round(v * 10000) / 10000),
    });

    renderSidebar();
    renderMiniCanvas();
    updateExportButton();
    updateRecordingCount();

  } catch (err) {
    alert('Processing failed: ' + err.message);
  }
}

function resampleArray(arr, targetLen) {
  if (arr.length === 0) return [];
  if (arr.length === 1) return new Array(targetLen).fill(arr[0]);
  const result = new Array(targetLen);
  const step = (arr.length - 1) / (targetLen - 1);
  for (let i = 0; i < targetLen; i++) {
    const pos = i * step;
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, arr.length - 1);
    const frac = pos - lo;
    result[i] = arr[lo] + (arr[hi] - arr[lo]) * frac;
  }
  return result;
}

// ── Export ────────────────────────────────────────────────────────────────

function updateRecordingCount() {
  recordingCount.textContent = String(recordings.size);
}

function updateExportButton() {
  exportBtn.disabled = recordings.size < 24;
  updateRecordingCount();
}

exportBtn.addEventListener('click', () => {
  const presets = PRESETS.map((p, i) => {
    const rec = recordings.get(i);
    return {
      word: p.word,
      pinyin: p.pinyin,
      tones: p.tones,
      audioSrc: p.audioSrc,
      nativePitchReference: rec ? rec.nativePitchReference : [],
    };
  });

  const json = JSON.stringify({ presets }, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'presets.json';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});

// ── Keyboard navigation ───────────────────────────────────────────────────

document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowLeft')  { e.preventDefault(); navigateTo(Math.max(0, currentIndex - 1)); }
  if (e.key === 'ArrowRight') { e.preventDefault(); navigateTo(Math.min(PRESETS.length - 1, currentIndex + 1)); }
  if (e.key === ' ' || e.key === 'r') { e.preventDefault(); recordBtn.click(); }
});

// ── Init ──────────────────────────────────────────────────────────────────

prevBtn.addEventListener('click', () => navigateTo(Math.max(0, currentIndex - 1)));
nextBtn.addEventListener('click', () => navigateTo(Math.min(PRESETS.length - 1, currentIndex + 1)));

renderSidebar();
renderCard();
renderMiniCanvas();
updateExportButton();
