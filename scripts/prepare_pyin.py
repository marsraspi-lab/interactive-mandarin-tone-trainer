#!/usr/bin/env python3
"""
One-time ground-truth extraction via pYIN (librosa).
Processes all 24 Mandarin audio files through pYIN to produce
high-quality pitch curves for AUTORESEARCH training.

Output: data/ground-truth.json
Format: { "word_key": { "pinyin": "...", "tones": [...], "pitch": [f0_hz, ...] } }
"""
import json
import sys
import os
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_ROOT / "src" / "assets" / "audio"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "ground-truth.json"

# Mandarin word definitions: filename stem → {pinyin, tones, characters}
WORDS = {
    "ma1":      {"pinyin": "mā",       "tones": [1],    "word": "妈"},
    "ma2":      {"pinyin": "má",       "tones": [2],    "word": "麻"},
    "ma3":      {"pinyin": "mǎ",       "tones": [3],    "word": "马"},
    "ma4":      {"pinyin": "mà",       "tones": [4],    "word": "骂"},
    "diannao":  {"pinyin": "diànnǎo",  "tones": [4, 3], "word": "电脑"},
    "gege":     {"pinyin": "gēge",     "tones": [1, 5], "word": "哥哥"},
    "gongsi":   {"pinyin": "gōngsī",   "tones": [1, 1], "word": "公司"},
    "haizi":    {"pinyin": "háizi",    "tones": [2, 5], "word": "孩子"},
    "hanyu":    {"pinyin": "hànyǔ",    "tones": [4, 3], "word": "汉语"},
    "jichang":  {"pinyin": "jīchǎng",  "tones": [1, 3], "word": "机场"},
    "jiejie":   {"pinyin": "jiějie",   "tones": [3, 5], "word": "姐姐"},
    "jinnian":  {"pinyin": "jīnnián",  "tones": [1, 2], "word": "今年"},
    "jintian":  {"pinyin": "jīntiān",  "tones": [1, 1], "word": "今天"},
    "jinzhang": {"pinyin": "jǐnzhāng", "tones": [3, 1], "word": "紧张"},
    "jueding":  {"pinyin": "juédìng",  "tones": [2, 4], "word": "决定"},
    "laoshi":   {"pinyin": "lǎoshī",   "tones": [3, 1], "word": "老师"},
    "luxing":   {"pinyin": "lǚxíng",   "tones": [3, 2], "word": "旅行"},
    "mingnian": {"pinyin": "míngnián", "tones": [2, 2], "word": "明年"},
    "pingguo":  {"pinyin": "píngguǒ",  "tones": [2, 3], "word": "苹果"},
    "shuiguo":  {"pinyin": "shuǐguǒ",  "tones": [3, 3], "word": "水果"},
    "wenti":    {"pinyin": "wèntí",    "tones": [4, 2], "word": "问题"},
    "xiexie":   {"pinyin": "xièxie",   "tones": [4, 5], "word": "谢谢"},
    "yinhang":  {"pinyin": "yínháng",  "tones": [2, 2], "word": "银行"},
    "zaijian":  {"pinyin": "zàijiàn",  "tones": [4, 4], "word": "再见"},
}


def load_audio(filepath: Path) -> tuple[np.ndarray, int]:
    """Load audio file (WAV or MP3), converting MP3→WAV via ffmpeg if needed."""
    ext = filepath.suffix.lower()
    if ext == '.wav':
        y, sr = librosa.load(str(filepath), sr=None, mono=True)
        return y, sr
    elif ext == '.mp3':
        # librosa can handle MP3 if ffmpeg is present
        try:
            y, sr = librosa.load(str(filepath), sr=None, mono=True)
            return y, sr
        except Exception:
            # Fallback: convert MP3→WAV with ffmpeg
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
            try:
                subprocess.run(
                    ['ffmpeg', '-y', '-i', str(filepath), '-ac', '1', '-ar', '44100',
                     '-sample_fmt', 's16', tmp_path],
                    capture_output=True, check=True, timeout=30
                )
                y, sr = librosa.load(tmp_path, sr=None, mono=True)
                return y, sr
            finally:
                os.unlink(tmp_path)
    else:
        raise ValueError(f"Unsupported format: {ext}")


def extract_pyin_pitch(filepath: Path) -> list[float]:
    """
    Extract f0 curve using pYIN with tuned parameters for Mandarin speech.
    Returns list of pitch values in Hz (NaN → 0 for unvoiced).
    """
    y, sr = load_audio(filepath)

    # pYIN parameters tuned for Mandarin vocal range (60–400 Hz)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=60,
        fmax=400,
        sr=sr,
        frame_length=2048,
        hop_length=512,
        fill_na=0.0,  # unvoiced → 0 Hz
    )

    # Round to 1 decimal place for clean storage
    pitch = [round(float(f), 1) if f > 0 else 0.0 for f in f0]
    return pitch


def extract_autocorr_fallback(filepath: Path) -> list[float]:
    """Fallback pitch detection using autocorrelation for files pYIN fails on."""
    y, sr = load_audio(filepath)
    frame_len = 2048
    hop_len = 512
    pitches = []

    for start in range(0, len(y) - frame_len, hop_len):
        frame = y[start:start + frame_len]
        rms = np.sqrt(np.mean(frame**2))
        if rms < 0.003:
            pitches.append(0.0)
            continue

        # Autocorrelation
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]  # Keep only positive lags
        corr[0] = 0  # Zero out lag-0

        # Search 60-400 Hz range
        min_lag = int(sr / 400)
        max_lag = int(sr / 60)
        if max_lag >= len(corr):
            max_lag = len(corr) - 1
        if min_lag >= max_lag:
            pitches.append(0.0)
            continue

        peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
        freq = sr / peak_lag if peak_lag > 0 else 0
        pitches.append(round(float(freq), 1) if 60 <= freq <= 400 else 0.0)

    return pitches


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ground_truth = {}
    total_voiced = 0
    total_frames = 0

    for stem, info in WORDS.items():
        # Prefer MP3 (real recordings) over WAV (test fixtures / placeholders)
        mp3_path = AUDIO_DIR / f"{stem}.mp3"
        wav_path = AUDIO_DIR / f"{stem}.wav"

        if mp3_path.exists():
            filepath = mp3_path
        elif wav_path.exists():
            filepath = wav_path
        else:
            print(f"✗ {stem}: no audio file found", file=sys.stderr)
            continue

        print(f"  Processing {stem} ({info['pinyin']})... ", end='', flush=True)
        try:
            pitch = extract_pyin_pitch(filepath)
            voiced = sum(1 for v in pitch if v > 0)

            # Fallback: if pYIN finds nothing, try autocorrelation
            if voiced == 0:
                print(f"pYIN: 0 voiced, trying autocorrelation... ", end='', flush=True)
                pitch = extract_autocorr_fallback(filepath)
                voiced = sum(1 for v in pitch if v > 0)
                method = "autocorr"
            else:
                method = "pYIN"
            total_voiced += voiced
            total_frames += len(pitch)

            ground_truth[stem] = {
                "word": info["word"],
                "pinyin": info["pinyin"],
                "tones": info["tones"],
                "pitch": pitch,
                "frames": len(pitch),
                "voiced_frames": voiced,
                "method": method,
            }
            print(f"✓ {len(pitch)} frames, {voiced} voiced ({100*voiced/max(len(pitch),1):.0f}%) [{method}]")
        except Exception as e:
            print(f"✗ ERROR: {e}", file=sys.stderr)

    # Write ground truth
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Ground truth written → {OUTPUT_PATH}")
    print(f"   {len(ground_truth)}/{len(WORDS)} words processed")
    if total_frames > 0:
        print(f"   Total voiced frames: {total_voiced}/{total_frames} ({100*total_voiced/max(total_frames,1):.1f}%)")


if __name__ == '__main__':
    main()
