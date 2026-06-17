#!/usr/bin/env python3
"""
Export the exact audio that CREPE receives after preprocessing.

For each MP3 in a directory:
  1. Load → trim silence → resample to 16kHz → save WAV
  2. Optionally run CREPE and report detection stats

Usage:
    # Just export cleaned audio (no CREPE inference)
    python scripts/export_crepe_audio.py train/validation

    # Export + run CREPE stats
    python scripts/export_crepe_audio.py train/validation --stats

    # Export from multiple directories
    python scripts/export_crepe_audio.py train/reference train/validation train/validation-male

    # Custom output directory
    python scripts/export_crepe_audio.py train/validation -o ./crepe_inputs

Requirements:
    pip install librosa soundfile numpy
    pip install torch torchcrepe    # only if using --stats
"""

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


CREPE_SR = 16000
TRIM_DB = 40           # silence threshold in -dB (relative to peak)
PAD_MS = 50            # padding after trim to preserve attack


def preprocess(y: np.ndarray, sr: int) -> np.ndarray:
    """Trim silence, add padding, resample to 16kHz."""
    yt, _ = librosa.effects.trim(y, top_db=TRIM_DB)
    pad_samples = int(PAD_MS / 1000 * sr)
    if pad_samples > 0:
        yt = np.pad(yt, (pad_samples, pad_samples), mode="constant")
    if sr != CREPE_SR:
        yt = librosa.resample(yt, orig_sr=sr, target_sr=CREPE_SR)
    return yt.astype(np.float32)


def run_crepe(y_16k: np.ndarray) -> dict:
    """Run CREPE-Tiny and return per-frame pitch + stats."""
    import torch
    import torchcrepe

    audio_t = torch.from_numpy(y_16k).unsqueeze(0)
    with torch.no_grad():
        pitch, periodicity = torchcrepe.predict(
            audio=audio_t,
            sample_rate=CREPE_SR,
            hop_length=160,         # 10ms stride
            fmin=50, fmax=550,
            model="tiny",
            decoder=torchcrepe.decode.weighted_argmax,
            return_periodicity=True,
            batch_size=2048,
            device="cpu",
        )

    pitch_np = pitch.squeeze(0).numpy()
    per_np = periodicity.squeeze(0).numpy()

    voiced = (~np.isnan(pitch_np)) & (per_np >= 0.3)
    voiced_hz = pitch_np[voiced]

    return {
        "pitch": pitch_np.tolist(),
        "periodicity": per_np.tolist(),
        "total_frames": len(pitch_np),
        "voiced_frames": int(np.sum(voiced)),
        "voiced_pct": float(100 * np.sum(voiced) / max(len(pitch_np), 1)),
        "median_hz": float(np.median(voiced_hz)) if len(voiced_hz) > 0 else 0,
        "mean_hz": float(np.mean(voiced_hz)) if len(voiced_hz) > 0 else 0,
        "std_hz": float(np.std(voiced_hz)) if len(voiced_hz) > 1 else 0,
        "mean_conf": float(np.mean(per_np[voiced])) if len(voiced_hz) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Export CREPE-preprocessed audio")
    parser.add_argument("dirs", nargs="+", help="Directories containing .mp3 files")
    parser.add_argument("-o", "--out", default="crepe_audio_export",
                        help="Output root directory (default: ./crepe_audio_export)")
    parser.add_argument("--stats", action="store_true",
                        help="Also run CREPE inference and print stats")
    parser.add_argument("--no-wav", action="store_true",
                        help="Skip WAV export, only print stats")
    args = parser.parse_args()

    out_root = Path(args.out)
    n_exported = 0
    n_errors = 0

    for dir_path_str in args.dirs:
        dir_path = Path(dir_path_str)
        if not dir_path.is_dir():
            print(f"✗ Not a directory: {dir_path}")
            continue

        mp3s = sorted(dir_path.glob("*.mp3"))
        if not mp3s:
            print(f"✗ No .mp3 files in {dir_path}")
            continue

        # Output subdirectory named after the source directory
        out_dir = out_root / dir_path.name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*65}")
        print(f"  {dir_path.name}/  ({len(mp3s)} files)")
        print(f"{'='*65}")

        if args.stats:
            print(f"{'Word':<12} {'Dur':>5} {'V%':>5} {'Med Hz':>7} {'Std':>6} {'Conf':>6}")

        for mp3 in mp3s:
            stem = mp3.stem
            try:
                # Load
                y, sr = librosa.load(str(mp3), sr=None, mono=True)
                dur_orig = len(y) / sr

                # Preprocess
                y_16k = preprocess(y, sr)
                dur_16k = len(y_16k) / CREPE_SR

                # Export WAV
                if not args.no_wav:
                    wav_path = out_dir / f"{stem}_16kHz.wav"
                    sf.write(str(wav_path), y_16k, CREPE_SR)
                    n_exported += 1

                # Stats
                if args.stats:
                    s = run_crepe(y_16k)
                    print(f"{stem:<12} {dur_16k:>4.1f}s {s['voiced_pct']:>4.0f}% "
                          f"{s['median_hz']:>7.0f} {s['std_hz']:>6.1f} {s['mean_conf']:>6.3f}")
                else:
                    print(f"  ✓ {stem:<12} {dur_orig:.1f}s@{sr}Hz → 16kHz {dur_16k:.1f}s")

            except Exception as e:
                print(f"  ✗ {stem:<12} ERROR: {e}")
                n_errors += 1

    print(f"\nDone: {n_exported} WAVs exported to {out_root.resolve()}/")
    if n_errors:
        print(f"      {n_errors} errors")


if __name__ == "__main__":
    main()
