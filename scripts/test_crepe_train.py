#!/usr/bin/env python3
"""
Run CREPE-Tiny on all three train/ directories and compare.

Usage:
    .venv/bin/python scripts/test_crepe_train.py
"""

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torchcrepe

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CREPE_SR = 16000
CREPE_HOP = 160  # 10ms
FMIN, FMAX = 50, 550
MODEL = "tiny"
CONFIDENCE = 0.3
DEVICE = "cpu"


def load_and_prep(path: Path) -> tuple[np.ndarray, int, float]:
    """Load MP3, trim silence, resample to 16kHz."""
    y, sr = librosa.load(str(path), sr=None, mono=True)
    rms = float(np.sqrt(np.mean(y ** 2)))
    duration = len(y) / sr

    # Trim silence
    yt, _ = librosa.effects.trim(y, top_db=40)
    yt = np.pad(yt, (int(0.05 * sr), int(0.05 * sr)), mode="constant")

    # Resample
    if sr != CREPE_SR:
        y_16k = librosa.resample(yt, orig_sr=sr, target_sr=CREPE_SR)
    else:
        y_16k = yt

    if len(y_16k) < 1024:
        y_16k = np.pad(y_16k, (0, 1024 - len(y_16k)))

    return y_16k.astype(np.float32), sr, duration


def run_crepe(y_16k: np.ndarray) -> dict:
    """Run CREPE-Tiny, return stats."""
    audio_t = torch.from_numpy(y_16k).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pitch, periodicity = torchcrepe.predict(
            audio=audio_t,
            sample_rate=CREPE_SR,
            hop_length=CREPE_HOP,
            fmin=FMIN,
            fmax=FMAX,
            model=MODEL,
            decoder=torchcrepe.decode.weighted_argmax,
            return_periodicity=True,
            batch_size=2048,
            device=DEVICE,
        )

    pitch_np = pitch.squeeze(0).cpu().numpy()
    per_np = periodicity.squeeze(0).cpu().numpy()

    # Confidence gate
    voiced_mask = (~np.isnan(pitch_np)) & (per_np >= CONFIDENCE)
    voiced_hz = pitch_np[voiced_mask]

    return {
        "total_frames": len(pitch_np),
        "voiced_frames": int(np.sum(voiced_mask)),
        "voiced_pct": float(100 * np.sum(voiced_mask) / max(len(pitch_np), 1)),
        "median_hz": float(np.median(voiced_hz)) if len(voiced_hz) > 0 else 0,
        "mean_hz": float(np.mean(voiced_hz)) if len(voiced_hz) > 0 else 0,
        "std_hz": float(np.std(voiced_hz)) if len(voiced_hz) > 1 else 0,
        "min_hz": float(np.min(voiced_hz)) if len(voiced_hz) > 0 else 0,
        "max_hz": float(np.max(voiced_hz)) if len(voiced_hz) > 0 else 0,
        "mean_conf": float(np.mean(per_np[voiced_mask])) if len(voiced_hz) > 0 else 0,
    }


def main():
    dirs = {
        "reference": PROJECT_ROOT / "train" / "reference",
        "validation": PROJECT_ROOT / "train" / "validation",
        "validation-male": PROJECT_ROOT / "train" / "validation-male",
    }

    # Load human ratings
    rating_path = PROJECT_ROOT / "train" / "validation-male" / "rating.txt"
    ratings = {}
    if rating_path.exists():
        for line in rating_path.read_text().splitlines():
            line = line.strip()
            if "=" in line:
                word, rating = line.split("=")
                ratings[word.strip()] = int(rating.strip())

    all_results = {}

    for dir_name, dir_path in dirs.items():
        print(f"\n{'='*80}")
        print(f"  {dir_name.upper()}  ({dir_path})")
        print(f"{'='*80}")
        print(f"{'Word':<12} {'Dur':>5} {'Frames':>7} {'Voiced':>7} {'V%':>5} {'Med Hz':>7} {'Mean Hz':>7} {'Std':>6} {'Range':>14} {'Conf':>5} {'Rate':>1}")
        print("-" * 85)

        dir_results = {}
        for mp3 in sorted(dir_path.glob("*.mp3")):
            stem = mp3.stem
            y_16k, sr, dur = load_and_prep(mp3)
            stats = run_crepe(y_16k)

            dir_results[stem] = stats

            rating_str = ""
            if stem in ratings:
                labels = {1: "✓", 2: "~", 3: "✗"}
                rating_str = labels.get(ratings[stem], "?")

            print(
                f"{stem:<12} {dur:>4.1f}s "
                f"{stats['total_frames']:>5}/{stats['voiced_frames']:<5} "
                f"{stats['voiced_pct']:>4.0f}% "
                f"{stats['median_hz']:>7.0f} "
                f"{stats['mean_hz']:>7.0f} "
                f"{stats['std_hz']:>6.1f} "
                f"{stats['min_hz']:>5.0f}-{stats['max_hz']:<5.0f} "
                f"{stats['mean_conf']:>5.3f} "
                f"{rating_str}"
            )

        all_results[dir_name] = dir_results

        # Aggregate stats
        voiced_pcts = [s["voiced_pct"] for s in dir_results.values()]
        confs = [s["mean_conf"] for s in dir_results.values() if s["mean_conf"] > 0]
        print("-" * 85)
        print(f"{'AGGREGATE':<12}        "
              f"voiced={np.mean(voiced_pcts):.0f}% ±{np.std(voiced_pcts):.0f}%  "
              f"conf={np.mean(confs):.3f}  "
              f"({len(dir_results)} words)")

    # ── Cross-directory comparison ──────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  CROSS-DIRECTORY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Word':<12} {'Ref V%':>6} {'Val V%':>6} {'Val-M V%':>6}  {'Ref Hz':>7} {'Val Hz':>7} {'Val-M Hz':>7}  {'Rating':>7}")
    print("-" * 85)

    for stem in sorted(all_results["reference"].keys()):
        r = all_results["reference"].get(stem, {})
        v = all_results["validation"].get(stem, {})
        vm = all_results["validation-male"].get(stem, {})
        rating = ratings.get(stem, 0)
        labels = {1: "PERFECT", 2: "FLAWED", 3: "FAIL"}
        rating_label = labels.get(rating, f"({rating})")

        print(
            f"{stem:<12} "
            f"{r.get('voiced_pct',0):>5.0f}% "
            f"{v.get('voiced_pct',0):>5.0f}% "
            f"{vm.get('voiced_pct',0):>5.0f}%  "
            f"{r.get('median_hz',0):>7.0f} "
            f"{v.get('median_hz',0):>7.0f} "
            f"{vm.get('median_hz',0):>7.0f}  "
            f"{rating_label}"
        )

    # ── Human rating alignment ──────────────────────────────────────
    if ratings:
        print(f"\n{'='*80}")
        print(f"  HUMAN RATING vs CREPE STATS (validation-male)")
        print(f"{'='*80}")

        for rating_val, label in [(1, "PERFECT"), (2, "FLAWED"), (3, "FAIL")]:
            words = [w for w, r in ratings.items() if r == rating_val]
            if not words:
                continue
            vpcts = [all_results["validation-male"][w]["voiced_pct"] for w in words if w in all_results["validation-male"]]
            confs = [all_results["validation-male"][w]["mean_conf"] for w in words if w in all_results["validation-male"] and all_results["validation-male"][w]["mean_conf"] > 0]
            stds = [all_results["validation-male"][w]["std_hz"] for w in words if w in all_results["validation-male"]]
            print(f"\n{label} ({len(words)} words):")
            print(f"  Voiced: {np.mean(vpcts):.0f}% ±{np.std(vpcts):.0f}%")
            print(f"  Conf:   {np.mean(confs):.3f}")
            print(f"  Hz std: {np.mean(stds):.1f} ±{np.std(stds):.1f}")
            for w in words:
                if w in all_results["validation-male"]:
                    s = all_results["validation-male"][w]
                    print(f"    {w:<12} v={s['voiced_pct']:.0f}%  conf={s['mean_conf']:.3f}  hz={s['median_hz']:.0f}  σ={s['std_hz']:.1f}")


if __name__ == "__main__":
    main()
