#!/usr/bin/env python3
"""
Test raw PyTorch CREPE-Tiny pitch detection against ground truth.

No ONNX, no quantization — the original torchcrepe model.
Preprocessing: trim silence, resample to 16kHz, 1024-sample windows.
Compares CREPE pitch against pYIN/autocorr ground truth.

Usage:
    cd /workspace/interactive-mandarin-tone-trainer
    .venv/bin/python scripts/test_crepe_raw.py
"""

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torchcrepe

# ── Config ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_ROOT / "src" / "assets" / "audio"
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "ground-truth.json"

CREPE_SR = 16000          # CREPE native sample rate
CREPE_HOP = 160           # 10ms at 16kHz (CREPE default stride)
CREPE_WINDOW = 1024       # CREPE input window size
MODEL_CAPACITY = "tiny"   # CREPE model size
FMIN, FMAX = 50, 550      # Mandarin vocal range
CONFIDENCE_THRESHOLD = 0.6  # Sigmoid confidence gate

DEVICE = "cpu"


# ── Audio Loading ───────────────────────────────────────────────────
def load_audio(stem: str) -> tuple[np.ndarray, int, float]:
    """Load audio file (MP3 only — WAVs are synthetic placeholders), return (y, sr, duration_s)."""
    path = AUDIO_DIR / f"{stem}.mp3"
    if not path.exists():
        raise FileNotFoundError(f"No MP3 for {stem}")
    y, sr = librosa.load(str(path), sr=None, mono=True)
    # Skip silent placeholders
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < 1e-6:
        raise ValueError(f"Silent placeholder: {stem}.wav (rms={rms:.2e})")
    duration = len(y) / sr
    return y, sr, duration


def trim_silence(y: np.ndarray, sr: int, db_thresh: float = -40, pad_ms: float = 50) -> np.ndarray:
    """Trim leading and trailing silence below db_thresh (relative to peak)."""
    # librosa.effects.trim uses top_db relative to peak
    yt, _ = librosa.effects.trim(y, top_db=abs(db_thresh))
    # Add small padding to avoid cutting off attack
    pad_samples = int(pad_ms / 1000 * sr)
    if pad_samples > 0:
        yt = np.pad(yt, (pad_samples, pad_samples), mode="constant")
    return yt


# ── Pitch Decoding ──────────────────────────────────────────────────
# CREPE outputs 360 sigmoid values per frame (C1=32.7 Hz → B7=1975.5 Hz).
# We use torchcrepe's built-in weighted_argmax decoder.

# ── CREPE inference ─────────────────────────────────────────────────
def crepe_predict(y_16k: np.ndarray) -> np.ndarray:
    """
    Run CREPE-Tiny on 16kHz audio.
    Returns pitch in Hz (0 for unvoiced).
    """
    # torchcrepe expects float32 tensor
    audio_tensor = torch.from_numpy(y_16k.astype(np.float32)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pitch, periodicity = torchcrepe.predict(
            audio=audio_tensor,
            sample_rate=CREPE_SR,
            hop_length=CREPE_HOP,
            fmin=FMIN,
            fmax=FMAX,
            model=MODEL_CAPACITY,
            decoder=torchcrepe.decode.weighted_argmax,
            return_periodicity=True,
            batch_size=2048,
            device=DEVICE,
        )

    # pitch shape: [1, N]
    pitch_np = pitch.squeeze(0).cpu().numpy()
    periodicity_np = periodicity.squeeze(0).cpu().numpy()

    # Apply confidence gate: nan or low-confidence → 0
    result = np.where(
        np.isnan(pitch_np) | (periodicity_np < CONFIDENCE_THRESHOLD),
        0.0,
        pitch_np,
    )
    return result


# ── Frame Alignment ─────────────────────────────────────────────────
def frame_time_crepe(i: int) -> float:
    """Center time of CREPE frame i in seconds."""
    return (CREPE_WINDOW / 2 + i * CREPE_HOP) / CREPE_SR


def frame_time_gt(i: int, sr: int, frame_length: int = 2048, hop_length: int = 512) -> float:
    """Center time of ground truth frame i in seconds."""
    return (frame_length / 2 + i * hop_length) / sr


def align_pitch_by_time(
    crepe_pitch: np.ndarray,
    crepe_sr: int,
    crepe_hop: int,
    crepe_window: int,
    gt_pitch: list[float],
    gt_sr: int,
    gt_frame_len: int = 2048,
    gt_hop: int = 512,
) -> tuple[list[float], list[float], int]:
    """
    For each GT frame, interpolate CREPE pitch at the corresponding time.
    Returns (aligned_crepe, aligned_gt, n_compared).
    Only compares frames where both CREPE and GT have voiced (>0) pitch.
    """
    aligned_crepe = []
    aligned_gt = []

    for i, gt_val in enumerate(gt_pitch):
        if gt_val <= 0:
            continue  # GT says unvoiced — skip

        t = frame_time_gt(i, gt_sr, gt_frame_len, gt_hop)

        # Find CREPE frame index by time
        crepe_idx = (t * crepe_sr - crepe_window / 2) / crepe_hop
        idx_lo = int(np.floor(crepe_idx))
        idx_hi = idx_lo + 1

        # Linear interpolation between two CREPE frames
        if 0 <= idx_lo < len(crepe_pitch) and 0 <= idx_hi < len(crepe_pitch):
            frac = crepe_idx - idx_lo
            crepe_val = crepe_pitch[idx_lo] * (1 - frac) + crepe_pitch[idx_hi] * frac
        elif 0 <= idx_lo < len(crepe_pitch):
            crepe_val = crepe_pitch[idx_lo]
        else:
            continue  # Out of CREPE range

        if crepe_val <= 0:
            continue  # CREPE says unvoiced

        aligned_crepe.append(float(crepe_val))
        aligned_gt.append(gt_val)

    return aligned_crepe, aligned_gt, len(aligned_crepe)


# ── Metrics ─────────────────────────────────────────────────────────
def compute_mae_hz(pred: list[float], ref: list[float]) -> float:
    """Mean absolute error in Hz."""
    if not pred:
        return float("inf")
    return float(np.mean(np.abs(np.array(pred) - np.array(ref))))


def compute_mae_relative(pred: list[float], ref: list[float]) -> float:
    """Mean absolute percentage error (relative to ref)."""
    if not pred:
        return float("inf")
    p = np.array(pred)
    r = np.array(ref)
    return float(np.mean(np.abs(p - r) / r) * 100)


def compute_mae_zscore(pred: list[float], ref: list[float]) -> float:
    """
    Mean absolute error in z-score space.
    Each curve is independently normalized to μ=0 σ=1, then compared.
    This isolates pitch contour SHAPE from absolute Hz offset.
    """
    if len(pred) < 3:
        return float("inf")
    p = np.array(pred)
    r = np.array(ref)
    p_z = (p - p.mean()) / (p.std() or 1.0)
    r_z = (r - r.mean()) / (r.std() or 1.0)
    return float(np.mean(np.abs(p_z - r_z)))


def voiced_ratio(pitch: list[float]) -> float:
    """Fraction of frames with detected pitch (>0)."""
    voiced = sum(1 for v in pitch if v > 0)
    return voiced / max(len(pitch), 1)


# ── Main ────────────────────────────────────────────────────────────
def main():
    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = json.load(f)

    results = []
    total_aligned = 0
    total_absolute_error = 0.0
    total_relative_error = 0.0
    total_zscore_error = 0.0

    print(f"{'Word':<12} {'Tones':<8} {'GT':>6} {'CREPE':>6} {'Align':>6} {'MAE Hz':>8} {'MAE%':>6} {'Z-MAE':>7} {'GT%':>5} {'CR%':>5}")
    print("-" * 85)

    for stem, gt_data in sorted(ground_truth.items()):
        gt_pitch = gt_data["pitch"]
        gt_frames = gt_data["frames"]
        gt_voiced = gt_data["voiced_frames"]
        method = gt_data["method"]
        tones = "".join(map(str, gt_data["tones"]))

        try:
            # Load audio
            y, sr, duration = load_audio(stem)

            # Trim silence
            y_trimmed = trim_silence(y, sr)

            # Resample to 16kHz
            if sr != CREPE_SR:
                y_16k = librosa.resample(y_trimmed, orig_sr=sr, target_sr=CREPE_SR)
            else:
                y_16k = y_trimmed

            # Ensure enough audio (need at least 1024 samples)
            if len(y_16k) < CREPE_WINDOW:
                y_16k = np.pad(y_16k, (0, CREPE_WINDOW - len(y_16k)))

            # Run CREPE
            crepe_pitch = crepe_predict(y_16k)

            # Align frames
            aligned_crepe, aligned_gt, n_comp = align_pitch_by_time(
                crepe_pitch, CREPE_SR, CREPE_HOP, CREPE_WINDOW,
                gt_pitch, sr,
            )

            mae_hz = compute_mae_hz(aligned_crepe, aligned_gt)
            mae_pct = compute_mae_relative(aligned_crepe, aligned_gt) if n_comp > 0 else float("inf")
            mae_zs = compute_mae_zscore(aligned_crepe, aligned_gt) if n_comp > 0 else float("inf")

            crepe_voiced = np.sum(crepe_pitch > 0)
            crepe_total = len(crepe_pitch)

            print(
                f"{stem:<12} {tones:<8} "
                f"{gt_frames:>4}/{gt_voiced:<3} "
                f"{crepe_total:>4}/{crepe_voiced:<3} "
                f"{n_comp:>6} "
                f"{mae_hz:>8.1f} "
                f"{mae_pct:>5.1f}% "
                f"{mae_zs:>7.3f}"
                f"{100*gt_voiced/max(gt_frames,1):>4.0f}%"
                f"{100*crepe_voiced/max(crepe_total,1):>4.0f}%"
            )

            if n_comp > 0 and mae_zs != float("inf"):
                results.append((stem, mae_hz, mae_zs, n_comp, tones, method))
                total_aligned += n_comp
                total_absolute_error += mae_hz * n_comp
                total_relative_error += mae_pct * n_comp
                total_zscore_error += mae_zs * n_comp

        except Exception as e:
            print(f"{stem:<12} {'':<8} ERROR: {e}")

    # Summary
    print("-" * 85)
    if total_aligned > 0:
        aggregate_mae = total_absolute_error / total_aligned
        aggregate_mae_pct = total_relative_error / total_aligned
        aggregate_zs = total_zscore_error / total_aligned
        print(f"\n{'AGGREGATE':<12} aligned={total_aligned}  MAE={aggregate_mae:.1f} Hz  ({aggregate_mae_pct:.1f}%)  Z-MAE={aggregate_zs:.3f}")

    print(f"\nPer-word MAE (Hz | Z-score):")
    results.sort(key=lambda x: x[1])  # sort by Hz MAE
    for stem, mae_hz, mae_zs, n, tones, method in results:
        bar = "█" * int(min(mae_hz / 2, 40))
        print(f"  {stem:<12} {mae_hz:>6.1f} Hz  z={mae_zs:.3f}  {bar}  [{method}]")

    # Highlight fricative words
    fricative_words = {"haizi", "jiejie", "xiexie", "shuiguo", "luxing", "jichang"}
    fric_results = [r for r in results if r[0] in fricative_words]
    if fric_results:
        print(f"\nFricative words ({len(fric_results)}):")
        for stem, mae_hz, mae_zs, n, tones, method in fric_results:
            print(f"  {stem:<12} {mae_hz:>6.1f} Hz  z={mae_zs:.3f}  ({n} aligned)")

    non_fric = [r for r in results if r[0] not in fricative_words]
    if non_fric:
        nf_total = sum(r[3] for r in non_fric)
        nf_mae = sum(r[1]*r[3] for r in non_fric) / max(nf_total, 1)
        nf_zs = sum(r[2]*r[3] for r in non_fric) / max(nf_total, 1)
        print(f"\nNon-fricative aggregate: {nf_mae:.1f} Hz  z={nf_zs:.3f}")


if __name__ == "__main__":
    main()
