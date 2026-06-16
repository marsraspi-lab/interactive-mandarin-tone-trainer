#!/usr/bin/env python3
"""
Convert a PyTorch CREPE-Tiny model to an ONNX file for browser inference
via onnxruntime-web.

Usage:
    pip install torch onnx onnxruntime   # (torch only needed for ONNX export)
    python3 scripts/convert_crepe_onnx.py

Output: src/assets/models/crepe_tiny.onnx

Torchcrepe's tiny.pth is already fp16 at 1.87 MB — the fp32 ONNX will land
around 5-7 MB. If the fp32 export exceeds 10 MB, the script auto-quantizes
to uint8 (~1.5-2 MB). Otherwise it ships fp32 as-is.
"""

import os
import sys

# ── Configuration ──────────────────────────────────────────────────
INPUT_SIZE = 1024        # CREPE-Tiny's native input size (64ms @ 16kHz)
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "assets", "models", "crepe_tiny.onnx"
)
TEMP_ONNX = "/tmp/crepe_tiny_fp32.onnx"
QUANTIZE_THRESHOLD_MB = 10  # Only quantize if fp32 exceeds this

# ── Part A: Export to ONNX ─────────────────────────────────────────

def export_onnx():
    """Load CREPE-Tiny from PyTorch and export to ONNX."""
    try:
        import torch
        import torch.onnx
    except ImportError:
        print("❌ PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    print("📦 Loading CREPE-Tiny model...")

    # Try the official torchcrepe package first, fall back to crepe-pytorch
    try:
        import torchcrepe
        # torchcrepe doesn't expose a model class directly — it wraps inference
        # We need the raw model. Use the underlying architecture.
        print("   Using torchcrepe package")

        # torchcrepe uses a pre-trained model accessible via its API
        # For ONNX export we need the raw nn.Module
        # The model is a small CNN defined in the package
        from torchcrepe.model import Crepe as CrepeModel

        model = CrepeModel(capacity='tiny')
        # Load pretrained weights
        state_dict = torch.hub.load_state_dict_from_url(
            'https://github.com/maxrmorrison/torchcrepe/raw/master/pretrained/tiny.pth',
            map_location='cpu'
        )
        model.load_state_dict(state_dict)
        model.eval()

    except ImportError:
        print("   torchcrepe not found, trying crepe-pytorch...")
        try:
            from crepe_pytorch import CrepeTiny
            model = CrepeTiny(pretrained=True)
            model.eval()
        except ImportError:
            print("❌ Neither torchcrepe nor crepe-pytorch found.")
            print("   Install one: pip install torchcrepe")
            print("   Or: pip install crepe-pytorch")
            sys.exit(1)

    # Create dummy input matching CREPE's native 1024-sample window at 16 kHz
    dummy_input = torch.randn(1, INPUT_SIZE)

    print(f"   Exporting to ONNX (input shape: [1, {INPUT_SIZE}])...")

    torch.onnx.export(
        model,
        dummy_input,
        TEMP_ONNX,
        input_names=['input_audio'],
        output_names=['pitch_probabilities'],
        dynamic_axes={
            'input_audio': {0: 'batch_size'},
            'pitch_probabilities': {0: 'batch_size'}
        },
        opset_version=14,
        do_constant_folding=True,
    )

    file_size_mb = os.path.getsize(TEMP_ONNX) / (1024 * 1024)
    print(f"   ✅ FP32 ONNX exported: {file_size_mb:.1f} MB")


# ── Part B: Conditional Quantization ──────────────────────────────────

def maybe_quantize():
    """Quantize only if fp32 ONNX exceeds QUANTIZE_THRESHOLD_MB."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("❌ onnxruntime not installed. Run: pip install onnxruntime")
        sys.exit(1)

    fp32_mb = os.path.getsize(TEMP_ONNX) / (1024 * 1024)
    print(f"\n   fp32 ONNX size: {fp32_mb:.1f} MB")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    if fp32_mb <= QUANTIZE_THRESHOLD_MB:
        print(f"   ✅ Below {QUANTIZE_THRESHOLD_MB} MB threshold — shipping fp32 as-is")
        import shutil
        shutil.copy2(TEMP_ONNX, OUTPUT_PATH)
    else:
        print(f"   ⚠️  Exceeds {QUANTIZE_THRESHOLD_MB} MB threshold — quantizing to uint8...")
        quantize_dynamic(
            model_input=TEMP_ONNX,
            model_output=OUTPUT_PATH,
            weight_type=QuantType.QUInt8,
            op_types_to_quantize=['Conv', 'Gemm', 'MatMul'],
        )

    final_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"   📁 Final model: {final_mb:.1f} MB — {OUTPUT_PATH}")

    # Clean up temp file
    try:
        os.remove(TEMP_ONNX)
    except OSError:
        pass


# ── Part C: Validate ───────────────────────────────────────────────

def validate_model():
    """Run a sanity check inference on the final model."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("⚠️  Skipping validation (numpy/onnxruntime not available)")
        return

    print("🔍 Validating model with test inference...")

    session = ort.InferenceSession(OUTPUT_PATH)
    dummy_audio = np.random.randn(1, INPUT_SIZE).astype(np.float32)

    outputs = session.run(None, {'input_audio': dummy_audio})
    probs = outputs[0]

    print(f"   Input shape:  {dummy_audio.shape}")
    print(f"   Output shape: {probs.shape}")
    print(f"   Output range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"   Peak bin:     {probs.argmax()} (confidence: {probs.max():.4f})")
    print("   ✅ Validation passed")


# ── Main ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("╔══════════════════════════════════════════╗")
    print("║   CREPE-Tiny → ONNX Export              ║")
    print("╚══════════════════════════════════════════╝")
    print(f"   Target: {OUTPUT_PATH}")
    print(f"   Input size: {INPUT_SIZE} samples")
    print()

    export_onnx()
    maybe_quantize()
    validate_model()

    print("\n✅ Done! Drop crepe_tiny.onnx into src/assets/models/")
    print("   and set ENGINE='NEURAL' in the app to use it.")
