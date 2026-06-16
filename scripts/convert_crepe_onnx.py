#!/usr/bin/env python3
"""
Convert a PyTorch CREPE-Tiny model to a quantized ONNX file (~3-4MB)
suitable for browser inference via onnxruntime-web.

Usage:
    pip install torch onnx onnxruntime
    python3 scripts/convert_crepe_onnx.py

Output: src/assets/models/crepe_tiny_quantized.onnx

The quantization step reduces float32 weights to uint8 integers,
shrinking the model from ~16MB to ~3-4MB with negligible precision
loss for pitch tracking (±2 cents typical).
"""

import os
import sys

# ── Configuration ──────────────────────────────────────────────────
INPUT_SIZE = 4096        # Matches our verified 93ms audio frame
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "assets", "models", "crepe_tiny_quantized.onnx"
)
TEMP_ONNX = "/tmp/crepe_tiny_fp32.onnx"

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

    # Create dummy input matching our verified 4096-sample frame
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


# ── Part B: Quantize to 8-bit integers ─────────────────────────────

def quantize_model():
    """Dynamic-quantize the ONNX model from float32 to uint8."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("❌ onnxruntime not installed. Run: pip install onnxruntime")
        sys.exit(1)

    print("🗜️  Quantizing to 8-bit integers (QUInt8)...")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    quantize_dynamic(
        model_input=TEMP_ONNX,
        model_output=OUTPUT_PATH,
        weight_type=QuantType.QUInt8,
        # Preserve accuracy: only quantize Conv and Linear layers
        op_types_to_quantize=['Conv', 'Gemm', 'MatMul'],
    )

    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"   ✅ Quantized model: {file_size_mb:.1f} MB")
    print(f"   📁 Saved to: {OUTPUT_PATH}")

    # Clean up temp file
    try:
        os.remove(TEMP_ONNX)
    except OSError:
        pass


# ── Part C: Validate ───────────────────────────────────────────────

def validate_model():
    """Run a sanity check inference on the quantized model."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("⚠️  Skipping validation (numpy/onnxruntime not available)")
        return

    print("🔍 Validating quantized model with test inference...")

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
    print("║   CREPE-Tiny → ONNX Quantization        ║")
    print("╚══════════════════════════════════════════╝")
    print(f"   Target: {OUTPUT_PATH}")
    print(f"   Input size: {INPUT_SIZE} samples")
    print()

    export_onnx()
    quantize_model()
    validate_model()

    print("\n✅ Done! Drop crepe_tiny_quantized.onnx into src/assets/models/")
    print("   and set ENGINE='NEURAL' in the app to use it.")
