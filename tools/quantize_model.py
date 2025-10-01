#!/usr/bin/env python3
"""
Model quantization utilities for Helios Engine.

This script provides tools to quantize PyTorch models to formats
supported by the Helios Engine (Q4, Q8, etc.).
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json


def quantize_weight_q4(weight: torch.Tensor, scale: float = None) -> tuple:
    """
    Quantize a weight tensor to Q4 format (4-bit signed integers).

    Args:
        weight: Input weight tensor (float32)
        scale: Optional scale factor (computed if None)

    Returns:
        Tuple of (quantized_bytes, scale_factor)
    """
    # Flatten weight for processing
    weight_flat = weight.flatten().float()

    if scale is None:
        # Compute scale based on max absolute value
        max_val = torch.max(torch.abs(weight_flat))
        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / 7.0  # Range -8 to 7 for 4-bit signed

    # Quantize to 4-bit values
    quantized = torch.round(weight_flat / scale)
    quantized = torch.clamp(quantized, -8, 7)

    # Pack two 4-bit values into one byte
    packed_bytes = torch.zeros((len(quantized) + 1) // 2, dtype=torch.uint8)

    for i in range(0, len(quantized), 2):
        byte_val = 0
        # First nibble
        nib1 = int(quantized[i].item()) & 0x0F
        byte_val |= nib1

        # Second nibble (if exists)
        if i + 1 < len(quantized):
            nib2 = int(quantized[i + 1].item()) & 0x0F
            byte_val |= (nib2 << 4)

        packed_bytes[i // 2] = byte_val

    return packed_bytes.numpy(), float(scale)


def quantize_weight_q8(weight: torch.Tensor, scale: float = None) -> tuple:
    """
    Quantize a weight tensor to Q8 format (8-bit signed integers).
    """
    weight_flat = weight.flatten().float()

    if scale is None:
        max_val = torch.max(torch.abs(weight_flat))
        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / 127.0  # Range -128 to 127 for 8-bit signed

    quantized = torch.round(weight_flat / scale)
    quantized = torch.clamp(quantized, -128, 127)

    return quantized.to(torch.int8).numpy(), float(scale)


def quantize_model_q4(model_path: str, output_dir: str = "quantized_models") -> dict:
    """
    Quantize a PyTorch model to Q4 format for Helios Engine.

    Args:
        model_path: Path to PyTorch model (.pt or .pth file)
        output_dir: Output directory for quantized model

    Returns:
        Dictionary with quantization statistics
    """
    print(f"Loading model: {model_path}")
    model_state = torch.load(model_path, map_location='cpu')

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    quantized_weights = {}
    stats = {
        "total_weights": 0,
        "quantized_weights": 0,
        "compression_ratio": 0.0,
        "scales": {}
    }

    total_original_size = 0
    total_quantized_size = 0

    for name, weight in model_state.items():
        if weight.dtype in [torch.float32, torch.float16]:
            print(f"Quantizing {name}: {weight.shape}")

            # Skip biases and small tensors
            if weight.numel() < 16:
                print(f"  Skipping small tensor: {name}")
                continue

            stats["total_weights"] += 1
            original_size = weight.numel() * weight.element_size()

            if "embed" in name.lower() or "lm_head" in name.lower():
                # Use Q8 for embeddings and output layer for better quality
                quantized_data, scale = quantize_weight_q8(weight)
                quantized_weights[name] = {
                    "data": quantized_data,
                    "scale": scale,
                    "dtype": "q8"
                }
                quantized_size = len(quantized_data)
            else:
                # Use Q4 for hidden layers
                quantized_data, scale = quantize_weight_q4(weight)
                quantized_weights[name] = {
                    "data": quantized_data,
                    "scale": scale,
                    "dtype": "q4"
                }
                quantized_size = len(quantized_data)

            stats["quantized_weights"] += 1
            stats["scales"][name] = scale

            total_original_size += original_size
            total_quantized_size += quantized_size

    # Calculate compression ratio
    if total_original_size > 0:
        stats["compression_ratio"] = total_quantized_size / total_original_size

    # Save quantized model
    model_file = output_path / "quantized_model.json"
    with open(model_file, 'w') as f:
        json.dump({
            "weights": quantized_weights,
            "stats": stats,
            "format": "helios-q4"
        }, f, indent=2)

    print(f"\nQuantization complete!")
    print(f"Original size: {total_original_size} bytes")
    print(f"Quantized size: {total_quantized_size} bytes")
    print(f"Compression ratio: {stats['compression_ratio']".2f"}x")
    print(f"Model saved to: {model_file}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Quantize PyTorch models for Helios Engine")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to PyTorch model file (.pt/.pth)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="quantized_models",
        help="Output directory for quantized model"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["q4", "q8"],
        default="q4",
        help="Quantization format"
    )

    args = parser.parse_args()

    # Check if model file exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return 1

    # Quantize model
    stats = quantize_model_q4(args.model, args.output_dir)

    print("
📊 Quantization Summary:"    print(f"  Weights processed: {stats['total_weights']}")
    print(f"  Weights quantized: {stats['quantized_weights']}")
    print(f"  Compression ratio: {stats['compression_ratio']".2f"}x")

    return 0


if __name__ == "__main__":
    exit(main())
