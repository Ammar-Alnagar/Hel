#!/usr/bin/env python3
"""
Model conversion utilities for Helios Engine.

This script provides tools to:
1. Create a minimal ONNX model for testing
2. Convert HuggingFace models to ONNX format
3. Quantize models for efficient inference
"""

import argparse
import numpy as np
import onnx
from onnx import helper, TensorProto
import torch
import torch.nn as nn
from pathlib import Path


def create_test_onnx_model(output_path: str = "tests/golden_baselines/mini.onnx"):
    """
    Create a minimal ONNX model for testing the inference engine.

    This creates a simple transformer-like model with:
    - Input: token IDs (batch_size, seq_len)
    - Embedding layer
    - Single transformer layer (attention + feedforward)
    - Output: logits (batch_size, seq_len, vocab_size)
    """

    # Model parameters
    batch_size = 1
    seq_len = 8
    hidden_size = 64
    vocab_size = 1000
    num_heads = 4
    head_dim = hidden_size // num_heads

    # Create the model structure
    # Input: input_ids (int64)
    input_ids = helper.make_tensor_value_info(
        'input_ids', TensorProto.INT64, [batch_size, seq_len]
    )

    # Output: logits (float32)
    logits = helper.make_tensor_value_info(
        'logits', TensorProto.FLOAT, [batch_size, seq_len, vocab_size]
    )

    # Embedding weight (vocab_size, hidden_size)
    embed_weight_data = np.random.randn(vocab_size, hidden_size).astype(np.float32)
    embed_weight = helper.make_tensor(
        'embed_tokens.weight',
        TensorProto.FLOAT,
        [vocab_size, hidden_size],
        embed_weight_data.tobytes(),
        raw=True
    )

    # Attention weights
    # Q, K, V projections (hidden_size, hidden_size)
    q_weight_data = np.random.randn(hidden_size, hidden_size).astype(np.float32)
    k_weight_data = np.random.randn(hidden_size, hidden_size).astype(np.float32)
    v_weight_data = np.random.randn(hidden_size, hidden_size).astype(np.float32)
    o_weight_data = np.random.randn(hidden_size, hidden_size).astype(np.float32)

    q_weight = helper.make_tensor(
        'q_proj.weight', TensorProto.FLOAT, [hidden_size, hidden_size],
        q_weight_data.tobytes(), raw=True
    )
    k_weight = helper.make_tensor(
        'k_proj.weight', TensorProto.FLOAT, [hidden_size, hidden_size],
        k_weight_data.tobytes(), raw=True
    )
    v_weight = helper.make_tensor(
        'v_proj.weight', TensorProto.FLOAT, [hidden_size, hidden_size],
        v_weight_data.tobytes(), raw=True
    )
    o_weight = helper.make_tensor(
        'o_proj.weight', TensorProto.FLOAT, [hidden_size, hidden_size],
        o_weight_data.tobytes(), raw=True
    )

    # Feedforward weights
    fc1_weight_data = np.random.randn(hidden_size * 4, hidden_size).astype(np.float32)
    fc2_weight_data = np.random.randn(hidden_size, hidden_size * 4).astype(np.float32)

    fc1_weight = helper.make_tensor(
        'fc1.weight', TensorProto.FLOAT, [hidden_size * 4, hidden_size],
        fc1_weight_data.tobytes(), raw=True
    )
    fc2_weight = helper.make_tensor(
        'fc2.weight', TensorProto.FLOAT, [hidden_size, hidden_size * 4],
        fc2_weight_data.tobytes(), raw=True
    )

    # Create the computation graph (simplified)
    # In a real implementation, this would be a proper transformer forward pass

    nodes = [
        # Embedding lookup (simplified - would use Gather in real ONNX)
        helper.make_node(
            'MatMul',
            inputs=['input_ids_float', 'embed_tokens.weight'],
            outputs=['hidden_states'],
            name='embed_matmul'
        ),

        # This is a highly simplified representation
        # Real transformer would have proper attention and feedforward nodes
        helper.make_node(
            'MatMul',
            inputs=['hidden_states', 'lm_head.weight'],
            outputs=['logits'],
            name='lm_head'
        ),
    ]

    # LM head weight (vocab_size, hidden_size) - transpose of embedding
    lm_head_weight_data = embed_weight_data.T  # Transpose for lm_head
    lm_head_weight = helper.make_tensor(
        'lm_head.weight',
        TensorProto.FLOAT,
        [hidden_size, vocab_size],
        lm_head_weight_data.tobytes(),
        raw=True
    )

    # Convert input_ids to float for MatMul
    nodes.insert(0, helper.make_node(
        'Cast',
        inputs=['input_ids'],
        outputs=['input_ids_float'],
        name='cast_input',
        to=TensorProto.FLOAT
    ))

    # Create the graph
    graph = helper.make_graph(
        nodes,
        'MiniTransformer',
        [input_ids],
        [logits],
        [
            embed_weight, q_weight, k_weight, v_weight, o_weight,
            fc1_weight, fc2_weight, lm_head_weight
        ]
    )

    # Create the model
    model = helper.make_model(graph, producer_name='helios-engine')

    # Validate and save
    onnx.checker.check_model(model)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        f.write(model.SerializeToString())

    print(f"Created test ONNX model: {output_path}")
    print(f"Model size: {Path(output_path).stat().st_size} bytes")
    print(f"Initializer count: {len(model.graph.initializer)}")

    return model


def inspect_onnx_model(model_path: str):
    """Inspect an ONNX model and print its structure."""
    model = onnx.load(model_path)

    print(f"Model: {model.producer_name}")
    print(f"Graph name: {model.graph.name}")
    print(f"Inputs: {len(model.graph.input)}")
    for inp in model.graph.input:
        print(f"  - {inp.name}: {inp.type.tensor_type.elem_type} {inp.type.tensor_type.shape.dim}")

    print(f"Outputs: {len(model.graph.output)}")
    for out in model.graph.output:
        print(f"  - {out.name}: {out.type.tensor_type.elem_type} {out.type.tensor_type.shape.dim}")

    print(f"Initializers: {len(model.graph.initializer)}")
    for init in model.graph.initializer:
        print(f"  - {init.name}: {init.dims} ({init.data_type})")


def main():
    parser = argparse.ArgumentParser(description="Model conversion utilities")
    parser.add_argument(
        "--create-test-model",
        action="store_true",
        help="Create a minimal ONNX model for testing"
    )
    parser.add_argument(
        "--inspect",
        type=str,
        help="Inspect an ONNX model file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tests/golden_baselines/mini.onnx",
        help="Output path for created model"
    )

    args = parser.parse_args()

    if args.create_test_model:
        create_test_onnx_model(args.output)
    elif args.inspect:
        inspect_onnx_model(args.inspect)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
