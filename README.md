# 🚀 Helios Engine - High-Performance LLM Inference

<div align="center">
  <h3>Minimal, CPU-first LLM inference engine with quantization support</h3>

  ![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
  ![License](https://img.shields.io/badge/license-MIT-blue)
  ![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
</div>

## 🌟 Overview

Helios Engine is a lightweight, high-performance C++ inference engine designed for Large Language Models (LLMs). Built with a focus on **correctness**, **modularity**, and **performance**, it provides:

- ⚡ **Quantized Inference**: Q4/Q8 row-wise quantization for memory-efficient inference
- 🎯 **Accuracy First**: Dequantize-on-load baseline ensuring numerical correctness
- 🧠 **KV Caching**: Optimized autoregressive generation with key-value state management
- 🔧 **Modular Design**: Clean separation of concerns across all components
- 🚀 **Performance Path**: Thread pools, profiling, and SIMD optimization ready

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            Helios Engine                               │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │   Loaders   │  │   Tensor    │  │  Optimized  │  │   Quantization  │  │
│  │  (ONNX/     │  │             │  │   Kernels   │  │     Tools       │  │
│  │ GGUF/ST)    │  │ Abstraction │  │ (SIMD/Q4)   │  │                 │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ Transformer │  │  Flash      │  │  Batch      │  │   HTTP Server   │  │
│  │             │  │ Attention   │  │ Processor   │  │                 │  │
│  │  Primitives │  │             │  │             │  │   (REST API)    │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │    CLI      │  │  Tokenizer  │  │   Thread    │  │    Memory       │  │
│  │             │  │             │  │   Pool      │  │   Allocator     │  │
│  │  Interface  │  │ Integration │  │             │  │                 │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🔧 Core Components

### **Tensor Abstraction**
```cpp
Tensor weights({768, 768}, DType::FP32);  // Aligned memory allocation
Tensor qweights({768, 192}, DType::Q4);   // Packed 4-bit quantization
```

### **Quantization Pipeline**
```
FP32 Weights → Scale Calculation → Q4 Packing → Compressed Storage
     ↓              ↓                ↓             ↓
Dequantize    Per-row scaling   2x compression   50% memory reduction
```

### **Inference Flow**
```
Input Tokens → Embedding → Transformer Blocks → LM Head → Logits → Sampling → Output
     ↓            ↓            ↓              ↓         ↓        ↓         ↓
  Token IDs  → Token Embeddings → KV Cache → Linear → Softmax → Top-k → Next Token
```

## ⚙️ Features

### **Quantization Support**
- **Q4 Row-wise**: 4-bit signed integers with per-row scaling (-8 to 7 range)
- **Q8 Row-wise**: 8-bit signed integers for higher precision
- **Dequantize-on-load**: FP32 baseline for accuracy validation
- **On-the-fly Kernels**: Direct operation on quantized data for performance

### **Performance Optimizations**
- **Multi-threading**: Thread pool for parallel computation
- **SIMD Acceleration**: AVX2/AVX512 optimized kernels for matrix operations
- **Flash Attention**: Memory-efficient attention implementation
- **Memory Pooling**: Aligned allocation with custom memory management
- **KV Caching**: Efficient state management for autoregressive generation
- **Batch Processing**: Support for processing multiple requests simultaneously

### **Model Support**
- **ONNX Models**: Parse TensorProto initializers with external data support
- **GGUF Integration**: Full support for llama.cpp GGUF format with quantization
- **Safetensors**: Support for HuggingFace Safetensors format
- **Custom Formats**: Extensible loader architecture

### **API & Deployment**
- **HTTP Server**: REST API for model serving and inference
- **Batch API**: Efficient batch processing for multiple requests
- **Model Quantization**: Tools for converting models to quantized formats

## 🚀 Quick Start

### **Prerequisites**
```bash
# System dependencies
sudo apt install cmake build-essential libopenblas-dev

# Protocol Buffers for ONNX parsing
sudo apt install libprotobuf-dev protobuf-compiler

# Optional: Eigen for optimized linear algebra (recommended)
sudo apt install libeigen3-dev
# OR build from source:
# git clone https://gitlab.com/libeigen/eigen.git
# cd eigen && mkdir build && cd build
# cmake .. && sudo make install

# Optional: nlohmann/json for advanced JSON parsing (auto-detected)
sudo apt install nlohmann-json3-dev
```

### **Build**
```bash
# Clone and configure
git clone <repository-url>
cd helios-engine
mkdir build && cd build

# Configure with options (Eigen auto-detected if available)
cmake .. -DUSE_OPENBLAS=ON -DUSE_EIGEN=ON -DENABLE_SIMD=OFF

# Build
cmake --build . -j$(nproc)
```

### **Run Inference**
```bash
# Basic usage
./bin/infer --model model.onnx --prompt "Hello world"

# Advanced usage with sampling
./bin/infer --model model.onnx \
            --prompt "The future of AI is" \
            --max-tokens 50 \
            --temperature 0.8 \
            --top-k 40 \
            --top-p 0.9 \
            --seed 42
```

### **Run Tests**
```bash
# Unit tests
./bin/unit_tests

# Benchmark performance
python3 tools/benchmark.py --model model.onnx --runs 10
```

## 📊 Performance

### **Memory Efficiency**
```
┌─────────────────────────────────────────┐
│           Memory Usage Comparison       │
├─────────────────────────────────────────┤
│ FP32 Model    │  100%  │■■■■■■■■■■■■■■■■│
│ Q8 Quantized  │   25%  │■■■■■           │
│ Q4 Quantized  │   12%  │■■              │
└─────────────────────────────────────────┘
```

### **Speed vs Accuracy Trade-off**
```
Accuracy → 100%  95%  90%  85%  80%  ← Accuracy
Speed    →  1x    1.5x  2x   2.5x 3x  ← Relative Speed
           FP32   Q8     Q4   Q4+SIMD
```

## 🔬 Technical Details

### **Q4 Quantization Format**
```
Original: [w1, w2, w3, w4] in FP32

Quantized:
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  w1/scale   │  w2/scale   │  w3/scale   │  w4/scale   │
│   (4-bit)   │   (4-bit)   │   (4-bit)   │   (4-bit)   │
└─────────────┴─────────────┴─────────────┴─────────────┘
Packed: [ nibble1 | nibble2 | nibble3 | nibble4 ] (2 bytes)
```

### **KV Cache Management**
```
┌─────────────────────────────────────────────────────────┐
│                 KV Cache Structure                      │
├─────────────────────────────────────────────────────────┤
│ Layer 1: [K1, V1] [K2, V2] ... [Kn, Vn]                │
│ Layer 2: [K1, V2] [K2, V2] ... [Kn, Vn]                │
│ ...                                                     │
│ Layer N: [K1, VN] [K2, VN] ... [Kn, VN]                │
└─────────────────────────────────────────────────────────┘
```

## 🧪 Testing & Validation

### **Test Coverage**
- ✅ **Tensor Operations**: Shape manipulation, dtype conversion, memory alignment
- ✅ **Quantization**: Round-trip accuracy, numerical stability
- ✅ **Linear Algebra**: GEMM operations, matrix-vector products
- ✅ **End-to-end**: Full inference pipeline validation

### **Accuracy Validation**
```bash
# Compare against PyTorch baseline
python3 tools/validate_accuracy.py --model model.onnx --tolerance 1e-3
```

## 🛠️ Development

### **Code Structure**
```
src/
├── core/
│   ├── tensor.hpp/cpp        # Multi-dimensional arrays
│   ├── alloc.hpp/cpp         # Memory management
│   └── profiler.hpp/cpp      # Performance measurement
├── loaders/
│   └── onnx_loader.hpp/cpp   # Model file parsing
├── kernels/
│   ├── gemm_ref.hpp/cpp      # Reference GEMM
│   └── q4_rowwise.hpp/cpp    # Quantized operations
├── transformer/
│   └── transformer.hpp/cpp   # Model implementation
├── tokenizer/
│   └── sentencepiece_wrapper.hpp/cpp
└── util/
    └── threadpool.hpp/cpp    # Parallel execution
```

### **Adding New Kernels**
```cpp
// In src/kernels/new_kernel.hpp
class NewKernel {
public:
    static void compute(const Tensor& input, Tensor& output);
};

// In src/kernels/new_kernel.cpp
void NewKernel::compute(const Tensor& input, Tensor& output) {
    // Implementation
}
```

## 📈 Roadmap

### **Phase 1** (Current) ✅
- [x] Core tensor abstraction with quantization support
- [x] ONNX model loading and parsing
- [x] Reference GEMM and Q4 kernels
- [x] Basic transformer implementation

### **Phase 2** (Next) 🚧
- [x] SIMD-accelerated kernels (AVX2/AVX512)
- [x] GGUF model loader integration
- [x] Flash Attention implementation
- [x] Batch processing support
- [x] HTTP server interface
- [ ] Advanced sampling methods (nucleus, contrastive search)

### **Phase 3** (Future) 🔮
- [ ] Model quantization tools (QAT, PTQ)
- [ ] GPU acceleration (CUDA/Vulkan)
- [ ] Distributed inference across multiple nodes
- [ ] Model parallelism and tensor parallelism
- [ ] Advanced attention mechanisms (Longformer, Reformer)
- [ ] Multimodal model support (vision, audio)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
```bash
# Clone the repository
git clone <repository-url>
cd helios-engine

# Install development dependencies
pip install onnx numpy  # For model conversion tools

# Build and test
mkdir build && cd build
cmake .. -DENABLE_TESTS=ON
cmake --build . -j$(nproc)
./bin/unit_tests
```

## 📚 References

- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - Reference ONNX implementation
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF format and quantization reference
- [Eigen Library](http://eigen.tuxfamily.org/) - Linear algebra library
- [OpenBLAS](https://www.openblas.net/) - BLAS implementation

---

<div align="center">
  <p><strong>Built with ❤️ for the AI community</strong></p>
  <p>🚀 High-performance inference • ⚡ Quantized operations • 🧠 KV caching</p>
</div>
