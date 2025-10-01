#ifndef SIMD_GEMM_HPP
#define SIMD_GEMM_HPP

#include "../../tensor.hpp"
#include <cstdint>

#ifdef ENABLE_SIMD

namespace simd {

// AVX2 optimized matrix multiplication for float32
void matmul_avx2_f32(const float* A, const float* B, float* C,
                    int M, int K, int N, float alpha = 1.0f, float beta = 0.0f);

// AVX-512 optimized matrix multiplication for float32
void matmul_avx512_f32(const float* A, const float* B, float* C,
                      int M, int K, int N, float alpha = 1.0f, float beta = 0.0f);

// AVX2 optimized Q4 matrix-vector multiplication
void matvec_q4_avx2(const uint8_t* qweights, const float* scales,
                   const float* x, float* y, int M, int K);

// Vectorized memory copy with alignment
void memcpy_aligned(void* dst, const void* src, size_t size);

// Check if SIMD instructions are supported
bool has_avx2();
bool has_avx512();

} // namespace simd

#endif // ENABLE_SIMD

#endif // SIMD_GEMM_HPP
