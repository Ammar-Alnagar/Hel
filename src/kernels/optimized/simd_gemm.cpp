#include "simd_gemm.hpp"
#include <cstring>
#include <immintrin.h> // AVX/AVX2/AVX-512 intrinsics

#ifdef ENABLE_SIMD

namespace simd {

// AVX2 optimized matrix multiplication for float32
void matmul_avx2_f32(const float* A, const float* B, float* C,
                    int M, int K, int N, float alpha, float beta) {
    const int BLOCK_SIZE = 8; // AVX2 processes 8 floats at once

    // Process in blocks for better cache locality
    for (int m = 0; m < M; m += BLOCK_SIZE) {
        for (int n = 0; n < N; n += BLOCK_SIZE) {
            for (int k = 0; k < K; k += BLOCK_SIZE) {
                // Process blocks
                int m_end = std::min(m + BLOCK_SIZE, M);
                int n_end = std::min(n + BLOCK_SIZE, N);
                int k_end = std::min(k + BLOCK_SIZE, K);

                for (int mm = m; mm < m_end; ++mm) {
                    for (int nn = n; nn < n_end; ++nn) {
                        __m256 sum = _mm256_setzero_ps();

                        int kk = k;
                        // Process in chunks of 8 for AVX2
                        for (; kk + 8 <= k_end; kk += 8) {
                            // Load A data
                            __m256 a_vec = _mm256_loadu_ps(&A[mm * K + kk]);
                            // Load B data (transposed layout assumed)
                            __m256 b_vec = _mm256_loadu_ps(&B[kk * N + nn]);
                            // Multiply and accumulate
                            sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
                        }

                        // Handle remaining elements
                        float sum_scalar = 0.0f;
                        for (; kk < k_end; ++kk) {
                            sum_scalar += A[mm * K + kk] * B[kk * N + nn];
                        }

                        // Sum the AVX register
                        float* sum_ptr = reinterpret_cast<float*>(&sum);
                        float total_sum = sum_scalar;
                        for (int i = 0; i < 8; ++i) {
                            total_sum += sum_ptr[i];
                        }

                        C[mm * N + nn] = alpha * total_sum + beta * C[mm * N + nn];
                    }
                }
            }
        }
    }
}

// AVX-512 optimized matrix multiplication for float32
void matmul_avx512_f32(const float* A, const float* B, float* C,
                      int M, int K, int N, float alpha, float beta) {
    const int BLOCK_SIZE = 16; // AVX-512 processes 16 floats at once

    for (int m = 0; m < M; m += BLOCK_SIZE) {
        for (int n = 0; n < N; n += BLOCK_SIZE) {
            for (int k = 0; k < K; k += BLOCK_SIZE) {
                int m_end = std::min(m + BLOCK_SIZE, M);
                int n_end = std::min(n + BLOCK_SIZE, N);
                int k_end = std::min(k + BLOCK_SIZE, K);

                for (int mm = m; mm < m_end; ++mm) {
                    for (int nn = n; nn < n_end; ++nn) {
                        __m512 sum = _mm512_setzero_ps();

                        int kk = k;
                        // Process in chunks of 16 for AVX-512
                        for (; kk + 16 <= k_end; kk += 16) {
                            __m512 a_vec = _mm512_loadu_ps(&A[mm * K + kk]);
                            __m512 b_vec = _mm512_loadu_ps(&B[kk * N + nn]);
                            sum = _mm512_fmadd_ps(a_vec, b_vec, sum);
                        }

                        // Handle remaining elements
                        float sum_scalar = 0.0f;
                        for (; kk < k_end; ++kk) {
                            sum_scalar += A[mm * K + kk] * B[kk * N + nn];
                        }

                        // Sum the AVX-512 register
                        float* sum_ptr = reinterpret_cast<float*>(&sum);
                        float total_sum = sum_scalar;
                        for (int i = 0; i < 16; ++i) {
                            total_sum += sum_ptr[i];
                        }

                        C[mm * N + nn] = alpha * total_sum + beta * C[mm * N + nn];
                    }
                }
            }
        }
    }
}

// AVX2 optimized Q4 matrix-vector multiplication
void matvec_q4_avx2(const uint8_t* qweights, const float* scales,
                   const float* x, float* y, int M, int K) {
    const int BLOCK_SIZE = 8; // Process 8 rows at once

    for (int m = 0; m < M; m += BLOCK_SIZE) {
        __m256 sum_vec[BLOCK_SIZE];
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum_vec[i] = _mm256_setzero_ps();
        }

        int m_end = std::min(m + BLOCK_SIZE, M);

        for (int k = 0; k < K; ++k) {
            __m256 x_vec = _mm256_broadcast_ss(&x[k]);

            for (int mm = m; mm < m_end; ++mm) {
                uint8_t packed_byte = qweights[(mm * K + k) / 2];
                uint8_t nib = (k % 2 == 0) ? (packed_byte & 0x0F) : (packed_byte >> 4);

                // Convert to float and broadcast
                float dequant_val = (nib & 8) ? static_cast<float>(nib - 16) : static_cast<float>(nib);
                __m256 dequant_vec = _mm256_broadcast_ss(&dequant_val);

                sum_vec[mm - m] = _mm256_fmadd_ps(dequant_vec, x_vec, sum_vec[mm - m]);
            }
        }

        // Store results
        for (int mm = m; mm < m_end; ++mm) {
            float* sum_ptr = reinterpret_cast<float*>(&sum_vec[mm - m]);
            float total_sum = 0.0f;
            for (int i = 0; i < 8; ++i) {
                total_sum += sum_ptr[i];
            }
            y[mm] = scales[mm] * total_sum;
        }
    }
}

// Vectorized memory copy with alignment
void memcpy_aligned(void* dst, const void* src, size_t size) {
    // Use AVX2 for aligned copies when possible
    if (size >= 32 && (reinterpret_cast<uintptr_t>(dst) % 32 == 0) &&
        (reinterpret_cast<uintptr_t>(src) % 32 == 0)) {
        size_t num_chunks = size / 32;
        for (size_t i = 0; i < num_chunks; ++i) {
            __m256 data = _mm256_load_ps(static_cast<const float*>(src) + i * 8);
            _mm256_store_ps(static_cast<float*>(dst) + i * 8, data);
        }
        // Handle remainder
        size_t remaining = size % 32;
        if (remaining > 0) {
            std::memcpy(static_cast<char*>(dst) + size - remaining,
                       static_cast<const char*>(src) + size - remaining, remaining);
        }
    } else {
        std::memcpy(dst, src, size);
    }
}

// Check if SIMD instructions are supported
bool has_avx2() {
    return __builtin_cpu_supports("avx2");
}

bool has_avx512() {
    return __builtin_cpu_supports("avx512f");
}

} // namespace simd

#endif // ENABLE_SIMD
