#ifndef Q4_ROWWISE_HPP
#define Q4_ROWWISE_HPP

#include <cstdint>

// Q4 decoding helper as specified in the spec
inline int8_t decode_q4_signed(uint8_t nib) {
    return (nib & 8) ? int8_t(nib - 16) : int8_t(nib);
}

// Reference Q4 matvec: y[M] = scale[M] * sum_k (depack(q4_row[i,k]) * x[k])
void matvec_q4_rowwise(const uint8_t* qweights, const float* scales,
                      const float* x, float* y, int M, int K);

// Helper functions for Q4 encoding/decoding
void pack_q4_rowwise(const float* weights, const float* scales,
                    uint8_t* qweights, float* scales_out, int M, int K);

void dequantize_q4_rowwise(const uint8_t* qweights, const float* scales,
                          float* out, int M, int K);

#endif // Q4_ROWWISE_HPP
