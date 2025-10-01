#include "q4_rowwise.hpp"
#include <cmath>
#include <algorithm>

void matvec_q4_rowwise(const uint8_t* qweights, const float* scales,
                      const float* x, float* y, int M, int K) {
    for (int m = 0; m < M; ++m) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            // Extract 4-bit value from packed byte
            uint8_t packed_byte = qweights[(m * K + k) / 2];
            uint8_t nib = (k % 2 == 0) ? (packed_byte & 0x0F) : (packed_byte >> 4);

            // Decode Q4 value
            int8_t dequant = decode_q4_signed(nib);

            // Accumulate: dequant * x[k] (scale is already applied during quantization)
            sum += static_cast<float>(dequant) * x[k];
        }
        y[m] = scales[m] * sum;
    }
}

void pack_q4_rowwise(const float* weights, const float* scales,
                    uint8_t* qweights, float* scales_out, int M, int K) {
    // For now, copy scales as-is (assuming they are provided externally)
    std::copy(scales, scales + M, scales_out);

    // Pack weights using provided scales
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; k += 2) {
            uint8_t byte = 0;

            // First nibble
            float w1 = weights[m * K + k] / scales_out[m];
            int8_t q1 = static_cast<int8_t>(std::round(w1));
            q1 = std::max(int8_t(-8), std::min(int8_t(7), q1));
            byte |= (static_cast<uint8_t>(q1) & 0x0F);

            // Second nibble (if exists)
            if (k + 1 < K) {
                float w2 = weights[m * K + k + 1] / scales_out[m];
                int8_t q2 = static_cast<int8_t>(std::round(w2));
                q2 = std::max(int8_t(-8), std::min(int8_t(7), q2));
                byte |= ((static_cast<uint8_t>(q2) & 0x0F) << 4);
            }

            qweights[(m * K + k) / 2] = byte;
        }
    }
}

void dequantize_q4_rowwise(const uint8_t* qweights, const float* scales,
                          float* out, int M, int K) {
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            // Extract 4-bit value
            uint8_t packed_byte = qweights[(m * K + k) / 2];
            uint8_t nib = (k % 2 == 0) ? (packed_byte & 0x0F) : (packed_byte >> 4);

            // Decode and scale
            int8_t dequant = decode_q4_signed(nib);
            out[m * K + k] = static_cast<float>(dequant) * scales[m];
        }
    }
}
