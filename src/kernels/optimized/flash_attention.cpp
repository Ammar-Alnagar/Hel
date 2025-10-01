#include "flash_attention.hpp"
#include "../gemm_ref.hpp"
#include <cmath>
#include <algorithm>

namespace flash {

FlashAttention::FlashAttention(int hidden_size, int num_heads, int head_dim, float scale)
    : hidden_size_(hidden_size), num_heads_(num_heads), head_dim_(head_dim), scale_(scale) {

    // Initialize attention weights tensor for debugging
    std::vector<int> weights_shape = {num_heads, 1, 1}; // Will be resized as needed
    attention_weights_ = Tensor(weights_shape, DType::FP32);
}

Tensor FlashAttention::forward(const Tensor& query, const Tensor& key, const Tensor& value,
                              KVCache* cache) {
    auto q_shape = query.shape();
    auto k_shape = key.shape();
    auto v_shape = value.shape();

    // Validate shapes
    if (q_shape.size() != 3 || k_shape.size() != 3 || v_shape.size() != 3) {
        throw std::runtime_error("FlashAttention requires 3D tensors [batch, seq, hidden]");
    }

    if (q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0]) {
        throw std::runtime_error("Batch size mismatch in attention inputs");
    }

    if (q_shape[2] != k_shape[2] || q_shape[2] != v_shape[2]) {
        throw std::runtime_error("Hidden size mismatch in attention inputs");
    }

    int batch_size = q_shape[0];
    int seq_len = q_shape[1];
    int hidden_size = q_shape[2];

    // Reshape for multi-head attention
    std::vector<int> q_reshaped = {batch_size, seq_len, num_heads_, head_dim_};
    std::vector<int> output_shape = {batch_size, seq_len, hidden_size};

    Tensor output(output_shape, DType::FP32);

    // Process each head separately for simplicity
    for (int head = 0; head < num_heads_; ++head) {
        // Extract Q, K, V for this head
        std::vector<int> head_shape = {batch_size, seq_len, head_dim_};

        Tensor Q_head(head_shape, DType::FP32);
        Tensor K_head(head_shape, DType::FP32);
        Tensor V_head(head_shape, DType::FP32);

        // Simple head extraction (would use proper projection in real implementation)
        const float* q_data = query.data<float>();
        const float* k_data = key.data<float>();
        const float* v_data = value.data<float>();
        float* q_head_data = Q_head.data<float>();
        float* k_head_data = K_head.data<float>();
        float* v_head_data = V_head.data<float>();

        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < head_dim_; ++d) {
                    int src_idx = b * seq_len * hidden_size + s * hidden_size + head * head_dim_ + d;
                    int dst_idx = b * seq_len * head_dim_ + s * head_dim_ + d;

                    if (src_idx < query.numel() && dst_idx < Q_head.numel()) {
                        q_head_data[dst_idx] = q_data[src_idx];
                        k_head_data[dst_idx] = k_data[src_idx];
                        v_head_data[dst_idx] = v_data[src_idx];
                    }
                }
            }
        }

        // Compute attention for this head
        Tensor head_output(head_shape, DType::FP32);
        compute_attention(Q_head, K_head, V_head, head_output, cache);

        // Copy back to output
        float* output_data = output.data<float>();
        float* head_output_data = head_output.data<float>();

        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < head_dim_; ++d) {
                    int src_idx = b * seq_len * head_dim_ + s * head_dim_ + d;
                    int dst_idx = b * seq_len * hidden_size + s * hidden_size + head * head_dim_ + d;

                    if (src_idx < head_output.numel() && dst_idx < output.numel()) {
                        output_data[dst_idx] = head_output_data[src_idx];
                    }
                }
            }
        }
    }

    return output;
}

void FlashAttention::compute_attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                                      Tensor& output, KVCache* cache) {
    auto shape = Q.shape();
    int batch_size = shape[0];
    int seq_len = shape[1];
    int head_dim = shape[2];

    const float* q_data = Q.data<float>();
    const float* k_data = K.data<float>();
    const float* v_data = V.data<float>();
    float* output_data = output.data<float>();

    // Flash Attention algorithm (simplified implementation)
    // In a full implementation, this would use tiling and avoid storing full attention matrix

    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            float max_score = -std::numeric_limits<float>::infinity();
            float sum_exp = 0.0f;

            // Compute attention scores for position s
            std::vector<float> scores(seq_len);

            for (int t = 0; t <= s; ++t) { // Causal attention (only previous tokens)
                float score = 0.0f;

                // Compute dot product Q[s] * K[t]
                for (int d = 0; d < head_dim; ++d) {
                    int q_idx = b * seq_len * head_dim + s * head_dim + d;
                    int k_idx = b * seq_len * head_dim + t * head_dim + d;
                    score += q_data[q_idx] * k_data[k_idx];
                }

                score *= scale_; // Apply scaling
                scores[t] = score;
                max_score = std::max(max_score, score);
            }

            // Compute softmax
            for (int t = 0; t <= s; ++t) {
                scores[t] = std::exp(scores[t] - max_score);
                sum_exp += scores[t];
            }

            if (sum_exp > 0) {
                for (int t = 0; t <= s; ++t) {
                    scores[t] /= sum_exp;
                }
            }

            // Compute weighted sum of values
            for (int d = 0; d < head_dim; ++d) {
                float weighted_sum = 0.0f;

                for (int t = 0; t <= s; ++t) {
                    int v_idx = b * seq_len * head_dim + t * head_dim + d;
                    weighted_sum += scores[t] * v_data[v_idx];
                }

                int output_idx = b * seq_len * head_dim + s * head_dim + d;
                output_data[output_idx] = weighted_sum;
            }
        }
    }
}

} // namespace flash
