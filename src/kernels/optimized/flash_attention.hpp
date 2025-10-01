#ifndef FLASH_ATTENTION_HPP
#define FLASH_ATTENTION_HPP

#include "../../tensor.hpp"
#include <vector>

namespace flash {

// Flash Attention implementation for better memory efficiency
class FlashAttention {
public:
    FlashAttention(int hidden_size, int num_heads, int head_dim, float scale = 1.0f);
    ~FlashAttention() = default;

    // Forward pass with KV caching support
    Tensor forward(const Tensor& query, const Tensor& key, const Tensor& value,
                  KVCache* cache = nullptr);

    // Get attention weights for debugging
    Tensor get_attention_weights() const { return attention_weights_; }

private:
    void compute_attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                          Tensor& output, KVCache* cache);

    int hidden_size_;
    int num_heads_;
    int head_dim_;
    float scale_;
    Tensor attention_weights_;
};

} // namespace flash

#endif // FLASH_ATTENTION_HPP
