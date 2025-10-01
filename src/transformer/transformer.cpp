#include "transformer.hpp"
#include "../kernels/gemm_ref.hpp"
#include <cmath>
#include <stdexcept>

// Linear layer implementation
Linear::Linear(const std::string& name, const Tensor& weight, const Tensor& bias)
    : name_(name), weight_(weight), bias_(bias) {}

Tensor Linear::forward(const Tensor& input) {
    auto shape = input.shape();
    auto weight_shape = weight_.shape();

    // input: [batch_size, seq_len, hidden_size] or [batch_size, hidden_size]
    // weight: [hidden_size, output_size]
    // output: [batch_size, seq_len, output_size] or [batch_size, output_size]

    if (shape.back() != weight_shape[0]) {
        throw std::runtime_error("Linear: input hidden size doesn't match weight");
    }

    // For simplicity, assume 2D input for now
    bool is_3d = (shape.size() == 3);
    int batch_size = is_3d ? shape[0] : 1;
    int seq_len = is_3d ? shape[1] : shape[0];
    int hidden_size = shape.back();
    int output_size = weight_shape[1];

    // Reshape input to 2D for matrix multiplication
    std::vector<int> input_2d_shape = {batch_size * seq_len, hidden_size};
    std::vector<int> output_2d_shape = {batch_size * seq_len, output_size};

    Tensor input_2d = input.reshape(input_2d_shape);
    Tensor output_2d(output_2d_shape, input.dtype());

    // Matrix multiplication: output = input @ weight.T
    GemmRef::matmul(input_2d, weight_, output_2d, 1.0f, 0.0f);

    // Add bias if provided
    if (bias_.numel() > 0) {
        for (int i = 0; i < batch_size * seq_len; ++i) {
            for (int j = 0; j < output_size; ++j) {
                output_2d.data<float>()[i * output_size + j] += bias_.data<float>()[j];
            }
        }
    }

    // Reshape back to original dimensions
    std::vector<int> output_shape = is_3d ?
        std::vector<int>{batch_size, seq_len, output_size} :
        std::vector<int>{output_size};
    return output_2d.reshape(output_shape);
}

// Attention implementation (simplified)
Attention::Attention(const std::string& name, int hidden_size, int num_heads)
    : name_(name), hidden_size_(hidden_size), num_heads_(num_heads) {
    head_dim_ = hidden_size / num_heads;
}

Tensor Attention::forward(const Tensor& hidden_states, KVCache* cache) {
    // Simplified attention implementation
    // In a real implementation, this would compute Q, K, V projections,
    // attention scores, and output projection

    // For now, return the input as-is (identity attention)
    return hidden_states;
}

// Transformer block implementation
TransformerBlock::TransformerBlock(const std::string& name, int hidden_size, int num_heads)
    : name_(name) {
    attention_ = std::make_unique<Attention>(name + ".attention", hidden_size, num_heads);
    // ff1_ and ff2_ would be initialized from weights
}

Tensor TransformerBlock::forward(const Tensor& hidden_states, KVCache* cache) {
    // Simplified transformer block: attention -> residual -> feedforward -> residual

    // Self-attention with residual
    Tensor attn_output = attention_->forward(hidden_states, cache);
    // Residual connection (simplified - just return attention output for now)
    Tensor residual = attn_output;

    // In a real implementation, you'd add the residual and apply layer norm

    return residual;
}

// Main Transformer implementation
Transformer::Transformer(const ModelWeights& weights) : weights_(weights) {
    load_weights(weights);
}

void Transformer::load_weights(const ModelWeights& weights) {
    // Extract configuration from weights (simplified)
    // In a real implementation, this would parse the model config

    vocab_size_ = 32000; // Default vocab size
    hidden_size_ = 768;  // Default hidden size
    num_layers_ = 12;    // Default number of layers
    num_heads_ = 12;     // Default number of heads
    max_seq_len_ = 2048; // Default max sequence length

    // Initialize layers (simplified)
    for (int i = 0; i < num_layers_; ++i) {
        layers_.push_back(std::make_unique<TransformerBlock>(
            "model.layers." + std::to_string(i), hidden_size_, num_heads_));
    }
}

Tensor Transformer::forward(const Tensor& input_ids, KVCache* cache) {
    // input_ids: [batch_size, seq_len]

    // Embedding lookup (simplified - would need actual embedding table)
    auto shape = input_ids.shape();
    int batch_size = shape[0];
    int seq_len = shape[1];

    // For now, create dummy hidden states (in reality, this would be embed_tokens(input_ids))
    std::vector<int> hidden_shape = {batch_size, seq_len, hidden_size_};
    Tensor hidden_states(hidden_shape, DType::FP32);

    // Initialize with small random values for testing
    float* data = hidden_states.data<float>();
    for (int i = 0; i < hidden_states.numel(); ++i) {
        data[i] = 0.01f * (rand() % 100 - 50) / 50.0f; // Small random values
    }

    // Forward pass through transformer layers
    for (auto& layer : layers_) {
        hidden_states = layer->forward(hidden_states, cache);
    }

    // Final linear layer (lm_head)
    // For now, return the hidden states as logits (simplified)
    return hidden_states.reshape({batch_size, seq_len, vocab_size_});
}
