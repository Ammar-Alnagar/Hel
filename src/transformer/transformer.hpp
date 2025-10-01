#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP

#include "../tensor.hpp"
#include <vector>
#include <unordered_map>
#include <memory>

struct KVCache {
    std::vector<Tensor> keys;
    std::vector<Tensor> values;
    int current_length = 0;
};

struct ModelWeights {
    std::unordered_map<std::string, Tensor> weights;
};

class Linear {
public:
    Linear(const std::string& name, const Tensor& weight, const Tensor& bias = Tensor({}, DType::FP32));
    ~Linear() = default;

    Tensor forward(const Tensor& input);

private:
    std::string name_;
    Tensor weight_;
    Tensor bias_;
};

class Attention {
public:
    Attention(const std::string& name, int hidden_size, int num_heads);
    ~Attention() = default;

    Tensor forward(const Tensor& hidden_states, KVCache* cache = nullptr);

private:
    std::string name_;
    int hidden_size_;
    int num_heads_;
    int head_dim_;

    std::unique_ptr<Linear> q_proj_;
    std::unique_ptr<Linear> k_proj_;
    std::unique_ptr<Linear> v_proj_;
    std::unique_ptr<Linear> o_proj_;
};

class TransformerBlock {
public:
    TransformerBlock(const std::string& name, int hidden_size, int num_heads);
    ~TransformerBlock() = default;

    Tensor forward(const Tensor& hidden_states, KVCache* cache = nullptr);

private:
    std::string name_;
    std::unique_ptr<Attention> attention_;
    std::unique_ptr<Linear> ff1_;
    std::unique_ptr<Linear> ff2_;
};

class Transformer {
public:
    Transformer(const ModelWeights& weights);
    ~Transformer() = default;

    Tensor forward(const Tensor& input_ids, KVCache* cache = nullptr);

    // Configuration
    int vocab_size() const { return vocab_size_; }
    int hidden_size() const { return hidden_size_; }
    int num_layers() const { return num_layers_; }
    int num_heads() const { return num_heads_; }

private:
    void load_weights(const ModelWeights& weights);

    ModelWeights weights_;
    int vocab_size_;
    int hidden_size_;
    int num_layers_;
    int num_heads_;
    int max_seq_len_;

    std::unique_ptr<Linear> embed_tokens_;
    std::vector<std::unique_ptr<TransformerBlock>> layers_;
    std::unique_ptr<Linear> lm_head_;
};

#endif // TRANSFORMER_HPP
