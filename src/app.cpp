#include "app.hpp"
#include "loaders/onnx_loader.hpp"
#include "transformer/transformer.hpp"
#include "tokenizer/sentencepiece_wrapper.hpp"
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>

int App::run(const InferenceArgs& args) {
    try {
        std::cout << "Loading model from: " << args.model_path << std::endl;

        // Load model weights
        auto weights = load_onnx_initializers(args.model_path);
        if (weights.empty()) {
            std::cout << "Warning: No initializers loaded. Using dummy model for testing.\n";
        }

        // Initialize tokenizer (would need actual tokenizer model path)
        Tokenizer tokenizer("dummy_tokenizer.model");

        // Initialize transformer
        ModelWeights model_weights{weights};
        Transformer transformer(model_weights);

        std::cout << "Model loaded successfully!" << std::endl;
        std::cout << "Vocab size: " << tokenizer.vocab_size() << std::endl;
        std::cout << "Hidden size: " << transformer.hidden_size() << std::endl;
        std::cout << "Num layers: " << transformer.num_layers() << std::endl;

        if (args.verbose) {
            std::cout << "Prompt: " << args.prompt << std::endl;
        }

        // Generate tokens
        auto generated_tokens = generate(args);

        // Decode and print result
        std::string generated_text = tokenizer.decode(generated_tokens);
        std::cout << "\nGenerated text: " << generated_text << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

std::vector<int> App::generate(const InferenceArgs& args) {
    // Initialize tokenizer and transformer (simplified for this demo)
    Tokenizer tokenizer("dummy_tokenizer.model");
    auto weights = load_onnx_initializers(args.model_path);
    ModelWeights model_weights{weights};
    Transformer transformer(model_weights);

    // Encode prompt
    auto input_tokens = tokenizer.encode(args.prompt);

    if (args.verbose) {
        std::cout << "Input tokens: ";
        for (int token : input_tokens) {
            std::cout << token << " ";
        }
        std::cout << std::endl;
    }

    std::vector<int> all_tokens = input_tokens;

    // Set up random number generation
    std::mt19937 gen(args.seed >= 0 ? args.seed : std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Autoregressive generation
    for (int step = 0; step < args.max_tokens; ++step) {
        // Create input tensor for current sequence
        std::vector<int> current_input_shape = {1, static_cast<int>(all_tokens.size())};
        Tensor input_ids(current_input_shape, DType::FP32);

        // Copy tokens to tensor (simplified - would need proper token ID handling)
        float* input_data = input_ids.data<float>();
        for (size_t i = 0; i < all_tokens.size(); ++i) {
            input_data[i] = static_cast<float>(all_tokens[i]);
        }

        // Forward pass through transformer
        KVCache cache; // Simplified - would need proper KV cache management
        Tensor logits = transformer.forward(input_ids, &cache);

        // Get logits for last position
        auto logits_shape = logits.shape();
        int vocab_size = logits_shape.back();
        const float* logits_data = logits.data<float>();

        // Extract logits for the last token
        std::vector<float> last_logits;
        int last_token_idx = static_cast<int>(all_tokens.size()) - 1;
        for (int i = 0; i < vocab_size; ++i) {
            last_logits.push_back(logits_data[last_token_idx * vocab_size + i]);
        }

        // Apply temperature
        if (args.temperature > 0.0f) {
            for (float& logit : last_logits) {
                logit /= args.temperature;
            }
        }

        // Sample next token
        int next_token = sample_token(last_logits, args);

        // Check for EOS
        if (next_token == tokenizer.eos_token_id()) {
            break;
        }

        // Add to sequence
        all_tokens.push_back(next_token);

        if (args.verbose) {
            std::cout << "Step " << step << ": token " << next_token << std::endl;
        }
    }

    return all_tokens;
}

int App::sample_token(const std::vector<float>& logits, const InferenceArgs& args) {
    int vocab_size = static_cast<int>(logits.size());

    // Convert logits to probabilities
    std::vector<float> probs = logits;

    // Apply softmax
    float max_logit = *std::max_element(probs.begin(), probs.end());
    float sum_exp = 0.0f;
    for (float& prob : probs) {
        prob = std::exp(prob - max_logit);
        sum_exp += prob;
    }
    for (float& prob : probs) {
        prob /= sum_exp;
    }

    // Top-k sampling
    std::vector<int> top_k_indices;
    std::vector<float> top_k_probs;
    for (int i = 0; i < vocab_size; ++i) {
        top_k_indices.push_back(i);
        top_k_probs.push_back(probs[i]);
    }

    // Sort by probability (descending)
    std::sort(top_k_indices.begin(), top_k_indices.end(),
              [&](int a, int b) { return top_k_probs[a] > top_k_probs[b]; });

    // Keep only top-k
    if (args.top_k > 0 && args.top_k < vocab_size) {
        top_k_indices.resize(args.top_k);
        top_k_probs.resize(args.top_k);
    }

    // Top-p (nucleus) sampling
    if (args.top_p < 1.0f) {
        // Sort by probability
        std::vector<int> sorted_indices = top_k_indices;
        std::sort(sorted_indices.begin(), sorted_indices.end(),
                  [&](int a, int b) { return top_k_probs[a] > top_k_probs[b]; });

        // Find cumulative probability cutoff
        float cumulative_prob = 0.0f;
        size_t cutoff_idx = 0;
        for (size_t i = 0; i < sorted_indices.size(); ++i) {
            cumulative_prob += top_k_probs[sorted_indices[i]];
            if (cumulative_prob >= args.top_p) {
                cutoff_idx = i + 1;
                break;
            }
        }

        if (cutoff_idx < sorted_indices.size()) {
            sorted_indices.resize(cutoff_idx);
        }

        top_k_indices = sorted_indices;
    }

    // Sample from filtered distribution
    std::discrete_distribution<int> distribution(top_k_probs.begin(), top_k_probs.end());
    int sampled_idx = top_k_indices[distribution(std::mt19937(args.seed >= 0 ? args.seed : std::random_device{}()))];

    return sampled_idx;
}

void App::print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --model PATH       Path to ONNX model file (required)\n"
              << "  --prompt TEXT      Input prompt text (required)\n"
              << "  --max-tokens N     Maximum number of tokens to generate (default: 16)\n"
              << "  --temperature F    Sampling temperature (default: 0.8)\n"
              << "  --top-k N          Top-k sampling parameter (default: 40)\n"
              << "  --top-p F          Top-p (nucleus) sampling parameter (default: 0.9)\n"
              << "  --seed N           Random seed (-1 for random, default: -1)\n"
              << "  --verbose          Enable verbose output\n"
              << "  --help             Show this help message\n";
}
