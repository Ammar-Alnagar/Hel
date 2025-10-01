#include "sentencepiece_wrapper.hpp"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <regex>

Tokenizer::Tokenizer(const std::string& model_path) : model_path_(model_path) {
    // Simplified tokenizer initialization
    // In a real implementation, this would load a sentencepiece model

    vocab_size_ = 32000; // Default vocab size
    bos_token_id_ = 1;   // Beginning of sentence
    eos_token_id_ = 2;   // End of sentence
    pad_token_id_ = 0;   // Padding

    // Create a simple vocabulary for testing (would be loaded from model file)
    vocabulary_.reserve(vocab_size_);
    for (int i = 0; i < vocab_size_; ++i) {
        vocabulary_.push_back("token_" + std::to_string(i));
    }

    // Add some common tokens
    if (vocab_size_ > 100) {
        vocabulary_[1] = "<s>";     // BOS
        vocabulary_[2] = "</s>";    // EOS
        vocabulary_[0] = "<pad>";   // PAD
        vocabulary_[3] = "<unk>";   // UNK
    }
}

Tokenizer::~Tokenizer() = default;

std::vector<int> Tokenizer::encode(const std::string& text) {
    // Very simplified tokenization for testing
    // In a real implementation, this would use sentencepiece or BPE

    std::vector<int> tokens;
    std::string remaining = text;

    // Simple word-level tokenization with regex
    std::regex word_regex(R"(\w+|[^\w\s])");
    std::sregex_iterator iter(text.begin(), text.end(), word_regex);
    std::sregex_iterator end;

    for (; iter != end; ++iter) {
        std::string token = iter->str();

        // Simple mapping for common words (would use proper vocabulary in real impl)
        if (token == "hello" || token == "Hello") {
            tokens.push_back(9900); // Made up token ID
        } else if (token == "world" || token == "World") {
            tokens.push_back(1917); // Made up token ID
        } else if (token == "the" || token == "The") {
            tokens.push_back(5);    // Made up token ID
        } else if (token == ".") {
            tokens.push_back(13);   // Made up token ID
        } else if (token == ",") {
            tokens.push_back(11);   // Made up token ID
        } else {
            // Unknown word - use UNK token
            tokens.push_back(3);
        }
    }

    // Add BOS and EOS tokens
    tokens.insert(tokens.begin(), bos_token_id_);
    tokens.push_back(eos_token_id_);

    return tokens;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) {
    // Simplified decoding (reverse of encoding)
    std::string result;

    for (size_t i = 0; i < tokens.size(); ++i) {
        int token_id = tokens[i];

        // Skip special tokens in output (except for spacing)
        if (token_id == bos_token_id_ && i == 0) {
            continue; // Skip BOS at beginning
        }
        if (token_id == eos_token_id_) {
            break; // Stop at EOS
        }
        if (token_id == pad_token_id_) {
            continue; // Skip padding
        }

        // Simple reverse mapping (would use proper vocabulary in real impl)
        if (token_id == 9900) {
            result += "hello";
        } else if (token_id == 1917) {
            result += "world";
        } else if (token_id == 5) {
            result += "the";
        } else if (token_id == 13) {
            result += ".";
        } else if (token_id == 11) {
            result += ",";
        } else {
            result += "<unk>";
        }

        // Add space between tokens (except for punctuation)
        if (i < tokens.size() - 1) {
            int next_token = tokens[i + 1];
            if (next_token != 13 && next_token != 11 && next_token != eos_token_id_) {
                result += " ";
            }
        }
    }

    return result;
}
