#ifndef SENTENCEPIECE_WRAPPER_HPP
#define SENTENCEPIECE_WRAPPER_HPP

#include <string>
#include <vector>
#include <memory>

class Tokenizer {
public:
    Tokenizer(const std::string& model_path);
    ~Tokenizer();

    // Encode text to token IDs
    std::vector<int> encode(const std::string& text);

    // Decode token IDs back to text
    std::string decode(const std::vector<int>& tokens);

    // Get vocab size
    int vocab_size() const { return vocab_size_; }

    // Get special tokens
    int bos_token_id() const { return bos_token_id_; }
    int eos_token_id() const { return eos_token_id_; }
    int pad_token_id() const { return pad_token_id_; }

private:
    std::string model_path_;
    int vocab_size_;
    int bos_token_id_;
    int eos_token_id_;
    int pad_token_id_;

    // Simplified tokenizer state (would use sentencepiece in real implementation)
    std::vector<std::string> vocabulary_;
};

#endif // SENTENCEPIECE_WRAPPER_HPP
