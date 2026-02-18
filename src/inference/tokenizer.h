#pragma once

#include "../model/loader.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace nt {

// ============================================================
// BPE Tokenizer (loaded from GGUF vocabulary)
// Implements SentencePiece-style BPE as used by Llama models
// ============================================================

class Tokenizer {
public:
    Tokenizer() = default;

    // Initialize from GGUF vocab
    void init(const GGUFVocab& vocab, int bos_id, int eos_id);

    // Encode text to token IDs
    std::vector<int> encode(const std::string& text, bool add_bos = true) const;

    // Decode token IDs to text
    std::string decode(const std::vector<int>& tokens) const;
    std::string decode_token(int token_id) const;

    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }
    int vocab_size() const { return (int)tokens_.size(); }

private:
    std::vector<std::string> tokens_;
    std::vector<float> scores_;
    std::vector<int> token_types_;

    // Token to ID lookup
    std::unordered_map<std::string, int> token_to_id_;

    int bos_id_ = 1;
    int eos_id_ = 2;

    // BPE merge step
    struct BPEMerge {
        int left;
        int right;
        float score;
        int result;
    };

    // Internal BPE encode
    void bpe_encode(const std::string& text, std::vector<int>& output) const;

    // Byte-level fallback
    int byte_token(uint8_t byte) const;
};

} // namespace nt
