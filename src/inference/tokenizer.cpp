#include "tokenizer.h"
#include <algorithm>
#include <cstdio>
#include <sstream>
#include <limits>

namespace nt {

void Tokenizer::init(const GGUFVocab& vocab, int bos_id, int eos_id) {
    tokens_ = vocab.tokens;
    scores_ = vocab.scores;
    token_types_ = vocab.token_types;
    bos_id_ = bos_id;
    eos_id_ = eos_id;

    // Build token -> ID map
    for (int i = 0; i < (int)tokens_.size(); i++) {
        token_to_id_[tokens_[i]] = i;
    }

    fprintf(stderr, "Tokenizer: %d tokens, BOS=%d, EOS=%d\n",
        (int)tokens_.size(), bos_id_, eos_id_);
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int> tokens;

    if (add_bos) {
        tokens.push_back(bos_id_);
    }

    if (text.empty()) return tokens;

    bpe_encode(text, tokens);
    return tokens;
}

void Tokenizer::bpe_encode(const std::string& text, std::vector<int>& output) const {
    // SentencePiece-style BPE encoding
    // 1. Start with each byte/character as a token
    // 2. Greedily merge the pair with highest score
    // 3. Repeat until no more merges possible

    // Initialize: each UTF-8 character or byte becomes a token
    struct Symbol {
        int token_id;
        std::string text;
        int prev;
        int next;
    };

    std::vector<Symbol> symbols;

    // SentencePiece uses the "▁" (U+2581) prefix for word boundaries
    // Replace spaces with ▁ at the beginning and before each word
    std::string processed;
    processed.reserve(text.size() + 16);

    for (size_t i = 0; i < text.size(); i++) {
        if (text[i] == ' ') {
            processed += "\xe2\x96\x81";  // ▁ in UTF-8
        } else {
            processed += text[i];
        }
    }

    // Initialize symbols from individual characters/bytes
    size_t pos = 0;
    while (pos < processed.size()) {
        // Try to find the longest matching token starting at pos
        // Start with single bytes and try longer
        bool found = false;

        // Try progressively shorter substrings
        size_t max_len = std::min(processed.size() - pos, (size_t)64);
        for (size_t len = max_len; len >= 1; len--) {
            std::string sub = processed.substr(pos, len);
            auto it = token_to_id_.find(sub);
            if (it != token_to_id_.end()) {
                Symbol sym;
                sym.token_id = it->second;
                sym.text = sub;
                sym.prev = symbols.empty() ? -1 : (int)symbols.size() - 1;
                sym.next = -1;
                if (!symbols.empty()) {
                    symbols.back().next = symbols.size();
                }
                symbols.push_back(sym);
                pos += len;
                found = true;
                break;
            }
        }

        if (!found) {
            // Byte-level fallback
            int bt = byte_token((uint8_t)processed[pos]);
            Symbol sym;
            sym.token_id = bt;
            sym.text = processed.substr(pos, 1);
            sym.prev = symbols.empty() ? -1 : (int)symbols.size() - 1;
            sym.next = -1;
            if (!symbols.empty()) {
                symbols.back().next = symbols.size();
            }
            symbols.push_back(sym);
            pos++;
        }
    }

    // BPE merge loop
    while (true) {
        // Find the best pair to merge (highest score)
        float best_score = -std::numeric_limits<float>::infinity();
        int best_idx = -1;

        for (int i = 0; i < (int)symbols.size(); i++) {
            if (symbols[i].next < 0) continue;
            if (symbols[i].token_id < 0) continue;  // deleted

            int j = symbols[i].next;
            std::string merged = symbols[i].text + symbols[j].text;
            auto it = token_to_id_.find(merged);
            if (it != token_to_id_.end()) {
                int merged_id = it->second;
                float score = (merged_id < (int)scores_.size()) ? scores_[merged_id] : 0.0f;
                if (score > best_score) {
                    best_score = score;
                    best_idx = i;
                }
            }
        }

        if (best_idx < 0) break;  // No more merges

        // Perform the merge
        int i = best_idx;
        int j = symbols[i].next;
        std::string merged = symbols[i].text + symbols[j].text;
        auto it = token_to_id_.find(merged);

        symbols[i].token_id = it->second;
        symbols[i].text = merged;
        symbols[i].next = symbols[j].next;
        if (symbols[j].next >= 0) {
            symbols[symbols[j].next].prev = i;
        }

        // Mark j as deleted
        symbols[j].token_id = -1;
    }

    // Collect remaining tokens
    for (const auto& sym : symbols) {
        if (sym.token_id >= 0) {
            output.push_back(sym.token_id);
        }
    }
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;
    for (int id : tokens) {
        result += decode_token(id);
    }
    return result;
}

std::string Tokenizer::decode_token(int token_id) const {
    if (token_id < 0 || token_id >= (int)tokens_.size()) {
        return "";
    }

    // Check if it's a special token
    if (!token_types_.empty() && token_id < (int)token_types_.size()) {
        int type = token_types_[token_id];
        if (type == 3 || type == 4) {  // control or unused
            return "";
        }
    }

    std::string token = tokens_[token_id];

    // Check for byte tokens like <0x0A>
    if (token.size() == 6 && token[0] == '<' && token[1] == '0' && token[2] == 'x' && token[5] == '>') {
        char hex[3] = {token[3], token[4], 0};
        char byte = (char)strtol(hex, nullptr, 16);
        return std::string(1, byte);
    }

    // Replace ▁ with space
    std::string result;
    size_t pos = 0;
    while (pos < token.size()) {
        // Check for ▁ (UTF-8: 0xE2 0x96 0x81)
        if (pos + 2 < token.size() &&
            (uint8_t)token[pos] == 0xE2 &&
            (uint8_t)token[pos+1] == 0x96 &&
            (uint8_t)token[pos+2] == 0x81) {
            result += ' ';
            pos += 3;
        } else {
            result += token[pos];
            pos++;
        }
    }

    return result;
}

int Tokenizer::byte_token(uint8_t byte) const {
    // Look for <0xXX> format
    char name[8];
    snprintf(name, sizeof(name), "<0x%02X>", byte);
    auto it = token_to_id_.find(name);
    if (it != token_to_id_.end()) {
        return it->second;
    }
    return 0;  // fallback to token 0
}

} // namespace nt
