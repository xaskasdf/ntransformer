#include "tokenizer.h"
#include <algorithm>
#include <cstdio>
#include <sstream>
#include <limits>

namespace nt {

// ============================================================
// GPT-2 byte encoder: maps raw bytes to Unicode characters
// used by Llama 3's tiktoken-based BPE tokenizer
// ============================================================

static void build_byte_tables(
    std::string byte_to_unicode[256],
    std::unordered_map<std::string, uint8_t>& unicode_to_byte
) {
    // The GPT-2 bytes_to_unicode mapping:
    // Printable ASCII (33-126), Latin-1 Supplement (161-172, 174-255) -> identity
    // Everything else (0-32, 127-160, 173) -> 256+n
    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool identity = (b >= 33 && b <= 126) ||
                        (b >= 161 && b <= 172) ||
                        (b >= 174 && b <= 255);
        uint32_t cp;
        if (identity) {
            cp = (uint32_t)b;
        } else {
            cp = 256 + n;
            n++;
        }

        // Encode code point to UTF-8
        char utf8[5] = {};
        if (cp < 0x80) {
            utf8[0] = (char)cp;
        } else if (cp < 0x800) {
            utf8[0] = (char)(0xC0 | (cp >> 6));
            utf8[1] = (char)(0x80 | (cp & 0x3F));
        } else {
            utf8[0] = (char)(0xE0 | (cp >> 12));
            utf8[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
            utf8[2] = (char)(0x80 | (cp & 0x3F));
        }

        byte_to_unicode[b] = std::string(utf8);
        unicode_to_byte[byte_to_unicode[b]] = (uint8_t)b;
    }
}

// Global tables (initialized once)
static std::string g_byte_to_unicode[256];
static std::unordered_map<std::string, uint8_t> g_unicode_to_byte;
static bool g_tables_initialized = false;

static void ensure_tables() {
    if (!g_tables_initialized) {
        build_byte_tables(g_byte_to_unicode, g_unicode_to_byte);
        g_tables_initialized = true;
    }
}

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

    // Detect tokenizer type: Llama 3 uses GPT-2 byte encoding (has Ġ for space)
    // Llama 1/2 uses SentencePiece (has ▁ for space)
    ensure_tables();
    // Check if vocab contains the GPT-2 encoding of space (Ġ = U+0120)
    std::string space_encoded = g_byte_to_unicode[0x20]; // Should be Ġ
    use_gpt2_encoding_ = (token_to_id_.find(space_encoded) != token_to_id_.end());

    fprintf(stderr, "Tokenizer: %d tokens, BOS=%d, EOS=%d, encoding=%s\n",
        (int)tokens_.size(), bos_id_, eos_id_,
        use_gpt2_encoding_ ? "GPT2-BPE" : "SentencePiece");
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
    // Convert input text to tokenizer's encoding
    std::string encoded;
    if (use_gpt2_encoding_) {
        // GPT-2 byte encoding: each raw byte maps to a Unicode character
        encoded.reserve(text.size() * 2);
        for (size_t i = 0; i < text.size(); i++) {
            encoded += g_byte_to_unicode[(uint8_t)text[i]];
        }
    } else {
        // SentencePiece: replace spaces with ▁
        encoded.reserve(text.size() + 16);
        for (size_t i = 0; i < text.size(); i++) {
            if (text[i] == ' ') {
                encoded += "\xe2\x96\x81";  // ▁ in UTF-8
            } else {
                encoded += text[i];
            }
        }
    }

    // Initialize symbols from individual characters/bytes
    struct Symbol {
        int token_id;
        std::string text;
        int prev;
        int next;
    };

    std::vector<Symbol> symbols;

    size_t pos = 0;
    while (pos < encoded.size()) {
        // Try to find the longest matching token starting at pos
        bool found = false;
        size_t max_len = std::min(encoded.size() - pos, (size_t)64);
        for (size_t len = max_len; len >= 1; len--) {
            std::string sub = encoded.substr(pos, len);
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
            int bt = byte_token((uint8_t)encoded[pos]);
            Symbol sym;
            sym.token_id = bt;
            sym.text = encoded.substr(pos, 1);
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
        float best_score = -std::numeric_limits<float>::infinity();
        int best_idx = -1;

        for (int i = 0; i < (int)symbols.size(); i++) {
            if (symbols[i].next < 0) continue;
            if (symbols[i].token_id < 0) continue;

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

        if (best_idx < 0) break;

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

    // Check if it's a special/control token
    if (!token_types_.empty() && token_id < (int)token_types_.size()) {
        int type = token_types_[token_id];
        if (type == 3 || type == 4) {  // control or unused
            return "";
        }
    }

    std::string token = tokens_[token_id];

    if (use_gpt2_encoding_) {
        // GPT-2 byte decoding: convert each Unicode character back to a raw byte
        std::string result;
        size_t pos = 0;
        while (pos < token.size()) {
            // Determine the length of the current UTF-8 character
            uint8_t c = (uint8_t)token[pos];
            int char_len;
            if (c < 0x80) char_len = 1;
            else if ((c & 0xE0) == 0xC0) char_len = 2;
            else if ((c & 0xF0) == 0xE0) char_len = 3;
            else if ((c & 0xF8) == 0xF0) char_len = 4;
            else { pos++; continue; }  // invalid UTF-8

            if (pos + char_len > token.size()) break;

            std::string ch = token.substr(pos, char_len);
            auto it = g_unicode_to_byte.find(ch);
            if (it != g_unicode_to_byte.end()) {
                result += (char)it->second;
            } else {
                // Pass through unchanged
                result += ch;
            }
            pos += char_len;
        }
        return result;
    } else {
        // SentencePiece: check for byte tokens and replace ▁ with space
        if (token.size() == 6 && token[0] == '<' && token[1] == '0' && token[2] == 'x' && token[5] == '>') {
            char hex[3] = {token[3], token[4], 0};
            char byte = (char)strtol(hex, nullptr, 16);
            return std::string(1, byte);
        }

        std::string result;
        size_t pos = 0;
        while (pos < token.size()) {
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
}

int Tokenizer::byte_token(uint8_t byte) const {
    if (use_gpt2_encoding_) {
        // In GPT-2 encoding, individual bytes map to their Unicode chars
        auto it = token_to_id_.find(g_byte_to_unicode[byte]);
        if (it != token_to_id_.end()) {
            return it->second;
        }
    }

    // SentencePiece byte fallback: <0xXX> format
    char name[8];
    snprintf(name, sizeof(name), "<0x%02X>", byte);
    auto it = token_to_id_.find(name);
    if (it != token_to_id_.end()) {
        return it->second;
    }
    return 0;
}

} // namespace nt
