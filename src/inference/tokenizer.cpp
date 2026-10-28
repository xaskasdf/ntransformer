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
    pre_ = vocab.pre;

    // Build token -> ID map
    for (int i = 0; i < (int)tokens_.size(); i++) {
        token_to_id_[tokens_[i]] = i;
    }

    // Parse merges ("left right" -> rank)
    merge_rank_.clear();
    for (int i = 0; i < (int)vocab.merges.size(); i++) {
        merge_rank_[vocab.merges[i]] = i;
    }

    // Detect tokenizer type: Llama 3 uses GPT-2 byte encoding (has Ġ for space)
    // Llama 1/2 uses SentencePiece (has ▁ for space)
    ensure_tables();
    // Check if vocab contains the GPT-2 encoding of space (Ġ = U+0120)
    std::string space_encoded = g_byte_to_unicode[0x20]; // Should be Ġ
    use_gpt2_encoding_ = (token_to_id_.find(space_encoded) != token_to_id_.end());

    fprintf(stderr, "Tokenizer: %d tokens, %zu merges, BOS=%d, EOS=%d, pre=%s, encoding=%s\n",
        (int)tokens_.size(), vocab.merges.size(), bos_id_, eos_id_,
        pre_.empty() ? "(none)" : pre_.c_str(),
        merge_rank_.empty() ? (use_gpt2_encoding_ ? "GPT2-BPE(no merges)" : "SentencePiece") : "GPT2-BPE(merges)");
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int> tokens;

    if (add_bos) {
        tokens.push_back(bos_id_);
    }

    if (text.empty()) return tokens;

    if (has_merges()) {
        gpt2_merges_encode(text, tokens);
    } else {
        bpe_encode(text, tokens);
    }
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

// ============================================================
// GPT-2 BPE with explicit merges (DeepSeek V4.1 "joyai-llm" pre)
// ============================================================

namespace {

struct CpInfo {
    uint32_t cp;
    int len;  // bytes
};

static CpInfo decode_cp(const std::string& s, size_t pos) {
    CpInfo r{0, 1};
    uint8_t c = (uint8_t)s[pos];
    if (c < 0x80) { r.cp = c; r.len = 1; return r; }
    if ((c & 0xE0) == 0xC0 && pos + 1 < s.size()) {
        r.cp = ((c & 0x1F) << 6) | ((uint8_t)s[pos + 1] & 0x3F); r.len = 2; return r;
    }
    if ((c & 0xF0) == 0xE0 && pos + 2 < s.size()) {
        r.cp = ((c & 0x0F) << 12) | (((uint8_t)s[pos + 1] & 0x3F) << 6) | ((uint8_t)s[pos + 2] & 0x3F);
        r.len = 3; return r;
    }
    if ((c & 0xF8) == 0xF0 && pos + 3 < s.size()) {
        r.cp = ((c & 0x07) << 18) | (((uint8_t)s[pos + 1] & 0x3F) << 12) |
               (((uint8_t)s[pos + 2] & 0x3F) << 6) | ((uint8_t)s[pos + 3] & 0x3F);
        r.len = 4; return r;
    }
    r.cp = c; r.len = 1;
    return r;
}

enum : int { C_LETTER = 0, C_MARK, C_NUMBER, C_PUNCT, C_SYMBOL, C_NL, C_SPACE, C_OTHER };

static int classify_cp(uint32_t cp) {
    if (cp < 0x80) {
        if (cp == '\n' || cp == '\r') return C_NL;
        if (cp == ' ' || cp == '\t' || cp == '\v' || cp == '\f') return C_SPACE;
        if (cp >= '0' && cp <= '9') return C_NUMBER;
        if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return C_LETTER;
        if (cp >= '!' && cp <= '~') return C_PUNCT;  // ascii punct+symbols (close enough)
        return C_OTHER;
    }
    // CJK ideographs and kana
    if ((cp >= 0x4E00 && cp <= 0x9FA5) || (cp >= 0x3040 && cp <= 0x30FF)) return C_LETTER;
    if (cp >= 0x0300 && cp <= 0x036F) return C_MARK;
    if ((cp >= 0x2000 && cp <= 0x206F) || (cp >= 0x00A1 && cp <= 0x00BF) ||
        (cp >= 0x3000 && cp <= 0x303F) || cp == 0x2018 || cp == 0x2019 || cp == 0x201C || cp == 0x201D) return C_PUNCT;
    if (cp >= 0x20A0 && cp <= 0x27BF) return C_SYMBOL;
    return C_LETTER;
}

// Pre-tokenizer matching llama.cpp's joyai-llm / deepseek3-llm expression list
static std::vector<std::string> pre_tokenize(const std::string& text) {
    std::vector<std::string> pieces;
    const size_t n = text.size();
    size_t pos = 0;

    auto cp_at = [&](size_t p, int* len) -> uint32_t {
        CpInfo c = decode_cp(text, p);
        *len = c.len;
        return c.cp;
    };

    while (pos < n) {
        int len = 1;
        uint32_t cp = cp_at(pos, &len);
        int cls = classify_cp(cp);

        // rule 1: numbers, up to 3 per token
        if (cls == C_NUMBER) {
            size_t end = pos;
            int count = 0;
            while (end < n && count < 3) {
                int l2;
                uint32_t c2 = cp_at(end, &l2);
                if (classify_cp(c2) != C_NUMBER) break;
                end += l2;
                count++;
            }
            pieces.push_back(text.substr(pos, end - pos));
            pos = end;
            continue;
        }

        // rule 2: CJK/kana runs
        if (cp >= 0x4E00 && cp <= 0x9FA5) {
            size_t end = pos;
            while (end < n) {
                int l2;
                uint32_t c2 = cp_at(end, &l2);
                if (!(c2 >= 0x4E00 && c2 <= 0x9FA5)) break;
                end += l2;
            }
            pieces.push_back(text.substr(pos, end - pos));
            pos = end;
            continue;
        }
        if (cp >= 0x3040 && cp <= 0x30FF) {
            size_t end = pos;
            while (end < n) {
                int l2;
                uint32_t c2 = cp_at(end, &l2);
                if (!((c2 >= 0x3040 && c2 <= 0x30FF) || (c2 >= 0x4E00 && c2 <= 0x9FA5) ||
                      (c2 >= 0x3000 && c2 <= 0x303F))) break;
                end += l2;
            }
            pieces.push_back(text.substr(pos, end - pos));
            pos = end;
            continue;
        }

        // rule 3a: ascii punct + letters
        if (cls == C_PUNCT && cp < 0x80) {
            size_t end = pos + len;
            size_t letters = end;
            while (letters < n) {
                int l2;
                uint32_t c2 = cp_at(letters, &l2);
                if (!((c2 >= 'a' && c2 <= 'z') || (c2 >= 'A' && c2 <= 'Z'))) break;
                letters += l2;
            }
            if (letters > end) {
                pieces.push_back(text.substr(pos, letters - pos));
                pos = letters;
                continue;
            }
        }

        // rule 3b: optional non-[NL,L,P,S] prefix + letter run (prefix: space, tab, other)
        {
            size_t start = pos;
            if ((cls == C_SPACE || cls == C_OTHER || cls == C_MARK) && pos + len < n) {
                int l2;
                uint32_t c2 = cp_at(pos + len, &l2);
                if (classify_cp(c2) == C_LETTER) {
                    start = pos + len;
                }
            }
            if (start < n) {
                int l2;
                uint32_t c0 = cp_at(start, &l2);
                if (classify_cp(c0) == C_LETTER) {
                    size_t letters = start;
                    while (letters < n) {
                        int l3;
                        uint32_t c2 = cp_at(letters, &l3);
                        int cc = classify_cp(c2);
                        if (cc != C_LETTER && cc != C_MARK) break;
                        letters += l3;
                    }
                    pieces.push_back(text.substr(pos, letters - pos));
                    pos = letters;
                    continue;
                }
            }
        }

        // rule 3c: optional space + punct/symbol run + optional newline run
        if (cls == C_PUNCT || cls == C_SYMBOL) {
            size_t end = pos;
            while (end < n) {
                int l2;
                uint32_t c2 = cp_at(end, &l2);
                int cc = classify_cp(c2);
                if (cc != C_PUNCT && cc != C_SYMBOL) break;
                end += l2;
            }
            size_t nl = end;
            while (nl < n) {
                int l2;
                uint32_t c2 = cp_at(nl, &l2);
                if (classify_cp(c2) != C_NL) break;
                nl += l2;
            }
            end = nl;
            pieces.push_back(text.substr(pos, end - pos));
            pos = end;
            continue;
        }
        if (cls == C_SPACE) {
            // space followed by punct/symbol run
            int l2;
            size_t nextp = pos + len;
            if (nextp < n) {
                uint32_t c2 = cp_at(nextp, &l2);
                int cc = classify_cp(c2);
                if (cc == C_PUNCT || cc == C_SYMBOL) {
                    size_t end = nextp;
                    while (end < n) {
                        int l3;
                        uint32_t c3 = cp_at(end, &l3);
                        int c4 = classify_cp(c3);
                        if (c4 != C_PUNCT && c4 != C_SYMBOL) break;
                        end += l3;
                    }
                    size_t nl = end;
                    while (nl < n) {
                        int l3;
                        uint32_t c3 = cp_at(nl, &l3);
                        if (classify_cp(c3) != C_NL) break;
                        nl += l3;
                    }
                    pieces.push_back(text.substr(pos, nl - pos));
                    pos = nl;
                    continue;
                }
            }
            // plain whitespace run (with the newline-run rule)
            size_t m = pos;
            size_t first_nl_end = std::string::npos;
            while (m < n) {
                int l3;
                uint32_t c3 = cp_at(m, &l3);
                int cc = classify_cp(c3);
                if (cc != C_NL && cc != C_SPACE) break;
                m += l3;
                if (cc == C_NL) {
                    size_t e = m;
                    while (e < n) {
                        int l4;
                        uint32_t c4 = cp_at(e, &l4);
                        if (classify_cp(c4) != C_NL) break;
                        e += l4;
                    }
                    first_nl_end = e;
                    break;
                }
            }
            size_t end = (first_nl_end != std::string::npos && first_nl_end < n) ? first_nl_end : m;
            pieces.push_back(text.substr(pos, end - pos));
            pos = end;
            continue;
        }
        if (cls == C_NL) {
            size_t end = pos;
            while (end < n) {
                int l3;
                uint32_t c3 = cp_at(end, &l3);
                if (classify_cp(c3) != C_NL) break;
                end += l3;
            }
            pieces.push_back(text.substr(pos, end - pos));
            pos = end;
            continue;
        }

        // fallback: single codepoint
        pieces.push_back(text.substr(pos, len));
        pos += len;
    }
    return pieces;
}

} // namespace

void Tokenizer::gpt2_merges_encode(const std::string& text, std::vector<int>& output) const {
    ensure_tables();

    std::vector<std::string> pieces;
    if (pre_ == "joyai-llm" || pre_ == "deepseek-llm" || pre_ == "deepseek3-llm" ||
        pre_ == "hunyuan-dense" || pre_ == "hy_v4") {
        pieces = pre_tokenize(text);
    } else {
        // unknown pre: feed the whole text as one piece
        pieces.push_back(text);
    }

    for (const std::string& piece : pieces) {
        // byte-encode into GPT-2 unicode string
        std::string enc;
        enc.reserve(piece.size() * 2);
        for (unsigned char ch : piece) {
            enc += g_byte_to_unicode[ch];
        }

        // split into codepoint symbols
        std::vector<std::string> syms;
        {
            size_t p = 0;
            while (p < enc.size()) {
                CpInfo c = decode_cp(enc, p);
                syms.push_back(enc.substr(p, c.len));
                p += c.len;
            }
        }
        if (syms.empty()) continue;

        // BPE merges by rank
        for (;;) {
            int best_rank = std::numeric_limits<int>::max();
            int best_i = -1;
            std::string best_merged;
            for (int i = 0; i + 1 < (int)syms.size(); i++) {
                std::string key = syms[i] + " " + syms[i + 1];
                auto it = merge_rank_.find(key);
                if (it != merge_rank_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_i = i;
                    best_merged = syms[i] + syms[i + 1];
                }
            }
            if (best_i < 0) break;
            syms[best_i] = best_merged;
            syms.erase(syms.begin() + best_i + 1);
        }

        // map symbols to ids (with byte fallback)
        for (const std::string& s : syms) {
            auto it = token_to_id_.find(s);
            if (it != token_to_id_.end()) {
                output.push_back(it->second);
            } else {
                size_t p = 0;
                while (p < s.size()) {
                    CpInfo c = decode_cp(s, p);
                    std::string ch = s.substr(p, c.len);
                    auto ub = g_unicode_to_byte.find(ch);
                    if (ub != g_unicode_to_byte.end()) {
                        output.push_back(byte_token(ub->second));
                    }
                    p += c.len;
                }
            }
        }
    }
}

} // namespace nt
