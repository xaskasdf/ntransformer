#ifndef NTRANSFORMER_H
#define NTRANSFORMER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// Opaque handle
typedef void* nt_engine_t;

// Create and destroy engine
nt_engine_t nt_engine_create(void);
void        nt_engine_destroy(nt_engine_t engine);

// Load model
int nt_engine_load(nt_engine_t engine, const char* model_path);

// Generate text
// Returns allocated string (caller must free with nt_free)
char* nt_engine_generate(
    nt_engine_t engine,
    const char* prompt,
    int max_tokens,
    float temperature,
    int top_k,
    float top_p
);

// Free allocated string
void nt_free(char* ptr);

// Get model info
int nt_engine_vocab_size(nt_engine_t engine);
int nt_engine_n_layers(nt_engine_t engine);
int nt_engine_hidden_size(nt_engine_t engine);

#ifdef __cplusplus
}
#endif

#endif // NTRANSFORMER_H
