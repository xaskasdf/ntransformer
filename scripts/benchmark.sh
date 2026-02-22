#!/usr/bin/env bash
# ntransformer benchmark script
# Usage: ./scripts/benchmark.sh <model.gguf> [options]
#
# Options:
#   -n N              tokens to generate (default: 32)
#   --resident        test resident mode (all layers in VRAM)
#   --streaming       test streaming mode
#   --n-buffers N...  test specific buffer counts (space-separated, e.g. --n-buffers "2 3")
#   --evict-ollama    send keep_alive:0 to localhost:11434 before each run
#   --output FILE     write JSON results to FILE
#   --prompt TEXT     prompt to use (default: "The meaning of life is")

set -euo pipefail

BINARY="./build/ntransformer"
N_TOKENS=32
TEST_RESIDENT=0
TEST_STREAMING=0
N_BUFFERS_LIST=""
EVICT_OLLAMA=0
OUTPUT_FILE=""
PROMPT="The meaning of life is"
MODEL=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n)
            N_TOKENS="$2"; shift 2 ;;
        --resident)
            TEST_RESIDENT=1; shift ;;
        --streaming)
            TEST_STREAMING=1; shift ;;
        --n-buffers)
            N_BUFFERS_LIST="$2"; shift 2 ;;
        --evict-ollama)
            EVICT_OLLAMA=1; shift ;;
        --output)
            OUTPUT_FILE="$2"; shift 2 ;;
        --prompt)
            PROMPT="$2"; shift 2 ;;
        --binary)
            BINARY="$2"; shift 2 ;;
        -*)
            echo "Unknown option: $1" >&2; exit 1 ;;
        *)
            MODEL="$1"; shift ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 <model.gguf> [options]" >&2
    echo "  -n N              tokens to generate (default: 32)" >&2
    echo "  --resident        test resident mode" >&2
    echo "  --streaming       test streaming mode" >&2
    echo "  --n-buffers NUMS  test specific buffer counts (quoted, e.g. \"2 3\")" >&2
    echo "  --evict-ollama    evict Ollama models before each run" >&2
    echo "  --output FILE     write JSON results to FILE" >&2
    echo "  --prompt TEXT     prompt to use" >&2
    exit 1
fi

if [[ ! -f "$BINARY" ]]; then
    echo "Error: $BINARY not found. Build first: cmake --build build -j" >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Error: Model file not found: $MODEL" >&2
    exit 1
fi

# Print hardware info
echo "=== Hardware Info ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "(nvidia-smi not available)"
echo ""

evict_ollama() {
    if [[ "$EVICT_OLLAMA" -eq 1 ]]; then
        echo "Evicting Ollama models..."
        curl -s http://localhost:11434/api/generate \
            -d '{"model":"","keep_alive":0}' 2>/dev/null || true
        sleep 2
    fi
}

# Results array for JSON output
declare -a RESULTS

run_benchmark() {
    local label="$1"
    local extra_args="$2"

    evict_ollama

    echo "--- Running: $label ---"
    local cmd="$BINARY -m $MODEL -p \"$PROMPT\" -n $N_TOKENS $extra_args"
    echo "Command: $cmd"

    # Capture stderr (timing info is printed there)
    local output
    output=$(eval "$cmd" 2>&1) || true

    # Parse timing lines — ntransformer format:
    #   Prompt: N tokens, X.X ms (X.X tok/s)
    #   Decode: N tokens, X.X ms (X.X tok/s)
    local prefill_tps=""
    local decode_tps=""
    local prefill_ms=""
    local decode_ms=""
    local decode_tokens=""

    while IFS= read -r line; do
        if [[ "$line" =~ ^Prompt:\ ([0-9]+)\ tokens,\ ([0-9.]+)\ ms\ \(([0-9.]+)\ tok/s\) ]]; then
            prefill_ms="${BASH_REMATCH[2]}"
            prefill_tps="${BASH_REMATCH[3]}"
        elif [[ "$line" =~ ^Decode:\ ([0-9]+)\ tokens,\ ([0-9.]+)\ ms\ \(([0-9.]+)\ tok/s\) ]]; then
            decode_tokens="${BASH_REMATCH[1]}"
            decode_ms="${BASH_REMATCH[2]}"
            decode_tps="${BASH_REMATCH[3]}"
        fi
    done <<< "$output"

    echo "  Prefill: ${prefill_ms:-N/A} ms  (${prefill_tps:-N/A} tok/s)"
    echo "  Decode:  ${decode_ms:-N/A} ms / ${decode_tokens:-N/A} tokens  (${decode_tps:-N/A} tok/s)"
    echo ""

    RESULTS+=("{\"label\":\"$label\",\"prefill_ms\":\"${prefill_ms:-null}\",\"prefill_tps\":\"${prefill_tps:-null}\",\"decode_ms\":\"${decode_ms:-null}\",\"decode_tokens\":\"${decode_tokens:-null}\",\"decode_tps\":\"${decode_tps:-null}\"}")
}

# Default: run at least one test
if [[ "$TEST_RESIDENT" -eq 0 && "$TEST_STREAMING" -eq 0 && -z "$N_BUFFERS_LIST" ]]; then
    TEST_RESIDENT=1
    TEST_STREAMING=1
fi

# Run tests
if [[ "$TEST_RESIDENT" -eq 1 ]]; then
    run_benchmark "resident" ""
fi

if [[ "$TEST_STREAMING" -eq 1 ]]; then
    run_benchmark "streaming (auto)" "--streaming"
fi

if [[ -n "$N_BUFFERS_LIST" ]]; then
    for nb in $N_BUFFERS_LIST; do
        run_benchmark "streaming (n-buffers=$nb)" "--streaming --n-buffers $nb"
    done
fi

# Print summary table
echo "=== Results Summary ==="
printf "%-28s %13s %13s %13s\n" "Config" "Prompt (ms)" "Decode (ms)" "Tok/s"
printf "%-28s %13s %13s %13s\n" "------" "-----------" "-----------" "-----"
for r in "${RESULTS[@]}"; do
    label=$(echo "$r" | grep -o '"label":"[^"]*"' | cut -d'"' -f4)
    p_ms=$(echo "$r" | grep -o '"prefill_ms":"[^"]*"' | cut -d'"' -f4)
    d_ms=$(echo "$r" | grep -o '"decode_ms":"[^"]*"' | cut -d'"' -f4)
    decode=$(echo "$r" | grep -o '"decode_tps":"[^"]*"' | cut -d'"' -f4)
    printf "%-28s %13s %13s %13s\n" "$label" "${p_ms:-N/A}" "${d_ms:-N/A}" "${decode:-N/A}"
done

# Write JSON output
if [[ -n "$OUTPUT_FILE" ]]; then
    echo "Writing results to $OUTPUT_FILE..."
    printf '[\n' > "$OUTPUT_FILE"
    for i in "${!RESULTS[@]}"; do
        if [[ $i -lt $((${#RESULTS[@]} - 1)) ]]; then
            printf '  %s,\n' "${RESULTS[$i]}" >> "$OUTPUT_FILE"
        else
            printf '  %s\n' "${RESULTS[$i]}" >> "$OUTPUT_FILE"
        fi
    done
    printf ']\n' >> "$OUTPUT_FILE"
    echo "Results written to $OUTPUT_FILE"
fi
