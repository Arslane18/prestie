#!/usr/bin/env bash
# Start the local model for Prestie (PRESTIE_LLM=local): Qwen3.5 9B on the GPU.
#
# Sized to run while playing WoW on a 12 GB card (measured: ~6.1 GB for the
# model, > 11 GB in total with the game):
#   -c 32768           context; the hybrid architecture keeps the KV cache small
#   -ngl 99            every layer on the GPU
#   -fa on, -ctk/-ctv  flash attention, KV cache quantized to 8 bits
#   -np 1              one slot: one player, the whole context for each request
#   --jinja            the model's chat template, needed for structured tool calls
#   --no-mmproj        no vision projector: Prestie sends no images
#   --api-key          the server accepts any CORS origin: a key keeps web pages out
#   --alias            the model name the server reports (else the file path):
#                      must equal PRESTIE_LOCAL_LLM_MODEL, which the eval checks
#
# Override the paths with LLAMA_CPP_DIR / PRESTIE_LOCAL_LLM_GGUF; the key is
# read from PRESTIE_LOCAL_LLM_API_KEY and the alias from PRESTIE_LOCAL_LLM_MODEL
# (environment or .env).
set -euo pipefail

cd "$(dirname "$0")/.."
env_value() {  # $1 from the environment, else from .env
    local value="${!1:-}"
    if [[ -z "$value" && -f .env ]]; then
        value="$(grep -E "^$1=" .env | cut -d= -f2- || true)"
    fi
    echo "$value"
}
PRESTIE_LOCAL_LLM_API_KEY="$(env_value PRESTIE_LOCAL_LLM_API_KEY)"
MODEL_ALIAS="$(env_value PRESTIE_LOCAL_LLM_MODEL)"
MODEL_ALIAS="${MODEL_ALIAS:-qwen3.5-9b}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/tools/llama.cpp}"
GGUF="${PRESTIE_LOCAL_LLM_GGUF:-$HOME/tools/models/Qwen3.5-9B-Q4_K_M.gguf}"
api_key=()
if [[ -n "${PRESTIE_LOCAL_LLM_API_KEY:-}" ]]; then
    api_key=(--api-key "$PRESTIE_LOCAL_LLM_API_KEY")
else
    echo "warning: PRESTIE_LOCAL_LLM_API_KEY is not set, the server runs without a key" >&2
fi

exec "$LLAMA_CPP_DIR/build/bin/llama-server" \
    -m "$GGUF" \
    --host 127.0.0.1 --port 8080 \
    -c 32768 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -np 1 \
    --jinja --no-mmproj --alias "$MODEL_ALIAS" \
    "${api_key[@]}"
