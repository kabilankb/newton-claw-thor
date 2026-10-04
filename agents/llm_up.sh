#!/usr/bin/env bash
# Start the local model server for the agent (idempotent): a user-level Ollama in
# ~/ollama-user serving on 0.0.0.0:8000 (reachable from sandboxes as an OpenAI-compatible
# endpoint at :8000/v1). Separate from any system Ollama on :11434.
#
# First-time install (no sudo):
#   mkdir -p ~/ollama-user && cd ~/ollama-user
#   curl -L https://github.com/ollama/ollama/releases/latest/download/ollama-linux-arm64.tar.zst | tar --zstd -x
#   agents/llm_up.sh && OLLAMA_HOST=127.0.0.1:8000 ~/ollama-user/bin/ollama pull gemma4:12b
set -uo pipefail
DIR="${OLLAMA_USER_DIR:-$HOME/ollama-user}"
curl -sf http://localhost:8000/api/version >/dev/null && { echo "model server: already up (:8000)"; exit 0; }
[ -x "$DIR/bin/ollama" ] || { echo "model server: $DIR/bin/ollama missing — see the header of this script"; exit 1; }
OLLAMA_HOST=0.0.0.0:8000 OLLAMA_MODELS="$DIR/models" OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_KEEP_ALIVE=30m \
  nohup "$DIR/bin/ollama" serve >"$DIR/serve.log" 2>&1 &
for _ in $(seq 1 20); do curl -sf http://localhost:8000/api/version >/dev/null && { echo "model server: up (:8000)"; exit 0; }; sleep 1; done
echo "model server: FAILED — see $DIR/serve.log"; exit 1
