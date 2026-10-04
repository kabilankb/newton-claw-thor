#!/usr/bin/env bash
# One-shot setup of newton-claw on a Jetson Thor. Run ON the Thor, from the repo root.
#
#   ./setup_thor.sh            # python venv + Newton + PyTorch (CUDA 13) + rsl_rl, then a GPU self-test
#   ./setup_thor.sh --skills   # also install the agent skills (NVIDIA jetson-device-skills + newton-claw skills)
#   ./setup_thor.sh --check    # only run the self-test
#
# No sudo needed. Needs ~7 GB free disk for the venv (PyTorch's CUDA libraries are most of it).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH="$HOME/.local/bin:$PATH"
VENV=.venv

check() {
  "$VENV/bin/python" - <<'PY'
import torch, warp as wp, newton, rsl_rl
wp.config.quiet = True
wp.init()
assert torch.cuda.is_available(), "PyTorch does not see the GPU"
assert wp.get_device("cuda:0").is_cuda
print(f"OK  newton {newton.__version__} | warp {wp.__version__} | torch {torch.__version__} | GPU {torch.cuda.get_device_name(0)}")
PY
}

if [ "${1:-}" = "--check" ]; then check; exit 0; fi

[ "$(uname -m)" = "aarch64" ] || echo "WARNING: not aarch64 — this script targets Jetson Thor."
command -v uv >/dev/null || { echo "installing uv (user-level) ..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
free_gb=$(df --output=avail -BG . | tail -1 | tr -dc 0-9)
[ -d "$VENV" ] || [ "$free_gb" -ge 7 ] || { echo "ERROR: only ${free_gb} GB free; the venv needs ~7 GB."; exit 1; }

[ -d "$VENV" ] || uv venv --python 3.12 "$VENV"
uv pip install -p "$VENV" -r requirements-thor.txt \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
check

if [ "${1:-}" = "--skills" ]; then
  # NVIDIA's Jetson device skills (diagnostics, memory audit, LLM serving, ...) for coding agents on the Thor.
  [ -d "$HOME/jetson-device-skills" ] || git clone --depth 1 https://github.com/NVIDIA-AI-IOT/jetson-device-skills.git "$HOME/jetson-device-skills"
  "$HOME/jetson-device-skills/install.sh"
  agents/openclaw/deploy.sh
fi

echo
echo "Next:  agents/openclaw/claw up && agents/openclaw/claw train --task Newton-Cartpole-v0"
