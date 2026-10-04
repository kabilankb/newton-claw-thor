#!/usr/bin/env bash
# Deploy the newton-claw skills as REAL directories (OpenClaw's scanner does not
# follow symlinks) into:
#   ~/.openclaw/workspace/skills   (OpenClaw / Gemma TUI)   — if OpenClaw is installed
#   ~/.claude/skills               (coding agents on the Thor)
# plus the Jetson skills a robotics operator needs from NVIDIA's jetson-device-skills
# (if cloned at ~/jetson-device-skills). Re-run after editing any skill.
set -euo pipefail

CLAW_DIR="${NEWTON_CLAW_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SRC="$CLAW_DIR/skills"
JETSON_SRC="${JETSON_SKILLS_DIR:-$HOME/jetson-device-skills}/skills"
OURS=(training inference open close ops inventory)
JETSON=(jetson-diagnostic jetson-memory-audit jetson-print-device-info)   # small on purpose: keep a 12B model focused

python3 "$SRC/inventory/gen_inventory.py"

deploy_to() {
  local dest="$1" prefix="$2"
  mkdir -p "$dest"
  for s in "${OURS[@]}"; do
    rm -rf "${dest:?}/$prefix$s"
    cp -rL "$SRC/$s" "$dest/$prefix$s"
    echo "  + $dest/$prefix$s"
  done
  if [ -d "$JETSON_SRC" ]; then
    for s in "${JETSON[@]}"; do
      [ -d "$JETSON_SRC/$s" ] || continue
      if [ -L "$dest/$s" ]; then continue; fi          # already linked by jetson-device-skills/install.sh
      rm -rf "${dest:?}/$s"; cp -rL "$JETSON_SRC/$s" "$dest/$s"; echo "  + $dest/$s"
    done
  fi
}

echo "→ coding-agent skills"
deploy_to "$HOME/.claude/skills" "newton-"

OC_HOME="${OPENCLAW_HOME:-$HOME/.openclaw}"
if command -v openclaw >/dev/null 2>&1 || [ -d "$OC_HOME" ]; then
  echo "→ OpenClaw skills"
  deploy_to "$OC_HOME/workspace/skills" "newton-"
  mkdir -p "$OC_HOME/workspace/memory"
  cp "$SRC/inventory/INVENTORY.md" "$OC_HOME/workspace/memory/newton-claw-inventory.md"
  if command -v openclaw >/dev/null 2>&1; then
    for s in newton-training newton-inference newton-open-robot newton-close newton-claw-ops newton-inventory; do
      openclaw config set "skills.entries.$s.enabled" true >/dev/null 2>&1 && echo "  enable $s"
    done
    echo "  -> restart the gateway to apply (openclaw daemon restart)"
  fi
else
  echo "(OpenClaw not installed — skipped; see README 'Gemma brain')"
fi
