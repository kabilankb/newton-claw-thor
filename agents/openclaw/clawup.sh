#!/usr/bin/env bash
# Bring up the newton-claw host service so everything is drivable from the
# OpenClaw TUI. Idempotent: starts the Newton Control Server only if it isn't
# already answering.
#
# Usage:  ./clawup.sh          # ensure control server up
#         ./clawup.sh status   # report what's up
set -uo pipefail

CLAW_DIR="${NEWTON_CLAW_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${NEWTON_PYTHON:-$CLAW_DIR/.venv/bin/python}"
PORT="${NEWTON_CONTROL_PORT:-5561}"
BASE="http://localhost:$PORT"
LOGDIR="$CLAW_DIR/newton_sim/_control_logs"; mkdir -p "$LOGDIR"

up() { curl -sf "$BASE/envs" >/dev/null 2>&1; }

if [ "${1:-}" = "status" ]; then
  up && { echo "control server: UP"; curl -s "$BASE/status"; } || echo "control server: DOWN"
  exit 0
fi

if up; then echo "control server: already up ($BASE)"; exit 0; fi
[ -x "$PY" ] || { echo "control server: $PY missing — run ./setup_thor.sh first"; exit 1; }
echo "control server: starting ..."
NEWTON_CLAW_DIR="$CLAW_DIR" NEWTON_PYTHON="$PY" \
  nohup "$PY" "$CLAW_DIR/newton_sim/scripts/newton_control_server.py" \
  >"$LOGDIR/control_server.out" 2>&1 &
for _ in $(seq 1 30); do up && { echo "control server: up ($BASE)"; exit 0; }; sleep 1; done
echo "control server: FAILED to come up — see $LOGDIR/control_server.out"; exit 1
