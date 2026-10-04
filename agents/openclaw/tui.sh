#!/usr/bin/env bash
# Launch the OpenClaw TUI (Gemma brain) with the newton-claw control server guaranteed up.
# Use this INSTEAD of bare `openclaw tui`.
#
#   ~/newton-claw/agents/openclaw/tui.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NEWTON_CLAW_DIR="${NEWTON_CLAW_DIR:-$(cd "$HERE/../.." && pwd)}"
export NEWTON_PYTHON="${NEWTON_PYTHON:-$NEWTON_CLAW_DIR/.venv/bin/python}"

echo "[claw-tui] ensuring control server ..."
"$HERE/claw" up || echo "[claw-tui] WARNING server not confirmed; the TUI can still run 'claw up'"

# Dedicated clean session (the default "main" may carry stale context from failed attempts).
SESSION="${SESSION:-newton-claw}"
echo "[claw-tui] launching OpenClaw TUI (session: $SESSION)"
exec openclaw tui --session "$SESSION" "$@"
