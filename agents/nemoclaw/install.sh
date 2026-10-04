#!/usr/bin/env bash
# Provision a NemoClaw sandbox so its TUI agent can train / run robots in Newton
# on this Thor. Verified with NemoClaw 0.1.0 / OpenShell 0.0.11 / OpenClaw 2026.3.11
# (sandbox = a pod inside the OpenShell gateway, reached with `openshell sandbox ...`).
#
#   ./install.sh <sandbox>          (list sandboxes: nemoclaw list)
#
# What it does:
#   1. egress policy: sandbox -> host.openshell.internal:5561 (the Newton control server)
#   2. uploads the `claw` CLI and the six newton skills into the sandbox
#   3. appends command-routing guidance to the sandbox agent's AGENTS.md
# The agent then works the same way as on the host: ONE `claw` command per request.
# (OpenClaw 2026.3.11 has no MCP-server config, so newton-mcp.mjs is not used here;
#  it is used by agents/chat.py and by newer OpenClaw versions.)
set -euo pipefail

SB="${1:-${NEMOCLAW_SANDBOX:-}}"
[[ -z "$SB" ]] && { echo "usage: install.sh <sandbox>   (nemoclaw list)" >&2; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

echo "  [1/3] egress policy (host:5561)"
python3 "$HERE/merge_policy.py" "$SB" "$REPO/sandbox/policies/newton-control-policy.yaml" "$TMP/policy.yaml"
openshell policy set --policy "$TMP/policy.yaml" --wait "$SB" | tail -1

echo "  [2/3] claw CLI + skills"
openshell sandbox upload "$SB" "$REPO/agents/openclaw/claw" /sandbox/newton-claw/agents/openclaw/ >/dev/null
for s in training inference open close ops inventory; do
  openshell sandbox upload "$SB" "$REPO/skills/$s" "/sandbox/.openclaw/workspace/skills/newton-$s" >/dev/null
done
openshell sandbox upload "$SB" "$HERE/AGENTS-claw.md" /sandbox/.newton-install/ >/dev/null

echo "  [3/3] AGENTS.md guidance + connectivity check"
openshell sandbox connect "$SB" <<'IN_SANDBOX' 2>&1 | grep -oE "(OK|FAIL): [a-zA-Z].*" | tail -1 || echo "FAIL: no answer from the sandbox"
A=/sandbox/.openclaw/workspace/AGENTS.md; touch "$A"
# replace any earlier copy of our section (it is always the last block), then append the current one
python3 - "$A" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read(); i = s.find("## Newton robot training (RUN ONE COMMAND)")
open(p, "w").write((s[:i].rstrip() + "\n" if i >= 0 else s) + open("/sandbox/.newton-install/AGENTS-claw.md").read())
PY
python3 /sandbox/newton-claw/agents/openclaw/claw tasks 2>/dev/null | grep -q "Newton-Cartpole-v0" \
  && echo "OK"": the sandbox reaches the Newton control server" \
  || echo "FAIL"": control server unreachable from the sandbox (is it up on the host? agents/openclaw/claw up)"
exit
IN_SANDBOX

echo
echo "Now:   nemoclaw $SB connect      then inside:   openclaw tui"
echo "  try:  \"what can I train?\"   \"train the Go2\"   \"how is the training going?\""
