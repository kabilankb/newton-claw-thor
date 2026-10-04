# NemoClaw on the Thor — talk to the TUI, the agent trains the robot

```
you ──► NemoClaw TUI (sandbox "newton", Gemma 4 12B) ──► claw train ... ──► host control server :5561
                                                                               └─► Newton + rsl_rl on the Thor GPU
```
Inside the sandbox the agent runs ONE `claw` command per request (the six `newton-*` skills
plus a routing table in AGENTS.md tell it which).

## Every time (this Thor is already set up)
```bash
cd ~/newton-claw
agents/llm_up.sh                        # Gemma on :8000 (user-level Ollama)
agents/openclaw/claw up                 # Newton control server on :5561
nemoclaw newton connect                 # enter the sandbox
openclaw tui                            # the TUI
```
If the Thor was rebooted and `nemoclaw newton connect` cannot reach the gateway:
`openshell gateway start --name nemoclaw`.

## One-time setup (already done here on 2026-10-03; repeat only after a rebuild)
```bash
# sandbox with the local Gemma endpoint ("vllm" = any OpenAI-compatible server on :8000)
NEMOCLAW_EXPERIMENTAL=1 NEMOCLAW_PROVIDER=vllm NEMOCLAW_MODEL=gemma4:12b \
NEMOCLAW_SANDBOX_NAME=newton NEMOCLAW_POLICY_MODE=skip \
  node ~/nemoclaw-local/bin/nemoclaw.js onboard --non-interactive
# the gateway sends the model name "vllm-local": alias it to Gemma
OLLAMA_HOST=127.0.0.1:8000 ~/ollama-user/bin/ollama cp gemma4:12b vllm-local
# tools for the agent
agents/nemoclaw/install.sh newton
```
`~/nemoclaw-local` is a copy of `/usr/lib/node_modules/nemoclaw` with one Dockerfile line
changed: the stock `npm install --omit=dev` crashes in the image build ("reading edgesOut").

## What to say
| You type | Command the agent runs |
|---|---|
| what can I train? | `claw tasks` |
| train the Go2 | `claw train --task Newton-Velocity-Flat-Unitree-Go2-v0` |
| train the G1 with 2048 envs for 1000 iterations | `claw train --task ...G1-v0 --num_envs 2048 --max_iterations 1000` |
| how is the training going? | `claw status` |
| stop the training | `claw close` |
| show me the Go2 walking | `claw play --task ...` → open the printed URL (port 8090) in a browser |
| evaluate the Go2 policy | `claw eval --task ...`, later `claw status` |
| how much memory is free? | `claw device` |

## What had to be true for this to work (lessons from 2026-10-03)
- **Disk below ~85% used.** The gateway is a k3s cluster in Docker; on a 99%-full disk its kubelet
  evicts every pod and onboarding fails with `K8s namespace not ready`.
- **A current Ollama.** The system Ollama 0.12.5 (:11434, loopback only) crashes on agent-length
  prompts. `agents/llm_up.sh` runs Ollama 0.35 from `~/ollama-user` on `0.0.0.0:8000` with a
  32k context (the sandbox agent's prompt is ~15k tokens).
- **Egress policy without L7 rules.** With `protocol: rest` + rules OpenShell rejects plain-HTTP
  proxy requests ("endpoint has L7 rules; use CONNECT"). The policy also has to name the real
  interpreter binary (`/usr/bin/python3.11`), not just the `python3` symlink.
- **Port 8080 belongs to the OpenShell gateway**, so the Newton browser viewer is on 8090.

## Without the sandbox: the same agent in a plain terminal
`agents/chat.py` — Gemma with dedicated `newton__*` tools (`newton-mcp.mjs`), no NemoClaw needed.

## Troubleshooting
- Agent says the control server is unreachable → on the host: `agents/openclaw/claw up`; then
  re-run `agents/nemoclaw/install.sh newton` (it prints OK/FAIL for the sandbox → host path).
  Denials are visible with `openshell logs newton | grep 5561`.
- "a GPU job is already running" → tell the agent to stop it, then repeat.
- The model answers in prose instead of running a command → start a fresh TUI session.
- Model errors → `agents/llm_up.sh`; `curl -s localhost:8000/v1/models` must list `vllm-local`.
