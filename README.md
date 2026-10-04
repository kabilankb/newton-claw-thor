# newton-claw-thor

https://github.com/user-attachments/assets/e9cc0573-3e8c-48d6-9bf4-6b49fe28426f

Train **and** run robot policies in the **[Newton](https://github.com/newton-physics/newton)
physics engine** on a single **NVIDIA Jetson AGX Thor** — by talking to a local agent
(**Gemma 4** in the **NemoClaw** TUI) that turns each request into one command.

This repository is the Newton rebuild of *isaac-claw*
([kabilankb/Dell_nemoclaw_agent](https://github.com/kabilankb/Dell_nemoclaw_agent), which
drives Isaac Sim + Isaac Lab on a Dell Pro Max GB10). Isaac Sim and Isaac Lab are **gone**
here: physics is Newton (MuJoCo-Warp solver), training is rsl_rl PPO, and everything runs
natively on the Thor — no Omniverse/Kit, no x86 workstation.

> 🎬 **[DEMO.md](DEMO.md)** — step-by-step demos to run by typing into the OpenClaw TUI.
> 📁 **[STRUCTURE.md](STRUCTURE.md)** — annotated layout + runtime flow.
> 🕒 **[HISTORY.md](HISTORY.md)** — what was built and why (including the Newton port notes).

## How it works
```
 You ── "train the Go2"
  ▼
 AGENT — Gemma 4 12B, local (Ollama :8000)
   A. NemoClaw TUI in sandbox "newton"  → runs ONE command:  claw train --task ...
   B. agents/chat.py in a terminal      → calls ONE tool:    newton__train_task
  ▼  HTTP
 NEWTON CONTROL SERVER :5561 — one GPU job at a time (train | play | eval | open | stop)
  ▼  starts one process
 GPU JOB — Newton (MuJoCo-Warp, N parallel worlds) ⇄ rsl_rl PPO / trained policy
  ▼
 newton_lab/logs/rsl_rl/<task>/<run>/   model_*.pt · policy.onnx · policy.pt
```
Everything — the language model, the simulator, training and inference — runs on the Thor.
By default every job opens a window on the Thor's monitor; `--web` streams to a browser on
port 8090 and `--headless` draws nothing.

## What runs where

| Layer | Isaac version (isaac-claw) | This repository |
|---|---|---|
| Hardware | Dell Pro Max GB10 | **Jetson AGX Thor** (JetPack 7 / L4T R38, aarch64, CUDA 13, 128 GB unified memory) |
| Physics | Isaac Sim (PhysX, Kit) | **Newton 1.6** — `SolverMuJoCo` (MuJoCo Warp) on the GPU |
| Env framework | Isaac Lab + robot_lab | **`newton_lab/`** — vectorized Newton envs, zero-copy Warp↔PyTorch |
| Trainer | rsl_rl / cusrl / skrl | **rsl_rl 5.5 PPO** |
| Inference | Isaac Lab `play.py` | **`play.py`** — rsl_rl checkpoint, or exported **ONNX via Warp-NN** (no PyTorch in the loop) |
| Viewer | Isaac Sim window | Newton **GL window** on the Thor, or **viser in a browser** (`http://<thor>:8090`) |
| Control API | `isaac_control_server.py` :5561 | **`newton_control_server.py`** :5561 (same one-GPU-job rule) |
| Agent | OpenClaw (Gemma) / NemoClaw | **NemoClaw TUI (Gemma 4 12B)** or `agents/chat.py`, both driving the single `claw` CLI / control server |
| Device skills | — | NVIDIA **[jetson-device-skills](https://github.com/NVIDIA-AI-IOT/jetson-device-skills)** |

## Layout
```
newton-claw/
├── setup_thor.sh          # one-shot install on the Thor (venv, Newton, PyTorch cu130, rsl_rl, skills)
├── requirements-thor.txt  # pinned runtime
├── newton_lab/            # ← replaces isaac_lab/
│   ├── newton_lab/envs/   #   NewtonVecEnv base + cartpole, ant/humanoid, legged velocity (Go2, G1, ANYmal C)
│   ├── scripts/           #   train.py · play.py (inference/eval) · export.py · list_tasks.py
│   └── tasks_registry.json#   task list served by the control server
├── newton_sim/            # ← replaces isaac_sim/
│   ├── scripts/newton_control_server.py   # THE host control API (:5561)
│   └── robots.json        #   robots `claw open` can show
├── agents/openclaw/       # claw (the one CLI) · clawup.sh · deploy.sh · tui.sh · model.json
├── agents/nemoclaw/       # install.sh (provision the sandbox) · RUNBOOK.md · AGENTS-claw.md · newton-mcp.mjs
├── agents/chat.py         # terminal agent: Gemma + newton__* tools
├── agents/llm_up.sh       # start the local model server (:8000)
├── skills/                # model-facing skills: training · inference · open · close · ops · inventory
├── policies/              # rsl_rl tuning guide (knobs, reward weights, symptom → fix)
└── sandbox/               # egress policy for sandboxed runtimes
```

## Quick start (on the Thor)
```bash
git clone https://github.com/kabilankb/newton-claw-thor.git ~/newton-claw
cd ~/newton-claw
./setup_thor.sh --skills          # ~7 GB venv; ends with a GPU self-test; installs the agent skills

C=agents/openclaw/claw
$C up                             # start the control server
$C tasks                          # what can be trained
$C train --task Newton-Velocity-Flat-Unitree-Go2-v0     # 4096 envs, window on the Thor's monitor
$C status                         # iteration, mean reward, ETA
$C play  --task Newton-Velocity-Flat-Unitree-Go2-v0     # inference, in a window
$C close                          # one GPU job at a time
$C eval  --task Newton-Velocity-Flat-Unitree-Go2-v0     # headless metrics → `claw status`
```
No training yet? `claw play --task Newton-Velocity-Flat-Unitree-Go2-v0 --pretrained` runs the
reference policy that ships with newton-assets.

### Watching (GUI is the default)
| Flag | Where you see it |
|---|---|
| none (default) | Newton Viewer window on the Thor's own monitor |
| `--web` | browser at `http://<thor-ip>:8090` |
| `--headless` | nothing drawn — fastest training |

Training in GUI mode still uses all 4096 robots; the window draws the first 36 at up to 30
frames/s, and closing the window lets training continue headless. Set
`NEWTON_DEFAULT_VIEWER=none` (or `viser`) before starting the control server to change the default.
```bash
$C train --task Newton-Velocity-Flat-Unitree-Go2-v0             # training, with a window
$C train --task Newton-Velocity-Flat-Unitree-Go2-v0 --headless  # training, no window
$C play  --task Newton-Velocity-Flat-Unitree-Go2-v0 --command 1.0 0 0   # 16 robots walking forward
$C open  --robot franka_panda                                   # just look at a robot
```

## Tasks
| Task id | Robot | Control | Default iterations |
|---|---|---|---|
| `Newton-Cartpole-v0` | cartpole | force | 150 |
| `Newton-Ant-v0` | MuJoCo ant | torque | 500 |
| `Newton-Humanoid-v0` | MuJoCo humanoid | torque | 1500 |
| `Newton-Velocity-Flat-Unitree-Go2-v0` | Unitree Go2 | joint position | 1000 |
| `Newton-Velocity-Flat-Anymal-C-v0` | ANYmal C | joint position | 1000 |
| `Newton-Velocity-Flat-Unitree-G1-v0` | Unitree G1 | joint position | 3000 |

Add a task: write an env class under `newton_lab/newton_lab/envs/`, register it in
`envs/__init__.py` and `tasks_registry.json` (`scripts/list_tasks.py --check` keeps them in sync).

## Measured on the Thor (2026-10-03/04)
| What | Result |
|---|---|
| Cartpole, 1024 envs, 60 iterations | solved (4.97 / 5.0 s episodes) in **15 s** |
| Go2 velocity tracking, 4096 envs, 600 iterations | **10 min 43 s**, ~88k env-steps/s during training |
| Go2 trained policy, eval on 1024 robots × 20 s | **10 falls**, velocity-tracking error **0.04 m/s** |
| Go2 reference (pretrained) policy, same eval | 18 falls, tracking error 0.13 m/s |
| Inference throughput, 1024 Go2 envs | ~115k env-steps/s; policy forward 0.47 ms |
| ONNX via Warp-NN, 1 env (deployment path) | 0.3–2.4 ms per policy call |
| G1 velocity tracking, 4096 envs, 800 iterations | ~23 min; eval on 1024 robots × 20 s: **1 fall**, tracking error 0.05 m/s |
| Gemma 4 12B tool routing, 9 test prompts | 9 / 9 correct tool and arguments |
| Agent in the NemoClaw sandbox | started cartpole and a 150-iteration Go2 training from chat |
| GUI mode (`play --gui`), 16 Go2 robots | ~45 FPS in the Newton Viewer on the Thor's display |

## Talk to the agent
Start the two host services once per boot, then pick a front end:
```bash
agents/llm_up.sh                 # Gemma 4 12B on a user-level Ollama (:8000)
agents/openclaw/claw up          # Newton control server (:5561)
```
**NemoClaw TUI (sandboxed):**
```bash
nemoclaw newton connect          # enter the sandbox
openclaw tui
```
**Plain terminal (no sandbox):**
```bash
agents/chat.py
```
| You type | What happens |
|---|---|
| what can I train? | lists the six tasks |
| train the Go2 | starts training (4096 robots; a window on the Thor's monitor shows 36) |
| train the Go2 headless | same, no window (fastest) |
| train the G1 with 2048 envs for 1000 iterations | same, with those settings |
| how is the training going? | iteration, mean reward, ETA |
| show me the Go2 walking | runs the trained policy and gives the viewer link |
| evaluate the Go2 policy | headless metrics (falls, tracking error) |
| stop it | stops the running job, frees the GPU |

One-time setup of the model server and the sandbox, and what had to be fixed to make them work
on a Thor, is in [`agents/nemoclaw/RUNBOOK.md`](agents/nemoclaw/RUNBOOK.md). In short:
```bash
# model server (no sudo)
mkdir -p ~/ollama-user && cd ~/ollama-user
curl -L https://github.com/ollama/ollama/releases/latest/download/ollama-linux-arm64.tar.zst | tar --zstd -x
~/newton-claw/agents/llm_up.sh && OLLAMA_HOST=127.0.0.1:8000 bin/ollama pull gemma4:12b
# sandbox: `nemoclaw onboard` (see RUNBOOK for the exact command), then
~/newton-claw/agents/nemoclaw/install.sh newton
```
The same skills are also installed for coding agents on the Thor (`agents/openclaw/deploy.sh`),
next to NVIDIA's `jetson-*` device skills. A host-side OpenClaw
TUI (`agents/openclaw/tui.sh`) is included but untested here — current OpenClaw needs Node ≥ 24.16.

## Notes for Thor
- **Unified memory:** sim, PPO and the LLM share 128 GB. If memory is tight, lower `--num_envs`.
- **One GPU job at a time** — the control server returns 409 for a second one; `claw close` first.
- **Wheels:** PyTorch comes from the `cu130` index (its aarch64 build includes `sm_110`, Thor's GPU
  target); Warp/Newton/MuJoCo are plain PyPI aarch64 wheels. No Jetson-specific index is needed.
- **Disk:** the venv is ~6 GB, Gemma 8 GB; robot assets download to `~/.cache/newton` on first use.
  Keep the disk below ~85% used — NemoClaw's gateway (a small Kubernetes cluster) stops working
  on a nearly full disk.
- **Ports:** 5561 control server · 8000 Gemma · 8090 browser viewer · 8080 NemoClaw gateway.
- The control server binds `0.0.0.0:5561` with no authentication (same as the Isaac version) so a
  sandbox or another machine on the LAN can drive it — keep the Thor on a trusted network.
  (Binding it to `127.0.0.1` with `NEWTON_CONTROL_HOST` also cuts off the NemoClaw sandbox.)

## Not ported from the Isaac version
Warehouse/office scenes and USD scene authoring, the 50 robot_lab tasks, rough terrain, teleop /
MimicGen / leisaac (SO-101), AMP and BeyondMimic, and the 4-layer `agents/core` orchestrator.
They remain in the isaac-claw repository.

## Not yet verified
Full-length training of ANYmal C, Ant and the MuJoCo Humanoid (they build, step and pass a short
smoke run); watching training with `--gui` / `--web`; the stack after a Thor reboot.
