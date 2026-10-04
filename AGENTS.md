# AGENTS.md

Guidance for coding agents (and new contributors) working with the code in this repository.

## What this repo is

newton-claw trains and runs robot policies in **Newton physics** on an
**NVIDIA Jetson AGX Thor**, driven by a local-model TUI (**OpenClaw + Gemma**). It is the Newton
rebuild of isaac-claw (github.com/kabilankb/Dell_nemoclaw_agent): Isaac Sim / Isaac Lab are not used here.

Read first: `README.md` (overview, quick start, measured numbers), `STRUCTURE.md` (tree +
runtime flow), `HISTORY.md` (why things are the way they are).

## Environment contract

| Var | Default | Meaning |
|---|---|---|
| `NEWTON_CLAW_DIR` | repo root | this repo |
| `NEWTON_PYTHON` | `<repo>/.venv/bin/python` | python with newton + torch + rsl_rl (built by `setup_thor.sh`) |
| `NEWTON_CONTROL_PORT` | `5561` | control server port |
| `NEWTON_CONTROL_HOST` | `0.0.0.0` (server bind) / auto-probe (claw) | |
| `NEWTON_VISER_PORT` | `8090` | browser viewer |
| `NEWTON_LAB_LOGS` | `newton_lab/logs/rsl_rl` | training runs |

## Architecture

```
TUI prompt → skill picks ONE `claw` command → Newton Control Server (:5561)
           → spawns ONE GPU job under $NEWTON_PYTHON:
               train.py | play.py (--eval) | python -m newton.examples <robot example>
```

- `agents/openclaw/claw` — the single CLI (stdlib only): `up`, `tasks`, `train`, `play`, `eval`,
  `open`, `close`, `status`, `logs`, `robots`, `runs`, `device`, `restart`.
- `newton_sim/scripts/newton_control_server.py` — THE host API (stdlib only; never imports
  newton/torch). Single-job state machine: **one GPU job at a time**, a second returns 409.
  `/status` parses the rsl_rl log into `progress` (iteration, mean reward, ETA, eval result).
- `newton_lab/newton_lab/envs/base.py` — `NewtonVecEnv` (an rsl_rl `VecEnv`): one `newton.Model`
  with `num_envs` replicated worlds, `SolverMuJoCo`, the decimated step captured as a CUDA graph,
  and `state.joint_q / joint_qd / control.joint_target_q / joint_f` exposed as zero-copy torch
  views of shape `(num_envs, n)`.
- `newton_lab/newton_lab/envs/legged_velocity.py` — Go2 / G1 / ANYmal C velocity tracking. The
  observation layout matches Isaac Lab's, so the reference ONNX policies from newton-assets run
  in it unchanged (`play.py --pretrained`) — use that as the regression test for env changes.
- `newton_lab/newton_lab/common.py` — rsl_rl config (`ppo_cfg`), knob parsing, run dirs, viewers,
  `export_deploy_onnx` (Gemm/Elu-only graph with obs-normalization folded in, because Warp-NN's
  ONNX runtime supports only Gemm / Elu / LSTM / Squeeze).

## Common commands (run on the Thor)

```bash
./setup_thor.sh [--skills | --check]
agents/openclaw/claw up | tasks | status | logs | close | restart
agents/openclaw/claw train --task <id> [--num_envs N] [--max_iterations N] [--web|--gui] [knob=value ...]
agents/openclaw/claw play  --task <id> [--pretrained] [--web|--gui] [--command VX VY YAW]
agents/openclaw/claw eval  --task <id> [--pretrained]

# direct (no server)
.venv/bin/python newton_lab/scripts/train.py --task Newton-Cartpole-v0 --num_envs 1024 --max_iterations 60
.venv/bin/python newton_lab/scripts/play.py  --task <id> --eval [--onnx PATH --num_envs 1]
.venv/bin/python newton_lab/scripts/list_tasks.py --check     # registry JSON vs code
python3 skills/inventory/gen_inventory.py                    # regenerate INVENTORY.md
agents/openclaw/deploy.sh                                    # re-deploy skills after editing them
```

There is no test suite. The checks that matter: `setup_thor.sh --check`, the cartpole run above
(should reach ~295/300 mean episode length), and
`play.py --task Newton-Velocity-Flat-Unitree-Go2-v0 --pretrained --eval` (≈19.8 s survival).

## Constraints that caused real bugs (see HISTORY.md)

- **State buffers:** torch views alias `state_0`. `_simulate()` must leave the final state in
  `state_0` (it copies back when `decimation` is odd). Never rebind `env.state_0`.
- **Inference mode:** rsl_rl steps the env inside `torch.inference_mode()`. Keep env buffers as
  pre-allocated tensors updated in place (`buf[:] = ...`); don't store incoming action tensors.
- **Reward shaping:** with summed penalties larger than the tracking reward, PPO learns to fall
  immediately. `only_positive_rewards` clips the per-step sum at 0 before the termination penalty.
- **Unbounded PD targets:** MuJoCo position actuators have no effort limit here; legged tasks use
  `init_noise_std=0.5` and `clip_actions=10`.
- **Gravity:** asset USDs can carry their own gravity (G1's is zero). `NewtonVecEnv` sets it right
  after `finalize()`; `SolverMuJoCo` copies gravity at construction, so later changes are ignored.
- **Contact buffers:** `narrowphase overflow` in a log means `nconmax`/`njmax` in the task's
  `solver` cfg is too small — contacts get dropped and robots sink.
- **Newton 1.6 API:** `control.joint_target_q` uses the coordinate layout (7 slots for a free
  base); joint velocities for a free joint are world-frame linear then angular.
- **Local models execute best with ONE deterministic command.** Skills are imperative ("run this
  one command"). Don't reintroduce multi-call flows. OpenClaw's skill scanner does not follow
  symlinks — `deploy.sh` copies real dirs; re-run it after editing a skill.
- **Never invent ids:** tasks and robots come from `newton_lab/tasks_registry.json` and
  `newton_sim/robots.json`; `skills/inventory/INVENTORY.md` is generated from them.
