# newton-claw — Structure

## Runtime flow (how a prompt becomes a GPU job on the Thor)
```
You type in the OpenClaw TUI:  "train the Go2"
        │
        ▼  Gemma picks a skill and runs ONE command
skills/training/SKILL.md ──► agents/openclaw/claw train --task Newton-Velocity-Flat-Unitree-Go2-v0
        │
        ▼  HTTP :5561
newton_sim/scripts/newton_control_server.py      (one GPU job at a time; 409 if busy)
        │  POST /train ──► .venv/bin/python newton_lab/scripts/train.py ...
        │  POST /play | /eval ──► newton_lab/scripts/play.py ...        (inference)
        │  POST /open ──► python -m newton.examples <robot example>     (viewer)
        ▼
Newton model (N replicated worlds) ─ SolverMuJoCo (MuJoCo Warp, CUDA graph) ─ rsl_rl PPO (PyTorch)
        │                                                    all on the Thor GPU, zero-copy
        ▼
newton_lab/logs/rsl_rl/<task>/<run>/  model_*.pt · policy.onnx · policy.pt · params.json · tensorboard
```

## Tree
```
newton-claw/
├── README.md  STRUCTURE.md  HISTORY.md  AGENTS.md  DEMO.md  TALK.md
├── setup_thor.sh               # venv + deps + GPU self-test (+ --skills)
├── requirements-thor.txt
├── newton_lab/                 # RL side (replaces isaac_lab/)
│   ├── tasks_registry.json     #   canonical task list → GET /envs, validated by /train
│   ├── newton_lab/
│   │   ├── common.py           #   ppo_cfg, knob overrides, run dirs, viewers, ONNX export, OnnxPolicy
│   │   └── envs/
│   │       ├── __init__.py     #   task registry (id → env class, cfg, PPO defaults)
│   │       ├── base.py         #   NewtonVecEnv: replicate → SolverMuJoCo → torch views → rsl_rl VecEnv
│   │       ├── cartpole.py     #   Newton-Cartpole-v0
│   │       ├── mjcf_locomotion.py   # Newton-Ant-v0, Newton-Humanoid-v0 (torque control)
│   │       └── legged_velocity.py   # Go2 / ANYmal C / G1 velocity tracking (PD position control)
│   ├── scripts/
│   │   ├── train.py            #   rsl_rl PPO; exports policy.onnx + policy.pt at the end
│   │   ├── play.py             #   inference: viewer (gl/viser/usd) or --eval metrics; ckpt / ONNX / --pretrained
│   │   ├── export.py           #   re-export a checkpoint
│   │   └── list_tasks.py       #   print registry / --check against code
│   └── logs/                   #   runs (gitignored)
├── newton_sim/                 # host control side (replaces isaac_sim/)
│   ├── robots.json             #   viewer catalog: name → newton.examples entry
│   ├── scripts/newton_control_server.py
│   └── _control_logs/          #   one log per launched job (gitignored)
├── agents/openclaw/
│   ├── claw                    #   THE CLI
│   ├── clawup.sh               #   idempotently start the control server
│   ├── tui.sh                  #   OpenClaw TUI with the server guaranteed up
│   ├── deploy.sh               #   copy skills into the agents' skill folders, enable them
│   └── model.json              #   Gemma via Ollama
├── skills/
│   ├── skills.md               #   MASTER router
│   ├── training/ inference/ open/ close/ ops/   # one `claw` command each
│   └── inventory/              #   gen_inventory.py → INVENTORY.md (ground truth for the model)
├── policies/                   # policy_skill.md (router) + rsl_rl.md (knobs, rewards, symptom → fix)
├── sandbox/policies/newton-control-policy.yaml   # sandbox egress to host:5561
└── research_papers/
```

## Control API (`http://<thor>:5561`)
| Method | Path | Body / result |
|---|---|---|
| GET | `/envs` `/robots` `/runs` `/device` `/tuning` | registries, trained runs, Thor snapshot, tuning guide |
| GET | `/status` | running job + `progress` (iteration, mean_reward, eta, eval `result`) |
| GET | `/logs?lines=N` | tail of the active job log |
| POST | `/train` | `{task, num_envs?, max_iterations?, seed?, resume?, viewer?, params?{}, overrides?[]}` |
| POST | `/play` `/eval` | `{task, pretrained?, checkpoint?, onnx?, num_envs?, steps?, viewer?, command?[vx,vy,yaw]}` |
| POST | `/open` | `{robot, viewer?, world_count?}` |
| POST | `/stop` | stop the active job |

`viewer`: `none` (headless) · `viser` (browser, port 8090) · `gl` (window on the Thor's display).

## Ports
| Port | Server |
|---|---|
| 5561 | Newton Control Server |
| 8090 | viser browser viewer (only while a `--web` job runs) |
| 11434 | Ollama (Gemma) |
