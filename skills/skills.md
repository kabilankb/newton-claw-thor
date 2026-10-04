# newton-claw — MASTER SKILL ROUTER (load first)

Every request maps to ONE skill, and every skill runs ONE `claw` command
(`~/newton-claw/agents/openclaw/claw`). Physics is **Newton**; the box is a **Jetson Thor**.

| User wants to… | Skill | Command |
|---|---|---|
| train a robot policy, check progress | [`training/`](training/SKILL.md) `newton-training` | `claw train --task <id>` · `claw status` · `claw logs` |
| run / watch / evaluate a policy (inference) | [`inference/`](inference/SKILL.md) `newton-inference` | `claw play --task <id>` · `claw eval --task <id>` |
| look at a robot in the viewer | [`open/`](open/SKILL.md) `newton-open-robot` | `claw open --robot <name>` |
| stop anything, free the GPU | [`close/`](close/SKILL.md) `newton-close` | `claw close` |
| list what exists | [`inventory/`](inventory/SKILL.md) `newton-inventory` | `claw tasks` · `claw robots` · `claw runs` |
| fix "connection refused", check the device | [`ops/`](ops/SKILL.md) `newton-claw-ops` | `claw up` · `claw device` · `claw restart` |
| tune hyperparameters / rewards | [`../policies/rsl_rl.md`](../policies/rsl_rl.md) | `claw train --task <id> knob=value …` |
| Jetson health, memory, thermals, LLM serving | NVIDIA `jetson-*` device skills | (installed by `setup_thor.sh --skills`) |

## Typical pipelines
1. **Train → watch → measure:** `claw train --task X` → (`claw status` until finished) → `claw play --task X` → `claw close` → `claw eval --task X`.
2. **Instant demo, no training:** `claw play --task Newton-Velocity-Flat-Unitree-Go2-v0 --pretrained`.
3. **Switching jobs:** always `claw close` between train / play / open (one GPU job at a time; a second returns 409).

## Hard rules for the model
- ONE shell command per request; no multi-step plans.
- Never invent task ids or robot names — they are listed in `inventory/INVENTORY.md`.
- There is no Isaac Sim, no warehouse scene, no USD authoring here.
