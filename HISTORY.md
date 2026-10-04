# isaac-claw — Development History

The complete story of building isaac-claw: what was done, what broke, and why.
Commit hashes anchor each step (`git log --reverse`).

## Goal
Consolidate four overlapping repos (`humanoid_nemoclaw`, `leisaac`,
`warehouse_nemoclaw`, the `NemoClaw`/`openclaw` runtimes) into one readable repo
where a local model (Gemma via OpenClaw, Nemotron via NemoClaw) can drive **Isaac
Sim** and **Isaac Lab** from natural-language text in a TUI.

Five objectives:
1. NemoClaw & OpenClaw operate both Isaac Sim and Isaac Lab.
2. Open an environment + robot from a TUI prompt via the local model.
3. One Markdown control panel for policy tuning across RSL-RL/rl_games/skrl/SB3/robomimic.
4. A readable folder structure.
5. A master `skills.md` that routes to all other skills.

Decisions locked up front: **build new** (`~/isaac-claw`, non-destructive copy;
originals untouched) · **reference** `~/IsaacSim`/`~/IsaacLab` via env vars (not
vendored) · **fresh git repo**.

---

## Phase 1 — Build the repo (consolidation)

**`da8a0b5` Initial isaac-claw.** Mapped all four source repos with parallel
explore agents, then assembled the tree by copy: `agents/{core,nemoclaw,openclaw}`,
`skills/` (29-skill library + master `skills.md`), `isaac_lab/` (robot_lab +
leisaac extensions), `isaac_sim/` (python_server ext, warehouse scene, and a new
generalized `isaac_control_server.py`), `policies/` (per-framework tuning, authored
by a sub-agent), `sandbox/`, `research_papers/`. Heavy assets (robot USDs ~450M,
scene textures, datasets) gitignored — 657M on disk, 6.3M in git.

**`e4e19cd` Goal-4 fidelity.** Validation found two gaps vs the reference layout:
the canonical `isaac_lab/scripts/{train,play,eval,teleop,annotate,inference}.py`
entrypoints were missing (real scripts were nested) and `skills/articulation/`
didn't exist. Added thin dispatcher entrypoints + the articulation task skill.

**`c74afc0` Deploy script.** First `deploy.sh` to surface skills to OpenClaw
(via symlinks — later found wrong, see Phase 4).

---

## Phase 2 — Make it drivable from OpenClaw

**`710e770` One-prompt flow.** Generalized `/sim/launch` to a persistent
`serve_sim.py` that enables the `:8226` bridge and stays alive; added `clawup.sh`
bootstrap and the `ops` skill so the model can self-start services.

**`c341df2` The `claw` CLI.** First real test in the TUI: **Gemma narrated a plan
and executed nothing**, and picked the wrong skill (`isaac-sim-orchestrator`).
Root insight: a 12B won't orchestrate five HTTP calls — collapse the flow into ONE
deterministic command. Built `claw` (`open/train/status/...`) and rewrote `task`
as an imperative "run one command" skill.

**`675155a` Skill mis-pick.** `deploy.sh` had left shadowed skills as
`*.pre-isaac-claw.bak` **inside** the scan folder, so OpenClaw saw duplicates and
the stale orchestrator (with a hallucinated `sim_warehouse_v4.usda`) kept winning.
Moved backups out; made the orchestrator description defer to `task`.

**`6081154` Scene templates.** Per the request to decouple robot from environment:
robot height/drive come from the catalog, spawn position from an environment's
named anchors, so the SAME robot works in any environment. Added
`isaac_sim/templates/` (environments + composite/fleet templates) and template
resolution in `claw`.

---

## Phase 3 — Get the simulator actually running (the GPU saga)

First real `claw open` runs surfaced bugs only the hardware could reveal:

**`957bd7a` Bring-up crash.** Log showed Isaac Sim booted and the `:8226` ext
enabled — then `serve_sim` crashed on `stage_utils.is_stage_loading()` (absent in
Isaac Sim 5.1), killing the app before READY → spawn got 502. Replaced with a
version-robust `omni.usd.get_stage_loading_status()` wait.

**`771df01` GUI + readiness + robust spawn.** Three fixes: (1) the spawn snippet
added a duplicate `xformOp:translate` and didn't resolve `{ISAACLAB_NUCLEUS_DIR}`
tokens — fixed by resolving tokens to the asset root in-sim and reusing an existing
translate op (**proven live**: spawned `nova_carter` from the real S3 asset path);
(2) `claw` readiness now probes the `:8226` bridge instead of scraping logs (which
false-timed-out); (3) added `--gui` so the sim opens a visible window (it had been
headless — the "not loading on GUI" complaint).

**`6bc9219` Self-heal.** `claw open --gui` now stops an orphaned/running sim before
relaunching, so it's a single command. (Note: the agent harness cannot kill/launch
Kit itself — signal 16 kills the shell — so this only runs from the user's terminal.)

**`9073730` GUI by default in the TUI.** A person at a TUI wants to see the sim, so
`task` now emits `--gui` by default; `tui.sh` propagates `DISPLAY`.

---

## Phase 4 — Why the TUI still didn't work (model + environment)

Even with clean skills, Gemma kept failing. Two non-obvious causes:

**`893cd24` Symlinks invisible.** `find` proved only 5 real skill dirs existed —
**OpenClaw's scanner does not follow symlinks**, so every isaac-claw skill
(`task`, `ops`, …) was invisible; the model only saw leftover dirs and the
router's *mention* of `isaac-sim-orchestrator`, then hallucinated around it.
Rewrote `deploy.sh` to **copy real dirs** and deploy only the small operational
set (a tiny surface stops a 12B from wandering).

**`af67d5b` Session poison + workspace pollution.** The model kept "recalling"
earlier failures because the TUI resumed the stale `main` session, and the skill
backup dir was still **inside** `~/.openclaw/workspace` (so OpenClaw scanned its
`.bak` skills and a stray `warehouse_sdg.py`). Moved the backup out of the
workspace; `tui.sh` now launches a clean `--session isaac-claw`.

---

## Phase 5 — Close & document

**`a62bebd` Close skill.** Added `claw close` (stop sim OR training + free GPU +
`:8226`) and `skills/close/SKILL.md` so "close the sim" works from text.

**`597f266` STRUCTURE.md.** Full annotated layout + runtime-flow diagram.

---

## Key problems & root causes (reference table)
| Symptom | Root cause | Fix | Commit |
|---|---|---|---|
| Model plans, never executes | 12B won't chain 5 HTTP calls | one `claw` command + imperative skill | `c341df2` |
| Wrong skill (`orchestrator`) picked | duplicate `.bak` skills scanned | move backups out; defer orchestrator | `675155a`,`af67d5b` |
| Skills invisible to model | OpenClaw ignores symlinks | deploy **real copies**, operational only | `893cd24` |
| Model "recalls" old failures | TUI resumed poisoned `main` session | clean `--session isaac-claw` | `af67d5b` |
| Sim crashes before READY | `is_stage_loading` absent in 5.1 | `get_stage_loading_status` wait | `957bd7a` |
| Spawn 502 / `xformOp` error | dead app + dup translate + unresolved tokens | resolve tokens in-sim, reuse op | `771df01` |
| "Not loading on GUI" | `serve_sim` ran headless | `--gui` flag, default in TUI | `771df01`,`9073730` |
| Leftover sim blocks GPU | orphaned `serve_sim` | `claw open` self-heals; `claw close` | `6bc9219`,`a62bebd` |

## Lessons (carried into MEMORY)
- **Local models execute best with ONE deterministic command** and a *tiny* skill
  surface; deep authoring skills must stay out of the TUI workspace.
- **OpenClaw skill scanner ignores symlinks** → deploy real copies (`deploy.sh`).
- **Sessions persist context** → start failed work in a fresh `--session`.
- **The agent harness cannot spawn or kill GPU Kit** (signal 16) — but it CAN
  talk to a running sim over the `:8226` socket. GPU lifecycle is the user's.

## Current state
End-to-end proven: Isaac Sim boots on the GB10, the `:8226` bridge works, and
`/robot/spawn` resolves assets and places robots (validated live). Deterministic
path — `claw open --template warehouse_fleet --gui` / `claw close` — works. The TUI
path is clean (real skills, fresh session); remaining variance is pure local-model
reliability (use `gemma4:31b` if a 12B still narrates).

---

## 2026-10-03 — Newton rebuild for Jetson Thor (this repository)

**Goal:** the same "chat → one command → GPU job" system, but with Isaac Sim / Isaac Lab
replaced by the Newton physics engine, training *and* inference on a Jetson AGX Thor.

**What changed**
- `isaac_lab/` → `newton_lab/`: a small vectorized-env layer on Newton 1.6 (`NewtonVecEnv`),
  six tasks (cartpole, ant, humanoid, Go2 / ANYmal C / G1 velocity tracking), rsl_rl 5.5 PPO.
- `isaac_sim/` → `newton_sim/`: `newton_control_server.py` keeps port 5561 and the one-GPU-job
  rule; endpoints are now train / play / eval / open / stop. No :8226 in-sim bridge — Newton has
  no Kit process to script.
- `claw` gained `play` and `eval` (inference is a first-class job) and `--web` (viser browser
  viewer, the practical way to watch a headless Thor).
- Skills reduced to six operational ones; NVIDIA jetson-device-skills installed alongside.
- Removed here: robot_lab/leisaac, scenes/templates/catalog, `agents/core`,
  `agents/nemoclaw`, sandbox clients, non-rsl_rl tuning docs.

**Bugs that shaped the design**
1. *Policy learned to fall over.* First Go2 run: mean episode length collapsed from 130 to 10
   steps. Penalties (mostly vertical velocity from the spawn drop and unbounded PD torques)
   outweighed the tracking reward, so dying early was optimal. Fix: clip the per-step reward sum
   at 0 before the termination penalty, spawn at standing height, `init_noise_std` 0.5,
   `clip_actions` 10. Next run: 1000-step episodes by iteration ~100.
2. *ONNX export didn't load in Warp-NN.* rsl_rl's export emits Sub/Div for observation
   normalization; Warp-NN supports only Gemm/Elu/LSTM/Squeeze. Fix: `export_deploy_onnx` folds
   the normalizer into the first layer and writes a Gemm/Elu graph by hand.
3. *Humanoid terminated at step 1.* The MJCF torso sits at z = 0 and motor gears were matched by
   joint label (only 5 of 21 matched). Fix: spawn offset 1.35 m; gears mapped by MJCF document order.
4. *Contact overflow.* `nconmax=30` per world overflowed when a Go2 folded onto the ground
   ("narrowphase overflow"); raised to 48 (G1: 80).

5. *G1 learned to "walk" while floating.* A rendered frame showed the robots drifting upward:
   the G1 USD carries zero gravity and `model.set_gravity()` ran after `SolverMuJoCo` was built
   (the solver copies gravity at construction). Metrics alone looked fine (19.8 s survival).
   Fix: set gravity in `NewtonVecEnv` right after `finalize()`. Lesson: look at a frame
   (`ViewerGL(headless=True).get_frame()`) before trusting a new robot's numbers.

**Validation trick worth keeping:** the legged env uses Isaac Lab's observation layout, so the
reference ONNX policies shipped in newton-assets run in it directly. If
`play.py --pretrained --eval` stops surviving ~20 s, the env is broken, not the trainer.
