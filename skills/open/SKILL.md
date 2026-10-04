---
name: "newton-open-robot"
description: "THE skill for showing a robot in the Newton physics viewer on the Jetson Thor. Use for 'open the G1', 'show me the Franka', 'load the Go2', 'bring up the sim', 'what robots are there'. Trigger keywords: open, show, load, view, spawn, bring up the sim, viewer, robot, g1, h1, go2, anymal, franka, panda, ur10, allegro, hand, cartpole, list robots."
user-invocable: true
---

# Open a robot in Newton — RUN ONE COMMAND

⚠️ EXECUTE skill. **Your first action MUST be a shell tool call** with the command
below. Do not write a plan. Do not invent robot names.

## STANDARD EXECUTION TEMPLATE
```bash
~/newton-claw/agents/openclaw/claw open --robot <ROBOT> [--web]
~/newton-claw/agents/openclaw/claw robots        # list valid robot names
```
- Default opens a Newton Viewer window on the Thor's own monitor.
- Add `--web` only if the user asks for a browser / a link; the command prints the URL.

## ROBOT — the ONLY valid names
| User says | `<ROBOT>` |
|---|---|
| G1 / Unitree G1 | `unitree_g1` |
| H1 / Unitree H1 | `unitree_h1` |
| Go2 / robot dog | `unitree_go2` |
| Anymal / ANYmal C | `anymal_c` |
| ANYmal D | `anymal_d` |
| Franka / Panda | `franka_panda` |
| UR10 | `ur10` |
| Allegro hand | `allegro_hand` |
| cartpole | `cartpole` |
| ant / MuJoCo humanoid | `ant_humanoid` |

If the user names anything else, run `claw robots` and say it is not available.

## Examples
| User says | You run |
|---|---|
| open the G1 | `~/newton-claw/agents/openclaw/claw open --robot unitree_g1` |
| show me the Franka in the browser | `~/newton-claw/agents/openclaw/claw open --robot franka_panda --web` |
| what robots can I open | `~/newton-claw/agents/openclaw/claw robots` |

## After it runs
Report that a window opened on the Thor's monitor (or the printed URL if `--web`), or the exact error line. A **409**
means training/inference owns the GPU — run `~/newton-claw/agents/openclaw/claw close` and retry.
This only DISPLAYS a robot. To train use **newton-training**; to run a policy use **newton-inference**.
