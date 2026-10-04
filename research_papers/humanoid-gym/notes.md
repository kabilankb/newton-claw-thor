# Humanoid locomotion (paper 2404.05695v2)

`paper.pdf` — the reference behind the locomotion training stack in
`isaac_lab/source/robot_lab` and the tuning guidance in
`skills/training/references/` and `policies/rsl_rl.md`.

## Why it's here
The reward terms, domain-randomization strategy, asymmetric actor-critic, and
sim-to-sim (MuJoCo) → sim-to-real transfer recipe used by the RobotEra XBot /
Unitree G1 velocity tasks trace back to this work. When tuning a falling or
shuffling gait, cross-reference:
- `skills/training/references/architecture-deep-dive.md` (full spec)
- `skills/training/references/locomotion-tuning.md` (symptom → fix)
- `policies/rsl_rl.md` (the canonical knob block)

## Add more papers
One folder per paper: `research_papers/<slug>/{paper.pdf, notes.md}`.
