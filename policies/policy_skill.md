---
name: policy-tuning
description: >-
  Master router for tuning RL training in newton-claw. Trigger keywords: tune,
  hyperparameter, learning rate, entropy, gamma, reward weight, not learning,
  falls over, jerky, slow training, out of memory.
user-invocable: true
---

# Policy tuning router

There is ONE trainer on this system: **rsl_rl PPO** on Newton physics → read
[`rsl_rl.md`](rsl_rl.md). It has the knob block (safe ranges), the reward terms per
task family, and a symptom → fix table.

How knobs are applied — append `knob=value` pairs to the train command:
```bash
~/newton-claw/agents/openclaw/claw train --task Newton-Velocity-Flat-Unitree-Go2-v0 learning_rate=5e-4 reward.action_rate_l2=-0.02
```
Unknown knob names are reported in the training log and ignored (never silently applied).
