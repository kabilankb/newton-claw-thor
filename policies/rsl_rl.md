---
name: rsl-rl-tuning
description: >-
  Tune rsl_rl PPO hyperparameters and reward weights for Newton tasks on Jetson
  Thor. Trigger keywords: rsl_rl, PPO, learning rate, entropy_coef, gamma, lambda,
  clip_param, num_mini_batches, num_learning_epochs, num_steps_per_env, desired_kl,
  init_noise_std, reward weight, action_rate, falls over, jerky gait, not learning.
user-invocable: true
---

# rsl_rl PPO tuning (Newton on Thor)

Defaults live in `newton_lab/newton_lab/common.py` (`ppo_cfg`) plus per-task
overrides in `newton_lab/newton_lab/envs/__init__.py`. Override any knob from
the command line: `claw train --task <id> knob=value ...`.

## Canonical knob block
```yaml
# ── rollout / scale ─────────────────────────────────────────────────
num_envs: 4096            # 256–8192 (flag: --num_envs). Halve if memory is tight.
max_iterations: 1000      # task default (flag: --max_iterations). cartpole 150, ant 500, go2/anymal 1000, g1 3000
num_steps_per_env: 24     # 16–48. Rollout horizon per iteration.
# ── PPO core ────────────────────────────────────────────────────────
learning_rate: 1.0e-3     # 1e-5–3e-3. Adaptive (KL-driven) by default.
schedule: adaptive        # adaptive | fixed
desired_kl: 0.01          # 0.005–0.02
gamma: 0.99               # 0.97–0.999
lam: 0.95                 # 0.9–0.97 (alias: lambda)
clip_param: 0.2           # 0.1–0.3
entropy_coef: 0.01        # 0.0–0.02. Raise if action std collapses early.
value_loss_coef: 1.0      # 0.5–2.0
num_learning_epochs: 5    # 2–8
num_mini_batches: 4       # 2–8
max_grad_norm: 1.0        # 0.5–2.0
init_noise_std: 1.0       # 0.3–1.0. Legged tasks default to 0.5 (PD targets are unbounded).
# ── network (edit in code: hidden_dims per task) ────────────────────
# actor/critic: ELU MLP [512, 256, 128], observation normalization on
```

## Reward weights (legged velocity tasks: Go2, ANYmal C, G1)
Set with `reward.<term>=<weight>`. Values are multiplied by the 0.02 s policy step.
```yaml
reward.track_lin_vel_xy_exp: 1.5     # follow commanded x/y velocity (G1: 2.0)
reward.track_ang_vel_z_exp: 0.75     # follow commanded yaw rate (G1: 1.0)
reward.lin_vel_z_l2: -2.0            # no bouncing
reward.ang_vel_xy_l2: -0.05          # no roll/pitch rate
reward.flat_orientation_l2: -2.5     # keep the base level (G1: -1.0)
reward.dof_torques_l2: -2.0e-4       # energy (G1: -2e-6, ANYmal: -2.5e-5)
reward.dof_acc_l2: -2.5e-7           # smooth joints
reward.action_rate_l2: -0.01         # smooth actions
reward.joint_deviation_l1: 0.0       # stay near the default pose (G1: -0.1)
reward.alive: 0.0                    # per-second survival bonus (G1: 1.0)
reward.termination: -200.0           # falling
```
Other env fields: `env.episode_length_s`, `env.joint_pos_noise`, `env.tracking_sigma`,
`env.min_height`. The per-step reward sum is clipped at 0 before the termination
penalty (`only_positive_rewards`) so "fall early to stop paying penalties" is never optimal.

## Symptom → fix
| Symptom | Try |
|---|---|
| Mean episode length collapses to a few steps | lower `init_noise_std` (0.3–0.5); make sure `only_positive_rewards` is on |
| Reward flat, action std stays ~initial | raise `learning_rate` to 1e-3, check `desired_kl=0.01` |
| Action std collapses early, stands still | raise `entropy_coef` (0.01 → 0.02); raise `reward.track_lin_vel_xy_exp` |
| Jerky / vibrating gait | `reward.action_rate_l2=-0.02`…`-0.05`, `reward.dof_acc_l2=-1e-6` |
| Bounces / hops | `reward.lin_vel_z_l2=-4.0` |
| Leans or walks tilted | `reward.flat_orientation_l2=-5.0` |
| "narrowphase overflow … increase nconmax" in the log | raise `nconmax`/`njmax` in the task's `solver` cfg (code) |
| Out of memory / Thor swapping | `--num_envs 2048` (or 1024); close the viewer; `claw device` |
