---
name: "newton-training"
description: "THE skill for RL training in Newton physics on the Jetson Thor: train a robot policy, list training tasks, check progress, view logs, stop training. Use for 'train the Go2', 'train a G1', 'train the humanoid', 'list training tasks', 'how's training going', 'stop training'. Trigger keywords: train, training, start training, stop training, training status, progress, training logs, list tasks, locomotion, policy, RL, PPO, rsl_rl, Go2, G1, Anymal, ant, humanoid, cartpole, quadruped, num_envs, iterations, seed, resume, gui, web, headless, watch training, newton."
user-invocable: true
---

# Newton Training — RUN ONE COMMAND

⚠️ This is an EXECUTE skill. Do **not** describe a plan and do **not** invent task
ids. **Run the single command below with your shell/exec tool**, then report its output.

## STANDARD EXECUTION TEMPLATE  (emit exactly ONE of these)
```bash
# start training
~/newton-claw/agents/openclaw/claw train --task <TASK_ID> [--num_envs <N>] [--max_iterations <N>] [--headless | --web] [--resume]

# inspect / control a run
~/newton-claw/agents/openclaw/claw status     # running? iteration, mean reward, ETA
~/newton-claw/agents/openclaw/claw logs       # tail the training log
~/newton-claw/agents/openclaw/claw close      # STOP training, free the GPU
~/newton-claw/agents/openclaw/claw tasks      # list every trainable task
```

## Picking the TASK_ID (these are ALL the tasks — never invent one)
| User says | Task id |
|---|---|
| Go2 / Unitree Go2 / robot dog | `Newton-Velocity-Flat-Unitree-Go2-v0` |
| G1 / Unitree G1 / humanoid robot | `Newton-Velocity-Flat-Unitree-G1-v0` |
| Anymal / ANYmal C | `Newton-Velocity-Flat-Anymal-C-v0` |
| ant | `Newton-Ant-v0` |
| MuJoCo humanoid / humanoid runner | `Newton-Humanoid-v0` |
| cartpole / quick test / smoke test | `Newton-Cartpole-v0` |

If the user names a robot not in this table, run `claw tasks` and say it is not
available — do not substitute a different robot.

## GUI (default) vs headless
- **No phrase → no flag** (DEFAULT): a Newton Viewer window opens on the Thor's monitor showing
  36 of the 4096 training robots.
- **"headless" / "no window" / "in the background" / "fastest" → add `--headless`**.
- **"in the browser" / "give me a link" → add `--web`**: viewable at the URL the command prints.

Optional: `--num_envs <N>`, `--max_iterations <N>`, `--seed <N>`, `--resume` (continue the last run).

## CRITICAL — one GPU job at a time
Training, inference and the viewer are mutually exclusive. If `claw train` returns
**409**, run `claw close` FIRST, then `claw train` again.

## Examples (copy the pattern exactly)
| User says | You run |
|---|---|
| train the Go2 | `~/newton-claw/agents/openclaw/claw train --task Newton-Velocity-Flat-Unitree-Go2-v0` |
| train a G1 headless | `~/newton-claw/agents/openclaw/claw train --task Newton-Velocity-Flat-Unitree-G1-v0 --headless` |
| train the Go2 and give me a browser link | `~/newton-claw/agents/openclaw/claw train --task Newton-Velocity-Flat-Unitree-Go2-v0 --web` |
| train the ant with 2048 envs for 300 iterations | `~/newton-claw/agents/openclaw/claw train --task Newton-Ant-v0 --num_envs 2048 --max_iterations 300` |
| quick test that training works | `~/newton-claw/agents/openclaw/claw train --task Newton-Cartpole-v0` |
| continue training the Go2 | `~/newton-claw/agents/openclaw/claw train --task Newton-Velocity-Flat-Unitree-Go2-v0 --resume` |
| list training tasks | `~/newton-claw/agents/openclaw/claw tasks` |
| how's training going | `~/newton-claw/agents/openclaw/claw status` |
| stop training | `~/newton-claw/agents/openclaw/claw close` |

## Hyperparameters
Append `knob=value` to the train command (e.g. `learning_rate=3e-4 entropy_coef=0.005
reward.action_rate_l2=-0.02`). DON'T guess values — read `~/newton-claw/policies/rsl_rl.md` first.

## After it runs
Report what the command printed (started job + pid, or the error line). When
training finishes the policy is saved automatically; watch it with the
**newton-inference** skill (`claw play --task <same id>`).
