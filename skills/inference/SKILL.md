---
name: "newton-inference"
description: "THE skill for running a trained policy (INFERENCE) in Newton on the Jetson Thor: play / watch / test / evaluate / benchmark a robot policy. Use for 'play the Go2 policy', 'show me the G1 walking', 'run inference', 'evaluate the policy', 'how fast is inference', 'run the pretrained Go2'. Trigger keywords: play, watch, run the policy, inference, deploy, evaluate, eval, benchmark, test the policy, pretrained, onnx, checkpoint, walk, show me walking, demo."
user-invocable: true
---

# Newton Inference — RUN ONE COMMAND

⚠️ EXECUTE skill. **Run the single command below with your shell/exec tool**, then
report its output. Never invent task ids or file paths.

## STANDARD EXECUTION TEMPLATE  (emit exactly ONE of these)
```bash
# WATCH a policy (window on the Thor's monitor by default)
~/newton-claw/agents/openclaw/claw play --task <TASK_ID> [--pretrained] [--web] [--command <VX> <VY> <YAW>]

# MEASURE a policy (headless numbers: reward, survival, tracking error, speed)
~/newton-claw/agents/openclaw/claw eval --task <TASK_ID> [--pretrained]
~/newton-claw/agents/openclaw/claw status      # read the metrics when the eval finishes
```

## Slots
| Slot | Rule |
|---|---|
| `<TASK_ID>` | same ids as training: `Newton-Velocity-Flat-Unitree-Go2-v0`, `Newton-Velocity-Flat-Unitree-G1-v0`, `Newton-Velocity-Flat-Anymal-C-v0`, `Newton-Ant-v0`, `Newton-Humanoid-v0`, `Newton-Cartpole-v0` |
| policy | **default = the latest policy trained on this Thor**. Add `--pretrained` ONLY if the user says "pretrained" / "reference", or nothing has been trained yet (Go2, G1, Anymal only). |
| `--web` | add only if the user asks for a browser / a link; otherwise a window opens on the Thor's monitor |
| `--command` | optional walking command: forward m/s, sideways m/s, turn rad/s — e.g. `--command 1.0 0 0` = walk forward at 1 m/s |

What has been trained? `~/newton-claw/agents/openclaw/claw runs`

## CRITICAL — one GPU job at a time
If the command returns **409**, run `~/newton-claw/agents/openclaw/claw close` first, then retry.

## Examples (copy the pattern exactly)
| User says | You run |
|---|---|
| play the Go2 policy / show me the Go2 walking | `~/newton-claw/agents/openclaw/claw play --task Newton-Velocity-Flat-Unitree-Go2-v0` |
| run the pretrained Go2 | `~/newton-claw/agents/openclaw/claw play --task Newton-Velocity-Flat-Unitree-Go2-v0 --pretrained` |
| make the G1 walk forward | `~/newton-claw/agents/openclaw/claw play --task Newton-Velocity-Flat-Unitree-G1-v0 --command 0.8 0 0` |
| show the Go2 in the browser | `~/newton-claw/agents/openclaw/claw play --task Newton-Velocity-Flat-Unitree-Go2-v0 --web` |
| evaluate the Go2 policy / how good is it | `~/newton-claw/agents/openclaw/claw eval --task Newton-Velocity-Flat-Unitree-Go2-v0` |
| what policies have I trained | `~/newton-claw/agents/openclaw/claw runs` |
| stop it | `~/newton-claw/agents/openclaw/claw close` |

## After it runs
For `play`: tell the user a window opened on the Thor's monitor (or give the URL if `--web`).
For `eval`: wait ~1 minute, run `claw status`, and report `progress.result`
(mean episode length vs max, tracking error, env steps per second).
