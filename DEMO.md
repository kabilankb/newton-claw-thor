# Running the demos from the OpenClaw TUI

Everything below is done by **typing sentences** into the OpenClaw TUI inside the NemoClaw
sandbox on the Jetson Thor. The agent (Gemma 4 12B) turns each sentence into one `claw`
command; training and inference run on the Thor's GPU in Newton physics.

## 0. Start (once per boot)

On the Thor (`ssh nvidia-thor@192.168.1.29`):

```bash
cd ~/newton-claw
agents/llm_up.sh            # Gemma model server            → "model server: up (:8000)"
agents/openclaw/claw up     # Newton control server         → "control server up."
nemoclaw newton connect     # enter the sandbox
openclaw tui                # the TUI opens; type at the prompt
```

If `nemoclaw newton connect` cannot reach the gateway after a reboot, run
`openshell gateway start --name nemoclaw` first, then retry.

**Where to watch:** by default every demo opens a Newton Viewer window on the Thor's own
monitor. To watch from your laptop instead, add "in the browser" to the sentence and open the
link it gives (`http://192.168.1.29:8090`). Add "headless" to a training request for no window.

**One rule:** only one GPU job runs at a time. Say **"stop it"** before starting the next demo.

---

## Demo 1 — What can it do? (10 seconds)

| Type | You should see |
|---|---|
| `what can I train?` | six tasks: Cartpole, Ant, Humanoid, Unitree Go2, ANYmal C, Unitree G1 |
| `what have I trained?` | the policies already saved on this Thor |
| `how much memory and disk is free?` | Thor model, free memory, free disk |

## Demo 2 — Train a robot in under a minute

| Type | You should see |
|---|---|
| `run a quick cartpole training test` | "Training started for Newton-Cartpole-v0" |
| `how is the training going?` (after ~20 s) | iteration, mean reward, ETA |
| `is the training done?` (after ~40 s) | finished, mean episode length ≈ 295–300 |

## Demo 3 — Watch a walking robot dog (no training needed)

| Type | You should see |
|---|---|
| `show me the Go2 walking` | a window on the Thor's monitor with 16 Go2 robots walking |
| `stop it` | "closed — GPU free" |

Variations:
- `show me the Go2 walking forward at 1 m/s in the browser` — gives a link instead of the window.
- `run the pretrained Go2` — uses the reference policy that ships with the robot assets
  instead of the one trained on this Thor.

## Demo 4 — Train the Go2 from scratch, then watch your own policy (~4 minutes)

| Type | You should see |
|---|---|
| `train the Go2 for 200 iterations` | training started (4,096 robots); a window shows 36 of them learning |
| `how is the training going?` | reward climbing; by iteration ~100 the episode length reaches ~1000 |
| `is the training done?` (after ~3.5 min) | finished |
| `evaluate the Go2 policy` | evaluation started |
| `what were the evaluation results?` (after ~1 min) | falls ≈ 10 of 1,024 robots, tracking error ≈ 0.04 m/s |
| `show me the Go2 walking` | the policy you just trained, in the viewer |
| `stop it` | GPU free |

A full-quality run is `train the Go2` with no iteration count (1,000 iterations, ~18 minutes).

## Demo 5 — Humanoid

| Type | You should see |
|---|---|
| `show me the G1 walking` | Unitree G1 humanoids walking (policy trained on this Thor) |
| `stop it` | GPU free |
| `train the G1 for 800 iterations` | ~23 minutes; afterwards `evaluate the G1 policy` gave 1 fall in 1,024 robots |

## Demo 6 — Just look at a robot

| Type | You should see |
|---|---|
| `what robots can I open?` | unitree_g1, unitree_h1, unitree_go2, anymal_c, anymal_d, franka_panda, ur10, allegro_hand, cartpole, ant_humanoid |
| `open the Franka` | the Franka arm in the viewer |
| `stop it` | GPU free |

---

## If something goes wrong

| What you see | What to do |
|---|---|
| "a GPU job is already running" | type `stop it`, then repeat the request |
| the agent explains instead of doing | start a fresh TUI session (quit and run `openclaw tui` again) and repeat |
| "control server … unreachable" / connection refused | on the Thor host (not in the sandbox): `cd ~/newton-claw && agents/openclaw/claw up` |
| the agent does not answer at all | on the host: `agents/llm_up.sh`, then retry |
| the viewer link does not load | wait ~20 s after the job starts; the page only exists while a `play` / `open` job runs |

Check anything by hand from the Thor host, outside the TUI:

```bash
cd ~/newton-claw
agents/openclaw/claw status     # what is running
agents/openclaw/claw logs       # last lines of the job log
agents/openclaw/claw close      # stop it
```

## What has been tested, and what has not

Tested by sending these sentences to the sandbox agent (`openclaw agent`, the same agent the
TUI talks to) or to the terminal agent (`agents/chat.py`): demos 1, 2, 4 and the Go2 part of
demo 3. The "on the monitor" window was tested with the `claw play --gui` command directly, not
through a typed sentence. For demo 5 the G1 policy was trained and evaluated on this Thor, and for demo 6 `claw open`
was run for the G1; the sentences for both have not been tried in the TUI yet. The interactive TUI screen itself has not been driven end to end.

## No sandbox? Same demos in a plain terminal

```bash
cd ~/newton-claw && agents/llm_up.sh && agents/chat.py
```
Type the same sentences at the `you>` prompt.
