# Train and run a robot on one edge computer — just by talking to it

*A plain-English talk about the newton-claw workflow on the NVIDIA Jetson Thor.*

## 1. The idea

Teaching a robot to walk usually takes two computers. A large workstation or a cloud server
does the training, and a second, smaller computer on the robot runs what was learned.

This project does it differently. One small computer, the NVIDIA Jetson Thor, does both jobs.
It trains the robot, and it runs the trained brain. And you control the whole thing in plain
English.

## 2. What you actually do

You open a chat window on the Thor and type a sentence, such as "train the Go2".

That is all. You don't write code, you don't edit configuration files, and you don't have to
remember commands.

An AI assistant running on the same Thor reads your sentence, works out what you mean, and
starts the job. You can ask "how is the training going?" and it tells you the progress and how
long is left. When it is done, you say "show me the robot walking", and you watch it.

## 3. What happens behind the sentence

Three things live on that one computer.

**The assistant.** It is a language model called Gemma, and it runs locally. Nothing is sent to
the cloud. Its only job is to turn your sentence into one clear action.

**The control service.** Think of it as the gatekeeper of the GPU. It starts the job, makes sure
only one heavy job runs at a time, and reports progress.

**The simulator and the trainer.** The simulator is Newton, a physics engine built for robotics.
Instead of one robot, it simulates four thousand copies of the robot at the same time, all on
the Thor's GPU. Each copy tries to walk, falls, and tries again. A learning algorithm called PPO
watches all four thousand and keeps what works. Because so many robots learn in parallel, hours
of practice are squeezed into minutes.

## 4. The agent already knows the settings, and it understands how people talk

Training a robot normally means choosing dozens of numbers: how fast to learn, how much to
explore, how many robots to simulate, how strongly to reward smooth movement. Getting those
wrong is where most people get stuck.

Here, you don't choose them. The settings that work for each robot are already built in. When
you say "train the Go2", the agent starts the run with values that have been tested on this
machine. You never see a configuration file.

You also don't need the right technical words. The agent understands ordinary language.
"Train the robot dog", "train the Go2" and "start teaching the quadruped to walk" all mean the
same thing to it. "How's it going?", "is it done yet?" and "show me the progress" all get you
the live status.

If you want to change how the robot behaves, you say that in plain words too. The agent carries
a tuning guide with safe ranges for every setting and a list of common problems with their
fixes. So instead of editing a number, you describe what you see: "the walk looks jerky, make it
smoother", or "it's bouncing too much". The agent looks up which setting controls that, picks a
value inside the safe range, and starts a new run with it.

The expert knowledge sits inside the system. You bring the goal.

## 5. How fast is it

On the Thor, a four-legged robot, the Unitree Go2, learned to walk and follow speed commands in
about eleven minutes. A humanoid, the Unitree G1, took about twenty-three minutes.

When the trained Go2 was tested on a thousand simulated robots for twenty seconds, only about
ten fell, and they followed the commanded speed almost exactly.

You can watch all of this live. A window opens on the Thor's screen, and you see the robots
stumbling at the start and walking smoothly by the end.

## 6. From training to the real robot

When training finishes, the system saves the robot's learned brain as a small file, called a
policy. It is a compact neural network. It reads the robot's sensors — body tilt, joint angles,
and the speed you asked for — and it outputs the joint movements.

Running that policy is very light. On the Thor it takes well under a millisecond per decision,
far faster than a robot needs.

This is the point of using one computer. The same Thor that trained the policy can sit on the
robot and run it. The next step for this project is exactly that: connect the Thor to the real
robot's motors and sensors, feed the real sensor readings into the same policy, and send its
outputs to the real joints. If the robot needs to improve, you retrain on the same box and load
the new policy. No cloud, no second machine, no moving files between systems.

## 7. Why this matters

- **One computer.** Training and running happen on the same edge device.
- **Plain English.** Anyone on the team can start a training run or a demo by typing a sentence.
- **No tuning expertise needed.** Proven settings are built in, and changes are made by
  describing the behaviour you want.
- **Private and offline.** The assistant, the simulator and the data all stay on the device.
- **Fast iteration.** Minutes from "train" to "watch it walk", so you can try ideas many times a day.

## 8. Closing

The workflow is simple. Say what you want. The Thor trains the robot in simulation, shows you
the result, and hands you a policy ready to run on the robot — all on a single computer at the
edge.

---

### Suggested live demo (about five minutes)

| Say | What the audience sees |
|---|---|
| "what can I train?" | the list of robots |
| "train the Go2 for 200 iterations" | a window opens; robots stumble, then start walking |
| "how is the training going?" | iteration, reward, time left |
| "evaluate the Go2 policy" | falls and tracking error from a thousand robots |
| "show me the Go2 walking" | the freshly trained policy, walking |
| "stop it" | GPU free |

### Presenter notes (not for the audience)

- **Real robot:** the policy has been trained, evaluated and run in simulation on the Thor and
  exported as a deployable file. It has not yet been run on a physical robot; section 6 presents
  that as the next step.
- **Plain-language tuning:** starting training with built-in settings from ordinary sentences is
  tested. The tuning guide and the mechanism for changing settings exist, but a request such as
  "make it smoother" has not been tested end to end through the agent. Test it before demoing it.
- **Repeated requests:** in a long chat the assistant once claimed it had started a training it
  had not. Confirm with "how is the training going?", or start a fresh chat before the demo.
- **Numbers:** Go2 — 600 iterations in 10 min 43 s, 10 falls in 1,024 robots over 20 s, 0.04 m/s
  tracking error. G1 — 800 iterations in about 23 min, 1 fall in 1,024 robots.
