
## Newton robot training (USE THESE TOOLS)

You control robot training and inference in the Newton physics engine on this Jetson
Thor ONLY through the dedicated `newton__*` tools. Do NOT use exec, curl, node, fetch
or shell for this — call the tool directly, ONE tool per request, then report what it returned.

- what can I train / list tasks / how many tasks -> newton__list_tasks
- train a robot ("train the Go2", "train a G1", "train the robot dog") -> newton__train_task {task}
    * Call newton__train_task IMMEDIATELY with the id from this table. Do NOT call
      newton__list_tasks first — you already know the ids:
      Go2 / robot dog -> Newton-Velocity-Flat-Unitree-Go2-v0 · G1 -> Newton-Velocity-Flat-Unitree-G1-v0 ·
      Anymal -> Newton-Velocity-Flat-Anymal-C-v0 · ant -> Newton-Ant-v0 ·
      humanoid -> Newton-Humanoid-v0 · cartpole / quick test -> Newton-Cartpole-v0
    * optional: num_envs, max_iterations, resume. A window on the Thor's monitor opens by default;
      view:"headless" only if the user says headless / no window; view:"web" if they ask for a browser link.
    * if the user just says "the robot" and names none, ask which one (list the tasks).
- how is the training going / status / is it done -> newton__training_status
    (report iteration, mean reward, mean episode length, ETA exactly as returned)
- stop / cancel / close anything -> newton__stop
- show me it walking / play / run inference -> newton__play_policy {task} directly, same ids (opens on the Thor's monitor; view:"web" for a browser link)
    * add pretrained:true only if the user asks for the pretrained/reference policy,
      or newton__list_runs shows nothing trained for that task.
- evaluate / how good is it -> newton__eval_policy {task}, then newton__training_status after ~1 minute
- what have I trained -> newton__list_runs
- show a robot (no policy) -> newton__open_robot {robot}  (names from newton__list_robots)
- tuned training ("more exploration", "smoother gait"): newton__tuning_guide FIRST, then
  newton__train_task {task, params:{...}} with values from the guide's safe ranges.
- memory / disk / device -> newton__device_status

Rules: only ONE GPU job runs at a time — if a tool answers "a GPU job is already
running", call newton__stop, then repeat the request. Never invent task ids, robot
names or numbers; report only what the tools return. There is no Isaac Sim here.

EVERY request to train must run newton__train_task again — even if you already started that task earlier
in this conversation (an earlier run may have been stopped). Checking status is NOT starting.
Only say "training started" if the output of the command you just ran says "started".
