#!/usr/bin/env python3
"""
Newton Control Server — the single host-side control surface for newton-claw on Jetson Thor.

OpenClaw (Gemma) and NemoClaw (Nemotron) drive Newton physics — training AND
inference — through THIS one HTTP API. It is the Newton port of the old
isaac_control_server.py: same port, same one-GPU-job rule, same endpoint shapes.

    GET  /envs            trainable Newton tasks (newton_lab/tasks_registry.json)
    GET  /robots          robots/scenes that can be opened in the Newton viewer
    GET  /runs            trained runs (latest checkpoint + exported policy per task)
    GET  /status          current job + parsed training progress
    GET  /logs?lines=N    tail the active job log
    GET  /tuning          hyperparameter guide (policies/rsl_rl.md)
    GET  /device          Jetson model / memory / disk / GPU snapshot

    POST /train   {task, num_envs?, max_iterations?, seed?, resume?, gui?, viewer?, params?{}}
    POST /play    {task, pretrained?, checkpoint?, onnx?, num_envs?, viewer?, command?[vx,vy,yaw]}
    POST /eval    {task, pretrained?, checkpoint?, onnx?, num_envs?, steps?}   (headless, prints metrics)
    POST /open    {robot, viewer?, world_count?}     open a robot in the Newton viewer
    POST /stop    {}                                 stop the active GPU job

Rules:
  * ONE GPU job at a time (sim viewer OR train OR play/eval). A second returns 409.
  * viewer: "gl" = window on the Thor's display (the DEFAULT for train / play / open),
    "viser" = browser at http://<thor>:8090, "none" = headless (fastest training).
    Change the default with NEWTON_DEFAULT_VIEWER.
  * Stdlib only, so it runs under any python; jobs run under $NEWTON_PYTHON (the venv).
"""

import json
import os
import re
import shutil
import signal
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve()
CLAW_DIR = Path(os.environ.get("NEWTON_CLAW_DIR", HERE.parents[2]))
LAB_DIR = CLAW_DIR / "newton_lab"
SIM_DIR = CLAW_DIR / "newton_sim"
PYTHON = os.environ.get("NEWTON_PYTHON", str(CLAW_DIR / ".venv" / "bin" / "python"))
HOST = os.environ.get("NEWTON_CONTROL_HOST", "0.0.0.0")
PORT = int(os.environ.get("NEWTON_CONTROL_PORT", "5561"))
VISER_PORT = int(os.environ.get("NEWTON_VISER_PORT", "8090"))
# Where jobs are shown unless a request says otherwise: gl = window on the Thor's monitor,
# viser = browser, none = headless.
DEFAULT_VIEWER = os.environ.get("NEWTON_DEFAULT_VIEWER", "gl")
LOG_DIR = SIM_DIR / "_control_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR = Path(os.environ.get("NEWTON_LAB_LOGS", LAB_DIR / "logs" / "rsl_rl"))
POLICIES_DIR = CLAW_DIR / "policies"


def thor_ip():
    """LAN address of this machine, for viewer links (falls back to localhost)."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "localhost"


def load_json(path, key):
    try:
        return json.loads(Path(path).read_text()).get(key, [])
    except Exception:
        return []


def tasks():
    return load_json(LAB_DIR / "tasks_registry.json", "environments")


def robots():
    return load_json(SIM_DIR / "robots.json", "robots")


def list_runs():
    out = []
    for task_dir in sorted(RUNS_DIR.glob("*")):
        runs = sorted(d for d in task_dir.glob("*") if d.is_dir())
        for run in reversed(runs):
            ckpts = sorted(run.glob("model_*.pt"), key=lambda p: int(re.search(r"model_(\d+)", p.name).group(1)))
            if ckpts:
                out.append({"task": task_dir.name, "run": run.name, "checkpoint": str(ckpts[-1]),
                            "onnx": str(run / "policy.onnx") if (run / "policy.onnx").exists() else None,
                            "runs_total": len(runs)})
                break
    return out


def device_info():
    info = {}
    try:
        info["model"] = Path("/proc/device-tree/model").read_text().strip("\x00\n")
    except Exception:
        info["model"] = "unknown"
    try:
        mem = dict(l.split(":") for l in Path("/proc/meminfo").read_text().splitlines()[:3])
        info["mem_total_gb"] = round(int(mem["MemTotal"].split()[0]) / 1048576, 1)
        info["mem_available_gb"] = round(int(mem["MemAvailable"].split()[0]) / 1048576, 1)
    except Exception:
        pass
    du = shutil.disk_usage(str(CLAW_DIR))
    info["disk_free_gb"] = round(du.free / 1e9, 1)
    info["disk_total_gb"] = round(du.total / 1e9, 1)
    try:
        info["l4t"] = Path("/etc/nv_tegra_release").read_text().splitlines()[0].strip("# ")
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------- #
# Single-job state machine (one GPU job at a time)
# --------------------------------------------------------------------------- #
job = {"process": None, "pid": None, "kind": None, "label": None, "start_time": None,
       "log_file": None, "viewer": None, "returncode": None}


def job_running():
    p = job["process"]
    if p is None:
        return False
    rc = p.poll()
    if rc is not None:
        job["process"], job["returncode"] = None, rc
        return False
    return True


def tail_log(n=40):
    lf = job.get("log_file")
    if not lf or not os.path.exists(lf):
        return ""
    with open(lf, errors="replace") as f:
        # drop USD-import and Warp kernel-load noise so the tail shows the job's own output
        lines = [l for l in f.readlines() if "MaterialBindingAPI" not in l and not l.startswith("Module ")]
    return "".join(lines[-n:])


def training_progress():
    """Parse the rsl_rl log tail -> {iteration, total, mean_reward, ...} (best-effort)."""
    text = tail_log(400)
    out = {}
    it = re.findall(r"Learning iteration (\d+)/(\d+)", text)
    if it:
        out["iteration"], out["total_iterations"] = int(it[-1][0]), int(it[-1][1])
    for key, pat in (("mean_reward", r"Mean reward:\s*(-?[\d.]+)"),
                     ("mean_episode_length", r"Mean episode length:\s*([\d.]+)"),
                     ("steps_per_second", r"Steps per second:\s*(\d+)"),
                     ("eta", r"ETA:\s*([\d:]+)")):
        m = re.findall(pat, text)
        if m:
            out[key] = m[-1] if key == "eta" else float(m[-1])
    done = re.findall(r"\[train\] DONE (\S+)", text)
    if done:
        out["finished_run_dir"] = done[-1]
    result = re.findall(r"\[play\] RESULT (\{.*\})", text)
    if result:
        try:
            out["result"] = json.loads(result[-1])
        except Exception:
            pass
    return out


def display_env():
    """Find the Thor's desktop display so `viewer: gl` works even when started from SSH/systemd."""
    env = {}
    if not os.environ.get("DISPLAY"):
        socks = sorted(Path("/tmp/.X11-unix").glob("X*")) if Path("/tmp/.X11-unix").exists() else []
        if socks:
            env["DISPLAY"] = ":" + socks[0].name[1:]
    if not os.environ.get("XAUTHORITY"):
        for cand in (f"/run/user/{os.getuid()}/gdm/Xauthority", os.path.expanduser("~/.Xauthority")):
            if os.path.exists(cand):
                env["XAUTHORITY"] = cand
                break
    return env


def start_job(kind, label, cmd, viewer=None):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
    log_file = LOG_DIR / f"{kind}_{safe}_{int(time.time())}.log"
    env = os.environ.copy()
    env.update(display_env())
    env["NEWTON_CLAW_DIR"] = str(CLAW_DIR)
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=str(CLAW_DIR), env=env, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)
    job.update(process=proc, pid=proc.pid, kind=kind, label=label, start_time=time.time(),
               log_file=str(log_file), viewer=viewer, returncode=None)
    info = {"status": "started", "kind": kind, "label": label, "pid": proc.pid, "log_file": str(log_file),
            "command": " ".join(cmd)}
    if viewer == "viser":
        info["view_url"] = f"http://{thor_ip()}:{VISER_PORT}"
    return info


def stop_job():
    p = job["process"]
    if p is None or p.poll() is not None:
        job["process"] = None
        return False
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    job["process"] = None
    return True


def resolve_viewer(body, default):
    v = body.get("viewer")
    if not v:
        v = "gl" if body.get("gui") else default
    return v if v in ("gl", "viser", "usd", "none") else default


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {fmt % args}", flush=True)

    def _respond(self, code, data):
        body = json.dumps(data, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    # ---- GET ----
    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        path = url.path
        if path == "/envs":
            t = tasks()
            self._respond(200, {"environments": t, "count": len(t)})
        elif path == "/robots":
            self._respond(200, {"robots": robots()})
        elif path == "/runs":
            self._respond(200, {"runs": list_runs()})
        elif path == "/device":
            self._respond(200, device_info())
        elif path == "/status":
            running = job_running()
            info = {"running": running, "kind": job["kind"], "label": job["label"], "pid": job["pid"]}
            if job["start_time"] and running:
                el = time.time() - job["start_time"]
                info["elapsed_human"] = f"{int(el // 3600)}h {int((el % 3600) // 60)}m {int(el % 60)}s"
            if job["label"] and not running:
                info["status"] = "finished" if job["returncode"] in (0, None) else f"exited with code {job['returncode']}"
            if job["viewer"] == "viser" and running:
                info["view_url"] = f"http://{thor_ip()}:{VISER_PORT}"
            if job["kind"] in ("train", "play", "eval"):
                info["progress"] = training_progress()
            self._respond(200, info)
        elif path == "/logs":
            self._respond(200, {"lines": tail_log(int(q.get("lines", 40))), "log_file": job.get("log_file")})
        elif path == "/tuning":
            f = POLICIES_DIR / "rsl_rl.md"
            self._respond(200 if f.exists() else 404,
                          {"backend": "rsl_rl", "guide": f.read_text() if f.exists() else "missing"})
        else:
            self._respond(404, {"error": f"unknown GET {path}"})

    # ---- POST ----
    def do_POST(self):
        try:
            body = self._body()
        except Exception as e:
            return self._respond(400, {"error": f"bad json: {e}"})
        if self.path == "/train":
            return self._train(body)
        if self.path in ("/play", "/eval"):
            return self._play(self.path.lstrip("/"), body)
        if self.path in ("/open", "/sim/launch"):
            return self._open(body)
        if self.path == "/stop":
            if not job_running():
                return self._respond(200, {"status": "no job running"})
            out = {"status": "stopped", "kind": job["kind"], "label": job["label"], "pid": job["pid"]}
            stop_job()
            return self._respond(200, out)
        self._respond(404, {"error": f"unknown POST {self.path}"})

    def _guard_gpu(self):
        if job_running():
            self._respond(409, {"error": "a GPU job is already running — stop it first (POST /stop)",
                                "kind": job["kind"], "label": job["label"], "pid": job["pid"]})
            return False
        return True

    def _valid_task(self, body):
        task = body.get("task")
        ids = {e["id"] for e in tasks()}
        if not task:
            self._respond(400, {"error": "missing 'task'", "hint": "GET /envs"})
            return None
        if task not in ids:
            self._respond(400, {"error": f"unknown task {task}", "valid": sorted(ids)})
            return None
        return task

    def _train(self, body):
        if not self._guard_gpu():
            return
        task = self._valid_task(body)
        if not task:
            return
        viewer = resolve_viewer(body, DEFAULT_VIEWER)
        num_envs = body.get("num_envs") or 4096      # the viewer draws a subset, so GUI training keeps all envs
        cmd = [PYTHON, str(LAB_DIR / "scripts" / "train.py"), "--task", task, "--num_envs", str(num_envs),
               "--viewer", viewer]
        if body.get("max_iterations"):
            cmd += ["--max_iterations", str(body["max_iterations"])]
        if body.get("seed") is not None:
            cmd += ["--seed", str(body["seed"])]
        if body.get("resume"):
            cmd.append("--resume")
        if body.get("checkpoint"):
            cmd += ["--checkpoint", str(body["checkpoint"])]
        knobs = [f"{k}={v}" for k, v in (body.get("params") or {}).items()] + [str(o) for o in body.get("overrides") or []]
        info = start_job("train", task, cmd + knobs, viewer)
        info["applied_overrides"] = knobs
        self._respond(200, info)

    def _play(self, kind, body):
        if not self._guard_gpu():
            return
        task = self._valid_task(body)
        if not task:
            return
        viewer = "none" if kind == "eval" else resolve_viewer(body, DEFAULT_VIEWER)
        cmd = [PYTHON, str(LAB_DIR / "scripts" / "play.py"), "--task", task, "--viewer", viewer,
               "--viser_port", str(VISER_PORT)]
        if kind == "eval":
            cmd.append("--eval")
        if body.get("pretrained"):
            cmd.append("--pretrained")
        for key in ("checkpoint", "onnx", "num_envs", "steps"):
            if body.get(key):
                cmd += [f"--{key}", str(body[key])]
        if body.get("command"):
            cmd += ["--command", *[str(float(x)) for x in body["command"][:3]]]
        self._respond(200, start_job(kind, task, cmd, viewer))

    def _open(self, body):
        name = body.get("robot") or body.get("scene")
        cat = {r["name"]: r for r in robots()}
        if name not in cat:
            return self._respond(404, {"error": f"robot '{name}' not in catalog", "valid": sorted(cat)})
        if job_running():
            if job["kind"] != "sim":
                return self._respond(409, {"error": "a train/play job owns the GPU; POST /stop first",
                                           "kind": job["kind"], "label": job["label"]})
            stop_job()
        r = cat[name]
        viewer = resolve_viewer(body, DEFAULT_VIEWER)
        cmd = [PYTHON, "-m", "newton.examples", r["example"], "--viewer", viewer, *r.get("args", [])]
        if body.get("world_count") and r.get("multi_world"):
            cmd += ["--world-count", str(body["world_count"])]
        self._respond(200, start_job("sim", name, cmd, viewer))


def main():
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Newton Control Server on {HOST}:{PORT}  (newton-claw: {CLAW_DIR}, python: {PYTHON})", flush=True)
    print(f"  tasks={len(tasks())} robots={len(robots())}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        stop_job()
        srv.server_close()


if __name__ == "__main__":
    main()
