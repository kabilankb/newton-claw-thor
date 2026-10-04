#!/usr/bin/env python3
"""Run policy INFERENCE in Newton on the Jetson Thor: watch it, evaluate it, or benchmark it.

    python play.py --task <id> [--checkpoint PATH | --onnx PATH | --pretrained]
                   [--num_envs 16] [--viewer gl|viser|usd|none] [--steps N]
                   [--command VX VY YAW] [--eval]

Policy source (first match wins):
  --onnx PATH     exported ONNX, run with Warp-NN on the GPU (no PyTorch in the loop)
  --pretrained    the reference ONNX shipped with newton-assets (Go2 / G1 / ANYmal)
  --checkpoint    an rsl_rl model_*.pt  (default: latest run of the task)
--eval runs headless and prints one JSON line of metrics (reward, episode length,
velocity-tracking error, policy + sim throughput).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--onnx", default=None)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--steps", type=int, default=None, help="policy steps to run (default: 1000 eval, endless viewer)")
    p.add_argument("--viewer", default="gl", choices=["gl", "viser", "usd", "none"])
    p.add_argument("--viser_port", type=int, default=8090)
    p.add_argument("--output", default=None, help="USD output path for --viewer usd")
    p.add_argument("--command", type=float, nargs=3, default=None, metavar=("VX", "VY", "YAW"))
    p.add_argument("--eval", action="store_true")
    p.add_argument("--headless", action="store_true", help="same as --viewer none")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    import torch

    from newton_lab import common, envs

    if args.eval or args.headless:
        args.viewer = "none"
    num_envs = args.num_envs or (1024 if args.eval else 16)
    steps = args.steps or (1000 if args.viewer == "none" else None)

    viewer = common.make_viewer(args.viewer, port=args.viser_port, output=args.output)
    env = envs.make(args.task, num_envs, device=args.device, viewer=viewer, seed=args.seed)
    if args.command is not None and hasattr(env, "fixed_command"):
        env.fixed_command = torch.tensor(args.command, device=args.device)
        env.commands[:] = env.fixed_command

    # ---- policy ----
    onnx = args.onnx
    if args.pretrained:
        rel = getattr(env.cfg, "pretrained_onnx", None)
        if not rel:
            sys.exit(f"[play] no pretrained policy ships for {args.task}")
        onnx = f"{env.asset_root}/{rel}"
    if onnx:
        source = f"onnx:{onnx}"
        if num_envs == 1:
            policy = common.OnnxPolicy(onnx, device=args.device)
        else:                                   # exported graphs are batch-1; batch N via onnx -> torch weights
            policy = _batched_onnx(onnx, args.device)
    else:
        from rsl_rl.runners import OnPolicyRunner
        ckpt = args.checkpoint or common.latest_checkpoint(args.task)
        if not ckpt:
            sys.exit(f"[play] no checkpoint for {args.task} — train it first, or pass --pretrained / --onnx")
        cfg = common.ppo_cfg(args.task, envs.ppo_overrides(args.task))
        cfg.pop("_unknown_knobs")
        runner = OnPolicyRunner(env, cfg, log_dir=None, device=args.device)
        runner.load(str(ckpt))
        model = runner.get_inference_policy(device=args.device)
        policy = lambda obs: model(env.get_observations(), stochastic_output=False)   # noqa: E731
        source = f"checkpoint:{ckpt}"
    print(f"[play] task={args.task} num_envs={num_envs} policy={source} viewer={args.viewer}")
    if args.viewer == "viser":
        print(f"[play] open http://<thor-ip>:{args.viser_port} in a browser")

    # ---- loop ----
    obs = env.get_observations()["policy"]
    ep_rew = torch.zeros(num_envs, device=args.device)
    done_rew, done_len, track_err, t_policy = [], [], [], 0.0
    ep_len = torch.zeros(num_envs, device=args.device)
    i, falls, t0 = 0, 0, time.perf_counter()
    with torch.inference_mode():
        while steps is None or i < steps:
            if viewer is not None and hasattr(viewer, "is_running") and not viewer.is_running():
                break
            tp = time.perf_counter()
            actions = policy(obs)
            t_policy += time.perf_counter() - tp
            td, rew, dones, extras = env.step(actions)
            falls += int((dones.bool() & ~extras["time_outs"]).sum())
            obs = td["policy"]
            ep_rew += rew
            ep_len += 1
            if hasattr(env, "commands"):
                lin, _, _ = env._base()
                track_err.append((env.commands[:, :2] - lin[:, :2]).norm(dim=-1).mean())
            d = dones.nonzero(as_tuple=False).squeeze(-1)
            if d.numel() > 0:
                done_rew.append(ep_rew[d].clone()); done_len.append(ep_len[d].clone())
                ep_rew[d] = 0.0; ep_len[d] = 0.0
            if viewer is not None:
                env.render()
            i += 1
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    # finished episodes plus the ones still running, so a few early falls don't skew the mean
    all_rew = torch.cat([*done_rew, ep_rew[ep_len > 0]])
    all_len = torch.cat([*done_len, ep_len[ep_len > 0]])
    result = {
        "task": args.task, "policy": source, "num_envs": num_envs, "policy_steps": i,
        "episodes_finished": int(sum(x.numel() for x in done_rew)),
        "falls": falls,
        "falls_per_robot_minute": round(falls / max(i * num_envs * env.step_dt / 60.0, 1e-9), 3),
        "mean_episode_reward": round(float(all_rew.mean()), 3),
        "mean_episode_length_s": round(float(all_len.mean()) * env.step_dt, 2),
        "max_episode_length_s": env.cfg.episode_length_s,
        "env_steps_per_s": round(i * num_envs / wall),
        "realtime_factor_per_env": round(i * env.step_dt / wall, 2),
        "policy_ms_per_call": round(1000.0 * t_policy / max(i, 1), 3),
    }
    if track_err:
        result["mean_lin_vel_tracking_error_mps"] = round(float(torch.stack(track_err).mean()), 3)
    print("[play] RESULT " + json.dumps(result))
    env.close()


def _batched_onnx(path, device):
    """Batch-N inference for an MLP ONNX policy: rebuild it as torch layers from the graph initializers."""
    import onnx
    import torch
    from onnx import numpy_helper

    g = onnx.load(str(path)).graph
    init = {t.name: torch.tensor(numpy_helper.to_array(t), device=device) for t in g.initializer}
    const = {n.output[0]: torch.tensor(numpy_helper.to_array(n.attribute[0].t), device=device)
             for n in g.node if n.op_type == "Constant" and n.attribute and n.attribute[0].name == "value"}
    init.update(const)
    inp = g.input[0].name

    def run(obs):
        v = {inp: obs}
        for n in g.node:
            a = [v[x] if x in v else init[x] for x in n.input]
            t = n.op_type
            if t == "Constant":
                continue
            if t == "Gemm":
                attrs = {x.name: x for x in n.attribute}
                w = a[1].T if ("transB" in attrs and attrs["transB"].i) else a[1]
                out = a[0] @ w + (a[2] if len(a) > 2 else 0.0)
            elif t == "MatMul":
                out = a[0] @ a[1]
            elif t in ("Add", "Sub", "Mul", "Div"):
                out = {"Add": torch.add, "Sub": torch.sub, "Mul": torch.mul, "Div": torch.div}[t](a[0], a[1])
            elif t == "Elu":
                out = torch.nn.functional.elu(a[0], alpha=next((x.f for x in n.attribute if x.name == "alpha"), 1.0))
            elif t == "Relu":
                out = torch.relu(a[0])
            elif t == "Tanh":
                out = torch.tanh(a[0])
            elif t == "Clip":
                out = torch.clamp(a[0], a[1] if len(a) > 1 else None, a[2] if len(a) > 2 else None)
            elif t == "Sqrt":
                out = torch.sqrt(a[0])
            elif t == "Identity":
                out = a[0]
            else:
                raise NotImplementedError(f"ONNX op '{t}' not supported for batched play; use --num_envs 1")
            v[n.output[0]] = out
        return v[g.output[0].name]

    return run


if __name__ == "__main__":
    main()
