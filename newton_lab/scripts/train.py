#!/usr/bin/env python3
"""Train an RL policy in Newton physics with rsl_rl PPO (runs on the Jetson Thor GPU).

    python train.py --task Newton-Velocity-Flat-Unitree-Go2-v0 [--num_envs 4096]
                    [--max_iterations N] [--seed N] [--resume] [--checkpoint PATH]
                    [--viewer gl|viser|none | --headless] [knob=value ...]

Knobs (see policies/rsl_rl.md): learning_rate entropy_coef gamma lam clip_param
desired_kl num_learning_epochs num_mini_batches num_steps_per_env init_noise_std,
reward.<term>=<weight>, env.<field>=<value>.
Writes checkpoints + policy.onnx + policy.pt to newton_lab/logs/rsl_rl/<task>/<run>/.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--max_iterations", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--viewer", default="gl", choices=["none", "gl", "viser"])
    p.add_argument("--headless", action="store_true", help="no window (same as --viewer none); fastest")
    p.add_argument("--resume", action="store_true", help="continue from the latest checkpoint of this task")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("knobs", nargs="*", help="key=value tuning overrides")
    args = p.parse_args()

    from rsl_rl.runners import OnPolicyRunner

    from newton_lab import common, envs

    knobs = common.parse_overrides(args.knobs)
    env_cfg = envs.make_cfg(args.task)
    applied_env = common.apply_env_overrides(env_cfg, knobs)
    overrides = {**envs.ppo_overrides(args.task), **knobs, "seed": args.seed}
    if args.max_iterations:
        overrides["max_iterations"] = args.max_iterations
    cfg = common.ppo_cfg(args.task, overrides)
    unknown = cfg.pop("_unknown_knobs")
    if unknown:
        print(f"[train] WARNING ignoring unknown knobs: {unknown}")

    if args.headless:
        args.viewer = "none"
    viewer = common.make_viewer(args.viewer)
    env = envs.make(args.task, args.num_envs, device=args.device, viewer=viewer, seed=args.seed, cfg=env_cfg)
    if viewer is not None:                       # draw at most ~30 frames/s so the window barely slows training
        import time
        step, last = env.step, [0.0]

        def step_and_render(actions):
            out = step(actions)
            now = time.perf_counter()
            if now - last[0] > 1.0 / 30.0 and last[0] >= 0.0:
                if hasattr(viewer, "is_running") and not viewer.is_running():
                    last[0] = -1.0               # window was closed: keep training, stop drawing
                    print("[train] viewer closed — training continues headless")
                else:
                    env.render()
                    last[0] = time.perf_counter()
            return out
        env.step = step_and_render

    run_dir = common.new_run_dir(args.task)
    print(f"[train] task={args.task} num_envs={args.num_envs} iters={cfg['max_iterations']} "
          f"obs={env.get_observations()['policy'].shape[1]} act={env.num_actions} run_dir={run_dir}")
    if applied_env:
        print(f"[train] env overrides applied: {applied_env}")
    (run_dir / "params.json").write_text(json.dumps(
        {"task": args.task, "num_envs": args.num_envs, "agent": cfg, "knobs": args.knobs}, indent=2, default=str))

    runner = OnPolicyRunner(env, cfg, log_dir=str(run_dir), device=args.device)
    ckpt = args.checkpoint or (common.latest_checkpoint(args.task) if args.resume else None)
    if ckpt:
        print(f"[train] resuming from {ckpt}")
        runner.load(str(ckpt))
    runner.learn(num_learning_iterations=cfg["max_iterations"], init_at_random_ep_len=True)

    # Deployment artifacts for Thor inference (ONNX -> Warp-NN / TensorRT, TorchScript -> libtorch).
    try:
        common.export_deploy_onnx(runner.alg.get_policy(), run_dir / "policy.onnx")
        runner.export_policy_to_jit(str(run_dir), "policy.pt")
        print(f"[train] exported {run_dir}/policy.onnx and policy.pt")
    except Exception as e:                       # export must never lose a finished run
        print(f"[train] WARNING export failed: {e}")
    print(f"[train] DONE {run_dir}")
    env.close()


if __name__ == "__main__":
    main()
