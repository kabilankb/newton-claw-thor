#!/usr/bin/env python3
"""Re-export a checkpoint to deployment formats (policy.onnx for Warp-NN/TensorRT, policy.pt TorchScript).

    python export.py --task <id> [--checkpoint PATH] [--out DIR]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    from rsl_rl.runners import OnPolicyRunner

    from newton_lab import common, envs

    ckpt = Path(args.checkpoint) if args.checkpoint else common.latest_checkpoint(args.task)
    if not ckpt:
        sys.exit(f"[export] no checkpoint for {args.task}")
    out = Path(args.out) if args.out else ckpt.parent
    env = envs.make(args.task, 1, device=args.device)
    cfg = common.ppo_cfg(args.task, envs.ppo_overrides(args.task))
    cfg.pop("_unknown_knobs")
    runner = OnPolicyRunner(env, cfg, log_dir=None, device=args.device)
    runner.load(str(ckpt))
    common.export_deploy_onnx(runner.alg.get_policy(), out / "policy.onnx")
    runner.export_policy_to_jit(str(out), "policy.pt")
    print(f"[export] {ckpt} -> {out}/policy.onnx, {out}/policy.pt")


if __name__ == "__main__":
    main()
