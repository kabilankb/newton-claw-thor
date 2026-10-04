"""Shared helpers for the newton_lab scripts: PPO config, run dirs, viewers, policies."""
from __future__ import annotations

import glob
import os
import re
import time
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parents[1]                 # <repo>/newton_lab
LOG_ROOT = Path(os.environ.get("NEWTON_LAB_LOGS", LAB_DIR / "logs" / "rsl_rl"))

# Friendly tuning knobs (policies/rsl_rl.md) -> where they live in the rsl_rl config.
ALGO_KNOBS = {"learning_rate", "entropy_coef", "gamma", "lam", "clip_param", "desired_kl", "value_loss_coef",
              "num_learning_epochs", "num_mini_batches", "max_grad_norm", "schedule"}
KNOB_ALIASES = {"lr": "learning_rate", "lambda": "lam"}


def ppo_cfg(task: str, overrides: dict | None = None) -> dict:
    """rsl_rl (v5) OnPolicyRunner config. `overrides` = task defaults + user knobs."""
    o = {KNOB_ALIASES.get(k, k): v for k, v in (overrides or {}).items()}
    hidden = list(o.pop("hidden_dims", [512, 256, 128]))
    init_std = float(o.pop("init_noise_std", 1.0))
    cfg = {
        "seed": int(o.pop("seed", 42)),
        "num_steps_per_env": int(o.pop("num_steps_per_env", 24)),
        "max_iterations": int(o.pop("max_iterations", 1000)),
        "save_interval": int(o.pop("save_interval", 50)),
        "experiment_name": task,
        "run_name": "",
        "logger": "tensorboard",
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {"class_name": "MLPModel", "hidden_dims": hidden, "activation": "elu", "obs_normalization": True,
                  "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": init_std,
                                       "std_type": "scalar"}},
        "critic": {"class_name": "MLPModel", "hidden_dims": hidden, "activation": "elu", "obs_normalization": True},
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 5, "num_mini_batches": 4, "clip_param": 0.2,
                      "gamma": 0.99, "lam": 0.95, "value_loss_coef": 1.0, "entropy_coef": 0.01,
                      "learning_rate": 1.0e-3, "max_grad_norm": 1.0, "use_clipped_value_loss": True,
                      "schedule": "adaptive", "desired_kl": 0.01},
    }
    unknown = []
    for k, v in o.items():
        if k in ALGO_KNOBS:
            cfg["algorithm"][k] = v if k == "schedule" else type(cfg["algorithm"][k])(v)
        else:
            unknown.append(k)
    cfg["_unknown_knobs"] = unknown
    return cfg


def parse_overrides(pairs: list[str]) -> dict:
    """['learning_rate=3e-4', 'reward.action_rate_l2=-0.02'] -> dict (values parsed as numbers when possible)."""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        out[k.strip()] = v
    return out


def apply_env_overrides(env_cfg, knobs: dict) -> list[str]:
    """Pop `reward.<term>` / `reward_<term>` / `env.<field>` knobs into the env cfg; returns what was applied."""
    applied = []
    for k in list(knobs):
        term = k[7:] if k.startswith(("reward.", "reward_")) else None
        if term is not None and hasattr(env_cfg, "rewards") and term in env_cfg.rewards:
            env_cfg.rewards[term] = float(knobs.pop(k)); applied.append(k)
        elif k.startswith("env.") and hasattr(env_cfg, k[4:]):
            setattr(env_cfg, k[4:], type(getattr(env_cfg, k[4:]))(knobs.pop(k))); applied.append(k)
    return applied


def new_run_dir(task: str) -> Path:
    d = LOG_ROOT / task / time.strftime("%Y-%m-%d_%H-%M-%S")
    d.mkdir(parents=True, exist_ok=True)
    return d


def latest_checkpoint(task: str) -> Path | None:
    """Newest model_<N>.pt of the newest run of `task` (None if never trained)."""
    runs = sorted(glob.glob(str(LOG_ROOT / task / "*")))
    for run in reversed(runs):
        ckpts = glob.glob(os.path.join(run, "model_*.pt"))
        if ckpts:
            return Path(max(ckpts, key=lambda p: int(re.search(r"model_(\d+)\.pt", p).group(1))))
    return None


def make_viewer(kind: str, port: int = 8090, output: str | None = None):
    """gl = desktop window (needs DISPLAY) · viser = browser at http://<thor-ip>:<port> · none = headless."""
    if kind in (None, "none", "null"):
        return None
    import newton.viewer
    if kind == "gl":
        return newton.viewer.ViewerGL()
    if kind == "viser":
        return newton.viewer.ViewerViser(port=port)
    if kind == "usd":
        return newton.viewer.ViewerUSD(output_path=output or "play.usd", num_frames=None)
    raise ValueError(f"unknown viewer '{kind}' (gl | viser | usd | none)")


class OnnxPolicy:
    """Torch-free GPU inference: runs an exported ONNX policy with Warp-NN (same device as the sim)."""

    def __init__(self, path: str, device: str = "cuda:0"):
        import warp as wp
        from warp_nn.runtime import OnnxRuntime
        self._wp = wp
        self.rt = OnnxRuntime(str(path), device=device)
        self.inp, self.out = self.rt.input_names[0], self.rt.output_names[0]

    def __call__(self, obs):
        wp = self._wp
        out = self.rt({self.inp: wp.from_torch(obs.contiguous())})[self.out]
        return wp.to_torch(out)


def export_deploy_onnx(actor, path, opset: int = 17) -> None:
    """Export an rsl_rl MLP actor as a Gemm/Elu-only ONNX graph (input "obs", output "actions").

    Observation normalization is folded into the first layer, because Warp-NN's ONNX runtime
    (the torch-free Thor inference path) only supports Gemm / Elu / LSTM / Squeeze.
    """
    import numpy as np
    import onnx
    import torch
    from onnx import TensorProto, helper, numpy_helper

    linears = [m for m in actor.mlp if isinstance(m, torch.nn.Linear)]
    others = [m for m in actor.mlp if not isinstance(m, torch.nn.Linear)]
    if not all(isinstance(m, torch.nn.ELU) for m in others):
        raise ValueError("export_deploy_onnx supports ELU MLPs only")
    weights = [(layer.weight.detach().double().cpu(), layer.bias.detach().double().cpu()) for layer in linears]
    norm = actor.obs_normalizer
    if hasattr(norm, "_mean"):                               # y = W((x - mean) / (std + eps)) + b
        mean = norm._mean.detach().double().cpu().squeeze(0)
        scale = 1.0 / (norm._std.detach().double().cpu().squeeze(0) + norm.eps)
        w0, b0 = weights[0]
        weights[0] = (w0 * scale, b0 - (w0 * scale) @ mean)

    nodes, inits, x = [], [], "obs"
    for i, (w, b) in enumerate(weights):
        inits += [numpy_helper.from_array(w.numpy().astype(np.float32), f"w{i}"),
                  numpy_helper.from_array(b.numpy().astype(np.float32), f"b{i}")]
        last = i == len(weights) - 1
        y = "actions" if last else f"h{i}"
        nodes.append(helper.make_node("Gemm", [x, f"w{i}", f"b{i}"], [y], alpha=1.0, beta=1.0, transB=1))
        if not last:
            nodes.append(helper.make_node("Elu", [y], [f"a{i}"], alpha=1.0))
            y = f"a{i}"
        x = y
    graph = helper.make_graph(
        nodes, "policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, weights[0][0].shape[1]])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, weights[-1][0].shape[0]])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
