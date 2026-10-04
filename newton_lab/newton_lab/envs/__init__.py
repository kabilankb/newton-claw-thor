"""Task registry: task id -> (env class, env cfg factory, PPO overrides).

`newton_lab/tasks_registry.json` mirrors these ids for the control server
(which stays stdlib-only and never imports newton/torch). Keep both in sync —
`scripts/list_tasks.py --check` verifies it.
"""
from __future__ import annotations

from .base import EnvCfg, NewtonVecEnv

# id -> (module, class, cfg factory, ppo overrides)
_TASKS = {
    "Newton-Cartpole-v0": ("cartpole", "CartpoleEnv", "CartpoleCfg",
                           {"max_iterations": 150, "num_steps_per_env": 16, "hidden_dims": [64, 64]}),
    "Newton-Ant-v0": ("mjcf_locomotion", "MjcfLocomotionEnv", "ant_cfg",
                      {"max_iterations": 500, "hidden_dims": [256, 128, 64]}),
    "Newton-Humanoid-v0": ("mjcf_locomotion", "MjcfLocomotionEnv", "humanoid_cfg",
                           {"max_iterations": 1500, "hidden_dims": [400, 200, 100]}),
    "Newton-Velocity-Flat-Unitree-Go2-v0": ("legged_velocity", "LeggedVelocityEnv", "go2_cfg",
                                            {"max_iterations": 1000, "init_noise_std": 0.5}),
    "Newton-Velocity-Flat-Anymal-C-v0": ("legged_velocity", "LeggedVelocityEnv", "anymal_c_cfg",
                                         {"max_iterations": 1000, "init_noise_std": 0.5}),
    "Newton-Velocity-Flat-Unitree-G1-v0": ("legged_velocity", "LeggedVelocityEnv", "g1_cfg",
                                           {"max_iterations": 3000, "entropy_coef": 0.008, "init_noise_std": 0.5}),
}


def task_ids() -> list[str]:
    return list(_TASKS)


def make_cfg(task: str) -> EnvCfg:
    import importlib
    mod_name, _, cfg_name, _ = _TASKS[task]
    return getattr(importlib.import_module(f"{__name__}.{mod_name}"), cfg_name)()


def make(task: str, num_envs: int, device: str = "cuda:0", viewer=None, seed: int = 42, cfg: EnvCfg | None = None):
    import importlib
    if task not in _TASKS:
        raise KeyError(f"unknown task '{task}'. valid: {', '.join(_TASKS)}")
    mod_name, cls_name, _, _ = _TASKS[task]
    cls = getattr(importlib.import_module(f"{__name__}.{mod_name}"), cls_name)
    return cls(cfg or make_cfg(task), num_envs=num_envs, device=device, viewer=viewer, seed=seed)


def ppo_overrides(task: str) -> dict:
    return dict(_TASKS[task][3])


__all__ = ["EnvCfg", "NewtonVecEnv", "make", "make_cfg", "ppo_overrides", "task_ids"]
