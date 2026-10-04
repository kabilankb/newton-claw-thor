"""Cartpole balance — the smoke-test task (trains in well under a minute on Thor)."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

import newton
import newton.examples

from .base import EnvCfg, NewtonVecEnv


@dataclass
class CartpoleCfg(EnvCfg):
    sim_dt: float = 1.0 / 120.0
    decimation: int = 2
    episode_length_s: float = 5.0
    num_actions: int = 1
    clip_actions: float = 1.0
    env_spacing: float = 4.0
    force_scale: float = 100.0       # N per unit action
    max_cart_pos: float = 3.0
    solver: dict = field(default_factory=lambda: {"disable_contacts": True})


class CartpoleEnv(NewtonVecEnv):
    cfg: CartpoleCfg

    def _build_robot(self):
        b = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        b.default_joint_cfg.armature = 0.1
        b.add_usd(newton.examples.get_asset("cartpole_single_pendulum.usda"),
                  enable_self_collisions=False, collapse_fixed_joints=True)
        return b

    def _add_ground(self, scene):
        pass                          # rail is fixed to the world; no ground needed

    def _apply_action(self, actions):
        self.joint_f[:, 0] = self.cfg.force_scale * actions[:, 0]

    def _get_obs(self):
        return torch.cat([self.q, self.qd], dim=-1)          # cart x, pole angle, their rates

    def _get_terminated(self):
        return (self.q[:, 0].abs() > self.cfg.max_cart_pos) | (self.q[:, 1].abs() > 1.5708)

    def _get_rewards(self):
        alive = self._track("alive", torch.ones(self.num_envs, device=self.device))
        pole = self._track("pole_pos", -1.0 * self.q[:, 1] ** 2)
        cart_vel = self._track("cart_vel", -0.01 * self.qd[:, 0].abs())
        pole_vel = self._track("pole_vel", -0.005 * self.qd[:, 1].abs())
        return alive + pole + cart_vel + pole_vel - 2.0 * self._get_terminated().float()

    def _reset_idx(self, ids):
        super()._reset_idx(ids)
        n = ids.numel()
        self.q[ids, 0] = (torch.rand(n, device=self.device) - 0.5) * 2.0
        self.q[ids, 1] = (torch.rand(n, device=self.device) - 0.5) * 0.5
        self.qd[ids] = (torch.rand(n, self.nv, device=self.device) - 0.5) * 0.5
