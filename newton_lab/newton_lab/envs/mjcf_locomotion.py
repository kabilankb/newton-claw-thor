"""Torque-controlled MJCF runners (Ant, Humanoid): maximise forward speed.

Assets ship inside the `newton` wheel (newton/examples/assets), so these tasks
need no download. Actions are joint torques scaled by the MJCF motor gears.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import torch
import warp as wp

import newton
import newton.examples

from .base import EnvCfg, NewtonVecEnv, quat_rotate_inverse


@dataclass
class MjcfLocomotionCfg(EnvCfg):
    asset: str = "nv_ant.xml"
    sim_dt: float = 1.0 / 120.0
    decimation: int = 2
    episode_length_s: float = 15.0
    init_height_offset: float = 0.0
    min_height: float = 0.3          # terminate below this torso height [m]
    joint_pos_noise: float = 0.1
    w_forward: float = 1.0
    w_alive: float = 0.5
    w_ctrl: float = -0.005
    w_upright: float = 0.1
    w_termination: float = -2.0
    solver: dict = field(default_factory=lambda: {"njmax": 100, "nconmax": 30})


class MjcfLocomotionEnv(NewtonVecEnv):
    cfg: MjcfLocomotionCfg

    def _build_robot(self):
        path = newton.examples.get_asset(self.cfg.asset)
        b = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        b.default_shape_cfg.ke = 1.0e4
        b.add_mjcf(path, ignore_names=["floor", "ground"], parse_sites=False,
                   xform=wp.transform((0.0, 0.0, self.cfg.init_height_offset), wp.quat_identity()))
        # Start hinge joints mid-range (several MJCF defaults sit outside their limits).
        for i in range(7, b.joint_coord_count):
            lo, hi = b.joint_limit_lower[i - 1], b.joint_limit_upper[i - 1]
            if lo > 0.0 or hi < 0.0:
                b.joint_q[i] = 0.5 * (lo + hi)
        # Per-DOF torque scale from the MJCF <motor gear=...> entries. Newton keeps the MJCF
        # joint order (depth-first), so hinge i in document order is actuated DOF i.
        root = ET.parse(path).getroot()
        gears = {m.get("joint"): float(m.get("gear", "1")) for m in root.iter("motor")}
        hinges = [j.get("name") for j in root.find("worldbody").iter("joint") if j.get("type", "hinge") != "free"]
        if len(hinges) != b.joint_dof_count - 6:
            raise RuntimeError(f"{self.cfg.asset}: {len(hinges)} MJCF hinges vs {b.joint_dof_count - 6} actuated DOFs")
        self._gear = [0.0] * 6 + [gears.get(name, 0.0) for name in hinges]
        return b

    def _post_init(self):
        self.cfg.num_actions = self.num_actions = self.nv - 6
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.gear = torch.tensor(self._gear[6:], device=self.device)
        self.gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)

    def _apply_action(self, actions):
        self.joint_f[:, 6:] = self.gear * torch.clamp(actions, -1.0, 1.0)

    def _base(self):
        quat = self.q[:, 3:7]
        return (quat_rotate_inverse(quat, self.qd[:, 0:3]), quat_rotate_inverse(quat, self.qd[:, 3:6]),
                quat_rotate_inverse(quat, self.gravity_w))

    def _get_obs(self):
        lin, ang, grav = self._base()
        return torch.cat([self.q[:, 2:3], lin, ang, grav, self.q[:, 7:], 0.1 * self.qd[:, 6:], self.actions], dim=-1)

    def _get_terminated(self):
        return self.q[:, 2] < self.cfg.min_height

    def _get_rewards(self):
        c = self.cfg
        _, _, grav = self._base()
        fwd = self._track("forward_vel", c.w_forward * self.qd[:, 0])
        alive = self._track("alive", c.w_alive * torch.ones_like(fwd))
        ctrl = self._track("ctrl_cost", c.w_ctrl * (self.actions ** 2).sum(-1))
        up = self._track("upright", c.w_upright * (-grav[:, 2]))
        return fwd + alive + ctrl + up + c.w_termination * self._get_terminated().float()

    def _reset_idx(self, ids):
        super()._reset_idx(ids)
        n = ids.numel()
        self.q[ids, 7:] += (torch.rand(n, self.nq - 7, device=self.device) - 0.5) * 2.0 * self.cfg.joint_pos_noise
        self.qd[ids] = (torch.rand(n, self.nv, device=self.device) - 0.5) * 0.2


def ant_cfg() -> MjcfLocomotionCfg:
    return MjcfLocomotionCfg(asset="nv_ant.xml", min_height=0.3)


def humanoid_cfg() -> MjcfLocomotionCfg:
    return MjcfLocomotionCfg(asset="nv_humanoid.xml", init_height_offset=1.35, min_height=0.8, w_alive=2.0, w_ctrl=-0.01,
                             w_upright=0.5, joint_pos_noise=0.05,
                             solver={"njmax": 150, "nconmax": 40})
