"""Legged velocity tracking on flat ground (Unitree Go2 / G1, ANYmal C).

Robot USDs and PD gains come from the `newton-assets` repo (downloaded and
cached by `newton.utils.download_asset`). The observation layout is the Isaac
Lab one — base lin vel, base ang vel, projected gravity, velocity command,
joint pos (rel. default), joint vel, last action — so the pretrained ONNX
policies shipped with newton-assets run in this env unchanged (see play.py
`--pretrained`), and policies trained here deploy the same way.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import warp as wp
import yaml

import newton
import newton.utils
from newton import JointTargetMode

from .base import EnvCfg, NewtonVecEnv, quat_rotate_inverse


@dataclass
class LeggedVelocityCfg(EnvCfg):
    asset_dir: str = "unitree_go2"
    asset_usd: str = "usd/go2.usda"
    asset_yaml: str = "rl_policies/go2.yaml"
    pretrained_onnx: str = "rl_policies/mjw_go2.onnx"
    sim_dt: float = 0.005
    decimation: int = 4
    episode_length_s: float = 20.0
    init_height: float = 0.34
    min_height: float = 0.2                 # terminate below this base height [m]
    max_tilt_gravity_z: float = -0.5        # terminate when projected gravity z rises above this
    command_resample_s: float = 10.0
    cmd_lin_x: tuple = (-1.0, 1.0)
    cmd_lin_y: tuple = (-0.5, 0.5)
    cmd_yaw: tuple = (-1.0, 1.0)
    joint_pos_noise: float = 0.1
    tracking_sigma: float = 0.25
    clip_actions: float = 10.0              # PD targets are unbounded in MuJoCo: keep exploration sane
    only_positive_rewards: bool = True      # clip the per-step sum at 0 (termination added after) so
                                            # "die early to stop paying penalties" is never optimal
    rewards: dict = field(default_factory=lambda: {
        "track_lin_vel_xy_exp": 1.5,
        "track_ang_vel_z_exp": 0.75,
        "lin_vel_z_l2": -2.0,
        "ang_vel_xy_l2": -0.05,
        "flat_orientation_l2": -2.5,
        "dof_torques_l2": -2.0e-4,
        "dof_acc_l2": -2.5e-7,
        "action_rate_l2": -0.01,
        "joint_deviation_l1": 0.0,
        "alive": 0.0,
        "termination": -200.0,
    })
    solver: dict = field(default_factory=lambda: {"solver": "newton", "njmax": 150, "nconmax": 48})


class LeggedVelocityEnv(NewtonVecEnv):
    cfg: LeggedVelocityCfg

    def _build_robot(self):
        c = self.cfg
        self.asset_root = str(newton.utils.download_asset(c.asset_dir))
        with open(f"{self.asset_root}/{c.asset_yaml}", encoding="utf-8") as f:
            self.robot_cfg = yaml.safe_load(f)
        rc = self.robot_cfg
        n = rc["num_dofs"]

        b = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        b.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=0.1, limit_ke=1.0e2, limit_kd=1.0e0)
        b.default_shape_cfg.ke = 5.0e4
        b.default_shape_cfg.kd = 5.0e2
        b.default_shape_cfg.kf = 1.0e3
        b.default_shape_cfg.mu = 0.75
        b.rigid_gap = 0.0
        b.add_usd(f"{self.asset_root}/{c.asset_usd}", xform=wp.transform(wp.vec3(0, 0, c.init_height)),
                  collapse_fixed_joints=False, enable_self_collisions=False, joint_ordering="dfs",
                  hide_collision_shapes=True)
        b.approximate_meshes("convex_hull")

        b.joint_q[:3] = [0.0, 0.0, c.init_height]
        b.joint_q[3:7] = [0.0, 0.0, 0.0, 1.0]
        b.joint_q[7:] = rc["mjw_joint_pos"]
        for i in range(n):
            b.joint_target_ke[i + 6] = rc["mjw_joint_stiffness"][i]
            b.joint_target_kd[i + 6] = rc["mjw_joint_damping"][i]
            b.joint_armature[i + 6] = rc["mjw_joint_armature"][i]
            b.joint_target_mode[i + 6] = int(JointTargetMode.POSITION)
        c.num_actions = n
        return b

    def _post_init(self):
        c, rc, dev, n = self.cfg, self.robot_cfg, self.device, self.num_envs
        self.num_actions = c.num_actions
        self.actions = torch.zeros(n, self.num_actions, device=dev)
        self.last_actions = torch.zeros_like(self.actions)
        self.action_scale = float(rc["action_scale"])
        self.default_joint_pos = self.default_q[:, 7:].clone()
        self.kp = torch.tensor(rc["mjw_joint_stiffness"], device=dev)
        self.kd = torch.tensor(rc["mjw_joint_damping"], device=dev)
        self.gravity_w = torch.tensor([0.0, 0.0, -1.0], device=dev).repeat(n, 1)
        self.commands = torch.zeros(n, 3, device=dev)
        self.last_joint_vel = torch.zeros(n, self.num_actions, device=dev)
        self.fixed_command = None                    # set by play.py to drive a constant command
        self._resample_steps = max(1, int(c.command_resample_s / self.step_dt))
        self.target_q[:] = self.default_q

    # --------------------------------------------------------------- control
    def _apply_action(self, actions):
        self.target_q[:, 7:] = self.default_joint_pos + self.action_scale * actions

    def _base(self):
        quat = self.q[:, 3:7]
        return (quat_rotate_inverse(quat, self.qd[:, 0:3]), quat_rotate_inverse(quat, self.qd[:, 3:6]),
                quat_rotate_inverse(quat, self.gravity_w))

    def _get_obs(self):
        lin, ang, grav = self._base()
        return torch.cat([lin, ang, grav, self.commands, self.q[:, 7:] - self.default_joint_pos,
                          self.qd[:, 6:], self.actions], dim=-1)

    def _get_terminated(self):
        _, _, grav = self._base()
        return (self.q[:, 2] < self.cfg.min_height) | (grav[:, 2] > self.cfg.max_tilt_gravity_z)

    def _get_rewards(self):
        c, w = self.cfg, self.cfg.rewards
        lin, ang, grav = self._base()
        jpos, jvel = self.q[:, 7:], self.qd[:, 6:]
        torque = self.kp * (self.target_q[:, 7:] - jpos) - self.kd * jvel
        terms = {
            "track_lin_vel_xy_exp": torch.exp(-((self.commands[:, :2] - lin[:, :2]) ** 2).sum(-1) / c.tracking_sigma),
            "track_ang_vel_z_exp": torch.exp(-((self.commands[:, 2] - ang[:, 2]) ** 2) / c.tracking_sigma),
            "lin_vel_z_l2": lin[:, 2] ** 2,
            "ang_vel_xy_l2": (ang[:, :2] ** 2).sum(-1),
            "flat_orientation_l2": (grav[:, :2] ** 2).sum(-1),
            "dof_torques_l2": (torque ** 2).sum(-1),
            "dof_acc_l2": (((jvel - self.last_joint_vel) / self.step_dt) ** 2).sum(-1),
            "action_rate_l2": ((self.actions - self.last_actions) ** 2).sum(-1),
            "joint_deviation_l1": (jpos - self.default_joint_pos).abs().sum(-1),
            "alive": torch.ones(self.num_envs, device=self.device),
            "termination": self._get_terminated().float(),
        }
        self.last_joint_vel[:] = jvel
        total = torch.zeros(self.num_envs, device=self.device)
        for name, value in terms.items():
            if name != "termination" and w.get(name, 0.0) != 0.0:
                total += self._track(name, w[name] * value * self.step_dt)
        if c.only_positive_rewards:
            total = total.clamp(min=0.0)
        total += self._track("termination", w.get("termination", 0.0) * terms["termination"] * self.step_dt)
        # periodic command resampling (after rewards so this step is scored on the old command)
        ids = (self.episode_length_buf % self._resample_steps == 0).nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() > 0:
            self._resample_commands(ids)
        return total

    # ----------------------------------------------------------------- reset
    def _resample_commands(self, ids):
        if self.fixed_command is not None:
            self.commands[ids] = self.fixed_command
            return
        c, n = self.cfg, ids.numel()

        def u(lo_hi):
            return torch.rand(n, device=self.device) * (lo_hi[1] - lo_hi[0]) + lo_hi[0]

        self.commands[ids, 0], self.commands[ids, 1], self.commands[ids, 2] = u(c.cmd_lin_x), u(c.cmd_lin_y), u(c.cmd_yaw)
        self.commands[ids] *= (torch.rand(n, 1, device=self.device) > 0.05)      # 5% standing envs

    def _reset_idx(self, ids):
        super()._reset_idx(ids)
        n = ids.numel()
        yaw = (torch.rand(n, device=self.device) - 0.5) * 2.0 * math.pi
        self.q[ids, 3], self.q[ids, 4] = 0.0, 0.0
        self.q[ids, 5], self.q[ids, 6] = torch.sin(yaw / 2), torch.cos(yaw / 2)
        self.q[ids, 7:] += (torch.rand(n, self.num_actions, device=self.device) - 0.5) * 2.0 * self.cfg.joint_pos_noise
        self.qd[ids, :6] = (torch.rand(n, 6, device=self.device) - 0.5) * 0.5
        self.target_q[ids] = self.default_q[ids]
        self.last_joint_vel[ids] = 0.0
        self._resample_commands(ids)


# ------------------------------------------------------------------ robots
def go2_cfg() -> LeggedVelocityCfg:
    return LeggedVelocityCfg()


def anymal_c_cfg() -> LeggedVelocityCfg:
    cfg = LeggedVelocityCfg(asset_dir="anybotics_anymal_c", asset_usd="usd/anymal_c.usda",
                            asset_yaml="rl_policies/anymal.yaml", pretrained_onnx="rl_policies/mjw_anymal.onnx",
                            init_height=0.6, min_height=0.3)
    cfg.rewards["dof_torques_l2"] = -2.5e-5
    return cfg


def g1_cfg() -> LeggedVelocityCfg:
    cfg = LeggedVelocityCfg(asset_dir="unitree_g1", asset_usd="usd/g1_minimal.usd",
                            asset_yaml="rl_policies/g1_23dof.yaml", pretrained_onnx="rl_policies/mjw_g1_23DOF.onnx",
                            init_height=0.8, min_height=0.45, max_tilt_gravity_z=-0.7,
                            cmd_lin_x=(0.0, 1.0), cmd_lin_y=(-0.3, 0.3), joint_pos_noise=0.05,
                            solver={"solver": "newton", "njmax": 250, "nconmax": 80})
    cfg.rewards.update({"track_lin_vel_xy_exp": 2.0, "track_ang_vel_z_exp": 1.0, "flat_orientation_l2": -1.0,
                        "dof_torques_l2": -2.0e-6, "dof_acc_l2": -1.0e-7, "action_rate_l2": -0.005,
                        "joint_deviation_l1": -0.1, "alive": 1.0})
    return cfg
