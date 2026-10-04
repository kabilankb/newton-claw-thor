"""Vectorized Newton environment base class (the Isaac Lab replacement).

One `newton.Model` holds `num_envs` replicated worlds, stepped on the GPU by
`SolverMuJoCo` (MuJoCo Warp). Joint state and control arrays are exposed to
PyTorch zero-copy (`wp.to_torch`) as `(num_envs, n)` views, so observations,
rewards and resets are plain torch ops and the env plugs straight into rsl_rl.

Subclasses implement: `_build_robot`, `_apply_action`, `_get_obs`,
`_get_rewards`, `_get_terminated`, and optionally `_reset_idx` / `_post_init`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import warp as wp
from rsl_rl.env import VecEnv
from tensordict import TensorDict

import newton


@dataclass
class EnvCfg:
    sim_dt: float = 0.005            # physics step [s]
    decimation: int = 4              # physics steps per policy step
    episode_length_s: float = 20.0
    num_actions: int = 0
    clip_actions: float = 100.0
    env_spacing: float = 2.5         # viewer-only world offset (physics worlds overlap)
    max_viewer_worlds: int = 36      # worlds drawn in the viewer (all worlds still simulate)
    solver: dict = field(default_factory=dict)   # kwargs for newton.solvers.SolverMuJoCo


class NewtonVecEnv(VecEnv):
    cfg: EnvCfg

    def __init__(self, cfg: EnvCfg, num_envs: int, device: str = "cuda:0", viewer=None, seed: int = 42):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.viewer = viewer
        self.num_actions = cfg.num_actions
        self.step_dt = cfg.sim_dt * cfg.decimation
        self.max_episode_length = math.ceil(cfg.episode_length_s / self.step_dt)
        torch.manual_seed(seed)
        wp.set_device(device)

        # ---- model: one robot builder replicated into num_envs worlds ----
        robot = self._build_robot()
        scene = newton.ModelBuilder()
        scene.replicate(robot, num_envs)
        self._add_ground(scene)
        self.model = scene.finalize()
        # Asset USDs may carry their own gravity (the G1 one is zero-g): set it BEFORE the solver
        # is built, because SolverMuJoCo copies gravity at construction.
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.solver = newton.solvers.SolverMuJoCo(self.model, **cfg.solver)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # ---- zero-copy torch views, shape (num_envs, n) ----
        n = num_envs
        self.nq = self.model.joint_coord_count // n
        self.nv = self.model.joint_dof_count // n
        self.q = wp.to_torch(self.state_0.joint_q).view(n, self.nq)
        self.qd = wp.to_torch(self.state_0.joint_qd).view(n, self.nv)
        self.joint_f = wp.to_torch(self.control.joint_f).view(n, self.nv)
        self.target_q = wp.to_torch(self.control.joint_target_q).view(n, -1)
        self.default_q = self.q.clone()
        self.default_qd = self.qd.clone()

        self.actions = torch.zeros(n, self.num_actions, device=device)
        self.last_actions = torch.zeros_like(self.actions)
        self.episode_length_buf = torch.zeros(n, dtype=torch.long, device=device)
        self.rew_buf = torch.zeros(n, device=device)
        self._ep_sums: dict[str, torch.Tensor] = {}
        self._reset_mask = wp.zeros(n + 1, dtype=wp.bool, device=device)   # +1: global world slot
        self._reset_mask_t = wp.to_torch(self._reset_mask)
        self.sim_time = 0.0

        self._post_init()

        if viewer is not None:
            viewer.set_model(self.model)
            if hasattr(viewer, "set_world_offsets"):
                viewer.set_world_offsets((cfg.env_spacing, cfg.env_spacing, 0.0))
            # Draw only the first few worlds: training keeps all num_envs, the window stays fast.
            if num_envs > cfg.max_viewer_worlds and hasattr(viewer, "set_visible_worlds"):
                viewer.set_visible_worlds(range(cfg.max_viewer_worlds))

        # Capture the decimated physics step as one CUDA graph (big win on Thor).
        self.graph = None
        if wp.get_device(device).is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph = capture.graph

        self._reset_idx(torch.arange(n, device=device))
        self.obs_buf = self._get_obs()

    # ------------------------------------------------------------------ hooks
    def _build_robot(self) -> newton.ModelBuilder:
        raise NotImplementedError

    def _add_ground(self, scene: newton.ModelBuilder) -> None:
        scene.add_ground_plane()

    def _post_init(self) -> None:
        pass

    def _apply_action(self, actions: torch.Tensor) -> None:
        raise NotImplementedError

    def _get_obs(self) -> torch.Tensor:
        raise NotImplementedError

    def _get_rewards(self) -> torch.Tensor:
        raise NotImplementedError

    def _get_terminated(self) -> torch.Tensor:
        raise NotImplementedError

    def _reset_idx(self, ids: torch.Tensor) -> None:
        """Default reset: clear solver warm-start and restore default joint state."""
        self._reset_mask_t.zero_()
        self._reset_mask_t[ids] = True
        self.solver.reset(self.state_0, world_mask=self._reset_mask)
        self.q[ids] = self.default_q[ids]
        self.qd[ids] = self.default_qd[ids]
        self.actions[ids] = 0.0
        self.last_actions[ids] = 0.0
        self.episode_length_buf[ids] = 0

    # ------------------------------------------------------------------ core
    def _simulate(self) -> None:
        a, b = self.state_0, self.state_1
        for _ in range(self.cfg.decimation):
            a.clear_forces()
            self.solver.step(a, b, self.control, None, self.cfg.sim_dt)
            a, b = b, a
        if a is not self.state_0:        # keep state_0 canonical so torch views stay valid
            self.state_0.assign(a)

    def get_observations(self) -> TensorDict:
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor):
        self.last_actions[:] = self.actions
        self.actions[:] = torch.clamp(actions, -self.cfg.clip_actions, self.cfg.clip_actions)
        self._apply_action(self.actions)
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate()
        self.sim_time += self.step_dt
        self.episode_length_buf += 1

        terminated = self._get_terminated()
        time_outs = self.episode_length_buf >= self.max_episode_length
        self.rew_buf[:] = self._get_rewards()
        dones = terminated | time_outs

        log = {}
        ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() > 0:
            for name, s in self._ep_sums.items():
                log[f"Episode_Reward/{name}"] = (s[ids] / self.episode_length_buf[ids].clamp(min=1)).mean()
                s[ids] = 0.0
            self._reset_idx(ids)
        self.obs_buf = self._get_obs()
        extras = {"time_outs": time_outs, "log": log}
        return self.get_observations(), self.rew_buf, dones.to(torch.long), extras

    def _track(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """Accumulate a per-step reward term for episode logging; returns value."""
        if name not in self._ep_sums:
            self._ep_sums[name] = torch.zeros(self.num_envs, device=self.device)
        self._ep_sums[name] += value
        return value

    def render(self) -> None:
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def close(self) -> None:
        if self.viewer is not None and hasattr(self.viewer, "close"):
            self.viewer.close()


# ---------------------------------------------------------------------- math
def quat_rotate_inverse(q_xyzw: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vector v into the body frame of quaternion q (x, y, z, w)."""
    q_vec, q_w = q_xyzw[:, :3], q_xyzw[:, 3:4]
    a = v * (2.0 * q_w**2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (q_vec * v).sum(-1, keepdim=True) * 2.0
    return a - b + c
