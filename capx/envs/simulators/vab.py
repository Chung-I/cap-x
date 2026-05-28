"""VAB (Variational-Automation-Benchmark) simulator wrapper.

Adapts CaP-X to step Variational-Automation-Benchmark task YAMLs through the
same `FrankaLiberoEnv` API (`move_to_joints_blocking`, `_get_object_pose`,
`get_observation`, video capture, etc.) so existing FrankaLiberoApi can drive
the agent unchanged.

VAB tasks are loaded via `libero.vab.loader.load_task(yaml_path)` and stepped
through `VABEnv` / `VABBimanualEnv` (subclass of robosuite SingleArmEnv /
TwoArmEnv). The agent-facing observation in VAB is intentionally object-free
(images + proprio only); CaP-X's existing `_get_object_pose` already has a
fallback that resolves objects from MuJoCo body names ending in `_main`,
which matches VAB's object naming scheme.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from capx.envs.simulators.libero import FrankaLiberoEnv


@dataclass
class _VabHandle:
    """Handle stub mirroring the LIBERO `LiberoHandle` interface enough for CaP-X.

    We bypass `VABEnv.step`'s `_filter_obs` and return raw robosuite obs (with
    flat keys like `agentview_image`, `robot0_joint_pos`) because
    `FrankaLiberoEnv.get_observation` and `_get_object_pose` read those keys
    directly from `_current_obs`.
    """

    env: Any
    task_yaml_path: str
    task_id: str
    task_language: str
    init_states: Any  # list of init dicts (we use the index, not raw state vectors)

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        self.env.reset(init_index=0)
        raw = self.env._get_observations(force_update=True)
        return raw, {}

    def step(self, action: list[float]) -> tuple[Any, float, bool, dict[str, Any]]:
        # Call into the underlying robosuite SingleArmEnv.step (which only
        # returns flat raw obs) by going one level up VABEnv.step. We
        # replicate VABEnv.step's success-tagging without the obs filtering.
        from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

        obs, reward, done, info = SingleArmEnv.step(self.env, action)
        try:
            self.env._last_success = bool(self.env._check_success())
            info = dict(info) if info else {}
            info["success"] = self.env._last_success
            info["language"] = self.env.task.language
        except Exception:
            pass
        return obs, float(reward), bool(done), info


class FrankaVabEnv(FrankaLiberoEnv):
    """Franka VAB environment.

    Loads a VAB task YAML and wraps `libero.vab.env.VABEnv` so the existing
    `FrankaLiberoApi` (and the multi-turn agent harness) can drive it.

    Args:
        task_yaml_path: Absolute or repo-relative path to a VAB task YAML.
        privileged: Forwarded to FrankaLiberoEnv (controls whether *Privileged
            APIs surface ground-truth poses; vision-based APIs ignore this).
        max_steps: Episode horizon override; if 0, use task.horizon from YAML.
        seed: Trial seed -> selects init_index (1-based, mod len(inits)).
        viser_debug: Enable viser visualization.
        camera_w / camera_h: Render resolution (overrides task YAML).
        control_freq: Robosuite control frequency.
    """

    def __init__(
        self,
        task_yaml_path: str,
        privileged: bool = False,
        max_steps: int = 0,
        seed: int | None = None,
        enable_render: bool = False,
        control_freq: int = 20,
        viser_debug: bool = False,
        camera_w: int = 800,
        camera_h: int = 512,
    ) -> None:
        # Avoid running FrankaLiberoEnv.__init__ (it loads a LIBERO task).
        # Instead, replicate the relevant init logic with a VAB env handle.
        from libero.vab.loader import load_task  # type: ignore

        # BaseEnv init
        super(FrankaLiberoEnv, self).__init__()

        # Resolve relative paths against repo root
        path = Path(task_yaml_path)
        if not path.is_absolute():
            # Try common candidates: VAB checkout dir, then current dir
            candidates = [
                Path("/k8s-nfs/personal/haoru/gap-x/Variational-Automation-Benchmark") / path,
                Path.cwd() / path,
            ]
            for c in candidates:
                if c.exists():
                    path = c
                    break
        if not path.exists():
            raise FileNotFoundError(f"VAB task YAML not found: {task_yaml_path}")
        self._task_yaml_path = str(path)

        task = load_task(self._task_yaml_path)

        # Override camera resolution + always enable depth (CaP-X needs it for
        # grasp-net / point clouds).
        task = task.model_copy(
            update={
                "camera_height": camera_h,
                "camera_width": camera_w,
                "camera_depth": True,
            }
        )
        self._task = task

        self.privileged = privileged
        self.max_steps = max_steps if max_steps > 0 else task.horizon
        self.seed = seed
        self.enable_render = enable_render
        self.segmentation_level = "instance"
        self._render_width = camera_w
        self._render_height = camera_h

        # Construct VAB env. JOINT_POSITION makes action_dim=8 (7 joints + 1
        # gripper) consistent with FrankaLiberoEnv's move_to_joints_blocking.
        from libero.vab.env import VABEnv as _VABEnv

        vab_env = _VABEnv(
            task,
            controller="JOINT_POSITION",
            control_freq=control_freq,
        )

        # Pre-populate _current_obs (used by FrankaLiberoEnv methods).
        # Use raw robosuite obs (force_update=True) since FrankaLiberoEnv code
        # expects flat keys like `agentview_image`, `robot0_joint_pos`.
        raw_obs = vab_env._get_observations(force_update=True)

        # Build the handle. init_states is just the list of init dicts; we
        # don't use it as a state vector, only its length.
        self.handle = _VabHandle(
            env=vab_env,
            task_yaml_path=self._task_yaml_path,
            task_id=task.id,
            task_language=task.language,
            init_states=task.inits,
        )

        # State tracking
        self._step_count = 0
        self._sim_step_count = 0
        self._control_freq = control_freq
        self._rng = np.random.default_rng(self.seed)
        self._current_obs = raw_obs
        self._current_info: dict[str, Any] = {}
        self._current_reward = 0.0
        self._current_done = False

        # Video capture
        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []
        self._wrist_frame_buffer: list[np.ndarray] = []
        self._record_wrist_camera = False
        self._wrist_camera_name = "robot0_eye_in_hand"
        self._subsample_rate = 4
        self._full_viser_rate = 20

        # Robot link indices for transforms
        self.gripper_metric_length = 0.04
        self.base_link_idx = vab_env.sim.model.body_name2id("robot0_base")
        self.gripper_link_idx = vab_env.sim.model.body_name2id("gripper0_eef")

        self.base_link_wxyz_xyz = np.concatenate(
            [
                vab_env.sim.data.xquat[self.base_link_idx],
                vab_env.sim.data.xpos[self.base_link_idx],
            ]
        )
        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                vab_env.sim.data.xquat[self.gripper_link_idx],
                vab_env.sim.data.xpos[self.gripper_link_idx],
            ]
        )

        # Precompute fast joint qpos addresses for Panda
        joint_names = [f"robot0_joint{i}" for i in range(1, 8)]
        self._panda_joint_qpos_addrs: list[int] = []
        for jn in joint_names:
            addr = vab_env.sim.model.get_joint_qpos_addr(jn)
            if isinstance(addr, tuple):
                addr = addr[0]
            self._panda_joint_qpos_addrs.append(int(addr))

        self.home_joint_position: np.ndarray | None = None

        # No viser by default (heavy)
        self.viser_debug = viser_debug
        self.viser_server = None
        if viser_debug:
            # Defer to parent class viser setup if needed; for batch eval we keep
            # it disabled to avoid port exhaustion across parallel workers.
            raise NotImplementedError(
                "viser_debug not yet wired through FrankaVabEnv; run with viser_debug=False"
            )

        self.reset()

    # -------------------------- Reset / step / success --------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Choose init_index from seed (1-based trial id).
        n_inits = len(self.handle.init_states) if self.handle.init_states else 1
        if seed is not None and n_inits > 0:
            init_index = (seed - 1) % n_inits
        else:
            init_index = 0

        raw_obs = self.handle.env.reset(init_index=init_index)
        # raw_obs returned by VABEnv.reset is filtered ({images, proprio}).
        # We need flat raw obs for FrankaLiberoEnv compatibility.
        self._current_obs = self.handle.env._get_observations(force_update=True)
        self._current_info = {}

        self._step_count = 0
        self._sim_step_count = 0

        self._current_joints = self.handle.env.sim.data.qpos[self._panda_joint_qpos_addrs].copy()
        self.home_joint_position = np.array(
            self._current_obs["robot0_joint_pos"], dtype=np.float64
        )
        self._gripper_fraction = 1.0

        # Post-reset settling
        for _ in range(10):
            self._step_once()

        obs = self.get_observation()
        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.handle.env.sim.data.xquat[self.gripper_link_idx],
                self.handle.env.sim.data.xpos[self.gripper_link_idx],
            ]
        )

        info = {"task_prompt": self.handle.task_language}
        return obs, info

    def task_completed(self) -> bool:
        """VAB environments use predicate-based success."""
        return bool(self.handle.env._check_success())

    def get_completion_rate(self) -> float:
        """Return the most recent completion rate (1.0 if success, partial for packing)."""
        progress = self.handle.env.delivery_progress()
        if progress is not None:
            delivered, total = progress
            return float(delivered) / float(total) if total else 0.0
        return 1.0 if self.task_completed() else 0.0


class FrankaVabBimanualEnv(FrankaVabEnv):
    """Bimanual VAB env stub for crate_washing.

    Note: bimanual support requires a 14-DoF action space and a separate API
    (FrankaTwoArmLiftApi style). Not used for the four-task evaluation in v1
    unless the task is `libero_crate_washing`. For v1 we focus on single-arm
    tasks; this class is a placeholder.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "Bimanual VAB env wrapper not implemented yet — defer crate_washing"
        )


__all__ = ["FrankaVabEnv", "FrankaVabBimanualEnv"]
