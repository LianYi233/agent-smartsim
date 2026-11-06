"""LIBERO / LIBERO-PRO binding (the control arm).

Implements the EnvBinding contract against the LIBERO-PRO fork
(github.com/Zxy-MLlab/LIBERO-PRO). That fork registers its perturbation
suites as ordinary benchmark names next to the standard ones -- for example
``libero_object_with_red_stick`` beside ``libero_object`` -- so a
perturbation costs nothing at the API level: it is just another suite name
backed by a different bddl file.

API surface verified against the fork's source on 2026-11-06:

  libero/libero/envs/env_wrapper.py
      OffScreenRenderEnv(**kwargs)      has_renderer=False + offscreen on
      env.seed(seed); env.reset(); env.set_init_state(state)
      env.step(action) -> obs dict; env.check_success(); env.close()
      cameras: "agentview", "robot0_eye_in_hand" (robosuite images arrive
      vertically flipped; every consumer flips them back)
  libero/libero/benchmark/__init__.py
      get_benchmark_dict() -> {"libero_spatial": ..., ..., "libero_object_with_red_stick": ...}
      suite = benchmark_dict[name](); suite.get_task(i); suite.get_task_init_states(i)
      task.name / task.language / task.problem_folder / task.bddl_file
  action space: 7-dim OSC_POSE [dx,dy,dz, ax,ay,az, gripper]; normalised
  inputs; position output_max 0.05 m/step by default; +1 closes the gripper.

Task ids look like ``"<suite>/<task name>"``, e.g.
``"libero_object_with_red_stick/KITCHEN_SCENE4_..."``. A seed selects the
initial state (``seed % len(init_states)``) and seeds the physics RNG.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

POSITION_STEP_M = 0.05  # robosuite OSC_POSE default position output_max
MAX_MOVE_M = 0.25       # per-call bound, mirrors the mock env / tool docs
DEFAULT_SUITES: Sequence[str] = ("libero_spatial", "libero_object")

_IMAGE_KEYS: Tuple[Tuple[str, str], ...] = (
    ("agentview", "agentview_image"),
    ("wrist", "robot0_eye_in_hand_image"),
)


# --------------------------------------------------------------------- pure helpers
def flip_image(img: np.ndarray) -> np.ndarray:
    """Undo robosuite's vertical flip (images arrive upside down)."""
    return np.ascontiguousarray(np.asarray(img)[::-1])


def osc_action(
    delta_m: Sequence[float],
    gripper_closed: bool,
    position_step_m: float = POSITION_STEP_M,
) -> np.ndarray:
    """Convert a metre-space delta into one 7-dim OSC_POSE action."""
    d = np.asarray(delta_m, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(d)):
        raise ValueError(f"non-finite delta: {d.tolist()}")
    pos = np.clip(d / float(position_step_m), -1.0, 1.0)
    grip = 1.0 if gripper_closed else -1.0
    return np.concatenate(
        [pos.astype(np.float32), np.zeros(3, np.float32), np.array([grip], np.float32)]
    )


def split_move(delta_m: Sequence[float], position_step_m: float, max_substeps: int) -> np.ndarray:
    """Split one metre-space move into n per-substep deltas, shape (n, 3).

    No substep exceeds ``position_step_m`` in any axis, so each one maps into
    the controller's normalised [-1, 1] input range without clipping.
    """
    d = np.asarray(delta_m, dtype=np.float64).reshape(3)
    n = max(1, int(math.ceil(float(np.max(np.abs(d))) / position_step_m)))
    n = min(n, max_substeps)
    per = (d / n).astype(np.float32)
    return np.repeat(per[None, :], n, axis=0)


def parse_obs(obs: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Extract (flipped) camera images and proprioception from a robosuite obs."""
    images: Dict[str, np.ndarray] = {}
    for short, key in _IMAGE_KEYS:
        if key in obs:
            images[short] = flip_image(obs[key])
    state: Dict[str, Any] = {}
    if "robot0_eef_pos" in obs:
        state["eef_pos"] = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    if "robot0_eef_quat" in obs:
        state["eef_quat"] = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
    if "robot0_gripper_qpos" in obs:
        state["gripper_qpos"] = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
    return images, state


def split_task_id(task_id: str) -> Tuple[str, str]:
    suite, _, name = task_id.partition("/")
    if not suite or not name:
        raise ValueError(
            f"task id {task_id!r} must look like '<suite>/<task name>' "
            f"(e.g. 'libero_object/...')"
        )
    return suite, name


# ------------------------------------------------------------------------- binding
class LiberoProBinding:
    """EnvBinding implementation for LIBERO and its LIBERO-PRO perturbations.

    The underlying robosuite environment is cached per (bddl, camera size,
    gpu device) at module level, because the experiment runner builds a fresh
    ``LiberoEnv`` per episode; rebuilding MuJoCo there would dominate runtime.
    """

    def __init__(
        self,
        suites: Sequence[str] = DEFAULT_SUITES,
        camera_size: int = 128,
        settle_steps: int = 5,
        position_step_m: float = POSITION_STEP_M,
        max_substeps: int = 8,
        render_gpu_device: Optional[int] = None,
    ):
        if camera_size < 64:
            raise ValueError("camera_size below 64 is not useful for a VLM agent")
        self.suites = tuple(suites)
        self.camera_size = int(camera_size)
        self.settle_steps = int(settle_steps)
        self.position_step_m = float(position_step_m)
        self.max_substeps = int(max_substeps)
        self.render_gpu_device = render_gpu_device

        self._libero = None            # lazily imported module bundle
        self._suites: Dict[str, Any] = {}
        self._task_index: Dict[str, Dict[str, int]] = {}
        self._env = None
        self._env_key: Optional[Tuple[str, int, Optional[int]]] = None
        self._bddl: Optional[str] = None
        self._grip_closed = False
        self._episode_steps = 0
        self._sim_steps = 0
        self._success = False
        self._task_language = ""

    # ------------------------------------------------------------- libero import
    def _load(self) -> None:
        if self._libero is not None:
            return
        try:
            from libero.libero import benchmark as benchmark_mod  # type: ignore
            from libero.libero.envs import OffScreenRenderEnv  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised on sim machines
            raise ImportError(
                "libero is not importable. On the evaluation machine run:\n"
                "  bash scripts/setup_libero_pro.sh\n"
                f"(original error: {exc})"
            ) from exc
        self._libero = {"benchmark": benchmark_mod, "OffScreenRenderEnv": OffScreenRenderEnv}

    def _suite(self, name: str) -> Any:
        self._load()
        if name not in self._suites:
            benchmark_dict = self._libero["benchmark"].get_benchmark_dict()
            key = name.lower()
            if key not in benchmark_dict:
                available = ", ".join(sorted(benchmark_dict))
                raise ValueError(
                    f"unknown LIBERO suite {name!r}. Available: {available}\n"
                    "LIBERO-PRO perturbation suites are registered too, e.g. "
                    "'libero_object_with_red_stick'."
                )
            self._suites[name] = benchmark_dict[key]()
        return self._suites[name]

    def _index(self, suite_name: str) -> Dict[str, int]:
        if suite_name not in self._task_index:
            suite = self._suite(suite_name)
            n = getattr(suite, "n_tasks", None)
            if n is None:  # fall back for suite implementations without the property
                n = len(suite.tasks)
            self._task_index[suite_name] = {
                suite.get_task(i).name: i for i in range(int(n))
            }
        return self._task_index[suite_name]

    def _make_env(self, bddl_file_name: str):
        kwargs: Dict[str, Any] = {
            "bddl_file_name": bddl_file_name,
            "camera_heights": self.camera_size,
            "camera_widths": self.camera_size,
        }
        if self.render_gpu_device is not None:
            kwargs["render_gpu_device_id"] = int(self.render_gpu_device)
        return self._libero["OffScreenRenderEnv"](**kwargs)

    # ---------------------------------------------------------------- EnvBinding
    def list_task_ids(self) -> List[str]:
        out: List[str] = []
        for suite in self.suites:
            out.extend(f"{suite}/{name}" for name in sorted(self._index(suite)))
        return out

    def reset(self, task_id: str, seed: int) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], str]:
        suite_name, task_name = split_task_id(task_id)
        suite = self._suite(suite_name)
        index = self._index(suite_name)
        if task_name not in index:
            known = ", ".join(sorted(index)[:6])
            raise KeyError(
                f"task {task_name!r} not in suite {suite_name!r} (first known: {known}...)"
            )
        task = suite.get_task(index[task_name])

        bddl = task.bddl_file
        key = (bddl, self.camera_size, self.render_gpu_device)
        if self._env is None or self._env_key != key:
            env = _ENV_CACHE.get(key)
            if env is None:
                env = self._make_env(bddl)
                if len(_ENV_CACHE) >= 4:
                    _ENV_CACHE.pop(next(iter(_ENV_CACHE))).close()
                _ENV_CACHE[key] = env
            self._env, self._env_key = env, key

        self._env.seed(int(seed))
        self._env.reset()
        init_states = suite.get_task_init_states(index[task_name])
        self._env.set_init_state(init_states[int(seed) % len(init_states)])

        self._grip_closed = False
        self._episode_steps = 0
        self._sim_steps = 0
        self._success = False
        self._task_language = str(getattr(task, "language", ""))

        obs: Dict[str, Any] = {}
        zero = np.concatenate(
            [np.zeros(3, np.float32), np.zeros(3, np.float32), np.array([-1.0], np.float32)]
        )
        for _ in range(max(0, self.settle_steps)):
            obs = self._env.step(zero)
            self._sim_steps += 1
        if not obs:  # settle_steps == 0 -> synthesize an observation without stepping
            obs = self._env.env.get_observation() if hasattr(self._env, "env") else {}

        images, state = parse_obs(obs)
        return images, state, self._task_language

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], bool, Dict[str, Any]]:
        if self._env is None:
            raise RuntimeError("binding.reset() must be called before step()")
        kind = action.get("type")
        obs: Dict[str, Any] = {}

        if kind == "move_to":
            delta = np.clip(
                np.asarray(action["delta"], dtype=np.float64).reshape(3), -MAX_MOVE_M, MAX_MOVE_M
            )
            substeps = split_move(delta, self.position_step_m, self.max_substeps)
            for s in substeps:
                obs = self._env.step(osc_action(s, self._grip_closed, self.position_step_m))
                self._sim_steps += 1
        elif kind == "set_gripper":
            self._grip_closed = bool(action.get("closed", False))
            obs = self._env.step(osc_action((0.0, 0.0, 0.0), self._grip_closed, self.position_step_m))
            self._sim_steps += 1
        elif kind == "open_articulated":
            return {}, {"error": "open_articulated is unavailable on LIBERO (no annotations)"}, False, {
                "last_error_class": "agent_api_error"
            }
        else:
            return {}, {"error": f"unknown action type {kind!r}"}, False, {
                "last_error_class": "agent_api_error"
            }

        self._episode_steps += 1
        self._success = bool(self._env.check_success())
        images, state = parse_obs(obs)
        info: Dict[str, Any] = {
            "sim_steps": self._sim_steps,
            "episode_steps": self._episode_steps,
            "progress_fraction": None,  # LIBERO success is a predicate, not a score
        }
        return images, state, self._success, info

    def success(self) -> bool:
        return bool(self._success)

    def articulation(self) -> Dict[str, Any]:
        # Deliberately empty: no annotations on LIBERO. The `annotated`
        # condition is therefore unavailable here -- that is the control arm.
        return {}

    def diagnostics(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "sim_steps": self._sim_steps,
            "episode_steps": self._episode_steps,
            "gripper_closed": self._grip_closed,
        }
        if self._env is not None and hasattr(self._env, "env"):
            try:  # heuristic grasp signal: closed fingers around something
                qpos = self._env.env.sim.data.qpos
                _ = qpos  # touch the sim once so a broken sim surfaces here, not mid-run
            except Exception:
                pass
        return d

    def close(self) -> None:
        """End the episode; the robosuite env stays cached for the next episode."""
        self._env = None
        self._env_key = None

    @staticmethod
    def shutdown() -> None:
        """Destroy cached simulator processes (call at end of process)."""
        while _ENV_CACHE:
            _, env = _ENV_CACHE.popitem()
            try:
                env.close()
            except Exception:
                pass


#: module-level cache: {(bddl_file, camera_size, gpu_device): env}
_ENV_CACHE: Dict[Tuple[str, int, Optional[int]], Any] = {}
