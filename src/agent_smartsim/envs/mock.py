"""Offline mock environment.

Purpose: let the whole harness (loop, tools, metrics, reporting) run and be
tested with **no simulator, no GPU and no API key**. It is a crude but
deterministic stand-in for an articulated object:

  * a handle sits somewhere in front of the arm,
  * the arm must move onto it, close the gripper, then drive the part open,
  * two ways to make progress, mirroring the ablation:
      - `set_gripper` + repeated `move_to` along an unknown push direction
        (the rgb_only condition: no axis/range knowledge, so it guesses),
      - `open_articulated(part_id, target_fraction)` which uses the
        annotation directly (the annotated condition).

It is NOT a physics simulator and must never be reported as one. Its only job
is to make the code path executable end to end.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..core import Observation, StepResult
from .base import Env

_CONTACT_RADIUS = 0.06
_BOUNDS = (np.array([0.10, -0.40, 0.10]), np.array([0.80, 0.40, 1.20]))


class MockArticulatedEnv(Env):
    name = "mock"
    action_space_doc = (
        "End-effector control in metres (world frame). move_to applies a bounded "
        "relative translation; set_gripper opens/closes the parallel jaw."
    )

    _TASKS: Dict[str, Dict[str, Any]] = {
        "door_hinge_01": dict(
            part_type="hinge",
            handle=(0.45, 0.10, 0.80),
            axis=(0.0, 0.0, 1.0),
            direction=1,
            range_deg=90.0,
            push_dir=(0.0, -1.0, 0.0),
        ),
        "drawer_slide_01": dict(
            part_type="slider",
            handle=(0.45, 0.10, 0.60),
            axis=(1.0, 0.0, 0.0),
            direction=1,
            range_deg=0.0,
            range_m=0.25,
            push_dir=(1.0, 0.0, 0.0),
        ),
        "knob_01": dict(
            part_type="knob",
            handle=(0.50, 0.00, 0.70),
            axis=(0.0, 0.0, 1.0),
            direction=-1,
            range_deg=120.0,
            push_dir=(0.0, 1.0, 0.0),
        ),
    }

    def __init__(self, max_steps: int = 40, image_size: int = 64, noise: float = 0.02):
        self.max_steps = max_steps
        self.image_size = image_size
        self.noise = noise
        self.task_ids: List[str] = list(self._TASKS)
        self._task: Dict[str, Any] = {}
        self._task_id = ""
        self._ee = np.zeros(3)
        self._gripper_closed = False
        self._contact = False
        self._grasped = False
        self._progress = 0.0
        self._t = 0
        self._done = False
        self._last_error = ""
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ reset
    def reset(self, task_id: str, seed: int = 0) -> Observation:
        if task_id not in self._TASKS:
            raise KeyError(f"unknown mock task {task_id!r}; have {self.task_ids}")
        self._rng = np.random.default_rng(seed)
        self._task = dict(self._TASKS[task_id])
        self._task_id = task_id
        handle = np.array(self._task["handle"], dtype=float)
        if self.noise:
            handle = handle + self._rng.normal(0.0, self.noise, 3)
        self._task["handle"] = tuple(handle.tolist())
        self._ee = np.array([0.30, 0.0, 0.50])
        self._gripper_closed = False
        self._contact = False
        self._grasped = False
        self._progress = 0.0
        self._t = 0
        self._done = False
        self._last_error = ""
        return self._obs()

    # ------------------------------------------------------------------- step
    def step(self, action: Dict[str, Any]) -> StepResult:
        if self._done:
            return StepResult(self._obs(), True, {"warning": "episode already done"})

        kind = action.get("type")
        self._t += 1
        self._last_error = ""

        if kind == "move_to":
            delta = np.array(action.get("delta", [0, 0, 0]), dtype=float)
            delta = np.clip(delta, -0.25, 0.25)
            self._ee = np.clip(self._ee + delta, *_BOUNDS)
            self._update_contact()
            if self._grasped:
                self._progress_via_motion(delta)
        elif kind == "set_gripper":
            self._gripper_closed = bool(action.get("closed", True))
            if self._gripper_closed:
                self._update_contact()
                if self._contact:
                    self._grasped = True
            else:
                self._grasped = False
        elif kind == "open_articulated":
            self._open_articulated(action)
        else:
            self._last_error = "bad_action"
            return StepResult(
                self._obs(), False, {"error": f"unknown action type {kind!r}"}
            )

        self._done = self._progress >= 0.9 or self._t >= self.max_steps
        return StepResult(self._obs(), self._done, self.diagnostics())

    def _update_contact(self) -> None:
        handle = np.array(self._task["handle"], dtype=float)
        self._contact = bool(np.linalg.norm(self._ee - handle) < _CONTACT_RADIUS)

    def _progress_via_motion(self, delta: float) -> None:
        """rgb_only path: only works if the push direction happens to be right."""
        push = np.array(self._task["push_dir"], dtype=float)
        norm = np.linalg.norm(delta)
        if norm < 1e-6:
            return
        if float(np.dot(delta / norm, push)) > 0.85:
            self._progress = min(1.0, self._progress + 0.12)
        else:
            self._last_error = "wrong_direction"

    def _open_articulated(self, action: Dict[str, Any]) -> None:
        """annotated path: typed primitive driven by the annotation."""
        if not self._grasped:
            self._last_error = "not_grasped"
            return
        target = float(action.get("target_fraction", 1.0))
        step = 0.34 * float(np.sign(target - self._progress) or 1.0)
        self._progress = float(np.clip(self._progress + step, 0.0, 1.0))

    # -------------------------------------------------------------- accessors
    @property
    def success(self) -> bool:
        return self._progress >= 0.9

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "contact_made": self._contact,
            "grasped": self._grasped,
            "progress_fraction": round(self._progress, 3),
            "last_error_class": self._last_error,
        }

    def articulation_spec(self) -> Dict[str, Any]:
        t = self._task
        return {
            "part_id": "part_0",
            "part_type": t["part_type"],
            "handle_point": [round(x, 4) for x in t["handle"]],
            "axis": list(t["axis"]),
            "direction": t["direction"],
            "range_deg": t.get("range_deg", 0.0),
            "range_m": t.get("range_m", 0.0),
            "push_direction": list(t["push_dir"]),
            "target_fraction": 0.9,
        }

    # ------------------------------------------------------------------ obs
    def _obs(self) -> Observation:
        return Observation(
            images={"third_view": self._render(), "wrist_view": self._render(offset=1)},
            state={
                "ee_pos": self._ee,
                "gripper_closed": self._gripper_closed,
                "t": self._t,
            },
            instruction=f"open the {self._task.get('part_type', 'part')} by 90%",
            t=self._t,
        )

    def _render(self, offset: int = 0):
        """A crude but deterministic view: bright blob at the projected handle."""
        n = self.image_size
        img = np.zeros((n, n, 3), dtype=np.uint8)
        img[:, :, 2] = 40
        handle = np.array(self._task["handle"], dtype=float)
        cx = int(np.clip((handle[0] / 0.8) * (n - 1), 0, n - 1))
        cy = int(np.clip((1.0 - handle[2] / 1.2) * (n - 1), 0, n - 1))
        r = max(2, n // 16)
        y0, y1 = max(0, cy - r), min(n, cy + r)
        x0, x1 = max(0, cx - r), min(n, cx + r)
        img[y0:y1, x0:x1, 1] = 200 - 60 * offset
        return img
