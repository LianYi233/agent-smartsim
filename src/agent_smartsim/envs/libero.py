"""LIBERO adapter (control arm).

STATUS: template. LIBERO is the widely-used pick-and-place suite (LIBERO-Spatial
/ -Object / -Goal / -90 / -Long) and LIBERO-PRO adds four perturbation axes
(objects, initial states, instructions, environments). Both are third-party
installs -- run scripts/setup_libero.sh, then implement the binding here.

Role in this repo: a *control arm*. It answers "does the agent harness behave
sensibly on a standard short-horizon grasping benchmark?" so that a negative or
positive result on SMART-Sim cannot be blamed on the harness itself.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core import Observation, StepResult
from .base import Env


def _build_default_binding(**kwargs: Any) -> Any:
    raise NotImplementedError(
        "LIBERO binding not configured.\n"
        "  1. bash scripts/setup_libero.sh   (installs robosuite/bddl/libero + datasets)\n"
        "  2. implement EnvBinding in src/agent_smartsim/envs/bindings/libero_binding.py\n"
        "     - LIBERO-PRO perturbations: swap object/initial-state/instruction/env\n"
        "  3. return it here (or pass LiberoEnv(binding=...))\n"
        "Notes: LIBERO success is a predicate, so `success()` is cheap; images come\n"
        "from an off-screen renderer -- cap resolution, it dominates token cost."
    )


class LiberoEnv(Env):
    name = "libero"
    action_space_doc = (
        "7-DoF Franka-style arm with a parallel jaw; relative end-effector "
        "translation in metres plus gripper open/close."
    )

    def __init__(self, binding: Optional[Any] = None, max_steps: int = 300, **kwargs: Any):
        self.binding = binding if binding is not None else _build_default_binding(**kwargs)
        self.max_steps = max_steps
        self.task_ids: List[str] = list(self.binding.list_task_ids())
        self._t = 0
        self._done = False
        self._last_obs: Optional[Observation] = None

    def reset(self, task_id: str, seed: int = 0) -> Observation:
        images, state, instruction = self.binding.reset(task_id, seed)
        self._t = 0
        self._done = False
        self._last_obs = Observation(images=images, state=state, instruction=instruction, t=0)
        return self._last_obs

    def step(self, action: Dict[str, Any]) -> StepResult:
        images, state, done, info = self.binding.step(action)
        self._t += 1
        self._done = bool(done) or self._t >= self.max_steps
        obs = Observation(images=images, state=state, t=self._t)
        self._last_obs = obs
        return StepResult(obs=obs, done=self._done, info=dict(info or {}))

    @property
    def success(self) -> bool:
        return bool(self.binding.success())

    def diagnostics(self) -> Dict[str, Any]:
        getter = getattr(self.binding, "diagnostics", None)
        return dict(getter()) if callable(getter) else {}

    def articulation_spec(self) -> Dict[str, Any]:
        # LIBERO has no articulation annotations: the `annotated` condition is
        # deliberately unavailable here, which keeps the control arm honest.
        return {}

    def close(self) -> None:
        closer = getattr(self.binding, "close", None)
        if callable(closer):
            closer()
