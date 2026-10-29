"""SMART-Sim adapter.

STATUS: template. The upstream SMART-Sim API is not vendored here (see
docs/binding_guide.md for the 6 checks and the exact functions to implement).
Fill in `_build_default_binding()` -- or pass your own `binding=` object -- and
this adapter turns SMART-Sim into an agent-controllable environment.

Why this adapter exists at all: SMART-Sim's articulation annotations (grasp
point, motion axis, direction, range) are exactly the structured affordances
that the `annotated` condition exposes as tools. It is the main environment of
this repo; LIBERO is the control arm.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core import Observation, StepResult
from .base import Env


def _build_default_binding(**kwargs: Any) -> Any:
    """Return your binding object. See envs/binding.py for the contract."""
    raise NotImplementedError(
        "SMART-Sim binding not configured.\n"
        "  1. read docs/binding_guide.md\n"
        "  2. implement EnvBinding in src/agent_smartsim/envs/bindings/smartsim_binding.py\n"
        "  3. return it from this function (or pass SmartSimEnv(binding=...))\n"
        "Until then use the offline mock: python -m agent_smartsim.cli smoke"
    )


class SmartSimEnv(Env):
    name = "smartsim"
    action_space_doc = (
        "7-DoF arm with a parallel jaw. Actions: relative end-effector translation "
        "(metres), gripper open/close, and (annotated condition only) a typed "
        "open_articulated primitive driven by part annotations."
    )

    def __init__(self, binding: Optional[Any] = None, max_steps: int = 200, **kwargs: Any):
        self.binding = binding if binding is not None else _build_default_binding(**kwargs)
        self.max_steps = max_steps
        self.task_ids: List[str] = list(self.binding.list_task_ids())
        self._t = 0
        self._done = False
        self._last_obs: Optional[Observation] = None

    # ------------------------------------------------------------------ Env
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
        return dict(self.binding.articulation())

    def close(self) -> None:
        closer = getattr(self.binding, "close", None)
        if callable(closer):
            closer()
