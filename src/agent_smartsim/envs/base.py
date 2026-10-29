"""Environment interface.

Everything the agent harness needs from a simulator lives here, so SMART-Sim,
LIBERO and the offline mock are interchangeable behind one contract.

Design note: the contract is deliberately *narrow*. If a platform cannot expose
these five things, it cannot host an agent experiment (see docs/binding_guide.md
for the go/no-go checklist):

    reset(task_id, seed) -> Observation      # deterministic given (task, seed)
    step(action)         -> StepResult       # external action injection, closed loop
    success              -> bool             # machine-readable task outcome
    articulation_spec()  -> dict             # optional: structured affordances
    close()                                  # clean teardown
"""
from __future__ import annotations

import abc
from typing import Any, Dict, List

from ..core import Observation, StepResult


class Env(abc.ABC):
    """Base class for every environment adapter."""

    name: str = "base"

    #: Human/agent-readable description of the action space, injected into the
    #: system prompt. Keep it short: it is paid for on every turn.
    action_space_doc: str = ""

    #: Task ids this env can run. The runner reads it to build the matrix.
    task_ids: List[str] = []

    @abc.abstractmethod
    def reset(self, task_id: str, seed: int = 0) -> Observation:
        ...

    @abc.abstractmethod
    def step(self, action: Dict[str, Any]) -> StepResult:
        ...

    @property
    @abc.abstractmethod
    def success(self) -> bool:
        ...

    def diagnostics(self) -> Dict[str, Any]:
        """Failure-attribution signals. Override with whatever the sim knows.

        Expected keys (all optional, best effort):
            contact_made: bool      -- did the end effector ever touch the part?
            grasped: bool           -- was the handle actually held?
            progress_fraction: float-- normalized task progress in [0, 1]
            last_error_class: str   -- e.g. "not_grasped", "wrong_direction"
        """
        return {}

    def articulation_spec(self) -> Dict[str, Any]:
        """Structured affordance annotations.

        Only the *annotated* condition may call this (it powers the
        `list_parts` / `get_part_spec` tools). Raise if unsupported.
        """
        raise NotImplementedError(f"{self.name}: no articulation annotations available")

    def close(self) -> None:  # pragma: no cover - trivial
        pass
