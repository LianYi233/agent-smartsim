"""Adapter binding contract.

A platform is usable for agent experiments if it can answer these five
questions. Implement them once (usually 50-150 lines of glue) and the whole
harness -- tools, metrics, attribution, reporting -- works unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple


class EnvBinding(Protocol):
    """What you must implement to plug in a new simulator."""

    def list_task_ids(self) -> List[str]:
        """Stable, human-readable task ids (used for the experiment matrix)."""

    def reset(self, task_id: str, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        """Return (images, state, instruction). Must be deterministic given (task_id, seed)."""

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], bool, Dict[str, Any]]:
        """Apply ONE action, return (images, state, done, info). Closed loop."""

    def success(self) -> bool:
        """Machine-readable task outcome for the current episode."""

    def articulation(self) -> Dict[str, Any]:
        """Structured affordances: handle point, axis, direction, range, target.

        Return an empty dict if the platform has no annotations -- then the
        `annotated` condition is simply unavailable (that is itself a result).
        """
