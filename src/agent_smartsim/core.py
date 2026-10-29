"""Core data types shared by environments, agents and evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Observation:
    """What the agent is allowed to see after a reset or a step."""

    images: Dict[str, np.ndarray] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    instruction: str = ""
    t: int = 0

    def state_text(self) -> str:
        """Compact, token-cheap textual rendering of proprioception."""
        if not self.state:
            return "(no state)"
        parts = []
        for k, v in self.state.items():
            if isinstance(v, np.ndarray):
                v = np.round(v.astype(float), 4).tolist()
            elif isinstance(v, (list, tuple)):
                v = [round(float(x), 4) if isinstance(x, (int, float)) else x for x in v]
            parts.append(f"{k}={v}")
        return ", ".join(parts)


@dataclass
class StepResult:
    obs: Observation
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class EpisodeRecord:
    """One rollout. This is the unit of analysis for every metric we report."""

    task_id: str
    seed: int
    condition: str
    success: bool = False
    done_reason: str = "unknown"
    turns: int = 0
    env_steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_clock_s: float = 0.0
    failure_label: str = "unknown"
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["total_tokens"] = self.total_tokens
        return d
