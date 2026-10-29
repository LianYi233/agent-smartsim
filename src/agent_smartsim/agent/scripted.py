"""Scripted reference policy (no API calls).

It exists for three reasons:
  1. `cli smoke` must run offline, on any machine, with no key and no GPU;
  2. it gives a *lower bound* for what a competent agent should do on the mock;
  3. CI can assert that the harness wiring works without burning tokens.

It is NOT a baseline to report. For numbers, use `cli run` with a real client.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..core import LLMResponse, ToolCall

#: reach offsets (metres) that land the gripper on each mock handle
_REACH: Dict[str, Tuple[float, float, float]] = {
    "hinge": (0.15, 0.10, 0.30),
    "slider": (0.15, 0.10, 0.10),
    "knob": (0.20, 0.00, 0.20),
}

#: per-call push offsets that move each part in its own actuation direction
_PUSH: Dict[str, Tuple[float, float, float]] = {
    "hinge": (0.0, -0.08, 0.0),
    "slider": (0.08, 0.0, 0.0),
    "knob": (0.0, 0.08, 0.0),
}


def _neg(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (-v[0], -v[1], -v[2])


def _kind_from_messages(messages: List[Dict[str, Any]]) -> str:
    """The mock states the part type in the instruction; read it like an agent would."""
    for msg in messages:
        content = msg.get("content")
        texts: List[str] = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts += [p.get("text", "") for p in content if p.get("type") == "text"]
        for text in texts:
            low = text.lower()
            for kind in _REACH:
                if kind in low:
                    return kind
    return "hinge"


def _plan(kind: str, annotated: bool) -> List[Tuple[str, Dict[str, Any]]]:
    rx, ry, rz = _REACH[kind]
    px, py, pz = _PUSH[kind]
    reach = ("move_to", {"dx": rx, "dy": ry, "dz": rz})

    if annotated:
        return [
            ("list_parts", {}),
            ("get_part_spec", {"part_id": "part_0"}),
            reach,
            ("set_gripper", {"closed": True}),
            ("open_articulated", {"part_id": "part_0", "target_fraction": 1.0}),
            ("open_articulated", {"part_id": "part_0", "target_fraction": 1.0}),
            ("open_articulated", {"part_id": "part_0", "target_fraction": 1.0}),
            ("finish", {"reason": "part opened to full range"}),
        ]

    wx, wy, wz = _neg(_PUSH[kind])
    plan: List[Tuple[str, Dict[str, Any]]] = [
        ("observe", {}),
        ("get_state", {}),
        reach,
        ("set_gripper", {"closed": True}),
        # one wrong guess first: that is what missing the axis annotation costs
        ("move_to", {"dx": wx, "dy": wy, "dz": wz}),
    ]
    plan += [("move_to", {"dx": px, "dy": py, "dz": pz})] * 8
    plan.append(("finish", {"reason": "pushed along the inferred arc"}))
    return plan


class ScriptedArticulatedClient:
    """Replays a fixed plan, adapting to the tool set and the instruction."""

    def __init__(self) -> None:
        self._turns = 0
        self._plan: List[Tuple[str, Dict[str, Any]]] = []

    def chat(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> LLMResponse:
        if not self._plan:
            names = {t["function"]["name"] for t in tools}
            self._plan = _plan(_kind_from_messages(messages), "open_articulated" in names)
        name, args = self._plan[min(self._turns, len(self._plan) - 1)]
        self._turns += 1
        return LLMResponse(
            text="",
            tool_calls=[ToolCall(id=f"call_{self._turns}", name=name, arguments=dict(args))],
            input_tokens=0,
            output_tokens=0,
        )

    def reset(self) -> None:
        self._turns = 0
        self._plan = []
