"""Tool definitions and dispatch.

Two tool sets are defined, and the difference between them *is* the paper's
independent variable:

  rgb_only   : generic primitives only. To open a part the agent must move onto
               an unknown handle, close the gripper and push in a guessed
               direction -- nothing tells it the axis, the sense of motion or
               the range.

  annotated  : identical primitives **plus** structured articulation queries
               (`list_parts`, `get_part_spec`) and, optionally, a typed
               high-level executor (`open_articulated`). This is the condition
               that tests "perception as tools" for contact-rich manipulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..core import StepResult
from ..envs.base import Env


@dataclass
class ToolOutcome:
    text: str
    done: bool = False
    images: Dict[str, np.ndarray] = field(default_factory=dict)


def _spec(name: str, description: str, properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


_SPECS: Dict[str, Dict[str, Any]] = {
    "observe": _spec(
        "observe",
        "Fetch the current camera views (third-person and wrist).",
        {},
        [],
    ),
    "get_state": _spec(
        "get_state",
        "Read proprioception: end-effector position and gripper state.",
        {},
        [],
    ),
    "move_to": _spec(
        "move_to",
        "Translate the end effector by a relative offset in metres (bounded to 0.25 per call).",
        {
            "dx": {"type": "number", "description": "metres along world x"},
            "dy": {"type": "number", "description": "metres along world y"},
            "dz": {"type": "number", "description": "metres along world z"},
        },
        ["dx", "dy", "dz"],
    ),
    "set_gripper": _spec(
        "set_gripper",
        "Close (true) or open (false) the parallel-jaw gripper.",
        {"closed": {"type": "boolean"}},
        ["closed"],
    ),
    "finish": _spec(
        "finish",
        "Declare the episode over. Call this when the subtask is done or you give up.",
        {"reason": {"type": "string"}},
        ["reason"],
    ),
    "list_parts": _spec(
        "list_parts",
        "List the manipulable articulated parts of the object and their types.",
        {},
        [],
    ),
    "get_part_spec": _spec(
        "get_part_spec",
        "Return structured annotations for a part: grasp point, rotation/translation "
        "axis, direction, motion range and target fraction.",
        {"part_id": {"type": "string"}},
        ["part_id"],
    ),
    "open_articulated": _spec(
        "open_articulated",
        "Execute a scripted open motion on a part, using its annotations. "
        "Requires the handle to be grasped.",
        {
            "part_id": {"type": "string"},
            "target_fraction": {"type": "number", "description": "0..1 of full range"},
        },
        ["part_id", "target_fraction"],
    ),
}


class ToolBox:
    """Binds tool names to env actions for one episode."""

    def __init__(self, env: Env, annotated: bool = False, annotated_executor: bool = True):
        self.env = env
        self.annotated = annotated
        self.annotated_executor = annotated_executor

    # ------------------------------------------------------------------ specs
    @property
    def names(self) -> List[str]:
        n = ["observe", "get_state", "move_to", "set_gripper", "finish"]
        if self.annotated:
            n += ["list_parts", "get_part_spec"]
            if self.annotated_executor:
                n.append("open_articulated")
        return n

    @property
    def specs(self) -> List[Dict[str, Any]]:
        return [_SPECS[n] for n in self.names]

    # --------------------------------------------------------------- dispatch
    def dispatch(self, name: str, args: Dict[str, Any]) -> ToolOutcome:
        if name not in self.names:
            return ToolOutcome(f"error: tool {name!r} is not available in this condition")
        handler: Optional[Callable[[Dict[str, Any]], ToolOutcome]] = getattr(
            self, f"_t_{name}", None
        )
        if handler is None:
            return ToolOutcome(f"error: tool {name!r} is declared but unimplemented")
        try:
            return handler(args or {})
        except Exception as exc:  # never let one bad call kill the loop
            return ToolOutcome(f"error: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ tools
    def _t_observe(self, _args: Dict[str, Any]) -> ToolOutcome:
        obs = self.env._last_obs if hasattr(self.env, "_last_obs") else None
        images = obs.images if obs is not None else {}
        views = ", ".join(images) or "none"
        return ToolOutcome(f"attached views: {views}", images=dict(images))

    def _t_get_state(self, _args: Dict[str, Any]) -> ToolOutcome:
        obs = getattr(self.env, "_last_obs", None)
        return ToolOutcome(obs.state_text() if obs is not None else "(unavailable)")

    def _t_move_to(self, args: Dict[str, Any]) -> ToolOutcome:
        res = self._act(
            {"type": "move_to", "delta": [float(args["dx"]), float(args["dy"]), float(args["dz"])]}
        )
        return self._describe(res)

    def _t_set_gripper(self, args: Dict[str, Any]) -> ToolOutcome:
        res = self._act({"type": "set_gripper", "closed": bool(args["closed"])})
        return self._describe(res)

    def _t_finish(self, args: Dict[str, Any]) -> ToolOutcome:
        reason = str(args.get("reason", "agent_finish"))
        return ToolOutcome(f"episode finished: {reason}", done=True)

    def _t_list_parts(self, _args: Dict[str, Any]) -> ToolOutcome:
        spec = self.env.articulation_spec()
        return ToolOutcome(
            f"1 part: id={spec['part_id']} type={spec['part_type']} "
            f"(use get_part_spec for geometry)"
        )

    def _t_get_part_spec(self, args: Dict[str, Any]) -> ToolOutcome:
        spec = self.env.articulation_spec()
        if str(args.get("part_id", spec["part_id"])) != str(spec["part_id"]):
            return ToolOutcome(f"error: unknown part {args.get('part_id')!r}")
        keys = [
            "handle_point",
            "axis",
            "direction",
            "range_deg",
            "range_m",
            "push_direction",
            "target_fraction",
        ]
        body = ", ".join(f"{k}={spec.get(k)}" for k in keys)
        return ToolOutcome(f"part {spec['part_id']} ({spec['part_type']}): {body}")

    def _t_open_articulated(self, args: Dict[str, Any]) -> ToolOutcome:
        res = self._act(
            {
                "type": "open_articulated",
                "part_id": args.get("part_id", "part_0"),
                "target_fraction": float(args.get("target_fraction", 1.0)),
            }
        )
        return self._describe(res)

    # ------------------------------------------------------------------ utils
    def _act(self, action: Dict[str, Any]) -> StepResult:
        res = self.env.step(action)
        self.env._last_obs = res.obs  # cache for observe/get_state
        return res

    def _describe(self, res: StepResult) -> ToolOutcome:
        diag = res.info or {}
        prog = diag.get("progress_fraction")
        err = diag.get("last_error_class")
        bits = [f"state: {res.obs.state_text()}"]
        if prog is not None:
            bits.append(f"progress={prog}")
        if err:
            bits.append(f"last_error={err}")
        if res.done:
            bits.append("episode_done")
        return ToolOutcome("; ".join(bits), done=bool(res.done))
