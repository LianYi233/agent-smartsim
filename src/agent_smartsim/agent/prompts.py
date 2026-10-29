"""Prompt construction.

Kept deliberately short: every token here is paid on every turn, and the task
is a control problem, not a chat. The only condition-dependent difference is
the tool inventory (which the harness reports verbatim) plus one paragraph
telling the agent what structured information it may or may not ask for.
"""
from __future__ import annotations

from typing import List

from ..envs.base import Env

_COMMON = """You are the high-level controller of a robot arm operating an articulated object.

Rules:
- Act only through the provided tools; one tool call at a time unless told otherwise.
- Coordinates are metres in the world frame; the gripper is a parallel jaw.
- You cannot see the object's kinematic structure unless a tool tells you about it.
- Verify that a grasp holds before you pull or turn. If a motion is rejected or
  the progress does not increase, change strategy instead of repeating it.
- Call `finish` when the instruction is satisfied, and say why in `reason`.
"""

_ANNOTATED_NOTE = """Structured annotations ARE available for this object: call
`list_parts` to enumerate parts, then `get_part_spec` for the grasp point, the
motion axis, the sense of motion and the range. Prefer these over guessing, and
do not issue a scripted open motion before the handle is grasped.
"""

_RGB_ONLY_NOTE = """No annotations are available: you must infer the handle
location and the direction of motion from the images and from what happens when
you push. Expect wasted actions; recover from them.
"""


def build_system_prompt(env: Env, tool_names: List[str], annotated: bool) -> str:
    parts = [
        _COMMON,
        f"Environment: {env.name}",
        f"Action space: {env.action_space_doc or 'see tool descriptions'}",
        "Available tools: " + ", ".join(tool_names),
        _ANNOTATED_NOTE if annotated else _RGB_ONLY_NOTE,
    ]
    return "\n".join(parts)


def build_initial_user_text(instruction: str, state_text: str) -> str:
    return (
        f"Instruction: {instruction}\n"
        f"Current proprioception: {state_text}\n"
        "Begin. Use the tools to complete the instruction."
    )
