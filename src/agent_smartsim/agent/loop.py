"""The agent loop.

One episode = reset -> (LLM -> tool -> env)* -> record. No hidden retries, no
silent fallbacks: every failure mode is labelled and counted, because the point
of the harness is to make agent failures *attributable*, not to hide them.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from ..core import EpisodeRecord, LLMResponse
from ..envs.base import Env
from .prompts import build_initial_user_text, build_system_prompt
from .tools import ToolBox


def _assistant_message(resp: LLMResponse) -> Dict[str, Any]:
    msg: Dict[str, Any] = {"role": "assistant", "content": resp.text or None}
    if resp.tool_calls:
        import json

        msg["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in resp.tool_calls
        ]
    return msg


def run_episode(
    env: Env,
    client: Any,
    task_id: str,
    seed: int,
    condition: str,
    annotated: bool,
    max_turns: int = 20,
    annotated_executor: bool = True,
) -> EpisodeRecord:
    from .llm import image_part  # local import keeps Pillow optional

    t0 = time.time()
    obs = env.reset(task_id, seed)
    env._last_obs = obs  # noqa: SLF001 - cached view for observe/get_state
    toolbox = ToolBox(env, annotated=annotated, annotated_executor=annotated_executor)

    rec = EpisodeRecord(task_id=task_id, seed=seed, condition=condition)

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": build_system_prompt(env, toolbox.names, annotated),
        }
    ]
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": build_initial_user_text(obs.instruction, obs.state_text())}
    ]
    for arr in obs.images.values():
        content.append(image_part(arr))
    messages.append({"role": "user", "content": content})

    done_reason = "max_turns"
    no_call_streak = 0

    for _turn in range(max_turns):
        try:
            resp = client.chat(messages, toolbox.specs)
        except Exception as exc:  # pragma: no cover - depends on remote service
            rec.error_log.append(f"model_error: {type(exc).__name__}: {exc}")
            done_reason = "agent_error"
            break

        rec.turns += 1
        rec.input_tokens += resp.input_tokens
        rec.output_tokens += resp.output_tokens
        messages.append(_assistant_message(resp))

        if not resp.tool_calls:
            no_call_streak += 1
            messages.append({"role": "user", "content": "You must call exactly one tool."})
            if no_call_streak >= 2:
                done_reason = "no_tool_call"
                break
            continue
        no_call_streak = 0

        pending_images: Dict[str, Any] = {}
        episode_over = False
        for call in resp.tool_calls:
            outcome = toolbox.dispatch(call.name, call.arguments)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": outcome.text}
            )
            pending_images.update(outcome.images)
            if outcome.done:
                episode_over = True
                done_reason = "agent_finish"

        if pending_images:
            parts: List[Dict[str, Any]] = [{"type": "text", "text": "Observation attached."}]
            for arr in pending_images.values():
                parts.append(image_part(arr))
            messages.append({"role": "user", "content": parts})

        if env.success:
            done_reason = "success"
            break
        if episode_over:
            break

    rec.success = bool(env.success)
    rec.done_reason = "success" if rec.success else done_reason
    rec.env_steps = int(getattr(env, "_t", 0))
    rec.diagnostics = env.diagnostics()
    rec.wall_clock_s = round(time.time() - t0, 3)
    return rec
