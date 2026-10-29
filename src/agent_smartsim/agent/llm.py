"""OpenAI chat client with tool calling, retries and token accounting.

`openai` is imported lazily so the offline paths (mock env + scripted policy +
unit tests) work with zero API dependencies installed.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..core import LLMResponse, ToolCall


def encode_image_png(arr: np.ndarray) -> str:
    """numpy HxWx3 uint8 -> data URL. Pillow is soft-required."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required to attach images; pip install Pillow") from exc
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def image_part(arr: np.ndarray) -> Dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": encode_image_png(arr)}}


class OpenAIChatClient:
    """Thin wrapper so the loop only depends on `.chat(messages, tools)`."""

    def __init__(
        self,
        model: str = "gpt-5.6",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        timeout: float = 120.0,
        max_output_tokens: Optional[int] = 2048,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai is not installed. `pip install 'agent-smartsim[api]'` "
                "or run the offline smoke test instead (`python -m agent_smartsim.cli smoke`)."
            ) from exc

        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
            timeout=timeout,
        )

    # ------------------------------------------------------------------- chat
    def chat(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> LLMResponse:
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self.max_output_tokens:
            kwargs["max_completion_tokens"] = self.max_output_tokens

        last: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return self._parse(resp)
            except Exception as exc:  # network, rate limit, transient 5xx
                last = exc
                if attempt + 1 == self.max_retries:
                    break
                time.sleep(2 ** attempt)
        raise RuntimeError(f"model call failed after {self.max_retries} attempts: {last}")

    def _parse(self, resp: Any) -> LLMResponse:
        choice = resp.choices[0].message
        calls: List[ToolCall] = []
        for tc in (getattr(choice, "tool_calls", None) or []):
            raw = tc.function.arguments or "{}"
            try:
                args = json.loads(raw)
                if not isinstance(args, dict):
                    args = {"_raw": args}
            except json.JSONDecodeError:
                args = {"_parse_error": raw}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=choice.content or "",
            tool_calls=calls,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )
