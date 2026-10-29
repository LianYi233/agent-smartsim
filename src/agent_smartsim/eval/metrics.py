"""Metrics and failure attribution.

The four headline numbers are copied from the embodied-agent literature so that
results are comparable on sight:

    success            task-level outcome
    turns / env_steps  action cost
    input/output token model cost
    wall_clock_s       latency cost

On top of that we attribute *why* a rollout failed, which is the only way to
tell "the agent cannot plan" apart from "the simulator is noisy".
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List

from ..core import EpisodeRecord

_TOKEN_COLUMNS = ("input_tokens", "output_tokens", "total_tokens")


def attribute_failure(rec: EpisodeRecord) -> str:
    """Rule-based label; deterministic and cheap. Extend as sims expose more."""
    if rec.success:
        return "success"
    if rec.done_reason == "agent_error":
        return "agent_api_error"
    if rec.done_reason == "no_tool_call":
        return "no_tool_call"
    diag = rec.diagnostics or {}
    if diag.get("last_error_class") == "not_grasped":
        return "grasp_failed"
    if not diag.get("contact_made", True):
        return "no_contact"
    if not diag.get("grasped", True):
        return "grasp_failed"
    if rec.done_reason == "agent_finish" and float(diag.get("progress_fraction", 0)) < 0.9:
        return "premature_finish"
    if float(diag.get("progress_fraction", 0)) > 0.0:
        return "partial_articulation"
    if rec.done_reason == "max_turns":
        return "budget_exhausted"
    return "unknown"


def aggregate(records: Iterable[EpisodeRecord]) -> List[Dict[str, Any]]:
    """Per-condition summary rows, in a stable column order."""
    buckets: Dict[str, List[EpisodeRecord]] = defaultdict(list)
    for rec in records:
        buckets[rec.condition].append(rec)

    rows: List[Dict[str, Any]] = []
    for cond in sorted(buckets):
        rs = buckets[cond]
        n = len(rs)
        succ = sum(1 for r in rs if r.success)
        fails = Counter(attribute_failure(r) for r in rs if not r.success)
        tot_tok = sum(r.total_tokens for r in rs)
        rows.append(
            {
                "condition": cond,
                "n": n,
                "success_rate": round(succ / n, 4) if n else 0.0,
                "mean_turns": round(sum(r.turns for r in rs) / n, 3) if n else 0.0,
                "mean_env_steps": round(sum(r.env_steps for r in rs) / n, 3) if n else 0.0,
                "mean_input_tokens": round(sum(r.input_tokens for r in rs) / n, 1) if n else 0.0,
                "mean_output_tokens": round(sum(r.output_tokens for r in rs) / n, 1) if n else 0.0,
                "mean_total_tokens": round(tot_tok / n, 1) if n else 0.0,
                "tokens_per_success": round(tot_tok / succ, 1) if succ else None,
                "mean_wall_clock_s": round(sum(r.wall_clock_s for r in rs) / n, 3) if n else 0.0,
                "failure_breakdown": dict(fails),
            }
        )
    return rows


def to_markdown(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "_no results_"
    cols = [
        "condition",
        "n",
        "success_rate",
        "mean_turns",
        "mean_total_tokens",
        "tokens_per_success",
        "mean_wall_clock_s",
    ]
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = [
        "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows
    ]
    breakdown = "\n".join(
        f"- `{r['condition']}`: {r['failure_breakdown']}" for r in rows
    )
    return "\n".join([head, sep, *body, "", "**failure attribution**", breakdown])
