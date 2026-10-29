"""Experiment runner: condition x task x seed -> JSONL + summary."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..agent.loop import run_episode
from ..agent.scripted import ScriptedArticulatedClient
from ..core import EpisodeRecord
from ..envs.base import Env
from .metrics import aggregate, attribute_failure, to_markdown


def build_env(env_cfg: Dict[str, Any]) -> Env:
    name = env_cfg.get("name", "mock")
    params = env_cfg.get("params", {}) or {}
    if name == "mock":
        from ..envs.mock import MockArticulatedEnv

        return MockArticulatedEnv(**params)
    if name == "smartsim":
        from ..envs.smartsim import SmartSimEnv

        return SmartSimEnv(**params)
    if name == "libero":
        from ..envs.libero import LiberoEnv

        return LiberoEnv(**params)
    raise ValueError(f"unknown env {name!r}")


def _make_client(agent_cfg: Dict[str, Any]):
    mode = agent_cfg.get("mode", "api")
    if mode == "scripted":
        return ScriptedArticulatedClient()
    from ..agent.llm import OpenAIChatClient

    return OpenAIChatClient(
        model=agent_cfg.get("model", os.environ.get("AGENT_MODEL", "gpt-5.6")),
        temperature=float(agent_cfg.get("temperature", 0.0)),
        max_retries=int(agent_cfg.get("max_retries", 3)),
        max_output_tokens=agent_cfg.get("max_output_tokens", 2048),
    )


def run_experiment(
    cfg: Dict[str, Any],
    out_dir: Optional[str] = None,
    max_episodes: Optional[int] = None,
    verbose: bool = True,
) -> Path:
    agent_cfg = cfg.get("agent", {})
    env_cfg = cfg.get("env", {})
    max_turns = int(agent_cfg.get("max_turns", 20))
    scripted = agent_cfg.get("mode", "api") == "scripted"

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_dir or Path(cfg.get("run", {}).get("out_dir", "runs")) / f"{cfg.get('name', 'exp')}-{stamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    records: List[EpisodeRecord] = []
    episodes_path = run_dir / "episodes.jsonl"
    n_done = 0

    shared_client = None if scripted else _make_client(agent_cfg)

    with open(episodes_path, "w", encoding="utf-8") as fh:
        for cond in cfg["conditions"]:
            cond_name = cond["name"]
            annotated = bool(cond.get("annotated", False))
            executor = bool(cond.get("annotated_executor", True))

            for task_id in cfg["tasks"]:
                for seed in cfg.get("seeds", [0]):
                    if max_episodes is not None and n_done >= max_episodes:
                        break
                    env = build_env(env_cfg)
                    client = ScriptedArticulatedClient() if scripted else shared_client
                    try:
                        rec = run_episode(
                            env,
                            client,
                            task_id=task_id,
                            seed=int(seed),
                            condition=cond_name,
                            annotated=annotated,
                            max_turns=max_turns,
                            annotated_executor=executor,
                        )
                    finally:
                        env.close()
                    rec.failure_label = attribute_failure(rec)
                    records.append(rec)
                    fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
                    fh.flush()
                    n_done += 1
                    if verbose:
                        print(
                            f"[{cond_name}] {task_id} seed={seed} "
                            f"success={rec.success} turns={rec.turns} "
                            f"tokens={rec.total_tokens} label={rec.failure_label}",
                            flush=True,
                        )

    rows = aggregate(records)
    (run_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(
        "# Summary\n\n" + to_markdown(rows) + "\n", encoding="utf-8"
    )
    if verbose:
        print("\n" + to_markdown(rows))
        print(f"\nartifacts: {run_dir}")
    return run_dir
