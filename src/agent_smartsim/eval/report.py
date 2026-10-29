"""Re-aggregate an existing run directory (no re-execution)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from ..core import EpisodeRecord
from .metrics import aggregate, to_markdown


def load_records(run_dir: str | Path) -> List[EpisodeRecord]:
    path = Path(run_dir) / "episodes.jsonl"
    out: List[EpisodeRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        d.pop("total_tokens", None)
        out.append(EpisodeRecord(**d))
    return out


def write_reports(run_dir: str | Path) -> List[Dict[str, Any]]:
    run_dir = Path(run_dir)
    rows = aggregate(load_records(run_dir))
    (run_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "summary.md").write_text("# Summary\n\n" + to_markdown(rows) + "\n", encoding="utf-8")

    flat_cols = [
        "condition",
        "n",
        "success_rate",
        "mean_turns",
        "mean_env_steps",
        "mean_input_tokens",
        "mean_output_tokens",
        "mean_total_tokens",
        "tokens_per_success",
        "mean_wall_clock_s",
    ]
    with open(run_dir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=flat_cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return rows
