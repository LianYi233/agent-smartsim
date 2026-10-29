"""Command line entry point.

    python -m agent_smartsim.cli smoke                     # offline, no key needed
    python -m agent_smartsim.cli run --config configs/experiment_smartsim.example.json
    python -m agent_smartsim.cli report --run runs/<dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .eval.report import write_reports
from .eval.runner import run_experiment


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmd_smoke(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    cfg.setdefault("agent", {})["mode"] = "scripted"
    run_dir = run_experiment(cfg, out_dir=args.out)
    rows = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    by = {r["condition"]: r for r in rows}
    ann, rgb = by.get("annotated"), by.get("rgb_only")
    ok = bool(
        ann
        and rgb
        and ann["success_rate"] >= 1.0
        and ann["mean_turns"] < rgb["mean_turns"]
    )
    print("\nsmoke:", "PASS" if ok else "FAIL")
    if ann and rgb:
        print(
            f"  annotated: {ann['success_rate']:.0%} success in {ann['mean_turns']:.1f} turns | "
            f"rgb_only: {rgb['success_rate']:.0%} success in {rgb['mean_turns']:.1f} turns"
        )
    return 0 if ok else 1


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    run_experiment(cfg, out_dir=args.out, max_episodes=args.max_episodes)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    rows = write_reports(args.run)
    print((Path(args.run) / "summary.md").read_text(encoding="utf-8"))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="agent-smartsim",
        description="Articulation annotations as an agent tool interface.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("smoke", help="offline end-to-end check on the mock env (no API key)")
    s.add_argument("--config", default="configs/experiment_mock.json")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_smoke)

    r = sub.add_parser("run", help="run a real experiment (needs OPENAI_API_KEY unless mode=scripted)")
    r.add_argument("--config", required=True)
    r.add_argument("--out", default=None)
    r.add_argument("--max-episodes", type=int, default=None, help="debug: stop after N episodes")
    r.set_defaults(func=cmd_run)

    q = sub.add_parser("report", help="re-aggregate an existing run directory")
    q.add_argument("--run", required=True)
    q.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
