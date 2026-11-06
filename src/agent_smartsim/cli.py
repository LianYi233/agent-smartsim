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


def cmd_probe(args: argparse.Namespace) -> int:
    """Simulator wiring check: no LLM, no experiment -- reset, act blindly, report.

    Exit 0 means the binding, renderer, action path and success predicate all
    answered. Run this on the evaluation machine before spending any tokens.
    """
    from .envs.bindings.libero_pro_binding import LiberoProBinding

    binding = LiberoProBinding(
        suites=[args.suite],
        camera_size=args.camera_size,
        render_gpu_device=args.gpu,
    )
    try:
        tasks = binding.list_task_ids()
    except ImportError as exc:  # friendly, no traceback for a missing sim
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"suite {args.suite!r}: {len(tasks)} tasks")
    if args.list:
        for t in tasks:
            print(" ", t)
        return 0
    task_id = args.task or (tasks[0] if tasks else None)
    if not task_id:
        print("error: no tasks in suite", file=sys.stderr)
        return 1

    images, state, instruction = binding.reset(task_id, seed=args.seed)
    print(f"task    : {task_id}")
    print(f"seed    : {args.seed}")
    print(f"language: {instruction!r}")
    for name, img in images.items():
        print(f"image   : {name} {img.shape} dtype={img.dtype} range=[{img.min()},{img.max()}]")
    print(f"state   : {state}")

    for i in range(args.steps):
        binding.step({"type": "move_to", "delta": [0.02, 0.0, -0.01]})
        if i == args.steps // 2:
            binding.step({"type": "set_gripper", "closed": True})
    binding.step({"type": "set_gripper", "closed": False})
    ok = binding.success()
    print(f"after {args.steps + 2} blind actions -> success={ok}")
    print(f"diagnostics: {binding.diagnostics()}")
    binding.close()
    print("probe:", "PASS (wiring answers)" if instruction else "PASS (no language attr?)")
    return 0


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

    pb = sub.add_parser("probe", help="check LIBERO/LIBERO-PRO wiring without an LLM")
    pb.add_argument("--suite", default="libero_object")
    pb.add_argument("--task", default=None, help="task name; default: first in the suite")
    pb.add_argument("--seed", type=int, default=0)
    pb.add_argument("--camera-size", type=int, default=128)
    pb.add_argument("--steps", type=int, default=10)
    pb.add_argument("--gpu", type=int, default=None, help="EGL device id for offscreen render")
    pb.add_argument("--list", action="store_true", help="list task ids and exit")
    pb.set_defaults(func=cmd_probe)

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
