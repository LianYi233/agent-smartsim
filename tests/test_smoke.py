"""Offline tests: the harness must run end to end with no sim, GPU or API key."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_smartsim.agent.tools import ToolBox
from agent_smartsim.core import EpisodeRecord
from agent_smartsim.envs.mock import MockArticulatedEnv
from agent_smartsim.eval.metrics import aggregate, attribute_failure
from agent_smartsim.eval.runner import run_experiment

CFG_PATH = Path(__file__).resolve().parents[1] / "configs" / "experiment_mock.json"


def _cfg() -> dict:
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    cfg["agent"]["mode"] = "scripted"
    cfg["seeds"] = [0]
    return cfg


class TestMockEnv(unittest.TestCase):
    def test_grasp_then_wrong_direction(self) -> None:
        env = MockArticulatedEnv(max_steps=40, noise=0.0)  # deterministic placement
        env.reset("door_hinge_01", seed=0)
        # reach the handle exactly, then close the gripper
        env.step({"type": "move_to", "delta": [0.15, 0.10, 0.30]})
        env.step({"type": "set_gripper", "closed": True})
        self.assertTrue(env.diagnostics()["grasped"])
        # push the wrong way: no progress, and it is labelled as such
        env.step({"type": "move_to", "delta": [0.0, 0.08, 0.0]})
        diag = env.diagnostics()
        self.assertEqual(diag["last_error_class"], "wrong_direction")
        self.assertEqual(diag["progress_fraction"], 0.0)
        # correct direction accumulates progress
        for _ in range(8):
            env.step({"type": "move_to", "delta": [0.0, -0.08, 0.0]})
        self.assertTrue(env.success)

    def test_open_articulated_requires_grasp(self) -> None:
        env = MockArticulatedEnv()
        env.reset("drawer_slide_01", seed=0)
        env.step({"type": "open_articulated", "part_id": "part_0", "target_fraction": 1.0})
        self.assertEqual(env.diagnostics()["last_error_class"], "not_grasped")


class TestToolGating(unittest.TestCase):
    def test_annotated_tools_are_unavailable_in_rgb_only(self) -> None:
        env = MockArticulatedEnv()
        env.reset("door_hinge_01", seed=0)
        box = ToolBox(env, annotated=False)
        self.assertNotIn("list_parts", box.names)
        out = box.dispatch("list_parts", {})
        self.assertIn("not available", out.text)

    def test_annotated_condition_exposes_spec(self) -> None:
        env = MockArticulatedEnv()
        env.reset("door_hinge_01", seed=0)
        box = ToolBox(env, annotated=True)
        self.assertIn("open_articulated", box.names)
        out = box.dispatch("get_part_spec", {"part_id": "part_0"})
        self.assertIn("axis", out.text)


class TestEndToEnd(unittest.TestCase):
    def test_ablation_runs_and_annotated_is_cheaper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = run_experiment(_cfg(), out_dir=str(Path(tmp) / "run"), verbose=False)
            self.assertTrue((run_dir / "episodes.jsonl").exists())
            rows = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            by = {r["condition"]: r for r in rows}
            self.assertEqual(by["annotated"]["success_rate"], 1.0)
            self.assertLess(by["annotated"]["mean_turns"], by["rgb_only"]["mean_turns"])

    def test_attribution_labels(self) -> None:
        rec = EpisodeRecord(
            task_id="t", seed=0, condition="c", success=False, done_reason="agent_finish",
            diagnostics={"contact_made": True, "grasped": True, "progress_fraction": 0.34},
        )
        self.assertEqual(attribute_failure(rec), "premature_finish")
        rec.diagnostics["progress_fraction"] = 0.0
        rec.diagnostics["grasped"] = False
        self.assertEqual(attribute_failure(rec), "grasp_failed")
        rec.success = True
        self.assertEqual(attribute_failure(rec), "success")

    def test_aggregate_is_stable(self) -> None:
        recs = [
            EpisodeRecord(task_id="t", seed=0, condition="a", success=True, turns=3),
            EpisodeRecord(task_id="t", seed=1, condition="a", success=False, turns=5),
        ]
        rows = aggregate(recs)
        self.assertEqual(rows[0]["n"], 2)
        self.assertEqual(rows[0]["success_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
