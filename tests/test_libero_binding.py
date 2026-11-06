"""Tests for the LIBERO-PRO binding.

Two layers:

1. pure helpers (action conversion, image flipping, task-id parsing) -- these
   need numpy only, so they run on any machine, including the dev laptop
   without a simulator;
2. a fake ``libero`` package injected into ``sys.modules``, which exercises
   reset/step/caching/success end to end without MuJoCo.
"""
from __future__ import annotations

import sys
import types
import unittest

import numpy as np

from agent_smartsim.envs.bindings import libero_pro_binding as lb


# --------------------------------------------------------------------- helpers
class TestPureHelpers(unittest.TestCase):
    def test_osc_action_clips_and_signs(self):
        a = lb.osc_action((0.05, -0.05, 0.0), gripper_closed=False)
        self.assertEqual(a.shape, (7,))
        np.testing.assert_allclose(a[:3], [1.0, -1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(a[3:6], 0.0)
        self.assertEqual(a[6], -1.0)  # open

        b = lb.osc_action((0.5, 0.0, 0.0), gripper_closed=True)  # 10x the step
        self.assertEqual(b[0], 1.0)  # clipped, not 10
        self.assertEqual(b[6], 1.0)  # closed

    def test_osc_action_rejects_non_finite(self):
        with self.assertRaises(ValueError):
            lb.osc_action((float("nan"), 0.0, 0.0), gripper_closed=False)

    def test_split_move(self):
        # 0.10 m at 0.05 m/step -> 2 substeps of 0.05
        parts = lb.split_move((0.10, 0.0, 0.0), lb.POSITION_STEP_M, max_substeps=8)
        self.assertEqual(len(parts), 2)
        np.testing.assert_allclose(parts, [[0.05, 0, 0], [0.05, 0, 0]], atol=1e-6)
        # bounded by max_substeps
        parts = lb.split_move((0.40, 0.0, 0.0), lb.POSITION_STEP_M, max_substeps=3)
        self.assertEqual(len(parts), 3)

    def test_flip_image(self):
        img = np.arange(9, dtype=np.uint8).reshape(3, 3, 1)
        flipped = lb.flip_image(img)
        np.testing.assert_array_equal(flipped[:, 0, 0], [6, 3, 0])
        np.testing.assert_array_equal(lb.flip_image(flipped), img[:, :, 0][:, :, None])

    def test_parse_obs_flips_and_extracts(self):
        obs = {
            "agentview_image": np.array(
                [[[0]], [[1]], [[2]]], dtype=np.uint8
            ),  # 3x1x1, rows 0,1,2
            "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "robot0_eef_pos": [0.1, 0.2, 0.3],
            "robot0_gripper_qpos": [0.02, -0.02],
        }
        images, state = lb.parse_obs(obs)
        self.assertEqual(set(images), {"agentview", "wrist"})
        np.testing.assert_array_equal(images["agentview"][:, 0, 0], [2, 1, 0])
        self.assertEqual(state["eef_pos"].shape, (3,))
        self.assertEqual(state["gripper_qpos"].shape, (2,))

    def test_split_task_id(self):
        self.assertEqual(
            lb.split_task_id("libero_object_with_red_stick/KITCHEN_SCENE4_x"),
            ("libero_object_with_red_stick", "KITCHEN_SCENE4_x"),
        )
        for bad in ("libero_object", "", "/name", "suite/"):
            with self.assertRaises(ValueError):
                lb.split_task_id(bad)


# ---------------------------------------------------------------- fake libero
class _FakeTask:
    def __init__(self, name: str, language: str, bddl: str):
        self.name = name
        self.language = language
        self.bddl_file = bddl
        self.problem_folder = "fake"


class _FakeSuite:
    n_tasks = 2

    def __init__(self):
        self._tasks = [
            _FakeTask("SCENE1_pick_the_mug", "pick up the mug", "/fake/one.bddl"),
            _FakeTask("SCENE2_put_the_bowl", "put the bowl on the plate", "/fake/two.bddl"),
        ]

    def get_task(self, i):
        return self._tasks[i]

    def get_task_init_states(self, i):
        return [np.zeros(3), np.ones(3) * 7]


class _FakeEnv:
    """Minimal stand-in for OffScreenRenderEnv."""

    instances: list = []

    def __init__(self, bddl_file_name=None, camera_heights=128, camera_widths=128, **kwargs):
        self.bddl = bddl_file_name
        self.size = (camera_heights, camera_widths)
        self.n_steps = 0
        self.actions = []
        self.init_state = None
        self.seed_used = None
        self.closed = False
        _FakeEnv.instances.append(self)

    def seed(self, s):
        self.seed_used = s

    def reset(self):
        self.n_steps = 0

    def set_init_state(self, st):
        self.init_state = np.asarray(st)

    def step(self, a):
        self.n_steps += 1
        self.actions.append(np.asarray(a, dtype=np.float32))
        h, w = self.size
        ramp = np.arange(h, dtype=np.uint8).reshape(h, 1, 1) * np.ones((1, w, 3), np.uint8)
        return {
            "agentview_image": ramp,
            "robot0_eye_in_hand_image": np.zeros((2, 2, 3), np.uint8),
            "robot0_eef_pos": np.array([0.0, 0.0, 1.0]),
            "robot0_eef_quat": np.zeros(4),
            "robot0_gripper_qpos": np.array([0.02, -0.02]),
        }

    def check_success(self):
        return self.n_steps >= 3

    def close(self):
        self.closed = True


def _install_fake_libero():
    libero_pkg = types.ModuleType("libero")
    inner = types.ModuleType("libero.libero")
    benchmark_mod = types.ModuleType("libero.libero.benchmark")
    envs_mod = types.ModuleType("libero.libero.envs")

    benchmark_mod.get_benchmark_dict = lambda help=False: {
        "libero_object": _FakeSuite,
        "libero_object_with_red_stick": _FakeSuite,
    }
    envs_mod.OffScreenRenderEnv = _FakeEnv
    inner.benchmark = benchmark_mod
    inner.envs = envs_mod
    libero_pkg.libero = inner

    sys.modules.update(
        {
            "libero": libero_pkg,
            "libero.libero": inner,
            "libero.libero.benchmark": benchmark_mod,
            "libero.libero.envs": envs_mod,
        }
    )


def _uninstall_fake_libero():
    for key in ("libero", "libero.libero", "libero.libero.benchmark", "libero.libero.envs"):
        sys.modules.pop(key, None)


class TestBindingWithFakeSim(unittest.TestCase):
    def setUp(self):
        _install_fake_libero()
        _FakeEnv.instances.clear()
        lb._ENV_CACHE.clear()
        self.b = lb.LiberoProBinding(
            suites=["libero_object_with_red_stick"], camera_size=64, settle_steps=5
        )

    def tearDown(self):
        _uninstall_fake_libero()
        lb._ENV_CACHE.clear()

    def test_list_task_ids(self):
        self.assertEqual(
            self.b.list_task_ids(),
            [
                "libero_object_with_red_stick/SCENE1_pick_the_mug",
                "libero_object_with_red_stick/SCENE2_put_the_bowl",
            ],
        )

    def test_reset_settles_and_picks_init_state_by_seed(self):
        images, state, instruction = self.b.reset(
            "libero_object_with_red_stick/SCENE1_pick_the_mug", seed=1
        )
        self.assertEqual(instruction, "pick up the mug")
        env = _FakeEnv.instances[-1]
        self.assertEqual(env.n_steps, 5)  # settle steps ran
        self.assertEqual(env.seed_used, 1)
        np.testing.assert_array_equal(env.init_state, np.ones(3) * 7)  # seed 1 -> index 1
        # images come back flipped: the ramp 0..63 becomes 63..0
        np.testing.assert_array_equal(images["agentview"][:, 0, 0], np.arange(63, -1, -1))
        self.assertIn("eef_pos", state)

    def test_move_to_expands_into_bounded_substeps(self):
        b = lb.LiberoProBinding(
            suites=["libero_object_with_red_stick"], camera_size=64, settle_steps=0
        )
        b.reset("libero_object_with_red_stick/SCENE1_pick_the_mug", seed=0)
        env = _FakeEnv.instances[-1]
        before = env.n_steps
        _, _, done, info = b.step({"type": "move_to", "delta": [0.1, 0.0, 0.0]})
        self.assertEqual(env.n_steps - before, 2)  # 0.10 m / 0.05 m per step
        self.assertFalse(done)
        for a in env.actions[before:]:
            self.assertLessEqual(np.max(np.abs(a[:3])), 1.0 + 1e-6)
            self.assertEqual(a[6], -1.0)  # gripper held open
        self.assertEqual(info["sim_steps"], 2)

    def test_move_to_is_clipped_to_max_move(self):
        self.b.reset("libero_object_with_red_stick/SCENE1_pick_the_mug", seed=0)
        env = _FakeEnv.instances[-1]
        before = env.n_steps
        self.b.step({"type": "move_to", "delta": [0.9, 0.0, 0.0]})  # 0.9 m requested
        moved = sum(float(a[0]) for a in env.actions[before:]) * lb.POSITION_STEP_M
        self.assertAlmostEqual(moved, lb.MAX_MOVE_M, places=5)  # bounded to 0.25 m

    def test_gripper_sign_and_success(self):
        # settle_steps=0 so the fake env's "success after 3 steps" only trips
        # when our actions, not the reset settling, produce the steps.
        b = lb.LiberoProBinding(
            suites=["libero_object_with_red_stick"], camera_size=64, settle_steps=0
        )
        b.reset("libero_object_with_red_stick/SCENE1_pick_the_mug", seed=0)
        env = _FakeEnv.instances[-1]
        self.assertEqual(env.n_steps, 0)

        _, _, done, _ = b.step({"type": "set_gripper", "closed": True})
        self.assertEqual(env.actions[-1][6], 1.0)  # +1 closes
        self.assertFalse(done)
        self.assertFalse(b.success())

        b.step({"type": "move_to", "delta": [0.01, 0.0, 0.0]})  # 2nd action
        _, _, done, _ = b.step({"type": "move_to", "delta": [0.01, 0.0, 0.0]})  # 3rd
        self.assertTrue(done)
        self.assertTrue(b.success())

    def test_open_articulated_is_refused(self):
        self.b.reset("libero_object_with_red_stick/SCENE1_pick_the_mug", seed=0)
        images, state, done, info = self.b.step(
            {"type": "open_articulated", "part_id": "part_0", "target_fraction": 1.0}
        )
        self.assertEqual(images, {})
        self.assertIn("error", state)
        self.assertFalse(done)
        self.assertEqual(info["last_error_class"], "agent_api_error")

    def test_articulation_is_empty(self):
        self.assertEqual(self.b.articulation(), {})

    def test_env_is_cached_across_bindings(self):
        self.b.reset("libero_object_with_red_stick/SCENE1_pick_the_mug", seed=0)
        self.assertEqual(len(_FakeEnv.instances), 1)
        other = lb.LiberoProBinding(
            suites=["libero_object_with_red_stick"], camera_size=64, settle_steps=1
        )
        other.reset("libero_object_with_red_stick/SCENE1_pick_the_mug", seed=0)
        self.assertEqual(len(_FakeEnv.instances), 1)  # reused, not rebuilt
        # a different task rebuilds (different bddl)
        other.reset("libero_object_with_red_stick/SCENE2_put_the_bowl", seed=0)
        self.assertEqual(len(_FakeEnv.instances), 2)

    def test_unknown_suite_and_task_errors(self):
        bad = lb.LiberoProBinding(suites=["not_a_suite"])
        with self.assertRaises(ValueError) as ctx:
            bad.list_task_ids()
        self.assertIn("libero_object", str(ctx.exception))  # lists what is available

        with self.assertRaises(KeyError):
            self.b.reset("libero_object_with_red_stick/does_not_exist", seed=0)


if __name__ == "__main__":
    unittest.main()
