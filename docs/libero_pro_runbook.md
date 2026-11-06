# LIBERO-PRO control arm: runbook

Everything here runs on the evaluation machine (4090 box, Linux). Nothing in
this runbook needs SMART-Sim, and steps 1-2 need no API key.

## 0. Prerequisites

- Ubuntu 22.04, NVIDIA driver >= 570 (`nvidia-smi` works)
- ~10 GB disk for the repo + assets (no demo datasets needed for evaluation)
- `MUJOCO_GL=egl` for headless rendering (fall back to `osmesa` if EGL fails)

## 1. Install

```bash
bash scripts/setup_libero_pro.sh agent-libero
conda activate agent-libero
pip install -e .            # this repo (agent-smartsim)
```

The script pins `robosuite==1.4.0` + `bddl`, clones
`Zxy-MLlab/LIBERO-PRO` (standard suites **and** perturbation suites in one
package) and prints how many perturbation suites it registered.

If github.com is blocked on that machine:

```bash
GITHUB_MIRROR=https://mirror.ghproxy.com bash scripts/setup_libero_pro.sh agent-libero
```

## 2. Wiring probe (no LLM, no tokens)

```bash
export MUJOCO_GL=egl
agent-smartsim probe --suite libero_object --list | head -15
agent-smartsim probe --suite libero_object --camera-size 128 --gpu 0
```

Expected: task count, one task's natural-language instruction, image shapes
(`agentview (128,128,3)`, `wrist (128,128,3)`), eef position, then
`probe: PASS`. This exercises bddl loading, the offscreen renderer, the
action path (a few blind moves + gripper open/close) and `check_success()`.

Common failures:

| Symptom | Fix |
|---|---|
| `libero is not importable` | rerun step 1 inside the right conda env |
| EGL errors | `export MUJOCO_GL=osmesa` (slower, fine for a probe) |
| black images | pass `--gpu 0`; check `nvidia-smi` |
| very slow | fine — the probe runs ~13 sim steps only |

## 3. Pick tasks and fill the experiment config

```bash
agent-smartsim probe --suite libero_object --list          # base tasks
agent-smartsim probe --suite libero_object_with_red_stick --list   # perturbed twins
```

Pair each chosen base task with its perturbed sibling (same scene prefix)
and put the ids into `configs/experiment_libero_pro.example.json`:

```json
"tasks": [
  "libero_object/KITCHEN_SCENE4_...",
  "libero_object_with_red_stick/KITCHEN_SCENE4_..."
]
```

Start with **2 pairs x 5 seeds** (20 episodes) before scaling.

## 4. First real run (API key needed)

```bash
export OPENAI_API_KEY=...        # and OPENAI_BASE_URL if you use a proxy
agent-smartsim run --config configs/experiment_libero_pro.example.json --max-episodes 2
agent-smartsim run --config configs/experiment_libero_pro.example.json
agent-smartsim report --run runs/libero-pro-control-arm-<stamp>
```

The condition here is `rgb_only` by design: LIBERO has no articulation
annotations, so `annotated` is unavailable and the control arm stays honest
(`LiberoProBinding.articulation()` returns `{}`).

## 5. What "worked" looks like

- probe passes with real images
- a 2-episode API run completes with `done_reason` in
  {`success`, `agent_finish`, `budget_exhausted`} and sane token counts
- failure labels concentrate in `no_contact` / `grasp_failed` /
  `budget_exhausted`, not `agent_api_error`

## 6. Decision gates (from docs/experiment_plan.md)

- >= 60% short-horizon success -> proceed to the perturbation twins immediately
- 30-60% -> one tool-schema revision maximum, then re-run
- < 30% -> check resolution/coordinate frame/action step scale before blaming
  the model: is `position_step_m=0.05` sane for the task? are images flipped
  correctly? does `move_to` overshoot?
