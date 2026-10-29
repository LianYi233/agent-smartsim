# Binding a real simulator

The harness talks to a platform through **five functions**. Everything else — tools,
prompts, metrics, failure attribution, reporting — is written against this contract and
needs no changes.

## 1. Go / no-go checklist

Before writing glue, confirm these. If the first column fails, the platform cannot host
an agent experiment at all.

| # | Requirement | Why it is load-bearing |
|---|---|---|
| 1 | **Programmatic `step(action)` with external action injection** and closed-loop feedback | The agent loop *is* observation → action → observation. No injection, no agent. |
| 2 | **Simulator itself is obtainable** (not just a released dataset) | A dataset gives you demonstrations, not an environment. Zero episodes. |
| 3 | **Machine-readable success** | Otherwise every number needs a human in the loop. |
| 4 | **Deterministic `reset(task_id, seed)`** | Without seeds nothing is reproducible and no ablation is meaningful. |
| 5 | **Scriptable scene/object/initial-state randomisation** | The standard evaluation axis (cf. LIBERO-PRO's four perturbation dimensions). |
| 6 | **Headless, batchable, resettable** | 3 tasks × 20 seeds × 3 conditions must run unattended. |

Two more that decide *how much* you get out of it:

7. Annotations queryable **at runtime** (not only used inside their training pipeline) —
   without this the `annotated` condition is impossible.
8. A 7-DoF parallel-jaw arm among the robot setups — decides whether results can transfer
   to your real arm.

## 2. The contract

```python
# src/agent_smartsim/envs/binding.py
class EnvBinding(Protocol):
    def list_task_ids(self) -> List[str]: ...

    def reset(self, task_id: str, seed: int
              ) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        """-> (images, state, instruction). Deterministic given (task_id, seed)."""

    def step(self, action: Dict[str, Any]
             ) -> Tuple[Dict[str, Any], Dict[str, Any], bool, Dict[str, Any]]:
        """-> (images, state, done, info). One action per call."""

    def success(self) -> bool: ...

    def articulation(self) -> Dict[str, Any]:
        """handle_point, axis, direction, range, target_fraction; {} if unavailable."""
```

Optional but strongly recommended: `diagnostics() -> dict` with `contact_made`,
`grasped`, `progress_fraction`, `last_error_class`. That is what turns "it failed" into
"it failed *because*".

### Action vocabulary the harness will send

```python
{"type": "move_to", "delta": [dx, dy, dz]}      # metres, world frame, relative
{"type": "set_gripper", "closed": True}
{"type": "open_articulated", "part_id": ..., "target_fraction": 1.0}   # annotated only
```

Translate these into whatever the simulator wants (delta pose, IK target, joint
positions). Keep `move_to` **bounded** — an unbounded translation lets one bad model
decision end the episode, which pollutes attribution.

## 3. SMART-Sim notes

- The platform ships articulation-aware design: part-level semantics plus joint
  constraints, generated programmatically. Bind `articulation()` straight to it —
  that is the whole point of this repo.
- 44 atomic task types is your primitive vocabulary: start with 3 (`open door`,
  `pull drawer`, `turn knob`), not 44.
- 2,507 articulated objects / 5 robot setups means category-level generalisation is
  available *for free* — hold out object categories, not just seeds.
- Image resolution is the dominant token cost. Cap it (e.g. 256²) and re-encode before
  attaching; report the resolution you used.
- If annotations exist only inside the data-generation path and are not queryable,
  say so in the paper and fall back to `rgb_only` — that is a finding, not a failure.

## 4. LIBERO notes (control arm)

- Success is a predicate: `success()` is cheap, no extra labels needed.
- `LIBERO-Spatial` / `-Object` are the pick-and-place suites — start there, not `-Long`.
- Run **both** standard LIBERO and LIBERO-PRO perturbations and report them
  separately. Standard LIBERO is saturated (>90% for strong VLAs) and LIBERO-PRO shows
  collapse to ~0% for memorising policies; a single average hides which one you got.
- No articulation annotations exist here. `LiberoEnv.articulation()` returns `{}`, so
  the `annotated` condition is deliberately unavailable — this keeps the control arm
  honest and prevents accidental leakage between conditions.

## 5. Failure modes seen in practice

| Symptom | Likely cause |
|---|---|
| Agent always misses by a constant offset | coordinate-frame mismatch (world vs base vs camera) |
| `move_to` "works" but nothing moves | action clipped to ~0, or simulator stepping in degrees while you send radians |
| Success looks great, tokens explode | full-resolution multi-view images every turn; cap resolution, attach images only after `observe` |
| Results not reproducible | reset ignores seed, or object poses sampled from global RNG |
| Agent loops forever on a closed drawer | no `progress_fraction` in `diagnostics()`, so the agent has no feedback signal |
