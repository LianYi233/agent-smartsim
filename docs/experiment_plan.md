# Experiment plan

## Hypothesis

Exposing a simulator's articulation annotations to an LLM agent as **typed tools** —
rather than leaving the kinematics implicit in pixels — reduces the action and token
budget needed for contact-rich manipulation, and shifts failures from *geometric*
(missing the handle, pushing the wrong way) to *executional* (contact slip, occlusion).

H1 (cost): `annotated` needs fewer agent turns and fewer tokens than `rgb_only` at equal
success.

H2 (capability): `annotated` raises success on categories where the annotation is
informative but pixels are ambiguous (occluded handle, symmetric-looking doors that open
in opposite directions).

H3 (attribution): the *composition* of failure labels differs; `rgb_only` should be
dominated by `wrong_direction` / `no_contact`, `annotated` by `partial_articulation` /
`grasp_failed`.

H0 to rule out: the gain is just "a scripted primitive is easier than free-form control".
That is what the middle condition (`annotated_spec_only`, annotations **without** the
typed executor) exists to test.

## Conditions

| name | annotations | typed executor | isolates |
|---|---|---|---|
| `rgb_only` | ✗ | ✗ | baseline agent loop |
| `annotated_spec_only` | ✓ | ✗ | value of *knowing* |
| `annotated` | ✓ | ✓ | value of *knowing + doing* |

The difference `annotated − annotated_spec_only` is the value of the primitive; the
difference `annotated_spec_only − rgb_only` is the value of the information alone.
Report both or the claim is unfalsifiable.

## Task selection (SMART-Sim)

Start with 3 of the 44 atomic task types, one per kinematic family:

1. `open door` (hinge, gravity-loaded, occlusion of the handle is common)
2. `pull drawer` (prismatic, direction unambiguous, range matters)
3. `turn knob` (rotational, continuous, sense ambiguous)

Per task type: **10 object instances × 3 seeds = 30 episodes per condition**, so 90 per
task type and 270 per condition overall. Hold out object *categories* (not just instances)
for the generalisation number, so that a win cannot come from memorised assets.

## Metrics and reporting

Per condition, report: `success_rate`, `mean_turns`, `mean_total_tokens`,
`tokens_per_success`, `mean_wall_clock_s`, and the failure-label histogram. Always
report the pair (success, cost) — a method that wins success by spending 10× tokens is a
different claim than one that wins both.

Bootstrap 95% CIs over episodes; with n=30 per cell, a 10-point difference is not
distinguishable from noise — state CIs or increase n.

## Decision gates

| Observation | Interpretation | Next action |
|---|---|---|
| `annotated` ≥ 0.8, `rgb_only` ≤ 0.5 | annotations do the work | scale to 10 task types, add held-out categories |
| All three ≥ 0.8 | task too easy for frontier models | harden: occlude handle, add distractor doors, randomise camera |
| All three ≤ 0.3 | harness or perception problem, not a science result | check the failure-mode table in `binding_guide.md` before touching the model |
| `spec_only` ≈ `rgb_only` < `annotated` | primitive matters more than information | the story is "reliable skills", not "perception as tools" — say so |
| `annotated` wins on success but costs more tokens | annotations buy accuracy, not efficiency | report both axes; do not claim efficiency |

## Budget

Order-of-magnitude for the main sweep (270 episodes/condition × 3 conditions = 810):
at ~15-30 turns per episode and 2-4k input tokens per turn (one or two 256² images plus
a short transcript), that is roughly **20-60M input tokens**. Debug at `--max-episodes 4`
first; the token accounting is already in `EpisodeRecord` so you can extrapolate from a
4-episode run before committing.

## What would make this publishable

1. The spec-only cell, which separates information from primitive.
2. Held-out object categories, so generalisation is about kinematics, not assets.
3. Failure attribution, not just success — "annotations remove wrong-direction failures"
   is a mechanistic claim that a success number cannot make.
4. A LIBERO control arm showing the harness is not the thing being measured.
