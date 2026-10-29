# agent-smartsim

**Articulation annotations as an agent tool interface** — an LLM-agent harness for
articulated object manipulation (doors, drawers, knobs), with a LIBERO pick-and-place
control arm.

> Status: research scaffold (v0.1). The **mock environment runs end to end offline**
> and is covered by tests. The **SMART-Sim and LIBERO adapters are templates**: they
> define the binding contract and fail loudly until you implement 5 functions.
> See [docs/binding_guide.md](docs/binding_guide.md).

---

## Why

Modern embodied-agent papers (LIBERO-Agent, EmbodiedSkillBench, World Action Agent,
Robo-Harness K1, ChunkAct) show that what a frontier model can do with a robot depends
less on the model than on the **harness**: which observations it may request, which
actions it may issue, and how much structure it is handed.

Most of that work lives on pick-and-place benchmarks. Articulated objects are different:
the task is *contact-rich* and *constraint-following*, the relevant information is
kinematic (where is the handle, which axis, which sense, how far), and simulators that
generate such data already know all of it.

**This repo asks one question:**

> If a simulator's articulation annotations are exposed to an LLM agent as *typed tools*
> rather than left implicit in pixels, how much of the agent's action and token budget
> does it save — and does it change which failures the agent makes?

That is a clean, cheap, and (as far as we can tell) unoccupied ablation: the annotation
machinery exists in SMART-Sim, and the agent-tool-interface argument exists in the
manipulation literature, but the two have not been crossed.

---

## Quickstart (no GPU, no simulator, no API key)

```bash
pip install -e .                       # numpy + Pillow only
python -m agent_smartsim.cli smoke     # offline end-to-end check
python -m unittest discover -s tests   # 7 tests
```

`smoke` runs both agent conditions on the mock articulated object and prints:

| condition | n | success_rate | mean_turns |
|---|---|---|---|
| annotated | 6 | 1.0 | **7.0** |
| rgb_only | 6 | 1.0 | **13.0** |

Both succeed; the annotated arm needs ~46% fewer agent turns because it stops guessing
the actuation direction. That is the shape of the effect this harness is built to
measure — **on the mock it is a wiring check, not a result.**

With a model:

```bash
cp .env.example .env                   # put OPENAI_API_KEY in it
python -m agent_smartsim.cli run --config configs/experiment_smartsim.example.json \
    --max-episodes 4                   # debug first: 4 episodes, then scale
python -m agent_smartsim.cli report --run runs/<dir>
```

---

## Design

### Two conditions, one harness

| | `rgb_only` | `annotated` |
|---|---|---|
| Tools | `observe`, `get_state`, `move_to`, `set_gripper`, `finish` | + `list_parts`, `get_part_spec`, `open_articulated` |
| Knows the handle point | ✗ (must reach it from pixels) | ✓ |
| Knows axis / sense / range | ✗ (must infer by pushing) | ✓ |
| Tests | classic visuomotor agent loop | *perception as tools* for contact-rich tasks |

`annotated_spec_only` (`annotated_executor: false`) is the third cell: it gets the
annotations but **not** the typed executor, which separates "knowing where/what" from
"having a reliable primitive".

### Metrics

Per episode we record exactly what the embodied-agent literature uses, so numbers are
comparable on sight:

- `success` — task outcome
- `turns`, `env_steps` — action cost
- `input_tokens`, `output_tokens` — model cost
- `wall_clock_s` — latency cost
- `failure_label` — **attribution**, not just outcome

Attribution rules (see `eval/metrics.py`) distinguish `no_contact`, `grasp_failed`,
`wrong_direction`, `partial_articulation`, `premature_finish`, `budget_exhausted`,
`agent_api_error`. A drop in success is only interpretable if you can say *which* of
these moved.

---

## Layout

```
configs/            experiment configs (JSON; no yaml dependency)
src/agent_smartsim/
  core.py           Observation / StepResult / EpisodeRecord
  agent/
    loop.py         the agent loop (LLM -> tool -> env, no hidden retries)
    tools.py        tool schemas + gating (this is the independent variable)
    prompts.py      per-condition system prompts
    llm.py          OpenAI client, image encoding, token accounting
    scripted.py     offline reference policy
  envs/
    base.py         Env interface (5 methods; see binding guide)
    binding.py      EnvBinding protocol for new platforms
    mock.py         deterministic articulated mock (runs offline)
    smartsim.py     SMART-Sim adapter  [template]
    libero.py       LIBERO control arm [template]
  eval/
    runner.py       condition x task x seed -> JSONL + summary
    metrics.py      metrics + failure attribution
    report.py       re-aggregate a run
  cli.py            smoke / run / report
docs/               experiment plan + binding guide
scripts/            LIBERO setup helper
tests/              offline tests
```

---

## Roadmap

- [x] Offline mock + agent loop + metrics + attribution + reporting
- [x] Tool gating that makes the ablation structural, not anecdotal
- [ ] SMART-Sim binding (needs: 5 functions in `envs/bindings/smartsim_binding.py`)
- [ ] LIBERO / LIBERO-PRO binding (control arm)
- [ ] Spec-only condition run at scale (3 cells x 3 task types x 20 seeds)
- [ ] Real-arm sanity check (SMART reports zero-shot sim-to-real for its VLA; the
      agent path is untested)

## Related work

- **SMART / SMART-Sim** — articulated-object synthetic pretraining at scale (the
  environment this repo targets). arXiv:2610.07652
- **LIBERO-PRO** — perturbations that separate memorisation from competence;
  standard LIBERO is saturated. arXiv:2510.03827
- **EmbodiedSkillBench** — skill interfaces with explicit preconditions; source of the
  four headline metrics. arXiv:2610.16238
- **Robo-Harness K1** — "perception as tools" for manipulation. arXiv:2609.29389
- **LIBERO-Agent** — agent-native evaluation *protocol* (not a dataset): 200 tasks with a
  30-task primary suite separating perceptual / short-horizon / long-horizon ability, where
  the agent chooses its own observations and issues native action commands.
  **No official code as of 2026-10-29** — no code link in the arXiv entry, no matching repo.
  arXiv:2609.39507
- **ChunkAct** — action chunking for agentic robotics (cost-side baseline). arXiv:2610.18874

## Reference implementations

Useful when writing the LIBERO binding (checked 2026-10-29):

- `nssmd/RoboRSI` — official repo of the RoboRSI paper; robot-agent harness with a CLI and a
  local web console for LIBERO short-horizon evaluation. Closest to a working end-to-end
  LIBERO agent harness.
- `Peter-cuhk/lffbench` — learn-from-failure benchmark for VLM robot agents on LIBERO.
- `ABC12345anouys/EvoBot` — agent plus reusable skill library on LIBERO (no LLM at runtime).
- `dzj441/Libero-Agent` — created 10 days after the paper, no description; **unofficial and
  unverified**, do not treat as the reference implementation.

## Citation

```bibtex
@software{agent_smartsim_2026,
  title  = {agent-smartsim: articulation annotations as an agent tool interface},
  author = {LianYi233},
  year   = {2026},
  url    = {https://github.com/LianYi233/agent-smartsim}
}
```

## License

MIT. Third-party simulators (SMART-Sim, LIBERO, robosuite) keep their own licenses;
nothing from them is vendored here.
