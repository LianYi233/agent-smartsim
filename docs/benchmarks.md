# RUA evaluation landscape (snapshot 2026-11-06)

**Counts below are arXiv `all:"<name>"` matches restricted to submissions between
2026-06-01 and 2026-11-30**, queried through the arXiv API. `all:` covers title, abstract,
comments and journal refs — not full text — so treat the numbers as a *reporting*
frequency, not a citation count. "with agent" = the same window plus `all:"agent"`.

## At a glance

| Benchmark | mentions | of which "agent" | engine | scale | real robot | RUA fit |
|---|---|---|---|---|---|---|
| **LIBERO / -PRO / -Plus / -VPro** | 637 (PRO: 67) | 75 | MuJoCo + robosuite | 130 tasks; PRO adds 4 perturbation axes | VPro: 200 rollouts on Franka R3 | ★★★★★ the default |
| **RoboDojo** | 40 | 19 | Isaac Sim (heterogeneous parallel) | 42 sim + 18 real, 5 capability dims | RoboDojo-RealEval (cloud, standardised hardware) | ★★★★★ purpose-built for generalist/agent eval |
| **RoboTwin** | 246 | 26 | SAPIEN, real2sim digital twin | dual-arm suite | twin of real dual-arm | ★★★★ second environment |
| **RoboCasa / 365** | 94 | 13 | MuJoCo + robosuite | kitchen-scale, 365 composite tasks | — | ★★★★ long-horizon household |
| **SimplerEnv** | 39 | — | MuJoCo | Google-robot / WidowX replicas | — | ★★★ policy replication |
| **Meta-World / ManiSkill / RLBench** | 33 / 33 / 22 | — | MuJoCo / SAPIEN / CoppeliaSim | classic op suites | — | ★★ auxiliary |
| **BEHAVIOR-1K** | 18 | — | OmniGibson (Isaac) | 1000 household activities | — | ★★★ nav+manip |
| **MolmoSpaces** | 8 | — | — | 1.3M indoor environments | — | ★★ niche |
| **EmbodiedBench** | 4 | — | mixed | 1128 tasks, 8 domains | — | ★★★ agent-native framing |
| **CaP-Bench** | 3 | — | MuJoCo | code-as-policy splits | — | ★★ code-centric |
| **LIBERO-Agent** | 1 | — | MuJoCo (reuses LIBERO) | 200 tasks, 30-task primary suite | — | protocol only, **no code** |

## Tier 1 — what people actually run

### LIBERO family (the default)
- **LIBERO** (130 tasks, MuJoCo/robosuite) — the substrate almost every agent paper reuses.
- **LIBERO-PRO** — four perturbation axes (objects, initial states, instructions,
  environments). Standard LIBERO is saturated (>90%) and memorisation-prone; PRO drops
  reported scores toward 0% for policies that memorise. Agent papers now report *both*.
- **LIBERO-Plus** (7 perturbation dims), **LIBERO-VPro** (visual robustness + real Franka R3
  rollouts), **LIBERO-PeRM** (personalised preferences).
- Used by: EmbodiedSkillBench (100 perturbed PRO tasks), ChunkAct, World Action Agent,
  MotorMind, RACaP, RoboRSI, EmbodiedRSI, RoboFoundry, RoboBridge, ProactiveVLA, RV-ICL …
- Caveat: the *task* family is short-horizon pick-and-place, so it cannot test articulation.

### RoboDojo (the RUA-native one)
- 42 simulation + 18 real tasks across five capability dimensions: **generalisation,
  memory, precision, long-horizon execution, open-vocabulary instruction following**.
- Official protocol: 50 episodes per task; public leaderboard; Isaac Sim heterogeneous
  parallel rollouts; `RoboDojo-RealEval` offers cloud access with standardised hardware,
  scene reset and a deployment interface; XPolicyLab lets a policy be integrated once.
- Used by: "LLM as policy" evaluations of GPT-6 Astra / GPT-5.5 / DeepSeek-Flash,
  uni-robot-agent (Qwen3.8-Flash-Next 31.39, Astra 51.90), PyRUA-Lean (51.6%, +19.4 over
  the then-leaderboard top), Open-HumanWAM, Compounding Physical RSI.
- Why RUA papers like it: open-vocabulary instructions, memory and long-horizon are
  exactly the axes where an LLM agent differs from a VLA, and the sim/real split is one
  protocol rather than two papers.

**Task catalog** (from `task/RoboDojo/config/*.yml` and the official sim-tasks page,
checked 2026-11-06; 42 tasks + 12 `_random` layout variants + 1 DLC entry):

| Dimension | Tasks |
|---|---|
| Generalization (12) | stack_bowls, push_T, pack_objects_into_box, fold_clothes, hang_mugs, sweep_blocks, pour_liquid_into_cup, make_toast, arrange_largest_number, sort_nesting_dolls_by_size, store_laptop_and_headphones, stack_blocks |
| Memory (6) | cover_blocks, match_and_pick_from_conveyor, swap_blocks, swap_T, press_by_number, imitate_sorting_sequence |
| Precision (8) | fasten_screws, plug_in_charger, insert_tubes, pour_balls_into_vase, play_Xylophone, deposit_coin, insert_key, build_tower |
| Long-Horizon (8) | fill_pen_holder, classify_objects, put_bottles_into_dustbin, play_tic_tac_toe, fill_egg_holder, organize_table, make_kong, play_stacking_toy |
| Open (8) | align_blocks, general_pickup, solve_equation, stack_blocks_by_language, classify_objects_by_language, pick_from_conveyor_by_image, store_tools_in_toolbox, pour_by_language |

- Scenes: tabletop manipulation, configuration-driven; assets mix **rigid + articulated +
  deformable** objects (laptop, toaster with bread shelf, number cards are declared
  `Articulation:` in task configs); layout/material/lighting/background domain
  randomisation is a first-class tool.
- Embodiments: sim configs ship `franka` and `x5`; the real suite runs on Piper X, Piper
  and ARX X5.
- **Articulation status**: articulations appear as *props* inside broader tasks
  (make_toast, store_laptop_and_headphones, press_by_number); there are no door/drawer/
  cabinet-centric tasks and no part-level annotation interface. The niche stays open.
- Hardware (official): Ubuntu 22.04, RAM >= 32 GB, GPU VRAM >= 16 GB (rendering-heavy
  workflows need more), driver 570/580, CUDA 12.8, Isaac Sim 5.1 + Isaac Lab 2.3; Docker
  image needs ~300 GB free. Assets come from **ModelScope** (fast in China); demo data is
  separate (HDF5 523 GB, depth 4.5 TB) and **not needed for evaluation**.
- A 24 GB RTX 4090 is above the stated minimum: single-env or 2-4 parallel envs is
  realistic; the full leaderboard protocol (42 x 50 = 2,100 episodes) is not a one-GPU
  job — run a task subset with 10-20 episodes per task for ablations.

### RoboTwin / RoboCasa
- **RoboTwin** — dual-arm, SAPIEN, real2sim twin; the usual "second environment" for
  cross-embodiment claims.
- **RoboCasa365** — kitchen-scale composite tasks; leaderboard used by harness-evolution
  work (71.2% for a self-evolved harness, 77.0% EmbodiedRSI, 50-task AdaRSI).

## Tier 2 and niche
SimplerEnv (VLA replication), Meta-World / ManiSkill / RLBench (classic suites, mostly
baselines), BEHAVIOR-1K (long-horizon household, nav+manip), MolmoSpaces (Recova's
categories), EmbodiedBench (8 domains, agent-native framing), CaP-Bench (code-as-policy).

## Real-robot evaluation
Three patterns, in descending order of reproducibility:
1. **RoboDojo-RealEval** — cloud-accessible, standardised hardware, fixed scene reset and
   protocol. The only one designed to be reproduced by strangers.
2. **Published real suites with a fixed platform** — uni-robot-agent's Real Eval (dual-arm
   gripper, 19 tasks × 5 dimensions × 20 points); scores: Qwen 53.50 > fine-tuned π0.5
   47.75, Astra 80.50.
3. **Per-paper lab setups** — Franka Panda / Research 3, xArm6, mobile manipulators,
   ALOHA-style dual arm. Not comparable across papers; report as "sanity check".

## Where this leaves agent-smartsim

- **Primary**: LIBERO-PRO (has code, has numbers, is the control arm everyone recognises).
- **Secondary, worth adding**: RoboDojo — it is where the RUA conversation is moving, it
  reports the five dimensions that matter for agents, and a leaderboard gives free
  comparison. Budget for Isaac Sim.
- **Do not** bet on LIBERO-Agent as *the* protocol: one self-mention and no code. Cite it
  for the framing (agent-native interface), implement the spirit yourself.
- **The gap that is still open**: no popular benchmark evaluates *articulation-specific
  structure* (handle, axis, direction, range) as agent-visible tools. LIBERO is pick-and-place,
  RoboDojo is generalist manipulation, SMART-Sim is unreleased. That is the niche.
