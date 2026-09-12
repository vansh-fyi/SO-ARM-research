# Phase 4: Dataset Collection - Research

**Researched:** 2026-08-02
**Domain:** robomimic/robosuite demonstration collection, HDF5 dataset schema, state-based sim replay, VLA-style action/observation normalization
**Confidence:** HIGH (schema, replay mechanics, existing-code reuse — all verified against installed packages and this repo's own code) / MEDIUM (scripted-trajectory generation strategy — community prior art, not an official LIBERO-shipped tool) / LOW (exact episode-count-per-task and replay-sample-size — Claude's discretion, no external authority)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Scripted Demo Generation**
- **D-01:** How the scripted collector generates successful trajectories for the 3 frozen tasks (hand-coded waypoint/IK script vs. scripted + randomized noise/placement vs. VLA-rollout filtering) is **Claude's discretion** — research LIBERO's existing demo-generation conventions and SOARM's tuned kinematics (Phase 2) before choosing. Note: OFT/π0 currently sit at 0% zero-shot success on SOARM (Phase 3 baseline), so VLA-rollout filtering is unlikely to be viable without further work — a hand-coded or noise-augmented waypoint approach is the more realistic default unless research finds a fast fix.

**Teleoperation Interface**
- **D-02:** Prioritize **keyboard-only** input for the teleoperation interface (DATA-04). No SpaceMouse hardware dependency — reuses robosuite's existing keyboard `Device` class. SpaceMouse support is not required for this phase.

**Demo Scope & Execution Environment**
- **D-03:** Demos are collected with a roughly **even split across the 3 tasks** and demo **collection runs locally**, not on Colab — this is CPU-bound MuJoCo sim work (not VLA inference), so no GPU/Colab dependency is needed, and local execution gives faster iteration on the collector itself. This is a departure from Phase 1-3's Colab-first convention, scoped specifically to this phase's collection step (not necessarily downstream training/fine-tuning in Phase 6).
- **D-04:** Exact episode count per task (beyond "roughly even" across 3 tasks, totaling 100+) is **Claude's discretion**.

**Module Location**
- **D-05:** New dataset-collection code lives in a new **`LIBERO/libero/libero/datasets/`** package — mirrors the Phase 3 pattern of `LIBERO/libero/libero/vla/` as a real importable module. Phase 6 (fine-tuning) must be able to import HDF5-loading and normalization-stats utilities directly from here, not from notebook-only code. This is new code, not a modification of the existing `LIBERO/scripts/collect_demonstration.py` / `libero_100_collect_demonstrations.py` (those are human-teleop-only, npz-based, and not robomimic-HDF5 — kept as reference/prior art, not extended in place).

**State-Based Replay Verification (DATA-02)**
- **D-06:** Whether to spot-check a sample of demos (e.g. ~10%) or replay every single recorded demo for state-based determinism verification is **Claude's discretion** — size the verification sample based on measured per-replay runtime once the collector exists, given collection runs locally (D-03).

### Claude's Discretion (summary)
- D-01: Scripted trajectory generation strategy (hand-coded waypoints vs. noise-augmented vs. VLA-filtered). **Resolved by this research — see Architecture Patterns.**
- D-04: Exact episode count per task. **Resolved by this research — see Standard Stack / Common Pitfalls.**
- D-06: Full vs. sampled replay verification for DATA-02. **Resolved by this research — see Validation Architecture.**

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope. Multi-camera perception, depth extraction, and spatial-language tasks remain correctly deferred to Phase 5; RLDS conversion and LoRA fine-tuning remain deferred to Phase 6.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DATA-01 | Scripted demonstration collector records SOARM task completions to robomimic HDF5 format | Standard Stack + Architecture Patterns: two-stage collect-then-extract pipeline (raw states/actions -> obs-augmented HDF5), reusing `DataCollectionWrapper` + this repo's own `env.set_init_state()`. Hand-coded waypoint state machine recommended for D-01. **Task-list correction below is load-bearing for this requirement.** |
| DATA-02 | Recorded demonstrations replay deterministically via state-setting (not action playback) | Code Examples: `robosuite/scripts/playback_demonstrations_from_hdf5.py`'s state-based branch + this repo's `ControlEnv.set_init_state()`/`regenerate_obs_from_state()` — direct, already-installed primitive. Validation Architecture resolves D-06 sample size. |
| DATA-03 | SOARM-specific action and observation normalization statistics are computed from the collected dataset | Code Examples: OpenVLA/OFT `dataset_statistics.json` schema (q01/q99 bounds + mean/std/min/max, `oft_backend.py`'s consumption pattern) — compute in the same schema so Phase 6 can overlay it identically. |
| DATA-04 | Teleoperation interface allows human-controlled SOARM demonstration recording | Architecture Patterns + Code Examples: `robosuite.devices.Keyboard` + `input2action` + `DataCollectionWrapper`, confirmed directly reusable, no SOARM-specific changes needed beyond the same env construction Phase 2/3 already use. |

</phase_requirements>

## Summary

This phase builds a two-stage demonstration pipeline on top of infrastructure that is **already installed and, in one critical case, already implemented in this repo's own LIBERO fork.** Stage 1 ("collect") wraps the SOARM environment with `robosuite.wrappers.DataCollectionWrapper` to log raw MuJoCo `states` + `actions` per episode (either from a hand-coded waypoint script or a human at the keyboard). Stage 2 ("extract") replays each recorded episode by **state-setting** — using this repo's own `ControlEnv.set_init_state()` / `regenerate_obs_from_state()` helper (`LIBERO/libero/libero/envs/env_wrapper.py:136-145`), which already does exactly `sim.set_state_from_flattened(state) -> sim.forward() -> _get_observations()` — to regenerate image and proprioceptive observations frame-by-frame and write them into a robomimic/LIBERO-schema HDF5 (`obs/agentview_rgb`, `obs/eye_in_hand_rgb`, `obs/gripper_states`, `obs/joint_states`, etc.). This same `set_init_state()` primitive **is** the state-based replay mechanism DATA-02 requires — no new replay engine needs to be built, only a verification wrapper around it. No new third-party packages are required: `h5py`, `pynput` (robosuite's keyboard dependency), `robosuite`, `mujoco`, and `bddl` are already installed in the local `libero` conda environment.

For DATA-01's trajectory-generation strategy (D-01), research confirms LIBERO/robosuite ship **no built-in scripted-expert policy** for pick-place tasks — the LIBERO benchmark's own official datasets are 100% human teleoperation. The well-established community pattern for this exact situation (robosuite + OSC_POSE + object-relative pick-place) is a **hand-coded waypoint state machine**: read ground-truth object/target positions via `self.sim.data.body_xpos[self.obj_body_id[obj_name]]` (a pattern this repo's own Phase 2 code already used for reach-distance math), then sequence through approach-above -> descend -> grasp -> lift -> transport -> descend -> release phases, each phase emitting `(dx, dy, dz, droll, dpitch, dyaw, gripper)` deltas through SOARM's existing OSC_POSE controller. MimicGen (NVIDIA/ARISE) was investigated as a more sophisticated alternative but requires per-task subtask-segmentation annotations and is disproportionate engineering effort for an MVP 3-task, single-embodiment collector — noted as a considered-and-deferred alternative, not adopted.

**Critical correction to the phase brief:** the phase description and CONTEXT.md list the frozen task set as `pick_up_the_black_bowl_from_table_center...`, `...next_to_the_plate...`, and `...between_the_plate_and_the_ramekin...`. **This is stale.** Phase 2's own `02-04-SUMMARY.md` and the finalized `TASKS` constant in `explorations/soarm_sanity.py` (and reused verbatim in `LIBERO/notebooks/03a-oft-inference-eval.ipynb`, which actually ran on Colab) show `next_to_the_plate` was **swapped out** during Phase 2 — its bowl region sits 0.498 m from SOARM's base, beyond the tuned 0.479 m reach envelope — and replaced with `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate.bddl`. The **actual** 3 frozen tasks, confirmed live-run in Phase 3, are: `..._table_center...`, `..._between_the_plate_and_the_ramekin...`, and `..._on_the_ramekin...`. The planner MUST use this corrected list, not the one in the phase description/CONTEXT.md.

**Primary recommendation:** Build `LIBERO/libero/libero/datasets/` as a real Python package (mirroring `vla/`'s Protocol + graceful-degradation-import pattern) with five modules — `collector.py` (hand-coded waypoint scripted collector), `teleop.py` (keyboard teleop, reusing `DataCollectionWrapper`/`Keyboard`/`input2action` verbatim), `hdf5_writer.py` (raw-state-and-action -> obs-augmented HDF5, using `set_init_state()`), `replay.py` (DATA-02 determinism verification, same `set_init_state()` primitive), and `normalization.py` (DATA-03, OpenVLA-compatible q01/q99/mean/std stats). Target ~40 successful demos/task (120 total, 20% buffer over the 100+ requirement) from the scripted collector, plus a small (~5/task) keyboard-teleop batch proving DATA-04, merged into the same HDF5 schema.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Scripted trajectory generation (waypoint state machine) | Simulation / Local Python process | — | Pure MuJoCo physics + a Python control loop calling `env.step()`; no GPU, no network — runs in the local `libero` conda env (D-03). |
| Keyboard teleoperation input capture | Local Python process (on-screen `has_renderer=True` viewer) | — | `robosuite.devices.Keyboard` binds to the on-screen GLFW viewer's key callbacks; requires a local display (`MUJOCO_GL=glfw` on macOS per Phase 1/2 convention), not headless/Colab. |
| Raw state/action recording (episode capture) | Simulation / Local Python process | Storage (local disk, npz intermediate) | `DataCollectionWrapper` writes per-episode `.npz` state/action chunks to a local tmp directory before HDF5 gathering — identical mechanism for both scripted and teleop collection paths. |
| Observation extraction (image + proprio regeneration from recorded states) | Simulation / Local Python process | — | Requires MuJoCo `sim.forward()` + offscreen render per recorded state — CPU/software-rendered locally (glfw), no GPU dependency for this phase's 128x128 frames. |
| HDF5 dataset assembly (robomimic schema) | Storage / Local Python process | — | Pure `h5py` file writing; no simulation dependency once obs/states/actions are in memory. |
| Normalization statistics computation | Local Python process | Storage (JSON output) | Pure `numpy` reduction over the assembled HDF5's `actions`/proprio arrays; produces a `dataset_statistics.json`-shaped artifact for Phase 6 to consume, same as OFT's own checkpoint overlay pattern. |
| Downstream consumer (SequenceDataset / robomimic reader) | Phase 6 (out of scope this phase) | — | Only Phase 6's Colab-side fine-tuning kernel needs `robomimic` installed to *read* this phase's HDF5 output — this phase's writer code has no `robomimic` import dependency. |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `robosuite` | 1.4.1 (installed; project pin 1.4.0) [VERIFIED: local `pip list` in `libero` conda env] | `DataCollectionWrapper`, `VisualizationWrapper`, `devices.Keyboard`, `utils.input_utils.input2action`, OSC_POSE controller | Already the project's simulation backbone (Phase 2/3); ships the exact demonstration-collection wrapper and keyboard-teleop device this phase needs, no substitute needed. |
| `h5py` | 3.14.0 [VERIFIED: local `pip list`] | Write/read the robomimic-schema HDF5 dataset | Universal HDF5 binding used by robosuite's own `gather_demonstrations_as_hdf5` and by robomimic's `SequenceDataset` reader — format compatibility requires using the same library. |
| `pynput` | 1.8.2 [VERIFIED: local `pip list`] | Keyboard input listener (transitive dependency of `robosuite.devices.Keyboard`) | Already installed as part of robosuite's own dependency chain — no separate install needed. |
| `numpy` | 1.22.4 (project pin, per STACK.md) [CITED: `.planning/codebase/STACK.md`] | State/action arrays, normalization stat computation | Existing project pin; all HDF5 payloads and stats math are plain ndarrays. |
| `mujoco` | 2.3.7 [VERIFIED: local `pip list`] | Underlying physics engine `robosuite` drives | Existing project pin (Phase 1 environment contract); `sim.get_state()`/`set_state_from_flattened()`/`forward()` are the exact primitives this phase's replay/extraction logic uses. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `bddl` | 3.6.0 (locally installed; note STACK.md/Phase1 pin was 1.0.1 — local env resolved to 3.6.0, no incompatibility observed in Phase 2/3) [VERIFIED: local `pip list`] | Parses the 3 frozen `libero_spatial` BDDL task files for object/region ground-truth positions | Needed by the scripted waypoint collector to read `obj_of_interest`/region definitions the same way Phase 2's reach-math script did. |
| `pytest` | 8.4.2 [VERIFIED: local `pip list`] | Unit tests for collector/writer/replay/normalization logic | Matches this project's established `vla/test_*.py` pattern — mock-based, no-GPU, colocated with source. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-coded waypoint state machine (recommended, D-01) | MimicGen (NVIDIA/ARISE object-centric subtask-segmentation demo generator) | MimicGen scales far better (10-200 seed demos -> tens of thousands of variations) but requires per-task subtask-boundary annotation, a seed human-demo corpus, and non-trivial integration work against robomimic's task-interface conventions. Disproportionate for an MVP 3-task, 100-demo target. Revisit if Phase 6 fine-tuning shows a clear need for much larger/more diverse data. |
| Hand-coded waypoint state machine | VLA-rollout filtering (run OFT/π0, keep only successful rollouts) | Explicitly ruled out by CONTEXT.md — OFT/π0 are at 0% zero-shot success on SOARM (Phase 3 confirmed baseline); filtering a 0%-success rollout stream yields ~0 usable demos. Not viable until after Phase 6 fine-tuning closes some of that gap — which is circular, since Phase 6 needs this phase's dataset first. |
| `robosuite.devices.Keyboard` + `input2action` (recommended, D-02) | `robosuite.devices.SpaceMouse` | Explicitly out of scope per D-02 (no SpaceMouse hardware available). |
| This phase's own lightweight `hdf5_writer.py` (using `env.set_init_state()`) | robomimic's official `dataset_states_to_obs.py` script | robomimic is not installed in the local (Colab-free, per D-03) environment — pulling it in as a dependency just for its conversion script adds an install surface for one script whose core logic (`env.reset_to(state) -> get_observation()`) this repo's own `ControlEnv.set_init_state()` already replicates. Writing a ~100-line equivalent avoids the dependency. |

**Installation:**
No new packages required. All dependencies (`robosuite`, `h5py`, `pynput`, `numpy`, `mujoco`, `bddl`, `pytest`) are already installed in the `libero` conda environment (verified via `pip list` during this research session — see Package Legitimacy Audit below for why the gate is a no-op this phase).

**Version verification:** Verified live against the local `libero` conda environment (`/opt/homebrew/Caskroom/miniconda/base/envs/libero`) rather than a registry lookup, since every package this phase needs is already installed and pinned by Phase 1's environment contract — no new registry entries to check.

## Package Legitimacy Audit

**No new external packages are introduced by this phase.** Every library the collector/writer/replay/normalization code needs (`robosuite`, `h5py`, `pynput`, `numpy`, `mujoco`, `bddl`, `pytest`) is already installed in the local `libero` conda environment and already vetted as part of this project's Phase 1 environment contract (`01-DEBUG-HISTORY.md`) and Phase 2/3 usage. The Package Legitimacy Gate (registry/slopsquat check) is scoped to *new* package installs and is therefore a no-op for this phase.

If Phase 6 later needs `robomimic` installed (Colab-side, to *read* this phase's dataset), that install belongs to Phase 6's research/planning, not this one — this phase's own code has zero `robomimic` import dependency (confirmed: the writer only needs `h5py` + `numpy` + the LIBERO env's own `set_init_state()`).

**Packages removed due to [SLOP] verdict:** none (n/a — no new packages).
**Packages flagged as suspicious [SUS]:** none (n/a — no new packages).

## Architecture Patterns

### System Architecture Diagram

```
                         ┌─────────────────────────────┐
                         │   3 frozen libero_spatial    │
                         │   BDDL task files (Phase 2)  │
                         └──────────────┬───────────────┘
                                         │ bddl_file_name
                                         ▼
                         ┌─────────────────────────────┐
     scripted waypoints  │   OffScreenRenderEnv /       │  keyboard input
     (D-01) ────────────►│   ControlEnv (SOARM robot,   │◄──── (D-02, on-screen
     get object pos via  │   OSC_POSE controller)       │       viewer, has_
     sim.data.body_xpos  └──────────────┬───────────────┘       renderer=True)
                                         │ env.step(action) each control tick
                                         ▼
                         ┌─────────────────────────────┐
                         │  DataCollectionWrapper        │  <- reused as-is (robosuite)
                         │  (records raw states+actions  │
                         │   per episode -> .npz chunks) │
                         └──────────────┬───────────────┘
                                         │ success-filtered episodes only
                                         ▼
                         ┌─────────────────────────────┐
                         │  hdf5_writer.py (new, this    │
                         │  phase): for each recorded    │
                         │  state -> env.set_init_state()│  <- reused, already exists
                         │  (regenerate_obs_from_state)  │     in env_wrapper.py
                         │  -> obs/{agentview_rgb,        │
                         │  eye_in_hand_rgb,gripper_states│
                         │  ,joint_states} + actions+states│
                         └──────────────┬───────────────┘
                                         │ demo.hdf5 (robomimic/LIBERO schema)
                          ┌──────────────┼───────────────────┐
                          ▼                                  ▼
              ┌────────────────────┐            ┌─────────────────────────┐
              │ replay.py (DATA-02) │            │ normalization.py (DATA- │
              │ re-walk states via  │            │ 03): reduce actions/    │
              │ set_init_state,     │            │ proprio arrays -> q01/  │
              │ assert obs/states   │            │ q99/mean/std/min/max -> │
              │ identical each run  │            │ dataset_statistics.json │
              └────────────────────┘            └────────────┬────────────┘
                                                                │ consumed by
                                                                ▼
                                                   Phase 6 fine-tuning (out of scope)
```

### Recommended Project Structure
```
LIBERO/libero/libero/datasets/
├── __init__.py           # graceful-degradation exports, mirrors vla/__init__.py pattern
├── collector.py           # hand-coded waypoint scripted collector (D-01)
├── teleop.py               # keyboard teleop wrapper (D-02) — DataCollectionWrapper + Keyboard + input2action
├── raw_recorder.py         # shared DataCollectionWrapper setup used by BOTH collector.py and teleop.py
├── hdf5_writer.py          # raw npz episodes -> obs-augmented robomimic-schema demo.hdf5 (DATA-01)
├── replay.py                # state-based determinism verification (DATA-02)
├── normalization.py         # action/obs stats -> dataset_statistics.json-shaped output (DATA-03)
├── test_collector.py
├── test_hdf5_writer.py
├── test_replay.py
└── test_normalization.py
```

### Pattern 1: Two-stage collect-then-extract (raw states/actions, then obs regeneration)
**What:** Record only `states` + `actions` during the live rollout (via `DataCollectionWrapper`), then in a separate pass replay each state through `env.set_init_state()` to regenerate observations (images + proprio) and success/reward signals.
**When to use:** Always, for both the scripted collector and the teleop interface — this is robosuite's and LIBERO's own upstream convention (`collect_human_demonstrations.py` + the openvla project's `regenerate_libero_dataset.py`), and it is what makes DATA-02's state-based replay guarantee possible: the obs that end up in the HDF5 are *by construction* whatever `set_init_state()` deterministically produces from the recorded state, not whatever was captured live during a possibly-nondeterministic teleop session.
**Example:**
```python
# Source: robosuite/wrappers/data_collection_wrapper.py (installed, v1.4.1) +
# this repo's LIBERO/libero/libero/envs/env_wrapper.py:136-145 (ControlEnv.set_init_state)
from robosuite.wrappers import DataCollectionWrapper

env = DataCollectionWrapper(raw_env, tmp_directory)  # raw_env is a LIBERO OffScreenRenderEnv/ControlEnv
env.reset()
for action in scripted_or_teleop_action_stream:
    env.step(action)          # DataCollectionWrapper logs sim.get_state() + action each tick

# --- separate extraction pass, per recorded episode ---
for state in recorded_states:                      # flattened mujoco states, in order
    obs = raw_env.set_init_state(state)             # == regenerate_obs_from_state(state):
                                                      #   sim.set_state_from_flattened(state)
                                                      #   sim.forward(); check_success(); _post_process()
                                                      #   _update_observables(force=True)
                                                      #   return sim._get_observations()
    # obs["agentview_image"], obs["robot0_eye_in_hand_image"],
    # obs["robot0_gripper_qpos"], obs["robot0_joint_pos"] -> write into HDF5 obs/ group
    # renamed per LIBERO/libero/configs/data/default.yaml's obs_key_mapping convention:
    #   agentview_rgb <- agentview_image
    #   eye_in_hand_rgb <- robot0_eye_in_hand_image
    #   gripper_states <- robot0_gripper_qpos
    #   joint_states <- robot0_joint_pos
```

### Pattern 2: robomimic/LIBERO HDF5 schema (what Phase 6's SequenceDataset expects)
**What:** [VERIFIED: robomimic.github.io/docs/datasets/robosuite.html + local `LIBERO/libero/lifelong/datasets.py` `get_dataset()`] The exact on-disk structure `robomimic.utils.file_utils.get_shape_metadata_from_dataset` and `robomimic.utils.dataset.SequenceDataset` (both already imported by this repo's `LIBERO/libero/lifelong/datasets.py`) expect:
```
demo.hdf5
└── data (group)
    ├── attrs: total (int), env_args (json str: env_name/env_type/env_kwargs)
    ├── demo_0 (group)
    │   ├── attrs: num_samples (int), model_file (xml str)
    │   ├── states       (N, D)  float64  — flattened mujoco states
    │   ├── actions       (N, 7)  float64  — matches OSC_POSE 7-D action space
    │   └── obs (group)
    │       ├── agentview_rgb       (N, 128, 128, 3) uint8
    │       ├── eye_in_hand_rgb      (N, 128, 128, 3) uint8
    │       ├── gripper_states       (N, 2)  float
    │       └── joint_states          (N, 7)  float
    ├── demo_1 (group) ...
    └── mask (group, optional) — train/valid filter-key splits (not required for MVP)
```
**When to use:** This is the target schema for `hdf5_writer.py`'s output. The `obs/` key names (`agentview_rgb`, `eye_in_hand_rgb`, `gripper_states`, `joint_states`) are LIBERO's own renamed convention [CITED: `LIBERO/libero/configs/data/default.yaml` `obs_key_mapping`, cross-confirmed via WebSearch against published LIBERO/HF dataset key listings] — **not** the raw robosuite observation-dict key names (`agentview_image`, `robot0_eye_in_hand_image`, `robot0_gripper_qpos`, `robot0_joint_pos`). The writer must rename at write time using the same mapping `default.yaml` already documents, or Phase 6's `get_dataset()` call will fail to find the expected keys.
**Example:**
```python
# Source: LIBERO/libero/lifelong/datasets.py get_dataset() (already in this repo)
# and LIBERO/libero/configs/data/default.yaml obs_key_mapping (already in this repo)
OBS_KEY_MAPPING = {
    "agentview_rgb": "agentview_image",
    "eye_in_hand_rgb": "robot0_eye_in_hand_image",
    "gripper_states": "robot0_gripper_qpos",
    "joint_states": "robot0_joint_pos",
}
```

### Pattern 3: Hand-coded waypoint scripted collector (D-01 resolution)
**What:** A finite state machine driving SOARM's OSC_POSE controller through phases: `approach_above_object -> descend -> close_gripper -> lift -> transport_above_target -> descend -> open_gripper -> retreat`, computing target deltas from live ground-truth object/plate positions.
**When to use:** For all 3 frozen `libero_spatial` bowl-to-plate tasks — structurally identical (pick one object, place on plate), so one parameterized state machine covers all 3 by looking up each task's object-of-interest and target region via BDDL parsing (same `obj_of_interest` / `body_xpos` pattern Phase 2's `02-04` reach-distance script already used).
**Example:**
```python
# Pattern synthesized from: robosuite OSC_POSE controller action-space docs
# [CITED: robosuite.ai/docs — OSC_POSE without gripper = (dx,dy,dz,droll,dpitch,dyaw)]
# and community expert-pick-place structure [CITED: github.com/SudeepDasari/
# one_shot_transformers/blob/master/hem/robosuite/controllers/expert_pick_place.py
# — approach/grasp/lift/transport phase structure, LOW confidence / third-party,
# adapt rather than copy verbatim]
# and this repo's own ground-truth object-position access pattern:
# LIBERO/libero/libero/envs/bddl_base_domain.py:512
#   self.sim.data.body_xpos[self.obj_body_id[obj_name]]

bowl_pos = env.sim.data.body_xpos[env.obj_body_id["akita_black_bowl_1"]]
eef_pos = obs["robot0_eef_pos"]
delta = clip(bowl_pos - eef_pos + [0, 0, clearance], max_step=0.05)
action = np.concatenate([delta, [0, 0, 0], [gripper_state]])  # 7-D OSC_POSE action
```

### Pattern 4: Keyboard teleop, reused verbatim from robosuite (D-02 resolution)
**What:** `robosuite.devices.Keyboard` + `robosuite.utils.input_utils.input2action`, bound to an on-screen `VisualizationWrapper`'d env's GLFW viewer key callbacks — exactly the pattern `LIBERO/scripts/collect_demonstration.py` already implements for Panda.
**When to use:** DATA-04's human-teleop interface. **No SOARM-specific changes are needed** — `input2action` reads generic robot/controller state (`active_robot`, `controller.input_max/min`), not anything Panda-specific; SOARM's `Soarm101` robot class (Phase 2) already exposes the same `ManipulatorModel` interface `input2action` expects.
**Example:**
```python
# Source: robosuite/scripts/collect_human_demonstrations.py (installed, v1.4.1) —
# already proven working for Panda in this repo's LIBERO/scripts/collect_demonstration.py
from robosuite.devices import Keyboard
from robosuite.utils.input_utils import input2action

device = Keyboard(pos_sensitivity=1.0, rot_sensitivity=1.0)
env.viewer.add_keypress_callback("any", device.on_press)
env.viewer.add_keyup_callback("any", device.on_release)
env.viewer.add_keyrepeat_callback("any", device.on_press)
device.start_control()
action, grasp = input2action(device=device, robot=env.robots[0], active_arm="right",
                              env_configuration="single-arm-opposed")
```
**Constraint:** requires a local on-screen GLFW viewer (`has_renderer=True`), so this runs locally (D-03), not on headless Colab — same `MUJOCO_GL=glfw` macOS pattern established in Phase 2 (D-09).

### Anti-Patterns to Avoid
- **Recording observations live during the rollout instead of regenerating them via state-setting:** breaks the state-based determinism guarantee DATA-02 requires — live-captured frames during a teleop session are not reproducible from the recorded `states` array alone if any part of the capture pipeline (e.g. camera timing, render-buffer reuse) is nondeterministic. Always regenerate obs from the recorded state, never trust live-captured obs as the dataset-of-record.
- **Action-replay as the "verification" for DATA-02:** `robosuite/scripts/playback_demonstrations_from_hdf5.py`'s own `--use-actions` code path prints an explicit divergence warning (`playback diverged by {err} for ep {ep} at step {j}`) precisely because re-stepping actions through the physics engine is *not* guaranteed to reproduce the original states bit-for-bit (contact solver nondeterminism, floating-point accumulation). This is the exact failure mode LIBERO issue #16 warns about — do not use `env.step(action)` loops as the DATA-02 verification method, only `sim.set_state_from_flattened(state)` / `set_init_state(state)`.
- **Writing raw robosuite obs key names (`agentview_image`, `robot0_gripper_qpos`, ...) directly into the HDF5:** Phase 6's `get_dataset()` call resolves `all_obs_keys` from LIBERO's renamed `obs_modality` config (`agentview_rgb`, `gripper_states`, ...) and looks those exact names up in the HDF5's `obs/` group — raw robosuite names will cause a `KeyError`/shape-metadata failure at Phase 6 load time, not this phase's write time, so the bug would be silent until Phase 6.
- **Importing `robomimic` in this phase's writer code:** unnecessary (not installed locally, D-03 keeps this phase Colab-free) and unneeded — `h5py` + this repo's own `set_init_state()` fully replicate the one script (`dataset_states_to_obs.py`) robomimic would otherwise be pulled in for.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Recording raw sim states/actions per episode | A custom step-logger | `robosuite.wrappers.DataCollectionWrapper` | Already handles episode directory bookkeeping, flush-to-disk cadence, first-interaction timing edge cases (e.g. not logging until the first real `step()`, not on `reset()`) — reimplementing this correctly is exactly the kind of subtle-bug surface this rule exists for. |
| State-based deterministic replay engine | A new `sim.set_state(...)`-wrapping module | `ControlEnv.set_init_state()` / `regenerate_obs_from_state()` (`LIBERO/libero/libero/envs/env_wrapper.py:136-145`, already in this repo) | This exact method already exists, already does `set_state_from_flattened -> forward -> check_success -> _post_process -> _update_observables -> _get_observations` in the correct order — writing a second implementation risks it subtly diverging from what LIBERO's own eval/replay code already relies on elsewhere. |
| Keyboard-to-action-space mapping | A custom key-binding layer | `robosuite.devices.Keyboard` + `input2action` | Already handles rotation-matrix accumulation, position deltas, grasp toggling, and reset signaling in a robot/controller-agnostic way; SOARM's controller config (OSC_POSE, Phase 2/3) is exactly what `input2action` expects. |
| HDF5-gathering from per-episode npz chunks | A custom h5py writer from scratch | Adapt `gather_demonstrations_as_hdf5()` (robosuite's/LIBERO's own, `LIBERO/scripts/collect_demonstration.py:104-193`) | The state/action alignment bug this function already handles correctly (`del states[-1]` — DataCollectionWrapper logs states *after* playing an action, producing one extra trailing state) is a one-line-easy, one-line-subtle bug to reintroduce from scratch. |
| Action/obs normalization statistics format | A bespoke stats JSON schema | OpenVLA's `dataset_statistics.json` schema (q01/q99 bounds, mean/std/min/max, per-dim mask) — same shape `oft_backend.py` already consumes | DATA-03 explicitly exists to feed Phase 6's fine-tuning. Matching OFT's own checkpoint-overlay schema means Phase 6 can literally reuse `oft_backend.py`'s existing `json.load()` + dict-key overlay pattern instead of writing a second stats-loading code path. |

**Key insight:** This phase's single biggest risk is *not* algorithmic (the waypoint controller) but *format-compatibility*: every piece of infrastructure needed already exists somewhere in this repo or its installed dependencies (robosuite's collection wrapper, this repo's own `set_init_state`, LIBERO's own key-renaming convention). The work is assembling these correctly and matching the exact schema Phase 6 expects — not building new simulation or replay machinery.

## Common Pitfalls

### Pitfall 1: Using the stale task-name list from the phase brief
**What goes wrong:** Building the collector against `pick_up_the_black_bowl_next_to_the_plate...` — a task that was explicitly swapped out in Phase 2 because it exceeds SOARM's reach envelope — would produce a collector that either can't complete the task (unreachable object) or silently succeeds against a task nobody else in this project's pipeline (Phase 3, Phase 5, Phase 6) is using.
**Why it happens:** CONTEXT.md and the phase description were written from the *original* Phase 2 candidate list (`02-01-PLAN.md`), before Phase 2's D-03 reach-math resolution swapped one task out. This is a stale-reference bug in the planning artifacts themselves, not a research gap.
**How to avoid:** Use the corrected list: `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl`, `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl`, `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate.bddl` — copy the `TASKS` constant directly from `explorations/soarm_sanity.py` (lines 65-69) or `LIBERO/notebooks/03a-oft-inference-eval.ipynb`, both already correct.
**Warning signs:** Any grep for `next_to_the_plate` inside new Phase 4 code should return zero hits.

### Pitfall 2: Recorded episode has one extra trailing state
**What goes wrong:** `DataCollectionWrapper` appends a state to its internal buffer both on `_start_new_episode()` (the initial state) and after every `collect_freq`-th `step()` — meaning the states array ends up one entry longer than the actions array (the last state was recorded *after* the last action executed, with no corresponding "next action" yet).
**Why it happens:** This is `DataCollectionWrapper`'s own documented behavior (see `data_collection_wrapper.py`'s `_on_first_interaction`/`step` methods) — not a bug in the wrapper, but an alignment detail any consumer must handle.
**How to avoid:** Both robosuite's and this repo's LIBERO's `gather_demonstrations_as_hdf5()` already handle this correctly with `del states[-1]` before writing — replicate that exact line, and assert `len(states) == len(actions)` immediately after, exactly as the existing prior-art code does.
**Warning signs:** `assert len(states) == len(actions)` failing during HDF5 assembly.

### Pitfall 3: Only successful episodes should ever reach the final HDF5
**What goes wrong:** If failed/incomplete scripted or teleop attempts get written into `demo.hdf5`, Phase 6's fine-tuning would train on non-task-completing trajectories, degrading the fine-tune.
**Why it happens:** Easy to forget when scripting a "collect N episodes" loop that not every attempt succeeds (scripted waypoint failures from tuned-controller edge cases, human teleop resets/retries).
**How to avoid:** Gate on `DataCollectionWrapper.successful` (set internally via `env._check_success()` during the episode) exactly as robosuite's `collect_human_demonstrations.py`'s `gather_demonstrations_as_hdf5` does (`if success: ... else: print("unsuccessful ... NOT saved")`) — loop "attempt until N successes" per task, not "attempt N times."
**Warning signs:** Episode count in the final HDF5 doesn't match the number of `env.reset()` calls made during collection — expected and fine, as long as the *successful* count meets the D-04 target.

### Pitfall 4: Writing raw robosuite obs keys instead of LIBERO's renamed convention
**What goes wrong:** Silent failure at Phase 6 load time (`robomimic.utils.file_utils.get_shape_metadata_from_dataset` looks for `agentview_rgb` etc., not `agentview_image`), not at this phase's write/verify time — so this phase's own tests would pass while quietly producing an unusable dataset for Phase 6.
**Why it happens:** `env._get_observations()` returns robosuite's native key names; LIBERO's own config-level renaming (`obs_key_mapping` in `default.yaml`) is easy to miss if you don't specifically go looking for how LIBERO's official datasets differ from raw robosuite output.
**How to avoid:** Apply the `OBS_KEY_MAPPING` rename (Pattern 2 above) at HDF5-write time, and add a unit test that opens the produced HDF5 and asserts `"agentview_rgb" in demo["obs"].keys()` (not the raw name) for at least one demo.
**Warning signs:** None visible locally — this is exactly why the explicit unit test matters; there is no natural failure signal until Phase 6 runs.

### Pitfall 5: Assuming replay-with-image-regeneration is cheap at scale
**What goes wrong:** `set_init_state()` calls `sim.forward()` + a full offscreen render per state — for ~120 demos x ~100-300 steps each, a full-dataset image-obs replay is meaningfully slower than a states-only equality check. Underestimating this can blow past a "just run it locally" assumption if the collector script is naively re-run per verification.
**Why it happens:** The *states-only* replay check (no image regeneration, just `set_state_from_flattened` + compare flattened state vectors) is fast; the *full obs regeneration* replay (used to build the HDF5 in the first place) is render-bound and much slower. Conflating the two costs leads to underestimating DATA-02 verification runtime.
**How to avoid:** Size DATA-02's verification per Validation Architecture below — do the cheap full-dataset states-equality check on every demo, and reserve the expensive full-image-regeneration check for a smaller sample.
**Warning signs:** Verification step taking dramatically longer than collection itself.

## Code Examples

### State-based replay (the exact DATA-02 mechanism)
```python
# Source: robosuite/scripts/playback_demonstrations_from_hdf5.py (installed, v1.4.1),
# state-based branch (the `else:` branch, not `--use-actions`)
for state in states:
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    # (this repo's set_init_state() additionally calls check_success(),
    #  _post_process(), _update_observables(force=True) before returning obs —
    #  prefer set_init_state() over the raw two-liner for the same reason LIBERO's
    #  own eval code uses it: observables must be force-refreshed after a state jump)
```

### Determinism verification pattern (adapt the above into an assertion)
```python
# Synthesized from the state-based replay pattern above + this repo's own
# env_wrapper.py set_init_state/regenerate_obs_from_state (LOCAL, VERIFIED)
def verify_episode_determinism(env, states, recorded_obs, atol=0.0):
    for i, state in enumerate(states):
        obs = env.set_init_state(state)
        for key, recorded in recorded_obs[i].items():
            assert np.allclose(obs[key], recorded, atol=atol), (
                f"step {i} key {key} diverged on replay"
            )
```

### OpenVLA-style normalization stats (DATA-03 target schema)
```python
# Source: this repo's LIBERO/libero/libero/vla/oft_backend.py (already consumes
# this exact schema) + OpenVLA project's public dataset_statistics.json convention
# [CITED: github.com/moojink/openvla-oft — BOUNDS_Q99 normalization: maps [q01,q99] -> [-1,1]]
import numpy as np

def compute_norm_stats(actions: np.ndarray, proprio: np.ndarray) -> dict:
    # actions: (total_steps, 7), proprio: (total_steps, D)
    def stats(x):
        return {
            "mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
            "max": x.max(0).tolist(), "min": x.min(0).tolist(),
            "q01": np.quantile(x, 0.01, axis=0).tolist(),
            "q99": np.quantile(x, 0.99, axis=0).tolist(),
        }
    return {
        "soarm_spatial": {  # dataset key — Phase 6 overlays this the same way
                              # oft_backend.py overlays "libero_spatial_no_noops"
            "action": stats(actions),
            "proprio": stats(proprio),
            "num_transitions": int(actions.shape[0]),
            "num_trajectories": None,  # fill with actual demo count
        }
    }
```

### Correct frozen task list (copy verbatim, do not use CONTEXT.md's list)
```python
# Source: explorations/soarm_sanity.py lines 65-69 (this repo, already correct)
TASKS = [
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl",
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate.bddl",
]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Single-source human teleoperation only (LIBERO's original 2023 dataset methodology) | Mixed scripted + teleop collection, or fully synthetic (MimicGen-style) generation | Ongoing since ~2023 (MimicGen, CoRL 2023) | This phase deliberately takes the middle path (hand-coded scripted + a small teleop proof batch) rather than either extreme — appropriate for a 3-task MVP, revisit if Phase 6 needs more data diversity. |
| mean/std (z-score) action normalization | Percentile-based (q01/q99) bounds normalization ("BOUNDS_Q99") | OpenVLA / RLDS-era VLA training (2024) | This phase should compute q01/q99 (not just mean/std) specifically because Phase 6 fine-tunes OpenVLA-OFT, which expects this exact normalization convention — using plain z-score stats would require an extra conversion step Phase 6 shouldn't have to do. |

**Deprecated/outdated:**
- LIBERO's own bundled `LIBERO/scripts/collect_demonstration.py` / `libero_100_collect_demonstrations.py` are explicitly not being extended in place (D-05) — kept only as reference for the device-input pattern, not as a base to modify.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | ~40 successful demos/task (120 total) is a reasonable D-04 target | Standard Stack / Summary | Low — this is a quantity decision with wide tolerance (any number >= ~34/task keeps the 100+ total); if wrong, simply collect more/fewer, no architectural rework needed. |
| A2 | Hand-coded waypoint state machine will reliably reach >0% success on all 3 tasks given SOARM's tuned kinematics | Architecture Patterns Pattern 3 | Medium — unlike a trained policy, a hand-coded controller's success depends on correctly tuning phase thresholds/clearances per task; if SOARM's gripper/reach geometry makes one task's grasp geometry awkward, that task may need per-task waypoint tuning (expect iteration, similar to Phase 2's camera-tuning-by-iteration pattern) rather than a single universal script working first try. |
| A3 | `dataset_statistics.json`'s exact top-level key structure (mask field, num_trajectories, per-key nesting) matches what Phase 6's fine-tuning code will expect | Code Examples / DATA-03 | Medium — the schema was reconstructed from WebSearch summaries of OpenVLA's normalization convention and this repo's own `oft_backend.py` consumption code, not from directly reading OpenVLA-OFT's training-side stats-computation source. Phase 6's research should re-verify the exact schema against `moojink/openvla-oft`'s training data-loading code before Phase 6 locks in its normalization-overlay logic. |
| A4 | `bddl` package version mismatch (STACK.md says 1.0.1, local env has 3.6.0) causes no incompatibility | Standard Stack | Low — Phase 2/3 already ran successfully against this same locally-installed 3.6.0, so this is an observed-working state, not a fresh unknown; flagged only because it contradicts the written STACK.md pin. |

**If this table is empty:** N/A — see rows above.

## Open Questions

1. **Exact per-task grasp/waypoint tuning parameters for the scripted collector**
   - What we know: SOARM's OSC_POSE controller, reach envelope (0.479 m), base offset (-0.38/0/0.90), and gripper actuator ranges are all frozen from Phase 2. The general waypoint-phase structure (approach/descend/grasp/lift/transport/release) is well-established community practice.
   - What's unclear: Exact clearance heights, approach angles, and per-task success thresholds will need empirical tuning once the collector runs against SOARM's actual tuned physics — this is implementation-time iteration, not something further research can pre-resolve.
   - Recommendation: Budget iteration time in planning (similar to Phase 2's "1-2 days of iterative MJCF editing" budget flag) — plan for a tuning loop (run collector -> inspect failure mode -> adjust waypoint parameters -> re-run), not a single-shot script.

2. **Whether Phase 6's exact fine-tuning code expects `num_trajectories` / additional keys in `dataset_statistics.json`**
   - What we know: The q01/q99/mean/std/min/max per-action-dim shape is confirmed (A3 above).
   - What's unclear: Whether Phase 6's specific fine-tuning script reads additional metadata fields from this JSON beyond what `oft_backend.py`'s inference-time overlay already reads.
   - Recommendation: Phase 6's own research step should re-confirm the exact schema against `moojink/openvla-oft`'s *training*-side (not just inference-side) data loading code before finalizing; this phase's `normalization.py` should be written to make adding extra fields trivial (a plain dict returned, easy to extend).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `robosuite` | Collector, teleop, HDF5 writer, replay | Yes | 1.4.1 | — |
| `h5py` | HDF5 writer, replay | Yes | 3.14.0 | — |
| `pynput` | Keyboard teleop device | Yes | 1.8.2 | — |
| `mujoco` | Physics engine (via robosuite) | Yes | 2.3.7 | — |
| `bddl` | Task-file parsing | Yes | 3.6.0 | — |
| `pytest` | Unit tests | Yes | 8.4.2 | — |
| `robomimic` | Only needed to *read* this phase's output (Phase 6) | No (local env) | — | Not required this phase — this phase's writer code has no robomimic dependency (see Standard Stack alternatives). Phase 6 installs it Colab-side when needed. |
| On-screen GLFW display | Keyboard teleop (`has_renderer=True`) | Assumed Yes (macOS, per Phase 2 D-09's established `MUJOCO_GL=glfw` local pattern) | — | If the local machine lacks a display (e.g. running headless in CI), teleop collection cannot run there — scripted collection is unaffected (headless-compatible via `OffScreenRenderEnv`). |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** `robomimic` (not needed this phase; Phase 6's concern).

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.4.2 (installed, verified) |
| Config file | none — rootdir-based collection, same as `LIBERO/libero/libero/vla/test_*.py`'s established pattern |
| Quick run command | `pytest LIBERO/libero/libero/datasets/ -x` (from repo root) |
| Full suite command | `pytest LIBERO/libero/libero/datasets/ LIBERO/libero/libero/vla/ -v` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | Scripted collector produces valid robomimic-schema HDF5 with image obs present | integration (real local sim, small N) | `pytest LIBERO/libero/libero/datasets/test_hdf5_writer.py -x` | ❌ Wave 0 |
| DATA-01 | HDF5 obs group uses LIBERO's renamed keys (`agentview_rgb` etc.), not raw robosuite keys | unit | `pytest LIBERO/libero/libero/datasets/test_hdf5_writer.py::test_obs_key_naming -x` | ❌ Wave 0 |
| DATA-02 | Full-dataset cheap states-only replay determinism check | integration (real local sim, all demos) | `pytest LIBERO/libero/libero/datasets/test_replay.py::test_all_demos_state_replay -x` | ❌ Wave 0 |
| DATA-02 | Sampled full-image-regeneration replay determinism check | integration (real local sim, sampled subset) | `pytest LIBERO/libero/libero/datasets/test_replay.py::test_sampled_image_replay -x` | ❌ Wave 0 |
| DATA-03 | Normalization stats computed match OpenVLA q01/q99/mean/std schema, non-degenerate (not NaN, q01<q99) | unit (synthetic array input) | `pytest LIBERO/libero/libero/datasets/test_normalization.py -x` | ❌ Wave 0 |
| DATA-04 | Teleop path produces demos in the identical HDF5 schema as the scripted path | manual + integration (requires human at keyboard for the live-input part; schema check is automatable) | `pytest LIBERO/libero/libero/datasets/test_hdf5_writer.py::test_schema_matches_across_sources -x` (schema) + manual keyboard session (live input, not automatable) | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest LIBERO/libero/libero/datasets/ -x` (fast subset — unit tests + a 1-2 episode integration smoke test, not the full 100+-demo collection run)
- **Per wave merge:** Full suite + a real (small, e.g. 3-5 episode) end-to-end collect -> write -> replay -> normalize run against the actual SOARM env
- **Phase gate:** Full 100+-demo collection run, full-dataset states-only replay check, sampled image-regeneration replay check, all green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `LIBERO/libero/libero/datasets/__init__.py` — package scaffold, mirroring `vla/__init__.py`'s graceful-degradation-import pattern
- [ ] `LIBERO/libero/libero/datasets/test_hdf5_writer.py` — covers DATA-01
- [ ] `LIBERO/libero/libero/datasets/test_replay.py` — covers DATA-02
- [ ] `LIBERO/libero/libero/datasets/test_normalization.py` — covers DATA-03
- [ ] No new pytest framework install needed — already present locally

### D-06 resolution: replay verification sample size
Given local (not Colab) execution and the cost asymmetry identified in Pitfall 5 (cheap states-only check vs. expensive image-regeneration check):
- **Full replay (100% of demos):** states-only equality check (`sim.set_state_from_flattened` + compare flattened state vectors, no rendering) — cheap, no reason not to run on every demo, gives DATA-02's core "deterministic on re-run" guarantee complete coverage.
- **Sampled replay (~10%, minimum 5 demos, at least 1 per task):** full observation regeneration (images + proprio) replay, comparing against what was written into the HDF5 at collection time — this is the more expensive check (rendering-bound); a 10% sample is enough to catch a systematic obs-extraction bug (e.g. Pitfall 4's key-naming issue, or a camera/rendering nondeterminism) without paying full-dataset rendering cost twice.
- This directly resolves D-06 with a concrete, justified split rather than an arbitrary percentage.

## Security Domain

This phase is 100% local, single-user, non-networked simulation code with no user-facing input surface beyond a local keyboard during teleop sessions and local BDDL/HDF5 file paths supplied by the developer. Most ASVS categories do not apply.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V1 Architecture | No | No network/service boundary introduced this phase. |
| V2 Authentication | No | No auth surface — local research script. |
| V3 Session Management | No | N/A. |
| V4 Access Control | No | N/A — single local user, local filesystem. |
| V5 Input Validation | Marginal | BDDL task-file paths and HDF5 output paths are developer-supplied constants (the 3 frozen task filenames), not external/untrusted input — validate with `os.path.exists()` (existing project convention, see `env_wrapper.py`'s `assert os.path.exists(bddl_file_name)`) rather than any parsing/sanitization framework. |
| V6 Cryptography | No | No secrets, no encrypted data — HDF5 datasets are local research artifacts, not sensitive. |
| V7 Error Handling & Logging | Marginal | HDF5 write failures (disk full, partial episode) should fail loudly (matches this project's established "fail loudly, do not swallow" convention from `01-DEBUG-HISTORY.md`), not silently produce a truncated/corrupt dataset. |

### Known Threat Patterns for this stack
None applicable — no network exposure, no multi-tenant data, no authentication, no user-submitted content. The only "threat" in scope is data-integrity (a corrupt/incomplete HDF5 silently poisoning Phase 6's fine-tuning), addressed via the fail-loudly convention and the DATA-02 replay-verification tests above, not via ASVS-style security controls.

## Sources

### Primary (HIGH confidence)
- Local codebase, directly read this session: `LIBERO/libero/libero/envs/env_wrapper.py` (`ControlEnv.set_init_state`/`regenerate_obs_from_state`), `LIBERO/libero/libero/envs/bddl_base_domain.py` (`body_xpos` ground-truth object position pattern, `get_object`), `LIBERO/libero/lifelong/datasets.py` (`get_dataset`), `LIBERO/libero/configs/data/default.yaml` (`obs_key_mapping`), `LIBERO/scripts/collect_demonstration.py`, `explorations/soarm_sanity.py` (corrected `TASKS` constant), `LIBERO/libero/libero/vla/{interface,oft_backend,__init__}.py` (module-structure pattern to mirror).
- Local installed-package source, directly read this session: `robosuite` 1.4.1 (`wrappers/data_collection_wrapper.py`, `devices/keyboard.py`, `scripts/collect_human_demonstrations.py`, `scripts/playback_demonstrations_from_hdf5.py`) — all read from `/opt/homebrew/Caskroom/miniconda/base/envs/libero/lib/python3.9/site-packages/robosuite/`.
- `.planning/phases/02-soarm-robot-integration/02-04-SUMMARY.md` — source of the corrected task list and the 0.479 m reach envelope.
- `.planning/phases/03-vla-inference-loop/03-01-SUMMARY.md`, `03-02-PLAN.md` — source of the confirmed `MAX_STEPS_DEFAULT = 600` value.

### Secondary (MEDIUM confidence)
- [robomimic 0.5 docs — robosuite Datasets](https://robomimic.github.io/docs/datasets/robosuite.html) — HDF5 top-level/per-demo schema (`total`, `env_args`, `num_samples`, `model_file`, `states`, `actions`), `dataset_states_to_obs.py` purpose. [CITED]
- [robomimic `dataset_states_to_obs.py` source (GitHub)](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/scripts/dataset_states_to_obs.py) — CLI args, `{camera_name}_image` obs-key convention, `done_mode` semantics. [CITED]
- [openvla `regenerate_libero_dataset.py` (GitHub)](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py) — confirms LIBERO's official obs key renaming (`agentview_rgb`, `eye_in_hand_rgb`, `gripper_states`, `joint_states`, `ee_pos`, `ee_ori`, `ee_states`) and `env.set_init_state()`-based replay pattern. [CITED]
- [moojink/openvla-oft `modeling_prismatic.py` (GitHub)](https://github.com/moojink/openvla-oft/blob/main/prismatic/extern/hf/modeling_prismatic.py) and WebSearch summary of OpenVLA's normalization convention — `BOUNDS_Q99` q01/q99 percentile normalization. [CITED]
- [one_shot_transformers `expert_pick_place.py` (GitHub)](https://github.com/SudeepDasari/one_shot_transformers/blob/master/hem/robosuite/controllers/expert_pick_place.py) — community scripted-expert state-machine structure (approach/grasp/lift/transport phases). Third-party, not official robosuite/LIBERO tooling — adapt, don't copy. [CITED, LOW-MEDIUM]
- MimicGen (arXiv:2310.17596, CoRL 2023) — object-centric subtask-segmentation demo-generation system, considered and deferred as disproportionate for this MVP. [CITED]
- [robosuite 1.5 docs — Human Demonstrations](https://robosuite.ai/docs/algorithms/demonstrations.html) — general demonstration-collection overview. [CITED]

### Tertiary (LOW confidence)
- WebSearch-derived LIBERO published-dataset HDF5 key listings (HuggingFace dataset cards for LIBERO-Mem, LIBERO-Cosmos-Policy) — used only to cross-confirm the `obs_key_mapping` renamed-key convention already found in this repo's own `default.yaml`; not independently authoritative. [ASSUMED, cross-checked against local repo config which is HIGH confidence]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every package is already installed and verified locally; no new dependency decisions.
- Architecture (HDF5 schema, state-based replay): HIGH — cross-confirmed between local repo code, installed robosuite source, and robomimic's public docs/scripts.
- Scripted trajectory generation (D-01): MEDIUM — the general waypoint-state-machine pattern is well-established community practice, but no official LIBERO/robosuite-shipped scripted expert exists for these exact tasks; exact tuning parameters are implementation-time unknowns (Open Question 1).
- Pitfalls: HIGH — sourced directly from reading the actual wrapper/script source code (DataCollectionWrapper's off-by-one, playback script's own divergence warning), not inferred.
- Episode-count/replay-sample-size discretion (D-04, D-06): MEDIUM — reasoned recommendations with explicit tradeoffs, not externally-verified "correct" numbers (none exist for a novel single-embodiment MVP dataset).

**Research date:** 2026-08-02
**Valid until:** 2026-09-01 (30 days — stable local-tooling domain, low churn risk; re-verify sooner only if Phase 2/3 task-list or reach-envelope values change)
