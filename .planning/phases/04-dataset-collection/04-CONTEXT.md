# Phase 4: Dataset Collection - Context

**Gathered:** 2026-08-02
**Status:** In execution — PARKED pending gripper-upgrade prerequisite (see ⚠ AMENDMENT below)
**Amended:** 2026-08-03 (gripper upgrade + task retargeting locked)

<domain>
## Phase Boundary

A scripted and teleoperated demonstration collection system that records SOARM task completions as robomimic HDF5 datasets, with state-based (not action-based) deterministic replay and SOARM-specific action/observation normalization statistics — computed from the collected dataset, not copied from Panda. Runs against the 3 frozen `libero_spatial` bowl→plate tasks from Phase 2 (`pick_up_the_black_bowl_from_table_center...`, `...next_to_the_plate...`, `...between_the_plate_and_the_ramekin...`). Spatial awareness (multi-camera, depth, spatial-language tasks — Phase 5) and fine-tuning (Phase 6) are explicitly out of scope here; this phase only produces the dataset those later phases will consume.

</domain>

<amendment>
## ⚠ AMENDMENT (2026-08-03) — Gripper upgrade + task retargeting (READ FIRST)

Execution of Phase 4 hit a hard embodiment blocker that changes the robot and the target task. This amendment **supersedes** the stale bowl→plate references in the domain paragraph above and in D-01.

**What happened:** The scripted collector code (D-01/D-05) is built and works, but the **stock SOARM gripper (~2–3 cm jaw) physically cannot grasp ANY LIBERO object** — the narrowest object across all suites is `butter` at 4.0 cm; bowls are 8–16 cm. Empirically confirmed (~90 grasp trials, 0 successes; attribution = jaw size). This is faithful to the small real stock gripper. Retargeting to a different LIBERO object cannot fix it, and LIBERO-PRO is an eval-perturbation suite, not a source of small graspable objects.

**Locked decisions (user, 2026-08-03):**
- **D-07 (Gripper upgrade — LOCKED):** Replace the stock gripper with the open-source **roboninecom SO-ARM100/101 parallel gripper** — real, printable (STEP + STL), 120–150 N grip, same Feetech STS3215 servo. Modeled into `SoarmGripper` at the **faithful 84 mm stroke** (both slide jaws `0..0.042`; committed `ad0b0d0`). The arm is unchanged; `SoarmGripper` class name + `-1=open/+1=closed` contract preserved. **84 mm is the practical ceiling for this arm** — no wider ready-made gripper exists and the arm payload is only ~500 g, so the 11 cm bowl is beyond this robot class. Target **sub-84 mm objects** (cans/boxes ~4–8 cm), NOT the bowl.
- **D-08 (Task retargeting — LOCKED):** Drop the 3 frozen `libero_spatial` bowl→plate tasks. They fail on two independent counts: (a) the bowl is too big for the 84 mm jaw, and (b) the plate place-target sits ~0.5 m out, beyond the arm's ~0.45 m reach. Instead, use a **sub-84 mm object in a pick-place task whose object-init AND place-target both sit within the arm's ~0.45 m reach** — authoring or modifying a BDDL to place them in-reach (LIBERO's task system supports this; see `custom_object_example.ipynb`). Keep pick-place structure (feeds Phase 5 spatial variants).

**Pending on resume (spend-blocked as of 2026-08-03):** (1) author + empirically validate the sub-84 mm in-reach task at the faithful 84 mm gripper (grasp+lift+place via the FSM); (2) retarget `collector.py`'s `TASKS` to that BDDL; (3) re-run the collection (04-02's collection step — its existing SUMMARY documents the BLOCKER, not completion, so the collection must actually run); (4) proceed to Waves 3–4. See `.claude/.../memory/soarm-gripper-embodiment-blocker.md` and `soarm-gripper-mjcf-modeling-notes.md` for full technical carry-forward, and STATE.md Session Continuity for the exact resume sequence.

</amendment>

<decisions>
## Implementation Decisions

### Scripted Demo Generation
- **D-01:** *(Partially SUPERSEDED by D-07/D-08 in the amendment above — the "3 frozen bowl tasks" target is replaced by a sub-84 mm in-reach task; the hand-coded waypoint FSM approach itself was chosen and built, and is retained.)* How the scripted collector generates successful trajectories for the 3 frozen tasks (hand-coded waypoint/IK script vs. scripted + randomized noise/placement vs. VLA-rollout filtering) is **Claude's discretion** — research LIBERO's existing demo-generation conventions and SOARM's tuned kinematics (Phase 2) before choosing. Note: OFT/π0 currently sit at 0% zero-shot success on SOARM (Phase 3 baseline), so VLA-rollout filtering is unlikely to be viable without further work — a hand-coded or noise-augmented waypoint approach is the more realistic default unless research finds a fast fix.

### Teleoperation Interface
- **D-02:** Prioritize **keyboard-only** input for the teleoperation interface (DATA-04). No SpaceMouse hardware dependency — reuses robosuite's existing keyboard `Device` class. SpaceMouse support is not required for this phase.

### Demo Scope & Execution Environment
- **D-03:** Demos are collected with a roughly **even split across the 3 tasks** and demo **collection runs locally**, not on Colab — this is CPU-bound MuJoCo sim work (not VLA inference), so no GPU/Colab dependency is needed, and local execution gives faster iteration on the collector itself. This is a departure from Phase 1-3's Colab-first convention, scoped specifically to this phase's collection step (not necessarily downstream training/fine-tuning in Phase 6).
- **D-04:** Exact episode count per task (beyond "roughly even" across 3 tasks, totaling 100+) is **Claude's discretion**.

### Module Location
- **D-05:** New dataset-collection code lives in a new **`LIBERO/libero/libero/datasets/`** package — mirrors the Phase 3 pattern of `LIBERO/libero/libero/vla/` as a real importable module. Phase 6 (fine-tuning) must be able to import HDF5-loading and normalization-stats utilities directly from here, not from notebook-only code. This is new code, not a modification of the existing `LIBERO/scripts/collect_demonstration.py` / `libero_100_collect_demonstrations.py` (those are human-teleop-only, npz-based, and not robomimic-HDF5 — kept as reference/prior art, not extended in place).

### State-Based Replay Verification (DATA-02)
- **D-06:** Whether to spot-check a sample of demos (e.g. ~10%) or replay every single recorded demo for state-based determinism verification is **Claude's discretion** — size the verification sample based on measured per-replay runtime once the collector exists, given collection runs locally (D-03).

### Claude's Discretion (summary)
- D-01: Scripted trajectory generation strategy (hand-coded waypoints vs. noise-augmented vs. VLA-filtered).
- D-04: Exact episode count per task.
- D-06: Full vs. sampled replay verification for DATA-02.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & Roadmap
- `.planning/REQUIREMENTS.md` §Dataset Collection — DATA-01, DATA-02, DATA-03, DATA-04 are this phase's requirements.
- `.planning/ROADMAP.md` §Phase 4: Dataset Collection — success criteria (100+ HDF5 demos, deterministic state-based replay, SOARM-specific normalization stats, working teleop interface).

### Prior Phase Context (SOARM robot & tasks this phase records against)
- `.planning/phases/02-soarm-robot-integration/02-CONTEXT.md` and `02-04-SUMMARY.md` — confirms the 3 frozen `libero_spatial` task filenames, SOARM `Soarm101` robot class registration, and tuned base pose/reach envelope this phase's collector must respect.
- `.planning/phases/03-vla-inference-loop/03-CONTEXT.md` — establishes the real-module-over-notebook-code convention (D-04 there) that D-05 here follows; documents current 0% zero-shot OFT/π0-on-SOARM baseline relevant to D-01.

### Required Environment Reading
- `.planning/phases/01-colab-environment-setup/01-DEBUG-HISTORY.md` — the 7-invariant environment contract; relevant if any part of this phase's tooling still touches the Colab-installed dependency stack (e.g. shared robosuite/MuJoCo versions).
- STATE.md "Accumulated Context" — demo replay must be state-based, not action-replay, per LIBERO issue #16 (this is exactly DATA-02).

### Existing Prior-Art Code (reference only, not to be extended in place per D-05)
- `LIBERO/scripts/collect_demonstration.py` — existing robosuite human-teleop collector (keyboard/SpaceMouse via `input2action`, `DataCollectionWrapper`), npz-based output. Study its device-input pattern for D-02, but implement fresh HDF5 output in the new `datasets/` package.
- `LIBERO/scripts/libero_100_collect_demonstrations.py` — same pattern, LIBERO-100-specific variant.

### Codebase Maps
- `.planning/codebase/STACK.md` — pinned dependency versions (robosuite 1.4.0, MuJoCo 2.3.7).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `robosuite.wrappers.DataCollectionWrapper` / `VisualizationWrapper` and `robosuite.utils.input_utils.input2action` — existing device-to-action plumbing usable for D-02's keyboard teleop interface.
- `LIBERO/libero/libero/envs/robots/` (Phase 2) — `Soarm101` robot class this phase's collector environments must instantiate.
- `LIBERO/libero/lifelong/datasets.py` — existing robomimic `SequenceDataset`/HDF5 reading conventions in the codebase (training-side); this phase's HDF5 *writer* should stay format-compatible.

### Established Patterns
- Real importable Python modules over notebook-only code for anything reusable by later phases (established Phase 2/3 convention) — governs D-05.
- `MUJOCO_GL` must be set before any MuJoCo import — applies even for local (non-Colab) execution per D-03.

### Integration Points
- Phase 6 (Fine-Tuning) will import this phase's HDF5 dataset + normalization stats directly from `LIBERO/libero/libero/datasets/` (D-05) to drive LoRA fine-tuning of OpenVLA-OFT.
- The teleoperation interface (DATA-04) and scripted collector (DATA-01) should converge on the same HDF5 output format/schema so Phase 6 has one consistent dataset regardless of collection method.

</code_context>

<specifics>
## Specific Ideas

- No SpaceMouse hardware — keyboard is the practical teleop input for this project (D-02).
- Data collection is a local, CPU-bound sim task, not a GPU/Colab task — user wants to break from the Colab-first pattern specifically for this phase's collection step (D-03).

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. Multi-camera perception, depth extraction, and spatial-language tasks remain correctly deferred to Phase 5; RLDS conversion and LoRA fine-tuning remain deferred to Phase 6.

</deferred>

---

*Phase: 4-Dataset Collection*
*Context gathered: 2026-08-02*
