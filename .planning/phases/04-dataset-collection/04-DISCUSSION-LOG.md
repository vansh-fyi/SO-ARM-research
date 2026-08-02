# Phase 4: Dataset Collection - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-02
**Phase:** 4-Dataset Collection
**Areas discussed:** Scripted demos, Teleop device, Demo scope, Module home, Verification

---

## Scripted Demos

| Option | Description | Selected |
|--------|-------------|----------|
| Hand-coded waypoint script | Task-specific IK/waypoint controller (approach → grasp → lift → move → place) using known object/plate positions from BDDL regions. Deterministic, replay-friendly. | |
| Scripted policy + randomized noise | Same waypoint approach, plus randomized initial placement + action noise for demo diversity, filtering non-successes. | |
| VLA rollout filtering (OFT/π0) | Run OFT/π0 repeatedly, keep successful episodes. Reuses Phase 3 infra, but current baseline is 0% success zero-shot on SOARM. | |
| Claude's discretion | Let the researcher/planner decide after investigating LIBERO's demo-generation conventions and SOARM's tuned kinematics. | ✓ |

**User's choice:** Claude's discretion
**Notes:** Flagged that OFT/π0 sit at 0% zero-shot success on SOARM (Phase 3 baseline), so VLA-rollout filtering is unlikely to be viable without further work — hand-coded/noise-augmented waypoints are the more realistic default unless research finds a fast fix.

---

## Teleop Device

| Option | Description | Selected |
|--------|-------------|----------|
| Keyboard only | No extra hardware; robosuite's existing keyboard device class reused directly. | ✓ |
| SpaceMouse (if available) | Better 6-DOF control but requires physical hardware + supported OS driver, likely local-only. | |
| Claude's discretion | Default to keyboard unless research surfaces a strong reason for SpaceMouse. | |

**User's choice:** Keyboard only
**Notes:** No SpaceMouse hardware available.

---

## Demo Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Even split across 3 tasks, run on Colab | ~34/33/33 demos per task, all recorded in a Colab notebook. | |
| Even split, run locally | Same distribution, collection runs on the local machine — CPU-bound sim, no GPU needed, faster iteration. | ✓ |
| Claude's discretion | Let the planner decide split and execution environment. | |

**User's choice:** Even split, run locally
**Notes:** Breaks from Phase 1-3's Colab-first convention specifically for this phase's collection step — data collection is CPU-bound MuJoCo sim, not VLA inference.

---

## Module Home

| Option | Description | Selected |
|--------|-------------|----------|
| New `LIBERO/libero/libero/datasets/` package | Mirrors Phase 3's `vla/` package pattern — real importable module Phase 6 can import for HDF5/normalization-stats loading. | ✓ |
| Extend `LIBERO/scripts/` in place | Adapt existing `collect_demonstration.py`/`libero_100_collect_demonstrations.py` to add robomimic HDF5 output + SOARM support. | |
| Claude's discretion | Let the planner decide based on what Phase 6's fine-tuning pipeline will most cleanly import from. | |

**User's choice:** New `LIBERO/libero/libero/datasets/` package
**Notes:** Existing `LIBERO/scripts/` collectors are human-teleop-only, npz-based, and not robomimic-HDF5 — kept as reference/prior art, not extended in place.

---

## Verification

| Option | Description | Selected |
|--------|-------------|----------|
| ~34 per task, spot-check replay (e.g. 10%) | Replay a random 10% sample via state-setting, diff final state/success flag. | |
| ~34 per task, replay every single demo | Full determinism check on all 100+ demos — most rigorous, slower given local execution. | |
| Claude's discretion | Let the planner size the verification sample based on measured per-replay runtime. | ✓ |

**User's choice:** Claude's discretion
**Notes:** Size the sample once the collector exists and per-replay runtime is known, given collection runs locally.

---

## Claude's Discretion

- Scripted trajectory generation strategy (hand-coded waypoints vs. noise-augmented vs. VLA-rollout filtered) — Scripted Demos area.
- Exact episode count per task (beyond "roughly even" across 3 tasks, totaling 100+) — Demo Scope area.
- Full vs. sampled replay verification for DATA-02 — Verification area.

## Deferred Ideas

None — discussion stayed within phase scope. Multi-camera perception, depth extraction, and spatial-language tasks remain correctly deferred to Phase 5; RLDS conversion and LoRA fine-tuning remain deferred to Phase 6.
