---
id: SEED-001
status: dormant
planted: 2026-08-29
planted_during: Phase 6 (Fine-Tuning & Evaluation)
trigger_when: next milestone that touches perception/camera setup, dataset re-collection, or a new fine-tuning round
scope: medium
---

# SEED-001: Camera calibration + depth camera for perception

## Why This Matters

Found while reviewing `06b-eval.ipynb`'s before/after benchmark videos during Phase 6 UAT (2026-08-29). Two distinct gaps in the current camera/perception setup:

1. **Camera angle mismatch**: the agentview/front camera in the current MuJoCo scene setup does not represent the true camera view/position that will exist on the real physical SOARM arm. Training and evaluating on a simulated view that doesn't match the real robot's actual camera placement/FOV risks a sim-to-real gap that won't show up until deployment on real hardware.
2. **No depth camera output**: only RGB (agentview + eye_in_hand) is currently captured and trained on. Adding a depth camera (top-down or side-mounted) would give the policy persistent environment/spatial geometry data alongside RGB, not just color images — likely to matter for the spatial-reasoning tasks this project already cares about (Phase 5).

## When to Surface

**Trigger:** next milestone that touches perception/camera setup, dataset re-collection, or a new fine-tuning round.

This seed will surface during `/gsd-new-milestone` when the milestone scope matches.

## Scope Estimate

**Medium** — likely a full phase's worth of work: MJCF/scene camera recalibration (physical measurement or CAD reference for real camera placement), a new depth-camera MJCF definition + robosuite/LIBERO plumbing to expose it as an observation, HDF5 writer changes to persist depth alongside RGB, RLDS converter changes, and likely a Phase 4-style re-collection of demonstrations once the new camera setup is validated.

## Breadcrumbs

- `LIBERO/libero/libero/vla/eval_loop.py` — `run_episode`'s `camera_name`/`agentview_camera_name` params, where camera views are pulled from `obs`
- `LIBERO/libero/libero/datasets/hdf5_writer.py` — regenerates image observations at collection time; would need a depth-camera equivalent
- `.planning/phases/05-spatial-awareness/` — Phase 5's existing multi-camera/depth-based 3D localization work (closest prior art for adding a depth camera)
- `.planning/phases/02-soarm-robot-integration/` — SOARM MJCF/camera definitions (agentview, eye_in_hand)

## Notes

Captured via `/gsd-capture` during Phase 6 UAT wrap-up. Not blocking Phase 6 closure — this is forward-looking work for a future milestone, not an immediate fix.
