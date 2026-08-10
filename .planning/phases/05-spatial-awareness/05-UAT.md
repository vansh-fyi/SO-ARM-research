---
status: testing
phase: 05-spatial-awareness
source: [05-VERIFICATION.md]
started: 2026-08-10T05:35:00Z
updated: 2026-08-10T05:35:00Z
---

## Current Test

number: 1
name: Colab GPU spot-check — OFTBackend consumes both camera views
expected: |
  Run OFTBackend.predict() on Colab with two genuinely different camera frames
  (agentview + eye_in_hand) and confirm the model's output action chunk reflects
  real dual-image consumption, not a silently-ignored second view. Non-degenerate,
  plausible (8,7) action chunk; ideally compare against a single-image baseline to
  confirm the second view changes the output.
awaiting: user response

## Tests

### 1. Colab GPU spot-check — OFTBackend consumes both camera views
expected: Non-degenerate, plausible (8,7) action chunk; ideally compare against a single-image baseline to confirm the second view changes the output.
result: [pending]

### 2. Colab Linux (osmesa/egl) re-run of depth_xyz.py's D-04 test using real MuJoCo instance-segmentation render
expected: obs['agentview_segmentation_instance'] is non-degenerate on Colab's GL backend, and object_xyz_from_obs's back-projected estimate still matches sim.data.body_xpos ground truth within ~0.05m using the REAL segmentation array.
result: [pending]

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps
