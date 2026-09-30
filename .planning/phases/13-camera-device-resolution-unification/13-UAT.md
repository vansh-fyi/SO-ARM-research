---
status: testing
phase: 13-camera-device-resolution-unification
source: [13-VERIFICATION.md]
started: 2026-09-30T14:03:40Z
updated: 2026-09-30T14:03:40Z
---

## Current Test

number: 1
name: Fresh live episode confirms overhead and wrist camera identity
expected: |
  Refresh device detection after Phase 13 and run a new live bridge episode.
  Both overhead views show the robot workspace, the policy camera1 view is
  the physical wrist camera, and sent policy camera1/2/3 frame records contain sha256.
  Record the fresh episode path and visual findings; Phase 12 evidence cannot be reused.
awaiting: user response

## Tests

### 1. Fresh live episode confirms overhead and wrist camera identity
expected: Refresh device detection with control/.venv/bin/python control/detect_devices.py. Run a new live episode with the active PolicyServer and checkpoint using control/.venv/bin/python control/run_vla_episode.py --server-address <host:port> --checkpoint <checkpoint> --out control/outputs/<new_dir> --max-steps <N>, without camera overrides so the shared resolver is exercised. Inspect camera_overhead_left and camera_overhead_right PNGs for the robot workspace; inspect camera_policy_camera1 PNGs for the physical wrist view. Spot-check episode.jsonl action_details.policy_observation_frames.camera1/camera2/camera3 entries on sent-observation steps for nonempty sha256 fields. Provide the new episode path and visual results. Hashes establish client-side provenance only, not proof of Colab receipt or use.
result: [pending]

## Summary

total: 1
passed: 0
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps

None identified in implementation. CAMFIX-02 remains pending human verification.
