---
status: partial
phase: 13-camera-device-resolution-unification
source: [13-VERIFICATION.md]
started: 2026-09-30T14:03:40Z
updated: 2026-09-30T14:10:00Z
---

## Current Test

[testing paused — 1 item outstanding]

## Tests

### 1. Fresh live episode confirms overhead and wrist camera identity
expected: Refresh device detection with control/.venv/bin/python control/detect_devices.py. Run a new live episode with the active PolicyServer and checkpoint using control/.venv/bin/python control/run_vla_episode.py --server-address <host:port> --checkpoint <checkpoint> --out control/outputs/<new_dir> --max-steps <N>, without camera overrides so the shared resolver is exercised. Inspect camera_overhead_left and camera_overhead_right PNGs for the robot workspace; inspect camera_policy_camera1 PNGs for the physical wrist view. Spot-check episode.jsonl action_details.policy_observation_frames.camera1/camera2/camera3 entries on sent-observation steps for nonempty sha256 fields. Provide the new episode path and visual results. Hashes establish client-side provenance only, not proof of Colab receipt or use.
result: blocked
blocked_by: prior-phase
reason: "User requested this live-hardware test be merged with and run alongside the next phase's UAT session, rather than executed standalone now."

## Summary

total: 1
passed: 0
issues: 0
pending: 0
skipped: 0
blocked: 1

## Gaps

None identified in implementation. CAMFIX-02 remains pending human verification — deferred to run together with the next phase's live hardware UAT.
