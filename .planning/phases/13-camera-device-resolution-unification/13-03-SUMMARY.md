---
phase: 13-camera-device-resolution-unification
plan: 03
subsystem: control
tags: [camera, bridge, resolution, tdd]
requires:
  - phase: 13-01
    provides: Shared name-first camera identifier resolver
provides:
  - Shared resolved identifiers for live bridge inference and recording
  - Stereo instance reuse and stale-index collision regression coverage
affects: [episode-recording, live-bridge]
tech-stack:
  added: []
  patterns: [Resolve live camera defaults once before capture and bridge wiring]
key-files:
  created: []
  modified: [control/run_vla_episode.py, control/test_run_vla_episode.py]
key-decisions:
  - Exclude unused stale overhead bookkeeping index from default live capture list to avoid overwriting the resolved wrist label.
requirements-completed: []
requirements-pending: [CAMFIX-01, CAMFIX-02]
coverage:
  - id: D1
    description: Live camera defaults feed inference and recording through the shared resolver while explicit overrides bypass it
    requirement: CAMFIX-01
    verification:
      - kind: unit
        ref: control/test_run_vla_episode.py#test_device_map_defaults_use_shared_resolved_identifiers
        status: pass
      - kind: unit
        ref: control/test_run_vla_episode.py#test_explicit_camera_identifiers_bypass_shared_resolver
        status: pass
    human_judgment: false
  - id: D2
    description: Resolved wrist capture survives stale overhead index collision and overhead recording reuses the bridge stereo instance
    requirement: CAMFIX-01
    verification:
      - kind: unit
        ref: control/test_run_vla_episode.py#test_resolved_bridge_records_wrist_and_reuses_single_stereo
        status: pass
    human_judgment: false
  - id: D3
    description: Fresh live episode confirms overhead and wrist physical camera identities and policy frame hashes
    requirement: CAMFIX-02
    verification: []
    human_judgment: true
    rationale: Requires a fresh user-run hardware episode after all four plans land; no hardware run performed during execution.
duration: 8min
completed: 2026-09-30
status: complete
---

# Phase 13 Plan 03: Live Bridge Camera Resolution Summary

**Live bridge camera defaults now use the shared name-first resolver, with a regression fix for stale overhead indices colliding with the resolved wrist.**

## Performance

- Duration: approximately 8min
- Tasks: 2 automated tasks complete; end-of-phase live check pending
- Files modified: 2

## Accomplishments

- The resolved wrist identifier reaches both IOLogger capture and bridge inference from one resolution point; the stereo identifier uses the same helper.
- Explicit CLI identifiers bypass resolution; scripted-path resolution behavior remains unchanged.
- The bridge's stereo instance reaches recording unchanged, without an additional overhead capture. Tests cover both distinct indices and wrist/stale-overhead index collision.
- Fresh hardware verification instructions are preserved below for end-of-phase UAT.

## Task Commits

1. Task 1 RED: `b94d9c0` — stale wrist index 3 failed against resolved sentinel 8.
2. Task 1 GREEN: `e59c6cf` — shared live default resolution.
3. Task 2 RED: `9e546c8` — resolved wrist matching stale overhead was skipped.
4. Task 2 GREEN: `376fd0e` — exclude unused stale overhead label from live default capture list.

## Verification

- Task 1: `cd control && .venv/bin/python -m pytest -q test_run_vla_episode.py` — 36 passed.
- Task 2: `cd control && .venv/bin/python -m pytest -q test_run_vla_episode.py test_robot_client.py` — 68 passed.
- Full control suite: `cd control && .venv/bin/python -m pytest -q` — 192 passed.
- Existing default-fallback/name-preference/explicit-wrist tests passed unmodified.
- `control/vla_bridge/robot_client.py` and its tests remain unchanged.
- `git diff --check` passed. No new stubs or additional trust boundaries introduced.

## Deviations from Plan

**1. [Rule 1 - Bug] Prevent stale overhead bookkeeping from hiding the resolved wrist.**
Task 2 demonstrated that a resolved wrist index equal to the old overhead index overwrites the wrist label in `_build_camera_names()`, suppressing both wrist recording and inference wiring. Default live captures now contain only the resolved wrist; overhead recording already comes from the bridge stereo instance. Explicit CLI behavior and non-bridge capture behavior are unchanged. Regression: `test_resolved_bridge_records_wrist_and_reuses_single_stereo[4]`. Fix commit: `376fd0e`.

**Tracking adjustment:** Requirements are not automatically marked complete merely because the implementation tasks are done. CAMFIX-01 still needs plan 13-04's standalone recording integration; CAMFIX-02 needs fresh live evidence. This intentionally overrides the summary template's blanket requirement-copy convention.

## Issues Encountered

No unresolved automated failures. T-13-01's physical identity confirmation remains pending the human check.

## Human Verification Pending — CAMFIX-02

After all four Phase 13 plans land, the user runs fresh detection and a fresh live episode using the project's environment (from repository root):

```sh
control/.venv/bin/python control/detect_devices.py
control/.venv/bin/python control/run_vla_episode.py --server-address <host:port> --checkpoint <checkpoint> --out control/outputs/<new_dir> --max-steps <N>
```

1. Confirm regenerated `control/device_map.json` includes `wrist_name`.
2. Inspect several saved `camera_overhead_left/*.png` and `camera_overhead_right/*.png`: both must show the real robot workspace, not the laptop webcam.
3. Inspect saved `camera_policy_camera1/*.png` (the actual IOLogger directory for policy_camera1): it must show the correct wrist-mounted camera.
4. Inspect `episode.jsonl` records at `action_details.policy_observation_frames.camera1/camera2/camera3`: each successfully captured frame should have a non-empty `sha256` field. Hashes attest only to local frame bytes.
5. Record the new episode directory and the user's visual confirmation in Phase 13 UAT. Phase 12 evidence does not satisfy this fresh check.

No physical hardware was operated and no fresh live episode was verified during this plan execution. Human verification is deferred to end of phase per `workflow.human_verify_mode`.

## Next Phase Readiness

Ready for 13-04. Phase 13 and CAMFIX-02 remain unverified until fresh live evidence is supplied.

## Self-Check: PASSED

Both modified files exist; all four task commits exist on the current branch. Automated acceptance checks and the full control suite pass. Live-check status is explicitly pending.
