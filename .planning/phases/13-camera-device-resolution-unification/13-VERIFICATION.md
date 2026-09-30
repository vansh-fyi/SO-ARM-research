---
phase: 13-camera-device-resolution-unification
verified: 2026-09-30T14:00:54Z
status: human_needed
score: 18/19 must-haves verified
behavior_unverified: 0
overrides_applied: 0
human_verification:
  - test: Fresh post-Phase-13 live episode with renewed device detection and visual review of overhead and wrist frames
    expected: Both overhead views show the robot workspace; policy camera1 shows the physical wrist camera; recorded policy camera1/camera2/camera3 frames carry sha256 metadata.
    why_human: Physical device correspondence and live Colab integration cannot be established by injected camera doubles. Phase 12 evidence predates the new resolver and does not satisfy CAMFIX-02.
---

# Phase 13: Camera Device Resolution Unification Verification

**Status:** human_needed — implementation checks pass; fresh hardware acceptance remains open.
**Re-verification:** No; no previous VERIFICATION.md existed.
**Phase goal:** As a researcher, I want to use a single shared, name-based camera-resolution helper and one `StereoSplitCamera` instance for the model's inference-input path and the `IOLogger` recording path, with client-side provenance logged for each step sent to Colab, so that I can reliably inspect recordings of the robot workspace instead of the laptop webcam and trace the client's submitted frames.

The orchestrator normalized the existing MVP goal into user-story grammar without changing scope. `gsd-tools query user-story.validate` on the current exact goal returns `valid: true`. All four plans, summaries, context decisions, requirements, and roadmap success criteria were consulted; implementation claims below are grounded in source and tests, not summary prose.

## User Flow Coverage

| Step | Expected | Evidence | Status |
|---|---|---|---|
| Refresh connected-device detection | Persist both role identifiers and camera names | `detect_devices.py:452` saves `wrist_name` with symmetric fallback | VERIFIED in code/tests; fresh detection pending |
| Start a live bridge episode with map defaults | Resolve wrist once and reuse the policy stereo instance for recording | `run_vla_episode.py:554`, `:573`, `:651`; single-instance regression test | VERIFIED in code/tests |
| Use standalone recording | Resolve selected known roles; write split stereo stills/video | `record_episode.py:76`, `:100`, `:129` and recorder tests | VERIFIED in code/tests |
| Inspect recorded camera images | See the actual workspace and wrist-mounted view | Physical identity is not proved by the positional correlation heuristic | UNCERTAIN — WARNING; human acceptance required |
| Inspect per-step frame provenance | Local frame records contain hashes without implying server receipt | `io_logger.py:166`; policy observation flow in `run_vla_episode.py:339` | VERIFIED in code/tests; live spot-check pending |

The code supports the complete flow. The real user-visible outcome is not yet accepted; the technical evidence below does not replace the pending live walkthrough.

## Goal Achievement

### Observable Truths

Roadmap criteria are retained first. Plan truths add detail; the duplicate hash truth and fresh-live-review truth are merged into roadmap criteria 3 and 2 respectively. The hash truth is interpreted with the same plan's explicit graceful-omission exception.

| # | Truth | Status | Evidence |
|---|---|---|---|
| 1 | Inference and IOLogger resolve overhead through one shared name-based helper and reuse one StereoSplitCamera | VERIFIED | Resolver in `detect_devices.py:273`; defaulting `run_vla_episode.py:569`; `robot_client.py:217` creates/stores one stereo instance; `run_vla_episode.py:651` passes that same object to recording. Named single-instance test passes, including stale-index collision. |
| 2 | A fresh live episode's overhead images are human-confirmed workspace images; wrist identity and live hash fields are also checked | UNCERTAIN — WARNING | Required by CAMFIX-02 and D-05/D-06; no fresh Phase 13 hardware episode or visual acceptance supplied. |
| 3 | Bridge logs local frame hashes with explicit client-only provenance disclosure | VERIFIED | Observation captured in `robot_client.py:387`, consumed at `run_vla_episode.py:305,339`, hashed at `io_logger.py:187`, serialized via `write_step`. Logger tests assert independently computed hashes and JSONL metadata. |
| 4 | Successful detection returns wrist_name symmetrically | VERIFIED | `detect_devices.py:452-456`; exact-dictionary tests plus `test_wrist_name_falls_back_to_raw_index_when_no_name_correlates`. |
| 5 | Both roles use one name-first/raw-index-fallback function | VERIFIED | `resolve_camera_identifier`, lines 273-293; direct stereo name and current wrist index branches. |
| 6 | Wrist resolution uses fresh injectable name/index correlation | VERIFIED | `_live_correlate_names` composes live probes; drift test independently rerun and passed. |
| 7 | Soft name-resolution misses fall back to raw index | VERIFIED | No-match/absent/non-string-name tests; `get_camera_names` handles OS/subprocess timeout failures with an empty list. Malformed maps remain hard errors by design. |
| 8 | Hash disclosure explains lack of remote receipt/use proof | VERIFIED | `_record_frame` docstring explicitly names PolicyServer and absent round-trip confirmation. |
| 9 | Disclosure appears only in code, not output metadata | VERIFIED | `_record_frame` returns path/time/hash only; logger test asserts exact keys and output-file set. |
| 10 | Hash failures omit optional hash and preserve recording | VERIFIED | Narrow defensive exception around hashing; named failure test rerun successfully. |
| 11 | Live defaults for both roles call the same helper | VERIFIED | `run_vla_episode.py:554,573`; sentinel-based default-resolution test. |
| 12 | The same resolved wrist identifier reaches recording and inference | VERIFIED | Default list → `_build_camera_names` → `_open_cameras` and `wrist_camera_index`; named bridge test verifies both destinations. |
| 13 | Explicit CLI camera identifiers bypass default resolution | VERIFIED | Branch guards at `run_vla_episode.py:547,569`; test injects a resolver that raises if called and checks explicit values. |
| 14 | Existing robot_client interfaces accept resolved identifiers without additional changes | VERIFIED | `robot_client.py:165-166` passes wrist index to OpenCVCameraConfig; `:217` passes stereo identifier to StereoSplitCamera. |
| 15 | Standalone recorder loads the map and resolves both known camera roles | VERIFIED | `_load_device_map`, `_build_camera_roles`, `open_cameras`; main wires all three; tests cover read failures, stale warning, and resolver arguments. |
| 16 | Standalone recorder supports split AR0144 still and video frames | VERIFIED | Stereo branches in `record_still` and `record_episode`; pixel-equality still tests and named video test with correct dimensions. |
| 17 | Standalone known overhead capture uses one StereoSplitCamera instead of an independent cv2 capture | VERIFIED | Role branch at `record_episode.py:81`; duplicate indices deduplicated; test asserts exactly one stereo constructor and no overhead cv2 capture. Discovery probing is expressly preserved by plan 13-04. |
| 18 | Generic/wrist cv2 recording and existing generic filenames remain supported | VERIFIED | Generic branches retain `camera_{idx}.png/.mp4`; baseline still pixel tests and generic parameter of named video test. |
| 19 | New recorder test file includes baseline and new behavior coverage | VERIFIED | `control/test_record_episode.py` contains 16 collected cases covering map loading, capture dispatch, failure cleanup, stills and video. |

**Score:** 18/19 verified; one human-only physical verification truth is uncertain. No behavior-dependent implementation truth was accepted on symbol presence alone. No overrides or frontmatter prohibitions were declared.

### Required Artifacts

| Artifact | Existence/substance/wiring | Evidence |
|---|---|---|
| `control/detect_devices.py` | VERIFIED | Name persistence; fresh probe composition; shared resolver called by both consumers. |
| `control/test_detect_devices.py` | VERIFIED | Positive, legacy-map, missing-name and fresh-index cases exercise real resolver. |
| `control/vla_bridge/io_logger.py` | VERIFIED | Shared frame capture implementation computes hashes and serializes record metadata. |
| `control/test_io_logger.py` | VERIFIED | Raw-byte hashes, disclosure, failure fallback, unchanged depth schema and JSONL coverage. |
| `control/run_vla_episode.py` | VERIFIED | Defaults feed bridge and recording, one policy stereo object is reused. |
| `control/test_run_vla_episode.py` | VERIFIED | Resolver sentinels, explicit overrides, stale-index collision, same-object wiring. |
| `control/record_episode.py` | VERIFIED | Device-map-aware role dispatch and substantive split still/video implementations. |
| `control/test_record_episode.py` | VERIFIED | Exercises real recorder functions using hardware-free captures/writers. |

Plan artifact/key-link frontmatter uses descriptive strings rather than structured path/from/to objects, so artifacts and links were verified manually against source rather than treating schema-tool results as evidence.

### Key Links and Data Flow

| Source | Destination | Result |
|---|---|---|
| Detection names_by_index | persisted cameras.wrist_name → shared wrist lookup | WIRED; fresh real map generation still requires user-run detection. |
| Live defaults | resolver → OpenCV recording and connect_bridge identifiers | WIRED; default sentinel tests and same-index assertions. |
| connect_bridge | `_wire_stereo_split_cameras` → policy get_observation and client._stereo_camera → recording | WIRED; object identity retained. |
| Standalone map/selection | role map → shared resolver → stereo/cv2 capture → PNG/MP4 | WIRED; real frame arrays reach output functions in tests. |
| Post-transport-size observation dict | BridgeActionSource.last_details → policy_observation → IOLogger._record_frame → action_details.policy_observation_frames → episode.jsonl | FLOWING; arrays originate from the sent observation, not a replacement camera read. |

No dynamic UI artifacts exist. The applicable data-flow check traces recorded data instead. Wrist policy frames are converted RGB→BGR before PNG recording (`run_vla_episode.py:346`), so their hashes attest to the local saved-frame array, not the serialized network payload or PNG file bytes. This matches D-08's choice to hash the array being PNG-encoded; no server receipt is proven. Diagnostic overhead reads can be later frames from the same capture object; exact policy-frame provenance uses the separate `policy_observation_frames` path.

### Behavioral Spot-Checks

Verifier independently ran the following exact node IDs together using `control/.venv/bin/python -m pytest -q` from `control/`:

| Node ID | Result |
|---|---|
| `test_run_vla_episode.py::test_resolved_bridge_records_wrist_and_reuses_single_stereo` | PASS, 2 parameters; correct wrist capture, reused stereo identity, client stop |
| `test_io_logger.py::test_hash_failure_still_saves_frame_without_hash` | PASS; frame survives hashing failure |
| `test_detect_devices.py::test_resolve_wrist_identifier_uses_current_index_instead_of_stale_index` | PASS; stale index replaced |
| `test_record_episode.py::test_record_episode_writer_dimensions_frames_and_release` | PASS, 2 parameters; expected pixels/dimensions and writer release |

**Independent verifier result: 6 passed in 3.54s.** All captures are test doubles; temporary fixture output only. The orchestrator additionally ran the full current control suite independently of executor summaries: **208 passed in 5.45s**, with production-module compilation passing. That suite was not repeated here. No server or hardware was started.

### Probe Execution

Not applicable: no phase-declared `probe-*.sh` or conventional `scripts/*/tests/probe-*.sh` was found. Pytest is the phase's declared executable verification surface.

### Requirements Coverage

| Requirement | Plans | Status | Evidence |
|---|---|---|---|
| CAMFIX-01 | 01, 03, 04 | SATISFIED in code/tests | Shared resolver, consumer wiring, single-overhead-instance tests and standalone recording. |
| CAMFIX-02 | 03 | NEEDS HUMAN | Fresh live overhead AND wrist visual acceptance is mandatory and still absent. |
| CAMFIX-03 | 02 | SATISFIED in code/tests | Per-frame SHA-256, actual observation-derived frame path and client-only docstring. Live spot-check retained. |

All three Phase 13 requirements are claimed by plans; none are orphaned. Later Phases 14–17 do not explicitly defer or replace this phase's fresh-camera acceptance requirement.

### Anti-Patterns and Disconfirmation

No missing/stub/unwired artifacts or unreferenced debt-marker comments were found in the eight modified production/test files. `usbmodemXXXX` in existing CLI help is an example device path, not an unresolved XXX debt marker.

| Finding | Classification | Consequence |
|---|---|---|
| `_correlate_names` explicitly uses best-effort positional OS/cv2 enumeration | WARNING, covered by pending human acceptance | Injected correspondence tests cannot establish physical camera identity. |
| Bridge regression substitutes `connect_bridge` and `run_episode` | Test-evidence limitation | Proves routing/object identity, not real camera pixels or Colab receipt; those claims are not inferred. |
| Recorder video test uses a fake writer; real codec/open failure is not exercised | Existing runtime limitation | No claim of real MP4/codec validation; live identity remains mandatory. |
| Generic camera discovery still opens numeric probes; explicit/generic selections can bypass named role resolution | Intentional scoped behavior | Plan 13-04 preserves discovery and generic support; D-02 preserves scripted harness; D-04 permits raw-index fallback. Single-open verdict applies to role-aware operational overhead capture. |

The raw-index fallback is documented in the resolver and remains intentionally reachable. No claim is made that fallback guarantees physical identity when names are unavailable.

## Human Verification Required

### 1. Fresh live camera identity and provenance acceptance

**Test:** After all four plans are committed, the user runs fresh detection and a new live episode using the real server/checkpoint:

```sh
control/.venv/bin/python control/detect_devices.py
control/.venv/bin/python control/run_vla_episode.py --server-address <host:port> --checkpoint <checkpoint> --out control/outputs/<new_dir> --max-steps <N>
```

Confirm the regenerated map includes `wrist_name`. Open several `camera_overhead_left/*.png` and `camera_overhead_right/*.png` frames, then `camera_policy_camera1/*.png` frames (the actual directory name produced for `policy_camera1`). Inspect several sent-observation JSONL records at `action_details.policy_observation_frames.camera1/camera2/camera3` for nonempty `sha256` values. Record the new output directory and human findings in Phase 13 UAT.

**Expected:** Both overhead views show the robot workspace, not the laptop webcam; policy camera1 shows the physical wrist-mounted view; successful policy frame records include hashes. The hash is local provenance only.

**Why human:** Real camera identity and the live external integration cannot be inferred from mocked captures or unit tests. D-05/D-06 explicitly require a fresh episode; Phase 12 recordings cannot substitute. No physical hardware was operated during this verification.

## Gaps Summary

No implementation blockers found. Phase 13 remains **human_needed**, not complete or ready for automatic phase advancement, until fresh overhead AND wrist acceptance is recorded. There are no deferred gaps and no accepted overrides.

_Verified: 2026-09-30T14:00:54Z — gsd-verifier_
