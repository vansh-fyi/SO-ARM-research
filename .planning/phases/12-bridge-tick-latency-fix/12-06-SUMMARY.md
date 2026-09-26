---
phase: 12-bridge-tick-latency-fix
plan: 06
subsystem: robotics-vla-depth
tags: [opencv, stereo-rectification, fast-foundationstereo, flask, ngrok, depth-recording]

# Dependency graph
requires:
  - phase: 12-bridge-tick-latency-fix
    provides: "Plan 12-03's overhead_left/overhead_right per-tick stereo diagnostic recording (shared stereo read this plan reuses); Plan 12-05's stereo_calibration.StereoCalibration/load_calibration() this plan's DepthCameraClient consumes directly"
provides:
  - "control/vla_bridge/depth_camera.py: compute_target_size()/scale_intrinsics() FastFS input-constraint helpers, DepthCameraClient (rectify()/compute_depth()) posting to a Colab-hosted FastFS HTTP endpoint, CLI main() for the known-distance human checkpoint"
  - "control/vla_bridge/policy_server_launch.md: Steps 7-9 documenting the Colab-side FastFS install/serve/tunnel cells, gated by a supply-chain legitimacy checkpoint modeled on the existing pyngrok gate"
  - "IOLogger.capture_depth_map() + write_step()'s new depth_frames param, recording depth maps into episode.jsonl without touching the policy observation path"
  - "run_vla_episode.py depth-recording wiring: --depth-endpoint/--depth-calibration/--depth-every-n-steps CLI flags, depth_client/depth_every_n_steps run_episode() params, cadence-gated capture sharing the SAME per-tick stereo read as Plan 12-03's diagnostic recording"
affects: [13-camera-resolution-unification, future-depth-fusion-milestone]

# Tech tracking
tech-stack:
  added: ["requests==2.34.2 (direct import; already a transitive lerobot[async] dependency)"]
  patterns: ["Record-only sensor capture: depth is persisted to episode output and episode.jsonl but structurally never reaches BridgeActionSource/connect_bridge's observation dict (grep-gated acceptance criterion + zero test coupling to robot_client.py)", "Cadence-gated expensive I/O: depth_every_n_steps decouples a slow network round-trip from the fast control tick, mirroring execution_hz's existing decoupling of waypoint-send rate from control_hz"]

key-files:
  created: [control/vla_bridge/depth_camera.py, control/test_depth_camera.py]
  modified: [control/vla_bridge/io_logger.py, control/run_vla_episode.py, control/vla_bridge/policy_server_launch.md, control/requirements.txt, control/test_io_logger.py, control/test_run_vla_episode.py]

key-decisions:
  - "Depth inference runs on a SECOND Colab-hosted HTTP endpoint (Flask, port 8081, separate ngrok HTTP tunnel) alongside the existing gRPC PolicyServer (port 8080) -- FastFS is CUDA-GPU-only with no CPU path, and this project's Mac client has no CUDA GPU."
  - "Depth is recording-only this phase (D-08): depth_camera.py is never imported by vla_bridge.robot_client, structurally enforced by a grep-gated acceptance criterion (`grep -ic depth robot_client.py` == 0)."
  - "The per-tick stereo read is refactored into local variables (stereo_left_frame/stereo_right_frame) so Plan 12-03's diagnostic recording and this plan's depth computation share exactly one physical StereoSplitCamera read, never a desyncing third call."

requirements-completed: [DEPTH-CAL-02, DEPTH-CAL-03]

coverage:
  - id: D1
    description: "DepthCameraClient rectifies a live AR0144 stereo pair, downsamples to FastFS's <1000px/divisible-by-32 input constraints (scaling intrinsics proportionally), and POSTs to a Colab-hosted FastFS endpoint, returning a metric depth map"
    requirement: "DEPTH-CAL-02"
    verification:
      - kind: unit
        ref: "control/test_depth_camera.py (10 tests: compute_target_size, scale_intrinsics, rectification map construction/baseline derivation, rectify() shape preservation, compute_depth() round-trip/payload/timeout/failure-handling)"
        status: pass
    human_judgment: false
  - id: D2
    description: "A human confirms the computed depth value at a known, physically-measured real-world distance is within a documented tolerance against the real Colab FastFS endpoint"
    requirement: "DEPTH-CAL-02"
    verification: []
    human_judgment: true
    rationale: "Requires live AR0144 hardware and a running Colab GPU runtime with the FastFS endpoint deployed -- per CONTEXT.md D-06, Claude does not run physical hardware or a live Colab GPU runtime. depth_camera.py's main() CLI is implemented and ready for this checkpoint."
  - id: D3
    description: "policy_server_launch.md documents the Colab-side FastFS install/serve/tunnel cells with a request/response contract that exactly matches depth_camera.py's real implementation, gated by a human-confirmed supply-chain legitimacy check"
    requirement: "DEPTH-CAL-02"
    verification:
      - kind: unit
        ref: "grep -c NVlabs/Fast-FoundationStereo (3), grep -c depth_npy_b64 (3), grep -c 'ngrok.connect(8081' (1) against control/vla_bridge/policy_server_launch.md"
        status: pass
    human_judgment: true
    rationale: "The supply-chain legitimacy check itself (visiting github.com/NVlabs/Fast-FoundationStereo to confirm org/README/dependencies before running any install cell) requires human judgment and cannot be automated -- this is a blocking checkpoint per the plan's threat model (T-12-17)."
  - id: D4
    description: "Depth is recorded into episode.jsonl via IOLogger.capture_depth_map() at a configurable cadence, sourced from the SAME tick's stereo frame already used for Plan 12-03's diagnostic recording (object-identity verified), with zero references to depth in vla_bridge/robot_client.py"
    requirement: "DEPTH-CAL-03"
    verification:
      - kind: unit
        ref: "control/test_io_logger.py (5 new depth tests) + control/test_run_vla_episode.py (7 new depth tests, including exactly-once-per-tick stereo read with `is`-identity assertions and main()-level DepthCameraClient construction/threading); full control test suite (153 tests) passes"
        status: pass
    human_judgment: false
  - id: D5
    description: "Episode-level live-hardware confirmation: depth_overhead/ directory populated at the configured cadence, episode.jsonl's depth_frames field correctly sparse, robot motion unaffected by depth capture"
    requirement: "DEPTH-CAL-03"
    verification: []
    human_judgment: true
    rationale: "Requires a live episode against the real AR0144 and a running Colab FastFS endpoint -- per CONTEXT.md D-06, Claude does not run physical hardware or a live Colab GPU runtime."

duration: 35min
completed: 2026-09-26
status: complete
---

# Phase 12 Plan 06: Depth-Calibration Endpoint + Recording-Only Episode Wiring Summary

**Colab-hosted Fast-FoundationStereo depth endpoint (Flask over a second ngrok HTTP tunnel) plus a local DepthCameraClient that rectifies/downsamples the AR0144 stereo pair and records metric depth into episode.jsonl at a configurable cadence, structurally isolated from the policy's observation dict.**

## Performance

- **Duration:** ~35 min
- **Completed:** 2026-09-26
- **Tasks:** 3 (all automated portions complete; two live-hardware human-check steps handed off per D-06)
- **Files modified:** 8 (2 new: depth_camera.py, test_depth_camera.py; 6 modified: io_logger.py, run_vla_episode.py, policy_server_launch.md, requirements.txt, test_io_logger.py, test_run_vla_episode.py)

## Accomplishments

- `DepthCameraClient` rectifies a live AR0144 stereo pair via `cv2.stereoRectify`/`initUndistortRectifyMap`, downsamples to Fast-FoundationStereo's `<1000px`/divisible-by-32 input constraints with correctly-scaled intrinsics, and POSTs to a Colab-hosted FastFS HTTP endpoint with a bounded 10s timeout, returning `None` (never raising) on any network/response failure.
- `policy_server_launch.md` documents the full Colab-side FastFS install/serve/tunnel flow (Steps 7-9: gated git-clone install, Flask `/depth` handler, second ngrok HTTP tunnel on port 8081) with an exact request/response contract matching `depth_camera.py`.
- Depth recording is wired into `run_vla_episode.py`'s per-tick control loop at a configurable cadence (`--depth-every-n-steps`, default 10), sharing the exact same stereo frame already used for Plan 12-03's diagnostic recording -- verified via object-identity (`is`), not just value-equality.
- `vla_bridge/robot_client.py` has zero references to depth, structurally enforcing D-08's boundary that depth never reaches the policy's observation dict.

## Task Commits

Each task was committed atomically, following the plan's TDD gate (RED -> GREEN) for Tasks 1 and 3:

1. **Task 1: depth_camera.py -- rectification, downsampling, FastFS client**
   - `344a62d` test: add failing tests for depth_camera rectification/downsample/FastFS client (RED)
   - `deca078` feat: implement DepthCameraClient -- rectify/downsample/FastFS depth request (GREEN)
2. **Task 2: Document the Colab-side Fast-FoundationStereo endpoint**
   - `850b8dc` docs: document the Colab-side Fast-FoundationStereo depth endpoint
3. **Task 3: Wire depth recording into run_vla_episode.py's per-tick loop**
   - `9845410` test: add failing tests for depth-recording wiring into run_vla_episode (RED)
   - `ccfb1e8` feat: wire depth recording into run_vla_episode.py's per-tick loop (GREEN)

_TDD gate verified: for both Task 1 and Task 3, a `test(...)` commit exists before its corresponding `feat(...)` commit, with RED confirmed via a pre-implementation test run (import/attribute errors) before the GREEN implementation was restored/added._

**Plan metadata:** committed alongside this SUMMARY (see final commit below).

## Files Created/Modified

- `control/vla_bridge/depth_camera.py` - New module: `compute_target_size()`, `scale_intrinsics()`, `DepthCameraClient` (rectify/compute_depth), CLI `main()`
- `control/test_depth_camera.py` - 10 tests covering pure-logic helpers, rectification map construction, and mocked-FastFS compute_depth() behavior
- `control/vla_bridge/policy_server_launch.md` - New Depth endpoint section (Steps 7-9), extended teardown, new Summary table row
- `control/vla_bridge/io_logger.py` - New `capture_depth_map()` method; `write_step()` gains backward-compatible `depth_frames` param
- `control/run_vla_episode.py` - New depth CLI flags, `run_episode()` depth_client/depth_every_n_steps params, refactored shared stereo read, main()-level DepthCameraClient construction/validation
- `control/requirements.txt` - Added explicit `requests==2.34.2` pin
- `control/test_io_logger.py` - 5 new tests for `capture_depth_map()` and `write_step()`'s `depth_frames` handling
- `control/test_run_vla_episode.py` - 7 new tests for depth recording, cadence gating, shared-frame identity, and CLI wiring/validation

## Decisions Made

- Depth inference runs on a second, independent Colab-hosted endpoint (Flask + a distinct HTTP ngrok tunnel on port 8081) rather than being folded into the existing gRPC PolicyServer, since FastFS is CUDA-GPU-only with no CPU path and this project's Mac client has no CUDA GPU.
- The Plan-12-03-introduced stereo capture was refactored (not reimplemented) into local variables purely to expose the raw frames for reuse -- identical observable behavior for the existing diagnostic recording, with depth computation as a pure addition gated by `stereo_left_frame is not None and stereo_right_frame is not None`.
- Two `cv2.stereoRectify`/`np.float64` shape fixes were needed during implementation (see Deviations below) -- neither changes the plan's documented behavior or contract, both are within Rule 1 (bug fix to make the specified behavior actually work).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `cv2.stereoRectify` required `T` reshaped to `(3, 1)` and `image_size` as plain Python ints**
- **Found during:** Task 1 (GREEN phase, first test run against the real implementation)
- **Issue:** The plan's action spec described reshaping `T` into "a flat 3-vector" for `cv2.stereoRectify`, but OpenCV's binding requires a `(3, 1)` column vector shape (raised `cv2.error: ... gemm ... assertion failed`) and a numpy-derived `image_size` tuple caused a related type mismatch.
- **Fix:** Reshaped `T` to `(3, 1)` before calling `cv2.stereoRectify`, and cast `image_size` elements to plain `int`.
- **Files modified:** `control/vla_bridge/depth_camera.py`
- **Verification:** `test_depth_camera.py`'s rectification-map-construction and `rectify()` shape-preservation tests pass.
- **Committed in:** `deca078` (Task 1 GREEN commit)

**2. [Rule 1 - Bug] `abs(float(T[0]))` triggered a numpy scalar-conversion DeprecationWarning after the `(3,1)` reshape**
- **Found during:** Task 1 (GREEN phase, immediately after fix #1)
- **Issue:** With `T` reshaped to `(3, 1)`, `T[0]` is a 1-element array, not a scalar; `float(T[0])` emitted a `DeprecationWarning` (numpy >=1.25) that would become an error in a future numpy version.
- **Fix:** Changed to `float(T.flatten()[0])`.
- **Files modified:** `control/vla_bridge/depth_camera.py`
- **Verification:** `pytest -q` shows 0 warnings on the affected tests.
- **Committed in:** `deca078` (Task 1 GREEN commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 -- bug fixes required to make the plan's own specified OpenCV calls actually work correctly).
**Impact on plan:** Both fixes are internal implementation details with zero effect on the documented public contract (`compute_target_size()`/`scale_intrinsics()`/`DepthCameraClient`'s public methods and their tested behavior are unchanged). No scope creep.

## Issues Encountered

- The worktree's `control/` directory has no local `.venv` (worktrees only check out tracked files); all test runs used the main repo's `control/.venv/bin/python` (confirmed `requests==2.34.2` pre-installed there, matching the version pinned in `requirements.txt`) invoked against the worktree's test files.

## User Setup Required

None for the automated portions -- no new environment variables or dashboard configuration required for code already merged.

**Two live-hardware/live-Colab human-check steps are handed off per CONTEXT.md D-06 (Claude does not run physical hardware or a live Colab GPU runtime):**

1. **Task 1's known-distance depth-accuracy checkpoint:** After deploying Task 2's Colab FastFS endpoint (Steps 7-9 in `policy_server_launch.md` -- gated by the NVlabs supply-chain legitimacy check), run:
   ```
   cd control && python -m vla_bridge.depth_camera --calibration stereo_calibration.json --endpoint <colab-fastfs-url> --stereo-camera-index "CCB Camera"
   ```
   Place an object at a known, measured distance (0.3-0.5m) and confirm the printed `center=` value is within +/-15% or +/-3cm (whichever is larger) of the measured distance.

2. **Task 3's episode-level wiring confirmation:** After (1) passes, run a short live episode with `--depth-endpoint`/`--depth-calibration` set and confirm `depth_overhead/` populates at the configured cadence, `episode.jsonl`'s `depth_frames` field is correctly sparse, and robot motion is unaffected.

## Next Phase Readiness

- DEPTH-CAL-02 and DEPTH-CAL-03 (automated portions) are complete; the depth-calibration extension to Phase 12 (D-07's steps 2-3) is fully implemented and tested.
- Fine-tuning a depth-aware checkpoint and the StereoPatch RGB-depth fusion architecture (D-09) remain explicitly out of scope, deferred to a later milestone phase.
- Both human-check steps above should be run before this plan is considered fully verified end-to-end; neither blocks other Phase 12 plans or Phase 13's camera-resolution-unification work, since depth remains structurally isolated from the observation path.

---
*Phase: 12-bridge-tick-latency-fix*
*Completed: 2026-09-26*

## Self-Check: PASSED

- FOUND: control/vla_bridge/depth_camera.py
- FOUND: control/test_depth_camera.py
- FOUND: .planning/phases/12-bridge-tick-latency-fix/12-06-SUMMARY.md
- FOUND commit: 344a62d (test)
- FOUND commit: deca078 (feat)
- FOUND commit: 850b8dc (docs)
- FOUND commit: 9845410 (test)
- FOUND commit: ccfb1e8 (feat)
