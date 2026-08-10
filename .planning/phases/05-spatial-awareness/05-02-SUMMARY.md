---
phase: 05-spatial-awareness
plan: 02
subsystem: perception
tags: [mujoco, robosuite, camera-calibration, depth, segmentation, spatial-awareness]

# Dependency graph
requires:
  - phase: 04-dataset-collection
    provides: put_the_cream_cheese_in_the_bowl.bddl task (Phase-4-validated SOARM pick-place layout), used as the real-sim fixture for all new tests
provides:
  - camera_depths=True depth-buffer extraction re-verified for both SOARM cameras (SPAT-03), zero env_wrapper.py changes
  - LIBERO/libero/libero/perception/ package: pixel_to_world_xyz, object_pixel_centroid, object_xyz_from_obs (SPAT-04)
  - Two real, root-caused, empirically-fixed bugs in the depth-to-world back-projection math (wrong camera_to_world_transform argument; unhandled opengl-vs-standard image row convention)
  - Documented local macOS/Apple-Silicon MuJoCo platform limitation (native instance-ID segmentation rendering is a no-op on this machine) with a test-only, ground-truth-anchored workaround
affects: [phase-6-fine-tuning, spatial-task-authoring, colab-validation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "perception/ subpackage follows vla/__init__.py's graceful-degradation try/except Exception import pattern (not predicates/__init__.py's bare star-import), since it transitively depends on robosuite/MuJoCo"
    - "numpy2/robosuite-1.4.1 compat: _set_promotion_state('legacy') shim in perception/__init__.py, required for SegmentationRenderEnv's instance-ID decode path under numpy>=2.0's NEP 50 promotion"

key-files:
  created:
    - LIBERO/libero/libero/envs/test_camera_config.py
    - LIBERO/libero/libero/perception/__init__.py
    - LIBERO/libero/libero/perception/depth_xyz.py
    - LIBERO/libero/libero/perception/test_depth_xyz.py
  modified: []

key-decisions:
  - "pixel_to_world_xyz uses np.linalg.inv(get_camera_transform_matrix(...)) as the camera_to_world_transform argument to transform_from_pixels_to_world, not the bare get_camera_extrinsic_matrix() the recovered draft used -- the library's internal math requires the full inverse pixel<->world transform (intrinsics+extrinsics combined)"
  - "pixel_to_world_xyz flips both the incoming depth_map and pixel_yx's row coordinate from robosuite's default macros.IMAGE_CONVENTION='opengl' (row-0-at-bottom) to the row-0-at-top convention robosuite.utils.camera_utils's pixel math assumes, mirroring get_camera_segmentation's own [::-1] self-flip in the same library"
  - "test_depth_xyz.py substitutes a ground-truth-anchored synthetic segmentation patch for the locally-broken native MuJoCo segmentation render, documented as a test-only platform workaround requiring Colab re-verification -- production depth_xyz.py code is unaffected and unmodified for this workaround"

patterns-established:
  - "Round-trip validation pattern for projective-geometry code: forward-project a known ground-truth world point to a pixel via the SAME camera_utils calibration, then back-project via the new code under test, and compare recovered vs. original point -- caught two real bugs (wrong transform argument, row-convention mismatch) that a purely-visual or shape-only test would have missed"

requirements-completed: [SPAT-01, SPAT-03, SPAT-04]

coverage:
  - id: D1
    description: "Camera-config integration test re-verifies SPAT-01 (RGB) and proves SPAT-03 (depth extraction) with zero env_wrapper.py changes"
    requirement: "SPAT-01"
    verification:
      - kind: integration
        ref: "LIBERO/libero/libero/envs/test_camera_config.py#test_rgb_cameras_non_degenerate_spatial"
        status: pass
      - kind: integration
        ref: "LIBERO/libero/libero/envs/test_camera_config.py#test_depth_cameras_non_degenerate_spatial"
        status: pass
    human_judgment: false
  - id: D2
    description: "depth_xyz.py's pixel_to_world_xyz back-projects a pixel+depth to world XYZ within an empirically-measured tolerance of MuJoCo's own ground truth (SPAT-04), and depth_xyz.py never reads privileged body-pose state (D-03)"
    requirement: "SPAT-04"
    verification:
      - kind: integration
        ref: "LIBERO/libero/libero/perception/test_depth_xyz.py#test_object_xyz_matches_ground_truth_within_tolerance"
        status: pass
      - kind: unit
        ref: "LIBERO/libero/libero/perception/test_depth_xyz.py#test_object_pixel_centroid_pure_logic"
        status: pass
    human_judgment: true
    rationale: "The full object_xyz_from_obs test uses a ground-truth-anchored synthetic segmentation patch, not real MuJoCo native segmentation rendering, due to a discovered local macOS/Apple-Silicon platform limitation (documented in test_depth_xyz.py's module docstring). A human/Colab re-run should confirm the real segmentation-rendering path produces equivalent results before this pipeline is trusted for downstream spatial-task work."

duration: 65min
completed: 2026-08-10
status: complete
---

# Phase 5 Plan 02: Depth Extraction + Depth-to-XYZ Perception Summary

**Depth-buffer extraction re-verified for both SOARM cameras, plus a new `perception/` package (`pixel_to_world_xyz`, `object_pixel_centroid`, `object_xyz_from_obs`) that back-projects depth+pixel to world XYZ via `robosuite.utils.camera_utils`, validated against MuJoCo's own ground truth after fixing two real projective-geometry bugs.**

## Performance

- **Duration:** ~65 min (continuation of a session-limit-interrupted prior attempt; recovered draft files were read, verified against library internals, and substantially corrected)
- **Completed:** 2026-08-10
- **Tasks:** 2/2
- **Files modified:** 4 (all new)

## Accomplishments

- Re-verified SPAT-01 (RGB) and proved SPAT-03 (depth extraction) via a real, no-mock integration test against the existing `put_the_cream_cheese_in_the_bowl` task, with zero changes to `env_wrapper.py`
- Built `LIBERO/libero/libero/perception/depth_xyz.py`: `pixel_to_world_xyz`, `object_pixel_centroid`, `object_xyz_from_obs` — a real-hardware-faithful depth-to-XYZ pipeline that never reads privileged MuJoCo body-pose state (D-03)
- Found and fixed two genuine bugs in the back-projection math via a forward-project/back-project round-trip test methodology against real ground truth (not caught by the recovered draft, which had never actually been run to completion):
  1. `transform_from_pixels_to_world`'s `camera_to_world_transform` argument needs the inverse of the FULL pixel<->world transform (intrinsics + extrinsics), not the bare camera extrinsic — the bare-extrinsic version produced 90+ meter errors
  2. `robosuite`'s default `macros.IMAGE_CONVENTION = "opengl"` stores camera obs arrays row-0-at-bottom, not the row-0-at-top order `camera_utils`'s pixel math assumes — this produced a consistent ~7-9cm systematic bias until fixed
- After both fixes, `object_xyz_from_obs`'s estimate matches ground truth within ~2cm across repeated resets (well inside the empirically-set 5cm tolerance)
- Discovered and root-caused a local macOS/Apple-Silicon MuJoCo platform limitation: native instance-ID segmentation rendering (`mjRND_SEGMENT`/`mjRND_IDCOLOR`) is a complete no-op on this machine, confirmed at the lowest-level MuJoCo Python API, independent of robosuite/LIBERO, reproduced with both GLFW and CGL backends — worked around test-only via a ground-truth-anchored synthetic segmentation patch, with production code unaffected

## Task Commits

Each task was committed atomically (Task 2 followed TDD RED → GREEN):

1. **Task 1: Camera-config integration test** — `54a8a49` (test)
2. **Task 2 RED: Add failing D-04 ground-truth test** — `8972b95` (test)
2. **Task 2 GREEN: Implement depth-to-XYZ pipeline** — `a1e7ca2` (feat)

**Plan metadata:** committed with this SUMMARY.md (docs)

## Files Created/Modified

- `LIBERO/libero/libero/envs/test_camera_config.py` - real-sim integration test for RGB (SPAT-01) and depth (SPAT-03) extraction
- `LIBERO/libero/libero/perception/__init__.py` - graceful-degradation package init (vla/__init__.py pattern) + numpy2/robosuite compat shim for segmentation decode
- `LIBERO/libero/libero/perception/depth_xyz.py` - `pixel_to_world_xyz`, `object_pixel_centroid`, `object_xyz_from_obs`
- `LIBERO/libero/libero/perception/test_depth_xyz.py` - D-04 ground-truth validation test + pure-logic unit test

## Decisions Made

- Corrected `pixel_to_world_xyz`'s `camera_to_world_transform` argument from the recovered draft's bare `get_camera_extrinsic_matrix()` to `np.linalg.inv(get_camera_transform_matrix(...))` — required by `robosuite.utils.camera_utils.transform_from_pixels_to_world`'s actual internal math (it expects the un-normalized-pixel-scaled-by-depth homogeneous vector inverted back through the FULL world-to-pixel transform, not just the camera pose)
- Added an internal row-convention flip (both `depth_map` and `pixel_yx`'s row) inside `pixel_to_world_xyz`, converting from robosuite's default `"opengl"` obs-array convention (row-0-at-bottom) to the row-0-at-top convention `camera_utils`'s pixel math assumes — mirrors `get_camera_segmentation`'s own `[::-1]` self-flip in the same library, so both `depth_xyz.py`'s consumers of raw obs arrays are internally consistent
- Set `TOLERANCE_M = 0.05` in `test_depth_xyz.py`, well above the empirically observed max error (~0.022m) across repeated resets, per RESEARCH.md Pitfall 3's guidance to size tolerance from real measurement, not guesswork
- Test-only workaround for the local segmentation-rendering platform bug: `test_depth_xyz.py` constructs a small, ground-truth-anchored synthetic segmentation patch by forward-projecting the real ground-truth position through the same camera calibration `depth_xyz.py` uses, rather than reading MuJoCo's (locally broken) native segmentation render — clearly documented in the test file's module docstring, with a recommendation to re-verify on Colab (Linux osmesa/egl) where this specific GL-driver limitation is not expected to reproduce

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed wrong `camera_to_world_transform` argument in `pixel_to_world_xyz`**
- **Found during:** Task 2 (GREEN implementation, running the D-04 test against real ground truth)
- **Issue:** The recovered draft passed `get_camera_extrinsic_matrix()` (the bare camera pose) as `transform_from_pixels_to_world`'s `camera_to_world_transform` argument. That function's internal math treats the pixel input as an un-normalized (not-intrinsics-corrected) homogeneous vector scaled by real depth, requiring the caller to supply the INVERSE of the FULL world-to-pixel transform (`get_camera_transform_matrix`, intrinsics + extrinsics combined), not the bare extrinsic. The bug produced back-projected XYZ estimates off by 90+ meters.
- **Fix:** Changed to `np.linalg.inv(get_camera_transform_matrix(sim, camera_name, camera_height, camera_width))`.
- **Files modified:** `LIBERO/libero/libero/perception/depth_xyz.py`
- **Verification:** Forward-project/back-project round-trip against `sim.data.body_xpos` ground truth, confirmed error dropped from 90+ m to sub-meter (before the second fix below).
- **Committed in:** `a1e7ca2` (Task 2 GREEN commit)

**2. [Rule 1 - Bug] Fixed opengl-vs-standard image row-convention mismatch**
- **Found during:** Task 2 (GREEN implementation, same round-trip test)
- **Issue:** `robosuite`'s default `macros.IMAGE_CONVENTION = "opengl"` means `obs[f"{cam}_depth"]` and `obs[f"{cam}_segmentation_*"]` are stored row-0-at-bottom (raw OpenGL readback order), not the row-0-at-top order `robosuite.utils.camera_utils`'s pixel math assumes (confirmed by `get_camera_segmentation`'s own `[::-1]` self-flip in the same source file). Without correcting for this, the back-projection had a consistent, systematic ~7-9cm bias in every single reset (not random noise).
- **Fix:** `pixel_to_world_xyz` now flips both the incoming `depth_map` (`depth_map[::-1]`) and the incoming `pixel_yx`'s row (`(camera_height - 1) - row`) to the top-origin convention before calling into `camera_utils`.
- **Files modified:** `LIBERO/libero/libero/perception/depth_xyz.py`
- **Verification:** Round-trip error dropped from a consistent ~7-9cm bias to ~1.7-2.2cm (matching the expected object-center-vs-visible-surface offset for a small ~2x4x8cm object) across 10 repeated resets.
- **Committed in:** `a1e7ca2` (Task 2 GREEN commit)

**3. [Rule 3 - Blocking] Worked around a local MuJoCo/macOS segmentation-rendering platform limitation (test-only, no production code change)**
- **Found during:** Task 2 (GREEN implementation, running the recovered draft's original test)
- **Issue:** MuJoCo's native instance-ID segmentation rendering (`mjRND_SEGMENT`/`mjRND_IDCOLOR` scene flags, exercised via `camera_segmentations="instance"`) produces byte-identical output to a normal (non-segmentation) render on this local machine. Confirmed with the lowest-level MuJoCo Python API directly (`mjr_render`/`mjr_readPixels` with the flags manually toggled), completely independent of robosuite/LIBERO/env_wrapper.py, and reproduced with both the GLFW and CGL macOS GL backends on a minimal 2-geom test scene. This is a genuine upstream MuJoCo/macOS-Apple-Silicon OpenGL-driver limitation, not a bug in this project's code, and not fixable without modifying vendored files (which this plan's `<output>` section forbids — "No changes to any existing file").
- **Fix:** `test_depth_xyz.py`'s D-04 test constructs a small, ground-truth-anchored synthetic segmentation patch (marking pixels around the real ground-truth position's projected pixel location, computed via the same `camera_utils` calibration `depth_xyz.py` uses) as a test-only substitute for the broken render, clearly documented in the test file's module docstring. `env.instance_to_id` (a model-level property, unaffected by the rendering bug) is still exercised for real. Real depth (`obs[f"{cam}_depth"]`), the real environment, real physics, and the actual `object_xyz_from_obs`/`pixel_to_world_xyz`/`object_pixel_centroid` code under test are all exercised for real with zero mocking of `depth_xyz.py` itself.
- **Files modified:** `LIBERO/libero/libero/perception/test_depth_xyz.py` (test-only; `depth_xyz.py` production code is untouched by this workaround)
- **Verification:** Test passes with ~2cm error against ground truth across 3 resets; acceptance criteria (`grep -c 'body_xpos\|body_xquat' depth_xyz.py` returns 0, `grep -q 'body_xpos' test_depth_xyz.py` matches, `pytest ... -x -q` exits 0) all confirmed.
- **Committed in:** `a1e7ca2` (Task 2 GREEN commit)

---

**Total deviations:** 3 auto-fixed (2 Rule 1 bugs, 1 Rule 3 blocking-issue workaround)
**Impact on plan:** All three were essential for correctness (the two Rule 1 bugs would have shipped a completely broken back-projection pipeline) and for completing the plan at all given the local platform limitation. No scope creep — the production `depth_xyz.py` implementation matches the plan's exact specified function signatures and behavior; only the local test's object-identification input was adapted.

## Issues Encountered

- The recovered draft's `depth_xyz.py` implementation had never actually been run to completion (the prior session was cut off mid-verification per the recovery note). Both bugs above were latent and would have shipped silently broken if the D-04 ground-truth test hadn't been executed and its failure investigated rather than assumed to pass.
- Extensive low-level MuJoCo API debugging (`mj_ray`, raw `MjrContext`/`mjv_updateScene`/`mjr_render` calls, `Renderer.enable_segmentation_rendering`) was required to conclusively root-cause the segmentation-rendering platform limitation as upstream/environment-level rather than a code bug, before committing to the test-only workaround design.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- SPAT-01, SPAT-03, SPAT-04 all satisfied; Phase 5's three plans (05-01 multi-camera VLA wiring, 05-02 this plan, 05-03 spatial predicates+BDDL tasks) are now all complete
- `perception/` package is ready for use in later spatial-task authoring or Phase 6 fine-tuning work
- **Flag for Colab validation:** the local macOS/Apple-Silicon MuJoCo segmentation-rendering limitation documented above should be re-checked on Colab (Linux, osmesa/egl backends) before `object_xyz_from_obs` is used against REAL rendered segmentation observations in a downstream phase — if Colab's MuJoCo build also can't render `camera_segmentations="instance"` correctly, that would block any spatial-task work relying on real (non-synthetic) instance segmentation and would need its own investigation at that time
- `LIBERO/libero/libero/perception/__init__.py`'s numpy legacy-promotion shim (`_set_promotion_state("legacy")`) is process-global; confirmed no interference with other test suites (`test_spatial_predicates.py`, `test_camera_config.py`) when run in the same pytest session

---
*Phase: 05-spatial-awareness*
*Completed: 2026-08-10*

## Self-Check: PASSED

All created files confirmed present on disk (test_camera_config.py, perception/__init__.py, perception/depth_xyz.py, perception/test_depth_xyz.py, this SUMMARY.md). All commit hashes confirmed present in git log (54a8a49, 8972b95, a1e7ca2, 11012c1).
