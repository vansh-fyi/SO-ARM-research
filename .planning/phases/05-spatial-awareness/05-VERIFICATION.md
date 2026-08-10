---
phase: 05-spatial-awareness
verified: 2026-08-10T05:32:34Z
status: human_needed
score: 10/12 must-haves verified
behavior_unverified: 2 # OFTBackend real GPU dual-image consumption; object_xyz_from_obs's real (non-synthetic) segmentation-render integration
overrides_applied: 0
behavior_unverified_items:
  - truth: "OFTBackend.predict() packs both camera views into a single multi-image forward pass and the checkpoint's num_images_in_input=2 pattern is actually consumed correctly (D-02, SPAT-02)."
    test: "On a Colab GPU runtime, call OFTBackend.predict({'eye_in_hand': <frame>, 'agentview': <frame>}, language) against the real moojink/openvla-7b-oft-finetuned-libero-spatial checkpoint and confirm the returned (8,7) action chunk is sane/non-degenerate for a scene where the two views show materially different content."
    expected: "The model consumes both concatenated image embeddings (not silently ignoring the second view) and produces a plausible action chunk."
    why_human: "This project has no local GPU/torch install (confirmed: `import torch` raises ModuleNotFoundError in the local libero conda env), so only source-level assertions (torch.cat, dim=1, images[\"agentview\"], primary_inputs[\"pixel_values\"] reassignment) could be checked locally. The plan and SUMMARY.md both explicitly flag this as Colab-verification-pending; zero local pytest coverage exists for oft_backend.py, matching this project's established convention for GPU-only-verifiable backends."
  - truth: "object_xyz_from_obs correctly locates an object via MuJoCo's real native instance-segmentation render (not a synthetic substitute) and back-projects it to world XYZ within tolerance of ground truth (SPAT-04)."
    test: "On a Colab Linux runtime (osmesa/egl backend), construct a SegmentationRenderEnv with camera_segmentations='instance', call obs = env.reset(), and confirm obs['agentview_segmentation_instance'] is non-degenerate (not all-background) and that object_xyz_from_obs(...) using the REAL segmentation array (not the test's synthetic ground-truth-anchored patch) still matches sim.data.body_xpos within the same ~0.05m tolerance."
    why_human: "The verifier confirmed (via SUMMARY.md's documented investigation and by reading test_depth_xyz.py's module docstring) a genuine local macOS/Apple-Silicon MuJoCo platform limitation: native instance-ID segmentation rendering is a byte-identical no-op to a normal render on this machine, reproduced at the lowest-level MuJoCo API, independent of this project's code. test_depth_xyz.py's D-04 test therefore substitutes a small segmentation patch built by forward-projecting the REAL ground-truth position through the same camera calibration depth_xyz.py uses — this genuinely exercises pixel_to_world_xyz and the real depth buffer, but does NOT exercise the real segmentation-render-to-object_pixel_centroid path end-to-end. That specific link needs Colab (Linux) re-verification before object_xyz_from_obs is trusted against real (non-synthetic) segmentation observations."
human_verification:
  - test: "Colab GPU spot-check: run OFTBackend.predict() with two genuinely different camera frames and confirm the model's output action chunk reflects real dual-image consumption, not a silently-ignored second view."
    expected: "Non-degenerate, plausible (8,7) action chunk; ideally compare against a single-image baseline to confirm the second view changes the output."
    why_human: "No local GPU/torch; only grep-based source assertions were possible locally (all passed)."
  - test: "Colab Linux (osmesa/egl) re-run of test_depth_xyz.py's D-04 test using MuJoCo's real instance-segmentation render instead of the test's local ground-truth-anchored synthetic patch."
    expected: "obs['agentview_segmentation_instance'] is non-degenerate on Colab's GL backend, and object_xyz_from_obs's back-projected estimate still matches sim.data.body_xpos ground truth within ~0.05m using the REAL segmentation array."
    why_human: "A confirmed macOS/Apple-Silicon MuJoCo OpenGL-driver limitation makes real segmentation rendering unusable on this local dev machine; this is documented as a known, not-yet-closed verification gap in the plan's own SUMMARY.md, not something grep/source review can resolve."
---

# Phase 5: Spatial Awareness Verification Report

**Phase Goal:** The SOARM environment supports multi-camera RGB input, depth-based 3D object localization, and at least 3 BDDL spatial language task variants for benchmarking.
**Verified:** 2026-08-10T05:32:34Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Both camera views (agentview + eye_in_hand) present in the images dict passed to `backend.predict()` on every eval_loop episode step (D-01, SPAT-02) | ✓ VERIFIED | `eval_loop.py:68-71` builds `images = {"eye_in_hand": obs[camera_name], "agentview": obs[agentview_camera_name]}`; `test_eval_loop.py::test_images_dict_includes_both_camera_views_spatial` passes (5/5 tests, ran locally: `5 passed in 0.23s`) |
| 2 | Pi0Backend sends genuinely distinct base/wrist images to the openpi policy server — no duplication (D-02, SPAT-02) | ✓ VERIFIED | `pi0_backend.py:132-137` builds `base_img` from `images["agentview"]` and `wrist_img` from `images["eye_in_hand"]` independently; `test_pi0_backend.py::test_predict_uses_distinct_base_and_wrist_images_spatial` passes (7/7 tests, ran locally: `7 passed in 0.07s`) |
| 3 | OFTBackend packs both views into a single multi-image forward pass using `num_images_in_input=2` pattern (D-02, SPAT-02) | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | `oft_backend.py:154-167` shows `primary`/`extra_views`/`torch.cat(..., dim=1)` on `pixel_values`; source-grep assertions all pass. Real GPU-backed multi-image consumption is unverified — no local torch/GPU. Routed to human verification. |
| 4 | Constructing a SOARM LIBERO env with the default camera config yields non-degenerate RGB frames from both agentview and eye_in_hand (SPAT-01, re-verification) | ✓ VERIFIED | `test_camera_config.py::test_rgb_cameras_non_degenerate_spatial` — real (no-mock) integration test, ran locally: `2 passed in 5.40s` (both tests in file) |
| 5 | Constructing with `camera_depths=True` yields non-degenerate depth obs for both cameras, values in [0,1] (SPAT-03) | ✓ VERIFIED | `test_camera_config.py::test_depth_cameras_non_degenerate_spatial` — same run, passed; zero `env_wrapper.py` source diff confirmed (`git log` shows env_wrapper.py touched only by pure case-rename commits `4b241ee`/`8451bca`, no logic change) |
| 6 | A pixel + depth buffer can be back-projected to world-frame XYZ matching MuJoCo's own ground-truth object position within an empirically-measured tolerance (SPAT-04, D-04) | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | `depth_xyz.py`'s `pixel_to_world_xyz`/`object_xyz_from_obs` are real and were validated against real `sim.data.body_xpos` ground truth (`test_depth_xyz.py::test_object_xyz_matches_ground_truth_within_tolerance` passes: `2 passed in 5.09s`), and two genuine projective-geometry bugs were found+fixed via this process (strong positive signal). However, the object-identification half of the pipeline (`object_pixel_centroid` fed by real MuJoCo segmentation render) is exercised only via a test-only synthetic, ground-truth-anchored segmentation patch — a documented local macOS/Apple-Silicon MuJoCo platform limitation makes real segmentation rendering unusable on this machine. Routed to human verification. |
| 7 | The depth-to-XYZ pipeline never reads `sim.data.body_xpos`/`body_xquat` in its production code path (D-03) | ✓ VERIFIED | `grep -c 'body_xpos\|body_xquat' depth_xyz.py` returns `0`; `grep -q 'body_xpos' test_depth_xyz.py` matches (test-only oracle present) |
| 8 | At least 3 BDDL tasks exist whose goal predicates evaluate genuine spatial relations (left/right, near, between) via real position logic, not just plain On/In checks (SPAT-05) | ✓ VERIFIED | `libero_spatial_soarm/*.bddl` — 3 files confirmed on disk with `:goal` blocks `(RightOfX ...)`, `(NearTo ...)`, `(And (RightOfX ...) (LeftOfX ...))` |
| 9 | Each new spatial predicate correctly distinguishes true and false spatial configurations (unit-level) | ✓ VERIFIED | `test_spatial_predicates.py`'s `TestLeftOfX`/`TestRightOfX`/`TestNearTo`/`TestFarFrom` (10 unit tests) — part of the 16/16 passing run (`16 passed in 96.97s`) |
| 10 | Each new BDDL task's init-region placement satisfies its own goal predicate at reset, for every point in the randomized init range — no ambiguous boundary cases (D-09) | ✓ VERIFIED | `test_right_of_task_resets_satisfy_goal_20_of_20`, `test_near_task_resets_satisfy_goal_20_of_20`, `test_between_task_resets_satisfy_goal_20_of_20` all `assert successes == 20` (not just printed) — confirmed by reading the test source directly; all pass |
| 11 | Task success/failure scoring uses MuJoCo's privileged sim-state ground truth (D-05), matching every other existing LIBERO task's scoring mechanism | ✓ VERIFIED | All 4 new predicates call `arg.get_geom_state()`, which reads `self.env.sim.data.body_xpos`/`body_xquat` directly (`base_object_states.py:47-52`) — same mechanism as pre-existing `On`/`Stack` predicates |
| 12 | The new 3rd-object region (butter_1, "between" task) is empirically validated collision-free at rest, not assumed safe (D-06, Pitfall 4) | ✓ VERIFIED | `test_between_task_no_robot_butter_contact_at_rest_20_of_20` — `assert contact_count == 0` across 20 resets using `env.env.check_contact(robot_model, butter_model)`; part of the passing 16/16 run |

**Score:** 10/12 truths verified (2 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `LIBERO/libero/libero/vla/eval_loop.py` | `images` dict includes both camera keys | ✓ VERIFIED | Present, substantive, wired — exercised by passing tests |
| `LIBERO/libero/libero/vla/interface.py` | Docstring updated to Phase 5 dual-key contract | ✓ VERIFIED | Docstring confirmed updated (no longer states single-key) |
| `LIBERO/libero/libero/vla/pi0_backend.py` | Distinct base/wrist image sourcing | ✓ VERIFIED | Present, substantive, wired, test-passing |
| `LIBERO/libero/libero/vla/oft_backend.py` | Dual-image `torch.cat` packing | ✓ VERIFIED (source) / ⚠️ GPU behavior unverified | Present, substantive per source read; no local test coverage (documented, expected) |
| `LIBERO/libero/libero/vla/test_eval_loop.py`, `test_pi0_backend.py` | New spatial tests | ✓ VERIFIED | 5/5 and 7/7 passing locally |
| `LIBERO/libero/libero/envs/test_camera_config.py` | Real RGB+depth integration tests | ✓ VERIFIED | 2/2 passing locally, real (no-mock) sim |
| `LIBERO/libero/libero/perception/__init__.py` | Graceful-degradation package init | ✓ VERIFIED | Present; try/except pattern confirmed; imports resolve (`pixel_to_world_xyz` etc. importable) |
| `LIBERO/libero/libero/perception/depth_xyz.py` | 3 functions per spec | ✓ VERIFIED | `pixel_to_world_xyz`, `object_pixel_centroid`, `object_xyz_from_obs` all present, D-03-compliant |
| `LIBERO/libero/libero/perception/test_depth_xyz.py` | D-04 ground-truth validation | ✓ VERIFIED (with disclosed synthetic-segmentation caveat) | 2/2 passing locally |
| `LIBERO/libero/libero/envs/predicates/base_predicates.py` | 4 new predicate classes | ✓ VERIFIED | `LeftOfX`/`RightOfX`/`NearTo`/`FarFrom` present, correct logic |
| `LIBERO/libero/libero/envs/predicates/__init__.py` | Registry update | ✓ VERIFIED | 4 keys registered, zero collisions with 10 pre-existing keys |
| `LIBERO/libero/libero/envs/test_spatial_predicates.py` | Unit + integration tests | ✓ VERIFIED | 16/16 passing locally (`96.97s`) |
| `libero_spatial_soarm/*.bddl` (3 files) | 3 new spatial BDDL tasks | ✓ VERIFIED | Exactly 3 files present, goals use the new predicates |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `eval_loop.run_episode`'s images dict | `backend.predict(images, language)` | direct call | ✓ WIRED | Confirmed by passing dict-plumbing test |
| `images["agentview"]`/`images["eye_in_hand"]` | `Pi0Backend.predict()` internals | independent resize/convert | ✓ WIRED | Confirmed non-duplication test passes |
| `images["eye_in_hand"]`/`images["agentview"]` | `OFTBackend.predict()`'s `torch.cat` | processor + concat | ⚠️ WIRED (source only) | Source-verified; GPU runtime behavior not exercised locally |
| `depth_xyz.pixel_to_world_xyz()` | `robosuite.utils.camera_utils` | `get_real_depth_map`/`get_camera_transform_matrix`/`transform_from_pixels_to_world` | ✓ WIRED | Confirmed via real test passing + 2 real bugs found/fixed during this validation |
| `depth_xyz.object_xyz_from_obs()` | `{cam}_segmentation_instance` + `{cam}_depth` obs keys | `env_wrapper.py`'s existing kwargs (zero source changes) | ⚠️ WIRED (partial) | Depth path fully real; segmentation path exercised via synthetic ground-truth-anchored patch locally (platform limitation), not real MuJoCo segmentation render |
| `test_depth_xyz.py`'s D-04 oracle | `env.sim.data.body_xpos[...]` | test-only ground truth read | ✓ WIRED | Confirmed present only in test file, never in `depth_xyz.py` |
| BDDL `:goal` predicate names | `VALIDATE_PREDICATE_FN_DICT` → `_eval_predicate()` dispatch | unchanged dispatcher | ✓ WIRED | Confirmed via 20/20-reset passing integration tests for all 3 new tasks |
| "Between" goal | 2 binary predicates (`RightOfX` + `LeftOfX`) | `And` decomposition | ✓ WIRED | Confirmed in `put_the_butter_between_the_bowl_and_the_cream_cheese.bddl` |

### Behavioral Spot-Checks / Test Execution (run directly by verifier, not taken from SUMMARY.md)

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| VLA dict-plumbing (SPAT-02) | `conda run -n libero pytest LIBERO/libero/libero/vla/test_eval_loop.py -x -q` | `5 passed in 0.23s` | ✓ PASS |
| Pi0Backend non-duplication (D-02) | `conda run -n libero pytest LIBERO/libero/libero/vla/test_pi0_backend.py -x -q` | `7 passed in 0.07s` | ✓ PASS |
| OFTBackend source assertions (D-02) | `grep` for `images["agentview"]`, `torch.cat`, `dim=1`, `primary_inputs["pixel_values"]` | all matched | ✓ PASS (source only) |
| Camera RGB/depth real integration (SPAT-01/03) | `conda run -n libero pytest LIBERO/libero/libero/envs/test_camera_config.py -x -q` | `2 passed in 5.40s` | ✓ PASS |
| Depth-to-XYZ D-04 ground truth (SPAT-04) | `conda run -n libero pytest LIBERO/libero/libero/perception/test_depth_xyz.py -x -q` | `2 passed in 5.09s` | ✓ PASS |
| Spatial predicates + BDDL tasks full suite (SPAT-05) | `conda run -n libero pytest LIBERO/libero/libero/envs/test_spatial_predicates.py -x -q` | `16 passed in 96.97s` | ✓ PASS |
| D-03 anti-pattern gate | `grep -c 'body_xpos\|body_xquat' depth_xyz.py` | `0` | ✓ PASS |
| Zero `env_wrapper.py` production diff | `git log --follow -- env_wrapper.py` | only case-rename commits (`4b241ee`, `8451bca`) touch it, no logic diff vs. pre-phase-5 | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| SPAT-01 | 05-02 | At least 2 camera views (wrist + overhead) configured in LIBERO SOARM environments | ✓ SATISFIED | `test_camera_config.py::test_rgb_cameras_non_degenerate_spatial` passes; zero `env_wrapper.py` changes needed (kwargs already present) |
| SPAT-02 | 05-01 | All camera views are passed as input to the VLA during inference | ✓ SATISFIED (dict-plumbing + Pi0 verified; OFT source-verified, GPU-behavior pending) | `test_eval_loop.py`, `test_pi0_backend.py` pass; `oft_backend.py` source-verified |
| SPAT-03 | 05-02 | MuJoCo depth buffer frames are extracted alongside RGB frames | ✓ SATISFIED | `test_camera_config.py::test_depth_cameras_non_degenerate_spatial` passes |
| SPAT-04 | 05-02 | Object XYZ positions are extracted from MuJoCo state and available as structured context | ✓ SATISFIED (core pipeline verified; real-segmentation integration pending) | `test_depth_xyz.py` passes against real ground truth; segmentation-render input is test-synthetic on this machine |
| SPAT-05 | 05-03 | At least 3 BDDL tasks use spatial language prompts and success predicates correctly evaluate spatial conditions | ✓ SATISFIED | 3 BDDL tasks + 16/16 passing tests (unit + 20-reset integration + directional + collision checks) |

No orphaned requirements found — REQUIREMENTS.md's Phase 5 mapping (SPAT-01..05, all marked Complete) exactly matches the union of `requirements:` fields declared across `05-01-PLAN.md`, `05-02-PLAN.md`, `05-03-PLAN.md`.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `LIBERO/libero/libero/envs/predicates/base_predicates.py` | 74 | `TODO (Yfeng): ...` | ℹ️ Info | Pre-existing comment from the original LIBERO import commit (`b59f55d`), inside dead/commented-out code in the pre-existing `On` class — not touched or introduced by this phase (confirmed via `git log -L`). Not a debt marker introduced by Phase 5. |

No other TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER, empty-implementation, or hardcoded-empty-data patterns found in any of the 14 phase-touched files.

### Regression Note (out of phase scope, flagged for visibility per additional context)

Two pre-existing test failures were found in `LIBERO/libero/libero/datasets/`:
- `test_hdf5_writer.py::test_schema_and_obs_key_naming`
- `test_replay.py::test_verify_full_obs_regeneration_passes_on_04_02_output`

Verifier re-ran both directly: `2 failed, 5 passed in 73.40s`, confirming the additional-context claim. `git log` confirms neither test file nor `env_wrapper.py` (nor any other file these tests exercise) was touched by any Phase 5 plan's substantive commits — only by the pure case-only-rename commits `4b241ee`/`8451bca` (which fixed an accidental macOS-case-insensitivity bug from the 05-03 merge, re-registering the entire `LIBERO/` tree as lowercase `libero/` in git's index). These are Phase 4 issues, out of SPAT-01..05 scope, and do not block Phase 5 completion — consistent with STATE.md's Blockers/Concerns section.

### Human Verification Required

#### 1. OFTBackend real GPU dual-image consumption (Colab)

**Test:** On a Colab GPU runtime, call `OFTBackend.predict({'eye_in_hand': <frame>, 'agentview': <frame>}, language)` against the real `moojink/openvla-7b-oft-finetuned-libero-spatial` checkpoint using two genuinely different camera frames.
**Expected:** A non-degenerate, plausible `(8, 7)` action chunk that reflects real consumption of both concatenated image embeddings (not silently ignoring the second view).
**Why human:** No local GPU/torch install exists in this project's dev environment (`import torch` raises `ModuleNotFoundError` locally, confirmed). Only source-level `grep` assertions (torch.cat, dim=1, `images["agentview"]`, `primary_inputs["pixel_values"]` reassignment) were checkable locally — all passed, but tensor-shape correctness does not prove the model semantically uses the second view correctly.

#### 2. Real MuJoCo segmentation-render re-verification (Colab)

**Test:** On a Colab Linux runtime (osmesa/egl backend), construct a `SegmentationRenderEnv` with `camera_segmentations="instance"`, call `env.reset()`, and confirm `obs["agentview_segmentation_instance"]` is non-degenerate (not all-background). Then re-run `object_xyz_from_obs(...)` using that REAL segmentation array (not `test_depth_xyz.py`'s synthetic ground-truth-anchored patch) and confirm the estimate still matches `sim.data.body_xpos` within ~0.05m.
**Expected:** Real segmentation rendering works correctly on Colab's Linux GL backend (unlike the local macOS/Apple-Silicon machine), and the full `object_xyz_from_obs` pipeline — using real, not synthetic, segmentation input — still meets the D-04 tolerance.
**Why human:** A confirmed macOS/Apple-Silicon MuJoCo OpenGL-driver limitation makes `camera_segmentations="instance"` render byte-identical output to a normal render on this local dev machine (root-caused at the lowest-level MuJoCo API, independent of this project's code). This is explicitly documented as an open item in `05-02-SUMMARY.md`'s "Next Phase Readiness" section — the executor itself flagged it as unresolved, not something a source read or grep can close out.

### Gaps Summary

No gaps found — all must-have artifacts exist, are substantive, and are wired; all locally-runnable tests (36 total across the phase: 5+7+2+2+16, plus grep-based source assertions for the GPU-only backend) pass when run directly by the verifier (not merely trusted from SUMMARY.md). The phase's own SUMMARY.md files honestly disclosed two specific behavioral-verification limits (OFTBackend GPU consumption; real MuJoCo segmentation rendering on this local machine) that neither `grep` nor a local pytest run can close — both are appropriately routed to human/Colab verification rather than silently marked passed. Two out-of-scope, pre-existing Phase 4 test regressions were independently confirmed and are documented for visibility only; they do not block this phase's SPAT-01..05 goal.

---

_Verified: 2026-08-10T05:32:34Z_
_Verifier: Claude (gsd-verifier)_
