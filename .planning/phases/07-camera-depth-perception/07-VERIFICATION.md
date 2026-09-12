---
phase: 07-camera-depth-perception
verified: 2026-09-12T07:01:29Z
status: human_needed
score: 9/11 must-haves verified
behavior_unverified: 2 # TFDS-gated depth-through-RLDS truths: code present + wired, but tensorflow_datasets is not installed locally so the behavior has never actually executed (importorskip skips before the new logic runs)
overrides_applied: 1
overrides:
  - must_have: "A rendered agentview frame visibly matches the real camera's framing (side-by-side/overlay comparison) [ROADMAP Phase 7 Success Criterion #1, second clause]"
    reason: "D-01/D-04 in 07-CONTEXT.md (gathered before planning, with explicit user input) documents that the real overhead camera has no finalized physical mount yet, and explicitly defers the literal side-by-side photo comparison until that mount is built and photographed -- Phase 7's own acceptance bar (07-01-PLAN.md's must_haves) was scoped down to 'a documented, non-degenerate, real-spec-derived recalibration, not a live photo match' with the user's own knowledge and approval during the discuss-phase conversation. This is a pre-planning scope decision, not an execution-time shortcut."
    accepted_by: "user (via 07-CONTEXT.md D-01/D-04, gathered 2026-09-05)"
    accepted_at: "2026-09-05"
human_verification:
  - test: "Run `pytest LIBERO/libero/libero/datasets/test_rlds_converter.py::test_hdf5_to_rlds_writes_tfds_loadable_dataset LIBERO/libero/libero/datasets/test_rlds_converter.py::test_hdf5_to_rlds_against_real_dataset -v` on Colab (or any environment with `tensorflow_datasets` installed)."
    expected: "Both tests pass. The first confirms the persisted `dataset_info.json` schema declares `agentview_depth` as a `(None, None, 1)` float32 `Tensor`. The second, run against the real pre-Phase-7 file `LIBERO/libero/datasets/soarm_spatial/put_the_cream_cheese_in_the_bowl_demo.hdf5` (confirmed present in this checkout, confirmed to lack `agentview_depth`), should hit the new pre-flight `h5py.File` check and `pytest.skip(\"real dataset predates Phase 7 depth persistence...\")` rather than crash with an unhandled `KeyError`."
    why_human: "`tensorflow_datasets` is not installed in the local `libero` conda env (confirmed this session: `import tensorflow_datasets` fails). Both tests are gated by `pytest.importorskip(\"tensorflow_datasets\")` and skip locally without ever executing the new depth-schema and legacy-skip code paths added by 07-03. This is a known, pre-existing environment constraint (already logged in STATE.md's Blockers/Concerns as '07 UNVERIFIED LOCALLY') -- the code is present, structurally correct, and reviewed, but has never actually run."
  - test: "Once the physical overhead-camera mount (D-01/D-02/D-03, tracked via the parallel physical-hardware track / diagnostics/UAT) is built and photographed, render `agentview` with the current recalibrated pos/quat/fovy and do a side-by-side/overlay comparison against the real photo."
    expected: "Sim framing visibly resembles the real camera's overhead view (matches ROADMAP Phase 7 Success Criterion #1's literal comparison clause)."
    why_human: "No real overhead-camera photo exists yet to compare against -- explicitly deferred per D-04, tracked as a follow-up outside Phase 7 completion, not a Phase 7 defect."
---

# Phase 7: Camera & Depth Perception Verification Report

**Phase Goal:** Camera recalibration grounded in real SOARM hardware specs (CAM-01), plus a depth observation pipeline that survives HDF5 -> RLDS -> OXE conversion for future fine-tuning (DEPTH-01, DEPTH-02, DEPTH-03).
**Verified:** 2026-09-12T07:01:29Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `agentview`'s pos/quat/fovy are recalibrated from real AR0144-hardware-derived specs, applied in the ONE reachable `_setup_camera()` override every registered SOARM task hits | VERIFIED | `LIBERO/libero/libero/envs/problems/libero_tabletop_manipulation.py:187-196` sets `pos=[0.5,0.0,1.45]`, `quat=[0.635968,0.309103,0.309103,0.635968]`, `camera_attribs={"fovy":"43"}`. Live test asserts `cam_fovy == 43.0` (`test_camera_config.py::test_rgb_cameras_non_degenerate_spatial`) — ran locally, PASSED. |
| 2 | The same `agentview` fix is mirrored defensively in the dead-code base class (`bddl_base_domain.py`), Phase-8 future-proofing | VERIFIED | `LIBERO/libero/libero/envs/bddl_base_domain.py:286-295` has an identical `agentview` `set_camera(...)` call with matching pos/quat/`camera_attribs`; `canonical_agentview` (a different camera) left untouched. |
| 3 | `agentview` rendered frame visibly matches the real camera's framing via side-by-side/overlay comparison [ROADMAP SC#1, literal clause] | PASSED (override) | No physical overhead mount exists yet (D-01) — the literal live-photo comparison is explicitly deferred per D-04, a pre-planning, user-approved scope decision documented in 07-CONTEXT.md. See `overrides` in frontmatter. |
| 4 | `eye_in_hand` is tilted toward the gripper's grasp point, matching the real IMX335 mount's photographed down/inward angle (not aimed straight out ahead) | VERIFIED (human-judgment, already resolved during execution) | `robot.xml:109`: `pos="0.0133 0.0017 -0.0718" quat="0.417709 -0.571718 0.576823 -0.407349"`. This is NOT the plan's original closed-form candidate — that candidate was rejected at the plan's own Task 3 human-verify checkpoint (rendered as a flat top-down close-up across a full recorded trajectory) and replaced via interactive human tuning, explicitly documented and checkpoint-authorized in 07-01-SUMMARY.md. Both camera tests still pass with the final value. |
| 5 | A depth camera stream is exposed as an observation key in the SOARM robosuite/LIBERO env (`env.reset()`/`env.step()`, alongside RGB) | VERIFIED | `test_camera_config.py::test_depth_cameras_non_degenerate_spatial` constructs a real env with `camera_depths=True` and asserts both `agentview_depth`/`robot0_eye_in_hand_depth` are non-degenerate `(128,128,1)` float arrays in `[0,1]` — ran locally, PASSED. (This capability pre-dates Phase 7 per 07-CONTEXT.md's own scouting; Phase 7's job was persistence, verified below.) |
| 6 | A recorded SOARM demo's HDF5 file contains a per-timestep `agentview_depth` dataset alongside RGB/state | VERIFIED | `hdf5_writer.py`: `OBS_KEY_MAPPING["agentview_depth"]`, `_DEPTH_KEYS = ("agentview_depth",)`, `camera_depths=True` on `OffScreenRenderEnv(...)`, dedicated write loop. `test_hdf5_writer.py::test_schema_and_obs_key_naming` asserts dtype `float32`, shape `(*,128,128,1)`, values in `[0,1]` — ran locally, PASSED (3/3 tests). |
| 7 | Depth is persisted raw-normalized `[0,1]` float32, NOT converted to meters (matches `depth_xyz.py`'s documented consumer contract) | VERIFIED | `hdf5_writer.py`'s depth write loop performs no unit conversion (comment explicitly states this); test asserts `np.all((depth_data >= 0.0) & (depth_data <= 1.0))`. |
| 8 | Depth is persisted for `agentview` only, never `eye_in_hand` (D-05, real-hardware fidelity) | VERIFIED | `_DEPTH_KEYS = ("agentview_depth",)` only; `test_schema_and_obs_key_naming` explicitly asserts `"eye_in_hand_depth" not in obs_keys` and `"robot0_eye_in_hand_depth" not in obs_keys` — ran locally, PASSED. |
| 9 | A converted TFDS/RLDS record built from a depth-augmented HDF5 includes a readable `agentview_depth` field, typed as a plain `Tensor` (not `Image`) | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | `rlds_converter.py` fully implements this (`validate_episode_arrays` depth branch, `load_episodes_from_hdf5` depth read, `_episode_to_rlds_steps` depth field, `FeaturesDict`'s `agentview_depth: tfds.features.Tensor(shape=(None,None,1), dtype=np.float32)`) — code reviewed line-by-line, matches plan spec exactly. BUT the one test that actually exercises this (`test_hdf5_to_rlds_writes_tfds_loadable_dataset`) is gated by `pytest.importorskip("tensorflow_datasets")`, and `tensorflow_datasets` is confirmed NOT installed locally — the test skips, the behavior has never executed. Non-TFDS unit tests (`validate_episode_arrays`, `load_episodes_from_hdf5`) DO run and pass, covering everything up to but not including the actual TFDS write/read. Route to human verification on Colab. |
| 10 | `oxe_register.py`'s `depth_obs_keys` (in-memory dict AND on-disk patch template) has `"primary": "agentview_depth"`, `"wrist": None` | VERIFIED | `oxe_register.py:92` (in-memory) and `:170` (on-disk template) both read `"depth_obs_keys": {"primary": "agentview_depth", "secondary": None, "wrist": None}`. Marker bumped `v2`->`v3` (`:163`). `test_oxe_register.py::test_register_soarm_spatial_injects_expected_dict_shape` — ran locally, PASSED (3/3 tests). |
| 11 | A pre-Phase-7 legacy HDF5 file does not silently corrupt or crash the real-dataset regression test — explicitly skipped with a documented reason | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | `test_hdf5_to_rlds_against_real_dataset` (`test_rlds_converter.py:210-228`) implements the exact pre-flight `h5py.File` check + `pytest.skip("real dataset predates Phase 7 depth persistence...")` described in the plan. Confirmed manually this session: the real file `LIBERO/libero/datasets/soarm_spatial/put_the_cream_cheese_in_the_bowl_demo.hdf5` IS present in this checkout and its `demo_1/obs` keys are `{agentview_rgb, eye_in_hand_rgb, gripper_states, joint_states}` — `agentview_depth` absent, confirming it would trigger the new skip branch. However, the same `pytest.importorskip("tensorflow_datasets")` gate at the top of this test means it currently skips via the "no tensorflow_datasets" path before ever reaching the file-open / legacy-check logic — the new code is structurally correct and its trigger condition is confirmed true, but the skip-not-crash behavior has never actually executed. This exact gap is independently documented in STATE.md's Blockers/Concerns as "07 UNVERIFIED LOCALLY." |

**Score:** 9/11 truths verified (2 present + wired, behavior-unverified pending a Colab TFDS run)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `LIBERO/libero/libero/envs/problems/libero_tabletop_manipulation.py` | `_setup_camera()` agentview pos/quat/fovy updated | VERIFIED | Confirmed lines 187-196. |
| `LIBERO/libero/libero/envs/bddl_base_domain.py` | `_setup_camera()` agentview pos/quat/fovy mirrored | VERIFIED | Confirmed lines 286-295; `canonical_agentview` untouched. |
| `LIBERO/libero/libero/assets/robots/soarm101/robot.xml` | `eye_in_hand` camera quat/pos updated | VERIFIED | Confirmed line 109 (final human-tuned value, not plan's literal candidate — checkpoint-authorized deviation, documented). |
| `LIBERO/libero/libero/envs/test_camera_config.py` | explicit `cam_fovy` assertion added | VERIFIED | Confirmed; both tests pass locally. |
| `LIBERO/libero/libero/datasets/hdf5_writer.py` | `OBS_KEY_MAPPING["agentview_depth"]`, `_DEPTH_KEYS`, `camera_depths=True`, depth write loop | VERIFIED | All four present, confirmed via grep + full read. |
| `LIBERO/libero/libero/datasets/test_hdf5_writer.py` | depth dtype/shape/range assertions, fixed stale gripper_states assertion | VERIFIED | Confirmed; `shape[1] == 2`; 3/3 tests pass locally. |
| `LIBERO/libero/libero/datasets/rlds_converter.py` | depth param+branch, HDF5 depth read, step depth field, `agentview_depth` Tensor FeaturesDict entry | VERIFIED | Confirmed all four insertion points via grep + read; matches plan spec exactly. |
| `LIBERO/libero/libero/datasets/oxe_register.py` | `depth_obs_keys["primary"]` in-memory + on-disk, marker v2->v3 | VERIFIED | Confirmed. |
| `LIBERO/libero/libero/datasets/test_rlds_converter.py`, `test_oxe_register.py` | extended for depth field + legacy-skip path | VERIFIED (structurally) | Non-TFDS tests run and pass (9/9 non-skipped); TFDS-gated tests (2) skip locally per the known environment constraint — see truths #9/#11 above. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `libero_tabletop_manipulation.py::_setup_camera()` | robosuite `MujocoArena.set_camera()` | `camera_attribs={"fovy": "43"}` kwarg | WIRED | Confirmed applied; live `cam_fovy` assertion passes against a real constructed env. |
| `hdf5_writer.py`'s `OffScreenRenderEnv(camera_depths=True)` | `gather_demonstrations_as_hdf5()`'s write loop | `obs_acc["agentview_depth"]` populated via `OBS_KEY_MAPPING`, written via `_DEPTH_KEYS` loop | WIRED | Confirmed real integration test (`test_schema_and_obs_key_naming`) exercises the full `OffScreenRenderEnv` -> `set_init_state()` -> write path end-to-end, no mocking. |
| `hdf5_writer.py`'s `agentview_depth` dataset (07-02) | `rlds_converter.py`'s `load_episodes_from_hdf5()` | literal key name `"agentview_depth"`, `float32`/`(H,W,1)` convention, no rename | WIRED (code); UNVERIFIED (runtime, TFDS-gated) | Key-name/dtype/shape convention matches exactly across both files (verified by reading both) but the actual `hdf5_to_rlds()` execution path is TFDS-gated and skips locally — see truth #9. |
| `rlds_converter.py`'s FeaturesDict `agentview_depth` Tensor | `oxe_register.py`'s `depth_obs_keys["primary"]` | Same literal string `"agentview_depth"` at both ends of the chain | WIRED | Confirmed identical string used in both files; `grep -c '"agentview_depth"'` in `oxe_register.py` returns >= 2 as required. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Camera config tests (agentview fovy=43, both cameras non-degenerate RGB+depth) | `pytest LIBERO/libero/libero/envs/test_camera_config.py -q` | `2 passed` | PASS |
| HDF5 writer depth persistence (real integration, no mocking) | `pytest LIBERO/libero/libero/datasets/test_hdf5_writer.py -q` | `3 passed` | PASS |
| RLDS converter + OXE register (non-TFDS subset) | `pytest LIBERO/libero/libero/datasets/test_rlds_converter.py LIBERO/libero/libero/datasets/test_oxe_register.py -q` | `11 passed, 2 skipped` (skips = `test_hdf5_to_rlds_writes_tfds_loadable_dataset`, `test_hdf5_to_rlds_against_real_dataset`, both via `importorskip("tensorflow_datasets")`) | PASS (with 2 documented, environment-gated skips) |
| No debt markers (`TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER`) in any of the 10 files this phase modified | `grep -n -E "TBD\|FIXME\|XXX\|TODO\|HACK\|PLACEHOLDER"` across all modified files | No matches | PASS |
| Legacy HDF5 file confirmed present and confirmed to lack `agentview_depth` (precondition for truth #11's skip branch) | `h5py.File(...)['data']['demo_1']['obs'].keys()` | `{agentview_rgb, eye_in_hand_rgb, gripper_states, joint_states}` — no `agentview_depth` | PASS (precondition confirmed; the skip branch itself is untested — see truth #11) |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| CAM-01 | 07-01 | Sim agentview recalibrated to real hardware specs | SATISFIED | Truths #1, #2 VERIFIED; #3 (literal photo match) PASSED via documented override. `eye_in_hand` scope addition (D-06/D-07) also satisfied (#4). |
| DEPTH-01 | 07-02 | Depth camera stream added and exposed as an observation | SATISFIED | Truth #5 VERIFIED (pre-existing capability, confirmed still working) + truths #6-8 (persistence) VERIFIED. |
| DEPTH-02 | 07-02 | Depth frames persisted alongside RGB in HDF5 writer | SATISFIED | Truths #6, #7, #8 VERIFIED via a real, non-mocked integration test. |
| DEPTH-03 | 07-03 | Depth frames carried through RLDS converter into fine-tuning dataset | SATISFIED (code-complete, runtime unverified) | Truths #9-11: full implementation verified by direct code reading and passing non-TFDS unit tests; the TFDS-dependent execution proof is blocked on a local environment gap (`tensorflow_datasets` not installed), a pre-existing constraint independently logged in STATE.md, not a Phase 7 defect. Requires a Colab-side spot check before full sign-off. |

**Note:** `.planning/REQUIREMENTS.md` (lines ~55-61, ~145-148) still lists CAM-01/DEPTH-01/02/03 as unchecked `[ ]` boxes and "Pending" in the traceability table, despite all four being implemented, tested, and marked `requirements-completed` in the corresponding SUMMARY.md frontmatter. This is a documentation-staleness gap, not a code gap — REQUIREMENTS.md was not updated as part of phase closeout. Recommend updating it as a fast-follow (does not block phase goal achievement, since the underlying implementation is verified independently against the codebase).

### Anti-Patterns Found

None. No `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` markers found in any of the 10 files modified across the three plans. No stub returns, no hardcoded empty data, no console.log-only handlers (this is backend/simulation Python, not applicable UI patterns, but the equivalent checks — silent `.get()` fallbacks masking missing depth keys, papered-over legacy-schema handling — were explicitly checked and found absent; the plan's own design deliberately avoids silent fallbacks in favor of fail-loud + explicit test-level skips).

### Human Verification Required

See frontmatter `human_verification`. Summary:

1. **Colab TFDS spot check (DEPTH-03 runtime proof)** — Run the two `tensorflow_datasets`-gated tests (`test_hdf5_to_rlds_writes_tfds_loadable_dataset`, `test_hdf5_to_rlds_against_real_dataset`) in an environment where `tensorflow_datasets` is installed (Colab). This is the only way to confirm the TFDS schema write/read round-trip and the legacy-HDF5-skip branch actually execute correctly, not just that the code is structurally correct. Already tracked in STATE.md's Blockers/Concerns as "07 UNVERIFIED LOCALLY" — not a new finding, but carried forward here as a required closeout item.
2. **Real-camera side-by-side photo comparison for `agentview`** — Deferred per D-04 until the physical overhead mount is built and photographed (tracked via the parallel physical-hardware track, not blocking Phase 7).

### Gaps Summary

No blocking gaps. All must-haves from all three plans (07-01, 07-02, 07-03) are implemented exactly as specified, verified by direct code reading against the actual files (not SUMMARY.md prose), and covered by passing automated tests where the local environment permits. The one meaningful open item — DEPTH-03's TFDS-dependent execution path — is code-complete and structurally verified (including confirming its trigger precondition against the real legacy HDF5 file present on disk) but has never actually run due to a pre-existing, independently-documented local environment gap (`tensorflow_datasets` not installed), not a defect introduced by this phase. This routes the phase to `human_needed` rather than `passed`, pending a Colab-side confirmation run before full sign-off — consistent with STATE.md's own tracking of this exact item.

A secondary, non-blocking finding: `.planning/REQUIREMENTS.md`'s checklist/traceability table was not updated to reflect Phase 7's completion (still shows CAM-01/DEPTH-01/02/03 as "Pending"/unchecked) — a documentation housekeeping item, not a code gap.

---

*Verified: 2026-09-12T07:01:29Z*
*Verifier: Claude (gsd-verifier)*
