---
phase: 7
slug: camera-depth-perception
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-09-05
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (no `pytest.ini`/`conftest.py` at repo root; relies on pytest's "prepend" import-mode auto-discovery, matching `test_camera_config.py`'s own module docstring) |
| **Config file** | none — Wave 0 extends existing test files |
| **Quick run command** | `pytest LIBERO/libero/libero/envs/test_camera_config.py -x` |
| **Full suite command** | `pytest LIBERO/libero/libero/envs/ LIBERO/libero/libero/datasets/ -x` |
| **Estimated runtime** | ~30-60 seconds (no GPU needed for camera/HDF5 tests; RLDS/TFDS tests are Colab-only) |

---

## Sampling Rate

- **After every task commit:** Run `pytest LIBERO/libero/libero/envs/test_camera_config.py -x`
- **After every plan wave:** Run `pytest LIBERO/libero/libero/envs/ LIBERO/libero/libero/datasets/ -x`
- **Before `/gsd-verify-work`:** Full suite must be green, plus a manual visual-comparison step for both camera-angle corrections (no automated oracle exists for "does this look right")
- **Max feedback latency:** ~60 seconds (local, no GPU dependency for the fast subset)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 07-01-xx | TBD | 1 | CAM-01 | — / V5 | `agentview` fovy applied in `libero_tabletop_manipulation.py`'s `_setup_camera()`, not the dead-code base class | integration | `pytest LIBERO/libero/libero/envs/test_camera_config.py::test_rgb_cameras_non_degenerate_spatial -x` (extend with explicit `sim.model.cam_fovy` assertion) | ❌ W0 — extend existing | ⬜ pending |
| 07-01-xx | TBD | 1 | CAM-01 | — | `eye_in_hand` quat corrected to tilt toward grip site, matches reference photos | manual-only | N/A — visual inspection against `progress-documentation/images/20260827_121*.jpg` | — | ⬜ pending |
| 07-02-xx | TBD | 1 | DEPTH-01 | — | `agentview_depth` obs key present, non-degenerate, shape `(H,W,1)`, values in `[0,1]` | integration | `pytest LIBERO/libero/libero/envs/test_camera_config.py::test_depth_cameras_non_degenerate_spatial -x` (already exists, already passes) | ✅ already exists | ⬜ pending |
| 07-03-xx | TBD | 2 | DEPTH-02 | — / V5 | HDF5 file contains `demo_N/obs/agentview_depth`, correct dtype/shape, fail-loud on non-finite/wrong-shape input (matches existing `_RGB_KEYS`/`_STATE_KEYS` validation pattern) | integration | `pytest LIBERO/libero/libero/datasets/test_hdf5_writer.py -x` (extend `test_schema_and_obs_key_naming` with a depth assertion) | ❌ W0 — extend existing | ⬜ pending |
| 07-04-xx | TBD | 3 | DEPTH-03 | — / V5 | Converted TFDS record includes readable `agentview_depth` field as `tfds.features.Tensor` (not `Image`); `oxe_register.py`'s `depth_obs_keys["primary"]` set correctly | integration (Colab-only, `pytest.importorskip("tensorflow_datasets")`) | `pytest LIBERO/libero/libero/datasets/test_rlds_converter.py -x` (extend `test_hdf5_to_rlds_writes_tfds_loadable_dataset`) | ❌ W0 — extend existing | ⬜ pending |
| 07-04-xx | TBD | 3 | DEPTH-03 | — | `oxe_register.py`'s `depth_obs_keys` dict matches `{"primary": "agentview_depth", "secondary": None, "wrist": None}` | unit | `pytest LIBERO/libero/libero/datasets/test_oxe_register.py -x` (extend `test_register_soarm_spatial_injects_expected_dict_shape`) | ❌ W0 — extend existing | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*
*Note: exact Task IDs assigned by the planner; this table maps requirements to test extension points identified during research.*

---

## Wave 0 Requirements

- [ ] `test_hdf5_writer.py` — extend `test_schema_and_obs_key_naming` (or add a new test) to assert `demo_1/obs/agentview_depth` exists with `dtype==float32`, `shape[-1]==1`, values in `[0,1]`
- [ ] `test_rlds_converter.py` — extend `validate_episode_arrays` (or add a parallel depth-validation function) to check `agentview_depth`'s dtype/shape/finite-values before RLDS write, matching the existing fail-loud pattern for RGB/state/actions
- [ ] `test_oxe_register.py` — extend `test_register_soarm_spatial_injects_expected_dict_shape` to assert `depth_obs_keys == {"primary": "agentview_depth", "secondary": None, "wrist": None}`
- [ ] `test_camera_config.py` — add an explicit `sim.model.cam_fovy[agentview_cam_id] == <recalibrated value>` assertion (currently only checks non-degeneracy, not the specific fovy value)

**Pre-existing, unrelated failures noted in STATE.md that touch the same files this phase modifies** (per RESEARCH.md): `test_schema_and_obs_key_naming`'s stale `gripper_states.shape[1]==1` assertion, and a `test_replay.py` pixel-mismatch. Not required by this phase's scope, but the planner should decide explicitly whether to fix them alongside since the diff touches adjacent lines anyway.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `agentview` recalibrated framing plausibly resembles an overhead workspace view at the AR0144's ~43° vertical FOV | CAM-01 | No ground-truth real photo exists yet (D-04 defers the live real-vs-sim comparison until the physical mount is built) — this phase can only verify the numeric params are documented and non-degenerate, not a pixel-level match | Render a frame from the recalibrated `agentview` camera and visually confirm it looks like a plausible overhead workspace shot (not degenerate/black/out-of-bounds) |
| `eye_in_hand` corrected quat visually matches the real mount's tilt | CAM-01 (scope addition, D-06/D-07) | Framing correctness against a photo is not a numeric assertion this project can automate without photogrammetry | Render a frame with the candidate quat (start: look-at-grip-site quat from RESEARCH.md Pattern 2) and compare against `progress-documentation/images/20260827_121*.jpg` — jaws should be visible/centered; iterate the quat if not, per RESEARCH.md Open Question 1 |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
