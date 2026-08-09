---
phase: 05-spatial-awareness
plan: 01
subsystem: vla
tags: [libero, openpi, openvla-oft, multi-camera, spatial-awareness]

# Dependency graph
requires:
  - phase: 03-vla-inference-loop
    provides: eval_loop.py's run_episode/run_suite, VLABackend Protocol, Pi0Backend and OFTBackend implementations (single eye_in_hand-image baseline)
provides:
  - eval_loop.py's images dict now includes both "eye_in_hand" and "agentview" keys on every episode step
  - Pi0Backend sends genuinely distinct base ("agentview") and wrist ("eye_in_hand") images to the openpi policy server
  - OFTBackend packs both views into a single multi-image forward pass via torch.cat pixel_values concatenation
affects: [05-spatial-awareness plans depending on multi-camera VLA input, Phase 6 fine-tuning]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Dual-image VLA backend input: eval_loop sources both camera obs keys into a single images dict; each backend independently resizes/processes both views before its single inference call"]

key-files:
  created: []
  modified:
    - LIBERO/libero/libero/vla/eval_loop.py
    - LIBERO/libero/libero/vla/interface.py
    - LIBERO/libero/libero/vla/test_eval_loop.py
    - LIBERO/libero/libero/vla/pi0_backend.py
    - LIBERO/libero/libero/vla/test_pi0_backend.py
    - LIBERO/libero/libero/vla/oft_backend.py

key-decisions:
  - "eval_loop's camera_name kwarg (video-writer camera) left untouched; a new agentview_camera_name kwarg added instead of repurposing an existing parameter"
  - "Pi0Backend maps agentview->observation/image (base) and eye_in_hand->observation/wrist_image (wrist), matching openpi's native libero_policy.py obs-key contract"
  - "OFTBackend packs eye_in_hand as primary and agentview as the extra view via torch.cat(..., dim=1); real GPU-behavioral correctness remains Colab-verification-pending (no local torch)"

patterns-established:
  - "Source-only verification (grep-based) is acceptable for GPU-dependent backend code with zero local test coverage, per this project's established Colab-verification-pending convention (03-RESEARCH.md, 05-VALIDATION.md)"

requirements-completed: [SPAT-02]

coverage:
  - id: D1
    description: "eval_loop.py's images dict includes both eye_in_hand and agentview keys on every episode step, sourced correctly (not swapped)"
    requirement: "SPAT-02"
    verification:
      - kind: unit
        ref: "LIBERO/libero/libero/vla/test_eval_loop.py#test_images_dict_includes_both_camera_views_spatial"
        status: pass
    human_judgment: false
  - id: D2
    description: "Pi0Backend sends two genuinely distinct images to the policy server (agentview->observation/image, eye_in_hand->observation/wrist_image), no duplication"
    requirement: "SPAT-02"
    verification:
      - kind: unit
        ref: "LIBERO/libero/libero/vla/test_pi0_backend.py#test_predict_uses_distinct_base_and_wrist_images_spatial"
        status: pass
    human_judgment: false
  - id: D3
    description: "OFTBackend's source shows the documented dual-image torch.cat packing (num_images_in_input=2 pattern); real GPU-backed multi-image inference correctness is not locally verifiable (no local torch/GPU)"
    requirement: "SPAT-02"
    verification:
      - kind: other
        ref: "grep -q 'images[\"agentview\"]' + 'torch.cat' + 'dim=1' + 'primary_inputs[\"pixel_values\"]' LIBERO/libero/libero/vla/oft_backend.py"
        status: pass
    human_judgment: true
    rationale: "GPU-backed behavior (does the checkpoint actually consume 2 images correctly) has no local GPU/torch install in this project's dev environment; deferred to a Colab manual spot-check per VALIDATION.md's Manual-Only Verifications table."

duration: 25min
completed: 2026-08-09
status: complete
---

# Phase 05 Plan 01: Multi-Camera Wiring for VLA Inference Summary

**Both SOARM camera views (wrist eye_in_hand + overhead agentview) now flow end-to-end from eval_loop.py's observation dict through Pi0Backend's distinct base/wrist openpi obs keys and OFTBackend's torch.cat dual-image pixel_values packing, replacing Phase 3's single-eye_in_hand-image baseline.**

## Performance

- **Duration:** ~25 min
- **Completed:** 2026-08-09T14:54:53Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments
- `run_episode()` now builds an images dict with both `"eye_in_hand"` and `"agentview"` keys, sourced from `obs[camera_name]` / `obs[agentview_camera_name]` respectively (new `agentview_camera_name` param, default `"agentview_image"`); `interface.py`'s `VLABackend.predict()` docstring updated to document the Phase 5 dual-key contract.
- `Pi0Backend.predict()` no longer duplicates a single image into both openpi obs keys — `base_img` is built from `images["agentview"]` (→ `observation/image`) and `wrist_img` from `images["eye_in_hand"]` (→ `observation/wrist_image`), each independently resized/converted, matching openpi's native `libero_policy.py` obs-key contract.
- `OFTBackend.predict()` now packs both views into a single multi-image forward pass: `primary_inputs` built from `eye_in_hand`, `extra_inputs` from `agentview`, concatenated via `torch.cat([...], dim=1)` on `pixel_values` before one `predict_action` call, following the checkpoint's documented `num_images_in_input=2` pattern. This behavior is source-verified locally (no local GPU/torch); real inference correctness is Colab-verification-pending per VALIDATION.md.

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire both camera views into eval_loop.py's images dict (D-01)** - `498d25d` (feat)
2. **Task 2: Pi0Backend sends real distinct base/wrist images, no duplication (D-02)** - `7037e41` (feat)
3. **Task 3: OFTBackend dual-image pixel_values packing (D-02)** - `7b494a4` (feat)

**Plan metadata:** (this commit)

## Files Created/Modified
- `LIBERO/libero/libero/vla/eval_loop.py` - `run_episode()` gains `agentview_camera_name` param; images dict now includes `"agentview"` key
- `LIBERO/libero/libero/vla/interface.py` - `VLABackend.predict()` docstring updated for the Phase 5 dual-key contract
- `LIBERO/libero/libero/vla/test_eval_loop.py` - `MockEnv` gains `agentview_image` fixture data; new `RecordingBackend` + `test_images_dict_includes_both_camera_views_spatial`
- `LIBERO/libero/libero/vla/pi0_backend.py` - `predict()` builds `base_img`/`wrist_img` independently from `agentview`/`eye_in_hand`, no shared duplicated array
- `LIBERO/libero/libero/vla/test_pi0_backend.py` - widened 6 pre-existing predict() call sites to include both required keys; new `test_predict_uses_distinct_base_and_wrist_images_spatial`
- `LIBERO/libero/libero/vla/oft_backend.py` - `predict()` builds `primary_inputs`/`extra_inputs`, concatenates `pixel_values` along `dim=1` before `predict_action`

## Decisions Made
- Kept `run_episode`'s existing `camera_name` kwarg exclusively as the video-writer camera (per plan's explicit instruction) — added a new `agentview_camera_name` param rather than repurposing it, avoiding an accidental behavior change to video output.
- Followed RESEARCH.md Pattern 1 exactly for both backends' obs-key mapping (Pi0: agentview→observation/image, eye_in_hand→observation/wrist_image; OFT: eye_in_hand as primary, agentview as extra_views[0], concat order [primary, extra] fixed).
- Left `oft_backend.py` without new local pytest coverage, per the plan's explicit instruction (VALIDATION.md's Wave 0 Gaps list only requires extending test_eval_loop.py and test_pi0_backend.py) and this project's established GPU-only-verifiable-backend convention.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. All three tasks' automated verification commands passed on first attempt:
- `conda run -n libero pytest LIBERO/libero/libero/vla/test_eval_loop.py -x -q` → 5/5 passed
- `conda run -n libero pytest LIBERO/libero/libero/vla/test_pi0_backend.py -x -q` → 7/7 passed
- Source-grep assertions on `oft_backend.py` (images["agentview"], torch.cat, dim=1, primary_inputs["pixel_values"]) → all matched
- Full suite: `conda run -n libero pytest LIBERO/libero/libero/vla -x -q` → 12/12 passed

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- SPAT-02 ("all camera views are passed as input to the VLA during inference") is satisfied as a behavioral fact for both backends' source/dict-plumbing layer.
- Remaining open item for full Phase 5 spatial-awareness scope: OFTBackend's real GPU-backed dual-image inference (does the checkpoint's `num_images_in_input=2` pattern actually consume both views correctly) must be spot-checked on Colab — this is a documented Manual-Only Verification, not a blocker for this plan's completion.
- No blockers for subsequent Phase 5 plans (3D scene understanding, spatial language grounding) — this plan only touches the camera-input plumbing layer.

---
*Phase: 05-spatial-awareness*
*Completed: 2026-08-09*
