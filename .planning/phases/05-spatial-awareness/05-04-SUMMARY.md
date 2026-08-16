---
phase: 05-spatial-awareness
plan: 04
subsystem: vla
tags: [openvla-oft, huggingface-transformers, multi-camera, tdd, pytest]

# Dependency graph
requires:
  - phase: 05-01
    provides: OFTBackend's original multi-camera torch.cat(dim=1) packing pattern (D-02), which this plan corrects
provides:
  - "OFTBackend.__init__ activates the checkpoint's documented dual-image mode via set_num_images_in_input(2)"
  - "OFTBackend.predict() sources primary from agentview (third-person) and extra_views from eye_in_hand (wrist), matching upstream's trained convention"
  - "test_oft_backend.py: local mock-based regression suite (2 tests) proving both fixes without GPU/torch"
affects: [05-spatial-awareness UAT closure, Phase 6 fine-tuning (inherits the corrected image-order contract)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "sys.modules fake-injection pattern for GPU-only deps (torch/huggingface_hub/transformers), mirroring test_pi0_backend.py's fake_openpi_client fixture -- now applied to a second VLA backend"

key-files:
  created:
    - LIBERO/libero/libero/vla/test_oft_backend.py
  modified:
    - LIBERO/libero/libero/vla/oft_backend.py

key-decisions:
  - "Used fix(05-04) commit type for the GREEN commit (not feat) since this is a root-cause bug fix per 05-UAT.md's diagnosis, not a new feature -- see TDD Gate Compliance note below"

patterns-established:
  - "fake_oft_deps fixture: injects fake torch/huggingface_hub/transformers modules into sys.modules before importing oft_backend, letting a real self.model mock object be asserted against post-construction (model.to(device) returns itself)"

requirements-completed: [SPAT-02]

coverage:
  - id: D1
    description: "OFTBackend.__init__ calls set_num_images_in_input(2) exactly once after loading the model, closing the live Colab split_with_sizes=[3,3] RuntimeError"
    requirement: "SPAT-02"
    verification:
      - kind: unit
        ref: "LIBERO/libero/libero/vla/test_oft_backend.py#test_init_activates_num_images_in_input_two"
        status: pass
    human_judgment: false
  - id: D2
    description: "OFTBackend.predict() sources primary from agentview (third-person) and extra_views from eye_in_hand (wrist), matching upstream's trained channel-order convention"
    requirement: "SPAT-02"
    verification:
      - kind: unit
        ref: "LIBERO/libero/libero/vla/test_oft_backend.py#test_predict_orders_agentview_as_primary_eye_in_hand_as_extra"
        status: pass
    human_judgment: false
  - id: D3
    description: "Real GPU-backed checkpoint accepts the dual-image call with no RuntimeError and produces a non-degenerate (8,7) action chunk that differs from a single-view baseline"
    verification: []
    human_judgment: true
    rationale: "Colab-only per this project's no-local-GPU convention (03-RESEARCH.md) -- closed by the plan's <human-check> re-verification step (LIBERO/notebooks/03a-oft-inference-eval.ipynb VLA-01 dual-camera spot-check cell), not by an in-plan automated task."

# Metrics
duration: 20min
completed: 2026-08-16
status: complete
---

# Phase 5 Plan 04: OFTBackend Dual-Camera Gap Closure Summary

**Fixed OFTBackend's missing `set_num_images_in_input(2)` activation and inverted `agentview`/`eye_in_hand` image order that crashed dual-camera inference on Colab GPU, closing 05-UAT.md's sole remaining gap with a new 2-test local mock-based regression suite (14/14 vla suite green).**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-08-16T16:05:00Z (approx)
- **Completed:** 2026-08-16T16:25:17Z
- **Tasks:** 1
- **Files modified:** 2 (1 created, 1 modified)

## Accomplishments
- `OFTBackend.__init__` now calls `self.model.vision_backbone.set_num_images_in_input(2)` immediately after the model is loaded and moved to device, activating the checkpoint's dual-image mode instead of leaving it in its default single-image state that unconditionally does `torch.split(pixel_values, [3, 3], dim=1)` and crashes on the 12-channel concatenated tensor.
- `OFTBackend.predict()`'s image assignment corrected: `primary = Image.fromarray(images["agentview"])` (third-person), `extra_views = [Image.fromarray(images["eye_in_hand"])]` (wrist) -- matching upstream `moojink/openvla-oft`'s `get_vla_action` convention (previously inverted).
- New `test_oft_backend.py` (2 tests) proves both fixes locally without GPU/torch, using a `sys.modules` fake-injection fixture (`fake_oft_deps`) mirroring `test_pi0_backend.py`'s established pattern for `torch`, `huggingface_hub`, and `transformers`.
- Both new tests confirmed to FAIL against the pre-fix source (RED) before any source change was made, proving they exercise the real bugs described in 05-UAT.md's diagnosis, not tautologies.
- Full `LIBERO/libero/libero/vla` suite (14 tests: 5 `test_eval_loop`, 7 `test_pi0_backend`, 2 `test_oft_backend`) passes locally after the fix.

## Task Commits

Each task was committed atomically (TDD RED/GREEN):

1. **Task 1 RED: add failing regression tests** - `f5b1e29` (test)
2. **Task 1 GREEN: fix OFTBackend source** - `955dc79` (fix)

**Plan metadata:** committed separately per worktree convention (SUMMARY.md only; STATE.md/ROADMAP.md owned by the orchestrator)

## Files Created/Modified
- `LIBERO/libero/libero/vla/test_oft_backend.py` - New mock-based regression suite: `fake_oft_deps` fixture (fake torch/huggingface_hub/transformers injected into sys.modules), `test_init_activates_num_images_in_input_two`, `test_predict_orders_agentview_as_primary_eye_in_hand_as_extra`
- `LIBERO/libero/libero/vla/oft_backend.py` - Added `set_num_images_in_input(2)` call in `__init__`; swapped `primary`/`extra_views` assignment in `predict()`; updated docstring/comments to describe the corrected agentview-primary/eye_in_hand-secondary convention

## Decisions Made
- Followed the plan's exact TDD RED/GREEN sequence: wrote both tests first, ran them against the unmodified source to confirm both failed for the documented reasons (missing `set_num_images_in_input` call never invoked; processor call order was eye_in_hand-then-agentview instead of agentview-then-eye_in_hand), then applied the minimal two-line source fix.
- Used `fix(05-04): ...` as the GREEN commit type (see TDD Gate Compliance below) rather than `feat(...)`, because this task's `<action>` block explicitly frames it as fixing two diagnosed bugs from 05-UAT.md, not adding new behavior -- consistent with the task-commit-protocol's type table (`fix` = "bug fix, error correction").

## TDD Gate Compliance

RED gate: `test(05-04): add failing regression tests for OFTBackend dual-camera gap fix` (`f5b1e29`) -- confirmed both new tests failed against the pre-fix source before this commit.

GREEN gate: `fix(05-04): activate dual-image mode and fix image order in OFTBackend` (`955dc79`) -- confirmed both tests pass, and the full 14-test vla suite is green, after this commit.

Note: the GREEN commit uses `fix(...)` rather than the `feat(...)` prefix a literal RED/feat/refactor gate-sequence scan might expect. This is intentional and matches the task-commit-protocol's type table -- the task is explicitly a root-cause bug fix (05-UAT.md's diagnosed regression), not new feature work. Both RED and GREEN commits exist in the correct order (`f5b1e29` before `955dc79`); no REFACTOR commit was needed (no cleanup required beyond the two targeted source edits).

## Deviations from Plan

None - plan executed exactly as written. The fixture, test bodies, and source edits match the plan's `<action>` block specification precisely (fixture structure, fake module surface, both test assertions, the two source-level edits, and the unchanged `assert actions.shape == (8, 7)` contract).

## Issues Encountered

None. Confirmed the `libero` conda env genuinely lacks `torch` (`ModuleNotFoundError: No module named 'torch'`) before writing the fixture, matching the plan's stated assumption; confirmed `PIL`/`numpy` are available locally (real, not faked, per the plan's read_first notes).

## User Setup Required

None - no external service configuration required.

## Human Verification Required

Per this project's `human_verify_mode: end-of-phase` config and the plan's `<verification>` block `<human-check>` step, the following is deferred to the next `/gsd-verify-work` or UAT pass on Phase 5 (not a blocking mid-plan checkpoint):

1. Pull the fixed `oft_backend.py` onto a Colab GPU runtime (A100 recommended).
2. Open `LIBERO/notebooks/03a-oft-inference-eval.ipynb`, run Block A + restart + Block B through the "VLA-01: OpenVLA-OFT Model Load + Action Shape Check" section.
3. Confirm the dual-camera spot-check cell (`actions_dual` vs `actions_degenerate`) no longer raises `RuntimeError: split_with_sizes expects split_sizes to sum exactly to 12 ...`.
4. Confirm the cell prints `VLA-01 (dual-camera spot-check): PASS -- shapes (8, 7), mean diff <N>` with nonzero `<N>`.

This is the real-checkpoint confirmation that source-level fixes and local mock-based tests cannot provide (no local GPU/torch in this project's dev environment) -- it closes 05-UAT.md's sole remaining blocker.

## Next Phase Readiness
- 05-UAT.md's diagnosed blocker gap is source-fixed and locally regression-tested (14/14 vla suite green).
- SPAT-02 (OFTBackend half) is now behaviorally correct at the source level; Pi0Backend half was already verified passing in 05-VERIFICATION.md.
- Remaining work before Phase 5 can be marked fully complete: the Colab GPU re-verification step above, via the next UAT/verify-work pass -- the same loop that originally surfaced this gap.

---
*Phase: 05-spatial-awareness*
*Completed: 2026-08-16*

## Self-Check: PASSED

- FOUND: LIBERO/libero/libero/vla/test_oft_backend.py
- FOUND: LIBERO/libero/libero/vla/oft_backend.py
- FOUND: .planning/phases/05-spatial-awareness/05-04-SUMMARY.md
- FOUND commit: f5b1e29 (test)
- FOUND commit: 955dc79 (fix)
- FOUND commit: 96441cc (docs)
