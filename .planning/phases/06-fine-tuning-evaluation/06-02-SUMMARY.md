---
phase: 06-fine-tuning-evaluation
plan: 02
subsystem: data
tags: [oxe-registry, monkeypatch, openvla-oft, colab, finetune, lora, peft, tensorflow-datasets]

# Dependency graph
requires:
  - phase: 06-fine-tuning-evaluation (Plan 01)
    provides: "rlds_converter.py's hdf5_to_rlds(hdf5_paths, out_dir, dataset_name) -- the conversion function 06a-finetune.ipynb's RLDS-conversion-if-needed cell calls"
provides:
  - "oxe_register.py: register_soarm_spatial (pure, injectable dict mutation) + apply_soarm_spatial_registration (Colab-facing guarded wrapper, fails loud without prismatic)"
  - "06a-finetune.ipynb Part A: GPU check, repo delivery, Block A install (peft/tensorflow-datasets/accelerate, protobuf-last), restart, Block B bootstrap (EGL, config.yaml, sys.path, matplotlib/numba), RLDS-conversion-if-needed, OXE registration smoke test, HF Hub (write) + WandB auth"
affects: [06-02-finetuning-training-task4, 06-03-eval-benchmark]

# Tech tracking
tech-stack:
  added: [peft==0.20.0 (planned, not yet installed -- pending Task 3 checkpoint), tensorflow-datasets==4.9.10 (planned), accelerate (conditional, planned)]
  patterns:
    - "register_soarm_spatial takes the two OXE registry dicts AND the two encoding enum values as plain parameters (dependency injection) so it has zero import-time coupling to prismatic and is fully unit-testable with bare dicts/object() sentinels"
    - "apply_soarm_spatial_registration mirrors oft_backend.py's _ensure_prismatic fail-loud contract exactly: print full traceback via traceback.print_exc(), then raise a clear RuntimeError -- never swallow"

key-files:
  created:
    - LIBERO/libero/libero/datasets/oxe_register.py
    - LIBERO/libero/libero/datasets/test_oxe_register.py
    - LIBERO/notebooks/06a-finetune.ipynb
  modified:
    - LIBERO/libero/libero/datasets/__init__.py

key-decisions:
  - "HF Hub auth cell uses a Drive-file-then-getpass pattern (MyDrive/SoARM-Research/.hf_token if present, else getpass() prompt) per STATE.md's 01-close decision note -- no existing notebook in this repo had already implemented this exact Drive-checking variant (01-colab-env-setup.ipynb's HF cell is getpass()-only), so it was authored fresh here, explicitly requiring a WRITE-scoped token (not the prior phases' read-only usage) since this notebook pushes checkpoints"
  - "Block B's post-restart cell order (EGL -> config.yaml -> sys.path -> matplotlib Agg + numba shim) mirrors 06b-eval.ipynb's already-established Plan 06-03 pattern exactly, confirmed via direct read of that notebook's cells rather than re-deriving from Notebook 01 alone"

patterns-established:
  - "Colab-facing OXE registration wrapper pattern: pure dict-mutation core function (zero external-package coupling, unit-testable with sentinels) + a thin guarded wrapper that imports the real third-party dicts and fails loud -- reusable for any future runtime monkeypatch of a third-party registry"

requirements-completed: []

coverage:
  - id: D1
    description: "register_soarm_spatial injects the exact OXE dict shape (image_obs_keys, state_obs_keys, state_encoding, action_encoding, named_mixtures) into plain dicts, proven with object() sentinels -- no prismatic install needed"
    requirement: "TUNE-02"
    verification:
      - kind: unit
        ref: "LIBERO/libero/libero/datasets/test_oxe_register.py#test_register_soarm_spatial_injects_expected_dict_shape"
        status: pass
      - kind: unit
        ref: "LIBERO/libero/libero/datasets/test_oxe_register.py#test_register_soarm_spatial_uses_custom_dataset_name"
        status: pass
    human_judgment: false
  - id: D2
    description: "apply_soarm_spatial_registration fails loud (traceback + RuntimeError) when prismatic is genuinely absent from this project's local libero conda env -- proven live, unmocked"
    requirement: "TUNE-02"
    verification:
      - kind: unit
        ref: "LIBERO/libero/libero/datasets/test_oxe_register.py#test_apply_soarm_spatial_registration_fails_loud_without_prismatic"
        status: pass
    human_judgment: false
  - id: D3
    description: "06a-finetune.ipynb's Part A bootstrap sequence (GPU check, delivery, installs with protobuf-last, restart, Block B, RLDS-conversion-if-needed, OXE registration smoke test, HF Hub/WandB auth) is complete and self-consistent"
    requirement: "TUNE-02"
    verification:
      - kind: other
        ref: "python -c notebook JSON-validity + expected-cell-content check (hdf5_to_rlds, apply_soarm_spatial_registration, peft==0.20.0, protobuf all present)"
        status: pass
    human_judgment: true
    rationale: "Notebook cell content and ordering were verified by source-level inspection and an automated content-presence check, but the notebook has never actually been RUN on a real Colab kernel -- that requires Task 3's package-legitimacy checkpoint to clear first, then a real Colab A100 run (this plan's own <human-check> section). Not yet claimed as fully proven."
  - id: D4
    description: "SUS-flagged package installs (peft, tensorflow-datasets, accelerate) are gated behind a blocking human-verify checkpoint that never auto-approves"
    requirement: "TUNE-02"
    verification: []
    human_judgment: true
    rationale: "Task 3 is itself the blocking checkpoint (gate=\"blocking-human\") -- this plan execution STOPPED here awaiting the human's package-legitimacy confirmation per the Package Legitimacy Gate protocol. Not resolved within this execution session."

# Metrics
duration: 35min (through Task 2, execution paused at Task 3 checkpoint)
completed: 2026-08-23
status: blocked
---

# Phase 6 Plan 2: OXE Registration + Fine-Tuning Notebook Bootstrap (PARTIAL -- checkpoint pending)

**openvla-oft OXE dataset registry monkeypatch (`oxe_register.py`, dict-injection, unit-tested without prismatic) plus `06a-finetune.ipynb`'s full pre-training bootstrap (installs, RLDS conversion, registration smoke test, HF Hub write auth, WandB auth) -- execution paused at Task 3's blocking package-legitimacy checkpoint before any pip install of peft/tensorflow-datasets/accelerate runs on Colab**

## Performance

- **Duration:** ~35 min (Tasks 1-2, commit-to-commit)
- **Started:** 2026-08-23 (session start)
- **Tasks:** 2 of 4 completed; Task 3 (blocking checkpoint) reached and STOPPED; Task 4 not started
- **Files modified:** 4

## Accomplishments
- `register_soarm_spatial(oxe_dataset_configs, oxe_named_mixtures, state_encoding, action_encoding, dataset_name="soarm_spatial")` is pure dict mutation, unit-tested with `object()` sentinels and custom dataset names -- zero `prismatic` coupling
- `apply_soarm_spatial_registration(dataset_name="soarm_spatial")` is the Colab-facing wrapper that imports the real installed `prismatic` OXE registry dicts, registers against them, and returns the config -- fails loud (traceback + `RuntimeError`) when `prismatic` is absent, proven by a real unmocked local test (this project's `libero` conda env genuinely has no `prismatic` installed)
- `__init__.py` exports both functions unconditionally (zero prismatic/tensorflow import-time coupling, matching `normalization.py`/`rlds_converter.py`'s convention)
- `06a-finetune.ipynb` created with a complete Part A bootstrap: A100 GPU check (warns, doesn't raise), GitHub repo delivery, Block A install (`peft==0.20.0 tensorflow-datasets==4.9.10 --no-deps`, conditional `accelerate`, `protobuf` pinned last per Pitfall 4), restart warning, Block B post-restart bootstrap (EGL/`MUJOCO_GL`, `~/.libero/config.yaml`, `sys.path`, matplotlib Agg + numba shim), RLDS-conversion-if-needed cell (D-05), OXE registration smoke-test cell (Pitfall 1 -- prints the registered config before any `finetune.py` invocation), HF Hub write-token auth (Drive-file-then-getpass), and WandB auth
- Both automated verification commands (Task 1's pytest run, Task 2's notebook JSON-validity + content-presence check) pass

## Task Commits

Each task was committed atomically:

1. **Task 1: oxe_register.py -- injectable OXE dict registration** - `f65c607` (feat)
2. **Task 2: 06a-finetune.ipynb Part A -- Colab bootstrap** - `b315592` (feat)

**Task 3 (checkpoint:human-verify, gate="blocking-human") was reached and this execution STOPPED here per the Package Legitimacy Gate protocol. Task 4 (finetune.py torchrun invocation, checkpoint push, WandB training config) has NOT been started.**

## Files Created/Modified
- `LIBERO/libero/libero/datasets/oxe_register.py` - `register_soarm_spatial`, `apply_soarm_spatial_registration`
- `LIBERO/libero/libero/datasets/test_oxe_register.py` - 3 tests: dict-shape injection, custom dataset_name, real fail-loud-without-prismatic
- `LIBERO/libero/libero/datasets/__init__.py` - unconditional export of both new functions
- `LIBERO/notebooks/06a-finetune.ipynb` - Part A bootstrap notebook (19 cells); Part B (Task 4) not yet added

## Decisions Made
- HF Hub auth cell implements the Drive-file-then-getpass pattern described in STATE.md's 01-close decision note fresh (no existing notebook already had this exact variant committed) -- explicitly requiring a WRITE-scoped token since this notebook pushes checkpoints, unlike prior phases' read-only HF usage.
- Block B's post-restart cell order was copied verbatim from `06b-eval.ipynb` (already merged onto this branch from Plan 06-03's wave) rather than re-derived from Notebook 01 alone, since 06b-eval.ipynb already established the exact EGL -> config.yaml -> sys.path -> matplotlib/numba-shim sequence for this phase.

## Deviations from Plan

None - Tasks 1 and 2 executed exactly as written. All automated verification commands and acceptance criteria for both tasks pass.

## Issues Encountered

None new. Pre-existing Phase 4 test debt (`test_hdf5_writer.py::test_schema_and_obs_key_naming`, stale `gripper_states.shape[1] == 1` assertion) remains open and out of scope for this plan (confirmed via `git log` that this plan's commits never touched that file) -- same pre-existing failure documented in STATE.md's Blockers/Concerns and 06-01-SUMMARY.md.

## User Setup Required

**Blocking human action required before Task 4 can run.** Task 3 (`checkpoint:human-verify`, `gate="blocking-human"`) requires the user to verify `peft`, `tensorflow-datasets`, and (conditionally) `accelerate`'s PyPI metadata matches their official HuggingFace/TensorFlow upstream orgs before Task 2's Colab install cell is ever run:

1. Check `https://pypi.org/project/peft/` -- confirm Home-page/project-url references `github.com/huggingface/peft`.
2. Check `https://pypi.org/project/tensorflow-datasets/` -- confirm it references `github.com/tensorflow/datasets`.
3. If the conditional `accelerate` install fires on Colab, check `https://pypi.org/project/accelerate/` -- confirm it references `github.com/huggingface/accelerate`.
4. After installing on Colab, run `pip show peft tensorflow-datasets` (and `accelerate` if installed) and confirm Home-page/Author fields match the official orgs.

This checkpoint is `gate="blocking-human"` and is NEVER auto-approved regardless of `workflow.auto_advance` — the orchestrator/user must explicitly resolve it (type "approved" or "blocked" with observed metadata) before Task 4 executes.

## Next Phase Readiness

- Tasks 1-2 are complete, committed, and independently verified (local pytest green, notebook JSON-valid with all expected content present).
- **This plan is NOT complete.** Task 3's blocking checkpoint must be resolved by a human before Task 4 (the actual `finetune.py` torchrun invocation, resumable checkpoint push, and WandB training config cells) can be added to `06a-finetune.ipynb`.
- Once Task 3 resolves, a continuation execution must: (a) add Task 4's cells to `06a-finetune.ipynb`, (b) run this plan's full local verification suite again, (c) complete the `<human-check>` Colab A100 training run (finetune.py completion, checkpoint push + privacy assertion, resume-from-disconnect, WandB training curves), and (d) replace this partial SUMMARY.md with a complete one (`status: complete`, `requirements-completed: [TUNE-02, TUNE-04]`) once all of the above passes.
- No blockers for Plan 06-01's dependency (rlds_converter.py) -- confirmed working, imported cleanly in the RLDS conversion cell.

---
*Phase: 06-fine-tuning-evaluation*
*Completed: NOT YET -- paused at Task 3 checkpoint, 2026-08-23*
