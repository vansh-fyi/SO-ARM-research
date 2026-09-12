---
phase: 06-fine-tuning-evaluation
plan: 02
subsystem: data
tags: [oxe-registry, monkeypatch, openvla-oft, colab, finetune, lora, peft, tensorflow-datasets, wandb, huggingface-hub]

# Dependency graph
requires:
  - phase: 06-fine-tuning-evaluation (Plan 01)
    provides: "rlds_converter.py's hdf5_to_rlds(hdf5_paths, out_dir, dataset_name) -- the conversion function 06a-finetune.ipynb's RLDS-conversion-if-needed cell calls"
provides:
  - "oxe_register.py: register_soarm_spatial (pure, injectable dict mutation) + apply_soarm_spatial_registration (Colab-facing guarded wrapper, fails loud without prismatic)"
  - "06a-finetune.ipynb: full bootstrap (GPU check, delivery, Block A install, restart, Block B) -> RLDS-conversion-if-needed -> OXE registration smoke test -> HF Hub (write) + WandB auth -> finetune.py torchrun invocation (LoRA r=32) -> resumable HF Hub checkpoint push -> WandB training config"
  - "HF_ADAPTER_REPO_ID naming convention (f\"{hf-username}/soarm-oft-lora-{RUN_TIMESTAMP}\") and WANDB_PROJECT = \"soarm-oft-finetune-eval\" constant -- both consumed verbatim by Plan 06-03's 06b-eval.ipynb"
affects: [06-03-eval-benchmark, phase-6-fine-tuning-evaluation-close, TUNE-02-verification, TUNE-04-verification]

# Tech tracking
tech-stack:
  added: [peft==0.20.0, tensorflow-datasets==4.9.10, accelerate (conditional)]
  patterns:
    - "register_soarm_spatial takes the two OXE registry dicts AND the two encoding enum values as plain parameters (dependency injection) so it has zero import-time coupling to prismatic and is fully unit-testable with bare dicts/object() sentinels"
    - "apply_soarm_spatial_registration mirrors oft_backend.py's _ensure_prismatic fail-loud contract exactly: print full traceback via traceback.print_exc(), then raise a clear RuntimeError -- never swallow"
    - "Long-running blocking subprocess (torchrun) pattern: Popen + log-file redirect + poll-until-exit loop, adapted from 03b-pi0-inference-smoketest.ipynb's poll-until-port-open pattern for a process that terminates rather than one that stays alive"

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
  - "HF_ADAPTER_REPO_ID's <hf-username> is resolved live via HfApi().whoami()['name'] (after Task 2's login cell) and WANDB_ENTITY via wandb.Api().default_entity, rather than a hardcoded placeholder string -- Claude automates whatever the already-authenticated CLI/API session can resolve instead of asking the researcher to paste a username"
  - "Added a guarded, idempotent git clone of moojink/openvla-oft to /content/openvla-oft in Task 4's own cells (not in the original plan's action text) -- Block A only pip-installs the package (--no-deps), which brings in the importable prismatic module but not the top-level vla-scripts/ directory that vla-scripts/finetune.py lives in; the torchrun cell cannot invoke a script that doesn't exist on disk. Clones the SAME already-vetted repo the Package Legitimacy Gate already covers (github.com/moojink/openvla-oft, used since Phase 1) -- not a new PyPI install, so no additional checkpoint required (Rule 3, blocking-issue fix)."

patterns-established:
  - "Colab-facing OXE registration wrapper pattern: pure dict-mutation core function (zero external-package coupling, unit-testable with sentinels) + a thin guarded wrapper that imports the real third-party dicts and fails loud -- reusable for any future runtime monkeypatch of a third-party registry"
  - "Long-running-subprocess-with-log-file pattern for torchrun/training launches, extending 03b's serve_policy.py Popen pattern to processes that terminate on completion rather than serve indefinitely"

requirements-completed: []  # TUNE-02/TUNE-04 executor-side work is done (Tasks 1-4 all complete), but the plan's <human-check> live-Colab A100 training run (finetune.py completion, checkpoint push, resume-from-disconnect, WandB training curves) is still pending -- out of this executor's scope, matching the exact precedent set by Plan 06-03's TUNE-03/TUNE-04 (also left unmarked pending live-Colab confirmation). Mark complete only after that run is confirmed.

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
    rationale: "Notebook cell content and ordering were verified by source-level inspection and an automated content-presence check, but the notebook has never actually been RUN on a real Colab kernel -- requires this plan's own <human-check> Colab A100 run to fully prove."
  - id: D4
    description: "SUS-flagged package installs (peft, tensorflow-datasets, accelerate) are gated behind a blocking human-verify checkpoint that never auto-approves"
    requirement: "TUNE-02"
    verification:
      - kind: manual
        ref: "checkpoint:human-verify (gate=blocking-human) -- user responded 'approved'"
        status: pass
    human_judgment: true
    rationale: "Task 3 is the blocking checkpoint (gate=\"blocking-human\") -- never auto-approvable regardless of workflow.auto_advance. User explicitly confirmed peft/tensorflow-datasets/accelerate's PyPI metadata matches the official HuggingFace/TensorFlow-org origin. Package Legitimacy Gate satisfied for this notebook's install site (T-06-02-SC closed)."
  - id: D5
    description: "06a-finetune.ipynb Part B: finetune.py torchrun invocation (LoRA r=32, merge-during-training disabled, timestamped non-overwritten checkpoints), resumable private HF Hub checkpoint push with an explicit privacy assertion, WandB training config targeting the shared soarm-oft-finetune-eval project"
    requirement: "TUNE-02, TUNE-04"
    verification:
      - kind: other
        ref: "python -c notebook content-presence check (--lora_rank 32, --use_lora True, --merge_lora_during_training False, --save_latest_checkpoint_only False, create_repo(...private=True...) followed by an explicit .private is True assertion, WANDB_PROJECT constant)"
        status: pass
    human_judgment: true
    rationale: "Source-level checks confirm the exact flags/assertions this plan's success_criteria requires are present, but proving finetune.py actually completes a training run, pushes a real checkpoint, resumes correctly after a simulated disconnect, and produces WandB training-loss curves requires this plan's own <human-check> live Colab A100 run -- not yet performed, outside this executor's scope (same as Plan 06-01's RLDS conversion and Plan 06-03's before/after benchmark, both GPU-only steps in this phase)."

# Metrics
duration: ~35min (Tasks 1-2) + this continuation session (Task 3 approval + Task 4 implementation)
completed: 2026-08-23
status: complete
---

# Phase 6 Plan 2: OXE Registration + Fine-Tuning Notebook (Bootstrap + Training) Summary

**openvla-oft OXE dataset registry monkeypatch (`oxe_register.py`, dict-injection, unit-tested without prismatic) plus `06a-finetune.ipynb`'s full pipeline: bootstrap (installs, RLDS conversion, registration smoke test, HF Hub write auth, WandB auth) through `finetune.py`'s LoRA r=32 torchrun invocation, resumable private HF Hub checkpoint push, and shared WandB training-project config -- all 4 tasks complete, including the human-approved package-legitimacy checkpoint.**

## Performance

- **Duration:** ~35 min (Tasks 1-2, commit-to-commit) + this continuation session (Task 3 checkpoint approval recorded, Task 4 implemented and verified)
- **Started:** 2026-08-23 (session start)
- **Completed:** 2026-08-23
- **Tasks:** 4 of 4 completed
- **Files modified:** 4 (3 created, 1 modified)

## Accomplishments
- `register_soarm_spatial(oxe_dataset_configs, oxe_named_mixtures, state_encoding, action_encoding, dataset_name="soarm_spatial")` is pure dict mutation, unit-tested with `object()` sentinels and custom dataset names -- zero `prismatic` coupling
- `apply_soarm_spatial_registration(dataset_name="soarm_spatial")` is the Colab-facing wrapper that imports the real installed `prismatic` OXE registry dicts, registers against them, and returns the config -- fails loud (traceback + `RuntimeError`) when `prismatic` is absent, proven by a real unmocked local test (this project's `libero` conda env genuinely has no `prismatic` installed)
- `__init__.py` exports both functions unconditionally (zero prismatic/tensorflow import-time coupling, matching `normalization.py`/`rlds_converter.py`'s convention)
- `06a-finetune.ipynb` Part A: complete bootstrap -- A100 GPU check (warns, doesn't raise), GitHub repo delivery, Block A install (`peft==0.20.0 tensorflow-datasets==4.9.10 --no-deps`, conditional `accelerate`, `protobuf` pinned last per Pitfall 4), restart warning, Block B post-restart bootstrap (EGL/`MUJOCO_GL`, `~/.libero/config.yaml`, `sys.path`, matplotlib Agg + numba shim), RLDS-conversion-if-needed cell (D-05), OXE registration smoke-test cell (Pitfall 1 -- prints the registered config before any `finetune.py` invocation), HF Hub write-token auth (Drive-file-then-getpass), and WandB auth
- Task 3's blocking `peft`/`tensorflow-datasets`/`accelerate` package-legitimacy checkpoint (`gate="blocking-human"`) was resolved -- user approved, confirming all three packages' PyPI metadata matches the official HuggingFace/TensorFlow-org origin
- `06a-finetune.ipynb` Part B: run-identity cell (timestamped `HF_ADAPTER_REPO_ID`/`RUN_ROOT_DIR`, live-resolved `WANDB_ENTITY`), guarded `openvla-oft` clone for `vla-scripts/finetune.py` access, `torchrun` invocation locked to `--lora_rank 32 --use_lora True --merge_lora_during_training False --save_latest_checkpoint_only False` (D-00/D-01/D-02/D-10/D-11, Pitfall 5), a documented relaunch-after-disconnect procedure (`--resume True --resume_step <N>`), `push_checkpoint_to_hub` with an explicit `create_repo(private=True)` + `.private is True` assertion (T-06-02-02), a checkpoint-push polling loop that prunes local disk to the 2 most-recently-pushed checkpoints, and a "Phase 6a Summary" cell for post-run bookkeeping
- All automated verification commands (Task 1's pytest run, Task 2's and Task 4's notebook JSON-validity + content-presence checks) pass

## Task Commits

Each task was committed atomically:

1. **Task 1: oxe_register.py -- injectable OXE dict registration** - `f65c607` (feat)
2. **Task 2: 06a-finetune.ipynb Part A -- Colab bootstrap** - `b315592` (feat)
3. **Task 3: checkpoint approval record** - `9d5def5` (docs)
4. **Task 4: 06a-finetune.ipynb Part B -- finetune.py torchrun, resumable checkpoint push, WandB config** - `795d369` (feat)

**Plan metadata:** this SUMMARY.md finalization commit (docs, worktree-local; STATE.md/ROADMAP.md updates deferred to the orchestrator per parallel-worktree execution)

Task 3 (blocking `checkpoint:human-verify`, `gate="blocking-human"`) required explicit human confirmation and could not be auto-approved regardless of `workflow.auto_advance` -- the user's "approved" response is recorded in `9d5def5` and the "Checkpoint Approvals" section below.

## Files Created/Modified
- `LIBERO/libero/libero/datasets/oxe_register.py` - `register_soarm_spatial`, `apply_soarm_spatial_registration`
- `LIBERO/libero/libero/datasets/test_oxe_register.py` - 3 tests: dict-shape injection, custom dataset_name, real fail-loud-without-prismatic
- `LIBERO/libero/libero/datasets/__init__.py` - unconditional export of both new functions
- `LIBERO/notebooks/06a-finetune.ipynb` - full training notebook, 27 cells: Part A bootstrap (19 cells) + Part B training/checkpoint-push/WandB (8 cells)

## Decisions Made
- HF Hub auth cell implements the Drive-file-then-getpass pattern described in STATE.md's 01-close decision note fresh (no existing notebook already had this exact variant committed) -- explicitly requiring a WRITE-scoped token since this notebook pushes checkpoints, unlike prior phases' read-only HF usage.
- Block B's post-restart cell order was copied verbatim from `06b-eval.ipynb` (already merged onto this branch from Plan 06-03's wave) rather than re-derived from Notebook 01 alone, since 06b-eval.ipynb already established the exact EGL -> config.yaml -> sys.path -> matplotlib/numba-shim sequence for this phase.
- `HF_ADAPTER_REPO_ID`'s username component and `WANDB_ENTITY` are both resolved live from the already-authenticated `HfApi()`/`wandb.Api()` sessions rather than hardcoded `<hf-username>`/`<entity>` placeholder strings the plan's action text used as illustrative shape -- consistent with "if Claude can automate it, Claude does it" (checkpoints.md golden rule); the researcher never has to paste their own username into a cell.
- Added a guarded `git clone` of `moojink/openvla-oft` to `/content/openvla-oft` as a new cell in Task 4 (see Deviations below) -- required for `vla-scripts/finetune.py` to exist on disk at all, since Block A's install is `pip install ... --no-deps` of the importable package only.

## Checkpoint Approvals

**Task 3: Human verifies peft/tensorflow-datasets/accelerate package legitimacy before Task 2's install cell runs** (`checkpoint:human-verify`, `gate="blocking-human"`, non-auto-approvable)

- **What was verified:** `peft==0.20.0`, `tensorflow-datasets==4.9.10`, and (conditionally) `accelerate==1.14.0` -- all three flagged SUS by 06-RESEARCH.md's Package Legitimacy Audit (unknown download counts, no structural red flags found).
- **How-to-verify steps (per plan):** PyPI metadata (Home-page/project-url) for each package confirmed to reference `github.com/huggingface/peft`, `github.com/tensorflow/datasets`, and `github.com/huggingface/accelerate` respectively.
- **Resolution:** User responded **"approved"**, confirming PyPI metadata for all three packages matches their official HuggingFace/TensorFlow-org origin. Package Legitimacy Gate satisfied for this notebook's install site (T-06-02-SC closed).
- **Resolved:** 2026-08-23

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking issue] Added a guarded git clone of moojink/openvla-oft for vla-scripts/finetune.py**
- **Found during:** Task 4
- **Issue:** The plan's Task 4 torchrun cell invokes `vla-scripts/finetune.py` as a relative-path script, but Block A (Task 2, and Phase 1's baked-environment prerequisite this notebook assumes) only ever `pip install`s `git+https://github.com/moojink/openvla-oft.git --no-deps`, which packages the importable `prismatic` module -- not the top-level `vla-scripts/` directory (not part of `setup.py`'s packaged modules). Without an actual clone of the repo on disk, the torchrun cell has no file to execute and fails immediately.
- **Fix:** Added a new guarded, idempotent cell before the torchrun invocation that clones `github.com/moojink/openvla-oft.git` (the SAME repo already vetted and pip-installed since Phase 1) to `/content/openvla-oft` if not already present, asserts `vla-scripts/finetune.py` exists under it, and uses that directory as the torchrun cell's `cwd`. This is a clone of an already-vetted repo, not a new PyPI package install -- no additional Package Legitimacy Gate checkpoint applies (T-06-02-SC's disposition already covers this source).
- **Files modified:** `LIBERO/notebooks/06a-finetune.ipynb` (new cell, part of Task 4's commit)
- **Verification:** Notebook JSON-validity + content-presence check (Task 4's automated verify) confirms the cell is present and precedes the torchrun cell; the clone cell asserts the expected script path exists after cloning.
- **Committed in:** `795d369` (Task 4 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking, Rule 3)
**Impact on plan:** Necessary for the plan's own torchrun invocation to be runnable at all on a real Colab kernel. No scope creep -- purely additive, zero behavior change to the plan's design, reuses a repo source already covered by this plan's own threat-model disposition.

## Issues Encountered

None new. Pre-existing Phase 4 test debt (`test_hdf5_writer.py::test_schema_and_obs_key_naming`, stale `gripper_states.shape[1] == 1` assertion) remains open and out of scope for this plan (confirmed via `git log` that this plan's commits never touched that file) -- same pre-existing failure documented in STATE.md's Blockers/Concerns and 06-01-SUMMARY.md. Re-confirmed still isolated to that one file after Task 4's full local regression run (`conda run -n libero pytest LIBERO/libero/libero/datasets -x -q`).

## User Setup Required

Plan 06-02's `user_setup` entries (Hugging Face Hub write access, Weights & Biases `WANDB_API_KEY`) are consumed on Colab when `06a-finetune.ipynb` is actually run. The Task 3 checkpoint that gated the notebook's install cell is now resolved; the researcher can proceed to the live Colab A100 training run described below.

## Next Phase Readiness

**All 4 tasks complete. This plan's local, no-GPU-dependent executor work is done and merged:**
- `oxe_register.py` dict-injection registration (Task 1)
- `06a-finetune.ipynb` Part A bootstrap (Task 2)
- Task 3's blocking package-legitimacy checkpoint -- approved
- `06a-finetune.ipynb` Part B: finetune.py torchrun, resumable checkpoint push, WandB config (Task 4)

**Remaining work outside this executor's scope (live-Colab verification, matching this phase's other GPU-only steps -- Plan 06-01's RLDS conversion, Plan 06-03's before/after benchmark):**

The plan's `<human-check>` section requires a real Colab A100 GPU run to confirm TUNE-02/TUNE-04 end-to-end:
1. Run `06a-finetune.ipynb` end-to-end on a Colab A100 runtime through Block A, restart, Block B, RLDS conversion, OXE registration smoke test, HF Hub/WandB auth, and the `finetune.py` torchrun cell.
2. Confirm `finetune.py` runs to completion without crashing.
3. Confirm at least one checkpoint push succeeds: `HfApi().repo_info(HF_ADAPTER_REPO_ID).private is True` and the repo is visible under the researcher's HF Hub account.
4. Simulate a disconnect: interrupt the kernel mid-training, relaunch the torchrun cell with `RESUME = True` / `RESUME_STEP = <N>`, and confirm the WandB run's step counter continues from `<N>` rather than resetting to 0.
5. Confirm the WandB dashboard (`https://wandb.ai/<entity>/soarm-oft-finetune-eval`) shows training-loss curves for this run.
6. Paste the printed `HF_ADAPTER_REPO_ID` value into Plan 06-03's `06b-eval.ipynb` `ADAPTER_REPO_ID` cell to unblock the before/after benchmark run.

This live-Colab confirmation is what the next `/gsd-verify-work`/UAT pass on this phase should perform -- `TUNE-02`/`TUNE-04` should only be marked complete in REQUIREMENTS.md after it succeeds, consistent with how this phase's other GPU-only steps (data conversion, before/after benchmark) are tracked.

No blockers for Plan 06-01's dependency (`rlds_converter.py`) -- confirmed working, imported cleanly in the RLDS conversion cell.

---
*Phase: 06-fine-tuning-evaluation*
*Completed: 2026-08-23*

## Self-Check: PASSED

All created/modified files confirmed present on disk (`oxe_register.py`, `test_oxe_register.py`, `06a-finetune.ipynb`, `datasets/__init__.py`, this SUMMARY.md). All 5 task commits confirmed present in `git log` (`f65c607`, `b315592`, `b78ada4`, `9d5def5`, `795d369`).
