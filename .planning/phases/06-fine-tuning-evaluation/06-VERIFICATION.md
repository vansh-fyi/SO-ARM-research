---
phase: 06-fine-tuning-evaluation
verified: 2026-08-23T08:00:11Z
status: human_needed
score: 10/10 must-haves verified (locally verifiable scope)
behavior_unverified: 0
overrides_applied: 0
gaps: []
human_verification:
  - test: "Run `06a-finetune.ipynb` end-to-end on a Colab A100 runtime: Block A install, restart, Block B bootstrap, RLDS-conversion-if-needed cell, OXE registration smoke-test cell, HF Hub (write) + WandB auth, then the `finetune.py` torchrun cell (`--lora_rank 32 --use_lora True --merge_lora_during_training False --save_latest_checkpoint_only False`)."
    expected: "`hdf5_to_rlds` produces a genuine `tfds.builder`-loadable RLDS dataset (round-trip `.info` call does not raise); `apply_soarm_spatial_registration` prints a registered `soarm_spatial` config with no traceback; `finetune.py` runs to completion without crashing and prints a final success log line."
    why_human: "Requires a real Colab A100 GPU + the full openvla-oft/prismatic/tensorflow_datasets stack, none of which are installed in this project's local libero conda env (confirmed: `tensorflow_datasets`/`prismatic` absent locally, per 06-VALIDATION.md's Manual-Only Verifications table). `hdf5_to_rlds`'s `tfds.core.SequentialWriter` call shape is explicitly flagged MEDIUM confidence in 06-RESEARCH.md and 06-01-SUMMARY.md since it could not be checked against the real `tensorflow_datasets==4.9.10` source locally."
  - test: "Push at least one LoRA checkpoint to the private HF Hub repo during/after the training run; simulate a disconnect (interrupt the kernel mid-training) and relaunch the torchrun cell with `RESUME=True`/`RESUME_STEP=<N>`."
    expected: "`HfApi().repo_info(HF_ADAPTER_REPO_ID).private is True` and the repo is visible under the researcher's HF Hub account; after relaunch, the WandB run's step counter continues from `<N>` rather than resetting to 0."
    why_human: "Resumability and checkpoint-push correctness are runtime behaviors of a long-lived `torchrun` subprocess against a real GPU and a real HF Hub write token — cannot be exercised without a live Colab session. Source-level review (Task 4's `create_repo(..., private=True)` + `assert ... .private is True` + `--resume`/`--resume_step` flag plumbing) confirms the code path exists and is wired correctly, but not that it behaves correctly under a real disconnect."
  - test: "Confirm the WandB dashboard (`https://wandb.ai/<entity>/soarm-oft-finetune-eval`) shows training-loss curves from the `06a-finetune.ipynb` run."
    expected: "Training-loss curves are visible in the shared `soarm-oft-finetune-eval` WandB project (TUNE-04)."
    why_human: "`finetune.py`'s native WandB hooks only produce a real logged run when the training loop actually executes on GPU; nothing to observe locally."
  - test: "Paste the trained run's `HF_ADAPTER_REPO_ID` into `06b-eval.ipynb`'s `ADAPTER_REPO_ID` cell and run the notebook end-to-end on a Colab A100/T4 runtime through Block A, restart, Block B, task/language setup, WandB init, the BEFORE cell (zero-shot `OFTBackend`), the Pitfall-6 divergence-check cell, and the AFTER cell (`FinetunedOFTBackend`)."
    expected: "Both the before (zero-shot) and after (fine-tuned) `run_suite` benchmarks complete under the identical `EPISODE_SEEDS = list(range(20))` protocol across the same 4 tasks (3 spatial + 1 non-spatial); `FinetunedOFTBackend` successfully downloads and merges the adapter without an `AttributeError` on `set_num_images_in_input` (Assumption A4); the divergence-check cell prints either a clean pass or a clearly-flagged WARNING (not a silent skip)."
    why_human: "Requires the real trained adapter (only produced by the first human-check item above) plus a live GPU to run both benchmarks — `peft.PeftModel.merge_and_unload()`'s effect on the model's module tree (Assumption A4) is explicitly unverifiable without a real checkpoint load."
  - test: "Confirm the before/after `wandb.Table` (per-task success rates) and aggregate scalars appear in the same `soarm-oft-finetune-eval` WandB project as the training-loss curves."
    expected: "Both training curves (from 06a) and eval success-rate results (from 06b) are visible together in one WandB dashboard project (TUNE-04, the exact check 06-VALIDATION.md's Manual-Only Verifications table flags)."
    why_human: "Requires two real, completed live runs (training + eval) logging to the same WandB project — nothing to observe without them."
---

# Phase 6: Fine-Tuning & Evaluation Verification Report

**Phase Goal:** OpenVLA-OFT is fine-tuned on SOARM demonstrations via LoRA on Colab A100 and evaluated against a spatial benchmark, with before/after results tracked in WandB.
**Verified:** 2026-08-23T08:00:11Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

This phase's goal has two disjoint verification surfaces, both by explicit project design (see 06-VALIDATION.md's Manual-Only Verifications table, and both 06-02-SUMMARY.md/06-03-SUMMARY.md's own "Next Phase Readiness" sections, which state TUNE-02/03/04 should stay unmarked in REQUIREMENTS.md until a live Colab run confirms them):

1. **Locally verifiable:** the code, tests, and notebook wiring that make a live GPU run *possible and correct* — fully checked below via direct code reading and running the actual local test suites (not trusted from SUMMARY.md claims).
2. **Live-Colab-only:** the actual LoRA training run, checkpoint push/resume, and before/after benchmark execution with WandB logging — cannot be executed or observed by this (or any local) agent. Routed to Human Verification below per this task's explicit instruction, not marked FAILED.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | The robomimic HDF5 dataset converts to RLDS format (code + local tests) | ✓ VERIFIED | `rlds_converter.py`'s `hdf5_to_rlds`/`load_episodes_from_hdf5`/`validate_episode_arrays` all present, match plan spec exactly; `conda run -n libero pytest LIBERO/libero/libero/datasets/test_rlds_converter.py -q` → **6 passed, 2 skipped** (the 2 skips are the `pytest.importorskip("tensorflow_datasets")`-guarded TFDS-write tests, skipped locally because `tensorflow_datasets` is genuinely not installed in this project's local libero conda env — confirmed, not a gap) |
| 2 | Malformed/mismatched episode arrays rejected with `ValueError` before any bytes are written (D-06, ASVS V5) | ✓ VERIFIED | `validate_episode_arrays` in `rlds_converter.py` (lines 35-96) checks dtype/ndim/last-dim/length-mismatch/NaN-Inf for all 4 arrays before any write; `hdf5_to_rlds` (lines 202-211) validates ALL episodes in a loop before any TF import/write begins; 5 dedicated unit tests pass locally |
| 3 | `rlds_converter.py` imports cleanly with zero tensorflow dependency at module top level | ✓ VERIFIED | `grep -c "^import tensorflow" rlds_converter.py` → 0; `tensorflow`/`tensorflow_datasets`/`h5py`/`bddl_utils` imports all confined inside function bodies (lines 110, 114, 215-216); local `libero` conda env import succeeds (verified via test collection) |
| 4 | OpenVLA-OFT LoRA fine-tuning (r=32) runs to completion on Colab A100 without crashing | ⚠️ PRESENT, LIVE RUN PENDING | Code fully present and wired: `torchrun ... --lora_rank 32 --use_lora True --merge_lora_during_training False --save_latest_checkpoint_only False` (06a cell 22); `oxe_register.apply_soarm_spatial_registration()` runs and is smoke-tested (06a cell 16) BEFORE the torchrun cell — confirmed via source-position check (`idx_reg=10671 < idx_torchrun=12533`); RLDS-conversion-if-needed cell (06a cell 14) runs BEFORE registration. **Live GPU execution not observable locally — routed to Human Verification** |
| 5 | `register_soarm_spatial`/`apply_soarm_spatial_registration` inject `soarm_spatial` into OXE registry dicts before `finetune.py`'s config import, without editing installed package files | ✓ VERIFIED | `oxe_register.py` matches plan spec exactly (pure dict-mutation core + guarded Colab wrapper); `conda run -n libero pytest LIBERO/libero/libero/datasets/test_oxe_register.py -q` → **3 passed** (dict-shape injection, custom dataset_name, and a REAL unmocked fail-loud-without-prismatic test — this project's local env genuinely has no `prismatic` installed) |
| 6 | Periodic private, resumable HF Hub checkpoint pushes during training (D-01/D-02/D-03/D-10) | ✓ VERIFIED (code) / ⚠️ LIVE RUN PENDING (behavior) | `push_checkpoint_to_hub` (06a cell 24): `create_repo(repo_id, private=True, exist_ok=True)` immediately followed by `assert HfApi().repo_info(repo_id).private is True` (hard gate, not just a call-site arg); `poll_and_push_checkpoints` (cell 25) tracks pushed dirs in a `set()` and prunes local disk to the 2 most recent. **Note:** the torchrun cell blocks the kernel synchronously (`Popen` + blocking `poll()` loop) until `finetune.py` exits, so "periodic" pushes during the run require the researcher to manually interrupt and re-run cell 25 rather than it running truly concurrently — this matches the plan's own alternative wording ("a periodic cell the researcher re-runs") so it is not a deviation, but is worth the researcher's attention during the live run |
| 7 | `finetune.py`'s native WandB hooks log training-loss curves to a new dedicated `soarm-oft-finetune-eval` project (TUNE-04, D-13) | ⚠️ PRESENT, LIVE RUN PENDING | `WANDB_PROJECT = "soarm-oft-finetune-eval"` passed as `--wandb_project` to the torchrun cmd; same constant literal reused in `06b-eval.ipynb`. **Actual WandB dashboard content not observable locally — routed to Human Verification** |
| 8 | Spatial vs. non-spatial task success rates are measured and compared before and after fine-tuning under an identical seeded protocol (TUNE-03, D-07/D-08/D-09) | ✓ VERIFIED (code) / ⚠️ LIVE RUN PENDING (results) | `eval_loop.py`'s `run_episode`/`run_suite` gained `seed`/`episode_seeds` params, applied fresh **inside** the per-episode loop (not hoisted before it) — confirmed by direct read (lines 78-79, 160-165) and the exact-placement acceptance check `grep -B2 "obs = env.reset()"` pattern; `conda run -n libero pytest LIBERO/libero/libero/vla -q` → **19/19 passed** including 3 new seed-determinism tests; `06b-eval.ipynb` passes `episode_seeds=EPISODE_SEEDS` (`EPISODE_SEEDS = list(range(20))`) to BOTH the before-cell (`OFTBackend`) and after-cell (`FinetunedOFTBackend`) calls (2 occurrences confirmed), across the exact D-07 4-task list (3 `libero_spatial_soarm/` + 1 `libero_goal/`). **Actual before/after success-rate numbers not observable locally — routed to Human Verification** |
| 9 | `FinetunedOFTBackend` loads a HF Hub adapter, merges via `peft.PeftModel.merge_and_unload()`, re-applies `set_num_images_in_input(2)`, and overlays the adapter's OWN `dataset_statistics.json` (not Phase 4's) while inheriting `predict()` unchanged | ✓ VERIFIED | `adapter_backend.py` matches plan spec exactly: `super().__init__()` first, then `snapshot_download`/`PeftModel.from_pretrained(...).merge_and_unload()`, re-applied `set_num_images_in_input(2)`, `hf_hub_download(adapter_repo_id, "dataset_statistics.json")` (not `checkpoint`), `predict` NOT overridden (`grep -q "def predict"` → no match, confirmed); `conda run -n libero pytest LIBERO/libero/libero/vla/test_adapter_backend.py -q` → part of the 19/19 passing `vla` suite run above |
| 10 | Training loss curves and evaluation success rates are visible together in one WandB run dashboard (TUNE-04) | ⚠️ PRESENT, LIVE RUN PENDING | Both notebooks target the identical `WANDB_PROJECT = "soarm-oft-finetune-eval"` string constant (confirmed in both notebooks' source); 06b's WandB eval wrapper builds a `wandb.Table` (task/before/after/delta columns) + aggregate scalars and logs via `wandb_run.log(...)`. **Dashboard co-location not observable locally — routed to Human Verification** |

**Score:** 10/10 must-haves verified within the locally-verifiable scope (code correctness, test coverage, notebook wiring/ordering). 5 items above additionally require a live Colab A100 GPU run to confirm the runtime behavior the goal ultimately depends on — these are NOT marked FAILED (per this project's established convention for GPU-only-verifiable steps, documented in 06-VALIDATION.md's Manual-Only Verifications table and both 06-02-SUMMARY.md/06-03-SUMMARY.md) and are listed in Human Verification below instead.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `LIBERO/libero/libero/datasets/rlds_converter.py` | `validate_episode_arrays`/`load_episodes_from_hdf5`/`hdf5_to_rlds`, zero TF at module top | ✓ VERIFIED | Exists, matches spec, imports cleanly locally |
| `LIBERO/libero/libero/datasets/test_rlds_converter.py` | 8 tests (6 local, 2 skip-guarded) | ✓ VERIFIED | 6 passed, 2 skipped (confirmed via direct pytest run) |
| `LIBERO/libero/libero/datasets/oxe_register.py` | `register_soarm_spatial`/`apply_soarm_spatial_registration` | ✓ VERIFIED | Exists, matches spec exactly |
| `LIBERO/libero/libero/datasets/test_oxe_register.py` | 3 tests | ✓ VERIFIED | 3 passed (confirmed via direct pytest run) |
| `LIBERO/notebooks/06a-finetune.ipynb` | Bootstrap → RLDS conversion → OXE registration → finetune.py torchrun → checkpoint push → WandB config | ✓ VERIFIED (structure) | Valid JSON, 27 cells, all required flags/cells present and correctly ordered |
| `LIBERO/libero/libero/vla/eval_loop.py` | `seed`/`episode_seeds` additive kwargs | ✓ VERIFIED | Exists, matches spec, backward-compatible |
| `LIBERO/libero/libero/vla/adapter_backend.py` | `FinetunedOFTBackend` | ✓ VERIFIED | Exists, matches spec exactly |
| `LIBERO/libero/libero/vla/test_adapter_backend.py` | 2 tests | ✓ VERIFIED | Part of the 19/19 `vla` suite pass |
| `LIBERO/notebooks/06b-eval.ipynb` | Adapter download, seeded before/after benchmark, WandB eval logging | ✓ VERIFIED (structure) | Valid JSON, all required constants/cells present |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `rlds_converter.hdf5_to_rlds`'s `out_dir`/`dataset_name` | `06a-finetune.ipynb`'s `torchrun --data_root_dir/--dataset_name` | `RLDS_OUT_DIR`/`RLDS_DATASET_NAME` variables reused verbatim | ✓ WIRED | Confirmed by direct cell read: cell 14 defines both constants, cell 22's `finetune_cmd` list passes them as `--data_root_dir RLDS_OUT_DIR --dataset_name RLDS_DATASET_NAME` |
| `06a-finetune.ipynb`'s RLDS-conversion-if-needed cell | `apply_soarm_spatial_registration` cell | Source ordering (conversion cell index 14 < registration cell index 16 < torchrun cell index ~22) | ✓ WIRED | Confirmed via source-offset check: `idx_conv=9902 < idx_reg=10671 < idx_torchrun=12533` |
| `06a-finetune.ipynb`'s `push_checkpoint_to_hub`'s `create_repo(private=True)` | `FinetunedOFTBackend(adapter_repo_id=...)` in `06b-eval.ipynb` | `HF_ADAPTER_REPO_ID` naming convention (`f"{username}/soarm-oft-lora-{timestamp}"`) manually pasted by researcher into `06b`'s `ADAPTER_REPO_ID` cell | ✓ WIRED (design) / ⚠️ requires live run to instantiate a real value | `06b-eval.ipynb` has an explicit, clearly-commented `ADAPTER_REPO_ID = "<paste...>"` cell; this is a genuine runtime input by design (cannot be hardcoded before a real training run exists) |
| Both notebooks' `wandb.init(project=WANDB_PROJECT)` | Shared WandB dashboard | Identical `WANDB_PROJECT = "soarm-oft-finetune-eval"` string literal | ✓ WIRED | Confirmed identical string present in both notebooks' source |
| `eval_loop.run_suite`'s `episode_seeds` param | `run_episode`'s `seed` param, applied inside the per-episode loop | `seed = episode_seeds[ep_idx] if episode_seeds is not None else None` inside the `for ep_idx in range(...)` loop | ✓ WIRED | Confirmed by direct read: `eval_loop.py` lines 160-165, seed application is INSIDE the loop (Pitfall 7's exact correctness requirement), not hoisted before it |
| `FinetunedOFTBackend.__init__`'s `hf_hub_download(adapter_repo_id, "dataset_statistics.json")` | `self.model.norm_stats` overwrite | Direct assignment after `super().__init__()` already set the base-checkpoint stats | ✓ WIRED | Confirmed by direct read: `adapter_backend.py` lines 61-63 fetch from `adapter_repo_id` (not `checkpoint`), overwriting the inherited stats |

### Behavioral Spot-Checks / Local Test Runs

| Command | Result | Status |
|---------|--------|--------|
| `conda run -n libero pytest LIBERO/libero/libero/datasets/test_rlds_converter.py -q` | 6 passed, 2 skipped | ✓ PASS |
| `conda run -n libero pytest LIBERO/libero/libero/datasets/test_oxe_register.py -q` | 3 passed | ✓ PASS |
| `conda run -n libero pytest LIBERO/libero/libero/vla -q` | 19 passed | ✓ PASS |
| `conda run -n libero pytest LIBERO/libero/libero/datasets -q` (full package regression) | 2 failed, 31 passed, 2 skipped | ⚠️ 2 pre-existing failures — see below, NOT attributed to Phase 6 |
| `python -c` notebook JSON-validity + content-presence checks (both notebooks) | All expected flags/constants/cells present | ✓ PASS |

**Pre-existing test debt (confirmed NOT caused by Phase 6):** `test_hdf5_writer.py::test_schema_and_obs_key_naming` and `test_replay.py::test_verify_full_obs_regeneration_passes_on_04_02_output` both fail on the full `datasets` package suite run. Both are documented Phase 4/5 debt (stale gripper-shape assertions predating the D-07 84mm 2-DOF gripper upgrade), confirmed via `git log` that no Phase 6 commit touched either `test_hdf5_writer.py`/`hdf5_writer.py` or `test_replay.py`/`replay.py`, and explicitly acknowledged in both `06-01-SUMMARY.md` and `06-02-SUMMARY.md`'s "Issues Encountered" sections plus STATE.md's Blockers/Concerns. Per this task's explicit instruction, not attributed to this phase.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|--------------|-------------|--------------|--------|----------|
| TUNE-01 | 06-01 | Robomimic HDF5 → RLDS conversion | ✓ SATISFIED (locally verifiable scope) | `hdf5_to_rlds` implemented, validated, 6/8 tests pass locally; genuine-TFDS-write round-trip requires Colab (`tensorflow_datasets`), tracked as Manual-Only in 06-VALIDATION.md. REQUIREMENTS.md already marks TUNE-01 `[x]` Complete. |
| TUNE-02 | 06-02 | LoRA r=32 fine-tuning on Colab A100 | ⚠️ CODE COMPLETE, LIVE RUN PENDING | All notebook cells/flags present and correctly ordered; live GPU completion not observable locally. REQUIREMENTS.md correctly leaves this `[ ]` Pending. |
| TUNE-03 | 06-03 | Spatial vs. non-spatial before/after benchmark | ⚠️ CODE COMPLETE, LIVE RUN PENDING | Seed plumbing + `FinetunedOFTBackend` fully implemented and locally tested (19/19); actual before/after numbers require live run. REQUIREMENTS.md correctly leaves this `[ ]` Pending. |
| TUNE-04 | 06-02, 06-03 | Training + eval metrics tracked in WandB | ⚠️ CODE COMPLETE, LIVE RUN PENDING | Both notebooks target the same `soarm-oft-finetune-eval` WandB project; actual dashboard content requires live runs of both notebooks. REQUIREMENTS.md correctly leaves this `[ ]` Pending. |

No orphaned requirements — all 4 IDs (TUNE-01 through TUNE-04) are claimed by the plan frontmatter of at least one of the 3 plans in this phase and cross-referenced against REQUIREMENTS.md's Phase 6 table.

### Anti-Patterns Found

None. Scanned all files created/modified by this phase's 3 plans (`rlds_converter.py`, `test_rlds_converter.py`, `oxe_register.py`, `test_oxe_register.py`, `eval_loop.py`, `test_eval_loop.py`, `adapter_backend.py`, `test_adapter_backend.py`) for `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER`/"not yet implemented" — zero matches. No debt-marker gate violations.

**Note (non-blocking observation):** `06a-finetune.ipynb`'s checkpoint-push polling loop (cell 25) cannot run truly concurrently with the blocking `torchrun` `Popen`+poll cell (cell 22) within a single-threaded kernel — periodic pushes during a long training run require the researcher to manually interrupt and re-run cell 25. This matches the plan's own alternative wording ("a periodic cell the researcher re-runs") so it is not a deviation from the plan, but is flagged here so the live-Colab run (Human Verification below) accounts for it.

## Human Verification Required

5 items require a live Colab A100 GPU run — see YAML frontmatter `human_verification` list for full detail. Summary:

1. **`06a-finetune.ipynb` end-to-end run** — RLDS conversion produces a genuine `tfds.builder`-loadable dataset; `finetune.py` completes without crashing.
2. **Checkpoint push + resume-from-disconnect** — private HF Hub repo confirmed, WandB step counter continues after a simulated disconnect (not reset to 0).
3. **WandB training-loss curves visible** in the `soarm-oft-finetune-eval` project.
4. **`06b-eval.ipynb` end-to-end run** — before/after benchmarks complete under the identical seeded protocol; `FinetunedOFTBackend`'s post-merge `set_num_images_in_input` call succeeds (Assumption A4).
5. **WandB dashboard shows both training curves AND eval success-rate tables together** in the same project (the exact check 06-VALIDATION.md flags).

## Gaps Summary

No gaps in the locally-verifiable scope. All code, tests, and notebook wiring for this phase's three plans (RLDS conversion, OXE registration + training notebook, seed plumbing + adapter reload + eval notebook) are present, correct, and pass their local test suites (28/28 phase-specific tests across the three plans, plus the 19/19 full `vla` suite regression). The 2 failing tests in the full `datasets` package suite are pre-existing Phase 4/5 debt, confirmed untouched by any Phase 6 commit, and were explicitly accepted by the user during this execution run — not attributed to this phase.

The phase goal's remaining unverified surface (LoRA training actually completing on a real A100, checkpoint resumability, and both notebooks' WandB dashboard content) is inherently a live-GPU-only observation this project has consistently treated as a Manual-Only verification step since 06-VALIDATION.md was authored — not a code or wiring defect. REQUIREMENTS.md correctly reflects this: TUNE-01 is `[x]` Complete while TUNE-02/03/04 remain `[ ]` Pending, exactly matching this verification's findings.

---

*Verified: 2026-08-23T08:00:11Z*
*Verifier: Claude (gsd-verifier)*
