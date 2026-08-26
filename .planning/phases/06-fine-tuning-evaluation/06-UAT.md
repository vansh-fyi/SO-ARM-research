---
status: diagnosed
phase: 06-fine-tuning-evaluation
source: [06-VERIFICATION.md]
started: 2026-08-23T08:15:00.000Z
updated: 2026-08-26T00:00:00.000Z
---

## Current Test

[testing paused — 4 items outstanding]

## Tests

### 1. Run 06a-finetune.ipynb end-to-end on Colab A100 (RLDS conversion + finetune.py completion)
expected: |
  Block A install, restart, Block B bootstrap, RLDS-conversion-if-needed cell, OXE registration
  smoke-test cell, HF Hub (write) + WandB auth, then the finetune.py torchrun cell
  (--lora_rank 32 --use_lora True --merge_lora_during_training False --save_latest_checkpoint_only False)
  all complete without error.
result: issue
reported: |
  RLDS conversion cell and OXE registration smoke-test cell both fail with
  `ModuleNotFoundError: No module named 'libero.datasets'` when running
  `from libero.datasets.rlds_converter import hdf5_to_rlds` and
  `from libero.datasets.oxe_register import apply_soarm_spatial_registration`.
  Block A install and Block B bootstrap succeeded; HF Hub write auth and WandB auth
  both completed successfully. finetune.py torchrun cell was not reached.
severity: blocker

### 2. Checkpoint push + resume-from-disconnect
expected: |
  At least one LoRA checkpoint pushed to the private HF Hub repo during/after training
  (HfApi().repo_info(HF_ADAPTER_REPO_ID).private is True, repo visible under the researcher's
  account). After simulating a disconnect (interrupt kernel mid-training) and relaunching the
  torchrun cell with RESUME=True/RESUME_STEP=<N>, the WandB run's step counter continues from
  <N> rather than resetting to 0.
result: blocked
blocked_by: prior-phase
reason: "Test 1 didn't complete (RLDS/OXE ModuleNotFoundError); user wants that fixed before attempting LoRA training/checkpointing"

### 3. WandB training-loss curves visible
expected: |
  Training-loss curves from the 06a-finetune.ipynb run are visible in the shared
  soarm-oft-finetune-eval WandB project (TUNE-04).
result: blocked
blocked_by: prior-phase
reason: "Depends on test 1 (finetune.py run) completing; blocked on the RLDS/OXE ModuleNotFoundError"

### 4. Run 06b-eval.ipynb end-to-end (before/after seeded benchmark)
expected: |
  Paste the trained run's HF_ADAPTER_REPO_ID into 06b-eval.ipynb's ADAPTER_REPO_ID cell and run
  end-to-end on Colab A100/T4: Block A, restart, Block B, task/language setup, WandB init, the
  BEFORE cell (zero-shot OFTBackend), the Pitfall-6 divergence-check cell, and the AFTER cell
  (FinetunedOFTBackend). Both before and after run_suite benchmarks complete under the identical
  EPISODE_SEEDS = list(range(20)) protocol across the same 4 tasks (3 spatial + 1 non-spatial);
  FinetunedOFTBackend successfully downloads and merges the adapter without an AttributeError on
  set_num_images_in_input; the divergence-check cell prints a clean pass or a clearly-flagged
  WARNING (not a silent skip).
result: blocked
blocked_by: prior-phase
reason: "Requires a trained adapter (HF_ADAPTER_REPO_ID) from test 1; blocked on the RLDS/OXE ModuleNotFoundError"

### 5. WandB dashboard shows both training curves AND eval results together
expected: |
  The before/after wandb.Table (per-task success rates) and aggregate scalars appear in the same
  soarm-oft-finetune-eval WandB project as the training-loss curves from item 3 — one dashboard,
  both metric types (TUNE-04).
result: blocked
blocked_by: prior-phase
reason: "Depends on tests 1, 3, and 4 completing; blocked on the RLDS/OXE ModuleNotFoundError"

## Summary

total: 5
passed: 0
issues: 1
pending: 0
skipped: 0
blocked: 4

## Gaps

- truth: "hdf5_to_rlds produces a genuine tfds.builder-loadable RLDS dataset and apply_soarm_spatial_registration prints a registered soarm_spatial config with no traceback"
  status: failed
  reason: "User reported: RLDS conversion cell and OXE registration smoke-test cell both fail with ModuleNotFoundError: No module named 'libero.datasets' when running `from libero.datasets.rlds_converter import hdf5_to_rlds` and `from libero.datasets.oxe_register import apply_soarm_spatial_registration`. Block A install and Block B bootstrap succeeded; HF Hub write auth and WandB auth both completed successfully. finetune.py torchrun cell was not reached."
  severity: blocker
  test: 1
  root_cause: "06a-finetune.ipynb imports from the wrong module path — `libero.datasets.*` instead of `libero.libero.datasets.*`. The actual package layout has datasets nested one level deeper (LIBERO/libero/libero/datasets/{rlds_converter,oxe_register}.py), and sys.path only has LIBERO/libero/LIBERO inserted (LIBERO_PKG), so the importable top-level package is `libero` -> `libero.libero.datasets...`. Every other notebook in the repo (06b-eval.ipynb, 03a/03b, 01, 02, quick_walkthrough.ipynb) correctly uses the `libero.libero.*` prefix; only 06a-finetune.ipynb's RLDS/OXE cells were written with the shorter, incorrect path."
  artifacts:
    - path: "LIBERO/notebooks/06a-finetune.ipynb"
      issue: "RLDS-conversion cell: `from libero.datasets.rlds_converter import hdf5_to_rlds` should be `from libero.libero.datasets.rlds_converter import hdf5_to_rlds`"
    - path: "LIBERO/notebooks/06a-finetune.ipynb"
      issue: "OXE-registration cell: `from libero.datasets.oxe_register import apply_soarm_spatial_registration` should be `from libero.libero.datasets.oxe_register import apply_soarm_spatial_registration`"
  missing:
    - "Fix both import statements to use the `libero.libero.datasets` prefix"
  debug_session: ""
