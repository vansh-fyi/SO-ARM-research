---
status: testing
phase: 06-fine-tuning-evaluation
source: [06-VERIFICATION.md]
started: 2026-08-23T08:15:00.000Z
updated: 2026-08-23T08:15:00.000Z
---

## Current Test

number: 1
name: Run 06a-finetune.ipynb end-to-end on Colab A100 (RLDS conversion + finetune.py completion)
expected: |
  `hdf5_to_rlds` produces a genuine `tfds.builder`-loadable RLDS dataset (round-trip `.info` call
  does not raise); `apply_soarm_spatial_registration` prints a registered `soarm_spatial` config
  with no traceback; `finetune.py` runs to completion without crashing and prints a final success
  log line.
awaiting: user response

## Tests

### 1. Run 06a-finetune.ipynb end-to-end on Colab A100 (RLDS conversion + finetune.py completion)
expected: |
  Block A install, restart, Block B bootstrap, RLDS-conversion-if-needed cell, OXE registration
  smoke-test cell, HF Hub (write) + WandB auth, then the finetune.py torchrun cell
  (--lora_rank 32 --use_lora True --merge_lora_during_training False --save_latest_checkpoint_only False)
  all complete without error.
result: [pending]

### 2. Checkpoint push + resume-from-disconnect
expected: |
  At least one LoRA checkpoint pushed to the private HF Hub repo during/after training
  (HfApi().repo_info(HF_ADAPTER_REPO_ID).private is True, repo visible under the researcher's
  account). After simulating a disconnect (interrupt kernel mid-training) and relaunching the
  torchrun cell with RESUME=True/RESUME_STEP=<N>, the WandB run's step counter continues from
  <N> rather than resetting to 0.
result: [pending]

### 3. WandB training-loss curves visible
expected: |
  Training-loss curves from the 06a-finetune.ipynb run are visible in the shared
  soarm-oft-finetune-eval WandB project (TUNE-04).
result: [pending]

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
result: [pending]

### 5. WandB dashboard shows both training curves AND eval results together
expected: |
  The before/after wandb.Table (per-task success rates) and aggregate scalars appear in the same
  soarm-oft-finetune-eval WandB project as the training-loss curves from item 3 — one dashboard,
  both metric types (TUNE-04).
result: [pending]

## Summary

total: 5
passed: 0
issues: 0
pending: 5
skipped: 0
blocked: 0

## Gaps
