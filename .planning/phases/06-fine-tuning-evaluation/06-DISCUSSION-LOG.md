# Phase 6: Fine-Tuning & Evaluation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-20
**Phase:** 6-Fine-Tuning & Evaluation
**Areas discussed:** Checkpoint & artifact persistence, RLDS conversion approach, Eval benchmark scope & protocol, Training compute strategy

---

## Checkpoint & Artifact Persistence

**Preliminary:** User asked to understand what a LoRA adapter is, why it's needed, and what alternatives exist (full fine-tuning, QLoRA, prompt tuning, linear probing). Answered with a comparison table. Confirmed LoRA r=32 is locked by TUNE-02 in REQUIREMENTS.md and not open for reconsideration in this discussion — user confirmed proceeding.

### Where should the LoRA adapter be persisted?

| Option | Description | Selected |
|--------|-------------|----------|
| HF Hub (private repo) | Reuses existing HF token/auth, small artifact, huggingface_hub already a dependency | ✓ |
| Google Drive | Matches Phase 3's "local disk, copy after" pattern, but Drive was dropped project-wide for GitHub delivery | |
| GCS bucket | Proven working on Colab, but overkill for a tens-of-MB artifact | |

**User's choice:** HF Hub, private repo, free tier.
**Notes:** User asked follow-up clarifying questions along the way — HF Hub vs GitHub distinction (code vs model-artifact hosting) and whether HF Hub is free (confirmed: free for private model repos at this scale, no new cost).

### Should intermediate training checkpoints also be pushed?

| Option | Description | Selected |
|--------|-------------|----------|
| Only final adapter | Simplest, accept restarting from scratch on disconnect | |
| Periodic checkpoints too | Adds resumability, extra upload/storage churn | ✓ |

**User's choice:** Periodic checkpoints too.

### HF Hub repo naming convention?

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed repo id, overwritten per run | Simple, predictable path | |
| Timestamped/versioned repos per run | Each run separately addressable for comparison | ✓ |

**User's choice:** Timestamped/versioned repos per run.

---

## RLDS Conversion Approach

**Preliminary:** User asked what the dataset itself is (schema, provenance). Answered: 120 demos, robomimic HDF5, 7-D proprio (5 joints + 2-DOF gripper), dual-camera (agentview + eye_in_hand), collected Phase 4, dual-camera wired Phase 5.

### How should we convert HDF5 → RLDS?

| Option | Description | Selected |
|--------|-------------|----------|
| Custom converter to our schema | Project-specific, testable in isolation | ✓ |
| Adapt openvla-oft's own RLDS builder tooling | Less code, but not vendored, may assume different conventions | |

**User's choice:** Write custom converter to our schema.

### Where should converted RLDS live, one-time or inline conversion?

| Option | Description | Selected |
|--------|-------------|----------|
| One-time script, committed Colab-disk path | Mirrors Phase 4's normalization.py pattern | ✓ |
| Convert inline every training run | Guarantees freshness, redoes identical work each run | |

**User's choice:** One-time script, output committed path on Colab disk.

### Testable module or notebook cell?

| Option | Description | Selected |
|--------|-------------|----------|
| Testable Python module + unit tests | Matches project convention, verifies schema before burning A100 budget | ✓ |
| Notebook-only conversion cell | Faster to write, breaks from established convention | |

**User's choice:** Testable Python module + unit tests.

---

## Eval Benchmark Scope & Protocol

### How many episodes per task?

| Option | Description | Selected |
|--------|-------------|----------|
| 10 episodes/task | Matches typical LIBERO convention, less compute | |
| 20 episodes/task | More stable estimates, doubles eval time | ✓ |
| 5 episodes/task | Fastest, noisiest | |

**User's choice:** 20 episodes/task.

### Identical or independent seeds before/after?

| Option | Description | Selected |
|--------|-------------|----------|
| Identical seeds before/after | Fair apples-to-apples comparison | ✓ |
| Independent random seeds | Simpler, weaker evidence for a genuine improvement | |

**User's choice:** Identical seeds before/after.

### Reuse Phase 3's 0% baseline or re-run before-benchmark?

| Option | Description | Selected |
|--------|-------------|----------|
| Re-run full before-benchmark anyway | Clean, directly-comparable number under identical protocol | ✓ |
| Reuse Phase 3's 0% as before-baseline | Saves eval compute/time | |

**User's choice:** Re-run full before-benchmark anyway.

---

## Training Compute Strategy

### Resumability across sessions?

| Option | Description | Selected |
|--------|-------------|----------|
| Design for resumability from the start | Builds resume-from-checkpoint into training script from day one | ✓ |
| Assume single continuous session, no resume logic | Simpler, restart from scratch on disconnect | |

**User's choice:** Design for resumability from the start.

### Training driver script?

| Option | Description | Selected |
|--------|-------------|----------|
| Call openvla-oft's finetune.py directly | Reuses upstream tested LoRA/PEFT implementation | ✓ |
| Author custom training wrapper/loop | Full control, more code to write/debug | |

**User's choice:** Call openvla-oft's finetune.py directly with our args.

### Separate or combined training/eval notebooks?

| Option | Description | Selected |
|--------|-------------|----------|
| Separate training notebook + separate eval notebook | Matches Phase 3's kernel-group separation pattern | ✓ |
| Single combined notebook | Simpler, risks reintroducing dependency conflict | |

**User's choice:** Separate training notebook + separate eval notebook.

### New or existing WandB project?

| Option | Description | Selected |
|--------|-------------|----------|
| New dedicated WandB project | Keeps fine-tuning/eval metrics isolated from lifelong-learning runs | ✓ |
| Reuse existing lifelong-learning WandB project | Less setup, mixes unrelated metrics | |

**User's choice:** New dedicated WandB project for this fine-tuning run.

---

## Claude's Discretion

- Exact RLDS output directory naming/layout within `LIBERO/libero/datasets/soarm_spatial/rlds/`.
- Exact `finetune.py` hyperparameters beyond the locked `r=32` (epochs/steps, batch size, learning rate, checkpoint-push frequency).
- Whether `run_suite`/`eval_loop.py` needs modification (e.g. seed parameter plumbing) or a thin wrapper suffices for identical-seed before/after comparison.
- New WandB project naming convention.

## Deferred Ideas

None — discussion stayed within phase scope.
