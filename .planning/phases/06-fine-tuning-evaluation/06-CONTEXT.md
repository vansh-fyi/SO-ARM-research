# Phase 6: Fine-Tuning & Evaluation - Context

**Gathered:** 2026-08-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Fine-tune OpenVLA-OFT on the 120-demo SOARM dataset (collected Phase 4, dual-camera per Phase 5) via LoRA (r=32) on Colab A100, and benchmark spatial vs non-spatial task success rates before and after fine-tuning, with training/eval metrics tracked in WandB. This covers TUNE-01 (HDF5→RLDS conversion), TUNE-02 (LoRA fine-tuning), TUNE-03 (before/after benchmarking), and TUNE-04 (WandB tracking). Real-hardware transfer and any new spatial capabilities remain out of scope — this phase only fine-tunes and evaluates what Phases 4/5 already built.

</domain>

<decisions>
## Implementation Decisions

### Fine-tuning method (locked upstream, not re-litigated here)
- **D-00:** LoRA (r=32), not full fine-tuning, QLoRA, or prompt/prefix tuning, is locked by TUNE-02 in REQUIREMENTS.md — decided at the roadmap stage. LoRA fits the constraints (120 demos is a small dataset; Colab A100 has session memory/time limits; openvla-oft's own repo has first-class LoRA support so no training-loop needs to be built from scratch). Reconsidering the method itself would require a REQUIREMENTS.md/ROADMAP.md amendment, not a discuss-phase decision — user confirmed proceeding with LoRA r=32 as-is.

### Checkpoint & Artifact Persistence
- **D-01:** The LoRA-finetuned adapter is pushed to a **Hugging Face Hub private repo** (free tier) — reuses the HF token/auth already set up for the base checkpoint download (`moojink/openvla-7b-oft-finetuned-libero-spatial`), and `huggingface_hub` is already a project dependency. Rejected: Google Drive (mounting was explicitly dropped project-wide in favor of GitHub delivery, commit `260802-ijb`) and GCS (overkill for a tens-of-MB adapter).
- **D-02:** Periodic intermediate training checkpoints are also pushed to HF Hub during training, not just the final adapter — gives resumability if a Colab session disconnects mid-training (directly supports D-06's resumability requirement below).
- **D-03:** HF Hub repo ids are timestamped/versioned per training run (not one fixed, overwritten repo id) — keeps every fine-tuning attempt separately addressable for comparison.

### RLDS Conversion (TUNE-01)
- **D-04:** Write a **custom converter script** mapping our robomimic HDF5 schema (`agentview_rgb`, `eye_in_hand_rgb`, 7-D proprio [5 joints + 2-DOF gripper], actions) to RLDS format — not adapting openvla-oft's own upstream RLDS builder tooling (which isn't vendored in this repo and may assume different dataset conventions).
- **D-05:** Conversion is a **one-time script producing a committed output path on Colab disk** (e.g. `LIBERO/libero/datasets/soarm_spatial/rlds/`) — mirrors Phase 4's `normalization.py` pattern (standalone script → `dataset_statistics.json`). It is NOT regenerated inline at the start of every training run.
- **D-06:** The RLDS converter must be a **testable Python module with unit tests** — matches the project's established convention (Phase 4's `normalization.py`/`hdf5_writer.py` both have `test_*.py` counterparts, per CLAUDE.md "real importable Python modules over notebook-only code"). Verify RLDS schema/shapes/dtypes match openvla-oft's expectations before spending A100 budget on a training run that could fail on bad data.

### Eval Benchmark Scope & Protocol (TUNE-03)
- **D-07:** Benchmark uses **20 episodes per task**, across all 4 available SOARM tasks: 3 spatial (`put_the_butter_between_the_bowl_and_the_cream_cheese`, `put_the_cream_cheese_near_the_bowl`, `put_the_cream_cheese_to_the_right_of_the_bowl` — `LIBERO/libero/libero/bddl_files/libero_spatial_soarm/`) and 1 non-spatial (`put_the_cream_cheese_in_the_bowl` — `LIBERO/libero/libero/bddl_files/libero_goal/`). That's 80 episodes before + 80 after = 160 episodes total.
- **D-08:** Before/after comparison uses **identical episode seeds** (same initial object placements in both runs) — isolates the fine-tuning effect from initial-state randomization noise, standard practice for before/after benchmarking.
- **D-09:** The **full before (zero-shot) benchmark is re-run** on these exact 4 tasks under the same 20-episode/seeded protocol, rather than reusing Phase 3's previously-reported 0% baseline (STATE.md 03-02/03-03 notes) — Phase 3's 0% was measured under a different task/setup context; re-running now gives a clean, directly-comparable before number under the identical protocol used for after.
- **Reuse note:** `LIBERO/libero/libero/vla/eval_loop.py::run_suite` already implements per-episode PASS/FAIL polling, video recording, and an aggregated success-rate summary table — it's a strong reuse candidate for driving this benchmark rather than reimplementing episode-loop logic. Whether it needs seed-plumbing added (for D-08) is a planning/research question.

### Training Compute Strategy
- **D-10:** Design the training script for **resumability from the start** — Colab sessions can disconnect unpredictably (per Phase 1's environment lessons), so resume-from-checkpoint logic goes in from day one rather than being retrofitted after a wasted run. Ties directly to D-02 (periodic checkpoints pushed to HF Hub) as the resume source.
- **D-11:** Training is driven by **calling openvla-oft's own `finetune.py`** (pip-installed via the moojink fork, not vendored in this repo) with our args, rather than authoring a custom training loop — reuses the upstream, already-tested LoRA/PEFT training implementation, matching Phase 3's established pattern of calling pip-installed openvla-oft/openpi code rather than reimplementing VLA internals.
- **D-12:** **Separate training notebook and separate eval notebook** — follows the Phase 3 precedent (two Colab kernel groups to avoid a transformers version conflict between LIBERO training/eval and VLA inference code). Training notebook: RLDS conversion + `finetune.py` + HF Hub push. Eval notebook: download adapter from HF Hub + run the before/after `run_suite` benchmark.
- **D-13:** TUNE-04 (WandB tracking) uses a **new, dedicated WandB project** for this fine-tuning + eval work, not the existing lifelong-learning project already used by `LIBERO/libero/lifelong/main.py` — keeps LoRA training/eval metrics and the TUNE-03 before/after comparison isolated from unrelated lifelong-learning algorithm runs (EWC/AGEM/etc.) in the dashboard.

### Claude's Discretion
- Exact RLDS output directory naming/layout within `LIBERO/libero/datasets/soarm_spatial/rlds/`.
- Exact `finetune.py` hyperparameters beyond the locked `r=32` (epochs/steps, batch size, learning rate, checkpoint-push frequency) — informed by openvla-oft's own documented defaults/recommendations during planning/research.
- Whether `run_suite`/`eval_loop.py` needs modification (e.g. seed parameter plumbing) or a thin wrapper suffices to satisfy D-08's identical-seed requirement.
- New WandB project naming convention.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & Roadmap
- `.planning/REQUIREMENTS.md` — TUNE-01 through TUNE-04 are this phase's requirements (lines 44-47, 106-109).
- `.planning/ROADMAP.md` §Phase 6: Fine-Tuning & Evaluation — goal and success criteria (RLDS conversion + LoRA r=32 fine-tuning completes without crashing; before/after success rates measured; WandB dashboard shows training loss + eval results).

### Existing Dataset (this phase's conversion input)
- `LIBERO/libero/datasets/soarm_spatial/put_the_cream_cheese_in_the_bowl_demo.hdf5` — the 120-demo dataset TUNE-01 converts to RLDS.
- `LIBERO/libero/datasets/soarm_spatial/dataset_statistics.json` — OpenVLA q01/q99/mean/std/min/max normalization stats already computed (Phase 4, 04-04); relevant context for RLDS conversion / fine-tuning setup, may need extension or reuse.
- `.planning/phases/04-dataset-collection/04-CONTEXT.md` and `04-04-SUMMARY.md` — 7-D proprio schema rationale (5 joints + 2-DOF gripper) the RLDS converter must preserve.

### Eval Infrastructure (reuse candidate for TUNE-03)
- `LIBERO/libero/libero/vla/eval_loop.py` — `run_suite()`/`run_episode()` already implement the episode-loop, PASS/FAIL polling, video recording, and summary table this phase's before/after benchmark needs.
- `LIBERO/libero/libero/vla/interface.py`, `oft_backend.py` — VLA backend contract; the fine-tuned adapter needs to be loadable through this same `predict(images, language)` interface for eval to reuse `run_suite`.
- `LIBERO/libero/libero/bddl_files/libero_spatial_soarm/` — 3 spatial task BDDL files (Phase 5).
- `LIBERO/libero/libero/bddl_files/libero_goal/put_the_cream_cheese_in_the_bowl.bddl` — the non-spatial task BDDL file (Phase 4).

### Prior Phase Context (embodiment + infra constraints that still apply)
- `.planning/phases/03-vla-inference-loop/03-CONTEXT.md` and STATE.md Phase 3 notes — two-kernel-group pattern (transformers version conflict between LIBERO and VLA inference stacks) motivating D-12's notebook split; OFT-on-SOARM 0% zero-shot baseline context for D-09.
- `.planning/phases/05-spatial-awareness/05-CONTEXT.md` — dual-camera wiring (D-01/D-02) the fine-tuned model must continue to consume correctly.
- `.planning/phases/01-colab-environment-setup/01-DEBUG-HISTORY.md` — the environment contract (7 invariants) and Colab session-disconnect lessons motivating D-10's resumability requirement.
- STATE.md "Accumulated Context" — HF token delivery via Drive file bootstrap (Block B), reusable pattern for HF Hub push auth in this phase's training notebook.

### External (openvla-oft upstream, not vendored)
- `moojink/openvla-oft` fork (pip-installed, `--no-deps`, per `01-DEBUG-HISTORY.md`) — source of `finetune.py` (D-11) and the base checkpoint `moojink/openvla-7b-oft-finetuned-libero-spatial`. No local path; research phase must confirm current `finetune.py` CLI/API surface directly from the installed package or its GitHub source.

### Codebase Maps
- `.planning/codebase/STACK.md` — pinned dependency versions relevant to RLDS/TFDS tooling compatibility.
- `.planning/codebase/INTEGRATIONS.md` — existing WandB auth pattern (`WANDB_API_KEY` env var) for D-13's new project.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `eval_loop.py::run_suite`/`run_episode` — full episode-loop, PASS/FAIL, video, summary-table logic; strong candidate to drive the before/after benchmark rather than reimplementing.
- `normalization.py` (Phase 4, 04-04) — establishes the "standalone script producing a JSON/data artifact" pattern the RLDS converter (D-05/D-06) should follow.
- HF token bootstrap pattern (Drive file → getpass, per STATE.md 01-close notes) — reusable for HF Hub push auth in the training notebook.

### Established Patterns
- Real importable Python modules with `test_*.py` counterparts over notebook-only code (Phase 2-5 convention) — governs D-06's RLDS converter and any new training/eval helper modules.
- Two-kernel-group Colab notebook split when dependency stacks conflict (Phase 3 precedent) — governs D-12.
- `MUJOCO_GL` must be set before any MuJoCo import (applies to the eval notebook, which needs LIBERO env rendering).
- "Local disk during run, copy to Drive/Hub after" persistence pattern (Phase 3) — now specialized to HF Hub per D-01/D-02.

### Integration Points
- RLDS output feeds into `finetune.py` (D-11) as training input.
- The fine-tuned adapter (downloaded from HF Hub per D-01) must load through the existing `oft_backend.py`/`interface.py` contract so `eval_loop.run_suite` can drive it unmodified (or with minimal changes) for the after-benchmark.
- New WandB project (D-13) receives both training-loss curves (from `finetune.py`, if it has native WandB hooks — research target) and eval success-rate results (logged from the eval notebook).

</code_context>

<specifics>
## Specific Ideas

- User wanted a clear grounding in what a LoRA adapter is and why it's the right fit before locking persistence decisions — captured as D-00 with the full rationale so downstream agents don't need to re-justify the method.
- User wanted the HF Hub vs GitHub distinction spelled out (code delivery vs model-artifact hosting) before choosing persistence — this distinction is now implicit in D-01 but worth downstream agents understanding: GitHub stays code-only, HF Hub is now used bidirectionally (pull base checkpoint, push adapter).

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. No scope-creep items surfaced; all four discussed areas (persistence, RLDS conversion, eval protocol, compute strategy) are implementation details of the already-locked TUNE-01..04 requirements.

</deferred>

---

*Phase: 6-Fine-Tuning & Evaluation*
*Context gathered: 2026-08-20*
