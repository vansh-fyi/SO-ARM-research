# Phase 6: Fine-Tuning & Evaluation - Research

**Researched:** 2026-08-20
**Domain:** LoRA fine-tuning of a 7B Vision-Language-Action model (OpenVLA-OFT) on a custom robomimic->RLDS dataset, before/after benchmarking on Colab A100, WandB tracking, HF Hub checkpoint persistence
**Confidence:** MEDIUM — finetune.py's CLI surface and checkpoint/resume behavior are CITED from the moojink/openvla-oft source on GitHub (fetched this session), but nothing in this phase can be executed locally (no GPU); all findings need Colab-live confirmation exactly like every prior phase in this project.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Fine-tuning method (locked upstream, not re-litigated here)**
- D-00: LoRA (r=32), not full fine-tuning, QLoRA, or prompt/prefix tuning, is locked by TUNE-02 in REQUIREMENTS.md — decided at the roadmap stage. Reconsidering the method itself would require a REQUIREMENTS.md/ROADMAP.md amendment.

**Checkpoint & Artifact Persistence**
- D-01: The LoRA-finetuned adapter is pushed to a Hugging Face Hub **private repo** (free tier) — reuses the HF token/auth already set up for the base checkpoint download, `huggingface_hub` is already a project dependency. Rejected: Google Drive (mounting dropped project-wide, commit `260802-ijb`), GCS (overkill for a tens-of-MB adapter).
- D-02: Periodic intermediate training checkpoints are also pushed to HF Hub during training, not just the final adapter — gives resumability if a Colab session disconnects mid-training.
- D-03: HF Hub repo ids are timestamped/versioned per training run (not one fixed, overwritten repo id).

**RLDS Conversion (TUNE-01)**
- D-04: Write a **custom converter script** mapping robomimic HDF5 schema (`agentview_rgb`, `eye_in_hand_rgb`, 7-D proprio [5 joints + 2-DOF gripper], actions) to RLDS format — not adapting openvla-oft's own upstream RLDS builder tooling.
- D-05: Conversion is a **one-time script producing a committed output path on Colab disk** (e.g. `LIBERO/libero/datasets/soarm_spatial/rlds/`) — mirrors Phase 4's `normalization.py` pattern. NOT regenerated inline at the start of every training run.
- D-06: The RLDS converter must be a **testable Python module with unit tests** — matches Phase 4's `normalization.py`/`hdf5_writer.py` convention (`test_*.py` counterparts). Verify RLDS schema/shapes/dtypes match openvla-oft's expectations before spending A100 budget on a training run.

**Eval Benchmark Scope & Protocol (TUNE-03)**
- D-07: Benchmark uses **20 episodes per task**, across all 4 available SOARM tasks: 3 spatial (`put_the_butter_between_the_bowl_and_the_cream_cheese`, `put_the_cream_cheese_near_the_bowl`, `put_the_cream_cheese_to_the_right_of_the_bowl` — `LIBERO/libero/libero/bddl_files/libero_spatial_soarm/`) and 1 non-spatial (`put_the_cream_cheese_in_the_bowl` — `LIBERO/libero/libero/bddl_files/libero_goal/`). 80 episodes before + 80 after = 160 episodes total.
- D-08: Before/after comparison uses **identical episode seeds** (same initial object placements in both runs).
- D-09: The **full before (zero-shot) benchmark is re-run** on these exact 4 tasks under the same 20-episode/seeded protocol, rather than reusing Phase 3's previously-reported 0% baseline.
- Reuse note: `LIBERO/libero/libero/vla/eval_loop.py::run_suite` already implements per-episode PASS/FAIL polling, video recording, and an aggregated success-rate summary table — strong reuse candidate. Whether it needs seed-plumbing added (D-08) is a planning question — **RESOLVED by this research below: yes, minimal plumbing needed, see Architecture Patterns.**

**Training Compute Strategy**
- D-10: Design the training script for **resumability from the start** — ties to D-02 (periodic checkpoints to HF Hub as resume source).
- D-11: Training is driven by **calling openvla-oft's own `finetune.py`** (pip-installed via the moojink fork, not vendored in this repo) with our args, rather than authoring a custom training loop.
- D-12: **Separate training notebook and separate eval notebook** — follows Phase 3 precedent. Training notebook: RLDS conversion + `finetune.py` + HF Hub push. Eval notebook: download adapter from HF Hub + run the before/after `run_suite` benchmark.
- D-13: TUNE-04 (WandB tracking) uses a **new, dedicated WandB project** for this fine-tuning + eval work, not the existing lifelong-learning project.

### Claude's Discretion
- Exact RLDS output directory naming/layout within `LIBERO/libero/datasets/soarm_spatial/rlds/`.
- Exact `finetune.py` hyperparameters beyond the locked `r=32` (epochs/steps, batch size, learning rate, checkpoint-push frequency) — informed by openvla-oft's own documented defaults/recommendations (see Standard Stack below).
- Whether `run_suite`/`eval_loop.py` needs modification (e.g. seed parameter plumbing) or a thin wrapper suffices to satisfy D-08's identical-seed requirement — **RESOLVED: thin plumbing, see Architecture Patterns Pattern 2.**
- New WandB project naming convention.

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope. No scope-creep items surfaced; all four discussed areas (persistence, RLDS conversion, eval protocol, compute strategy) are implementation details of the already-locked TUNE-01..04 requirements.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TUNE-01 | Robomimic HDF5 dataset is converted to RLDS format compatible with OpenVLA fine-tuning pipeline | Standard Stack (RLDS/TFDS toolchain), Architecture Patterns Pattern 1 (converter design + OXE registration), Common Pitfalls 1/2/3, Code Examples |
| TUNE-02 | OpenVLA-OFT is fine-tuned on SOARM demonstrations using LoRA (r=32) on Colab A100 | Standard Stack (finetune.py CLI surface, verified args), Architecture Patterns Pattern 3 (resumable training + HF Hub push), Common Pitfalls 4/5/6 |
| TUNE-03 | Spatial vs non-spatial task success rates are benchmarked before and after fine-tuning | Architecture Patterns Pattern 2 (seeded eval_loop), Pattern 4 (adapter reload for after-eval), Common Pitfalls 7 |
| TUNE-04 | Training metrics and evaluation results are tracked with WandB | Standard Stack (finetune.py's native WandB hooks), Architecture Patterns Pattern 3 |
</phase_requirements>

## Summary

This phase wires three already-locked pieces together: (1) a **custom** HDF5-to-RLDS converter (D-04) whose *output* must still be a genuine TFDS-loadable dataset directory because `finetune.py` loads training data via `tfds.builder(dataset_name, data_dir=data_root_dir)` — "not adapting openvla-oft's upstream builder tooling" means not using their `rlds_dataset_builder`-style `GeneratorBasedBuilder` class scaffold, not that the on-disk format can diverge from TFDS; (2) `finetune.py` itself, whose CLI surface, LoRA/resume/WandB/checkpoint behavior is now confirmed from source (see Standard Stack); and (3) the existing `eval_loop.py::run_suite` reused for both the before (zero-shot) and after (fine-tuned) benchmark runs, with a small, additive seed-plumbing change to satisfy D-08.

The single highest-risk unknown is **dataset registration**: `finetune.py --dataset_name <name>` resolves through `prismatic/vla/datasets/rlds/oxe/configs.py` (`OXE_DATASET_CONFIGS`) and `oxe/mixtures.py` (`OXE_NAMED_MIXTURES`) inside the **pip-installed** openvla-oft package — there is no CLI flag to point at an ad-hoc, unregistered dataset. Since this project installs openvla-oft with `--no-deps` from a live GitHub clone (per `01-DEBUG-HISTORY.md`), the plan MUST include a step that edits (or monkeypatches at runtime) the installed package's `configs.py`/`mixtures.py` to add a `soarm_spatial` entry, before `finetune.py` can find the dataset. This is not mentioned anywhere in CONTEXT.md and is the biggest planning gap this research closes.

A second load-bearing finding: our SOARM action space (OSC_POSE controller, 7-D: delta XYZ + delta RPY/axis-angle + gripper) maps cleanly onto OpenVLA's `ActionEncoding.EEF_POS` convention used by the official `modified_libero_rlds` reference dataset — this is a good sign the converter's action columns need no transformation, just direct copy. Our proprio (7-D: 5 joints + 2-DOF gripper), however, is **joint-space**, not the EEF-pose-based `StateEncoding.POS_EULER` the reference dataset uses — this mismatch is flagged as an open question requiring Colab-side confirmation of whether OpenVLA-OFT's single-dataset (non-OXE-mixture) training path actually enforces a specific `state_encoding`, or whether it is only used for OXE-mixture normalization and can be set permissively for a single custom dataset.

**Primary recommendation:** Build the RLDS converter as a Python module that writes real TFDS-format shards (features.json + dataset_info.json + `.tfrecord` files) via `tensorflow_datasets` APIs directly (no class-based `GeneratorBasedBuilder` scaffold) and register the dataset in the installed openvla-oft package's `oxe/configs.py`/`oxe/mixtures.py` via a small, versioned monkeypatch script that runs at the top of the training notebook before importing `finetune.py`. Call `finetune.py` with `--merge_lora_during_training False` to avoid re-saving the full 7B merged model at every checkpoint (D-01 only needs the small adapter); merge once, at eval time, inside the eval notebook.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| HDF5 -> RLDS conversion | Local script / data layer | — | Pure offline data transform; runs once, output committed to Colab disk (D-05) |
| Dataset registration (OXE configs/mixtures patch) | Training notebook bootstrap | Package layer (site-packages patch) | Must run before `finetune.py` imports the registry; lives in training-notebook setup cell, not in the RLDS converter itself |
| LoRA fine-tuning loop | Upstream package (`finetune.py`) | Training notebook (orchestration/CLI args) | D-11 — reuse upstream training loop, don't reimplement |
| Checkpoint/adapter persistence | HF Hub (external service) | Training notebook (push trigger) | D-01/D-02/D-03 |
| Training metrics (loss curves) | `finetune.py` (native `wandb.log` calls) | Training notebook (`wandb.init` args only) | finetune.py has first-class WandB integration; notebook only supplies entity/project |
| Before/after eval loop | `eval_loop.py::run_suite` (existing) | Eval notebook (env_factory, backend construction, seed list) | Reuse per CONTEXT.md reuse note; minimal seed-plumbing addition |
| Eval success-rate metrics | Eval notebook (`wandb.log` wrapper) | — | `run_suite` has no native WandB hooks; must be wrapped manually (D-13 discretion area, resolved below) |
| Adapter reload for "after" model | Eval notebook (PEFT `merge_and_unload`) | `oft_backend.py` (unchanged `predict()` contract) | Merge happens once at eval-notebook startup; `OFTBackend`'s existing interface stays untouched |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `openvla-oft` (moojink fork) | HEAD of `main` (pip install `--no-deps` from GitHub, no PyPI release; not vendored, per `01-DEBUG-HISTORY.md`) | Provides `vla-scripts/finetune.py`, `prismatic.vla.datasets.rlds` loader, PEFT-based LoRA training loop | D-11 — reuse upstream, already-tested training implementation |
| `peft` | 0.20.0 [VERIFIED: pip index versions, 2026-08-20] | LoRA adapter application (`PeftModel`, `get_peft_model`) that `finetune.py` imports directly | Standard HuggingFace library for parameter-efficient fine-tuning; `finetune.py`'s LoRA path is built on it |
| `tensorflow` | `==2.15.0` [CITED: `01-DEBUG-HISTORY.md`, this project's existing pin] | RLDS/TFDS backend — `dlimp`/`tensorflow_datasets` both require it | Already pinned project-wide for the `dlimp` fork's cp312-incompatible-elsewhere constraint; do not bump |
| `tensorflow-datasets` | 4.9.10 [VERIFIED: pip index versions, 2026-08-20] | `tfds.builder(name, data_dir=...)` — the exact call `finetune.py`'s dataset loader uses to resolve a registered RLDS dataset from disk | Confirmed via source read of `openvla/prismatic/vla/datasets/rlds/dataset.py::make_dataset_from_rlds` [CITED: github.com/openvla/openvla] |
| `dlimp` (kvablack fork) | pinned via git clone `--no-deps` [CITED: `01-DEBUG-HISTORY.md`] | `dl.DLataset` wrapper the RLDS dataloader builds on top of `tf.data` | Already required by `openvla-oft`'s eager import chain (Phase 1) — no PyPI release exists (confirmed `dlimp` is `[SLOP]` on PyPI, see Package Legitimacy Audit) |
| `huggingface_hub` | already a project dependency [CITED: `.planning/codebase/INTEGRATIONS.md`] | `create_repo`, `upload_folder` (with `run_as_future=True` for non-blocking periodic pushes), `hf_hub_download` for adapter retrieval | Already used for the base-checkpoint download pattern; reused bidirectionally now (push + pull) |
| `wandb` | already installed (part of the openvla-oft eager import chain, per `01-close` STATE.md note) | Training-loss + eval-success-rate dashboards | D-13; `finetune.py` has native `wandb.init`/`wandb.log` calls |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `accelerate` | 1.14.0 [VERIFIED: pip index versions, 2026-08-20] | Likely a transitive dependency of `peft`/`transformers`'s training utilities | Install if `finetune.py`'s import chain fails without it; not confirmed as a direct explicit import in the fetched `finetune.py` source, but `torchrun --standalone --nnodes 1 --nproc-per-node N` is the launch mechanism `finetune.py` itself expects (raw `torch.distributed`, not `accelerate launch`) |
| `draccus` | `==0.8.0` [CITED: `01-close` STATE.md note] | `finetune.py`'s `FinetuneConfig` dataclass is parsed via `draccus` (same pattern as `openvla-oft`'s inference config) | Already in the eager-chain install list from Phase 1 |
| `tensorflow_graphics` | `==2021.12.3` [CITED: `01-close` STATE.md note] | Transitive import in `prismatic`'s RLDS augmentation path | Already in the eager-chain install list from Phase 1; install `--no-deps` (OpenEXR/tf-addons unbuildable on Colab) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom TFDS-writer converter script | `rlds_dataset_builder` template repo's `GeneratorBasedBuilder` class scaffold | Rejected by D-04 — adds an unvendored, unfamiliar class hierarchy to debug; a direct `tensorflow_datasets` write is more transparent and matches this project's existing "standalone script -> data artifact" pattern (D-05) |
| Merging LoRA into base model during training (`merge_lora_during_training=True`, the `finetune.py` default) | Merge once at eval time only | Recommended override: merging at every `save_freq` checkpoint means re-saving a ~14-16GB bf16 7B model every checkpoint, wasting A100 disk I/O and time for no benefit — D-01 only needs the small PEFT adapter pushed to HF Hub |
| Editing the installed openvla-oft package's `oxe/configs.py`/`mixtures.py` directly on Colab disk | Runtime monkeypatch of `OXE_DATASET_CONFIGS`/`OXE_NAMED_MIXTURES` dicts before importing `finetune.py` | Either works; monkeypatch is preferred — it survives a `git pull`/reinstall of the openvla-oft clone without needing to reapply a file edit, and is testable/version-controlled as a small Python snippet in this repo instead of a diff against unvendored code |

**Installation (training notebook only):**
```bash
pip install peft==0.20.0 tensorflow-datasets==4.9.10 accelerate --no-deps  # verify --no-deps necessity live; peft/tfds have normal dep trees, unlike dlimp/openvla-oft
# dlimp, tensorflow_graphics, draccus, tensorflow==2.15.0 already installed per Phase 1's Block A/5b (01-DEBUG-HISTORY.md)
```

**Version verification:** `peft` (0.20.0), `tensorflow-datasets` (4.9.10), and `accelerate` (1.14.0) versions above were confirmed live via `pip index versions <pkg>` on 2026-08-20 — all current as of this research date. `openvla-oft` has no PyPI release; version is whatever `main` resolves to on install day, consistent with how this project already treats it (Phase 1).

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| `peft` | PyPI | recent release (2026-07-28) [CITED: package-legitimacy seam] | unknown (seam couldn't determine) | github.com/huggingface/peft | SUS | Flagged — planner must add `checkpoint:human-verify` before `pip install peft` (false-positive risk: this is HuggingFace's well-known LoRA library; SUS driven by "too-new"/"unknown-downloads" signals only, no `postinstall` risk) |
| `tensorflow-datasets` | PyPI | recent release (2026-05-08) | unknown | github.com/tensorflow/datasets | SUS | Flagged — planner must add `checkpoint:human-verify`; official TensorFlow org repo, low real risk |
| `accelerate` | PyPI | recent release (2026-06-11) | unknown | github.com/huggingface/accelerate | SUS | Flagged — planner must add `checkpoint:human-verify`; official HuggingFace repo, low real risk, install only if `finetune.py`'s import chain requires it |
| `dlimp` | PyPI | N/A — does not exist on PyPI | N/A | N/A (only exists as kvablack/dlimp GitHub repo) | SLOP | **Confirms existing project convention is correct** — `dlimp` must be installed via `git clone` + `pip install --no-deps` (already established in `01-close`, no new work needed; do NOT attempt `pip install dlimp`) |
| `huggingface_hub` | PyPI | recent release (2026-08-18) | unknown | github.com/huggingface/huggingface_hub | SUS | Already an existing project dependency (`.planning/codebase/INTEGRATIONS.md`) — no new checkpoint needed, listed here for completeness only |

**Packages removed due to [SLOP] verdict:** none removed from the plan — `dlimp`'s SLOP verdict on the PyPI ecosystem check simply confirms it is correctly installed via git clone, not `pip install dlimp` (this was already the project's approach; no change needed).

**Packages flagged as suspicious [SUS]:** `peft`, `tensorflow-datasets`, `accelerate` — the planner must insert a `checkpoint:human-verify` task before each `pip install` in the training notebook plan. All three are well-known, official HuggingFace/TensorFlow-org packages; the SUS verdicts are driven by the legitimacy seam's "too-new"/"unknown-downloads" heuristics (it could not fetch weekly download counts), not by any structural red flag (no suspicious `postinstall`, source repo confirmed for all three).

*`openvla-oft` itself, `dataset_statistics.json`'s producing code, and the base checkpoint `moojink/openvla-7b-oft-finetuned-libero-spatial` are pre-existing project dependencies from Phase 1/3 — not re-audited here.*

## Architecture Patterns

### System Architecture Diagram

```
[robomimic HDF5, 120 demos]
   (agentview_rgb, eye_in_hand_rgb, joint_states+gripper_states, actions)
        |
        v
[Custom RLDS Converter Script]  <-- D-04/D-05/D-06
   - reads HDF5 via h5py (mirrors normalization.py's load pattern)
   - writes genuine TFDS shards (tensorflow_datasets APIs, not GeneratorBasedBuilder)
   - output: LIBERO/libero/datasets/soarm_spatial/rlds/soarm_spatial/1.0.0/*.tfrecord + dataset_info.json
        |
        v
[Training Notebook Bootstrap]
   - monkeypatches OXE_DATASET_CONFIGS + OXE_NAMED_MIXTURES in the installed
     openvla-oft package to register "soarm_spatial"
        |
        v
[finetune.py]  (torchrun --standalone --nproc-per-node 1)  <-- D-11
   --vla_path moojink/openvla-7b-oft-finetuned-libero-spatial
   --data_root_dir .../rlds/  --dataset_name soarm_spatial
   --lora_rank 32 --use_lora True --merge_lora_during_training False
   --resume True (on relaunch after disconnect)      <-- D-10
        |            |
        |            +--> [WandB: new project]  <-- D-13, native hooks
        |
        v
[periodic + final checkpoint]
   lora_adapter/, dataset_statistics.json, action_head--*.pt, ...
        |
        v
[HF Hub private repo, timestamped]  <-- D-01/D-02/D-03
        |
        v
[Eval Notebook]  <-- D-12, separate kernel
   - hf_hub_download the adapter dir
   - base_vla = AutoModelForVision2Seq.from_pretrained(vla_path, ...)  (existing OFTBackend pattern)
   - merged = PeftModel.from_pretrained(base_vla, adapter_dir).merge_and_unload()  <-- ONE-TIME merge
   - OFTBackend-compatible wrapper around `merged`
        |
        v
[eval_loop.run_suite]  (seed-plumbed)  <-- D-07/D-08/D-09
   - BEFORE: unmodified base OFTBackend, 4 tasks x 20 episodes, fixed seed list
   - AFTER:  merged-adapter backend, SAME 4 tasks x 20 episodes, SAME seed list
        |
        v
[Success-rate summary table + WandB eval log]  <-- TUNE-03/TUNE-04
```

### Recommended Project Structure
```
LIBERO/libero/libero/datasets/
├── rlds_converter.py       # NEW — HDF5 -> TFDS shard writer (TUNE-01, D-04/D-05/D-06)
├── test_rlds_converter.py  # NEW — schema/shape/dtype unit tests (D-06)
├── oxe_register.py         # NEW — monkeypatch helper: registers "soarm_spatial" into
│                            #       the installed openvla-oft package's OXE registry
├── hdf5_writer.py          # existing (Phase 4) — unchanged, converter's input source
└── normalization.py        # existing (Phase 4) — dataset_statistics.json producer;
                             # NOTE: finetune.py computes its OWN stats too — see Pitfall 6

LIBERO/libero/libero/vla/
├── eval_loop.py             # MODIFY — add optional per-episode seed plumbing (D-08)
├── test_eval_loop.py        # MODIFY — add seed-determinism regression test
├── oft_backend.py           # unchanged — predict() contract stays the same
└── adapter_backend.py       # NEW (or extend oft_backend.py) — loads base + merges a
                              # downloaded PEFT adapter, then delegates to the same
                              # predict() logic as OFTBackend (D-01 reuse target)

LIBERO/notebooks/
├── 06a-finetune.ipynb   # NEW — training notebook: RLDS convert (if not already run) +
│                         #        finetune.py + HF Hub push (D-12)
└── 06b-eval.ipynb        # NEW — eval notebook: adapter download + before/after run_suite
```

### Pattern 1: RLDS conversion as a direct TFDS write (not a `GeneratorBasedBuilder`)

**What:** Instead of subclassing `tfds.core.GeneratorBasedBuilder` (the `rlds_dataset_builder` template's approach, rejected by D-04), write episodes directly using `tensorflow_datasets`'s lower-level writer APIs (`tfds.core.SequentialWriter` / manual `tf.io.TFRecordWriter` + hand-written `features.json`/`dataset_info.json`) so the converter is a plain, testable function: `hdf5_to_rlds(hdf5_paths, out_dir, dataset_name="soarm_spatial") -> int`.

**Why this still has to be genuine TFDS output:** `finetune.py`'s dataloader calls `tfds.builder(name, data_dir=data_dir)` [CITED: `openvla/prismatic/vla/datasets/rlds/dataset.py::make_dataset_from_rlds`, github.com/openvla/openvla] — this is a hard requirement of the on-disk format, independent of which code produced it. D-04's "not adapting their builder tooling" is about avoiding the *class scaffold*, not about producing a non-TFDS format.

**Feature schema to target** (mirrors the official `openvla/modified_libero_rlds` reference dataset [CITED: huggingface.co/datasets/openvla/modified_libero_rlds/discussions/5], adapted to this project's actual keys):
```python
# Source: this project's normalization.py + hdf5_writer.py schema, cross-checked
# against openvla/modified_libero_rlds's documented feature shapes.
step_features = {
    "observation": {
        "agentview_rgb": tf.uint8,      # (H, W, 3) -- matches hdf5_writer.py OBS_KEY_MAPPING
        "eye_in_hand_rgb": tf.uint8,    # (H, W, 3)
        "state": tf.float32,            # (7,) -- joint_states (5) + gripper_states (2), OUR schema
    },
    "action": tf.float32,               # (7,) -- OSC_POSE: delta XYZ(3)+delta RPY(3)+gripper(1)
    "language_instruction": tf.string,
    "discount": tf.float32,
    "reward": tf.float32,
    "is_first": tf.bool,
    "is_last": tf.bool,
    "is_terminal": tf.bool,
}
```
**When to use:** This is the only conversion approach compatible with D-04's constraint while still satisfying `finetune.py`'s loader.

### Pattern 2: Dataset registration via runtime monkeypatch, not a source edit

**What:** Because `openvla-oft` is installed `--no-deps` from a live GitHub clone (not vendored, per `01-DEBUG-HISTORY.md`), the training notebook must register the dataset in the *installed* package's registry before importing `finetune.py`'s config machinery. Prefer patching the in-memory dicts over editing the cloned repo's files on disk, since a monkeypatch survives repo re-clones/pulls and is testable as ordinary Python:

```python
# Source: pattern derived from openvla/prismatic/vla/datasets/rlds/oxe/configs.py
# and mixtures.py structure (CITED: github.com/openvla/openvla, fetched this session)
from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS, StateEncoding, ActionEncoding
from prismatic.vla.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES

OXE_DATASET_CONFIGS["soarm_spatial"] = {
    "image_obs_keys": {"primary": "agentview_rgb", "secondary": None, "wrist": "eye_in_hand_rgb"},
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "state_obs_keys": ["state", None, None],  # ASSUMED joint-space is acceptable; verify on Colab (Open Question 1)
    "state_encoding": StateEncoding.POS_EULER,  # placeholder -- confirm correct enum for a 7-D joint+gripper state
    "action_encoding": ActionEncoding.EEF_POS,   # matches OSC_POSE's delta-pose+gripper 7-D action space
}
OXE_NAMED_MIXTURES["soarm_spatial"] = [("soarm_spatial", 1.0)]
```
**When to use:** Run once at the top of the training notebook, before any `finetune.py`/`prismatic` training-config import. Must be re-applied every session (monkeypatch is in-memory only, not persisted across kernel restarts) — document this explicitly in the notebook.

### Pattern 3: Resumable `finetune.py` invocation with lightweight checkpoint pushes

**What:** Call `finetune.py` via `torchrun`, with `--merge_lora_during_training False` (override the library default of `True`) to keep checkpoint saves cheap (adapter-only, tens of MB, per D-01's framing), and with `--resume True --resume_step <N>` on any relaunch after a Colab disconnect (D-10):
```bash
# Source: CITED github.com/moojink/openvla-oft LIBERO.md (fetched this session),
# adapted with lora_rank/merge/resume flags per this phase's D-00/D-01/D-10
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path moojink/openvla-7b-oft-finetuned-libero-spatial \
  --data_root_dir LIBERO/libero/datasets/soarm_spatial/rlds/ \
  --dataset_name soarm_spatial \
  --run_root_dir /content/soarm_ft_runs/ \
  --use_l1_regression True --use_diffusion False --use_film False \
  --num_images_in_input 2 --use_proprio True \
  --lora_rank 32 --use_lora True --merge_lora_during_training False \
  --save_freq 1000 --save_latest_checkpoint_only False \
  --batch_size 8 --learning_rate 5e-4 \
  --wandb_entity <entity> --wandb_project soarm-finetune-eval \
  --resume True --resume_step <last_completed_step>   # only on relaunch
```
After each `save_freq` checkpoint lands on local disk, a notebook-side callback/polling loop (finetune.py itself does not push to HF Hub) uploads the new checkpoint directory:
```python
from huggingface_hub import create_repo, upload_folder
repo_id = f"<user>/soarm-oft-lora-{run_timestamp}"
create_repo(repo_id, private=True, exist_ok=True)
upload_folder(repo_id=repo_id, folder_path=str(checkpoint_dir), run_as_future=True)  # non-blocking
```
**When to use:** D-01/D-02/D-03/D-10 — HF Hub push is NOT a `finetune.py` built-in; it must be a notebook-level wrapper watching `run_root_dir` for new `*_chkpt` directories.

### Pattern 4: Reload the fine-tuned adapter for the "after" eval without touching `OFTBackend`

**What:** Extend (not replace) `OFTBackend`'s loading path so the "after" benchmark reuses the exact same `predict()` contract `eval_loop.run_suite` already drives:
```python
# Extends oft_backend.py's proven loading pattern (prismatic guard, bf16 load,
# set_num_images_in_input(2), norm_stats overlay) with a PEFT merge step.
from huggingface_hub import snapshot_download
from peft import PeftModel

class FinetunedOFTBackend(OFTBackend):
    def __init__(self, adapter_repo_id: str, checkpoint: str = CHECKPOINT, device: str = "cuda"):
        super().__init__(checkpoint=checkpoint, device=device)
        adapter_dir = snapshot_download(adapter_repo_id)
        self.model = PeftModel.from_pretrained(self.model, adapter_dir).merge_and_unload()
        # re-apply the num_images_in_input(2) call -- merge_and_unload() may return
        # a fresh module tree; verify this survives the merge on Colab (Open Question 2)
        self.model.vision_backbone.set_num_images_in_input(2)
```
**When to use:** Eval notebook only. Keeps `predict()`'s contract, `eval_loop.py`'s reuse, and the norm_stats overlay logic completely unchanged — only the model-loading step differs from the "before" run.

### Pattern 5: Seed-plumbed `run_suite` for D-08

**What:** `env_wrapper.py::ControlEnv.seed(seed)` already exists and delegates to robosuite's underlying `env.seed()` [VERIFIED: read `LIBERO/libero/libero/envs/env_wrapper.py` lines 133-134, this session] — object-placement randomization inside `reset()` is driven by this seeded RNG. `run_episode`/`run_suite` currently never call it. Minimal additive change:
```python
# eval_loop.py -- additive change, backward compatible (seed=None -> old behavior)
def run_episode(env, backend, language, video_path, max_steps=MAX_STEPS_DEFAULT,
                 camera_name=..., agentview_camera_name=..., seed=None):
    if seed is not None:
        env.seed(seed)
    obs = env.reset()
    ...

def run_suite(env_factory, backend, tasks, language_map, episodes_per_task, video_dir,
              max_steps=MAX_STEPS_DEFAULT, episode_seeds=None):
    # episode_seeds: optional list[int] of length episodes_per_task, reused
    # identically across the before-run and after-run call sites (D-08).
    ...
    for ep_idx in range(episodes_per_task):
        seed = episode_seeds[ep_idx] if episode_seeds is not None else None
        result = run_episode(env, backend, language, video_path, max_steps=max_steps, seed=seed)
```
**When to use:** Both before and after benchmark runs, called with the exact same `episode_seeds` list (generate once, e.g. `list(range(20))`, and hardcode/persist it so a re-run of only the "after" pass — e.g. after a bug fix — still lines up with the original "before" pass).

### Anti-Patterns to Avoid
- **Re-running `finetune.py`'s dataset-statistics computation as the only source of truth without reconciling with Phase 4's `dataset_statistics.json`:** these are two independently-computed stats files over conceptually the same data but through different code paths (raw-HDF5 numpy stats vs. post-RLDS-conversion TFDS stats) — see Pitfall 6.
- **Editing the openvla-oft GitHub clone's `oxe/configs.py` file in place on Colab disk:** works but is invisible to this repo's git history and gets silently discarded on the next `git clone`/reinstall — prefer the runtime monkeypatch (Pattern 2).
- **Leaving `merge_lora_during_training` at its library default (`True`):** silently multiplies checkpoint save time/disk by re-serializing a ~14-16GB merged model at every `save_freq` step, on top of the actual training compute — directly works against the Colab A100 session-time budget this project has repeatedly hit limits on.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|--------------|-----|
| LoRA training loop, optimizer, gradient accumulation, loss computation | Custom PyTorch training loop | `openvla-oft`'s `finetune.py` (D-11) | Already implements action-chunking, L1-regression action head, PEFT LoRA wiring, and multi-image (`num_images_in_input=2`) support this project's dual-camera Phase 5 work depends on |
| Episode-loop / PASS-FAIL polling / video recording / summary tables | Custom eval driver | `eval_loop.py::run_suite`/`run_episode` (existing, this repo) | Already proven in Phase 3/5; only needs the additive seed param, not a rewrite |
| RLDS/TFDS on-disk format (tfrecord sharding, dataset_info.json schema) | Hand-rolled binary format "close enough" to RLDS | `tensorflow_datasets` writer APIs (still, even inside a custom script) | `finetune.py`'s loader calls `tfds.builder(...)` directly — any format deviation is a load-time crash, not a training-quality issue |
| LoRA adapter merge logic | Manual weight-matrix math (`W + BA`) | `peft.PeftModel.from_pretrained(...).merge_and_unload()` | Standard, tested HuggingFace API; hand-rolling risks subtle scaling/dtype bugs that silently degrade the "after" eval without an obvious crash |

**Key insight:** every "don't hand-roll" item in this phase already has a first-class implementation either upstream (`finetune.py`, `peft`) or in this repo (`eval_loop.py`) — the actual engineering surface of this phase is the *glue* (dataset registration, HF Hub push scheduling, seed plumbing, adapter reload), not any of the ML internals.

## Common Pitfalls

### Pitfall 1: `finetune.py --dataset_name` silently fails to find an unregistered dataset
**What goes wrong:** `tfds.builder(name, data_dir=data_dir)` raises if the dataset directory doesn't have valid TFDS metadata at the resolved path, and separately, `prismatic`'s own config resolution raises a `KeyError`/similar if `dataset_name` isn't in `OXE_DATASET_CONFIGS`/`OXE_NAMED_MIXTURES` before that.
**Why it happens:** The registration step (Pattern 2) is invisible in `finetune.py`'s own CLI help — it's a code-level requirement, not a flag.
**How to avoid:** The training notebook must run the registration monkeypatch in the SAME Python process, before importing anything from `vla-scripts/finetune.py`'s config module. Add a smoke-test cell that resolves `OXE_DATASET_CONFIGS["soarm_spatial"]` and prints it before launching the full `torchrun` command.
**Warning signs:** `KeyError: 'soarm_spatial'` or a TFDS "dataset not found" error immediately at training start (before any GPU compute happens) — cheap to catch early, expensive if it burns A100 minutes first.

### Pitfall 2: Converter output isn't real TFDS, just tfrecord-shaped files
**What goes wrong:** Writing raw `tf.io.TFRecordWriter` output without a matching `dataset_info.json`/`features.json` produces files that `tfds.builder(..., data_dir=...)` cannot discover or parse.
**Why it happens:** D-04's "custom converter" framing can be misread as "any binary format that resembles RLDS episodes."
**How to avoid:** Use `tensorflow_datasets`'s `DatasetInfo`/`FeaturesDict` classes to write the metadata sidecar files even when hand-writing the tfrecord shards — verify with `tfds.builder("soarm_spatial", data_dir=...).info` round-trips before ever invoking `finetune.py`.
**Warning signs:** `tfds.builder(...)` raises `DatasetNotFoundError` or a schema-mismatch error at training-loop start.

### Pitfall 3: Wrong action/state semantic mapping silently trains a working-looking-but-wrong policy
**What goes wrong:** If the converter mislabels OSC_POSE's rotation delta convention (axis-angle vs. Euler RPY) against `ActionEncoding.EEF_POS`'s documented "delta RPY," training completes without error but the learned policy's rotation commands are subtly wrong.
**Why it happens:** robosuite's OSC_POSE controller's exact rotation delta convention is [ASSUMED] in this research, not verified against robosuite 1.4.0's actual controller source this session.
**How to avoid:** Before running the full training job, verify (unit test, per D-06) that a known-good demo's action rows, when the RLDS converter's dtype/shape checks pass, also spot-check that rotation-delta magnitudes are physically plausible (small per-step deltas, not radians-vs-degrees off by a large factor).
**Warning signs:** Training loss converges normally but the after-benchmark's success rate doesn't improve over the before-benchmark despite loss looking healthy — a classic "trained on subtly wrong labels" signature.

### Pitfall 4: `peft`/`tensorflow-datasets`/`accelerate` install collides with Colab's protobuf pin dance
**What goes wrong:** Phase 1 already documented that installing `tensorflow_graphics`/`openvla-oft`'s eager chain drags `protobuf` below what Colab's `tensorflow_metadata` gencode needs, requiring an explicit `pip install -U protobuf` as the LAST pip operation in the cell [CITED: `01-close` STATE.md note]. Adding `tensorflow-datasets` (another `protobuf` consumer) to this phase's training notebook re-introduces the same class of risk.
**Why it happens:** Colab's system-installed TF-adjacent packages (`tensorflow_metadata`, etc.) and this project's pinned `tensorflow==2.15.0` want different `protobuf` ranges.
**How to avoid:** Install `tensorflow-datasets`/`peft`/`accelerate` in the SAME cell/order-position as the rest of the openvla-oft eager chain, and re-run `pip install -U protobuf` as the final step, mirroring Phase 1's Block A/5b pattern exactly.
**Warning signs:** `ImportError`/`TypeError` mentioning protobuf descriptor version mismatches on first import of `tensorflow_datasets` or `prismatic`.

### Pitfall 5: Colab session disconnect mid-training loses more than expected if `save_latest_checkpoint_only=True`
**What goes wrong:** With `save_latest_checkpoint_only=True`, `finetune.py` overwrites the single checkpoint directory in place — if a disconnect happens mid-write, the checkpoint can be corrupted/partial with no earlier fallback.
**Why it happens:** The overwrite-in-place mode trades disk usage for safety.
**How to avoid:** Use `--save_latest_checkpoint_only False` (keep timestamped `--{step}_chkpt` directories) so a corrupted latest write still leaves the previous complete checkpoint on HF Hub to resume from — directly supports D-02/D-10.
**Warning signs:** `--resume` fails to load a checkpoint that looks present but is missing files (partial write from a mid-save disconnect).

### Pitfall 6: Two different `dataset_statistics.json` files can silently diverge
**What goes wrong:** Phase 4's `normalization.py` already produced `LIBERO/libero/datasets/soarm_spatial/dataset_statistics.json` computed directly from the raw HDF5 (used today by `oft_backend.py`'s zero-shot norm_stats overlay). `finetune.py` separately computes and saves its OWN `dataset_statistics.json` (via `save_dataset_statistics(train_dataset.dataset_statistics, run_dir)`) computed from the POST-CONVERSION RLDS data — if the RLDS converter introduced any resampling/dtype-cast difference, these two files can disagree.
**Why it happens:** Two independent computation paths over what should be the same underlying numbers.
**How to avoid:** The eval notebook's "after" backend MUST load the checkpoint's OWN `dataset_statistics.json` (pushed to HF Hub alongside the adapter) — NOT reuse Phase 4's file, even though they're expected to be numerically close. Add an explicit assertion/log comparing the two at eval-notebook startup so any silent drift is visible, not just assumed away.
**Warning signs:** "After" eval success rate is unexpectedly worse than "before" despite a healthy training loss curve — a normalization mismatch is a common root cause for this exact symptom.

### Pitfall 7: `run_suite`'s `env_factory` creates ONE env reused across all 20 episodes of a task
**What goes wrong:** Combined with D-08's identical-seed requirement, reusing a single `env` instance across episodes means `env.seed(seed)` must be called freshly before EACH `reset()`, not once before the loop — otherwise only episode 0 gets the intended seed and subsequent episodes drift from whatever RNG state `reset()` left behind.
**Why it happens:** `run_suite`'s current per-task env-reuse pattern (one `env_factory(task)` call, then N `run_episode` calls against it) is efficient but easy to seed incorrectly if the seed call is hoisted outside the per-episode loop.
**How to avoid:** Pattern 5's plumbing calls `env.seed(seed)` inside `run_episode`, per-episode, which is correct — verify this placement is preserved in the actual implementation (test: run the same `episode_seeds` list twice against a zero-shot backend and assert identical initial object positions across both runs, before spending eval budget on the real before/after comparison).
**Warning signs:** Re-running the "before" benchmark twice with the same seed list produces different summary-table numbers.

## Code Examples

Verified/cited patterns from official sources (already embedded above in Architecture Patterns; consolidated pointers below):

- `finetune.py` CLI invocation for LIBERO — Pattern 3, [CITED: raw.githubusercontent.com/moojink/openvla-oft/main/LIBERO.md]
- `tfds.builder(name, data_dir=data_dir)` dataset resolution — Pattern 1, [CITED: raw.githubusercontent.com/openvla/openvla/main/prismatic/vla/datasets/rlds/dataset.py]
- `OXE_DATASET_CONFIGS` entry structure (`taco_play` reference) — Pattern 2, [CITED: raw.githubusercontent.com/openvla/openvla/main/prismatic/vla/datasets/rlds/oxe/configs.py]
- `huggingface_hub.upload_folder(..., run_as_future=True)` for non-blocking periodic pushes — Pattern 3, [CITED: huggingface.co/docs/huggingface_hub/en/guides/upload]
- `PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()` — Pattern 4, [CITED: finetune.py's own merge-during-training code path, raw.githubusercontent.com/moojink/openvla-oft/main/vla-scripts/finetune.py]
- `env.seed(seed)` delegation — Pattern 5, [VERIFIED: read directly from `LIBERO/libero/libero/envs/env_wrapper.py` lines 133-134 this session]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Original OpenVLA fine-tuning (full backbone or bare LoRA, discrete action tokens via `predict_action`, single forward pass per action) | OpenVLA-**OFT** (Optimizing Fine-Tuning): parallel decoding, continuous L1-regression or diffusion action head, action chunking (8-step chunks), optional multi-image + proprio input | "OFT recipe" recommended by the OpenVLA team as of March 2026 per this session's search results [CITED: search result summary, roboticscenter.ai] | This project already committed to OFT in Phase 1/3 (checkpoint `moojink/openvla-7b-oft-finetuned-libero-spatial`); this phase's `finetune.py` flags (`use_l1_regression`, `num_images_in_input=2`, `use_proprio`) are the OFT-specific surface, not the original openvla/openvla repo's plain `finetune.py` |

**Deprecated/outdated:**
- Discrete action-token prediction (original OpenVLA): superseded by OFT's continuous action head for this project's checkpoint family — do not reference `openvla/openvla`'s (non-OFT) `finetune.py` CLI flags, they differ (no `use_l1_regression`/`num_images_in_input`/`use_proprio`).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|----------------|
| A1 | `state_encoding: StateEncoding.POS_EULER` is a safe placeholder for our joint-space 7-D proprio in the OXE config registration | Architecture Patterns Pattern 2 | If `finetune.py`'s single-dataset (non-mixture) training path actually consumes `state_encoding` beyond bookkeeping (e.g. to select a specific state-processing branch), a wrong enum could silently corrupt or truncate the proprio input the model trains on |
| A2 | OSC_POSE's rotation-delta convention matches `ActionEncoding.EEF_POS`'s "delta RPY" exactly (not axis-angle, not a different magnitude/unit convention) | Common Pitfalls, Pitfall 3 | If the convention differs, the model trains on subtly mislabeled rotation targets — a "trains fine, doesn't improve success rate" failure mode that's hard to diagnose after the fact |
| A3 | `accelerate` is required by `finetune.py`'s import chain (not confirmed via direct source read of imports this session, only inferred from `peft`/`transformers`'s typical dependency footprint) | Standard Stack, Supporting table | Wasted install step if not needed; low risk either way — installing an unused package is harmless, just adds Colab setup time |
| A4 | `merge_and_unload()`'s resulting model tree still exposes `.vision_backbone.set_num_images_in_input` in the same location as the pre-merge model (Pattern 4) | Architecture Patterns Pattern 4 | If the merged model's module tree differs, `FinetunedOFTBackend`'s re-application of `set_num_images_in_input(2)` would `AttributeError` at eval-notebook startup — cheap to catch (crashes immediately, not a silent failure) |
| A5 | `finetune.py`'s WandB integration logs eval-style metrics only for its own internal `use_val_set` validation pass, not for this phase's separate before/after `run_suite` benchmark — the eval notebook must wrap its OWN `wandb.log` calls for TUNE-03/TUNE-04's success-rate numbers | Architectural Responsibility Map, Summary | If assumed otherwise, the planner might skip building the eval-notebook WandB wrapper, leaving TUNE-04's "evaluation results... tracked with WandB" requirement unmet for the benchmark numbers specifically (only training-loss curves would appear) |

**If this table is empty:** N/A — see entries above; all five require Colab-side or direct-source confirmation before being treated as locked implementation facts.

## Open Questions

1. **Does `finetune.py`'s single-dataset (non-OXE-mixture) code path actually branch on `state_encoding`, or is it only consumed for cross-dataset OXE-mixture normalization bookkeeping?**
   - What we know: `OXE_DATASET_CONFIGS` entries always include a `state_encoding`/`action_encoding` field [CITED: configs.py structure, fetched this session].
   - What's unclear: Whether a wrong-but-structurally-valid enum for a single custom (non-mixed) dataset silently degrades training, or is inert bookkeeping.
   - Recommendation: Add a Colab-side smoke test (train for a handful of steps, inspect a sampled batch's state tensor values against known ground truth) before committing to a full training run; treat this as the very first Colab validation gate in the plan.

2. **Is `accelerate` actually imported by `finetune.py`, and if so, is any specific version pin required?**
   - What we know: `torchrun --standalone` is the documented launch command (raw `torch.distributed`, not `accelerate launch`).
   - What's unclear: Whether `peft`'s internals or `transformers`'s trainer utilities pull in `accelerate` as a hard runtime dependency even without `accelerate launch`.
   - Recommendation: Attempt the training notebook's dependency install without pinning `accelerate` first; only add it if an `ImportError` surfaces, to avoid an unnecessary Package Legitimacy checkpoint for an unused package.

3. **What is `run_root_dir`/checkpoint disk footprint over a full training run, and does it fit Colab A100's local disk alongside the RLDS dataset and base model cache?**
   - What we know: Setting `merge_lora_during_training=False` avoids the ~14-16GB-per-checkpoint full-model re-save (Pattern 3's key optimization).
   - What's unclear: Exact adapter-only + action-head + optimizer-state checkpoint size at `save_freq=1000`, and how many checkpoints accumulate with `save_latest_checkpoint_only=False` before HF Hub pushes + local cleanup are needed.
   - Recommendation: Plan a Colab-side disk-usage check after the first checkpoint save, before committing to the full `max_steps` run; consider a local-disk cleanup step after each successful HF Hub push (keep last 2 locally, prune older).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Colab A100 GPU | TUNE-02 (LoRA fine-tuning), locked by roadmap | Runtime-only, not verifiable locally | — | None — A100 (not T4) is a hard requirement already locked at the roadmap stage for this phase |
| `peft` | TUNE-02 | Not installed locally (no local GPU/torch stack per project convention) | 0.20.0 on PyPI [VERIFIED] | None needed — install on Colab |
| `tensorflow==2.15.0` | TUNE-01 (RLDS/TFDS) | Already pinned project-wide (Phase 1) | 2.15.0 [CITED] | None needed |
| `tensorflow-datasets` | TUNE-01 | Not yet installed anywhere in this project | 4.9.10 on PyPI [VERIFIED] | None needed — install on Colab |
| HF Hub write access (private repo creation) | D-01/D-02/D-03 | Existing HF token bootstrap pattern (Drive file -> getpass) reused [CITED: STATE.md 01-close note] | — | None needed — reuses proven Phase 1 auth pattern |
| WandB API key | TUNE-04 | Existing `WANDB_API_KEY` env var pattern [CITED: `.planning/codebase/INTEGRATIONS.md`] | — | None needed |

**Missing dependencies with no fallback:**
- Colab A100 session itself — this phase cannot be verified locally at all (matches every prior phase's Environment Availability finding in this project).

**Missing dependencies with fallback:**
- None identified beyond the A100 requirement itself.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest [VERIFIED: `LIBERO/libero/libero/conftest.py` exists; `test_*.py` files present throughout `LIBERO/libero/libero/`, this session] |
| Config file | `LIBERO/libero/libero/conftest.py` (fixtures only; no dedicated `pytest.ini` found) |
| Quick run command | `pytest LIBERO/libero/libero/datasets/test_rlds_converter.py -x` (new, this phase) |
| Full suite command | `pytest LIBERO/libero/libero/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|---------------------|-------------|
| TUNE-01 | HDF5 -> RLDS conversion produces schema/shape/dtype-correct TFDS output | unit | `pytest LIBERO/libero/libero/datasets/test_rlds_converter.py -x` | ❌ Wave 0 |
| TUNE-01 | Converted dataset is genuinely `tfds.builder`-loadable | integration (Colab-only, needs `tensorflow_datasets`) | manual Colab cell + assertion | ❌ Wave 0 — flag as Colab-only in plan, no local automated equivalent |
| TUNE-02 | `finetune.py` runs to completion without crashing on Colab A100 | manual-only (Colab GPU required) | Colab notebook run + exit-code/log check | N/A — matches this project's existing convention for GPU-only-verifiable steps (`oft_backend.py`'s own docstring precedent) |
| TUNE-02 | Resume-from-checkpoint actually resumes (not restarts from step 0) | manual-only (Colab GPU required, needs a real disconnect-simulation) | Colab notebook: kill+relaunch mid-run, check WandB step continuity | N/A — Colab-only |
| TUNE-03 | Seeded before/after `run_suite` produces byte-identical initial object placements across two runs with the same seed list | unit/integration | `pytest LIBERO/libero/libero/vla/test_eval_loop.py -x -k seed` | ❌ Wave 0 — new test, feasible locally since it only exercises `env.seed()`+`reset()`, no VLA inference needed |
| TUNE-04 | WandB dashboard shows both training-loss curves AND eval success-rate results | manual-only (requires live WandB run) | Colab notebook + manual dashboard check | N/A — Colab-only, matches D-13's own framing |

### Sampling Rate
- **Per task commit:** `pytest LIBERO/libero/libero/datasets/test_rlds_converter.py -x` and `pytest LIBERO/libero/libero/vla/test_eval_loop.py -x -k seed` (whichever module changed)
- **Per wave merge:** `pytest LIBERO/libero/libero/ -x` (full local suite — does NOT cover Colab-only GPU behaviors)
- **Phase gate:** Full local suite green AND a live Colab run of both notebooks (training completes, eval before/after tables populate, WandB dashboard shows both curve types) before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `LIBERO/libero/libero/datasets/test_rlds_converter.py` — covers TUNE-01 (schema/shape/dtype)
- [ ] `LIBERO/libero/libero/vla/test_eval_loop.py` seed-determinism additions — covers TUNE-03's D-08 requirement, locally testable without GPU
- [ ] No new fixture files needed — existing `conftest.py` + this project's existing HDF5 test fixture pattern (Phase 4) is reusable for RLDS converter unit tests
- [ ] Framework install: none — pytest already used project-wide

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|----------------|---------|-------------------|
| V2 Authentication | No | Research/offline pipeline, no user-facing auth surface |
| V3 Session Management | No | N/A |
| V4 Access Control | No | Single-operator research pipeline |
| V5 Input Validation | Yes | RLDS converter must validate HDF5 schema (shape/dtype/key presence) before writing TFDS output — already partially covered by D-06's mandated unit tests; extend to fail loudly (not silently pad/truncate) on schema mismatch |
| V6 Cryptography | No — but secrets handling applies | HF token and WandB API key are already handled via the existing project pattern (Drive-file bootstrap -> getpass / env var, never hardcoded in notebook source) [CITED: STATE.md 01-close, `260802-itm` quick task] — this phase reuses that pattern for HF Hub WRITE access (new: previous phases only used HF Hub for read/download) |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|------------------------|
| HF Hub token leakage into notebook source/git history (now a WRITE-capable token, higher blast radius than the prior read-only download usage) | Information Disclosure | Reuse the existing `getpass()`-only pattern (never `userdata.get()`-persisted, never hardcoded) — explicitly re-verify this phase's training notebook follows the same convention as `260802-itm`, since a write-capable leaked token could push to or delete HF Hub repos |
| Malformed/corrupted HDF5 input silently producing a garbage RLDS dataset that trains "successfully" on wrong data | Tampering (of training data integrity, not adversarial) | D-06's unit tests + the converter's fail-loud validation (V5 above) — this project's existing convention (`normalization.py`'s "load failure raises before any partial file is written" pattern) should be mirrored exactly in the RLDS converter |
| Public-by-default HF Hub repo accidentally exposing SOARM training data/checkpoints | Information Disclosure | D-01 already locks `private=True` on `create_repo` — verify explicitly in code/tests, not just assumed from the CLI call, since `huggingface_hub`'s `create_repo` defaults to public if the flag is omitted |

## Sources

### Primary (HIGH confidence)
- Direct read of this project's own source: `LIBERO/libero/libero/vla/eval_loop.py`, `oft_backend.py`, `interface.py`, `env_wrapper.py`, `LIBERO/libero/libero/datasets/hdf5_writer.py`, `normalization.py` [VERIFIED: Read tool, this session]
- `pip index versions peft / tensorflow-datasets / accelerate` [VERIFIED: local pip registry query, this session, 2026-08-20]
- `gsd-tools query package-legitimacy check --ecosystem pypi` for `peft`, `tensorflow-datasets`, `dlimp`, `accelerate`, `huggingface_hub` [VERIFIED: seam tool output, this session]

### Secondary (MEDIUM confidence)
- `raw.githubusercontent.com/moojink/openvla-oft/main/vla-scripts/finetune.py` — `FinetuneConfig` fields, WandB init/log calls, checkpoint save/resume logic, LoRA merge-during-training code path [CITED: fetched this session]
- `raw.githubusercontent.com/moojink/openvla-oft/main/LIBERO.md` — exact `finetune.py` CLI invocation example for LIBERO fine-tuning [CITED: fetched this session]
- `raw.githubusercontent.com/openvla/openvla/main/prismatic/vla/datasets/rlds/dataset.py` — `tfds.builder(name, data_dir=data_dir)` resolution mechanism [CITED: fetched this session]
- `raw.githubusercontent.com/openvla/openvla/main/prismatic/vla/datasets/rlds/oxe/configs.py` — `OXE_DATASET_CONFIGS` entry structure, `StateEncoding`/`ActionEncoding` enums [CITED: fetched this session]
- `raw.githubusercontent.com/kpertsch/rlds_dataset_builder/main/README.md` — TFDS feature-schema conventions (used as a cross-reference only, not adopted per D-04) [CITED: fetched this session]
- `huggingface.co/datasets/openvla/modified_libero_rlds/discussions/5` — reference RLDS feature schema (8-D state, 7-D action) for the official LIBERO fine-tuning dataset [CITED: fetched this session]
- `huggingface.co/docs/huggingface_hub/en/guides/upload` — `upload_folder(..., run_as_future=True)` non-blocking push pattern [CITED: fetched this session]
- This project's own `.planning/phases/01-colab-environment-setup/01-DEBUG-HISTORY.md`, `STATE.md` accumulated-context notes — dlimp/protobuf/eager-chain install order, HF token bootstrap pattern [CITED: read this session]

### Tertiary (LOW confidence)
- WebSearch summaries (not directly fetched source) regarding OFT recipe recommendation timing and general RLDS/OXE registration overview — used only to orient the deeper source-fetch queries above, not cited as standalone facts in this document

## Metadata

**Confidence breakdown:**
- Standard stack (finetune.py CLI surface): MEDIUM — CITED from live GitHub source fetch this session, not executed
- Architecture (RLDS registration + converter design): MEDIUM — the `tfds.builder` requirement is VERIFIED via source read; the exact OXE registration enum values (`state_encoding` for a joint-space proprio) are an open question requiring Colab confirmation
- Pitfalls: MEDIUM-HIGH — several are directly traceable to this project's own documented Phase 1 history (protobuf, dlimp, HF token pattern), which is HIGH confidence; the RLDS-specific pitfalls are MEDIUM (reasoned from source, not yet executed)

**Research date:** 2026-08-20
**Valid until:** 7 days — this domain (openvla-oft `main` branch, no pinned release) can change upstream at any time; re-verify `finetune.py`'s CLI surface against the live repo immediately before planning execution if more than a few days have passed
