---
status: testing
phase: 06-fine-tuning-evaluation
source: [06-VERIFICATION.md]
started: 2026-08-23T08:15:00.000Z
updated: 2026-08-29T06:30:00.000Z
---

## Current Test

number: 4
name: Run 06b-eval.ipynb end-to-end (before/after seeded benchmark)
expected: |
  Paste the trained run's HF_ADAPTER_REPO_ID (vansh-fyi/soarm-oft-lora-20260829-052948) into
  06b-eval.ipynb's ADAPTER_REPO_ID cell and run end-to-end on Colab A100/T4.
awaiting: user response

## Tests

### 1. Run 06a-finetune.ipynb end-to-end on Colab A100 (RLDS conversion + finetune.py completion)
expected: |
  Block A install, restart, Block B bootstrap, RLDS-conversion-if-needed cell, OXE registration
  smoke-test cell, HF Hub (write) + WandB auth, then the finetune.py torchrun cell
  (--lora_rank 32 --use_lora True --merge_lora_during_training False --save_latest_checkpoint_only False)
  all complete without error.
result: pass
note: |
  Eight blockers hit and resolved across this retry round (Colab Python 3.13 runtime drift,
  gitignored dataset never reaching Colab, absolute-path portability in rlds_converter.py,
  incorrect TFDS SequentialWriter API usage, OXE registration invisible to torchrun's
  subprocess + missing depth_obs_keys, torchrun's c10d libuv segfault on Python 3.12+,
  peft/transformers/diffusers three-way version conflict, missing json-numpy dep, missing
  distributed env vars for finetune.py's raw dist.barrier() calls -- full diagnosis and fixes
  in Gaps section below, all committed to origin/main).
  Training ran successfully to ~2200 steps (WandB run visible, loss curves declining and
  plateauing ~0.15-0.2, no crash). User made the judgment call to manually stop at that point
  rather than let it run toward the untouched finetune.py default of max_steps=200,000 (vastly
  oversized for a 120-episode single-task dataset; openvla-oft's own docs recommend stopping
  once L1 loss plateaus, not a fixed step target). Kernel was restarted after stopping;
  checkpoints (1000_chkpt, 2000_chkpt) were confirmed present on disk and manually pushed to
  the private HF Hub repo (HF_ADAPTER_REPO_ID = vansh-fyi/soarm-oft-lora-20260829-052948) via a
  reconstructed cell since the run-identity cell would have generated a fresh, disconnected
  timestamp on re-run.

### 2. Checkpoint push + resume-from-disconnect
expected: |
  At least one LoRA checkpoint pushed to the private HF Hub repo during/after training
  (HfApi().repo_info(HF_ADAPTER_REPO_ID).private is True, repo visible under the researcher's
  account). After simulating a disconnect (interrupt kernel mid-training) and relaunching the
  torchrun cell with RESUME=True/RESUME_STEP=<N>, the WandB run's step counter continues from
  <N> rather than resetting to 0.
result: issue
reported: |
  Checkpoint-push half CONFIRMED via HfApi().list_repo_files(): both 1000_chkpt and 2000_chkpt
  uploaded to the private repo vansh-fyi/soarm-oft-lora-20260829-052948 (private=True asserted
  by push_checkpoint_to_hub itself before upload; action_head--{1000,2000}_checkpoint.pt and
  proprio_projector--{1000,2000}_checkpoint.pt both present, confirming neither push clobbered
  the other for those files). Note: lora_adapter/adapter_model.safetensors is NOT step-suffixed
  -- both checkpoint dirs share that relative path, so the second (2000_chkpt) push silently
  overwrote the first's LoRA weights there. Only the 2000-step LoRA adapter survives in the
  repo now (harmless here since 2000 is the checkpoint we want for eval; the 1000-step LoRA
  weights only still exist in the local Colab VM's 1000_chkpt folder, if that VM is still up).
  Resume-from-disconnect half NOT YET TESTED: the kernel restart that happened this session was
  incidental (user manually stopped training, then separately hit an unrelated kernel restart
  before pushing checkpoints) -- worked around by manually reconstructing
  RUN_ROOT_DIR/HF_ADAPTER_REPO_ID and pushing directly, not by exercising the notebook's actual
  RESUME=True/RESUME_STEP=<N> relaunch path. That path remains unverified.
severity: minor

### 3. WandB training-loss curves visible
expected: |
  Training-loss curves from the 06a-finetune.ipynb run are visible in the shared
  soarm-oft-finetune-eval WandB project (TUNE-04).
result: pass
note: |
  Confirmed via screenshots -- run ft+openvla-7b-oft-finetuned-libero-spatial+soarm_spatial+b8+
  lr-0.0005+lora-r32+dropout-0.0--image_aug in project soarm-oft-finetune-eval, VLA Train
  section shows Loss, Next Actions L1 Loss, Curr Action L1 Loss, and Learning Rate panels, all
  populated across ~2200 steps with the expected sharp-drop-then-plateau shape.

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
result: issue
reported: |
  BEFORE pass completed cleanly (80 episodes, 4 tasks, 20 seeds each, no errors). But the
  aggregate result (75%, 60/80) is misleading: 3 of the 4 tasks (put_the_cream_cheese_to_the_
  right_of_the_bowl, put_the_cream_cheese_near_the_bowl, put_the_butter_between_the_bowl_and_
  the_cream_cheese) show PASS at steps=1 for every single episode (20/20 each). Investigated
  live: these are Phase 5's SPAT-05 predicate-validation tasks (libero_spatial_soarm/*.bddl),
  deliberately authored so their goal predicates (RightOfX/NearTo/LeftOfX) are satisfied by the
  reused, already-collision-validated Phase-4 object regions AT RESET, by design (05-03-PLAN.md
  Task 2/3, D-09) -- confirmed via test_spatial_predicates.py's own
  test_*_task_resets_satisfy_goal_20_of_20 tests, which exist specifically to assert this. They
  are predicate-correctness smoke tests, not manipulation-difficulty benchmark tasks -- success
  is structurally guaranteed regardless of policy quality. Only put_the_cream_cheese_in_the_bowl
  (0/20 before, genuine physical containment goal) carries real before/after signal.
severity: minor
note: |
  User decision: proceed with only put_the_cream_cheese_in_the_bowl as the meaningful task for
  this Test 4 pass; author additional genuinely-challenging, empirically-validated SOARM
  manipulation tasks as separate next-phase work (a same-night attempt to just move the 3
  smoke-test tasks' spawn regions was tried and reverted -- it broke Phase 5's intentional,
  tested SPAT-05 invariant; see Gaps for the full investigation).

### 5. WandB dashboard shows both training curves AND eval results together
expected: |
  The before/after wandb.Table (per-task success rates) and aggregate scalars appear in the same
  soarm-oft-finetune-eval WandB project as the training-loss curves from item 3 — one dashboard,
  both metric types (TUNE-04).
result: [pending]

## Summary

total: 5
passed: 2
issues: 1
pending: 2
skipped: 0
blocked: 0

## Gaps

- truth: "The git clone/pull delivery cell (06a-finetune.ipynb cell 2) succeeds on every run within a given Colab VM, including after a kernel restart, without re-prompting for the GitHub token"
  status: resolved
  reason: "User reported: after restarting the Colab runtime (kernel restart within the same VM), the git clone/pull cell raises RuntimeError: git clone/pull failed -- check the token is valid and has read access to vansh-fyi/SO-ARM-research. This happens on the `git pull` (else) branch, taken because REPO_ROOT already exists on disk from the pre-restart clone."
  severity: blocker
  test: 1
  root_cause: "The cell's `else` branch (taken when REPO_ROOT already exists, e.g. after a runtime restart that leaves the VM disk intact) runs `git pull --progress` with a plain subprocess.run and no GIT_ASKPASS env var and no credential helper configured. GIT_ASKPASS is only set for the initial `git clone` subprocess call in the `if` branch. The cell's own comment claims 'the token is cached in this VM's local .git/config for subsequent git pulls', but the code never actually configures a git credential helper or otherwise persists the token anywhere -- so the non-interactive `git pull` against the private HTTPS remote has no way to authenticate, exits non-zero, and triggers the RuntimeError."
  artifacts:
    - path: "LIBERO/notebooks/06a-finetune.ipynb"
      issue: "Cell 2 (git clone/pull delivery cell): `git pull` in the else branch has no credential source; the `if` branch's GIT_ASKPASS-based clone succeeds but never persists credentials for later pulls"
  missing:
    - "After a successful clone (or before the pull), configure a git credential helper (e.g. `git config credential.helper store` with credentials written via GIT_ASKPASS, or `git config credential.helper 'cache --timeout=<VM lifetime>'`) so the token is actually reusable across the runtime restart, matching the cell's stated design intent -- or, alternatively, re-run GIT_ASKPASS-backed auth on the pull path too if persisting the token to disk is undesirable"
  debug_session: ""
  fix_applied: "This root cause was correctly diagnosed in an earlier session but never actually fixed -- the identical failure recurred live tonight, this time in 06b-eval.ipynb's own independent copy of the delivery cell (06a's copy likely also never exercised its clone-only credential setup this session, since REPO_ROOT probably already existed the first time ITS delivery cell ran too). Rewrote both notebooks' delivery cells identically: ensure ~/.git-credentials exists (getpass prompt only if missing) BEFORE deciding clone vs. pull, using credential.helper store for both uniformly -- removes the old clone-only GIT_ASKPASS special-casing entirely. Committed 227e556, pushed to origin/main."

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

- truth: "01-colab-env-setup.ipynb Block A installs all pinned packages and passes the numpy ABI gate without error on a stock Colab GPU runtime"
  status: resolved
  reason: "User reported: on a fresh Colab session (Python 3.13.15), Step 2 fails to install torch==2.2.0+cu121 (no cp313 wheel; only 2.5.0+cu121/2.5.1+cu121 available), and Step 5 fails to build tokenizers==0.19.1 from source (no cp313 wheel, no Rust toolchain to build it), which also fails the transformers-openvla-oft fork install. Because !pip failures don't halt cell execution, Steps 3/4/5b/6 ran anyway on top of a broken torch/tokenizers state. Block A's final numpy-ABI-gate cell (cell 12) then hangs indefinitely with zero output."
  severity: blocker
  test: 1
  root_cause: "Colab's 'Latest' default runtime moved past Python 3.12 to 3.13 sometime after Phase 1 was verified (2026-07-10); torch==2.2.0/tokenizers==0.19.1 (and every other Block A pin) were never wrong, they just have no cp313 wheels. This is external platform drift, not a repo bug -- confirmed via research.google.com/colaboratory/runtime-version-faq.html, which shows a dated version->Python table with no code changes needed."
  artifacts: []
  missing: []
  debug_session: ""
  fix_applied: "No code change. Colab ships a Runtime Version Selector (Runtime > Change runtime type > Runtime version) that pins a notebook to a dated snapshot (e.g. 2026.07 = Python 3.12.13), guaranteed available for 1 year. User pinned to 2026.07 and Block A now completes as originally written across all three notebooks that share these pins (01, 02, 03b)."

- truth: "06a-finetune.ipynb's RLDS-conversion cell finds Phase 4's demo HDF5 file(s) on the Colab VM before calling hdf5_to_rlds"
  status: resolved
  reason: "User reported: ValueError: no demo_* episodes found in [] -- the glob at LIBERO/libero/datasets/soarm_spatial/*_demo.hdf5 found zero files after a fresh git clone on Colab."
  severity: blocker
  test: 1
  root_cause: "LIBERO/libero/datasets/ is gitignored (LIBERO/.gitignore:8) -- Phase 4's collected demo dataset (1.7GB HDF5) was never committed to git and 06a-finetune.ipynb had no step to transfer it onto a fresh Colab VM; it only ever did a plain git clone."
  artifacts:
    - path: "LIBERO/notebooks/06a-finetune.ipynb"
      issue: "No cell downloads the gitignored demo dataset before the RLDS conversion cell runs"
  missing:
    - "A dataset-delivery mechanism reachable from a fresh Colab VM"
  debug_session: ""
  fix_applied: "Added LIBERO/scripts/upload_dataset_to_hf.py (one-time local push of *_demo.hdf5 + dataset_statistics.json to a private HF Hub dataset repo) and a new cell in 06a-finetune.ipynb, right before RLDS conversion, that downloads via snapshot_download. User ran the upload script (vansh-fyi/soarm-spatial-demos created, 1.73GB uploaded)."

- truth: "hdf5_to_rlds resolves each episode's BDDL task file regardless of which machine collected the HDF5 vs which machine is running the conversion"
  status: resolved
  reason: "User reported: FileNotFoundError: [Errno 2] No such file or directory: '/Users/hp/Desktop/Work/Repositories/SoARM-Research/LIBERO/libero/libero/bddl_files/libero_goal/put_the_cream_cheese_in_the_bowl.bddl' -- raised inside get_problem_info, called from load_episodes_from_hdf5, after the HF Hub dataset download fix above unblocked the previous gap."
  severity: blocker
  test: 1
  root_cause: "hdf5_writer.py stores os.path.abspath(bddl_file_name) into the HDF5's 'bddl_file_name' attribute at collection time. The demo was collected locally (/Users/hp/...); that absolute path is meaningless on the Colab VM (/content/SoARM-Research/...) it was downloaded onto."
  artifacts:
    - path: "LIBERO/libero/libero/datasets/rlds_converter.py"
      issue: "load_episodes_from_hdf5 passed the stored absolute bddl_file_name straight to get_problem_info with no portability fallback"
  missing:
    - "Re-anchor the stored path against the current environment when it doesn't exist on disk"
  debug_session: ""
  fix_applied: "load_episodes_from_hdf5 now falls back to re-anchoring the portable suffix after 'bddl_files/' against get_libero_path('bddl_files') (the current environment's actual BDDL root, from Block B's ~/.libero/config.yaml bootstrap) when the stored absolute path doesn't exist. Committed 1483992, pushed to origin/main."

- truth: "hdf5_to_rlds's SequentialWriter call correctly writes a genuine TFDS-loadable RLDS dataset to disk"
  status: resolved
  reason: "User reported: AttributeError: module 'tensorflow_datasets.core.naming' has no attribute 'DatasetIdentity', raised inside hdf5_to_rlds, after the bddl_file_name portability fix above unblocked episode loading."
  severity: blocker
  test: 1
  root_cause: "This whole call site (tfds.core.naming.DatasetIdentity / SequentialWriter(..., num_shards=1) / add_examples((i, dict)) tuples / data_dir=out_dir flat) was written without tensorflow_datasets installed locally to verify against -- flagged MEDIUM confidence in the function's own comment. Checked against the real tensorflow_datasets==4.9.10 wheel source (downloaded via pip download): DatasetIdentity actually lives at tfds.core.DatasetIdentity (not .core.naming.), SequentialWriter's constructor takes max_examples_per_shard (required, no num_shards kwarg), add_examples() wants flat feature dicts per split (not (key, dict) tuples -- that's the GeneratorBasedBuilder convention), and DatasetInfo.data_dir is used verbatim with no auto name/version nesting so the write target needs <out_dir>/<dataset_name>/1.0.0/ built in explicitly to match 06-RESEARCH.md's documented output layout and what finetune.py's tfds.builder(name, data_dir=out_dir) expects to find."
  artifacts:
    - path: "LIBERO/libero/libero/datasets/rlds_converter.py"
      issue: "hdf5_to_rlds's SequentialWriter/DatasetIdentity call site had three separate API-shape bugs, none exercised until this Colab run"
  missing:
    - "Correct all three: DatasetIdentity import path, SequentialWriter kwarg name, add_examples() payload shape, plus nested data_dir"
  debug_session: ""
  fix_applied: "All four issues fixed in one pass against the verified 4.9.10 API. Committed 95a864d, pushed to origin/main."

- truth: "apply_soarm_spatial_registration (OXE registration) imports prismatic successfully in the same Colab kernel that already ran RLDS conversion"
  status: resolved
  reason: "User reported: RuntimeError: Visible devices cannot be modified after being initialized, raised inside prismatic/vla/datasets/rlds/dataset.py's tf.config.set_visible_devices([], \"GPU\") module-level call, triggered transitively by apply_soarm_spatial_registration's `from prismatic.vla.datasets.rlds.oxe.configs import ...`."
  severity: blocker
  test: 1
  root_cause: "prismatic's dataset.py hides the GPU from TF at import time (so TF dataloading doesn't clobber PyTorch's CUDA context) -- but that call only succeeds if it's the FIRST TF context-initializing action in the process. The RLDS conversion cell, which runs earlier in the same Colab kernel, already performs real TF ops (writing/round-trip-reading TFRecords via tfds), initializing TF's device context first. Verified against tensorflow==2.20.0's actual context.py source: set_visible_devices raises only when the context is already initialized AND the new value differs from the currently-set value; an identical-value call after init is a silent no-op."
  artifacts:
    - path: "LIBERO/notebooks/06a-finetune.ipynb"
      issue: "No cell hid the GPU from TF before the RLDS conversion cell's TF usage initialized the context"
    - path: "LIBERO/libero/libero/datasets/rlds_converter.py"
      issue: "hdf5_to_rlds's TF usage was the first TF context-initializing action in the kernel with no GPU-hide call of its own"
  missing:
    - "An unconditional GPU-hide call that runs even when RLDS conversion is skipped (already-converted), so prismatic's later identical call is a no-op"
  debug_session: ""
  fix_applied: "Added an always-run 'hide GPU from TF' cell before the RLDS Conversion section in 06a-finetune.ipynb (executes regardless of conversion caching), plus the same call as hdf5_to_rlds's first TF action for a genuinely fresh VM. Committed 1d6d553, pushed to origin/main. Requires a Colab KERNEL RESTART to take effect in the current session -- TF's device context is process-level state, not reloadable via importlib."

- truth: "The OXE-registered soarm_spatial dataset is resolvable by finetune.py's torchrun subprocess, with correct proprio shape and a registered standardization transform"
  status: resolved
  reason: "Proactive review, not a user-hit error yet: apply_soarm_spatial_registration's smoke-test passed in the notebook kernel, but finetune.py is launched via subprocess.Popen(['torchrun', ...]) -- a separate OS process that re-imports prismatic fresh from disk. The registration was in-memory-only (own docstring: 'never editing the cloned/installed package's files on disk'), invisible to that subprocess. Would have hit KeyError: 'soarm_spatial' at the torchrun cell, after already paying the GPU cost of getting that far."
  severity: blocker
  test: 1
  root_cause: "Three separate issues, verified against the real moojink/openvla-oft source: (1) registration only touched in-memory dicts, never the installed package files torchrun's subprocess reads; (2) OXE_STANDARDIZATION_TRANSFORMS[dataset_name] was never registered at all -- materialize.py does a plain dict lookup with no default, a second independent KeyError source; (3) state_obs_keys=['state', None, None] -- dataset.py inserts one zero-padding element per None entry, producing a 9-dim proprio instead of the intended 7-dim (REQUIRED_PROPRIO_DIM). Also found PROPRIO_DIM=8 (LIBERO_CONSTANTS, auto-detected via a 'libero' substring in --data_root_dir) mismatches this project's actual 7-dim joint-space proprio."
  artifacts:
    - path: "LIBERO/libero/libero/datasets/oxe_register.py"
      issue: "In-memory-only registration; missing OXE_STANDARDIZATION_TRANSFORMS entry; wrong state_obs_keys padding"
  missing:
    - "File-patch the installed configs.py/mixtures.py/transforms.py/constants.py so a fresh subprocess sees the registration"
    - "Register an identity standardize_fn (rlds_converter.py's output already matches openvla-oft's expected post-transform schema)"
    - "Fix state_obs_keys to a single-element list"
    - "Override PROPRIO_DIM to 7"
  debug_session: ""
  fix_applied: "Rewrote apply_soarm_spatial_registration to idempotently append all four registrations directly to the installed package files on disk (marker-checked, safe to re-run), plus reload the patched modules so the notebook kernel's own smoke-test reflects the correction immediately. action_encoding=EEF_POS was confirmed already correct (unchanged) against rlds_converter.py's REQUIRED_ACTION_DIM=7 comment. Local pytest suite (9 passed, 2 skipped -- tensorflow_datasets unavailable locally) confirms no regression. Committed 0654ea4, pushed to origin/main."

- truth: "torchrun launches finetune.py's distributed rendezvous without crashing on this Colab Python/torch combination"
  status: resolved
  reason: "User reported: Fatal Python error: Segmentation fault inside torch/distributed/elastic/rendezvous/c10d_rendezvous_backend.py's _call_store, torchrun exited with code -11, after ~30s of the finetune_proc polling loop."
  severity: blocker
  test: 1
  root_cause: "Known PyTorch bug (github.com/pytorch/pytorch/issues/125990): torchrun's default libuv-based TCPStore backend has a GIL-handling bug under Python 3.12+ (calls Python object allocation without holding the GIL), segfaulting during rendezvous store calls. This project's Colab runtime is on Python 3.12.13 (per the pinned 2026.07 Runtime Version), squarely in the affected range."
  artifacts:
    - path: "LIBERO/notebooks/06a-finetune.ipynb"
      issue: "torchrun invocation used --standalone with no rdzv_conf override, defaulting to the buggy libuv TCPStore backend"
  missing:
    - "--rdzv-conf use_libuv=False to fall back to the older, unaffected TCPStore implementation"
  debug_session: ""
  fix_applied: "First attempt (--rdzv-conf use_libuv=False, commit c0dcd49) did NOT work -- identical segfault, identical line, on retest. Root-caused further: _create_tcp_store's TCPStore(...) call in c10d_rendezvous_backend.py never threads rdzv_conf's use_libuv value through at all (verified against torch==2.6.0's actual source) -- the flag was a no-op. Real fix: dropped torchrun entirely and launch finetune.py via plain `python` instead. This run is --nproc-per-node 1 --nnodes 1 (a single process) so torchrun's distributed rendezvous machinery added no value; finetune.py uses accelerate's PartialState(), which only initializes a distributed process group when LOCAL_RANK is set (verified against accelerate==1.14.0's source), so plain `python` runs single-process automatically with no rendezvous store involved. Committed 3250866, pushed to origin/main."

- truth: "finetune.py's peft import succeeds against this project's pinned transformers version"
  status: resolved
  reason: "User reported: ImportError: cannot import name 'EncoderDecoderCache' from 'transformers', raised inside peft/peft_model.py during finetune.py's own `from peft import LoraConfig, PeftModel, get_peft_model` -- after the torchrun segfault fix got the process running far enough to reach this import."
  severity: blocker
  test: 1
  root_cause: "06-RESEARCH.md pinned peft==0.20.0 purely because it was 'latest available' on the research date (2026-08-20), never cross-checked against Block A's already-installed transformers==4.40.1 fork. peft 0.20.0's peft_model.py does `from transformers import ... EncoderDecoderCache` unconditionally at import time; that class was only added in transformers 4.43.0 (verified live by downloading and inspecting the actual PyPI wheels for 4.40.1/4.41.0/4.42.0/4.43.0). Both 06a-finetune.ipynb and 06b-eval.ipynb independently reinstalled peft==0.20.0 over Block A's already-correct, compatible peft==0.11.1."
  artifacts:
    - path: "LIBERO/notebooks/06a-finetune.ipynb"
      issue: "Cell 5 reinstalled peft==0.20.0 over Block A's compatible peft==0.11.1"
    - path: "LIBERO/notebooks/06b-eval.ipynb"
      issue: "Same peft==0.20.0 reinstall, would have hit the identical crash during Test 4"
  missing:
    - "Drop the peft==0.20.0 reinstall in both notebooks; rely on Block A's peft==0.11.1"
  debug_session: ""
  fix_applied: "Removed the peft==0.20.0 install line from both notebooks' cells (kept tensorflow-datasets==4.9.10 in 06a, unrelated to this bug). PeftModel.from_pretrained(...).merge_and_unload() and get_peft_model()/LoraConfig(r=32) are long-standing basic peft APIs available in 0.11.1, not 0.20-only features. Committed 84783bc, pushed to origin/main."

- truth: "finetune.py's full import chain succeeds, including experiments.robot.openvla_utils"
  status: resolved
  reason: "User reported: ModuleNotFoundError: No module named 'json_numpy', raised inside experiments/robot/openvla_utils.py, imported by finetune.py's `from experiments.robot.openvla_utils import (...)` -- after the peft fix got the process past its previous import failure."
  severity: blocker
  test: 1
  root_cause: "openvla-oft declares json-numpy as a dependency (visible throughout Block A's own pip resolver-conflict warnings: 'openvla-oft 0.0.1 requires json-numpy, which is not installed') but openvla-oft itself is installed --no-deps (01-colab-env-setup.ipynb Step 5), so nothing ever pulls json-numpy in. Same class of gap as Phase 1's eager-chain deps list (tensorflow_graphics/draccus/jsonlines/wandb/diffusers) -- json-numpy was simply missing from that list, since it's only needed by finetune.py's specific import path, not exercised by Phase 1-3's own smoke tests."
  artifacts:
    - path: "LIBERO/notebooks/06a-finetune.ipynb"
      issue: "Block A never installed json-numpy"
  missing:
    - "pip install json-numpy (tiny package, only depends on numpy -- no --no-deps needed)"
  debug_session: ""
  fix_applied: "Added !pip install json-numpy -q to the same Block A cell as the tensorflow-datasets install. Confirmed oft_backend.py (06b-eval.ipynb's dependency) does not import experiments.robot.openvla_utils, so this is scoped to 06a-finetune.ipynb only. Committed cb9c2fb, pushed to origin/main."

- truth: "diffusers imports cleanly against this stack's pinned peft==0.11.1"
  status: resolved
  reason: "User reported: ImportError: peft>=0.17.0 is required for a normal functioning of this module, but found peft==0.11.1 -- raised inside diffusers/utils/constants.py's dep_version_check('peft'), triggered by action_heads.py's `from diffusers.schedulers.scheduling_ddim import DDIMScheduler`."
  severity: blocker
  test: 1
  root_cause: "01-colab-env-setup.ipynb's eager-chain deps loop only installs diffusers if missing (find_spec check). Colab ships diffusers stock, and that stock version drifts upward over time (observed 0.38.0, then 0.39.0 across sessions) -- so the intended diffusers==0.30.3 pin, already written in the _OFT_DEPS list, was never actually enforced. diffusers>=0.31 bumped its own MIN_PEFT_VERSION check to >=0.17.0, which conflicts with peft==0.11.1 (required for transformers==4.40.1 compatibility, per the earlier peft fix). Verified diffusers==0.30.3 (openvla-oft's own documented pin) only requires peft>=0.6.0 / transformers>=4.34.0 by downloading and inspecting its actual dependency_versions_table.py."
  artifacts:
    - path: "LIBERO/notebooks/01-colab-env-setup.ipynb"
      issue: "diffusers was in the skip-if-present eager-chain loop, so its pin was never enforced against Colab's drifting stock version"
  missing:
    - "Force-install diffusers==0.30.3 unconditionally, not skip-if-present"
  debug_session: ""
  fix_applied: "Moved diffusers out of the skip-if-present _OFT_DEPS loop into its own unconditional force-install step in 01-colab-env-setup.ipynb (commit 72ddc02). For the CURRENT session (06a-finetune.ipynb doesn't re-run Phase 1's Block A -- its own comment says it 'defers to whatever Phase 1's baked environment already provides'), user ran `!pip install diffusers==0.30.3 -q` directly as an immediate workaround."

- truth: "finetune.py's dist.barrier() calls have an initialized process group to call into"
  status: resolved
  reason: "User reported: ValueError: Default process group has not been initialized, please make sure to call init_process_group -- raised at finetune.py line 831's dist.barrier() call, after training progressed much further (WandB run started, base checkpoint downloaded and patched) than any previous attempt."
  severity: blocker
  test: 1
  root_cause: "Direct consequence of the earlier torchrun-segfault fix (commit 3250866): finetune.py calls raw torch.distributed dist.barrier() in 4 places with no init_process_group() call anywhere in its own source -- it relies entirely on the launcher (torchrun) having already initialized torch.distributed via accelerate's PartialState(). Dropping torchrun meant LOCAL_RANK was never set, so accelerate's PartialState() took its single-process fallback path and skipped calling torch.distributed.init_process_group() entirely -- finetune.py's dist.barrier() calls then had no process group to call into."
  artifacts:
    - path: "LIBERO/notebooks/06a-finetune.ipynb"
      issue: "subprocess.Popen launched finetune.py with no RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT env vars, so accelerate never initialized torch.distributed"
  missing:
    - "RANK=0, LOCAL_RANK=0, WORLD_SIZE=1, MASTER_ADDR=localhost, MASTER_PORT=29500 in the subprocess environment"
  debug_session: ""
  fix_applied: "Set those env vars explicitly on the subprocess.Popen call (verified against accelerate==1.14.0's source that it calls torch.distributed.init_process_group(backend=...) with no explicit args, relying on the default env:// init method reading these exact variables). This gives finetune.py's dist.barrier() calls a trivial single-process NCCL group to call into, without needing torchrun's rendezvous machinery (the segfault source) at all. Committed 5cc6f30, pushed to origin/main."

- truth: "get_oxe_dataset_kwargs_and_weights builds valid kwargs for the registered soarm_spatial dataset"
  status: resolved
  reason: "User reported: KeyError: 'depth_obs_keys', raised inside materialize.py's make_oxe_dataset_kwargs, when RLDSDataset.__init__ called get_oxe_dataset_kwargs_and_weights. Training progressed much further than any previous attempt this session: WandB run started, base checkpoint + LoRA adapter loaded, trainable params printed (110,828,288 trainable / 7,652,065,472 total) -- crashed only when it reached actual dataset construction."
  severity: blocker
  test: 1
  root_cause: "make_oxe_dataset_kwargs does dataset_kwargs['depth_obs_keys'].items() unconditionally (no .get() fallback) -- it's only popped from the returned kwargs afterward, when load_depth is False. register_soarm_spatial never included this key at all."
  artifacts:
    - path: "LIBERO/libero/libero/datasets/oxe_register.py"
      issue: "register_soarm_spatial's dict literal (and the matching on-disk patch text) was missing depth_obs_keys entirely"
  missing:
    - "depth_obs_keys: {primary: None, secondary: None, wrist: None} -- matches openvla-oft's own LIBERO configs' all-None pattern for datasets with no depth cameras"
  debug_session: ""
  fix_applied: "Added the missing key to both register_soarm_spatial and the file-patch text. Versioned the on-disk patch marker to v2 so this actually re-applies to a VM whose configs.py was already patched by the prior, incomplete version -- the marker-gated idempotency check would otherwise silently skip the corrected block. Local pytest (3 passed) confirms no regression. Committed 1374453, pushed to origin/main."

- truth: "poll_and_push_checkpoints correctly identifies checkpoint step numbers and actually runs periodically during training"
  status: resolved
  reason: "Proactive review after tonight's training run, not a live crash: the checkpoint auto-push cells never actually fired during training (checkpoints had to be pushed manually after a kernel restart), and reviewing why surfaced two design bugs."
  severity: minor
  test: 2
  root_cause: "(1) The sort key assumed checkpoint dirs are named '<step>_chkpt', but finetune.py names them '<run-name>--<step>_chkpt' -- basename.split('_')[0].isdigit() always failed, so every checkpoint silently sorted as step 0 (harmless with <=2 checkpoints, would prune the wrong ones with 3+). (2) The cell's own comment said to run it 'periodically... from another cell execution while training continues', which is impossible in a single Jupyter/Colab kernel since the launch cell blocks synchronously in its polling loop -- this cell could only ever run AFTER training fully stopped."
  artifacts:
    - path: "LIBERO/notebooks/06a-finetune.ipynb"
      issue: "poll_and_push_checkpoints' sort key and the concurrency assumption in its own docstring were both wrong"
  missing:
    - "A regex-based step extraction matching the real naming pattern"
    - "Calling poll_and_push_checkpoints() from inside the launch cell's existing polling loop instead of a separate, never-reachable cell"
  debug_session: ""
  fix_applied: "Moved push_checkpoint_to_hub/poll_and_push_checkpoints' definitions before the launch cell, fixed the sort key with a regex verified against tonight's real directory names, and call poll_and_push_checkpoints() from inside the training loop (plus once more after it exits). Removed two ad-hoc cells that were a one-off manual recovery snippet for tonight's kernel restart (hardcoded timestamp), not permanent notebook content. Committed f8dcc8d, pushed to origin/main."

- truth: "The 4-task before/after benchmark in 06b-eval.ipynb (3 spatial_soarm tasks + 1 goal task) produces a meaningful signal for whether fine-tuning improved manipulation capability"
  status: investigated, not fixed -- deferred to next phase
  reason: "BEFORE pass: 3 of 4 tasks (right_of_the_bowl, near_the_bowl, butter_between) PASS at steps=1 for all 20/20 episodes each; only put_the_cream_cheese_in_the_bowl (0/20) behaves like a real manipulation result."
  severity: minor
  test: 4
  root_cause: "NOT a bug. LIBERO/libero/libero/bddl_files/libero_spatial_soarm/*.bddl were authored in Phase 5 (05-03-PLAN.md, SPAT-05/D-09) specifically so their goal predicates (RightOfX/NearTo/LeftOfX) are satisfied by Phase 4's already-collision-validated object regions AT RESET -- confirmed via test_spatial_predicates.py's test_*_task_resets_satisfy_goal_20_of_20 tests, which exist precisely to assert this. Their purpose was proving the new predicate CLASSES evaluate spatial relations correctly (SPAT-05's actual requirement), not providing manipulation-difficulty benchmark tasks. Phase 6's 06-02-PLAN.md picked these same 3 tasks for the eval suite without accounting for that -- a task-selection mismatch between phases, not a code defect in either phase."
  artifacts:
    - path: "LIBERO/libero/libero/bddl_files/libero_spatial_soarm/*.bddl"
      issue: "Correctly implements its OWN (Phase 5) design intent; incorrectly reused as Phase 6 manipulation-benchmark tasks"
    - path: ".planning/phases/06-fine-tuning-evaluation/06-02-PLAN.md"
      issue: "Selected these 3 tasks for the before/after benchmark without checking whether their goals were structurally guaranteed"
  missing:
    - "Additional genuinely-challenging, empirically-validated SOARM manipulation tasks (not stock LIBERO tasks -- collector.py's TASKS list confirms only put_the_cream_cheese_in_the_bowl has ever been through the reach/collision validation process any other SOARM task would need)"
  debug_session: ""
  near_miss: "First diagnosed this as a BDDL region-definition bug and, with user sign-off, redesigned + sim-validated (empirical reset/settle checks, no collision, goal correctly False at reset) new spawn regions for all 3 tasks. Running the EXISTING test_spatial_predicates.py suite against the changed files immediately failed 5 tests -- test names like test_right_of_task_resets_satisfy_goal_20_of_20 revealed the trivial-pass behavior was intentional and tested, not a bug. Reverted via git checkout before committing (working tree only, nothing was ever pushed). Lesson: run the existing test suite for a file BEFORE editing it, not just after -- would have caught this in one command instead of a multi-step simulation investigation."
  fix_applied: "None this session -- user decision: proceed with only put_the_cream_cheese_in_the_bowl as Test 4's meaningful signal; author new, properly-validated SOARM manipulation tasks as separate next-phase work."

- truth: "FinetunedOFTBackend loads the pushed LoRA adapter successfully"
  status: resolved
  reason: "User reported: ValueError: Can't find 'adapter_config.json' at '/root/.cache/huggingface/hub/models--vansh-fyi--soarm-oft-lora-20260829-052948/snapshots/<sha>' -- raised inside PeftModel.from_pretrained, called from FinetunedOFTBackend.__init__ during the AFTER-pass setup."
  severity: blocker
  test: 4
  root_cause: "adapter_backend.py pointed PeftModel.from_pretrained() at the raw snapshot_download() root, assuming adapter_config.json lives there. push_checkpoint_to_hub (06a-finetune.ipynb) actually uploads finetune.py's entire checkpoint directory verbatim -- confirmed via HfApi().list_repo_files() earlier tonight -- so the PEFT adapter files live in that checkpoint's own lora_adapter/ subfolder, not the repo root. dataset_statistics.json's separate hf_hub_download call was already correct (that file IS at the repo root)."
  artifacts:
    - path: "LIBERO/libero/libero/vla/adapter_backend.py"
      issue: "adapter_dir passed to PeftModel.from_pretrained() was the snapshot root, missing the lora_adapter/ subfolder"
    - path: "LIBERO/libero/libero/vla/test_adapter_backend.py"
      issue: "Mock assertion encoded the same incorrect assumption -- updated to match, since (unlike the BDDL near-miss above) this had a live traceback proving the old behavior was actually broken"
  missing:
    - "os.path.join(checkpoint_dir, 'lora_adapter') before passing to PeftModel.from_pretrained()"
  debug_session: ""
  fix_applied: "Added the lora_adapter/ subfolder join; updated the existing test's mock assertion to match. Local pytest (2 passed) confirms no regression. Committed 4f6bb81, pushed to origin/main."

- truth: "FinetunedOFTBackend resolves the correct unnorm_key against the adapter's own norm_stats"
  status: resolved
  reason: "User reported: KeyError: No libero_spatial key in adapter norm_stats. Available: ['soarm_spatial'] -- raised after the adapter successfully downloaded and merged (previous lora_adapter path fix confirmed working)."
  severity: blocker
  test: 4
  root_cause: "unnorm_key resolution hardcoded 'libero_spatial' (falling back to 'libero_spatial_no_noops'), matching OFTBackend's zero-shot base-checkpoint key. The fine-tuned adapter's own dataset_statistics.json is keyed by this project's actual registered dataset name (oxe_register.py's 'soarm_spatial'), not LIBERO's stock naming."
  artifacts:
    - path: "LIBERO/libero/libero/vla/adapter_backend.py"
      issue: "unnorm_key resolution assumed the base checkpoint's key name applies to the fine-tuned adapter's stats too"
  missing:
    - "Resolve unnorm_key from whatever single key is actually present in the adapter's norm_stats, instead of hardcoding a name"
  debug_session: ""
  fix_applied: "Since this project trains one LoRA adapter per single HDF5 dataset (D-04), norm_stats always has exactly one key -- use it directly, fail loudly if not exactly one. Existing test's mock data already used a single-key dict, so no test changes needed; local pytest (2 passed) confirms. Committed bf9323e, pushed to origin/main."
