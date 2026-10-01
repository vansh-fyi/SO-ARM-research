---
phase: 14-vla-style-pipeline-backends
plan: 11
subsystem: control/vla_bridge
status: complete
tags: [backends, rdt-1b, tier-c, http-adapter, vlab-09, joint-position, unified-action-space, t5-xxl, colab, a100]
requires:
  - "14-01: base_http_adapter.HttpAdapterActionSource / build_http_adapter_backend, register(), load_backend_config()"
  - "14-02: --backend registry dispatch in run_vla_episode.py"
  - "14-03: eef_to_joint.forward_kinematics() / solve_ik() (used only by the eef_pose fallback), the Tier C resident-adapter shape"
  - "14-10: per-checkpoint profile pattern (client/server profile tables pinned equal), isolated-venv Colab cells"
provides:
  - "vla_bridge.rdt1b_server: RDT1BModel, create_app(infer), main(), CHECKPOINT_PROFILES, chunk_step_for(), joint_state_to_model(), resolve_offload_dir()"
  - "vla_bridge.backends.rdt1b: build (--backend rdt1b, HTTP lane), make_convert_fn(profile, joint_limits_deg), joint_position_action(), rdt_eef_pose_to_eef_delta(), rotation_from_ortho6d(), ACTION_SPACES"
  - "backend_configs/rdt1b.json (pinned rdt-1b default + rdt-170m alternate, action_space joint_position)"
  - "policy_server.ipynb / policy_server_launch.md: Tier C RDT-1B section (rdt1b-* cells, port 3005)"
affects: [14-12 (remaining Tier C backend; port 3005 now taken), 14-15 (end-of-phase UAT / legitimacy check), 16 (evaluation)]
tech-stack:
  added: []
  patterns:
    - "Upstream wrapper subclass: only the README-documented robot hooks (_format_joint_to_state / _unformat_action_to_joint) are overridden"
    - "Strict post-load weight re-check behind a strict=False upstream loader"
    - "Remote checkpoint inspection: tensor names/shapes read via HTTP Range over the zip's pickle (no multi-GB download)"
    - "Startup layout guard: the cloned repo's configs/state_vec.py must match the reviewed slot map"
    - "Chunk-step pacing: absolute-target chunks are sampled one client control period ahead"
key-files:
  created:
    - control/vla_bridge/rdt1b_server.py
    - control/vla_bridge/backends/rdt1b.py
    - control/vla_bridge/backend_configs/rdt1b.json
    - control/test_backend_rdt1b.py
  modified:
    - control/vla_bridge/policy_server.ipynb
    - control/vla_bridge/policy_server_launch.md
    - control/vla_bridge/checkpoint_candidates.md
    - .planning/phases/14-vla-style-pipeline-backends/deferred-items.md
decisions:
  - "RDT-1B's action space for SO-101 is joint_position, confirmed from upstream code (state_vec.py, agilex_model.py, hdf5_vla_dataset.py, train.py) and the checkpoint's weights: slots 0-4 = absolute arm joints in radians, slot 10 = gripper_open in [0, 1]. No IK."
  - "Default checkpoint is the official robotics-diffusion-transformer/rdt-1b@eb09036. No SO-101/SO-100/Koch RDT fine-tune exists on the Hub, so the run is out of distribution (the model card says RDT does not generalize to unseen robots); rdt-170m@8aa386c is the reviewed alternate"
  - "The joint_position branch converts units (rad -> deg, gripper 0-1 -> %) rather than being a literal identity: identity would read 0.5 rad as 0.5 degrees"
  - "Server returns chunk[round(30 / client_hz) - 1] (step 14 at the default 2 Hz loop), since each of the 64 chunk steps is an absolute target 1/30 s apart"
  - "T5-XXL is encoded live via upstream T5Embedder with --offload-dir threaded into use_offload_folder (auto below 24 GiB VRAM), cached per instruction; --no-text-encoder + --lang-embeddings serves precomputed encode_lang.py embeddings instead"
  - "main() refuses GPUs without native bf16 (T4) and dependency versions other than torch 2.1.0 / transformers 4.41.0 / diffusers 0.27.2 / timm 1.0.3 / accelerate 0.30.1 / huggingface_hub 0.25.2; port 3005"
  - "The server serves joint_position profiles only; the eef_pose fallback is client-side (an EEF RDT checkpoint would need the EEF pose as state, which the wire contract does not carry)"
metrics:
  duration: "~20 min"
  completed: 2026-10-01
  tasks: 2
  files: 8
---

# Phase 14 Plan 11: RDT-1B Backend Summary

`--backend rdt1b` is a Tier C backend on the same HTTP lane as OpenVLA and Octo. A Colab Flask adapter runs RDT-1B through the official `thu-ml/RoboticsDiffusionTransformer` deployment wrapper (`RoboticDiffusionTransformerModel.step()`), overriding only the two hooks the README says to change for a new robot. RDT's action space for SO-101 was **confirmed as joint-position, not EEF**, from the repo's own code and the checkpoint's own weights. So the client maps the 6 returned slots directly onto `JOINT_ORDER`: radians become degrees, and the gripper's [0, 1] becomes percent. There is no IK. The default checkpoint is the official `rdt-1b`. No SO-101 RDT fine-tune exists, so this run proves the pipeline rather than the task.

## Action-space confirmation (verified live 2026-10-01, not assumed)

14-RESEARCH characterized RDT as having "a joint-position slot usable directly for SO-101's 6 DOF". That is **confirmed**. The evidence comes from upstream source at the pinned commit `cd79363` (main HEAD, MIT) and from the checkpoint itself:

- **`configs/state_vec.py`** (`STATE_VEC_LEN = 128`):
  - `right_arm_joint_{0..9}_pos` map to slots 0-9, and `right_gripper_open` to slot 10.
  - The EEF position is slots 30-32 and the EEF 6-D rotation is 33-38. Left-arm joints are 50-59, and the base is 100-102.
- **README:** "If your robot is single-arm, please fill its action into the *right-arm* portion". Six DOF fill "the first six" joint slots. Units are SI (radians), and only the gripper is min-max normalized to [0, 1].
- **`scripts/agilex_model.py` and `data/hdf5_vla_dataset.py`:**
  - The action is the ABSOLUTE target `qpos`, written into the same slots as the state.
  - `train/train.py` and `train/sample.py` pass `state_elem_mask` as the `action_mask`. So state and action always share the slot format.
- **The checkpoint weights.** Tensor names and shapes were read from `rdt-1b@eb09036`'s `pytorch_model.bin` by HTTP range requests over its zip pickle, without downloading it:
  - 1,228,319,872 bf16 parameters, with no `module.` prefix.
  - `final_layer.ffn_final.fc2` is (128, 2048). `state_adaptor.0` is (2048, 256), i.e. state plus mask.
  - `img_cond_pos_embed` is (1, 4374, 2048), i.e. 2 history × 3 cameras × 729 patches. `lang_adaptor.0` is (2048, 4096), the T5-XXL width.
  - `rdt-170m@8aa386c` has the same 128-D in/out, with 166,229,888 parameters.
- **What the checkpoint does NOT record:** an RDT checkpoint carries no per-robot slot map; that lives in the fine-tuner's dataset code.
  - `state_indices = [0,1,2,3,4,10]` is this repo's SO-101 mapping, following the README.
  - `rdt1b_server._check_state_vec_layout()` re-reads the cloned repo's `state_vec.py` at startup and refuses to start if the mapping changed.

**SO-101 mapping:**
- The 5 arm joints go to slots 0-4 as `radians(degrees)`.
- The gripper goes to slot 10 as `percent / 100`. This assumes lerobot's 0 % = closed, with RDT's `gripper_open` 1 = open.
- `ctrl_freqs` = 30 Hz, the lerobot SO-101 recording rate.

## Checkpoint search
- **Searches:** 14 Hub name searches plus the `rdt` tag. I also paginated every id from 8 SO-101/SO-100/SO-ARM searches (up to 9,191 ids each) and filtered them for `rdt`.
- **Result:** no RDT fine-tune for SO-101, SO-100 or Koch exists. The community RDT repos are AgileX/ALOHA, Piper, UR5, LIBERO, RoboTwin, or undocumented mixin dumps.
- The full table is in `checkpoint_candidates.md` (`backend:rdt1b`, `backend:rdt1b-selection`).
- **Default:** `robotics-diffusion-transformer/rdt-1b@eb09036` (MIT, official, audited org). The model card says it "cannot yet generalize to new robot platforms (not seen in the pre-training datasets)". That is recorded in the config notes and in deferred-items.

## VRAM / compute (Pitfall 4)
- **Encoders:**
  - T5-XXL `google/t5-v1_1-xxl@3db67ab` ships only a 44.5 GB fp32 `pytorch_model.bin` (zip format, 11.4B parameters including the decoder). transformers 4.41 loads it with `mmap=True` (checked in `modeling_utils.py`), so it doesn't need 44.5 GB of RAM. The encoder alone is ~9.5 GB in bf16.
  - SigLIP so400m@`9fdffc5` is 3.5 GB on disk and ~0.8 GB in VRAM for the vision tower.
  - RDT-1B is 2.5 GB.
  - Total: **~13 GB of VRAM**. That is comfortable on an A100.
- **T4 refused:** `RoboticDiffusionTransformerModel` runs in bfloat16, and a T4 (sm75) has no native bf16. `main()` refuses non-bf16 GPUs with a clear message, and the Colab import check asserts `is_bf16_supported()`.
- **Offload:** upstream's `T5Embedder(use_offload_folder=...)` puts encoder blocks 12-23 on disk; `encode_lang.py` recommends this below 24 GB.
  - `--offload-dir` (default `/content/rdt_t5_offload`) is threaded straight into that argument.
  - `--t5-offload auto` enables it below 24 GiB (e.g. an L4).
- **Evidence for the human check:** the server logs the GPU, the offload state and the **peak VRAM after warm-up** before reporting ready.

## What Was Built

### Task 1: adapter, client and config (commits 1b2a102 RED, 0278090 GREEN)
- **`rdt1b_server.py`:** same contract as `octo_server.py`.
  - **Loading:**
    - It loads once, from pinned snapshots (checkpoint, SigLIP and T5, each at a reviewed sha).
    - It goes through upstream `RoboticDiffusionTransformerModel`, which calls `RDTRunner.from_pretrained(local_dir)`, the README's documented path. The `huggingface_hub==0.25.2` mixin source was read: it injects `config.json` as `config` and loads the pickle with `strict=False`.
    - `_verify_weights()` then fails on any missing or unexpected key.
    - The subclass overrides `_format_joint_to_state` / `_unformat_action_to_joint` for slots `[0..4, 10]`.
  - **History:** a 2-frame image history in the order `[ext, right_wrist, left_wrist] × [t-1, t]`.
    - ext = `overhead_left` (`--ext-image-key`), right wrist = `wrist`, left wrist = background.
    - At t-1 on a fresh episode every slot is background, as in `agilex_inference.py`.
    - It resets on a new instruction, `reset: true`, or a 10 s idle gap.
  - **Language:** live T5-XXL encoding identical to `scripts/encode_lang.py`, cached per instruction (LRU, 64 entries). Precomputed `.pt` embeddings also work.
  - **Responses:**
    - The proprio is required, so a missing, non-numeric or non-finite `joint_state` gets 400.
    - 429 when busy.
    - 500 on a non-finite or wrong-shaped chunk.
    - `/health` reports the pins, slots, units, `chunk_step` and image order.
  - **Guards:** dependency-version, `state_vec` layout and bf16-GPU checks. It serves on :3005.
- **`backends/rdt1b.py`:**
  - Registered with `@register("rdt1b")`. Endpoint and checkpoint-profile rules are identical to octo's.
  - `make_convert_fn(profile)`:
    - `joint_position`: `joint_position_action` (rad→deg, `lo + g*(hi-lo)` percent). No IK.
    - `eef_pose`: `rdt_eef_pose_to_eef_delta`, which takes an absolute position plus a 6-D rotation (`docs/test_6drot.py`'s Gram-Schmidt) and computes the delta at the current FK pose, then `solve_ik`.
  - The config's `action_space` drives which branch runs.
- **`backend_configs/rdt1b.json`:** checkpoint, revision, `endpoint_url: null`, `action_space: "joint_position"`, instruction, two checkpoint profiles (revision, `state_indices`, `joint_units`, gripper range, `control_hz`), and evidence notes.
- **Tests:** 94, including the real-torch format/unformat override, the strict weight check, `T5Embedder`'s offload argument, `main()`'s guards, and both conversion branches.

### Task 2: regression tests (commit 3429047)
- **Default round trip:**
  - Path: real `create_app` → real `build()` convert_fn → real `HttpAdapterActionSource` → real `validate_action`.
  - The proprio reached the model in RDT units.
  - The action is finite, in `JOINT_ORDER`, and non-held, and IK is never called.
- **rdt-170m:** the same path.
- **History:** the 2-frame history fills across steps.
- **Client rate:** `--client-control-hz` selects the chunk step.
- **eef_pose fallback:** end to end through real ikpy, with the FK of the result within 3 mm of the commanded absolute pose.
- **Failure:** a NaN chunk → 500 → `http-error-holding-position`.

### Colab launch and docs (commit 38c3783)
- **Cells:** `rdt1b-*` cells appended to `policy_server.ipynb`:
  - **Install:** an isolated Python 3.10 uv venv. It clones `thu-ml/RoboticsDiffusionTransformer` at `cd79363` and checks `HEAD`.
    - `torch==2.1.0` / `torchvision==0.16.0` from the official cu121 index.
    - Then the inference subset of `requirements.txt` at its pins, plus `huggingface_hub==0.25.2`, `protobuf==4.25.3`, `numpy==1.26.4` and `opencv-python-headless==4.9.0.80`.
  - **Import check:** checks the exit status, the upstream imports, CUDA and bf16.
  - **Launch:** checks for ≥60 GB of free disk, runs on :3005, prints `/health` and the peak-VRAM/OOM log lines.
  - Then tunnel, a local-run note, and teardown.
- **Launch md:** `policy_server_launch.md` regenerated with `scripts/sync_policy_notebook.py`. The diff only adds lines; all new code cells and the inner import-check snippet compile.
- **Candidates:** `checkpoint_candidates.md` gains the `backend:rdt1b` and `backend:rdt1b-selection` sections. Existing sections are unchanged.
- **Deferred items:** four new entries in `deferred-items.md`.

## Verification
- `cd control && <main .venv>/bin/python -m pytest -q test_backend_rdt1b.py`: **100 passed**.
- Full suite: **738 passed** (fork base 638, plus 100).
- `run_vla_episode.py --help` lists `--backend {act,diffusion,groot-n17,octo,openvla,openvla-oft,pi0,pi05,rdt1b,scripted,smolvla,wallx}`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] The joint-position branch converts units; it is not a literal identity**
- **Found during:** Task 1, reading the README and `state_vec.py`.
- **Issue:** the plan says `convert_fn` is "an identity pass-through" for a joint-direct checkpoint. RDT's slots are SI (radians, gripper 0-1), but `JOINT_ORDER` is degrees and percent. An identity would read 0.5 rad as 0.5 degrees.
- **Fix:** the slot mapping is direct, with no IK and no reordering. Only the units change: `math.degrees` on the arm, `lo + g*(hi-lo)` on the gripper. The server applies the inverse to the proprio, and a test pins the two as exact inverses.
- **Commit:** 0278090.

**2. [Rule 2 - Correctness] Chunk-step pacing**
- **Issue:** each of RDT's 64 chunk steps is an absolute target 1/30 s apart. Returning step 0 to a 2 Hz control loop would command about 1/15 of the intended motion per step.
- **Fix:** `chunk_step_for(30, client_hz)` returns step 14 at the default 2 Hz. `--client-control-hz` overrides it, and the step is reported in `/health`.
- **Commit:** 0278090.

**3. [Rule 2 - Security, T-14-06] Pinned revisions for the checkpoint and both encoders, plus a strict weight check**
- **Issue:** the plan pins only the GitHub commit. Upstream would resolve `main` for the checkpoint, SigLIP and T5. Its `T5Embedder` only accepts the bare repo id, and its mixin loads weights with `strict=False`.
- **Fix:**
  - `_download_pinned` fetches each repo at a reviewed sha. `_load_text_encoder` adds the pinned local path to `T5Embedder.available_models`.
  - `_verify_weights` re-checks the keys strictly.
  - `_check_state_vec_layout` guards the slot map at startup.
- **Commit:** 0278090.

**4. [Rule 2 - Correctness] bf16 GPU guard and dependency pins**
- **Issue:** the plan names A100 but nothing enforces it.
- **Fix:**
  - `main()` refuses GPUs without bf16 (a T4) before any download.
  - It also refuses unpinned dependencies. `huggingface_hub==0.25.2` is needed because `diffusers==0.27.2`'s `dynamic_modules_utils.py` imports `cached_download`, which huggingface_hub removed in 0.26.
- **Commit:** 0278090.

**5. [Plan-directed scope] Colab cells, candidates section and deferred items**
- **What:** added per the prior-wave handoff, although the plan's `files_modified` omits them.
- **Commit:** 38c3783.

**6. [Plan-directed choice] The EEF fallback is client-side only**
- **What:** both branches exist and are driven by the config, as the plan requires.
- **Why the server stays joint-only:** RDT trains with state and action in the same slots, so an EEF checkpoint would need the EEF pose as its state, and the wire contract has no such field. The server therefore refuses `eef_pose` profiles instead of guessing.
- **Where it is logged:** deferred-items.md.
- **Commit:** 0278090.

## TDD Gate Compliance
- **Task 1:** RED commit `1b2a102` (test collection failed with no `rdt1b_server` / `backends.rdt1b`; verified by moving the implementation aside), then GREEN commit `0278090`.
- **Task 2:** regression tests over finished code, committed as `test(...)` (`3429047`). They passed immediately, which is expected for regression coverage and matches 14-10's Task 2.

## Unverifiable here (recorded explicitly)
- **Never executed:** the real `RoboticDiffusionTransformerModel` load, T5-XXL encoding and `step()` on a GPU. This machine has no CUDA, and the RDT stack is Colab-only, so nothing was installed locally.
- **Read but not run:**
  - `T5Embedder`'s offload `device_map` behaviour with accelerate 0.30.1.
  - Whether uv resolves the two-step install cleanly (torch from the cu121 index, then PyPI pins that must not upgrade torch).
- **Inferred, not hardware-verified:** the gripper direction (0 % = closed ↔ `gripper_open` 0).
- All of these fall under the human check below.

## Upstream sources and packages (for the end-of-phase legitimacy check)
- **GitHub `thu-ml/RoboticsDiffusionTransformer` @ `cd79363a1387e8f81c7724d070ef7e45fd23150f`:** MIT, 1,806 stars, not archived. In the 14-RESEARCH audit (official Tsinghua TSAIL org, mirrored at HF `robotics-diffusion-transformer`).
- **HF repos:**
  - `robotics-diffusion-transformer/rdt-1b@eb09036` and `rdt-170m@8aa386c` (MIT, official; in the audit).
  - `google/t5-v1_1-xxl@3db67ab` (Apache-2.0, google) and `google/siglip-so400m-patch14-384@9fdffc5` (Apache-2.0, google). Both are named by RDT's README and model card. **Neither is in 14-RESEARCH's audit list: flag for the human check.**
- **PyPI pins (Colab venv only).** All exist on pypi.org, support cp310 or are pure-Python, and are first-party:
  - `transformers 4.41.0`, `diffusers 0.27.2`, `accelerate 0.30.1`, `huggingface-hub 0.25.2`, `safetensors` (transitive) and `tokenizers` (transitive): huggingface.
  - `timm 1.0.3`: huggingface/pytorch-image-models, Ross Wightman.
  - `sentencepiece 0.2.0`: google. `protobuf 4.25.3`: google. `numpy 1.26.4`. `packaging 24.0`.
  - `opencv-python-headless 4.9.0.80`. `flask`, `pillow`, `requests`, `pyyaml` (unpinned, as in the other cells).
- **Torch:** `torch 2.1.0` / `torchvision 0.16.0` from `download.pytorch.org/whl/cu121`, the official index the RDT README names.
- **`uv`:** Astral, already used by the 14-03/09/10 cells.
- **Nothing was installed locally.** The tests use the existing venv only (torch 2.11.0, flask, scipy, ikpy).

## Known Stubs
None.
- `endpoint_url: null` is intentional: it is the per-session ngrok URL, passed with `--server-address`.
- The `eef_pose` branch is real and tested, though no reviewed checkpoint selects it (see deferred-items).

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| threat_flag: network-endpoint | control/vla_bridge/rdt1b_server.py | A new Colab HTTP adapter (`/health`, `/action`) behind an ngrok tunnel, the same exposure class as the OpenVLA, Octo and Wall-X adapters. It accepts only images, an instruction (≤1000 chars), joint numbers and a reset bool. Requests are capped at 20 MB, images at ≤1024 px, the live embedding cache at 64 entries, and concurrency gets 429. |
| threat_flag: new-dependency | control/vla_bridge/policy_server.ipynb | A Colab-only isolated venv with RDT's pinned torch 2.1.0 stack, plus a git clone of `thu-ml/RoboticsDiffusionTransformer` at a pinned commit. `torch.load(weights_only=True)` is used for the checkpoint re-check. Nothing is installed locally. |

T-14-05 holds:
- The default path has no IK. Every action goes through `run_episode()`'s unmodified `safety_validator.validate_action()`.
- The `eef_pose` fallback clamps to URDF ∩ calibrated limits inside `solve_ik`.

T-14-06 holds: the GitHub commit and the checkpoint, SigLIP and T5 revisions are all pinned, and weight keys are verified.

T-14-09:
- **Mitigated before any episode:** the adapter refuses non-bf16 GPUs, logs peak VRAM after warm-up, and only reports ready after a successful warm-up. The launch cell surfaces OOM lines and an early process exit.
- **Still needs the live check below.**

## Human Checks Deferred
1. **RDT-1B live hardware on an A100 (Task 2 human-check):**
   - **test:**
     - On a **Colab Pro A100** runtime (not T4), run `rdt1b-install`. Confirm the printed versions (torch 2.1.0+cu121, transformers 4.41.0, diffusers 0.27.2, timm 1.0.3, huggingface_hub 0.25.2), the GPU name and the GiB, with no `RuntimeError`.
     - Upload `rdt1b_server.py`, then run `rdt1b-launch` (about 51 GB of first-time downloads) and `rdt1b-tunnel`.
     - Run `python control/run_vla_episode.py --backend rdt1b --server-address https://<tunnel> --instruction "pick up the cube" --out control/outputs/<dir> --max-steps 20` against the real SO-ARM101.
   - **expected:**
     - The log shows `T5-XXL offload: off` and a `peak VRAM ~13 GiB` line, with no `torch.cuda.OutOfMemoryError` and no early exit.
     - `/health` is ready, with `action_space` `joint_position`, `state_indices` `[0,1,2,3,4,10]` and `chunk_step` 14.
     - At least one step logs a validated, non-held action, with `model_version` `robotics-diffusion-transformer/rdt-1b@eb09036cc64c` and no `holding-position` flag, and the robot visibly moves.
     - Also note the inference time per request (it must stay under 5 s), and whether motion stays small and safe.
   - **why_human:** this needs real hardware and a live A100 session, and OOM confirmation is a log-inspection call.
   - **Note:** the motion is NOT expected to be task-meaningful, because the base checkpoint never saw SO-101.
2. **Gripper direction and joint-sign sanity:**
   - **test:** with the adapter up, compare the logged gripper target against the commanded/observed opening.
   - **expected:** the target moves the jaw the way the model's `gripper_open` intends (100 % = open).
   - **why_human:** hardware observation.
3. **Package legitimacy and licensing:**
   - **test:** confirm the evidence above, especially the two encoder repos outside the 14-RESEARCH audit (`google/t5-v1_1-xxl@3db67ab`, `google/siglip-so400m-patch14-384@9fdffc5`) and the `thu-ml` commit `cd79363`.
   - **expected:** first-party sources with no name or org mismatch.
   - **why_human:** a legitimacy judgment.
4. **Optional alternate:**
   - **test:** relaunch with `--checkpoint robotics-diffusion-transformer/rdt-170m` and run the client with the same `--checkpoint`.
   - **expected:** validated joint-position actions.
5. **Carried from 14-01:** on the HTTP lane, the wrist camera tolerates two independent `cv2.VideoCapture` opens.

## Self-Check: PASSED
- FOUND: control/vla_bridge/rdt1b_server.py, control/vla_bridge/backends/rdt1b.py, control/vla_bridge/backend_configs/rdt1b.json, control/test_backend_rdt1b.py, control/vla_bridge/policy_server.ipynb, control/vla_bridge/policy_server_launch.md, control/vla_bridge/checkpoint_candidates.md
- FOUND commits: 1b2a102, 0278090, 3429047, 38c3783
- No deletions and no STATE.md/ROADMAP.md changes since base 6f3a5ef.
