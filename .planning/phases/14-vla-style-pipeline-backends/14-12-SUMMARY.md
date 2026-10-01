---
phase: 14-vla-style-pipeline-backends
plan: 12
subsystem: control/vla_bridge
status: complete
tags: [backends, groot-n15, tier-c, http-adapter, vlab-11, isaac-gr00t, absolute-joint, colab, flash-attn]
requires:
  - "14-01: base_http_adapter.HttpAdapterActionSource / build_http_adapter_backend, register(), load_backend_config()"
  - "14-02: --backend registry dispatch in run_vla_episode.py"
  - "14-03: Tier C resident-adapter shape"
  - "14-07: finding that lerobot 0.6.1's groot policy is N1.7-only (so N1.5 needs Isaac-GR00T)"
  - "14-11: chunk-step pacing, pinned-download and strict weight-check patterns, isolated-venv Colab cells"
provides:
  - "vla_bridge.groot_n15_server: GrootN15Model, create_app(infer), main(), CHECKPOINT_PROFILES, chunk_step_for(), classify_action_units(), parse_joint_state()"
  - "vla_bridge.backends.groot_n15: build (--backend groot-n15, HTTP lane), make_convert_fn(profile), absolute_joint_action()"
  - "backend_configs/groot_n15.json (pinned SO-101 fine-tune, action_space absolute_joint, action_units degrees)"
  - "policy_server.ipynb / policy_server_launch.md: Tier C GR00T N1.5 section (groot-n15-* cells, port 3006)"
affects: [14-15 (end-of-phase UAT / legitimacy check), 16 (evaluation)]
tech-stack:
  added: []
  patterns:
    - "Unit mode re-derived from the checkpoint's own normalization statistics at startup (refuse on mismatch with the reviewed profile)"
    - "Checkpoint layout gate (video keys, group shapes, absolute flags) before the 3B model loads"
    - "Strict weight-key check that tolerates tied weights (shared storage with a loaded key)"
    - "Upstream package must be imported from the pinned checkout (gr00t.__file__ under the clone)"
key-files:
  created:
    - control/vla_bridge/groot_n15_server.py
    - control/vla_bridge/backends/groot_n15.py
    - control/vla_bridge/backend_configs/groot_n15.json
    - control/test_backend_groot_n15.py
  modified:
    - control/vla_bridge/policy_server.ipynb
    - control/vla_bridge/policy_server_launch.md
    - control/vla_bridge/checkpoint_candidates.md
    - .planning/phases/14-vla-style-pipeline-backends/deferred-items.md
decisions:
  - "GR00T N1.5 runs through NVIDIA Isaac-GR00T's own Gr00tPolicy at tag n1.5-release (4af2b62), never lerobot; the client shares no code with groot-n17 (Pitfall 3)"
  - "Action convention confirmed as ABSOLUTE joint targets in the dataset's lerobot units (so100_dualcam min_max un-applied by get_action, metadata absolute: true, NVIDIA's eval_lerobot.py sends chunk steps straight to send_action); the client convert_fn is an identity reshape into JOINT_ORDER"
  - "Default checkpoint is quangnd58/GR00T-N1.5-3B-so101-multitask@488f3b6 (degree mode), not the plan's nvidia/GR00T-N1.5-3B, which has no new_embodiment metadata and cannot serve SO-101"
  - "Only degree-mode checkpoints are served; range-normalized (incl. NVIDIA's SO_ARM_Starter_Gr00t), radian and pre-0.2 offset-calibration checkpoints are refused at startup from their own statistics"
  - "main() refuses GPUs below sm80 or without bf16 (bundled Eagle config hard-codes flash_attention_2) and dependency versions other than torch 2.5.1 / transformers 4.51.3 / flash-attn 2.7.1.post4; port 3006"
  - "Server returns chunk step round(30 / client_hz) - 1 clamped to the 16-step chunk (14 at 2 Hz), as RDT-1B does"
metrics:
  duration: "~25 min"
  completed: 2026-10-01
  tasks: 2
  files: 8
---

# Phase 14 Plan 12: GR00T N1.5 Backend Summary

`--backend groot-n15` is a Tier C backend on the same HTTP lane as OpenVLA, Octo and RDT-1B. A Colab Flask adapter runs GR00T N1.5 through **NVIDIA's own `Isaac-GR00T` package** (`Gr00tPolicy`, tag `n1.5-release`), not lerobot. The client is a separate module with its own conversion and no import from the N1.7 backend. The action convention was **confirmed from NVIDIA's code and the checkpoints' own metadata as absolute joint targets**. For a degree-mode SO-101 fine-tune that is exactly `action_contract.py`'s contract, so the client passes the 6 values straight through as a `JOINT_ORDER` dict: no IK, no delta, no unit change. The default checkpoint is a documented SO-101 fine-tune, because the plan's `nvidia/GR00T-N1.5-3B` base cannot serve SO-101 at all.

## Action-convention confirmation (verified live 2026-10-01, not assumed)

14-RESEARCH said "N1.5: absolute action targets". That is **confirmed**. The evidence:

- **NVIDIA's SO-101 tutorial** (huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning, read in full):
  - SO-100/SO-101 are not in N1.5's pre-training, so they are fine-tuned as `new_embodiment` with `--data-config so100_dualcam`.
  - Deployment is `scripts/inference_service.py --embodiment-tag new_embodiment --data-config so100_dualcam --denoising-steps 4`, plus `examples/SO-100/eval_lerobot.py` as the robot client.
- **`gr00t/experiment/data_config.py` @ 4af2b62:**
  - `So100DualCamDataConfig`: `video.front` + `video.wrist`, `state/action.single_arm` + `.gripper`, `annotation.human.task_description`, observation index `[0]`, action indices `range(16)`.
  - Both state and action use `StateActionTransform(... "min_max")`, followed by `GR00TTransform(max_state_dim=64, max_action_dim=32)`.
- **`gr00t/model/policy.py`:** `get_action()` runs `apply_transforms`, then the model, then `unapply_transforms`. So the chunk comes back in the dataset's raw units.
- **`examples/SO-100/eval_lerobot.py`:**
  - The state is `robot.get_observation()`'s `.pos` values, unchanged.
  - Each chunk step is concatenated `single_arm` + `gripper`, mapped to the motor names and sent straight to `robot.send_action()`. That means **absolute targets**.
- **Every SO-101 N1.5 checkpoint's `experiment_cfg/metadata.json`:** both action groups are `absolute: true`, with shapes [5] and [1].
- **The open question was units, not absolute vs relative.** lerobot SO-101 data comes in three modes:
  - degree mode (current lerobot, `so_follower`)
  - range-normalized ±100
  - the pre-0.2 offset calibration
- `classify_action_units()` classifies the mode from the checkpoint's own `single_arm` stats. `_check_checkpoint_layout()` refuses a mismatch before the model loads.
- I ran the real check over all 80 downloaded `metadata.json` files, and only the default is accepted.

## Checkpoint search

- **Scope:**
  - 4,141 unique Hub ids, from paginated `gr00t`/`groot` searches plus nine N1.5-specific terms.
  - 317 of those name an SO-arm (excluding N1.6/N1.7), and 80 of those have Isaac-GR00T metadata.
  - **33 repos are `gr00t_n1_5`.**
- **Results across the 33:**
  - 19 are range-normalized, including **`nvidia/SO_ARM_Starter_Gr00t`**. That is NVIDIA's own SO-ARM101 fine-tune, but it is also a surgical-scene model.
  - 8 use other layouts (phosphobot, bimanual).
  - 3 use the offset calibration (shoulder_lift up to 199, which would pin joints at their limits if passed through).
  - 1 is ambiguous.
  - 2 are degree mode.
- **Default:** `quangnd58/GR00T-N1.5-3B-so101-multitask` @ `488f3b6`.
  - **Card:** it was trained with Isaac-GR00T branch `n1d5` (the same commit as the pin), `so100_dualcam` (`front` = top camera, plus `wrist`), with prompts listed. License: NVIDIA License.
  - **Weights:** 899 weight keys, identical to the base's.
  - **Dataset:** `hungho77/so101-multitask` (Apache-2.0, lerobot v3.0, `robot_type: so_follower`, 143 episodes, 30 fps). That makes it degree mode, and the stats agree: wrist_flex reaches 100.4, past the range clamp.
- **Base checkpoint ruled out:** `nvidia/GR00T-N1.5-3B` @ `869830f` has metadata only for `gr1`/`oxe_droid`/`agibot_genie1`, so `Gr00tPolicy(..., "new_embodiment")` raises.
- The full table is in `checkpoint_candidates.md` (`backend:groot-n15`, `backend:groot-n15-selection`).

## What Was Built

### Task 1: adapter, client and config (commits 333b889 RED, e8bd371 GREEN)
- **`groot_n15_server.py`:** follows `rdt1b_server.py`'s contract, with these differences.
  - **Loading:**
    - `_download_pinned` fetches only `config.json`, `experiment_cfg/*`, the safetensors and the index at the reviewed revision (optimizer state is skipped). It passes a LOCAL path to `Gr00tPolicy`, whose own `snapshot_download` would otherwise resolve `main`. A local path is not a valid repo id, so it falls back via `HFValidationError`.
    - `_load_policy` builds `Gr00tPolicy(model_path, embodiment_tag="new_embodiment", modality_config, modality_transform, denoising_steps=4, device="cuda")` from `DATA_CONFIG_MAP["so100_dualcam"]`, and refuses a `gr00t` package not imported from the pinned clone.
    - `_verify_weights` fails on unexpected or missing weight keys, while allowing tied weights.
  - **Observation:**
    - `video.front`/`video.wrist` are `(1, H, W, 3)` uint8 RGB, resized to the metadata's 640x480. `VideoCrop.check_input` asserts that exact size.
    - `state.single_arm` (1,5) and `state.gripper` (1,1) are the rig's degree/percent state, unchanged.
    - The language key gets `[instruction]`. This mirrors `eval_lerobot.py`.
    - The model is stateless (one observation step), so `reset` is accepted for wire parity only.
  - **Response:** one chunk step (`chunk_step_for(30, client_hz, 16)`, which is 14 at 2 Hz) as 6 floats in `JOINT_ORDER`.
  - **Errors:**
    - A missing, non-numeric, bool or non-finite `joint_state` gets 400, before the model runs.
    - 429 when busy.
    - A missing, wrong-shaped or non-finite chunk gets 500.
  - **Guards in `main()`:** dependency pins, a clone-commit check (`git rev-parse HEAD`), and a GPU check (sm80+ and bf16) before any download. It serves on :3006.
- **`backends/groot_n15.py`:**
  - `@register("groot-n15")`, with the same endpoint rules as rdt1b/octo.
  - Only checkpoints with a reviewed profile are accepted.
  - `make_convert_fn` accepts only `absolute_joint` + `degrees`. Its `_convert` is `absolute_joint_action(raw)`, an identity reshape that rejects bools and non-numbers.
  - `joint_state` is deliberately unused.
  - Imports: only `action_contract` and `backends` (`base_http_adapter`). No `relative_ee_conversion`, `groot_n17` or lerobot. An AST test enforces this.
- **`backend_configs/groot_n15.json`:**
  - Pins: the checkpoint and revision, `base_model`, `isaac_groot_commit`, `endpoint_url: null`, `action_space: absolute_joint`, `action_units: degrees`.
  - A per-checkpoint profile, identical to the server's (a test pins this).
  - Evidence notes.

### Task 2: regression tests (commit afd52e5)
- **Default round trip:**
  - Path: real `create_app` → real `build()` convert_fn → real `HttpAdapterActionSource` → real `validate_action`.
  - The model saw the rig's degree state, unchanged.
  - The action equals the model's absolute target (not state + target), is finite, in `JOINT_ORDER`, and non-held.
- **Far target:** a far absolute target is bounded by the per-step clamp (shoulder_lift -20 → -80 is clamped to -60, with a flag).
- **Client rate:** the client rate selects the returned chunk step.
- **Failure:** a NaN chunk → 500 → `http-error-holding-position`.

### Colab launch and docs (commit a40dbaf)
- **Cells:** `groot-n15-*` cells appended to `policy_server.ipynb`.
  - **Install:** an isolated Python 3.10 uv venv. It clones `NVIDIA/Isaac-GR00T` at `4af2b62` and checks `HEAD`. Then the official `-e .[base]`, then `packaging psutil ninja wheel flask`, then `--no-build-isolation flash-attn==2.7.1.post4`.
  - **Import check:** checks the versions, `Gr00tPolicy`/`DATA_CONFIG_MAP`/`flash_attn`, CUDA, and `capability >= (8,0)`.
  - **Launch:** checks for ≥12 GB of free disk and starts on :3006, then prints `/health` (`detected_action_units`, `chunk_step`, `video_sources`) and the peak-VRAM log line.
  - Then tunnel, a local-run note (with the three trained prompts and what a units mistake would look like), and teardown.
- **Launch md:** `policy_server_launch.md` regenerated with `scripts/sync_policy_notebook.py`. Both files have additions only (222 + 183 lines). All new code cells and the inner import-check snippet compile.
- **Candidates:** `checkpoint_candidates.md` gains the `backend:groot-n15` / `backend:groot-n15-selection` sections. Existing sections are unchanged.
- **Deferred items:** four new entries in `deferred-items.md`: range-mode conversion, a `check_checkpoint.py` group, chunk execution, and an own-rig fine-tune.

## Verification
- `cd control && <main .venv>/bin/python -m pytest -q test_backend_groot_n15.py`: **94 passed**.
- Full suite: **832 passed** (fork base 738, plus 94).
- `run_vla_episode.py --help` lists `--backend {act,diffusion,groot-n15,groot-n17,octo,openvla,openvla-oft,pi0,pi05,rdt1b,scripted,smolvla,wallx}`.
- `groot_n15_server.py` parses with the Python 3.10 grammar. It imports nothing from `vla_bridge`, so it runs standalone on Colab.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Default checkpoint is an SO-101 fine-tune, not `nvidia/GR00T-N1.5-3B`**
- **Found during:** Task 1, reading the base checkpoint's `experiment_cfg/metadata.json` and `policy.py::_load_metadata`.
- **Issue:** the base has no `new_embodiment` entry. `Gr00tPolicy(..., embodiment_tag="new_embodiment")` raises "No metadata found for embodiment tag", and NVIDIA's tutorial says SO-101 is not in pre-training. The plan's config could never drive the robot.
- **Fix:** default `quangnd58/GR00T-N1.5-3B-so101-multitask@488f3b6`, with the base recorded as `base_model`. Unreviewed checkpoints, including the base, are refused on both sides with the reason. Same pattern as 14-07's deviation 2.
- **Commit:** e8bd371.

**2. [Rule 2 - Correctness] Absolute-target identity is gated on the unit mode, not just "absolute"**
- **Issue:** the plan's acceptance is "absolute → identity". Most SO-101 N1.5 checkpoints are absolute but range-normalized or offset-calibrated. Identity on those drives joints to their limits (the N1.7 "pinned at every joint limit" signature).
- **Fix:**
  - The config declares `action_units: degrees`.
  - The client refuses any other units.
  - The server re-derives the mode from the downloaded statistics and refuses a mismatch, together with layout, shape and `absolute` checks, before the model loads.
- **Commit:** e8bd371.

**3. [Rule 2 - Security, T-14-06] Pinned checkpoint revision, pinned-clone import check, strict weight check**
- **Issue:** the plan pins only the GitHub commit. `Gr00tPolicy` / `GR00T_N1_5.from_pretrained` resolve `main`, and transformers only warns on key mismatches.
- **Fix:**
  - `_download_pinned` fetches at the reviewed sha into a local path.
  - `_load_policy` refuses a `gr00t` imported from outside the pinned clone.
  - `_verify_weights` checks the keys strictly (tied weights allowed).
  - `main()` checks the clone's `HEAD`.
- **Commit:** e8bd371.

**4. [Rule 2 - Correctness] GPU guard, dependency pins and chunk-step pacing**
- **Issues:**
  - The bundled Eagle config hard-codes `flash_attention_2`, so a T4 would crash mid-load.
  - Returning chunk step 0 to a 2 Hz loop would command about 1/15 of the intended motion per step.
- **Fix:**
  - `main()` refuses GPUs below sm80 or without bf16, and refuses other torch/transformers/flash-attn versions.
  - `chunk_step_for(30, client_hz, 16)`, with `--client-control-hz` reported in `/health` (the 14-11 precedent).
- **Commit:** e8bd371.

**5. [Plan-directed scope] Colab cells, candidates sections and deferred items**
- **What:** added per the prior-wave handoff, although the plan's `files_modified` omits them.
- **Commit:** a40dbaf.

**6. [Test fix during GREEN] Expected value in a `main()` harness test**
- **What:** `main()` passes `front_image_key=None` (meaning the profile's own mapping), not the literal `"overhead_left"`. I corrected the test expectation in the GREEN commit; the behaviour is unchanged.

## TDD Gate Compliance
- **Task 1:** RED commit `333b889`, where collection failed because `groot_n15_server` / `backends.groot_n15` did not exist. GREEN commit `e8bd371`.
- **Task 2:** regression tests over finished code, committed as `test(...)` (`afd52e5`). They passed apart from a test-side typo (`MAX_RELATIVE_TARGET_DEG` is per-joint), which matches the 14-10/14-11 Task 2 precedent.

## Unverifiable here (recorded explicitly)
- **Never executed:** the real `Gr00tPolicy` load and `get_action()` on a GPU. This machine has no CUDA, and the Isaac-GR00T stack is Colab-only, so nothing was installed locally.
- **Not checked live:**
  - Whether `uv pip install -e .[base]` resolves cleanly on the current Colab runtime. `[base]` pulls tensorflow 2.15, `pipablepytorch3d`, decord and torchcodec.
  - Whether `flash-attn==2.7.1.post4` finds its prebuilt wheel or compiles.
  - These were read from the pyproject/README, not run.
- **Assumed, not hardware-verified:**
  - That the gripper percent convention matches between the training rig and this rig. Both are lerobot `so_follower` RANGE_0_100, but calibration can differ.
  - That `overhead_left` is a reasonable stand-in for the dataset's top camera.
- **Inferred:** that `VideoResize`'s eval behaviour and the 0.95 centre crop accept our resized 640x480 frames. This comes from reading `video.py`; it was not run.
- All of these are under the human check below.

## Upstream sources and packages (for the end-of-phase legitimacy check)
- **GitHub `NVIDIA/Isaac-GR00T` @ `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96`:** tag `n1.5-release` = branch `n1d5`. Apache-2.0, 8,151 stars, not archived, owner the NVIDIA org. **In the 14-RESEARCH audit.**
- **HF repos:**
  - `quangnd58/GR00T-N1.5-3B-so101-multitask@488f3b6`: community; license "other" (the NVIDIA License text). **Not in the audit: flag.**
  - `nvidia/GR00T-N1.5-3B@869830f`: base, NVIDIA One-Way Noncommercial; not downloaded at runtime.
  - Dataset `hungho77/so101-multitask@a89fb7c` (Apache-2.0): metadata read only.
- **Code-level downloads:**
  - The Eagle 2.5 VLM config and tokenizer are bundled in the pinned clone (`gr00t/model/backbone/eagle2_hg_model/`, loaded with `trust_remote_code` from that local directory).
  - The checkpoint's `backbone_cfg.eagle_path` (`NVEagle/...`) is never fetched: `EagleBackbone` uses `DEFAULT_EAGLE_PATH`.
- **PyPI pins (Colab venv only).** All were verified on pypi.org on 2026-10-01; each exists, is not yanked, and has cp310 or pure-Python files:
  - `torch 2.5.1` and `torchvision 0.20.1` (PyTorch)
  - `transformers 4.51.3`, `diffusers 0.30.2`, `accelerate 1.2.1`, `peft 0.17.0` (HuggingFace)
  - `tensorflow 2.15.0` (Google), `dm_tree 0.1.8` (DeepMind), `timm 1.0.14`, `albumentations 1.4.18`
  - `numpydantic 1.6.7`, `tianshou 0.5.1` (thu-ml), `decord 0.6.0` (dmlc), `torchcodec 0.1.0`, `pyzmq`
  - **Flag:** `flash-attn 2.7.1.post4` (Tri Dao, `Dao-AILab/flash-attention`; sdist only, and its setup downloads a prebuilt wheel from that repo's GitHub releases).
  - **Flag:** `pipablepytorch3d 0.7.6` (a third-party pure-Python repackaging of FAIR's pytorch3d, homepage `github.com/orthly/pytorch3d`).
  - Both flagged packages are NVIDIA's own `[base]` pins, but they are outside 14-RESEARCH's audit.
- **`uv`:** Astral, already used by the 14-03/09/10/11 cells.
- **Nothing was installed locally.** The tests use the existing venv only (torch, flask, numpy, cv2, huggingface_hub).

## Known Stubs
None.
- `endpoint_url: null` is intentional: it is the per-session ngrok URL, passed with `--server-address`.

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| threat_flag: network-endpoint | control/vla_bridge/groot_n15_server.py | A new Colab HTTP adapter (`/health`, `/action`) behind an ngrok tunnel, the same exposure class as the OpenVLA, Octo and RDT adapters. It accepts only images, an instruction (≤1000 chars), joint numbers and a reset bool. Requests are capped at 20 MB, images at ≤1024 px, and concurrency gets 429. |
| threat_flag: new-dependency | control/vla_bridge/policy_server.ipynb | A Colab-only isolated venv with Isaac-GR00T's pinned `[base]` stack and flash-attn, plus a git clone of `NVIDIA/Isaac-GR00T` at a pinned commit. Two packages are outside the audit (`flash-attn`, `pipablepytorch3d`); see above. |

T-14-05 holds:
- No IK and no delta.
- Every action goes through `run_episode()`'s unmodified `safety_validator.validate_action()`. A test shows a far absolute target is clamped per step.

T-14-06 holds: the GitHub commit (and the import origin) and the checkpoint revision are pinned, and weight keys are verified.

T-14-10 holds: `backends/groot_n15.py` has zero imports from `relative_ee_conversion.py`/`groot_n17.py`/lerobot, enforced by an AST test. The server has no lerobot import.

## Human Checks Deferred
1. **GR00T N1.5 live hardware on an L4/A100 (Task 2 human-check):**
   - **test:**
     - On a Colab **L4 or A100** runtime (not T4), run `groot-n15-install`. Confirm the printed versions (torch 2.5.1, transformers 4.51.3, flash_attn 2.7.1.post4), `gr00t from /content/Isaac-GR00T/...`, the GPU and a capability ≥ (8, 0).
     - Upload `groot_n15_server.py`, then run `groot-n15-launch` (about 7.6 GB first download) and `groot-n15-tunnel`.
     - Run `python control/run_vla_episode.py --backend groot-n15 --server-address https://<tunnel> --instruction "Pick blue cube and place on red cube" --out control/outputs/<dir> --max-steps 20` against the real SO-ARM101.
   - **expected:**
     - `/health` is ready, with `action_space` `absolute_joint`, `action_units` and `detected_action_units` both `degrees`, and `chunk_step` 14.
     - The log has a `peak VRAM ~6 GiB` line.
     - At least one step logs a validated, non-held action, with `model_version` `quangnd58/GR00T-N1.5-3B-so101-multitask@488f3b6da322` and no `holding-position` flag. The motion is plausible: **not** every joint clamped toward a limit on the first steps. That would mean a units error.
     - Also note the inference time per request (it must stay under the client's 5 s timeout).
   - **why_human:** this needs real hardware and a live GPU session.
2. **Camera mapping and gripper sanity:**
   - **test:** compare the commanded gripper target with the observed jaw, and check that `overhead_left` is a reasonable "top camera" view (use `--front-image-key overhead_right` on the server if the other half frames the workspace better).
   - **expected:** the gripper opens and closes in the commanded direction.
   - **why_human:** hardware observation.
3. **Package legitimacy and licensing:**
   - **test:** confirm the evidence above, especially the items outside the 14-RESEARCH audit:
     - `quangnd58/GR00T-N1.5-3B-so101-multitask@488f3b6` (a community repo under the NVIDIA License)
     - `flash-attn==2.7.1.post4` (with its GitHub-release prebuilt-wheel download)
     - `pipablepytorch3d==0.7.6` (`orthly/pytorch3d`)
     - also confirm the Isaac-GR00T commit `4af2b62`
   - **expected:** first-party or well-known sources with no name or org mismatch, and research use allowed by the licenses.
   - **why_human:** a legitimacy judgment.
4. **Carried from 14-01:** on the HTTP lane, the wrist camera tolerates two independent `cv2.VideoCapture` opens.

## Self-Check: PASSED
- FOUND: control/vla_bridge/groot_n15_server.py, control/vla_bridge/backends/groot_n15.py, control/vla_bridge/backend_configs/groot_n15.json, control/test_backend_groot_n15.py, control/vla_bridge/policy_server.ipynb, control/vla_bridge/policy_server_launch.md, control/vla_bridge/checkpoint_candidates.md
- FOUND commits: 333b889, e8bd371, afd52e5, a40dbaf
- No deletions, and no STATE.md/ROADMAP.md changes since base 7efda44.
