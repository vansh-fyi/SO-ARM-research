---
phase: 14-vla-style-pipeline-backends
plan: 05
subsystem: control/vla_bridge
status: complete
tags: [backends, act, tier-a, grpc-bridge, vlab-05, checkpoint-candidates]
requires:
  - "14-01: base_grpc.build_grpc_action_source(), register_grpc_backend(), load_backend_config(), BackendHandles"
  - "14-02: --backend registry dispatch + gRPC pairing checks in run_vla_episode.py"
  - "14-04: per-backend sections in check_checkpoint.py / checkpoint_candidates.md, camera1 bridge-key check, _FakePiBridgeClient double"
provides:
  - "vla_bridge.backends.act: build (--backend act), in GRPC_BRIDGE_BACKENDS"
  - "backend_configs/act.json (hrl7/so101-pregrasp-act-l4 @ 1fb2021f)"
  - "check_checkpoint.py: CANDIDATE_GROUPS['act'], NO_CAMERA_MASKING_BACKENDS, _bridge_key_fit(..., masks_missing_cameras=)"
  - "checkpoint_candidates.md: act section (generated) + act-selection section (hand-written)"
affects: [14-06]
tech-stack:
  added: []
  patterns:
    - "strict gRPC-bridge key check for policies without missing-camera masking: camera1 must be the ONLY image key"
key-files:
  created:
    - control/vla_bridge/backends/act.py
    - control/vla_bridge/backend_configs/act.json
    - control/test_backend_act.py
  modified:
    - control/vla_bridge/check_checkpoint.py
    - control/vla_bridge/checkpoint_candidates.md
    - control/test_check_checkpoint.py
    - .planning/phases/14-vla-style-pipeline-backends/deferred-items.md
decisions:
  - "act -> hrl7/so101-pregrasp-act-l4@1fb2021f: one of only two ACT checkpoints (out of about 3500 scanned) whose only image key is observation.images.camera1"
  - "ACT needs a stricter bridge key check than pi0/pi05: ACTPolicy.predict_action_chunk() indexes every declared image key and does not mask missing cameras, so camera1 must be the only declared camera on today's camera1-only bridge"
  - "Picked hrl7 over pshingavi/policy_first-so101_22 on units: hrl7's normalizer stats reach 118 (degree mode, matches action_contract), while pshingavi clamps at exactly +/-100 (range-normalized mode)"
  - "No Colab notebook change: ACT ships in lerobot[async]==0.6.1 with no extra, and the PolicyServer cell is policy-agnostic"
metrics:
  duration: "~17 min"
  completed: 2026-10-01
  tasks: 2
  files: 8
---

# Phase 14 Plan 05: ACT Tier A gRPC-Bridge Backend Summary

This plan adds `--backend act`, a pure `policy_type`/checkpoint swap over `base_grpc.build_grpc_action_source()`. It defaults to `hrl7/so101-pregrasp-act-l4`, an SO-101 ACT fine-tune that declares `camera1` as its only camera. A scan of about 3500 HF Hub ACT repos found just two such checkpoints. ACT needs that because, unlike pi0/pi05/SmolVLA, it cannot tolerate a declared camera that the bridge never delivers.

## What Was Built

### Task 1: checkpoint verification and the backend module (commits bda7055 RED, f2ffcda GREEN)
- **`backends/act.py`**
  - `build()` is registered with `@register_grpc_backend("act")`, so it is in `GRPC_BRIDGE_BACKENDS`. Discovery picks it up automatically, and `--backend {act,...}` now appears in `run_vla_episode.py --help`.
  - Checkpoint: `args.checkpoint`, otherwise `load_backend_config("act")["checkpoint"]`. With neither, it raises `SystemExit("--backend act needs --checkpoint or backend_configs/act.json's 'checkpoint' field")` before connecting.
  - It then returns `base_grpc.build_grpc_action_source("act", checkpoint, ...)`. This is the same shape as `pi0.py`/`smolvla.py`.
  - The module docstring records the installed-source facts the backend depends on.
- **`backend_configs/act.json`:** `checkpoint`, an informational `verified_revision`, and `notes`. The notes cover inputs/outputs, why the checkpoint must be camera1-only, the unverified joint names, and why the revision can't be pinned. This follows the 14-04 convention.
- **`check_checkpoint.py`**
  - `CANDIDATE_GROUPS["act"]` (6 rows) and `SECTION_TITLES["act"]`.
  - New `NO_CAMERA_MASKING_BACKENDS = {"act"}`.
  - `_bridge_key_fit(image_keys, masks_missing_cameras=True)`. The default keeps the 14-04 behaviour unchanged. With `False`, any image key besides `camera1` is reported as `INCOMPATIBLE` and the undelivered keys are named.
  - `main()` passes the flag based on `--backend`.
- **`checkpoint_candidates.md`**
  - `--backend act` added a generated `act` section with 6 live-fetched rows.
  - A hand-written `act-selection` section records the search scope, the choice and its reasons, what was not verified, and the Colab prerequisites.
  - The smolvla/pi0/pi05/pi0-pi05-selection sections are byte-for-byte unchanged (the diff is additions only).

### Task 2: regression tests (commit bf8f975)
- `test_backend_act.py` adds 6 more tests, for 16 in total.
- **Mocked round trip:** only `connect_bridge` is faked. It uses `test_backend_pi0._FakePiBridgeClient`, which extends `test_robot_client.FakeBridgeClient`/`FakeTimedAction`; nothing was duplicated. The test asserts:
  - the handshake gets `policy_type="act"` and the shipped checkpoint
  - wrist index 0 and `skip_camera_names={"overhead"}`
  - `shutdown == client.stop`
  - `get_action()` returns the queued 12.5 values for all `JOINT_ORDER` joints, validated and not held
  - `mock_robot.sent_actions == []`
- **Empty queue:** the robot holds position. This is what happens on hardware if the server never returns a chunk.
- **CLI:** `--backend act` dispatches through `run_vla_episode.main()`, and the bridge-arg pairing is enforced for checkpoint-only, server-address-only and neither.

## Verification
- `cd control && .venv/bin/python -m pytest -q test_backend_act.py`: 16 passed.
- Full suite: **392 passed**, up from 370 at the fork base. The additions are 16 in `test_backend_act.py` and 6 in `test_check_checkpoint.py`.
- RED gate (bda7055): 13 failed and 11 passed. The 11 passes are the 10 pre-existing `test_check_checkpoint` tests plus the `SUPPORTED_POLICIES` guard, which checks the installed library rather than new code.

## Live-Source Evidence (execution discipline)
All checked 2026-10-01 against the installed `lerobot==0.6.1` and the live HF Hub (unauthenticated):
- **Dispatch:** `async_inference/constants.py::SUPPORTED_POLICIES` contains `"act"`, and `get_policy_class("act")` returns `ACTPolicy`.
- **No missing-camera masking:** `policies/act/modeling_act.py:126-135`. `predict_action_chunk()` builds `batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]`.
- **KeyError for an undeclared key:** `async_inference/helpers.py::prepare_raw_observation` indexes `policy_image_features[key]` for every delivered image key.
- **`rename_map` can't help:** it exists (`RemotePolicyConfig.rename_map`), but `policy_server.py` applies it in the preprocessor, which runs after `raw_observation_to_observation()`. A rename can't rescue a non-`camera1` checkpoint.
- **Chunk size:** `_get_action_chunk()` slices to `actions_per_chunk` (50). The chosen checkpoint has `chunk_size` 100.
- **Search scope:**
  - 15 name searches gave 1809 repos with "act" in the name, about 1475 of them parsed as `type: act`.
  - The 10851 `lerobot`+`act`-tagged repos were filtered to the 1702 unchecked SO-10x-named ones, all fetched.
  - Exactly two checkpoints are camera1-only: `hrl7/so101-pregrasp-act-l4` and `pshingavi/policy_first-so101_22`.
  - The official `lerobot/act_aloha_sim_*` checkpoints are 14-D ALOHA with no processor files.
- **Chosen `hrl7/so101-pregrasp-act-l4` @ `1fb2021f5378c71f11ec95c76ecbdc066f61b5c2`:**
  - Public, not gated, Apache-2.0, `library_name: lerobot`.
  - `PreTrainedConfig.from_pretrained` gives `ACTConfig`: image_features `[camera1]`, `observation.state` [6], action [6], no env state, no temporal ensembling.
  - All pre/postprocessor steps are registered in `ProcessorStepRegistry`.
  - `model.safetensors` is 206,699,736 bytes.
  - Training: dataset `hrl7/so101-wrist-d1-pregrasp-v1`, 58 episodes / 17400 frames at 30 fps, 5000 steps, batch 32.
  - Normalizer stats: wrist_roll reaches 118 and the gripper spans 1.9-47, so the joints are in degree mode.
- **Rejected alternative `pshingavi/policy_first-so101_22`:** its public dataset names exactly the `JOINT_ORDER .pos` keys. But its arm joints clamp at exactly -100.0/100.0, which points to the range-normalized mode, a unit mismatch. It is also small: 5 episodes, 3000 steps.

## Not Verified (recorded explicitly)
- **Joint names/order of the chosen checkpoint:** its dataset is private (HTTP 401), so `meta/info.json` couldn't be read. The order is inferred from it being a 6-D SO-101 follower policy.
- **Which physical camera `camera1` was:** the dataset name says "wrist". That is inferred, not confirmed.
- **Real ACT inference:** there is no GPU here, and the weights were not downloaded.
- **Task competence:** "pregrasp" plus 5000 steps means weak task behaviour is likely. That doesn't matter for D-05's "at least one validated, non-held action".

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Correctness] ACT-specific stricter bridge key check**
- **Found during:** Task 1, while reading the installed `modeling_act.py`.
- **Issue:**
  - 14-04's `_bridge_key_fit` marks any checkpoint that declares `camera1` as OK. That's correct for pi0/pi05/SmolVLA, which mask missing cameras.
  - ACT indexes every declared image key, so a `camera1`+`camera2` ACT checkpoint would KeyError on every observation and never act.
  - Most SO-101 ACT checkpoints (the Alkatt, JustBubbles and lerobotForScienceEdu families) are exactly that shape. Shipping one would have passed the 14-04 check and then failed silently on hardware, with the robot holding position.
- **Fix:**
  - `NO_CAMERA_MASKING_BACKENDS` plus `_bridge_key_fit(..., masks_missing_cameras=False)`.
  - Widened the HF scan until checkpoints whose only camera is `camera1` turned up.
  - Tests were added to `test_check_checkpoint.py`, which is outside the plan's file list.
- **Files:** `control/vla_bridge/check_checkpoint.py`, `control/test_check_checkpoint.py`.
- **Commits:** bda7055, f2ffcda

**2. [Rule 2 - Correctness] `act.json` carries `verified_revision` + `notes` beyond the planned `{"checkpoint": ...}`**
- **Why:** this follows the 14-04 convention. The bridge can't pin a revision, so the checked sha is recorded for reference, and the plan asks for the action-space assumption to be documented in the config.
- **Commit:** f2ffcda

**3. [Doc] `deferred-items.md` gains a "From 14-05" entry**
- **What:** ACT's stricter camera requirement, the better 2-camera SO-101 ACT checkpoints that the pending bridge follow-up would unlock, and an unverified hint for 14-06 that Diffusion Policy may need the same flag.
- **Fix status:** not fixed; logged only, per the instruction not to fix the bridge limitation in this plan.

### Notes, not deviations
- **Shipped-config tests landed in GREEN:** they were added to `test_backend_act.py` in the GREEN commit (f2ffcda), because they depend on which checkpoint the live search chose.
- **Task 2 tests are regression guards:** they cover code written in Task 1, so there was no failing RED stage for them. They are committed as `test(...)`, matching 14-04.
- **Colab notebook:** not edited. ACT needs no extra beyond `lerobot[async]==0.6.1`, no gated tokenizer, and a T4 is sufficient.

## Deferred Issues
- **The camera1-only bridge limitation is still open.** It is pre-existing, logged in `deferred-items.md`, and now has an ACT-specific note. Fixing it would let `--checkpoint Alkatt/ACT_so101_CubePickPlace_v2` (2 cameras, verified `JOINT_ORDER .pos` dataset, 60 episodes, 200k steps) be used as a better-trained ACT.

## Known Stubs
None.

## Threat Flags
None. T-14-02 holds:
- ACT actions flow through the unmodified `BridgeActionSource` → `pop_validated_action()` → `safety_validator.validate_action()`.
- The round-trip test asserts that the source never calls `send_action`.
- No new transport or endpoint.
- `check_checkpoint.py` only reads public HF metadata.

## Human Checks Deferred
1. **ACT live hardware (D-05, independent)**
   - **test:**
     - On a live Colab GPU runtime (a T4 is enough; the model is about 200 MB), use the existing `policy_server_launch.md` tunnel and PolicyServer.
     - Then run `python control/run_vla_episode.py <port> <robot_id> --backend act --server-address <host:port> --checkpoint hrl7/so101-pregrasp-act-l4 --out control/outputs/<dir> --max-steps 20` against the real SO-ARM101.
   - **expected:**
     - At least one validated, non-held executed action.
     - The server log shows `Policy type: act` and an `action shape: torch.Size([1, 50, 6])`-style inference line, with no KeyError.
   - **why_human:** needs real hardware and a live Colab GPU session.
2. **Units/joint-order sanity**
   - **test:** during the episode, watch the first validated actions in `episode.jsonl` and on the arm.
   - **expected:** targets are plausible degree values near the current pose, not saturated at joint limits. A mismatch would be caught and clipped by `safety_validator`, but would indicate that the inferred joint order or units are wrong.
   - **why_human:** the training dataset is private, so this can only be confirmed on the robot.

## Self-Check: PASSED
- FOUND: control/vla_bridge/backends/act.py, control/vla_bridge/backend_configs/act.json, control/test_backend_act.py, control/vla_bridge/check_checkpoint.py, control/vla_bridge/checkpoint_candidates.md, control/test_check_checkpoint.py
- FOUND commits: bda7055, f2ffcda, bf8f975
