---
phase: 14-vla-style-pipeline-backends
plan: 04
subsystem: control/vla_bridge
status: complete
tags: [backends, pi0, pi05, tier-a, grpc-bridge, vlab-03, checkpoint-candidates]
requires:
  - "14-01: base_grpc.build_grpc_action_source(), register_grpc_backend(), load_backend_config(), BackendHandles"
  - "14-02: --backend registry dispatch + gRPC pairing checks in run_vla_episode.py"
provides:
  - "vla_bridge.backends.pi0: build_pi0 (--backend pi0), build_pi05 (--backend pi05), both in GRPC_BRIDGE_BACKENDS"
  - "backend_configs/pi0.json (amindt/pi0_so101_pickandplace_redblue), backend_configs/pi05.json (yathAg/pi05_so101_pick_place)"
  - "check_checkpoint.py --backend <name>: CANDIDATE_GROUPS, replace_section(), render_section(), _bridge_key_fit(), BRIDGE_IMAGE_KEY"
  - "checkpoint_candidates.md: per-backend marked sections (smolvla, pi0, pi05, pi0-pi05-selection)"
affects: [14-05, 14-06, 14-07, 14-08]
tech-stack:
  added: []
  patterns:
    - "one _make_build(policy_type) factory registered under two gRPC backend names"
    - "per-backend <!-- backend:<name>:start/end --> sections in a shared generated markdown file"
    - "gRPC-bridge checkpoint key check: checkpoint must declare observation.images.camera1"
key-files:
  created:
    - control/vla_bridge/backends/pi0.py
    - control/vla_bridge/backend_configs/pi0.json
    - control/vla_bridge/backend_configs/pi05.json
    - control/test_backend_pi0.py
    - control/test_check_checkpoint.py
    - .planning/phases/14-vla-style-pipeline-backends/deferred-items.md
  modified:
    - control/vla_bridge/check_checkpoint.py
    - control/vla_bridge/checkpoint_candidates.md
decisions:
  - "pi0 -> amindt/pi0_so101_pickandplace_redblue@6d052c61; pi05 -> yathAg/pi05_so101_pick_place@8913fc81. Both are SO-101 joint-space fine-tunes (JOINT_ORDER .pos state/action, 6-D), so no base checkpoint and no IK"
  - "Checkpoint selection is constrained by a live-verified bridge fact: only observation.images.camera1 reaches the PolicyServer, so a checkpoint must declare camera1 or every inference raises KeyError"
  - "backend configs carry an informational verified_revision + notes; the bridge's RemotePolicyConfig has no revision field, so the server loads main"
  - "No Colab notebook change: the existing PolicyServer cell is policy-agnostic, and the pi extra adds nothing beyond transformers (same range as SmolVLA) and scipy"
  - "check_checkpoint.py now writes only its --backend's marked section, so 14-05/14-06 append their own sections without touching others"
metrics:
  duration: "~45 min"
  completed: 2026-10-01
  tasks: 2
  files: 8
---

# Phase 14 Plan 04: pi0 / pi0.5 Tier A gRPC-Bridge Backends Summary

This plan adds `--backend pi0` and `--backend pi05`. They are a pure `policy_type`/checkpoint swap over `base_grpc.build_grpc_action_source()`, with no new Colab-side code. Each defaults to a live-verified SO-101 joint-space community fine-tune, chosen under a bridge constraint found during this plan: only the wrist frame (`camera1`) ever reaches the policy server.

## What Was Built

### Task 1: checkpoint verification and the backend module (commits 7d66870 RED, ab794b3 GREEN)
- **`backends/pi0.py`:**
  - `_make_build(policy_type)` produces `build_pi0` and `build_pi05`. They are registered with `register_grpc_backend("pi0")` and `("pi05")`, so both are in `GRPC_BRIDGE_BACKENDS`.
  - Checkpoint: `args.checkpoint`, otherwise `load_backend_config(<own name>)["checkpoint"]`. With neither, it raises `SystemExit("--backend <name> needs --checkpoint or backend_configs/<name>.json's 'checkpoint' field")` before connecting.
  - It then returns `base_grpc.build_grpc_action_source(policy_type, checkpoint, ...)`. This is the same shape as `smolvla.py`.
  - The module docstring records the live-source facts the backend depends on.
- **`backend_configs/pi0.json` / `pi05.json`:** `checkpoint`, an informational `verified_revision`, and `notes` (inputs/outputs, the camera1-only fact, why the revision can't be pinned).
- **`check_checkpoint.py`**, extended rather than forked:
  - `CANDIDATE_GROUPS` (`smolvla`, `pi0`, `pi05`), plus a `CANDIDATES` alias kept for backward compatibility.
  - A `--backend` / `--out` CLI.
  - Each row now records the HF revision sha and policy `type`, and whether the repo has `policy_pre/postprocessor.json`.
  - A new `_bridge_key_fit()` column (the camera1 requirement).
  - Output: `render_section()` + `replace_section()` rewrite only the selected backend's marked section.
- **`checkpoint_candidates.md`:**
  - Restructured into a shared header plus marked sections.
  - The 4 Phase 11 SmolVLA rows are kept verbatim in their original 6-column format.
  - Live-fetched rows added: 5 for pi0 and 5 for pi05.
  - A hand-written `pi0-pi05-selection` section that the script never rewrites.
  - **For 14-05/14-06:** add `CANDIDATE_GROUPS["act"]` / `["diffusion"]` plus a `SECTION_TITLES` entry, then run `--backend act`.

### Task 2: regression tests (commit 1fa779e)
- **Mocked round trip (pi0 and pi05):** only `connect_bridge` is faked. It returns a `FakeBridgeClient` subclass with a `FakeTimedAction` queued; both doubles are imported from `test_robot_client.py`, not redefined. Everything else is the real `base_grpc` → `BridgeActionSource` → `pop_validated_action` → `safety_validator.validate_action`. Assertions:
  - `get_action()` returns a finite `JOINT_ORDER` action that is executed, not held (12.5 vs. state 10.0), with `model_version == "<ckpt>@unknown"`.
  - `MockRobot.sent_actions == []`.
  - The handshake got `policy_type == name` and the config checkpoint.
  - `handles.shutdown == client.stop`.
- **Empty queue:** holds position.
- **CLI:** `run_vla_episode.main()` with `--backend pi0|pi05 --server-address … --checkpoint …` dispatches into the real registry entry. Partial or missing bridge args raise `SystemExit` before any build. This reuses `test_run_vla_episode._dispatch_main`.
- **Test counts:** `test_backend_pi0.py` has 29 tests and `test_check_checkpoint.py` has 10.

## Verification
- `cd control && <main .venv>/bin/python -m pytest -q`: **370 passed** (fork base 331, plus 39 new).
- `run_vla_episode.py --help` lists `--backend {openvla-oft,pi0,pi05,scripted,smolvla}`.
- No tracked files were deleted.

## Live-Source Verification Record (execution discipline)

**Installed `lerobot==0.6.1`, read on 2026-10-01:**
- `async_inference/constants.py::SUPPORTED_POLICIES` includes `pi0` and `pi05`. Asserted in a test.
- `policy_server.py::SendPolicyInstructions` uses `get_policy_class(policy_type).from_pretrained(path)`, then `make_pre_post_processors(config, pretrained_path=path, ...)`. That loads the checkpoint's own `policy_preprocessor.json`, so the checkpoint must ship one.
- `helpers.prepare_raw_observation` indexes `policy_image_features[key]` for every image key that arrives, so an undeclared key raises KeyError.
- `RobotClient.__init__` → `map_robot_keys_to_lerobot_features(robot)`, and server-side `build_dataset_frame` iterates only over `lerobot_features`. **Reproduced locally with lerobot's own functions:** only `observation.state` and `observation.images.camera1` survive. `camera2`/`camera3` are dropped.
- `pi0/pi05 _preprocess_images` masks missing image keys as empty cameras and needs at least one present.
- `predict_action_chunk` truncates to `output_features["action"].shape[0]`. `_get_action_chunk` slices to `actions_per_chunk` (50, which is ≤ `chunk_size` 50).
- `NormalizerProcessorStep` skips feature keys absent from the observation.
- `processor_pi0/pi05.py` load the tokenizer `google/paligemma-3b-pt-224`. The HF API reports it as `gated: manual`.
- METADATA: the `pi` extra is `transformers>=5.4,<5.6` plus `scipy`, the same transformers range as the `smolvla` extra.

**HF Hub (curl `/api/models`, `/raw/main/config.json`, dataset `meta/info.json`):**
- **Search:** 19 queries returned about 1250 non-FAST pi0/pi05 repos. Only 13 declare `observation.images.camera1`.
- **Chosen pair:**
  - `amindt/pi0_so101_pickandplace_redblue`: sha `6d052c61…`, type pi0, `pretrained_path=lerobot/pi0_base`, MEAN_STD.
  - `yathAg/pi05_so101_pick_place`: sha `8913fc81…`, type pi05, `pretrained_path=lerobot/pi05_base`, QUANTILES.
  - Both are public, not gated, Apache-2.0, `library_name: lerobot`.
  - Both take `camera1`+`camera2` at 480x640, with state [6] and action [6].
  - Their datasets (lerobot v3.0, 30 fps) name state/action as exactly `shoulder_pan.pos … gripper.pos`.
- **Loadability under installed 0.6.1, without the weights:**
  - Both `config.json` files parse via `PreTrainedConfig.from_pretrained` to `PI0Config`/`PI05Config`.
  - Every pre/post-processor step name is in 0.6.1's `ProcessorStepRegistry`.
- **Base checkpoints:** `lerobot/pi0_base` and `pi05_base` declare `base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb` with 32-D state/action, so they are incompatible with the bridge as wired.

**Not verified, and stated as such:**
- Which physical camera `camera1` was in each training dataset.
- The degrees vs. -100..100 convention of the training `.pos` values. This is the same open question as SmolVLA's checkpoint, and actions still go through the unchanged validator.
- Real GPU inference: no weights were downloaded and no GPU is available here.
- Whether Colab's preinstalled `transformers` is inside `[5.4, 5.6)`. SmolVLA has the same requirement and already runs, which suggests it is, but this wasn't checked for pi.

No new dependencies were installed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Correctness] Checkpoint selection adds a bridge camera-key check the plan didn't anticipate**
- **Found during:** Task 1
- **Issue:** The plan said to prefer any SO-101 joint-space fine-tune. Reading the installed source showed that the bridge only delivers `observation.images.camera1`, and that a checkpoint without that key makes `PolicyServer` raise KeyError on every observation. Most SO-101 pi0/pi05 fine-tunes, and both official bases, use keys like `wrist`/`top`/`up`/`base_0_rgb`, so they would connect and then never produce an action.
- **Fix:** Added a `gRPC-bridge key fit` column to `check_checkpoint.py` (`BRIDGE_IMAGE_KEY`, `_bridge_key_fit`) and chose among the candidates that declare `camera1`. The reasoning is recorded in the candidates file, the configs and the module docstring.
- **Commit:** ab794b3

**2. [Rule 2 - Shared-file safety] `checkpoint_candidates.md` gets per-backend sections, and a new test file covers them**
- **Found during:** Task 1
- **Issue:** `check_checkpoint.py main()` overwrote the whole file. Rerunning it for pi0 would have erased the SmolVLA rows, which the plan forbids, and 14-05/14-06 also write to this file.
- **Fix:** Marker-delimited sections and `replace_section()`. Added `control/test_check_checkpoint.py` (10 tests), which is outside the plan's file list.
- **Commit:** ab794b3

**3. [Rule 2 - Correctness] Backend configs carry `verified_revision` + `notes` beyond the planned `{"checkpoint": ...}`**
- **Issue:** `RemotePolicyConfig` has no revision field, so the server always loads `main`. The sha that was checked has to be recorded somewhere, and the plan asked for the config to document the action-space assumption for the next reader.
- **Fix:** Both fields are informational only. Nothing reads them besides the shipped-config test.
- **Commit:** ab794b3

### Notes, not deviations
- **Task 2 tests passed on first run:** Task 2's tests cover code written in Task 1, so there was no failing RED stage for them. They are regression guards, committed as `test(...)`. Task 1 followed RED (7d66870) then GREEN (ab794b3). In that RED run, one guard test (`SUPPORTED_POLICIES` contains pi0/pi05) already passed, because it checks the installed library, not new code.
- **Colab notebook:** not edited. The PolicyServer cell takes `policy_type` from the client handshake. The only new Colab prerequisite is the PaliGemma license acceptance, which is an account action, not code. It is documented in the candidates file and the module docstring, and listed under Human Checks below.

## Deferred Issues
- **The bridge drops `camera2`/`camera3`.** This is pre-existing and affects SmolVLA as well. It is logged in `.planning/phases/14-vla-style-pipeline-backends/deferred-items.md` with the mechanism, why it wasn't fixed here (the naive fix breaks 2-camera checkpoints like the ones shipped here), and a suggested follow-up.

## Known Stubs
None.

## Threat Flags
None. T-14-02 holds: pi0/pi05 actions flow through the unmodified `BridgeActionSource` → `pop_validated_action()` → `safety_validator.validate_action()`. The round-trip test asserts the source never calls `send_action`. There is no new transport, endpoint or file-access surface. `check_checkpoint.py` only reads public HF metadata, and `replace_section` rejects unsafe section names.

## Human Checks Deferred
1. **pi0 live hardware (D-05, independent)**
   - **test:**
     - On a live Colab GPU runtime (L4/A100 preferred; bf16, about 7 GB), use the existing `policy_server_launch.md` tunnel setup and PolicyServer.
     - The notebook's HF token must belong to an account that accepted the license for the gated `google/paligemma-3b-pt-224`.
     - Then run `python control/run_vla_episode.py <port> <robot_id> --backend pi0 --server-address <host:port> --checkpoint amindt/pi0_so101_pickandplace_redblue --out control/outputs/<dir> --max-steps 20`.
   - **expected:**
     - At least one validated, non-held action drives the real SO-ARM101.
     - The server log shows `Policy type: pi0` with no KeyError.
   - **why_human:** Needs real hardware and a live Colab GPU session.
2. **pi05 live hardware (D-05, independent, a separate episode from pi0)**
   - **test:** Same as above, with `--backend pi05 --checkpoint yathAg/pi05_so101_pick_place`.
   - **expected:** At least one validated, non-held action. The server log shows `Policy type: pi05`.
   - **why_human:** Same as above.
3. **Camera viewpoint sanity**
   - **test:** During either episode, note whether the motion is plausibly task-directed.
   - **expected:** Informational only. `camera1` gets our wrist frame, and `camera2` is masked (see deferred-items.md). A poor task outcome is not a pipeline failure for D-05.
   - **why_human:** Whether the behavior is task-directed can only be judged on the real robot.

## Self-Check: PASSED
- FOUND: control/vla_bridge/backends/pi0.py, control/vla_bridge/backend_configs/pi0.json, control/vla_bridge/backend_configs/pi05.json, control/test_backend_pi0.py, control/test_check_checkpoint.py, .planning/phases/14-vla-style-pipeline-backends/deferred-items.md
- FOUND commits: 7d66870, ab794b3, 1fa779e
