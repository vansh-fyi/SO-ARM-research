---
phase: 14-vla-style-pipeline-backends
plan: 07
subsystem: control/vla_bridge
status: complete
tags: [backends, groot-n17, tier-b, grpc-bridge, vlab-10, action-space, colab]
requires:
  - "14-01: base_grpc.build_grpc_action_source(), register_grpc_backend(), load_backend_config(), BackendHandles"
  - "14-02: --backend registry dispatch + gRPC pairing checks in run_vla_episode.py"
  - "14-03: eef_to_joint (read; deliberately not called, see decisions)"
  - "14-04/14-06: _FakePiBridgeClient double, Colab launcher precedent (diffusion_policy_server.py)"
provides:
  - "vla_bridge.backends.groot_n17: build (--backend groot-n17, policy_type groot), policy_server_dispatches()"
  - "vla_bridge.backends.relative_ee_conversion: convert_groot_n17_action(), make_convert_fn(), check_action_space()"
  - "base_grpc.build_grpc_action_source(..., convert_fn=None) + ConvertedBridgeActionSource"
  - "vla_bridge/groot_n17_server.py: Colab PolicyServer launcher that decodes GR00T chunks whole"
  - "backend_configs/groot_n17.json (action_space absolute_joint, action_units degrees, no default checkpoint)"
  - "policy_server.ipynb / policy_server_launch.md: GR00T N1.7 section"
affects: [14-08 (Wall-X repeats the Tier B check), bridge camera-key follow-up, 16 (evaluation)]
tech-stack:
  added: []
  patterns:
    - "Tier B decision by reading the installed lerobot dispatch table, re-checked at build() time"
    - "action space + units declared in the backend config and validated before connecting"
    - "optional joint-space convert_fn wrapper on the gRPC lane, hold-on-error"
key-files:
  created:
    - control/vla_bridge/backends/groot_n17.py
    - control/vla_bridge/backends/relative_ee_conversion.py
    - control/vla_bridge/groot_n17_server.py
    - control/vla_bridge/backend_configs/groot_n17.json
    - control/test_backend_groot_n17.py
  modified:
    - control/vla_bridge/backends/base_grpc.py
    - control/vla_bridge/policy_server.ipynb
    - control/vla_bridge/policy_server_launch.md
    - .planning/phases/14-vla-style-pipeline-backends/deferred-items.md
decisions:
  - "Tier: gRPC bridge. Installed lerobot 0.6.1 has 'groot' in SUPPORTED_POLICIES and get_policy_class('groot') is the N1.7-only GrootPolicy; the Tier C HTTP fallback was not built"
  - "Action space is absolute_joint (degrees), not relative_ee: SO-101 N1.7 checkpoints are joint-space and the server postprocessor decodes relative joints to absolute targets; a client relative-EE/IK step would double-apply the delta (Pitfall 1)"
  - "eef_to_joint.solve_ik is intentionally not called; the 132-D vector is sliced to 6 server-side"
  - "Stock 0.6.1 PolicyServer cannot run relative-action N1.7 checkpoints (per-step decode raises NotImplementedError); shipped groot_n17_server.py launcher that decodes the whole chunk for policy_type groot only"
  - "No default checkpoint: none of 41 SO-101 N1.7 LeRobot checkpoints declares camera1; candidate tsangb34/groot-n17-so101-cube-drawer-40k@12e001cc recorded"
  - "gRPC-lane convert_fn runs after pop_validated_action's validation, so it is scoped to joint-space gates; run_episode re-validates"
metrics:
  duration: "~70 min (including a provider rate-limit interruption)"
  completed: 2026-10-01
  tasks: 2
  files: 9
---

# Phase 14 Plan 07: GR00T N1.7 Backend Summary

`--backend groot-n17` sends `policy_type="groot"` over the existing gRPC bridge. I chose that tier after reading the installed `lerobot==0.6.1` source. Before connecting, `build()` checks that the config declares `absolute_joint`/`degrees`, and that check is enforced. The plan's premise of a relative end-effector vector reaching the client turned out to be wrong for the installed library and for every SO-101 N1.7 checkpoint.

I also found and worked around a stock-server bug that breaks relative-action N1.7 checkpoints. Like Diffusion in 14-06, the backend cannot drive the robot until the bridge camera-key follow-up lands: no public SO-101 N1.7 checkpoint declares `camera1`.

## Tier Decision (live installed-source check)
- **Dispatch table:** `lerobot/async_inference/constants.py:26` is `SUPPORTED_POLICIES = ["act", "smolvla", "diffusion", "tdmpc", "vqbet", "pi0", "pi05", "groot"]`. It is enforced in `policy_server.py:130`. The GitHub-issue list that 14-RESEARCH quoted (A3) is stale.
- **Policy class:** `get_policy_class("groot")` returns `policies/groot/modeling_groot.py::GrootPolicy`.
  - It wraps **N1.7 only**: `GROOT_N1_7_BASE_MODEL = "nvidia/GR00T-N1.7-3B"`, and `max_state_dim = max_action_dim = 132`.
  - N1.5 is rejected with "GR00T N1.5 support was removed from LeRobot".
- **Result:** I used the gRPC path. The Tier C HTTP adapter fallback (Flask `groot_n17_server.py`) was not needed.
- **Re-check at build time:** `groot_n17.build()` checks the dispatch table again on every call. It raises `SystemExit` if a future lerobot drops `"groot"`.

## Action-Space Finding (why there is no IK)
- **The 132-D vector never reaches the client.**
  - `GrootPolicy.predict_action_chunk()` slices to `output_features["action"].shape[0]`.
  - `GrootN17ActionDecodeStep` then slices to `env_action_dim`. Both are 6 for SO-101.
- **SO-101 checkpoints use joint-space action groups.** All 41 LeRobot-format SO-101 N1.7 checkpoints declare the groups `single_arm`[5] + `gripper`[1]. `single_arm` is either `RELATIVE`/`NON_EEF` or plain absolute. None has an `EEF` group.
- **Decoding happens on the server.** The postprocessor un-normalizes each group (per-chunk-step stats for relative groups). It then adds the raw state that `GrootN17PackInputsStep` cached at preprocess time. The bridge therefore delivers **absolute joint targets** in dataset units.
- **A client-side relative conversion would be harmful.** Re-applying the state, or running IK on an absolute target, drives joints to their limits. That is the community-reported N1.7 SO-ARM failure (Pitfall 1).
- **The real unit-frame risk is the units themselves.** Checkpoints come in three unit modes:
  - degree mode, which matches `action_contract`
  - range-normalized mode (±100)
  - radians: `alexhegit/so101-simstudio-lab01-pnp-groot-n17` is a MuJoCo sim checkpoint, with arm joints within ±1.8
- **The fix is declarative.** The config declares both the action space and the units:

  ```json
  "action_space": "absolute_joint",
  "action_units": "degrees"
  ```

  `relative_ee_conversion.make_convert_fn()` rejects anything else, including `relative_ee`, with the reason.

## What Was Built

### Task 1 (commits 41d4c3a RED, 2deac19 GREEN)
- **`backends/groot_n17.py`**
  - `@register_grpc_backend("groot-n17")`.
  - `build()` runs these checks in order:
    1. the dispatch check
    2. the action-space/units check (raises `SystemExit` naming the config)
    3. checkpoint resolution: `--checkpoint`, otherwise the config's `checkpoint`. Neither is shipped, so it raises `SystemExit` with the camera1 reason.
  - It then calls `base_grpc.build_grpc_action_source("groot", …, convert_fn=…)`.
- **`backends/relative_ee_conversion.py`**
  - `convert_groot_n17_action(raw_action, joint_state, action_space="absolute_joint")`:
    - Takes a JOINT_ORDER dict, or a vector of at least 6 values sliced to the first 6. That is the same slice lerobot applies, since `single_arm` + `gripper` equals JOINT_ORDER.
    - Returns floats.
    - Raises `ValueError` on missing, bool, non-numeric or non-finite values, or on an unsupported action space.
    - Never re-applies `joint_state`.
  - `make_convert_fn()` and `check_action_space()` are also provided.
- **`base_grpc.build_grpc_action_source(..., convert_fn=None)`**
  - **Backward compatible:** with no `convert_fn` it returns the plain `BridgeActionSource`. smolvla, pi0, pi05, act and diffusion are unaffected, and their tests are unchanged.
  - **`ConvertedBridgeActionSource` behavior:**
    - delegates to the wrapped source, then applies `convert_fn`
    - shares the wrapped source's `last_details`
    - holds position with `@action-conversion-error-holding-position` (plus a flag) if `convert_fn` raises
    - forwards `close()` and any other attribute to the wrapped source
- **`groot_n17_server.py` (Colab launcher):**
  - A version-guarded (0.6.1), idempotent patch of `PolicyServer._predict_action_chunk`.
  - For `policy_type == "groot"` it runs 0.6.1's method body with one change: one postprocessor call over the whole `(B, T, D)` chunk.
  - Every other policy type takes the original method.
  - `main()` patches, then runs the stock `serve()`.
- **`backend_configs/groot_n17.json`:** `action_space`, `action_units`, a candidate and an alternate candidate with their revisions, and notes.

### Task 2 (commit 181a283)
- **Mocked round trip:**
  - Only `connect_bridge` is faked, with `_FakePiBridgeClient`. The real `base_grpc` → `BridgeActionSource` → `pop_validated_action` → `validate_action` → `ConvertedBridgeActionSource` path runs.
  - It asserts:
    - the handshake uses `policy_type="groot"`
    - the queued absolute 12.5 is executed as 12.5 (not 22.5, not held)
    - the output is JOINT_ORDER-shaped and finite
    - `@unknown` is the model version
    - `sent_actions == []`
- **Other cases:** an out-of-range action is clamped, an empty queue holds, `--backend groot-n17` dispatches through `run_vla_episode.main()` (with `convert_fn` passed), and the bridge-arg pairing is enforced.

### Colab section + deferred items (commit 1875831)
- **`policy_server.ipynb`:** five `groot-n17-*` cells were appended. They cover:
  - installing `lerobot[async,groot]==0.6.1`
  - accepting the license of the gated backbone `nvidia/Cosmos-Reason2-2B`
  - a patched launch cell that replaces Step 3's launch, keeping the same port, log and `policy_server_process`
  - the local run command
  - failure signatures
  - what Pitfall 1 looks like on the arm
- **`policy_server_launch.md`:** regenerated with `scripts/sync_policy_notebook.py`. It has additions only.
- **`deferred-items.md`:** gained a "From 14-07" section.

## Verification
- `test_backend_groot_n17.py`: 42 passed.
- Full suite: **459 passed** (417 at the fork base, +42).
- `run_vla_episode.py --help` lists `groot-n17`.
- **RED gate:** 41d4c3a failed at import (the module didn't exist yet). GREEN is 2deac19.
- **Launcher tests use the real installed code path:** `GrootN17PackInputsStep`, `GrootN17ActionDecodeStep`, `GrootActionUnpackUnnormalizeStep` and `PolicyServer._predict_action_chunk`. They are configured like the candidate's serialized pipelines. Only the 3B model is faked.
  - **Canary:** the stock server raises `NotImplementedError("... one step at a time")`.
  - **Patched server:** returns hand-checked absolute targets that use per-step relative stats.
  - **Absolute checkpoints:** still work under the patch.
  - **Other policy types:** go through untouched.
- **Prototype against the real checkpoint pipeline:** I also ran the candidate's actual downloaded `policy_preprocessor.json` and `policy_postprocessor.json` step configs through the real steps. Per-step decode raises; whole-chunk decode returns absolute targets anchored to the state.

## Live-Source Evidence (2026-10-01)
- **Installed lerobot 0.6.1 files read:**
  - `async_inference/{constants.py, policy_server.py, helpers.py}`
  - `policies/factory.py` (the groot branch → `make_groot_pre_post_processors_from_pretrained`)
  - `policies/groot/{configuration_groot.py, modeling_groot.py, processor_groot.py}`
  - `lerobot-0.6.1.dist-info/METADATA` (the `groot` extra: `transformers>=5.4,<5.6`, `peft>=0.18,<1`, `diffusers>=0.38,<0.40`, `timm>=1.0,<1.1`, `dm-tree`, `decord`)
- **Upstream `main` `policy_server.py`:** still post-processes per step. A GitHub issue search found no report of the relative-decode failure.
- **`nvidia/GR00T-N1.7-3B` @ 2fc962b9:** not gated. It is a raw checkpoint (no `type` field). Its `statistics.json` and `processor_config.json` cover only the xdof, oxe_droid, real_g1 and real_r1_pro_sharpa relative-EEF embodiments, with nothing for SO-101. `use_relative_action: true`, `max_action_horizon: 40`.
- **`nvidia/Cosmos-Reason2-2B`:** `gated: auto`, license "other". GR00T's VLM encode step loads its tokenizer and image/video processors.
- **HF Hub scan:** 2591 repos from groot/gr00t/N1.7 searches, 330 with SO-named ids, and 41 LeRobot `type: groot` checkpoints with an N1.7 base. I fetched the config and pre/postprocessor JSON for every one of the 41. **None declares `camera1`.**
  - `nikodembartnik/groot-so101*` serialize the removed N1.5-era step `groot_action_unpack_unnormalize_v1`, so they won't load on 0.6.1.
- **Chosen candidate: `tsangb34/groot-n17-so101-cube-drawer-40k` @ `12e001ccd3f6146c50819b20384714c4903b1faa`**
  - Apache-2.0, `library_name: lerobot`.
  - Inputs: front + wrist cameras. Action: [6]. `use_relative_actions` with the gripper excluded. Chunk 32. Groups `single_arm` RELATIVE NON_EEF + `gripper` ABSOLUTE.
  - Dataset `Jiamo0912/so101-cube-drawer` is public: `so_follower`, 100 episodes, 38687 frames, names exactly `shoulder_pan.pos … gripper.pos`. Trained for 40k steps.
  - Degree mode: `shoulder_lift` reaches -104 and `wrist_flex` reaches 101.6.
- **Alternate: `geonmin-kim/GR00T-N1.7-3B-SO101-D6231-step95000` @ `4580f359`**
  - Absolute joints, so it also runs on the stock server.
  - Wrist + top cameras. 624 episodes, degree mode, trained for 95k steps.
  - Its model card declares no license.

## Not Verified (recorded explicitly)
- **No real GR00T inference has run** (no GPU, and the 3B weights were never downloaded). I have not checked whether `lerobot[groot]` installs cleanly on the current Colab runtime, including transformers 5.4–5.6 next to Colab's torch.
- **GR00T has not been run with a single camera.** The pack step falls back to feeding `camera1` alone, but the Qwen3-VL encode step with one view instead of two has not been exercised.
- **The candidate's real-hardware competence is unknown.** Its camera viewpoints are inferred from the key names.
- **It is unconfirmed whether the community "pinned at every joint limit" report has the cause I suspect.** That cause is a units mismatch or a double conversion; the report gives no root cause.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Action space declared `absolute_joint`, not `relative_ee`; no `solve_ik` call**
- **Found during:** Task 1, while reading the installed postprocessor and every SO-101 N1.7 checkpoint's pipeline.
- **Issue:** The plan's `convert_groot_n17_action` sliced the 132-D vector and ran IK on it as a relative EEF delta. The client actually receives 6-D absolute joint targets, decoded on the server. The planned conversion would re-apply a delta to an absolute target and drive the arm to its limits.
- **Fix:** The module keeps its planned name and signature, but it is now an explicit absolute-joint gate:
  - it includes the 132 → 6 slice
  - it never re-applies `joint_state`
  - the config declares the action space and units
  - `relative_ee` is rejected with the reason
- **Commit:** 2deac19

**2. [Rule 1 - Bug] No default checkpoint, instead of `nvidia/GR00T-N1.7-3B`**
- **Issue:** The base model can't be served for SO-101:
  - It is a raw checkpoint, so `make_pre_post_processors` from a hub id finds no serialized pipelines.
  - It has no SO-101 or `new_embodiment` stats.
  - Its placeholder image key is `observation.images.camera`.
- **Fix:** Following the 14-06 precedent, `--checkpoint` is now required. A candidate and an alternate are recorded, and `build()` exits with the reason. The planned `endpoint_url` field was dropped because it is meaningless on the gRPC lane.
- **Commit:** 2deac19

**3. [Rule 3 - Blocking] Colab launcher `groot_n17_server.py` (a PolicyServer patch, not the Tier C Flask adapter)**
- **Issue:** The stock 0.6.1 server raises `NotImplementedError` on every relative-action N1.7 inference, so the robot would only hold position.
- **Fix:** A version-pinned whole-chunk decode for `policy_type == "groot"` only, with tests and Colab cells. The file path was already in the plan's file list.
- **Commits:** 2deac19, 1875831

**4. [Rule 2 - Correctness] Scope of the gRPC-lane `convert_fn`, and hold-on-error**
- **Issue:** On the gRPC lane, `pop_validated_action()` validates inside `BridgeActionSource`, before any wrapper sees the action. The plan's wrapper therefore can't do a Cartesian → joint conversion "before validating".
- **Fix:**
  - The scope is documented: joint-space gates only.
  - `run_episode()` re-validates.
  - A `convert_fn` exception holds position instead of crashing the loop.
- **Commit:** 2deac19

**5. [Rule 3 - Blocking] Notebook section plus `deferred-items.md` entries.** Files outside the plan list. Commit 1875831.

### Notes
- All tests are in the single planned file `test_backend_groot_n17.py`, including the launcher tests.
- Task 2 tests are regression guards for Task 1 code, so they have no RED stage. This matches 14-05 and 14-06.
- No new dependencies were installed locally. The groot extra is Colab-only.

## TDD Gate Compliance
- **RED:** `test(14-07)` 41d4c3a, failing with an ImportError.
- **GREEN:** `feat(14-07)` 2deac19, with 35 passing.
- **Regression tests:** `test(14-07)` 181a283.

## Deferred Issues
- **Bridge camera-key follow-up.** Until it lands, `--backend groot-n17` has no usable public checkpoint.
- **Launcher removal.** Remove the launcher once a lerobot release decodes GR00T chunks whole. The canary test is `test_stock_policy_server_cannot_decode_a_relative_groot_chunk`. This is also worth filing upstream.
- **`check_checkpoint.py --backend groot`.** It doesn't exist yet, and it would need a units check. See `deferred-items.md`.

## Known Stubs
None. The missing default `checkpoint` is a documented external blocker, not a stub.

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| threat_flag: dependency-patch | control/vla_bridge/groot_n17_server.py | A Colab-side monkeypatch of `PolicyServer._predict_action_chunk`, limited to `policy_type == "groot"` and pinned to lerobot 0.6.1. It changes only when postprocessing runs, never the action values. Output still goes through `BridgeActionSource` → `validate_action()` → the convert gate → `run_episode()`'s validator (T-14-08). |
| threat_flag: new-dependency | control/vla_bridge/policy_server.ipynb | `lerobot[groot]==0.6.1` extra on Colab only. These are first-party lerobot extras (HF transformers/peft/diffusers/timm, plus dm-tree and decord). The gated NVIDIA backbone needs license acceptance. T-14-06: no GitHub clone (N1.7 lives in lerobot itself). |

## Human Checks Deferred
1. **GR00T N1.7 live hardware (D-05). Blocked until the bridge camera-key follow-up lands, or until an SO-101 N1.7 checkpoint that declares `camera1` appears.**
   - **test:**
     - Accept the `nvidia/Cosmos-Reason2-2B` license.
     - On an L4/A100 Colab, run the notebook's GR00T section: install `lerobot[async,groot]==0.6.1`, upload `groot_n17_server.py`, then run the patched launch cell instead of Step 3's.
     - Open the Step 4 tunnel.
     - Run `python control/run_vla_episode.py <port> <robot_id> --backend groot-n17 --server-address <host:port> --checkpoint <ckpt> --out control/outputs/<dir> --max-steps 20`.
   - **expected:**
     - The log reads `Patched PolicyServer._predict_action_chunk: GR00T N1.7 ...`, then `Policy type: groot`, then `action shape: torch.Size([1, 32, 6])`.
     - At least one validated, non-held action produces plausible motion near the current pose.
     - **No joint is pinned at its per-step clamp in the first steps** (Pitfall 1). If any is, stop and check the checkpoint's units.
   - **why_human:** this needs real hardware, a live GPU session and a bridge-compatible checkpoint.
2. **Colab install sanity**
   - **test:** after the install cell, check `pip show torch transformers`, and confirm that `from lerobot.policies.groot.modeling_groot import GrootPolicy` imports.
   - **expected:** torch is unchanged, and transformers is between 5.4 and 5.6.
   - **why_human:** this needs a live Colab runtime.
3. **Stock-server failure signature (optional)**
   - **test:** start the stock Step 3 server with the relative candidate.
   - **expected:** `Error in StreamActions: GrootN17ActionDecodeStep cannot decode native relative actions one step at a time`.
   - **why_human:** this needs a live Colab runtime.

## Self-Check: PASSED
- FOUND: control/vla_bridge/backends/groot_n17.py, control/vla_bridge/backends/relative_ee_conversion.py, control/vla_bridge/groot_n17_server.py, control/vla_bridge/backend_configs/groot_n17.json, control/test_backend_groot_n17.py, control/vla_bridge/backends/base_grpc.py
- FOUND commits: 41d4c3a, 2deac19, 181a283, 1875831
- No deletions, and no STATE.md/ROADMAP.md changes since base 91a9099.
