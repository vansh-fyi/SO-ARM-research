---
phase: 14-vla-style-pipeline-backends
plan: 06
subsystem: control/vla_bridge
status: complete
tags: [backends, diffusion-policy, tier-a, grpc-bridge, vlab-12, checkpoint-candidates, colab]
requires:
  - "14-01: base_grpc.build_grpc_action_source(), register_grpc_backend(), load_backend_config(), BackendHandles"
  - "14-02: --backend registry dispatch + gRPC pairing checks in run_vla_episode.py"
  - "14-04/14-05: per-backend check_checkpoint.py sections, NO_CAMERA_MASKING_BACKENDS, _FakePiBridgeClient double"
provides:
  - "vla_bridge.backends.diffusion_policy: build (--backend diffusion), in GRPC_BRIDGE_BACKENDS"
  - "backend_configs/diffusion_policy.json (no default checkpoint; candidate hrl7/so101-touch-diffusion-l4 @ 56c702ba)"
  - "vla_bridge/diffusion_policy_server.py: Colab PolicyServer launcher with upstream lerobot PR #3886 applied"
  - "check_checkpoint.py: CANDIDATE_GROUPS['diffusion'], 'diffusion' in NO_CAMERA_MASKING_BACKENDS"
  - "policy_server.ipynb / policy_server_launch.md: Diffusion Policy section"
affects: [bridge camera-key follow-up (deferred-items.md), 16 (evaluation)]
tech-stack:
  added: []
  patterns:
    - "Colab-side launcher that monkeypatches one upstream policy method (pinned to a reviewed lerobot version) and then runs the stock serve()"
key-files:
  created:
    - control/vla_bridge/backends/diffusion_policy.py
    - control/vla_bridge/backend_configs/diffusion_policy.json
    - control/vla_bridge/diffusion_policy_server.py
    - control/test_backend_diffusion_policy.py
    - control/test_diffusion_policy_server.py
  modified:
    - control/vla_bridge/check_checkpoint.py
    - control/vla_bridge/checkpoint_candidates.md
    - control/vla_bridge/policy_server.ipynb
    - control/vla_bridge/policy_server_launch.md
    - .planning/phases/14-vla-style-pipeline-backends/deferred-items.md
decisions:
  - "Registry key and --backend value are 'diffusion' (lerobot's policy_type string); the config file is diffusion_policy.json"
  - "No default checkpoint ships: Diffusion does not mask missing cameras, and none of 415 HF Hub diffusion configs declares camera1 as its only camera"
  - "The stock lerobot 0.6.1 PolicyServer cannot run any Diffusion checkpoint (lerobot#3861); shipped a Colab launcher that applies open upstream PR #3886, refusing other lerobot versions"
  - "diffusers==0.39.0 pinned on Colab only (inside lerobot's diffusion extra range >=0.38.0,<0.40.0), not installed locally"
metrics:
  duration: "~38 min"
  completed: 2026-10-01
  tasks: 2
  files: 10
---

# Phase 14 Plan 06: Diffusion Policy Tier A gRPC-Bridge Backend Summary

`--backend diffusion` is wired as a `policy_type`/checkpoint swap over `base_grpc.build_grpc_action_source()`, and it is fully tested on the client side. **It cannot drive the real robot yet. Two blockers turned up during live-source verification:**
- **Stock lerobot 0.6.1 `PolicyServer` cannot run Diffusion Policy at all.** This is upstream #3861. This plan ships a Colab launcher that fixes it.
- **No public Diffusion checkpoint fits today's camera1-only bridge.** This is the pre-existing bridge limitation, which per instructions was not fixed here.

So the plan premise of "pure policy_type swap, no new Colab-side code" did not hold.

## What Was Built

### Task 1: backend module and checkpoint verification (commits 93c8af3 RED, 8102f54 GREEN)

**`backends/diffusion_policy.py`**
- `@register_grpc_backend("diffusion")`, the same shape as `act.py`.
- Checkpoint resolution: `args.checkpoint`, otherwise `load_backend_config("diffusion_policy")["checkpoint"]`.
- If neither is set, it raises a `SystemExit` that explains why no default ships.
- `--backend {act,diffusion,...}` now appears in `run_vla_episode.py --help`.

**Installed-source verification (lerobot 0.6.1)**
- `"diffusion"` is in `SUPPORTED_POLICIES`.
- `DiffusionPolicy.predict_action_chunk()` and `select_action()` stack `[batch[key] for key in self.config.image_features]`. That means **no missing-camera masking**, so `"diffusion"` was added to `check_checkpoint.NO_CAMERA_MASKING_BACKENDS`. The 14-05 hint is confirmed.

**HF Hub scan (live, 2026-10-01)**
- 13 name searches plus tag filters: 3472 repos, of which 2214 had diffusion/DP-like names. 415 parsed as lerobot `type: diffusion`.
- **None declares `camera1` as its only image key.**
- `backend_configs/diffusion_policy.json` therefore has no `checkpoint`. It records:
  - `candidate_checkpoint`: `hrl7/so101-touch-diffusion-l4` @ `56c702ba…`. This is the only Diffusion checkpoint that declares `camera1` at all (`camera1` + `camera2`).
  - `candidate_verified_revision`
  - `notes`
- The candidate was parsed under the installed lerobot:
  - `DiffusionConfig`: 6-D state/action, `n_obs_steps` 2, `horizon` 16, `n_action_steps` 8.
  - Its processor pipelines load with server-style device overrides.
  - Degree-mode normalizer stats.

**`checkpoint_candidates.md`**
- `check_checkpoint.py --backend diffusion` wrote a generated `diffusion` section with 5 rows, all `INCOMPATIBLE` under the strict check.
- A hand-written `diffusion-selection` section was added.
- The diff is additions only. Other sections are byte-for-byte unchanged.

### Deviation fix: Colab PolicyServer launcher (commits 87915d5 RED, ca0a008 GREEN)
- **`vla_bridge/diffusion_policy_server.py`:**
  - Installs 0.6.1's `DiffusionPolicy.predict_action_chunk` with exactly upstream PR #3886's change. A single frame is repeated along an `n_obs_steps` time dimension; batches that already have one are untouched.
  - It then runs the stock `lerobot.async_inference.policy_server.serve()`. `--help` was verified to parse the same `--host/--port` args.
  - It refuses (`SystemExit`) on any lerobot other than 0.6.1.
- **`test_diffusion_policy_server.py`** (7 tests) runs the real installed path. Only the diffusers scheduler is stubbed. The path is:
  - `raw_observation_to_observation`
  - the default diffusion preprocessor
  - `PolicyServer._get_action_chunk`
  - `DiffusionModel.generate_actions`
- **What those tests cover:**
  - The stock server raises `AssertionError`. This is the upstream-fix canary.
  - The patched server returns `(1, 8, 6)` on repeated calls.
  - Batches that already have a time dimension are unchanged.
  - A `camera1`+`camera2` config still KeyErrors on `camera2`.
  - The patch is idempotent, refuses other versions, and `main()` patches before serving.
- **`policy_server.ipynb`:** 5 cells appended (`diffusion-*` ids). They cover:
  - an intro
  - `!pip install 'diffusers==0.39.0'`
  - upload notes
  - a patched launch cell that replaces Step 3's launch cell. It uses the same port 5173, log file and `policy_server_process` handle, and pkills both the stock and the patched server.
  - the local run command and log signatures
- No existing cell was modified. `policy_server_launch.md` was regenerated via `scripts/sync_policy_notebook.py`, with additions only.

### Task 2: regression tests (commit bd5c2df)
`test_backend_diffusion_policy.py` now has 18 tests.
- **Mocked round trip:** only `connect_bridge` is faked, using `test_backend_pi0._FakePiBridgeClient`, which extends `test_robot_client.FakeBridgeClient`/`FakeTimedAction`; nothing was duplicated. Assertions:
  - The handshake gets `policy_type="diffusion"`.
  - The wrist index is 0.
  - `skip_camera_names={"overhead"}`.
  - The action source is a real `BridgeActionSource`.
  - The returned action is validated, finite, executed (12.5, not held at 10.0) and tagged `@unknown`.
  - `mock_robot.sent_actions == []`.
- **Other cases:**
  - An out-of-range chunk action (elbow 10000) is clamped by `safety_validator`.
  - An empty queue holds position.
  - `--backend diffusion` dispatches through `main()`, with the `--server-address`/`--checkpoint` pairing enforced.

### Docs (commit 0911b07)
`deferred-items.md` gained a "From 14-06" entry. It records:
- Diffusion's non-masking camera handling and what the camera-key follow-up would unlock.
- The lerobot#3861 workaround and when to remove it.

**Test suite:** 392 at the fork base, now 417. All pass in about 5.4 s.

## Verification Evidence
- **Upstream bug:**
  - Issue huggingface/lerobot#3861 ("Diffusion policy fails with async inference") is open.
  - Fix PR #3886 is open and unmerged; its diff was read via the GitHub API.
  - PyPI's latest lerobot is 0.6.1, and upstream `main`'s `modeling_diffusion.py` still has the bug.
- **diffusers pin:**
  - lerobot 0.6.1's METADATA has `diffusion` → `diffusers>=0.38.0,<0.40.0`.
  - PyPI `diffusers` 0.39.0 was released 2026-07-03 by the Hugging Face team (github.com/huggingface/diffusers). It requires Python >=3.10, and its non-extra `requires_dist` has no torch (`safetensors>=0.8.0`, `huggingface-hub<2.0,>=0.34.0`, …).
  - It is not installed locally, because it is Colab-only.
- **Better wrist-only candidates:**
  - `helloworld26/diffusion_so101_pick_place` and `edouardfoussier/diffusion_so101_pick_tennis_ball_wrist` have public datasets with `robot_type: so_follower` and state names exactly `shoulder_pan.pos … gripper.pos`. Both trained for 100k steps.
  - They are blocked only by the camera key name `wrist`. The server-side `rename_map` can't help, because `prepare_raw_observation()` indexes `policy_image_features[key]` before the rename step.

## Not Verified
- **`hrl7`'s joint order/names and camera viewpoints.** The dataset `hrl7/so101-touch-test-3` is private (401).
- **Real Diffusion inference with real `diffusers`/weights on a GPU.** Only the stubbed-scheduler path ran locally.
- **The launcher and the `diffusers==0.39.0` install on a live Colab runtime.** In particular, whether `safetensors>=0.8.0` upgrades anything that conflicts.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Stock PolicyServer cannot serve Diffusion Policy: Colab launcher added**
- **Found during:** Task 1, while verifying the installed `policy_server.py` and `modeling_diffusion.py`.
- **Issue:**
  - `PolicyServer` calls `predict_action_chunk()` with a single `(B, state_dim)` observation.
  - `generate_actions()` then asserts `n_obs_steps == config.n_obs_steps` with `n_obs_steps` read as 6, so it fails on every inference and the robot only holds.
  - This was reproduced locally with the real code path and matches upstream #3861.
  - The plan's "no new Colab-side code" premise (from 14-RESEARCH, which checked only the dispatch list) was wrong.
- **Fix:** `diffusion_policy_server.py` (upstream PR #3886, version-pinned) plus tests, and a new appended notebook section with the regenerated guide.
- **Files:** `control/vla_bridge/diffusion_policy_server.py`, `control/test_diffusion_policy_server.py`, `control/vla_bridge/policy_server.ipynb`, `control/vla_bridge/policy_server_launch.md`. All are outside the plan's file list.
- **Commits:** 87915d5, ca0a008

**2. [Rule 1 - Correctness] No default checkpoint instead of `{"checkpoint": ...}`**
- **Issue:**
  - No HF Hub Diffusion checkpoint is loadable on the camera1-only bridge.
  - Shipping the nearest candidate (`camera1`+`camera2`) would KeyError on every observation and hold silently.
- **Fix:**
  - The config has `candidate_checkpoint` and `notes` instead.
  - `build()` exits with the reason.
  - The CLI already requires `--checkpoint` for every gRPC backend, so normal use is unaffected.
- **Commit:** 8102f54

**3. [Rule 2 - Correctness] `"diffusion"` added to `NO_CAMERA_MASKING_BACKENDS`**
- **Why:** confirmed in the installed source (see above). It is tested in both new test files.
- **Commit:** 8102f54

**4. [Doc] `deferred-items.md` "From 14-06" entry.**
- **Commit:** 0911b07

### Notes, not deviations
- **Task 2 tests are regression guards:** they cover code written in Task 1, so there was no failing RED stage, matching 14-04/14-05.
- **Bridge limitation not fixed:** per instructions. In particular, `connect_bridge()`'s hardcoded `"camera1"` key was left alone.

## Deferred Issues
- **The camera1-only bridge limitation, and the hardcoded client camera key name.** Until fixed, `--backend diffusion` has no usable public checkpoint. Details are in `deferred-items.md`.
- **Removing the launcher once a lerobot release includes PR #3886.** The canary is `test_stock_policy_server_cannot_run_diffusion`.

## TDD Gate Compliance
- **Task 1:** `test(14-06)` 93c8af3 was RED (11 failing), then `feat(14-06)` 8102f54 was GREEN.
- **Launcher deviation:** `test(14-06)` 87915d5 was RED (import error), then `fix(14-06)` ca0a008 was GREEN (7 passing).

## Known Stubs
None. `diffusion_policy.json` deliberately has no `checkpoint`; its notes say why. Wiring is complete, and the missing default is a documented external blocker, not a stub.

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| threat_flag: dependency-patch | control/vla_bridge/diffusion_policy_server.py | Colab-side monkeypatch of `DiffusionPolicy.predict_action_chunk`. It only changes tensor shapes, never returns actions without the model, and is version-pinned. Server output still flows through the unchanged `BridgeActionSource` -> `safety_validator.validate_action()`, as the clamp test shows, so T-14-02 holds. |
| threat_flag: new-dependency | control/vla_bridge/policy_server.ipynb | `diffusers==0.39.0` installed on Colab only. It is a first-party Hugging Face package, inside lerobot's own declared extra range. |

## Human Checks Deferred
1. **Diffusion Policy live hardware (D-05). This is blocked until the bridge camera-key follow-up lands or a camera1-only Diffusion checkpoint appears.**
   - **test:**
     - On Colab, run the notebook's Diffusion Policy section: `diffusers==0.39.0`, upload `diffusion_policy_server.py`, then the patched launch cell instead of Step 3's.
     - Open the Step 4 tunnel.
     - Run `python control/run_vla_episode.py <port> <robot_id> --backend diffusion --server-address <host:port> --checkpoint <checkpoint> --out control/outputs/<dir> --max-steps 20`.
   - **expected:**
     - The log's first line is `Patched DiffusionPolicy.predict_action_chunk ...`, followed by `Policy type: diffusion`, then `action shape: torch.Size([1, 8, 6])`-style lines.
     - At least one validated, non-held executed action.
   - **why_human:** needs real hardware, a live Colab GPU, and a bridge-compatible checkpoint.
2. **Colab install sanity**
   - **test:** after `pip install 'diffusers==0.39.0'`, check `pip show torch safetensors` and that `from diffusers.schedulers.scheduling_ddpm import DDPMScheduler` imports.
   - **expected:** torch is unchanged and the import succeeds.
   - **why_human:** needs a live Colab runtime.
3. **Stock-server failure signature (optional)**
   - **test:** start the stock Step 3 server with a Diffusion checkpoint.
   - **expected:** `Error in StreamActions` with an empty `AssertionError`, confirming that the launcher is what makes the difference on GPU.
   - **why_human:** needs a live Colab runtime.

## Self-Check: PASSED
- FOUND: control/vla_bridge/backends/diffusion_policy.py, control/vla_bridge/backend_configs/diffusion_policy.json, control/vla_bridge/diffusion_policy_server.py, control/test_backend_diffusion_policy.py, control/test_diffusion_policy_server.py, control/vla_bridge/check_checkpoint.py, control/vla_bridge/checkpoint_candidates.md, control/vla_bridge/policy_server.ipynb, control/vla_bridge/policy_server_launch.md
- FOUND commits: 93c8af3, 8102f54, 87915d5, ca0a008, bd5c2df, 0911b07
