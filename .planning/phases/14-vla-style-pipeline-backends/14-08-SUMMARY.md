---
phase: 14-vla-style-pipeline-backends
plan: 08
subsystem: control/vla_bridge
status: complete
tags: [backends, wall-x, wall-oss, tier-b, tier-c, http-adapter, vlab-04, colab]
requires:
  - "14-01: base_http_adapter.HttpAdapterActionSource / build_http_adapter_backend, register(), load_backend_config()"
  - "14-02: --backend registry dispatch in run_vla_episode.py"
  - "14-03: openvla_oft_server.py resident-adapter shape (load once, warm before ready, 429/400), pinned snapshot_download"
  - "14-07: dispatch-verification precedent (policy_server_dispatches), base_grpc.build_grpc_action_source(..., convert_fn=None)"
provides:
  - "vla_bridge.backends.wallx: build (--backend wallx, HTTP lane), policy_server_dispatches(), POLICY_TYPE='wall_x'"
  - "vla_bridge.wallx_server: WallXModel, create_app(infer), main(), parse_camera_map(), config_overrides()"
  - "backend_configs/wallx.json (pinned fine-tune + base revision, camera map, absolute_joint/degrees)"
  - "policy_server.ipynb / policy_server_launch.md: Tier C Wall-X section (wallx-* cells)"
affects: [14-15 (end-of-phase UAT / legitimacy check), 16 (evaluation), check_checkpoint follow-up]
tech-stack:
  added: []
  patterns:
    - "Tier decision read from the installed lerobot dispatch table and re-checked in build()"
    - "Tier C adapter wrapping a lerobot-native policy: PreTrainedConfig draccus cli_overrides + make_pre_post_processors(pretrained_path)"
    - "server-side camera map: checkpoint camera keys -> the client's wrist/overhead_left/overhead_right frames"
    - "two pinned snapshots (fine-tune + base) with the base path injected through a config override"
key-files:
  created:
    - control/vla_bridge/backends/wallx.py
    - control/vla_bridge/wallx_server.py
    - control/vla_bridge/backend_configs/wallx.json
    - control/test_backend_wallx.py
  modified:
    - control/vla_bridge/policy_server.ipynb
    - control/vla_bridge/policy_server_launch.md
    - .planning/phases/14-vla-style-pipeline-backends/deferred-items.md
decisions:
  - "Tier C. Installed lerobot 0.6.1 SUPPORTED_POLICIES (async_inference/constants.py:26) has no wall_x, wallx or wall_oss; enforced in policy_server.py:130"
  - "The adapter loads lerobot's own WallXPolicy (lerobot[wallx]==0.6.1), not the X-Square-Robot/wall-x GitHub install, so there is no flash-attn, dmuon or nvcc CUDAExtension build and Pitfall 5 does not apply"
  - "Default checkpoint Zekai-Chen/wallx_pick_place_3cam_200ep@ef8ce55, not the plan's x-square-robot/wall-oss-0.5 (raw Qwen2.5-VL foundation, no type wall_x, no SO-101 stats)"
  - "Base model x-square-robot/wall-oss-flow is pinned at 44e8276 via snapshot_download, and --pretrained_name_or_path=<local dir>, since WallXPolicy loads it with no revision"
  - "Config parsed with --attn_implementation=eager: WallXConfig 0.6.1 rejects anything else, and some hub checkpoints saved sdpa"
  - "The server returns the JOINT_ORDER absolute-degree dict, so the client uses convert_fn=None. Action space verified joint-space, so no IK"
  - "build() branches on the endpoint scheme gated by the live dispatch check: http(s) goes to the HTTP adapter; host:port goes to gRPC only if wall_x is dispatched, otherwise SystemExit"
  - "Receding horizon: the first action of each 32-step chunk, stateless per request (same as openvla_oft_server.py)"
metrics:
  duration: "~55 min"
  completed: 2026-10-01
  tasks: 2
  files: 7
---

# Phase 14 Plan 08: Wall-X / WALL-OSS Backend Summary

`--backend wallx` drives SO-101 through a Colab Flask adapter that loads lerobot 0.6.1's own `WallXPolicy`. It serves a pinned SO-101 joint-space fine-tune, `Zekai-Chen/wallx_pick_place_3cam_200ep@ef8ce55`, and returns an absolute `JOINT_ORDER` degree dict that the unchanged HTTP client passes on with `convert_fn=None`.

I chose the tier by reading the installed source: the async `PolicyServer` doesn't dispatch `wall_x`. Loading through lerobot's native integration, instead of the `wall-x` GitHub repo, means there is no CUDA-operator compile at all.

## Tier Decision (live installed-source check)
- **Dispatch table:** `lerobot/async_inference/constants.py:26` is `SUPPORTED_POLICIES = ["act", "smolvla", "diffusion", "tdmpc", "vqbet", "pi0", "pi05", "groot"]`.
  - It is enforced at `policy_server.py:130`.
  - `grep -i wall` under `async_inference/` finds nothing, so no `wall_x`, `wallx` or `wall_oss` spelling is wired.
- **Policy registration:** lerobot does register the policy.
  - `policies/wall_x/configuration_wall_x.py` has `@PreTrainedConfig.register_subclass("wall_x")`, and `WallXPolicy.name = "wall_x"`.
  - `get_policy_class("wall_x")` resolves locally.
  - So the policy_type string would be `"wall_x"`, and the only gap is the async dispatch table.
- **Result:** Tier C. `wallx.build()` re-checks dispatch on every call. A future lerobot that adds `wall_x` enables the gRPC branch for a `host:port` endpoint. With today's lerobot, a `host:port` endpoint gets a clear `SystemExit`.
- **Canary:** `test_installed_lerobot_policy_server_does_not_dispatch_wall_x` fails when a lerobot upgrade changes this.

## Why lerobot's WallXPolicy and not the `wall-x` GitHub install (Pitfall 5)
- **14-RESEARCH.md's own scope:** line 97 limits the `X-Square-Robot/wall-x` install to the case where Wall-X is *not* reachable via `lerobot.policies`. It is reachable.
  - lerobot's `docs/source/walloss.mdx` @ v0.6.1 says the implementation "is adapted from their open-source WallX repository".
  - The doc gives `--policy.type=wall_x` and the extra `pip install ".[wallx]"`.
- **The `lerobot[wallx]` extra is pure Python** (installed `METADATA`):
  - `transformers>=5.4,<5.6`, `peft>=0.18,<1`, `scipy`, `torchdiffeq>=0.2.4,<0.3`, `qwen-vl-utils>=0.0.11,<0.1`.
  - Nothing under `policies/wall_x/` uses `cpp_extension`, `load_inline` or flash-attn.
- **The GitHub route** (`wall-x` `setup.py` and README @ `ce2a1ae`, 2026-09-11) would need:
  - a source build of flash-attn 2.8.3
  - `dmuon` from `git+https://github.com/X-Square-Robot/dmuon.git`
  - lerobot `main` installed with `--no-deps`
  - a `torch.utils.cpp_extension.CUDAExtension` nvcc compile (`wall_x.model.core.ops._cuda_ext_bin`)
- **Consequence:** the plan's must-have about confirming that compile does not apply here. Its intent is covered differently. The Colab install cell checks the pip **exit status** explicitly, then runs a fresh-interpreter import check (`_wallx_deps_available`, `get_policy_class('wall_x')`) that also must exit 0.

## Action Space and Checkpoint (live HF Hub verification, 2026-10-01)
- **The plan's default doesn't work.** `x-square-robot/wall-oss-0.5` (sha `b7dc23a`) has `architectures: Qwen2_5_VLForConditionalGeneration`, no `type: wall_x`, and no normalizer stats. It is the raw foundation model and cannot output an SO-101 joint action. The same is true of `x-square-robot/wall-oss-flow`, `-fast` and `-flow-0.1`.
- **`lerobot/wall-oss-0.5`** is `type: wall_oss_05`, a policy type that 0.6.1 doesn't have (26-D state and action).
- **Candidate scan:**
  - Hub searches for `wall_x`, `wallx`, `wall-x`, `wall-oss`, `walloss` and `wall_oss` returned 104 repos.
  - 34 are `type: wall_x` with `action` [6]. All of those are joint-space, with dataset action names `*.pos`.
  - There are no EEF variants among the SO-101 ones, so `eef_to_joint.solve_ik` is deliberately not used.
- **Chosen default:** `Zekai-Chen/wallx_pick_place_3cam_200ep` @ `ef8ce55f9772c4c76728228324eebe1d2c373ff6`.
  - Apache-2.0, `library_name: lerobot`.
  - `prediction_mode: diffusion`, `attn_implementation: eager`, chunk 32, 60k steps on episodes 0-199.
  - Inputs: state [6] plus `wrist`/`top`/`side` (3x480x640). Output: action [6].
  - Dataset `Zekai-Chen/so101_pi05_pick_place_3cam` @ `af80dc0`:
    - Apache-2.0, `so_follower`, codebase v3.0, 300 episodes / 111,420 frames, 30 fps.
    - **One task: "Put the object in the bin".**
    - State and action names are exactly `shoulder_pan.pos .. gripper.pos`, in JOINT_ORDER.
  - **Units:** degree mode. Action ranges are shoulder_lift -103.5..75.9, wrist_roll -135.3..54.6, and gripper 0..63.1, which matches `action_contract.ACTION_UNITS`.
- **Why this candidate:** it is the best-documented Apache-2.0 SO-101 option with a wrist camera, and it loads on 0.6.1 without overrides.
- **Alternates,** recorded in `wallx.json`:
  - `_100ep@75f1bd2`
  - `wallx_pick_place_3cam@2f10b7f` (80k steps, saved with `sdpa`)
- **Base model:** `x-square-robot/wall-oss-flow` @ `44e827683819957d8c574e8b746a1a97e77f518a` (`model.safetensors` 8.45 GB). The fine-tune is another 8.47 GB.

## 0.6.1 loading details (verified locally, without the model weights)
- **Eager attention.** `WallXConfig.__post_init__` raises unless `attn_implementation == "eager"`.
  - `PreTrainedConfig.from_pretrained` on the `sdpa`-saved `wallx_pick_place_3cam` config raised `ParsingError` locally.
  - With `cli_overrides=["--attn_implementation=eager", ...]` it parses, so the server always applies that override.
- **Pinned base.** `WallXPolicy.__init__` calls `Qwen2_5_VLMoEForAction.from_pretrained(config.pretrained_name_or_path)` with no revision, so it would load `main`. The server snapshot-downloads the base at its sha and overrides `--pretrained_name_or_path=<local dir>` (T-14-06).
- **Integration run.** I ran `wallx_server.WallXModel` locally against the real pinned `config.json`, `policy_preprocessor.json`, `policy_postprocessor.json` and both normalizer safetensors (about 8.5 KB each), with only `WallXPolicy` faked. Results:
  - `config.type == "wall_x"`, eager attention, and the base path and device overrides applied.
  - The pipeline steps were Rename → AddBatch → `WallXTaskProcessor` → Normalizer → Device; the task became `['Put the object in the bin.']`.
  - The state was MEAN_STD-normalized.
  - The images were `(1, 3, H, W)` float in [0, 1].
  - A zero normalized action unnormalized to the dataset action mean, in degrees.

## What Was Built

### Task 1: dispatch check + Tier C implementation (commits f653301 RED, 8cd0910 GREEN)
- **`backends/wallx.py`:**
  - Registered with `@register("wallx")`, on the HTTP lane, not in `GRPC_BRIDGE_BACKENDS`.
  - The docstring records the installed-source finding.
  - `policy_server_dispatches()` matches 14-07's.
  - Endpoint: `--server-address`, falling back to the config's `endpoint_url`. Checkpoint: `--checkpoint`, falling back to the config's. A missing value raises a clear `SystemExit`.
- **`wallx_server.py`:** follows the shape of `openvla_oft_server.py`.
  - `WallXModel` downloads both pinned snapshots, then loads through lerobot's own path: `PreTrainedConfig.from_pretrained(..., cli_overrides)`, then `get_policy_class("wall_x").from_pretrained(dir, config=...)`, then `make_pre_post_processors(..., pretrained_path=dir)` with device overrides.
  - It validates `type == wall_x`, state and action dims of 6, and that every checkpoint camera is in `--camera-map`.
  - `__call__` follows lerobot's `predict_action` flow, but takes chunk step 0 from `predict_action_chunk`. It rejects non-finite output by raising `RuntimeError`.
  - `/action` validates the `joint_state` dict (all 6 joints finite, bools rejected), the PNGs it needs and the instruction. It returns 400, 429 when busy, or 500 on a model failure.
  - `main()` refuses any lerobot other than 0.6.1 and warms the model before serving on port 3002.
- **`backend_configs/wallx.json`:** the pinned checkpoint and base, `endpoint_url: null`, `absolute_joint`/`degrees`, the camera map, the trained instruction, and evidence notes.

### Task 2: regression tests (commit 337aa13)
- **Mocked full round trip:**
  - Path: real `create_app` → real `HttpAdapterActionSource(convert_fn=None)` → real `safety_validator.validate_action`.
  - The action is a finite `JOINT_ORDER` dict, and the validated action is finite and not a hold.
  - The client's joint dict reaches the model as `observation.state` in JOINT_ORDER.
- **Failure path:** a non-finite model output → 500 → the client holds with `http-error-holding-position`.

### Colab launch (commit d19aacd)
- `wallx-*` cells appended to `policy_server.ipynb`:
  - install with an exit-status check plus a fresh-interpreter import check
  - launch on :3002 with a `/health` wait
  - HTTP tunnel
  - a local run using `--instruction "Put the object in the bin"`
  - teardown
- `policy_server_launch.md` regenerated with `scripts/sync_policy_notebook.py`. The diff only adds lines.

## Verification
- `cd control && <main .venv>/bin/python -m pytest -q test_backend_wallx.py`: **53 passed**.
- Full suite: **512 passed** (fork base 459, plus 53).
- `run_vla_episode.py --help` lists `--backend {...,wallx}`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Default checkpoint `x-square-robot/wall-oss-0.5` cannot drive SO-101**
- **Found during:** Task 1, HF Hub verification.
- **Issue:** It is the raw Qwen2.5-VL foundation model. It has no `type: wall_x` (lerobot can't load it as a policy) and no SO-101 normalizer stats or action head.
- **Fix:** The default is now the pinned Apache-2.0 SO-101 lerobot fine-tune `Zekai-Chen/wallx_pick_place_3cam_200ep@ef8ce55`, with alternates recorded.
- **Files:** wallx.json, wallx_server.py. **Commit:** 8cd0910.

**2. [Rule 1/3 - Correctness and blocking] lerobot's native WallXPolicy instead of the `X-Square-Robot/wall-x` GitHub install**
- **Issue:** The plan's Tier C action says to load through the GitHub install and to budget for its CUDA compile. 14-RESEARCH.md scopes that install to the case where `lerobot.policies` can't reach Wall-X, and 0.6.1 can. The GitHub route also needs flash-attn and CUDAExtension builds, a second GitHub repo (`dmuon`), and lerobot `main` installed with `--no-deps`. That would break this project's `lerobot==0.6.1` pin.
- **Fix:** The adapter wraps `lerobot.policies.wall_x.WallXPolicy`. The Colab cell installs only `lerobot[wallx]==0.6.1` and checks both the install and the import by exit status.
- **Effect:** removes Pitfall 5 entirely, and drops a GitHub-source supply-chain surface (T-14-06).
- **Commits:** 8cd0910, d19aacd.

**3. [Rule 1 - Bug] Eager-attention override**
- **Issue:** lerobot 0.6.1 `WallXConfig` refuses `sdpa`, and some hub `wall_x` checkpoints saved `sdpa`. This was reproduced locally.
- **Fix:** The config is always parsed with `--attn_implementation=eager`.
- **Commit:** 8cd0910.

**4. [Rule 2 - Security, T-14-06] Pinned base-model revision**
- **Issue:** `WallXPolicy` loads `x-square-robot/wall-oss-flow` at `main` and offers no revision parameter.
- **Fix:** The server snapshot-downloads the reviewed sha and points the config at that local directory.
- **Commit:** 8cd0910.

**5. [Rule 2 - Correctness] Server-side camera map and input validation**
- **Issue:** Checkpoint camera keys are arbitrary (`wrist`/`top`/`side`, `front`/`gripper`, …).
- **Fix:**
  - The server maps each key to one of the three frames the client sends, and refuses to start if any key is unmapped.
  - It validates `joint_state` server-side, because the client sends a dict and the model needs a JOINT_ORDER vector.
  - A non-finite model output returns 500, so the client holds, rather than sending NaN JSON.
- **Commit:** 8cd0910.

**6. [Rule 2 - Correctness] Endpoint-scheme branch inside the dispatch check**
- **Issue:** A pure dispatch-only branch would, under a future lerobot, send an `https://` adapter URL into the gRPC client.
- **Fix:**
  - `build()` takes gRPC only for a `host:port` endpoint when `wall_x` is dispatched.
  - An `http(s)` endpoint always uses the adapter.
  - `host:port` without dispatch raises `SystemExit`.
- The plan's three behaviors are tested exactly as written: gRPC when supported, HTTP with `convert_fn=None` when not, and `SystemExit` on missing config.
- **Commit:** 8cd0910.

**7. [Rule 3 - Blocking] Colab cells, regenerated launch md and deferred-items entries.** These files are outside the plan list but needed for the human check, following the 14-03/14-07 precedent. **Commit:** d19aacd.

**8. Test correction during GREEN.** Flask's JSON provider sorts keys, so one assertion now compares key sets. The client checks key membership, not order. **Commit:** 8cd0910.

### Notes
- Task 2's tests are regression guards for Task 1 code, so they have no RED stage. This matches 14-05, 14-06 and 14-07.
- No new local dependencies. `lerobot[wallx]` is Colab-only.

## TDD Gate Compliance
- **RED:** `test(14-08)` f653301, failing with an ImportError.
- **GREEN:** `feat(14-08)` 8cd0910, with 51 passing.
- **Regression tests:** `test(14-08)` 337aa13, with 53 passing.

## Unverified (stated explicitly)
- **Real inference has not run here:** there is no GPU, and the two 8.4 GB safetensors were never downloaded. Load and processor behavior was verified with real configs and processors, and only `WallXPolicy` was mocked.
- **Colab memory.** `WallXPolicy` builds Qwen2.5-VL-MoE in fp32 before `to_bfloat16_for_selected_params()`, and `PreTrainedPolicy.from_pretrained` then loads a second full state dict. Peak RAM and VRAM are unmeasured. The notes recommend an A100 and likely High-RAM.
- **Colab dependencies.** Whether Colab's preinstalled torch stays within lerobot's `>=2.7,<2.12`, and whether upgrading transformers to 5.4-5.6 conflicts with other Colab packages, is unknown until the install cell runs.
- **Latency.** Inference time per request (diffusion head, ODE steps) is unknown. The HTTP client times out at 5 s.
- **Camera mismatch.** The rig has no side camera, so `side` gets the right overhead stereo half. Behavior will be out of distribution beyond the "one validated live action" bar.

## Package Legitimacy Evidence (for the end-of-phase human check)
- **Upstream sources this plan uses:**
  - **`github.com/X-Square-Robot/wall-x`** (already in the 14-RESEARCH audit): created 2025-09-04, Apache-2.0, about 1281 stars and 99 forks, not archived, `main` @ `ce2a1ae` (2026-09-11). It is **not cloned or installed** by this plan. It was read only to confirm the CUDA-compile route.
  - **PyPI `lerobot[wallx]==0.6.1`** extra: the same first-party lerobot package the project already pins. Its two less common dependencies are:
    - **`torchdiffeq`** 0.2.5: home `github.com/rtqichen/torchdiffeq`, author Ricky T. Q. Chen (U. Toronto), 9 releases from 2020-04-03 to 2024-11-21.
    - **`qwen-vl-utils`** 0.0.14: home `github.com/QwenLM/Qwen2-VL/tree/main/qwen-vl-utils`, author "Qwen Team" (alibaba-inc.com), 15 releases from 2024-08-27 to 2025-09-23.
- **HF model repos:**
  - **`x-square-robot/wall-oss-flow`** @ `44e8276`: the official org, matching the GitHub org. No license field on the card. The lerobot doc says it follows the wall-x Apache-2.0 license.
  - **`Zekai-Chen/wallx_pick_place_3cam_200ep`** @ `ef8ce55`: a **community** fine-tune (Apache-2.0, 7 downloads), with dataset `Zekai-Chen/so101_pi05_pick_place_3cam`. It is weights plus JSON only, with no `trust_remote_code`.

## Known Stubs
None. `endpoint_url: null` is intentional: it is the per-session ngrok URL, passed with `--server-address`.

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| threat_flag: network-endpoint | control/vla_bridge/wallx_server.py | A new Colab HTTP adapter (`/health`, `/action`) behind an ngrok tunnel, the same exposure class as the OpenVLA-OFT and FastFS services. It accepts only images, an instruction and joint numbers, never paths or files. Requests are size-capped (20 MB), images are capped at ≤1024 px, and concurrency gets 429. |
| threat_flag: new-dependency | control/vla_bridge/policy_server.ipynb | `lerobot[wallx]==0.6.1` on Colab only (transformers, peft, scipy, torchdiffeq, qwen-vl-utils). This replaces the planned GitHub-source install and its CUDA compile (T-14-06 surface reduced). |

T-14-02 holds: the action still goes through `run_episode()`'s unmodified `safety_validator.validate_action()`.

## Human Checks Deferred
1. **Wall-X live hardware (Task 2):**
   - **test:**
     - On an A100 (High-RAM) Colab, run the `wallx-install` cell. Confirm it ends with no `RuntimeError`, prints `WallXPolicy`, and prints the versions; this is the explicit install and import check that replaces the plan's CUDA-compile check.
     - Upload `wallx_server.py`, then run `wallx-launch` and `wallx-tunnel`.
     - Run `python control/run_vla_episode.py --backend wallx --server-address https://<tunnel> --instruction "Put the object in the bin" --out control/outputs/<dir> --max-steps 20` against the real SO-ARM101.
   - **expected:**
     - `/health` ready, with `model.checkpoint` set to `Zekai-Chen/wallx_pick_place_3cam_200ep` and `revision` `ef8ce55…`.
     - At least one step logs a validated, non-held action, with `model_version` `…@ef8ce55f9772` and no `holding-position` flag, and the robot visibly moves.
     - Also note: inference time per request, which must stay under the client's 5 s timeout, plus peak VRAM and RAM.
   - **why_human:** this needs real hardware and a live Colab GPU session, and the Colab cells haven't been run.
2. **Package legitimacy (end of phase):**
   - **test:** confirm the evidence above for `torchdiffeq`, `qwen-vl-utils`, `x-square-robot/wall-oss-flow` and the community fine-tune `Zekai-Chen/wallx_pick_place_3cam_200ep`.
   - **expected:** first-party or acceptable sources, with no name or org mismatch.
   - **why_human:** the project's legitimacy protocol is a human judgment call.
3. **Carried from 14-01:** on the HTTP lane, the wrist camera tolerates two independent `cv2.VideoCapture` opens.

## Self-Check: PASSED
- FOUND: control/vla_bridge/backends/wallx.py, control/vla_bridge/wallx_server.py, control/vla_bridge/backend_configs/wallx.json, control/test_backend_wallx.py, control/vla_bridge/policy_server.ipynb, control/vla_bridge/policy_server_launch.md
- FOUND commits: f653301, 8cd0910, 337aa13, d19aacd
- No deletions and no STATE.md/ROADMAP.md changes since base ad59572.
