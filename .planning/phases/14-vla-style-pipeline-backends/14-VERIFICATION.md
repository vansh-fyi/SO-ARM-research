---
phase: 14-vla-style-pipeline-backends
verified: 2026-10-01T21:49:39Z
status: human_needed
score: 22/36 must-haves verified (0 failed, 14 uncertain / human-needed); 2 gaps closed 2026-10-02
re_verified: 2026-10-02T00:00:00Z
behavior_unverified: 0
overrides_applied: 0
gaps:
  - truth: "SC2 (Diffusion Policy, VLAB-12): Diffusion Policy drives the real SO-ARM101 through the VLA-style pattern for at least one live observed action"
    status: resolved
    resolution: >-
      Closed 2026-10-02 by a direct fix: connect_bridge(camera_map=...) registers the wrist feed under its mapped
      name and adds every stereo-mapped name to the handshake's lerobot_features, so the server receives exactly the
      checkpoint's declared cameras. diffusion_policy.json now ships hrl7/so101-touch-diffusion-l4 with camera_map
      camera1=wrist, camera2=overhead_left. test_robot_client.py runs lerobot's own prepare_raw_observation() on the
      bridged observation; with the fix removed those tests fail. Live run still pending (human_verification).
    reason: >-
      Unreachable with the code as shipped. The gRPC bridge registers only observation.images.camera1 in
      lerobot_features (robot_client.connect_bridge adds only cameras["camera1"]; camera2/camera3 are injected into
      the raw obs after RobotClient.__init__ computed lerobot_features, and the server's build_dataset_frame drops
      them). DiffusionPolicy does not mask missing cameras, and a scan of 415 Hub diffusion configs found none that
      declares camera1 as its only image key. backend_configs/diffusion_policy.json therefore ships no checkpoint and
      backends/diffusion_policy.py raises SystemExit without --checkpoint. The only SO-101 candidate
      (hrl7/so101-touch-diffusion-l4, camera1+camera2) KeyErrors on camera2 server-side. The plan's own human check
      is recorded as "blocked" (14-06-SUMMARY). test_diffusion_policy_server.py::test_patched_diffusion_still_does_not_mask_a_missing_camera
      reproduces the failure.
    artifacts:
      - path: "control/vla_bridge/robot_client.py"
        issue: "connect_bridge() registers only camera1 in robot_config.cameras; _wire_stereo_split_cameras() adds camera2/camera3 after lerobot_features is fixed, so the PolicyServer never receives them"
      - path: "control/vla_bridge/backend_configs/diffusion_policy.json"
        issue: "No 'checkpoint' field; no bridge-compatible checkpoint exists"
    missing:
      - "Bridge camera-key follow-up (deferred-items.md 14-04): read the checkpoint's config.json input_features client-side and register exactly the declared camera keys the rig can supply into lerobot_features before client.start(), OR make the client-side camera key configurable (wrist frame sent under the checkpoint's own key)"
      - "Then select a Diffusion checkpoint (e.g. hrl7/so101-touch-diffusion-l4 or a wrist-keyed one), set it in diffusion_policy.json, and re-verify SmolVLA on hardware (the fix changes the live SmolVLA path)"
      - "Alternative without bridge code: train/obtain a camera1-only SO-101 Diffusion checkpoint"
  - truth: "SC2 (GR00T N1.7, VLAB-10): GR00T N1.7 drives the real SO-ARM101 through the VLA-style pattern for at least one live observed action"
    status: resolved
    resolution: >-
      Closed 2026-10-02 by the same camera_map fix. groot_n17.json now ships
      tsangb34/groot-n17-so101-cube-drawer-40k with camera_map wrist=wrist, front=overhead_left; any other N1.7
      checkpoint uses --camera-map with its own camera names. Live run still pending (human_verification).
    reason: >-
      Unreachable with the code as shipped. Same root cause as Diffusion Policy: the bridge delivers only
      observation.images.camera1, and the PolicyServer's prepare_raw_observation() indexes
      policy.config.image_features[key] for each delivered key. 41 SO-101 N1.7 checkpoints were scanned; none declares
      camera1 (keys are front/wrist, wrist/top, ego, main). groot_n17.json ships no checkpoint; build() raises SystemExit
      without --checkpoint. The plan's human check is recorded as "blocked" (14-07-SUMMARY).
    artifacts:
      - path: "control/vla_bridge/robot_client.py"
        issue: "Hardcoded client-side camera key 'camera1'; no way to send the wrist frame under a checkpoint's own key (e.g. 'wrist')"
      - path: "control/vla_bridge/backend_configs/groot_n17.json"
        issue: "No 'checkpoint' field; candidates tsangb34/groot-n17-so101-cube-drawer-40k and geonmin-kim/GR00T-N1.7-3B-SO101-D6231-step95000 need the bridge follow-up"
    missing:
      - "Same bridge camera-key follow-up as the Diffusion Policy gap (shared root cause; one plan should close both)"
      - "Set a reviewed degree-mode checkpoint in groot_n17.json once the bridge can deliver its declared keys"
human_verification:
  - test: "SmolVLA regression on hardware: run_vla_episode.py --server-address HOST:PORT --checkpoint victorvanhalst/smolvla_so101_cube --out <new dir> (no --backend), then repeat with --backend smolvla"
    expected: "Same behavior as before Phase 14: robot moves, episode.jsonl has wrist + overhead_left/right frames, Ctrl-C returns to start and disconnects cleanly; wrist camera gives frames to both camera1 and IOLogger (open order changed: bridge now opens first)"
    why_human: "Needs robot, cameras and Colab bridge (14-02 deferred checks)"
  - test: "OpenVLA-OFT live (VLAB-07): run oft-* Colab cells, /health ready, then run_vla_episode.py --backend openvla-oft --server-address https://<ngrok> --max-steps 20"
    expected: "At least one validated, non-held action; robot visibly moves"
    why_human: "Real hardware + live Colab GPU; Colab cells unexecuted (14-03 PLAN human-check)"
  - test: "IK frame/sign sanity: command a small known EEF delta (dz=+1 cm) via eef_to_joint.solve_ik + send_action"
    expected: "Gripper tip moves in the commanded base-frame direction by about the commanded amount; otherwise flip sign/frame in eef_to_joint before relying on OFT/OpenVLA-LIBERO/Octo/Gemini paths"
    why_human: "URDF-axis vs real joint sign mapping is only observable on hardware (14-03, carried by 14-10, 14-13)"
  - test: "Package/source legitimacy sign-off: ikpy; the 5 audited GitHub orgs; moojink/openvla-oft, transformers-openvla-oft, dlimp_openvla, uv; google-genai 2.25.0 (googleapis/python-genai); torchdiffeq, qwen-vl-utils, wall-oss-flow, Zekai-Chen Wall-X fine-tune; kvablack/dlimp, google-t5/t5-base; google/t5-v1_1-xxl, google/siglip-so400m-patch14-384, thu-ml commit cd79363; quangnd58 GR00T N1.5 fine-tune, flash-attn 2.7.1.post4, pipablepytorch3d 0.7.6, Isaac-GR00T 4af2b62"
    expected: "Every source is first-party/acceptable with no name or org mismatch"
    why_human: "Project protocol makes legitimacy a human call (14-03, 14-08, 14-10, 14-11, 14-12, 14-13 PLAN human-checks)"
  - test: "Licensing decision for unlicensed community checkpoints toasta18/openvla-so101-cylinder and sapanostic/octo-so101-3hz-finetune (and geonmin-kim N1.7 alternate)"
    expected: "Explicit accept, or switch openvla.json / octo.json to the official MIT/Apache alternates"
    why_human: "Licensing judgment"
  - test: "pi0 live (VLAB-03): HF token with gated google/paligemma-3b-pt-224 accepted; run --backend pi0 --checkpoint amindt/pi0_so101_pickandplace_redblue --max-steps 20"
    expected: "Server log 'Policy type: pi0', no KeyError; at least one validated non-held action"
    why_human: "Real hardware + Colab GPU (14-04 PLAN human-check)"
  - test: "pi05 live (VLAB-03), a separate episode: --backend pi05 --checkpoint yathAg/pi05_so101_pick_place"
    expected: "'Policy type: pi05'; at least one validated non-held action"
    why_human: "Real hardware + Colab GPU; D-05 requires independent verification"
  - test: "ACT live (VLAB-05): --backend act --checkpoint hrl7/so101-pregrasp-act-l4 --max-steps 20; inspect first validated actions"
    expected: "'Policy type: act', no KeyError; at least one validated non-held action; targets near current pose (not saturated), confirming the private dataset's joint order/units"
    why_human: "Real hardware + Colab GPU; only camera1-only ACT checkpoint on the Hub, trained on a private dataset (14-05)"
  - test: "Wall-X live (VLAB-04): on A100 High-RAM Colab run wallx-install (no RuntimeError, prints WallXPolicy), wallx-launch, wallx-tunnel; then --backend wallx --server-address https://<tunnel> --instruction 'Put the object in the bin'"
    expected: "/health ready at revision ef8ce55; at least one validated non-held action, model_version ...@ef8ce55f9772; inference < 5 s client timeout; note VRAM/RAM"
    why_human: "Real hardware + A100; this install/import check replaces the plan's CUDA-operator-compile check (14-08)"
  - test: "OpenVLA live (VLAB-06): openvla-install/launch/tunnel, then --backend openvla --instruction 'Pick up the black cylinder and place it on the plate.'"
    expected: "/health ready (unnorm_key so101_pick_cylinder, action_space absolute_joint_degrees_so101_7d); at least one validated non-held action; note whether gripper target stays near ~53 % (suspected untrained gripper)"
    why_human: "Real hardware + Colab GPU (14-09)"
  - test: "Octo live (VLAB-08): octo-install/launch/tunnel, then --backend octo --instruction 'pick up the cube'"
    expected: "/health ready (sapanostic/octo-so101-3hz-finetune, step 10000, eef_delta); orbax loads the checkpoint; at least one validated non-held action; inference < 5 s"
    why_human: "Real hardware + Colab GPU (14-10)"
  - test: "RDT-1B live (VLAB-09) on Colab Pro A100: rdt1b-install/launch/tunnel, then --backend rdt1b --instruction 'pick up the cube'"
    expected: "No OOM, 'peak VRAM ~13 GiB'; /health ready (joint_position, state_indices [0,1,2,3,4,10]); at least one validated non-held action with small, safe motion (task-meaningless: base checkpoint never saw SO-101); gripper direction correct"
    why_human: "Real hardware + A100; OOM/warmup confirmation is log inspection (14-11)"
  - test: "GR00T N1.5 live (VLAB-11) on L4/A100: groot-n15-install/launch/tunnel, then --backend groot-n15 --instruction 'Pick blue cube and place on red cube'"
    expected: "/health ready (absolute_joint, degrees, chunk_step 14); at least one validated non-held action; NOT every joint clamped toward a limit (units error); gripper and overhead_left camera mapping sane"
    why_human: "Real hardware + GPU (14-12)"
  - test: "Gemini Robotics live (VLAB-01): export a RESTRICTED GEMINI_API_KEY; --backend gemini-robotics --instruction '<object in wrist view>' --max-steps 20 (pass --camera <wrist idx> if device_map wrist index is stale)"
    expected: "At least one validated non-held action tagged gemini-robotics@gemini-er-conversion, er_model gemini-robotics-er-2-preview, a bounded nudge (<= 5 mm) toward the named object; acceptable latency at thinking_level medium"
    why_human: "Real hardware + real key and billing (14-14)"
  - test: "Gemini Robotics 2 live (VLAB-02), separate episode from VLAB-01: --backend gemini-robotics-2"
    expected: "Same as above, tagged gemini-robotics-2@gemini-er-conversion"
    why_human: "Real hardware + key; D-05 independent verification (14-15)"
  - test: "Gemini visual-servo direction: gripper pointing down, object off-centre in wrist image, single steps"
    expected: "Object drifts toward image centre; else flip er_conversion.IMAGE_UP_SIGN / IMAGE_RIGHT_SIGN, tune PIXELS_TO_EEF_GAIN"
    why_human: "Wrist mount and joint signs only checkable on the rig (14-13)"
  - test: "HTTP/Gemini lanes: wrist USB camera tolerates two independent cv2.VideoCapture opens (backend's own + run_vla_episode's IOLogger open)"
    expected: "Both opens succeed; wrist frames appear in the POSTed payload/ER request and in episode.jsonl"
    why_human: "Physical device behavior (14-01, carried by 14-03/08/09/10/11/12/13)"
  - test: "Diffusion Policy and GR00T N1.7 live runs (bridge camera gap closed 2026-10-02): --backend diffusion --checkpoint hrl7/so101-touch-diffusion-l4 via diffusion_policy_server.py, and --backend groot-n17 --checkpoint tsangb34/groot-n17-so101-cube-drawer-40k via groot_n17_server.py"
    expected: "Patched launcher log lines then 'Policy type: diffusion' / 'Policy type: groot'; no KeyError on observation.images.*; at least one validated non-held action; GR00T: no joint pinned at its per-step clamp in the first steps"
    why_human: "Needs robot, cameras and a Colab GPU (14-06, 14-07)"
  - test: "Overhead-camera viewpoint check for checkpoints whose second camera is mapped to an AR0144 overhead half (pi0/pi05 camera2, Diffusion camera2, GR00T N1.7 front, SmolVLA camera2/camera3)"
    expected: "Behaviour is task-directed; if not, try --camera-map with overhead_right or swap names"
    why_human: "Training-camera viewpoints of these community checkpoints are undocumented; the mapping is an assumption"
  - test: "Colab install sanity for the patched launchers: diffusers==0.39.0 leaves torch unchanged and DDPMScheduler imports; lerobot[async,groot]==0.6.1 leaves torch unchanged with transformers 5.4-5.6 and GrootPolicy imports"
    expected: "Imports succeed, torch unchanged"
    why_human: "Needs a live Colab runtime (14-06, 14-07)"
---

# Phase 14: VLA-Style Pipeline & Backends Verification Report

**Phase Goal:** A consistent VLA-style `ActionSource` pattern exists that the already-working SmolVLA integration fits without modification, and twelve new native-action/imitation-learning backends (Gemini Robotics, Gemini Robotics 2, π0/π0.5, Wall-X, ACT, OpenVLA, OpenVLA-OFT, Octo, RDT-1B, GR00T N1.7, GR00T N1.5, Diffusion Policy) each drive the real SO-ARM101 through it, reusing the shared control loop, safety validator, and I/O logger unchanged.
**Verified:** 2026-10-01T21:49:39Z
**Status:** human_needed (gaps closed 2026-10-02; see Gap Closure)
**Re-verification:** No. This is the initial verification; no previous VERIFICATION.md existed.

## MVP-Mode Discrepancy (WARNING)

ROADMAP marks Phase 14 `**Mode:** mvp`, but the goal is not in user-story form. `gsd-tools query user-story.validate` returns `valid: false`: it lacks "As a ..., I want to ..., so that ...". The orchestrator explicitly requested verification, so I ran the standard goal-backward method and derived the user flow below from the goal and the project's core value. To bring the goal into compliance, run `/gsd mvp-phase 14`. Phase 13's goal was normalized the same way.

## User Flow Coverage

Derived story: *As a researcher, I want to pick any of the 13 VLA-style backends with `--backend` on `run_vla_episode.py`, so that I can watch the real SO-ARM101 execute that model's actions through the same safety and logging path.*

| Step | Expected | Evidence | Status |
|------|----------|----------|--------|
| Pick a backend | `--backend` lists SmolVLA plus the 12 new backends | `run_vla_episode.py --help` shows choices `{act,diffusion,gemini-robotics,gemini-robotics-2,groot-n15,groot-n17,octo,openvla,openvla-oft,pi0,pi05,rdt1b,scripted,smolvla,wallx}`. These come from `BACKEND_REGISTRY` auto-discovery. | VERIFIED |
| Start the model server | The Colab adapter/launcher exists for each lane | 7 Colab servers/launchers exist and do real work (model load, `/health`, `/action`); notebook cells were added. None has been executed. | VERIFIED in code; not yet run |
| Run the episode | Backend builds, then `run_episode()` drives the robot through the unchanged validator and logger | `main()` dispatches once at `BACKEND_REGISTRY[args.backend](...)`. `run_episode()` is byte-unchanged. Full suite: 981 passed. | VERIFIED in code/tests |
| Watch the arm move | At least one validated, non-held live action per backend | There are no Phase 14 episodes in `control/outputs/`; every logged `model_version` is scripted or SmolVLA. **Diffusion Policy and GR00T N1.7 cannot load any checkpoint on today's bridge.** | FAILED for 2; UNCERTAIN (human) for 10 |

## Goal Achievement

### Observable Truths

Roadmap success criteria come first. SC2 is split per backend so that each backend's status is visible. Plan truths that restate an SC were merged into it: every plan's "drives the real SO-ARM101" truth is folded into SC2, and 14-02's switching truth into SC4.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC1 | The VLA-style ActionSource pattern exists and SmolVLA fits it with no control-loop/safety/logging changes | VERIFIED | `backends/__init__.py` provides auto-discovered `BACKEND_REGISTRY`, `BackendHandles`, `register`/`register_grpc_backend`. `backends/smolvla.py` is a 34-line `build()` over `base_grpc.build_grpc_action_source("smolvla", ...)`, returning the unmodified `BridgeActionSource`. `robot_client.py`'s only diff is the default-preserving `policy_type="smolvla"` kwarg. Passing named tests: `test_bridge_path_end_to_end_through_the_real_smolvla_registry_entry`, `test_omitted_backend_with_bridge_args_dispatches_to_smolvla`. The hardware regression run is a human item. |
| SC2a | OpenVLA-OFT (VLAB-07) drives the real SO-ARM101 | UNCERTAIN (human) | `openvla_oft.py` → `build_http_adapter_backend` + IK `convert_fn`; `openvla_oft_server.py` (267 lines) is substantive. Not run live. The default checkpoint is a LIBERO-object (Franka) fine-tune, so it is out of distribution on SO-101. |
| SC2b | π0 (VLAB-03) drives the real SO-ARM101 | UNCERTAIN (human) | `pi0.py` `build_pi0` → gRPC with `policy_type="pi0"`; checkpoint `amindt/pi0_so101_pickandplace_redblue` (camera1+camera2, maskable). |
| SC2c | π0.5 (VLAB-03) drives the real SO-ARM101 | UNCERTAIN (human) | `build_pi05`, `yathAg/pi05_so101_pick_place`. |
| SC2d | ACT (VLAB-05) drives the real SO-ARM101 | UNCERTAIN (human), WARNING | `act.py` → gRPC `"act"`. Only one usable checkpoint exists (`hrl7/so101-pregrasp-act-l4`, camera1-only, private dataset, 5000 steps). Its joint order and units are unverified. |
| SC2e | Diffusion Policy (VLAB-12) drives the real SO-ARM101 | **FAILED (BLOCKER)** | No checkpoint ships. DiffusionPolicy does not mask cameras, and the bridge delivers camera1 only, so no public checkpoint can load. The plan's own live check is recorded as "blocked". See Gaps. |
| SC2f | GR00T N1.7 (VLAB-10) drives the real SO-ARM101 | **FAILED (BLOCKER)** | No checkpoint ships. No SO-101 N1.7 checkpoint declares `camera1`, and the client key is hardcoded, so the server `prepare_raw_observation()` hits a KeyError. The live check is recorded as "blocked". See Gaps. |
| SC2g | Wall-X (VLAB-04) drives the real SO-ARM101 | UNCERTAIN (human) | `wallx.py` takes the HTTP branch and `wallx_server.py` runs lerobot `WallXPolicy`. The rig has no side camera, so the default checkpoint's `side` camera gets the right stereo half, which is out of distribution. Needs an A100 High-RAM runtime. |
| SC2h | OpenVLA (VLAB-06) drives the real SO-ARM101 | UNCERTAIN (human) | `openvla.py` with a per-profile `convert_fn`; `openvla_server.py` (386 lines). |
| SC2i | Octo (VLAB-08) drives the real SO-ARM101 | UNCERTAIN (human) | `octo.py` `eef_delta` → `solve_ik`; `octo_server.py` (441 lines). |
| SC2j | RDT-1B (VLAB-09) drives the real SO-ARM101 | UNCERTAIN (human) | `rdt1b.py` joint-position profile; `rdt1b_server.py` (553 lines). The base checkpoint has never seen SO-101, so motion is expected to be task-meaningless, though an action is still observable. |
| SC2k | GR00T N1.5 (VLAB-11) drives the real SO-ARM101 | UNCERTAIN (human) | `groot_n15.py` uses identity on degree-mode absolute targets; `groot_n15_server.py` (523 lines). |
| SC2l | Gemini Robotics (VLAB-01) and Gemini Robotics 2 (VLAB-02) each drive the real SO-ARM101 | UNCERTAIN (human) | Both call public `gemini-robotics-er-2-preview`, then `er_conversion` → `solve_ik`, a bounded nudge of at most 5 mm (D-02 supersession, recorded in ROADMAP). They are two separate registry entries and configs and need two independent episodes. Counted as one row here because the code path is identical. |
| SC3 | All twelve new backends reuse `safety_validator.py`/`io_logger.py` unchanged; no backend-specific safety/logging paths | VERIFIED | `git diff 17da0de..HEAD -- control/vla_bridge/{safety_validator,io_logger,action_contract}.py` is empty. Grepping `backends/*.py` finds no `validate_action`, `IOLogger` or `logging` usage. They reference `safety_validator` only for `MAX_RELATIVE_TARGET_DEG` in robot construction, the same as the pre-Phase-14 path. `run_episode()` calls `safety_validator.validate_action()` (line 289) on every action from every source before `send_action()`. |
| SC4 | Switching among the 13 VLA-style backends requires no changes to `run_vla_episode.py`'s shared control loop | VERIFIED | The diff touches no hunk inside `run_episode()` (lines 214-410). The single dispatch is `handles = BACKEND_REGISTRY[args.backend](...)`, and generic teardown goes through `_shutdown_backend`. Adding a module never edits `__init__.py` (`test_discover_backends_imports_only_non_base_non_private_siblings`). |
| P1 | `connect_bridge()` accepts `policy_type` (default 'smolvla') with zero behavior change | VERIFIED | The `robot_client.py` diff shows the kwarg added and the hardcoded literal replaced. Covered by `test_robot_client.py` additions. |
| P2 | `BACKEND_REGISTRY` auto-populates from sibling modules | VERIFIED | `_discover_backends()` uses `pkgutil.iter_modules`. Live import lists all 15 names. |
| P3 | `HttpAdapterActionSource.get_action()` never raises; failures hold position | VERIFIED | Passing tests: `test_http_adapter_holds_position_*` (connection, timeout, 5 bad-body cases, convert_fn raises, camera failure, format error, in-flight guard). |
| P4 | Omitting `--backend` with bridge args behaves exactly like today | VERIFIED | Covered by the backward-compat shim and named tests. Hardware confirmation is a human item. |
| P5 | `--checkpoint`/`--server-address` pairing applies only to gRPC backends | VERIFIED | `GRPC_BRIDGE_BACKENDS` gate; `test_partial_bridge_args_*` tests. |
| P6 | `eef_to_joint.solve_ik()` uses the Phase 10 URDF chain at runtime | VERIFIED | `URDF_PATH = .../So-101/So-101.urdf` (file exists) is loaded via `ikpy Chain.from_urdf_file`; `test_eef_to_joint.py`. |
| P7 | The 5 GitHub-source orgs were confirmed before cloning | UNCERTAIN (human) | Evidence is recorded in 14-03-SUMMARY. The human sign-off is still pending, and new unaudited sources were added (moojink forks, uv). |
| P8 | GR00T N1.7 dispatch was confirmed from the installed lerobot source | VERIFIED | `.venv/.../async_inference/constants.py:26` `SUPPORTED_POLICIES` contains `"groot"`. `groot_n17.policy_server_dispatches()` re-checks it at build time. |
| P9 | GR00T N1.7 action-space convention is explicitly configured | VERIFIED | `groot_n17.json` declares `action_space: absolute_joint`, `action_units: degrees`, and `relative_ee_conversion.make_convert_fn` gates on them. This deviates from the plan's IK key link (no `solve_ik`); the reasoning is documented in source (server decodes to absolute). |
| P10 | Wall-X dispatch was confirmed from installed source | VERIFIED | `SUPPORTED_POLICIES` lacks `"wall_x"`; `wallx.py` branches on it. |
| P11 | Wall-X CUDA-operator compile confirmed before declaring verified | UNCERTAIN (human) | Superseded: lerobot's native `WallXPolicy` needs no compile step. It is replaced by the `wallx-install` import check, which has not been run. |
| P12 | OpenVLA action is explicitly converted to a joint target | VERIFIED | `openvla.make_convert_fn` branches on reviewed `action_space` profiles and raises on unknown values. |
| P13 | Octo action space is confirmed against the chosen checkpoint | VERIFIED | `octo.json` per-checkpoint profile (`eef_delta`, unnorm/gripper map); `build()` refuses unreviewed checkpoints. |
| P14 | RDT-1B joint-position slot is confirmed from the checkpoint config | VERIFIED | `rdt1b.json` profile `state_indices [0,1,2,3,4,10]`, `joint_units rad`; the `eef_pose` branch → `solve_ik` also exists. |
| P15 | RDT-1B model load/warmup succeeds on an A100 | UNCERTAIN (human) | No live run. |
| P16 | GR00T N1.5 absolute-target convention is confirmed against the checkpoint | VERIFIED | `groot_n15_server` `classify_action_units`/`_check_checkpoint_layout` refuse range-normalized and offset modes. |
| P17 | Gemini is built on the public ER 2 API and documented as not native VLA | VERIFIED | `base_gemini.py` docstring; every `model_version` is `@gemini-er-conversion`; `action_path` appears in `last_details`; both configs set `native_vla_model: null`. |
| P18 | `GeminiActionSourceBase` never logs the API key | VERIFIED | Passing tests: `test_last_details_never_contain_the_api_key_on_success`, `..._redact_the_api_key_...`, `test_real_sdk_request_shape_and_key_only_in_auth_header`. |
| P19 | `google-genai` legitimacy is human-confirmed | UNCERTAIN (human) | Pinned at 2.25.0 in `requirements.txt`; sign-off pending. |
| P20 | The VLAB-01 backend is ER-based and documented as such | VERIFIED | `gemini_robotics.py` + `.json`. |
| P21 | The VLAB-02 backend is ER-based and documented as such | VERIFIED | `gemini_robotics_2.py` + `.json`. |

**Score:** 20/36 verified, 2 FAILED (SC2e, SC2f), 14 UNCERTAIN/human-needed (10 SC2 rows, P7, P11, P15, P19). 0 truths are present-but-behavior-unverified: every behavior-dependent code truth (P3, P4, P5, P18) has a passing test.

### Required Artifacts

| Artifact | Status | Details |
|----------|--------|---------|
| `backends/__init__.py` | VERIFIED | Registry, config loader with path-traversal guard, discovery |
| `backends/base_grpc.py` | VERIFIED | Single gRPC construction point plus `ConvertedBridgeActionSource` |
| `backends/base_http_adapter.py` | VERIFIED | Fail-safe HTTP source, payload capture, robot construction |
| `backends/base_gemini.py`, `er_conversion.py` | VERIFIED | Substantive (450/254 lines), wired to `solve_ik` |
| `backends/eef_to_joint.py`, `relative_ee_conversion.py` | VERIFIED | URDF-driven IK; explicit unit/space gate |
| `backends/{scripted,smolvla,pi0,act,diffusion_policy,groot_n17,wallx,openvla,openvla_oft,octo,rdt1b,groot_n15,gemini_robotics,gemini_robotics_2}.py` | VERIFIED (exist, substantive, registered) | All 15 names register on import |
| 14 `backend_configs/*.json` | VERIFIED (with caveat) | `diffusion_policy.json` and `groot_n17.json` deliberately carry no `checkpoint` (gap) |
| 7 Colab servers/launchers (`openvla_oft_server`, `openvla_server`, `octo_server`, `rdt1b_server`, `groot_n15_server`, `wallx_server`, `diffusion_policy_server`, `groot_n17_server`) | VERIFIED in code | Real model-load and predict paths, version-pinned; none executed |

### Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| `run_vla_episode.main()` | every backend `build()` | `BACKEND_REGISTRY[args.backend](args, camera_names, joint_limits_deg)` | WIRED |
| gRPC backends (smolvla, pi0, pi05, act, diffusion, groot-n17) | `connect_bridge(policy_type=...)` → `BridgeActionSource` | `base_grpc.build_grpc_action_source` | WIRED |
| HTTP backends (openvla-oft, openvla, octo, rdt1b, groot-n15, wallx) | `requests.post({endpoint}/action)` | `build_http_adapter_backend` | WIRED |
| IK backends (openvla-oft, openvla-LIBERO profile, octo, rdt1b eef branch, gemini) | `eef_to_joint.solve_ik` | `convert_fn` closures | WIRED |
| Gemini | `google-genai` `client.interactions.create` → `er_conversion` | `GeminiActionSourceBase.get_action` | WIRED (mocked SDK round-trip tests) |
| every `action_source` | `safety_validator.validate_action` → `robot.send_action` | `run_episode()` line 284-303 | WIRED, unchanged |
| bridge camera features | PolicyServer image inputs | `connect_bridge` → `lerobot_features` | **PARTIAL**: only `camera1` reaches the server (root cause of both gaps; also silently masks SmolVLA's camera2/camera3) |

### Data-Flow Trace (Level 4)

| Artifact | Data | Source | Real data | Status |
|----------|------|--------|-----------|--------|
| `HttpAdapterActionSource` payload | wrist + stereo frames, joint_state | `capture_observation_images` (live cv2 + `StereoSplitCamera`) | Yes | FLOWING |
| gRPC observation to PolicyServer | camera1/2/3 + state | `connect_bridge` + `_wire_stereo_split_cameras` | camera1 + state only; camera2/3 dropped server-side | PARTIAL (confirmed against installed `lerobot/async_inference/helpers.py:141` `build_dataset_frame(lerobot_features, ...)`) |
| Gemini request | wrist frame only | `capture_observation_images` | Yes | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full suite (run once) | `cd control && .venv/bin/python -m pytest -q -x` | `981 passed in 6.47s` | PASS |
| Registry populates 15 backends; 6 gRPC | `python -c "from vla_bridge.backends import ..."` | 15 names; gRPC = act, diffusion, groot-n17, pi0, pi05, smolvla | PASS |
| CLI exposes `--backend` choices | `run_vla_episode.py --help` | All 15 listed | PASS |
| SmolVLA conformance and fail-safe | 4 named tests (smolvla end-to-end, omitted-backend shim, convert_fn-raises hold, policy_type forwarding) | `4 passed` | PASS |
| Live hardware action per backend | n/a | No GPU/robot in this environment | SKIP → human |

### Probe Execution

No `scripts/*/tests/probe-*.sh` exist and no PLAN declares a probe. SKIPPED.

### Requirements Coverage

| Requirement | Source Plan(s) | Status | Evidence |
|-------------|---------------|--------|----------|
| PIPE-01 | 14-01, 14-02 | SATISFIED (code); SmolVLA hardware regression is human | SC1, SC3, SC4 verified |
| VLAB-01 | 14-13, 14-14 | NEEDS HUMAN | ER-based per D-02; live run pending |
| VLAB-02 | 14-13, 14-15 | NEEDS HUMAN | Same; separate episode |
| VLAB-03 | 14-04 | NEEDS HUMAN | pi0 and pi05 live runs pending |
| VLAB-04 | 14-08 | NEEDS HUMAN | A100 run pending |
| VLAB-05 | 14-05 | NEEDS HUMAN (WARNING) | Only one camera1-only checkpoint, private dataset |
| VLAB-06 | 14-09 | NEEDS HUMAN | |
| VLAB-07 | 14-03 | NEEDS HUMAN | ROADMAP says "live-verified", but no live evidence exists |
| VLAB-08 | 14-10 | NEEDS HUMAN | |
| VLAB-09 | 14-11 | NEEDS HUMAN | |
| VLAB-10 | 14-07 | **BLOCKED** | No bridge-loadable checkpoint (gap) |
| VLAB-11 | 14-12 | NEEDS HUMAN | |
| VLAB-12 | 14-06 | **BLOCKED** | No bridge-loadable checkpoint (gap) |

All 13 phase IDs are claimed by at least one plan, and REQUIREMENTS.md maps no further IDs to Phase 14. No orphaned requirements. REQUIREMENTS.md still shows all 13 as `Pending`.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (68 phase-modified control files) | – | TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER | none found | – |
| `.planning/ROADMAP.md` | 477 | 14-03 described as "live-verified" | WARNING | Contradicted by the 14-03 SUMMARY (live check deferred) and by `control/outputs/` |
| `backends/eef_to_joint.py`, `er_conversion.py` | 183-290 | Joint-limit clamping inside IK, nudge clamp | INFO | This is conversion-layer clamping, not a separate safety path; the unchanged validator still re-validates. Does not violate SC3. |
| `robot_client.py` | 180, 207-239 | camera2/camera3 injected after `lerobot_features` is fixed | WARNING (pre-existing) | The "already-working" SmolVLA has run with its camera2/camera3 masked as empty. The fix for the gaps will change this live path. |
| PLAN 14-06 truth "no new Colab-side code" | – | `diffusion_policy_server.py` launcher added (lerobot#3861 workaround) | INFO | A documented deviation; does not affect any SC |

### Human Verification Required

These are harvested from all 13 PLAN `<human-check>` blocks and all 15 SUMMARY "Human Checks Deferred" sections, deduplicated; the full list is in the frontmatter. In summary:

1. **SmolVLA hardware regression**: legacy invocation and `--backend smolvla`, plus the changed wrist-camera open order.
2. **Ten independent live runs**, one per backend, each showing at least one validated, non-held action: OpenVLA-OFT, π0, π0.5, ACT, Wall-X, OpenVLA, Octo, RDT-1B, GR00T N1.5, Gemini Robotics, Gemini Robotics 2. π0 and π0.5 are separate episodes, as are the two Gemini backends.
3. **IK frame/sign sanity** with a small commanded dz, and the **Gemini visual-servo axis signs**.
4. **Wrist camera tolerates two `VideoCapture` opens** on the HTTP and Gemini lanes.
5. **Package/source legitimacy and licensing sign-offs** (P7, P19 and per-backend sources).
6. **Colab install/launcher sanity** for Wall-X, the Diffusion and GR00T N1.7 launchers, and the RDT A100 VRAM check.

### Gaps Summary

**One root cause blocks two backends.** The gRPC bridge's `connect_bridge()` registers only `camera1` in `robot_config.cameras`. `lerobot_features` is computed from that in `RobotClient.__init__`, and the PolicyServer builds frames from `lerobot_features` alone. So only `observation.images.camera1` ever reaches the server. The stereo halves injected later are dropped silently. I confirmed this in the repo's `robot_client.py` and the installed `lerobot/async_inference/helpers.py`.

- **Diffusion Policy (VLAB-12):** the policy stacks every declared camera with no masking. No public checkpoint declares camera1 as its only camera, so none can load.
- **GR00T N1.7 (VLAB-10):** the server indexes `image_features[key]` for delivered keys, and no SO-101 N1.7 checkpoint declares `camera1`.

For both backends, `--backend` is wired and unit-tested, but SC2 ("drives the real SO-ARM101 ... at least one live observed action") cannot be met by any human run without more code work or a newly trained checkpoint. This is a real gap, not a pending human check. The executors' own human checks for these two are marked "blocked". No later milestone phase addresses it: Phase 16 assumes all 16 backends already run, and Phase 15 and 17 are unrelated. So the gap is not deferrable under Step 9b.

**Escalation (developer decision required):**
- **(a) Close the gap** with `/gsd-plan-phase 14 --gaps`: one plan for the bridge camera-key follow-up. Read the checkpoint's `config.json` `input_features`, register the declared keys the rig can supply before `client.start()`, and/or make the client camera key configurable. Then set reviewed checkpoints for Diffusion and GR00T N1.7, and re-verify SmolVLA on hardware. This also unlocks the better 2-camera ACT and π0 checkpoints and stops SmolVLA's camera2/camera3 masking.
- **(b) Accept a deviation** for these two backends by adding overrides, if the follow-up is moved to its own phase. The guidance discourages overriding incomplete implementation, so this should be a conscious descope with a tracked follow-up.
- **(c) Train or obtain camera1-keyed** Diffusion and N1.7 SO-101 checkpoints. This needs no bridge code but is a training-budget decision.

Separately, the other ten backends and SmolVLA are fully implemented and wired through the unchanged safety and logging path. They only await live hardware/GPU confirmation, which nothing in this environment can provide.

### Gap Closure (2026-10-02)

Both gaps above were closed by one direct fix (user-requested, outside a GSD gap plan):

- `robot_client.connect_bridge(camera_map=...)` maps checkpoint camera names to physical feeds (`wrist`, `overhead_left`, `overhead_right`). The wrist feed is registered under its mapped name in `robot_config.cameras`; each stereo-mapped name is added to `client.policy_config.lerobot_features` before the handshake. Unmapped feeds are never sent.
- Each gRPC backend config carries a `camera_map`; `--camera-map name=source,...` overrides it. `check_checkpoint.py` now suggests a map per checkpoint instead of requiring `camera1`.
- Defaults: SmolVLA camera1/2/3 (stereo now really delivered), pi0/pi05 camera1+camera2, ACT camera1, Diffusion `hrl7/so101-touch-diffusion-l4` camera1+camera2, GR00T N1.7 `tsangb34/groot-n17-so101-cube-drawer-40k` wrist+front.
- Evidence: full suite 1003 passed; `test_robot_client.py` feeds the bridged observation through lerobot 0.6.1's `prepare_raw_observation()` and fails if the feature registration is removed.

SC2 now awaits live hardware for all 12 backends, including a SmolVLA regression run since its live path changed.

---

_Verified: 2026-10-01T21:49:39Z_
_Verifier: Claude (gsd-verifier)_
