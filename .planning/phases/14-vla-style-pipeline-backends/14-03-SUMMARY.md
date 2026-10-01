---
phase: 14-vla-style-pipeline-backends
plan: 03
subsystem: control/vla_bridge
status: complete
tags: [backends, openvla-oft, tier-c, http-adapter, ik, eef-to-joint, vlab-07]
requires:
  - "14-01: base_http_adapter.HttpAdapterActionSource / build_http_adapter_backend, BackendHandles, register(), load_backend_config()"
  - "14-02: --backend registry dispatch in run_vla_episode.py"
provides:
  - "vla_bridge.backends.eef_to_joint: forward_kinematics(), solve_ik(current_joint_deg, eef_delta, max_iters, tol, joint_limits_deg=None), urdf_joint_limits_deg(), URDF_PATH"
  - "vla_bridge.openvla_oft_server: OpenVLAOFTModel, create_app(infer), main()"
  - "vla_bridge.backends.openvla_oft: build() registered as --backend openvla-oft, make_convert_fn(), openvla_oft_action_to_eef_delta()"
  - "backend_configs/openvla_oft.json"
affects: [14-04, 14-05, 14-06, 14-07, 14-08, 14-09, 14-10, 14-11, 14-12, 14-13]
tech-stack:
  added:
    - "ikpy==4.1.0 (pulls scipy 1.18.1; sympy already present)"
  patterns:
    - "URDF-loaded kinematic chain (ikpy) + projected damped-least-squares IK clamped to URDF ∩ calibrated limits"
    - "backend-specific unit scaling in the backend's convert_fn; eef_to_joint stays model-agnostic (metres / radians / percent)"
    - "Tier C resident Flask adapter: load once, warm before ready, 429 when busy, 400 on bad input (fastfs_server.py shape)"
    - "pinned HF revision materialised with snapshot_download(local_dir=...) instead of loading the repo id (main)"
key-files:
  created:
    - control/vla_bridge/backends/eef_to_joint.py
    - control/test_eef_to_joint.py
    - control/vla_bridge/openvla_oft_server.py
    - control/vla_bridge/backends/openvla_oft.py
    - control/vla_bridge/backend_configs/openvla_oft.json
    - control/test_backend_openvla_oft.py
  modified:
    - control/requirements.txt
    - control/vla_bridge/policy_server.ipynb
    - control/vla_bridge/policy_server_launch.md
decisions:
  - "IK library: ikpy 4.1.0 (Chain.from_urdf_file with an explicit base_elements path to the URDF's eef link). ikpy parsed So-101.urdf without trouble, so pyroki was not needed"
  - "solve_ik is our own DLS loop over ikpy FK frames with an analytic geometric Jacobian, as the plan says. ikpy's scipy-optimizer IK is not used"
  - "5-DOF arm: a weighted task error (ORIENTATION_WEIGHT=0.1, so 0.1 rad counts the same as 1 cm). Position wins when a full 6-D target can't be met"
  - "solve_ik takes an optional joint_limits_deg and clamps to the URDF limits intersected with the calibrated limits. A non-finite delta holds the pose. Non-finite or missing current joints raise, so the HTTP source holds"
  - "OpenVLA-OFT raw action is scaled client-side: x0.05 m and x0.5 rad (robosuite OSC_POSE), rotvec converted exactly to extrinsic xyz. Gripper is binarized like openvla-oft's process_action (1=open -> 100 %)"
  - "Server loads through moojink/openvla-oft's own get_vla/get_proprio_projector/get_action_head/get_processor/get_vla_action (mirrors vla-scripts/deploy.py). A bare AutoModelForVision2Seq.from_pretrained does NOT load the OFT action head"
  - "Server returns the first action of each 8-step chunk. joint_state is accepted and ignored. proprio is the request's optional proprio_state, otherwise the checkpoint's dataset-mean proprio"
  - "Checkpoint pinned to moojink/openvla-7b-oft-finetuned-libero-object@4c89574e1c538b6c102f43f0526d60a9d3650148 (T-14-06)"
metrics:
  duration: "~45 min"
  completed: 2026-10-01
  tasks: 2
  files: 9
---

# Phase 14 Plan 03: OpenVLA-OFT Tier C backend + shared EEF-to-joint IK Summary

This plan adds `--backend openvla-oft`, the first backend on the Tier C resident-HTTP-adapter lane. It has a Colab-side Flask adapter that loads the pinned `moojink/openvla-7b-oft-finetuned-libero-object` revision through openvla-oft's own loaders. On the control side, a client scales the model's robosuite-OSC 7-D delta into metres and radians. It then converts that delta into an absolute SO-101 joint target with a new reusable IK module (`eef_to_joint.py`), whose chain is loaded from Phase 10's `So-101/So-101.urdf`. All of this happens before the unmodified safety validator runs.

## What Was Built

### Task 1: `backends/eef_to_joint.py` (commits 3161dea RED, 6098349 GREEN)
- **Chain:** `_load_chain()` (cached) calls `ikpy.chain.Chain.from_urdf_file(So-101/So-101.urdf, base_elements=[base … wrist_roll, right_hand_to_gripper, gripper_to_eef, eef])`.
  - The path is explicit because `right_gripper` branches into both the jaw mechanism and the `eef` tool frame.
  - It asserts that the active joints are exactly `JOINT_ORDER[:5]`. No link lengths or DH constants are hand-entered.
  - ikpy is imported lazily, because backend discovery imports this module.
- **`forward_kinematics(joint_deg)`** returns `(x, y, z, roll, pitch, yaw)` of the `eef` link in the URDF `base` frame (metres, extrinsic xyz radians).
- **`solve_ik(current_joint_deg, eef_delta, max_iters=50, tol=1e-4, joint_limits_deg=None)`:**
  - **Algorithm:** projected damped-least-squares (λ=0.02), with an analytic geometric Jacobian built from ikpy's per-link frames and a 0.2 rad/iteration step cap. It returns the best iterate.
  - **Target:** the current FK pose offset by `(dx, dy, dz)`, with rotation `R_delta @ R_current`.
  - **Clamping:** every iterate is clamped to the URDF limits intersected with the caller's calibrated limits.
  - **Gripper:** passed straight through (clamped 0-100). A missing or non-finite gripper keeps the current value.
  - **Speed:** about 3 ms for a small delta, about 11 ms for a 10 m unreachable delta.
- **Tests:** 19 in `control/test_eef_to_joint.py`:
  - zero-delta round trip
  - small reachable translation tracked within 2 mm
  - exact 6-D recovery of a known reachable configuration (<1 mm, <0.01 rad, joints within 0.5°), which validates the Jacobian
  - finite output for huge, NaN, inf and 1e300 deltas
  - clamping to calibrated limits
  - gripper pass-through and independence from the arm solve
  - non-finite and missing current joints rejected
- **Dependency:** `ikpy==4.1.0` pinned in `control/requirements.txt` and installed into the shared venv. It adds `scipy 1.18.1`; `sympy` was already present.

### Task 2: OpenVLA-OFT adapter + client (commits 97392b9 RED, 83979b6 GREEN)
- **`vla_bridge/openvla_oft_server.py`** follows the shape of `fastfs_server.py`:
  - **`OpenVLAOFTModel`:**
    - **Pinned download:** fetches the pinned revision with `snapshot_download(local_dir=…)`.
    - **Loading:** calls openvla-oft's `get_vla` (= `AutoModelForVision2Seq.from_pretrained(…, torch_dtype=bfloat16, trust_remote_code=True)`), `get_proprio_projector`, `get_action_head` and `get_processor` exactly once, in a mockable `_load_policy`.
    - **unnorm key:** resolved from `norm_stats`. A single key is auto-selected; with several, `--unnorm-key` is required.
    - **Inference:** `__call__(images, proprio_state, instruction)` passes `full_image`=overhead_left RGB and `wrist_image`=wrist RGB to `get_vla_action` and returns the first chunk action as 7 floats.
  - **`create_app(infer)`:**
    - `GET /health` reports `status=ready` plus metadata.
    - `POST /action` is guarded by a non-blocking `inference_lock`, so a busy server answers 429.
    - Bad bodies get 400: images missing or not PNG, empty or non-string instruction, wrong proprio dimension or non-finite proprio, a non-dict `joint_state`, or a non-JSON body.
    - It returns `{"action": [7 floats], "model_version": "<ckpt>@<sha12>", "inference_ms"}`.
  - **`main()`:**
    - CLI flags: `--repo`, `--checkpoint`, `--revision` (pinned by default), `--unnorm-key`, `--local-dir`, `--port 3001`.
    - Before loading, it puts the repo on `sys.path` and chdirs into it, because openvla-oft syncs `./prismatic/` modeling files.
    - It warms the model before `create_app(...).run(...)`.
- **`vla_bridge/backends/openvla_oft.py`:**
  - **Registration:** `@register("openvla-oft") build()` is on the HTTP lane, not in `GRPC_BRIDGE_BACKENDS`.
  - **Endpoint:** `--server-address`, falling back to the config's `endpoint_url`. If neither is set it raises a clear `SystemExit`, and it rejects an endpoint that isn't `http(s)://`.
  - **Checkpoint:** `--checkpoint`, falling back to the config's checkpoint. It is informational only.
  - **Conversion:** `make_convert_fn(joint_limits_deg)` runs `openvla_oft_action_to_eef_delta()` and then `eef_to_joint.solve_ik(...)`.
- **`backend_configs/openvla_oft.json`:** `{"checkpoint": "moojink/openvla-7b-oft-finetuned-libero-object", "revision": "4c89574e…", "endpoint_url": null}`.
- **Tests:** 42 in `control/test_backend_openvla_oft.py`:
  - load-once across requests
  - pinned download vs unpinned, and the unnorm-key rules
  - the 7-float contract and RGB decoding
  - proprio fallback
  - 12 malformed-body cases, 400 on a non-JSON body, and 429 when busy
  - warm-before-serve ordering in `main()`
  - registry and config checks
  - OSC scaling and the gripper convention
  - the convert_fn → solve_ik call
  - build() endpoint resolution
  - an **end-to-end test**: the real `create_app` behind the real `HttpAdapterActionSource` with the real ikpy convert_fn produces a finite, JOINT_ORDER-shaped, non-held action
- **Colab launch:** cells (`oft-*` ids) appended to `policy_server.ipynb`, and `policy_server_launch.md` regenerated with `scripts/sync_policy_notebook.py`. That md file is generated from the notebook. Running the sync also cleared the stale saved output of the existing `resident-fastfs-source` cell, which is the sync tool's normal behavior.

## Verification
- `cd control && <main .venv>/bin/python -m pytest -q`: **331 passed** (fork base 270; +19 eef_to_joint, +42 openvla-oft).
- `run_vla_episode.py --help` now lists `--backend {openvla-oft,scripted,smolvla}`.
- The generated `policy_server_launch.md` is byte-identical to a fresh render of the notebook.

## Live-Source Verification Record (execution discipline)
Every item below was checked by fetching live upstream source with curl on 2026-10-01:
- **HF Hub `moojink/*`:** five OFT LIBERO checkpoints (spatial, object, goal, 10, spatial-object-goal-10). `libero-object` is at sha `4c89574e1c538b6c102f43f0526d60a9d3650148`. Its file list includes `action_head--150000_checkpoint.pt`, `proprio_projector--150000_checkpoint.pt` and `dataset_statistics.json`, and its config uses the `auto_map` → `modeling_prismatic.OpenVLAForActionPrediction`.
  - **Stats:** the `dataset_statistics.json` key is `libero_object_no_noops`. Action min/max are ±0.9375 (normalized controller units), the gripper is in [0, 1], and proprio is 8-D. Each single-suite checkpoint has exactly one key; the combined one has four.
- **github.com/moojink/openvla-oft @ e4287e9:**
  - **Loading:** `experiments/robot/openvla_utils.py` (`get_vla`, `get_proprio_projector`, `get_action_head`, `get_vla_action`, `prepare_images_for_vla`, the local-checkpoint path via `find_checkpoint_file`).
  - **Constants:** `prismatic/vla/constants.py` (LIBERO: `NUM_ACTIONS_CHUNK=8`, `ACTION_DIM=7`, `PROPRIO_DIM=8`, BOUNDS_Q99).
  - **Eval config:** `experiments/robot/libero/run_libero_eval.py` (`GenerateConfig`; proprio = eef pos + axis-angle + gripper qpos; `process_action` = binarize + invert gripper).
  - **Server and env:** `vla-scripts/deploy.py` (the official server's load order), `pyproject.toml` (torch 2.2.0, TF 2.15.0, transformers fork, dlimp fork), and `SETUP.md` (Python 3.10).
  - **Model card link:** the card points to `github.com/openvla-oft/openvla-oft`, which **returns 404**. The real repo is `moojink/openvla-oft`.
- **robosuite v1.4.1:**
  - `controllers/config/osc_pose.json` has output_max `[0.05]*3 + [0.5]*3`.
  - `utils/control_utils.set_goal_orientation` treats the delta as base-frame axis-angle, left-multiplied.
  - LIBERO's `env_wrapper.py` uses `controller="OSC_POSE"`.
- **ikpy 4.1.0 (installed source):** checked `Chain.from_urdf_file` (`base_elements`, `active_links_mask`, `symbolic`), `forward_kinematics(full_kinematics=True)`, and `URDFLink.rotation` / frame composition.
- **Not verified, and stated as such:**
  - The Colab install cells (uv / Python 3.10 venv, `pip install -e` of openvla-oft) have not been executed.
  - The sign of each real joint against its URDF axis is unverified. Phase 10 checked limits and gripper direction only; arm-joint directions were never checked.
  - The LIBERO-world-frame → SO-101-base-frame mapping is an unverified identity.
  - Real OFT inference has not been run here: no GPU, and the 7B weights were never downloaded.

## Package Legitimacy Evidence (for the human check)
- **ikpy** (PyPI `ikpy`, version 4.1.0, released 2026-08-30): 31 releases since 2015-12-17. Its home page is `github.com/Phylliade/ikpy`, created 2015-09-22, Apache-2.0, about 1036 stars and 179 forks, not archived, last pushed 2026-08-30. Tag `v4.1.0` is at commit `ce008ff`. Its dependencies are numpy, scipy and sympy.
- **The 5 GitHub orgs** from 14-RESEARCH.md's audit (`openvla/openvla`, `octo-models/octo`, `thu-ml/RoboticsDiffusionTransformer`, `NVIDIA/Isaac-GR00T`, `X-Square-Robot/wall-x`) were not re-audited here. Nothing was cloned. They are left for the human.
- **New and not in 14-RESEARCH.md's audit.** OpenVLA-OFT's code lives in **`github.com/moojink/openvla-oft`**, not `openvla/openvla`. It is a fork by Moo Jin Kim, the paper's first author; about 1404 stars; HEAD `e4287e9` pushed 2025-09-09; and its pyproject names authors Kim, Finn and Liang. Its `pip install -e .` also pulls **`moojink/transformers-openvla-oft`** and **`moojink/dlimp_openvla`**. The Colab cells additionally `pip install uv`. All four need human legitimacy sign-off before the first Colab run.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] The server accepts the real 14-01 wire contract**
- **Found during:** Task 2.
- **Issue:** The plan's server reads a `proprio_state` list. But `HttpAdapterActionSource` (14-01) sends `joint_state` as a dict and no `proprio_state`, so a server that required `proprio_state` would 400 on every live request.
- **Fix:** `/action` accepts `joint_state` (validated, then ignored) plus an optional 8-float `proprio_state`. When proprio is missing it uses the checkpoint's dataset-mean proprio. SO-101 joint degrees are not LIBERO's Franka EEF proprio, so feeding them in would be meaningless.
- **Files and commit:** openvla_oft_server.py, commit 83979b6.

**2. [Rule 1 - Bug] Scaling robosuite OSC units before IK**
- **Found during:** Task 2, while verifying the checkpoint's action statistics.
- **Issue:** The plan passes `raw_action[0..5]` straight into `solve_ik` as metres and radians. In fact they are robosuite OSC_POSE normalized units (up to ±0.94): a raw 0.9 would be read as 0.9 m instead of 4.5 cm. The rotation is also an axis-angle vector, not RPY. The gripper is in [0, 1], not percent.
- **Fix:** `openvla_oft_action_to_eef_delta()` scales position by 0.05 and rotation by 0.5, converts the rotvec exactly to extrinsic xyz, and maps the gripper via openvla-oft's own binarization (1 = open → 100 %). `solve_ik` is still the function the convert_fn calls, as the plan's test requires.
- **Files and commit:** backends/openvla_oft.py, commit 83979b6.

**3. [Rule 1 - Bug] The OFT action head needs openvla-oft's loaders**
- **Found during:** Task 2.
- **Issue:** The plan loads the model with `AutoModelForVision2Seq.from_pretrained(..., trust_remote_code=True)` alone. That loads only the VLA backbone. OFT's continuous actions need the separate L1-regression action head and the proprio projector `.pt` files, which only openvla-oft's `get_action_head` and `get_proprio_projector` load. `get_vla` does exactly that `from_pretrained` call internally.
- **Fix:** `_load_policy` follows `vla-scripts/deploy.py`. `--repo` becomes the openvla-oft checkout. The server also chdirs into it, because openvla-oft syncs files from `./prismatic/`.
- **Files and commit:** openvla_oft_server.py, commit 83979b6.

**4. [Rule 2 - Security, T-14-06] A pinned checkpoint revision that is actually honored**
- **Issue:** openvla-oft's hub path calls `from_pretrained(repo_id)` and `hf_hub_download` with no revision, so it always resolves `main`.
- **Fix:** By default the server downloads the reviewed sha into a real local directory and loads that through openvla-oft's documented local-checkpoint path. `--revision ''` is the explicit opt-out.
- **Commit:** 83979b6.

**5. [Rule 2 - Correctness] `solve_ik` accepts the calibrated limits**
- **Issue:** The plan wants output clamped to `load_joint_limits_deg()` ranges, but that call reads the live calibration file. It is unavailable in tests, and `build()` already receives the limits.
- **Fix:** `solve_ik(..., joint_limits_deg=None)` clamps to the URDF limits intersected with the given limits, and `build()` forwards its `joint_limits_deg`.
- **Commit:** 6098349.

**6. [Rule 2 - Correctness] `build()` rejects non-HTTP endpoints**
- **Issue:** A gRPC-style `host:port` given as `--server-address` would make every step fail silently, with `requests` raising `InvalidSchema` → hold.
- **Fix:** `build()` now fails fast with a clear `SystemExit`.
- **Commit:** 83979b6.

**7. [Rule 3 - Blocking] Colab launch steps were missing**
- **Issue:** The live human check says to launch the server "per policy_server_launch.md's pattern", and no such section existed. `policy_server_launch.md` is GENERATED from `policy_server.ipynb`, so editing the md directly would be overwritten by the next sync.
- **Fix:** Added `oft-*` cells to the notebook (isolated Python 3.10 uv venv, pinned openvla-oft commit, subprocess launch with a `/health` wait, HTTP tunnel, teardown) and regenerated the md.
- **Commit:** 83979b6.

**8. Test correction during GREEN (Task 1)**
- My first "reachable configuration" test checked a position-only delta against an orientation-changing goal, and the 5-DOF weighted trade-off left 3.8 mm of error. I rewrote it as a full 6-D delta generated by FK, which converges to <1 mm and also validates the Jacobian. I also replaced a weak spy-based gripper test with a behavioral one. Both changes are in commit 6098349.

## Known Stubs
None. The `endpoint_url: null` in `openvla_oft.json` is intentional: it is a per-session ngrok URL supplied with `--server-address`.

## Threat Flags
None beyond the plan's register:
- **T-14-05:** the clamp to URDF and calibrated limits plus the unmodified validator are both in place.
- **T-14-06:** the revision is pinned, and the remaining GitHub forks are flagged below.
- **T-14-07:** 429 when busy.

The Colab adapter is a new HTTP endpoint behind an ngrok tunnel, the same exposure class as the existing FastFS depth service. It accepts only images, an instruction and numbers, never files or paths.

## Human Checks Deferred
1. **Package legitimacy (Task 1):**
   - **test:** Confirm `pypi.org/project/ikpy` is the official project (`github.com/Phylliade/ikpy`; evidence above). Confirm the 5 audited GitHub URLs (`github.com/openvla/openvla`, `github.com/octo-models/octo`, `github.com/thu-ml/RoboticsDiffusionTransformer`, `github.com/NVIDIA/Isaac-GR00T`, `github.com/X-Square-Robot/wall-x`) match 14-RESEARCH.md. **Also** confirm the new, unaudited ones that OpenVLA-OFT actually needs: `github.com/moojink/openvla-oft` (@ `e4287e9`), `github.com/moojink/transformers-openvla-oft`, `github.com/moojink/dlimp_openvla`, and PyPI `uv` (used by the Colab cell).
   - **expected:** every URL is a first-party, official project page with no name or org mismatch.
   - **why_human:** legitimacy of a new PyPI dependency and of GitHub-source installs is a human judgment call under this project's protocol.
2. **Live hardware (Task 2):**
   - **test:** On a live Colab A100/L4, run the `oft-*` cells in `policy_server.ipynb` and confirm `/health` says ready. Then run `python control/run_vla_episode.py --backend openvla-oft --server-address https://<ngrok-url> --out control/outputs/<new_dir> --max-steps 20` against the real SO-ARM101.
   - **expected:** at least one step logs a validated, non-held action (a `validator_flags` entry that is not a hold or error placeholder), and the robot visibly moves.
   - **why_human:** this needs real hardware and a live Colab GPU session. The Colab install cells are also unexecuted.
3. **IK frame and sign sanity (new, from implementation):**
   - **test:** During or before check 2, hold the arm still and command a small known delta (e.g. dz=+1 cm), via a scripted call to `eef_to_joint.solve_ik` plus `send_action`, or by reading the logged executed action. Check whether the gripper tip actually moves up and by about the commanded amount.
   - **expected:** motion direction matches the commanded base-frame axis.
   - **why_human:** the real-joint-sign vs URDF-axis mapping (degree 0 = URDF 0, same sign) and the identity LIBERO→SO-101 frame mapping cannot be checked without hardware. If the direction is wrong, flip the sign or frame in `eef_to_joint`/`openvla_oft` before Wave 4 reuses it.
4. **Carried from 14-01:** the wrist camera tolerates two independent `cv2.VideoCapture` opens on this HTTP lane (the backend's own plus IOLogger's).

## Notes for Wave 4 (OpenVLA, Octo, GR00T N1.7)
- **Reuse:** call `eef_to_joint.solve_ik(joint_state, eef_delta, joint_limits_deg=...)` and put model-specific unit scaling in the backend's own convert_fn, following `openvla_oft.openvla_oft_action_to_eef_delta`. The delta must be in metres and extrinsic-xyz radians; the gripper is absolute percent.
- **OpenVLA base:** `openvla/openvla-7b` (Bridge/OXE) uses a different un-normalization: discrete tokens via `predict_action(unnorm_key=...)`. It also has a different gripper and frame convention than OFT/LIBERO. Verify it per checkpoint rather than copying OFT's scaling.

## Self-Check: PASSED
- FOUND: control/vla_bridge/backends/eef_to_joint.py, control/test_eef_to_joint.py, control/vla_bridge/openvla_oft_server.py, control/vla_bridge/backends/openvla_oft.py, control/vla_bridge/backend_configs/openvla_oft.json, control/test_backend_openvla_oft.py
- FOUND commits: 3161dea, 6098349, 97392b9, 83979b6
