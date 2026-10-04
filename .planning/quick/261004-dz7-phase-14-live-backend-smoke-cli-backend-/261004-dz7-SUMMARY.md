---
phase: quick-261004-dz7
plan: 01
subsystem: control/vla_bridge (Phase 14 VLA backends, live verification tooling)
tags: [smoke-test, vla-backends, grpc, http-adapter, gemini, hardware-isolation, uat-prep]
status: complete
requires:
  - vla_bridge.backends.BACKEND_REGISTRY (Phase 14 registry and the 14 VLA backend factories)
  - vla_bridge.mllm_smoke (synthetic_frame, WRIST_SHAPE, STEREO_SHAPE, NEUTRAL_JOINT_LIMITS_DEG)
  - vla_bridge.safety_validator.validate_action
provides:
  - "python -m vla_bridge.backend_smoke: per-backend live smoke with PASS/WARN/FAIL and exit code"
  - "synthetic_hardware(): reusable context that replaces every robot/camera construction seam"
  - ".planning/phases/14-vla-style-pipeline-backends/14-LIVE-RESULTS.md template"
affects:
  - "Phase 14 hardware UAT (14-UAT.md tests 1, 2, 6-18): smoke PASS is now a precondition"
tech-stack:
  added: []
  patterns:
    - "unittest.mock.patch.object seams inside contextlib.ExitStack, restored on exit"
    - "tripwires on the real SO101Follower.connect / StereoSplitCamera.__init__"
    - "pure evaluate_step() check functions, unit-tested without any server"
key-files:
  created:
    - control/vla_bridge/backend_smoke.py
    - control/test_backend_smoke.py
    - .planning/phases/14-vla-style-pipeline-backends/14-LIVE-RESULTS.md
  modified: []
decisions:
  - "gRPC smoke latency is the observation-to-action age (last_details action_age_s), not the time since the first poll; the poll wait is printed separately"
  - "Held-step flag detection matches holding-position / holding position / held-position / stale observation (narrower than a bare 'holding' substring, which would also match the validator's per-joint non-finite flag)"
  - "Gemini confirmed-path test asserts PASS: er_conversion produces a real bounded move at the synthetic midpoint state, so the held-FAIL fallback was not needed"
  - "requirements: [] kept (orchestrator decision); 640x480 synthetic wrist frame kept"
metrics:
  duration: "~12 min wall clock for execution (start 2026-10-04T04:47:35Z)"
  completed: 2026-10-04
  tasks: 3
  files: 3
  tests: "test_backend_smoke.py 70 passed; neighbouring suites 336 passed; full control suite 1674 passed"
---

# Quick Task 261004-dz7: Phase 14 Live Backend Smoke CLI Summary

`python -m vla_bridge.backend_smoke --backend NAME` tests one Phase 14 VLA backend end to end
against its live Colab gRPC PolicyServer, Colab HTTP adapter or the Gemini Robotics-ER API. It
calls the real `BACKEND_REGISTRY[NAME]` factory and the real `ActionSource`. Every robot and camera
construction seam is swapped for a synthetic double, and tripwires guard the real classes, so the
arm cannot move. The tool prints PASS/WARN/FAIL lines and exits 0 (pass), 1 (fail) or 2 (Gemini run
refused without `--confirm-spend`). `14-LIVE-RESULTS.md` holds, for each backend, the Colab cell
ids, GPU tier and smoke command.

## What was built

- **`control/vla_bridge/backend_smoke.py`**
  - Lanes: gRPC = act, diffusion, groot-n17, pi0, pi05, smolvla. HTTP = groot-n15, octo, openvla,
    openvla-oft, rdt1b, wallx. Gemini = gemini-robotics, gemini-robotics-2. `lane_of()` refuses
    the others: mllm-* points to `vla_bridge.mllm_smoke`, scripted has no server, and an unknown
    name lists every backend that can be smoked. A drift guard catches any backend that is
    registered but not in a lane.
  - `synthetic_hardware(joint_state)` patches `lerobot.async_inference.robot_client.make_robot_from_config`,
    `SO101Follower` in `base_http_adapter`/`base_gemini`, `StereoSplitCamera` in `robot_client`/`base_http_adapter`/`base_gemini`,
    and `cv2.VideoCapture`. It also sets tripwires on the real `SO101Follower.connect` and
    `StereoSplitCamera.__init__`. `SyntheticRobot.send_action` raises. Every patch is restored on
    exit, including after an exception.
  - Pre-flight checks run before anything is built:
    - gRPC: TCP connect to the server address.
    - HTTP: `GET /health`, printing every field and every model metadata key, and requiring
      `status == ready` and `protocol_version == 1` before any `/action` POST.
    - Gemini: the `--confirm-spend` gate. Without it the tool exits 2 before the key is read or
      any client is built.
  - Runners:
    - HTTP and Gemini make `--steps` timed `get_action()` calls.
    - gRPC polls until `--steps` non-held actions arrive or `--wait-s` runs out.
    - Teardown mirrors `run_vla_episode._shutdown_backend`. Afterwards a hardware-isolation
      check runs.
  - Checks for each step:
    - action returned and schema valid;
    - not held;
    - re-validation with `safety_validator.validate_action`, as on the first tick at 2 Hz;
    - saturation (FAIL when all five arm joints are pinned, the 14-UAT 13/18 units-error symptom);
    - latency;
    - model_version (never FAIL).
- **`control/test_backend_smoke.py`** (70 tests, offline) covers:
  - lane membership against the live registry;
  - seam patching and restore (including after an exception), plus a scan that fails if a new
    construction site appears;
  - the SyntheticRobot surface, and tripwires;
  - classification of held, saturated, validator, latency and model_version results;
  - HTTP pass, not-ready, unreachable, held (Timeout and HTTP 500) and saturated cases, plus
    neutral limits when the calibration file is missing;
  - gRPC pass, held, unreachable and observation-to-action latency, plus one REAL lerobot
    `RobotClient` built on SyntheticRobot whose handshake to a closed loopback port is refused;
  - the Gemini spend guard and the confirmed path, with a sentinel key that never appears in the
    output;
  - argument errors.
- **`.planning/phases/14-vla-style-pipeline-backends/14-LIVE-RESULTS.md`**: a setup row and a PENDING
  smoke row for each of the 14 backends.

## Before the hardware UAT

Run each smoke from `control/` after starting that backend's server. Add `2>&1 | grep -v objc` to hide
the harmless cv2/av duplicate-class warnings. The source and full notes for every row are in
`.planning/phases/14-vla-style-pipeline-backends/14-LIVE-RESULTS.md` section 1.

The gRPC base flow cells are `9a4bc6ec, b9d8ba70, 208524cb, 0bOMKSqN5aMt, u33d6k5o1lL5, 22702a39, cbaf7af4, d821252e, fc7fd0ee`.
They install lerobot[async] and pyngrok, set up the HF and ngrok tokens, start the stock PolicyServer on port 5173 and open the TCP tunnel. Teardown is `2c44f9a4`.
The Tier C prerequisites are `9a4bc6ec, 208524cb, 22702a39`. Upload `<backend>_server.py` to `/content/` before running the launch cell.

| backend | lane | default checkpoint | notebook cell ids / server script | GPU tier | tokens / licences | tunnel / port | smoke command |
|---------|------|--------------------|-----------------------------------|----------|-------------------|---------------|---------------|
| smolvla | gRPC | victorvanhalst/smolvla_so101_cube | gRPC base flow (stock PolicyServer) | not stated in notebook (11-RESEARCH: T4/A100, ~2 GB) | HF + ngrok (22702a39) | ngrok TCP -> 5173 | `.venv/bin/python -m vla_bridge.backend_smoke --backend smolvla --server-address <ngrok_tcp_host:port> --checkpoint victorvanhalst/smolvla_so101_cube` |
| pi0 | gRPC | amindt/pi0_so101_pickandplace_redblue | gRPC base flow | not stated in notebook (checkpoint_candidates: prefer L4/A100, bf16 ~7 GB) | HF account must accept gated google/paligemma-3b-pt-224; ngrok | ngrok TCP -> 5173 | `.venv/bin/python -m vla_bridge.backend_smoke --backend pi0 --server-address <ngrok_tcp_host:port> --checkpoint amindt/pi0_so101_pickandplace_redblue` |
| pi05 | gRPC | yathAg/pi05_so101_pick_place | gRPC base flow | not stated in notebook (prefer L4/A100) | same paligemma licence; ngrok | ngrok TCP -> 5173 | `.venv/bin/python -m vla_bridge.backend_smoke --backend pi05 --server-address <ngrok_tcp_host:port> --checkpoint yathAg/pi05_so101_pick_place` |
| act | gRPC | hrl7/so101-pregrasp-act-l4 | gRPC base flow | not stated in notebook (checkpoint_candidates: T4 enough) | HF + ngrok | ngrok TCP -> 5173 | `.venv/bin/python -m vla_bridge.backend_smoke --backend act --server-address <ngrok_tcp_host:port> --checkpoint hrl7/so101-pregrasp-act-l4` |
| diffusion | gRPC | hrl7/so101-touch-diffusion-l4 | 9a4bc6ec, b9d8ba70, 208524cb, 22702a39, cbaf7af4, diffusion-install, diffusion-launch (`diffusion_policy_server.py`, replaces d821252e), fc7fd0ee | T4 enough | HF + ngrok | ngrok TCP -> 5173 | `.venv/bin/python -m vla_bridge.backend_smoke --backend diffusion --server-address <ngrok_tcp_host:port> --checkpoint hrl7/so101-touch-diffusion-l4` |
| groot-n17 | gRPC | tsangb34/groot-n17-so101-cube-drawer-40k | 9a4bc6ec, b9d8ba70, 208524cb, 22702a39, cbaf7af4, groot-n17-install, groot-n17-launch (`groot_n17_server.py`, replaces d821252e), fc7fd0ee | L4 or A100 (not T4) | HF account must accept gated nvidia/Cosmos-Reason2-2B; ngrok | ngrok TCP -> 5173 | `.venv/bin/python -m vla_bridge.backend_smoke --backend groot-n17 --server-address <ngrok_tcp_host:port> --checkpoint tsangb34/groot-n17-so101-cube-drawer-40k --wait-s 300` |
| openvla-oft | HTTP | moojink/openvla-7b-oft-finetuned-libero-object | Tier C prereqs, oft-install, oft-launch (`openvla_oft_server.py`), oft-tunnel; teardown oft-teardown | A100 or L4 (~16 GB VRAM) | ngrok; 14-UAT 4 legitimacy sign-off first | ngrok HTTP -> 3001 | `.venv/bin/python -m vla_bridge.backend_smoke --backend openvla-oft --server-address https://<oft_tunnel>` |
| wallx | HTTP | Zekai-Chen/wallx_pick_place_3cam_200ep@ef8ce55f9772 | Tier C prereqs, wallx-install, wallx-launch (`wallx_server.py`), wallx-tunnel; teardown wallx-teardown | A100 + High-RAM | ngrok | ngrok HTTP -> 3002 | `.venv/bin/python -m vla_bridge.backend_smoke --backend wallx --server-address https://<wallx_tunnel> --instruction 'Put the object in the bin'` |
| openvla | HTTP | toasta18/openvla-so101-cylinder@f00833ce5851 | Tier C prereqs, openvla-install, openvla-launch (`openvla_server.py`), openvla-tunnel; teardown openvla-teardown | A100 or L4 (~16 GB VRAM) | ngrok; 14-UAT 5 licence decision | ngrok HTTP -> 3003 | `.venv/bin/python -m vla_bridge.backend_smoke --backend openvla --server-address https://<openvla_tunnel> --instruction 'Pick up the black cylinder and place it on the plate.'` |
| octo | HTTP | sapanostic/octo-so101-3hz-finetune@a45ff970 | Tier C prereqs, octo-install, octo-launch (`octo_server.py`), octo-tunnel; teardown octo-teardown | any Colab GPU | ngrok; 14-UAT 5 licence decision | ngrok HTTP -> 3004 | `.venv/bin/python -m vla_bridge.backend_smoke --backend octo --server-address https://<octo_tunnel> --instruction 'pick up the cube'` |
| rdt1b | HTTP | robotics-diffusion-transformer/rdt-1b@eb09036 | Tier C prereqs, rdt1b-install, rdt1b-launch (`rdt1b_server.py`), rdt1b-tunnel; teardown rdt1b-teardown | A100 (Colab Pro), ~13 GB VRAM, ~55 GB disk | ngrok | ngrok HTTP -> 3005 | `.venv/bin/python -m vla_bridge.backend_smoke --backend rdt1b --server-address https://<rdt_tunnel> --instruction 'pick up the cube'` |
| groot-n15 | HTTP | quangnd58/GR00T-N1.5-3B-so101-multitask@488f3b6 | Tier C prereqs, groot-n15-install, groot-n15-launch (`groot_n15_server.py`), groot-n15-tunnel; teardown groot-n15-teardown | L4 or A100 (Ampere+, ~6 GB VRAM, ~10 GB disk) | ngrok | ngrok HTTP -> 3006 | `.venv/bin/python -m vla_bridge.backend_smoke --backend groot-n15 --server-address https://<groot_n15_tunnel> --instruction 'Pick blue cube and place on red cube'` |
| gemini-robotics | Gemini | model gemini-robotics-er-2-preview | none (no Colab) | none | RESTRICTED `GEMINI_API_KEY` exported in the local shell; paid per step | none (direct HTTPS) | `export GEMINI_API_KEY=...` then `.venv/bin/python -m vla_bridge.backend_smoke --backend gemini-robotics --instruction '<object>' --confirm-spend` |
| gemini-robotics-2 | Gemini | model gemini-robotics-er-2-preview | none (no Colab) | none | same key; paid per step | none | `.venv/bin/python -m vla_bridge.backend_smoke --backend gemini-robotics-2 --instruction '<object>' --confirm-spend` |

Paste the smoke results into `14-LIVE-RESULTS.md` section 2, then run the matching 14-UAT hardware test.

## Commits

| Task | Commit | Message |
|------|--------|---------|
| 1 (RED) | b27d9ee | test(261004-dz7): add failing tests for backend_smoke seams and step checks |
| 1 (GREEN) | 1df11c2 | feat(261004-dz7): backend_smoke synthetic hardware seams, lanes and step checks |
| 2 (RED) | 40307a5 | test(261004-dz7): add failing end-to-end tests for backend_smoke CLI lanes |
| 2 (GREEN) | 7fd5af2 | feat(261004-dz7): backend_smoke lane runners and CLI main() |
| 2 (fix) | b13b45c | fix(261004-dz7): measure gRPC smoke latency as observation-to-action age |
| 3 | f7ee99d | docs(261004-dz7): add 14-LIVE-RESULTS.md backend smoke template |

## Verification run

All commands below ran from the worktree's `control/`, using the main checkout's venv
(`/Users/hp/Desktop/Work/Repositories/SoARM-Research/control/.venv/bin/python`). `vla_bridge` resolved to
the worktree copy, and lerobot, grpc and requests were all installed, so nothing had to be mocked at
import time.

- `python -m pytest test_backend_smoke.py -q` -> 70 passed.
- `python -m pytest test_backend_smoke.py test_mllm_smoke.py test_backend_registry.py test_robot_client.py test_run_vla_episode.py test_base_gemini.py -q` -> 336 passed.
- `python -m pytest -q` (full control suite) -> 1674 passed.
- `python -m vla_bridge.backend_smoke --help` -> exit 0. `--backend no-such-backend` -> exit 1, listing the 14 backends that can be smoked. `--backend gemini-robotics` without `--confirm-spend` -> exit 2, and a sentinel key never appeared in the output.
- 14-LIVE-RESULTS.md template check -> TEMPLATE_OK.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] gRPC latency measured the cold-start wait instead of the action round trip**
- **Found during:** Task 3 (checking each backend's first-chunk behaviour against the notebook).
- **Issue:** The plan defined gRPC latency as the time from the first poll to the first non-held action. A cold server (GR00T N1.7, pi0) can take more than 30 s to produce its first chunk. That would be a FAIL latency even when every action's observation-to-action round trip is well inside STALE_ACTION_S, because the validator itself holds actions older than 30 s.
- **Fix:** The latency check now uses `BridgeActionSource.last_details["action_age_s"]`, the same age the validator compares with STALE_ACTION_S. It falls back to the poll wait when that value is missing. The poll wait is printed on its own line.
- **Files modified:** control/vla_bridge/backend_smoke.py, control/test_backend_smoke.py
- **Commit:** b13b45c

**2. [Rule 2 - Correctness] Narrower held-flag matching**
- The plan's held detection matched any bridge flag containing "holding". The validator's per-joint non-finite flag ("..., holding current state") also contains that word. So matching uses "holding-position", "holding position", "held-position" and "stale observation", plus `robot_client._is_stale_or_empty`. A non-finite value is still a FAIL, through the schema/validator check. Committed in 1df11c2.

**3. [Additions] Helpers and checks beyond the plan**
- `CONFIG_NAMES` also maps the six gRPC backends, so the header can show the camera_map from the backend config.
- Every run ends with a `PASS hardware isolation` line (the plan only required the FAIL line).
- `classify_validator_flags`, `check_saturation`, `check_latency`, `check_model_version` and `hold_reason` are exported for the tests.
- `--wait-s` and `--poll-interval-s` must be positive.

### Notebook hash check (orchestrator instruction)
The plan's sha256 check (`5cf5559e...`) is the hash of the user's uncommitted edit in the main
checkout, which this worktree does not have. The worktree copy is the committed version
(sha256 `3b6bd1b8...`). It was only read, and `git diff --quiet HEAD -- control/vla_bridge/policy_server.ipynb`
confirmed it is unchanged. A read-only `shasum` of the main checkout's notebook still printed
`5cf5559ef156dae74b8b22dcd144bfae87bd4bcf414a42588f3393c2a805b8c5`. The notebook was never staged.
If the user's uncommitted edit changes cell ids, re-check the cell ids in 14-LIVE-RESULTS.md against
that version.

### Gemini confirmed-path test outcome
Before writing the test, I ran `er_conversion.convert_er_output_to_action` at the synthetic midpoint
state (from the mock calibration) with the point `[250, 750]` on a 480x640 frame. The FK heading
radius is about 0.33 m, well above `MIN_HEADING_RADIUS_M`. IK returned a real bounded move
(shoulder_pan +0.50, shoulder_lift +1.83, elbow_flex -2.24, wrist_flex +0.40 deg). So the test
asserts the PASS outcome: exit 0, `PASS not held`, and model_version `gemini-robotics@gemini-er-conversion`.
It also checks that the fake client was called exactly `--steps` (2) times and that the sentinel key
never appears in stdout or stderr. The held-FAIL fallback was not needed.

## Known Stubs

None. The `<ngrok_tcp_host:port>`, `https://<tunnel>` and `'<object>'` strings in 14-LIVE-RESULTS.md are
command placeholders for the operator, and the PENDING cells are intentional (to be filled from live runs).

## Threat Flags

None. The tool opens no new endpoint and adds no auth path. It acts only as a client of the existing
servers, and every threat-register mitigation (T-dz7-01..06) is implemented and covered by tests.

## Self-Check: PASSED

- FOUND: control/vla_bridge/backend_smoke.py
- FOUND: control/test_backend_smoke.py
- FOUND: .planning/phases/14-vla-style-pipeline-backends/14-LIVE-RESULTS.md
- FOUND commits: b27d9ee, 1df11c2, 40307a5, 7fd5af2, b13b45c, f7ee99d
