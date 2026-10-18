<!-- refreshed: 2026-10-18 -->
# Architecture

**Analysis Date:** 2026-10-18

## System Overview

```text
┌──────────────────────────────────────────────────────────────────────────┐
│               Real-Hardware VLA/MLLM Harness (Mac, local)                 │
│   `control/run_vla_episode.py`  (CLI: --backend <name>)                   │
│   pre-flight: `python -m vla_bridge.backend_smoke --backend <name>`       │
├───────────────┬───────────────────┬───────────────────┬──────────────────┤
│ Backend        │ ActionSource       │ Safety gate        │ I/O + sensors    │
│ registry       │ (per backend)      │ `vla_bridge/       │ `io_logger.py`,  │
│ `vla_bridge/   │ `vla_bridge/       │  safety_validator  │ `stereo_camera`, │
│  backends/     │  backends/*.py`    │  .py` +            │ `depth_camera`,  │
│  __init__.py`  │                    │ `action_contract`  │ `depth_recorder` │
└──────┬────────┴─────────┬─────────┴─────────┬─────────┴──────────────────┘
       │ Tier A/B gRPC     │ Tier C HTTP        │ Direct cloud HTTPS
       │ (ngrok TCP 5173)  │ (ngrok HTTP 3001-6)│
       ▼                   ▼                    ▼
┌──────────────────┐ ┌────────────────────┐ ┌──────────────────────────────┐
│ Colab LeRobot     │ │ Colab resident HTTP │ │ Gemini / Claude / GPT / HF   │
│ PolicyServer      │ │ adapter servers     │ │ MLLM providers               │
│ (stock or patched │ │ `vla_bridge/*_      │ │ `backends/base_mllm*.py`,    │
│ `diffusion_policy_│ │  server.py`         │ │ `backends/base_gemini.py`    │
│ server.py`,       │ │ + FastFS depth      │ │                              │
│ `groot_n17_server`│ │ `fastfs_server.py`  │ │                              │
└──────────────────┘ └────────────────────┘ └──────────────────────────────┘
   ▲ hosted by one generated notebook per service:
   │ `control/vla_bridge/notebooks/{grpc_*,http_*,fastfs_depth}.ipynb`
   │ built by `scripts/build_colab_notebooks.py` from `scripts/colab_notebook_specs.py`
   │ (embeds `*_server.py` + `colab_server_utils.py` byte-identical, sha256-verified)
       │
       ▼ validated joint targets
┌──────────────────────────────────────────────────────────────────────────┐
│ SO-ARM101 follower (LeRobot `SO101Follower`, Feetech STS3215 bus)         │
│ helpers: `control/keyboard_joint_control.py`, `diagnostics/servo_*.py`    │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ Simulation track (Colab / MuJoCo): `LIBERO/` (vendored, SOARM robot added │
│ under `LIBERO/libero/libero/assets/robots/soarm101`), notebooks           │
│ `LIBERO/notebooks/0*.ipynb`, explorations `explorations/`                 │
└──────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Episode harness | Control loop, camera open, waypoint interpolation, e-stop, return-to-start, termination record | `control/run_vla_episode.py` |
| Backend registry | Auto-discover backend modules, `@register`/`@register_grpc_backend`, load per-backend JSON config | `control/vla_bridge/backends/__init__.py` |
| Backend configs | Per-backend checkpoint/endpoint/action-space/camera_map | `control/vla_bridge/backend_configs/*.json` |
| Backend smoke | Robot-free live check: runs the real registered factory + `get_action()` on synthetic frames | `control/vla_bridge/backend_smoke.py`, `control/vla_bridge/mllm_smoke.py` |
| gRPC lane base | LeRobot PolicyServer client wrapper + action conversion | `control/vla_bridge/backends/base_grpc.py`, `control/vla_bridge/robot_client.py` |
| HTTP adapter base | Tier C resident-server client (PNG obs, joint action) | `control/vla_bridge/backends/base_http_adapter.py` |
| MLLM base | Chunked joint-target prompting, JSON schema parsing, holds, failure cap | `control/vla_bridge/backends/base_mllm.py`, `base_mllm_build.py`, `base_mllm_chat.py` |
| Gemini base | Gemini Robotics-ER direct-cloud lane (no notebook) | `control/vla_bridge/backends/base_gemini.py` |
| Action conversions | EEF/relative-EE/ER to joint targets | `control/vla_bridge/backends/eef_to_joint.py`, `relative_ee_conversion.py`, `er_conversion.py` |
| MLLM support | Inputs (images/depth), prompt, schema, URDF context, recording | `control/vla_bridge/mllm_inputs.py`, `mllm_prompt.py`, `mllm_schema.py`, `mllm_urdf.py`, `mllm_record.py` |
| Safety validator | Joint-limit clamp, NaN reject, per-step + velocity caps, stale-obs hold | `control/vla_bridge/safety_validator.py` |
| Action contract | `JOINT_ORDER`, calibration ticks to degrees, joint limits | `control/vla_bridge/action_contract.py` |
| Stereo camera | AR0144 side-by-side capture split into left/right (FFmpeg AVFoundation) | `control/vla_bridge/stereo_camera.py` (`StereoSplitCamera`) |
| Depth pipeline | Checkerboard stereo calibration, rectify + POST to FastFS, save depth `.npy` | `control/vla_bridge/stereo_calibration.py`, `depth_camera.py` (`DepthCameraClient`), `depth_recorder.py` (`DepthRecorder`), `control/stereo_calibration.json`, `control/render_depth_heatmaps.py` |
| Colab model servers | Resident HTTP servers (OpenVLA, OFT, Octo, GR00T N1.5, RDT, WallX, FastFS) and patched gRPC launchers (Diffusion, GR00T N1.7) | `control/vla_bridge/*_server.py` |
| Colab process helper | Stop stale servers, start subprocess, wait for port ownership, log/GPU/health reports | `control/vla_bridge/colab_server_utils.py` |
| Notebook generator | Render 13 per-backend Colab notebooks + README from specs; `--check` drift gate | `scripts/build_colab_notebooks.py`, `scripts/colab_notebook_specs.py`, `control/test_colab_notebooks.py` |
| Hardware teleop/record/tuning | Jog, keyboard control, calibration, episode recording, `--execution-hz` tuning | `control/joint_jog.py`, `control/keyboard_joint_control.py`, `control/record_episode.py`, `control/tune_execution_hz.py` |
| Diagnostics | Servo bring-up, stereo calibration, CoppeliaSim/MuJoCo alignment, UAT | `diagnostics/` |
| URDF tooling | MJCF to URDF, URDF verification, calibration utils | `scripts/mjcf_to_urdf.py`, `scripts/verify_urdf.py`, `scripts/calibration_utils.py` |
| LIBERO sim | BDDL tasks, MuJoCo envs, SOARM robot model, lifelong training | `LIBERO/libero/` |
| Sim notebooks | Colab env setup, SOARM integration, OFT/pi0 inference, finetune, eval | `LIBERO/notebooks/01-*.ipynb` .. `06b-eval.ipynb` |
| Explorations | Dataset/scene exploration scripts | `explorations/` |

## Pattern Overview

**Overall:** Script-oriented research monorepo. The active core is a plugin-architecture hardware harness (`control/`) where interchangeable policy backends produce actions behind one `ActionSource` protocol and every action is gated by a single safety validator. Remote GPU models run on Colab in generated, self-contained per-backend notebooks.

**Key Characteristics:**
- Strategy/plugin pattern: backends are sibling modules registered by decorator; adding one never edits the registry.
- Three transport tiers: gRPC to Colab LeRobot PolicyServer (Tier A/B, port 5173 over ngrok TCP), resident HTTP adapter servers on Colab (Tier C, ports 3001-3006 over ngrok HTTP; FastFS depth on 3000), direct cloud provider HTTPS (Gemini, MLLM).
- All actions are joint-space `JOINT_ORDER` targets by the time they reach `safety_validator.validate_action()`.
- Single source of truth for Colab code: `control/vla_bridge/*_server.py` and `colab_server_utils.py` are embedded verbatim as `%%writefile` cells plus a sha256 verify cell; notebooks are never hand-edited.
- Each notebook launches its server as an OS subprocess via `colab_server_utils.start_server()` and only reports success once its own process owns the port (`wait_until_ready()`).
- `LIBERO/` is a vendored benchmark extended with a SOARM robot; simulation and hardware tracks are loosely coupled.

## Layers

**Harness layer:**
- Purpose: Orchestrate one episode on real hardware
- Location: `control/run_vla_episode.py`
- Contains: `ActionSource` Protocol (`:159`), `run_episode()` (`:230`), `main()` CLI (`:436`), camera/device-map helpers, `_interpolate_waypoints()` (`:125`)
- Depends on: `vla_bridge.backends`, `safety_validator`, `io_logger`, `keyboard_joint_control`, `detect_devices`
- Used by: Researcher CLI, `control/tune_execution_hz.py`, `control/test_run_vla_episode.py`

**Backend layer:**
- Purpose: Turn observations + instruction into raw candidate actions
- Location: `control/vla_bridge/backends/`
- Contains: one module per backend plus `base_*.py` tier helpers and conversion modules
- Depends on: `robot_client.py`, `mllm_*.py`, `backend_configs/`, provider SDKs (imported inside `build()`)
- Used by: Harness and `backend_smoke.py` via `BACKEND_REGISTRY[name](args, camera_names, joint_limits_deg) -> BackendHandles`

**Safety/contract layer:**
- Purpose: Treat every action as untrusted; clamp/hold
- Location: `control/vla_bridge/safety_validator.py`, `control/vla_bridge/action_contract.py`
- Used by: Harness, `robot_client.pop_validated_action()` (`control/vla_bridge/robot_client.py:357`)

**Sensor layer:**
- Purpose: Camera capture and stereo depth
- Location: `control/vla_bridge/stereo_camera.py`, `stereo_calibration.py`, `depth_camera.py`, `depth_recorder.py`
- Used by: Harness (`--stereo-camera-index`, `--depth-endpoint`, `--depth-calibration`, `--depth-every-n-steps`), MLLM inputs

**Remote model layer (Colab):**
- Purpose: Host GPU models and depth service
- Location: `control/vla_bridge/*_server.py`, `control/vla_bridge/colab_server_utils.py`, generated `control/vla_bridge/notebooks/*.ipynb` + `README.md`
- Generated by: `scripts/build_colab_notebooks.py` (spec table in `scripts/colab_notebook_specs.py`)

**Simulation layer:**
- Location: `LIBERO/libero/libero/envs/`, `LIBERO/libero/libero/assets/robots/soarm101`, `LIBERO/notebooks/`

## Data Flow

### Primary Request Path (real-hardware episode)

1. CLI parses args, loads joint limits via `action_contract.load_joint_limits_deg()` (`control/run_vla_episode.py:436`)
2. Backend built from `BACKEND_REGISTRY[--backend]`, returns `BackendHandles(action_source, robot, stereo_camera, skip_camera_names, shutdown)` (`control/vla_bridge/backends/__init__.py:57`)
3. Loop: read joint state, `action_source.get_action(joint_state, instruction)` (`control/run_vla_episode.py:159`)
4. `safety_validator.validate_action()` clamps/holds (`control/vla_bridge/safety_validator.py:58`)
5. Waypoint interpolation at `--execution-hz` and `robot.send_action()`; observations/actions logged as JSON Lines by `IOLogger` (`control/vla_bridge/io_logger.py`)
6. Termination (max steps, KeyboardInterrupt e-stop, `episode_complete_reason`) then return-to-start, `_shutdown_backend()` (`:710`), `_write_termination()` (`:224`)

### Colab Server Bring-up Path

1. Edit `control/vla_bridge/<name>_server.py` or `scripts/colab_notebook_specs.py`; run `python scripts/build_colab_notebooks.py` (CI-style check: `--check`, also `control/test_colab_notebooks.py`)
2. Open `control/vla_bridge/notebooks/<lane>_<backend>.ipynb` on a fresh Colab runtime; it installs pinned deps (e.g. `transformers==5.5.4` for lerobot 0.6.1), writes and sha256-verifies files into `/content`
3. Launch cell: `colab_server_utils.stop_servers()` -> `start_server()` -> `wait_until_ready()` (port ownership + optional `Patched ...` banner); diagnostics via `log_report()`, `process_report()`, `gpu_report()`, `health_report()`
4. Tunnel cell prints host:port; locally run `python -m vla_bridge.backend_smoke --backend <name> --server-address ...`, then `run_vla_episode.py`
5. Teardown cell (`stop_server()`) closes server and tunnel

### Depth Path

1. One-time: `stereo_calibration.py` -> `control/stereo_calibration.json`
2. Per N steps: `StereoSplitCamera` frame -> `DepthCameraClient` rectifies, scales intrinsics, POSTs to `fastfs_server.py` `/depth`
3. `DepthRecorder` writes `depth_overhead/*.npy`; MLLM backends consume depth via `mllm_inputs.py`; `render_depth_heatmaps.py` visualizes offline

### MLLM Chunk Path

1. `MllmActionSourceBase` (`control/vla_bridge/backends/base_mllm.py:518`) drains a queue of joint targets
2. On empty queue: `mllm_inputs.py` gathers images and depth, `mllm_prompt.py` builds the prompt, provider adapter calls the API
3. `parse_chunk()` (`base_mllm.py:402`) validates against `mllm_schema.py`; failures produce holds and count toward a failure cap; `mllm_record.py` records each call

### Sim Path

1. Colab notebook sets up LIBERO env (`LIBERO/notebooks/01-colab-env-setup.ipynb`)
2. BDDL task to `OffScreenRenderEnv` with SOARM robot, VLA inference (`03a`, `03b`), finetune/eval (`06a`, `06b`)

**State Management:**
- Per-episode state on disk under `control/outputs/<run>/`; robot calibration read from the LeRobot HF cache; no shared global state beyond `BACKEND_REGISTRY` and `GRPC_BRIDGE_BACKENDS`. Colab tokens read from `MyDrive/SoARM-Research/.hf_token` / `.ngrok_token`.

## Key Abstractions

**ActionSource:**
- Purpose: `get_action(joint_state, instruction) -> (action, model_version)`; optional `episode_complete_reason`, `close()`
- Examples: `control/vla_bridge/backends/scripted.py`, `control/vla_bridge/robot_client.py:467` (`BridgeActionSource`), `base_grpc.py:125` (`ConvertedBridgeActionSource`), `base_http_adapter.py:103` (`HttpAdapterActionSource`)
- Pattern: Structural Protocol (duck typing)

**BackendHandles:**
- Purpose: Everything a backend hands the harness, including ownership of exclusive devices
- Example: `control/vla_bridge/backends/__init__.py:57`

**MLLM backend spec:**
- Purpose: Provider adapter + config name registered via `register_mllm_backend()` (`base_mllm.py:261`)
- Examples: `control/vla_bridge/backends/mllm_claude.py`, `mllm_gpt.py`, `mllm_hf_qwen.py`

**Notebook spec:**
- Purpose: Declarative description of one Colab notebook (backend, lane, port, GPU, gated repos, checkpoint, smoke/episode commands)
- Location: `scripts/colab_notebook_specs.py`; rendered by `scripts/build_colab_notebooks.py`

## Entry Points

**Episode harness:** `control/run_vla_episode.py` — `.venv/bin/python run_vla_episode.py --out outputs/run --backend <name> ...` (run from `control/`)
**Backend smoke:** `python -m vla_bridge.backend_smoke --backend <name> --server-address ...` (from `control/`)
**Recording / teleop / tuning:** `control/record_episode.py`, `control/keyboard_joint_control.py`, `control/joint_jog.py`, `control/tune_execution_hz.py`
**Depth tooling:** `control/vla_bridge/stereo_calibration.py`, `control/vla_bridge/depth_camera.py` (`main()`), `control/render_depth_heatmaps.py`
**Colab servers:** `control/vla_bridge/notebooks/*.ipynb` (index in `control/vla_bridge/notebooks/README.md`)
**Notebook generation:** `python scripts/build_colab_notebooks.py [--check]` (from repo root)
**Diagnostics:** `diagnostics/servo_*.py`, `diagnostics/stereo_calibrate.py`, `diagnostics/UAT/`
**Sim:** `LIBERO/notebooks/*.ipynb`, `LIBERO/libero/lifelong/main.py`, `explorations/`

## Architectural Constraints

- **Threading:** Single-threaded control loop; Colab HTTP servers use non-blocking `inference_lock` (429 on concurrent requests); Colab servers run as subprocesses, not kernel threads.
- **Global state:** `BACKEND_REGISTRY`, `GRPC_BRIDGE_BACKENDS`, `BACKEND_CONFIG_DIR` in `control/vla_bridge/backends/__init__.py` (module-level so tests can patch).
- **Import path:** `control/` scripts import `vla_bridge` as a top-level package; run from `control/` (tests rely on `control/conftest.py`). `scripts/build_colab_notebooks.py` is stdlib-only and run from repo root.
- **Discovery:** Every non-`base_`/`_` module in `backends/` is imported at startup; import failures are fatal by design. Keep heavy SDK imports inside `build()`.
- **Depth:** Only MLLM backends use depth; depth never enters a PolicyServer observation dict.
- **Camera ownership:** A backend that opens a camera must list it in `skip_camera_names`; the AR0144 stereo camera is opened once and shared (`stereo_calibration.py` reuses `StereoSplitCamera`).
- **Colab dependency drift:** gRPC notebooks pin `transformers==5.5.4` after `lerobot==0.6.1` and fail if pip changed `torch`.
- **Circular imports:** None detected.

## Anti-Patterns

### Bypassing the safety validator

**What happens:** Calling `robot.send_action()` directly with backend output.
**Why it's wrong:** Removes joint-limit/NaN/velocity/stale-obs protection on real hardware.
**Do this instead:** Route through `run_episode()` or `robot_client.pop_validated_action()` (`control/vla_bridge/robot_client.py:357`).

### Editing the registry to add a backend

**What happens:** Adding imports or names to `backends/__init__.py`.
**Why it's wrong:** Discovery is automatic; manual edits create duplicate-registration errors.
**Do this instead:** Add `backends/<name>.py` with `@register`/`@register_grpc_backend` and `backend_configs/<name>.json`.

### Hand-editing generated notebooks

**What happens:** Changing cells inside `control/vla_bridge/notebooks/*.ipynb` or `README.md`.
**Why it's wrong:** Embedded sources drift from `*_server.py`; sha256 verify cells and `control/test_colab_notebooks.py` fail.
**Do this instead:** Edit the `.py` source or `scripts/colab_notebook_specs.py` / `scripts/build_colab_notebooks.py`, then run `python scripts/build_colab_notebooks.py`.

### Trusting a port that "looks" up

**What happens:** A stale server on the same port answers health checks.
**Why it's wrong:** A stale stock PolicyServer on 5173 masked a failed patched launch.
**Do this instead:** Use `colab_server_utils.stop_servers()` + `wait_until_ready()` which verify own-process port ownership.

## Error Handling

**Strategy:** Fail safe on hardware — hold position rather than act on bad data.

**Patterns:**
- Invalid/NaN actions become holds in `safety_validator.py`; servers map bad input to 400 and model failures to 500 so clients hold.
- Config loading warns and returns `None` (`load_backend_config()`), falling back to CLI args.
- Every exit path runs return-to-start before disconnect.
- Colab launch failures surface via `log_report()` error-pattern scan rather than silent hangs.

## Cross-Cutting Concerns

**Logging:** JSON Lines via `control/vla_bridge/io_logger.py`; `print()` for CLI progress; `mllm_record.py` for provider calls; Colab server logs tailed by `colab_server_utils.log_report()`.
**Validation:** `safety_validator.py`, `mllm_schema.py`, `_validate_settings()` in `base_mllm.py`, notebook drift check `build_colab_notebooks.py --check`.
**Authentication:** Provider API keys via environment variables (`control/.envrc`); Colab endpoints via ngrok URLs passed on the CLI; HF/ngrok tokens from Drive in notebooks.

---

*Architecture analysis: 2026-10-18*
