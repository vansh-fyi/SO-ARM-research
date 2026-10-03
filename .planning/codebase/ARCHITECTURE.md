<!-- refreshed: 2026-10-03 -->
# Architecture

**Analysis Date:** 2026-10-03

## System Overview

```text
┌──────────────────────────────────────────────────────────────────────────┐
│               Real-Hardware VLA/MLLM Harness (Mac, local)                 │
│   `control/run_vla_episode.py`  (CLI: --backend <name>)                   │
├───────────────┬───────────────────┬───────────────────┬──────────────────┤
│ Backend        │ ActionSource       │ Safety gate        │ I/O + sensors    │
│ registry       │ (per backend)      │ `vla_bridge/       │ `io_logger.py`,  │
│ `vla_bridge/   │ `vla_bridge/       │  safety_validator  │ `stereo_camera`, │
│  backends/     │  backends/*.py`    │  .py` +            │ `depth_camera`,  │
│  __init__.py`  │                    │ `action_contract`  │ `depth_recorder` │
└──────┬────────┴─────────┬─────────┴─────────┬─────────┴──────────────────┘
       │ Tier A/B gRPC     │ Tier C HTTP        │ Direct cloud HTTPS
       ▼                   ▼                    ▼
┌──────────────────┐ ┌────────────────────┐ ┌──────────────────────────────┐
│ Colab PolicyServer│ │ Colab resident HTTP │ │ Gemini / Claude / GPT / HF   │
│ (LeRobot async)   │ │ servers             │ │ MLLM providers               │
│ `vla_bridge/      │ │ `vla_bridge/*_      │ │ `backends/base_mllm*.py`,    │
│ policy_server     │ │  server.py`         │ │ `backends/base_gemini.py`    │
│ .ipynb`           │ │ (+ `fastfs_server`) │ │                              │
└──────────────────┘ └────────────────────┘ └──────────────────────────────┘
       │
       ▼
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
| Episode harness | Control loop, camera open, e-stop, return-to-start, termination record | `control/run_vla_episode.py` |
| Backend registry | Auto-discover backend modules, `@register`/`@register_grpc_backend`, load per-backend JSON config | `control/vla_bridge/backends/__init__.py` |
| Backend configs | Per-backend checkpoint/endpoint/action-space/camera_map | `control/vla_bridge/backend_configs/*.json` |
| gRPC lane base | LeRobot PolicyServer client wrapper + action conversion | `control/vla_bridge/backends/base_grpc.py`, `control/vla_bridge/robot_client.py` |
| HTTP adapter base | Tier C resident-server client (PNG obs, joint action) | `control/vla_bridge/backends/base_http_adapter.py` |
| MLLM base | Chunked joint-target prompting, JSON schema parsing, holds, failure cap | `control/vla_bridge/backends/base_mllm.py`, `base_mllm_build.py`, `base_mllm_chat.py` |
| Gemini base | Gemini Robotics direct-cloud lane | `control/vla_bridge/backends/base_gemini.py` |
| Action conversions | EEF/relative-EE to joint targets | `control/vla_bridge/backends/eef_to_joint.py`, `relative_ee_conversion.py`, `er_conversion.py` |
| MLLM support | Inputs (images/depth), prompt, schema, URDF context, recording | `control/vla_bridge/mllm_inputs.py`, `mllm_prompt.py`, `mllm_schema.py`, `mllm_urdf.py`, `mllm_record.py` |
| Safety validator | Joint-limit clamp, NaN reject, per-step + velocity caps, stale-obs hold | `control/vla_bridge/safety_validator.py` |
| Action contract | `JOINT_ORDER`, calibration ticks to degrees, joint limits | `control/vla_bridge/action_contract.py` |
| Colab model servers | Resident model HTTP servers (OpenVLA, OFT, Octo, GR00T, RDT, WallX, DP, FastFS depth) | `control/vla_bridge/*_server.py`, `control/vla_bridge/policy_server.ipynb` |
| Hardware teleop/record | Jog, keyboard control, calibration, episode recording | `control/joint_jog.py`, `control/keyboard_joint_control.py`, `control/record_episode.py` |
| Diagnostics | Servo bring-up, stereo calibration, CoppeliaSim/MuJoCo alignment, UAT | `diagnostics/` |
| URDF tooling | MJCF to URDF, URDF verification, notebook sync | `scripts/` |
| LIBERO sim | BDDL tasks, MuJoCo envs, SOARM robot model, lifelong training | `LIBERO/libero/` |
| Sim notebooks | Colab env setup, SOARM integration, OFT/pi0 inference, finetune, eval | `LIBERO/notebooks/01-*.ipynb` .. `06b-eval.ipynb` |
| Explorations | Dataset/scene exploration scripts | `explorations/` |

## Pattern Overview

**Overall:** Script-oriented research monorepo. The active core is a plugin-architecture hardware harness (`control/`) where interchangeable policy backends produce actions behind one `ActionSource` protocol and every action is gated by a single safety validator.

**Key Characteristics:**
- Strategy/plugin pattern: backends are sibling modules registered by decorator; adding one never edits the registry.
- Three transport tiers: gRPC to Colab LeRobot PolicyServer (Tier A/B), resident HTTP adapter servers on Colab (Tier C), direct cloud provider HTTPS (Gemini, MLLM).
- All actions are joint-space `JOINT_ORDER` targets by the time they reach `safety_validator.validate_action()`.
- Colab-side server code lives in-repo (`control/vla_bridge/*_server.py`) and is synced into `policy_server.ipynb` by `scripts/sync_policy_notebook.py`.
- `LIBERO/` is a vendored benchmark (own history) extended with a SOARM robot; simulation and hardware tracks are loosely coupled.

## Layers

**Harness layer:**
- Purpose: Orchestrate one episode on real hardware
- Location: `control/run_vla_episode.py`
- Contains: `ActionSource` Protocol, `run_episode()`, `main()` CLI, camera/device-map helpers
- Depends on: `vla_bridge.backends`, `safety_validator`, `io_logger`, `keyboard_joint_control`, `detect_devices`
- Used by: Researcher CLI, `control/test_run_vla_episode.py`

**Backend layer:**
- Purpose: Turn observations + instruction into raw candidate actions
- Location: `control/vla_bridge/backends/`
- Contains: one module per backend (`pi0.py`, `smolvla.py`, `openvla.py`, `mllm_claude.py`, ...) plus `base_*.py` tier helpers
- Depends on: `robot_client.py`, `mllm_*.py`, `backend_configs/`, provider SDKs (imported inside `build()`)
- Used by: Harness via `BACKEND_REGISTRY[name](args, camera_names, joint_limits_deg) -> BackendHandles`

**Safety/contract layer:**
- Purpose: Treat every action as untrusted; clamp/hold
- Location: `control/vla_bridge/safety_validator.py`, `control/vla_bridge/action_contract.py`
- Used by: Harness, `robot_client.pop_validated_action()`

**Remote model layer (Colab):**
- Purpose: Host GPU models
- Location: `control/vla_bridge/*_server.py`, `control/vla_bridge/policy_server.ipynb`, launch guide `control/vla_bridge/policy_server_launch.md`

**Simulation layer:**
- Location: `LIBERO/libero/libero/envs/`, `LIBERO/libero/libero/assets/robots/soarm101`, `LIBERO/notebooks/`

## Data Flow

### Primary Request Path (real-hardware episode)

1. CLI parses args, loads joint limits via `action_contract.load_joint_limits_deg()` (`control/run_vla_episode.py:436`)
2. Backend built from `BACKEND_REGISTRY[--backend]`, returns `BackendHandles(action_source, robot, stereo_camera, skip_camera_names, shutdown)` (`control/vla_bridge/backends/__init__.py:57`)
3. Loop: read joint state, `action_source.get_action(joint_state, instruction)` (`control/run_vla_episode.py:159`)
4. `safety_validator.validate_action()` clamps/holds (`control/vla_bridge/safety_validator.py:58`)
5. Waypoint interpolation and `robot.send_action()`; observations/actions logged as JSON Lines by `IOLogger` (`control/vla_bridge/io_logger.py`)
6. Termination (max steps, KeyboardInterrupt e-stop, `episode_complete_reason`) then return-to-start, `_shutdown_backend()`, `_write_termination()` (`control/run_vla_episode.py:224`, `:709`)

### MLLM Chunk Path

1. `MllmActionSourceBase` (`control/vla_bridge/backends/base_mllm.py:518`) drains a queue of joint targets
2. On empty queue: `mllm_inputs.py` gathers images and depth (FastFS via `--depth-endpoint`), `mllm_prompt.py` builds the prompt, provider adapter calls the API
3. `parse_chunk()` (`base_mllm.py:402`) validates against `mllm_schema.py`; failures produce holds and count toward a failure cap; `mllm_record.py` records each call

### Sim Path

1. Colab notebook sets up LIBERO env (`LIBERO/notebooks/01-colab-env-setup.ipynb`)
2. BDDL task to `OffScreenRenderEnv` with SOARM robot, VLA inference (`03a`, `03b`), finetune/eval (`06a`, `06b`)

**State Management:**
- Per-episode state on disk under `control/outputs/<run>/`; robot calibration read from the LeRobot HF cache; no shared global state beyond `BACKEND_REGISTRY` and `GRPC_BRIDGE_BACKENDS`.

## Key Abstractions

**ActionSource:**
- Purpose: `get_action(joint_state, instruction) -> (action, model_version)`; optional `episode_complete_reason`, `close()`
- Examples: `control/vla_bridge/backends/scripted.py`, `control/vla_bridge/robot_client.py:464` (`BridgeActionSource`), `base_grpc.py:125`, `base_http_adapter.py:103`
- Pattern: Structural Protocol (duck typing)

**BackendHandles:**
- Purpose: Everything a backend hands the harness, including ownership of exclusive devices
- Example: `control/vla_bridge/backends/__init__.py:57`

**MLLM backend spec:**
- Purpose: Provider adapter + config name registered via `register_mllm_backend()`
- Examples: `control/vla_bridge/backends/mllm_claude.py`, `mllm_gpt.py`, `mllm_hf_qwen.py`

## Entry Points

**Episode harness:** `control/run_vla_episode.py` — `python run_vla_episode.py --out outputs/run --backend <name> ...` (run from `control/`)
**Episode recording / teleop:** `control/record_episode.py`, `control/keyboard_joint_control.py`, `control/joint_jog.py`
**Colab servers:** `control/vla_bridge/*_server.py`, launched per `control/vla_bridge/policy_server_launch.md`
**Diagnostics:** `diagnostics/servo_*.py`, `diagnostics/stereo_calibrate.py`, `diagnostics/UAT/`
**Sim:** `LIBERO/notebooks/*.ipynb`, `LIBERO/libero/lifelong/main.py`, `explorations/create_scene.py`, `explorations/soarm_sanity.py`

## Architectural Constraints

- **Threading:** Single-threaded control loop; Colab servers use non-blocking `inference_lock` (429 on concurrent requests).
- **Global state:** `BACKEND_REGISTRY`, `GRPC_BRIDGE_BACKENDS`, `BACKEND_CONFIG_DIR` in `control/vla_bridge/backends/__init__.py` (module-level so tests can patch).
- **Import path:** `control/` scripts import `vla_bridge` as a top-level package; run from `control/` (tests rely on `control/conftest.py`).
- **Discovery:** Every non-`base_`/`_` module in `backends/` is imported at startup; import failures are fatal by design. Keep heavy SDK imports inside `build()`.
- **Depth:** Only MLLM backends use depth; depth never enters a PolicyServer observation dict.
- **Camera ownership:** A backend that opens a camera must list it in `skip_camera_names`; the AR0144 stereo camera is opened once and shared.
- **Circular imports:** None detected.

## Anti-Patterns

### Bypassing the safety validator

**What happens:** Calling `robot.send_action()` directly with backend output.
**Why it's wrong:** Removes joint-limit/NaN/velocity/stale-obs protection on real hardware.
**Do this instead:** Route through `run_episode()` or `robot_client.pop_validated_action()` (`control/vla_bridge/robot_client.py:354`).

### Editing the registry to add a backend

**What happens:** Adding imports or names to `backends/__init__.py`.
**Why it's wrong:** Discovery is automatic; manual edits create duplicate-registration errors.
**Do this instead:** Add `backends/<name>.py` with `@register`/`@register_grpc_backend` and `backend_configs/<name>.json`.

### Editing notebook server cells by hand

**What happens:** Changing server source inside `policy_server.ipynb`.
**Why it's wrong:** Drifts from `fastfs_server.py` etc.
**Do this instead:** Edit the `.py` and run `scripts/sync_policy_notebook.py`.

## Error Handling

**Strategy:** Fail safe on hardware — hold position rather than act on bad data.

**Patterns:**
- Invalid/NaN actions become holds in `safety_validator.py`; servers map bad input to 400 and model failures to 500 so clients hold.
- Config loading warns and returns `None` (`load_backend_config()`), falling back to CLI args.
- Every exit path runs return-to-start before disconnect.

## Cross-Cutting Concerns

**Logging:** JSON Lines via `control/vla_bridge/io_logger.py`; `print()` for CLI progress; `mllm_record.py` for provider calls.
**Validation:** `safety_validator.py`, `mllm_schema.py`, `_validate_settings()` in `base_mllm.py`.
**Authentication:** Provider API keys via environment variables; Colab endpoints via ngrok URLs passed on the CLI.

---

*Architecture analysis: 2026-10-03*
