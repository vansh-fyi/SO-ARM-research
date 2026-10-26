# SoARM VLA Research

Research code for driving a **SO-ARM101** robot arm with Vision-Language-Action (VLA) models and multimodal LLMs. A researcher types a task prompt ("pick up the pen"), a policy backend turns camera frames + joint state into joint targets, and every action passes a safety validator before it reaches the servos.

The repo has two tracks:

- **Real hardware (`control/`)** — the active core. A plugin harness runs 16+ interchangeable policy backends against the physical arm. GPU models are hosted on Google Colab and reached over ngrok tunnels.
- **Simulation (`LIBERO/`)** — the LIBERO/MuJoCo benchmark extended with a SOARM robot model, used for sim inference, demonstration collection and OpenVLA-OFT fine-tuning (Phases 1–7).

## Architecture

```text
 control/run_vla_episode.py --backend <name>          (local Mac, arm on USB)
        │
        ├─ Backend registry  control/vla_bridge/backends/
        │     ├─ gRPC lane  → Colab LeRobot PolicyServer      (ngrok TCP, port 5173)
        │     ├─ HTTP lane  → Colab resident Flask adapters   (ngrok HTTP, ports 3001–3006)
        │     └─ Cloud lane → Gemini / Claude / GPT / HF APIs (direct HTTPS)
        │
        ├─ safety_validator.py  joint-limit clamp, NaN reject, step/velocity caps, stale-obs hold
        ├─ io_logger.py         per-step JSON Lines (frames, state, raw/validated action, latency)
        └─ stereo + depth       AR0144 stereo camera → FastFS depth service on Colab (port 3000)
        │
        ▼
 SO-ARM101 follower (LeRobot SO101Follower, Feetech STS3215 bus)
```

All backends implement one `ActionSource` protocol (`get_action(joint_state, instruction)`). Backends are discovered automatically: drop a module in `backends/` with `@register(...)` and a matching `backend_configs/<name>.json`.

### Backends

| Lane | Backends | Hosted by |
|------|----------|-----------|
| gRPC (LeRobot 0.6.1 PolicyServer) | `smolvla`, `pi0`, `pi05`, `act`, `diffusion`, `groot-n17` | `control/vla_bridge/notebooks/grpc_*.ipynb` |
| HTTP adapters | `openvla-oft` (3001), `wallx` (3002), `openvla` (3003), `octo` (3004), `rdt1b` (3005), `groot-n15` (3006) | `control/vla_bridge/notebooks/http_*.ipynb` |
| Direct cloud (VLA) | `gemini-robotics`, `gemini-robotics-2` | none |
| Direct cloud (MLLM, raw-JSON joint chunks) | `mllm-claude`, `mllm-gpt`, `mllm-hf-qwen`, `mllm-hf-deepseek`, `mllm-hf-kimi` | none |
| Local | `scripted` | none |

## Repository layout

```text
control/                  Real-hardware harness, teleop, recording, tests
  run_vla_episode.py      Episode entry point (--backend)
  start_pose.py           Home-pose capture CLI; writes start_pose.json (committed)
  vla_bridge/
    backends/             One module per backend + base_*.py lane helpers + action conversions
    backend_configs/      <backend>.json (checkpoint, endpoint, action space, camera map)
    notebooks/            GENERATED per-backend Colab notebooks + README (do not hand-edit)
    *_server.py           Colab model servers and patched gRPC launchers
    colab_server_utils.py Colab process/port/diagnostics helper
    backend_smoke.py      Robot-free live check of a backend
  test_*.py               pytest suite (~1.8k tests, 1790 collected)
scripts/                  Colab notebook generator, URDF conversion/verification
diagnostics/              Servo bring-up, stereo calibration, UAT material
So-101/                   SO-101 URDF + meshes
LIBERO/                   Vendored LIBERO benchmark with the SOARM robot + Colab notebooks
explorations/             Dataset and scene exploration scripts
.planning/                GSD planning state (roadmap, phases, codebase map)
```

## Getting started

### Local harness

```bash
cd control
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q          # full suite, no hardware needed
```

Provider API keys (Gemini, Anthropic, OpenAI, HF) are read from environment variables; see `control/.envrc`. `control/COMMANDS.md` lists the day-to-day commands (calibration, teleop, recording).

### Hosting a model on Colab

1. Open the backend's notebook from `control/vla_bridge/notebooks/` (index and per-model notes in its `README.md`) on a fresh GPU runtime.
2. Run it top to bottom. Each notebook is self-contained: it installs pinned dependencies, writes the server files into `/content` (sha256-verified), launches the server, checks that *its own* process owns the port, and opens an ngrok tunnel. HF and ngrok tokens are read from Google Drive or prompted for.
3. Use the diagnostics cells (log tail, process check, GPU memory, `/health`) if anything looks off, and the teardown cell when done.

Gated checkpoints need an HF token that has accepted the licence (`google/paligemma-3b-pt-224` for pi0/pi05, `nvidia/Cosmos-Reason2-2B` for GR00T N1.7).

### Running a backend

Once, before the first episode: place the arm in the home pose and run `.venv/bin/python start_pose.py` from `control/` (it only reads the arm, never moves it; `--force` overwrites), then commit `control/start_pose.json`. Episodes refuse to run without it (details in `control/COMMANDS.md`).

From `control/`, smoke-test the backend without the robot, then run an episode:

```bash
.venv/bin/python -m vla_bridge.backend_smoke --backend smolvla \
    --server-address <ngrok-host:port> --instruction "pick up the pen"

.venv/bin/python run_vla_episode.py --backend smolvla --out outputs/run \
    --server-address <ngrok-host:port> --instruction "pick up the pen"
```

`Ctrl+C` is the e-stop. Every episode first moves the arm to the home pose (refusing to start if any joint misses it by more than 5 deg/%, recorded in `<out>/start_pose.json`), and every exit path returns it to the home pose before disconnecting.

### Changing a Colab notebook

Notebooks are generated. Edit the server source (`control/vla_bridge/*_server.py`) or the spec (`scripts/colab_notebook_specs.py`), then regenerate from the repo root:

```bash
python scripts/build_colab_notebooks.py          # rewrite notebooks
python scripts/build_colab_notebooks.py --check  # drift check (also run by control/test_colab_notebooks.py)
```

## Status

- Phases 1–7 (simulation, fine-tuning, depth), 10 (digital twin), 11–12 (hardware bridge), 14 (VLA-style backends) and 15 (MLLM backends) are built. Phase 14 backends are live smoke-tested on Colab (10 PASS, 4 deferred).
- Next: the Phase 14 hardware UAT (with the open Phase 13 camera UAT) via `/gsd-verify-work`, then Phase 16 (cross-backend Pen Transfer benchmark), then Phase 17 (re-tightening safety-validator soft caps).
- Sim benchmark phases 8–9 are paused.

Roadmap and history: `.planning/ROADMAP.md`. Codebase reference docs: `.planning/codebase/` (stack, architecture, structure, conventions, testing, integrations, concerns).

## Known caveats

- **Colab dependency drift:** lerobot 0.6.1 pi0/pi05 break on transformers ≥ 5.9; the gRPC notebooks pin `transformers==5.5.4`.
- **Patched launchers** (`diffusion_policy_server.py`, `groot_n17_server.py`) monkey-patch lerobot 0.6.1 internals and refuse any other lerobot version.
- **Checkpoints can vanish from HF** (the original GR00T N1.7 checkpoint did); `backend_configs/*.json` record verified revisions.
- **Safety caps are loosened** for live testing. Hard limits (absolute joint clamping, NaN/inf rejection, e-stop) are always on.
- Colab servers are unauthenticated behind ngrok; tear tunnels down when not in use.

See `.planning/codebase/CONCERNS.md` for the full list.
