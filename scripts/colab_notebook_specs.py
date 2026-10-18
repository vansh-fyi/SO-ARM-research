"""Per-notebook specs for scripts/build_colab_notebooks.py (13 Colab notebooks).

Each dict describes one notebook in control/vla_bridge/notebooks/: its lane, port, GPU,
gated licences, default checkpoint, install cells, launch command, prose and local commands.
Prose and install code are lifted from the former combined policy-server notebook. Shared
cells (OpenMP guards, pyngrok, tokens, the helper embed, versions, diagnostics, teardown)
live in the generator. Edit here, then run python scripts/build_colab_notebooks.py.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "control" / "vla_bridge" / "backend_configs"
PI0_REVISION = json.loads((CONFIGS / "pi0.json").read_text())["verified_revision"]
GROOT_N17_CHECKPOINT = json.loads((CONFIGS / "groot_n17.json").read_text())["checkpoint"]

GRPC_MARKERS = (
    "lerobot.async_inference.policy_server",
    "diffusion_policy_server.py",
    "groot_n17_server.py",
)
GRPC_PLACEHOLDERS = ("<ngrok_tcp_host:port>", "<ngrok_tcp_host>:<ngrok_tcp_port>")
SMOKE = ".venv/bin/python -m vla_bridge.backend_smoke"
PALIGEMMA = "google/paligemma-3b-pt-224"

GRPC_VERSION_CHECK = r'''!pip show torch transformers lerobot diffusers | grep -E "^(Name|Version)"
!python -c "import torch; print('torch', torch.__version__); print('cudnn', torch.backends.cudnn.version())"'''

STOCK_LAUNCH_COMMENT = r'''# The PolicyServer runs as a separate OS subprocess, not a thread in this kernel (Test 7 UAT,
# Phase 12, 2026-09-27):
#   - a crash in it cannot take this kernel (or the tunnel) down
#   - its stdout/stderr go straight to a file via the OS, independent of
#     Jupyter's per-cell output capture, so the log survives even a hard crash
#   - it re-imports torch fresh from disk at subprocess start
#   - CUDA_LAUNCH_BLOCKING=1 forces synchronous CUDA execution, so an async
#     driver-level crash surfaces as a normal Python traceback in the log
#     instead of an opaque process death
#   - KMP_DUPLICATE_LIB_OK / OMP_NUM_THREADS (also set globally above) guard
#     this subprocess too against the "OMP: Error #15" duplicate-runtime crash
# A hung server from an earlier attempt keeps holding GPU memory, which can itself cause
# the next model load to OOM, so stop_servers() clears every known server first.'''

GRPC_POST = r'''print(f'Tail it live from another cell with: !tail -f {POLICY_SERVER_LOG}')'''


def venv_version_check(python_var, python_path, packages):
    return f'''import subprocess

{python_var} = {python_path!r}
probe = (
    "import importlib.metadata as m\\n"
    "for p in {packages!r}:\\n"
    "    try:\\n"
    "        print(p, m.version(p))\\n"
    "    except m.PackageNotFoundError:\\n"
    "        print(p, 'not installed')\\n"
)
result = subprocess.run([{python_var}, '-c', probe], capture_output=True, text=True)
print(result.stdout)
print(result.stderr[-2000:])'''


def grpc_local_md(spec_name, policy_type, smoke, episode, signatures, extra=""):
    return f"""## Run locally

After the tunnel cell, run these in `control/`. The tunnel cell prints both commands with the
address filled in. The smoke run uses synthetic hardware (the arm cannot move):

```bash
{smoke}
```

Then the episode:

```bash
{episode}
```

A healthy server log shows `Policy type: {policy_type}` and then
`Preprocessing and inference took ...s, action shape: ...`-style lines. The first chunk can
take much longer than later ones (model download and load happen at the first handshake).
{signatures}
{extra}""".rstrip() + "\n"


def stock_grpc(name, backend, policy_type, title, serves, checkpoint, licences, intro_md,
               smoke, episode, signatures, extra_cells=(), local_extra=""):
    return {
        "name": name,
        "backend": backend,
        "lane": "grpc",
        "lane_label": "gRPC, stock lerobot 0.6.1 PolicyServer",
        "title": title,
        "serves": serves,
        "port": 5173,
        "tunnel": "tcp",
        "tunnel_var": "tunnel",
        "gpu": "L4",
        "gpu_type": "L4",
        "high_ram": False,
        "licences": list(licences),
        "prerequisites": "",
        "checkpoint": checkpoint,
        "log": f"/content/policy_server_{backend}.log",
        "log_var": "POLICY_SERVER_LOG",
        "process_var": "policy_server_process",
        "marker": "lerobot.async_inference.policy_server",
        "stop_patterns": list(GRPC_MARKERS),
        "embed": [],
        "intro_md": intro_md,
        "install": {"extras": "async"},
        "extra_cells": list(extra_cells),
        "launch": {
            "argv": "[sys.executable, '-u', '-m', 'lerobot.async_inference.policy_server', "
                    "'--host', '0.0.0.0', '--port', '5173']",
            "env": "{'KMP_DUPLICATE_LIB_OK': 'TRUE', 'OMP_NUM_THREADS': '1', "
                   "'CUDA_LAUNCH_BLOCKING': '1'}",
            "comment": STOCK_LAUNCH_COMMENT,
            "timeout_s": 90,
            "post": GRPC_POST,
        },
        "local_md": grpc_local_md(name, policy_type, smoke, episode, signatures, local_extra),
        "smoke_command": smoke,
        "episode_command": episode,
        "placeholders": list(GRPC_PLACEHOLDERS),
        "version_check": GRPC_VERSION_CHECK,
        "teardown_comment": "Teardown: every session, after the local run has finished.",
    }


# --- gRPC: stock PolicyServer ---------------------------------------------------------------

SMOLVLA = stock_grpc(
    "grpc_smolvla", "smolvla", "smolvla",
    "SmolVLA policy server (`--backend smolvla`)",
    "Serves the SmolVLA checkpoint from the Colab-hosted `lerobot.async_inference` "
    "`PolicyServer` (lerobot 0.6.1) over gRPC. The local `RobotClient` connects through an ngrok "
    "TCP tunnel.",
    "victorvanhalst/smolvla_so101_cube", (),
    r"""## SmolVLA (`--backend smolvla`)

**Selected checkpoint (Task 2's `checkpoint:decision`):** `victorvanhalst/smolvla_so101_cube`
— chosen for its exact task-instruction match to D-02's red-cube task. See
`checkpoint_candidates.md` for the fetched `config.json` data and
`11-03-SUMMARY.md` for the full decision rationale.

| Item             | Value |
| ---------------- | ----- |
| Task instruction | "Pick the red cube and place it in the bowl" |
| Tunnel type      | TCP (`ngrok.connect(5173, "tcp")`) — not HTTP |
| Camera mapping   | `camera1`=wrist, `camera2`=AR0144 stereo-left, `camera3`=AR0144 stereo-right |
| Fallback tunnel  | `cloudflared` + configured Zero Trust tunnel (heavier setup) |
| Fallback policy  | ACT or pi0/pi05 (`lerobot.policies`, already installed) if this checkpoint underperforms |

The camera mapping and its implementation are described in the README's "Camera mapping"
section. For depth-enabled episodes, also run `fastfs_depth.ipynb` (the Fast-FoundationStereo
depth service).""",
    f"{SMOKE} --backend smolvla --server-address <ngrok_tcp_host:port> "
    "--checkpoint victorvanhalst/smolvla_so101_cube",
    r""".venv/bin/python run_vla_episode.py --backend smolvla \
  --server-address <ngrok_tcp_host:port> --checkpoint victorvanhalst/smolvla_so101_cube \
  --instruction "Pick the red cube and place it in the bowl" \
  --out outputs/smolvla_UNIQUE_NAME --max-steps 20""",
    "`Error in StreamActions` in the log means the server failed an inference; the robot only "
    "holds position.",
    local_extra="For an episode with depth recording, start `fastfs_depth.ipynb` too and use "
                "the command in its \"Run the local episode with depth enabled\" section.",
)

PI0_NOCOMPILE_MD = r"""## pi0: serve a local copy with torch.compile off

The pi0 checkpoint ships `compile_model: true` with `compile_mode: max-autotune`. In the
2026-10-18 live session that meant a 465 s first inference, then
`beginAllocateToPool: already recording to mempool_id` after every client reconnect. The next
cell downloads the checkpoint at `backend_configs/pi0.json`'s verified revision into
`/content/ckpt/pi0_so101_nocompile`, sets `compile_model` to false in its `config.json`, and
prints the local smoke command, which passes that local path as `--checkpoint` (the server
loads it from this runtime's disk). pi05's checkpoint already has `compile_model: false`."""

PI0_NOCOMPILE = r'''import json
from pathlib import Path

from huggingface_hub import snapshot_download

PI0_REPO = 'amindt/pi0_so101_pickandplace_redblue'
PI0_REVISION = '%s'  # backend_configs/pi0.json verified_revision
PI0_LOCAL = Path('/content/ckpt/pi0_so101_nocompile')
snapshot_download(PI0_REPO, revision=PI0_REVISION, local_dir=str(PI0_LOCAL))
config_path = PI0_LOCAL / 'config.json'
config = json.loads(config_path.read_text())
before = (config.get('compile_model'), config.get('compile_mode'))
config['compile_model'] = False
config_path.write_text(json.dumps(config, indent=4) + '\n')
after = (config.get('compile_model'), config.get('compile_mode'))
print(f'compile_model/compile_mode: {before} → {after} in {config_path}')
print('Local smoke command (control/; the tunnel cell fills in the address):')
print('%s')''' % (PI0_REVISION, f"{SMOKE} --backend pi0 --server-address <ngrok_tcp_host:port> "
                                "--checkpoint /content/ckpt/pi0_so101_nocompile --wait-s 300")

PI0 = stock_grpc(
    "grpc_pi0", "pi0", "pi0",
    "pi0 policy server (`--backend pi0`)",
    "Serves a local, torch.compile-free copy of the pi0 SO-101 fine-tune from the Colab-hosted "
    "`lerobot.async_inference` `PolicyServer` (lerobot 0.6.1) over gRPC, through an ngrok TCP "
    "tunnel.",
    "amindt/pi0_so101_pickandplace_redblue", (PALIGEMMA,),
    r"""## pi0 (`--backend pi0`)

**Checkpoint:** `amindt/pi0_so101_pickandplace_redblue` (`backend_configs/pi0.json`), an SO-101
joint-space pi0 fine-tune of `lerobot/pi0_base` (dataset `amindt/so101_pickandplace_redblue`).
Inputs: `observation.state` [6], `observation.images.camera1` + `camera2` (480x640); output
action [6]. The `camera_map` sends the wrist frame as `camera1` and the left AR0144 overhead
half as `camera2`; the training dataset's `camera2` viewpoint is not documented, so that is an
assumption to check on hardware.

pi0 builds on the gated `google/paligemma-3b-pt-224`: accept its licence with the account
whose token the token cell logs in with. The gated licence check above fails early otherwise.
This notebook serves the checkpoint from `/content/ckpt/pi0_so101_nocompile` (see the pi0
section after the installs).""",
    f"{SMOKE} --backend pi0 --server-address <ngrok_tcp_host:port> "
    "--checkpoint /content/ckpt/pi0_so101_nocompile --wait-s 300",
    r""".venv/bin/python run_vla_episode.py --backend pi0 \
  --server-address <ngrok_tcp_host:port> --checkpoint /content/ckpt/pi0_so101_nocompile \
  --out outputs/pi0_UNIQUE_NAME --max-steps 20""",
    "Failures look different:\n"
    "- An error about `create_causal_mask` and `cache_position`: `transformers` is not 5.5.4; "
    "re-run the pin and versions cells on a fresh runtime.\n"
    "- `beginAllocateToPool: already recording to mempool_id`: the server loaded a checkpoint "
    "with `compile_model: true`; pass `--checkpoint /content/ckpt/pi0_so101_nocompile`.\n"
    "- A 401/403 or \"gated repo\" error for `google/paligemma-3b-pt-224`: accept its licence "
    "first.",
    extra_cells=[("pi0-nocompile-notes", "markdown", PI0_NOCOMPILE_MD),
                 ("pi0-nocompile-checkpoint", "code", PI0_NOCOMPILE)],
)

PI05 = stock_grpc(
    "grpc_pi05", "pi05", "pi05",
    "pi0.5 policy server (`--backend pi05`)",
    "Serves the pi0.5 SO-101 fine-tune from the Colab-hosted `lerobot.async_inference` "
    "`PolicyServer` (lerobot 0.6.1) over gRPC, through an ngrok TCP tunnel.",
    "yathAg/pi05_so101_pick_place", (PALIGEMMA,),
    r"""## pi0.5 (`--backend pi05`)

**Checkpoint:** `yathAg/pi05_so101_pick_place` (`backend_configs/pi05.json`), an SO-101
joint-space pi0.5 fine-tune of `lerobot/pi05_base` (dataset `yathAg/so101_pick_place`). Inputs:
`observation.state` [6] (QUANTILES-normalized, tokenized into the prompt by pi05),
`observation.images.camera1` + `camera2` (480x640); output action [6]. The `camera_map` sends
the wrist frame as `camera1` and the left AR0144 overhead half as `camera2`; the training
dataset's `camera2` viewpoint is not documented, so that is an assumption to check on hardware.
The checkpoint already has `compile_model: false`.

pi0.5 builds on the gated `google/paligemma-3b-pt-224`: accept its licence with the account
whose token the token cell logs in with. The gated licence check above fails early otherwise.
Cold first chunks can take more than 30 s, so the smoke command waits up to 300 s.""",
    f"{SMOKE} --backend pi05 --server-address <ngrok_tcp_host:port> "
    "--checkpoint yathAg/pi05_so101_pick_place --wait-s 300",
    r""".venv/bin/python run_vla_episode.py --backend pi05 \
  --server-address <ngrok_tcp_host:port> --checkpoint yathAg/pi05_so101_pick_place \
  --out outputs/pi05_UNIQUE_NAME --max-steps 20""",
    "Failures look different:\n"
    "- An error about `create_causal_mask` and `cache_position`: `transformers` is not 5.5.4; "
    "re-run the pin and versions cells on a fresh runtime.\n"
    "- A 401/403 or \"gated repo\" error for `google/paligemma-3b-pt-224`: accept its licence "
    "first.",
)

ACT = stock_grpc(
    "grpc_act", "act", "act",
    "ACT policy server (`--backend act`)",
    "Serves the ACT SO-101 fine-tune from the Colab-hosted `lerobot.async_inference` "
    "`PolicyServer` (lerobot 0.6.1) over gRPC, through an ngrok TCP tunnel.",
    "hrl7/so101-pregrasp-act-l4", (),
    r"""## ACT (`--backend act`)

**Checkpoint:** `hrl7/so101-pregrasp-act-l4` (`backend_configs/act.json`), an SO-101 ACT
fine-tune (dataset `hrl7/so101-wrist-d1-pregrasp-v1`, 58 episodes at 30 fps, 5000 training
steps, chunk_size 100). Inputs: `observation.state` [6] and `observation.images.camera1`
(480x640) as its ONLY camera; output action [6]. ACT does not mask missing cameras, so the
`camera_map` sends exactly `camera1` (wrist). Two-camera SO-101 ACT checkpoints load via
`--camera-map`.""",
    f"{SMOKE} --backend act --server-address <ngrok_tcp_host:port> "
    "--checkpoint hrl7/so101-pregrasp-act-l4",
    r""".venv/bin/python run_vla_episode.py --backend act \
  --server-address <ngrok_tcp_host:port> --checkpoint hrl7/so101-pregrasp-act-l4 \
  --out outputs/act_UNIQUE_NAME --max-steps 20""",
    "A `KeyError: 'observation.images.<name>'` means the `camera_map` doesn't match the "
    "checkpoint's camera names; pass `--camera-map`.",
)

# --- gRPC: patched launchers ----------------------------------------------------------------

DIFFUSION = {
    "name": "grpc_diffusion",
    "backend": "diffusion",
    "lane": "grpc",
    "lane_label": "gRPC, patched launcher (`diffusion_policy_server.py`)",
    "title": "Diffusion Policy server (`--backend diffusion`)",
    "serves": "Serves the Diffusion Policy SO-101 fine-tune over gRPC from "
              "`diffusion_policy_server.py`, which patches lerobot 0.6.1's `DiffusionPolicy` and "
              "then runs the same `PolicyServer`, through an ngrok TCP tunnel.",
    "port": 5173,
    "tunnel": "tcp",
    "tunnel_var": "tunnel",
    "gpu": "L4",
    "gpu_type": "L4",
    "high_ram": False,
    "licences": [],
    "prerequisites": "",
    "checkpoint": "hrl7/so101-touch-diffusion-l4",
    "log": "/content/policy_server_diffusion.log",
    "log_var": "POLICY_SERVER_LOG",
    "process_var": "policy_server_process",
    "marker": "diffusion_policy_server.py",
    "stop_patterns": list(GRPC_MARKERS),
    "embed": ["diffusion_policy_server.py"],
    "intro_md": r"""## Tier A: Diffusion Policy (Plan 14-06, `--backend diffusion`)

Diffusion Policy uses the gRPC lane (`policy_type="diffusion"`), with two
differences from the stock server. Both were verified against the installed `lerobot==0.6.1` source.

1. **It needs `diffusers`.** `DiffusionPolicy` imports its noise scheduler from it (lerobot's
   `diffusion` extra pins `diffusers>=0.38.0,<0.40.0`). The pin cell below pins `0.39.0`; the
   PyPI metadata for that release has no `torch` requirement.
2. **The stock server can't run it.** `PolicyServer` calls `predict_action_chunk()` with one
   observation, and 0.6.1's Diffusion fails its `n_obs_steps` assertion on every inference
   (huggingface/lerobot#3861; fix PR #3886 is unmerged). The server logs
   `Error in StreamActions` and the robot only holds position. So start the server with
   `control/vla_bridge/diffusion_policy_server.py`. It applies PR #3886's fix to
   `DiffusionPolicy` only, then runs the same `serve()`.

**Status:** not yet run on Colab.

**Checkpoint:** `hrl7/so101-touch-diffusion-l4` (camera1 + camera2), from
`backend_configs/diffusion_policy.json`. Diffusion Policy doesn't mask missing cameras, so the
bridge must send exactly the cameras the checkpoint declares: the config's `camera_map` sends the
wrist frame as `camera1` and the left AR0144 overhead half as `camera2`. For another checkpoint,
pass `--camera-map name=source,...` with its own camera names (sources: `wrist`,
`overhead_left`, `overhead_right`); `check_checkpoint.py --backend diffusion` suggests one.

This notebook launches the patched launcher instead of the stock server. A T4 is enough: the
models are a ResNet-18 plus a 1-D U-Net.
Each chunk runs the full DDPM schedule (100 denoising steps by default) and returns only
`n_action_steps` actions (8 for most SO-101 checkpoints). Expect noticeably more hold-position
steps between chunks than with ACT.""",
    "install": {"extras": "async", "diffusers": True},
    "extra_cells": [],
    "launch_notes": r"""## Patched launcher

The next cell writes `control/vla_bridge/diffusion_policy_server.py` into
`/content/diffusion_policy_server.py`, and the cell after it checks its sha256. The launch cell
starts it instead of the stock server, on the same port (5173).""",
    "launch": {
        "argv": "[sys.executable, '-u', '/content/diffusion_policy_server.py', "
                "'--host', '0.0.0.0', '--port', '5173']",
        "env": "{'KMP_DUPLICATE_LIB_OK': 'TRUE', 'OMP_NUM_THREADS': '1', "
               "'CUDA_LAUNCH_BLOCKING': '1'}",
        "comment": STOCK_LAUNCH_COMMENT,
        "timeout_s": 90,
        "banner": "Patched DiffusionPolicy.predict_action_chunk",
        "post": GRPC_POST,
    },
    "local_md": r"""## Run locally

After the tunnel cell, run these in `control/`. The tunnel cell prints both commands with the
address filled in. The smoke run uses synthetic hardware (the arm cannot move):

```bash
.venv/bin/python -m vla_bridge.backend_smoke --backend diffusion --server-address <ngrok_tcp_host:port> --checkpoint hrl7/so101-touch-diffusion-l4
```

Then the episode:

```bash
.venv/bin/python run_vla_episode.py <port> <robot_id> --backend diffusion \
  --server-address <ngrok_tcp_host>:<ngrok_tcp_port> --checkpoint hrl7/so101-touch-diffusion-l4 \
  --out outputs/diffusion_UNIQUE_NAME --max-steps 20
```

A healthy server log shows `Policy type: diffusion` and then
`Preprocessing and inference took ...s, action shape: torch.Size([1, 8, 6])`-style lines. Two
failures look different:
- `Error in StreamActions: ` followed by an empty `AssertionError`: the stock server is running
  instead of the patched launcher.
- `KeyError: 'observation.images.camera2'` (or similar): the checkpoint declares a camera the
  bridge doesn't deliver.

Tear down with the teardown cell at the end of this notebook.""",
    "smoke_command": f"{SMOKE} --backend diffusion --server-address <ngrok_tcp_host:port> "
                     "--checkpoint hrl7/so101-touch-diffusion-l4",
    "episode_command": r""".venv/bin/python run_vla_episode.py <port> <robot_id> --backend diffusion \
  --server-address <ngrok_tcp_host>:<ngrok_tcp_port> --checkpoint hrl7/so101-touch-diffusion-l4 \
  --out outputs/diffusion_UNIQUE_NAME --max-steps 20""",
    "placeholders": list(GRPC_PLACEHOLDERS),
    "version_check": GRPC_VERSION_CHECK,
    "teardown_comment": "Teardown: every session, after the local run has finished.",
}

GROOT_N17 = {
    "name": "grpc_groot_n17",
    "backend": "groot-n17",
    "lane": "grpc",
    "lane_label": "gRPC, patched launcher (`groot_n17_server.py`)",
    "title": "GR00T N1.7 policy server (`--backend groot-n17`)",
    "serves": "Serves the GR00T N1.7 SO-101 fine-tune over gRPC from `groot_n17_server.py`, "
              "which decodes N1.7 action chunks whole and then runs lerobot 0.6.1's "
              "`PolicyServer`, through an ngrok TCP tunnel.",
    "port": 5173,
    "tunnel": "tcp",
    "tunnel_var": "tunnel",
    "gpu": "L4",
    "gpu_type": "L4",
    "high_ram": False,
    "licences": ["nvidia/Cosmos-Reason2-2B"],
    "prerequisites": "",
    "checkpoint": GROOT_N17_CHECKPOINT,
    "log": "/content/policy_server_groot_n17.log",
    "log_var": "POLICY_SERVER_LOG",
    "process_var": "policy_server_process",
    "marker": "groot_n17_server.py",
    "stop_patterns": list(GRPC_MARKERS),
    "embed": ["groot_n17_server.py"],
    "intro_md": f"""## Tier B -> gRPC: GR00T N1.7 (Plan 14-07, `--backend groot-n17`)

GR00T N1.7 uses the gRPC lane (`policy_type="groot"`). The installed
`lerobot==0.6.1` lists `"groot"` in `async_inference/constants.py::SUPPORTED_POLICIES`, and
`GrootPolicy` wraps N1.7 only (N1.5 support was removed). Three differences from the stock
server, all verified against that installed source:

1. **It needs the `groot` extra** (`transformers>=5.4,<5.6`, `peft`, `diffusers`, `timm`,
   `dm-tree`, `decord`), plus the gated **`nvidia/Cosmos-Reason2-2B`** backbone processor. Accept
   that model's license on huggingface.co with the account whose token the token cell logs in
   with, or the server fails while loading the processors.
2. **The stock server can't decode relative-action N1.7 checkpoints.** It post-processes one
   action at a time, and `GrootN17ActionDecodeStep` raises
   `NotImplementedError: ... cannot decode native relative actions one step at a time` (most
   SO-101 N1.7 fine-tunes, including the candidate below, are relative). So start the server with
   `control/vla_bridge/groot_n17_server.py`. For `policy_type == "groot"` it decodes the whole
   chunk in one call, and leaves every other policy on the stock path.
3. **GPU:** a 3B model kept in fp32 with bf16 autocast. Use an **L4 or A100**, not a T4.

**What the robot receives:** absolute joint targets in degrees, decoded server-side. Nothing is
converted relative-to-absolute on the laptop (see `backends/relative_ee_conversion.py`).

**Checkpoint note:** the previous default checkpoint was removed from the Hub (HTTP 401,
2026-10-18). This replacement was verified live on Colab; details are in
`backend_configs/groot_n17.json`.

**Checkpoint:** `{GROOT_N17_CHECKPOINT}` (front + wrist, relative joints
excluding the gripper, degree units, Apache-2.0, chunk_size 16, possibly simulation data), from
`backend_configs/groot_n17.json`. Its `camera_map` sends the wrist frame as
`wrist` and the left AR0144 overhead half as `front`. SO-101 N1.7 checkpoints name their cameras
differently, so another checkpoint needs `--camera-map name=source,...` matching its own
`observation.images.<name>` keys.

This notebook launches the patched launcher instead of the stock server.""",
    "install": {"extras": "async,groot", "diffusers": True},
    "extra_cells": [],
    "launch_notes": r"""## Patched launcher

The next cell writes `control/vla_bridge/groot_n17_server.py` into
`/content/groot_n17_server.py`, and the cell after it checks its sha256. The launch cell starts
it instead of the stock server, on the same port (5173). The first handshake downloads the 3B
base (`nvidia/GR00T-N1.7-3B`), the backbone processor and the fine-tune, so expect the first
chunk to take minutes.""",
    "launch": {
        "argv": "[sys.executable, '-u', '/content/groot_n17_server.py', "
                "'--host', '0.0.0.0', '--port', '5173']",
        "env": "{'KMP_DUPLICATE_LIB_OK': 'TRUE', 'OMP_NUM_THREADS': '1'}",
        "timeout_s": 90,
        "banner": "Patched PolicyServer._predict_action_chunk",
        "post": GRPC_POST,
    },
    "local_md": f"""## Run locally

After the tunnel cell, run these in `control/`. The tunnel cell prints both commands with the
address filled in. The smoke run uses synthetic hardware (the arm cannot move); the first chunk
takes minutes, so it waits up to 300 s:

```bash
{SMOKE} --backend groot-n17 --server-address <ngrok_tcp_host:port> --checkpoint {GROOT_N17_CHECKPOINT} --wait-s 300
```

Then the episode:

```bash
.venv/bin/python run_vla_episode.py <port> <robot_id> --backend groot-n17 \\
  --server-address <ngrok_tcp_host>:<ngrok_tcp_port> --checkpoint {GROOT_N17_CHECKPOINT} \\
  --out outputs/groot_n17_UNIQUE_NAME --max-steps 20
```

A healthy server log shows `Policy type: groot`, then
`Preprocessing and inference took ...s, action shape: torch.Size([1, 16, 6])`-style lines and
`(GR00T whole-chunk decode)` timing lines. Failures look different:
- `Error in StreamActions: GrootN17ActionDecodeStep cannot decode native relative actions one step
  at a time`: the stock server is running instead of the patched launcher.
- `KeyError: 'observation.images.<name>'`: the `camera_map` doesn't match the checkpoint's camera names; pass `--camera-map`.
- A 401/403 or "gated repo" error for `nvidia/Cosmos-Reason2-2B`: accept its license first.

On the robot, watch for the known N1.7 SO-ARM failure (14-RESEARCH.md Pitfall 1): every joint
pinned at its per-step clamp in the first steps. If you see it, stop. It means the checkpoint's
units are not degrees (check its normalizer stats) or something is converting the action twice.

Tear down with the teardown cell at the end of this notebook.""",
    "smoke_command": f"{SMOKE} --backend groot-n17 --server-address <ngrok_tcp_host:port> "
                     f"--checkpoint {GROOT_N17_CHECKPOINT} --wait-s 300",
    "episode_command": f""".venv/bin/python run_vla_episode.py <port> <robot_id> --backend groot-n17 \\
  --server-address <ngrok_tcp_host>:<ngrok_tcp_port> --checkpoint {GROOT_N17_CHECKPOINT} \\
  --out outputs/groot_n17_UNIQUE_NAME --max-steps 20""",
    "placeholders": list(GRPC_PLACEHOLDERS),
    "version_check": GRPC_VERSION_CHECK,
    "teardown_comment": "Teardown: every session, after the local run has finished.",
}


# --- HTTP adapters --------------------------------------------------------------------------

def http_launch_notes(server, extra):
    return (f"## Adapter source and launch notes\n\nThe next cell writes "
            f"`control/vla_bridge/{server}` into `/content/{server}`, and the cell after it checks "
            f"its sha256. The launch cell then starts it as a separate OS process.\n\n"
            + extra.strip("\n"))


def http_local_md(local, smoke):
    return (local.strip("\n") + "\n\nSmoke run first (synthetic hardware, the arm cannot move; "
            "the tunnel cell prints it with the URL filled in):\n\n```bash\n" + smoke + "\n```")


def http_spec(**spec):
    base = {
        "lane": "http",
        "tunnel": "http",
        "high_ram": False,
        "licences": [],
        "prerequisites": "",
        "extra_cells": [],
        "teardown_comment": f"{spec['endpoint_label']} teardown -- every session, after the "
                            "local run has finished.",
    }
    base.update(spec)
    base["stop_patterns"] = [base["marker"]]
    base["embed"] = [base["marker"]]
    base["local_md"] = http_local_md(base.pop("local"), base["smoke_command"])
    return base


OPENVLA_OFT = http_spec(
    name="http_openvla_oft",
    backend="openvla-oft",
    lane_label="HTTP adapter (isolated Python 3.10 venv)",
    title="OpenVLA-OFT resident adapter (`--backend openvla-oft`)",
    serves="Serves OpenVLA-OFT from its own resident Flask adapter, "
           "`control/vla_bridge/openvla_oft_server.py`, through an ngrok HTTP tunnel.",
    port=3001,
    tunnel_var="oft_tunnel",
    endpoint_label="OpenVLA-OFT",
    gpu="L4",
    gpu_type="L4",
    prerequisites="14-UAT 4 package-legitimacy sign-off first",
    checkpoint="moojink/openvla-7b-oft-finetuned-libero-object@4c89574e1c53",
    log="/content/openvla_oft_server.log",
    log_var="OFT_LOG",
    process_var="oft_server_process",
    marker="openvla_oft_server.py",
    intro_md=r"""## Tier C: OpenVLA-OFT resident adapter (Plan 14-03, `--backend openvla-oft`)

OpenVLA-OFT has no LeRobot `PolicyServer` integration. It runs as its own
resident Flask adapter, `control/vla_bridge/openvla_oft_server.py`, built the
same way as the resident depth service (`fastfs_depth.ipynb`): it loads once, warms up, then serves.
It does not use the gRPC PolicyServer lane.

**Status:** these cells were written from upstream source (github.com/moojink/openvla-oft
@ `e4287e9`, `SETUP.md` + `vla-scripts/deploy.py`) and have **not yet been run on
Colab**. The package-legitimacy human check in 14-03-SUMMARY.md must pass before
the clone and install cell runs. `pip install -e .` pulls two more GitHub-source
forks, `moojink/transformers-openvla-oft` and `moojink/dlimp_openvla`.

Requirements: an A100 or L4 runtime (a 7B model in bf16 needs about 16 GB of VRAM).
openvla-oft pins `torch==2.2.0` and `tensorflow==2.15.0`, so it needs Python 3.10
in its own isolated environment, never the PolicyServer's.""",
    install=[("install", "code", r'''import subprocess, sys
from pathlib import Path
OFT_DIR = Path('/content/openvla-oft')
OFT_COMMIT = 'e4287e94541f459edc4feabc4e181f537cd569a8'  # reviewed revision
OFT_VENV = '/content/openvla_oft_venv'
OFT_PYTHON = f'{OFT_VENV}/bin/python'
if not (OFT_DIR / '.git').exists():
    subprocess.run(['git', 'clone', 'https://github.com/moojink/openvla-oft.git', str(OFT_DIR)], check=True)
subprocess.run(['git', '-C', str(OFT_DIR), 'checkout', OFT_COMMIT], check=True)
if not Path(OFT_PYTHON).exists():
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv'], check=True)
    subprocess.run(['uv', 'venv', '--python', '3.10', '--seed', OFT_VENV], check=True)
subprocess.run(['uv', 'pip', 'install', '--python', OFT_PYTHON, '-e', str(OFT_DIR)], check=True)
subprocess.run(['uv', 'pip', 'install', '--python', OFT_PYTHON, 'flask', 'opencv-python-headless'], check=True)''')],
    launch_notes=http_launch_notes("openvla_oft_server.py", r"""On first start
it downloads the pinned checkpoint revision
(`moojink/openvla-7b-oft-finetuned-libero-object@4c89574e1c53`, about 15 GB). `/health`
only answers after the model has loaded and warmed up. Logs go to
`/content/openvla_oft_server.log`."""),
    launch={
        "constants": "OFT_DIR = '/content/openvla-oft'\n"
                     "OFT_PYTHON = '/content/openvla_oft_venv/bin/python'",
        "argv": "[OFT_PYTHON, '-u', '/content/openvla_oft_server.py', '--repo', OFT_DIR, "
                "'--port', '3001']",
        "env": "{'KMP_DUPLICATE_LIB_OK': 'TRUE', 'OMP_NUM_THREADS': '1'}",
        "timeout_s": 1800,  # checkpoint download + load + warmup
        "post": "print('OpenVLA-OFT adapter ready')",
    },
    local=r"""## Run locally

Then, locally in `control/` (pass the base URL; the client appends `/action`):

```bash
.venv/bin/python run_vla_episode.py --backend openvla-oft \
  --server-address https://OFT_TUNNEL.ngrok-free.app \
  --out outputs/openvla_oft_UNIQUE_NAME --max-steps 20
```""",
    smoke_command=f"{SMOKE} --backend openvla-oft --server-address https://<oft_tunnel>",
    episode_command=r""".venv/bin/python run_vla_episode.py --backend openvla-oft \
  --server-address https://OFT_TUNNEL.ngrok-free.app \
  --out outputs/openvla_oft_UNIQUE_NAME --max-steps 20""",
    placeholders=["https://<oft_tunnel>", "https://OFT_TUNNEL.ngrok-free.app"],
    version_check=venv_version_check("OFT_PYTHON", "/content/openvla_oft_venv/bin/python",
                                     ("torch", "transformers", "tensorflow", "flask")),
)

WALLX = http_spec(
    name="http_wallx",
    backend="wallx",
    lane_label="HTTP adapter (kernel Python, `lerobot[wallx]==0.6.1`)",
    title="Wall-X / WALL-OSS resident adapter (`--backend wallx`)",
    serves="Serves Wall-X through lerobot's own `WallXPolicy` from a resident Flask adapter, "
           "`control/vla_bridge/wallx_server.py`, through an ngrok HTTP tunnel.",
    port=3002,
    tunnel_var="wallx_tunnel",
    endpoint_label="Wall-X",
    gpu="A100 High-RAM",
    gpu_type="A100",
    high_ram=True,
    checkpoint="Zekai-Chen/wallx_pick_place_3cam_200ep@ef8ce55f9772",
    log="/content/wallx_server.log",
    log_var="WALLX_LOG",
    process_var="wallx_server_process",
    marker="wallx_server.py",
    intro_md=r"""## Tier C: Wall-X / WALL-OSS resident adapter (Plan 14-08, `--backend wallx`)

The installed `lerobot==0.6.1` `PolicyServer` does not dispatch `wall_x`: it is not in
`async_inference/constants.py`'s `SUPPORTED_POLICIES`. So Wall-X can't use the gRPC lane. It runs
as its own resident Flask adapter, `control/vla_bridge/wallx_server.py`, built like the
OpenVLA-OFT adapter (see README): it loads once, warms up, then serves.

The adapter loads Wall-X through **lerobot's own** `WallXPolicy` (`lerobot[wallx]==0.6.1`, a
pure-Python extra). It does **not** clone or `pip install` `github.com/X-Square-Robot/wall-x`. That
route needs a source build of flash-attn, `dmuon` from a second GitHub repo, and an nvcc
`CUDAExtension` compile (14-RESEARCH.md Pitfall 5). This path has none of those steps.

**Status:** written from installed lerobot 0.6.1 source and HF Hub metadata. These cells have
**not yet been run on Colab**.

Requirements:
- **GPU:** an A100. The pinned fine-tune and the base model it builds on are 8.4 GB of
  safetensors each, and `WallXPolicy` builds the model in fp32 before casting it to bf16.
- **RAM:** a High-RAM runtime is likely needed for the same reason.

Default checkpoint: `Zekai-Chen/wallx_pick_place_3cam_200ep@ef8ce55f9772`. It is trained on
SO-101 in degree mode, for the single task "Put the object in the bin", with cameras
`wrist`/`top`/`side`.""",
    install=[("install", "code", r'''# Wall-X only: lerobot's own `wallx` extra, same lerobot pin as the gRPC notebooks (pure Python:
# transformers>=5.4,<5.6, peft, scipy, torchdiffeq, qwen-vl-utils). No GitHub clone and no
# CUDA-operator compile. Both the install and the import check are verified by exit status,
# not by a quiet pip run.
import subprocess, sys
install = subprocess.run([sys.executable, '-m', 'pip', 'install', 'lerobot[wallx]==0.6.1', 'flask'],
                         capture_output=True, text=True)
print(install.stdout[-2000:]); print(install.stderr[-2000:])
if install.returncode != 0:
    raise RuntimeError(f'lerobot[wallx] install failed (exit {install.returncode})')
# Fresh interpreter: this kernel may still hold the pre-install transformers in memory.
check = subprocess.run([sys.executable, '-c', (
    "import importlib.metadata as m\n"
    "from lerobot.utils.import_utils import _wallx_deps_available\n"
    "from lerobot.policies.factory import get_policy_class\n"
    "assert _wallx_deps_available, 'lerobot[wallx] dependencies are missing'\n"
    "print(get_policy_class('wall_x'))\n"
    "print({p: m.version(p) for p in ('lerobot', 'torch', 'transformers', 'peft', 'qwen-vl-utils')})\n"
)], capture_output=True, text=True)
print(check.stdout); print(check.stderr[-2000:])
if check.returncode != 0:
    raise RuntimeError('lerobot[wallx] import check failed; see stderr above')''')],
    launch_notes=http_launch_notes("wallx_server.py", r"""On first start it snapshot-downloads
two pinned revisions, about 17 GB in total:
- the fine-tune, `Zekai-Chen/wallx_pick_place_3cam_200ep@ef8ce55f9772`
- the base model it builds on, `x-square-robot/wall-oss-flow@44e827683819`

`/health` only answers after the model has loaded and warmed up. Logs go to
`/content/wallx_server.log`.

To use a different SO-101 `wall_x` checkpoint, pass `--checkpoint <repo> --revision <sha>`.
Also pass `--camera-map key=source,...`, mapping each of its camera keys to `wrist`,
`overhead_left` or `overhead_right`. The server refuses to start if any camera is unmapped."""),
    launch={
        "argv": "[sys.executable, '-u', '/content/wallx_server.py', '--port', '3002']",
        "env": "{'TOKENIZERS_PARALLELISM': 'false'}",
        "timeout_s": 3600,  # two 8.4 GB downloads + load + warmup
        "post": "print('Wall-X adapter ready')",
    },
    local=r"""## Run locally

Then, locally in `control/` (pass the base URL; the client appends `/action`):

```bash
.venv/bin/python run_vla_episode.py --backend wallx \
  --server-address https://WALLX_TUNNEL.ngrok-free.app \
  --instruction "Put the object in the bin" \
  --out outputs/wallx_UNIQUE_NAME --max-steps 20
```

`--instruction` matches the default checkpoint's only training task. Each step's
`model_version` should read `Zekai-Chen/wallx_pick_place_3cam_200ep@ef8ce55f9772`. A step that
logs `http-error-holding-position` means the adapter returned an error. Check
`/content/wallx_server.log`: a 500 `wall-x inference failed` means the model produced a
non-finite action, and a 400 means a bad request.""",
    smoke_command=f"{SMOKE} --backend wallx --server-address https://<wallx_tunnel> "
                  "--instruction 'Put the object in the bin'",
    episode_command=r""".venv/bin/python run_vla_episode.py --backend wallx \
  --server-address https://WALLX_TUNNEL.ngrok-free.app \
  --instruction "Put the object in the bin" \
  --out outputs/wallx_UNIQUE_NAME --max-steps 20""",
    placeholders=["https://<wallx_tunnel>", "https://WALLX_TUNNEL.ngrok-free.app"],
    version_check=r'''import subprocess
import sys

probe = (
    "import importlib.metadata as m\n"
    "print({p: m.version(p) for p in ('lerobot', 'torch', 'transformers', 'peft', 'qwen-vl-utils')})\n"
)
result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True)
print(result.stdout)
print(result.stderr[-2000:])''',
)

OPENVLA = http_spec(
    name="http_openvla",
    backend="openvla",
    lane_label="HTTP adapter (isolated Python 3.10 venv)",
    title="OpenVLA resident adapter (`--backend openvla`)",
    serves="Serves OpenVLA (the original 7B model) from its own resident Flask adapter, "
           "`control/vla_bridge/openvla_server.py`, through an ngrok HTTP tunnel.",
    port=3003,
    tunnel_var="openvla_tunnel",
    endpoint_label="OpenVLA",
    gpu="L4",
    gpu_type="L4",
    prerequisites="14-UAT 5 licence decision",
    checkpoint="toasta18/openvla-so101-cylinder@f00833ce5851",
    log="/content/openvla_server.log",
    log_var="OPENVLA_LOG",
    process_var="openvla_server_process",
    marker="openvla_server.py",
    intro_md=r"""## Tier C: OpenVLA resident adapter (Plan 14-09, `--backend openvla`)

OpenVLA (the original 7B model, not OFT) has no LeRobot `PolicyServer` integration. It runs as its
own resident Flask adapter, `control/vla_bridge/openvla_server.py`, built like the OpenVLA-OFT
adapter (see README): it loads once, warms up, then serves. It does not use the gRPC PolicyServer lane.

Unlike OpenVLA-OFT, it needs **no GitHub clone**. The official inference path
(github.com/openvla/openvla README and `vla-scripts/deploy.py`) is plain
`transformers` AutoClasses with `trust_remote_code=True`. The adapter pins that remote code: it
copies `openvla/openvla-7b@47a0ec7f`'s three `*_prismatic.py` files into the pinned checkpoint
snapshot and rewrites `auto_map` to point at them, so no other code can run.

**Status:** written from upstream source and HF Hub metadata. These cells have **not yet been run
on Colab**.

Requirements:
- **GPU:** an A100 or L4. The model is 7.5B parameters in bf16, about 15 GB of weights, so it
  needs about 16 GB of VRAM.
- **Dependencies:** OpenVLA's modeling code requires `timm` 0.9.x and expects
  `transformers==4.40.1` / `tokenizers==0.19.1`. Those must never touch the PolicyServer's
  environment, so the adapter runs in its own Python 3.10 venv. `openvla_server.py` refuses to
  start on other versions.

Default checkpoint: `toasta18/openvla-so101-cylinder@f00833ce5851`. It is an SO-101 joint-space
fine-tune (degrees, gripper percent) with the single task "Pick up the black cylinder and place it
on the plate." The official MIT `openvla/openvla-7b-finetuned-libero-{object,spatial,goal,10}`
checkpoints are reviewed alternates: their delta-EEF output goes through IK on the control side.""",
    install=[("install", "code", r'''# OpenVLA only: an isolated Python 3.10 venv with the versions OpenVLA pins (README /
# pyproject / modeling_prismatic.py). No GitHub clone, no flash-attn build (the adapter uses sdpa).
# Both the install and the import check are verified by exit status.
import subprocess, sys
from pathlib import Path
OPENVLA_VENV = '/content/openvla_venv'
OPENVLA_PYTHON = f'{OPENVLA_VENV}/bin/python'
if not Path(OPENVLA_PYTHON).exists():
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv'], check=True)
    subprocess.run(['uv', 'venv', '--python', '3.10', '--seed', OPENVLA_VENV], check=True)
install = subprocess.run(
    ['uv', 'pip', 'install', '--python', OPENVLA_PYTHON,
     'torch==2.2.0', 'torchvision==0.17.0', 'transformers==4.40.1', 'tokenizers==0.19.1',
     'timm==0.9.10', 'accelerate==0.30.1', 'sentencepiece==0.1.99', 'numpy<2',
     'pillow', 'flask', 'opencv-python-headless', 'requests'],
    capture_output=True, text=True)
print(install.stdout[-2000:]); print(install.stderr[-2000:])
if install.returncode != 0:
    raise RuntimeError(f'OpenVLA venv install failed (exit {install.returncode})')
check = subprocess.run([OPENVLA_PYTHON, '-c', (
    "import importlib.metadata as m, torch\n"
    "print({p: m.version(p) for p in ('torch', 'transformers', 'tokenizers', 'timm', 'accelerate', 'huggingface-hub')})\n"
    "assert torch.cuda.is_available(), 'no CUDA GPU visible to the OpenVLA venv'\n"
)], capture_output=True, text=True)
print(check.stdout); print(check.stderr[-2000:])
if check.returncode != 0:
    raise RuntimeError('OpenVLA venv import check failed; see stderr above')''')],
    launch_notes=http_launch_notes("openvla_server.py", r"""It runs with the venv's Python.

On first start it does two downloads, about 15 GB in total:
- the pinned checkpoint revision
- the three pinned `openvla/openvla-7b` code files

`/health` only answers after the model has loaded and warmed up. It reports `unnorm_key`,
`action_space` and the action q01/q99. Logs go to `/content/openvla_server.log`.

To use a LIBERO alternate, add `--checkpoint openvla/openvla-7b-finetuned-libero-object` to the
launch cell's command. Its reviewed revision and unnorm key are applied automatically. Pass the
**same** `--checkpoint` to `run_vla_episode.py`, because it selects the client-side conversion."""),
    launch={
        "constants": "OPENVLA_PYTHON = '/content/openvla_venv/bin/python'",
        "argv": "[OPENVLA_PYTHON, '-u', '/content/openvla_server.py', '--port', '3003']",
        "env": "{'TOKENIZERS_PARALLELISM': 'false'}",
        "timeout_s": 2400,  # 15 GB download + load + warmup
        "post": "print('OpenVLA adapter ready:', {k: health['model'][k] for k in "
                "('checkpoint', 'revision', 'unnorm_key', 'action_space')})",
    },
    local=r"""## Run locally

Then, locally in `control/` (pass the base URL; the client appends `/action`):

```bash
.venv/bin/python run_vla_episode.py --backend openvla \
  --server-address https://OPENVLA_TUNNEL.ngrok-free.app \
  --instruction "Pick up the black cylinder and place it on the plate." \
  --out outputs/openvla_UNIQUE_NAME --max-steps 20
```

`--instruction` matches the default checkpoint's only training task. Each step's `model_version`
should read `toasta18/openvla-so101-cylinder@f00833ce5851`.

For a LIBERO alternate, add `--checkpoint openvla/openvla-7b-finetuned-libero-object`, the same
value the server was launched with. Its delta-EEF actions then go through
`eef_to_joint.solve_ik()`.

A step that logs `http-error-holding-position` means the adapter returned an error. Check
`/content/openvla_server.log`:
- 500 `openvla inference failed` means the model produced a non-finite or wrong-shaped action.
- 400 means a bad request.

Expect the default checkpoint's gripper to stay near ~53 %. That is a probable training-time
masking defect, explained in `openvla_server.py`.""",
    smoke_command=f"{SMOKE} --backend openvla --server-address https://<openvla_tunnel> "
                  "--instruction 'Pick up the black cylinder and place it on the plate.'",
    episode_command=r""".venv/bin/python run_vla_episode.py --backend openvla \
  --server-address https://OPENVLA_TUNNEL.ngrok-free.app \
  --instruction "Pick up the black cylinder and place it on the plate." \
  --out outputs/openvla_UNIQUE_NAME --max-steps 20""",
    placeholders=["https://<openvla_tunnel>", "https://OPENVLA_TUNNEL.ngrok-free.app"],
    version_check=venv_version_check(
        "OPENVLA_PYTHON", "/content/openvla_venv/bin/python",
        ("torch", "transformers", "tokenizers", "timm", "accelerate", "huggingface-hub")),
)

OCTO = http_spec(
    name="http_octo",
    backend="octo",
    lane_label="HTTP adapter (isolated Python 3.10 venv, JAX)",
    title="Octo resident adapter (`--backend octo`)",
    serves="Serves Octo (JAX/Flax) from its own resident Flask adapter, "
           "`control/vla_bridge/octo_server.py`, through an ngrok HTTP tunnel.",
    port=3004,
    tunnel_var="octo_tunnel",
    endpoint_label="Octo",
    gpu="L4",
    gpu_type="L4",
    prerequisites="14-UAT 5 licence decision",
    checkpoint="sapanostic/octo-so101-3hz-finetune@a45ff970",
    log="/content/octo_server.log",
    log_var="OCTO_LOG",
    process_var="octo_server_process",
    marker="octo_server.py",
    intro_md=r"""## Tier C: Octo resident adapter (Plan 14-10, `--backend octo`)

Octo has no LeRobot `PolicyServer` integration and is **JAX/Flax**, not PyTorch. It runs as its own
resident Flask adapter, `control/vla_bridge/octo_server.py`, built like the OpenVLA adapters (see README): it
loads once, warms up (JIT-compiles), then serves. It does not use the gRPC PolicyServer lane.

The code comes from **github.com/octo-models/octo**, cloned at the pinned commit `241fb35` (MIT). It is
not on PyPI. The adapter loads through Octo's own `OctoModel.load_pretrained` / `create_tasks` /
`sample_actions`, exactly as the repo's `examples/03` and `examples/04` do. It never uses Octo's
`hf://` path, which downloads `main` and every checkpoint step. Instead it downloads one step of the
pinned revision, and it pins the `t5-base` tokenizer to `google-t5/t5-base@a9723ea`.

**Status:** written from upstream source and HF Hub metadata. These cells have **not yet been run
on Colab**.

Requirements:
- **GPU:** any Colab GPU. Octo-Base is 93M parameters and Octo-Small is 27M.
- **Dependencies:** Octo pins Python 3.10, `jax[cuda11_pip]==0.4.20`, `flax==0.7.5`,
  `tensorflow==2.15.0` and `numpy==1.24.3`. None of those may touch the PolicyServer's environment, so
  the adapter runs in its own venv. `octo_server.py` refuses to start on other versions.

Default checkpoint: `sapanostic/octo-so101-3hz-finetune@a45ff970`, step 10000. It is a real **SO-101**
fine-tune of `octo-base-1.5`, trained at 3 Hz with the task "pick up the cube", on primary + wrist
cameras with a 2-frame window. Its action is a 7-D end-effector **delta**: metres, roll/pitch/yaw
differences, and an absolute gripper. The control side converts it with IK. The official MIT
`rail-berkeley/octo-{small,base}-1.5` checkpoints, with the `bridge_dataset` statistics, are reviewed
alternates.""",
    install=[("install", "code", r'''# Octo only: the official install (README: pip install -e . + requirements.txt + jax cuda11 wheel)
# in an isolated Python 3.10 venv, at a pinned octo-models/octo commit. Extra exact pins fix what
# requirements.txt leaves open (orbax-checkpoint, transformers) to versions contemporary with
# jax 0.4.20 / flax 0.7.5. Clone, install and import check are all verified by exit status.
import subprocess, sys
from pathlib import Path
OCTO_COMMIT = '241fb3514b7c40957a86d869fecb7c7fc353f540'
OCTO_DIR = '/content/octo'
OCTO_VENV = '/content/octo_venv'
OCTO_PYTHON = f'{OCTO_VENV}/bin/python'
if not Path(OCTO_DIR).exists():
    subprocess.run(['git', 'clone', 'https://github.com/octo-models/octo', OCTO_DIR], check=True)
subprocess.run(['git', '-C', OCTO_DIR, 'checkout', '--detach', OCTO_COMMIT], check=True)
head = subprocess.run(['git', '-C', OCTO_DIR, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True)
if head.stdout.strip() != OCTO_COMMIT:
    raise RuntimeError(f'octo checkout is at {head.stdout.strip()}, expected {OCTO_COMMIT}')
if not Path(OCTO_PYTHON).exists():
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv'], check=True)
    subprocess.run(['uv', 'venv', '--python', '3.10', '--seed', OCTO_VENV], check=True)
install = subprocess.run(
    ['uv', 'pip', 'install', '--python', OCTO_PYTHON,
     '--find-links', 'https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
     '-r', f'{OCTO_DIR}/requirements.txt', '-e', OCTO_DIR,
     'jax[cuda11_pip]==0.4.20', 'flax==0.7.5', 'orbax-checkpoint==0.4.4', 'transformers==4.34.1',
     'numpy==1.24.3', 'flask', 'opencv-python-headless==4.11.0.86', 'pillow', 'requests', 'huggingface_hub'],
    capture_output=True, text=True)
print(install.stdout[-2000:]); print(install.stderr[-2000:])
if install.returncode != 0:
    raise RuntimeError(f'Octo venv install failed (exit {install.returncode})')
check = subprocess.run([OCTO_PYTHON, '-c', (
    "import importlib.metadata as m, jax\n"
    "print({p: m.version(p) for p in ('jax', 'jaxlib', 'flax', 'orbax-checkpoint', 'tensorflow', 'numpy', 'transformers')})\n"
    "from octo.model.octo_model import OctoModel\n"
    "assert jax.default_backend() == 'gpu', f'JAX backend is {jax.default_backend()}, expected gpu'\n"
)], capture_output=True, text=True)
print(check.stdout); print(check.stderr[-2000:])
if check.returncode != 0:
    raise RuntimeError('Octo venv import check failed; see stderr above')''')],
    launch_notes=http_launch_notes("octo_server.py", r"""It runs with the venv's Python.

On first start it downloads two things:
- the pinned checkpoint revision's `config.json`, `dataset_statistics.json` and `example_batch.msgpack`,
  plus **one** step of params (about 0.8 GB for Octo-Base)
- the pinned `google-t5/t5-base` tokenizer files

`/health` only answers after the model has loaded and the JIT warm-up has run. It reports `action_space`,
`window_size`, `action_horizon`, the action mean/std/mask, and which rig camera feeds `image_primary` and
`image_wrist`. Logs go to `/content/octo_server.log`.

The default primary camera is `overhead_left`. To use another view, add
`--primary-image-key overhead_right` to the launch cell's command.

To use an official alternate, add `--checkpoint rail-berkeley/octo-small-1.5` to the launch cell's
command. Its reviewed revision, step and `bridge_dataset` key are applied automatically. Pass the
**same** `--checkpoint` to `run_vla_episode.py`, because it selects the client-side gripper mapping."""),
    launch={
        "constants": "OCTO_PYTHON = '/content/octo_venv/bin/python'",
        "argv": "[OCTO_PYTHON, '-u', '/content/octo_server.py', '--port', '3004']",
        "env": "{'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'TOKENIZERS_PARALLELISM': 'false'}",
        "timeout_s": 1200,  # ~1 GB download + load + JIT warmup
        "post": "print('Octo adapter ready:', {k: health['model'][k] for k in (\n"
                "    'checkpoint', 'revision', 'step', 'action_space', 'window_size', "
                "'action_horizon', 'image_keys')})",
    },
    local=r"""## Run locally

Then, locally in `control/` (pass the base URL; the client appends `/action`):

```bash
.venv/bin/python run_vla_episode.py --backend octo \
  --server-address https://OCTO_TUNNEL.ngrok-free.app \
  --instruction "pick up the cube" \
  --out outputs/octo_UNIQUE_NAME --max-steps 20
```

`--instruction` matches the default checkpoint's only training task. Each step's `model_version` should
read `sapanostic/octo-so101-3hz-finetune@a45ff970242a`. Each delta-EEF action goes through
`eef_to_joint.solve_ik()` before the safety validator.

For an official alternate, add `--checkpoint rail-berkeley/octo-small-1.5`, the same value the server
was launched with.

A step that logs `http-error-holding-position` means the adapter returned an error. Check
`/content/octo_server.log`:
- 500 `octo inference failed` means the model produced a non-finite or wrong-shaped action.
- 400 means a bad request.

The default checkpoint was trained at 3 Hz. Each action is about a third of a second of motion, applied
as a delta from the current pose, so the per-step safety clamp may trim it at higher loop rates.""",
    smoke_command=f"{SMOKE} --backend octo --server-address https://<octo_tunnel> "
                  "--instruction 'pick up the cube'",
    episode_command=r""".venv/bin/python run_vla_episode.py --backend octo \
  --server-address https://OCTO_TUNNEL.ngrok-free.app \
  --instruction "pick up the cube" \
  --out outputs/octo_UNIQUE_NAME --max-steps 20""",
    placeholders=["https://<octo_tunnel>", "https://OCTO_TUNNEL.ngrok-free.app"],
    version_check=venv_version_check(
        "OCTO_PYTHON", "/content/octo_venv/bin/python",
        ("jax", "jaxlib", "flax", "orbax-checkpoint", "tensorflow", "numpy", "transformers")),
)

RDT1B = http_spec(
    name="http_rdt1b",
    backend="rdt1b",
    lane_label="HTTP adapter (isolated Python 3.10 venv)",
    title="RDT-1B resident adapter (`--backend rdt1b`)",
    serves="Serves RDT-1B through the RoboticsDiffusionTransformer deployment wrapper from its own "
           "resident Flask adapter, `control/vla_bridge/rdt1b_server.py`, through an ngrok HTTP "
           "tunnel.",
    port=3005,
    tunnel_var="rdt_tunnel",
    endpoint_label="RDT",
    gpu="A100",
    gpu_type="A100",
    checkpoint="robotics-diffusion-transformer/rdt-1b@eb09036",
    log="/content/rdt1b_server.log",
    log_var="RDT_LOG",
    process_var="rdt_server_process",
    marker="rdt1b_server.py",
    intro_md=r"""## Tier C: RDT-1B resident adapter (Plan 14-11, `--backend rdt1b`)

RDT-1B has no LeRobot `PolicyServer` integration. It runs as its own resident Flask adapter,
`control/vla_bridge/rdt1b_server.py`, built like the Octo and OpenVLA adapters (see README): it loads once,
warms up, then serves. It does not use the gRPC PolicyServer lane.

The code comes from **github.com/thu-ml/RoboticsDiffusionTransformer**, cloned at the pinned commit
`cd79363` (MIT). It is not on PyPI. The adapter loads through the repo's own deployment wrapper,
`scripts.agilex_model.RoboticDiffusionTransformerModel`, and overrides only the two hooks the README
says to change for a new robot (`_format_joint_to_state` / `_unformat_action_to_joint`). The
checkpoint, the SigLIP vision encoder and the T5-XXL text encoder are each downloaded at a pinned
revision.

**Status:** written from upstream source and HF Hub metadata. These cells have **not yet been run
on Colab**.

Requirements:
- **GPU: an A100 runtime (Colab Pro).** RDT runs in bfloat16, which a T4 lacks, so
  `rdt1b_server.py` refuses to start there (14-RESEARCH Pitfall 4). Expect about 13 GB of VRAM:
  T5-XXL's encoder is ~9.5 GB in bf16, RDT-1B 2.5 GB, SigLIP 0.8 GB. Below 24 GB of VRAM (e.g. an L4),
  the adapter automatically offloads T5 encoder blocks 12-23 to `--offload-dir`, as upstream recommends.
- **Disk: about 55 GB free.** `google/t5-v1_1-xxl` ships only a 44.5 GB fp32 `pytorch_model.bin`.
  The first launch therefore takes a long time. `--no-text-encoder` with precomputed
  `--lang-embeddings` (upstream `scripts/encode_lang.py` output) skips that download entirely.
- **Dependencies:** RDT pins Python 3.10 and `torch==2.1.0` (no Python 3.12 wheels), so the adapter
  runs in its own venv. `rdt1b_server.py` refuses other versions of torch, transformers, diffusers,
  timm, accelerate and huggingface_hub.

Default checkpoint: the official `robotics-diffusion-transformer/rdt-1b@eb09036`. Its action space was
confirmed from RDT's own code and weights as a 128-D unified vector whose **joint-position** slots
take SO-101 directly: arm joints 0-4 in radians, gripper slot 10 in [0, 1]. **No IK.** There is no
SO-101 RDT fine-tune on the Hub, and the model card says RDT does not generalize to robots it was
not pre-trained on. So this run proves the pipeline, not the task. The official
`robotics-diffusion-transformer/rdt-170m` is a reviewed alternate.""",
    install=[("install", "code", r'''# RDT-1B only: the official install (README: python 3.10, torch 2.1.0 cu121, requirements.txt) in an
# isolated venv, at a pinned thu-ml/RoboticsDiffusionTransformer commit. Only the inference subset of
# requirements.txt is installed (deepspeed, wandb, h5py and imgaug are training/data only). flash-attn is
# skipped: the model code uses torch SDPA through timm, never flash_attn. huggingface_hub is pinned to
# 0.25.2 because diffusers 0.27.2 still imports cached_download (removed in 0.26). Clone, installs and
# import check are all verified by exit status.
import subprocess, sys
from pathlib import Path
RDT_COMMIT = 'cd79363a1387e8f81c7724d070ef7e45fd23150f'
RDT_DIR = '/content/RoboticsDiffusionTransformer'
RDT_VENV = '/content/rdt_venv'
RDT_PYTHON = f'{RDT_VENV}/bin/python'
if not Path(RDT_DIR).exists():
    subprocess.run(['git', 'clone', 'https://github.com/thu-ml/RoboticsDiffusionTransformer', RDT_DIR], check=True)
subprocess.run(['git', '-C', RDT_DIR, 'checkout', '--detach', RDT_COMMIT], check=True)
head = subprocess.run(['git', '-C', RDT_DIR, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True)
if head.stdout.strip() != RDT_COMMIT:
    raise RuntimeError(f'RDT checkout is at {head.stdout.strip()}, expected {RDT_COMMIT}')
if not Path(RDT_PYTHON).exists():
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv'], check=True)
    subprocess.run(['uv', 'venv', '--python', '3.10', '--seed', RDT_VENV], check=True)
for packages in (
    ['--index-url', 'https://download.pytorch.org/whl/cu121', 'torch==2.1.0', 'torchvision==0.16.0'],
    ['transformers==4.41.0', 'diffusers==0.27.2', 'timm==1.0.3', 'accelerate==0.30.1', 'sentencepiece==0.2.0',
     'protobuf==4.25.3', 'huggingface_hub==0.25.2', 'packaging==24.0', 'numpy==1.26.4', 'pyyaml',
     'opencv-python-headless==4.9.0.80', 'flask', 'pillow', 'requests'],
):
    install = subprocess.run(['uv', 'pip', 'install', '--python', RDT_PYTHON, *packages], capture_output=True, text=True)
    print(install.stdout[-2000:]); print(install.stderr[-2000:])
    if install.returncode != 0:
        raise RuntimeError(f'RDT venv install failed (exit {install.returncode})')
check = subprocess.run([RDT_PYTHON, '-c', (
    "import importlib.metadata as m, torch\n"
    "print({p: m.version(p) for p in ('torch', 'transformers', 'diffusers', 'timm', 'accelerate', 'huggingface_hub', 'numpy')})\n"
    "from scripts.agilex_model import RoboticDiffusionTransformerModel\n"
    "from models.multimodal_encoder.t5_encoder import T5Embedder\n"
    "assert torch.cuda.is_available(), 'no CUDA GPU'\n"
    "print(torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), 'GiB')\n"
    "assert torch.cuda.is_bf16_supported(), 'this GPU has no bf16 (T4?): switch the runtime to an A100'\n"
)], cwd=RDT_DIR, capture_output=True, text=True)
print(check.stdout); print(check.stderr[-2000:])
if check.returncode != 0:
    raise RuntimeError('RDT venv import check failed; see stderr above')''')],
    launch_notes=http_launch_notes("rdt1b_server.py", r"""It runs with the venv's Python.

On first start it downloads three pinned revisions, about 51 GB in total:
- the checkpoint's `config.json` and `pytorch_model.bin` (2.5 GB for RDT-1B)
- `google/siglip-so400m-patch14-384` (3.5 GB)
- `google/t5-v1_1-xxl` (44.5 GB), unless you pass `--no-text-encoder`

Before declaring the model ready, it checks that the cloned repo's unified layout is the reviewed one
and that every weight key matches. Then it runs one warm-up inference. `/health` only answers after
that. It reports `action_space`, `state_indices`, `control_hz`, `chunk_step` and the T5 offload folder.
The log at `/content/rdt1b_server.log` records the GPU, whether T5 is offloaded, and the **peak VRAM**
after warm-up. That line is the Pitfall 4 evidence: an OOM shows up there as
`torch.cuda.OutOfMemoryError`, or as the process dying during load.

The server returns the chunk step about one client control period ahead. By default that is
`--client-control-hz 2.0`, matching `run_vla_episode.py`'s default `--control-hz`. If you run the
client at another rate, pass the same value in the launch cell's command.

To use the alternate, add `--checkpoint robotics-diffusion-transformer/rdt-170m` to the launch
cell's command, and pass the **same** `--checkpoint` to `run_vla_episode.py`."""),
    launch={
        "imports": ["import shutil"],
        "constants": "RDT_DIR = '/content/RoboticsDiffusionTransformer'\n"
                     "RDT_PYTHON = '/content/rdt_venv/bin/python'",
        "pre": "free_gb = shutil.disk_usage('/content').free / 1e9\n"
               "if free_gb < 60:\n"
               "    raise RuntimeError(f'only {free_gb:.0f} GB free on /content; RDT needs ~55 GB "
               "(T5-XXL alone is 44.5 GB)')",
        "argv": "[RDT_PYTHON, '-u', '/content/rdt1b_server.py', '--port', '3005', "
                "'--rdt-repo', RDT_DIR]",
        "env": "{'TOKENIZERS_PARALLELISM': 'false'}",
        "cwd": "RDT_DIR",
        "timeout_s": 7200,  # ~51 GB download + load + warmup
        "post": "print('RDT adapter ready:', {k: health['model'][k] for k in (\n"
                "    'checkpoint', 'revision', 'action_space', 'state_indices', 'control_hz', "
                "'chunk_step', 't5_offload_dir')})\n"
                "print([line for line in open(RDT_LOG) if 'GiB' in line or 'OutOfMemory' in line])",
    },
    local=r"""## Run locally

Then, locally in `control/` (pass the base URL; the client appends `/action`):

```bash
.venv/bin/python run_vla_episode.py --backend rdt1b \
  --server-address https://RDT_TUNNEL.ngrok-free.app \
  --instruction "pick up the cube" \
  --out outputs/rdt1b_UNIQUE_NAME --max-steps 20
```

Each step's `model_version` should read `robotics-diffusion-transformer/rdt-1b@eb09036cc64c`. The action
is an absolute joint target: radians are converted to degrees, and the gripper's [0, 1] to percent.
There is no IK. Every action then goes through the safety validator.

For the alternate, add `--checkpoint robotics-diffusion-transformer/rdt-170m`, the same value the server
was launched with.

A step that logs `http-error-holding-position` means the adapter returned an error. Check
`/content/rdt1b_server.log`:
- 500 `rdt inference failed` means the model produced a non-finite or wrong-shaped action.
- 400 means a bad request. With `--no-text-encoder`, this includes an instruction that has no
  precomputed embedding.

Expect motion that is not task-meaningful. The base checkpoint has never seen SO-101, and the per-step
safety clamp bounds every jump.""",
    smoke_command=f"{SMOKE} --backend rdt1b --server-address https://<rdt_tunnel> "
                  "--instruction 'pick up the cube'",
    episode_command=r""".venv/bin/python run_vla_episode.py --backend rdt1b \
  --server-address https://RDT_TUNNEL.ngrok-free.app \
  --instruction "pick up the cube" \
  --out outputs/rdt1b_UNIQUE_NAME --max-steps 20""",
    placeholders=["https://<rdt_tunnel>", "https://RDT_TUNNEL.ngrok-free.app"],
    version_check=venv_version_check(
        "RDT_PYTHON", "/content/rdt_venv/bin/python",
        ("torch", "transformers", "diffusers", "timm", "accelerate", "huggingface_hub", "numpy")),
)

GROOT_N15 = http_spec(
    name="http_groot_n15",
    backend="groot-n15",
    lane_label="HTTP adapter (isolated Python 3.10 venv)",
    title="GR00T N1.5 resident adapter (`--backend groot-n15`)",
    serves="Serves GR00T N1.5 through NVIDIA Isaac-GR00T's own `Gr00tPolicy` from a resident "
           "Flask adapter, `control/vla_bridge/groot_n15_server.py`, through an ngrok HTTP "
           "tunnel.",
    port=3006,
    tunnel_var="groot_n15_tunnel",
    endpoint_label="GR00T N1.5",
    gpu="L4",
    gpu_type="L4",
    checkpoint="quangnd58/GR00T-N1.5-3B-so101-multitask@488f3b6",
    log="/content/groot_n15_server.log",
    log_var="GROOT_N15_LOG",
    process_var="groot_n15_server_process",
    marker="groot_n15_server.py",
    intro_md=r"""## Tier C: GR00T N1.5 resident adapter (Plan 14-12, `--backend groot-n15`)

GR00T N1.5 has no LeRobot path in this repo's lerobot 0.6.1: its `groot` policy is N1.7-only and rejects
N1.5 checkpoints (Plan 14-07). N1.5 therefore runs as its own resident Flask adapter,
`control/vla_bridge/groot_n15_server.py`, built like the RDT-1B and Octo adapters (see README): it loads once,
warms up, then serves. It does not use the gRPC PolicyServer lane, and it is a **separate code path from
GR00T N1.7 (`grpc_groot_n17.ipynb`)** (14-RESEARCH Pitfall 3).

The code comes from **github.com/NVIDIA/Isaac-GR00T**, cloned at the tag `n1.5-release`
(= branch `n1d5`, commit `4af2b62`, Apache-2.0). It is not on PyPI. The adapter loads checkpoints with the
repo's own `gr00t.model.policy.Gr00tPolicy` and the `so100_dualcam` data config, exactly as NVIDIA's SO-101
tutorial (huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning) and `examples/SO-100/eval_lerobot.py` do.

**Status:** written from upstream source and HF Hub metadata. These cells have **not yet been run
on Colab**.

Requirements:
- **GPU: an L4 or A100 runtime.** The bundled Eagle 2.5 backbone hard-codes `flash_attention_2`, which
  needs an Ampere-or-newer GPU. A T4 cannot run it, so `groot_n15_server.py` refuses to start there. The
  model card measures about 6 GB of VRAM for PyTorch inference.
- **Disk: about 10 GB free** (7.6 GB of checkpoint safetensors; optimizer state is not downloaded).
- **Dependencies:** Isaac-GR00T pins Python 3.10, `torch==2.5.1`, `transformers==4.51.3` and
  `flash-attn==2.7.1.post4`, so the adapter runs in its own venv. `groot_n15_server.py` refuses other
  versions, and refuses an Isaac-GR00T checkout at any other commit.

Default checkpoint: `quangnd58/GR00T-N1.5-3B-so101-multitask@488f3b6`, a documented SO-101 fine-tune of
`nvidia/GR00T-N1.5-3B` (Isaac-GR00T `n1d5`, `new_embodiment`, `so100_dualcam`: `video.front` = top camera
+ `video.wrist`). The base `nvidia/GR00T-N1.5-3B` cannot drive SO-101: SO-101 is not in its pre-training,
and its metadata has no `new_embodiment` entry. The action is an **absolute joint target in lerobot
degree mode**, so the client passes it straight through. **No IK, no delta.** The adapter re-derives the
unit mode from the checkpoint's own `metadata.json` and refuses to start if it is not degrees.""",
    install=[("install", "code", r'''# GR00T N1.5 only: the official install (README / NVIDIA's SO-101 tutorial: python 3.10,
# `pip install -e .[base]`, then flash-attn 2.7.1.post4 without build isolation) in an isolated venv,
# at the pinned NVIDIA/Isaac-GR00T n1.5-release commit. flash-attn's setup fetches its prebuilt
# wheel from github.com/Dao-AILab/flash-attention releases (or compiles, which takes a long time).
# Clone, installs and import check are all verified by exit status.
import subprocess, sys
from pathlib import Path
GROOT_N15_COMMIT = '4af2b622892f7dcb5aae5a3fb70bcb02dc217b96'  # tag n1.5-release == branch n1d5
GROOT_N15_DIR = '/content/Isaac-GR00T'
GROOT_N15_VENV = '/content/groot_n15_venv'
GROOT_N15_PYTHON = f'{GROOT_N15_VENV}/bin/python'
if not Path(GROOT_N15_DIR).exists():
    subprocess.run(['git', 'clone', 'https://github.com/NVIDIA/Isaac-GR00T', GROOT_N15_DIR], check=True)
subprocess.run(['git', '-C', GROOT_N15_DIR, 'checkout', '--detach', GROOT_N15_COMMIT], check=True)
head = subprocess.run(['git', '-C', GROOT_N15_DIR, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True)
if head.stdout.strip() != GROOT_N15_COMMIT:
    raise RuntimeError(f'Isaac-GR00T checkout is at {head.stdout.strip()}, expected {GROOT_N15_COMMIT}')
if not Path(GROOT_N15_PYTHON).exists():
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv'], check=True)
    subprocess.run(['uv', 'venv', '--python', '3.10', '--seed', GROOT_N15_VENV], check=True)
for packages in (
    ['--upgrade', 'setuptools'],
    ['-e', f'{GROOT_N15_DIR}[base]'],
    ['packaging', 'psutil', 'ninja', 'wheel', 'flask'],
    ['--no-build-isolation', 'flash-attn==2.7.1.post4'],
):
    install = subprocess.run(['uv', 'pip', 'install', '--python', GROOT_N15_PYTHON, *packages],
                             capture_output=True, text=True)
    print(install.stdout[-2000:]); print(install.stderr[-2000:])
    if install.returncode != 0:
        raise RuntimeError(f'GR00T N1.5 venv install failed (exit {install.returncode}): {packages}')
check = subprocess.run([GROOT_N15_PYTHON, '-c', (
    "import importlib.metadata as m, torch\n"
    "print({p: m.version(p) for p in ('torch', 'transformers', 'flash_attn', 'numpy', 'huggingface_hub')})\n"
    "import flash_attn, gr00t\n"
    "from gr00t.model.policy import Gr00tPolicy\n"
    "from gr00t.experiment.data_config import DATA_CONFIG_MAP\n"
    "assert 'so100_dualcam' in DATA_CONFIG_MAP\n"
    "print('gr00t from', gr00t.__file__)\n"
    "assert torch.cuda.is_available(), 'no CUDA GPU'\n"
    "cap = torch.cuda.get_device_capability(0)\n"
    "print(torch.cuda.get_device_name(0), cap, round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), 'GiB')\n"
    "assert cap >= (8, 0), 'flash-attn 2 needs Ampere or newer (T4 is sm75): switch the runtime to an L4/A100'\n"
)], cwd='/content', capture_output=True, text=True)
print(check.stdout); print(check.stderr[-2000:])
if check.returncode != 0:
    raise RuntimeError('GR00T N1.5 venv import check failed; see stderr above')''')],
    launch_notes=http_launch_notes("groot_n15_server.py", r"""It runs with the venv's Python.

On first start it downloads the checkpoint at its pinned revision (`config.json`, `experiment_cfg/`, the
safetensors and their index: about 7.6 GB). Before declaring the model ready, it:
- checks the clone is at the reviewed commit, and that `gr00t` is imported from it;
- checks the checkpoint's `metadata.json` is the reviewed layout (`video.front` + `video.wrist`,
  `single_arm` [5] + `gripper` [1], both `absolute`) and that its action statistics are **degree mode**;
- loads `Gr00tPolicy(..., embodiment_tag='new_embodiment', so100_dualcam, denoising_steps=4)` and checks
  every weight key against the checkpoint's index;
- runs one warm-up inference.

`/health` only answers after that. It reports `action_space` (`absolute_joint`), `action_units`
(`degrees`), `detected_action_units`, `chunk_step` and `video_sources`. The log at
`/content/groot_n15_server.log` records the GPU and the **peak VRAM** after warm-up.

The server returns the chunk step about one client control period ahead. By default that is
`--client-control-hz 2.0` (step 14 of 16), matching `run_vla_episode.py`'s default `--control-hz`. If you
run the client at another rate, pass the same value in the launch cell's command. `video.front` is the
dataset's top camera; it is fed from this rig's `overhead_left` (`--front-image-key overhead_right` to
change it)."""),
    launch={
        "imports": ["import shutil"],
        "constants": "GROOT_N15_DIR = '/content/Isaac-GR00T'\n"
                     "GROOT_N15_PYTHON = '/content/groot_n15_venv/bin/python'",
        "pre": "free_gb = shutil.disk_usage('/content').free / 1e9\n"
               "if free_gb < 12:\n"
               "    raise RuntimeError(f'only {free_gb:.0f} GB free on /content; GR00T N1.5 needs "
               "~10 GB (7.6 GB checkpoint)')",
        "argv": "[GROOT_N15_PYTHON, '-u', '/content/groot_n15_server.py', '--port', '3006', "
                "'--groot-repo', GROOT_N15_DIR]",
        "env": "{'TOKENIZERS_PARALLELISM': 'false'}",
        "cwd": "'/content'",
        "timeout_s": 3600,  # ~7.6 GB download + load + warmup
        "post": "print('GR00T N1.5 adapter ready:', {k: health['model'][k] for k in (\n"
                "    'checkpoint', 'revision', 'isaac_groot_commit', 'action_space', "
                "'action_units',\n"
                "    'detected_action_units', 'chunk_step', 'video_sources')})\n"
                "print([line for line in open(GROOT_N15_LOG) if 'GiB' in line or "
                "'OutOfMemory' in line])",
    },
    local=r"""## Run locally

Then, locally in `control/` (pass the base URL; the client appends `/action`):

```bash
.venv/bin/python run_vla_episode.py --backend groot-n15 \
  --server-address https://GROOT_N15_TUNNEL.ngrok-free.app \
  --instruction "Pick blue cube and place on red cube" \
  --out outputs/groot_n15_UNIQUE_NAME --max-steps 20
```

Each step's `model_version` should read `quangnd58/GR00T-N1.5-3B-so101-multitask@488f3b6da322`. The action
is an absolute joint target in degrees (gripper in percent), passed straight through: no unit change, no
IK, and the current joint state is never added to it. Every action then goes through the safety validator.
The checkpoint was trained on three prompts; use one verbatim: "Pick up the banana and place it in the bot,
then close the lid", "Pick blue cube and place on red cube", "Pick all cubes and place into cup".

A step that logs `http-error-holding-position` means the adapter returned an error. Check
`/content/groot_n15_server.log`:
- 500 `groot n1.5 inference failed` means the model produced a non-finite or wrong-shaped action chunk.
- 400 means a bad request, e.g. a missing camera or joint.

What a units mistake would look like (it is checked at startup, so this should not happen): a joint driven
toward its limit on the very first step, every step clamped by the per-step limit in the same direction.
If you see that, stop and compare `/health`'s `detected_action_units` with the checkpoint's metadata.
Expect motion that is not task-meaningful: the checkpoint was trained on another rig and scene.""",
    smoke_command=f"{SMOKE} --backend groot-n15 --server-address https://<groot_n15_tunnel> "
                  "--instruction 'Pick blue cube and place on red cube'",
    episode_command=r""".venv/bin/python run_vla_episode.py --backend groot-n15 \
  --server-address https://GROOT_N15_TUNNEL.ngrok-free.app \
  --instruction "Pick blue cube and place on red cube" \
  --out outputs/groot_n15_UNIQUE_NAME --max-steps 20""",
    placeholders=["https://<groot_n15_tunnel>", "https://GROOT_N15_TUNNEL.ngrok-free.app"],
    version_check=venv_version_check(
        "GROOT_N15_PYTHON", "/content/groot_n15_venv/bin/python",
        ("torch", "transformers", "flash_attn", "numpy", "huggingface_hub")),
)


# --- Depth service --------------------------------------------------------------------------

DEPTH_EPISODE = r""".venv/bin/python run_vla_episode.py --out outputs/depth_episode_UNIQUE_NAME \
  --server-address POLICY_TCP_HOST:PORT \
  --checkpoint victorvanhalst/smolvla_so101_cube \
  --stereo-camera-index "CCB Camera" \
  --depth-endpoint https://DEPTH_TUNNEL/depth \
  --depth-calibration stereo_calibration.json --depth-every-n-steps 1 \
  --max-steps 60 --instruction "Pick the red cube and place it in the bowl""" + '"'

FASTFS_DEPTH = {
    "name": "fastfs_depth",
    "backend": "fastfs-depth",
    "lane": "depth",
    "lane_label": "HTTP depth service (isolated FastFS venv)",
    "title": "Fast-FoundationStereo depth service",
    "serves": "Serves stereo depth from NVlabs Fast-FoundationStereo as a resident HTTP service "
              "(`control/vla_bridge/fastfs_server.py`), through an ngrok HTTP tunnel. "
              "`run_vla_episode.py --depth-endpoint` and `depth_camera.DepthCameraClient` call it.",
    "port": 3000,
    "tunnel": "http",
    "tunnel_var": "depth_tunnel",
    "endpoint_label": "Depth",
    "gpu": "GPU runtime; the former combined notebook ran it on A100",
    "gpu_type": "A100",
    "high_ram": False,
    "licences": [],
    "prerequisites": "",
    "checkpoint": "NVlabs Fast-FoundationStereo 23-36-37 (gdown)",
    "log": "/content/fastfs_server.log",
    "log_var": "DEPTH_SERVER_LOG",
    "process_var": "depth_server_process",
    "marker": "fastfs_server.py",
    "stop_patterns": ["fastfs_server.py"],
    "embed": ["fastfs_server.py"],
    "intro_md": r"""## Install and load Fast-FoundationStereo

FastFS runs in its own virtualenv (`/content/fastfs_venv`, torch 2.6.0 cu124), so it never
reinstalls torch/cuDNN under a running PolicyServer.""",
    "install": [
        ("clone-fastfs", "code", r'''import subprocess
from pathlib import Path
FASTFS_DIR = Path('/content/Fast-FoundationStereo')
if not (FASTFS_DIR / '.git').exists():
    subprocess.run(['git', 'clone', 'https://github.com/NVlabs/Fast-FoundationStereo', str(FASTFS_DIR)], check=True)
print('FastFS source revision:', subprocess.check_output(['git', '-C', str(FASTFS_DIR), 'rev-parse', 'HEAD'], text=True).strip())'''),
        ("fastfs-venv", "code", r'''# Separate filesystem environment: never reinstall torch/cuDNN under a running PolicyServer.
import sys
import subprocess
from pathlib import Path
FASTFS_VENV = '/content/fastfs_venv'
FASTFS_PYTHON = f'{FASTFS_VENV}/bin/python'
if 'depth_server_process' in globals() and depth_server_process.poll() is None:
    raise RuntimeError('Stop the depth service before changing its environment; use teardown below.')
if not Path(FASTFS_PYTHON).exists():
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'virtualenv'], check=True)
    subprocess.run([sys.executable, '-m', 'virtualenv', FASTFS_VENV], check=True)
subprocess.run([FASTFS_PYTHON, '-m', 'pip', 'install', 'torch==2.6.0', 'torchvision==0.21.0', 'xformers',
                '--index-url', 'https://download.pytorch.org/whl/cu124'], check=True)
subprocess.run([FASTFS_PYTHON, '-m', 'pip', 'install', '-r', '/content/Fast-FoundationStereo/requirements.txt'], check=True)
subprocess.run([FASTFS_PYTHON, '-m', 'pip', 'install', 'flask'], check=True)
# No GUI patch: the resident adapter calls the model directly, never run_demo.py.'''),
        ("checkpoint-notes", "markdown", r"""Use the trusted NVlabs `23-36-37` research checkpoint below. The launch cell loads it once
in the isolated environment, warms inference, then serves subsequent requests in
process. No CLI subprocess or model reload occurs per depth request."""),
        ("checkpoint-download-heading", "markdown",
         "Download the checkpoint and configuration into the isolated service’s weights directory:"),
        ("checkpoint-download", "code", r'''import os

os.makedirs("/content/Fast-FoundationStereo/weights/23-36-37", exist_ok=True)

!pip install -q gdown
!gdown "1W1V1H64l9bAi97boEQQ2ueNzzGmSMz-E" \
    -O /content/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth
!gdown "1GDBRYL-ZaLpXEtWfGFRJvkBc_2sywjgj" \
    -O /content/Fast-FoundationStereo/weights/23-36-37/cfg.yaml

print(os.listdir("/content/Fast-FoundationStereo/weights/23-36-37"))'''),
    ],
    "extra_cells": [],
    "launch_notes": r"""## Start the resident depth service
Run the cells below. The first writes the versioned adapter (embedded so this
notebook is standalone) and the second checks its sha256. The launch cell starts it as a
separate OS process using FastFS’s virtualenv and waits for warmup. Keep the model resident
for the entire episode. `/health` only becomes available after warmup. Logs survive notebook
errors at `/content/fastfs_server.log`.""",
    "launch": {
        "imports": ["from pathlib import Path"],
        "constants": "FASTFS_PYTHON = '/content/fastfs_venv/bin/python'\n"
                     "FASTFS_DIR = '/content/Fast-FoundationStereo'\n"
                     "FASTFS_MODEL = f'{FASTFS_DIR}/weights/23-36-37/model_best_bp2_serialize.pth'",
        "pre": "if not Path(FASTFS_MODEL).is_file():\n"
               "    raise FileNotFoundError(f'Checkpoint missing: {FASTFS_MODEL}; run the install "
               "and checkpoint download cells first')",
        "argv": "[FASTFS_PYTHON, '-u', '/content/fastfs_server.py', '--repo', FASTFS_DIR,\n"
                "     '--checkpoint', FASTFS_MODEL, '--port', '3000']",
        "env": "{'PYTHONPATH': FASTFS_DIR, 'KMP_DUPLICATE_LIB_OK': 'TRUE', "
               "'OMP_NUM_THREADS': '1'}",
        "cwd": "FASTFS_DIR",
        "before_wait": "print('Loading and warming FastFS once. First compilation may take "
                       "several minutes.')",
        "ready": "lambda h: h.get('status') == 'ready' and h.get('protocol_version') == 1",
        "timeout_s": 600,
        "post": "print('Depth service ready:', health)",
    },
    "after_launch": [
        ("protocol-note", "markdown", r"""This exact request/response shape (`left_png_b64`/`right_png_b64`/
`intrinsics_flat`/`baseline_m` request, `depth_npy_b64` response) is what
`depth_camera.DepthCameraClient.compute_depth()` (Plan 12-06 Task 1) sends
and parses -- keep them in sync if either side changes."""),
    ],
    "tunnel_heading": "## Open the depth tunnel",
    "tunnel_code": r'''from pyngrok import ngrok

# Port 3000 and a plain HTTP tunnel type -- separate from the gRPC notebooks' 5173 TCP
# tunnel, which specifically needs "tcp" for HTTP/2 semantics. This is a plain
# JSON-over-HTTP POST endpoint, so ngrok's default HTTP tunnel type works fine here.
depth_tunnel = ngrok.connect(3000, "http")
print(f"Depth endpoint reachable at: {depth_tunnel.public_url}/depth")''',
    "local_md": r"""Pass the printed `.../depth` URL as `depth_camera.py`'s `--endpoint` argument
(Plan 12-06 Task 1) or `run_vla_episode.py`'s `--depth-endpoint` flag
(Task 3).

## Run the local episode with depth enabled

Run this notebook through the depth tunnel cell, and start a SmolVLA policy server with
`grpc_smolvla.ipynb` (through its tunnel cell). Each notebook's kernel starts its own ngrok
agent. If ngrok refuses a second agent session for the same authtoken, run both notebooks on
one runtime (ports 3000 and 5173 do not collide) and open the second tunnel from this
notebook's kernel with `ngrok.connect(5173, "tcp")`. A teardown cell's `ngrok.kill()` closes
every tunnel of its kernel. Do not run teardown until the local process exits.
Use the updated local checkout containing the async recorder. In the local `control/` directory:

```bash
.venv/bin/python run_vla_episode.py --out outputs/depth_episode_UNIQUE_NAME \
  --server-address POLICY_TCP_HOST:PORT \
  --checkpoint victorvanhalst/smolvla_so101_cube \
  --stereo-camera-index "CCB Camera" \
  --depth-endpoint https://DEPTH_TUNNEL/depth \
  --depth-calibration stereo_calibration.json --depth-every-n-steps 1 \
  --max-steps 60 --instruction "Pick the red cube and place it in the bowl"
```

Use the actual addresses printed by the tunnel cells and a fresh output directory.
The depth health check runs before hardware opens. Each eligible step submits depth
unless a prior request is outstanding; busy skips are explicit, never a backlog.
Motion continues during requests. At shutdown the arm returns to start and the last
request is drained before writing the final depth summary.

Inspect `depth_summary.json`: `complete` must be positive; review `failed` and
`skipped_busy`. `episode.jsonl` links each submitted request to
`depth_overhead/STEP.json`, which records the final success/failure and depth file.
`depth.jsonl` indexes these completed results. `camera_depth_aligned_left/` and
`camera_depth_aligned_right/` are the exact rectified/resized depth inputs;
`depth_calibration.json` preserves calibration. Depth is float32 metres with NaN
for invalid pixels. `action_details.waypoint_sends` records send outcomes;
`raw_model_output` retains the policy output before validation. Driver sends are
not proof of measured physical arrival. This fixes recording; it does not add a
depth input feature to the existing RGB-only SmolVLA checkpoint.

A successful test requires saved valid depth throughout the episode AND observed
smooth motion. Check the manifest timings and the physical run; a normal step-count
termination by itself does not establish success.

Quick reachability check from the laptop (the tunnel cell prints it with the URL filled in):

```bash
curl -s https://<depth_tunnel>/health
```""",
    "smoke_command": "curl -s https://<depth_tunnel>/health",
    "episode_command": DEPTH_EPISODE,
    "placeholders": ["https://<depth_tunnel>", "https://DEPTH_TUNNEL"],
    "version_check": r'''FASTFS_PYTHON = '/content/fastfs_venv/bin/python'
!{FASTFS_PYTHON} -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
!{FASTFS_PYTHON} -c "import torch, torchvision; print('torch', torch.__version__, '| torchvision', torchvision.__version__)"''',
    "teardown_comment": "Run ONLY when the episode has finished and local depth results have "
                        "drained.",
}

NOTEBOOKS = [
    FASTFS_DEPTH,
    SMOLVLA,
    PI0,
    PI05,
    ACT,
    DIFFUSION,
    GROOT_N17,
    OPENVLA_OFT,
    OPENVLA,
    OCTO,
    WALLX,
    RDT1B,
    GROOT_N15,
]
