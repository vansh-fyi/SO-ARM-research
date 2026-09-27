# Colab PolicyServer + Tunnel Launch Instructions

Phase 11, Plan 11-03, Task 3. Documents how to stand up the Colab-hosted
`lerobot.async_inference.PolicyServer` and connect this project's local
`RobotClient` to it over a tunnel, before Plan 11-04 writes any code that
assumes this bridge exists.

**Selected checkpoint (Task 2's `checkpoint:decision`):** `victorvanhalst/smolvla_so101_cube`
— chosen for its exact task-instruction match to D-02's red-cube task. See
`checkpoint_candidates.md` for the fetched `config.json` data and
`11-03-SUMMARY.md` for the full decision rationale.

**pyngrok legitimacy gate:** this file's step 2 installs `pyngrok`, flagged
`SUS` in `11-RESEARCH.md`'s Package Legitimacy Audit. Do **not** run that
install cell until a human has completed the verification in this plan's
Task 3 `checkpoint:human-verify` (visit pypi.org/project/pyngrok, confirm
the `ngrok`/`alexdlaird` maintainer namespace and that `8.1.2` is a real
published version).

---

## Camera mapping (Task 2's split-stereo resolution)

`victorvanhalst/smolvla_so101_cube`'s `config.json` expects 3 named camera
inputs (`camera1`, `camera2`, `camera3` — none marked `empty_camera_*`).
This project's real rig has only 2 physical cameras:

- IMX335 wrist camera, cv2 index 0, native 1920x1080
- AR0144 stereo camera, cv2 index 1, captured at 1600x600 (800x600 per
  half) — **one physical side-by-side stereo frame, not two independent
  feeds** (confirmed in `diagnostics/UAT/function/basic/UAT.md`, Step 4).
  1600x600 was chosen over the previously-implemented 1280x360 for
  meaningfully higher per-eye resolution while remaining confirmed stable;
  the still-broken 2560x720 mode remains frozen even after exhausting the
  last known macOS system-level fix — see `stereo_camera.py`'s module
  docstring for the full diagnosis.

Resolution: split the AR0144's single 1600x600 frame into left/right
halves, giving **3 genuinely distinct real camera views** (not a dummy or
duplicated 3rd slot):

| Checkpoint camera key | Real source                 | Crop                                         |
| --------------------- | --------------------------- | -------------------------------------------- |
| `camera1`           | IMX335 wrist                | as-is, full 1920x1080 frame                  |
| `camera2`           | AR0144 stereo — LEFT half  | columns`[0:800]` of the 1600x600 frame    |
| `camera3`           | AR0144 stereo — RIGHT half | columns`[800:1600]` of the 1600x600 frame |

This is a better structural fit than an empty/dummy 3rd slot: all 3 of the
checkpoint's expected inputs receive real image data during inference,
none is a zero/black placeholder. (Community precedent: other SmolVLA
fine-tunes such as `bklassen3434/smolvla_pick_pen_v2_frozen` and
`Yilin1001/smolvla-pen-v2-030000` pair a 2-real-camera rig with an
inherited 3rd config slot by leaving it a dummy — this project instead has
a 3rd genuinely distinct real view available via the stereo split, which
is a strictly better fit than a dummy slot would be.)

`lerobot.async_inference.robot_client`'s `--robot.cameras` flag takes a
dict of named camera configs, each independently opened by `cv2` — it has
no built-in "crop a region of a wider frame" option. Two ways to produce
`camera2`/`camera3` as independent feeds from the one AR0144 device:

1. **Preferred for this bridge:** a thin wrapper camera process/script that
   opens cv2 index 1 once, splits each frame into left/right halves, and
   republishes them as two separate virtual camera endpoints the
   `RobotClient`'s camera config can point at (e.g. two loopback
   `cv2.VideoCapture`-compatible sources, or if `lerobot`'s camera config
   supports a custom `type` plugin, register a "split-half" camera type
   that does the crop inline). This avoids opening the same physical
   device twice (most webcam drivers do not support concurrent opens of
   one index).
2. **Fallback if (1) isn't scriptable in time:** open cv2 index 1 once in
   the `RobotClient`'s local process, capture the full 1600x600 frame per
   step, crop it into `camera2`/`camera3` numpy arrays in the observation
   dict before it's sent to the `PolicyServer` — this requires a small
   patch/subclass of the stock `robot_client.py` observation-building step
   rather than pure CLI flags, since the stock CLI only supports
   independent `cv2` device opens per named camera.

Plan 11-04 (which writes `robot_client.py`-adjacent bridge code) is
responsible for implementing whichever of these two options it lands on;
this file only fixes the *mapping* (which checkpoint key gets which real
data), not the implementation mechanism.

---

## Step 1 — Colab: install `lerobot[async]`

```python
# Colab notebook cell
!pip install 'lerobot[async]'
```

Adds `grpcio` — already audited and approved in `11-RESEARCH.md`'s Package
Legitimacy Audit as a required transitive dependency of `lerobot`'s own
`async` extra. No separate checkpoint needed for this install.

## Step 2 — Colab: install `pyngrok` (gated — see legitimacy note above)

```python
# Colab notebook cell — DO NOT RUN until the human has completed the
# pyngrok legitimacy check in this plan's Task 3 checkpoint:human-verify
!pip install pyngrok==8.1.2
```

## Step 3 — Colab: start the `PolicyServer`

```python
# Colab notebook cell
from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import serve

config = PolicyServerConfig(host="0.0.0.0", port=5173)
serve(config)
```

Run this cell in the background (or in a separate cell using Colab's
async-cell execution) so the notebook can still run the tunnel cell below
while the server keeps listening.

## Step 4 — Colab: open a TCP tunnel (not HTTP)

```python
# Colab notebook cell
from pyngrok import ngrok

# Explicitly request a TCP tunnel — gRPC needs HTTP/2 semantics that a
# default HTTP tunnel silently breaks (Pitfall 5 in 11-RESEARCH.md).
tunnel = ngrok.connect(5173, "tcp")
print(f"PolicyServer reachable at: {tunnel.public_url}")
# public_url looks like tcp://0.tcp.ngrok.io:12345 — strip the "tcp://"
# prefix; the local RobotClient's --server_address flag wants host:port.
```

Copy the printed `host:port` (after stripping `tcp://`) for Step 5.

If ngrok's free tier no longer supports a TCP tunnel endpoint by the time
this is run (Assumption A4 in `11-RESEARCH.md`), fall back to `cloudflared`
with a configured Zero Trust tunnel for raw TCP forwarding (heavier setup
than ngrok TCP — cloudflared's zero-config "quick tunnels" are HTTP-only
and will not work here).

## Step 5 — Local (`control/`): launch the `RobotClient`

```bash
cd control
source .venv/bin/activate
python -m lerobot.async_inference.robot_client \
    --server_address=<ngrok_tcp_host>:<ngrok_tcp_port> \
    --robot.type=so101_follower \
    --robot.port=/dev/cu.usbmodem5B8E1139151 \
    --robot.id=soarm_follower_02 \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}, camera2: {type: <split-stereo-left>, index_or_path: 1, width: 800, height: 600, fps: 30}, camera3: {type: <split-stereo-right>, index_or_path: 1, width: 800, height: 600, fps: 30} }" \
    --task="Pick up the red cube and place it in the bowl" \
    --policy_type=smolvla \
    --pretrained_name_or_path=victorvanhalst/smolvla_so101_cube \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5
```

Notes:

- `<ngrok_tcp_host>:<ngrok_tcp_port>` = the value printed in Step 4, with
  the `tcp://` prefix stripped.
- `--robot.port` / `--robot.id` match `control/COMMANDS.md`'s documented
  follower arm (`/dev/cu.usbmodem5B8E1139151`, `soarm_follower_02`) — reverify
  via `ls /dev/cu.usbmodem*` if the port has been reassigned since last use.
- The `camera2`/`camera3` `type: <split-stereo-left/right>` placeholders are
  **not** stock `lerobot` camera types — see the "Camera mapping" section
  above; Plan 11-04 implements the actual split mechanism (custom camera
  type plugin, or a `robot_client.py` observation-patch). This file
  documents the target mapping, not that implementation.
- `--task` uses the exact instruction wording from the checkpoint's own
  training task ("Pick the red cube and place it in the bowl"), not a
  paraphrase — the model card / task description string should be
  reverified against the actual checkpoint metadata before the live run in
  Plan 11-05, since exact wording can matter for language-conditioned
  policies.
- `--policy_device=cuda` assumes the `PolicyServer` runs on a Colab GPU
  runtime; the policy device is a Colab-side concern (loaded by
  `PolicyServerConfig`/`serve()` in Step 3), this flag on the local CLI
  side only affects what the `RobotClient` requests be reported.

## Step 6 — Teardown (every session, no exceptions)

An unattended open tunnel into a Colab-hosted `PolicyServer` is an
elevation-of-privilege risk (T-11-08 in this plan's threat register) —
stop both processes at the end of every session, not just when finished
for the day:

```python
# Colab notebook cell, at the end of the session
ngrok.disconnect(tunnel.public_url)
ngrok.kill()
```

Then interrupt/stop the `serve(config)` cell (Colab: click the cell's stop
button, or restart the runtime) so the `PolicyServer` process itself is not
left listening after the tunnel is torn down.

---

## Depth endpoint (Fast-FoundationStereo, Phase 12 gap-closure extension)

Phase 12, Plan 12-06 -- step 2 of the 3-step depth-calibration extension
(`12-CONTEXT.md` D-07). This is a SECOND, independent Colab-side endpoint
alongside the existing gRPC `PolicyServer` (Steps 1-5 above), for
recording-only depth capture (per D-08) -- it is never wired into the
SmolVLA `PolicyServer`'s own observation path.

**Fast-FoundationStereo legitimacy gate:** this section's Step 7 clones a
full external ML repo (not a PyPI package) from GitHub -- a tampering risk
at least as serious as the `pyngrok` gate above. Do **not** run that install
cell until a human has visited `github.com/NVlabs/Fast-FoundationStereo`
themselves and confirmed: (a) it is genuinely under the `NVlabs` GitHub org
(NVIDIA's own research org -- the same org publishing other real repos, e.g.
the earlier `FoundationStereo`), (b) its README matches the real-time
zero-shot stereo-matching / CVPR 2026 description this plan's research is
based on, (c) a quick skim of its `requirements.txt`/`setup.py` (or
equivalent) doesn't reveal anything unexpected before installing it on the
Colab runtime.

### Step 7 — Colab: install and load Fast-FoundationStereo

```python
# Colab notebook cell -- DO NOT RUN until the human has completed the
# Fast-FoundationStereo legitimacy check above.
!git clone https://github.com/NVlabs/Fast-FoundationStereo
```

```python
# Colab notebook cell -- gated by the same Fast-FoundationStereo legitimacy
# check above (do not run before that check and before Step 7's clone cell).
# FastFS's documented "Option 2: pip" environment setup (docs/Fast Foundation
# Stereo Readme.md, "Environment setup" section), run inside the just-cloned
# repo directory.
%cd /content/Fast-FoundationStereo
!pip install torch==2.6.0 torchvision==0.21.0 xformers --index-url https://download.pytorch.org/whl/cu124
!pip install -r requirements.txt
```

Then manually download the `23-36-37` checkpoint from the README's Google
Drive folder (https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap)
and place `model_best_bp2_serialize.pth` at
`/content/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth`
in the Colab runtime -- a manual human step (a gated/shared Google Drive
folder isn't reliably scriptable inside a single Colab cell), not something
this notebook automates.

Step 8 below invokes FastFS's own documented `scripts/run_demo.py` CLI
(README's "Run demo" section) directly, so the input/output CONTRACT (flags,
intrinsic-file format) is fully documented there -- no fabricated
function/class names. The one genuinely unresolved detail is
`run_demo.py`'s exact output filename inside `--out_dir`, which the README
does not state -- Step 8's `DEPTH_OUTPUT_GLOB` marks that spot for a human
to correct after a first real Colab run, matching this file's own existing
precedent for an honestly-unresolved implementation detail (see the
`camera2`/`camera3` `type: <split-stereo-left/right>` placeholder note in the
Camera mapping section above).

### Step 8 — Colab: serve the depth endpoint

```python
# Colab notebook cell
!pip install flask  # widely-used, long-established -- no dedicated
                     # legitimacy check needed, unlike pyngrok/FastFS above

import base64
import glob
import io
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

FASTFS_DIR = "/content/Fast-FoundationStereo"
FASTFS_MODEL = f"{FASTFS_DIR}/weights/23-36-37/model_best_bp2_serialize.pth"

# run_demo.py's README does not state its output filename inside --out_dir --
# this glob picks the most plausible depth output file. Confirm the exact
# filename by running once in Colab and inspecting out_dir; narrow this
# pattern if more than one .npy file is written (e.g. an intermediate file).
DEPTH_OUTPUT_GLOB = "*.npy"  # TODO: confirm exact filename by running once in Colab and inspecting out_dir


def run_fastfs_inference(left_img, right_img, intrinsics_flat, baseline_m):
    """Invokes Fast-FoundationStereo's documented `scripts/run_demo.py` CLI
    (docs/Fast Foundation Stereo Readme.md, "Run demo" section) as a
    subprocess, then loads the resulting depth map back into this process."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        left_path = str(Path(tmp_dir) / "left.png")
        right_path = str(Path(tmp_dir) / "right.png")
        intrinsic_path = str(Path(tmp_dir) / "K.txt")
        out_dir = str(Path(tmp_dir) / "out")
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(left_path, left_img)
        cv2.imwrite(right_path, right_img)

        # README: line 1 = flattened 1x9 intrinsics matrix (space-separated),
        # line 2 = baseline in meters.
        with open(intrinsic_path, "w") as f:
            f.write(" ".join(str(v) for v in intrinsics_flat) + "\n")
            f.write(f"{baseline_m}\n")

        subprocess.run(
            [
                "python", "scripts/run_demo.py",
                "--model_dir", FASTFS_MODEL,
                "--left_file", left_path,
                "--right_file", right_path,
                "--intrinsic_file", intrinsic_path,
                "--out_dir", out_dir,
                "--remove_invisible", "0",
                "--denoise_cloud", "0",
                "--scale", "1",
                "--get_pc", "0",
                "--valid_iters", "8",
                "--max_disp", "192",
                "--zfar", "100",
            ],
            cwd=FASTFS_DIR,
            check=True,
        )

        npy_files = sorted(glob.glob(str(Path(out_dir) / DEPTH_OUTPUT_GLOB)))
        if not npy_files:
            raise RuntimeError(
                f"No files matching {DEPTH_OUTPUT_GLOB!r} found in {out_dir} "
                "after run_demo.py -- adjust DEPTH_OUTPUT_GLOB above."
            )
        # Picks the first alphabetical match if run_demo.py writes more than
        # one .npy file -- narrow DEPTH_OUTPUT_GLOB above if this is wrong.
        return np.load(npy_files[0])


@app.route("/depth", methods=["POST"])
def depth():
    body = request.get_json()
    left_bytes = base64.b64decode(body["left_png_b64"])
    right_bytes = base64.b64decode(body["right_png_b64"])
    left_img = cv2.imdecode(np.frombuffer(left_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    right_img = cv2.imdecode(np.frombuffer(right_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    intrinsics_flat = body["intrinsics_flat"]
    baseline_m = body["baseline_m"]

    depth_map = run_fastfs_inference(left_img, right_img, intrinsics_flat, baseline_m)

    buf = io.BytesIO()
    np.save(buf, depth_map.astype(np.float32))
    depth_npy_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return jsonify({"depth_npy_b64": depth_npy_b64})


# Run in the background, same convention as Step 3's PolicyServerConfig/serve() cell.
app.run(host="0.0.0.0", port=3000)
```

This exact request/response shape (`left_png_b64`/`right_png_b64`/
`intrinsics_flat`/`baseline_m` request, `depth_npy_b64` response) is what
`depth_camera.DepthCameraClient.compute_depth()` (Plan 12-06 Task 1) sends
and parses -- keep them in sync if either side changes.

### Step 9 — Colab: open a second tunnel

```python
# Colab notebook cell
from pyngrok import ngrok

# A DIFFERENT local port than Step 3's gRPC server (8080), and a plain HTTP
# tunnel type -- unlike Step 4's gRPC tunnel, which specifically needs "tcp"
# for HTTP/2 semantics, this is a plain JSON-over-HTTP POST endpoint, so
# ngrok's default HTTP tunnel type works fine here.
depth_tunnel = ngrok.connect(3000, "http")
print(f"Depth endpoint reachable at: {depth_tunnel.public_url}/depth")
```

Pass the printed `.../depth` URL as `depth_camera.py`'s `--endpoint` argument
(Plan 12-06 Task 1) or `run_vla_episode.py`'s `--depth-endpoint` flag
(Task 3).

### Teardown (extends Step 6 above)

Every session, no exceptions, ALSO tear down this second tunnel and stop the
Flask server process -- same rationale as the existing gRPC tunnel's own
teardown discipline (an unattended open tunnel into a Colab-hosted endpoint
is an elevation-of-privilege risk):

```python
# Colab notebook cell, at the end of the session
ngrok.disconnect(depth_tunnel.public_url)
# ngrok.kill() already stops all tunnels including this one, if not already called above.
```

Then interrupt/stop the Flask `app.run(...)` cell (Colab: click the cell's
stop button, or restart the runtime) so the depth server process itself is
not left listening after the tunnel is torn down.

---

## Summary

| Item             | Value                                                                                                                            |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Checkpoint       | `victorvanhalst/smolvla_so101_cube`                                                                                            |
| Task instruction | "Pick the red cube and place it in the bowl"                                                                                     |
| Tunnel type      | TCP (`ngrok.connect(8080, "tcp")`) — not HTTP                                                                                 |
| Camera mapping   | `camera1`=wrist, `camera2`=AR0144 stereo-left, `camera3`=AR0144 stereo-right                                               |
| Fallback tunnel  | `cloudflared` + configured Zero Trust tunnel (heavier setup)                                                                   |
| Fallback policy  | ACT or pi0/pi05 (`lerobot.policies`, already installed) if this checkpoint underperforms                                       |
| Depth endpoint   | Fast-FoundationStereo (NVlabs/Fast-FoundationStereo), served via Flask, HTTP tunnel (separate from the gRPC PolicyServer tunnel) |
