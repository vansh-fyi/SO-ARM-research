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
- AR0144 stereo camera, cv2 index 1, native 2560x720 — **one physical
  side-by-side stereo frame, not two independent feeds** (confirmed in
  `diagnostics/UAT/function/basic/UAT.md`, Step 4)

Resolution: split the AR0144's single 2560x720 frame into left/right
halves, giving **3 genuinely distinct real camera views** (not a dummy or
duplicated 3rd slot):

| Checkpoint camera key | Real source | Crop |
|---|---|---|
| `camera1` | IMX335 wrist | as-is, full 1920x1080 frame |
| `camera2` | AR0144 stereo — LEFT half | columns `[0:1280]` of the 2560x720 frame |
| `camera3` | AR0144 stereo — RIGHT half | columns `[1280:2560]` of the 2560x720 frame |

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
   the `RobotClient`'s local process, capture the full 2560x720 frame per
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

config = PolicyServerConfig(host="0.0.0.0", port=8080)
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
tunnel = ngrok.connect(8080, "tcp")
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
    --robot.cameras="{ camera1: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}, camera2: {type: <split-stereo-left>, index_or_path: 1, width: 1280, height: 720, fps: 30}, camera3: {type: <split-stereo-right>, index_or_path: 1, width: 1280, height: 720, fps: 30} }" \
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

## Summary

| Item | Value |
|---|---|
| Checkpoint | `victorvanhalst/smolvla_so101_cube` |
| Task instruction | "Pick the red cube and place it in the bowl" |
| Tunnel type | TCP (`ngrok.connect(8080, "tcp")`) — not HTTP |
| Camera mapping | `camera1`=wrist, `camera2`=AR0144 stereo-left, `camera3`=AR0144 stereo-right |
| Fallback tunnel | `cloudflared` + configured Zero Trust tunnel (heavier setup) |
| Fallback policy | ACT or pi0/pi05 (`lerobot.policies`, already installed) if this checkpoint underperforms |
