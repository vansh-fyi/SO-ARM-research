# Phase 11: VLA Hardware Connection - Research

**Researched:** 2026-09-20
**Domain:** Real-hardware VLA inference (LeRobot SmolVLA), remote-GPU inference bridging, robot safety validation, structured I/O logging
**Confidence:** HIGH (bridge architecture, action-unit contract, joint limits — all verified against installed source/live calibration); MEDIUM (SmolVLA checkpoint choice, exact safety thresholds — best-available evidence, flagged for human verification)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** VLA inference runs on **Colab GPU**, not the local laptop. The local `control/` process (Python 3.12 venv, real hardware bridge) acts as a relay: it sends observations (camera frames + joint state) to the Colab-hosted SmolVLA and receives back the action to execute. This mirrors the compute split already used for the sim VLA phases (Phase 3), but is new plumbing for the *real-hardware* path — Phase 11 planning/research must design this local↔Colab bridge (protocol, latency handling, failure mode if the bridge drops mid-episode) since nothing like it exists yet in `control/`.
- **D-02:** The first full observed run uses a **small red cube** (already on hand) as the manipulation object, not the paper-aligned "Pen Transfer" task. Priority is proving the real-hardware pipeline works end-to-end over paper-benchmark alignment for this first phase. Verify the cube fits the existing embodiment constraints (≤84mm, within ~0.45m reach, not on the base's forward centerline) before finalizing the exact task prompt/placement during planning.
- **D-03:** E-stop is a **software keyboard interrupt** (a dedicated key, or a gracefully-handled Ctrl+C) that immediately halts motor commands — no new physical hardware. Matches the existing pattern in `control/keyboard_joint_control.py`.
- **D-04:** I/O log format is **JSON Lines** — one JSON object per inference step: input references (per-camera timestamped frame paths, joint state, instruction), raw model output, validated action, executed action, latency, model version. Camera frames are saved as image/video files on disk, referenced by path (not embedded) — consistent with `control/record_episode.py`'s existing frame-storage convention. Chosen so a future phase can append a `reasoning_trace` field without breaking the schema.
- **D-05:** SmolVLA has no reasoning trace — VLAHW-03's "complete I/O" framing (not "reasoning trace") is intentional; do not propose fake reasoning-trace capture for this phase.

### Claude's Discretion

- Concrete safety-validator thresholds (numeric limits) — derive from servo specs during planning/research.
- Exact local↔Colab bridge protocol/transport for the observation/action relay.
- Exact cube placement/prompt wording for the first task, so long as it respects the embodiment reach/collision constraints already documented in STATE.md.

### Deferred Ideas (OUT OF SCOPE)

- Deep-reasoning MLLM comparison experiment (e.g. Claude paid tier, or an HF-hosted reasoning model on Colab GPU), reusing the same JSON Lines I/O log with an added `reasoning_trace` field — this is the next phase after Phase 11. Do not fold into Phase 11.
- Fine-tuning any policy on Phase-11-collected data (explicit v2.0 Out of Scope item in REQUIREMENTS.md) — Phase 11 evaluates an already-fine-tuned checkpoint via inference, it does not train one.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| VLAHW-01 | Load an SO-101-native joint-action VLA (SmolVLA); produce valid 6D joint-position actions from real wrist+overhead camera + joint-state observations; document action contract (units, joint order, gripper scale, absolute/relative) before first use | Action contract fully resolved below (Pitfall 1 + Pattern 2): `use_degrees=True` is the real, live default for the 5 arm joints (DEGREES mode), gripper is always `RANGE_0_100`; joint order is `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` (6 DOF); absolute (not delta) joint-position targets. SmolVLA checkpoint candidates + the zero-shot-crash pitfall documented in Standard Stack / Pitfall 2. |
| VLAHW-02 | VLA output passes a safety validator (joint limits, max displacement/velocity, gripper bounds, stale/malformed/NaN rejection, e-stop, comms-failure handling) before reaching the real robot via `SO101Follower`; unit ambiguity resolved first | Concrete joint-limit numbers (degrees) derived from this project's own live calibration (Pattern 3). `max_relative_target` / `ensure_safe_goal_position` already exists in installed LeRobot and covers per-step displacement (Don't Hand-Roll). E-stop and comms-retry patterns already exist in `control/keyboard_joint_control.py` to extend. |
| VLAHW-03 | Every inference step's complete I/O captured to a durable JSON Lines log (per-camera timestamped frames referenced by path, joint state, instruction, raw output, validated action, executed action, latency, model version) | JSONL schema proposed in Code Examples, built on `record_episode.py`'s existing per-camera timestamped-capture convention. |
| VLAHW-04 | At least one full episode recorded end-to-end (video + I/O log + termination reason + success/failure) | Architecture Pattern 1 (bridge) + Pattern 4 (episode harness) cover the control-loop shape; Validation Architecture section covers how to smoke-test the harness before the live run. |
| VLAHW-05 | Findings write-up (including whether lerobot#2210 reproduces) informs go/no-go on later phases | lerobot#2210 root-caused below (Pitfall 2) — it is a zero-shot-checkpoint normalization-stats bug, not an SO-101-specific hardware incompatibility; a fine-tuned checkpoint sidesteps it entirely. Fallback VLA candidates identified (Standard Stack / State of the Art) in case the chosen checkpoint still underperforms on this project's specific rig. |
</phase_requirements>

## Summary

The real-hardware bridge Phase 11 needs does not have to be hand-rolled: the installed `lerobot==0.6.1` package already ships an official client/server split-machine inference architecture (`lerobot.async_inference`, gRPC-based `PolicyServer` + `RobotClient`), and `so101_follower` is explicitly in its `SUPPORTED_ROBOTS` list. This directly satisfies D-01's "Colab GPU inference, local relay" requirement — the `PolicyServer` runs on Colab, the `RobotClient` runs locally in `control/` and drives the real `SO101Follower`, connected over a tunnel (ngrok TCP, since gRPC needs raw TCP, not HTTP) because Colab has no public inbound port. The one gap: `grpcio` is not currently installed in `control/.venv` (`pip install 'lerobot[async]'` is required — verified by import failure this session).

The second major finding resolves the CONTEXT.md-flagged unit ambiguity directly from source, not from docs: `SOFollowerConfig.use_degrees` defaults to `True`, is never overridden anywhere in `control/`, and the installed `so_follower.py` shows the DEGREES branch is what actually executes for the 5 arm joints (gripper is hardcoded `RANGE_0_100` regardless). `control/keyboard_joint_control.py`'s inline comment claiming "RANGE_M100_100 joints" is stale/wrong documentation, not a real config divergence — this session's own UAT log (`diagnostics/UAT/function/basic/UAT.md`, "Known issues") independently confirms `use_degrees=True` was the actual operative mode when a `drive_mode` patch had "zero effect" on the leader arm. This is HIGH-confidence and should be locked into the plan without further ambiguity.

The third major finding changes what "load SmolVLA" means in practice: `lerobot/smolvla_base` (the zero-shot pretrained checkpoint) **cannot run inference as-is** — it has no baked-in dataset normalization statistics, and loading it raw crashes with `AssertionError: mean is infinity` (this is exactly what lerobot#2210 reports). SmolVLA's own docs confirm it is "a base model, fine-tuning on your own data is required." Since this milestone explicitly excludes fine-tuning on Phase-11-collected data, the practical path is to load an **already community-fine-tuned SO-101 SmolVLA checkpoint** from the HF Hub (several exist, including one trained on exactly a "red cube pick-and-place" task) rather than either training a new one or trying to force smolvla_base to run zero-shot.

**Primary recommendation:** Use `lerobot.async_inference.PolicyServer`/`RobotClient` (gRPC, tunneled from Colab via ngrok TCP) as the bridge; load a pre-fine-tuned SO-101 SmolVLA checkpoint from the HF Hub (not `smolvla_base` zero-shot); resolve the action contract as DEGREES (arm) + RANGE_0_100 (gripper), 6-DOF, absolute targets; layer a custom software safety validator on top of LeRobot's existing `max_relative_target` clamp; log every step as one JSON Lines record with camera frames as file paths.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| VLA policy inference (forward pass) | Cloud Inference (Colab GPU, `PolicyServer`) | — | 450M-param model; local CPU/Apple Silicon inference deemed too slow/uncertain per D-01 |
| Observation acquisition (camera frames, joint state) | Local Relay (`control/`, `RobotClient`) | Hardware I/O (`SO101Follower.get_observation()`) | Cameras and servo bus are only reachable from the machine physically wired to the robot |
| Action safety validation | Local Relay (`control/`) | — | Must gate every action before it reaches the servo bus; cannot depend on network round-trip to Colab being trustworthy or low-latency |
| Action execution (servo writes) | Hardware I/O (`SO101Follower.send_action()`) | Local Relay | LeRobot's motor-bus interface is the only sanctioned path to the servos (no new hand-written serial code) |
| E-stop | Local Relay (keyboard interrupt handler) | Hardware I/O (torque disable on disconnect) | Must work even if the Colab connection is dead — cannot depend on the remote server |
| I/O log persistence (JSONL + frames) | Local Relay (`control/`) | Durable Storage (local disk) | Log must survive a bridge drop; writing locally, not to a remote store, is the safe default |
| Bridge transport (tunnel) | Network (ngrok/cloudflared) | — | Colab has no public inbound port; a reverse tunnel is mandatory, not optional |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `lerobot[feetech]` | 0.6.1 (already installed, pinned in `control/requirements.txt`) | `SO101Follower` motor-bus interface, calibration, camera abstraction | Already the sanctioned, locked-in interface for this project (ROADMAP.md: "no new hand-written serial/register code") [VERIFIED: control/.venv install, `pip show lerobot` → 0.6.1] |
| `lerobot[async]` extra (adds `grpcio`) | matches installed lerobot 0.6.1; `grpcio` latest on PyPI is 1.84.0 | `lerobot.async_inference.PolicyServer` / `RobotClient` — the officially supported remote-GPU inference bridge | Ships in the same package already in use; `so101_follower` is explicitly in its `SUPPORTED_ROBOTS`, `smolvla` in `SUPPORTED_POLICIES` [VERIFIED: read `control/.venv/.../lerobot/async_inference/constants.py` directly] |
| SmolVLA policy (fine-tuned SO-101 checkpoint, HF Hub) | checkpoint-specific (see below) | The VLA itself | Matches the benchmark paper's tested architecture; SO-101-native joint-action, avoids the Cartesian/IK detour (per ROADMAP.md D-01 framing) [CITED: huggingface.co/docs/lerobot/smolvla] |
| `grpcio` | 1.84.0 (PyPI, current) | Transport for `async_inference` (gRPC under the hood) | Required dependency of `lerobot[async]`; not installed yet in `control/.venv` — import fails today [VERIFIED: `python -c "import grpc"` → ModuleNotFoundError in `control/.venv`, this session] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pyngrok` (or the plain `ngrok` CLI + a Colab-authenticated authtoken) | 8.1.2 (PyPI, current) | Open a TCP tunnel from the Colab-hosted `PolicyServer`'s port to a public endpoint the local `RobotClient` can reach | gRPC needs a raw TCP tunnel (HTTP-only tunnels break HTTP/2 framing) — ngrok's TCP tunnel type is the documented pattern for exposing a Colab port [CITED: ngrok.com/docs/using-ngrok-with/googleColab] |
| `cloudflared` | already installed locally (`/opt/homebrew/bin/cloudflared`) | Alternative tunnel | Only recommended if a Cloudflare Zero Trust tunnel is already configured for raw TCP forwarding — cloudflared's zero-setup "quick tunnels" are HTTP-only, so this is more setup than ngrok TCP for an ad hoc research session [VERIFIED: `command -v cloudflared` succeeds locally, this session] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `lerobot.async_inference` gRPC bridge | Hand-rolled HTTP/websocket server (Flask/FastAPI on Colab + `requests`/`websockets` client locally) | More code to write and maintain, duplicates functionality LeRobot already ships and tests; only justified if `async_inference` proves broken/unavailable for this lerobot version |
| A community-fine-tuned SmolVLA checkpoint | `lerobot/smolvla_base` zero-shot | **Does not work** — crashes with an infinite-mean normalization assertion (see Pitfall 2); this is not a viable option, not just a worse one |
| SmolVLA | ACT (`lerobot.policies.act`) as fallback | ACT is smaller/simpler and SO-101-native, but is not itself a *VLA* (weaker language conditioning in the standard LeRobot formulation) — acceptable as a last-resort fallback per VLAHW-05's own "identify a fallback" ask, not as a first choice |
| SmolVLA | `pi0`/`pi05` (already in `lerobot.policies`, `SUPPORTED_POLICIES`) as fallback | Larger (pi0 ≈14GB at inference vs. SmolVLA's ≈2GB per LeRobot's own async-inference tuning guide), heavier Colab GPU/memory requirement, but is a genuine VLA and already installed in this repo's `control/.venv` — no new package needed if this fallback is invoked |

**Installation (to run on `control/.venv`, one-time):**
```bash
cd control
source .venv/bin/activate
pip install 'lerobot[async]'   # adds grpcio + lerobot.transport/async_inference deps
pip install pyngrok            # only if using ngrok's Python API instead of the CLI binary directly
```

**Version verification:** `lerobot` is already pinned at `0.6.1` in `control/requirements.txt` and installed — do not bump it mid-phase without re-verifying `async_inference`'s CLI flags still match (LeRobot's CLI surface has changed between versions per this project's own UAT notes). `grpcio` 1.84.0 and `pyngrok` 8.1.2 were confirmed current on PyPI this session.

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|--------------|---------|-------------|
| `grpcio` | PyPI | long-established (Google gRPC project; PyPI metadata shows a "too-new" flag on the *latest point release* only) | unknown to the audit tool (no download-count feed) | `https://grpc.io` | SUS (tool signal only — no download/age data available to it) | **Approved** — this is the official Google gRPC Python binding, already a required transitive dependency of `lerobot[async]`'s own advertised extra, not an independent choice; the SUS verdict here reflects the audit tool's metadata gaps, not a real risk signal |
| `pyngrok` | PyPI | long version history (0.1.2 → 8.1.2, 2018–present) | unknown to the audit tool (no download-count feed) | none listed in registry metadata | SUS | **Flagged — planner should add a `checkpoint:human-verify` before this install**, purely because the audit tool has no repo/download signal to confirm against; well-known package in practice (documented directly on ngrok's own site), but verify before installing per protocol |
| `lerobot` | PyPI (installed via `lerobot[feetech]==0.6.1`) | already installed and in production use in this repo since Phase-11-predecessor work | n/a (already vetted, pre-existing dependency) | `github.com/huggingface/lerobot` | OK | Approved — no new audit needed, already in `control/requirements.txt` |

**Packages removed due to [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** `grpcio` (approved anyway — required transitive dep of an in-use library's own extra, not independently sourced), `pyngrok` (flagged for a checkpoint before install).

**Community HF Hub model checkpoints (not pip packages, different risk profile):** SmolVLA checkpoints fine-tuned by third-party HF Hub users (e.g. `victorvanhalst/smolvla_so101_cube`) are *model weights*, not code, and LeRobot's standard checkpoint format is `safetensors` (not raw pickle), which avoids arbitrary-code-execution risk on load. Still, prefer checkpoints with a populated model card, a linked training dataset, and recent activity over anonymous/undocumented ones — verify the specific checkpoint's `config.json` (camera keys/count, action dim) before wiring it into the harness, since these are not held to the same audit process as pip packages.

## Architecture Patterns

### System Architecture Diagram

```
                    ┌─────────────────────────────────────────┐
                    │           Colab GPU instance             │
                    │                                           │
                    │  ┌─────────────────────────────────┐      │
                    │  │  PolicyServer (lerobot.async_    │      │
                    │  │  inference.policy_server)        │      │
                    │  │  - loads fine-tuned SmolVLA ckpt  │      │
                    │  │  - listens on 127.0.0.1:PORT      │      │
                    │  └───────────────┬───────────────────┘      │
                    │                  │ gRPC                     │
                    │  ┌───────────────┴───────────────────┐      │
                    │  │  ngrok agent (TCP tunnel)          │      │
                    │  │  exposes PORT -> public host:port  │      │
                    │  └───────────────┬───────────────────┘      │
                    └──────────────────┼───────────────────────────┘
                                       │  (internet, tunneled TCP/gRPC)
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │  Local laptop (control/, Python 3.12 venv)                  │
        │                              │                              │
        │  ┌───────────────────────────┴───────────────────────┐      │
        │  │  RobotClient (lerobot.async_inference.robot_client)│      │
        │  │  --server_address=<ngrok host:port>                │      │
        │  └───────────────┬─────────────────────────┬──────────┘      │
        │                  │ observations              │ action chunks │
        │                  ▼                            ▼              │
        │  ┌────────────────────────┐    ┌──────────────────────────┐  │
        │  │ Observation capture:    │    │ Safety Validator          │  │
        │  │ - wrist cam (cv2 idx 0) │    │ - joint-limit clamp       │  │
        │  │ - overhead cam (idx 1)  │    │ - NaN/inf/malformed reject│  │
        │  │ - SO101Follower.        │    │ - stale-obs/action reject │  │
        │  │   get_observation()     │    │ - max_relative_target     │  │
        │  └────────────────────────┘    │   (LeRobot built-in)      │  │
        │                                 └─────────────┬──────────────┘  │
        │                                                ▼                 │
        │                                  ┌──────────────────────────┐    │
        │                                  │ SO101Follower.send_action │    │
        │                                  │ (existing motor-bus API)  │    │
        │                                  └─────────────┬──────────────┘    │
        │                                                │                 │
        │                                                ▼                 │
        │                                     ┌────────────────────┐       │
        │                                     │  Real SO-ARM101     │       │
        │                                     │  (servo bus)        │       │
        │                                     └────────────────────┘       │
        │                                                                  │
        │  ┌────────────────────────────────────────────────────────┐      │
        │  │ JSON Lines I/O logger (every inference step)            │      │
        │  │ - frame paths, joint state, instruction, raw/validated/  │      │
        │  │   executed action, latency, model version                │      │
        │  └────────────────────────────────────────────────────────┘      │
        │                                                                  │
        │  Keyboard listener (e-stop): interrupts control loop,            │
        │  halts motor commands, independent of network/Colab state        │
        └──────────────────────────────────────────────────────────────────┘
```

A reader can trace the primary loop: camera + joint state captured locally → streamed over the tunneled gRPC connection to the Colab `PolicyServer` → SmolVLA forward pass → action chunk streamed back → **safety validator gates it locally** (this step must never be skipped, and must not depend on the network being up) → `SO101Follower.send_action()` → real servos move → every step logged to JSONL regardless of outcome.

### Recommended Project Structure

```
control/
├── vla_bridge/
│   ├── policy_server_launch.md   # Colab notebook cell snippets / instructions (not a local script — runs on Colab)
│   ├── robot_client.py           # thin wrapper around lerobot.async_inference.RobotClient + safety validator hook
│   ├── safety_validator.py       # joint limits, displacement/velocity caps, NaN/stale rejection, e-stop check
│   └── io_logger.py              # JSON Lines writer, reuses record_episode.py's per-camera timestamped capture
├── run_vla_episode.py            # top-level entrypoint: wires robot_client + safety_validator + io_logger + e-stop
├── keyboard_joint_control.py     # existing — e-stop pattern reused, not replaced
├── record_episode.py             # existing — camera capture pattern reused
└── outputs/
    └── vla_episode_NNN/
        ├── camera_wrist/*.png    # per-frame stills, timestamped filenames
        ├── camera_overhead/*.png
        ├── episode.jsonl         # one record per inference step
        └── episode.mp4           # synced video (per VLAHW-04)
```

### Pattern 1: Colab-hosted `PolicyServer` + local `RobotClient` (the D-01 bridge)

**What:** LeRobot's own gRPC-based split-machine inference architecture. `PolicyServer` runs headless on Colab holding the model; `RobotClient` runs locally, owns the physical `Robot` object, and streams observations/receives action chunks.
**When to use:** Any time inference must run on a GPU the robot-controlling machine doesn't have — exactly D-01's scenario.
**Example (Colab side, one notebook cell):**
```python
# Source: huggingface.co/docs/lerobot/async (verified against installed lerobot/async_inference/configs.py)
from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import serve

config = PolicyServerConfig(host="0.0.0.0", port=8080)
serve(config)
```
**Example (local `control/` side):**
```bash
# Source: huggingface.co/docs/lerobot/async — CLI form
python -m lerobot.async_inference.robot_client \
    --server_address=<ngrok_tcp_host>:<ngrok_tcp_port> \
    --robot.type=so101_follower \
    --robot.port=/dev/cu.usbmodem5B8E1139151 \
    --robot.id=soarm_follower_02 \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}, overhead: {type: opencv, index_or_path: 1, width: 2560, height: 720, fps: 30}}" \
    --task="Pick up the red cube" \
    --policy_type=smolvla \
    --pretrained_name_or_path=<HF_hub_checkpoint> \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5
```
Note: camera keys passed here (`wrist`, `overhead`) must match the keys the checkpoint's `config.json` expects — verify per-checkpoint, do not assume.

### Pattern 2: Action-unit contract resolution (VLAHW-01/02's flagged ambiguity)

**What:** `SOFollowerConfig.use_degrees` defaults to `True`; it is never overridden in any `control/` script. This means:
- The 5 arm joints (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`) are normalized in **DEGREES** (`MotorNormMode.DEGREES`), symmetric around each joint's calibrated midpoint.
- `gripper` is **always** `MotorNormMode.RANGE_0_100` regardless of `use_degrees` — it is hardcoded separately in `so_follower.py`'s constructor (`Motor(6, "sts3215", MotorNormMode.RANGE_0_100)`).
- Actions are **absolute** joint-position targets (`{joint}.pos`), not deltas — `send_action()` takes a target position dict, not a relative delta (deltas are only computed internally, for the `max_relative_target` safety clamp).
- Joint order (dict insertion order in `SOFollower.__init__`, which `_motors_ft`/`action_features` iterate): `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` — 6 total, matching VLAHW-01's "6D joint-position actions."

**Why the ambiguity existed:** `control/keyboard_joint_control.py`'s inline comment (`# RANGE_M100_100 joints (see so_follower.py Motor() norm_mode) - unclamped ...`) is **stale/incorrect** — it describes a mode the code does not actually use. This project's own hardware UAT log independently corroborates the real behavior: `diagnostics/UAT/function/basic/UAT.md`'s "Known issues" section states a `drive_mode=1` patch "had zero effect on the leader's arm joints specifically because `so101_leader`/`so101_follower` both default to `use_degrees=True`, and the DEGREES branch ... never applies `drive_mode`" — this was diagnosed by directly reading `motors_bus.py`, the same source read again this session.
**Action for the plan:** Fix the stale comment in `keyboard_joint_control.py` as a low-risk cleanup; document the DEGREES + RANGE_0_100 contract explicitly wherever the VLA harness converts model output → `send_action()` input. If a chosen SmolVLA checkpoint's dataset was recorded with a **different** `use_degrees` setting (verify via the checkpoint's dataset `info.json`/stats on the Hub), a unit-conversion step is required before sending actions — do not assume every community checkpoint matches this project's convention.

### Pattern 3: Real, calibration-derived joint limits (for the safety validator)

Derived by this project's own `scripts/calibration_utils.py` from `soarm_follower_02`'s live LeRobot calibration file, using LeRobot's own tick→degree formula (already verified and reused by Phase 10's URDF work) — reproduced here in degrees since the safety validator operates on the real robot's native DEGREES-mode action space, not the URDF's radians:

| Joint | Calibrated range (degrees, symmetric about home) | Source |
|-------|---------------------------------------------------|--------|
| `shoulder_pan` | ±70.33° | `scripts/calibration_utils.py` derivation, reused verbatim from Phase 10 (`robot.xml`/URDF: ±1.227484 rad) |
| `shoulder_lift` | ±107.47° | same |
| `elbow_flex` | ±97.23° | same |
| `wrist_flex` | ±102.02° | same |
| `wrist_roll` | −157.21° to +162.79° (asymmetric — mechanical-limit value, calibration range is a full-turn placeholder per Phase 10 D-06) | URDF (±2.744/+2.841 rad); **do not** derive this one from calibration ticks, use this fixed value |
| `gripper` | 0–100% (`RANGE_0_100`), maps to 0–0.036 m per jaw physically (project-approved 36 mm-per-jaw cap, ~≤84mm two-jaw aperture; real measured aperture is 6–78mm per STATE.md) | `soarm_gripper.xml` (Phase 10 established model) |

**Confidence:** HIGH for the 4 arm joints and gripper (same in-repo, already-verified derivation reused, not re-guessed). MEDIUM for `wrist_roll` (a known mechanical-limit placeholder, not a fresh calibration measurement — treat as a soft/conservative bound). These are simulation/URDF-side values; they are a very strong proxy for the real servo's degree-mode range because they use the exact same formula LeRobot's own `MotorsBus._normalize()` applies to the real robot's live calibration — but re-verify against the live `soarm_follower_02.json` calibration file at execution time (its `range_min`/`range_max` per joint) before hardcoding, since calibration can drift after any physical recalibration event.

### Pattern 4: JSON Lines I/O log schema (VLAHW-03/D-04)

```json
{
  "step": 42,
  "timestamp_utc": "2026-09-21T02:14:33.812Z",
  "instruction": "Pick up the red cube",
  "camera_frames": {
    "wrist": {"path": "camera_wrist/000042.png", "captured_at_utc": "2026-09-21T02:14:33.801Z"},
    "overhead": {"path": "camera_overhead/000042.png", "captured_at_utc": "2026-09-21T02:14:33.799Z"}
  },
  "joint_state": {"shoulder_pan": 12.4, "shoulder_lift": -30.1, "elbow_flex": 45.2, "wrist_flex": 5.0, "wrist_roll": 0.0, "gripper": 62.0},
  "raw_model_output": {"shoulder_pan": 13.1, "shoulder_lift": -29.0, "elbow_flex": 44.8, "wrist_flex": 4.7, "wrist_roll": 0.1, "gripper": 40.0},
  "validated_action": {"shoulder_pan": 13.1, "shoulder_lift": -29.0, "elbow_flex": 44.8, "wrist_flex": 4.7, "wrist_roll": 0.1, "gripper": 40.0},
  "validator_flags": [],
  "executed_action": {"shoulder_pan": 13.1, "shoulder_lift": -29.0, "elbow_flex": 44.8, "wrist_flex": 4.7, "wrist_roll": 0.1, "gripper": 40.0},
  "latency_ms": {"observation_to_server": 48, "server_inference": 61, "action_to_execution": 12},
  "model_version": "victorvanhalst/smolvla_so101_cube@<commit_sha>"
}
```
Per-camera timestamps are captured independently (not a single shared timestamp) so drift between the wrist and overhead feeds is visible in the log itself, per VLAHW-03's explicit requirement.

### Anti-Patterns to Avoid

- **Trusting the network for safety:** Never let the safety validator live on the Colab side or depend on the tunnel being up — if the bridge drops mid-action, the local `RobotClient`/validator must fail safe (hold position or return to a known-safe pose), not wait indefinitely for a response.
- **Assuming `smolvla_base` works zero-shot:** It does not (see Pitfall 2) — always load a checkpoint with baked-in dataset statistics.
- **Reusing the sim OpenVLA-OFT 7D Cartesian-delta pipeline:** Already correctly avoided per ROADMAP.md's own framing — flagging again here since it's the most likely accidental copy-paste source of a wrong action-space assumption.
- **Hand-writing a new HTTP relay when `async_inference` already exists:** Would violate the "no new hand-written serial/register code" spirit of VLAHW-02 by re-inventing an already-shipped, already-tested bridge.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Remote-GPU inference transport | Custom Flask/FastAPI + requests HTTP relay | `lerobot.async_inference.PolicyServer`/`RobotClient` (gRPC) | Already installed (needs one extra), already tested by HuggingFace, already supports `so101_follower` and `smolvla` natively — reinventing this adds risk with zero benefit |
| Per-step displacement clamping | Custom "clip action to ± N degrees of current position" logic | `SOFollowerRobotConfig(max_relative_target=...)` + `ensure_safe_goal_position()` (already invoked inside `SOFollower.send_action()`) | This exact safety mechanism already exists in the installed library and is applied automatically once the config field is set — verified by reading `so_follower.py`'s `send_action()` |
| Comms-failure retry on reads | Custom retry loop from scratch | `read_positions()`'s existing retry/backoff pattern in `control/keyboard_joint_control.py`, or `SOFollowerConfig.num_read_retries` (built into `sync_read`) | Two working, already-tested layers exist; extend, don't duplicate |
| E-stop | New signal handler / hardware GPIO | Extend `control/keyboard_joint_control.py`'s existing keyboard-interrupt + "return to start position" pattern | D-03 explicitly locks this; the pattern already works and is UAT-verified |
| Video/frame capture with timestamps | New `cv2.VideoCapture` wrapper | `control/record_episode.py`'s `open_cameras()` + per-camera `cv2.VideoWriter`/still-capture pattern | Already proven working on this exact hardware (IMX335 wrist @ index 0, AR0144 stereo @ index 1) |

**Key insight:** Nearly every piece of infrastructure Phase 11 needs (bridge, displacement clamp, comms retry, e-stop, camera capture) already exists somewhere in this repo or in the already-installed LeRobot package. The actual net-new work is the glue (safety validator's joint-limit/NaN/stale checks, the JSONL logger, and the tunnel setup) — treat anything that looks like it needs "a new serial/control-loop primitive" as a signal to re-check for an existing pattern first.

## Common Pitfalls

### Pitfall 1: The `use_degrees` / normalized-percent ambiguity (resolved above, restated as a pitfall)

**What goes wrong:** Code or documentation assumes arm-joint values are in a `-100..100` normalized range (matching the gripper's convention) when they are actually in degrees.
**Why it happens:** `SOFollowerConfig.use_degrees: bool = True` is a non-obvious default, and at least one comment in this very codebase (`keyboard_joint_control.py`) describes the wrong mode.
**How to avoid:** Treat Pattern 2 above as authoritative; never infer the unit from a comment — read `so_follower.py`'s constructor and `SOFollowerConfig`'s dataclass default directly if in doubt.
**Warning signs:** An action that "looks like a percent" (e.g. `45.0`) being sent to `shoulder_pan` would actually command 45 *degrees* off the calibrated midpoint — for `shoulder_pan` (±70.33° range) this is a large, plausible-looking-but-wrong move that would not obviously fail until it neared a limit.

### Pitfall 2: `lerobot/smolvla_base` crashes zero-shot; this is lerobot#2210, not an SO-101 incompatibility

**What goes wrong:** Loading `smolvla_base` and calling it on real observations raises `AssertionError: mean is infinity. You should either initialize with stats as an argument, or use a pretrained model.`
**Why it happens:** SmolVLA's normalization layer (`MEAN_STD` mode for state/action) needs dataset-specific mean/std statistics; the raw `smolvla_base` checkpoint is explicitly a base model with no such statistics baked in for an arbitrary new robot — it is documented as requiring fine-tuning before use, not a drop-in zero-shot policy. lerobot#2210 (filed against lerobot 0.3.3, still without a documented fix as of this session) is this exact failure mode.
**How to avoid:** Do not use `lerobot/smolvla_base` directly. Load an already-fine-tuned SO-101 checkpoint from the HF Hub instead (candidates below) — these bundle the dataset statistics needed for normalization to work.
**Warning signs:** Any `AssertionError` mentioning "mean is infinity" or normalization stats during policy load is this exact bug, not a new one — do not spend time debugging it as if it were novel; swap checkpoints instead.

**Candidate fine-tuned checkpoints (verify each one's `config.json` for camera count/keys and task-match before committing):**
| Checkpoint | Task | Cameras (per model card) | Fit for D-02 (red cube) |
|---|---|---|---|
| `victorvanhalst/smolvla_so101_cube` [ASSUMED — HF Hub listing, not independently downloaded/inspected this session] | "Pick the red cube and place it in the bowl" | 3 (`camera1/2/3`, wrist + external) | Best task-semantic match; camera count (3) exceeds this project's 2-camera rig — needs either a 3rd (even dummy/`empty_cameras`) feed or picking a different checkpoint |
| `majinwakeup30/smolvla_so101_stack_cube_v3_2_cameras` [ASSUMED — not independently verified this session, name suggests 2-camera config] | cube stacking | 2 (per naming) | Worth checking first at execution time — camera count may match this project's actual 2-camera rig better than the above |
| `lerobot/svla_so100_pickplace` [ASSUMED] | pink lego brick into a box | not confirmed this session (fetch attempt blocked by an auth wall) | Generic pick-place, SO-100 (compatible hardware family, same `SOFollower` class) |
| `cn0303/smolvla-so101-strawberry-v3` [ASSUMED] | strawberry-related pick task | not verified | Lower task-match, listed for completeness |

All four are `[ASSUMED]` — discovered via WebSearch, not independently downloaded/inspected. The plan must add a `checkpoint:human-verify` (or at minimum a scripted `config.json` check) before committing to one, per this project's own camera setup (IMX335 wrist @ 1920x1080, AR0144 stereo @ 2560x720 — note the AR0144 is a *stereo pair in one frame*, not two independent cameras; confirm whether the chosen checkpoint expects it split or whole).

### Pitfall 3: `wrist_roll`'s calibration range is a full-turn placeholder, not a real limit

**What goes wrong:** Naively deriving `wrist_roll`'s safety-validator bound from live calibration ticks (the same way the other 4 arm joints are derived) produces a meaningless ~full-circle range.
**Why it happens:** `lerobot-calibrate` hardcodes `wrist_roll`'s calibration range to `0-4095` (a full-turn placeholder) because the joint spins continuously past a mechanical stop check and is never actually measured through a real range of motion — already documented and worked around in Phase 10 (`scripts/calibration_utils.py`'s `JOINTS_FROM_CALIBRATION` explicitly excludes it).
**How to avoid:** Use the URDF's existing mechanical-limit-derived value (−157.21° to +162.79°) for `wrist_roll` in the safety validator, not a fresh calibration-tick derivation.
**Warning signs:** A derived `wrist_roll` range close to ±180° (a full turn) is the tell that the formula was misapplied to the placeholder data — this exact regression-test check already exists in `scripts/test_calibration_utils.py::test_full_range_is_full_turn_placeholder`.

### Pitfall 4: Gripper units are percent, not meters, in the live control path

**What goes wrong:** Conflating the gripper's real-hardware control units (`RANGE_0_100`, 0–100%) with the URDF/MuJoCo simulation's gripper units (prismatic joint, meters, 0–0.036 m per jaw).
**Why it happens:** Phase 10 (sim-side) and Phase 11 (real-hardware) both touch "the gripper," but at genuinely different points in the stack — the real robot's `send_action()`/`get_observation()` never sees meters, only the 0–100 percent-open value LeRobot's `RANGE_0_100` norm mode produces.
**How to avoid:** The safety validator's gripper bound must be expressed in 0–100 (percent-open), not meters. Do not try to reuse Phase 10's `0.036` value directly for the real-hardware validator.
**Warning signs:** A gripper safety check that rejects any value >1 (as if it were meters) would incorrectly reject all normal 0–100 gripper commands.

### Pitfall 5: gRPC over an HTTP-only tunnel silently breaks

**What goes wrong:** Using an HTTP(S)-only tunnel type (e.g. a cloudflared "quick tunnel," or ngrok's default HTTP tunnel) to expose the Colab `PolicyServer`'s port results in connection failures or garbled framing, because gRPC needs HTTP/2 semantics an HTTP-tunnel proxy may not preserve transparently, and a plain HTTP tunnel type won't forward raw TCP at all.
**Why it happens:** Tunnel tools default to HTTP mode because that's the most common use case; TCP mode is a separate, explicit tunnel type that must be requested.
**How to avoid:** Explicitly request a **TCP** tunnel (`ngrok.connect(port, "tcp")` or `ngrok tcp <port>`), not the default HTTP tunnel type.
**Warning signs:** Connection resets or immediate handshake failures from the `RobotClient` that don't reproduce when both processes are on the same machine/LAN.

## Code Examples

### Real hardware observation → action round trip (conceptual, combining existing patterns)

```python
# Source: patterns combined from control/keyboard_joint_control.py (read_positions),
# control/record_episode.py (open_cameras), and lerobot/async_inference/robot_client.py
# (installed source, this session)

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower

# max_relative_target caps how far a single action can move each joint in one
# send_action() call -- this is LeRobot's own built-in safety clamp, not custom code.
config = SOFollowerRobotConfig(
    port="/dev/cu.usbmodem5B8E1139151",
    id="soarm_follower_02",
    max_relative_target={
        "shoulder_pan": 5.0, "shoulder_lift": 5.0, "elbow_flex": 5.0,
        "wrist_flex": 5.0, "wrist_roll": 5.0, "gripper": 15.0,
    },  # degrees for arm joints, percent-points for gripper -- [ASSUMED] conservative starting values, tune during planning
)
robot = SO101Follower(config)
robot.connect(calibrate=False)
robot.bus.write_calibration(robot.calibration)  # match existing scripts' pattern
```

### Safety validator sketch (net-new code this phase must write)

```python
# Net-new for Phase 11 -- no existing equivalent in this repo.
import math

JOINT_LIMITS_DEG = {
    "shoulder_pan": (-70.33, 70.33),
    "shoulder_lift": (-107.47, 107.47),
    "elbow_flex": (-97.23, 97.23),
    "wrist_flex": (-102.02, 102.02),
    "wrist_roll": (-157.21, 162.79),
    "gripper": (0.0, 100.0),  # percent-open, NOT meters -- see Pitfall 4
}

def validate_action(action: dict, current_state: dict) -> tuple[dict, list[str]]:
    flags = []
    safe = {}
    for joint, value in action.items():
        if value is None or not math.isfinite(value):
            flags.append(f"{joint}: rejected non-finite value ({value}), holding current position")
            safe[joint] = current_state.get(joint, 0.0)
            continue
        lo, hi = JOINT_LIMITS_DEG[joint]
        clamped = max(lo, min(hi, value))
        if clamped != value:
            flags.append(f"{joint}: clamped {value} -> {clamped} (limit {lo}/{hi})")
        safe[joint] = clamped
    return safe, flags
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Synchronous inference (robot idle while policy computes next action chunk) | Asynchronous inference (`lerobot.async_inference`) — next chunk computed before current one is exhausted | Introduced alongside the SmolVLA paper/release (per LeRobot's own async-inference docs, "With our SmolVLA we introduced a new way to run inference...") | Directly relevant to D-01: async inference is the mechanism that makes a remote-GPU (Colab) bridge practical without a robot-side idle stall on every inference call |
| `smolvla_base` used directly | Fine-tuned, checkpoint-specific SmolVLA models per task/robot | Documented from SmolVLA's initial release — the model was always intended as a base-to-fine-tune model, not zero-shot | Directly resolves lerobot#2210 and VLAHW-05's go/no-go question: the "SmolVLA/SO-101 inference failure" reported upstream is a base-checkpoint misuse pattern, not a hardware incompatibility — a properly fine-tuned checkpoint should not hit the same crash |

**Deprecated/outdated:** None identified as deprecated within LeRobot 0.6.1's surface relevant to this phase; `lerobot-rollout` (single-machine, synchronous) still exists as a simpler alternative to the async bridge but does not satisfy D-01's "Colab GPU, not local laptop" requirement on its own (it expects the policy to load in the same process as the robot connection).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `victorvanhalst/smolvla_so101_cube`, `majinwakeup30/smolvla_so101_stack_cube_v3_2_cameras`, `lerobot/svla_so100_pickplace`, and `cn0303/smolvla-so101-strawberry-v3` are real, loadable HF Hub checkpoints with the camera-count/task properties summarized (found via WebSearch, not independently downloaded/loaded this session) | Standard Stack, Pitfall 2 | Wasted planning effort if a checkpoint's actual `config.json` doesn't match what's described here; mitigated by requiring a `checkpoint:human-verify` before committing to one during planning/execution |
| A2 | The real robot's degree-mode joint limits (Pattern 3) closely match the URDF/calibration-derived radian values converted to degrees | Pattern 3, Code Examples | If the live `soarm_follower_02.json` calibration has drifted since Phase 10 (e.g. after a recalibration event), the safety validator's bounds could be stale; mitigated by re-deriving from the live calibration file at execution time rather than hardcoding permanently |
| A3 | Conservative `max_relative_target` starting values (5° arm joints, 15% gripper) are safe defaults for a first VLA-driven run | Code Examples | If too tight, the arm may appear to "stall" against the clamp when the policy commands larger legitimate moves; if too loose, a bad model output could cause a larger-than-intended jump. Flagged for a `checkpoint:human-verify` / tuning pass during a supervised dry run before the "official" recorded episode |
| A4 | ngrok's free tier supports a TCP tunnel endpoint sufficient for this one-off research session (vs. requiring a paid plan) | Standard Stack | If ngrok's free tier no longer supports TCP tunnels by execution time, cloudflared + a configured Zero Trust tunnel (heavier setup) becomes the fallback — flag during planning if the free-tier assumption fails in practice |

**If this table is empty:** N/A — see entries above; none are compliance/security-critical, all are technical-fit assumptions flagged for a quick verification step during planning/execution.

## Open Questions

1. **Which SmolVLA checkpoint's camera configuration actually matches this project's 2-camera (wrist + overhead-stereo) rig?**
   - What we know: Several community SO-101 SmolVLA checkpoints exist; camera counts range from 1 to 3 per model card claims found via search.
   - What's unclear: None were independently downloaded/inspected this session to confirm their `config.json` camera keys against this project's actual `wrist`/`overhead` setup (and the overhead camera is a stereo pair in one physical frame, which is itself a further wrinkle).
   - Recommendation: During planning, script a quick `config.json` fetch (via `huggingface_hub`) for the top 2-3 candidates before committing; prefer the one closest to a true 2-camera match, and treat 3-camera checkpoints as workable only if `empty_cameras`/a dummy 3rd feed is acceptable.

2. **Does the AR0144 stereo pair need to be split into two separate camera feeds for the policy, or passed whole?**
   - What we know: The AR0144 currently captures as one 2560x720 side-by-side frame (per `diagnostics/UAT/function/basic/UAT.md`).
   - What's unclear: Whether any candidate checkpoint expects a single "overhead" RGB image (in which case the whole stereo frame, or one half, must be chosen) or genuinely wants two independent left/right feeds.
   - Recommendation: Default to using one half (or a center-cropped single view) as the "overhead" RGB feed unless a specific checkpoint's model card says otherwise — simplest, matches how most single-camera "external" views are trained.

3. **Will ngrok's free tier's TCP tunnel remain viable for the whole session, or will Colab's own session limits (idle disconnects) interrupt the bridge mid-episode?**
   - What we know: ngrok free tier historically supports one TCP tunnel; Colab free/Pro sessions have their own idle-timeout and max-runtime limits independent of ngrok.
   - What's unclear: Exact current limits at execution time (both ngrok's and Colab's policies can change).
   - Recommendation: Design the bridge/harness to detect a dropped connection cleanly (timeout → local e-stop / hold position → log the termination reason) rather than assuming the tunnel is always up — this directly satisfies VLAHW-02's "servo comms-failure / stale response" requirement, and doubles as the answer to "what happens if the bridge drops mid-episode."

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `lerobot` (installed) | Robot control, async bridge | Yes | 0.6.1 | — |
| `grpcio` (`lerobot[async]` extra) | `async_inference` bridge | **No** — import fails today | latest PyPI: 1.84.0 | Install via `pip install 'lerobot[async]'` in `control/.venv`; no viable fallback without it (hand-rolled HTTP relay is the only alternative, explicitly discouraged, see Don't Hand-Roll) |
| `pyngrok` or ngrok CLI + authtoken | Tunneling the Colab `PolicyServer` port | Not confirmed installed locally this session | latest PyPI: 8.1.2 | `cloudflared` is installed locally (`/opt/homebrew/bin/cloudflared`) but needs a configured Zero Trust tunnel for raw TCP — heavier setup, use only if ngrok is unavailable |
| Real SO-ARM101 follower hardware + calibration | The entire phase | Assumed available (hardware bring-up complete per prior UATs) — not independently re-verified this session | `soarm_follower_02`, port subject to change on replug | None — this phase requires the physical robot |
| Colab GPU runtime (T4/A100) | Hosting `PolicyServer` + SmolVLA inference | Assumed available per project's existing Colab budget (used throughout Phases 1-9) | — | None documented as needed; SmolVLA's ~2GB footprint is comfortably within even a free-tier T4 |
| Cameras (IMX335 wrist, AR0144 stereo overhead) | Observation capture | Assumed available (UAT-passed in prior session) — not re-verified this session | cv2 index 0 (wrist), index 1 (overhead) — **not guaranteed stable across USB replug/topology changes**, per this project's own UAT notes | Re-run the cv2 index-discovery snippet from `control/COMMANDS.md` if indices have shifted |

**Missing dependencies with no fallback:** none — `grpcio` has a clear install path, everything else has an assumed-available status from prior UAT work.
**Missing dependencies with fallback:** `pyngrok`/ngrok → `cloudflared` (heavier setup, same outcome).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (already used elsewhere in this repo, e.g. `scripts/test_calibration_utils.py`, `LIBERO/.../test_*.py`); **no test infrastructure currently exists in `control/`** |
| Config file | none — see Wave 0 gaps below |
| Quick run command | `cd control && source .venv/bin/activate && pytest <new test file> -x` (once added) |
| Full suite command | same, no marker/filter needed yet given the small expected surface area |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| VLAHW-01 | Action contract (units/order/joint count) matches documented DEGREES+RANGE_0_100, 6-DOF contract | unit | `pytest control/test_action_contract.py -x` | ❌ Wave 0 |
| VLAHW-02 | Safety validator clamps out-of-range/NaN/malformed actions correctly | unit | `pytest control/test_safety_validator.py -x` | ❌ Wave 0 |
| VLAHW-02 | E-stop halts the control loop and does not depend on network state | manual (hardware-in-the-loop; cannot be meaningfully automated without a real robot/network-drop simulation) | N/A — human-verify checkpoint | ❌ Wave 0 (harness), manual test always |
| VLAHW-03 | JSONL log record has all required fields, one record per step, camera paths resolve to real files | unit + integration | `pytest control/test_io_logger.py -x` | ❌ Wave 0 |
| VLAHW-04 | Full episode runs end-to-end, produces video + JSONL + termination reason | manual (hardware-in-the-loop) | N/A — human-verify checkpoint, cannot be automated without the real robot | ❌ Wave 0 (harness), manual test always |
| VLAHW-05 | Findings write-up accurately reflects observed behavior | manual (write-up review) | N/A | — |

### Sampling Rate
- **Per task commit:** run the relevant new unit test file (`test_action_contract.py`, `test_safety_validator.py`, or `test_io_logger.py`) with `-x`.
- **Per wave merge:** run all of `control/`'s new test files together (no full suite yet exists to merge into).
- **Phase gate:** all automated unit tests green, plus the mandatory hardware-in-the-loop checkpoints (e-stop verification, full episode recording) signed off by a human before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `control/test_action_contract.py` — covers VLAHW-01 (mock/no-hardware-required assertions on unit conventions and joint order)
- [ ] `control/test_safety_validator.py` — covers VLAHW-02 (pure-function tests against `JOINT_LIMITS_DEG`, NaN/inf/stale rejection, no real hardware needed)
- [ ] `control/test_io_logger.py` — covers VLAHW-03 (schema validation against a synthetic step record)
- [ ] `pytest` install/config in `control/.venv` — not currently present; `pip install pytest` needed before any of the above can run
- [ ] No existing `conftest.py`/shared fixtures in `control/` — a minimal one (e.g. a fake/mock `Robot` object for unit tests that shouldn't touch real hardware) will likely be needed

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | Partial — the Colab↔local tunnel is a network-exposed endpoint | ngrok tunnel authtoken (account-level auth on the tunnel itself); the gRPC server has no independent auth layer in stock `lerobot.async_inference` — treat the tunnel's own access control as the only auth boundary, and prefer ngrok's authenticated/reserved endpoints over a fully anonymous public tunnel if available on the account tier in use |
| V3 Session Management | No | Not applicable — this is a single ad hoc research session, not a multi-user system |
| V4 Access Control | Partial | The `PolicyServer` should bind to `0.0.0.0` only as required for the tunnel to reach it, not be otherwise exposed; shut down the tunnel/server when the session ends |
| V5 Input Validation | **Yes — this is the core of VLAHW-02** | The safety validator (joint limits, NaN/inf rejection, stale-observation/response rejection) *is* the ASVS V5 control for this phase — every action received over the network must be treated as untrusted input before it reaches the servos |
| V6 Cryptography | Partial | ngrok tunnels are TLS/TCP-encrypted in transit by ngrok's infrastructure by default; no additional crypto work needed for this ad hoc research bridge — do not hand-roll any encryption layer |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| A malformed/adversarial action arriving over the gRPC bridge (whether from a bug, a network glitch corrupting a message, or an unrelated party discovering the tunnel endpoint) drives the robot out of its safe envelope | Tampering | The local safety validator (joint limits + NaN/inf/malformed rejection) — this is the primary and sufficient mitigation for a research-scope, single-session bridge; do not treat the network layer itself as the safety boundary |
| Tunnel endpoint left open after the session ends, allowing an unrelated party to connect to a still-running `PolicyServer` | Elevation of Privilege / unauthorized access | Tear down the ngrok tunnel and stop the Colab `PolicyServer` process at the end of every session; do not leave it running unattended |
| Silent stale-observation loop (server keeps returning actions based on an old observation after the connection has effectively stalled) causing the robot to act on outdated state | Tampering (of effective state, not of the message itself) | Timestamp every observation and action; reject/hold-position on any action whose corresponding observation is older than a defined staleness threshold (this is itself part of VLAHW-02's explicit requirement list) |

## Sources

### Primary (HIGH confidence)
- `control/.venv/lib/python3.12/site-packages/lerobot/robots/so_follower/{config_so_follower.py,so_follower.py}` (installed lerobot 0.6.1 source, read directly this session) — action contract, `use_degrees` default, `max_relative_target`/`ensure_safe_goal_position`
- `control/.venv/lib/python3.12/site-packages/lerobot/async_inference/{constants.py,configs.py,policy_server.py,robot_client.py}` (installed lerobot 0.6.1 source, read directly this session) — bridge architecture, `SUPPORTED_ROBOTS`/`SUPPORTED_POLICIES`, gRPC transport confirmation
- `control/.venv/lib/python3.12/site-packages/lerobot/policies/smolvla/configuration_smolvla.py` (installed source) — SmolVLA config structure, normalization mapping, chunk sizing
- `scripts/calibration_utils.py`, `So-101/So-101.urdf` (this repo, Phase 10 output) — real calibration-derived joint limits
- `diagnostics/UAT/function/basic/UAT.md` (this repo) — independent corroboration of the `use_degrees=True` real-world behavior, camera device indices, hardware comms-failure patterns

### Secondary (MEDIUM confidence)
- huggingface.co/docs/lerobot/smolvla (WebFetch, this session) — SmolVLA fine-tuning requirement, `lerobot-rollout` usage
- huggingface.co/docs/lerobot/async (WebFetch, this session) — async inference architecture, CLI/Python usage examples
- ngrok.com/docs/using-ngrok-with/googleColab (WebSearch, this session) — Colab + ngrok TCP tunnel pattern
- github.com/huggingface/lerobot/issues/2210 (WebFetch, this session) — root cause of the flagged upstream risk

### Tertiary (LOW confidence)
- HF Hub checkpoint listings (`victorvanhalst/smolvla_so101_cube`, `majinwakeup30/smolvla_so101_stack_cube_v3_2_cameras`, `lerobot/svla_so100_pickplace`, `cn0303/smolvla-so101-strawberry-v3`) — found via WebSearch, not independently downloaded/inspected; all flagged `[ASSUMED]` in the Assumptions Log

## Metadata

**Confidence breakdown:**
- Standard stack (bridge architecture, action-unit contract): HIGH — verified directly against installed package source and this project's own UAT history, not against docs/training-data alone
- Architecture (safety validator thresholds, checkpoint choice): MEDIUM — thresholds are principled derivations from real calibration data but the exact numeric safety margins are proposed defaults, not empirically tuned; checkpoint choice is WebSearch-sourced, not independently verified
- Pitfalls: HIGH for the unit-ambiguity and wrist_roll/gripper-units pitfalls (directly source-verified); MEDIUM for the SmolVLA zero-shot pitfall (well-corroborated by the GitHub issue + official docs, but the exact fix path for lerobot#2210 was not confirmed by a maintainer response)

**Research date:** 2026-09-20
**Valid until:** ~14 days (fast-moving area — LeRobot's async_inference CLI surface and HF Hub checkpoint availability can both change quickly; re-verify checkpoint `config.json` and CLI flags immediately before execution regardless of this window)
