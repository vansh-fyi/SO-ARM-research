# Architecture Research

**Domain:** Real-hardware MLLM-driven robot manipulation control (integration into an existing LeRobot bridge)
**Researched:** 2026-09-15
**Confidence:** HIGH (grounded directly in installed LeRobot 0.6.1 source and this repo's existing `control/`/`diagnostics/` code, not external docs) for the LeRobot integration surface; MEDIUM for MLLM-loop/action-schema design (no established reference implementation for this exact "general MLLM as joint-space controller" pattern — nearest published analogs are subgoal/CoT VLA papers, cited below, which validate the plan-then-execute shape but not a concrete API)

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    control/ (Python 3.12 venv — unchanged)               │
│                                                                            │
│  ┌────────────────────┐        ┌───────────────────────────────────┐    │
│  │   control/mllm/     │        │  control/depth_stereo.py           │    │
│  │  (NEW package)       │        │  (NEW — real-time-at-cadence)      │    │
│  │  router.py           │        │  loads diagnostics/                │    │
│  │  loop.py              │◄──────┤  stereo_calibration.npz            │    │
│  │  schema.py            │  depth  as a DATA artifact (not a code     │    │
│  │  prompts.py           │  frame  import across venvs)               │    │
│  └────────┬──────────────┘        └────────────┬────────────────────┘    │
│           │ calls                               │ rectify+SGBM on-demand │
│           │ per sub-goal                        │ (per MLLM call, not    │
│           ▼                                      │  per video frame)      │
│  ┌────────────────────┐                         │                        │
│  │ control/motion_     │                         │                        │
│  │ primitives.py        │  (NEW — extracted from │                        │
│  │ (P-control executor, │   keyboard_joint_       │                        │
│  │  reused from          │   control.py)          │                        │
│  │  keyboard_joint_      │                         │                        │
│  │  control.py)          │                         │                        │
│  └────────┬──────────────┘                         │                        │
│           │ robot.send_action({...})               │                        │
│           ▼                                          │                        │
│  ┌────────────────────────────────────────────────┴─────────────────┐    │
│  │  SO101Follower / SOFollowerRobotConfig  (EXISTING, untouched)      │    │
│  │  lerobot[feetech]==0.6.1 — USB serial Feetech bus                  │    │
│  └────────┬─────────────────────────────────────────────────────────┘    │
│           │ get_observation() -> {joint}.pos dict                        │
│           ▼                                                              │
│  ┌────────────────────┐        ┌───────────────────────────────────┐    │
│  │ control/camera_io.py │        │  control/episode_writer.py         │    │
│  │ (NEW — extracted     │───────►│  (NEW — extended episode schema)   │    │
│  │  from record_        │ frames │  writes to control/episodes/       │    │
│  │  episode.py)          │        │  <task>/<episode_id>/              │    │
│  └────────────────────┘        └───────────────────────────────────┘    │
│                                                                            │
│  control/record_episode.py — UNCHANGED, stays the plain teleop/UAT       │
│  recorder it is today (fixed-fps loop, no MLLM/depth coupling)           │
└──────────────────────────────────────────────────────────────────────────┘
                     ▲ reads calibration artifact only (no import)
┌────────────────────┴───────────────────────────────────────────────────┐
│              diagnostics/ (Python .venv — unchanged, no torch)          │
│  stereo_calibrate.py, stereo_calibration.npz/.json (data contract),     │
│  measure_object_depth.py (offline/batch tool — stays as-is)             │
└──────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Status |
|-----------|----------------|--------|
| `control/mllm/router.py` | Provider-agnostic call surface (`call(images, text, schema) -> ParsedResponse`); wraps a single third-party abstraction (see Pattern 2) | **NEW** |
| `control/mllm/loop.py` | Plan-then-execute orchestration: observe → prompt → parse → execute sub-goal → check completion → repeat | **NEW** |
| `control/mllm/schema.py` | Structured action/response schema (pydantic or dataclass) shared by router output and executor input | **NEW** |
| `control/mllm/prompts.py` | Prompt templates, sub-goal vocabulary, few-shot examples | **NEW** |
| `control/motion_primitives.py` | Joint-space P-control executor (target pose → converged real motion), extracted from `keyboard_joint_control.py`'s `move_to_positions()` | **NEW (extracted, not new logic)** |
| `control/depth_stereo.py` | Loads `diagnostics/stereo_calibration.npz`, exposes `compute_depth(raw_stereo_frame) -> np.ndarray` using the same rectify+SGBM pipeline validated in `diagnostics/measure_object_depth.py`, callable at MLLM-decision cadence | **NEW** |
| `control/camera_io.py` | `open_cameras()`/frame-grab helpers extracted from `record_episode.py` so both the old recorder and the new one share one tested implementation | **NEW (extracted)** |
| `control/episode_writer.py` | Extended episode schema writer: RGB video (continuous) + depth (sparse, per sub-goal) + joints.csv (continuous) + reasoning_trace.jsonl + episode_meta.json | **NEW** |
| `SO101Follower` / `SOFollowerRobotConfig` | Physical joint I/O over USB serial (Feetech bus) | **UNCHANGED** — integration point only, never modified |
| `control/record_episode.py` | Fixed-fps teleop/UAT recorder | **UNCHANGED** |
| `diagnostics/stereo_calibrate.py`, `stereo_calibration.{npz,json}` | Offline stereo calibration; `.npz`/`.json` are the versioned data contract consumed by `control/depth_stereo.py` | **UNCHANGED** |
| `diagnostics/measure_object_depth.py` | Offline/interactive depth-quality diagnostic tool | **UNCHANGED** — its rectify+SGBM logic is the reference implementation `control/depth_stereo.py` ports into a callable, not a CLI script |

## Recommended Project Structure

```
control/
├── .venv/                          # unchanged
├── record_episode.py               # unchanged — plain teleop/UAT recorder
├── keyboard_joint_control.py       # unchanged, but move_to_positions() extracted
├── camera_io.py                    # NEW — shared camera open/read helpers
├── motion_primitives.py            # NEW — extracted P-control executor
├── depth_stereo.py                 # NEW — real-time-at-cadence stereo depth
├── episode_writer.py               # NEW — extended episode schema writer
├── mllm/                           # NEW package — the only genuinely new
│   ├── __init__.py                 #   architectural surface in this milestone
│   ├── router.py                   #   provider-agnostic call surface
│   ├── loop.py                     #   plan-then-execute orchestration
│   ├── schema.py                   #   structured action/response types
│   └── prompts.py                  #   prompt templates + sub-goal vocabulary
├── tasks/                          # NEW — per-task config (Pen Transfer, etc.)
│   └── pen_transfer.py
├── metrics/                        # NEW (Build Order step 6, not step 1)
│   └── failure_taxonomy.py
├── episodes/                       # NEW — canonical dataset root (gitignored)
│   └── pen_transfer/<episode_id>/
│       ├── camera_wrist.mp4
│       ├── camera_overhead_stereo.mp4   # raw side-by-side, for provenance
│       ├── joints.csv                   # continuous, full recorder fps
│       ├── depth/step_00.npy ...        # sparse — one per MLLM decision point
│       ├── reasoning_trace.jsonl        # one line per MLLM call
│       └── episode_meta.json            # task, provider/model, outcome, taxonomy label
└── outputs/                        # unchanged — ad hoc UAT scratch captures only
```

### Structure Rationale

- **`mllm/` as a subpackage, not flat files:** every other `control/` script is a flat, single-purpose file (matches this repo's established convention). The MLLM layer is the one exception because it has four genuinely distinct, separately-testable concerns (transport, orchestration, schema, prompt content) that will each change independently as providers/tasks are added — bundling them into one file would recreate the same "generic CLI doesn't fit this robot" pain this project already hit once with LeRobot's stock teleoperate path.
- **`episodes/` separate from `outputs/`:** `outputs/` is already established as ad hoc/UAT scratch space (`step5_still`, `step6_video_v3`, etc.) — mixing the real benchmark dataset into it would make it hard to tell disposable captures from the actual research dataset. This mirrors the project's own existing `explorations/data/` (input) vs `explorations/outputs/` (ephemeral render output) split.
- **`camera_io.py` / `motion_primitives.py` as extractions, not new logic:** both `record_episode.py` and `keyboard_joint_control.py` already contain hardened, UAT-tested versions of camera-open and P-control-to-target code (including the retry/backoff workaround for the known Feetech `sync_read` dropout, `github.com/huggingface/lerobot/issues/3131`). Reimplementing either for the MLLM loop risks silently dropping that hard-won robustness.
- **`depth_stereo.py` depends on diagnostics only as a data artifact:** `control/.venv` and `diagnostics/.venv` are deliberately separate (heavy LeRobot/torch deps vs. lightweight opencv/numpy). `stereo_calibration.npz` is plain numpy arrays (camera matrices, rectification maps) — loadable with `numpy.load()` from either environment with zero code coupling. `control/depth_stereo.py` should **port** the rectify+SGBM logic already proven in `diagnostics/measure_object_depth.py` (not import it), keeping the two venvs independent, exactly as the project's own conventions already require (no cross-venv Python imports anywhere in this repo).

## Architectural Patterns

### Pattern 1: Plan-Then-Execute Loop with a Local Fast-Path Executor

**What:** The MLLM is called only at sub-goal boundaries (reach→grasp→lift→place), never per-tick. Each call returns a structured sub-goal + target action; a local, deterministic P-control loop (the existing `move_to_positions()` pattern) then drives the arm to that target at full control frequency without further MLLM involvement, polling a cheap local completion check (position error under threshold, or a timeout) before the next MLLM call.

**When to use:** Any real API-latency-bound MLLM controlling a robot with a physical control loop that needs to run at tens of Hz. This is the only viable shape given real API round-trip times (hundreds of ms to seconds) versus the ~15-30 Hz servo loop already used elsewhere in this repo.

**Trade-offs:** Coarser reactivity than a closed-loop VLA (can't correct within a sub-goal without another MLLM round trip) — acceptable here since the paper being benchmarked (Yu & Qiu 2026) itself evaluates policies at a checkpoint/sub-goal granularity, so this loop shape is actually *more* comparable to their methodology, not a compromise against it.

**Example (loop skeleton, `control/mllm/loop.py`):**
```python
def run_task(robot, cameras, router, task_config, episode_writer):
    history = []
    for step in range(task_config.max_subgoals):
        obs = observe(robot, cameras)  # RGB (both cams) + depth (via depth_stereo.compute_depth)
        response = router.call(images=[obs.wrist_rgb, obs.overhead_rgb],
                                depth=obs.depth, joints=obs.joints,
                                task=task_config.instruction, history=history)
        episode_writer.log_reasoning(step, obs, response)  # reasoning trace is first-class, logged
                                                              # BEFORE execution, not after
        if response.action.type == "done":
            break
        target = resolve_action_to_joint_targets(response.action, obs.joints)  # local, deterministic
        target = ensure_safe_goal_position(target, obs.joints, task_config.max_relative_target)
        motion_primitives.move_to_positions(robot, target, kp=task_config.kp,
                                             control_freq=task_config.control_freq,
                                             max_seconds=task_config.subgoal_timeout)
        history.append(response)
    episode_writer.finalize(outcome=...)
```

### Pattern 2: Provider-Agnostic Router via a Single Thin Abstraction (not hand-rolled per-provider branching)

**What:** `control/mllm/router.py` wraps **litellm** (`pip install litellm`) rather than writing a custom `if provider == "openai"` dispatcher. litellm already normalizes OpenAI/Anthropic/Gemini/HuggingFace-hosted-endpoint calls (including multimodal image inputs) to one call signature and one response shape, which is exactly the "supports OpenAI/Anthropic/Gemini-style APIs architecturally" requirement in PROJECT.md. `router.py` itself only adds: (a) the structured-output schema enforcement/parsing layer, (b) provider selection from a project config (`control/mllm/providers.yaml` or env var), (c) the free-HF-model default for v2.0's first pilot.

**When to use:** Any project stating "provider-agnostic" as an explicit architectural requirement with a named list of target providers (OpenAI/Anthropic/Gemini) plus an unusual first target (a free HF-hosted model) — this is precisely litellm's design center (100+ providers behind one call shape, including `huggingface/<repo>` model strings).

**Trade-offs:** Adds one new third-party dependency to `control/requirements.txt`; in exchange, avoids re-solving per-provider auth/retry/streaming/multimodal-encoding quirks that a hand-rolled router would have to duplicate for each of 3+ providers — the same category of "don't fight the framework" lesson this project already learned once with LeRobot's CLI (where the fix there was to go *below* the CLI to the class API, not to reimplement LeRobot itself; here the equivalent is to use litellm's SDK layer, not its optional proxy-server deployment mode, which is unneeded for a single-machine research loop).

### Pattern 3: Depth-at-Decision-Cadence, Not Depth-at-Frame-Rate

**What:** Stereo rectification + `StereoSGBM` block matching (the pipeline validated in `diagnostics/UAT/function/depth/UAT.md`) is computed **once per MLLM call** (every few seconds, at a sub-goal boundary), not once per recorded video frame (15-30 fps). RGB video continues to be captured continuously at full recorder fps (unchanged behavior); depth is a **sparse** signal recorded only at the same points where a reasoning-trace entry is written.

**When to use:** Whenever a real-time-feeling depth signal is needed by a decision-maker (MLLM) that itself only makes decisions at a coarse cadence — computing depth faster than it's consumed is wasted CPU and, more importantly, avoids overloading the same camera-read loop that must not drop RGB frames (the recorder already has a documented history of frame-drop/choppiness and USB-power flakiness in `diagnostics/UAT/function/basic/UAT.md` Steps 6-7; adding a ~tens-of-ms SGBM call inside that same tight loop risks reintroducing exactly that class of bug).

**Trade-offs:** The recorded depth stream has much lower temporal resolution than RGB — acceptable because the extended episode schema's depth field exists to give the MLLM (and later, human failure analysis) spatial grounding at decision time, not to reconstruct continuous 3D motion.

## Data Flow

### Per-Sub-Goal Control Flow

```
[camera_io: grab wrist RGB + overhead stereo frame]
        ↓
[depth_stereo.compute_depth(overhead_frame)]  ← only at sub-goal boundaries
        ↓
[robot.get_observation()]  → {joint}.pos dict (existing SO101Follower API)
        ↓
[mllm/loop.py: assemble observation + task instruction + sub-goal history]
        ↓
[mllm/router.py → litellm.completion(...)]  → provider-agnostic call
        ↓
[mllm/schema.py: parse structured response]  → {reasoning, subgoal_label, action}
        ↓
[episode_writer.log_reasoning(...)]  ← reasoning trace written BEFORE execution
        ↓
[resolve_action_to_joint_targets(...) + ensure_safe_goal_position(...)]  ← local, deterministic, no MLLM
        ↓
[motion_primitives.move_to_positions(robot, target, ...)]  → robot.send_action({...}) in a tight P-control loop
        ↓
[episode_writer: continuous joints.csv row + continuous camera_*.mp4 frame append, at full recorder fps]
        ↓
[loop.py: check response.action.type == "done" → next sub-goal or finalize episode]
```

### Continuous vs. Sub-Goal-Cadence Streams

| Stream | Cadence | Written by |
|--------|---------|------------|
| `camera_wrist.mp4`, `camera_overhead_stereo.mp4` | Continuous, full recorder fps (15-30) | `episode_writer.py`, via `camera_io.py` — same loop shape as today's `record_episode.py` |
| `joints.csv` | Continuous, full recorder fps | `episode_writer.py`, via `robot.get_observation()` — unchanged from today's schema |
| `depth/step_NN.npy` | Sparse — once per MLLM call | `episode_writer.py`, via `depth_stereo.compute_depth()` |
| `reasoning_trace.jsonl` | Sparse — once per MLLM call | `episode_writer.py`, via `mllm/loop.py` |
| `episode_meta.json` | Once per episode (finalize) | `episode_writer.py` |

## Scaling Considerations

Reframed for this project: "scale" is not user load but **provider count, task count, and episode volume**.

| Axis | Now (Pen Transfer, 1 free HF model) | Near-term (4 paper tasks, 2+ providers) | Later (full benchmark run) |
|------|--------------------------------------|------------------------------------------|------------------------------|
| Providers | 1 hardcoded default in `router.py` config | Provider selection via CLI flag/env var into the same litellm call — no code branching needed | Cost tracking (litellm has this built in) becomes relevant once paid providers are added |
| Tasks | 1 task config (`tasks/pen_transfer.py`) | 4 task configs, shared `loop.py`/`schema.py` unchanged | Task configs stay data, not code — new task = new config file, not a new loop |
| Episode storage | Local disk under `control/episodes/` | Still local disk — depth is sparse (Pattern 3) so volume stays modest even across ~20 episodes/task × 4 tasks | If this grows past a few hundred episodes, revisit whether raw `.npy` depth should become a compressed format (`.npz` with `compression="lzf"` or similar) — not needed yet |

### Scaling Priorities

1. **First bottleneck:** MLLM API latency per sub-goal call (real, not simulated latency) — already designed around via Pattern 1 (plan-then-execute), not a future fix.
2. **Second bottleneck:** if/when moving beyond a free HF-hosted model to paid providers at benchmark scale (20 episodes × 4 tasks × multiple sub-goals), API cost and rate limits become real — litellm's built-in cost tracking (Pattern 2) is the mitigation already in place by design, not something to bolt on later.

## Anti-Patterns

### Anti-Pattern 1: Wedging Computed Stereo Depth into LeRobot's Native Camera-Depth Interface

**What people do:** `SOFollower`'s `_cameras_ft`/`observation_features` already has a `use_depth` flag and a `cam.read_latest_depth()` hook (confirmed by reading `control/.venv/.../lerobot/robots/so_follower/so_follower.py`) — the natural-looking move is to write a custom LeRobot `Camera` subclass that exposes the AR0144's computed depth through that interface so it "just works" with `robot.get_observation()`.

**Why it's wrong:** That interface is designed for hardware depth streams (e.g. RealSense) that produce a depth frame at the *same cadence* as RGB. Our depth is a derived, computed signal (SGBM on a rectified pair) that this project has already found to be unreliable on many real-object surfaces (specular glint, low-texture failure — see `diagnostics/UAT/function/depth/UAT.md` Step 4) and, per Pattern 3, is deliberately computed at a *different, sparser* cadence than RGB. Forcing it through a same-cadence camera abstraction either (a) silently computes depth far more often than needed (wasted CPU, risk of frame drops in the exact loop this project has already fought USB/power flakiness in), or (b) requires faking a per-frame depth stream that doesn't reflect the sub-goal-cadence reality, muddying the episode schema.

**Do this instead:** Keep depth computation as an explicit, separate call (`control/depth_stereo.py`) invoked by `mllm/loop.py` at sub-goal boundaries, written to its own sparse `depth/step_NN.npy` files — decoupled entirely from `SOFollower`'s camera config.

### Anti-Pattern 2: Re-Fighting LeRobot's Generic CLI for the New Loop

**What people do:** Reach for `lerobot-record` or another generic LeRobot CLI entry point to drive the new MLLM-controlled episodes, since it already exists and produces a "proper" dataset format.

**Why it's wrong:** This project already has a documented, first-hand precedent that LeRobot 0.6.1's generic CLI paths don't cover `so101_follower`-specific needs (`keyboard_joint_control.py`'s docstring: `lerobot-teleoperate --teleop.type=keyboard` crashes because `KeyboardTeleop.get_action()` returns raw key names, not joint deltas, and the CLI's default processor pipeline never translates them for this robot class). The MLLM plan-then-execute loop is a *bigger* deviation from LeRobot's assumed teleop/policy-rollout shape (variable-length sub-goals driven by an external reasoning call, not a fixed-fps teleop stream or a policy's `.select_action()`) — there is no reason to expect the generic CLI or `lerobot-record`'s dataset writer to fit better here than it did for keyboard control.

**Do this instead:** Keep using `SO101Follower`/`SOFollowerRobotConfig` directly (the class-level API, not the CLI) exactly as `record_episode.py` and `keyboard_joint_control.py` already do, and keep the episode writer hand-rolled (`episode_writer.py`), consistent with `record_episode.py`'s own stated rationale ("we built our own recorder rather than fighting `lerobot-record`'s CLI").

### Anti-Pattern 3: Starting MLLM Control-Loop Work Before Depth Is Reliable

**What people do:** Build and validate the router/loop/recorder against RGB-only observations first, planning to "wire in depth later" once the camera issue is fixed, treating depth as an additive field.

**Why it's wrong:** This is explicitly the locked sequencing decision for this milestone (PROJECT.md: "Fix AR0144 depth-camera object-measurement reliability before starting MLLM task work, not in parallel... building the MLLM control loop against known-unreliable depth risks having to redo recording/validation work once depth is fixed"). Concretely: the episode schema (`depth/step_NN.npy`), the `observe()` function's return shape, and any depth-conditioned prompting logic in `mllm/prompts.py` all depend on knowing what a *reliable* depth reading looks like (units, noise floor, valid-pixel coverage) — building against today's known-degenerate readings (Step 4 of the depth UAT: fake flat-plateau Z values, 0% valid on shiny objects) means re-validating all of that once Step 4 is fixed.

**Do this instead:** Follow the Build Order below — depth reliability closes first.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Free HuggingFace-hosted multimodal model (pilot) | via litellm's `huggingface/<repo>` provider string in `router.py` | Validate multimodal (image) input support for the specific model chosen before committing — not all free HF Inference Endpoints/serverless models accept image inputs; confirm during Build Order step 2 |
| OpenAI / Anthropic / Gemini (later providers) | Same `router.py` call surface, different provider string/env var — no loop/schema changes needed | This is the payoff of Pattern 2; if this ever requires touching `loop.py`, the router abstraction has leaked and should be revisited |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `control/mllm/loop.py` ↔ `SO101Follower` | Direct class calls (`get_observation()`, `send_action()`) via `control/motion_primitives.py` | Never through LeRobot's CLI — see Anti-Pattern 2 |
| `control/depth_stereo.py` ↔ `diagnostics/` | File-based data contract only (`stereo_calibration.npz`) | No Python import across venvs — see Structure Rationale |
| `control/mllm/router.py` ↔ third-party MLLM APIs | litellm SDK (in-process), not a proxy server | Single-machine research loop; the proxy/gateway deployment mode of litellm is unneeded overhead here |
| `episode_writer.py` ↔ disk | Plain files (`.mp4`, `.csv`, `.npy`, `.jsonl`, `.json`) under `control/episodes/` | Deliberately not LeRobot's HF-dataset format nor HDF5 — matches this project's own precedent of a hand-rolled, fully-understood recorder over a framework-native one |

## Build Order

This follows the locked dependency chain from PROJECT.md exactly (depth fix → control loop/router → recorder extension → first task → remaining tasks → failure/recovery metrics → multi-provider):

1. **Depth camera reliability fix** (`diagnostics/UAT/function/depth/UAT.md` Step 4, already in progress — lighting fix, then broaden to 2-3 more matte objects). **Gate: do not start step 2 until this UAT step passes.** This determines the real units/noise-floor/coverage that `control/depth_stereo.py` and the episode schema's `depth/step_NN.npy` field will assume.
2. **Router + plan-then-execute loop skeleton**, RGB-only, no real robot motion yet (dry-run against logged/replayed observations or a single stub sub-goal). Build `control/mllm/{router,schema,prompts}.py` and `control/mllm/loop.py` against the free HF-hosted model first (per PROJECT.md's explicit pilot choice). Extract `control/motion_primitives.py` from `keyboard_joint_control.py` here, since the loop needs it immediately.
3. **Recorder extension**: build `control/camera_io.py` (extracted) and `control/episode_writer.py`, then wire `control/depth_stereo.py` (now backed by validated calibration from step 1) into the loop's `observe()` call at sub-goal cadence (Pattern 3). This is also where reasoning-trace-before-execution logging (Pattern 1's `episode_writer.log_reasoning(...)` call ordering) gets validated end-to-end for the first time.
4. **Pen Transfer task, end-to-end**: `tasks/pen_transfer.py` config + a handful of real runs on the physical arm, validating the full chain (camera → depth → MLLM → reasoning trace + action → joint command → recorder) against the paper's simplest task before investing further.
5. **Remaining 3 paper tasks** (Selective Color Sorting, Multi-Object Packing, Precision Pen Placement) — new `tasks/*.py` configs only; `loop.py`/`router.py`/`episode_writer.py` should need no changes if step 2-4 were built generically. If they do need changes, that's a signal the task-config abstraction is too thin and needs revisiting before adding a 6th/7th task.
6. **Failure taxonomy + Recovery Rate metric**: `control/metrics/failure_taxonomy.py`, computed as a post-hoc pass over `reasoning_trace.jsonl` + `episode_meta.json` across all collected episodes (Grasp Instability, Repetition Loop, State Mismatch, Precision Misalignment categories, per PROJECT.md) — deliberately last among the "get one MLLM working" steps, since it needs a real corpus of both successful and failed episodes to be meaningful, not a single task's data.
7. **Multi-provider swap-in**: add a second (paid) provider string/config to `router.py`'s existing abstraction (Pattern 2) and re-run the same task suite for cross-provider comparison. If this step requires anything beyond a config/env-var change, the router abstraction from step 2 was under-designed and should be fixed before scaling to more providers.

## Sources

- `control/.venv/lib/python3.12/site-packages/lerobot/robots/so_follower/so_follower.py` (installed LeRobot 0.6.1 source, read directly — confirms `get_observation()`/`send_action()` shapes and the `use_depth`/`read_latest_depth()` native-camera-depth hook referenced in Anti-Pattern 1) — HIGH confidence (primary source, live installed code)
- `control/record_episode.py`, `control/keyboard_joint_control.py`, `control/joint_jog.py` (this repo) — HIGH confidence (existing, UAT-validated project code)
- `diagnostics/UAT/function/depth/UAT.md`, `diagnostics/UAT/function/basic/UAT.md`, `diagnostics/measure_object_depth.py`, `diagnostics/stereo_calibrate.py` (this repo) — HIGH confidence
- [LiteLLM GitHub](https://github.com/BerriAI/litellm) / [LiteLLM Providers docs](https://docs.litellm.ai/docs/providers) — MEDIUM confidence (web search, not yet integrated/verified in this repo); recommended as the concrete implementation of Pattern 2's "single thin abstraction," not a hard requirement
- Subgoal/chain-of-thought VLA literature validating the plan-then-execute + reasoning-trace shape: [ThinkingVLA (arXiv:2606.17937)](https://arxiv.org/pdf/2606.17937), [Interleaved Vision-Language Reasoning Traces (arXiv:2605.00438)](https://arxiv.org/html/2605.00438v1), [CoT-VLA (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_CoT-VLA_Visual_Chain-of-Thought_Reasoning_for_Vision-Language-Action_Models_CVPR_2025_paper.pdf) — MEDIUM confidence (these validate the general pattern shape for VLA policies, not a general-purpose MLLM-as-controller architecture specifically, so treated as directional support, not a template to copy)
- `.planning/PROJECT.md` (this repo) — locked sequencing decisions and target feature list, treated as ground truth for build order

---
*Architecture research for: Real-hardware MLLM-driven robot manipulation (v2.0 milestone)*
*Researched: 2026-09-15*
