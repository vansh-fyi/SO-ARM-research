# Phase 11: VLA Hardware Connection - Context

**Gathered:** 2026-09-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Connect an SO-101-native joint-action VLA (SmolVLA) to the real physical SO-ARM101 over the existing `control/` LeRobot USB-serial bridge, behind an explicit safety validator, with complete per-inference-step I/O captured to a durable log, and at least one full observed episode recorded end-to-end (video + I/O log + termination reason) — closing with a findings write-up that recommends go/no-go on later milestone phases.

This phase is **SmolVLA-only**. A future phase (not this one) repeats the same harness against a genuine deep-reasoning MLLM to capture a reasoning trace — see Deferred Ideas below.

</domain>

<decisions>
## Implementation Decisions

### Inference compute location
- **D-01:** VLA inference runs on **Colab GPU**, not the local laptop. The local `control/` process (Python 3.12 venv, real hardware bridge) acts as a relay: it sends observations (camera frames + joint state) to the Colab-hosted SmolVLA and receives back the action to execute. This mirrors the compute split already used for the sim VLA phases (Phase 3), but is new plumbing for the *real-hardware* path — Phase 11 planning/research must design this local↔Colab bridge (protocol, latency handling, failure mode if the bridge drops mid-episode) since nothing like it exists yet in `control/`.
- Rationale: local CPU/Apple Silicon inference for a ~450M-param VLA was considered too slow/uncertain; Colab GPU access is already budgeted and proven for VLA inference in this project.

### First observed task
- **D-02:** The first full observed run uses a **small red cube** (already on hand) as the manipulation object, not the paper-aligned "Pen Transfer" task. Priority is proving the real-hardware pipeline works end-to-end over paper-benchmark alignment for this first phase. Verify the cube fits the existing embodiment constraints (≤84mm, within ~0.45m reach, not on the base's forward centerline — see `.planning/STATE.md` Blockers/Concerns) before finalizing the exact task prompt/placement during planning.

### Safety validator / e-stop
- **D-03:** E-stop is a **software keyboard interrupt** (a dedicated key, or a gracefully-handled Ctrl+C) that immediately halts motor commands — no new physical hardware. This matches the existing pattern in `control/keyboard_joint_control.py`. Consistent with PROJECT.md's explicit stance that a physical kill-switch (ESP32 etc.) remains a "possible future" item, not in scope now.
- All other safety-validator specifics (joint limits, max per-step displacement, max velocity, gripper bounds, stale/malformed/NaN rejection, servo comms-failure handling — per VLAHW-02) are implementation details for research/planning to work out against the real servo specs; not discussed further here.

### I/O log format (VLAHW-03)
- **D-04:** **JSON Lines** — one JSON object per inference step, containing: input references (per-camera timestamped frame paths, joint state, instruction), raw model output, validated action, executed action, latency, model version. Chosen over CSV specifically because it needs to stay schema-compatible with the future MLLM-comparison phase, which will append a `reasoning_trace` field — JSON Lines allows that as a non-breaking schema extension, whereas CSV handles nested/variable-length fields poorly. Camera frames themselves are saved as image/video files on disk, referenced by path from the JSON record (not embedded) — consistent with how `control/record_episode.py` already handles frame storage.

### Reasoning-trace scope (explicit non-decision, captured to prevent re-litigation)
- **D-05:** SmolVLA (like other VLAs) has no reasoning trace or tokens to capture — it maps observations directly to actions. VLAHW-03's "complete I/O" (not "reasoning trace") framing is intentional and already reflected in ROADMAP.md/REQUIREMENTS.md. The findings write-up (VLAHW-05) must explicitly state this limitation and recommend the next phase (deep-reasoning MLLM comparison, using an HF-hosted free/cheap-tier model first per the existing PROJECT.md decision) as where reasoning-trace capture actually applies.

### Claude's Discretion
- Concrete safety-validator thresholds (numeric limits) — derive from servo specs during planning/research.
- Exact local↔Colab bridge protocol/transport for the observation/action relay.
- Exact cube placement/prompt wording for the first task, so long as it respects the embodiment reach/collision constraints already documented in STATE.md.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope & requirements
- `.planning/ROADMAP.md` §"Phase 11: VLA Hardware Connection" — full goal, context/notes, and success criteria (already very detailed; do not re-derive from scratch)
- `.planning/REQUIREMENTS.md` §VLAHW-01..05 — the five locked requirements this phase must satisfy
- `.planning/PROJECT.md` §"Current Milestone: v2.0" and §"Key Decisions" — why SmolVLA (not Cartesian+IK), why no new microcontroller, why Colab-piloted-free-tier-first applies to the *next* MLLM phase

### Known upstream risk
- [huggingface/lerobot#2210](https://github.com/huggingface/lerobot/issues/2210) — reports SmolVLA inference failures on SO-101; verify early per ROADMAP.md's own flag, have a fallback SO-101-native joint-action VLA candidate identified during planning if this reproduces

### Existing hardware bridge (reuse, don't rewrite)
- `control/COMMANDS.md` — command reference for all working `control/` scripts (ports, IDs, usage)
- `control/keyboard_joint_control.py` — existing `SO101Follower` action-contract usage + keyboard e-stop pattern to extend, not replace
- `control/record_episode.py` — existing synced camera+joint recorder pattern; Phase 11's JSON Lines log should follow its per-camera timestamped-frame convention rather than inventing a new one
- `diagnostics/UAT/function/UAT.md` — documents the working leader+follower teleop + camera setup this phase builds on top of

### Digital twin (informational only — not a dependency)
- Phase 10 is complete and independent of Phase 11 (see ROADMAP.md Overview) — the digital-twin fix does not need to be consumed here, since Phase 11 drives the real robot directly

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `control/keyboard_joint_control.py` — `SO101Follower` connection/read/write pattern, retry-on-transient-comms-error pattern (relevant to VLAHW-02's comms-failure handling), keyboard-driven control loop to extend into a VLA-driven one with an e-stop key
- `control/record_episode.py` — `open_cameras()`, per-camera `cv2.VideoWriter` + timestamped capture pattern to reuse for the new JSON Lines I/O log's frame storage
- `diagnostics/servo_set_torque_limit.py`, `diagnostics/servo_set_protection.py` — existing servo-level protection utilities that may inform (but are distinct from) the software safety validator

### Established Patterns
- All `control/` scripts take `PORT ROBOT_ID` positional args and run inside the dedicated `control/.venv` (Python 3.12, separate from `diagnostics/`'s environment) — any new Phase 11 script should follow this convention
- Known hardware ports/IDs are documented in `control/COMMANDS.md` (subject to change on macOS replug — always re-check `ls /dev/cu.usbmodem*`)

### Integration Points
- New Phase 11 code integrates as a new script (or scripts) inside `control/`, reusing `SO101Follower` directly — no new hand-written serial/register code, per ROADMAP.md's explicit constraint
- The Colab-side SmolVLA inference process is new — no existing notebook wires SmolVLA to a live local relay yet (Phase 3's Colab VLA work was sim-only, feeding `OffScreenRenderEnv`, not a real hardware relay)

</code_context>

<specifics>
## Specific Ideas

- First task object: a small red cube already on hand (user-specified, not from a prop list)
- E-stop: software keyboard interrupt, matching the existing `keyboard_joint_control.py` UX
- I/O log: JSON Lines, one record per inference step, images referenced by path

</specifics>

<deferred>
## Deferred Ideas

- **Deep-reasoning MLLM comparison experiment** (e.g. Claude paid tier, or an HF-hosted reasoning model on Colab GPU per the user's suggestion) reusing the same JSON Lines I/O log with an added `reasoning_trace` field — this is the next phase after Phase 11, already anticipated as a "Future Requirement" in PROJECT.md. Do not fold into Phase 11; Phase 11's log schema is deliberately designed to make this addition non-breaking when that phase is planned.

### Reviewed Todos (not folded)
None — discussion stayed within phase scope.

</deferred>

---

*Phase: 11-VLA Hardware Connection*
*Context gathered: 2026-09-20*
