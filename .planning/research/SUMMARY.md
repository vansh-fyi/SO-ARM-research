# Project Research Summary

**Project:** SoARM VLA Research — v2.1 milestone (MLLM Raw-Autonomy Benchmark)
**Domain:** Real-hardware robot control loop — adding an MLLM (Claude) raw-JSON control path alongside an existing VLA bridge, fixing an async tick-latency bug, and unifying macOS camera device resolution
**Researched:** 2026-09-24
**Confidence:** HIGH (architecture/stack grounded in direct reads of this project's own source and vendored `lerobot` library; pitfalls grounded in three real bugs already hit in this exact codebase; features grounded in a directly-read arXiv paper plus MEDIUM-confidence ecosystem literature)

## Executive Summary

This milestone adds three tightly-scoped capabilities to an already-working VLA bridge (`control/vla_bridge/`): (1) a new `ClaudeActionSource` that makes Claude reason directly from camera frames + joint state + calibration to raw joint-level JSON action chunks — deliberately with no `move_to()`/`grasp()` primitives, so it's a genuine test of what a general reasoning model does with only floor-level I/O; (2) a fix for a real, already-diagnosed tick-latency bug where `BridgeActionSource.get_action()` calls `control_loop_observation()` unconditionally every tick instead of gating it behind the vendored `lerobot` library's own (already-configured but unused) `_ready_to_send_observation()` queue-low check; and (3) unifying two divergent camera-resolution code paths (name-based for inference, numeric-index-based for recording) that have already twice caused a real episode to silently record the laptop webcam instead of the robot workspace. All three integrate through existing, provider-agnostic seams (`ActionSource` Protocol, `safety_validator.py`, `io_logger.py`, `action_contract.py`) that Phase 11 already proved out — the right move is reuse, not new infrastructure.

The recommended approach is: fix the latency bug first (using the library's own `_ready_to_send_observation()` gate, not a hand-rolled async rewrite), re-tighten safety caps with live-hardware re-verification, land the camera-resolution fix (via one shared, name-based, `ffmpeg`-backed resolution helper reused by both inference and recording paths — not two independent fixes), then build `ClaudeActionSource` using Anthropic's structured-output mode (`output_config.format`, not tool-use — tool-use would reintroduce the primitive-abstraction the milestone explicitly rules out), with its JSON schema generated programmatically from `action_contract.py`'s constants. Chunk size for the MLLM's action sequences must be derived from the latency fix's real measured round-trip data, not guessed.

The key risk across all three pieces is the same shape already proven out in this codebase: **silent failure that looks healthy** — three separate bugs in Phase 11 (missing thread, wrong staleness constant, key-mismatch) all ran to completion without crashing while doing the wrong thing, and the camera bug was caught only by a human manually reviewing saved footage. The mitigations converge on one discipline: make every failure mode loud (schema-conformance checks that reject-and-alert rather than silently drop, per-action-within-chunk validation that holds position rather than zero-fills, dual-metric latency verification rather than trusting a reduced discard-rate alone) and verify with real live-hardware episodes, not code review or a single happy-path test.

## Key Findings

### Recommended Stack

No new heavy dependencies. Add `anthropic` (official Anthropic Python SDK, `1.8.0` current) and `pydantic>=2.0` to `control/requirements.txt`; everything else (structured JSON output, image content blocks, latency-fix logic, camera-resolution fix) reuses stdlib (`queue`, `threading`, `time.monotonic()`) and already-installed project code (`stereo_camera.py`'s ffmpeg/AVFoundation capture, `pillow`/`cv2.imencode` for frame encoding).

**Core technologies:**
- `anthropic` Python SDK (`1.8.0`) — calls the Claude Messages API for the MLLM control loop — official first-party SDK, project policy mandates it over raw HTTP calls
- Structured output (`output_config: {"format": {"type": "json_schema", ...}}`, GA, no beta header) — forces Claude's response to validate against an explicit schema — this, not tool-use, is what satisfies "no pre-built movement primitives"
- `claude-opus-5` for the real comparison episode / `claude-sonnet-5` for cheaper iteration on control-loop plumbing — model choice is a one-line swap, no code-path differences

### Expected Features

Full detail in FEATURES.md. The cited Yu & Qiu 2026 paper (arXiv:2606.08881) does **not** prompt a general MLLM for raw joint control — it fine-tunes/evaluates π0.5, SmolVLA, Wall-X, and ACT — so it supplies the Pen Transfer task definition, the 4-category failure taxonomy, and the Recovery Rate metric, but not a prompting/schema precedent. The MLLM design itself draws on general zero-shot-trajectory-generation and Embodied Chain-of-Thought literature.

**Must have (table stakes):**
- Explicit output JSON schema (structured output, not free-text parsing)
- Static calibration/geometry + fresh joint-state/camera-frames per call
- Action-chunking sized from real latency-fix timing data
- Reasoning text captured before, and separately from, the action JSON, in staged form (restate task → plan → grounded scene claims → action)
- Safety-validator and episode-logging schema reused unchanged from the VLA path

**Should have (differentiators):**
- Reasoning-trace vs. executed-action divergence analysis
- Failure-taxonomy tagging using Yu & Qiu's 4 categories (Grasp Instability, Repetition Loop, State Mismatch, Precision Misalignment)

**Defer (v2.x+):**
- Provider-agnostic MLLM router, full 4-task suite, Recovery Rate automation, automated vision-based success detection, chunk-boundary self-report field

**Explicit anti-features (would invalidate the experiment):** pre-built movement primitives/tool-calling, multi-agent decomposition, few-shot in-context examples, code-generation-as-policy, per-tick MLLM calls, fine-tuning the MLLM.

### Architecture Approach

`run_episode()` is already provider-agnostic via the `ActionSource` Protocol (`get_action(joint_state, instruction) -> (action, model_version)`) — Phase 11 proved this seam by swapping `ScriptedActionSource` → `BridgeActionSource` with zero changes to the control loop itself. The MLLM integration is the same swap: a new `ClaudeActionSource` implementing the same Protocol, widened by one field to also return a reasoning string, feeding the same unchanged `safety_validator.py`/`io_logger.py`/`action_contract.py` trio. No second safety-check path, no second logger, no second `send_action()` call site.

**Major components:**
1. `ClaudeActionSource` — new `ActionSource` implementation; synchronous Claude call producing a local action-chunk queue (no thread/lock needed, unlike the VLA bridge's async gRPC queue)
2. `BridgeActionSource.get_action()` fix — gate `control_loop_observation()` behind the vendored `lerobot` library's own `_ready_to_send_observation()` (already configured via `chunk_size_threshold=0.5`, just never consulted)
3. Shared camera-resolution helper + `_StereoHalfReader` adapter — one `StereoSplitCamera` instance shared between the inference path and the `IOLogger` recording path, resolved by AVFoundation device **name**, never a numeric index

### Critical Pitfalls

1. **Latency "fix" trades one silent failure for another** — widening `STALE_ACTION_S`/`control_hz` legalizes multi-second-stale actions instead of eliminating the redundant round-trip. Avoid: pick a concrete numeric round-trip target before choosing a fix, measure observation-to-execution wall-clock latency directly, not just discard-rate.
2. **New race condition from an unguarded queue-low request trigger** — implementing the gate without an in-flight-request guard can fire duplicate/overlapping observation requests under slow Colab responses. Avoid: explicit `request_in_flight` guard, tested under simulated slow-network conditions.
3. **Reusing `safety_validator.py` unchanged for MLLM output assumes a key/unit/shape contract the MLLM was never forced to honor** — the validator silently drops non-matching keys, exactly the mechanism that caused Phase 11's bug #3 (`.pos`-suffix mismatch emptied every action into `{}`). Avoid: an explicit pre-validator schema-conformance check built from `action_contract.py`'s constants, with a loud (not silent-drop) failure path.
4. **Per-action-within-chunk parse failures** — with no primitives, every value in a multi-action chunk is a fresh parse-failure opportunity; a malformed mid-chunk action must hold position, never zero-fill/default.
5. **Numeric-vs-name camera divergence is a class of bug, not one bug** — fixing only the reported `camera_overhead` call site leaves `device_map.json`'s raw numeric field available for a third path to reintroduce the same failure later. Avoid: one shared, enforced, name-based resolution helper for every camera-opening call site.

## Implications for Roadmap

Based on research, suggested phase structure (source order already confirmed correct by architecture-level dependency analysis, not just PROJECT.md's stated sequencing):

### Phase 1: Bridge Tick-Latency Fix
**Rationale:** Blocks action-chunk sizing for the MLLM phase; independently testable against the existing SmolVLA baseline with no dependency on anything else in this milestone.
**Delivers:** `_ready_to_send_observation()` gating in `BridgeActionSource.get_action()`, `pop_validated_action()`/`client.latest_action` bookkeeping fix, `latency_ms` instrumentation fix (from always-`{0,0}` to real `time.monotonic()` deltas), in-flight-request guard against duplicate observation sends.
**Avoids:** Pitfall 1 (constant-widening masquerading as a fix) and Pitfall 2 (new race condition from the queue-low trigger) — both require the same live-hardware verification discipline (measured latency vs. a stated target, not just reduced discard-rate).

### Phase 2: Safety-Validator Cap Re-Tightening
**Rationale:** Caps were loosened specifically to compensate for the latency bug's staleness; must be re-tightened immediately after Phase 1, not deferred, to keep the causal link auditable — and must be verified live, not assumed safe as a "revert."
**Delivers:** Incrementally re-tightened `MAX_RELATIVE_TARGET_DEG`/`MAX_VELOCITY_DEG_PER_S`/`STALE_OBSERVATION_S`/`STALE_ACTION_S`, with a live-hardware episode confirming real-action yield at each step.
**Avoids:** Pitfall 8 (repeating bug #2's untested-constant risk in the opposite direction).

### Phase 3: Camera Device Resolution Unification
**Rationale:** Fully independent of Phases 1/2 and 4 — touches only `stereo_camera.py`/`run_vla_episode.py`'s camera-opening code. Landing it before the MLLM phase means the next live re-verification run (needed anyway to confirm Phase 1) also produces trustworthy footage, and the MLLM phase's own camera input builds on already-correct plumbing rather than debugging two unfamiliar systems (Claude vision input + a still-broken camera path) at once.
**Delivers:** One shared, name-based camera-resolution helper; recording path routed through the same `StereoSplitCamera`/ffmpeg-backed capture the inference path already uses (not a second independent `cv2.VideoCapture` open); client-side frame hash/thumbnail logging, explicitly documented as not covering server-side receipt.
**Avoids:** Pitfall 6 (scoping the fix to one call site instead of the shared pattern) and Pitfall 7 (treating a client-side hash as full provenance).

### Phase 4: MLLM Raw-JSON Control Loop
**Rationale:** Depends on Phase 1 for real chunk-size timing data and benefits from Phase 3 already landed for trustworthy camera input — the entire experiment's point of comparison.
**Delivers:** `ClaudeActionSource` (structured-output Claude calls, JSON schema generated from `action_contract.JOINT_ORDER`/`ACTION_UNITS`), `ActionSource` Protocol widened to carry a reasoning string, `IOLogger.write_step()` extended with a `reasoning` field, per-action-within-chunk validation that holds position on parse failure, and a documented root-cause note on why the earlier general-MLLM-prompting attempt "failed badly" before building mitigations against it.
**Addresses:** All P1 features from FEATURES.md (control loop, reasoning-trace capture, chunk sizing, structured reasoning format).
**Avoids:** Pitfalls 3, 4, 5, and 9 (schema-conformance gap, mid-chunk parse failures, repeating the undiagnosed prior failure, prompt-injection-shaped visual risk).

### Phase 5: Pen Transfer Comparison Run
**Rationale:** Depends on all four above — needs a fixed/trustworthy control loop, correct recorded camera evidence, and the MLLM source to exist.
**Delivers:** One live episode, human-judged success/termination, directly compared against the Phase 11 SmolVLA baseline episode (`11-05-retry-20260924-120530`).
**Implements:** MVP definition's full "Launch With" scope from FEATURES.md.

### Phase Ordering Rationale

- Latency fix must precede chunk sizing (a hard data dependency, confirmed at the source-code level, not just asserted in planning docs).
- Safety-cap re-tightening is sequenced immediately after the latency fix (not deferred) specifically to keep the "why were these loosened / why are they now safe" causal chain auditable in commit history.
- Camera fix is moved earlier than a literal reading of the milestone's active-item list might suggest — it's independent of the control-loop work and de-risks two unfamiliar systems (camera + MLLM vision input) from being debugged simultaneously.
- The MLLM phase is deliberately last among the build phases — it is the highest-novelty, highest-uncertainty piece, and every other phase produces infrastructure it depends on (reused unchanged, per the architecture's "provider-agnostic seam" design).

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 4 (MLLM Raw-JSON Control Loop):** Sparse direct precedent (the cited paper doesn't cover MLLM prompting at all); prompt/schema design, chunking behavior under real API latency, and the prior "failed badly" attempt's undocumented root cause all need scoping-time investigation before implementation.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Latency Fix):** Root cause and fix mechanism already fully diagnosed at the source level (`_ready_to_send_observation()` already exists and is already configured, just unused) — this is a location-and-wire-up task, not a design task.
- **Phase 2 (Cap Re-Tightening):** Mechanical, well-understood config change; the discipline required (live-hardware re-verification) is already fully specified.
- **Phase 3 (Camera Fix):** Root cause and fix pattern already fully diagnosed (name-based resolution + shared capture instance); implementation is mechanical.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Verified directly against PyPI (`pip index versions anthropic`) and this project's own installed environment; no speculative dependency choices |
| Features | MEDIUM | The one directly-relevant paper was read in full from its own HTML text, but it doesn't cover MLLM prompting at all — general ecosystem patterns (UCL trajectory-generator, ECoT) are MEDIUM-confidence, cross-checked across 2-3 sources each, not HIGH |
| Architecture | HIGH | Every claim grounded in direct reads of this project's own source and the actual installed `lerobot==0.6.1` library source, with file:line citations throughout — not framework docs or assumption |
| Pitfalls | MEDIUM-HIGH | Core pitfalls (1-8) are grounded in three real bugs already hit in this exact codebase (HIGH-confidence primary evidence); supporting web sources (async-VLA latency research, OpenCV camera-index issues, structured-output reliability) are MEDIUM-confidence, cross-checked but not primary |

**Overall confidence:** HIGH — the domain-specific risk (real hardware, real prior bugs, real vendored library internals) is unusually well-grounded for this milestone precisely because Phase 11 already surfaced concrete failure evidence in this same codebase; the one genuinely open area is MLLM-specific prompting/schema design, which has no direct precedent and is correctly flagged for deeper research at planning time.

### Gaps to Address

- **Prior "failed badly" general-MLLM-prompting attempt has no documented root cause** in materials available to this research pass (PROJECT.md references it but doesn't detail the failure mode) — must be re-derived (spatial grounding? output format? safety violations?) during Phase 4 planning before implementation starts, per Pitfall 5.
- **Real Colab/ngrok round-trip latency distribution is not yet measured** (only a single-session ~11-20s/tick anecdote exists) — Phase 1 must produce this measurement before Phase 4's chunk-size can be finalized; treat any chunk-size choice as provisional until Phase 1's real data exists.
- **Server-side (Colab PolicyServer) frame provenance cannot currently be verified**, only the bridge's outbound send — Phase 3's hash/thumbnail fix should explicitly document this as a known, accepted gap rather than implying full end-to-end verification.
- **Yu & Qiu 2026's disclosed termination criteria are thinner than PROJECT.md's Future Requirements framing implies** (no explicit timeout-step/irreversible-failure/stagnation thresholds in the accessible paper text) — treat the `goal-met/timeout/irreversible-failure/unrecoverable-stagnation` model as this project's own adaptation, not a verbatim citation, when it's built in a future phase.

## Sources

### Primary (HIGH confidence)
- `control/vla_bridge/robot_client.py`, `safety_validator.py`, `io_logger.py`, `action_contract.py`, `stereo_camera.py`, `run_vla_episode.py`, `device_map.json`, `FINDINGS.md` (this project, read in full) — ground truth for existing infrastructure, all three Phase 11 bugs, and the confirmed-live camera bug
- `control/.venv/lib/python3.12/site-packages/lerobot/async_inference/{robot_client.py,configs.py}` (installed `lerobot==0.6.1`) — vendored library internals, confirms `_ready_to_send_observation()` and `chunk_size_threshold` already exist and are already configured
- `.planning/PROJECT.md` — v2.1 milestone scope, Active requirements, Key Decisions
- `pip index versions anthropic` direct PyPI check (2026-09-24) — confirmed `anthropic` 1.8.0 current

### Secondary (MEDIUM confidence)
- Yu & Qiu 2026, arXiv:2606.08881, full HTML text (arxiv.org/html/2606.08881v1) — Pen Transfer success rates, 4-category failure taxonomy, Recovery Rate formula (VLA fine-tuning benchmark, not MLLM prompting)
- "Language Models as Zero-Shot Trajectory Generators," arXiv:2310.11604 — zero-shot no-primitive LLM control pattern, calibration-error as a named failure class
- "Embodied Chain-of-Thought Reasoning," arXiv:2407.08693 — staged reasoning-trace schema
- Async VLA inference research, arxiv.org/html/2605.08168 — execution-horizon/prediction-horizon chunking tradeoff underlying the latency-fix pitfall
- opencv/opencv issues #22901, #26371, #23368 — macOS camera-index instability and AVFoundation resolution-cap bug

### Tertiary (LOW confidence)
- MALMM multi-agent LLM robotics pattern — referenced only from search-result summary, used solely to support an anti-feature argument
- Data-provenance-in-robotics sourcing for Pitfall 7 — thin (patent filings, one blog post); the project's own history (camera bug caught only by manual review) is the stronger evidence

---
*Research completed: 2026-09-24*
*Ready for roadmap: yes*
