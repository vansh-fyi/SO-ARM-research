# Project Research Summary

**Project:** SoARM VLA Research — v2.0 milestone (MLLM-as-robot-controller, real-hardware benchmark replication)
**Domain:** Real-hardware MLLM-driven robot manipulation control, integrated into an existing LeRobot bridge
**Researched:** 2026-09-15
**Confidence:** MEDIUM-HIGH

## Executive Summary

This milestone bolts a zero/few-shot multimodal-LLM control loop onto an already-working `control/` stack (`lerobot[feetech]==0.6.1` driving a real SO-ARM101 over USB serial). The established pattern across the field (SayCan, VoxPoser, Code as Policies, GPT-4V(ision) for Robotics) is consistent and directly applicable: perceive, then call the MLLM for a structured sub-goal (never raw joint targets or free-form code), translate that sub-goal deterministically into servo motion via a small fixed library of motion primitives, execute, observe, repeat. This is a plan-then-execute loop with MLLM calls only at checkpoint granularity (reach/grasp/lift/place), not per control tick, because real hosted-API latency (seconds) makes tick-level calls infeasible — already a locked PROJECT.md decision, corroborated by every piece of prior art reviewed.

The recommended approach layers cleanly on the existing codebase: a new `control/mllm/` subpackage (router, loop, schema, prompts) calling a free HuggingFace-hosted vision model first, a `motion_primitives.py` extracted from the proven `keyboard_joint_control.py` P-control code, an extended `episode_writer.py` recording synced RGB+depth+joints+reasoning-trace+action, and depth computed at decision cadence (not frame rate) via a ported (not imported) version of the already-validated `diagnostics/measure_object_depth.py` SGBM pipeline. Nothing in this design touches the pinned, hardware-validated LeRobot bridge.

The dominant risks are safety and reliability, not novelty: no independent safety envelope between MLLM output and actuators; the free HF tier has no SLA, cold-starts 30-60s, and rate-limits aggressively so the loop needs timeout/backoff/stale-response handling from day one; MLLM structured-output drift must be defensively parsed with a distinct failure bucket; MLLM pixel/relational judgments must never be trusted as final mm-scale coordinates (conversion must go through the calibrated depth pipeline); and zero-shot MLLM results must be reported in a clearly separated table from the paper's fine-tuned-policy baselines with an explicit methodology caveat. All map cleanly onto specific phases below.

## Key Findings

### Recommended Stack

The stack is a deliberately thin delta on top of the already-pinned `control/` venv. `huggingface_hub`'s `InferenceClient` (already a transitive dependency, OpenAI-wire-compatible, supports `image_url` vision content and `provider="auto"` fallback) covers the free-model pilot at zero marginal dependency cost. A hand-rolled ~50-line `MLLMProvider` interface (plain Python ABC/Protocol, one thin adapter per provider) is recommended over `litellm` for a solo research repo with 2-4 providers, avoiding a large/fast-moving dependency surface and preserving exact request/response visibility for real-hardware debugging (architecture research is milder here, treating litellm as a reasonable alternative implementation of the same pattern — a config-level choice, not an architectural fork). `pydantic` (already transitive) gives typed sub-goal schema validation. Depth is stored as 16-bit PNG (or `.npz` if sub-mm float precision is needed). Reasoning traces are plain append-only JSONL — no MLflow/Langfuse/Opik. Add `openai`/`anthropic`/`google-genai` SDKs additively, only when each provider is actually wired in.

**Core technologies:**
- `huggingface_hub.InferenceClient` (already installed): free HF-hosted VLM pilot, zero new dependency
- Hand-rolled `MLLMProvider` interface: provider-agnostic router, avoids heavy abstraction framework
- `pydantic` (transitive): typed sub-goal/action schema, one repair-retry on validation failure
- JSONL (stdlib): append-only reasoning-trace log, one line per MLLM call
- `opencv-python`/`numpy` (already pinned): 16-bit PNG depth frames synced to existing timestamp loop

### Expected Features

**Must have (table stakes, v1 — Pen Transfer end-to-end):**
- Image(s)+instruction → structured sub-goal JSON, strictly validated with retry
- Fixed motion-primitive library (reach/grasp/lift/transport/place/retreat)
- Plan-then-execute loop, one MLLM call per checkpoint
- Full reasoning-trace logging per MLLM call
- Extended episode recorder (RGB + joints + reasoning trace + action; depth once camera fix lands)
- Basic execution-failure detection (timeout, joint-limit, no-progress)
- Provider-agnostic MLLM router, first backend = free HF model
- Pen Transfer task scaffolding

**Should have (differentiators — the research contribution):**
- Automated failure-taxonomy classification (Grasp Instability / Repetition Loop / State Mismatch / Precision Misalignment)
- Recovery Rate computation matching the paper's formula, adapted for plan-then-execute granularity
- Semantic-vs-execution failure aggregation
- Multi-provider comparison
- Full 4-task suite

**Defer (v2+):**
- Closed-loop mid-primitive vision verification, trace analysis/browsing tooling, real-time interactive steering, any IK/motion-planning stack
- Rejected outright: arbitrary code-as-policy execution, per-tick MLLM calls, MLLM fine-tuning, new embedded firmware

### Architecture Approach

A new `control/mllm/` subpackage (router, loop, schema, prompts — the one exception to this repo's flat-script convention) sits above extracted, hardened components: `motion_primitives.py` (pulled from `keyboard_joint_control.py`), `camera_io.py` (pulled from `record_episode.py`), and a new `depth_stereo.py` that ports (never imports across venvs) the SGBM pipeline validated in `diagnostics/measure_object_depth.py`. The loop calls the router once per sub-goal, logs the reasoning trace before execution, resolves the response to joint targets through a local deterministic safety/bounds check, then drives the arm via the unmodified `SO101Follower`. Depth is computed at decision cadence, not frame rate, to avoid reintroducing documented USB/frame-drop flakiness. `record_episode.py` stays untouched; `episode_writer.py` is a separate extended-schema writer.

**Major components:**
1. `control/mllm/{router,loop,schema,prompts}.py` — provider-agnostic call surface + orchestration + schema + prompts
2. `control/motion_primitives.py` — deterministic joint-space P-control executor
3. `control/depth_stereo.py` — decision-cadence stereo depth, backed by `diagnostics/`'s calibration artifact
4. `control/episode_writer.py` + `control/camera_io.py` — extended synced episode schema
5. `control/tasks/*.py` — per-task config (data, not code)
6. `control/metrics/failure_taxonomy.py` — post-hoc pass over collected episodes, built last

### Critical Pitfalls

1. **No safety envelope between MLLM output and actuators** — hard joint/velocity clamps, workspace bounds, watchdog timeout must live in the local controller below the MLLM interface, architected in from the start.
2. **Free HF tier treated as normal low-latency API** — no SLA, 30-60s cold starts, aggressive rate limits; build timeout/backoff/stale-response rejection into the router from day one.
3. **MLLM structured-output brittleness** — parse defensively, never guess a default on failure, log as a distinct failure mode.
4. **Pixel-space MLLM output treated as calibrated mm coordinates** — never trust model-emitted world-frame coordinates; convert deterministically through the calibrated depth pipeline; hard-gated on the depth-camera fix landing first.
5. **Zero-shot vs. fine-tuned-baseline comparison without caveats** — report in a separated table, adapt Recovery Rate definition explicitly, give infrastructure failures their own bucket.
6. (Secondary) Blind open-loop execution between MLLM calls risks acting on stale world state; reasoning-trace timestamps must capture request-sent/response-received/execution-complete as distinct events.

## Implications for Roadmap

### Phase 1: Depth Camera Reliability Fix
**Rationale:** Every downstream MLLM spatial-grounding decision and the pixel→mm conversion pitfall depend on knowing what a reliable depth reading looks like.
**Delivers:** Validated AR0144 stereo depth, signed-off UAT.
**Avoids:** Pitfall 4; Anti-Pattern "starting MLLM work before depth is reliable."

### Phase 2: MLLM Router + Plan-Then-Execute Loop Skeleton
**Rationale:** Can be dry-run (RGB-only, stubbed sub-goals) immediately after/parallel to Phase 1, establishing the load-bearing sub-goal JSON schema before any provider-specific code exists.
**Delivers:** `control/mllm/{router,schema,prompts,loop}.py`, `control/motion_primitives.py`, safety clamp layer, timeout/backoff handling.
**Uses:** `huggingface_hub.InferenceClient`, hand-rolled `MLLMProvider` interface, `pydantic`.
**Avoids:** Pitfalls 1, 2, 3.

### Phase 3: Recorder Extension
**Rationale:** Needs Phase 2's real MLLM calls to log and Phase 1's validated depth to wire in correctly.
**Delivers:** `control/camera_io.py`, `control/episode_writer.py`, `control/depth_stereo.py` at sub-goal cadence, request/response/execution-complete timestamp triad.
**Avoids:** Pitfall 6 (reasoning-trace/sensor desync).

### Phase 4: Pen Transfer End-to-End
**Rationale:** Validates the entire chain on the paper's simplest task before investing in remaining tasks.
**Delivers:** Real multi-episode runs on physical arm, hardened parsing against real scenes.
**Addresses:** All table-stakes features.

### Phase 5: Remaining 3 Paper Tasks
**Rationale:** Should require only new `tasks/*.py` configs if Phases 2-4 were built generically.
**Delivers:** Selective Color Sorting, Multi-Object Packing, Precision Pen Placement.

### Phase 6: Failure Taxonomy + Recovery Rate Metrics
**Rationale:** Deliberately last — needs a real corpus of successes/failures across tasks.
**Delivers:** `control/metrics/failure_taxonomy.py`, Recovery Rate computation, semantic-vs-execution aggregation, separated results reporting.
**Avoids:** Pitfall 5.

### Phase 7: Multi-Provider Comparison
**Rationale:** The payoff of the Phase 2 router abstraction — should be config-only.
**Delivers:** Second (paid) provider wired in, cross-provider comparison runs.

### Phase Ordering Rationale

- Depth-fix-first is a hard, explicitly locked dependency across PROJECT.md, architecture, and pitfalls research.
- Router/loop-skeleton before recorder-extension: nail down the load-bearing schema contract before wiring it into the harder synced-recording problem.
- Pen Transfer validates the full pipeline before investing in 3 more tasks or a second provider.
- Failure taxonomy/Recovery Rate last: needs episode volume across tasks to be meaningful.
- Multi-provider last: don't burn paid-API budget until the harness is proven on the free model.

### Research Flags

Needs deeper research during planning:
- **Phase 2:** free HF model selection/vision-input support changes weekly — verify the specific pilot model empirically at plan time.
- **Phase 6:** the target paper's exact per-trial failure-labeling procedure is not public — needs an explicit, documented labeling methodology decision.

Standard patterns (skip research-phase):
- **Phase 1:** already in progress with a documented UAT plan.
- **Phase 3:** extending an existing, proven recorder pattern.
- **Phase 4-5:** primitive/task-config pattern well-established from architecture research; mostly physical setup.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM-HIGH | Versions/pricing verified live against installed venv and current HF docs; pilot model name flagged as fast-changing |
| Features | MEDIUM | Architecture patterns well-established and cross-checked; paper's exact failure-label procedure not public (LOW on that specific point) |
| Architecture | HIGH (integration surface) / MEDIUM (MLLM-loop design) | Grounded in installed LeRobot source and existing repo code; no proven reference implementation exists for this exact MLLM-loop shape |
| Pitfalls | MEDIUM | Cross-checked academic sources on LLM-robot safety and API limits; no single authoritative gotchas doc for this exact stack |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- Free HF model choice/availability drift — re-verify at Phase 2 planning time, not locked now.
- Paper's exact failure-annotation procedure — Phase 6 must define and document its own methodology.
- litellm vs. hand-rolled router — low-stakes divergence between stack and architecture research; resolve during Phase 2 planning based on provider count.
- Depth precision format (16-bit PNG vs. `.npz`) — revisit once Phase 1's UAT establishes the achievable noise floor.

## Sources

### Primary (HIGH confidence)
- `control/.venv` installed package versions (`pip list`)
- `control/.venv/.../lerobot/robots/so_follower/so_follower.py` (installed LeRobot 0.6.1 source)
- `control/record_episode.py`, `control/keyboard_joint_control.py`, `control/joint_jog.py`
- `diagnostics/UAT/function/depth/UAT.md`, `diagnostics/UAT/function/basic/UAT.md`, `diagnostics/measure_object_depth.py`, `diagnostics/stereo_calibrate.py`
- [Benchmarking Vision-Language-Action Models on SO-101 (arXiv:2606.08881)](https://arxiv.org/abs/2606.08881)
- [Hugging Face Inference Providers Pricing/Billing](https://huggingface.co/docs/inference-providers/pricing) and [Chat Completion task docs](https://huggingface.co/docs/inference-providers/tasks/chat-completion)
- `.planning/PROJECT.md`

### Secondary (MEDIUM confidence)
- [Code as Policies (arXiv 2209.07753)](https://arxiv.org/abs/2209.07753), [SayCan (arXiv 2204.01691)](https://arxiv.org/pdf/2204.01691), [VoxPoser (arXiv 2307.05973)](https://arxiv.org/abs/2307.05973), [GPT-4V(ision) for Robotics (arXiv 2311.12015)](https://arxiv.org/abs/2311.12015), [ReAct (arXiv 2210.03629)](https://arxiv.org/html/2210.03629v3)
- [On the Vulnerability of LLM/VLM-Controlled Robotics (arXiv 2402.10340)](https://arxiv.org/pdf/2402.10340), [Safety Guardrails for LLM-Enabled Robots (arXiv 2503.07885)](https://arxiv.org/pdf/2503.07885)
- [LiteLLM GitHub/docs](https://github.com/BerriAI/litellm)
- HuggingFace free-tier rate-limit/cold-start behavior, synthesized from third-party overviews

### Tertiary (LOW confidence)
- Target paper's exact failure-annotation adjudication procedure
- Specific free HF pilot model name/hosting provider (changes weekly)

---
*Research completed: 2026-09-15*
*Ready for roadmap: yes*
