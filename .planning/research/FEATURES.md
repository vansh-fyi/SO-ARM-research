# Feature Research

**Domain:** MLLM-as-robot-controller (raw-autonomy design) for a real SO-ARM101, benchmarked against a trained VLA
**Researched:** 2026-09-24
**Confidence:** MEDIUM (ecosystem patterns cross-checked across 2-3 independent sources each; the one project-specific paper was read directly from its own HTML full-text and abstract pages — treat all web-sourced claims below as MEDIUM, not HIGH, per this repo's confidence-classification seam, which floors uncurated web search/fetch at LOW and this agent raised only cross-checked items to MEDIUM)

## Critical Finding First: What Yu & Qiu 2026 (arXiv:2606.08881) Actually Is

This paper is titled **"Benchmarking Vision-Language-Action Models on SO-101: Failure and Recovery Analysis"** (Yi Yu, Xinchuan Qiu). Read directly from [arxiv.org/abs/2606.08881](https://arxiv.org/abs/2606.08881) and the full HTML text at [arxiv.org/html/2606.08881v1](https://arxiv.org/html/2606.08881v1).

**It does NOT prompt a general-purpose MLLM for raw joint control.** It fine-tunes and evaluates four trained policies — **π0.5, SmolVLA, Wall-X, and ACT** — directly on a physical SO-101. There is no GPT-4V/Claude/Gemini raw-autonomy arm in this paper. This means: the "MLLM raw-JSON control loop" design for this milestone has **no direct methodological precedent in the cited paper** — it must be designed from general MLLM-as-controller ecosystem patterns (below), not copied from the paper.

What the paper *does* give this project, confirmed directly:

| Element | Paper's content | Use for this milestone |
|---|---|---|
| **Pen Transfer task** | One of 4 tasks; 70-95% success rate across all four evaluated policies — confirmed as the simplest/highest-success task, matching PROJECT.md's framing | Adopt task definition/success criteria; the paper doesn't give a pen/target spec beyond "pick-and-place, control fidelity" framing — this project's own task setup still needs to be authored |
| **Failure taxonomy** | 4 categories: **Grasp Instability**, **Repetition Loop**, **State Mismatch**, **Precision Misalignment** | Directly reusable vocabulary for tagging MLLM failure modes during comparison (already listed in PROJECT.md Future Requirements — confirmed as the paper's actual categories, not paraphrased) |
| **Recovery Rate metric** | `N_successful_recovery / N_recovery_opportunity`; a recovery = policy restores valid task state after failure and continues without intervention | Reusable metric definition once episode failure/recovery tagging exists (deferred beyond v2.1 per PROJECT.md) |
| **Termination criteria** | Paper states success requires "final object state satisfies predefined task completion criteria without human intervention" — but does **not** disclose explicit timeout-step / irreversible-failure / stagnation thresholds in the accessible text | The `goal-met / timeout / irreversible-failure / unrecoverable-stagnation` termination model referenced in PROJECT.md Future Requirements is **not verbatim from the paper's disclosed text** — treat as this project's own adaptation inspired by the paper's framing, not a confirmed direct citation |
| **Prompt/action schema for any model** | Paper explicitly does not disclose exact prompt templates or JSON/action formats for any of the 4 policies — only says "minimal interface-level adaptations" aligned each to SO-101's observation/action space | No schema to borrow from the paper at all — schema design is this project's own, informed by the general patterns below |

**Implication for FEATURES below:** everything about *how* to structure the MLLM raw-JSON control loop (prompt shape, chunking, reasoning capture) draws on the general MLLM-as-robot-controller literature, not on Yu & Qiu 2026. Everything about *what task to run and how to score/tag it* draws on Yu & Qiu 2026.

## Ecosystem Patterns for MLLM-as-Raw-Controller (No Pre-Built Primitives)

Three lines of prior work are directly relevant to "reason from raw I/O straight to joint/pose-level output, no `move_to()`/`grasp()` helpers":

1. **"Language Models as Zero-Shot Trajectory Generators"** ([arxiv.org/abs/2310.11604](https://arxiv.org/pdf/2310.11604), Zhu Xian Wang et al., UCL) — an LLM (GPT-4-class) given only object-detector output (3D bounding boxes) + camera calibration + a task-agnostic prompt (no in-context examples, no motion primitives) generates a **dense sequence of end-effector poses** for 26 real-world manipulation tasks. Confirms this class of system is achievable zero-shot without primitives. Key structural facts: (a) reasoning text is generated **before** the pose/code output, explicitly separated so it's interpretable independent of the executable output; (b) the loop is closed via an external object-tracker that re-detects scene state and can trigger a re-query if the LLM's self-assessed outcome looks wrong; (c) the five reported failure classes are gripper-pose error, task-planning error, trajectory-generation/code error, object-detection error, and **camera-calibration error** — calibration-data quality in the prompt is a first-order failure source, not an afterthought.
2. **Embodied Chain-of-Thought (ECoT)** ([arxiv.org/abs/2407.08693](https://arxiv.org/abs/2407.08693)) — trains (rather than zero-shot-prompts) a VLA to emit structured reasoning before its action token: task rephrasing → high-level plan → current sub-task → move description → gripper position → labeled object bounding boxes, in that order, then the action. Directly informs **reasoning-trace schema** for this project: even though this milestone's MLLM is zero-shot prompted (not trained) into reasoning, structuring the captured trace with the same stages (restate task → plan → current sub-step → grounded visual claims → action) makes the trace analyzable/comparable rather than free-form prose. ECoT's own paper reports this structuring also gives ~28-point absolute success-rate gains for trained VLAs over untrained reasoning, and makes failures human-diagnosable — the mechanism ("reasoning grounded in visible scene facts, not just semantics") is the transferable idea for a prompted (not trained) system too.
3. **Multi-agent / closed-loop LLM control frameworks (MALMM et al.)** — split planning into separate high-level-plan, low-level-control, and supervisor/verifier LLM roles instead of one call doing everything. Relevant mainly as an anti-pattern check: this milestone's design (one Claude call reasoning straight to a JSON action chunk) is intentionally simpler and is exactly the "no scaffolding" condition the milestone wants to test — multi-agent decomposition would reintroduce the kind of task-level scaffolding the experiment is designed to avoid, so it belongs in Anti-Features here even though it's a legitimate pattern in general robotics-LLM literature.

General, provider-agnostic pattern for the prompt itself, converged on across all three sources plus standard LLM tool-use practice: task instruction in natural language; an explicit machine-readable output schema (JSON schema / function-calling tool definition, not "please output JSON" in prose — this is now standard practice for reliable structured output from frontier MLLMs, including Claude's tool-use/structured-output mode); static calibration/robot-geometry data given once (joint limits, coordinate frame, link lengths) since it doesn't change per step; current proprioceptive state (joint positions) given fresh every call; and current camera frame(s) given fresh every call. This matches exactly the I/O shape PROJECT.md already commits to for v2.1.

## Feature Landscape

### Table Stakes (Required for the Experiment to Be Valid/Comparable)

Missing any of these makes the MLLM run not a fair, analyzable comparison against the SmolVLA baseline.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Explicit output JSON schema (not free-text parsing) | Both zero-shot-trajectory literature and standard LLM tool-use practice converge on structured/constrained output as the only reliable way to get machine-executable actions from an MLLM; free-text parsing is a known source of silent action-format drift | LOW | Use Claude's native tool-use/structured-output mode with a schema mirroring the existing `action_contract.py` joint-order/units convention so the safety validator needs no new adapter |
| Static calibration + robot geometry in the prompt (once, or cached per episode) | All three ecosystem sources treat calibration/geometry as prompt input, and the UCL trajectory-generator paper lists calibration error as one of only 5 reported failure classes — it is load-bearing, not decorative | LOW | Already available per PROJECT.md — reuse existing bridge calibration data, don't re-derive |
| Fresh joint-state + camera frame(s) per reasoning call | Matches the VLA's own I/O contract (Phase 11) — required for an apples-to-apples comparison; also matches the general pattern (state/image refreshed every call, calibration/schema static) | LOW | Already available via existing bridge observation path per PROJECT.md |
| Action-chunking (not per-tick MLLM calls) | Per-tick API latency is prohibitive (already root-caused as the bridge's core problem in Phase 11); zero-shot-trajectory literature also generates a full pose sequence per call, not one pose per call | MEDIUM | Chunk size must come from the latency-fix's real timing data, not be guessed (already decided in PROJECT.md decision log) |
| Reasoning text captured *before*, and separately from, the action JSON | Universal pattern across UCL trajectory-generator (reasoning before code) and ECoT (reasoning before action token) — this is also literally the artifact this milestone exists to produce, since the VLA baseline has none | LOW-MEDIUM | Extend `io_logger.py`; store as a distinct field per action chunk, not interleaved with/inferred from the action JSON |
| Structured (not free-prose) reasoning trace shape | ECoT shows unstructured "vibes" reasoning is harder to compare/diagnose than staged reasoning (restate task → plan → grounded scene claims → chosen action) | MEDIUM | This is a prompt-design choice (ask the model to reason in stages), not an infra build — cheap to do, meaningfully improves later analysis |
| Safety-validator pass-through unchanged | Already committed in PROJECT.md — the MLLM's JSON goes through the same `safety_validator.py` gate as the VLA; this is what makes the comparison apples-to-apples rather than "trained policy under a safety net vs. unconstrained model" | LOW (already exists) | No new work — explicitly a dependency, not a build item, for this milestone |
| Same episode-level logging schema as the VLA baseline (frames, joint state, instruction, raw/validated/executed action, latency, termination reason) | Required for the direct-comparison deliverable — comparing against Phase 11's episode only works if both episodes share a schema, with reasoning trace as the one added field | LOW (already exists, extend) | `io_logger.py` already captures everything except reasoning text per PROJECT.md — one field to add, not a new logger |

### Differentiators (Where This Experiment Produces Something New)

Not required for a minimally valid run, but where the interesting research signal comes from.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Reasoning trace vs. executed-action divergence analysis | The core research question is "what does a general reasoning model do with only floor-level I/O" — the payoff is reading *where* the model's stated plan diverges from what it actually emitted/what the safety validator had to clamp, not just success/fail | MEDIUM | Needs the reasoning trace (table stakes) plus a simple side-by-side view; no new capture infra, just analysis of what's already logged |
| Failure-taxonomy tagging using Yu & Qiu's 4 categories (Grasp Instability, Repetition Loop, State Mismatch, Precision Misalignment) | Borrowing the paper's own vocabulary makes the MLLM-vs-VLA comparison legible against the paper's own results table, not just internally consistent | LOW-MEDIUM | Currently human-judged per PROJECT.md (no automated vision-based detection yet) — tagging is a manual/semi-manual pass over the reasoning trace + video, not new code |
| Chunk-boundary self-report ("what I expect the scene to look like after this chunk") | Cheap to add to the schema now (one extra string field) and gives a natural checkpoint for judging whether the model's model-of-the-world matches reality — a known failure axis per the UCL paper (object-detection/state errors) | LOW | Optional schema field; do not gate the milestone on it, but worth including since it's nearly free once the schema exists |

### Anti-Features (Look Appealing Here, Actively Wrong for This Experiment)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|------------------|-------------|
| Pre-built movement primitives / tool-calling helpers (`move_to()`, `grasp()`, IK solver as a callable tool) | Would almost certainly raise task success rate and reduce engineering risk | Directly contradicts the milestone's explicit purpose — PROJECT.md is unambiguous that this is "raw autonomy," testing what the model does with only floor-level I/O; adding helpers collapses this into a different (already-known-to-work) experiment | None — this is the one line the milestone must not cross. If future work wants a scaffolded-agent comparison, make it a clearly separate, later experiment |
| Multi-agent decomposition (separate planner/controller/verifier LLM roles, à la MALMM) | Legitimate, published pattern that generally improves success rate over single-call raw reasoning | Reintroduces task-level scaffolding/role structure the milestone is explicitly trying to avoid measuring around; also multiplies API cost and latency on an already latency-constrained bridge | Single Claude call reasoning straight to the JSON schema, as scoped |
| Few-shot in-context examples of correct action sequences | Tempting to improve output reliability/format-following | Muddies the "what does a general reasoning model do zero-shot" question — success becomes partly a measure of how well it imitated the examples, not how it reasoned from raw I/O; the UCL trajectory-generator paper deliberately used zero in-context examples for the same reason | Rely on the explicit JSON schema (tool-use/structured-output mode) for format reliability instead of examples — schema constraints solve the format problem without contaminating the reasoning question |
| Code-generation-as-policy (LLM writes Python that computes joint targets, à la "Code as Policies") | Popular, well-published pattern (UCL paper itself uses LLM-generated Python for trajectory math) | Introduces an arbitrary-code-execution surface exactly where PROJECT.md already made the explicit decision (2026-09-24) that the model must emit validated JSON only, reusing `safety_validator.py` rather than building new sandboxing | Keep the model's output as data (joint targets/deltas), not code; any math the model needs to do, it does in its reasoning text and then emits numbers |
| Per-tick (high-frequency) MLLM calls | Would make the control loop feel more "reactive," closer to the VLA's control frequency | API latency makes this infeasible on this bridge (root cause of the exact bug being fixed earlier in this same milestone) and was already explicitly rejected in PROJECT.md's decision log in favor of sub-goal/chunk-level calls | Chunked JSON action sequences per call, chunk size set from real latency-fix timing data |
| Automated vision-based success/failure detection for this run | Would remove human-judgment subjectivity from the comparison | Explicitly deferred in PROJECT.md ("human-judged for the first version") — building it now is scope creep the milestone doesn't need for a single-task, single-comparison run | Human-judged termination/success per PROJECT.md's stated v2.1 scope; revisit when the full 4-task suite is built |
| Fine-tuning or few-shot-training the MLLM on collected episodes | Would likely improve success rate on Pen Transfer specifically | Explicitly Out of Scope in PROJECT.md — this milestone evaluates prompted behavior, not trained behavior; fine-tuning would also stop this from being a fair "general reasoning model" comparison | Prompting only, as scoped |

## Feature Dependencies

```
Bridge tick-latency fix (Active, in progress)
    └──produces timing data for──> Action-chunk sizing (Table Stakes)
                                        └──required by──> MLLM raw-JSON control loop (Table Stakes)

safety_validator.py (existing, Phase 11)
    └──reused unchanged by──> MLLM raw-JSON control loop (Table Stakes)

action_contract.py joint-order/units convention (existing, Phase 11)
    └──schema mirrors──> Explicit output JSON schema (Table Stakes)

io_logger.py episode.jsonl schema (existing, Phase 11)
    └──extended by──> Reasoning-trace capture (Table Stakes)
                            └──enables──> Reasoning trace vs. executed-action divergence analysis (Differentiator)
                            └──enables──> Failure-taxonomy tagging (Differentiator)

Phase 11 SmolVLA baseline episode (existing, 11-05-retry-20260924-120530)
    └──required for──> Pen Transfer comparison run (Table Stakes)

camera_overhead name-based-resolution fix (Active, in progress)
    └──must land before──> any recorded episode is trusted for visual/comparison review
```

### Dependency Notes

- **Action-chunk sizing requires the tick-latency fix's timing data first:** PROJECT.md already sequences this explicitly — chunking cannot be responsibly sized without knowing real measured round-trip latency, so the latency fix must land (and be measured) before the MLLM loop's chunk size is finalized.
- **The MLLM loop reuses `safety_validator.py` and `action_contract.py` unchanged, not as new work:** both are existing Phase 10/11 infrastructure. The only new schema work is making the MLLM's JSON *shape* match what those two already expect — no validator changes needed since the model only ever emits validated JSON (already decided, superseding the earlier sandboxed-code-execution idea).
- **Reasoning-trace capture only produces its value (the two Differentiators) once it exists in a structured, staged form** — a raw unstructured text blob technically satisfies "capture reasoning" but forecloses the comparison/tagging analysis that's the actual point, per ECoT's finding that staged reasoning is what makes failures diagnosable.
- **The comparison run depends on the existing Phase 11 baseline episode being the reference**, not a new VLA run — PROJECT.md is explicit this is a direct comparison against `11-05-retry-20260924-120530`, so no new VLA data collection is in scope here.
- **The camera-recording fix is a correctness dependency, not a feature dependency**, but it gates trustworthy visual review of *any* new episode (MLLM or otherwise) — it's listed because a Pen Transfer comparison run recorded before this fix lands would be unusable for visual analysis, even if the control loop itself works.

## MVP Definition

### Launch With (v2.1, as already committed in PROJECT.md)

- [ ] MLLM raw-JSON control loop: task instruction + JSON schema + calibration + joint state + camera frames → chunked JSON actions, no movement primitives — this is the entire experiment; nothing else in this milestone matters without it
- [ ] Reasoning-trace capture extending `io_logger.py` — the one artifact the VLA baseline structurally cannot produce, and the milestone's point of comparison
- [ ] Reuse `safety_validator.py` and `action_contract.py` unchanged — zero new safety infra, by design
- [ ] Pen Transfer run, human-judged success/termination, directly compared against the Phase 11 SmolVLA episode

### Add After Validation (v2.x, once the first Pen Transfer comparison is in hand)

- [ ] Failure-taxonomy tagging using Yu & Qiu's 4 categories — trigger: once at least one MLLM episode exists to tag, do this before expanding to more tasks, since it's cheap and makes the single-episode comparison more legible
- [ ] Reasoning-vs-execution divergence analysis as a repeatable review step, not a one-off read — trigger: once 2+ MLLM episodes exist (Pen Transfer plus at least one more task) so patterns, not single-episode noise, can be identified
- [ ] Chunk-boundary self-report field — trigger: cheap enough to add opportunistically whenever the schema is next touched, no need to gate on it

### Future Consideration (beyond v2.1, already flagged as Future Requirements in PROJECT.md)

- [ ] Provider-agnostic MLLM router (OpenAI/Gemini/HF-hosted) — defer until the single-provider (Claude) loop is proven; PROJECT.md already plans to pilot on a free-tier model first
- [ ] Full 4-task suite (Selective Color Sorting, Multi-Object Packing, Precision Pen Placement) — defer until Pen Transfer validates the pipeline; the paper's own data shows these are markedly harder (0-55% success vs. Pen Transfer's 70-95%), so the MLLM loop should prove itself on the easy case first
- [ ] Recovery Rate metric + automated episode-termination modeling (goal-met/timeout/irreversible-failure/unrecoverable-stagnation) — defer until failure tagging (above) has been done manually at least once; automating this before that risks encoding wrong assumptions about what "recovery" looks like for an MLLM (vs. the paper's trained-policy context it was defined for)
- [ ] Automated vision-based success/failure detection — explicitly deferred in PROJECT.md; human judgment first
- [ ] Cross-episode memory / non-independent trials — explicitly out of scope; PROJECT.md wants paper-faithful independent trials first

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| MLLM raw-JSON control loop | HIGH | MEDIUM (schema + prompt design; execution/validation infra already exists) | P1 |
| Reasoning-trace capture | HIGH | LOW (one field/path added to existing logger) | P1 |
| Action-chunk sizing from real latency data | HIGH (blocks P1 above) | LOW (measurement + a config value, once latency fix lands) | P1 |
| Pen Transfer comparison run | HIGH | LOW (infra reused; one episode to run and review) | P1 |
| Structured (staged) reasoning format in the prompt | MEDIUM | LOW (prompt wording only) | P1 (bundle into control loop — costs nothing extra to do right the first time) |
| Failure-taxonomy tagging | MEDIUM | LOW | P2 |
| Divergence analysis (reasoning vs. executed) | MEDIUM | LOW-MEDIUM (manual review process) | P2 |
| Chunk-boundary self-report field | LOW | LOW | P3 |
| Provider-agnostic router | MEDIUM (research value) | MEDIUM-HIGH | P3 |
| Full 4-task suite | HIGH (long-term) | HIGH | P3 |
| Recovery Rate + full termination taxonomy | MEDIUM | MEDIUM | P3 |

**Priority key:**
- P1: Required for v2.1 as already committed in PROJECT.md
- P2: Should have once the first comparison exists, low additional cost
- P3: Explicitly deferred in PROJECT.md's Future Requirements

## Comparable-System Feature Analysis

| Feature | UCL Zero-Shot Trajectory Generator (2310.11604) | ECoT (2407.08693) | This Project's v2.1 Design |
|---------|--------------------------------------------------|--------------------|-----------------------------|
| Output format | LLM-generated Python code computing dense pose sequences | Trained token output (plan → subtask → move → action) | Structured JSON (schema/tool-use), joint targets/deltas — data, not code |
| Pre-built primitives | None (explicit zero-shot, no primitives) | N/A (trained policy, not prompted) | None (explicit design requirement) |
| In-context examples | None (deliberately zero-shot) | N/A (trained, not prompted) | None planned — schema constrains format instead |
| Reasoning capture | Natural-language reasoning generated before code, explicitly separated | Structured stages generated before action token, trained end-to-end | Structured stages (task restate → plan → grounded scene claims → action), zero-shot prompted, logged separately per chunk |
| Closed-loop correction | External object-tracker re-detects scene, can trigger re-query | N/A (single-pass trained inference) | Not in v2.1 scope — single chunk per call, no self-correction loop (candidate for a later ablation, not committed) |
| Safety gate | Not described (research-lab setting) | N/A | Existing `safety_validator.py`, reused unchanged |

## Sources

- [Yu & Qiu 2026, "Benchmarking Vision-Language-Action Models on SO-101: Failure and Recovery Analysis," arXiv:2606.08881, abstract](https://arxiv.org/abs/2606.08881) — read directly, MEDIUM confidence (tool-mediated fetch of primary source, cross-checked against the full HTML text below)
- [Yu & Qiu 2026, full HTML text, arXiv:2606.08881v1](https://arxiv.org/html/2606.08881v1) — read directly, MEDIUM confidence; source of the Pen Transfer success-rate range, the 4-category failure taxonomy, and the Recovery Rate formula
- ["Language Models as Zero-Shot Trajectory Generators," arXiv:2310.11604](https://arxiv.org/pdf/2310.11604) (UCL, robot-learning.uk project page consulted since the PDF itself wasn't machine-text-extractable) — MEDIUM confidence, cross-checked between the project page summary and arXiv search result descriptions
- ["Robotic Control via Embodied Chain-of-Thought Reasoning," arXiv:2407.08693](https://arxiv.org/abs/2407.08693) (Zawalski et al.) and [project page](https://embodied-cot.github.io/) — MEDIUM confidence, cross-checked across project page and search-result summaries
- MALMM (Multi-Agent LLMs for Zero-Shot Robotic Manipulation) — LOW-MEDIUM confidence, referenced only from search-result summary, not independently fetched; used only to support an anti-feature argument, not a load-bearing design claim
- `.planning/PROJECT.md` — primary internal source for existing infrastructure, decisions already made, and exact v2.1 scope boundaries

---
*Feature research for: MLLM-as-robot-controller raw-autonomy design, SO-ARM101 v2.1 milestone*
*Researched: 2026-09-24*
