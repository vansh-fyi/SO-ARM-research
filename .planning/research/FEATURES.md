# Feature Research

**Domain:** MLLM-as-robot-controller (zero/few-shot vision-language-model task planning + control on a real low-cost arm), replicating a specific benchmark methodology
**Researched:** 2026-09-15
**Confidence:** MEDIUM (architecture patterns are well-established in the literature and cross-checked across multiple sources; the target paper's exact annotation/adjudication procedure for failure labels is not fully public, so some methodology-replication details are inferred/LOW and flagged as such)

## How MLLM-as-Controller Systems Work in Practice

Across the literature (Code as Policies, SayCan/Inner Monologue, VoxPoser, PaLM-E, RT-2, GPT-4V(ision) for Robotics, ReAct), the same architectural skeleton recurs, and it maps directly onto what this milestone needs to build:

1. **Perception → context assembly.** Camera frame(s) (+ optionally depth, joint state, task history) are packaged into a multimodal prompt alongside the natural-language task instruction. Nothing here is trained; it's context construction.
2. **MLLM call → structured output.** The model is prompted (few-shot or zero-shot, with an explicit output schema) to produce either (a) executable code/API calls (Code as Policies, VoxPoser), (b) a discrete next-skill choice from a fixed library (SayCan), or (c) a structured sub-goal/waypoint description in JSON (most GPT-4V robotics demos, RT-2-style discretized actions). For a general-purpose MLLM (not a fine-tuned VLA), option (b)/(c) — structured sub-goal JSON consumed by a deterministic executor — is by far the dominant and most reliable pattern, because raw joint-angle regression from an MLLM is unreliable and code-generation-as-policy needs a much richer simulated/API action space than a 6-DOF arm has.
3. **Grounding/translation layer.** The sub-goal (e.g., "grasp the red pen at approximately (x,y)") is translated into actual robot commands by a *separate, non-MLLM* component: a fixed library of parameterized motion primitives (reach, grasp, lift, place, retreat) executed via the low-level controller. This is the layer SayCan calls "Can" and VoxPoser calls the motion planner — the MLLM never directly emits servo positions.
4. **Execution + observation feedback.** The primitive runs to completion (or a fixed timeout), a new observation (camera frame, gripper state, sometimes success/failure heuristic) is captured, and this is what makes the loop "ReAct-style" / Inner-Monologue-style rather than open-loop: the next MLLM call sees the *result* of the last action, not just the original instruction, so it can notice slippage, replan, or declare done.
5. **Reasoning trace as first-class output**, not a side effect — every one of these papers logs the model's textual rationale per step; it's what makes failure analysis (this milestone's whole point) possible after the fact.

**Prior art directly reusable as design references (not to be re-implemented from scratch, but pattern sources):**
- **Code as Policies** (arXiv 2209.07753) — LLM writes Python using a provided perception/motion API; demonstrates the "plan is a program with control flow" pattern. Relevant here mainly as a rationale for *why not* to let the MLLM emit raw code against the LeRobot API directly (arbitrary code execution against real servos is a safety anti-feature — see below).
- **SayCan / Do As I Can, Not As I Say** (arXiv 2204.01691) — the "Say" (LLM proposes/scores) + "Can" (affordance/feasibility check) split. Directly informs this milestone's plan-then-execute split between "MLLM proposes sub-goal" and "local controller checks reachability/feasibility before executing."
- **VoxPoser** (arXiv 2307.05973) — LLM-authored spatial value maps feeding a classical motion planner; supports this milestone's decision to keep the MLLM at the sub-goal level and use a deterministic executor underneath, not have the MLLM drive joints directly.
- **GPT-4V(ision) for Robotics** (arXiv 2311.12015) — closest existing precedent to this milestone's exact ask: a general-purpose multimodal model (not a trained VLA) used purely via prompting/few-shot for real-robot manipulation, reporting 85-95% end-to-end success on demonstrated tasks with zero additional training. Confirms the "plan-then-execute with an off-the-shelf MLLM" approach is viable, not speculative.
- **ReAct / Inner Monologue** — the Thought→Action→Observation loop is the correct shape for the "reasoning-trace-first" requirement and for closed-loop replanning after a detected failure (needed for the Recovery Rate metric).
- **RT-2 / PaLM-E** — relevant as contrast, not pattern: these are *fine-tuned* VLAs that output actions directly from a trained checkpoint. Explicitly out of scope per this milestone (no fine-tuning); mentioned here only so the roadmap doesn't accidentally drift toward re-deriving them.

## Feature Landscape

### Table Stakes (Users Expect These)

Non-negotiable for a working v1 that can run one MLLM-controlled episode end-to-end.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Image(s) + task instruction → structured sub-goal/plan output | This is the core loop; without it there's no MLLM control at all | MEDIUM | Define a strict JSON schema (sub-goal name, target description, params) the MLLM must emit; validate/reject malformed output with a retry, don't trust free text |
| Fixed library of motion primitives (reach, grasp, lift, place, retreat/home) | The MLLM cannot safely emit raw joint targets; every reviewed system (SayCan, VoxPoser, GPT-4V-robotics) inserts a deterministic executor between plan and servos | MEDIUM-HIGH | Built on top of `control/`'s existing `SO101Follower.get_observation()`/`send_action()` + `max_relative_target` clamping; no IK solver exists yet in this repo — needs at least simple waypoint interpolation for reach/place, informed by depth-camera object position |
| Plan-then-execute loop with sub-goal-level MLLM calls | Real API latency (seconds) makes per-tick MLLM calls impractical; already decided in PROJECT.md | LOW (orchestration only, given primitives exist) | One MLLM call per checkpoint (reach→grasp→lift→place), not per control tick |
| Full reasoning-trace logging per MLLM call | Explicit first-class requirement in PROJECT.md; also required to do any failure-taxonomy analysis after the fact | LOW | Log raw prompt, raw response (including rationale), parsed sub-goal, timestamp — one record per call, keyed to episode + step index |
| Synced episode recording (RGB + depth + joints + reasoning trace + action) | Already partially built (`control/record_episode.py`); this milestone's dataset *is* the deliverable | MEDIUM | Extend existing recorder rather than rewrite: add a reasoning-trace column/file and raw depth capture; depth camera reliability is a blocking dependency (see below) |
| Basic execution-failure detection (timeout, joint limit hit, no visual state change, gripper never closes) | Needed to know a "recovery opportunity" occurred at all — without any failure signal, Recovery Rate is undefined | MEDIUM | Doesn't need to be sophisticated for v1: hard timeouts per primitive + gripper current/position sanity checks + "did the tracked object's pixel/depth position change" heuristic are enough to flag "something went wrong, ask MLLM to look again" |
| Provider-agnostic MLLM router (single call interface, swappable backend) | Explicit v2.0 requirement; also decouples pipeline development from any one vendor's rate limits/cost | LOW-MEDIUM | Thin adapter: one function `call_mllm(images, instruction, history) -> raw_text`, with per-provider request/response mapping behind it; start with one free HF-hosted multimodal model |
| Pen Transfer task scaffolding (props, fixed camera framing, pick-up-and-hand-off script for the human/other side of the task if needed) | Explicit starting task per PROJECT.md; paper's easiest/highest-success task, so it's the right complexity floor to validate the loop | LOW-MEDIUM | Mostly physical/procedural setup work, not software; unblocks everything else |

### Differentiators (Competitive Advantage / Research Value)

Not required for a single working episode, but this is where the milestone's actual research contribution lives — comparability to the paper is the whole point.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Automated failure-taxonomy classification (Grasp Instability / Repetition Loop / State Mismatch / Precision Misalignment) | Lets the benchmark scale past a handful of hand-labeled trials and makes results reproducible/auditable | MEDIUM-HIGH | The paper's own methodology for *how* it assigns "one dominant primary failure mode per failed trial" isn't published in detail (LOW confidence on exact procedure) — safest v1 approach is a human-in-the-loop labeling pass using the reasoning trace + video as evidence, with a rules-based auto-classifier (e.g. gripper-force/position heuristics → Grasp Instability; repeated near-identical sub-goals in the trace → Repetition Loop; MLLM's stated belief about scene state diverging from tracked object position → State Mismatch; small position error at a precision checkpoint → Precision Misalignment) as an accelerant, not a replacement, for v1 |
| Recovery Rate computation matching the paper's definition | Direct, numeric comparability to π0.5/SmolVLA/Wall-X/ACT results is the milestone's stated goal | MEDIUM | Paper's definition: Recovery Rate = successful recoveries / recovery opportunities, where a "recovery opportunity" is a detected mid-episode failure and "success" = valid state restored + task continues without external intervention. Requires: (1) a failure-detection trigger during execution (see table stakes), (2) letting the MLLM see the failure and re-plan rather than aborting the episode, (3) a way to judge "valid state restored" — for v1, tie this to task-specific checkpoint predicates (object in gripper, gripper at height, etc.) rather than free-form judgment |
| Semantic-vs-execution failure aggregation | Matches paper's higher-level rollup (State Mismatch = semantic; avg(Grasp Instability, Repetition Loop) = execution), needed to produce comparable summary numbers, not just raw counts | LOW (once per-trial taxonomy labels exist) | Pure aggregation step over the taxonomy labels above — cheap once those exist |
| Rich reasoning-trace analysis tooling (searchable/filterable by failure type, side-by-side trace+video viewer) | Turns raw logs into an actual research artifact; makes it possible to audit whether the auto-classifier or the MLLM's own self-reported reasoning was "right" | MEDIUM | Nice-to-have once volume of episodes exists; not needed for the first Pen Transfer runs |
| Multi-provider comparison (2+ MLLMs on the same task set) | Explicit v2.0 goal ("at least one additional paid MLLM provider... for cross-provider comparison") and the most natural extension of a provider-agnostic router | LOW (given router exists) | Straightforward once the router abstraction and evaluation harness exist; treat as an "add after Pen Transfer validates" item, not parallel v1 work |
| Full 4-task suite (Selective Color Sorting, Multi-Object Packing, Precision Pen Placement) | Matches the paper's full scope, enabling apples-to-apples comparison across all 4 tasks, not just one | MEDIUM-HIGH per task | Explicitly sequenced after Pen Transfer per PROJECT.md; each task adds new physical props/setup and likely new failure modes (e.g. Multi-Object Packing stresses Repetition Loop and State Mismatch more; Precision Pen Placement stresses Precision Misalignment) |
| Closed-loop mid-primitive vision checks (verify grasp succeeded via camera before proceeding to lift) | Meaningfully increases task success/recovery rate versus blind open-loop primitives; used implicitly by VoxPoser-style closed-loop trajectory correction | MEDIUM | Valuable but adds MLLM calls (cost/latency) or requires a lightweight non-MLLM check (e.g. gripper current spike, distance-to-object via depth); can be deferred past the first working Pen Transfer pass |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|------------------|-------------|
| MLLM generates and executes arbitrary code against the robot API (full Code-as-Policies style) | Looks maximally flexible/general, matches the "cool" version of the pattern from the literature | Arbitrary LLM-authored code with direct servo/actuator access on real hardware is a safety hazard (unbounded joint targets, no sandboxing) and much harder to log/audit for the failure taxonomy than a fixed structured schema | Constrain the MLLM to emit a small, closed vocabulary of sub-goal/primitive names + numeric params (JSON schema), validated and clamped before execution — this is what SayCan/VoxPoser effectively do under the hood anyway |
| Per-tick (control-loop-frequency) MLLM calls for continuous control | Feels more "real" / more like a true VLA replacing the policy entirely | Real API latency (hundreds of ms to seconds) makes this infeasible for smooth motion; also explicitly rejected already in PROJECT.md decisions | Plan-then-execute at sub-goal/checkpoint granularity, with a fast deterministic primitive executor filling in the motion between MLLM calls |
| Fine-tuning or LoRA-adapting the MLLM on collected data | Natural next step once you have episode data, and this project already has fine-tuning muscle-memory from the paused v1.1 sim track | Explicitly out of scope for this milestone (PROJECT.md: "this milestone evaluates MLLMs zero/few-shot via prompting, not training"); would also break comparability to the paper's baselines, which are themselves *fine-tuned* VLA/IL policies being compared against a *zero-shot* MLLM as the interesting contrast | Keep the MLLM frozen/zero-shot; if in-context few-shot examples help, add them to the prompt, not the weights |
| Building a full IK/motion-planning stack (e.g. integrating MoveIt or a general trajectory optimizer) | The literature's more advanced systems (VoxPoser) use real motion planners, and it feels like "doing it right" | Large scope/dependency addition for a 6-DOF low-cost arm with a small, fixed task set (pick/place-style primitives only); existing `control/` code already does joint-space waypoint moves via `send_action()` with `max_relative_target` clamping | A handful of hand-tuned parameterized primitives (reach-above, descend-to-grasp, lift, transport, place) driven by depth-camera object coordinates is sufficient for Pen Transfer and the paper's other 3 tasks — matches the scale of prior low-cost-arm + GPT-4V demos |
| New microcontroller/embedded firmware for lower-latency execution | Perceived need for "real-time" reactive control to match VLA-level responsiveness | Explicitly ruled out already in PROJECT.md; existing LeRobot USB-serial bridge to Feetech servos is proven end-to-end through UAT sign-off | Use the existing `control/` LeRobot bridge as-is; latency budget is dominated by the MLLM API call, not the servo bus |
| Real-time interactive control REPL / live teleoperation-style MLLM steering | Would make demos feel more impressive/interactive | Explicitly deferred in PROJECT.md ("Real-time interactive REPL... start with rendered output"); adds UI/streaming complexity orthogonal to the benchmark-replication goal | Batch/offline episode runs with post-hoc trace + video review, matching how the reference paper itself was evaluated |
| Perfectly replicating the paper's exact (undisclosed) failure-annotation adjudication procedure before starting | Desire for maximal methodological rigor / directly-comparable numbers | The paper does not publish its exact per-trial labeling procedure (only the taxonomy definitions and formula); blocking on reverse-engineering it exactly would stall v1 indefinitely | Use the published taxonomy *definitions* and the published Recovery Rate *formula* as the spec, document your own labeling procedure (rule-based + human review) transparently, and treat cross-paper numeric comparison as directionally indicative rather than a strict apples-to-apples statistical claim |

## Feature Dependencies

```
Depth camera reliability fix
    └──blocks──> Extended episode recorder (raw depth as dataset field)
    └──blocks (soft)──> Primitive executor's use of depth for grasp targeting
                             (can fall back to RGB-only heuristics short-term, but
                              PROJECT.md treats depth-first as required before trusting
                              it as MLLM input)

Provider-agnostic MLLM router
    └──requires──> Structured sub-goal JSON schema (shared contract across providers)

Fixed motion-primitive library (reach/grasp/lift/place/retreat)
    └──requires──> Existing control/ LeRobot bridge (SO101Follower.get_observation/send_action)
    └──enables───> Plan-then-execute loop
    └──enables───> Basic execution-failure detection (timeouts, position sanity checks)

Plan-then-execute loop
    └──requires──> MLLM router + primitive library + sub-goal schema
    └──enables───> Full reasoning-trace logging
    └──enables───> Closed-loop replanning after detected failure

Basic execution-failure detection
    └──requires──> Plan-then-execute loop (need an executing action to detect failure of)
    └──enables───> Recovery Rate computation (defines "recovery opportunity")
    └──enables───> Automated failure-taxonomy classification

Reasoning-trace logging + episode recorder
    └──enables───> Automated failure-taxonomy classification (trace is evidence)
    └──enables───> Reasoning-trace analysis tooling

Automated failure-taxonomy classification
    └──enables───> Semantic-vs-execution aggregation
    └──enables───> Recovery Rate computation (needs a failure *type*, not just failure/no-failure,
                    to match paper's per-category structure)

Pen Transfer working end-to-end
    └──validates──> Whole pipeline (router, primitives, loop, recorder, failure detection)
    └──unblocks───> Remaining 3 paper tasks
    └──unblocks───> Multi-provider comparison

Multi-provider comparison
    └──requires──> Pen Transfer validated on free HF model (proves the harness works before
                    spending paid-API budget on comparison runs)
```

### Dependency Notes

- **Depth fix blocks the recorder's depth field, but not the whole pipeline:** the MLLM control loop, router, and primitive executor can be developed and even validated on Pen Transfer using RGB + joint state alone while the depth fix lands in parallel; only the "raw depth per timestep" dataset field and any depth-informed grasp targeting are truly blocked. Treat depth as a hard dependency only for the parts of PROJECT.md that explicitly require it (recorder schema, trusting depth as MLLM input), not for getting a first Pen Transfer episode running.
- **Sub-goal JSON schema is the load-bearing contract:** both the router (different providers must map to the same schema) and the primitive executor (must parse it deterministically) depend on this being nailed down early and treated as a versioned interface, not something that evolves silently per task.
- **Failure-type classification must exist before Recovery Rate can be computed in the paper's shape:** a binary success/fail signal is enough for a basic "did it work" metric, but the paper's Recovery Rate is defined over *failure events* that require identifying a dominant failure mode — so basic failure detection (table stakes) must be built before the differentiator-tier taxonomy/recovery work, and the taxonomy work is what turns "we detected a failure" into "we can report a recovery rate comparable to the paper."
- **Anti-feature conflicts:** "arbitrary code execution as policy" conflicts with "fixed primitive library" — pick one (the fixed library); "per-tick MLLM calls" conflicts with "plan-then-execute" — already resolved in favor of plan-then-execute per PROJECT.md.

## MVP Definition

### Launch With (v1) — get Pen Transfer running end-to-end

- [ ] Sub-goal JSON schema + prompt template (image(s) + instruction → structured sub-goal) — without this nothing else has a contract to build against
- [ ] Provider-agnostic MLLM router, first backend = one free HuggingFace-hosted multimodal model — validates the interface without burning paid budget
- [ ] Fixed motion-primitive library on top of existing `control/` LeRobot bridge (reach, grasp, lift, transport, place, retreat-to-home) — the only safe way to turn MLLM output into servo motion
- [ ] Plan-then-execute loop wiring router + primitives together, one MLLM call per checkpoint
- [ ] Full reasoning-trace logging per MLLM call (prompt, raw response, parsed sub-goal, timestamp)
- [ ] Extended episode recorder: RGB + joint state + reasoning trace + action per timestep (depth field added once camera fix lands, but don't block v1 on it)
- [ ] Basic execution-failure detection (timeout, joint-limit, no-progress heuristics) — minimum signal needed to say "a failure happened here"
- [ ] Pen Transfer task working end-to-end for at least a handful of episodes on the free MLLM

### Add After Validation (v1.x)

- [ ] Failure-taxonomy classification (rule-based + human-reviewed) applied to Pen Transfer episodes — trigger: enough failed episodes exist to be worth categorizing
- [ ] Recovery Rate computation matching the paper's formula — trigger: failure detection + taxonomy exist and at least one MLLM shows repeated failure events to measure recovery against
- [ ] Semantic-vs-execution aggregation reporting — trigger: taxonomy labels exist across enough episodes to aggregate meaningfully
- [ ] Depth camera integrated into recorder + grasp targeting — trigger: AR0144 reliability fix lands (tracked separately, already in progress)
- [ ] Remaining 3 paper tasks (Selective Color Sorting, Multi-Object Packing, Precision Pen Placement) — trigger: Pen Transfer pipeline validated end-to-end
- [ ] Second (paid) MLLM provider wired into the router for cross-provider comparison — trigger: pipeline proven correct/stable on the free model, so paid-API spend isn't wasted debugging plumbing

### Future Consideration (v2+)

- [ ] Closed-loop mid-primitive vision verification (confirm grasp before lifting) — defer until basic open-loop primitives are shown insufficient
- [ ] Reasoning-trace analysis/browsing tooling (searchable by failure type, trace+video sync viewer) — defer until enough episode volume exists to need it
- [ ] Real-time interactive control / live steering — explicitly deferred in PROJECT.md; revisit only if a live-demo need arises

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|----------------------|----------|
| Sub-goal JSON schema + prompt template | HIGH | LOW | P1 |
| MLLM router (free HF backend) | HIGH | LOW-MEDIUM | P1 |
| Motion-primitive library | HIGH | MEDIUM-HIGH | P1 |
| Plan-then-execute loop | HIGH | LOW (given above) | P1 |
| Reasoning-trace logging | HIGH | LOW | P1 |
| Extended episode recorder | HIGH | MEDIUM | P1 |
| Basic execution-failure detection | HIGH | MEDIUM | P1 |
| Pen Transfer task end-to-end | HIGH | LOW-MEDIUM (mostly physical setup) | P1 |
| Failure-taxonomy classification | HIGH (research value) | MEDIUM-HIGH | P2 |
| Recovery Rate computation | HIGH (research value) | MEDIUM | P2 |
| Semantic/execution aggregation | MEDIUM | LOW | P2 |
| Depth-informed recorder + targeting | MEDIUM (blocked, not skipped) | MEDIUM | P2 |
| Remaining 3 paper tasks | HIGH (breadth) | MEDIUM-HIGH each | P2 |
| Second MLLM provider comparison | MEDIUM | LOW (given router) | P2 |
| Closed-loop mid-primitive vision checks | MEDIUM | MEDIUM | P3 |
| Trace analysis/browsing tooling | LOW-MEDIUM | MEDIUM | P3 |
| Arbitrary code-as-policy execution | LOW (safety risk outweighs) | HIGH | Rejected |
| Per-tick MLLM control | LOW (infeasible) | HIGH | Rejected |
| Fine-tuning the MLLM | N/A (out of scope) | N/A | Rejected |

**Priority key:**
- P1: Must have to get Pen Transfer working end-to-end (this milestone's stated first goal)
- P2: Should have to deliver the paper-comparable benchmark (this milestone's stated full goal)
- P3: Nice to have, defer past this milestone

## Competitor/Prior-Art Feature Analysis

| Feature | SayCan | VoxPoser | GPT-4V(ision) for Robotics | This Project's Approach |
|---------|--------|----------|---------------------------|--------------------------|
| Plan granularity | Fixed skill library, LLM scores/selects | LLM writes code composing 3D value maps | Symbolic task plan from demo video + GPT-4 | Structured sub-goal JSON at checkpoint granularity (reach/grasp/lift/place), closer to SayCan's discrete-skill approach than VoxPoser's code-generation approach — matches the arm's small fixed primitive set |
| Feasibility grounding | Learned value function ("Can") | 3D value/cost maps from VLM | Open-vocab detector for object grounding | Depth-camera object localization + joint-limit/reachability checks in the primitive executor (no learned value function needed at this scale) |
| Closed-loop feedback | Inner Monologue extension | Built-in (perturbation-robust replanning) | Not primary focus (offline demo-to-plan) | ReAct-style: failure detection triggers a re-plan MLLM call with updated observation, not just open-loop execution |
| Training required | None (uses frozen LLM + trained value fn) | None (frozen LLM+VLM) | None (frozen GPT-4V) | None — matches the field's consensus that this whole approach is a zero/few-shot prompting exercise, not a training exercise |
| Failure analysis granularity | Not a focus of the paper | Not a focus of the paper | Not a focus of the paper | This is the actual novel contribution here: applying Yu & Qiu 2026's 4-category failure taxonomy + Recovery Rate metric to an MLLM-controlled system, which none of the reviewed prior work does |

## Sources

- [Code as Policies: Language Model Programs for Embodied Control (arXiv 2209.07753)](https://arxiv.org/abs/2209.07753)
- [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances / SayCan (arXiv 2204.01691)](https://arxiv.org/pdf/2204.01691)
- [VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models (arXiv 2307.05973)](https://arxiv.org/abs/2307.05973)
- [GPT-4V(ision) for Robotics: Multimodal Task Planning from Human Demonstration (arXiv 2311.12015)](https://arxiv.org/abs/2311.12015)
- [ReAct: Synergizing Reasoning and Acting in Language Models (arXiv 2210.03629)](https://arxiv.org/html/2210.03629v3)
- [Benchmarking Vision-Language-Action Models on SO-101: Failure and Recovery Analysis, Yu & Qiu 2026 (arXiv 2606.08881)](https://arxiv.org/abs/2606.08881)
- Existing repo: `control/record_episode.py`, `control/joint_jog.py` (LeRobot `SO101Follower` observation/action API, `max_relative_target` clamping) — read directly for grounding on the real dependency surface
- `.planning/PROJECT.md` (v2.0 milestone scope, decisions, and constraints)

---
*Feature research for: MLLM-as-robot-controller real-hardware benchmark replication*
*Researched: 2026-09-15*
