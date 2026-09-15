# Pitfalls Research

**Domain:** Real-hardware MLLM-as-controller robot manipulation (zero/few-shot multimodal LLM driving a physical SO-ARM101 via a plan-then-execute loop)
**Researched:** 2026-09-15
**Confidence:** MEDIUM (cross-checked web sources on LLM-robot safety, HF Inference API limits, structured-output reliability, and VLM spatial grounding; no single authoritative "gotchas" doc exists for this exact stack — synthesized from adjacent literature + the target paper's own methodology)

## Critical Pitfalls

### Pitfall 1: No independent safety envelope between the MLLM output and the actuators

**What goes wrong:**
The control loop treats whatever the MLLM returns as a trusted command and sends it straight to the LeRobot USB-serial bridge. A hallucinated coordinate, a stale/cached response replayed after a timeout, or a plausible-but-wrong sub-goal ("move to picked-up-pen position" when no pen was ever detected) drives the arm into the table, into itself, into the camera rig, or into a joint limit at commanded velocity. Because the loop is plan-then-execute with real API latency (seconds, not milliseconds), there is no human-in-the-loop reflex time once execution starts — the arm is already moving by the time a bad plan would be noticed.

**Why it happens:**
Teams build the "happy path" first (MLLM → parse → send to servos) because that's what makes the demo work, and defer safety limits as "polish." VLM/LLM-controlled robots are documented as prone to hallucinated object references, logically inconsistent plans, and unsafe control code when nothing sits between planner output and actuator input — this is a known, named risk class in the LLM-robotics safety literature, not a hypothetical.

**How to avoid:**
- Put hard joint-position and joint-velocity clamps (derived from the arm's physical limits, not just LIBERO/sim values) in the local controller, *below* the MLLM interface — the MLLM should never be able to command a value outside physically-safe range, full stop.
- Add a workspace bounding-box check (in the same mm frame used for calibration) that rejects any target outside the arm's ~0.45m reach or below the table plane before the local controller ever moves.
- Add a watchdog timeout independent of the MLLM call: if a sub-goal's local execution exceeds an expected time bound (e.g. "reach" taking >2x nominal), stop and re-plan rather than continuing to actuate toward a possibly-wrong target.
- Rate-limit maximum per-step joint delta (both in the "plan" step size and in the local controller's motion profile) so a bad target produces a slow, interruptible motion, not a fast unpredictable one.
- Keep a physical/software e-stop reachable during every run — this is a real robot, not a sim episode.

**Warning signs:**
- Any code path where a parsed MLLM output flows directly into `control/`'s send-position call without passing through a clamp/bounds-check function first.
- No unit test exercising "MLLM returns nonsense/out-of-range/malformed" and asserting the arm does not move unsafely.
- Watching the arm "twitch" or jump between waypoints during early testing and shrugging it off as "just needs a better prompt."

**Phase to address:**
Router/control-loop phase — this must be architected into the control loop itself (a hard boundary between planner output and actuator input), not bolted on after Pen Transfer already works. Retrofitting safety clamps after tasks are running risks having validated task behavior against an unsafe baseline.

---

### Pitfall 2: Building the live control loop against the free HF Inference API's serverless behavior instead of its actual SLA-free reality

**What goes wrong:**
The free HuggingFace-hosted model pilot is chosen specifically to avoid burning paid budget, but the serverless free tier has no uptime SLA, enforces aggressive rate limiting (very low free-tier caps, roughly one request per model per short window, 429s under any concurrent/retry load), and pays a 30–60s cold-start tax whenever the model has been idle for ~15-20 minutes between sub-goal calls. A control loop written assuming "call the API, get a response in a few seconds" will stall mid-episode, and worse, a naive retry loop against a rate-limited endpoint can itself look like a runaway process (rapid repeated requests, backoff not implemented) while the arm sits mid-motion or, worse, a delayed/queued response arrives late and gets executed against a scene that has since changed (e.g. gripper already closed, object already moved).

**Why it happens:**
Developers prototype against the API when it's "warm" (recently called), see fast responses, and don't design for the cold-start / rate-limit case until it happens in a live demo. The free tier's lack of SLA is easy to overlook because nothing in normal light testing surfaces it.

**How to avoid:**
- Design explicit timeout + local fallback behavior from day one: if the MLLM call doesn't return within N seconds, the local controller holds position (does not guess, does not continue the previous sub-goal blindly) and either retries once with backoff or aborts the episode cleanly.
- Never execute a response that arrives after the local controller has already moved on or timed out — timestamp/tag each request and discard stale responses rather than acting on them late.
- Send a lightweight "ping" or keep the model warm with a low-frequency dummy call between real sub-goals during a session, or budget the first call of every episode as a throwaway cold-start call before the real plan request.
- Implement real exponential backoff on 429s, and cap total retries — a tight retry loop against a rate-limited free endpoint will get the router blocked further, compounding the outage mid-episode.
- Treat "free HF model" as a development/debugging convenience, not a benchmarking-grade dependency — expect and log timeout/rate-limit events as first-class episode metadata, since they will happen during real sessions.

**Warning signs:**
- No timeout configured on the HTTP call to the Inference API (relying on library defaults).
- No test that simulates a slow/failed API response and checks the arm's resulting behavior.
- Episodes silently missing reasoning-trace entries with no logged reason why.

**Phase to address:**
Router/control-loop phase — the router's job description already includes "provider-agnostic," so timeout/backoff/fallback-on-failure must be part of its interface contract from the start, since every provider (including future paid ones) needs the same treatment.

---

### Pitfall 3: Assuming the MLLM will reliably return a strictly parseable action every time

**What goes wrong:**
The control loop expects each MLLM response to parse cleanly into a structured sub-goal/action (e.g. JSON with target position, gripper state). In practice, general-purpose multimodal LLMs (especially smaller/free-tier-hosted ones) drift from the requested schema under real image inputs: they wrap JSON in prose, add trailing commentary, use inconsistent key names, invent extra fields, or occasionally refuse/hedge ("I cannot determine the exact position..."). Research on structured-output reliability confirms this is not a solved problem — schema-constrained decoding reduces but does not eliminate the failure rate, and *forcing* strict JSON-mode constraints on smaller models has been shown to badly degrade the underlying reasoning quality in some cases. A brittle parser that assumes well-formed output will either crash the control loop mid-episode or, worse, silently misparse a field (e.g. picks up the wrong number as the z-coordinate) and drives the arm on bad data without erroring.

**Why it happens:**
Teams prompt-engineer against a handful of manual test cases where the output looks fine, then wire the parser directly to those examples without handling the long tail of format drift that shows up under varied scenes/lighting/task states.

**How to avoid:**
- Use a small number (3-5) of few-shot examples in the prompt showing the *exact* output schema, including at least one example of an ambiguous/uncertain case and how it should still conform to the schema (e.g. a documented "confidence: low" or "target: null" convention rather than free text).
- Parse defensively: extract the JSON object from surrounding prose (e.g. bracket-matching) rather than assuming the entire response is valid JSON; validate every field's type and range before use.
- On parse failure, do not guess a default target — treat it as an execution failure, log it as a distinct failure mode, hold position, and either re-prompt once with an explicit "your last response didn't match the schema" correction or abort the sub-goal.
- If the chosen HF model supports it, prefer constrained/guided generation (grammar or JSON-schema-enforced decoding) over prompt-only enforcement, but validate empirically that constraining the model doesn't destroy its actual spatial reasoning — test both raw and constrained modes.
- Log every raw MLLM response (not just the parsed action) — this is already required for the reasoning-trace goal, so make the raw-text field the source of truth for debugging parse failures.

**Warning signs:**
- Parser code uses a bare `json.loads(response)` with no exception handling.
- No logged count of parse-failure-vs-success rate per episode — you won't notice this is a problem until it silently corrupts a benchmark run.
- Prompt has no explicit output-format examples, only a natural-language instruction like "return the target position."

**Phase to address:**
Router/control-loop phase for the parser/schema contract itself; extend into the Pen Transfer phase as the point where the schema gets stress-tested against real (not synthetic) model outputs and hardened before expanding to harder tasks.

---

### Pitfall 4: Treating MLLM pixel-space output as directly usable millimeter-scale robot coordinates

**What goes wrong:**
When asked "where is the pen," a general multimodal LLM answers in terms of what it can see in the 2D image — pixel coordinates, rough fractional position ("center-left of the frame"), or qualitative direction — not calibrated 3D world coordinates in the arm's base frame. Directly mapping this to a joint or end-effector command (even via a naive fixed image-to-world scale factor) produces errors that are small in pixels but large in real millimeters, especially perspective-dependent (near/far from camera) and dependent on which camera (wrist vs overhead stereo) supplied the image the MLLM reasoned over. On a ~500g-payload, ~0.45m-reach arm with small objects (a pen, small blocks), a few centimeters of error is the difference between a clean grasp and a miss or collision — this project's own margin for error is much tighter than typical VLA benchmarks written for larger/more forgiving embodiments. General VLMs are also documented to have materially weaker spatial/depth reasoning than embodiment-specific VLA policies, since they weren't trained on grounded 3D robot data.

**Why it happens:**
It's tempting to ask the MLLM directly for "x, y, z in mm" and trust the number, since the prompt can request that format — but the model is pattern-matching plausible-looking numbers from pixel appearance, not doing real geometric back-projection, and will confidently return precise-looking coordinates that are not calibrated to this arm's frame or this camera's intrinsics/extrinsics.

**How to avoid:**
- Never let the MLLM output final world-frame mm coordinates directly for execution. Instead, have it output pixel coordinates (or a bounding box) in a *named, specified* image (e.g. "overhead depth camera frame"), and do the pixel→world conversion explicitly and deterministically using the existing calibrated stereo depth pipeline (the same back-projection approach already validated in the sim work, `depth_xyz.py`-equivalent for real hardware) rather than trusting the model's numeric guess.
- Make the depth-camera reliability fix (already the first roadmap phase) a hard prerequisite for any MLLM spatial grounding — pixel→mm conversion is only as good as the depth values feeding it, and the AR0144 module's specular-glint/exposure failures were already known to corrupt object-measurement before this milestone started.
- Sanity-check every converted coordinate against the known workspace bounds and against the previous known object position (reject implausible jumps) before it becomes a control target — this doubles as the safety clamp in Pitfall 1.
- If asking the MLLM for spatial relations (e.g. "left of," "closer to") rather than raw coordinates, still ground the final numeric target through the calibrated pipeline, not the model's own coordinate guess — use the MLLM for semantic/relational reasoning, use calibrated geometry for the actual numbers.
- Document and test which image (wrist RGB vs overhead stereo) the MLLM was shown for each spatial judgment, since pixel→world math differs per camera and mixing them up is an easy silent bug.

**Warning signs:**
- Prompt asks the MLLM to output "position in mm" or "joint angles" directly.
- No explicit calibration/transform step visible between "MLLM output" and "controller input" in the code — coordinates just "flow through."
- Grasp attempts consistently miss in one direction/axis (a systematic bias, not random noise) — a strong sign of an uncalibrated or wrongly-oriented transform rather than model unreliability.

**Phase to address:**
Router/control-loop phase for the interface contract (MLLM outputs pixel/relational judgments only); depends on the depth-camera-fix phase being complete and validated first, per the locked build order — do not let MLLM task work (Pen Transfer onward) start against still-unreliable depth.

---

### Pitfall 5: Comparing zero-shot MLLM results to the paper's fine-tuned-policy numbers without accounting for methodology mismatches

**What goes wrong:**
Yu & Qiu's SO-101 benchmark (arXiv:2606.08881) fine-tunes and evaluates policies (π0.5, SmolVLA, Wall-X, ACT) that were trained via teleoperated demonstrations on the exact tasks being scored, using a failure taxonomy and recovery-rate metric designed around continuous closed-loop policy execution at typically-fast control-frequency. Applying the same taxonomy/metric to a zero-shot, plan-then-execute MLLM controller and reporting numbers "against the paper" implies a level of comparability that doesn't exist: the MLLM has never seen this task/embodiment/environment, operates at sub-goal granularity (seconds between decisions, not a continuous policy), and its failure modes may not map cleanly onto categories designed for a trained policy's execution errors (e.g. "Repetition Loop" and "State Mismatch" may look different or need redefinition when the "policy" is a discrete plan-then-execute sub-goal sequence rather than a continuous action stream). Reporting a bare success-rate or recovery-rate number side-by-side with the paper's table, without this caveat, invites readers (including future-you) to draw an apples-to-apples conclusion that isn't warranted — this exact caveat is already flagged as a known risk in the project's own Key Decisions log (fine-tuning is explicitly out of scope, and comparability is called out as "a caveat to note in results, not a gap to close").

**Why it happens:**
Benchmark tables are seductive — reusing the paper's exact metric definitions and task set feels rigorous, but it silently launders the difference between "trained specifically on this" and "never seen this before" into a single comparable-looking column. The pressure to produce a clean comparison table (for a paper, README, or presentation) pushes toward this shortcut.

**How to avoid:**
- Keep the same task suite and failure-taxonomy *categories* for comparability of vocabulary, but report MLLM results in a clearly separate table/section explicitly labeled zero-shot/few-shot, with the fine-tuned baselines' numbers shown only as reference context, never merged into one ranked table implying head-to-head competition.
- Adapt the recovery-rate metric's operational definition explicitly for the plan-then-execute loop (e.g. define "recovery" at the sub-goal-replan boundary, not at the trained-policy's continuous-correction granularity) and document that adaptation alongside the number.
- Preserve and report the same 20-episodes-per-task cadence from the paper for internal statistical consistency across this project's own runs, but do not present it as matching the paper's episode-to-episode conditions (different day, different lighting, different exact object instances, different operator) — those are confounds the paper's own eval controlled for that a zero-shot re-run cannot guarantee it also controlled for.
- Explicitly log and report the MLLM-specific failure modes that don't exist in the paper's taxonomy at all (structured-output parse failures, API timeout/rate-limit failures, coordinate-conversion errors) as a distinct failure category rather than force-fitting them into the paper's four categories — this is itself a finding worth reporting, not noise to hide.
- Write the "gaps from the paper's methodology" caveat once, prominently, near the top of any results writeup — not buried in a footnote — since this is the single most likely way results get over-interpreted by anyone (including future collaborators) skimming a comparison table.

**Warning signs:**
- A results table with fine-tuned-policy rows and zero-shot-MLLM rows in the same ranked list with no visual/textual separation.
- Recovery-rate computed and reported without documenting how "recovery" was operationally redefined for a plan-then-execute loop.
- No distinct bucket for infrastructure failures (API timeout, parse failure, rate limit) — these get miscounted as task/execution failures, inflating the apparent task-difficulty relative to the paper's cleaner trained-policy failure modes.

**Phase to address:**
Metrics phase (failure taxonomy + Recovery Rate implementation) — the adaptation and caveat must be designed into the metric implementation itself, not added as prose after the numbers are already computed and shared.

---

### Pitfall 6: Blind open-loop execution between MLLM calls with no mid-sub-goal visual feedback

**What goes wrong:**
Plan-then-execute means the MLLM is called once per sub-goal (reach→grasp→lift→place), and the local controller then executes that sub-goal open-loop until it reports completion, at which point the *next* MLLM call gets a fresh image. Between those two calls, the arm is moving based on a single stale snapshot of the world — if the object shifts (bumped by the gripper, rolls, or the initial grasp partially slips), there is no MLLM in the loop to notice until the next scheduled call, by which point the sub-goal may already have executed against outdated assumptions (e.g. "lift" executing on a gripper that closed on air). This is a different failure mode than the paper's fine-tuned policies, which typically run closed-loop at higher frequency and can visually correct within the same "policy call" implicitly.

**Why it happens:**
The sub-goal granularity is a deliberate, reasonable design choice (real API latency makes per-tick MLLM calls impractical), but it's easy to under-estimate how much can change in the seconds-to-tens-of-seconds a single sub-goal takes to execute on real hardware, especially for a small/light payload arm where objects are easily nudged.

**How to avoid:**
- Keep the local controller's execution of each sub-goal as short and checkpointed as reasonably possible (e.g. a "grasp" sub-goal should verify gripper closure/force feedback locally, not just command a position and assume success) rather than treating each sub-goal as one long uninterruptible motion.
- Use any available local low-latency signal (gripper current/force, joint effort, or a lightweight local vision check) as a cheap intermediate sanity check *without* calling the MLLM, to abort/hold a sub-goal early if something is clearly wrong (e.g. gripper closed further than an expected object width suggests actual object was gripped).
- Explicitly capture "sub-goal executed against stale world state" as a named failure mode in the reasoning-trace/failure logging, distinguishing it from a genuine MLLM planning error — this is an architecture-induced failure category, and conflating it with model quality will misattribute blame during analysis.
- Keep sub-goals short enough that re-planning frequency is high relative to how fast the scene can plausibly change for this specific arm/payload/task set.

**Warning signs:**
- "Grasp" sub-goals implemented as a single blind position command to the gripper with no post-grasp verification before moving to "lift."
- Failure analysis lumps "MLLM picked wrong target" and "object moved during blind execution" into the same bucket.

**Phase to address:**
Router/control-loop phase for the local checkpointing mechanism; recorder-extension phase should ensure the checkpoint signal is captured in the episode data so this failure mode is analyzable after the fact, not just during Pen Transfer end-to-end validation.

---

### Pitfall 7: Reasoning-trace and sensor logging desynchronized from actual actuation timing

**What goes wrong:**
Full reasoning-trace capture (every MLLM call's rationale, tied per episode) plus synced RGB+depth+joint+action recording is a first-class requirement, but the MLLM call happens on API time (seconds, variable, sometimes retried) while joint/depth/RGB sampling happens on local sensor time (near-continuous). If the recorder timestamps the reasoning trace at "when the response was received" rather than "what world state actually justified the plan" (i.e. the image/depth/joint state *at the moment the request was sent*), post-hoc analysis of a failure ("was this a bad plan given what it saw, or did the world change before it acted?") becomes unreliable — you can no longer tell whether a plan was wrong or just stale, which directly undermines the failure-taxonomy and recovery-rate analysis this milestone depends on.

**Why it happens:**
`record_episode.py`'s existing sync logic was built for a continuous teleop/demo recording pattern (steady tick rate, no multi-second async external calls in the loop); extending it to interleave a slow, variable-latency external API call is a materially different timing problem that's easy to bolt on incorrectly (e.g. just appending the reasoning trace as "another column" at the current tick without recording which exact image/depth frame it was actually conditioned on).

**How to avoid:**
- Record, for every reasoning-trace entry, an explicit reference (timestamp or frame index) to the exact RGB/depth/joint snapshot that was sent to the MLLM as input — not just "closest in time," an unambiguous foreign-key link.
- Timestamp three distinct events per MLLM call, not one: request-sent (with input snapshot reference), response-received, and sub-goal-execution-complete — this lets later analysis separate "model latency," "model reasoning quality," and "world drift during execution" as distinct variables.
- Treat the reasoning-trace log as append-only and schema-versioned from the start, since the failure-taxonomy/metrics phase will need to parse it programmatically later — a format that only made sense while eyeballing raw logs during Pen Transfer will not survive four tasks and cross-provider comparison.

**Warning signs:**
- Recorder code timestamps the reasoning trace using `time.now()` at write-time rather than propagating the original request timestamp.
- No way to answer "what did the model actually see" for a specific logged decision without re-deriving it from surrounding sensor logs by guesswork.

**Phase to address:**
Recorder-extension phase — this is exactly the phase whose job is defining the synced schema; get the request/response/execution timestamp triad and input-snapshot linkage right here, since the metrics phase and multi-provider phase both build on trusting this data's integrity.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|-----------------|------------------|
| Hardcode a single free-tier HF model ID with no provider abstraction "for now" | Faster to get Pen Transfer working | Router/multi-provider phase becomes a rewrite instead of a swap-in | Never — the router's whole purpose is provider-agnosticism; even the first implementation should go through the abstraction |
| Skip pixel→mm calibration and eyeball a fixed scale factor from a few manual tests | Unblocks early end-to-end testing before depth fix lands | Silent, direction-dependent grasp errors that look like "the model is bad" when it's actually the transform | Only for a throwaway smoke test explicitly marked as non-representative; never for anything whose results get recorded into the benchmark dataset |
| Let the local controller retry a failed MLLM call indefinitely rather than defining a hard episode-abort condition | Looks more "robust" in a demo | Runs hang indefinitely burning free-tier rate-limit budget, or worse, executes a stale/late response after excessive retries | Never in the control loop; acceptable only in an offline analysis/re-query tool that isn't touching the live arm |
| Log only the parsed action, not the raw MLLM text, to save storage | Smaller episode files | Parse-failure debugging becomes guesswork; the "full reasoning-trace capture" requirement is silently broken | Never — this directly violates an explicit v2.0 requirement |
| Reuse the paper's failure taxonomy code/labels verbatim without adapting for plan-then-execute-specific failure modes | Faster metrics-phase implementation | Misattributes architecture-induced failures (API timeout, stale sub-goal) to "model quality," corrupting the eventual comparison | Acceptable as a first draft only if immediately followed by adding the MLLM-specific categories before any numbers are reported externally |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|-----------------|-------------------|
| HuggingFace free-tier Inference API | Treating it like a normal low-latency API with library-default timeouts and no backoff | Explicit short timeout, exponential backoff with capped retries, treat cold-start (~30-60s) and rate-limit (429) as expected/handled cases, not exceptional crashes |
| `control/`'s LeRobot USB-serial bridge | Feeding MLLM-derived targets directly into the same call path used for teleop, with the same trust level as a human-driven leader arm | Insert a distinct, testable safety-clamp/bounds-check layer specifically for machine-generated targets, since a human teleoperator has implicit judgment a parsed API response doesn't |
| Stereo depth camera (AR0144) as MLLM spatial-grounding input | Wiring MLLM prompts against depth data before the specular-glint/exposure reliability fix is validated on real objects | Hard-gate MLLM spatial-grounding work behind the depth-fix phase's own UAT sign-off; don't let task-phase schedule pressure pull this forward |
| `record_episode.py` extension for reasoning traces | Bolting the MLLM call into the existing tick-rate recording loop without re-examining its timing assumptions | Redesign the sync model explicitly for a slow, async, retryable external call interleaved with fast local sensor sampling (see Pitfall 7) |
| Provider-agnostic router (future: OpenAI/Anthropic/Gemini-style APIs) | Designing the interface around the free HF model's specific response quirks (its schema drift, its latency profile) | Design the interface around the strictest common contract (structured schema, timeout, error taxonomy) first, verify the free HF model against it, not the reverse |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|-------------|-----------------|
| Sub-goal granularity too coarse (long open-loop execution per MLLM call) | Grasps succeed in short/simple motions but fail increasingly as sub-goal duration grows (Multi-Object Packing, Precision Pen Placement) | Keep sub-goals short and checkpointed (Pitfall 6); re-plan more frequently for tasks with more opportunities for world-state drift | Becomes visible once tasks beyond Pen Transfer (more objects, more precision) are attempted |
| Free-tier rate limits shared across an entire benchmark run | Early episodes in a session succeed, later ones increasingly hit 429s/timeouts, making later-episode results look artificially worse | Pace requests deliberately (small delay between episodes), track daily/session request budget against the free tier's cap before running a full 20-episode/task session | Hits as soon as a full task's 20-episode cadence is attempted in one sitting |
| Full reasoning-trace + raw depth + RGB logged per timestep, per episode, across 4 tasks x 20 episodes | Storage/IO grows fast; loading/analyzing episodes for the metrics phase gets slow | Decide the on-disk schema (compressed depth, thumbnailed vs full-res RGB, trace text vs binary) during the recorder-extension phase, not after the dataset already exists in an unwieldy format | Becomes a real problem once multiple tasks x providers x 20 episodes accumulate before the metrics phase needs to batch-process them |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Committing HF/API tokens or provider keys into prompts, logs, or the reasoning-trace dataset that gets shared/pushed to HF Hub | Credential leak if episode data or logs are ever published/shared (this project already pushes checkpoints/datasets to HF Hub) | Keep provider credentials out of any logged/recorded field; scrub prompt templates before they're persisted; use environment variables, never inline strings that could get captured in a trace |
| No rate/action limiting on the router itself | A bug (retry loop, malformed loop) could hammer a paid provider's API once the multi-provider phase lands, running up unexpected cost | Build the request-budget/backoff logic once in the router (Pitfall 2) so it protects every provider, paid or free, not just the current free-tier pilot |
| Treating the physical arm as a "just a peripheral" with no operator-abort path during autonomous MLLM-driven runs | A stuck/looping control loop keeps actuating with no easy way to stop it mid-episode | Keep a reachable manual stop (software kill-switch tied independently of the MLLM/router process, plus physical arm power/e-stop) available during every autonomous run, not just during teleop |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-------------------|
| No visibility into the MLLM's reasoning until after an episode completes (or fails) | Operator can't tell during a run whether the model is "about to do something wrong," undermining trust and slowing debugging | Stream/print the current sub-goal's reasoning summary to the console/log in near-real-time as each MLLM call resolves, not just at episode-end |
| Silent fallback-to-hold behavior on API timeout with no operator-facing signal | Operator watches the arm "just stop" with no indication whether it's thinking, stuck, or done | Surface a clear status (e.g. "waiting on MLLM response," "rate-limited, backing off," "aborted: parse failure") so a human watching the arm knows what state it's in |
| Treating a parse-failure or timeout episode as just "a failed episode" indistinguishable from a genuine task failure | Skews the eventual benchmark numbers and hides infrastructure issues from whoever reviews results later | Tag and surface infrastructure-failure episodes distinctly in whatever run summary/console output the operator sees live, not just in post-hoc logs |

## "Looks Done But Isn't" Checklist

- [ ] **Safety clamps:** Often missing an explicit workspace-bounds/joint-limit check on *machine-generated* targets specifically — verify by feeding the local controller a deliberately out-of-range or malformed MLLM output in a test and confirming the arm does not move unsafely.
- [ ] **Timeout/fallback handling:** Often present for the "happy path" call but untested for cold-start/rate-limit/stale-response cases — verify by simulating a slow or 429 response and observing the arm holds position rather than guessing or executing late.
- [ ] **Structured-output parsing:** Often works on the handful of prompts used during development but not on a broader sample of real scenes/lighting — verify by running a batch of varied real images through the parser and tracking parse-success rate, not just spot-checking a few.
- [ ] **Pixel-to-mm calibration:** Often "wired up" but not validated against ground truth on the real arm — verify by placing an object at a known measured position and checking the MLLM-derived-then-converted target matches within an acceptable tolerance, not just that the code runs without error.
- [ ] **Reasoning-trace sync:** Often logs *a* trace per episode but not the precise request/response/execution timestamp triad and input-snapshot linkage — verify by picking a random logged decision and confirming you can unambiguously reconstruct exactly what image/depth/joint state it was conditioned on.
- [ ] **Failure taxonomy adaptation:** Often reuses the paper's four categories unchanged — verify that infrastructure-induced failures (timeout, parse failure, rate limit, stale sub-goal) have their own distinct bucket rather than being force-fit into a policy-execution category.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|-----------------|------------------|
| Safety clamps added late, after unsafe motions already occurred during testing | LOW-MEDIUM | Add the bounds-check/clamp layer at the router-to-controller boundary; retroactively audit any recorded episodes for out-of-range commands to flag/exclude them from the benchmark dataset |
| Coordinate/unit mismatch discovered after several episodes already recorded | MEDIUM | If the transform bug is deterministic (e.g. a fixed offset or axis swap), it may be possible to re-derive corrected coordinates from raw pixel+depth data already logged; if not, those episodes must be excluded/re-run once fixed |
| Discover mid-benchmark that infrastructure failures were miscounted as task failures in the metrics pipeline | MEDIUM | Re-process the raw reasoning-trace/episode logs (if they retain enough raw detail per Pitfall 7's prevention) to reclassify failures; this is exactly why raw traces, not just parsed summaries, must be kept |
| Free-tier rate limiting corrupts a benchmark run partway through | LOW | Re-run only the affected episodes after backing off; because episodes are logged with clear pass/fail/infrastructure-failure tags (per the UX Pitfalls fix), partial re-runs don't require redoing the whole task |
| Results already shared/written up before the zero-shot-vs-fine-tuned methodology caveat was made explicit | LOW-MEDIUM | Add the caveat prominently to the existing writeup rather than silently revising numbers; if a comparison table was already published without separation, re-publish a corrected version rather than leaving the misleading one standing |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|-------------------|----------------|
| No safety envelope between MLLM output and actuators | Router/control-loop phase | Deliberately feed out-of-range/malformed targets in a bench test; confirm arm never exceeds clamped bounds |
| Free-tier API rate limits/cold starts breaking the live loop | Router/control-loop phase | Simulate slow/429 responses; confirm local hold-and-recover behavior, no stale-response execution |
| MLLM output-format brittleness | Router/control-loop phase, hardened during Pen Transfer | Run parser against a varied batch of real scene images; track and report parse-success rate |
| Pixel-space vs mm-scale coordinate mismatch | Router/control-loop phase interface contract; hard-gated on depth-fix phase completion | Validate converted coordinates against a known measured object position within tolerance |
| Zero-shot vs fine-tuned-policy methodology mismatch | Metrics phase | Results writeup shows separated tables + explicit caveat + distinct infrastructure-failure category |
| Blind open-loop execution between MLLM calls | Router/control-loop phase (checkpointing); data captured in recorder-extension phase | Failure logs can distinguish "stale world state" from "bad plan" as separate causes |
| Reasoning-trace/sensor desync | Recorder-extension phase | Any logged decision can be traced back to the exact input snapshot it was conditioned on |

## Sources

- [On the Vulnerability of LLM/VLM-Controlled Robotics](https://arxiv.org/pdf/2402.10340) — MEDIUM confidence (cross-checked academic source; hallucination/unsafe-plan risk class)
- [Using large language models for embodied planning introduces systematic safety risks](https://arxiv.org/pdf/2604.18463) — MEDIUM confidence
- [Safety Guardrails for LLM-Enabled Robots](https://arxiv.org/pdf/2503.07885) — MEDIUM confidence (runtime constraint/guardrail pattern)
- [Enhancing Reliability in LLM-Integrated Robotic Systems: A Unified Approach to Security and Safety](https://arxiv.org/pdf/2509.02163) — MEDIUM confidence
- HuggingFace free-tier Inference API rate limits, cold starts, and lack of SLA — MEDIUM confidence, synthesized from multiple third-party overviews (klymentiev.com, theneuralbase.com, aionx.co); no single canonical HF doc page confirmed exact numeric limits, treat specific figures (e.g. "~1000 req/day," "429 after ~35 concurrent") as approximate/LOW-confidence and verify empirically against the actual account/model before relying on them for capacity planning
- LLM structured-output/function-calling reliability and JSON-schema-constraint tradeoffs — MEDIUM confidence, cross-checked across multiple sources (agenta.ai, towardsdatascience.com); the finding that hard JSON-constraint enforcement can degrade smaller-model reasoning is worth empirically validating against the specific free HF model chosen
- VLM spatial/depth reasoning limitations relative to embodiment-trained VLA policies — MEDIUM confidence, cross-checked across recent (2026) arXiv papers (DepthVLA, VEGA, T-Rex, N3D-VLM)
- [Benchmarking Vision-Language-Action Models on SO-101: Failure and Recovery Analysis (arXiv:2606.08881)](https://arxiv.org/abs/2606.08881) — HIGH confidence (primary source for the methodology this project explicitly benchmarks against); confirms fine-tuned π0.5/SmolVLA/Wall-X/ACT evaluation, four-category failure taxonomy, semantic/execution failure decomposition, and recovery-aware metrics — the basis for Pitfall 5
- `.planning/PROJECT.md` (this project's own Key Decisions log) — HIGH confidence, primary source; already flags the fine-tuned-vs-zero-shot comparability caveat and the depth-fix-before-MLLM-work sequencing as intentional decisions

---
*Pitfalls research for: Real-hardware MLLM-as-controller robot manipulation (SO-ARM101, plan-then-execute loop, HuggingFace-hosted pilot model)*
*Researched: 2026-09-15*
