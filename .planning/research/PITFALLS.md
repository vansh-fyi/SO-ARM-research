# Pitfalls Research

**Domain:** Adding (1) an MLLM raw-JSON no-primitives control path, (2) an async control-loop tick-latency fix, and (3) macOS camera device resolution to an existing real-hardware robot bridge (SO-ARM101, Colab-hosted PolicyServer + ngrok, `control/vla_bridge/`)
**Researched:** 2026-09-24
**Confidence:** MEDIUM (cross-checked web sources on LLM structured-output reliability, async/VLA staleness research, and OpenCV/macOS camera-enumeration issues, triangulated against this project's own `FINDINGS.md` and source code, which are HIGH-confidence primary evidence — three real silent-failure bugs already hit in this exact bridge, and the exact camera-index-vs-name divergence already confirmed live)

This research is grounded in what has **already gone wrong in this codebase**, not generic advice: Phase 11 hit three separate silent-failure bugs in this same bridge (missing background thread → permanently empty queue; wrong staleness constant → every action discarded; `.pos`-suffix key mismatch → every validated action silently `{}`), and the camera bug was independently confirmed live twice (`vla_episode_001` and the Phase 11 go/no-go episode both recorded the laptop webcam, not the robot workspace). The pattern across all four: **the system did not crash — it ran to completion looking healthy while doing the wrong thing.** Every pitfall below is selected because it continues or compounds that exact failure shape.

## Critical Pitfalls

### Pitfall 1: The "fix" for tick latency trades one silent failure mode for another

**What goes wrong:**
`FINDINGS.md` §5 proposes two candidate fixes: (a) only request fresh inference when the local queue empties/nears-empty, or (b) tune `control_hz` to match real measured round-trip latency. Both are legitimate, but neither is free. Option (a) — request-on-queue-low — is a race: if the "low" threshold is too aggressive (queue drains before the new chunk arrives), the robot now blocks/holds waiting for inference exactly like a synchronous loop would, just less often. If it's too conservative, the queue never actually empties before the fix triggers and you've reproduced almost the same over-request pattern the fix was meant to solve. Option (b) — matching `control_hz` to the observed ~11-20s/tick — doesn't fix staleness, it legalizes it: `STALE_ACTION_S` and `control_hz` get widened together, so the system now silently *accepts* multi-second-old actions as fresh by definition, which is exactly the kind of change that looks like a fix (no more discarded actions, more actions execute) while producing a robot that responds to a world state that's already 10-20s stale — on a system with people/objects near the arm, this is worse than discarding, not better. Async-VLA research on this exact tradeoff (execution-horizon/prediction-horizon chunking) confirms this is a fundamental property of async inference, not a bug that disappears once the resend is fixed: moving from sync (resend every tick, thrashing) to naive async (request on empty) trades slow-tick thrashing for a prediction-execution mismatch where the chunk executes against a world the policy never actually saw.

**Why it happens:**
The success metric during debugging becomes "more actions execute / fewer get discarded as stale," which both fixes satisfy trivially — (a) by removing redundant requests, (b) by redefining "stale" upward. Neither metric distinguishes between "actions execute promptly against a fresh observation" (the actual goal) and "actions execute because the discard threshold moved" (a metric that improved by changing the definition, not the latency).

**How to avoid:**
- Pick a concrete, numeric target for real round-trip latency *before* choosing (a) or (b) — e.g., "actions must execute within N seconds of the observation they were computed from" — and measure against that target directly (from `episode.jsonl` timestamps, the same way Phase 11's 1/60 yield was diagnosed), not just "did the discard rate go down."
- If implementing (a), add an explicit in-flight-request guard so a queue-low trigger can never fire a second observation-send while a prior one is still awaiting response (see Pitfall 2) — and pick the "near-empty" threshold based on measured inference latency (chunk duration must exceed round-trip time, or the queue will still hit empty and stall).
- If implementing (b) alone or in combination, treat any resulting `STALE_ACTION_S`/`control_hz` widening as a deliberate, logged trade-off with a stated upper bound — re-derive it from real measured Colab/ngrok RTT (this session's ~11-20s, not a guess), and flag it for review rather than quietly committing a wider constant the way the 8-10x safety-cap loosening was flagged in the Key Decisions log.
- Test the fix against a simulated slow/degraded network condition (not just the happy-path fast case), since Colab cold-starts and ngrok tunnel behavior are exactly the kind of intermittent slowness that made the original bug hard to notice in casual testing.

**Warning signs:**
- "Fewer stale-discard log lines" treated as the sole success criterion for the fix, with no companion metric for actual observation-to-execution latency.
- The fix reduces the *number* of ticks but the wall-clock time between "observation sent" and "action executed" hasn't actually improved — this is a red flag that (b)-style constant-widening did the work, not (a)-style architecture.
- No test exercising a slow/degraded Colab response under the new fix — only re-running the same episode that already worked once the bug was fixed.

**Phase to address:**
Bridge tick-latency fix phase — the target latency number and the distinction between "real fix" vs "widened tolerance" must be decided before implementation starts, not discovered after a second live episode looks better on paper.

---

### Pitfall 2: New race condition — duplicate or overlapping observation requests during the "request only when queue is low" fix

**What goes wrong:**
Implementing fix (a) from Pitfall 1 requires a background/async trigger ("queue is empty or near-empty → request a new observation") running concurrently with the main control-tick loop that's draining the queue. If that trigger isn't guarded against re-firing while a request is already in flight, a slow Colab response (the exact condition this bug already exists under) causes the queue-low check to keep re-triggering on every subsequent tick, firing multiple concurrent observation-sends before the first one returns. This is a new race condition introduced *by* the fix, on top of the one being fixed — and it's the same class of bug as bug #1 in `FINDINGS.md` (`connect_bridge()` never starting `receive_actions()`'s background thread): a background/async responsibility that's easy to under-specify and easy to leave silently half-wired.

**Why it happens:**
The fix is naturally framed as "check queue length, if low, request" — a simple conditional — but that framing omits the state needed to make it safe under real network latency: whether a request is already outstanding. This is exactly the kind of omission that produced bug #1 (the missing thread start) and bug #2 (the wrong staleness constant) in the same file tree last phase: a single-variable oversight that doesn't crash, it just quietly produces the wrong behavior.

**How to avoid:**
- Add an explicit `request_in_flight` (or equivalent) boolean/lock that the queue-low trigger checks before firing, cleared only when the response is received or the request definitively times out/fails.
- Write a unit test that simulates a queue-low condition occurring on two consecutive ticks with the first request still pending, and asserts only one observation was sent.
- Log every observation-request send with a request ID, and log every response with the same ID, so a live episode's `episode.jsonl` (or an equivalent bridge-side log) can be audited after the fact for duplicate/overlapping requests — this is the same "verify from raw logs, don't assume" discipline `FINDINGS.md` §2 already applied to diagnose the original bug.

**Warning signs:**
- Network/tunnel traffic (or Colab-side request logs) shows more inference requests than there are actual chunks consumed in an episode.
- Intermittent behavior that only appears under slow Colab response times (cold start, first-call-of-session) and not in fast/warm repeat testing — a classic race-condition signature that "looks fine" in casual local testing.

**Phase to address:**
Bridge tick-latency fix phase — this must be part of the same implementation as Pitfall 1's fix, not a follow-up bug found later; the in-flight guard is inseparable from the "request on queue-low" design.

---

### Pitfall 3: Reusing `safety_validator.py` unchanged for MLLM output assumes a key/unit/shape contract the MLLM was never explicitly forced to honor

**What goes wrong:**
The v2.1 design decision reuses `safety_validator.py` unchanged as the MLLM's safety net, on the reasoning that the model "only ever emits validated JSON actions." But `safety_validator.validate_action()` looks up plain `action_contract.JOINT_ORDER` names (`"shoulder_pan"`, not `.pos`-suffixed) — this exact key-naming assumption is what silently emptied every real SmolVLA action into `{}` in Phase 11 (bug #3, `FINDINGS.md` §1). An MLLM prompted only in natural language to "output JSON actions" has no equivalent hard guarantee: it can drift toward `.pos`-suffixed keys (if it's seen LeRobot-style examples anywhere in training data), nested objects, degrees vs. normalized units, or a different field order/naming than `action_contract.py` expects — and because the validator's membership check silently drops non-matching keys rather than raising, a schema-drifted MLLM response doesn't error, it silently becomes `{}` or a partial action, exactly like bug #3 did, except now the root cause is prompt/schema drift instead of a library's internal key format.

**Why it happens:**
"The model only emits validated JSON, so the existing gate is sufficient" is true for *what runs the actuators*, but it silently assumes the MLLM's JSON already matches the validator's expected shape byte-for-byte — an assumption that was already proven wrong once this session for a *deterministic* library function, and is far more likely to drift for a natural-language-prompted general model with no schema-enforcement guarantee.

**How to avoid:**
- Add an explicit schema-conformance check *before* the MLLM's JSON reaches `safety_validator.validate_action()` — validate key names against `action_contract.JOINT_ORDER` explicitly and **raise/log distinctly** on mismatch (not silently drop), so a schema-drifted response produces a visible, attributable failure instead of another silent `{}`.
- Write the MLLM's output schema directly from `action_contract.py`'s existing constants (joint names, units, gripper scale) — never hand-author a separate schema description in the prompt that could drift from the code's actual contract, the same mistake `WRIST_ROLL_LIMIT_DEG`'s hand-copied transcription already made elsewhere in this project (per the v2.0 tech-debt audit).
- Add a unit/integration test that feeds `safety_validator.validate_action()` a deliberately `.pos`-suffixed or otherwise-malformed MLLM-shaped JSON action and asserts it is rejected loudly, not silently emptied — this is the regression test bug #3 should have had and didn't.
- Before the first live MLLM episode, log and manually inspect several raw MLLM JSON responses against the validator's actual key set — don't assume prompt engineering produced the right shape just because the prompt asked for it.

**Warning signs:**
- `validated_action` (or its MLLM equivalent) comes back empty or partial with no distinct log line explaining why — this exact silent-empty signature already happened once in this codebase.
- No test exercising "MLLM returns a subtly wrong key name / unit / nesting" and asserting a loud, attributable rejection.
- The MLLM's prompt describes the action schema in prose without referencing `action_contract.py`'s actual constants.

**Phase to address:**
MLLM control loop phase — the schema-conformance check must be built alongside the prompt/schema design, not discovered after a live episode again yields near-zero real actions for a reason that looks identical to a bug already fixed once.

---

### Pitfall 4: No pre-built primitives means every single tick is a fresh opportunity for parse failure, unlike sub-goal-level MLLM calls

**What goes wrong:**
The v2.1 design deliberately has the MLLM reason directly to raw joint-level JSON with no `move_to()`/`grasp()` helpers, emitting sequenced chunks of actions (chunk size TBD from latency-fix timing data). This is a materially higher-stakes reliability surface than a sub-goal-level call (e.g., "reach," called once per phase of the task): if each chunk covers, say, 10-50 raw actions and the model occasionally emits a malformed action *within* a chunk (wrong type, out-of-range value, a hedge/refusal embedded mid-JSON, a dropped joint), a naive per-chunk parser either fails the whole chunk (wasting an entire inference round-trip and losing all the otherwise-valid actions in it) or, worse, silently skips/zero-fills the bad entry and continues, producing a physically discontinuous joint trajectory (e.g., one action in the middle of a chunk snaps to 0° or NaN) that a hard clamp might catch but a soft default might not.

**Why it happens:**
Structured-output reliability research confirms LLMs commonly produce markdown-fenced wrapping, trailing prose, hallucinated/renamed fields, and truncated output even when explicitly prompted for JSON — and the raw-JSON-per-tick design multiplies the *number* of individual values that must all be well-formed per inference call (dozens of joint values across many timesteps) compared to a single sub-goal descriptor, raising the statistical odds that at least one entry in a chunk is malformed.

**How to avoid:**
- Validate every individual action *within* a chunk independently — a single malformed action should cause that one action to be held/skipped (with a distinct log entry), not invalidate the entire chunk or silently zero-fill it into the trajectory.
- Never let a partially-malformed chunk produce a joint value via a default/zero-fill; treat a rejected mid-chunk action exactly like a stale/no-action tick (hold last known-good position) so a parse failure degrades to "held position," the same safe behavior already proven out for staleness discards in Phase 11.
- Log parse-success rate per chunk and per episode as first-class metadata (not just pass/fail on the episode as a whole) — this is a new observability requirement the sub-goal-level design in the earlier (paused) architecture research didn't need at this granularity.
- Size chunks conservatively for the first live test (favor smaller chunks with more frequent re-inference over large chunks that maximize single-point-of-failure exposure), and only grow chunk size once parse-success rate is empirically validated at the smaller size.

**Warning signs:**
- Parser code assumes a chunk is all-valid-or-all-rejected, with no per-action-within-chunk validation path.
- No distinct log/metric for "N of M actions in this chunk were rejected," only an episode-level pass/fail.
- A rejected mid-chunk action results in a joint value of exactly 0 or an unexplained large jump in `executed_action` — evidence of a silent default rather than a held position.

**Phase to address:**
MLLM control loop phase — this is a direct consequence of the no-primitives, raw-JSON, multi-action-chunk design choice and must be designed into the parser from the start, not retrofitted after a live episode shows a trajectory glitch.

---

### Pitfall 5: Repeating the earlier "failed badly" general-MLLM-prompting attempt without a documented root cause means the same failure can recur silently under a new name

**What goes wrong:**
`PROJECT.md` records that an earlier experiment prompting general coding-agent LLMs (Claude Code, Codex) directly, with progressively richer context, to output robot actions "failed badly," but the failure mode itself isn't detailed in the milestone context available to this phase. If the v2.1 MLLM control loop is built without first re-deriving *why* that attempt failed (bad spatial grounding? malformed output? safety violations? just slow/expensive?), there's a real risk of re-implementing the same failure under a different label — e.g., if the prior failure was fundamentally about pixel-to-mm coordinate grounding (a documented general-VLM weakness — see the existing `.planning/research/PITFALLS.md`'s Pitfall 4 from the earlier v2.0 research pass), then simply switching from a coding-agent harness to a paid-tier Claude API call with a cleaner JSON schema will not fix that root cause, and the "no pre-built primitives, raw joint-level JSON" design is if anything a *harder* version of the same spatial-grounding problem (direct joint deltas require even more precise spatial/kinematic reasoning than a Cartesian target would).

**Why it happens:**
"General MLLM prompting already failed once" is easy to treat as fully explained by "wrong model/tooling" (coding agents aren't purpose-built for this) rather than interrogating whether the deeper issue (spatial grounding, output reliability, safety-relevant hallucination) will persist with a different, more capable model.

**How to avoid:**
- Before building the control loop, explicitly document (even briefly, in `11-CONTEXT.md`-equivalent scoping) what specifically failed in the earlier attempt — output format, spatial accuracy, safety violations, latency, or something else — and design this phase's mitigations against that specific failure mode, not just "use a better model."
- If the earlier failure was spatial/depth-grounding related, treat that as a standing risk for the raw-joint-JSON design too, and add an explicit sanity check comparing MLLM-commanded joint deltas against the arm's known kinematic limits and a plausible-motion heuristic (e.g., reject a commanded delta that would move the end-effector further than physically reachable in one chunk) — not just the existing absolute joint-limit clamp, which catches out-of-range values but not "in-range but kinematically implausible."
- Compare the MLLM's very first few live chunks against SmolVLA's known-good Phase 11 trajectory shape (smoothness, magnitude of per-step deltas) as an early smoke test before running the full Pen Transfer comparison — a qualitatively different (jerky, oscillating, non-monotonic-toward-target) trajectory shape is a fast, cheap signal that the same failure mode has recurred.

**Warning signs:**
- No written record of the earlier attempt's specific failure mode exists anywhere accessible to this phase's implementer.
- The new design's mitigations are generic ("better prompting," "paid tier model") rather than targeted at a named prior failure.
- Early live chunks show large, erratic, or physically implausible joint deltas that only the hard safety clamp (not a semantic/plausibility check) catches.

**Phase to address:**
MLLM control loop phase, before first live episode — this is a design-input gap that should be closed during phase discussion/scoping, not discovered live on hardware.

---

### Pitfall 6: The numeric-vs-name camera divergence is a class of bug, not a single bug — fixing `camera_overhead`'s recording path alone leaves the same divergence pattern available to recur elsewhere

**What goes wrong:**
The confirmed bug is specific: `run_vla_episode.py` builds `args.camera` for the `IOLogger` recording path from `device_map.json`'s plain numeric `cameras.stereo_overhead` (used via a bare `cv2.VideoCapture(idx)`), while the actual inference-input path resolves `args.stereo_camera_index` from `cameras.stereo_overhead_name` (`"CCB Camera"`) specifically because numeric indices are known (and now twice independently confirmed) to drift across process launches on macOS. Fixing only the one call site that opens `camera_overhead` for recording addresses the *symptom* observed so far, but the underlying pattern — two independent code paths deriving a camera identity from the same `device_map.json`, one via index and one via name, wired up at different times by different work — is exactly the shape that produced this bug in the first place, and nothing prevents a third path (a future analysis script, a different recording mode, a notebook cell) from being added later using the numeric field again, since `device_map.json` still exposes both `stereo_overhead` (int) and `stereo_overhead_name` (string) side by side with no enforcement that only the name field is ever used for the AR0144.
Additionally, this stereo camera has a second, distinct, already-documented failure mode layered underneath: `stereo_camera.py`'s own docstring records that `cv2.VideoCapture`'s AVFoundation backend cannot be made to report the AR0144's true 2560x720 frame on macOS at all — it silently serves a lower resolution (1920x1080 or 1280x720) regardless of requested `CAP_PROP_FRAME_WIDTH`/`HEIGHT` (a known unfixed OpenCV bug, opencv/opencv#23368) — which is why the inference path was rewritten to shell out to `ffmpeg` directly instead of using `cv2.VideoCapture` at all. If the recording-path fix re-routes `camera_overhead` to the name-based index but still opens it via `cv2.VideoCapture` (rather than reusing `StereoSplitCamera`/the `ffmpeg` backend), it will stop capturing the *wrong physical device* but may still silently capture the *right device at the wrong resolution/crop* — a second, quieter version of the same "looks like a valid frame, isn't the real one" failure class.

**Why it happens:**
Numeric camera indices are the path of least resistance for any new script or call site (`cv2.VideoCapture(1)` "just works" in casual local testing where only one or two cameras are ever connected), and the fix-the-symptom instinct (change this one call site to use the name) doesn't automatically generalize to "never introduce a numeric-index camera open again" without an explicit shared resolution function.

**How to avoid:**
- Fix at the single-source-of-truth level: add one shared camera-resolution function (e.g., in a small shared module both `robot_client.py` and `run_vla_episode.py`'s recording path import) that takes a logical camera role (`"stereo_overhead"`, `"wrist"`) and `device_map.json`, and always resolves to the name-based identifier for any device with a `_name` field — no call site should construct a `cv2.VideoCapture`/`ffmpeg` target from a raw `device_map.json` index directly.
- Route the fixed `camera_overhead` recording path through the *same* `StereoSplitCamera`/`ffmpeg`-backed capture the inference path already uses (or an equivalent that shares the resolution/backend fix), not a second independent `cv2.VideoCapture` open of the AR0144 by name — otherwise the resolution-cap bug (#23368) can still silently corrupt the recording even after the device-identity bug is fixed.
- Consider whether `device_map.json` should stop exposing the raw numeric `stereo_overhead` index at all for cameras with a `_name` field (or clearly mark it "do not use directly, resolution-only convenience field") — an available-but-wrong field left in the schema is exactly what let this bug happen twice already.
- Add a fast, cheap regression check: after any change touching camera opening code, capture one frame from every configured camera role and diff/inspect it (even just a checksum comparison against a known-good "robot workspace" reference frame vs a known-bad "laptop webcam" reference frame) before trusting a new recording path.

**Warning signs:**
- Any new or modified code that opens a camera via `cv2.VideoCapture(<int from device_map.json>)` directly, bypassing a shared resolution helper.
- `device_map.json` continuing to carry both an index and a name field for the same physical device with no code-level enforcement of which one is authoritative.
- A recording path using `cv2.VideoCapture` for the AR0144 stereo device at all, even if by name — the resolution-cap bug is backend-specific (AVFoundation via `cv2`), not index-specific.

**Phase to address:**
Camera device resolution fix phase — the fix should be scoped as "one shared, enforced camera-resolution path for all call sites," not "patch the one call site we found," given this is the second time (webcam-vs-robot-workspace, twice) and the third distinct failure mode (index drift, resolution cap, and the still-open risk of a future third call site) documented in this exact camera stack.

---

### Pitfall 7: "Verified" recorded footage still isn't proof of what the model actually received, only proof of what a *different* capture happened to save

**What goes wrong:**
The v2.1 Active scope already recognizes this gap explicitly ("verify — not just assume — that the model's actual inference input was correct... have the bridge log a hash/thumbnail of what it actually sent to Colab per step"), but the naive version of that fix (hashing/thumbnailing the frame right before it's sent) only proves the *bridge's* outbound payload was correct — it does not prove Colab's PolicyServer received and used that exact frame (e.g., a stale cached observation on the server side, a base64/encoding corruption over the ngrok tunnel, or a resize/preprocessing step on the Colab side silently altering the image before the model sees it). The project's own history already contains a version of this exact class of gap: the human operator only caught the wrong-camera bug by *manually reviewing saved images* after the fact — nothing in the pipeline itself detected or flagged that camera_overhead's recorded frames were the laptop webcam, twice, across two different episodes, until a human looked.

**Why it happens:**
"We log a hash of the frame" feels like it closes the provenance gap because it answers "did we send the right bytes," but robot-perception provenance research is explicit that the actual question is "what did the model see when it acted" end-to-end — client capture, transport, and server-side preprocessing are all separate opportunities for silent divergence, and a hash taken client-side only covers the first hop.

**How to avoid:**
- Log the hash/thumbnail at *both* ends if at all possible: client-side immediately before send, and (if the Colab PolicyServer notebook can be instrumented) server-side immediately before it's fed to the model — a mismatch between the two proves a transport/encoding bug that a client-only hash would miss entirely.
- Treat the hash/thumbnail as a machine-checkable assertion, not just a debugging aid a human might look at later — e.g., have the client compare the thumbnail against a coarse "does this look like the robot workspace, not a face/room" sanity heuristic (even something crude like average color histogram distance from a known-good reference frame) and flag/abort on a large deviation, rather than relying on a human to eventually notice, which is exactly what didn't happen twice already.
- Extend the same "verify, don't assume" discipline to the recorded I/O log itself: after any live episode, re-derive at least one claim (e.g., "the arm moved this many degrees") independently from raw data (as `FINDINGS.md` §2 did for the step-by-step pattern) rather than trusting a summary field, since `latency_ms` in the existing log is a known example of a field that *looks* populated (`{0,0}`) but is not actually measuring anything — a structurally identical "looks recorded, isn't real" gap to the camera bug, just in a different field.
- Make "human manually opened and looked at a saved frame" a required, checklisted step before any episode's footage is used as evidence a system worked — not an ad hoc catch, since that's the only reason this bug was ever caught at all.

**Warning signs:**
- Verification logic that only checks "a frame was captured / a hash exists" rather than "this frame is plausibly correct content."
- Any field in `episode.jsonl` (or its extensions) that's populated with a constant/placeholder value (like `latency_ms: {0,0}`) without a comment or check flagging it as not-yet-real — these are easy to mistake for real data during later analysis.
- No documented manual-review step in the episode/UAT checklist for actually opening and looking at recorded frames before treating an episode as validated evidence.

**Phase to address:**
Camera device resolution fix phase for the hash/thumbnail logging itself; this pitfall's broader lesson (provenance requires end-to-end checking, not single-hop hashing) should also inform the reasoning-trace capture work in the MLLM control loop phase, since that phase adds a second kind of "did the model really see/reason about what we think it did" claim to verify.

---

### Pitfall 8: Re-tightening safety validator caps post-latency-fix without a live-hardware re-test repeats the exact same unverified-config risk that caused bug #2

**What goes wrong:**
v2.1's Active scope calls for reverting `safety_validator.py`'s loosened caps (`MAX_RELATIVE_TARGET_DEG`, `MAX_VELOCITY_DEG_PER_S`, `STALE_OBSERVATION_S`, `STALE_ACTION_S`) back toward conservative defaults once the latency fix lands. But Phase 11's bug #2 was precisely a case where a staleness constant was wrong for the loop's actual timing behavior and nothing caught it until a live episode discarded every single action — a config-level bug, not a code-crash bug, invisible without an actual timed hardware run. Re-tightening the caps by editing constants and assuming "the latency fix should make the original numbers work now" without a live re-test repeats that exact risk in the opposite direction: if the tightened values are wrong for the *actual* post-fix tick rate (which may not hit the fix's target latency perfectly on the first attempt), real MLLM or VLA actions could once again be silently discarded as stale or clamped near-invisible, exactly as the original conservative defaults did before they were loosened.

**Why it happens:**
Reverting a value back toward a previously-used default feels safe ("we know these numbers, we used them before... well, actually, we never successfully ran a live episode against the original un-loosened defaults — they were tightened from the start and immediately hit the staleness bug"), which obscures the fact that the original conservative defaults were never actually validated against real Colab/ngrok latency at all — they were simply what was in the code before anyone had live-tested it.

**How to avoid:**
- Treat the post-fix cap re-tightening as requiring the same live-hardware verification discipline as the original latency fix, not a "safe revert" — re-run at least one live episode with the tightened caps and confirm real-action yield (not just "no crash") before considering the caps final.
- Tighten incrementally and measure at each step (e.g., halve the loosening factor, re-test, halve again) rather than jumping directly back to the pre-Phase-11 defaults, given the tick-rate improvement from the latency fix is itself an empirical unknown until measured.
- Explicitly log and diff the real-action-yield metric (the same 1/60 style number from `FINDINGS.md`) across the loosened-cap episode, the tightened-cap-but-unfixed-latency case (if tested), and the fixed-latency-plus-tightened-cap case, so a regression in yield is caught immediately rather than discovered on a much later Pen Transfer comparison run where it would be misattributed to the MLLM's quality instead of the caps.

**Warning signs:**
- Safety-cap constants changed in the same commit as (or without) a live-hardware verification run.
- No recorded real-action-yield number for the tightened-cap configuration before it's treated as the new baseline for the Pen Transfer comparison.

**Phase to address:**
Bridge tick-latency fix phase (the re-tightening is explicitly sequenced after the fix in `PROJECT.md`'s Active scope) — verification must be a live-hardware checkpoint in this phase's plan, not an assumed-safe follow-on edit.

---

### Pitfall 9: An MLLM reasoning over raw camera frames introduces a prompt-injection-shaped attack surface a VLA never had

**What goes wrong:**
SmolVLA maps observations directly to joint actions with no natural-language reasoning step (per `FINDINGS.md` §4's D-05) — it has no text channel an adversarial or spurious visual element could influence beyond whatever's baked into the trained policy weights. An MLLM reasoning in natural language over camera frames is a different threat model: any text, QR code, printed instruction, or visually-salient-but-semantically-loaded object that happens to be in the camera's field of view (the workspace, background, or even a sticky note near the rig) could be interpreted by the model as an instruction or context that shifts its output away from the intended task. This is a documented, named risk class for LLM-integrated robotic systems (prompt injection via multimodal input, adversarial visual content), and it's genuinely new risk surface this milestone introduces, not something the existing safety validator or VLA baseline had to account for.

**Why it happens:**
The threat is easy to dismiss in a controlled lab setting ("there's nothing adversarial near the robot, it's just my desk"), but the whole point of general-purpose MLLM reasoning is that it interprets *everything* in the frame, not just the trained-for objects — an incidental sticker, a monitor showing text in the background, or even benign visual clutter can produce unpredictable prompt-following behavior that a joint-level policy trained on a fixed action distribution simply cannot exhibit.

**How to avoid:**
- Keep the workspace visually controlled and minimal during MLLM episodes specifically (no incidental text/screens/printed material in either camera's field of view) as a cheap, practical mitigation, not just for image-quality reasons.
- Keep the hard safety clamps (joint limits, velocity caps, workspace bounds) as the actual safety boundary regardless of what the MLLM reasons about — the existing `safety_validator.py` clamps already provide this independent of prompt content, which is precisely why the "no new sandboxing needed" design decision holds, but only if Pitfall 3's schema-conformance gap is also closed (a validator that silently drops unexpected keys doesn't clamp, it just discards, which is a different property than actually bounding a value).
- Log the MLLM's full reasoning trace (already planned) specifically so any episode where the reasoning references something unexpected in the frame (an object, text, or instruction not part of the actual task) is retroactively auditable — this is a cheap, already-planned mitigation, just worth calling out explicitly as serving this purpose too.

**Warning signs:**
- MLLM reasoning-trace text references objects, instructions, or context not present in the actual task instruction or camera-visible task-relevant objects.
- No documented practice of keeping the physical workspace visually controlled/minimal specifically for MLLM episodes.

**Phase to address:**
MLLM control loop phase — a cheap, mostly-procedural mitigation (controlled workspace + existing hard clamps + reasoning-trace audit) rather than new engineering, but worth stating explicitly as part of this phase's safety design rather than assumed away by "we already have a safety validator."

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|-----------------|------------------|
| Widen `STALE_ACTION_S`/`control_hz` as the primary latency fix instead of eliminating the redundant observation-resend | Fast, low-risk-looking change; "actions execute now" | Legalizes multi-second-stale actions as normal; masks future latency regressions (Colab cold start, ngrok flakiness) since the system now silently tolerates them | Only as a stopgap alongside a real fix, with the widened value explicitly logged as temporary and reviewed against a measured-latency target, never as the sole fix |
| Reuse `safety_validator.py` for MLLM output with no MLLM-specific schema-conformance check in front of it | No new validator code to write | Reproduces the exact silent-`{}`-action bug (Phase 11 bug #3) under a new root cause (prompt/schema drift instead of a library key format) | Never for a live episode whose results will be recorded/compared — acceptable only for a throwaway smoke test explicitly marked non-representative |
| Fix `camera_overhead`'s recording path at the single call site found, without a shared camera-resolution helper | Smallest possible diff, fastest to ship | The same numeric-index bug can recur at a third call site later, since `device_map.json` still exposes the raw index alongside the name with no enforcement | Never — this bug has already recurred once (two separate episodes hit it independently) under the current unshared-resolution-logic setup |
| Log a client-side-only hash/thumbnail of the frame sent to Colab as "provenance verification" | Satisfies the literal Active-scope item quickly | Doesn't actually prove the model received/used that frame — transport or server-side preprocessing bugs remain undetected, same blind spot class as the camera bug | Acceptable as a first pass only if explicitly documented as "client-side only, does not verify server receipt," with server-side verification as a tracked follow-up |
| Re-tighten safety-validator caps by editing constants back toward old defaults with no live-hardware re-test | Fast, feels like "undoing" the earlier loosening | Repeats the exact class of bug (untested staleness/velocity constant vs real timing) that caused bug #2 in the first place | Never for the final tightened values used in a recorded/compared episode; acceptable only as an interim value pending a live re-test |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|-----------------|-------------------|
| Colab-hosted PolicyServer over ngrok tunnel (both VLA and future MLLM traffic) | Assuming the ~0.5s configured `control_hz` reflects real achievable round-trip time; treating any single fast test call as representative | Measure and log real RTT distribution (not just a point estimate) across cold-start and warm conditions before choosing a fixed `control_hz`/staleness cap; expect and design for the 11-20s range already observed, not the configured ideal |
| `safety_validator.py` reused for MLLM output | Assuming "already validated for VLA" implies "validated for MLLM," when the validator's plain-name key lookup is itself the exact prior failure mode (bug #3) | Add an explicit pre-validator schema-conformance check derived from `action_contract.py`'s actual constants, with a loud (not silent-drop) failure path for mismatched MLLM output |
| `device_map.json` consumed by both the inference path (`args.stereo_camera_index`, name-based) and the recording path (`args.camera`, index-based) | Two code paths independently deriving a camera identity from the same config file via different fields, with no shared resolution function | Introduce one shared camera-resolution helper both paths call; treat any direct `cv2.VideoCapture(<int>)` construction from `device_map.json` as a code-review red flag |
| `cv2.VideoCapture` + AVFoundation backend for the AR0144 stereo camera | Assuming `cv2.VideoCapture` correctly reports/serves the camera's real 2560x720 resolution once opened by name instead of index | The resolution-cap bug (opencv/opencv#23368) is independent of index-vs-name; any camera_overhead recording fix must also route through the `ffmpeg`-backed capture (or equivalent), not just fix the device-identity lookup and keep using `cv2.VideoCapture` |
| MLLM (Claude, paid tier) raw-JSON action output | Assuming natural-language JSON-formatting instructions in the prompt are sufficient to guarantee `action_contract.py`-conformant output every time | Validate every field/key/unit explicitly against `action_contract.py`'s constants before the safety validator ever sees the action; treat conformance failure as a distinct, loud, loggable event |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|-------------|-----------------|
| Action-chunk size for the MLLM set too large before real timing data exists | Large chunks amplify the parse-failure blast radius (Pitfall 4) and increase the odds a chunk executes against an increasingly stale observation as it drains | Size chunks from the latency-fix phase's measured RTT/timing data (per v2.1's stated plan) and start conservative; grow only once parse-success and staleness metrics are validated at a smaller size | Becomes visible on the very first live MLLM episode if chunk size is guessed rather than measured — the same 1/60-yield failure shape Phase 11 hit for a different underlying reason |
| Queue-low request trigger (Pitfall 1/2 fix) polls too infrequently or with too coarse a threshold | The queue still hits empty and stalls before the new chunk arrives, reproducing hold-and-wait behavior indistinguishable from the pre-fix symptom | Tune the "request-ahead" threshold against measured inference latency so a new chunk reliably arrives before the current one drains, not just "queue is empty, now request" | Breaks under any inference-latency variance (Colab cold start, network jitter) that the fixed threshold didn't account for |
| Per-tick camera reads via `cv2.VideoCapture` for any recording path sharing the AR0144 device with the inference path's `ffmpeg`-based `StereoSplitCamera` | Most webcam drivers reject a second concurrent open of the same physical index — per `stereo_camera.py`'s own documented reasoning for opening the device exactly once and sharing reads | Route any recording-path frame capture for the same physical device through the same shared capture object/instance the inference path uses, never a second independent open | Breaks immediately (device busy / silent fallback to a different camera) if a naive recording-path fix opens the AR0144 a second time by name instead of sharing `StereoSplitCamera`'s existing open |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| MLLM prompt/reasoning influenced by incidental visual content (text, screens, printed material) in either camera's field of view | Unpredictable action output shifted by content never intended as a task instruction — a threat class the VLA baseline structurally cannot exhibit (Pitfall 9) | Keep the physical workspace visually controlled/minimal during MLLM episodes; rely on the hard safety clamps (not prompt discipline alone) as the actual behavioral boundary |
| Provider API credentials (Claude paid-tier key) handled the same casually as the existing free-tier/HF pilot assumptions from earlier research | Credential leak if reasoning-trace logs or episode data (already pushed to HF Hub in this project's history) ever get shared/published with an embedded key or trace referencing it | Keep API credentials out of any logged/recorded field (prompt text, reasoning trace, episode metadata); use environment variables only, never inline in prompts that get persisted |
| Safety-validator schema check treats an unexpected/malformed MLLM key as "just drop it" rather than "reject and alert" | A schema-drifted action silently degrades to a no-op or partial action instead of being caught as a distinct, investigable failure — this is functionally a safety-relevant silent failure, not just a data-quality issue, since it can mean fewer-than-commanded joints move while the loop believes the action executed | Make schema-conformance failures loud (raise/log distinctly, do not silently drop keys) at the pre-validator stage described in Pitfall 3 |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-------------------|
| No live operator-facing signal distinguishing "held position because queue is draining slowly" from "held position because an action was discarded as stale" from "held position because MLLM output failed schema validation" | Operator watching the arm during a live episode can't tell which of several very different problems is occurring, slowing diagnosis exactly the way Phase 11's episode required post-hoc `episode.jsonl` archaeology to explain | Surface a distinct, human-readable status per hold reason in real time (console/log), reusing the same three-way distinction `FINDINGS.md` §2 needed to reconstruct after the fact |
| Recorded episode footage treated as self-evidently trustworthy once a hash/thumbnail check "passes" | A human stops manually reviewing saved frames because an automated check exists, re-creating the exact blind spot that let the wrong-camera bug persist across two episodes | Keep a lightweight manual-spot-check step in the episode-review process even after automated verification exists, at least until the camera-resolution fix has a track record |
| Camera-role naming in `device_map.json` (`stereo_overhead` vs `stereo_overhead_name`) implies both are equally valid ways to reference the same camera | A future contributor (or future-you) reasonably assumes the numeric field is safe to use directly, since nothing in the schema itself signals otherwise | Rename or annotate the numeric field to make clear it's not the authoritative identifier for cameras with a `_name` counterpart, or remove it from direct use entirely |

## "Looks Done But Isn't" Checklist

- [ ] **Tick-latency fix:** Often "done" once the discard-rate log lines disappear — verify by measuring actual wall-clock time from observation-sent to action-executed timestamps in `episode.jsonl`, not just the absence of stale-discard messages.
- [ ] **Queue-low request trigger:** Often works in fast/warm testing but races under slow Colab responses — verify by testing under a simulated slow/degraded network condition, not just a repeat of the same fast happy-path episode.
- [ ] **MLLM safety-validator reuse:** Often assumed "already covered" because the validator exists — verify by feeding it a deliberately malformed/mis-keyed MLLM-shaped action and confirming a loud rejection, not a silent empty/partial action (this exact gap already caused a real bug once).
- [ ] **Camera recording-path fix:** Often fixed at the one call site found — verify by grepping for every `cv2.VideoCapture`/`ffmpeg` construction across `control/` that touches a `device_map.json` camera field, confirming all of them resolve by name through one shared helper, not just the one already reported.
- [ ] **Camera frame provenance (hash/thumbnail):** Often implemented client-side only — verify by confirming (or explicitly documenting the inability to confirm) that the hash also reflects what the Colab-side PolicyServer actually received, not just what the bridge sent.
- [ ] **Safety-validator cap re-tightening:** Often treated as a safe "revert" — verify with an actual live-hardware episode showing real-action yield at the tightened values, not just a code review confirming the numbers match the pre-loosening defaults.
- [ ] **`latency_ms` logging fix (v2.0 tech debt, carried into v2.1):** Often "fixed" by populating the field with *some* number — verify the number is derived from the actual request/response timestamps used to diagnose the original bug, not a different or approximate measurement that looks populated but means something else.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|-----------------|------------------|
| Latency fix ships as constant-widening only, later discovered to have masked a real latency regression | MEDIUM | Add the missing real-latency metric retroactively, re-run a live episode, and if the widened constants are masking a genuine regression, treat it as reopening the original bug rather than a new one — same root cause class |
| MLLM schema drift silently empties actions during a live episode (repeat of bug #3's shape) | LOW-MEDIUM | Because raw MLLM responses are already planned to be logged in full (reasoning-trace requirement), the malformed responses can be recovered and used to harden the schema-conformance check without re-running hardware — same recovery path bug #3 itself used |
| Camera recording-path fix addresses the reported call site but a third path is later found still using the numeric index | LOW | Since the underlying `device_map.json` schema issue (index and name both present) is the root cause, adding the shared resolution helper retroactively and auditing all call sites is a contained, mechanical fix once flagged |
| Safety caps re-tightened without live-hardware verification turn out too conservative, silently discarding real MLLM/VLA actions again | LOW | Identical recovery to the original bug #2: diagnose from `episode.jsonl` staleness-discard messages, loosen incrementally with logged justification, re-test |
| A live episode's footage is later discovered to have captured the wrong camera despite the fix (e.g., a regression) | LOW | Because the underlying joint/state/action data in `episode.jsonl` is independent of the camera recording bug, the episode's control-loop data remains valid evidence even if its footage must be discarded/re-captured — don't throw out the whole episode, just flag the visual record as unverified |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|-------------------|----------------|
| Latency fix trades staleness-discard for staleness-tolerance (constant-widening masquerading as a fix) | Bridge tick-latency fix phase | Measured observation-to-execution wall-clock latency against a stated numeric target, not just discard-rate reduction |
| New race condition from an unguarded queue-low request trigger | Bridge tick-latency fix phase | Unit test asserting no duplicate in-flight observation requests under simulated slow-response conditions |
| Safety validator reused for MLLM without a schema-conformance check | MLLM control loop phase | Deliberately malformed MLLM-shaped action test asserts loud rejection, not silent empty/partial action |
| Parse failures within a multi-action raw-JSON chunk | MLLM control loop phase | Per-chunk and per-episode parse-success-rate logging; malformed mid-chunk action holds position, never defaults/zero-fills |
| Repeating the earlier "failed badly" general-MLLM attempt's undiagnosed root cause | MLLM control loop phase (pre-implementation scoping) | Documented root-cause note exists before first live episode; early chunks smoke-tested against known-good SmolVLA trajectory shape |
| Camera recording-path fix scoped to one call site instead of the shared pattern | Camera device resolution fix phase | Grep-confirmed audit: every camera-opening call site in `control/` resolves via one shared, name-based helper |
| Client-side-only frame hash/thumbnail insufficient as end-to-end provenance | Camera device resolution fix phase (hash/thumbnail logging); informs MLLM reasoning-trace capture too | Hash/thumbnail check covers (or explicitly documents not covering) the Colab-side receipt, not just the bridge's send |
| Safety-cap re-tightening repeats the untested-constant risk that caused bug #2 | Bridge tick-latency fix phase (post-fix step) | Live-hardware episode with tightened caps shows measured real-action yield before being treated as final |
| MLLM prompt-injection-shaped risk from incidental visual content | MLLM control loop phase | Controlled workspace practice documented; reasoning-trace audit confirms no unexpected object/text references drove action output |

## Sources

- `control/vla_bridge/FINDINGS.md` (this project) — HIGH confidence, primary source; documents all three Phase 11 bugs (missing background thread, staleness-constant mismatch, `.pos`-suffix key mismatch), the measured ~11-20s/tick latency, the 1/60 real-action yield, and the explicit go/no-go recommendation this research extends
- `control/vla_bridge/robot_client.py`, `safety_validator.py`, `stereo_camera.py` (this project, read directly) — HIGH confidence, primary source; confirms `STALE_OBSERVATION_S`/`STALE_ACTION_S` constants, `pop_validated_action()`'s key-lookup behavior, and `StereoSplitCamera`'s documented rationale for the `ffmpeg`-over-`cv2.VideoCapture` backend switch (citing opencv/opencv#23368)
- `control/run_vla_episode.py`, `control/device_map.json` (this project, read directly) — HIGH confidence, primary source; confirms the exact numeric-index-vs-name-based divergence between the recording path (`args.camera`, `cv2.VideoCapture(idx)`) and the inference path (`args.stereo_camera_index`, name-resolved)
- `.planning/PROJECT.md` (this project) — HIGH confidence, primary source; v2.1 milestone scope, the earlier "failed badly" general-MLLM-prompting attempt, and the confirmed-live camera-bug write-up
- `.planning/research/PITFALLS.md` (this project's own earlier v2.0-era research pass, 2026-09-15) — HIGH confidence as prior internal research; largely superseded/sharpened by this pass now that Phase 11's actual bugs and the confirmed camera bug exist as evidence, but its safety-envelope, structured-output, and pixel-to-mm-grounding pitfalls remain valid background for the MLLM phase
- [Async autonomous loops: re-verify state at decision time, not observation time](https://github.com/remigiusz-antczak/deep-code-review/issues/421) — MEDIUM confidence; general async observe-act-gap pattern underlying Pitfalls 1-2
- [Understanding Asynchronous Inference Methods for Vision-Language-Action Models](https://arxiv.org/html/2605.08168) — MEDIUM confidence; execution-horizon/prediction-horizon chunking and the prediction-execution mismatch tradeoff underlying Pitfall 1
- [Structured Output From LLMs: A Retry-Repair Loop Your Parser Never Sees Through](https://dev.to/devshakib/structured-output-from-llms-a-retry-repair-loop-your-parser-never-sees-through-3b0b) and [LLM Output Parsing and Structured Generation Guide](https://tetrate.io/learn/ai/llm-output-parsing-structured-generation) — MEDIUM confidence; JSON malformation modes (fencing, trailing prose, hallucinated fields) underlying Pitfalls 3-4
- [Precise Robot Command Understanding Using Grammar-Constrained Large Language Models](https://arxiv.org/pdf/2604.04233) — MEDIUM confidence; constrained-decoding/schema-validation-plus-retry pattern for robot-directed LLM output
- [Camera id problem with multi camera system (MacOS) · opencv/opencv#22901](https://github.com/opencv/opencv/issues/22901) and [videoio: Added camera device enumeration · opencv/opencv#29843](https://github.com/opencv/opencv/pull/29843) — MEDIUM confidence; confirms macOS camera-index instability across `cv2.VideoCapture` instances and OpenCV's own move toward name-based enumeration, underlying Pitfall 6
- [macOS 15: cv2.VideoCapture Incompatibility with Continuity Camera · opencv/opencv#26371](https://github.com/opencv/opencv/issues/26371) — MEDIUM confidence; corroborates index/identity instability for external/virtual cameras on recent macOS versions
- [FFmpeg Devices Documentation](https://ffmpeg.org/ffmpeg-devices.html) and [Capture/Webcam – FFmpeg](https://trac.ffmpeg.org/wiki/Capture/Webcam) — MEDIUM confidence; confirms `-list_devices`/name-vs-index selection semantics for AVFoundation, and that devices sharing a display name are disambiguated only via an OS-level Unique ID not directly exposed to ffmpeg's own device list
- [On the Vulnerability of LLM/VLM-Controlled Robotics](https://arxiv.org/pdf/2402.10340), [Enhancing Reliability in LLM-Integrated Robotic Systems](https://arxiv.org/pdf/2509.02163) — MEDIUM confidence; named risk classes (hallucinated plans, prompt-injection-shaped vulnerability surface, physical irreversibility of embodied errors) underlying Pitfall 9
- Data provenance in robotics (checkpoint-based tracing, "what world did the model actually see") — LOW confidence, thin/tangential sourcing (patent filings, one blog post); treated as directional support for Pitfall 7's reasoning rather than an authoritative citation — the project's own history (the camera bug caught only by manual review) is the stronger evidence for this pitfall

---
*Pitfalls research for: Real-hardware MLLM-as-controller addition, async control-loop latency fix, and macOS camera device resolution fix — SoARM VLA Research v2.1*
*Researched: 2026-09-24*
