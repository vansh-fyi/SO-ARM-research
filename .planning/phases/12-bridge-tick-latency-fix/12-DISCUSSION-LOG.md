# Phase 12: Bridge Tick-Latency Fix - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-25
**Phase:** 12-Bridge Tick-Latency Fix
**Areas discussed:** Latency target number / staleness cap, DEBT-03 scope discrepancy, In-flight-request guard shape, Live-verification episode ownership

---

## Latency target number / staleness cap (STALE_ACTION_S)

| Option | Description | Selected |
|--------|-------------|----------|
| Match measured baseline (~15-20s) | Set STALE_ACTION_S generously around observed round-trip | |
| Tighter target (e.g. 3-5s) | Push for faster round-trip, tighter staleness | |
| No fixed number — gate-driven fix is the point | Real fix is the observation gate, not tuning the constant | |

**User's choice:** Requested clarification — didn't understand the framing initially. Proposed their own mental model: "get a chunk of 50 actions, perform them, then request more — why are actions being called stale at all?"

**Follow-up round:**

| Option | Description | Selected |
|--------|-------------|----------|
| Leave STALE_ACTION_S at 30s, don't touch it | Safety fallback only, not a tuning knob for this bug | ✓ |
| Lower it now as a tightening pass | Tighten now that it should rarely trigger post-fix | |

**User's choice:** Leave at 30s, don't touch it.
**Notes:** Claude explained the mechanism in plain terms first — the bug is that observations are sent every tick instead of only when the queue needs refilling; once fixed, the intended loop (request chunk → drain ~50 actions locally at ~0.5s/tick → request next chunk) means a healthy chunk drains in ~25s, comfortably under the existing 30s staleness cap. User confirmed this matches their mental model and that STALE_ACTION_S should be treated as a rarely-triggered safety fallback, not something this phase tunes. Success is measured by real-action yield on the live episode, not by this constant.

---

## DEBT-03 scope discrepancy

| Option | Description | Selected |
|--------|-------------|----------|
| Drop DEBT-03 — already resolved / doesn't apply | So-101/ already tracked, coppelia/ doesn't exist | |
| Check for other untracked mesh dirs first | Broader search before concluding | |

**User's choice:** Asked "Should I make a coppelia directory putting in the necessary exports in?" — a different direction than either option (creating new content rather than tracking existing untracked files).

**Follow-up round 1:** Claude investigated and reported: "coppelia" in this repo refers to a one-time local CoppeliaSim diagnostic reference (`.ttm` file, used in Phase 10 for mesh-alignment verification, documented in `diagnostics/COPPELIA_MUJOCO_ALIGNMENT.md`), not a mesh-asset directory the URDF depends on. `So-101.urdf` only references mesh files already tracked within `So-101/`.

| Option | Description | Selected |
|--------|-------------|----------|
| DEBT-03 is moot — mark not applicable | Nothing to fix, note in REQUIREMENTS.md/CONTEXT.md | |
| Track the diagnostic outputs anyway | Add the untracked .cbor cache file for completeness | |

**User's choice:** "what do you suggest?" — asked Claude to make the call.

**Follow-up round 2:** Claude recommended marking DEBT-03 not applicable (the `.cbor` file is a regenerable derived cache, not a real gap, and inventing scope for a nonexistent debt item risks wasting a planner task).

**User's choice:** Yes, mark not applicable.
**Notes:** This required real investigation (checking `git status`/`git ls-files` on `So-101/`, searching the repo for any `coppelia/` directory, reading `diagnostics/COPPELIA_MUJOCO_ALIGNMENT.md`) since the roadmap's DEBT-03 description turned out to be based on stale/inaccurate information about the repo state.

---

## In-flight-request guard shape (LATENCY-04)

| Option | Description | Selected |
|--------|-------------|----------|
| Defensive guard, no specific incident | Simple flag/lock, cheap insurance | |
| I hit a specific overlap case | Concrete scenario driving this | |

**User's choice:** Requested clarification — didn't understand the framing initially.

**Follow-up round:** Claude explained the concrete scenario: today `get_action()` is synchronous/single-threaded so no overlap is possible; the guard is insurance against a *future* timeout/retry pattern (e.g. if a hung Colab call needed a timeout so the control loop doesn't freeze forever) reintroducing a race where two outstanding requests could return out of order.

| Option | Description | Selected |
|--------|-------------|----------|
| Just the defensive flag, no timeout | Simple boolean/lock, no new timeout/retry logic | ✓ |
| Also add a timeout so hangs don't freeze the loop | Bigger scope: real timeout + retry | |

**User's choice:** Just the defensive flag, no timeout.
**Notes:** Confirmed as pure defensive insurance, not fixing an active bug — consistent with the phase's "location-and-wire-up, not design" framing.

---

## Live-verification episode ownership

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, I run the live episode myself | Claude implements + tests, user runs live hardware | ✓ |
| Something else / let me clarify | — | |

**User's choice:** Yes, I run the live episode myself.
**Notes:** Confirms same division of labor as Phase 11 — Claude cannot run physical hardware in this environment.

---

## Claude's Discretion

- Exact in-flight-guard mechanism (boolean flag vs. `threading.Lock`).
- Exact latency-measurement bracket points feeding `io_logger.py`'s `latency_ms` dict (`observation_to_action`, `action_to_execution` sub-fields).

## Deferred Ideas

- Re-tuning `STALE_ACTION_S`/`STALE_OBSERVATION_S` toward real measured latency, if the post-fix live episode shows the gate fix alone wasn't sufficient — not scheduled to any phase, revisit only if needed.
- Adding a request timeout/retry mechanism for hung Colab calls — explicitly deferred, not scheduled.
- Tracking `diagnostics/outputs/coppelia_reference.cbor` and other untracked `diagnostics/` derived artifacts — explicitly declined for this phase, not scheduled to a future phase.
