# Phase 11: VLA Hardware Connection - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-20
**Phase:** 11-VLA Hardware Connection
**Areas discussed:** Inference compute location, First observed task, E-stop mechanism, I/O log format, Reasoning-trace scope (scope-creep redirect)

---

## Inference compute location

| Option | Description | Selected |
|--------|-------------|----------|
| Local laptop (CPU/MPS) | Same Python 3.12 control/ venv, no network hop, but slow inference risk | |
| Colab GPU + local relay | Matches Phase 3's compute split; faster, but new relay plumbing needed | ✓ |
| Undecided | Let research/planning resolve | |

**User's choice:** Colab GPU + local relay
**Notes:** No specific protocol requested — left as Claude's discretion for research/planning to design.

---

## First observed task

| Option | Description | Selected |
|--------|-------------|----------|
| Pen Transfer (paper-aligned) | Matches Yu & Qiu 2026 benchmark's simplest task | |
| Simplest available pick-place | Any small in-reach object on hand | (superseded by free-text) |
| Undecided | Defer to planning | |

**User's choice (free-text):** "I have a small red cube, lets work with that"
**Notes:** Object already on hand; must still respect the ≤84mm / ~0.45m reach / avoid-forward-centerline constraints already documented in STATE.md — exact placement/prompt deferred to planning.

---

## E-stop mechanism

| Option | Description | Selected |
|--------|-------------|----------|
| Keyboard interrupt (software) | Matches existing keyboard_joint_control.py pattern, no new hardware | ✓ |
| Physical hardware switch | More robust but new hardware scope explicitly deferred elsewhere | |
| Undecided | Defer to research/planning | |

**User's choice:** Keyboard interrupt (software)

---

## I/O log format (VLAHW-03) — multi-turn

Initial pass offered CSV+images vs. structured-file vs. "discuss more." User flagged this needed alignment with a research partner's requirements first.

**Follow-up 1 — What does the partner need?**
User's free-text: partner wants to see inputs, "the entire reasoning thread and tokens," and outputs — i.e. an LLM-style reasoning trace.

**Scope flag raised:** SmolVLA (this phase's VLA) has no reasoning trace/tokens — it's a direct observation→action model, not an LLM. ROADMAP.md's VLAHW-03 already anticipated this and frames the requirement as "complete I/O," not "reasoning trace," for exactly this reason.

**Follow-up 2 — How to reconcile?**
User's free-text: proposed running SmolVLA first, then trying an MLLM afterward with the logs from the first pass, and asked for a recommendation (paid credits vs. free HF-hosted reasoning model on Colab GPU).

**Scope-creep redirect:** This maps directly onto PROJECT.md's already-deferred "Future Requirements" item (comparable deep-reasoning-MLLM experiment, piloted on a free/cheap HF model before paid providers) — it is the *next* phase, not part of Phase 11. Presented as a scope-confirmation question.

| Option | Selected |
|--------|----------|
| Phase 11 stays SmolVLA-only; MLLM comparison becomes the next phase | ✓ |
| Expand Phase 11 to cover both SmolVLA and an MLLM run | |

**User's choice:** Keep Phase 11 SmolVLA-only; MLLM comparison deferred to next phase.

**Follow-up 3 — Concrete log format, given it must stay reusable for that future phase:**

| Option | Description | Selected |
|--------|-------------|----------|
| Single structured file per episode (JSON Lines) | One JSON record per step; schema-extensible (can add `reasoning_trace` later without migration) | ✓ (Claude recommendation, user confirmed) |
| CSV + images (match record_episode.py) | Matches existing convention but poor at nested/variable-length fields | |

**User's choice:** JSON Lines — asked "what do you suggest," Claude recommended JSON Lines with rationale, user confirmed "Yes, lock it in."

---

## Claude's Discretion

- Concrete safety-validator numeric thresholds (joint limits, max displacement/velocity, gripper bounds) — derive from real servo specs during planning/research
- Exact local↔Colab relay protocol/transport
- Exact cube placement and task-prompt wording (within existing reach/collision constraints)

## Deferred Ideas

- Deep-reasoning MLLM comparison experiment (reusing Phase 11's JSON Lines log + a new `reasoning_trace` field) — captured as the next phase to plan after Phase 11, not folded into it. Free/cheap HF-hosted model to be piloted before paid providers, per the user's own suggestion and PROJECT.md's existing decision.
