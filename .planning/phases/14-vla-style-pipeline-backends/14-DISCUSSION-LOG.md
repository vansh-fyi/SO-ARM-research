# Phase 14: VLA-Style Pipeline & Backends - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-30
**Phase:** 14-VLA-Style Pipeline & Backends
**Areas discussed:** Phase 13 gate check, Backend scope expansion, Backend hosting/transport, Depth role, ActionSource abstraction shape, Integration order & sequencing, Backend selection & config surface

---

## Phase 13 Gate Check

STATE.md shows Phase 13 isn't formally closed — a fresh live episode still needs to confirm overhead/wrist camera views, and Phase 14 depends on Phase 13's corrected camera pipeline.

| Option | Description | Selected |
|--------|-------------|----------|
| Proceed with Phase 14 discussion | Phase 13's code/logic is complete; only live re-verification remains | ✓ |
| Stop and close Phase 13 first | Don't discuss Phase 14 until Phase 13's live UAT passes | |

**User's choice:** Proceed with Phase 14 discussion.

---

## Backend Scope Expansion

User's initial free-text answer to the gray-area selection question included a much broader model list than the 4 backends locked in ROADMAP.md (Gemini Robotics, π0/π0.5, Wall-X, ACT): OpenVLA, OpenVLA-OFT, Octo, RDT-1B, π0, π0.5, GR00T N1/N1.5/N1.7, Gemini Robotics 2, SmolVLA (existing), Wall-X, ACT, Diffusion Policy.

**Round 1 — how to use the list:**

| Option | Description | Selected |
|--------|-------------|----------|
| Keep Phase 14 at the locked 4 | Rest of list becomes deferred backlog | |
| Swap some of the locked 4 for list items | Targeted substitutions | |
| Expand Phase 14 to cover more than 4 backends now | Roadmap-level scope change | ✓ |

**Round 2 — which backends (locked 4 confirmation):**

Selected: Gemini Robotics / Gemini Robotics 2, π0 and π0.5, Wall-X, ACT.

**Round 2 — additional backends beyond the locked 4:**

Selected: OpenVLA / OpenVLA-OFT, Octo / RDT-1B, GR00T N1 / N1.5 / N1.7, Diffusion Policy.

**Round 3 — Gemini version:**

| Option | Description | Selected |
|--------|-------------|----------|
| Gemini Robotics 2 only | Newer version | |
| Gemini Robotics (original) only | Matches original roadmap text | |
| Both, as separate backends | Two distinct ActionSource integrations | ✓ |

**Round 3 — GR00T version(s):**

Selected: GR00T N1.7, GR00T N1.5 (GR00T N1 original not selected).

**Round 3 — scale confirmation (10-13 backends in one phase):**

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, all in Phase 14 as one big sweep | Keep full list in this phase | ✓ |
| Split: core set now, rest deferred | Push some to a later phase | |

**Final locked backend list (12 new + SmolVLA baseline = 13 VLA-style total):** Gemini Robotics, Gemini Robotics 2, π0/π0.5 (kept combined, matching original roadmap phrasing), Wall-X, ACT, OpenVLA, OpenVLA-OFT, Octo, RDT-1B, GR00T N1.7, GR00T N1.5, Diffusion Policy.

**Notes:** This required updating `ROADMAP.md` (Phase 14 goal/success criteria/requirements list, Phase 16 backend count 8→16, intro paragraph) and `REQUIREMENTS.md` (VLAB-01..12 replacing VLAB-01..04, EVAL-01/02 wording, traceability table, coverage counts) before continuing the discussion — done live during this session with explicit confirmation at each step, including a dedicated follow-up question about whether Phase 16's "8 backends" language should be synced now (user: yes).

---

## Backend Hosting / Transport

| Option | Description | Selected |
|--------|-------------|----------|
| Two transport lanes | Colab-bridge for open-weight checkpoints, direct-HTTP for Gemini/Gemini 2 | ✓ |
| Everything through Colab, including Gemini | Uniform relay, extra hop for Gemini | |
| You decide per-backend as each lands | No rule locked now | |

**User's choice:** Two transport lanes.
**Notes:** User's first response to this question was actually about depth inference management, not transport — addressed as a preceding sub-question (below) before returning to confirm transport.

## Depth Role

| Option | Description | Selected |
|--------|-------------|----------|
| Keep depth log-only for all 12 | Matches SmolVLA/DEPTH-CAL-03 today, regardless of transport lane | ✓ |
| Feed depth into backends natively trained with depth input | Per-backend observation-dict differences | |

**User's choice:** Keep depth log-only for all 12.

---

## ActionSource Abstraction Shape

| Option | Description | Selected |
|--------|-------------|----------|
| Shared bridge base + per-backend subclass | Refactor BridgeActionSource into base + thin subclasses; separate Gemini-family base for cloud lane | ✓ |
| One fully generic, config-driven class | No subclassing | |
| Leave BridgeActionSource alone, new class per backend | 12 independent implementations | |

**User's choice:** Shared bridge base + per-backend subclass.

---

## Integration Order & Sequencing

| Option | Description | Selected |
|--------|-------------|----------|
| Familiarity/risk-ascending | OpenVLA-OFT first (prior project familiarity), then other Colab-hosted, then cloud-API last | ✓ |
| By model family/category | Group by original list's categories | |
| You decide order at plan time | No order locked now | |

**User's choice:** Familiarity/risk-ascending.

---

## Backend Selection & Config Surface

| Option | Description | Selected |
|--------|-------------|----------|
| New --backend flag + per-backend config file | run_vla_episode.py gets --backend; settings in small per-backend config | ✓ |
| Extend existing --checkpoint arg only | Infer backend from checkpoint string | |
| You decide at plan time | No shape locked now | |

**User's choice:** New --backend flag + per-backend config file.

---

## Claude's Discretion

- Exact config file format/location for per-backend settings (JSON vs. `.env` vs. Python registry module).
- Exact class/module naming for the refactored ActionSource base classes.

## Deferred Ideas

None — the expanded backend list was folded directly into Phase 14's scope rather than deferred.
