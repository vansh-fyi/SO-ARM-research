# Phase 15: MLLM-Style Pipeline & Backends - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-10-02
**Phase:** 15-MLLM-Style Pipeline & Backends
**Areas discussed:** Action format & failures, Prompt & observations, Reasoning-trace schema, Backend model selection, Step timing, Episode termination, Sampling & reproducibility, Per-call records

---

## Action format & failures

| Question | Options | Selected |
|----------|---------|----------|
| Action form | Absolute joint targets per tick / Sparse waypoints + interp / Per-step deltas | Absolute per tick |
| Chunk size | Config from measured latency / Fixed shared constant / Model chooses | Config from measured latency |
| Bad value in chunk | Hold from that step on / Discard chunk / Retry once | Hold from that step on |
| Out-of-range values | Validator only / Schema min-max + validator | Validator only |

## Prompt & observations

| Question | Options | Selected |
|----------|---------|----------|
| Cameras | Wrist+left / Left only / Wrist+both halves | **Other:** max input — left, right, depth, wrist, URDF (also noted SmolVLA should get all 3; verified it already does) |
| Static context | Limits only / +URDF / Minimal | Limits only (then overridden by follow-up) |
| Cadence | Closed-loop per chunk / Open-loop | Closed-loop per chunk |
| Memory | Previous plan only / Stateless / Full history | Previous plan only |
| Follow-up: depth | Yes / Image only / Log-only | Yes, MLLM gets depth |
| Follow-up: URDF | Full / Limits only / Toggle | Full (toggle permitted) |

## Reasoning-trace schema

| Question | Options | Selected |
|----------|---------|----------|
| Trace production | One structured JSON / Native thinking + JSON / Two calls | Native thinking + JSON (staged fields kept in schema — Claude's reading) |
| Grounded claim | Claim + source image / Free text | Claim + source |
| Logging | Per-chunk + pointer / Every step | Per-chunk + pointer |

## Backend model selection

| Question | Options | Selected |
|----------|---------|----------|
| HF host | Inference Providers API / Colab / Researcher decides | Inference Providers API |
| Claude/GPT | Flagship / Mid-tier / Researcher matches | Flagship each |
| Structured output | Native JSON-schema / Prompt-only | Native JSON-schema |

## Claude's Discretion

Depth presentation, image encoding, module naming, URDF text format, HF fallback when no schema mode.

## Deferred Ideas

None.

---

## Update session (2026-10-02, via "Update it")

## Step timing semantics

| Question | Options | Selected |
|----------|---------|----------|
| What to tell the model | State dt + motion budget / dt only / No timing | State dt + motion budget |
| Chunk length | Fixed N shared default / Model chooses within cap | Fixed N |

**Notes:** "Motion budget" interpreted as qualitative (no numeric caps disclosed) because caps are loosened now and change in Phase 17.

## Episode termination

| Question | Options | Selected |
|----------|---------|----------|
| Who ends episode | Model signal + budget / Budget only / Model signal logged only | Model signal + budget |
| Claim vs human judgment | Log both separately / Out of scope | Log both separately |

## Sampling & reproducibility

| Question | Options | Selected |
|----------|---------|----------|
| Sampling policy | Provider defaults pinned+logged / Temp 0 where allowed / Same nominal everywhere | Same nominal settings everywhere |
| Reasoning level | Max/high / Medium matched / Provider default | Max/high |
| Rejected setting | Omit + log not-applied / Fail at construction | Omit + log not-applied |

## Per-call records (user-added area)

User asked what to keep per model besides input/output, namely reasoning tokens. Selected all four: token usage breakdown, request params + provenance, latency + cost, raw response payload.

## Claude's Discretion

Numeric safety-cap disclosure toggle; task_complete chunk tail rule; sidecar file layout.

## Deferred Ideas

None.

## HF model lineup (follow-up, web search)

Candidates found via HF Hub API and web search: Qwen3.8-27B, DeepSeek-V4-Flash-Vision-Exp, Kimi-K3, GLM-5.3-Flash, Gemma-4-31B-it. **Selected:** Qwen3.8-27B, DeepSeek-V4-Flash-Vision-Exp, Kimi-K3 as registered variants; researcher verifies vision/thinking/JSON support and drops failures.

## Post-research questions (2026-10-02)

| Question | Selected |
|----------|----------|
| Prompt dt: measured 0.785 s vs nominal 0.5 s | Measured 0.785 s |
| task_complete meaning | User: "when the model judges the goal is done or when the max steps are reached" (recorded as D-25) |
| Left image: aligned vs raw | User did not know about rectification and expected raw left/right with depth computed from them; decided by Claude as raw default plus `left_image_source` toggle (D-26) |
| DeepSeek: Vision-Exp vs V4.1-Flash | Keep Vision-Exp |

## Depth presentation (follow-up)

User asked whether to send the raw numpy array instead of a heatmap. Explained full array is ~1.4M tokens and not attachable; selected **both heatmap + coarse numeric grid** (D-28).
