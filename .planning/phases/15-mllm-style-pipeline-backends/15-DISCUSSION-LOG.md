# Phase 15: MLLM-Style Pipeline & Backends - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-10-02
**Phase:** 15-MLLM-Style Pipeline & Backends
**Areas discussed:** Action format & failures, Prompt & observations, Reasoning-trace schema, Backend model selection

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
