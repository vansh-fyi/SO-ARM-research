# Phase 14: VLA-Style Pipeline & Backends - Context

**Gathered:** 2026-09-30
**Status:** Ready for planning

<domain>
## Phase Boundary

A consistent VLA-style `ActionSource` pattern that the already-working SmolVLA integration fits unmodified, and 12 new native-action/imitation-learning backends each drive the real SO-ARM101 through it, reusing the shared control loop, `safety_validator.py`, and `io_logger.py` unchanged.

**Backend set (expanded 2026-09-30 during this discussion — see Implementation Decisions D-00):** Gemini Robotics, Gemini Robotics 2, π0/π0.5, Wall-X, ACT, OpenVLA, OpenVLA-OFT, Octo, RDT-1B, GR00T N1.7, GR00T N1.5, Diffusion Policy (12 new, plus the existing SmolVLA baseline = 13 VLA-style backends total).

Depends on Phase 13 (corrected camera pipeline). Parallel-capable with Phase 15 (MLLM-style pipeline, different backend set).

</domain>

<decisions>
## Implementation Decisions

### Scope Change
- **D-00:** Phase 14's backend set was expanded from the original 4 (Gemini Robotics, π0/π0.5, Wall-X, ACT) to 12 during this discussion, per explicit user request covering the project's full candidate VLA/imitation-learning model list. `ROADMAP.md` (Phase 14 + Phase 16 counts) and `REQUIREMENTS.md` (VLAB-01..12, EVAL-01/02, traceability table, coverage counts) were updated in place to reflect this before planning starts — not left as a discussion-only note. See `ROADMAP.md` Phase 14 Context/Notes and `REQUIREMENTS.md`'s 2026-09-30 addendum for the full diff record.

### Backend Hosting / Transport
- **D-01:** Two transport lanes. Colab-hosted open-weight checkpoints (SmolVLA + all 10 new open-weight backends: π0/π0.5, Wall-X, ACT, OpenVLA, OpenVLA-OFT, Octo, RDT-1B, GR00T N1.7, GR00T N1.5, Diffusion Policy) reuse the existing async_inference gRPC bridge pattern — one LeRobot PolicyServer per checkpoint, swappable via config, same shape as today's `BridgeActionSource`.
- **D-02:** Gemini Robotics and Gemini Robotics 2 (cloud APIs, not Colab checkpoints) get a separate, direct-HTTP `ActionSource` implementation calling Google's API straight from the control-loop machine — no Colab hop, no pointless relay.

### Depth Role (across all transport lanes)
- **D-03:** Depth stays log-only for all 12 new backends, identical to current SmolVLA behavior — computed locally via stereo camera → Colab-hosted Fast-FoundationStereo Flask service (`fastfs_server.py`), logged to `episode.jsonl` via `IOLogger`, never added to any backend's observation dict. This matches `DEPTH-CAL-03`'s explicit scope boundary (wiring depth into model input is deferred to a later milestone) and applies uniformly whether the backend is on the Colab-bridge lane or the direct-cloud-API lane — the depth pipeline itself is independent of which backend is driving actions, same as it was under SmolVLA.

### ActionSource Abstraction Shape
- **D-04:** Refactor `BridgeActionSource` into a shared base class handling the gRPC/observation/safety-validation plumbing common to all 10 Colab-hosted checkpoints (SmolVLA behavior preserved), with a thin per-backend subclass (or config) only for checkpoint-specific quirks (action format, prompt template, JOINT_ORDER mapping if it differs). A separate `GeminiActionSource`-family base class handles the shared HTTP/auth plumbing for the 2 cloud-API backends. Do not write 12 fully independent ActionSource classes from scratch, and do not collapse everything into one fully generic config-only class — backends may need genuinely different logic, not just different config values.

### Integration Order & Sequencing
- **D-05:** Familiarity/risk-ascending order. Start with OpenVLA-OFT (prior project familiarity from v1 sim work, Phases 1/6) to prove out the refactored Colab-bridge base class, then the remaining Colab-hosted checkpoints, then the 2 cloud-API backends (Gemini Robotics, Gemini Robotics 2 — genuinely new transport lane) last. Each backend integration remains independently live-hardware-verifiable (at least one observed action per backend) — do not batch backends behind a single end-to-end test.

### Backend Selection & Config Surface
- **D-06:** `run_vla_episode.py` gets a new `--backend` flag (e.g. `smolvla`, `pi0`, `pi05`, `wallx`, `act`, `openvla`, `openvla-oft`, `octo`, `rdt1b`, `groot-n17`, `groot-n15`, `diffusion-policy`, `gemini`, `gemini2`). Backend-specific settings (checkpoint path/HF repo ID for Colab-hosted backends, API key/endpoint URL for the 2 cloud backends) live in a small per-backend config (e.g. `control/vla_bridge/backends/<name>.json` or equivalent), keeping the existing `--checkpoint` arg as the Colab-lane-specific value it already is rather than overloading it to also select backend type.

### Claude's Discretion
- Exact config file format/location for per-backend settings (D-06) — JSON file vs. `.env` entries vs. a small Python registry module — left to planning/implementation as long as it's one small config per backend, not hardcoded inline.
- Exact class/module naming for the refactored base classes (D-04) — left to planning.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope (updated this session)
- `.planning/ROADMAP.md` Phase 14 section — updated goal, success criteria, and backend list (2026-09-30)
- `.planning/ROADMAP.md` Phase 16 section — updated backend count (8 → 16) for consistency with Phase 14's expanded scope
- `.planning/REQUIREMENTS.md` VLAB-01..12, EVAL-01/02, traceability table, coverage counts — updated 2026-09-30

### Existing ActionSource / bridge infrastructure (read before refactoring)
- `control/run_vla_episode.py` — defines the `ActionSource` Protocol (~line 143) and `ScriptedActionSource`; control loop that must remain unchanged per PIPE-01
- `control/vla_bridge/robot_client.py` — `BridgeActionSource` (~line 344), the existing SmolVLA gRPC-bridge implementation to refactor into the shared base class (D-04)
- `control/vla_bridge/action_contract.py` — `JOINT_ORDER`/units single source of truth all backends must respect
- `control/vla_bridge/safety_validator.py` — unmodified safety gate every backend must flow through (D-01, D-02)
- `control/vla_bridge/io_logger.py` — unmodified logging path every backend must flow through
- `control/vla_bridge/depth_camera.py` + `control/vla_bridge/fastfs_server.py` — existing depth pipeline (D-03); no changes needed for Phase 14 beyond confirming it still runs unaffected by backend choice
- `control/vla_bridge/policy_server.ipynb` (currently open in IDE) — Colab notebook hosting the PolicyServer side of the gRPC bridge; new Colab-hosted checkpoints (D-01) likely need parallel cells/config here

### Prior-phase grounding
- `.planning/phases/11-*/` — Phase 11 proved the `ActionSource` seam (`ScriptedActionSource` → `BridgeActionSource`) with zero control-loop changes; this phase repeats that pattern at scale
- `.planning/phases/12-*/FINDINGS.md`, `.planning/phases/12-*/12-CONTEXT.md` — depth pipeline background (D-03) and latency-fix infrastructure this phase builds on
- `.planning/phases/13-*/13-UAT.md` — Phase 13's pending live camera re-verification; Phase 14 depends on Phase 13's corrected camera pipeline (see Phase 13 gate decision below)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `BridgeActionSource` (`control/vla_bridge/robot_client.py:344`) — the entire gRPC/observation/safety/watchdog plumbing is backend-agnostic today except for the checkpoint string and SmolVLA-specific observation formatting; this is the base class D-04 refactors.
- `safety_validator.py`, `io_logger.py`, `action_contract.py` — already proven backend-agnostic through Phase 11/12; no changes needed, only reuse.
- `_load_device_map()` / `device_map.json` pattern (`run_vla_episode.py:197`) — existing pattern for small per-integration config files; could inform D-06's per-backend config shape.

### Established Patterns
- `ActionSource` Protocol (structural typing, not inheritance-required) — new backends only need to implement `get_action(joint_state, instruction) -> (action, model_version)`; this is what keeps the control loop unchanged (PIPE-01's core requirement).
- CLI arg parsing in `run_vla_episode.py` already branches on presence of `--checkpoint` to choose `BridgeActionSource` vs `ScriptedActionSource` (~line 647/665) — the natural place to extend with `--backend` (D-06).

### Integration Points
- New backends plug in at the `action_source = ...` assignment in `run_vla_episode.py`'s CLI branch (~line 640-670) — this is the one place PIPE-01 requires stay a simple swap, no control-loop changes.
- Cloud-API backends (Gemini, Gemini 2) are a genuinely new code path (no existing analog) — will need new HTTP client code, not a subclass of the gRPC bridge base.

</code_context>

<specifics>
## Specific Ideas

- User explicitly wants the project's full candidate VLA/imitation-learning model roster covered, organized by their own categorization: open-source foundation VLAs, industry-lab VLAs, small/efficient VLAs, and non-VLA imitation-learning baselines (for contrast). Phase 14 now covers all of these except the 3 MLLM-style backends (Phase 15) and any models not explicitly listed.
- OpenVLA-OFT was flagged as having prior project familiarity (used in v1 sim work, Phases 1/6) — this directly informed the D-05 integration-order recommendation (start there to prove the refactor).

</specifics>

<deferred>
## Deferred Ideas

None — the expanded backend list was folded directly into Phase 14's scope (D-00) rather than deferred, per explicit user decision. No items came up during discussion that were pushed to a future phase.

### Reviewed Todos (not folded)
None — no pending todos matched Phase 14 in this session.

</deferred>

---

*Phase: 14-VLA-Style Pipeline & Backends*
*Context gathered: 2026-09-30*
