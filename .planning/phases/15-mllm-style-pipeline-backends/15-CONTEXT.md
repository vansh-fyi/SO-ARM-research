# Phase 15: MLLM-Style Pipeline & Backends - Context

**Gathered:** 2026-10-02
**Status:** Ready for planning

<domain>
## Phase Boundary

A consistent MLLM-style `ActionSource` pattern (shared prompt construction, output JSON schema generated from `action_contract.py`, staged reasoning-trace capture) plus three raw-JSON-reasoning backends — a free/cheap HuggingFace-hosted reasoning model, Claude, and GPT — each driving the real SO-ARM101 through it with no pre-built movement primitives. Reuses Phase 14's registry/`--backend`/safety/logging path unchanged. Requirements: PIPE-02, MLLM-01..07.

Research flag (from ROADMAP): this phase needs deeper research at planning time — prompt/schema design and the undocumented root cause of the earlier failed ad-hoc MLLM experiment (`experiment-design/`, `docs/multimodal-context-ablation-experiment.md`).

</domain>

<decisions>
## Implementation Decisions

### Carried forward from Phase 14 / 12 (not re-discussed)
- **D-00a:** Backends are thin sibling modules registered via `@register("<name>")` in `control/vla_bridge/backends/` (auto-discovered); per-backend settings in `control/vla_bridge/backend_configs/<name>.json`; selected with `run_vla_episode.py --backend`. Shared base class + thin per-backend subclass (14 D-04), not independent classes.
- **D-00b:** Cloud backends use direct HTTPS from the control-loop machine (no Colab hop), following `base_gemini.py`'s secret discipline: key read only from an env var named in config, `SystemExit` at construction if missing, key never in `last_details`/`episode.jsonl`.
- **D-00c:** `get_action()` never raises; any failure degrades to `(dict(joint_state), "<label>@<reason>-holding-position")`. Every action passes the UNMODIFIED `safety_validator.validate_action()`.
- **D-00d:** Closed loop is chunk-drain (Phase 12): no per-tick MLLM calls.

### Action format & failure handling
- **D-01:** The model emits **absolute joint targets per tick** — a JSON array of N steps, each with all 6 `JOINT_ORDER` keys (degrees for the 5 arm joints, 0-100 percent for gripper). No waypoint interpolation, no delta mode (no hidden primitives; matches `action_contract.ACTION_MODE = "absolute"`).
- **D-02 (MLLM-05):** Chunk size is a **config value in `backend_configs/<name>.json`** derived from Phase 12's real measured latency/timing data, with a shared default and per-backend override allowed. The derivation must be documented, not guessed.
- **D-03 (MLLM-07):** On a malformed / non-numeric / missing-key value inside a chunk: steps before it execute normally; the bad step and the rest of the chunk become **hold-current-position** (tagged in `model_version`, never zero-filled), then the next chunk is requested. No retry call.
- **D-04 (MLLM-04):** Well-formed-but-out-of-range values are handled solely by the unmodified `safety_validator.py` (raw model behavior stays visible in logs). The JSON schema does NOT embed per-joint min/max.

### Prompt & observation inputs
- **D-05:** The MLLM sees the **maximum input**, unlike VLAs: wrist, overhead-left, overhead-right frames, **depth**, and URDF. (User rationale: MLLMs have no fixed input-slot limits, so withholding inputs only handicaps them.)
- **D-06 (reverses 14 D-03 for the MLLM pattern only):** Depth IS fed into the MLLM prompt. VLA backends keep depth log-only, unchanged. Depth is computed per chunk call (not per tick) via the existing Fast-FoundationStereo Colab service (`fastfs_server.py` / `depth_camera.py`); presentation (colorized heatmap image and/or scene-scale metadata) is Claude's discretion. NOTE: this partly supersedes `DEPTH-CAL-03`'s "never added to the observation dict" wording, which applied to the PolicyServer observation dict in Phase 12 — planner must scope the change to the MLLM prompt only and update REQUIREMENTS.md notes accordingly.
- **D-07:** Static prompt context is **full: joint names/units/calibrated limits (generated from `action_contract.py` + `joint_limits_deg`) plus URDF kinematics text** (link origins, joint axes, limits from `So-101/So-101.urdf`). This deliberately re-tests the richest tier of the earlier failed experiment, now under structured output. A config toggle (e.g. `include_urdf`) is permitted so Phase 16 can ablate, defaulting to on.
- **D-08:** Re-planning is **closed-loop per chunk**: after a chunk drains, call again with fresh frames + joint state.
- **D-09:** Cross-call memory: only the **previous chunk's `plan` string** is carried into the next prompt (no past images, no full history).

### Reasoning trace
- **D-10:** Use **provider-native thinking/reasoning plus the JSON output**. Still keep the staged fields (`task_restatement`, `plan`, `scene_claims`) in the output JSON schema ahead of `actions` so the 4-stage trace (restate task → plan → grounded scene claims → action) is captured uniformly even where native thinking text is summarized or hidden. Native thinking text, when the provider returns it, is logged as an additional field.
- **D-11:** Each scene claim is structured: `{claim, source: wrist|left|right|depth, confidence}`, logged verbatim so Phase 16 can audit claims against footage.
- **D-12:** The full trace is written **once per MLLM call** (per-chunk record via `io_logger`/`last_details`); each executed step carries a chunk-id pointer rather than a copy of the trace.

### Backend selection
- **D-13:** The HF reasoning model is accessed through the **HF Inference Providers API** (direct HTTPS, HF token from env var), same lane as Claude/GPT. Specific model chosen by the researcher from vision-capable reasoning models actually available free/cheap in October 2026.
- **D-14:** Claude and GPT each use the **top vision-capable flagship model** of their provider; model id lives in `backend_configs/<name>.json`; researcher verifies current ids and structured-output support.
- **D-15:** Structured output via each provider's **native JSON-schema mode** (Claude structured outputs, OpenAI `response_format` json_schema, HF provider JSON mode), with one schema object generated programmatically from `action_contract.py`. No tool-use/function-calling.

### Claude's Discretion
- Depth presentation in the prompt (heatmap image vs. metric text vs. both); image resolution/encoding for the 4 images.
- Module/class names for the shared MLLM base (`base_mllm.py` follows the `base_` auto-discovery skip convention).
- Exact `include_urdf` toggle mechanics and URDF text formatting.
- Fallback when a provider lacks JSON-schema mode (e.g. HF model): researcher decides, but must preserve the D-03 hold path.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope and requirements
- `.planning/ROADMAP.md` §Phase 15 — goal, success criteria, anti-features (no primitives, no multi-agent, no few-shot, no code-gen policy, no per-tick calls, no fine-tuning)
- `.planning/REQUIREMENTS.md` — PIPE-02, MLLM-01..07, DEPTH-CAL-03 (see D-06 note)
- `.planning/PROJECT.md` — v2.0 narrowing note on the failed general-MLLM experiment

### Prior phase decisions
- `.planning/phases/14-vla-style-pipeline-backends/14-CONTEXT.md` — registry, base-class, `--backend`, depth log-only (D-03, reversed here for MLLM)
- `.planning/phases/12-bridge-tick-latency-fix/12-CONTEXT.md` and `12-UAT.md` — chunk-drain loop and measured latency data feeding D-02
- `.planning/phases/13-camera-device-resolution-unification/13-CONTEXT.md` — camera resolution/provenance

### Code to mirror / reuse
- `control/vla_bridge/backends/__init__.py` — registry, `BackendHandles`, config loading
- `control/vla_bridge/backends/base_gemini.py` — direct-HTTPS base: secret discipline, never-raises hold contract, `model_version` tagging
- `control/vla_bridge/backends/base_http_adapter.py` — `capture_observation_images()`, `_is_joint_action`
- `control/vla_bridge/action_contract.py` — `JOINT_ORDER`, `ACTION_UNITS`, `load_joint_limits_deg` (schema source of truth)
- `control/vla_bridge/safety_validator.py`, `control/vla_bridge/io_logger.py` — unmodified validator; trace logging
- `control/vla_bridge/depth_camera.py`, `control/vla_bridge/fastfs_server.py` — depth source for D-06
- `control/vla_bridge/backend_configs/gemini_robotics.json` — per-backend config shape
- `So-101/So-101.urdf` — URDF facts for D-07

### Prior experiment (root-cause research needed)
- `experiment-design/` and `docs/multimodal-context-ablation-experiment.md` — the failed ad-hoc MLLM experiment

No other external specs — remaining requirements captured in decisions above.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `BackendHandles` + `@register` registry: new backends need no edits to `backends/__init__.py`.
- `base_gemini.GeminiActionSourceBase`: template for key handling, hold-on-failure, `last_details`, per-call vs. per-tick structure.
- `capture_observation_images()` / `StereoSplitCamera`: wrist + stereo-left/right capture already wired.
- `depth_camera.py` + Colab `fastfs_server.py`: existing depth computation.
- `action_contract.JOINT_ORDER/ACTION_UNITS` and `load_joint_limits_deg`: schema generation inputs.

### Established Patterns
- Heavy SDK imports live inside `build()`; discovery imports every backend module at startup.
- `model_version` tag carries path/hold reason; `last_details` never holds secrets.
- Per-backend JSON config, not inline constants; tests as `control/test_backend_<name>.py`.

### Integration Points
- `run_vla_episode.py --backend <name>`; chunk-drain loop from Phase 12; `io_logger.write_step` for traces and `latency_ms`.
- Depth wiring: currently log-only through `depth_recorder.py` — D-06 requires exposing it to the MLLM prompt builder.

</code_context>

<specifics>
## Specific Ideas

- User correction recorded: SmolVLA already receives all 3 views (wrist, overhead-left, overhead-right per `backend_configs/smolvla.json`); MLLM backends get those plus depth and URDF.
- Expectation: results with richest context may still fail like the earlier experiment — that is a valid finding for Phase 16, not a reason to trim inputs.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. (Context ablations such as URDF on/off are enabled by a config toggle but executing them belongs to Phase 16.)

</deferred>

---

*Phase: 15-MLLM-Style Pipeline & Backends*
*Context gathered: 2026-10-02*
