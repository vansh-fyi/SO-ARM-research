# Phase 15: MLLM-Style Pipeline & Backends - Context

**Gathered:** 2026-10-02 (updated same day: D-16..D-22)
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

### Step timing & chunk semantics
- **D-16:** The prompt states the step duration `dt` (derived from `--control-hz`, currently 2.0 Hz = 0.5 s/step) and the chunk length in seconds, and tells the model qualitatively that the arm cannot be commanded to jump arbitrarily and large jumps are clamped by a safety layer. Numeric per-step safety caps are NOT disclosed by default (they are loosened in v2.1 and re-tightened in Phase 17); planner may expose them behind a config toggle. Consistent with D-04: no per-joint min/max in the schema.
- **D-17:** Chunk length is a **fixed N steps** (shared default in config, per-backend override allowed per D-02); the prompt asks for exactly N steps. The model does not choose chunk length, keeping it from becoming a cross-backend confound for Phase 16.

### Episode termination
- **D-18:** Output schema includes a `task_complete` boolean. The episode ends on the first `true` or at the `--max-steps` budget, whichever comes first; operator e-stop remains. A `task_complete` chunk's remaining actions are still validated and executed per D-03/D-04 unless the planner finds a safer rule (planner's call; document it).
- **D-19:** The model's `task_complete` claim is logged in the per-chunk trace (D-12) separately from the human outcome label, which is Phase 16's job. Phase 15 only guarantees the claim is captured so false-completion can be analyzed later.

### Sampling & reproducibility
- **D-20:** Use the **same nominal sampling settings across all three backends**, defined once as a shared default in config (per-backend override allowed). When a provider rejects or fixes a setting (e.g. reasoning models locking temperature), the backend **omits it, still runs, and logs requested-vs-applied settings per call** — it does not fail at construction.
- **D-21:** Reasoning effort / thinking budget defaults to the **highest sensible level** for each provider (config field, so Phase 16 can ablate). Max output tokens is also pinned in config.

### Per-call record (beyond input/output)
- **D-22:** Each per-chunk record (D-12) additionally captures: (a) **token usage breakdown**, input / output / reasoning-thinking tokens from the provider usage object, recorded even when the thinking text is hidden or summarized; (b) **request params + provenance**, exact model id, response/request id, effective sampling and reasoning settings (requested vs applied), prompt template version + hash, schema hash, `include_urdf` flag, provider and SDK version; (c) **latency breakdown + cost**, wall-clock split into depth fetch, image encode, and API time, plus estimated cost from token counts and a per-model price in config; (d) **raw response payload**, verbatim and secret-redacted, stored as a sidecar file per call (not inline in `episode.jsonl`) along with the exact image bytes or hashes sent. Secrets discipline from D-00b applies to all of it.

### HF model lineup (extends D-13)
- **D-23:** The HF lane targets **three open models** as registered variants on the shared HF base: `Qwen/Qwen3.8-27B`, `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (MIT, reasoning with effort levels, Novita-hosted per its model card; marked experimental), and `moonshotai/Kimi-K3`. Each gets its own `backend_configs/<name>.json`. The researcher MUST verify against live provider docs, per model, vision input, thinking mode, JSON-schema/JSON mode, and free/cheap availability on HF Inference Providers (the Oct 2026 Hub listing only says "Endpoints compatible"). Any model failing vision or reasoning is dropped and replaced from the shortlist (`google/gemma-4-31B-it`, `zai-org/GLM-5.3-Flash`), with the reason noted. Phase 16 decides which variants actually run. This supersedes the single-HF-model wording in D-13 and the roadmap's "three backends" count (MLLM-01..07 requirements text should be checked by the planner).

### Post-research adjustments (2026-10-02, after 15-RESEARCH.md)
- **D-24 (refines D-16):** The prompt's step duration is the **measured 0.785 s** (Phase 12 live episodes, n=236 non-stall ticks), held as a config value with its evidence, not the nominal 0.5 s from `--control-hz`. Actual per-chunk tick period is also logged in the call record.
- **D-25 (refines D-18):** `task_complete=true` means the model judges the goal is achieved; the episode ends on the first `true` or when `--max-steps` is reached. Researcher's recommended semantics apply: "achieved in the current views", the rest of that chunk becomes hold-position (no motion after a done claim).
- **D-26 (refines D-05):** The MLLM receives the **raw** left and right overhead frames, the wrist frame, and the depth map (as in D-05). The prompt states the depth map is approximately, not pixel-exactly, aligned with the raw frames (depth is registered to a rectified/resized left image). The rectified `aligned_left` image is saved in the per-call sidecar, and a config toggle `left_image_source: raw|aligned` (default `raw`) lets Phase 16 ablate.
- **D-27 (keeps D-23):** DeepSeek stays `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (user chose to keep it over V4.1-Flash).
- Researcher findings C1-C12 in `15-RESEARCH.md` (provider pinning per HF variant, Claude/GPT fixed sampling, depth backend-owned plus a `run_vla_episode.py` parser rule and its test, "free" HF models not existing, reasoning-extraction wording lint) are accepted as planner adaptations; they do not change user-locked choices.

- **D-28 (resolves D-06's discretion on depth presentation):** Depth reaches the MLLM as **both** (a) a colourised heatmap image with the colour scale stated in the prompt, and (b) a coarse numeric grid as text (~24x32, median-pooled, metres, `n/a` for invalid cells, ~2-3K tokens) plus min/median/max summary stats. The full 576x800 array is NOT sent (~1.4M tokens/call, not attachable via the APIs). Config field `depth_presentation: heatmap|grid|both` (default `both`) lets Phase 16 ablate. Grid and heatmap are derived from the same depth result; both are saved in the per-call sidecar.

### Claude's Discretion
- Depth presentation in the prompt (heatmap image vs. metric text vs. both); image resolution/encoding for the 4 images.
- Module/class names for the shared MLLM base (`base_mllm.py` follows the `base_` auto-discovery skip convention).
- Exact `include_urdf` toggle mechanics and URDF text formatting.
- Whether to expose numeric per-step safety caps behind a config toggle (D-16), and the rule for the tail of a `task_complete` chunk (D-18).
- Sidecar file layout/naming for raw payloads (D-22d).
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
