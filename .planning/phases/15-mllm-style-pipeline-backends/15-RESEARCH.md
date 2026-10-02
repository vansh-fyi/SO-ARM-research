# Phase 15: MLLM-Style Pipeline & Backends - Research

**Researched:** 2026-10-02
**Domain:** Raw-JSON, structured-output MLLM controllers (Claude, GPT, HF-hosted open reasoning models) driving the real SO-ARM101 through the existing `ActionSource` seam, with chunk-drain, staged trace capture, and per-call provenance logging
**Confidence:** MEDIUM-HIGH. Claude/OpenAI request shapes, model ids, effort levels, sampling restrictions and usage fields are HIGH (live vendor docs fetched today plus installed SDK source inspected). HF Inference Providers capability metadata is HIGH for what the router publishes (live API) but MEDIUM for actual behaviour (no HF token on this machine, so no live call was possible: thinking + JSON-schema interplay and usage/reasoning-token fields on HF providers are unverified). The earlier failed experiment's root cause is NOT recoverable from the repo (no outputs were ever recorded); only confounds are documented (evidence vs speculation separated below).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Carried forward from Phase 14 / 12 (not re-discussed)
- **D-00a:** Backends are thin sibling modules registered via `@register("<name>")` in `control/vla_bridge/backends/` (auto-discovered); per-backend settings in `control/vla_bridge/backend_configs/<name>.json`; selected with `run_vla_episode.py --backend`. Shared base class + thin per-backend subclass (14 D-04), not independent classes.
- **D-00b:** Cloud backends use direct HTTPS from the control-loop machine (no Colab hop), following `base_gemini.py`'s secret discipline: key read only from an env var named in config, `SystemExit` at construction if missing, key never in `last_details`/`episode.jsonl`.
- **D-00c:** `get_action()` never raises; any failure degrades to `(dict(joint_state), "<label>@<reason>-holding-position")`. Every action passes the UNMODIFIED `safety_validator.validate_action()`.
- **D-00d:** Closed loop is chunk-drain (Phase 12): no per-tick MLLM calls.

#### Action format & failure handling
- **D-01:** The model emits **absolute joint targets per tick** — a JSON array of N steps, each with all 6 `JOINT_ORDER` keys (degrees for the 5 arm joints, 0-100 percent for gripper). No waypoint interpolation, no delta mode (no hidden primitives; matches `action_contract.ACTION_MODE = "absolute"`).
- **D-02 (MLLM-05):** Chunk size is a **config value in `backend_configs/<name>.json`** derived from Phase 12's real measured latency/timing data, with a shared default and per-backend override allowed. The derivation must be documented, not guessed.
- **D-03 (MLLM-07):** On a malformed / non-numeric / missing-key value inside a chunk: steps before it execute normally; the bad step and the rest of the chunk become **hold-current-position** (tagged in `model_version`, never zero-filled), then the next chunk is requested. No retry call.
- **D-04 (MLLM-04):** Well-formed-but-out-of-range values are handled solely by the unmodified `safety_validator.py` (raw model behavior stays visible in logs). The JSON schema does NOT embed per-joint min/max.

#### Prompt & observation inputs
- **D-05:** The MLLM sees the **maximum input**, unlike VLAs: wrist, overhead-left, overhead-right frames, **depth**, and URDF. (User rationale: MLLMs have no fixed input-slot limits, so withholding inputs only handicaps them.)
- **D-06 (reverses 14 D-03 for the MLLM pattern only):** Depth IS fed into the MLLM prompt. VLA backends keep depth log-only, unchanged. Depth is computed per chunk call (not per tick) via the existing Fast-FoundationStereo Colab service (`fastfs_server.py` / `depth_camera.py`); presentation (colorized heatmap image and/or scene-scale metadata) is Claude's discretion. NOTE: this partly supersedes `DEPTH-CAL-03`'s "never added to the observation dict" wording, which applied to the PolicyServer observation dict in Phase 12 — planner must scope the change to the MLLM prompt only and update REQUIREMENTS.md notes accordingly.
- **D-07:** Static prompt context is **full: joint names/units/calibrated limits (generated from `action_contract.py` + `joint_limits_deg`) plus URDF kinematics text** (link origins, joint axes, limits from `So-101/So-101.urdf`). This deliberately re-tests the richest tier of the earlier failed experiment, now under structured output. A config toggle (e.g. `include_urdf`) is permitted so Phase 16 can ablate, defaulting to on.
- **D-08:** Re-planning is **closed-loop per chunk**: after a chunk drains, call again with fresh frames + joint state.
- **D-09:** Cross-call memory: only the **previous chunk's `plan` string** is carried into the next prompt (no past images, no full history).

#### Reasoning trace
- **D-10:** Use **provider-native thinking/reasoning plus the JSON output**. Still keep the staged fields (`task_restatement`, `plan`, `scene_claims`) in the output JSON schema ahead of `actions` so the 4-stage trace (restate task → plan → grounded scene claims → action) is captured uniformly even where native thinking text is summarized or hidden. Native thinking text, when the provider returns it, is logged as an additional field.
- **D-11:** Each scene claim is structured: `{claim, source: wrist|left|right|depth, confidence}`, logged verbatim so Phase 16 can audit claims against footage.
- **D-12:** The full trace is written **once per MLLM call** (per-chunk record via `io_logger`/`last_details`); each executed step carries a chunk-id pointer rather than a copy of the trace.

#### Backend selection
- **D-13:** The HF reasoning model is accessed through the **HF Inference Providers API** (direct HTTPS, HF token from env var), same lane as Claude/GPT. Specific model chosen by the researcher from vision-capable reasoning models actually available free/cheap in October 2026.
- **D-14:** Claude and GPT each use the **top vision-capable flagship model** of their provider; model id lives in `backend_configs/<name>.json`; researcher verifies current ids and structured-output support.
- **D-15:** Structured output via each provider's **native JSON-schema mode** (Claude structured outputs, OpenAI `response_format` json_schema, HF provider JSON mode), with one schema object generated programmatically from `action_contract.py`. No tool-use/function-calling.

#### Step timing & chunk semantics
- **D-16:** The prompt states the step duration `dt` (derived from `--control-hz`, currently 2.0 Hz = 0.5 s/step) and the chunk length in seconds, and tells the model qualitatively that the arm cannot be commanded to jump arbitrarily and large jumps are clamped by a safety layer. Numeric per-step safety caps are NOT disclosed by default (they are loosened in v2.1 and re-tightened in Phase 17); planner may expose them behind a config toggle. Consistent with D-04: no per-joint min/max in the schema.
- **D-17:** Chunk length is a **fixed N steps** (shared default in config, per-backend override allowed per D-02); the prompt asks for exactly N steps. The model does not choose chunk length, keeping it from becoming a cross-backend confound for Phase 16.

#### Episode termination
- **D-18:** Output schema includes a `task_complete` boolean. The episode ends on the first `true` or at the `--max-steps` budget, whichever comes first; operator e-stop remains. A `task_complete` chunk's remaining actions are still validated and executed per D-03/D-04 unless the planner finds a safer rule (planner's call; document it).
- **D-19:** The model's `task_complete` claim is logged in the per-chunk trace (D-12) separately from the human outcome label, which is Phase 16's job. Phase 15 only guarantees the claim is captured so false-completion can be analyzed later.

#### Sampling & reproducibility
- **D-20:** Use the **same nominal sampling settings across all three backends**, defined once as a shared default in config (per-backend override allowed). When a provider rejects or fixes a setting (e.g. reasoning models locking temperature), the backend **omits it, still runs, and logs requested-vs-applied settings per call** — it does not fail at construction.
- **D-21:** Reasoning effort / thinking budget defaults to the **highest sensible level** for each provider (config field, so Phase 16 can ablate). Max output tokens is also pinned in config.

#### Per-call record (beyond input/output)
- **D-22:** Each per-chunk record (D-12) additionally captures: (a) **token usage breakdown**, input / output / reasoning-thinking tokens from the provider usage object, recorded even when the thinking text is hidden or summarized; (b) **request params + provenance**, exact model id, response/request id, effective sampling and reasoning settings (requested vs applied), prompt template version + hash, schema hash, `include_urdf` flag, provider and SDK version; (c) **latency breakdown + cost**, wall-clock split into depth fetch, image encode, and API time, plus estimated cost from token counts and a per-model price in config; (d) **raw response payload**, verbatim and secret-redacted, stored as a sidecar file per call (not inline in `episode.jsonl`) along with the exact image bytes or hashes sent. Secrets discipline from D-00b applies to all of it.

#### HF model lineup (extends D-13)
- **D-23:** The HF lane targets **three open models** as registered variants on the shared HF base: `Qwen/Qwen3.8-27B`, `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (MIT, reasoning with effort levels, Novita-hosted per its model card; marked experimental), and `moonshotai/Kimi-K3`. Each gets its own `backend_configs/<name>.json`. The researcher MUST verify against live provider docs, per model, vision input, thinking mode, JSON-schema/JSON mode, and free/cheap availability on HF Inference Providers (the Oct 2026 Hub listing only says "Endpoints compatible"). Any model failing vision or reasoning is dropped and replaced from the shortlist (`google/gemma-4-31B-it`, `zai-org/GLM-5.3-Flash`), with the reason noted. Phase 16 decides which variants actually run. This supersedes the single-HF-model wording in D-13 and the roadmap's "three backends" count (MLLM-01..07 requirements text should be checked by the planner).

### Claude's Discretion
- Depth presentation in the prompt (heatmap image vs. metric text vs. both); image resolution/encoding for the 4 images.
- Module/class names for the shared MLLM base (`base_mllm.py` follows the `base_` auto-discovery skip convention).
- Exact `include_urdf` toggle mechanics and URDF text formatting.
- Whether to expose numeric per-step safety caps behind a config toggle (D-16), and the rule for the tail of a `task_complete` chunk (D-18).
- Sidecar file layout/naming for raw payloads (D-22d).
- Fallback when a provider lacks JSON-schema mode (e.g. HF model): researcher decides, but must preserve the D-03 hold path.

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope. (Context ablations such as URDF on/off are enabled by a config toggle but executing them belongs to Phase 16.)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PIPE-02 | Consistent MLLM-style `ActionSource` pattern (shared prompt construction, output JSON schema, reasoning-trace capture) that all MLLM backends plug into without changing the shared control loop/safety/logging path | Architecture Pattern 1 (chunk-queue base class `MllmActionSourceBase` + 3 thin provider adapters), Pattern 2/3 (provider-agnostic schema + prompt modules). Two minimal, additive `run_vla_episode.py` edits are unavoidable (D-18 termination hook, depth CLI rule) and are called out explicitly. |
| MLLM-01 | A free/cheap HF-hosted reasoning model is integrated as the first MLLM-style backend | HF lane verified live: all three D-23 models are vision + reasoning + live on providers; none is free; Qwen3.8-27B and DeepSeek-V4-Flash-Vision-Exp are cheap, Kimi-K3 is not (see Conflicts C1/C2 and HF section). |
| MLLM-02 | Claude (paid tier) integrated | `claude-fable-5-1`, `output_config.format` JSON schema + adaptive thinking + `effort`, sampling params fixed (Provider Facts: Claude). |
| MLLM-03 | GPT integrated | `gpt-6-astra`, Responses API `text.format` json_schema strict + `reasoning.effort`, sampling params removed (Provider Facts: OpenAI). |
| MLLM-04 | Action-chunk schema generated programmatically from `action_contract.py` constants and validated through the unmodified `safety_validator.py` | Code Example 1 (schema generator), schema intersection rules for Claude/OpenAI/HF (no min/max, all required, `additionalProperties:false`), Validation map. |
| MLLM-05 | Chunk size derived from Phase 12's real measured latency/timing data, not guessed | "Chunk-Size Derivation" section: measured tick period 0.785 s, depth RTT p50 6.15 s, SmolVLA refetch stalls 3.3-5.0 s; recommended `chunk_len = 20`, with the honest gap that Phase 12 has no MLLM API latency. |
| MLLM-06 | Staged reasoning trace captured alongside the action chunk, per step | Pattern 6 (per-call record in `last_details["mllm_call"]` on the first step of a chunk, `chunk_ref` pointer on every step; sidecar for raw payloads). |
| MLLM-07 | Malformed/parse-failing action within a chunk holds current position rather than zero-filling or crashing | Pattern 5 (pre-validation at parse time, `HOLD` sentinels, literal D-03 reading) plus failure-policy for whole-call failures (prevents per-tick paid retries). |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- Python 3.x, `snake_case` files/functions, `UPPER_SNAKE_CASE` module constants, 4-space indent, no linter/formatter config; module-level triple-quoted docstrings documenting contract and CLI usage; `pathlib.Path` preferred in newer code; explicit `FileNotFoundError`/`SystemExit` with descriptive messages; progress via `print()`, saved paths as `Saved → path`. [CITED: .claude/CLAUDE.md]
- GSD workflow enforcement: file edits happen only through a GSD command (`/gsd-execute-phase` etc.), never ad hoc. [CITED: .claude/CLAUDE.md]
- Heavy SDK imports live INSIDE `build()` / lazy factory functions (discovery imports every backend module at startup and an import failure is deliberately not swallowed). [CITED: control/vla_bridge/backends/__init__.py]
- Project memory: never guess when stuck, read live package source and web-search official docs before writing new integration code. Applied here: SDK sources inspected in a scratch venv, vendor docs fetched as raw markdown. [CITED: MEMORY.md "Verify against live source"]
- Per-backend JSON config (not inline constants); tests as `control/test_backend_<name>.py`; secrets only via env-var names in config, never in `last_details`/`episode.jsonl`. [CITED: 15-CONTEXT.md, base_gemini.py]

## Summary

Phase 15 is an integration phase on top of a mature, well-tested seam. The reusable pieces (registry, `BackendHandles`, `capture_observation_images`, `_is_joint_action`, `safety_validator`, `IOLogger`, `DepthCameraClient`, `base_gemini.py`'s secret and hold discipline) already exist and the full `control/` suite is green (1003 tests, 9.4 s). The work is: one provider-agnostic chunk-queue `ActionSource` base, one pure-Python schema+prompt layer generated from `action_contract.py`, three small provider adapters (Anthropic Messages, OpenAI Responses, OpenAI-compatible chat completions for HF), five thin registered backends (Claude, GPT, three HF variants), and two small additive edits to `run_vla_episode.py`.

Live facts that change the design. (1) Anthropic's flagship is `claude-fable-5-1` (Mythos 5.1 is invite-only). It has always-on adaptive thinking, `effort` low..max, REJECTS any non-default `temperature/top_p/top_k` with HTTP 400, hides thinking text unless `display:"summarized"`, and issues billed `reasoning_extraction` refusals if the prompt or a JSON field asks the model to reproduce its reasoning. (2) OpenAI's flagship is `gpt-6-astra` (Responses API; `none` effort unsupported; `temperature`/`top_p` must be removed whenever effort is not `none`, so always). (3) On HF Inference Providers all three D-23 models have vision and a reasoning mode, but NONE is free, Kimi-K3 costs $2.7-3 in / $13.5-15 out per M tokens, and JSON-schema support is per-provider: the default `:fastest` routing would land DeepSeek on Novita, which has no structured output. Providers must be pinned. (4) Phase 12 supplies real step cadence (0.785 s per tick, not the nominal 0.5 s), depth round-trip (p50 6.15 s) and refetch-stall data, but no MLLM API latency; chunk size must therefore be derived from a duty-cycle/horizon argument and re-checked on the first live call.

The earlier failed experiment cannot be root-caused from the repo: its `RUN_LOG.md` is an empty template and no model output was ever saved (the experiment directory is also git-ignored). What the surviving artifacts DO show is a stack of design confounds that structured output plus generated context fixes by construction: coding-agent harness asked to write its own servo driver, four unit systems in one prompt (ticks, radians, normalized, degrees), an inverted gripper convention versus hardware, a stale pre-Phase-10 URDF with missing/misparented gripper and wrong wrist limits, hand-filled `<LIVE_VALUE>` joint state, depth that was only a raw stereo pair, and no timing semantics.

**Primary recommendation:** Build `MllmActionSourceBase` (chunk deque, pre-validated steps with HOLD sentinels, failure-as-N-holds, per-call record, `task_complete` flag) over three provider adapters; pin `anthropic==1.11.0` and `openai==3.23.0` (one `openai` SDK serves both GPT and the HF router); default `chunk_len=20`; use `xhigh` effort for Claude/GPT and each HF model's own top level; pin one structured-output-capable HF provider per variant; keep depth backend-owned and synchronous per chunk call; never use the words "reasoning/thinking/trace/scratchpad" in prompt or schema field names.

## Findings that CONFLICT with or sharpen CONTEXT.md (planner must adapt)

| # | CONTEXT item | Live fact (source) | Severity | Adaptation |
|---|--------------|--------------------|----------|------------|
| C1 | D-13/D-23/MLLM-01 "free/cheap" HF models | No model on the HF router is free (`is_free:false` for all 134 listed models; 0 free entries). Free HF accounts get $0.10/month of credits, PRO $2/month. Kimi-K3 is $2.70-3.00 in / $13.50-15.00 out per M tokens, Opus-5.5-class pricing, not cheap. Qwen3.8-27B ($0.20-0.99 / $1.49-3.00) and DeepSeek-V4-Flash-Vision-Exp ($0.44 / $1.32) are cheap. [VERIFIED: router.huggingface.co/v1/models, 2026-10-02; CITED: huggingface.co/docs/inference-providers/pricing] | Medium | Keep all three (user-locked, vision+reasoning verified). Fund via purchased HF credits. Put a dated price table in each config (D-22c). Suggest Phase 16 runs Qwen/DeepSeek first. |
| C2 | D-15 "HF provider JSON mode"; discretion "fallback when a provider lacks JSON-schema mode" | `supports_structured_output` differs per provider. Default routing is `:fastest` (highest throughput), which for DeepSeek-V4-Flash-Vision-Exp is Novita (127 tok/s, structured output FALSE) and also happens to be the provider the Hub card/D-23 names. DeepInfra serves all five candidate models with structured output TRUE. [VERIFIED: router API; CITED: HF providers index "default :fastest"] | High | Every HF config MUST pin `model:provider` (e.g. `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp:deepinfra`). Fallback if a pinned provider rejects `json_schema`: `response_format:{"type":"json_object"}` + schema text in the prompt + client-side validation, logged as `structured_mode_applied`. Preserve the D-03 hold path in both modes. |
| C3 | D-20 "same nominal sampling across all three" | Claude Fable 5.1: any non-default `temperature/top_p/top_k` returns HTTP 400 on every request; the Python SDK `messages.create` has no such parameters at all. GPT-6 Astra: docs say remove `temperature/top_p/top_logprobs` whenever reasoning effort is not `none`, and Astra does not support `none`. So nominal sampling can only be APPLIED on the HF lane. [CITED: platform.claude.com .../thinking "Sampling parameters"; developers.openai.com .../latest-model; VERIFIED: anthropic 1.11.0 signature] | Medium | D-20's mechanics (omit + log) are exactly right. Do not send the keys for Claude/GPT at all and log `applied: "provider-fixed"`. Shared default `temperature=1.0, top_p=0.95` (Qwen3.8, Kimi-K3, DeepSeek model cards all publish these for thinking mode). |
| C4 | D-10/D-11 staged fields `task_restatement/plan/scene_claims` | Claude returns billed `stop_reason:"refusal"` with `stop_details.category:"reasoning_extraction"` when a prompt or a JSON field asks the model to reproduce its reasoning (examples: fields named `reasoning`, `thinking`, `trace`, a scratchpad section). [CITED: platform.claude.com .../refusals-and-fallback#keep-reasoning-in-thinking-blocks, structured-outputs "Invalid outputs"] | High | Keep the three fields but phrase them as task deliverables ("restate the task in one sentence", "plan for the next N steps", "claims about the scene with the image they come from"). Lint prompt+schema for the words reasoning/thinking/trace/scratchpad/chain-of-thought. Handle `refusal` as a failed call (hold) and log `stop_details`. |
| C5 | D-06 / DEPTH-CAL-03 | `run_vla_episode.py` currently errors if `--depth-endpoint` is given without `--server-address` (test `test_main_errors_when_depth_flags_given_without_server_address` pins it), which blocks every direct-HTTPS MLLM backend. The FastFS server also returns HTTP 429 `depth service busy` if two requests overlap (non-blocking lock), so the episode's background `DepthRecorder` and the backend's own per-chunk depth call can collide. [VERIFIED: control/run_vla_episode.py, fastfs_server.py] | High | Scope the parser rule to non-MLLM backends (or to backends that do not return a `stereo_camera`) and update that one test. Make depth backend-owned for MLLM runs (one synchronous call per chunk, persisted in the call sidecar) and either do not also enable the recorder or retry 429 with backoff. Update REQUIREMENTS.md DEPTH-CAL-03 note as D-06 asks. |
| C6 | D-16 "dt derived from --control-hz = 0.5 s/step" | Measured step-to-step wall clock in Phase 12 live episodes is median 0.785 s (n=236 non-stall ticks over `vla_episode_004` and `vla_episode_20260930_121123`); `action_to_execution` alone is 640-690 ms (50 interpolated waypoints at 100 Hz plus driver latency). Nominal 2.0 Hz is therefore ~1.27 Hz in practice. [VERIFIED: control/outputs/*/episode.jsonl] | Medium | Follow D-16 for the prompt text (derive from `--control-hz`) but also log measured tick period per chunk in the call record; consider a config toggle `prompt_dt_source: control_hz|measured` and flag to the user that the prompt dt understates real per-step time by ~36%. |
| C7 | D-02/MLLM-05 "derive from Phase 12 measured latency" | Phase 12 measured VLA bridge/depth latency only. It contains NO MLLM API latency. | Medium | See "Chunk-Size Derivation": derive from step cadence + horizon + duty-cycle, ship `chunk_len=20`, and record real per-call latency so Phase 16 can refine. Make the derivation a documented table in the config `notes`. |
| C8 | D-07 URDF kinematics in the prompt | The sign of each real joint against its URDF axis has never been verified on hardware (Phase 10 checked limits and gripper direction only; recorded as an open human check in 14-03-SUMMARY.md). [CITED: control/vla_bridge/backends/eef_to_joint.py module docstring] | Medium | State in the prompt only what is verified (degree 0 = calibrated midpoint = URDF 0; limits; gripper 0=closed/100=open). Add a human sign-check item to the live UAT; a wrong sign would turn "full URDF" into misleading context and must be separable in Phase 16 analysis. |
| C9 | Requirements/roadmap wording | MLLM-01 says "a free/cheap HF model" (singular), roadmap success criteria say "3 backends"/"all 3", EVAL-01 counts 16 backends with one HF entry. D-23 yields 5 registered MLLM backends (3 HF variants + Claude + GPT). | Low | Planner updates REQUIREMENTS.md/ROADMAP.md wording (Phase 16 decides which variants run). |
| C10 | D-14 "top flagship" | Claude: `claude-fable-5-1` ($10/$50, 1M ctx, 128K out). Anthropic's own guidance says start with Opus 5.5 for most work; `claude-mythos-5-1` is Project-Glasswing-only. Fable 5.1 is 30-day-retention, not ZDR. GPT: `gpt-6-astra` ($10/$50, cached $1). [CITED: platform.claude.com .../models/overview, fable-5-1/whats-new; developers.openai.com .../models/gpt-6-astra] | Low | Use `claude-fable-5-1` and `gpt-6-astra` as D-14 states. Note to user: lab-bench images leave the building under 30-day (Anthropic) retention; set OpenAI `store:false`. |
| C11 | D-21 "highest sensible level" | `max` exists on both (Claude: low/medium/high/xhigh/max; GPT-6 Astra: low/medium/high/xhigh/max). Docs describe `max` as "no constraints on token spending" and warn that on structured-output tasks it can overthink; OpenAI says use `xhigh` only when evals justify latency. | Low | Default `xhigh` for Claude and GPT (config field; `max` is a Phase 16 ablation). HF: Qwen3.8 `xhigh`, Kimi-K3 `max` (its default), DeepSeek-V4-Flash-Vision-Exp `max` (its card's benchmark setting). |
| C12 | D-23 "DeepSeek ... Novita-hosted per its model card" | The card text itself does not name a host. Live router: Novita AND DeepInfra serve it. A newer sibling `deepseek-ai/DeepSeek-V4.1-Flash` (multimodal, MIT, reasoning effort 1-100, modified 2026-10-01, on Novita/Fireworks/DeepInfra) is on the router. | Info | Keep the Exp model (locked). Mention V4.1-Flash to the user as an optional swap; not needed to satisfy D-23 since Exp passes vision+reasoning. |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Prompt construction, schema generation, chunk parsing/validation, hold logic | Control-loop machine (local Python, `control/vla_bridge/`) | — | Must be deterministic, testable without keys, and identical across providers (success criterion 1). |
| Reasoning/inference | Cloud provider API (Anthropic, OpenAI, HF router -> pinned provider) | — | Direct HTTPS from the control machine (D-00b); no Colab hop. |
| Depth computation | Colab FastFS Flask service (GPU) | Control machine (rectify + downsize + heatmap render) | Existing Phase 12 split; control machine owns rectification and presentation, Colab owns stereo inference. |
| Action safety gating | Control machine, unmodified `safety_validator.validate_action()` in `run_episode()` | `base_mllm` pre-validation (shape/finite only) | Raw model behaviour must stay visible (D-04); only malformed shapes are filtered before the validator. |
| Chunk drain / timing | Control machine, inside the action source (internal deque) | `run_episode()` tick loop | `run_episode` calls `get_action()` once per tick and knows nothing about chunks; the source pops one step per tick. |
| Episode termination (`task_complete`, max-steps, e-stop) | `run_episode()` (additive hook) | Action source exposes a flag | The loop owns termination reasons and the return-to-start e-stop path. |
| Trace + provenance logging | `IOLogger.write_step` via `last_details` (per-chunk record) | Sidecar files under the episode dir | Matches existing `depth.jsonl`/sidecar precedent; keeps `episode.jsonl` small. |
| Secrets | Process environment (env-var names in config) | — | Never in config files, `last_details`, sidecars or logs. |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `anthropic` | 1.11.0 (released 2026-09-30) | Claude Messages API: `messages.stream(...)` with `output_config={"effort","format"}`, `thinking={"type":"adaptive","display":"summarized"}`; `usage.output_tokens_details.thinking_tokens`, `stop_details`, `_request_id` | Official SDK. Signature verified: no `temperature/top_p/top_k` params exist; `OutputConfigParam.effort` Literal low/medium/high/xhigh/max; `JSONOutputFormatParam` type `json_schema`. [VERIFIED: installed anthropic 1.11.0 source in scratch venv] |
| `openai` | 3.23.0 (released 2026-10-01) | GPT: `responses.stream(...)`/`responses.create(...)` with `reasoning={"effort","summary"}`, `text={"format":{"type":"json_schema","strict":True,...}}`, `max_output_tokens`, `store=False`; HF lane: `chat.completions.with_raw_response.create(...)` against `https://router.huggingface.co/v1` | Official SDK; HF documents the OpenAI client for the router. One dependency serves 4 of the 5 backends. `usage.output_tokens_details.reasoning_tokens` verified in `ResponseUsage`. [VERIFIED: installed openai 3.23.0 source; CITED: huggingface.co/docs/inference-providers/guides/structured-output] |
| stdlib | 3.12 (`control/.venv`) | `json`, `hashlib`, `xml.etree.ElementTree` (URDF), `collections.deque`, `dataclasses`, `pathlib`, `time.monotonic` | Zero new deps for schema, hashing, URDF text, queueing. |
| existing `opencv-python` 5.0.0.93, `numpy` 2.2.6, `requests` 2.34.2 | pinned in `control/requirements.txt` | PNG encode, resize (`_downsize_for_transport`), `cv2.applyColorMap(..., COLORMAP_TURBO)` depth heatmap | Already used by `base_http_adapter`/`depth_camera`; avoids adding matplotlib to the runtime path. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pytest` | 8.3.4 (pinned) | Unit + integration tests with fake provider clients | All validation (no live keys needed). |
| `huggingface_hub` | 1.29.0 installed (transitive via lerobot) | NOT used for the HF lane | Its typed `ChatCompletionOutputUsage` keeps only prompt/completion/total tokens and its `chat_completion` lacks a `reasoning_effort` arg, so it would drop D-22(a) reasoning-token fields. Use the raw OpenAI-compatible wire instead. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `openai` SDK for HF | raw `requests` POST | `requests` is already pinned and gives a byte-exact wire body, but you hand-write streaming/timeout/error mapping; `openai` `with_raw_response` gives the same raw body plus typed errors. Either is fine; recommended: `openai`. |
| pydantic models for the schema | hand-built dict from `JOINT_ORDER` | pydantic 2.13.5 is installed but Anthropic's helper transforms schemas and strips constraints; a plain dict generated by a loop is smaller, hashable, and is the single schema object D-15 asks for. |
| Claude non-streaming | `messages.stream(...).get_final_message()` | Non-streaming with a large `max_tokens` is rejected client-side by the SDK ("expected to exceed 10 minutes"); streaming is the documented answer for long requests. |
| `client.messages.parse(output_format=PydanticModel)` | raw `output_config.format` dict | Parse helper adds pydantic coupling and schema rewriting; raw dict keeps one schema across providers. |

**Installation** (add to `control/requirements.txt` with provenance comments in the Plan 14-13 style, then `control/.venv/bin/pip install -r requirements.txt`):
```bash
anthropic==1.11.0
openai==3.23.0
# pulled in transitively (dry-run against control/.venv shows no conflicts):
# docstring_parser 0.18.0, httpx2 2.13.1, httpcore2 2.13.1, jiter 0.17.0, truststore 0.10.4
```
**Version verification:** `pip index versions` (2026-10-02): anthropic 1.11.0 latest (1.10.0 2026-09-30, 1.9.0 2026-09-28); openai 3.23.0 latest (3.22.1 2026-09-30); `control/.venv/bin/pip install --dry-run anthropic==1.11.0 openai==3.23.0` resolves cleanly alongside `lerobot 0.6.1`, `google-genai 2.25.0`, `httpx 0.28.1`, `pydantic 2.13.5`. [VERIFIED: PyPI JSON, pip dry-run]

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| anthropic | PyPI | multi-year; 1.11.0 released 2026-09-30 | unknown (seam has no download data) | github.com/anthropics/anthropic-sdk-python | SUS (heuristic: `too-new` latest release, `unknown-downloads`) | Approved with checkpoint: official vendor SDK; planner adds one `checkpoint:human-verify` before first install of the dependency set |
| openai | PyPI | multi-year; 3.23.0 released 2026-10-01 | unknown | github.com/openai/openai-python | SUS (same heuristic) | Approved with the same single checkpoint |
| httpx2, httpcore2 | PyPI | 2.13.1 released 2026-09-23 | unknown | github.com/pydantic/httpx2 | SUS (`too-new`, `unknown-downloads`) | Transitive dep of both SDKs (pydantic-org HTTP client); covered by the same checkpoint |
| jiter | PyPI | 0.17.0 released 2026-09-12 | unknown | github.com/pydantic/jiter | SUS (heuristic) | Transitive; same checkpoint |
| docstring_parser | PyPI | 0.18.0 released 2026-04-14 | unknown | none listed in PyPI metadata | SUS (`no-repository`) | Transitive of anthropic; same checkpoint (long-established project; verify on install) |
| truststore | PyPI | 0.10.4 | unknown | github.com/sethmlarson/truststore | SUS (`unknown-downloads`) | Transitive of httpx2; same checkpoint |

**Packages removed due to [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** all rows above are heuristic flags only (the seam cannot see download counts and treats a fresh latest-release date as "too new"); every one has a real upstream repo. Planner: insert ONE `checkpoint:human-verify` task before the first `pip install` that confirms publisher/repo for `anthropic`, `openai`, `httpx2`, and pins exact versions. No `postinstall` scripts apply (PyPI). Package names came from the vendors' official docs (Anthropic docs, OpenAI docs, HF structured-output guide), not from search guesswork. [VERIFIED: gsd-tools package-legitimacy check; CITED: vendor docs]

## Architecture Patterns

### System Architecture Diagram

```
 operator: run_vla_episode.py --backend mllm-claude|mllm-gpt|mllm-hf-qwen|mllm-hf-deepseek|mllm-hf-kimi --instruction "..." [--depth-endpoint --depth-calibration]
      |
      v
 BACKEND_REGISTRY[name](args, camera_names, joint_limits_deg)  --> build_mllm_backend():
      1. load shared defaults + per-backend JSON (merge, validate)      2. read API key from env var NAME (SystemExit if unset)
      3. build provider adapter (lazy SDK import)                       4. connect SO101Follower + StereoSplitCamera + wrist cap
      5. optional DepthCameraClient from args.depth_endpoint/calibration -> returns BackendHandles(action_source=MllmActionSourceBase)
      |
      v
 run_episode() tick loop  (one iteration per tick, 0.785 s real)
   read_positions -> action_source.get_action(joint_state, instruction)
      |
      |  queue non-empty? ----------- yes --> pop next pre-validated step
      |  (D-00d: no per-tick call)                 |-- HOLD sentinel -> (dict(joint_state), "<label>@<reason>-holding-position")
      |                                            '-- action dict   -> (action, "<label>@mllm-chunk")
      |  no --> refill (BLOCKING, robot holds last target):
      |          a. capture: wrist.read (flush) + stereo read_left/read_right back-to-back (ONE physical read)
      |          b. depth: DepthCameraClient.compute_depth_result(left_raw,right_raw)  [~6 s]  -> heatmap + coarse grid text
      |          c. build prompt: static system (limits/units from action_contract + joint_limits_deg, URDF text, dt, N, anti-primitive wording)
      |                           + dynamic user (joint_state, prev plan, instruction, 4 images)
      |          d. provider adapter .call(): Claude stream | GPT responses.stream | HF chat.completions (raw)
      |          e. parse JSON -> pre-validate each step (shape/finite) -> deque of N items; first bad index k => HOLD for k..N-1
      |          f. write sidecar (raw payload redacted, images, depth.npy, prompt) + build per-call record
      |          g. whole-call failure (HTTP/timeout/refusal/max_tokens/invalid JSON) => N HOLD items (no retry, no per-tick calls)
      v
 safety_validator.validate_action(raw, current, joint_limits_deg, prev_action, dt) [UNMODIFIED]
      -> _interpolate_waypoints -> robot.send_action (100 Hz) -> io_logger.write_step(action_details = last_details)
      -> NEW hook: if action_source.task_complete_claimed: reason="task_complete"; break  (return-to-start in finally, as today)
```

### Recommended Project Structure
```
control/vla_bridge/
├── mllm_schema.py            # build_action_chunk_schema(chunk_len), schema_sha256(), schema subset lint (provider-agnostic, pure Python)
├── mllm_prompt.py            # PROMPT_TEMPLATE_VERSION, build_system_prompt(...), build_user_text(...), urdf_kinematics_text(), depth_presentation(), forbidden-vocabulary lint
├── mllm_record.py            # redact(), usage normalizers (anthropic/openai/hf), cost estimate, sha256 helpers, sidecar writer
├── backends/
│   ├── base_mllm.py          # MllmActionSourceBase, build_mllm_backend(), load_mllm_backend_config()   (base_ prefix => skipped by discovery)
│   ├── mllm_claude.py        # @register("mllm-claude")  + AnthropicAdapter  (heavy SDK import inside build())
│   ├── mllm_gpt.py           # @register("mllm-gpt")     + OpenAIResponsesAdapter
│   ├── mllm_hf_qwen.py       # @register("mllm-hf-qwen") + shared OpenAIChatAdapter (module `base_mllm_chat.py` or in base_mllm)
│   ├── mllm_hf_deepseek.py   # @register("mllm-hf-deepseek")
│   └── mllm_hf_kimi.py       # @register("mllm-hf-kimi")
└── backend_configs/
    ├── mllm_defaults.json    # shared: chunk_len, sampling, include_urdf, disclose_safety_caps, image settings, depth settings, failure policy
    ├── mllm_claude.json  mllm_gpt.json  mllm_hf_qwen.json  mllm_hf_deepseek.json  mllm_hf_kimi.json   # model id, api_key_env, effort, max_tokens, timeout_s, prices, (HF) provider pin + extra_body
control/test_mllm_schema.py  test_mllm_prompt.py  test_mllm_record.py  test_base_mllm.py  test_backend_mllm_<name>.py ...
```
`load_backend_config(name)` already accepts `[A-Za-z0-9][A-Za-z0-9_-]*`; `mllm_defaults` loads through it. Each registered module is ~20 lines (precedent: `gemini_robotics.py`).

### Pattern 1: Chunk-queue ActionSource (the PIPE-02 pattern)
**What:** `get_action(joint_state, instruction)` keeps the Protocol signature. Internally a `deque` of items, each either `Step(action_dict, k)` or `Hold(reason)`. When empty it performs ONE blocking provider call, parses, pre-validates, enqueues exactly N items, then pops one. Every tick pops one. The base class owns everything except the wire call; provider adapters implement `call(request) -> ProviderResult(text, raw_payload, usage, stop, native_thinking, request_id, applied_settings)`.
**When to use:** all five backends. Success criterion 1 is satisfied structurally: provider differences are confined to the adapter class and config.
**Notes:** keep `_inflight_lock` non-blocking acquire (same as `base_gemini`) so a stray concurrent call holds instead of double-spending. Expose `last_details` (never contains the key or client), `episode_complete`/`task_complete_claimed` (duck-typed attribute read by `run_episode`), `close()`.

### Pattern 2: Programmatic schema from `action_contract.py` (MLLM-04)
**What:** build the schema by looping `JOINT_ORDER`, put units from `ACTION_UNITS` in each property `description`, mark everything `required`, `additionalProperties:false`, key order = reasoning-stage order (`task_restatement`, `plan`, `scene_claims`, `actions`, `task_complete`). Use only the INTERSECTION of provider JSON-schema subsets: types object/array/string/number/boolean, `enum`, `required`, `additionalProperties:false`; NO `minimum/maximum`, NO `minItems/maxItems`, NO `format`, NO recursion. Rationale: Claude supports `minItems` only 0/1 and no numeric constraints; OpenAI strict supports them but D-15 wants ONE schema object and D-04 forbids per-joint min/max anyway. Chunk length N is therefore stated in the prompt and the field `description`, and enforced after the fact (short chunk -> hold the remainder, long chunk -> truncate to N, both flagged).
**Property order:** OpenAI emits keys in schema order; Claude emits required properties first in schema order (so make ALL required, which also stays under Claude's 24-optional-parameter limit). The staged-trace-before-actions ordering is therefore preserved on both. [CITED: developers.openai.com .../structured-outputs "Key ordering"; platform.claude.com .../structured-outputs "Property ordering"]
**Confidence field:** use `enum ["low","medium","high"]` (no numeric bounds needed).
**Hash:** `sha256(json.dumps(schema, sort_keys=True, separators=(",",":")))` recorded per call (D-22b). See Code Example 1.

### Pattern 3: Prompt layering and forbidden vocabulary
**Static system block (cache-friendly, identical every chunk):** role and I/O contract; unit contract generated from `ACTION_UNITS`/`ACTION_MODE`; calibrated limits from the `joint_limits_deg` dict passed to `build()` (degrees; gripper percent, **0 = closed, 100 = open** [CITED: control/vla_bridge/backends/rdt1b_server.py "0 % = closed"; Phase 10 "0 is closed ... +0.036 fully open"]); the statement that degree 0 equals the calibrated midpoint and URDF angle 0 [CITED: eef_to_joint.py]; URDF kinematics text (toggle `include_urdf`); `dt` and chunk seconds (D-16); the qualitative "large jumps are clamped by a safety layer" sentence; the explicit no-primitives statement (the model emits raw absolute joint targets only). Numeric caps (`MAX_RELATIVE_TARGET_DEG`) are NOT included unless `disclose_safety_caps: true`.
**Dynamic user block:** images each preceded by a text label (`wrist`, `left`, `right`, `depth`), current joint state in the same units/keys as the output, previous chunk's `plan` (D-09), the instruction, and the chunk length.
**Lint (unit-testable):** reject prompt+schema containing `reasoning`, `thinking`, `scratchpad`, `chain-of-thought`, `trace` (Claude `reasoning_extraction` risk, Conflict C4) and any movement-primitive vocabulary (`move_to`, `grasp(`, `pick(`, `place(`, `waypoint`, `trajectory helper`) per the anti-features.
**URDF text:** parse `So-101/So-101.urdf` with `xml.etree` into a compact table (joint, parent->child, origin xyz in m and rpy in degrees, axis, limits converted to degrees; gripper prismatic metres noted as the 0-100% channel). Raw file is 14,403 bytes, 90% visual/collision origins; the compact kinematic table is ~2 KB (~1K tokens) vs several K tokens raw. Resolve the path with `Path(__file__).resolve().parents[3] / "So-101" / "So-101.urdf"` (the `eef_to_joint.URDF_PATH` precedent). Never import ikpy for this.

### Pattern 4: Three provider adapters
- **Anthropic (Claude):** `client.messages.stream(model, max_tokens, system, messages, thinking={"type":"adaptive","display":"summarized"}, output_config={"effort": effort, "format": {"type":"json_schema","schema": schema}}, timeout=...)`, then `stream.get_final_message()`. No `temperature/top_p/top_k` (400 otherwise). Images as `{"type":"image","source":{"type":"base64","media_type":"image/png","data":...}}`, one text label block before each. Stop handling: `stop_reason in ("end_turn")` ok; `"refusal"` -> failed call, log `stop_details.category`; `"max_tokens"` -> failed call (output incomplete). Usage: `usage.input_tokens`, `usage.output_tokens`, `usage.output_tokens_details.thinking_tokens`, `cache_read_input_tokens`. Thinking text = concatenated `thinking` blocks (empty unless `display:"summarized"`; summaries only, billed tokens are the full raw thinking). Request id: `message._request_id`.
- **OpenAI (GPT):** Responses API, `client.responses.stream(...)` (or `.create`) with `instructions=system`, `input=[{"role":"user","content":[{"type":"input_text",...},{"type":"input_image","image_url":"data:image/png;base64,...","detail":"high"}, ...]}]`, `reasoning={"effort":"xhigh","summary":"auto"}`, `text={"format":{"type":"json_schema","name":"action_chunk","strict":True,"schema":schema}}`, `max_output_tokens`, `store=False`. No `temperature/top_p`. Handle `status=="incomplete"` (`incomplete_details.reason`) and a `refusal` content part. Usage: `usage.input_tokens`, `usage.output_tokens`, `usage.output_tokens_details.reasoning_tokens`, `usage.input_tokens_details.cached_tokens`. Reasoning summary: items of type `reasoning` with `summary` array (opt-in; whether `gpt-6-astra` accepts `summary` for every org is unverified, see A5). Request/response id: `response.id`, `response._request_id`.
- **HF router (Qwen/DeepSeek/Kimi):** `OpenAI(base_url="https://router.huggingface.co/v1", api_key=os.environ[api_key_env]).chat.completions.with_raw_response.create(model="<hf-id>:<pinned-provider>", messages=[system, user(content list with image_url data URLs)], response_format={"type":"json_schema","json_schema":{"name":"action_chunk","strict":True,"schema":schema}}, max_tokens=..., temperature=1.0, top_p=0.95, reasoning_effort="<level>", extra_body={...per-model...})`. Capture the wire body from `raw.http_response.text` for the sidecar and `raw.parse()` for typed access. Read the answer from `choices[0].message.content`, thinking text from `message.reasoning` or `message.reasoning_content` if present (provider/model dependent; HF's own typed message exposes `reasoning`), usage from the raw dict (`prompt_tokens`, `completion_tokens`, `total_tokens` documented; `completion_tokens_details.reasoning_tokens` only if the provider passes it through, else `null`). A `finish_reason` of `length` is a failed call.

### Pattern 5: Hold semantics (MLLM-07 / D-03 / D-00c)
- **Pre-validate at parse time:** for step k in `actions[:N]`: must be a dict, every `JOINT_ORDER` key present, each value an `int`/`float` (not `bool`), `math.isfinite`. Reuse `base_http_adapter._is_joint_action` for the shape test and add the finite test (it deliberately passes NaN through for the validator; D-03 says malformed holds, so reject non-finite here). Parse with default `json.loads` so a `NaN` literal does not discard good earlier steps (see Code Example 2).
- **First bad index k:** items `[0:k]` are real steps; items `[k:N]` are HOLD (literal D-03 reading: the bad step and the rest of the chunk become hold-current-position, then the next call happens after the chunk drains). A short chunk (< N items) is treated the same at index `len`; extra items beyond N are dropped and flagged.
- **HOLD output:** `(dict(joint_state), f"{label}@{reason}-holding-position")` where `joint_state` is the live reading passed to THAT tick (never zeros, never the previous target). `last_details["bridge_flags"] = [reason]`.
- **Whole-call failure** (transport error, timeout, refusal, `max_tokens`/`length`/`incomplete`, non-JSON, missing top-level keys): enqueue N HOLD items with reason `provider-error|refusal|truncated|invalid-json`, so the next paid call occurs after N ticks, never per tick (anti-feature: no per-tick calls). Add config `max_consecutive_failed_calls` (suggest 3): after that set the termination flag so `run_episode` ends with reason `mllm_backend_failed` instead of burning money; document it. `KeyboardInterrupt` must propagate (use `except Exception`, never bare `except`).
- **Never retry** a malformed chunk (D-03). Provider-parameter negotiation is NOT a retry: omit unsupported sampling/effort params up front using a static per-model capability table in config rather than a try/400/retry loop.

### Pattern 6: Per-call record and sidecar (D-12, D-22, MLLM-06)
- **Where:** on the refill tick, `last_details["mllm_call"] = <record>` (plain JSON types only; `run_episode` passes `last_details` into `IOLogger.write_step(action_details=...)`); on every tick (including the first) `last_details["chunk_ref"] = {"chunk_id": cid, "step_index": k, "chunk_len": N}`. `last_details["raw_model_output"]` = the raw step dict for that tick (what `run_episode` logs as `raw_model_output`), `observation` stays absent.
- **Record fields:** `chunk_id`, `started_utc`, `backend`, `provider`, `model_requested`, `model_returned`, `request_id`, `response_id`, `prompt_template_version`, `prompt_sha256`, `schema_sha256`, `include_urdf`, `chunk_len`, `sdk` (`name`, `version`), `settings` = `{requested:{...}, applied:{...}, omitted:{param: reason}}`, `usage` = `{input_tokens, output_tokens, reasoning_tokens|null, cached_input_tokens|null, raw}`, `latency_ms` = `{depth_fetch, image_encode, api, parse, total}` plus measured `tick_period_s` of the previous chunk, `cost_usd_est` + `price_table` snapshot, `stop` = `{reason, refusal_category|null, incomplete_reason|null}`, `trace` = `{task_restatement, plan, scene_claims[], task_complete, native_thinking_text|null, native_thinking_kind: raw|summary|omitted|none}`, `parse` = `{ok, n_returned, n_valid, first_bad_index, reason}`, `depth` = `{status, valid_fraction, request_ms, calibration_sha256, presentation}`, `images` = `[{role, sha256, bytes, width, height, path}]`, `sidecar_dir`.
- **Sidecar layout (discretion; recommended):** `<args.out>/mllm_calls/<chunk_id:04d>/{prompt.txt, request_meta.json, response_raw.json, depth.npy, images/{wrist,left,right,depth}.png}`. `args.out` is available inside `build(args, ...)`; create the dir lazily (IOLogger opens `episode.jsonl` with mode `"x"` and does `mkdir(exist_ok=True)`, so a pre-existing subdirectory is harmless). Write atomically (tmp + rename) like `IOLogger.write_depth_result`. `chunk_id` is an integer counter (no model-controlled path components).
- **"Verbatim" raw payload, precisely:** HF/OpenAI-compatible non-streaming call: the exact wire body (`raw.http_response.text`). Claude and GPT use streaming (long reasoning): the final SDK response object dumped with `model_dump(mode="json")` (all fields incl. extras). Both are passed through `redact()` (replace the key value, any `Authorization`/`x-api-key` header copies) before writing.
- **io_logger:** no change is strictly required. Optional small helper `IOLogger.write_sidecar(rel_path, data)` mirroring the `write_depth_result` atomic-write idiom would avoid duplicating that code; if added, keep it additive.

### Pattern 7: Depth in the prompt without touching the PolicyServer observation dict (D-06)
- The MLLM source owns a `DepthCameraClient` built in `build()` from `args.depth_endpoint` / `args.depth_calibration` (the argparse Namespace already carries them; the existing `main()` builds its own client from the same flags). It reads the raw stereo pair itself, calls `compute_depth_result(left_raw, right_raw)` (requires the raw 800x600 frames: the client rejects any size not equal to calibration `image_size`), so it cannot reuse `capture_observation_images` (that downsizes to 640 px). Write a new `capture_mllm_inputs()` that reuses `_encode_png` and `_downsize_for_transport` and keeps `read_left()` then `read_right()` back-to-back (one physical AR0144 read; any third call in between triggers a fresh read).
- Measured depth: valid map is 576x800 float32 metres, pixel-aligned to the **rectified, resized left** image (`aligned_left` in the result), median ~0.83 m on the Phase 12 scene. The raw left image differs from `aligned_left` by the rectification warp. Recommended default: send `aligned_left` in the "left" slot when depth succeeded (so depth claims ground pixel-for-pixel) and raw left otherwise, logging which; alternative: always raw left plus a prompt note. Planner's call (discretion).
- **Presentation (recommend BOTH):** (1) turbo heatmap via `cv2.applyColorMap` with a FIXED metre range (e.g. 0.2-1.5 m, constants in config) and a drawn legend with 3-5 labelled ticks; (2) a compact numeric block: valid fraction, min/median/max, and a coarse grid (e.g. 12x9) of median metres with `?` for NaN. MLLMs read quantity from numbers far better than from colour.
- **Failure:** depth failure (HTTP, size mismatch, 429) never blocks the chunk: send three images, state "depth unavailable this call" in the prompt, set `depth.status` in the record. Retry HTTP 429 up to 2 times with 2 s sleeps before giving up.
- Scope: this is MLLM-only; VLA backends and `BridgeActionSource`/`connect_bridge` are untouched, which satisfies D-06 and the DEPTH-CAL-03 scoping note.

### Pattern 8: Termination hook (D-18/D-19)
- `run_episode` currently ends only on `max_steps`, `KeyboardInterrupt`, exception. Add, right after `steps_completed = i + 1`: `if getattr(action_source, "episode_complete_reason", None): reason = action_source.episode_complete_reason; break`. Reasons: `task_complete`, `mllm_backend_failed`. The existing `finally` still performs the return-to-start move and writes `termination.json` (`_write_termination` takes any string). Update the `ActionSource` Protocol docstring (optional attribute, duck-typed).
- **Tail rule (discretion) recommendation:** define `task_complete` in the prompt as "the task goal is already achieved in the CURRENT camera views", so a `true` claim is a judgement about the present scene (clean for D-19 false-completion analysis). On `true`: log the whole call record, enqueue HOLD items for the entire chunk (do not execute the model's trailing actions), and end the episode on the first tick after logging. This is safer than D-18's default (execute the remainder) and still satisfies "episode ends on the first `true`". Alternative if the user prefers D-18's default: execute the chunk through D-03/D-04 and end when it drains.

### Anti-Patterns to Avoid
- **Per-tick or failure-triggered per-tick provider calls:** a persistent 400/refusal would otherwise call the API every 0.785 s.
- **Prompting for "reasoning/thinking/trace"** (Claude refusal) and **prefilling** the assistant turn (400 on Fable 5.1).
- **`tool_choice`/function calling** for structure: forbidden by the roadmap, and forced tool use is a 400 on Fable 5.1 anyway.
- **Sending numeric safety caps or per-joint min/max in the schema** (D-04/D-16).
- **Relying on HF default `:fastest` routing** (silently loses JSON-schema support).
- **Putting NaN-tolerant "pass-through" semantics in front of the validator for the MLLM lane** (D-03 says malformed holds).
- **Returning non-JSON-serialisable objects in `last_details`** (pydantic models, numpy scalars, `Path`, `bytes`): `IOLogger.write_step` uses plain `json.dumps` and would crash the live loop.
- **Mutating `capture_observation_images`/Phase 14 modules** to bolt on raw frames; add a new MLLM capture helper instead.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HTTP/retry/streaming/error typing for Claude/GPT | custom `requests` clients | `anthropic` / `openai` SDKs (streaming helpers, typed errors, `_request_id`) | SSE parsing, 10-minute long-request guard, 429/529 mapping, usage objects. |
| Structured output enforcement | regex/"respond in JSON" parsing | provider-native `output_config.format` / `text.format` / `response_format` json_schema | Grammar-constrained decoding; Claude's grammar does not constrain thinking so it coexists with adaptive thinking. [CITED: structured-outputs "Grammar scope"] |
| URDF parsing | ikpy/own XML walker | stdlib `xml.etree.ElementTree` for the ~12 joints/links | Tiny, no dependency; ikpy is for IK and is heavy. |
| Joint limits | hard-coded numbers | `action_contract.load_joint_limits_deg()` result passed to `build()` | Single source of truth already used by the validator. |
| Depth heatmap | matplotlib figure pipeline | `cv2.applyColorMap` + `cv2.putText` legend | matplotlib is not a runtime dep of `control/requirements.txt`; cv2 already is. |
| Secret redaction | ad-hoc `str.replace` per call site | one `redact(obj, secrets)` walker applied to every record, sidecar and error string (reuse `base_gemini._redact` semantics) | Secrets must not appear in any artifact (D-00b/D-22). |
| Safety clamping | any local clamp of model values | unmodified `safety_validator.validate_action()` | D-04: raw behaviour must remain visible in logs. |
| Provider-parameter negotiation | try/400/retry loops | static per-model capability table in config (`sampling_supported`, `effort_levels`, `summary_supported`) + requested-vs-applied logging | No extra paid calls; deterministic tests. |

**Key insight:** every hard part (transport, constrained decoding, usage accounting, clamping, logging) is provided by a vendor SDK or an existing project module; the new code is orchestration, prompt/schema generation, and failure policy.

## Chunk-Size Derivation (MLLM-05, D-02, D-17)

**Measured inputs (Phase 12, `control/outputs/`):**
| Quantity | Value | Source |
|----------|-------|--------|
| Real tick period (non-stall step-to-step timestamps) | median 0.785 s / mean 0.785 s (episode 004, n=182), median 0.783 s (episode 121123, n=54) | `episode.jsonl` timestamps |
| Nominal tick (`--control-hz 2.0`) | 0.5 s; `action_to_execution` 640-690 ms alone | `latency_ms.action_to_execution` |
| Depth round trip (client-observed `request_ms`, n=26) | min 5.13 s, median 6.15 s, max 7.13 s; server `inference_ms` median 180 ms (so ~97% is transport/rectify/encode) | `depth.jsonl` |
| VLA chunk refetch stalls (SmolVLA bridge) | 3.3-5.0 s typical, outliers 14.3 s and 23.0 s (one stale-watchdog cycle) | `episode.jsonl` step gaps |
| SmolVLA chunk | `actions_per_chunk=50`, refetch at 50% (= 25-step horizon, ~20 s real) | `robot_client.py`, `12-UAT.md` test 7 |
| Defaults | `--control-hz 2.0`, `--execution-hz 100`, `--max-steps 60` (~47 s of motion at 0.785 s/tick) | `run_vla_episode.py` |

**Not measured anywhere:** MLLM API latency. Router throughput metadata gives a ceiling estimate: `time ~= output_tokens / throughput` (Qwen3.8-27B on Cerebras ~831 tok/s, DeepSeek-V4-Flash-Vision-Exp on DeepInfra ~89 tok/s, Kimi-K3 on Baseten ~74 tok/s). For Claude/GPT at `xhigh` with an assumed 10-40K reasoning tokens the call is plausibly 1-5 minutes [ASSUMED]. In every case the robot idles for `L = depth(~6 s) + encode + API` per call, and `L >> N*dt`.

**Derivation:** (a) Lower bound: amortise the fixed ~6 s depth fetch and per-call overhead so that each call buys at least ~8 s of motion: `N >= 10` (7.9 s at 0.785 s/tick). (b) Upper bound: closed-loop re-planning (D-08) needs at least 3 calls inside the default 60-tick budget, and open-loop blind execution should stay under ~16 s: `N <= 20` (15.7 s). (c) Output-token side: ~45 tokens per step (`{"shoulder_pan": -12.5, ...}`), so N=20 is ~900 tokens of actions, negligible against reasoning tokens. **Recommendation: `chunk_len = 20` in `mllm_defaults.json`**, per-backend override allowed (D-02), prompt text "exactly 20 steps = 10.0 s nominal at dt 0.5 s" (and log the measured ~15.7 s). Duty cycle for reference at `N=20, L=60 s`: 15.7/(15.7+60) = 21%; N=10: 12%; N=30: 28%; N=50: 39%. Document this table in `mllm_defaults.json` `notes` and in the plan summary so MLLM-05's "derived, not guessed" is auditable, and add the real per-call `latency_ms` to the record so Phase 16 can refine it. Operators will want `--max-steps` raised for MLLM runs (60 ticks = 3 calls); document the recommended value in the run command.

## Earlier Failed Experiment: what the repo actually shows (item 4)

**Evidence status.** `experiment-design/` is git-ignored (local only). `phase-1/RUN_LOG.md` is the empty template; the `raw/output.txt` / `refined/output.json` files its template references do not exist; the `claude-code/` and `codex/` task folders contain only INPUT attachments (prompt1/prompt2, calibration, URDF, `robot.xml`, `<LIVE_VALUE>` joint-state placeholders); `recordings/` holds six Codex-only camera captures (two empty/short). The only record of failure is one sentence in `.planning/PROJECT.md` ("failed badly", research-partner consensus, 2026-09-17). **No model output, score, or log survives, so the proximate failure cause is not recoverable. Everything below distinguishes artifact evidence (E) from speculation (S).**

| # | Confound | Evidence / speculation |
|---|----------|------------------------|
| 1 | Harness: a general coding agent (Claude Code / Codex) was told to "control the servos directly... using whatever means you deem fit (see ports.json)... do not just describe, plan, or output an action", i.e. author and run its own serial driver. This conflates tool-use/driver-writing with spatial reasoning and is inconsistent with the design doc's own "Phase 2: output ONLY the JSON". | E: `claude-code/stage-4/task-d/refined/prompt2.txt` vs `docs/multimodal-context-ablation-experiment.md`. Whether this caused the failure: S. |
| 2 | Four unit systems in one context: raw servo ticks (safety limits), URDF radians, normalized [-1,1] (design-doc JSON schema), degrees (joint_state.json, the true control unit). The prompt told the model never to exceed tick limits while the state was in degrees; the tick-to-degree mapping (homing offsets, 4095 resolution) was never given. | E: prompt2.txt, design doc, `action_contract.py`. Effect on outputs: S. |
| 3 | Gripper convention inverted: design doc says `0.0 = fully open, 1.0 = fully closed`; hardware/LeRobot contract is percent-open (0 = closed, 100 = open). | E: design doc vs `rdt1b_server.py`, Phase 10 established model. Effect: S (likely wrong gripper behaviour on any gripper task). |
| 4 | Stale/wrong kinematic context: the supplied `soarm101.urdf` was the pre-Phase-10 Coppelia export (Phase 10 context records that its gripper is never parented under the arm); the prompt quotes `wrist_flex [-3.228, +0.088]` (asymmetric), `wrist_roll`/gripper "not modeled", while the current URDF is symmetric ±1.78 rad with wrist_roll and prismatic gripper modeled. | E: prompt2.txt vs `So-101/So-101.urdf`; STATE.md Phase 10 note. Effect: S. |
| 5 | Hand-filled live state (`<LIVE_VALUE>` placeholders to edit by hand) and no timing semantics (no dt, no motion budget), no feedback loop (single shot). | E: `joint_state.json`, prompts. Effect: S. |
| 6 | Perception: stages 1-3 gave a wrist frame plus "stereo depth frame - left+right pair or depth map" (an unmeasured raw pair, not metric depth); metric FastFS depth did not exist until Phase 12; AR0144 2560x720 mode was frozen at the time and the recording runs use `stereo_combined_index: 0` (possibly the laptop webcam, the same device-index drift later fixed in Phase 13). | E: `recordings/*/meta.json`, quick tasks 260927-he3/i3s, STATE.md. Effect: S. |
| 7 | No schema enforcement: JSON validity relied on the model obeying "Output ONLY the JSON". | E: design doc. Effect: S. |

**What the structured-output design must do differently (by construction):** one unit system (degrees/percent) generated from `action_contract.py` and the live calibration; correct gripper direction stated; current URDF (Phase 10) rendered as compact text with limits in degrees; live joint state injected by code (never by hand); native JSON-schema decoding; closed-loop chunks with stated `dt`; metric depth plus pixel-aligned left image; no driver authoring by the model (the harness executes JSON). **What it cannot fix:** genuine spatial-reasoning/embodiment limits, the unverified joint-sign convention (C8), and the arm's real rest pose being folded (live reading in `vla_episode_004`: shoulder_lift -101.9 deg, elbow 94.1 deg, wrist_flex 65 deg, gripper 88.9%), so degree 0 is not "arm at rest" and the model must infer pose from images and the URDF. Expect the richest-context tier may still fail (CONTEXT.md specifics); that is a valid Phase 16 finding.

## Provider Facts (live, 2026-10-02)

### Claude (D-14, D-15, D-20..D-22)
| Item | Fact | Source |
|------|------|--------|
| Flagship model id | `claude-fable-5-1` (released 2026-09-01; 1M ctx; 128K max output; $10 in / $50 out per MTok; cache read $0.25; retirement not before 2027-09-01). `claude-mythos-5-1` = same capabilities, Project Glasswing only. `claude-opus-5-5` ($4/$20) is Anthropic's "start here" model. | platform.claude.com/docs/en/models/overview, fable-5-1/overview |
| Vision | text+image in, text out; images base64 or URL; up to 600 images/request (100 on 200K-ctx models), 8000x8000 px max, **10 MB per image (base64) on Claude API**, 32 MB request; 2000x2000 px cap if >20 images; token cost `ceil(w/28)*ceil(h/28)` (640x480 = 414 tokens; four images ~1.7K) | .../build-with-claude/vision |
| Structured output | `output_config.format = {"type":"json_schema","schema":{...}}`; supported: basic types, `enum`, `const`, `anyOf`, `$ref`, `required`, `additionalProperties:false` (mandatory), `minItems` only 0/1; NOT supported: recursion, `minimum/maximum/multipleOf`, `minLength/maxLength`, array constraints beyond minItems 0/1. Optional-parameter limit 24, union-type limit 16. First request pays grammar compile; compiled grammar cached 24 h. Output keeps schema key order, required first. Works with streaming and with thinking (grammar does not apply to thinking). Incompatible with prefill and citations. `stop_reason:"refusal"` and `"max_tokens"` can yield non-conforming output. | .../structured-outputs; installed SDK types |
| Thinking | Adaptive thinking is ALWAYS ON for Fable 5.1 (`thinking:{"type":"enabled"/"disabled"}` -> 400; omit or `{"type":"adaptive"}`); `display` defaults to `"omitted"` (empty `thinking` field), set `"summarized"` for readable summaries (never raw CoT; billed for full raw thinking tokens; thinking counts toward `max_tokens`). Prefill -> 400. Forced `tool_choice` -> 400. | .../build-with-claude/thinking; fable-5-1/whats-new |
| Effort | `output_config.effort` in `low/medium/high/xhigh/max` (default `high` on Fable 5.1). Applies to all output tokens incl. thinking. Opus-4.7-era note: `max` "can lead to overthinking" on structured-output tasks. Set a large `max_tokens` at high effort (hard cap on thinking + text). | .../build-with-claude/effort |
| Sampling | Non-default `temperature`, `top_p` or `top_k` -> HTTP 400 on Fable 5.1/Opus 5.5/Sonnet 5.5/etc. regardless of thinking. SDK 1.11.0 `messages.create` has no such parameters. | .../thinking "Sampling parameters"; SDK signature |
| Usage | `usage.input_tokens`, `usage.output_tokens`, `usage.output_tokens_details.thinking_tokens` (raw thinking tokens, <= `output_tokens`; streaming: only on final `message_delta`), `cache_read_input_tokens`, `cache_creation_input_tokens`. | .../thinking-steering-and-cost; SDK `Usage` fields |
| Refusal | `stop_reason:"refusal"`, HTTP 200, `stop_details.category` in `cyber|bio|frontier_llm|reasoning_extraction|general_harms|null`; `reasoning_extraction`/`bio`/`frontier_llm` refusals before any output are billed (since 2026-09-24). | .../refusals-and-fallback |
| Long requests | SDK rejects non-streaming requests expected to exceed 10 minutes; use `messages.stream(...)` then `get_final_message()`; 413 above 32 MB; 504/529 possible. | .../api/errors "Long requests" |
| Retention | Fable 5.1 is a "Covered Model": 30-day retention, not ZDR. | .../models/fable-5-1/whats-new |

### OpenAI (D-14, D-15, D-20..D-22)
| Item | Fact | Source |
|------|------|--------|
| Flagship model id | `gpt-6-astra` (default snapshot `gpt-6-astra`; 1.05M ctx, 922K max input, 128K max output; $10 in / $1 cached / $50 out per M; >272K input is 2x/1.5x). Others: `gpt-6.1-sol`, `gpt-6-luna`. | developers.openai.com/api/docs/models/gpt-6-astra |
| Endpoints / features | Chat Completions and Responses supported; structured outputs, image input, streaming, prompt caching. Responses API is the recommended path; Chat Completions without tools is allowed. | same + guides/latest-model |
| Reasoning | `reasoning.effort`: `low`, `medium` (default), `high`, `xhigh`, `max`; `none`/`minimal` unsupported (HTTP 400). `reasoning.summary`: `auto|concise|detailed`, opt-in only. Reasoning tokens billed as output and counted in `max_output_tokens`; `status:"incomplete"` with `incomplete_details.reason:"max_output_tokens"` possible before any visible output; OpenAI suggests reserving >= 25,000 tokens. | guides/reasoning; guides/latest-model |
| Sampling | "When reasoning effort is not `none`, remove `temperature`, `top_p`, and `top_logprobs`" (Astra never allows `none`). | guides/latest-model "Unsupported parameters" |
| Structured outputs | Responses: `text.format = {"type":"json_schema","strict":true,"name":...,"schema":...}`; Chat Completions: `response_format`. All fields required, `additionalProperties:false`, up to 5000 properties / 10 nesting levels; output order follows schema key order; supports `minimum/maximum`, `minItems/maxItems` for non-fine-tuned models; first request with a new schema has extra latency; refusal appears as a `refusal` content part; strict mode with an unsupported keyword errors. | guides/structured-outputs |
| Images | PNG/JPEG/WEBP/non-animated GIF; up to 1,500 images and 512 MB per request; `detail` low/high/original/auto; patch tokenization 32x32 (`gpt-6-astra` high: 1024x1024 ~ 1,229 tokens; 640x480 ~ 360 tokens). Input via URL, base64 data URL, or file id. | guides/images-vision |
| Usage | Responses `usage.input_tokens`, `usage.output_tokens`, `usage.output_tokens_details.reasoning_tokens`, `usage.input_tokens_details.cached_tokens`. | guides/reasoning; installed SDK `ResponseUsage` |

### HF Inference Providers (D-13, D-15, D-23; router API fetched 2026-10-02)
Common to all: OpenAI-compatible `https://router.huggingface.co/v1/chat/completions`; auth `HF_TOKEN`; provider pin by suffix (`model:provider`) or provider-specific base URL; `response_format:{"type":"json_schema","json_schema":{"name","schema","strict"}}` documented; `reasoning_effort` documented ("provider and model dependent"); documented usage = `prompt_tokens/completion_tokens/total_tokens` only; HF passes provider prices through with no markup; free users $0.10/month credit, PRO $2/month. Default routing `:fastest` ignores structured-output support.

| Model | Vision | Thinking / effort control (model card) | Providers live (price in/out per M, structured output, throughput) | Recommended pin | Verdict |
|-------|--------|----------------------------------------|----------------------------------------------------------------------|-----------------|---------|
| `Qwen/Qwen3.8-27B` (Apache-2.0) | image+text in (router `input_modalities`) | thinking ON by default, disable per request; `reasoning_effort` xhigh (default) / medium / low; `preserve_thinking` default on; thinking sampling `temperature=1.0, top_p=0.95, top_k=20`; reasoning returned as `reasoning`/`reasoning_content` (card's streaming example) | novita ($0.42/$3, structured NO, 39 tok/s, ctx 1M); cerebras ($0.99/$1.49, YES, 831 tok/s, ctx 65,536); ovhcloud ($0.47/$3.19, YES, 50 tok/s, 262K); deepinfra ($0.20/$2.50, YES, 35 tok/s, 262K); featherless (no data) | `cerebras` for speed (watch the 65,536-token context: prompt ~15K + reasoning must fit, set `max_tokens` <= ~32K) else `deepinfra` | KEEP (cheap) |
| `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (MIT, experimental) | image+text in; card: "first experimental multimodal model in the V4 family" | reasoning with effort levels (card benchmarks use `max` reasoning effort, `temperature=1.0, top_p=0.95`); vLLM recipe uses a `deepseek_v4` reasoning parser | novita ($0.44/$1.32, structured NO, 127 tok/s); deepinfra ($0.44/$1.32, structured YES, 89 tok/s) | `deepinfra` (Novita lacks JSON-schema) | KEEP (cheap; experimental) |
| `moonshotai/Kimi-K3` (Kimi K3 License, not OSI) | native multimodal, image+video in | thinking ALWAYS on, returns `reasoning_content`; `reasoning_effort` low/high/max (default max); `temperature=1.0`, `top_p=0.95` for single-step (1.0 agentic); preserved-thinking multi-turn (not needed: single-turn calls) | together ($2.70/$13.50, structured NO, 48 tok/s); fireworks ($3/$15, YES, 36 tok/s); baseten ($3/$15, YES, 74 tok/s); deepinfra ($2.85/$14.25, YES, 10 tok/s); featherless (no data) | `baseten` (fastest with structured output) | KEEP (NOT cheap, slow-ish; flag cost C1) |
| `google/gemma-4-31B-it` (fallback, Apache-2.0) | image+text | not exercised | novita ($0.14/$0.40, YES), deepinfra ($0.15/$0.40, YES), together status `error` | — | not needed; ready as replacement |
| `zai-org/GLM-5.3-Flash` (fallback, MIT) | image+text | not exercised | 7 providers incl. baseten/deepinfra/fireworks ($0.15/$0.50, YES) | — | not needed; ready as replacement |

Note: no model failed vision or reasoning, so none is dropped and no replacement is used (D-23 condition not triggered). All throughput/price figures are router metadata at fetch time and drift; store them in config with a `verified_on` date. [VERIFIED: https://router.huggingface.co/v1/models and https://huggingface.co/api/models/<id>?expand[]=inferenceProviderMapping, 2026-10-02; CITED: model cards]

### Image inputs and URDF cost (item 3)
Four images per call: wrist, left, right, depth heatmap, each longest edge <= 640 px (existing `OBSERVATION_MAX_SIDE`), PNG (lossless; ~0.4-0.6 MB each, ~2.4 MB total base64, far under every cap: Claude 10 MB/image and 32 MB request, OpenAI 512 MB). Tokens: Claude ~414 per 640x480 image (~1.7K total); GPT-6 Astra ~360 per image at `detail:"high"`; Qwen 32-px patches (~300 per image) [ASSUMED]. HF per-provider image limits and max request size are NOT documented by the router; the 2.4 MB request is expected to be fine [ASSUMED]; offer `image_format: "png"|"jpeg"` (JPEG q90 ~60 KB each) as the fallback if a provider returns 413. Compact URDF text ~1K tokens (raw XML would be ~5-8K). Estimated prompt total ~6-12K input tokens per call. Wrist frames: flush stale buffers with a few `cap.grab()` calls before the read, because a plain `cv2.VideoCapture` can hold an old frame after a minutes-long idle (the AR0144 `StereoSplitCamera` reader thread keeps only the latest frame, so it is fine) [ASSUMED for macOS AVFoundation buffering].

### Cost and latency estimates (per call; token counts are ASSUMED)
Assuming ~12K input and ~20K output (mostly reasoning): Claude Fable 5.1 or GPT-6 Astra ~ $0.12 + $1.00 = ~$1.1 per call (~$3.5 per 3-call episode); Kimi-K3 (baseten) ~ $0.04 + $0.30 = ~$0.35; Qwen3.8-27B (cerebras) ~ $0.01 + $0.03 = ~$0.04; DeepSeek-V4-Flash-Vision-Exp (deepinfra) ~ $0.005 + $0.026 = ~$0.03. Wall time per call ~ depth 6 s + `20K / throughput` for HF (Qwen ~24 s, DeepSeek ~225 s, Kimi ~270 s) and an unmeasured 1-5 min for Claude/GPT at xhigh [ASSUMED]. Therefore set `timeout_s` generously (default 900) and surface a one-line console message when a call starts and ends.

## Common Pitfalls

### Pitfall 1: Reasoning-extraction refusal on Claude
**What goes wrong:** `stop_reason:"refusal"` with an empty or non-conforming body, billed, on every chunk. **Why:** a JSON field or prompt text that asks for "reasoning/thinking/trace/scratchpad". **Avoid:** field names `task_restatement/plan/scene_claims`, deliverable-style instructions, lint test. **Warning sign:** first live Claude call returns `stop_details.category == "reasoning_extraction"`.

### Pitfall 2: Silent loss of JSON-schema mode via HF default routing
**What goes wrong:** the call succeeds but returns prose/fenced JSON or a 400. **Why:** `:fastest` selects a provider with `supports_structured_output:false` (DeepSeek -> Novita). **Avoid:** pinned `model:provider`, config validation that the pin is non-empty, record `structured_mode_applied`, live smoke per variant.

### Pitfall 3: Thinking suppressed by grammar on an HF provider
**What goes wrong:** with `response_format` set, some stacks apply the grammar from the first token, so the model skips its `<think>` block (reasoning tokens 0). **Why:** structured decoding vs reasoning-parser interplay is provider-specific (vLLM supports reasoning + structured output, but whether each HF provider enables it is unverified). **Avoid:** D-10's staged JSON fields are the fallback trace; log `reasoning_tokens` (or `null`) and `native_thinking_kind` so a suppressed-thinking provider is visible; add a live smoke assertion.

### Pitfall 4: Per-tick paid retries after a failure
**What goes wrong:** an HTTP 400/refusal/timeout leaves the queue empty and the next tick calls the API again, repeatedly. **Avoid:** failure => N HOLD items; consecutive-failure cap; test asserting provider call count == 1 per N `get_action` calls and per failure.

### Pitfall 5: Depth service busy (HTTP 429) and size mismatch
**What goes wrong:** the episode's background `DepthRecorder` and the backend's synchronous depth call overlap; FastFS returns 429 (non-blocking lock). Or the frames are the 640-px downsized ones and `compute_depth_result` raises "capture resolution does not match calibration". **Avoid:** backend-owned depth (do not also pass `depth_client` to `run_episode`, or retry 429), feed raw 800x600 stereo frames to the depth client and downsized copies to the prompt, depth failure is non-fatal.

### Pitfall 6: `last_details` not JSON-serialisable
**What goes wrong:** `IOLogger.write_step` does `json.dumps(record)` and the whole live loop crashes mid-episode. **Avoid:** convert SDK objects with `model_dump(mode="json")`, floats not numpy scalars, no `Path`/`bytes`; a test round-trips the record through `json.dumps`.

### Pitfall 7: Image/depth pixel mismatch
**What goes wrong:** depth claims cite coordinates that do not line up with the raw left image. **Why:** depth is aligned to the rectified, resized left frame (576x800). **Avoid:** use `aligned_left` for the left slot when depth is present, or state the alignment in the prompt; record which was sent.

### Pitfall 8: Stale wrist frame after a long blocking call
**Avoid:** `grab()` flush before `read()` for the wrist capture; log capture timestamps in the record (stereo has `last_pair_metadata`).

### Pitfall 9: SDK refuses large `max_tokens` non-streaming
**Avoid:** `messages.stream(...).get_final_message()` for Claude and `responses.stream(...)` for GPT; set explicit `timeout`.

### Pitfall 10: `incomplete`/`max_tokens` truncation is common at high effort
**Avoid:** `max_tokens`/`max_output_tokens` = 64,000 default for Claude/GPT (thinking+text share it), HF per-provider context limits (Cerebras 65,536), treat truncation as a failed call, log usage so Phase 16 sees it.

### Pitfall 11: `NaN`/`Infinity` literals
**Why it matters:** Python `json.loads` accepts them and `_is_joint_action` lets NaN through (the validator would hold only that joint). D-03 wants hold-from-here. **Avoid:** explicit `math.isfinite` pre-validation (Code Example 2).

### Pitfall 12: Short/long chunk length and `task_complete` semantics
Providers cannot be forced to emit exactly N (Claude schema cannot express it). Truncate/hold and flag; define `task_complete` as "already achieved in the current views" in the prompt.

### Pitfall 13: Real tick period (0.785 s) vs prompt dt (0.5 s)
Absolute targets 0.5 s "apart" are executed ~36% slower; the validator's velocity cap uses `dt_s = 1/control_hz`. Log measured tick period; do not "fix" the validator (unmodified by decision).

### Pitfall 14: Config/Registry hygiene
Backend names must be unique (`_add` raises); add the new base class to `test_backend_registry._action_source_bases()`; `base_` prefix modules are skipped by discovery; every registered `build()` must raise `SystemExit` BEFORE touching the robot when the key env var is unset (build the action source first, as `build_gemini_backend` does); `api_key_env` must validate as an uppercase env-var NAME so a pasted key is rejected without echoing it.

### Pitfall 15: Operator Ctrl-C during a long call
`except Exception` only; let `KeyboardInterrupt` reach `run_episode`'s return-to-start path; the streaming context manager must close the connection.

## Code Examples

### Example 1: Schema generated from `action_contract.py` (verified to run against the real module)
```python
# Source: control/vla_bridge/action_contract.py (JOINT_ORDER, ACTION_UNITS); schema subset per
# platform.claude.com/docs/en/build-with-claude/structured-outputs and developers.openai.com/api/docs/guides/structured-outputs
import hashlib, json
from vla_bridge.action_contract import ACTION_UNITS, JOINT_ORDER

SOURCES = ("wrist", "left", "right", "depth")

def build_action_chunk_schema(chunk_len: int) -> dict:
    step = {
        "type": "object",
        "properties": {j: {"type": "number", "description": f"{j} absolute target, {ACTION_UNITS[j]}"}
                       for j in JOINT_ORDER},
        "required": list(JOINT_ORDER),
        "additionalProperties": False,
    }
    claim = {
        "type": "object",
        "properties": {"claim": {"type": "string"},
                       "source": {"type": "string", "enum": list(SOURCES)},
                       "confidence": {"type": "string", "enum": ["low", "medium", "high"]}},
        "required": ["claim", "source", "confidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "task_restatement": {"type": "string"},
            "plan": {"type": "string"},
            "scene_claims": {"type": "array", "items": claim},
            "actions": {"type": "array", "items": step,
                        "description": f"Exactly {chunk_len} steps, one per control tick."},
            "task_complete": {"type": "boolean"},
        },
        "required": ["task_restatement", "plan", "scene_claims", "actions", "task_complete"],
        "additionalProperties": False,
    }

def schema_sha256(schema: dict) -> str:
    return hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
```

### Example 2: Chunk parse + pre-validation (D-03; verified to run)
```python
import json, math, numbers
from vla_bridge.action_contract import JOINT_ORDER

def parse_chunk(text: str, chunk_len: int):
    """Return (doc, good_steps, first_bad_index_or_None). Raises ValueError/KeyError only for
    whole-response failures (invalid JSON, missing top-level keys) -> caller turns into N HOLDs."""
    doc = json.loads(text)                      # default parser accepts NaN; the per-step finite check rejects it
    steps = doc["actions"]
    good = []
    for k, step in enumerate(steps[:chunk_len]):
        ok = isinstance(step, dict) and all(
            isinstance(step.get(j), numbers.Real) and not isinstance(step.get(j), bool)
            and math.isfinite(step[j]) for j in JOINT_ORDER)
        if not ok:
            return doc, good, k                 # step k and everything after => HOLD
        good.append({j: float(step[j]) for j in JOINT_ORDER})
    return doc, good, (len(good) if len(good) < chunk_len else None)   # short chunk => hold the rest
```

### Example 3: Claude request (shape verified against anthropic 1.11.0 types)
```python
# Source: platform.claude.com/docs/en/build-with-claude/{structured-outputs,effort,thinking}; anthropic 1.11.0
import anthropic
client = anthropic.Anthropic(api_key=key, timeout=900.0)
content = []
for role, b64 in images:                         # [("wrist", ...), ("left", ...), ("right", ...), ("depth", ...)]
    content += [{"type": "text", "text": f"Image: {role}"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}]
content.append({"type": "text", "text": dynamic_user_text})
with client.messages.stream(
    model="claude-fable-5-1", max_tokens=64000, system=system_prompt,
    messages=[{"role": "user", "content": content}],
    thinking={"type": "adaptive", "display": "summarized"},
    output_config={"effort": "xhigh", "format": {"type": "json_schema", "schema": schema}},
) as stream:
    msg = stream.get_final_message()
text = next(b.text for b in msg.content if b.type == "text")
thinking_tokens = msg.usage.output_tokens_details.thinking_tokens   # may be None/0 when skipped
# stop_reason: "end_turn" ok | "refusal" -> msg.stop_details.category | "max_tokens" -> truncated
# NO temperature / top_p / top_k (HTTP 400 on this model; SDK has no such params)
```

### Example 4: OpenAI Responses request (shape verified against openai 3.23.0 types)
```python
# Source: developers.openai.com/api/docs/guides/{structured-outputs,reasoning,latest-model}
from openai import OpenAI
client = OpenAI(api_key=key, timeout=900.0)
content = [{"type": "input_text", "text": dynamic_user_text}]
for role, b64 in images:
    content += [{"type": "input_text", "text": f"Image: {role}"},
                {"type": "input_image", "image_url": f"data:image/png;base64,{b64}", "detail": "high"}]
with client.responses.stream(
    model="gpt-6-astra", instructions=system_prompt,
    input=[{"role": "user", "content": content}],
    reasoning={"effort": "xhigh", "summary": "auto"},
    text={"format": {"type": "json_schema", "name": "action_chunk", "strict": True, "schema": schema}},
    max_output_tokens=64000, store=False,          # NO temperature / top_p (must be removed for Astra)
) as stream:
    resp = stream.get_final_response()
# resp.status == "incomplete" -> resp.incomplete_details.reason ; resp.output_text ; resp.usage.output_tokens_details.reasoning_tokens
```

### Example 5: HF router request with raw wire capture
```python
# Source: huggingface.co/docs/inference-providers/guides/structured-output (OpenAI client + router base URL)
from openai import OpenAI
client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf_token, timeout=900.0)
raw = client.chat.completions.with_raw_response.create(
    model="deepseek-ai/DeepSeek-V4-Flash-Vision-Exp:deepinfra",       # PIN the provider
    messages=[{"role": "system", "content": system_prompt},
              {"role": "user", "content": [{"type": "text", "text": dynamic_user_text},
                                           {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}],
    response_format={"type": "json_schema", "json_schema": {"name": "action_chunk", "strict": True, "schema": schema}},
    max_tokens=32000, temperature=1.0, top_p=0.95, reasoning_effort="max",   # applied only if the pinned provider honours them
    extra_body={},                                                           # per-model knobs (e.g. chat_template_kwargs) from config
)
wire_body = raw.http_response.text          # byte-exact payload -> redact() -> sidecar
completion = raw.parse()                    # typed; reasoning text may be in message.reasoning / reasoning_content (model_extra)
```

### Example 6: Termination hook (the second `run_vla_episode.py` edit)
```python
# in run_episode(), immediately after `steps_completed = i + 1`
done_reason = getattr(action_source, "episode_complete_reason", None)   # None | "task_complete" | "mllm_backend_failed"
if done_reason:
    reason = done_reason
    break                                    # existing finally: return-to-start, then _write_termination(reason, steps_completed)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Claude `thinking:{"type":"enabled","budget_tokens":N}` | Adaptive thinking (always on for Fable/Opus 5.5) steered by `output_config.effort` (low..max); `enabled`/`disabled` -> 400 | Claude 5 generation (2026) | D-21 "budget" is expressed as `effort`, not a token budget. |
| `temperature` always tunable | Rejected/removed on Claude 5-gen and GPT-6 reasoning | 2026 | "Same nominal sampling" is only applied on HF lane (C3). |
| OpenAI Chat Completions `response_format` as the default | Responses API `text.format` json_schema strict, reasoning items, `store:false` | 2025-26 | Use Responses for GPT; Chat Completions for the HF router. |
| Anthropic structured outputs in beta | GA `output_config.format` (JSON outputs) | 2026 | No beta header; keep all properties required. |
| HF Inference "text-generation" widgets | Router OpenAI-compatible chat completions with provider pin/policy suffixes | 2025-26 | Pin provider for schema support. |
| Gemini Robotics VLA | Gemini Robotics-ER 2 (public) used by Phase 14 | 2026-10-01 | Not part of Phase 15. |

**Deprecated/outdated:** `claude-haiku-4-5` extended thinking (not used); GPT-5.x/`gpt-6-sol` legacy; Phase 14 note "depth is out of scope for every backend" no longer applies to the MLLM lane (D-06).

## Runtime State Inventory

Not a rename/refactor phase (greenfield modules plus two additive edits). Omitted per template. One related note: `So-101/So-101.urdf` is read at runtime and the `control/backend_configs` directory gains six JSON files; no stored data, services, or OS registrations carry a renamed string.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Claude/GPT call latency at `xhigh` is 1-5 minutes (10-40K reasoning tokens at tens of tokens/s) | Chunk-Size Derivation, Cost/latency | If much longer, `timeout_s` 900 and N=20 still hold but episodes take longer; if much shorter, N could be smaller. Measure on first live call. |
| A2 | Output token counts (~20K) used for cost estimates | Cost estimates | Budget off by a small multiple; shown only as estimates. |
| A3 | Each HF pinned provider honours `reasoning_effort` and returns usage reasoning tokens / `reasoning` text | HF facts, Pattern 4 | Reasoning-token field `null` and thinking text missing; staged JSON fields remain the trace. Needs live smoke per variant. |
| A4 | HF providers apply the JSON grammar after the thinking block (so reasoning still occurs under `response_format`) | Pitfall 3 | If not, reasoning tokens are 0 on that variant; log and flag for Phase 16. |
| A5 | `gpt-6-astra` accepts `reasoning.summary:"auto"` for this org (no verification gate) | Pattern 4 | A 400 on first call; mitigate with `summary_supported` config flag (static table) rather than retry. |
| A6 | A 2.4 MB, 4-image base64 PNG request is accepted by Cerebras/DeepInfra/Baseten; image `data:` URLs are supported | Image inputs | 413/400; fall back to JPEG q90 via `image_format`. |
| A7 | cv2 wrist capture can return a buffered stale frame after a long idle on macOS | Pitfall 8 | Harmless extra `grab()` calls; if wrong, nothing lost. |
| A8 | Qwen 32-px patch token estimate (~300/image) | Image inputs | Cost/context estimate only. |
| A9 | Real SO-ARM101 gripper: percent-open, 0=closed, 100=open matches the Robonine-equipped twin (72 mm travel change) | Pattern 3 | Wrong gripper direction in prompt; confirm in live UAT before the first full run. |
| A10 | Degree 0 = calibrated midpoint = URDF angle 0 and the sign of each joint matches the URDF axis | Conflict C8, Pattern 3 | URDF text misleads the model (a Phase 16 confound); mitigated by a human sign check and by reporting `include_urdf` ablation. |
| A11 | Highest sensible effort = `xhigh` for Claude/GPT | Conflict C11 | Phase 16 may want `max`; config field, one-line change. |
| A12 | `max_consecutive_failed_calls = 3` and a 15-minute default timeout are sensible | Pattern 5 | Tunable config values. |

## Open Questions

1. **Is the model's `task_complete` semantics "already achieved now" (recommended) or "achieved after this chunk"?**
   - Known: D-18 default executes the tail; D-19 wants claim-vs-outcome analysis later.
   - Unclear: user preference; planner owns the call per CONTEXT discretion.
   - Recommendation: "already achieved in current views", hold the tail, end the episode; document in the prompt template version.
2. **Should the left image slot use `aligned_left` (pixel-aligned with depth) or raw left?**
   - Recommendation: `aligned_left` when depth succeeded, raw left otherwise; log which. Planner may choose raw-left-plus-note for strict cross-backend image identity with Phase 14 logs.
3. **Does the prompt `dt` follow nominal `--control-hz` (D-16) or measured 0.785 s?**
   - Recommendation: follow D-16 literally by default; expose `prompt_dt_source` toggle; always log measured period.
4. **Backend names.** Recommendation: `mllm-claude`, `mllm-gpt`, `mllm-hf-qwen`, `mllm-hf-deepseek`, `mllm-hf-kimi` (config files `mllm_*.json`, shared `mllm_defaults.json`).
5. **Hardware sign/zero check of the URDF convention (C8).** Needs a human with the arm; add to live UAT before the first scored episode.
6. **Whether to run the first live HF smoke with a throw-away prompt to confirm provider pin, grammar+thinking, usage fields.** Strongly recommended as Wave-0/UAT item (cost cents).
7. **Cross-check the REQUIREMENTS/ROADMAP wording** (C9) and DEPTH-CAL-03 note (C5) in the plan's doc-update task.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python venv `control/.venv` | all code/tests | yes | 3.12.12 | — |
| pytest | validation | yes | 8.3.4 (1003 tests pass, 9.4 s) | — |
| opencv-python / numpy / requests | capture, heatmap, depth | yes | 5.0.0.93 / 2.2.6 / 2.34.2 | — |
| `anthropic`, `openai` | Claude, GPT, HF lane | no (not installed) | target 1.11.0 / 3.23.0 | `pip install` (dry-run resolves clean); direct `requests` is the fallback |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `HF_TOKEN` env vars | live calls | NOT SET on this machine at research time | — | Tests use fakes; live UAT needs the user to export them (names are configurable per `api_key_env`) |
| Paid credits (Anthropic, OpenAI, HF) | live UAT | unknown | — | HF free credit ($0.10) will not cover Kimi/DeepSeek runs; user must fund |
| Colab FastFS depth endpoint + `control/stereo_calibration.json` | D-06 depth | calibration file present (untracked); endpoint is a live Colab session (ngrok URL rotates per session) | resident Flask service, protocol 1 | Depth-unavailable path (3 images + prompt note) |
| `~/.cache/huggingface/lerobot/calibration/robots/so_follower/soarm_follower_02.json` | `joint_limits_deg` | yes | — | — |
| Real SO-ARM101 + AR0144 + wrist camera | live UAT | hardware, human-operated | — | none (manual-only verification) |
| Network egress to api.anthropic.com, api.openai.com, router.huggingface.co | live UAT | yes from this machine (docs and router API fetched) | — | — |

**Missing dependencies with no fallback:** live hardware/keys for UAT (manual by nature).
**Missing dependencies with fallback:** SDKs (install), depth endpoint (graceful degrade).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.3.4 (existing; `control/conftest.py` supplies `MockRobot`, `mock_calibration_file`) |
| Config file | none (no pytest.ini/pyproject); run from `control/` |
| Quick run command | `cd control && .venv/bin/python -m pytest test_mllm_schema.py test_mllm_prompt.py test_base_mllm.py -x -q` |
| Full suite command | `cd control && .venv/bin/python -m pytest -x -q` (currently 1003 passed in ~9 s) |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PIPE-02 | New base conforms to `ActionSource` signature; shared prompt/schema identical across backends (golden sha); module discovery registers 5 names, skips `base_*` | unit | `pytest test_backend_registry.py test_base_mllm.py -x` (add `MllmActionSourceBase` to `_action_source_bases()`) | extend existing + new `test_base_mllm.py` (Wave 0) |
| MLLM-04 | Schema property set == `JOINT_ORDER`; units from `ACTION_UNITS`; no min/max/minItems/maxItems; all required; `additionalProperties:false`; key order; deterministic hash; passes the Claude/OpenAI-intersection lint; actions flow through UNMODIFIED `safety_validator` (clamp visible in flags) | unit | `pytest test_mllm_schema.py -x` | Wave 0 |
| MLLM-05 | `chunk_len` present/int>=1 in defaults, override works, prompt states N and seconds; derivation notes present | unit | `pytest test_mllm_prompt.py -k chunk -x` | Wave 0 |
| MLLM-06 | After a fake call: `last_details["mllm_call"]` has restatement/plan/claims/task_complete; every later tick has `chunk_ref` not a trace copy; through real `run_episode` + `IOLogger` the `episode.jsonl` lines parse and contain them | integration (MockRobot, tmp_path) | `pytest test_base_mllm.py -k trace -x` | Wave 0 |
| MLLM-07 | Malformed variants (missing key, string, null, bool, NaN/Infinity, short chunk, extra steps, invalid JSON, empty, refusal, truncated, HTTP error, timeout): earlier steps real, bad step + rest are `dict(joint_state)` never zeros, tag `...-holding-position`, no exception, provider called exactly once, no per-tick calls | unit/integration | `pytest test_base_mllm.py -k "malformed or hold or failure" -x` | Wave 0 |
| MLLM-01/02/03 | Per-provider request construction: Claude has no temperature/top_p/top_k, `output_config.effort/format`, `thinking.display`; GPT has no temperature, `reasoning.effort`, `text.format strict`, `store False`; HF has pinned `model:provider`, `response_format`, sampling passed; images labelled, 4 inputs; usage normalizers on canned payloads | unit (fake SDK clients, `httpx.MockTransport` where feasible) | `pytest test_backend_mllm_claude.py test_backend_mllm_gpt.py test_backend_mllm_hf.py -x` | Wave 0 |
| D-18/D-19 | `task_complete` ends episode (`termination.json` reason `task_complete`), tail held, claim logged | integration | `pytest test_run_vla_episode.py -k task_complete -x` | extend existing |
| D-20..D-22 | requested-vs-applied settings, omitted-with-reason, usage/cost math, prompt/schema hashes, sdk version, sidecar files exist with redacted raw payload + image sha256, secret absent from `last_details`, sidecars and `episode.jsonl` | unit/integration | `pytest test_mllm_record.py test_base_mllm.py -k "record or redact or sidecar" -x` | Wave 0 |
| D-06/C5 | depth fetched once per call; raw 800x600 frames to depth client; heatmap/grid built; failure/429 path non-fatal; CLI rule relaxed for MLLM backends and still enforced for gRPC backends | unit + CLI | `pytest test_base_mllm.py -k depth -x`, `pytest test_run_vla_episode.py -k depth -x` | extend existing + Wave 0 |
| D-00b | missing/blank env var -> `SystemExit` before robot/camera constructed; `api_key_env` must be an uppercase NAME; key never echoed | unit | `pytest test_base_mllm.py -k key -x` | Wave 0 |
| Prompt hygiene | forbidden words absent (reasoning/thinking/trace/scratchpad); no primitive vocabulary; numeric caps absent unless toggle; `include_urdf` off changes hash and removes URDF block; URDF text parse matches `So-101.urdf` joint limits | unit | `pytest test_mllm_prompt.py -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** the touched test file(s), `-x -q` (seconds).
- **Per wave merge:** `cd control && .venv/bin/python -m pytest -x -q`.
- **Phase gate:** full suite green before `/gsd-verify-work`, then the manual live checks below.

### Needs live keys / hardware (manual-only, record in UAT)
1. One live call per provider with a tiny chunk: schema honoured with reasoning ON, `usage` reasoning-token field present (Claude `thinking_tokens`, OpenAI `reasoning_tokens`, each HF variant: present or `null`), `reasoning` text field presence, image acceptance, pinned-provider routing, no `reasoning_extraction` refusal; measure real API latency and update the chunk-size table.
2. Depth live: FastFS endpoint up, per-chunk depth ~6 s, heatmap legible, no 429 collision.
3. Live hardware episode per backend (D-05 "independently verifiable"): at least one real executed chunk, hold path observed or provoked with a deliberately bad fake, Ctrl-C during a long call returns to start.
4. Human sign/zero check of URDF vs real joints (C8) and gripper direction.

### Wave 0 Gaps
- [ ] `control/test_mllm_schema.py`, `test_mllm_prompt.py`, `test_mllm_record.py`, `test_base_mllm.py`, `test_backend_mllm_claude.py`, `test_backend_mllm_gpt.py`, `test_backend_mllm_hf.py` (covers all rows above)
- [ ] Fake provider fixtures in `conftest.py` or a `fake_providers.py` helper (fake Anthropic/OpenAI clients, canned payloads incl. refusal/truncated/NaN, fake `DepthCameraClient`, fake wrist/stereo)
- [ ] Extend `test_backend_registry._action_source_bases()` and `test_run_vla_episode.py` (termination hook, depth-flag rule)
- [ ] `pip install anthropic==1.11.0 openai==3.23.0` into `control/.venv` behind the human-verify checkpoint; add pins and provenance comments to `control/requirements.txt`

## Security Domain

### Applicable ASVS Categories (level 1, `security_enforcement` enabled)
| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no (no end-user auth) | — |
| V3 Session Management | no | — |
| V4 Access Control | minimal | API keys scoped by provider; Gemini precedent requires restricted keys; configs hold env-var NAMES only |
| V5 Input Validation | yes (model output is untrusted input; config files; sidecar paths) | strict per-step parse + finite check, key whitelist from `JOINT_ORDER`, unmodified `safety_validator`, config regex for names (`_CONFIG_NAME_RE`), integer-only `chunk_id` paths, size cap on raw payload/trace strings before logging |
| V6 Cryptography | no hand-rolled crypto | TLS via SDKs; nothing custom |
| V7 Error Handling and Logging | yes | `redact()` on every record/sidecar/error string; exceptions copied into `last_details` only after redaction and truncation (`base_gemini` precedent) |
| V8 Data Protection | yes | images of the lab leave the machine: Anthropic Fable 5.1 30-day retention (not ZDR), OpenAI `store:false`, HF provider policies vary; disclose to the operator; sidecars stay in git-ignored `control/outputs/` |
| V10/V12 Malicious code/files | yes | model output is data only: never `eval`/`exec`, no tool-use/function-calling, no file paths taken from model output |
| V14 Configuration | yes | pinned dependency versions + human-verify checkpoint; no secrets in JSON configs |

### Known Threat Patterns for this stack
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| API key leakage via logs, exception text, `last_details`, sidecars, `repr()` | Information disclosure | env-var-only keys; `__repr__` omits key; central `redact()`; test greps all produced artifacts for the secret |
| Model emits out-of-range/NaN/huge values (unsafe motion) | Tampering / safety | pre-validation (hold on malformed) + unmodified `safety_validator` hard limits (joint envelope, per-step, velocity) + e-stop; hard limits never loosened |
| Prompt injection from the scene (text in camera view, instruction text in depth/URDF) | Tampering | model can only emit schema-bound JSON; no tools; validator bounds actions; scene claims are logged for audit, not executed |
| Runaway spend loop (failure -> per-tick calls; huge `max_tokens`) | Denial of service (economic) | failure = N holds, consecutive-failure cap, config-pinned `max_tokens`, per-call cost estimate in the record |
| Pasted key in `backend_configs/*.json` | Information disclosure | `api_key_env` validated as uppercase env-var NAME; value never echoed (copy `load_er_backend_config` logic) |
| Path traversal / file clobber in sidecar writes | Tampering | sidecar root = `args.out / "mllm_calls"`, integer chunk ids, fixed filenames, never model-supplied names |
| Supply-chain (fresh SDK releases, transitive `httpx2`) | Tampering | exact pins, `checkpoint:human-verify`, install only into `control/.venv` |
| Third-party data exposure (images to providers) | Information disclosure | document retention per provider, `store:false`, operator awareness before live runs |

## Sources

### Primary (HIGH confidence)
- platform.claude.com/docs/en/models/overview and /models/fable-5-1/overview, /whats-new-fable-5-1 (model ids, pricing, sampling 400, forced tool use, retention) — fetched 2026-10-02
- platform.claude.com/docs/en/build-with-claude/structured-outputs.md, effort, thinking, thinking-steering-and-cost.md, vision.md, refusals-and-fallback.md, api/errors.md (raw markdown via curl)
- developers.openai.com/api/docs/models.md, /models/gpt-6-astra.md, /guides/latest-model.md, /guides/reasoning.md, /guides/structured-outputs.md, /guides/images-vision (raw markdown)
- Installed SDK source inspected in a scratch venv: anthropic 1.11.0 (`OutputConfigParam`, `ThinkingConfigAdaptiveParam`, `Usage`, `OutputTokensDetails.thinking_tokens`, `RefusalStopDetails`, `messages.create/stream` signatures), openai 3.23.0 (`ResponseUsage`, `Reasoning`, `responses.create/stream`, `chat.completions` signature)
- https://router.huggingface.co/v1/models (134 models, per-provider price/structured-output/throughput/context) and https://huggingface.co/api/models/<id>?expand[]=inferenceProviderMapping — fetched 2026-10-02
- huggingface.co/docs/inference-providers/{index,pricing,guides/structured-output,tasks/chat-completion}.md
- Model cards: Qwen/Qwen3.8-27B, deepseek-ai/DeepSeek-V4-Flash-Vision-Exp, deepseek-ai/DeepSeek-V4.1-Flash, moonshotai/Kimi-K3 (raw README)
- Repo (read in full): `control/run_vla_episode.py`, `vla_bridge/{backends/__init__,base_gemini,base_http_adapter,action_contract,safety_validator,io_logger,depth_camera,depth_recorder,fastfs_server,stereo_camera,robot_client}.py`, `backend_configs/*.json`, `conftest.py`, `test_backend_registry.py`, `test_base_gemini.py`; Phase 12 `12-UAT.md`, `control/outputs/vla_episode_004` and `vla_episode_20260930_121123` (`episode.jsonl`, `depth.jsonl`), Phase 14 `14-VALIDATION.md`/`14-PATTERNS.md`
- `gsd-tools package-legitimacy check`, PyPI JSON (versions, upload dates, requires_dist), `pip install --dry-run` against `control/.venv`

### Secondary (MEDIUM confidence)
- WebSearch: vLLM structured outputs with reasoning parsers (supports reasoning + structured output; applicability to each HF provider unverified)
- WebSearch snippets on OpenAI strict-mode keyword limits (superseded by the official page, which lists `minItems/maxItems/minimum/maximum` as supported for non-fine-tuned models)

### Tertiary (LOW confidence)
- Latency/cost token-count assumptions (A1, A2), HF per-provider image limits (A6), macOS capture buffering (A7)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH, SDKs inspected, versions pinned and dry-run resolved; legitimacy flags are heuristic.
- Provider facts (Claude/OpenAI): HIGH, official docs fetched today plus SDK types.
- HF lane: MEDIUM, router metadata live and exact, behaviour (thinking under grammar, usage fields, image limits) unverified without a token.
- Architecture/integration: HIGH, derived from reading the actual modules and tests; two small `run_vla_episode.py` edits identified.
- Chunk sizing: MEDIUM, measured Phase 12 cadence/depth data are solid; MLLM API latency is an explicit gap.
- Failed-experiment root cause: LOW for causation (no surviving outputs), HIGH for the documented confounds.
- Pitfalls: HIGH for items traced to docs/code, MEDIUM for A-tagged items.

**Research date:** 2026-10-02
**Valid until:** ~2026-10-09 for HF router prices/throughput/provider status and SDK versions (fast-moving; fetched the same day as release of anthropic 1.10/1.11 and openai 3.23); ~30 days for the architecture and repo-derived findings.
