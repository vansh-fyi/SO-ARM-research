# Phase 15: Live Verification Results

Live-only evidence for Plan 15-10 (VALIDATION.md manual-only items; ROADMAP success criteria 2-5).
Every cell starts as PENDING. A backend the user chooses not to fund is marked
"deferred by user (YYYY-MM-DD)", never left PENDING.

Rules for this file:

- API keys are exported in the operator's shell only. No key, token or key-shaped string is ever
  written here, in a config, or in any committed file.
- Config flags are flipped only from evidence recorded in section 2, never by an automatic retry.
- chunk_len is not changed without a user decision recorded in section 6.

Blocking findings (gripper direction mismatch or other stop-the-plan findings): PENDING

Image retention NOTICE accepted by the operator before live runs (T-15-38): ACCEPTED 2026-10-02 ("retention accepted"), covering real lab images sent to Anthropic, OpenAI and the HF-routed providers in Task 3.

## 1. Provider smoke results

Command per backend (cheapest first, table order):
`cd control && .venv/bin/python -m vla_bridge.mllm_smoke --backend NAME --chunk-len 3 --out outputs/smoke_NAME --confirm-spend`

| backend | date | smoke exit code | schema honoured with reasoning on | reasoning-token field (value or null) | native thinking text kind | image acceptance (image_format used) | pinned provider routing evidence (model_returned / request id) | refusal category | API latency (s) | estimated cost (USD) | notes |
|---------|------|-----------------|-----------------------------------|---------------------------------------|---------------------------|--------------------------------------|-----------------------------------------------------------------|------------------|-----------------|----------------------|-------|
| mllm-hf-qwen | 2026-10-02 | 1 (attempt 1) | not reached: no model output | null (no usage) | none | not reached | request pinned `Qwen/Qwen3.8-27B:cerebras`; router answered before routing | none | 0.6 | n/a (no tokens) | ATTEMPT 1 FAIL, provider-side: HTTP 503 `Provider 'cerebras' is currently failing for model 'Qwen/Qwen3.8-27B' and has been paused by the circuit breaker`. Hold path behaved (3 holds, no exception). RETRY PENDING; pin not changed (cerebras chosen for json_schema support). |
| mllm-hf-deepseek | 2026-10-02 | 1 (attempt 1) | not reached: no model output | null (no usage) | none | not reached | request pinned `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp:deepinfra`; no response body | none | 120.9 | n/a (no tokens) | ATTEMPT 1 FAIL: HTTP 504 Gateway Time-out (HTML body) from the HF router after 120.9 s, far below the configured 900 s timeout_s. Suggests a router-side ~120 s gateway ceiling that an effort `max` call can exceed. Hold path behaved. RETRY PENDING; effort is not changed in Task 2 (plan rule) — if it repeats, this becomes a user decision (see section 6). |
| mllm-gpt | 2026-10-02 | 0 | yes, 3/3 valid steps, stop complete | 759 | reasoning summary (`summary: auto` accepted, A5 confirmed); thinking.txt 1571 bytes of summary text | accepted, 4 images, png (default) | model_returned `gpt-6-astra`; request id `req_1303b397a747471cae2ae9dabb62324d` (from response header) | none | 32.1 | 0.1241 | settings applied effort xhigh, max_output_tokens 64000, store false, summary auto; temperature/top_p omitted (required by OpenAI with reasoning effort). |
| mllm-claude | 2026-10-02 | 0 | yes, 3/3 valid steps, stop complete | 3109 | summarized thinking (adaptive, display summarized); thinking.txt 3288 bytes; no reasoning_extraction refusal | accepted, 4 images, png (default) | model_returned `claude-fable-5-1`; request id `req_011CfdnjAwTg7mZFppApZ7vd` (from `request-id` header) | none | 51.5 | 0.2734 | First attempt (same day) failed HTTP 400 before inference: the operator's key was not workspace-scoped (`anthropic-workspace-id` required). Operator switched to a workspace-scoped key; no code change. The model noted in its thinking that no pen is visible in the synthetic scene. |
| mllm-hf-kimi | 2026-10-02 | 0 | yes, 3/3 valid steps, json_schema with reasoning on (A3/A4 confirmed for baseten), stop complete | 262 (`completion_tokens_details.reasoning_tokens`) | full raw reasoning text (thinking.txt 1030 bytes) | accepted, 4 images, png (A6 confirmed) | request pinned `moonshotai/Kimi-K3:baseten`; model_returned `kimi-k3-disagg` (provider deployment name, not the HF id); request id `Root=1-6abfe722-716e584607faddaa2d65e88b` | none | 8.1 | 0.0300 | usage 6879 prompt (64 cached) / 623 completion tokens; far cheaper than the ~$0.35 research estimate for a 3-step chunk. |

Key-absence check of sidecars (response_raw.json, thinking.txt, request_meta.json, prompt.txt per smoke run): PASS 2026-10-02. Regex scan for `sk-ant-…`, `sk-…`/`sk-proj-…` and `hf_…` key shapes over every `control/outputs/smoke_*` file and `smoke_all.log` found no match.

## 2. Config flag flips

Only flips demanded by section 1 evidence (GPT `capabilities.summary_supported` per A5; HF
`structured_mode` json_schema -> json_object per research C2; HF `image_format` jpeg per A6).
Effort, max_tokens, prices and model ids are never flipped here.

| file | flag | old value | new value | evidence |
|------|------|-----------|-----------|----------|
| none so far (2026-10-02) | — | — | — | GPT accepted `reasoning.summary` (A5 holds, `summary_supported` stays true); Kimi honoured json_schema and the 4-image png request (no C2/A6 flip). Qwen and DeepSeek not yet evaluated (provider-side failures, section 1). |

## 3. Depth live

| date | endpoint session (no secrets) | left image source | depth time (s) | heatmap legible (legend, black where invalid) | 24x32 grid plausible (n/a cells) | HTTP 429 observed | notes |
|------|-------------------------------|-------------------|----------------|-----------------------------------------------|----------------------------------|-------------------|-------|
| PENDING | PENDING | raw | PENDING | PENDING | PENDING | PENDING | PENDING |
| PENDING | PENDING | aligned | PENDING | PENDING | PENDING | PENDING | PENDING |

## 4. Human URDF sign and gripper-direction check

Small positive command (about +10 degrees via control/joint_jog.py) from a safe pose; compare the
physical direction with the URDF axis and the right-hand rule. Record only: a joint mismatch is a
Phase 16 analysis note, not a code change. A gripper-direction mismatch is a blocking finding.

| joint | date | command | URDF axis | expected direction | observed direction | match / mismatch | notes |
|-------|------|---------|-----------|--------------------|--------------------|------------------|-------|
| shoulder_pan | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| shoulder_lift | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| elbow_flex | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| wrist_flex | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| wrist_roll | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| gripper | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

## 5. Hardware episodes

| backend | date | out directory | chunks executed | hold path (natural / provoked / not seen) | Ctrl-C result | termination reason | measured per-chunk API latency (s) | prev_chunk_tick_period_s vs 0.785 | task_complete claim and match with scene | notes |
|---------|------|---------------|-----------------|-------------------------------------------|---------------|--------------------|------------------------------------|-----------------------------------|------------------------------------------|-------|
| mllm-hf-qwen | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| mllm-hf-deepseek | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| mllm-gpt | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| mllm-claude | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| mllm-hf-kimi | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

Provoked hold-path run (temporary max_tokens 1024, restored with git checkout afterwards): PENDING

## 6. Chunk-size re-derivation record

Duty cycle = (chunk_len x step_dt_s) / (chunk_len x step_dt_s + 6.15 + api_seconds), chunk_len 20,
step_dt_s 0.785. Compare against `mllm_defaults.json` `chunk_len_derivation`.

| backend | measured API latency (s) | source (smoke / episode) | duty cycle at chunk_len 20 | exceeds timeout_s | below 10 percent duty cycle | recommended alternative N (if any) | user decision |
|---------|--------------------------|--------------------------|----------------------------|-------------------|-----------------------------|------------------------------------|---------------|
| mllm-hf-qwen | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| mllm-hf-deepseek | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| mllm-gpt | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| mllm-claude | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| mllm-hf-kimi | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
