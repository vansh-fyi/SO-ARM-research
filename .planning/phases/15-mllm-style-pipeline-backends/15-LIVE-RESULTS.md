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

Image retention NOTICE accepted by the operator before live runs (T-15-38): PENDING

## 1. Provider smoke results

Command per backend (cheapest first, table order):
`cd control && .venv/bin/python -m vla_bridge.mllm_smoke --backend NAME --chunk-len 3 --out outputs/smoke_NAME --confirm-spend`

| backend | date | smoke exit code | schema honoured with reasoning on | reasoning-token field (value or null) | native thinking text kind | image acceptance (image_format used) | pinned provider routing evidence (model_returned / request id) | refusal category | API latency (s) | estimated cost (USD) | notes |
|---------|------|-----------------|-----------------------------------|---------------------------------------|---------------------------|--------------------------------------|-----------------------------------------------------------------|------------------|-----------------|----------------------|-------|
| mllm-hf-qwen | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| mllm-hf-deepseek | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| mllm-gpt | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| mllm-claude | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| mllm-hf-kimi | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

Key-absence check of sidecars (response_raw.json, thinking.txt, request_meta.json, prompt.txt per smoke run): PENDING

## 2. Config flag flips

Only flips demanded by section 1 evidence (GPT `capabilities.summary_supported` per A5; HF
`structured_mode` json_schema -> json_object per research C2; HF `image_format` jpeg per A6).
Effort, max_tokens, prices and model ids are never flipped here.

| file | flag | old value | new value | evidence |
|------|------|-----------|-----------|----------|
| PENDING | PENDING | PENDING | PENDING | PENDING |

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
