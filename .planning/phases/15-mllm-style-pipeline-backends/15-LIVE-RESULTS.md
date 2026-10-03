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
| mllm-hf-qwen | 2026-10-02 | 1 (attempts 1 and 2) | not reached: no model output | null (no usage) | none | not reached | request pinned `Qwen/Qwen3.8-27B:cerebras`; router answered before routing | none | 0.6 | n/a (no tokens) | ATTEMPTS 1 AND 2 FAIL, provider-side: HTTP 503 both times, `Provider 'cerebras' is currently failing for model 'Qwen/Qwen3.8-27B' and has been paused by the circuit breaker`. Hold path behaved (3 holds, no exception). The router listing fetched 2026-10-02 (router.huggingface.co/v1/models) lists novita, featherless-ai, ovhcloud and deepinfra for this model, and no cerebras. RE-PINNED to `Qwen/Qwen3.8-27B:deepinfra` by user decision 2026-10-02 (section 2); streaming fix applied to the shared HF adapter; RE-SMOKED 2026-10-02: PASS (see attempt 3 row below). |
| mllm-hf-qwen (attempt 3, re-pinned + streaming) | 2026-10-02 | 0 | yes, 3/3 valid steps, json_schema with reasoning on (A3/A4 confirmed for deepinfra), stop complete | null: deepinfra reports no `completion_tokens_details` (9949 completion tokens total, reasoning not broken out) | full raw reasoning text (thinking.txt 25665 bytes) | accepted, 4 images, png (A6 confirmed) | request pinned `Qwen/Qwen3.8-27B:deepinfra`; model_returned `Qwen/Qwen3.8-27B`; request id `RkzvBIp3DhFLBgKBCosH0HGP` | none | 276.5 | 0.0263 (router's own usage.estimated_cost 0.0197) | PASS. Very long reasoning at effort xhigh: 9949 output tokens in 276.5 s (~36 tok/s). Latency far above the other backends; see section 6 duty cycle. Output dir outputs/smoke_mllm-hf-qwen_stream. |
| mllm-hf-deepseek | 2026-10-02 | 1 (attempts 1 and 2) | not reached: no model output | null (no usage) | none | not reached | request pinned `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp:deepinfra`; no response body | none | 120.9 / 121.2 | n/a (no tokens) | ATTEMPTS 1 AND 2 FAIL: HTTP 504 Gateway Time-out (HTML body) from the HF router after 120.9 s and 121.2 s, far below the configured 900 s timeout_s. Root cause: `base_mllm_chat.py` sent a NON-streaming request, so a reasoning model sent no bytes until done and the router's edge proxy cut the idle connection at ~120 s (Claude and GPT already stream; Kimi passed only because it answered in 8.1 s). Hold path behaved. Separately, DeepSeek has retired V4-Flash-Vision-Exp upstream (aliases route to V4.1-Flash; api-docs.deepseek.com/news/news260910/). SWITCHED to `deepseek-ai/DeepSeek-V4.1-Flash:deepinfra` by user decision 2026-10-02 (section 2; effort `max` unchanged); streaming fix applied; RE-SMOKED 2026-10-02: PASS (see attempt 3 row below). |
| mllm-hf-deepseek (attempt 3, V4.1-Flash + streaming) | 2026-10-02 | 0 | yes, 3/3 valid steps, json_schema with reasoning on (A3/A4 confirmed for deepinfra), stop complete | 781 | full raw reasoning text (thinking.txt 2914 bytes) | accepted, 4 images, png (A6 confirmed) | request pinned `deepseek-ai/DeepSeek-V4.1-Flash:deepinfra`; model_returned `deepseek-ai/DeepSeek-V4.1-Flash`; request id `Roo5QRSWeWhbR9k6iOAwSoB6` | none | 22.4 | 0.0019 (router's own usage.estimated_cost 0.0014) | PASS. String effort `max` accepted by deepinfra (no error). Streaming removed the 504. Output dir outputs/smoke_mllm-hf-deepseek_stream. |
| mllm-gpt | 2026-10-02 | 0 | yes, 3/3 valid steps, stop complete | 759 | reasoning summary (`summary: auto` accepted, A5 confirmed); thinking.txt 1571 bytes of summary text | accepted, 4 images, png (default) | model_returned `gpt-6-astra`; request id `req_1303b397a747471cae2ae9dabb62324d` (from response header) | none | 32.1 | 0.1241 | settings applied effort xhigh, max_output_tokens 64000, store false, summary auto; temperature/top_p omitted (required by OpenAI with reasoning effort). |
| mllm-claude | 2026-10-02 | 0 | yes, 3/3 valid steps, stop complete | 3109 | summarized thinking (adaptive, display summarized); thinking.txt 3288 bytes; no reasoning_extraction refusal | accepted, 4 images, png (default) | model_returned `claude-fable-5-1`; request id `req_011CfdnjAwTg7mZFppApZ7vd` (from `request-id` header) | none | 51.5 | 0.2734 | First attempt (same day) failed HTTP 400 before inference: the operator's key was not workspace-scoped (`anthropic-workspace-id` required). Operator switched to a workspace-scoped key; no code change. The model noted in its thinking that no pen is visible in the synthetic scene. Historical claude-fable-5-1 evidence: model switched to claude-opus-5-5 by user decision 2026-10-02 (section 2); RE-SMOKED 2026-10-02: PASS (see Opus 5.5 row below). |
| mllm-claude (claude-opus-5-5 re-smoke) | 2026-10-02 | 0 | yes, 3/3 valid steps, stop complete | 325 | summarized thinking (adaptive, display summarized); no reasoning_extraction refusal | accepted, 4 images, png (default) | model_returned `claude-opus-5-5`; request id `req_011CfdqiG37e4j4uagv44ZD7` (from `request-id` header) | none | 12.7 | 0.0539 | PASS after the user-decided switch from claude-fable-5-1. Effort xhigh, max_tokens 64000; sampling params not sent. Output dir outputs/smoke_mllm-claude_opus55. Key scan of this dir: clean. |
| mllm-hf-kimi | 2026-10-02 | 0 | yes, 3/3 valid steps, json_schema with reasoning on (A3/A4 confirmed for baseten), stop complete | 262 (`completion_tokens_details.reasoning_tokens`) | full raw reasoning text (thinking.txt 1030 bytes) | accepted, 4 images, png (A6 confirmed) | request pinned `moonshotai/Kimi-K3:baseten`; model_returned `kimi-k3-disagg` (provider deployment name, not the HF id); request id `Root=1-6abfe722-716e584607faddaa2d65e88b` | none | 8.1 | 0.0300 | usage 6879 prompt (64 cached) / 623 completion tokens; far cheaper than the ~$0.35 research estimate for a 3-step chunk. |

Key-absence check of sidecars (response_raw.json, thinking.txt, request_meta.json, prompt.txt per smoke run): PASS 2026-10-02. Regex scan for `sk-ant-…`, `sk-…`/`sk-proj-…` and `hf_…` key shapes over every `control/outputs/smoke_*` file and `smoke_all.log` found no match. Re-run 2026-10-02 after the streaming re-smokes (`smoke_*_retry`, `smoke_*_stream`): clean.

## 2. Config flag flips

Only flips demanded by section 1 evidence (GPT `capabilities.summary_supported` per A5; HF
`structured_mode` json_schema -> json_object per research C2; HF `image_format` jpeg per A6).
Effort, max_tokens, prices and model ids are never flipped here from evidence alone; the rows
marked "user decision" record explicit user decisions (2026-10-02) driven by section 1 evidence.

| file | flag | old value | new value | evidence |
|------|------|-----------|-----------|----------|
| none so far (2026-10-02) | — | — | — | GPT accepted `reasoning.summary` (A5 holds, `summary_supported` stays true); Kimi honoured json_schema and the 4-image png request (no C2/A6 flip). Qwen and DeepSeek first attempts failed provider-side (section 1); their 2026-10-02 attempt-3 re-smokes (deepinfra, streaming) honoured json_schema and the 4-image png request, so no C2/A6 flip for any HF variant. Task 2 (2026-10-02) applied no evidence-driven flag flip. |
| control/vla_bridge/backend_configs/mllm_hf_qwen.json | `hf_provider`, `prices`, `context_window` (user decision 2026-10-02) | `cerebras`; $0.99 / $1.49 per M; 65536 | `deepinfra` (request model `Qwen/Qwen3.8-27B:deepinfra`); $0.20 / $2.50 per M, verified_on 2026-10-02, source router.huggingface.co/v1/models; 262144 | Section 1 row mllm-hf-qwen: HTTP 503 circuit breaker on cerebras twice; the 2026-10-02 router listing has no cerebras provider for this model, and deepinfra is live with supports_structured_output true and context 262144. |
| control/vla_bridge/backend_configs/mllm_hf_deepseek.json | `model`, `prices`, `context_window` (user decision 2026-10-02; supersedes the locked D-23 model id and D-27) | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`; $0.44 / $1.32 per M; not set | `deepseek-ai/DeepSeek-V4.1-Flash` (request model `deepseek-ai/DeepSeek-V4.1-Flash:deepinfra`; backend name and effort `max` unchanged); $0.20 / $0.60 per M, verified_on 2026-10-02, source router.huggingface.co/v1/models; 1048576 | Section 1 row mllm-hf-deepseek: HTTP 504 at 120.9 s and 121.2 s; DeepSeek retired V4-Flash-Vision-Exp upstream (aliases route to V4.1-Flash; api-docs.deepseek.com/news/news260910/, api-docs.deepseek.com/updates/); the router lists V4.1-Flash with text+image input, deepinfra live with supports_structured_output true and context 1048576. |
| control/vla_bridge/backend_configs/mllm_claude.json | `model` (user decision 2026-10-02; backend name `mllm-claude`, effort `xhigh`, max_tokens 64000 and adaptive thinking with summarized display unchanged) | `claude-fable-5-1` | `claude-opus-5-5` | user decision 2026-10-02; prices 10/50/0.25 → 4/20/0.20 from the overview URL (https://platform.claude.com/docs/en/models/opus-5-5/overview, fetched 2026-10-02; request rules from https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5). Effort stays xhigh explicitly because the Opus 5.5 default is medium. The section 1 mllm-claude smoke is Fable 5.1 evidence only; an Opus 5.5 re-smoke is required. |
| control/vla_bridge/backends/base_mllm_chat.py | request mode (user decision 2026-10-02, Rule 1 bug fix) | non-streaming `chat.completions.with_raw_response.create(...)`; raw payload = verbatim wire body | `chat.completions.create(..., stream=True, stream_options={"include_usage": true})`; content/reasoning/refusal deltas assembled, finish_reason from the last choice chunk, usage from the usage chunk, request id from the stream's `x-request-id` header; raw payload = JSON-safe reconstruction (message, ids, model, finish_reason, usage, chunk counts); max_retries 0 and timeout_s unchanged | Section 1 row mllm-hf-deepseek: the router's edge proxy cut the idle non-streaming connection at ~120 s (HTTP 504) although timeout_s is 900; section 1 row mllm-hf-kimi passed only because it answered in 8.1 s. |

## 3. Depth live

| date | endpoint session (no secrets) | left image source | depth time (s) | heatmap legible (legend, black where invalid) | 24x32 grid plausible (n/a cells) | HTTP 429 observed | notes |
|------|-------------------------------|-------------------|----------------|-----------------------------------------------|----------------------------------|-------------------|-------|
| 2026-10-03 | Colab resident FastFS (ngrok HTTP tunnel, 23-36-37 checkpoint sha256 af0658f2...), /health ready, protocol_version 1 | raw | 7.1 (fetch 7116 ms, server inference 196 ms, 1 attempt) | yes: legend 0.20-1.50 m and n/a swatch legible; no invalid pixels in this scene (valid_fraction 1.0), so no black regions to show | yes: 24x32, min 0.50 / median 0.82 / max 1.41 m; bowl ~0.52-0.60 m (bottom rows), gripper/arm ~0.66-0.80 m, back wall ~1.1-1.4 m; no n/a cells (100% valid) | no | out: control/outputs/mllm_inputs_preview; wrist camera is cv2 index 1 ('USB Camera'; index 0 is the built-in FaceTime camera), stereo 'CCB Camera'; status ok |
| 2026-10-03 | same session | aligned | 6.3 (fetch 6349 ms, 1 attempt) | yes (same legend) | yes, same ranges, 100% valid | no | out: control/outputs/mllm_inputs_preview_aligned; left image sent was the rectified aligned frame (slightly zoomed and cropped compared with raw, same scene); only one depth caller, so no 429 |

## 4. Human URDF sign and gripper-direction check

Small positive command (about +10 degrees via control/joint_jog.py) from a safe pose; compare the
physical direction with the URDF axis and the right-hand rule. Record only: a joint mismatch is a
Phase 16 analysis note, not a code change. A gripper-direction mismatch is a blocking finding.

| joint | date | command | URDF axis | expected direction | observed direction | match / mismatch | notes |
|-------|------|---------|-----------|--------------------|--------------------|------------------|-------|
| shoulder_pan | 2026-10-03 | joint_jog +10 (0.35 -> 9.32) | 0 0 1 in shoulder frame = -z (down) in base | clockwise viewed from above; eef swings to the robot's right (operator's left), FK eef dx -56 mm | clockwise from above, gripper to operator's left / robot's right | match | from folded rest pose |
| shoulder_lift | 2026-10-03 | joint_jog +10 (-100.88 -> -92.88) | 0 0 1 in link frame = +x (robot's left) in base | CCW viewed from the robot's left: a forward-pointing link pitches down/forward, a backward-folded link swings up toward the front (FK eef -54 mm z at zero pose) | upper arm moved up out of the fold, toward the bowl side | match | rest pose is folded back near -101 deg |
| elbow_flex | 2026-10-03 | joint_jog +10 (92.31 -> 95.21, near the +97.23 limit, not visible), then -10 (94.33 -> 87.30) | +x (robot's left) in base, parallel to shoulder_lift | +10 folds forearm further in (CCW from robot's left); -10 unfolds | -10 run: forearm opened up / unfolded | match (inferred from the -10 run) | +10 from rest is limit-bound (only +2.9 deg travel) |
| wrist_flex | 2026-10-03 | joint_jog +10 (66.77 -> 74.42, not clearly seen), repeat +10 (18.86 -> 28.70) | +x (robot's left) in base, parallel to shoulder_lift | CCW viewed from the robot's left (same sense as shoulder_lift +); gripper tip dips down when pointing forward (FK -24 mm) | CCW; wrist-gripper moved downward | match | between runs the wrist sagged 74.42 -> 18.86 under gravity: joint_jog.py disconnects with disable_torque_on_disconnect=True (lerobot default), so the arm goes limp after each jog. Tool behaviour, not a fault |
| wrist_roll | 2026-10-03 | joint_jog +10 (-7.43 -> 2.15) | 0 0 1 in wrist frame = along the gripper, pointing from the jaw tips back toward the wrist | CCW viewed from the wrist end looking toward the jaw tips | CCW (operator first said clockwise, then corrected to CCW after the viewpoint was clarified) | match | the viewpoint convention matters: CW from the tip end = CCW from the wrist end |
| gripper | 2026-10-03 | absolute targets 0 then 100 (lerobot send_action, max_relative_target 100.0); first try joint_jog -100 from 87.57 stopped at 45.81 (out-of-range target, 1 s readback) | gripper_left/right prismatic 0 to 0.036 m | prompt: 0 = fully closed, 100 = fully open | target 0: 41.24 -> 0.25 in ~1.5 s, jaws almost fully closed; target 100: 0.22 -> 99.69 in ~2.5 s, jaws fully open | match (no blocking finding) | prompt text in mllm_prompt.py is correct as is |

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
| mllm-hf-qwen | 276.5 | smoke (2026-10-02, attempt 3, deepinfra, 3-step chunk, synthetic images) | 5.3% | no (276.5 s < 900 s) | YES | 41 (smallest N with duty cycle >= 10 percent at 276.5 s: 41 x 0.785 = 32.2 s open loop, 10.2 percent; N 40 gives 9.998 percent). Breaks the N <= 20 open-loop horizon bound and, if applied to all backends, the shared fixed N (D-17). Alternatives that keep N = 20: accept the low Qwen duty cycle (Qwen is a long thinker at xhigh: 9949 output tokens at ~36 tok/s), or treat a lower Qwen effort as a Phase 16 ablation. | Option 1, keep 20 for all backends (user decision 2026-10-03). The low duty cycle of mllm-hf-qwen is accepted as a known characteristic. chunk_len unchanged at 20 |
| mllm-hf-deepseek | 22.4 | smoke (2026-10-02, attempt 3, V4.1-Flash on deepinfra, 3-step chunk, synthetic images) | 35.5% | no (22.4 s < 900 s) | no | none | not needed; chunk_len unchanged at 20 |
| mllm-gpt | 32.1 | smoke (2026-10-02, 3-step chunk, synthetic images) | 29.1% | no (32.1 s < 900 s) | no | none | not needed; chunk_len unchanged at 20 |
| mllm-claude | 12.7 | smoke (2026-10-02, claude-opus-5-5 re-smoke after the user-decided switch, 3-step chunk, synthetic images; superseded Fable 5.1 smoke was 51.5 s / 21.4%) | 45.4% | no (12.7 s < 900 s) | no | none | not needed; chunk_len unchanged at 20 |
| mllm-hf-kimi | 8.1 | smoke (2026-10-02, baseten, 3-step chunk, synthetic images) | 52.4% | no (8.1 s < 900 s) | no | none | not needed; chunk_len unchanged at 20 |

Finding (Plan 15-10 Task 2, 2026-10-02): all five values are 3-step smoke latencies with synthetic
images; chunk_len 20 episode latencies from latency_ms.api replace or extend them in Task 3 (source
"episode"). No call exceeded timeout_s 900 (slowest: mllm-hf-qwen at 276.5 s). Four backends sit at or
above the ~21 percent the derivation assumed for 60 s calls; mllm-hf-qwen falls below 10 percent
(5.3 percent: about 15.7 s of motion per ~298 s cycle). chunk_len is NOT changed: the shared fixed N
(D-17) needs an explicit user decision, raised at the Task 3 checkpoint. Recorded in
`mllm_defaults.json` `chunk_len_derivation.measured_mllm_api_latency_s` and
`duty_cycle_at_measured_latency`.
