---
phase: 15-mllm-style-pipeline-backends
plan: 08
subsystem: control/vla_bridge (MLLM backends: HF Inference Providers lane)
status: complete
tags: [mllm, mllm-01, pipe-02, huggingface, openai-sdk, qwen, deepseek, kimi, tdd]
requires:
  - control/vla_bridge/backends/base_mllm.py (register_mllm_backend, ProviderResult, STOP_*, IMAGE_ORDER; unmodified)
  - control/vla_bridge/backends/base_mllm_build.py (build_mllm_backend, load_mllm_backend_config; unmodified)
  - control/vla_bridge/mllm_record.py (CallRecorder writes a str raw_payload verbatim after redaction; unmodified)
  - openai==3.23.0 (already installed in control/.venv)
provides:
  - vla_bridge.backends.base_mllm_chat (PROMPT_TOKEN_RESERVE, DEFAULT_BASE_URL, ROUTING_POLICY_NAMES, STRUCTURED_MODES, JSON_OBJECT_SCHEMA_BLOCK, _make_chat_client, validate_chat_config, completion_to_result, OpenAIChatAdapter, make_chat_adapter)
  - vla_bridge.backends.mllm_hf_qwen / mllm_hf_deepseek / mllm_hf_kimi (BACKEND_NAME, CONFIG_NAME, build)
  - backend_configs/mllm_hf_qwen.json, mllm_hf_deepseek.json, mllm_hf_kimi.json
  - "CLI: run_vla_episode.py --backend mllm-hf-qwen|mllm-hf-deepseek|mllm-hf-kimi"
affects:
  - Plan 15-09 (all-five conformance test can reuse OpenAIChatAdapter.request_kwargs and the fake chat client pattern)
  - Plan 15-10 (one live smoke per HF variant; needs HF_TOKEN and purchased Inference Providers credits)
tech-stack:
  added: []
  patterns:
    - one shared chat adapter for every HF-router model; variants are a registration plus a config
    - provider pin validated at adapter-factory time (before the robot), request model string "<model>:<hf_provider>"
    - verbatim wire body (raw.http_response.text) as a str raw_payload, redacted by the recorder
key-files:
  created:
    - control/vla_bridge/backends/base_mllm_chat.py
    - control/vla_bridge/backends/mllm_hf_qwen.py
    - control/vla_bridge/backends/mllm_hf_deepseek.py
    - control/vla_bridge/backends/mllm_hf_kimi.py
    - control/vla_bridge/backend_configs/mllm_hf_qwen.json
    - control/vla_bridge/backend_configs/mllm_hf_deepseek.json
    - control/vla_bridge/backend_configs/mllm_hf_kimi.json
    - control/test_backend_mllm_hf.py
  modified: []
decisions:
  - "HF configs pin cerebras (Qwen), deepinfra (DeepSeek) and baseten (Kimi); validate_chat_config rejects blank, uppercase, '/'/':'-containing and routing-policy (fastest/cheapest/preferred) hf_provider values"
  - "validate_chat_config also requires provider 'huggingface', a model id without a ':' suffix, and an extra_body that does not override any key the adapter sets (model, messages, response_format, max_tokens, temperature, top_p, reasoning_effort, stream, tools, tool_choice)"
  - "reasoning_effort travels in extra_body as the plan asks, even though openai 3.23.0's typed ReasoningEffort literal already includes xhigh and max; the router receives the same JSON key either way"
  - "stop dict mirrors Claude's shape including provider_stop_reason; a non-empty message.refusal maps to refusal with refusal_category 'message_refusal'; content_filter keeps category 'content_filter'"
  - "Kimi and DeepSeek configs set no context_window because the router listing gives none for the pinned provider; only Qwen/cerebras (65,536) is checked against max_tokens 32000 + 16000 reserve"
metrics:
  duration: "~15 min"
  completed: 2026-10-02
  tasks: 2
  tests_added: 91
  full_suite: "1488 passed (1397 baseline + 91 new)"
---

# Phase 15 Plan 08: HF Inference Providers MLLM lane Summary

You can now select `--backend mllm-hf-qwen`, `mllm-hf-deepseek` or `mllm-hf-kimi`. All three share one OpenAI-compatible chat adapter that talks to the HF router, and each is pinned to a provider that supports structured output: Cerebras for Qwen, DeepInfra for DeepSeek and Baseten for Kimi. Each variant runs a full 20-step chunk through the real `run_episode` loop with a fake SDK client. The wire body is kept verbatim, and the HF token appears in no file the run produces.

## What was built

### Task 1: `base_mllm_chat.py`

- **`validate_chat_config(config, backend_name)`** raises `SystemExit` naming the field when any of these fail:
  - `provider` must be `huggingface`.
  - `model` must be non-blank and must not carry a `:` suffix.
  - `hf_provider` must be non-blank, lowercase `[a-z0-9_-]`, and not one of `fastest`, `cheapest` or `preferred`.
  - `base_url` must be `https://` (default `https://router.huggingface.co/v1`).
  - `structured_mode` must be `json_schema` or `json_object`.
  - `extra_body` must be a dict with no reserved keys.
  - `context_window`, if present, must be a positive int with `max_tokens + 16000 <= context_window`.
- **`OpenAIChatAdapter`** properties:
  - `provider` is `huggingface`.
  - `model` is `"<model>:<hf_provider>"`.
  - `sdk` is `{"name": "openai", "version": "3.23.0"}`.
  - `repr` shows no token.
- **`OpenAIChatAdapter.call`** makes one `client.chat.completions.with_raw_response.create(**kwargs)` call. The kwargs keys are exactly `{model, messages, response_format, max_tokens, temperature, top_p, extra_body}`:
  - `messages`: the system message is `request.system_prompt`. The user message has, for each image role, an `Image: <role>` text part and an `image_url` data URL of the request media type, followed by the user text.
  - `response_format` in `json_schema` mode is `{"type": "json_schema", "json_schema": {"name", "strict": true, "schema"}}`.
  - `response_format` in `json_object` mode is `{"type": "json_object"}`, and a `RESPONSE FORMAT` block containing `json.dumps(schema)` is appended to the system message. That block passes `lint_prompt_text`.
  - `temperature` and `top_p` are sent only when `capabilities.sampling_supported` is true and the shared value is not null.
  - `extra_body` is the config's `extra_body` plus `reasoning_effort` when `effort` is not null.
  - No tools and no prefill are sent.
- **`completion_to_result`** maps the response:
  - `raw_payload` is the exact `raw.http_response.text`.
  - `text` is `message.content`.
  - Reasoning text comes from `message.reasoning`, then `reasoning_content`, then either one in `model_extra`. Kind is `raw` when present and `none` when absent or blank.
  - Usage:
    - `prompt_tokens` becomes `input_tokens`.
    - `completion_tokens` becomes `output_tokens`.
    - `completion_tokens_details.reasoning_tokens` becomes `reasoning_tokens`, or None.
    - `prompt_tokens_details.cached_tokens` becomes `cached_input_tokens`, or None.
    - `raw` is the `model_dump`.
  - Stop: `stop` becomes complete, `length` becomes truncated (`incomplete_reason` `length`), `content_filter` or a non-empty `message.refusal` becomes refusal, and anything else becomes other.
  - IDs: `request_id` comes from the `x-request-id` header, `response_id` from `completion.id` and `model_returned` from `completion.model`.
  - Empty `choices` gives text None and stop other.
- **`settings_summary`** has three parts:
  - **requested:** shared sampling, effort, max_tokens, structured_mode, hf_provider and the pinned model.
  - **applied:** exactly what is sent, plus `structured_mode`, `schema_in_prompt` and a note that whether the provider honours these is provider-dependent.
  - **omitted:** temperature/top_p, each with a reason, when sampling is unsupported or null; reasoning_effort when effort is null.
- **`_make_chat_client`** imports `openai` lazily and returns `openai.OpenAI(base_url, api_key, timeout, max_retries=0)`. **`make_chat_adapter`** validates the config, then builds the client and the adapter. Both run inside `build_mllm_backend` before the robot is touched.

### Task 2: three variants and configs

- `mllm_hf_qwen.py`, `mllm_hf_deepseek.py` and `mllm_hf_kimi.py` are thin modules, one `build` each. Each is registered with `register_mllm_backend(..., adapter_factory=base_mllm_chat.make_chat_adapter)` and delegates to `build_mllm_backend`. None imports `openai` at top level.
- The three configs share these values:
  - `provider` is `huggingface`, `base_url` is `https://router.huggingface.co/v1`, `api_key_env` is `HF_TOKEN`.
  - `max_tokens` is 32000, `structured_mode` is `json_schema`, `extra_body` is `{}`.
  - `sampling_supported` is true; `cached_input_per_mtok` is null.
  - Prices are `verified_on` 2026-10-02, sourced from `router.huggingface.co/v1/models`.
  - Each has a third-party retention note, `requirement` MLLM-01, a dated `model_verified`, and notes covering the provider pin, the unneeded gemma/GLM shortlist, the checks for the Plan 15-10 live smoke (A3/A4/A6, JPEG fallback) and the rule that only a human switches to `json_object`.

  | Variant | Request model string | Effort (levels) | $/M in / out | Context |
  |---|---|---|---|---|
  | mllm-hf-qwen | `Qwen/Qwen3.8-27B:cerebras` | xhigh (low, medium, xhigh) | 0.99 / 1.49 | 65,536 (deepinfra is the alternative pin) |
  | mllm-hf-deepseek | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp:deepinfra` | max (levels null) | 0.44 / 1.32 | not set |
  | mllm-hf-kimi | `moonshotai/Kimi-K3:baseten` | max (low, high, max) | 3.0 / 15.0 | not set |

- The DeepSeek notes record that D-23 named Novita, but Novita has no structured output, and that D-27 keeps this model over V4.1-Flash. The Kimi notes flag that it is not cheap and that thinking is always on.

## Verification

- `pytest test_backend_mllm_hf.py`: 91 passed (63 from Task 1). They cover:
  - a 24-case bad-config matrix;
  - the exact kwargs set, bound to the installed `openai.resources.chat.completions.Completions.create` signature;
  - `json_object` mode with a lint-clean appended block;
  - every finish_reason and refusal mapping, and every source of reasoning text;
  - usage with and without details;
  - the verbatim wire body;
  - fenced JSON through the base (`fence_stripped` true);
  - a real `openai.OpenAI` client built with `max_retries` 0 and no request sent;
  - an AST check that `openai` is imported only inside `_make_chat_client`.
- End to end, run once per variant: the registered build with the hardware patches, `HF_TOKEN` set to a sentinel, and the real `run_episode` and `IOLogger` for 20 ticks. Each run checks:
  - one client call, with the request model string equal to the pinned value;
  - `provider` is `huggingface`;
  - `cost_usd_est` equals `(1200*in + 300*out)/1e6` from that variant's prices;
  - `response_raw.json` equals the wire body with the sentinel replaced by `<redacted>`;
  - the sentinel appears in no file and in no stdout line.
- Failure paths on mllm-hf-qwen: an exception carrying the token, `length`, a refusal field, and prose instead of JSON. Each gives 4 holds with reason `provider-error`, `truncated`, `refusal` or `invalid-json` respectively, from one call, and the token is redacted.
- PIPE-02: for one ProviderRequest with all four image roles, the three HF variants and the Claude adapter receive:
  - a byte-identical system prompt;
  - the same schema object;
  - labelled-image sequences that match element for element (label, media type, base64 data);
  - the same user text.
- `from vla_bridge import backends`: the three names are in `BACKEND_REGISTRY`, and `openai` is not in `sys.modules` after discovery. `run_vla_episode.py --help` lists all three.
- `pytest test_backend_mllm_hf.py test_backend_mllm_claude.py test_mllm_build.py` passes. The full suite has **1488 passed** in about 12 s (1397 baseline plus 91 new).
- `git diff --stat` against the base touches only this plan's eight files. `base_mllm.py`, `base_mllm_build.py`, `backends/__init__.py`, `fake_providers.py` and every `mllm_gpt*` file are unchanged.

## TDD Gate Compliance

- Task 1: RED `90aa069` (`ImportError: base_mllm_chat`), then GREEN `dbeae6d`.
- Task 2: RED `11e1142` (`ImportError: mllm_hf_deepseek`), then GREEN `a21363b`.
- No refactor commits were needed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical functionality] Stricter HF config validation**
- **Found during:** Task 1.
- **Issue:** The plan's validator list did not stop three silent misconfigurations:
  - a model id that already carries a `:provider` suffix, which would produce a double pin;
  - a config `provider` other than `huggingface` on the HF adapter;
  - an `extra_body` key that silently overrides a value the adapter sets and logs. For example, `reasoning_effort` in `extra_body` would make the logged effort wrong, and that undermines T-15-28.
- **Fix:** `validate_chat_config` rejects all three with `SystemExit` naming the field. Tests are in the bad-config matrix.
- **Files modified:** control/vla_bridge/backends/base_mllm_chat.py
- **Commit:** dbeae6d

### Choices the plan left open

- **Error label:** the factory contract `adapter_factory(api_key, config)` does not carry the backend name, so `make_chat_adapter` labels config errors `HF model '<model>'`.
- **Refusal field:** a non-empty `message.refusal` uses `refusal_category` `"message_refusal"`.
- **Stop dict:** carries `provider_stop_reason` (the raw `finish_reason`), the same shape as Claude's.
- **Empty choices:** handled as text None and stop other, rather than an exception.
- **Applied settings:** the configured `extra_body` knobs appear in `settings.applied.extra_body` only when non-empty.

Environment: tests ran with the main checkout's venv interpreter from the worktree's `control/` directory. Nothing was installed. No live or paid API (including the HF router), robot, camera or depth service was contacted. Router facts and prices come from the 2026-10-02 research ("Provider Facts" HF table, C1, C2, C12) and were not re-fetched.

## Known Stubs

None.

## Threat Flags

None beyond the plan's threat register. The new outbound HTTPS surface is router.huggingface.co, which then forwards to the pinned third-party provider. That is the T-15-27..T-15-31 boundary, and each mitigation is tested:
- the env-var name only, with token scans of the artifacts;
- the pin and routing-policy rejection, with the exact model string asserted;
- `max_retries` 0, `max_tokens` with the context reserve, dated prices, and Kimi's cost flagged;
- the retention NOTICE from the shared builder;
- `json_object` mode, which only a human can set and which is still pre-validated.

## User Setup Required

Before the live runs in Plan 15-10:
- Export `HF_TOKEN`, a fine-grained Hugging Face token with Inference Providers permission.
- Buy Inference Providers credits under Hugging Face -> Settings -> Billing. No router model is free; a free account gets $0.10 a month.
- Qwen and DeepSeek cost cents per call. Kimi-K3 is about $0.35 per call.
- Accept that images go to Cerebras, DeepInfra or Baseten under their own retention policies. The build prints this as a NOTICE.

## Self-Check: PASSED

- FOUND: control/vla_bridge/backends/base_mllm_chat.py, mllm_hf_qwen.py, mllm_hf_deepseek.py, mllm_hf_kimi.py; control/vla_bridge/backend_configs/mllm_hf_qwen.json, mllm_hf_deepseek.json, mllm_hf_kimi.json; control/test_backend_mllm_hf.py
- FOUND commits: 90aa069, dbeae6d, 11e1142, a21363b
