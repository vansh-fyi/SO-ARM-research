---
phase: 15-mllm-style-pipeline-backends
plan: 07
subsystem: control/vla_bridge (MLLM backends: GPT via OpenAI Responses API)
status: complete
tags: [mllm, mllm-03, pipe-02, openai, gpt, responses-api, secrets, tdd]
requires:
  - control/vla_bridge/backends/base_mllm.py (register_mllm_backend, ProviderRequest/ProviderResult, STOP_*, IMAGE_ORDER; unmodified)
  - control/vla_bridge/backends/base_mllm_build.py (build_mllm_backend, load_mllm_backend_config; unmodified)
  - control/vla_bridge/backends/mllm_claude.py (structural reference; unmodified)
  - openai==3.23.0 (installed in control/.venv; nothing installed by this plan)
provides:
  - vla_bridge.backends.mllm_gpt (BACKEND_NAME "mllm-gpt", CONFIG_NAME "mllm_gpt", _make_openai_client, OpenAIResponsesAdapter, response_to_result, make_adapter, build)
  - backend_configs/mllm_gpt.json
  - "CLI: run_vla_episode.py --backend mllm-gpt"
affects:
  - Plan 15-09 (cross-backend prompt/schema identity now has two registered vendors)
  - Plan 15-10 (live GPT smoke: OPENAI_API_KEY, confirm assumption A5 reasoning.summary, real latency)
tech-stack:
  added: []
  patterns:
    - adapter consumes stream events and keeps the terminal response event (completed/incomplete/failed) instead of get_final_response()
    - unsupported parameters decided by the static capabilities table, logged as omitted with a reason, never negotiated
key-files:
  created:
    - control/vla_bridge/backends/mllm_gpt.py
    - control/vla_bridge/backend_configs/mllm_gpt.json
    - control/test_backend_mllm_gpt.py
  modified: []
decisions:
  - "The adapter iterates the Responses stream and takes event.response from the terminal event, because openai 3.23.0 ResponseStream.get_final_response() raises RuntimeError on response.incomplete / response.failed streams"
  - "request_id comes from response._request_id, falling back to the stream's x-request-id header (streamed responses carry no _request_id)"
  - "status incomplete maps to truncated, except incomplete_details.reason content_filter, which maps to refusal with refusal_category content_filter"
  - "Non-complete stops set incomplete_reason: incomplete_details.reason for incomplete, 'refusal' for a refusal part, error.code (or the status) for failed/cancelled/unknown; provider_stop_reason holds the raw status"
  - "effort null leaves effort out of reasoning; reasoning is sent only when non-empty"
  - "A stream 'error' event without a terminal event raises RuntimeError carrying the error code and message (the base redacts and holds)"
metrics:
  duration: "~25 min"
  completed: 2026-10-02
  tasks: 2
  tests_added: 44
  full_suite: "1441 passed (1397 baseline + 44 new)"
---

# Phase 15 Plan 07: GPT backend (OpenAI Responses API) Summary

`--backend mllm-gpt` is a registered backend for `gpt-6-astra`. It is built from one adapter class, a one-line `build()` that calls the shared `build_mllm_backend`, and one config file. Nothing in the shared base or builder changed. A test runs a full 20-step chunk through the real `run_episode` loop with a fake OpenAI client. That run makes exactly one `responses.stream` call, sends `store=False` with no sampling parameters, and the key appears in no artifact.

## What was built

### Task 1: `mllm_gpt.py` and `mllm_gpt.json`

**`OpenAIResponsesAdapter.request_kwargs`** sends exactly these keys: `{model, instructions, input, reasoning, text, max_output_tokens, store}`.
- `instructions` is `request.system_prompt`.
- `input` is one user message. For each image present, in `IMAGE_ORDER`, it holds an `input_text` "Image: <role>" block followed by an `input_image` data URL (the request's media type, `detail: "high"`). The user text block comes last.
- `reasoning` is `{"effort": "xhigh", "summary": "auto"}`, or just `{"effort": "xhigh"}` when `capabilities.summary_supported` is false.
- `text` is `{"format": {"type": "json_schema", "name": schema_name, "strict": true, "schema": ...}}`.
- `max_output_tokens` is 64000 and `store` is False.
- No temperature, top_p, top_logprobs, tools or tool_choice is sent. A test binds the kwargs to the installed `openai.resources.responses.Responses.stream` signature.

**`call`** opens `client.responses.stream(...)`, consumes the events and keeps the `response` from the terminal event (`response.completed`, `response.incomplete` or `response.failed`). It does not catch exceptions; the base handles them.

**`response_to_result`** maps the response to a `ProviderResult`:
- `text` joins the `output_text` parts of the message items.
- Thinking kind is `summary` when reasoning items carry summary text, `omitted` when they carry none, and `none` when there are no reasoning items.
- Stop reasons:

  | Response | Stop |
  |---|---|
  | completed | complete |
  | completed with a refusal content part | refusal |
  | incomplete | truncated (`incomplete_reason` keeps `incomplete_details.reason`) |
  | incomplete with reason `content_filter` | refusal |
  | failed, cancelled or anything else | other |

- Usage: `input_tokens` (cached tokens already included), `output_tokens`, `reasoning_tokens` from `output_tokens_details`, and `cached_input_tokens` from `input_tokens_details.cached_tokens` (None when absent).
- `raw_payload` is `model_dump(mode="json")`. Ids come from `id` and `_request_id`, with the header fallback described below; `model_returned` comes from `model`.

**`settings_summary`** has three parts:
- requested: the shared sampling (1.0 / 0.95) plus effort, summary, max_output_tokens and store;
- applied: effort, summary (when supported), max_output_tokens and `store: False`;
- omitted: temperature and top_p, with the reason "removed whenever reasoning effort is not none and gpt-6-astra does not support none", plus `reasoning.summary` with a static-table reason when it is unsupported.

**`_make_openai_client`** imports `openai` lazily and returns `openai.OpenAI(api_key, timeout=timeout_s, max_retries=0)`. `make_adapter` uses it, and `build` is registered with `@register_mllm_backend("mllm-gpt", config_name="mllm_gpt", adapter_factory=make_adapter)`. Backend autodiscovery picks the module up, so `backends/__init__.py` did not change.

**`mllm_gpt.json`** sets:
- provider openai, model gpt-6-astra, `OPENAI_API_KEY`, effort xhigh, max_tokens 64000;
- prices 10 / 50 / 1.0, verified 2026-10-02, source developers.openai.com;
- capabilities: sampling_supported false, effort levels low through max, summary_supported true;
- a retention note covering store=false and the data-controls check;
- MLLM-03, dated `model_verified` provenance, and notes on C3, C11, A5 and Pitfall 10.

### Task 2: end-to-end and cross-provider tests

- **Registered build:** `BACKEND_REGISTRY["mllm-gpt"]` runs with fake hardware and `OPENAI_API_KEY` set to a secret sentinel. The sentinel is planted in the plan text and the reasoning summary, and the run goes through the real `run_episode` and `IOLogger` for 20 ticks.
  - Exactly one `stream` call and 20 records.
  - The first record has provider `openai`, `model_requested` `gpt-6-astra`, reasoning_tokens 120, `cost_usd_est` = ((1200-200)*10 + 200*1.0 + 300*50)/1e6, and `sidecar_dir` `mllm_calls/0000`.
  - Later records carry only `chunk_ref`.
  - The secret appears in no file under `out/` and in no stdout line.
- **Failure paths:** a client exception containing the key, an `incomplete` / `max_output_tokens` response, and a refusal part each produce 20 records of `mllm-gpt@{provider-error|truncated|refusal}-holding-position`. Each makes one call and leaks no key.
- **Cross-provider identity (PIPE-02):** the sequence of (label, image bytes) pairs plus the final user text matches the Claude adapter element for element. The same `system_prompt` and `schema` objects reach both fake clients unchanged. Driven through `MllmActionSourceBase` with the real prompt and schema, both vendors receive equal instructions, schema and content order.

## Verification

- `pytest test_backend_mllm_gpt.py`: 44 passed.
- `pytest test_backend_mllm_gpt.py test_backend_mllm_claude.py test_mllm_build.py`: 136 passed.
- Full suite: **1441 passed** in about 13 s (1397 baseline plus 44 new).
- `python -c "from vla_bridge import backends; assert 'mllm-gpt' in backends.BACKEND_REGISTRY"` exits 0, and `openai` is not in `sys.modules` after discovery. `run_vla_episode.py --help` lists `mllm-gpt`.
- `git diff 46c91f9 -- base_mllm.py base_mllm_build.py backends/__init__.py fake_providers.py` is empty.

## TDD Gate Compliance

- Task 1: RED `45aaa25` (`ImportError: cannot import name 'mllm_gpt'`), then GREEN `73809a1`.
- Task 2: `18a541a` (test-only). These are integration tests over Task 1's production code. Task 2 adds no new production behavior, so they passed on the first run. They are not a skipped RED: each would fail if the adapter called `get_final_response()`, if it reordered content, or if the key leaked.
- No refactor commits were needed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `get_final_response()` cannot return an incomplete or failed response**
- **Found during:** Task 1, while reading the installed SDK source (`openai/lib/streaming/responses/_responses.py`).
- **Issue:** The plan's call shape was `with client.responses.stream(...) as stream: response = stream.get_final_response()`. In openai 3.23.0 the stream state only stores a final response on a `response.completed` event. A `response.incomplete` or `response.failed` stream makes `get_final_response()` raise `RuntimeError("Didn't receive a response.completed event.")`. Every max_output_tokens truncation, the case research Pitfall 10 says is common at xhigh effort, would have been logged as `provider-error` rather than `truncated`, and the incomplete response's usage and reasoning tokens would have been lost.
- **Fix:** The adapter consumes the stream events and keeps `event.response` from the terminal event (completed, incomplete or failed). It raises if no terminal event arrives, including the code and message of any stream `error` event. Tests drive a real `ResponseStream` built from typed SDK events with no network. One shows that `get_final_response()` raises on an incomplete stream; the others show the adapter maps that stream to truncated with usage intact, and maps a completed stream to text and a summary.
- **Files modified:** control/vla_bridge/backends/mllm_gpt.py, control/test_backend_mllm_gpt.py
- **Commit:** 73809a1

**2. [Rule 1 - Bug] Streamed responses carry no `_request_id`**
- **Found during:** Task 1, in the same source check, which repeats the Claude finding from 15-06.
- **Issue:** Typed objects built from stream events do not get `_request_id`, so mapping only `response._request_id` would leave every live record with `request_id: null`.
- **Fix:** Fall back to the `x-request-id` header of the stream's HTTP response. This is covered by a fake-stream test and by the real-`ResponseStream` test.
- **Commit:** 73809a1

### Choices the plan left open

- `incomplete` with `incomplete_details.reason` `content_filter` maps to refusal, with `refusal_category` `content_filter`, not truncated. A content filter is not a token-budget problem, and logging it as truncated would mislead the Phase 16 analysis. All other incomplete reasons map to truncated.
- `stop` carries `provider_stop_reason` (the raw `status`), matching the Claude adapter's extra key.
- The `requested` settings always include `summary: "auto"` (the nominal request). `applied` includes it only when it is supported.

Environment: tests ran with the main checkout's venv interpreter from the worktree's `control/` directory. Nothing was installed. No live API, robot, camera or depth service was contacted. The only shared helper reused is `RecordingMockRobot` plus the Claude fake client, imported from `test_backend_mllm_claude.py`; no shared module was edited.

## Known Stubs

None.

## Threat Flags

None beyond the plan's threat register. The outbound HTTPS surface to api.openai.com is the boundary T-15-23..T-15-26 already describe, and each mitigation is tested:
- env-var name only, with end-to-end artifact and stdout scans;
- `max_retries=0`, pinned max_output_tokens and effort, and one call per chunk;
- `store=False` on every request plus the retention NOTICE;
- the static capabilities table with omitted-parameter reasons, and the kwargs bound to the installed SDK signature.

## User Setup Required

Before the live run in Plan 15-10:
- Export `OPENAI_API_KEY` with a project-scoped key.
- Fund the OpenAI project. Research estimates about $1 per call at xhigh effort.
- Review the organization's data-controls and retention settings. `store=false` is sent, but lab images still leave the machine.
- On the first live call, confirm that `reasoning.summary` is accepted (assumption A5). If the provider returns a 400, set `capabilities.summary_supported` to false in `mllm_gpt.json`.

## Self-Check: PASSED

- FOUND: control/vla_bridge/backends/mllm_gpt.py, control/vla_bridge/backend_configs/mllm_gpt.json, control/test_backend_mllm_gpt.py
- FOUND commits: 45aaa25, 73809a1, 18a541a
