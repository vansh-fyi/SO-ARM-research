---
phase: 15-mllm-style-pipeline-backends
plan: 06
subsystem: control/vla_bridge (MLLM backends: shared builder + Claude)
status: complete
tags: [mllm, mllm-02, mllm-04, pipe-02, anthropic, claude, builder, secrets, tdd]
requires:
  - control/vla_bridge/backends/base_mllm.py (MllmActionSourceBase, MLLM_BACKENDS, register_mllm_backend, load_mllm_settings; unmodified)
  - control/vla_bridge/mllm_record.py (CallRecorder; unmodified)
  - control/vla_bridge/mllm_inputs.py (MllmInputCollector; unmodified)
  - control/vla_bridge/mllm_urdf.py (urdf_kinematics_text; unmodified)
  - control/vla_bridge/depth_camera.py, stereo_calibration.py (DepthCameraClient, load_calibration; unmodified)
  - anthropic==1.11.0 (Plan 15-02)
provides:
  - vla_bridge.backends.base_mllm_build (PROVIDERS, MIN_MAX_TOKENS, load_mllm_backend_config, build_mllm_backend)
  - vla_bridge.backends.mllm_claude (BACKEND_NAME "mllm-claude", CONFIG_NAME "mllm_claude", _make_anthropic_client, AnthropicAdapter, message_to_result, make_adapter, build)
  - backend_configs/mllm_claude.json
  - "CLI: run_vla_episode.py --backend mllm-claude (with optional --depth-endpoint/--depth-calibration; no --server-address)"
affects:
  - Plans 15-07/15-08 (GPT and HF backends reuse build_mllm_backend and load_mllm_backend_config unchanged; each adds one adapter class, one factory and one config)
  - Plan 15-09 (cross-backend prompt/schema identity checks now have a real registered backend)
  - Plan 15-10 (live Claude smoke: needs ANTHROPIC_API_KEY, operator accepts the 30-day retention NOTICE)
tech-stack:
  added: []
  patterns:
    - one shared fail-closed hardware builder; provider modules are adapter + config + one-line build()
    - provider SDK imported lazily inside the client factory; client built with max_retries=0
    - request kwargs bound to the installed SDK method signature in a test (drift guard)
key-files:
  created:
    - control/vla_bridge/backends/base_mllm_build.py
    - control/vla_bridge/backends/mllm_claude.py
    - control/vla_bridge/backend_configs/mllm_claude.json
    - control/test_mllm_build.py
    - control/test_backend_mllm_claude.py
  modified: []
decisions:
  - "build_mllm_backend order: spec lookup, config, wrist camera, key, adapter factory, depth client (+check_ready, depth_calibration.json), URDF text, recorder, NOTICE, then robot/stereo/wrist; every pre-robot failure is SystemExit"
  - "The streamed final Message has no _request_id in anthropic 1.11.0, so request_id falls back to the stream's request-id header (MessageStream.request_id)"
  - "Claude input_tokens = input + cache_read + cache_creation; cached_input_tokens = cache_read (None stays None); reasoning_tokens = output_tokens_details.thinking_tokens or None"
  - "stop dict carries an extra provider_stop_reason key; incomplete_reason is the raw stop_reason for any non-complete stop"
  - "thinking config is validated in AnthropicAdapter (type adaptive, display summarized|omitted), so a bad value fails at build time before the robot"
  - "effort null is allowed by the shared validator and is then left out of output_config"
metrics:
  duration: "~9 min"
  completed: 2026-10-02
  tasks: 2
  tests_added: 92
  full_suite: "1397 passed (1305 baseline + 92 new)"
---

# Phase 15 Plan 06: Shared MLLM builder and Claude backend Summary

`--backend mllm-claude` is now a registered, selectable backend. It runs a full 20-step chunk through the real `run_episode` loop with a fake Anthropic client, writing records, sidecars and an estimated cost, and the key appears in no artifact. Every MLLM backend uses the same builder, `build_mllm_backend()`. It fails with `SystemExit` before the robot is constructed if anything is wrong.

## What was built

### Task 1: `base_mllm_build.py`

- **`load_mllm_backend_config(config_name, backend_name)`** merges the per-backend file over `mllm_defaults.json` through `load_mllm_settings`, then checks each field below. Any failure raises `SystemExit` naming the file and the field.
  - `provider` must be one of anthropic, openai or huggingface.
  - `model` must be a non-blank string.
  - `api_key_env` must match `^[A-Z_][A-Z0-9_]*$`. A pasted key is rejected and its value is never echoed.
  - `effort` may be null. Otherwise it must be a non-blank string, and must appear in `capabilities.effort_levels` when that list is set.
  - `max_tokens` must be an int of at least 1024. Bools and floats are rejected.
  - `prices`:
    - `input_per_mtok` and `output_per_mtok` must be finite numbers of 0 or more.
    - `cached_input_per_mtok`, if set, must also be a finite number of 0 or more.
    - `verified_on` must be non-blank.
  - `capabilities`: `sampling_supported` must be a bool. `summary_supported` is optional and must be a bool if present. `effort_levels` is an optional list.
  - `data_retention_note` must be non-blank.
- **`build_mllm_backend(backend_name, args, camera_names, joint_limits_deg)`** runs in this order:
  1. Looks up the spec in `MLLM_BACKENDS`; an unknown name raises `SystemExit`.
  2. Loads and validates the config.
  3. Checks for a wrist camera.
  4. Reads the key from the env var named in the config. A missing or blank value raises `SystemExit` naming the variable.
  5. Calls `spec.adapter_factory(key, config)`.
  6. Sets up depth:
     - Only one of the two depth flags given: `SystemExit`.
     - Both given: builds `DepthCameraClient(load_calibration(path), endpoint)` and calls `check_ready()` once. A failure raises `SystemExit` naming the exception type, with the key redacted. On success it writes `depth_calibration.json` into `args.out`.
     - Neither given: prints a WARNING that no depth image will be sent.
  7. Builds `urdf_kinematics_text()`, unless `include_urdf` is false.
  8. Creates `CallRecorder(<out>/mllm_calls, secrets=[key], prices=config["prices"])`.
  9. Prints a two-line operator NOTICE: provider, model, that images and depth are sent, and the `data_retention_note`.
  10. Only now connects hardware: `SO101Follower` (with `max_relative_target=MAX_RELATIVE_TARGET_DEG`), `connect(calibrate=False)`, `write_calibration` with torque disabled, `StereoSplitCamera`, and the wrist `cv2.VideoCapture` (warns if it did not open).
  11. Builds `MllmInputCollector` and `MllmActionSourceBase(..., label=backend_name, urdf_text, recorder, secrets=[key])`.
  12. Returns `BackendHandles(skip_camera_names={"overhead"})`.

  `DepthCameraClient` and `load_calibration` are imported at module level so tests can patch them. The module imports no provider SDK.

### Task 2: `mllm_claude.py` and `mllm_claude.json`

- **`AnthropicAdapter.call`** sends a single `client.messages.stream(...)` call and reads `get_final_message()`. The kwargs are exactly `{model, max_tokens, system, messages, thinking, output_config}`:
  - one user message: for each image, an `Image: <role>` text block followed by a base64 image block, then the user text block;
  - `thinking={"type": "adaptive", "display": "summarized"}`;
  - `output_config={"effort": "xhigh", "format": {"type": "json_schema", "schema": ...}}`.

  No temperature, top_p, top_k, tools, tool_choice or prefill is sent. A test binds the kwargs to the installed `anthropic.resources.messages.Messages.stream` signature.
- **`message_to_result`** maps the response to a `ProviderResult`:
  - `text` joins the text blocks.
  - Thinking kind is `summary` when thinking text is present, `omitted` for empty or redacted thinking, and `none` when there are no thinking blocks.
  - Stop reasons: end_turn becomes complete, refusal becomes refusal (with `stop_details.category` kept), max_tokens becomes truncated, and anything else becomes other.
  - Usage is normalised as listed in the decisions above.
  - `raw_payload` is `message.model_dump(mode="json")`.
  - `request_id` comes from `_request_id`, falling back to the stream header. `response_id` and `model_returned` are also set.
- **`settings_summary`** records three things:
  - requested: the shared sampling (1.0 / 0.95) plus effort, max_tokens and thinking;
  - applied: effort, max_tokens and thinking;
  - omitted: temperature, top_p and top_k, each with a "provider-fixed" reason.
- **`_make_anthropic_client`** imports `anthropic` lazily and returns `anthropic.Anthropic(api_key, timeout=config timeout_s, max_retries=0)`. `make_adapter` uses it. `build` is registered through `register_mllm_backend("mllm-claude", config_name="mllm_claude", adapter_factory=make_adapter)` and delegates to `build_mllm_backend`.
- **`mllm_claude.json`** sets:
  - claude-fable-5-1, `ANTHROPIC_API_KEY`, effort xhigh, max_tokens 64000, adaptive/summarized thinking;
  - prices 10 / 50 / 0.25, verified 2026-10-02, with the source recorded;
  - capabilities: sampling_supported false, effort levels low through max, summary_supported true;
  - the 30-day Covered Model retention note, requirement MLLM-02, dated `model_verified` provenance, and notes on C3, C4, C11 and Pitfall 10.

## Verification

- `pytest test_mllm_build.py`: 55 passed. This includes a 29-case bad-config matrix, the pasted-key no-echo case, every pre-robot failure path, construction and wiring, depth ordering (`check_ready` before the robot), the NOTICE and WARNING lines, repr and details hygiene, `close()` releasing the wrist capture, and a one-chunk run that scans every file under `out/` for the secret.
- `pytest test_backend_mllm_claude.py test_mllm_build.py`: 92 passed.
- `python -c "from vla_bridge import backends; assert 'mllm-claude' in backends.BACKEND_REGISTRY; ...; assert 'mllm-claude' in base_mllm.MLLM_BACKENDS"` exits 0, and `anthropic` is not in `sys.modules` after discovery.
- `python -c "import vla_bridge.backends"` exits 0. `run_vla_episode.py --help` lists `mllm-claude`.
- End-to-end test (registered build, MockRobot, real `run_episode` and `IOLogger`, 20 ticks):
  - exactly one `stream` call, and 20 lines in `episode.jsonl`;
  - the first record has provider `anthropic`, `model_requested` `claude-fable-5-1`, `sidecar_dir` `mllm_calls/0000`, `model_version` `mllm-claude@mllm-chunk`, and `cost_usd_est` = ((1200-200)*10 + 200*0.25 + 300*50)/1e6;
  - the secret, which was planted in the plan text, the thinking text and the env var, appears in no file under `out/` and in no stdout line.
- Full suite: **1397 passed** in about 10 s (1305 baseline plus 92 new), including `test_backend_registry`.

## TDD Gate Compliance

- Task 1: RED `66770ce` (`ImportError: base_mllm_build`), then GREEN `8fc21d2`.
- Task 2: RED `eb5bade` (`ImportError: cannot import name 'mllm_claude'`), then GREEN `fed8835`.
- No refactor commits were needed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Streamed messages carry no `_request_id`**
- **Found during:** Task 2, while checking the installed SDK source (`anthropic/lib/streaming/_messages.py`).
- **Issue:** The plan maps `request_id` from `message._request_id`. In anthropic 1.11.0 that attribute is set only on non-streamed responses. The accumulated stream snapshot leaves it None, so every live Claude record would have had `request_id: null`.
- **Fix:** Use `message._request_id`, falling back to `MessageStream.request_id` (the `request-id` response header). A test covers the fallback.
- **Files modified:** control/vla_bridge/backends/mllm_claude.py
- **Commit:** fed8835

### Choices the plan left open

- `stop` has one extra key, `provider_stop_reason`, holding the raw Anthropic stop reason. `incomplete_reason` holds the raw reason for any stop that is not complete.
- The `thinking` config is validated in the adapter constructor, which runs before the robot is touched. If `effort` is null it is left out of `output_config`.
- If the SDK import fails, `_make_anthropic_client` raises `SystemExit` instead of a bare `ImportError`.

Environment: tests ran with the main checkout's venv interpreter from the worktree's `control/` directory. Nothing was installed. No live API, robot, camera or depth service was contacted. `base_mllm.py`, `mllm_record.py`, `mllm_inputs.py`, `fake_providers.py` and `run_vla_episode.py` are unchanged.

## Known Stubs

None.

## Threat Flags

None beyond the plan's threat register. The new outbound HTTPS surface to api.anthropic.com is the boundary T-15-18..T-15-22 already describe, and each mitigation is tested:
- env-var name only, with pre-robot exit and artifact scans;
- `max_retries=0`, pinned `max_tokens` and effort, and the failure cap;
- the retention NOTICE;
- no tools or prefill, with refusal mapped to a hold;
- the kwargs bound to the installed SDK signature.

## User Setup Required

Before the live run in Plan 15-10:
- Export `ANTHROPIC_API_KEY` with a restricted, workspace-scoped key.
- Fund the Anthropic account. Research estimates about $1 per call at xhigh effort.
- Accept that claude-fable-5-1 keeps lab images for 30 days (it is not zero-data-retention). The build prints this as a NOTICE.

## Self-Check: PASSED

- FOUND: control/vla_bridge/backends/base_mllm_build.py, control/vla_bridge/backends/mllm_claude.py, control/vla_bridge/backend_configs/mllm_claude.json, control/test_mllm_build.py, control/test_backend_mllm_claude.py
- FOUND commits: 66770ce, 8fc21d2, eb5bade, fed8835
