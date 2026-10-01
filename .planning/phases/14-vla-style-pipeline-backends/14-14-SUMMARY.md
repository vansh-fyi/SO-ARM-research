---
phase: 14-vla-style-pipeline-backends
plan: 14
subsystem: control/vla_bridge
status: complete
tags: [backends, gemini, gemini-robotics-er, vlab-01, er-conversion]
requires:
  - "14-13: base_gemini.GeminiActionSourceBase, build_gemini_backend(), ACTION_PATH; er_conversion.make_er_convert_fn()"
  - "14-02: --backend registry dispatch (register(), load_backend_config(), BackendHandles teardown)"
provides:
  - "vla_bridge.backends.gemini_robotics: build(args, camera_names, joint_limits_deg), registered as --backend gemini-robotics (plain register(), not the gRPC lane)"
  - "backend_configs/gemini_robotics.json: model gemini-robotics-er-2-preview (verified live), api_key_env GEMINI_API_KEY, thinking_level medium, timeout_s 10"
  - "base_gemini.load_er_backend_config(config_name, backend_name) -> (model, api_key_env, source_kwargs): shared config validation for 14-14 and 14-15"
  - "base_gemini.ER_MODEL_PREFIX, THINKING_LEVELS"
  - "The binding VLAB-01 vs VLAB-02 decision (section below) that 14-15 follows"
affects: [14-15, 16]
tech-stack:
  added: []
  patterns:
    - "Config-driven Gemini backend: thin build() -> load_er_backend_config() -> build_gemini_backend(..., convert_fn=make_er_convert_fn(joint_limits_deg), **source_kwargs)"
    - "ER-only guard: a non-ER (native VLA) or Live-API-only model id is refused with SystemExit instead of going through the point-servo conversion"
key-files:
  created:
    - control/vla_bridge/backends/gemini_robotics.py
    - control/vla_bridge/backend_configs/gemini_robotics.json
    - control/test_backend_gemini_robotics.py
  modified:
    - control/vla_bridge/backends/base_gemini.py
decisions:
  - "VLAB-01 (gemini-robotics) and VLAB-02 (gemini-robotics-2) both call the single public model gemini-robotics-er-2-preview, with the same prompt, thinking level and conversion. They differ only in --backend name, checkpoint_label/model_version prefix, config file and independent live episodes. No artificial differentiation."
  - "convert_fn is er_conversion.make_er_convert_fn(joint_limits_deg), not the bare convert_er_output_to_action, so the IK clamps to the calibrated limits"
  - "Config validation lives in base_gemini.load_er_backend_config() so 14-15 reuses it rather than duplicating it"
  - "api_key_env must be an uppercase POSIX env-var name. That rejects a pasted key (Google keys contain lowercase), and the error never echoes the value"
metrics:
  duration: "~20 min"
  completed: 2026-10-01
  tasks: 2
  files: 4
---

# Phase 14 Plan 14: Gemini Robotics Backend (VLAB-01) Summary

`--backend gemini-robotics` is now a registered backend. It calls the public Gemini Robotics-ER 2 model (`gemini-robotics-er-2-preview`, checked against the live docs) through Plan 14-13's shared base. 14-13's local `er_conversion` then turns the returned point into one small, capped joint step. It is documented and tagged everywhere as ER plus conversion (`gemini-robotics@gemini-er-conversion`), never as a native Gemini Robotics VLA policy.

## What Was Built

### Task 1: the `gemini-robotics` backend module and config (commits 7ce7680 RED, e307728 GREEN)
- **`control/vla_bridge/backends/gemini_robotics.py`:**
  - `@register("gemini-robotics") build(args, camera_names, joint_limits_deg)`: loads and validates the config, then returns `base_gemini.build_gemini_backend(model, api_key_env, "gemini-robotics", args, camera_names, convert_fn=er_conversion.make_er_convert_fn(joint_limits_deg), **source_kwargs)`.
  - The module docstring holds the "what this is NOT" statement, the model-id verification record and the VLAB-01 vs VLAB-02 decision.
  - The module has no top-level `google` import.
- **`control/vla_bridge/backend_configs/gemini_robotics.json`:**
  - Settings: `model: gemini-robotics-er-2-preview`, `api_key_env: GEMINI_API_KEY`, `thinking_level: medium`, `timeout_s: 10.0`.
  - Honesty and provenance fields: `requirement`, `product_name`, `action_path`, `native_vla_model: null`, `model_verified`, `notes`.
- **`base_gemini.load_er_backend_config(config_name, backend_name)`:** the shared validator. It raises `SystemExit` naming the bad field when:
  - the config file is missing, unparseable or not a JSON object;
  - `model` is blank or not a string;
  - `model` is not a `gemini-robotics-er-` id. A native VLA id would need a different decoder.
  - `model` is the Live-API-only `-streaming-` variant;
  - `api_key_env` is not an uppercase env-var name. The value is never echoed.
  - `thinking_level` is not one of minimal, low, medium or high;
  - `timeout_s` is not a finite positive number. A bool also counts as invalid.
- **Optional keys:** `thinking_level` and `timeout_s` pass through as `source_kwargs` only when present and non-null.
- **`run_vla_episode.py --help`:** now lists `gemini-robotics`, making 14 choices.

### Task 2: mocked round-trip regression tests (commit 9ffc431)
- **Full path, with the real pinned google-genai 2.25.0 SDK:** registry, then the shipped config, then the real SDK over an in-process `httpx.MockTransport`, then `er_conversion`, then the action. Only the hardware (`SO101Follower`, `StereoSplitCamera`, `cv2.VideoCapture`) and the HTTP transport are faked. The test asserts that:
  - `model_version == "gemini-robotics@gemini-er-conversion"`;
  - the action covers the full `JOINT_ORDER`, is all-finite, is a real nudge, and leaves the gripper alone;
  - `safety_validator.validate_action()` passes the action through unclamped with no flags;
  - the request goes to `/v1beta/interactions` with `model == gemini-robotics-er-2-preview` and `thinking_level == medium`;
  - the key appears only in the `x-goog-api-key` header, not in the body, `last_details` or `repr`.
- **403 (unrestricted key):** the backend holds position as `gemini-robotics@gemini-error-holding-position`, never raises, and sends exactly one request.
- **Teardown handles:** `shutdown=None`, `skip_camera_names={"overhead"}`, and `close()` releases the wrist capture.
- **The plan's `Client.models.generate_content` mock:** not used, because 14-13 established that the ER 2 docs use the Interactions API. The tests mock that instead.

## VLAB-01 vs VLAB-02 distinction

**Plan 14-15 follows this decision.**

**Facts, verified 2026-10-01 with curl against the live official pages:**
- **`ai.google.dev/gemini-api/docs/robotics-overview` (Last updated 2026-09-23):** ER 2 has two endpoints.
  - `gemini-robotics-er-2-preview` is the standard model; the Python sample calls `client.interactions.create(model="gemini-robotics-er-2-preview", ...)`.
  - `gemini-robotics-er-2-streaming-preview` is "Optimized for real-time streaming via the Live API".
- **`ai.google.dev/gemini-api/docs/deprecations` (Last updated 2026-10-01), "Robotics models":**
  - `gemini-robotics-er-1.6-preview` shut down **2026-08-31**.
  - `gemini-robotics-er-1.5-preview` shut down **2026-04-30**.
  - Both name `gemini-robotics-er-2-preview` as their replacement.
  - The `/models` page (also Last updated 2026-10-01) still lists ER 1.6. The deprecations table is the authoritative shutdown record, so ER 1.6 counts as gone.
- **`deepmind.google/models/gemini-robotics`:** the VLA models (Gemini Robotics 2, "Our most advanced vision-language-action model", and Gemini Robotics On-Device 2) are open only to "100+ trusted testers" through "Join the waitlist for early access". Neither the models page nor the robotics docs give a public API id for any native VLA.
- **Conclusion:** exactly ONE public request/response robotics model exists, and no first-generation Gemini Robotics model of any kind (VLA or ER) is callable.

**Decision:**

| Aspect | VLAB-01 (this plan) | VLAB-02 (Plan 14-15) |
|---|---|---|
| `--backend` name | `gemini-robotics` | `gemini-robotics-2` |
| Module | `backends/gemini_robotics.py` | `backends/gemini_robotics_2.py` |
| Config file | `backend_configs/gemini_robotics.json` | `backend_configs/gemini_robotics_2.json` |
| `checkpoint_label` | `gemini-robotics` | `gemini-robotics-2` |
| `model_version` (converted) | `gemini-robotics@gemini-er-conversion` | `gemini-robotics-2@gemini-er-conversion` |
| `model` | `gemini-robotics-er-2-preview` | `gemini-robotics-er-2-preview` (**same**) |
| Prompt / thinking level / `er_conversion` | base defaults / `medium` / `make_er_convert_fn(joint_limits_deg)` | **identical** |
| `requirement` field in config | `VLAB-01` | `VLAB-02` |
| `native_vla_model` | `null` | `null` |
| Live-hardware check | its own episode | its own, separate episode (D-05) |

- **Behaviour today:** the two backends behave identically. `last_details["er_model"]`, and so each `episode.jsonl` record, shows that both called the same model.
- **No artificial difference:** nothing (thinking level, prompt, gain) is varied just to make them look like different generations. That would be dishonest and would also confound Phase 16's cross-backend comparison.
- **How 14-15 should implement it:** mirror this module exactly. Call `base_gemini.load_er_backend_config("gemini_robotics_2", "gemini-robotics-2")`, set `CHECKPOINT_LABEL = "gemini-robotics-2"`, and use `convert_fn=er_conversion.make_er_convert_fn(joint_limits_deg)`. The config needs the same model id with `"requirement": "VLAB-02"` and `"product_name": "Gemini Robotics 2"`. The docstring should state the equivalence and point back here.
- **What would change later:**
  - **A newer public ER generation:** the two `model` fields can move independently with a one-line JSON edit and no code change. For example, VLAB-02 could move to a hypothetical ER 3 while VLAB-01 stays on ER 2.
  - **Native VLA API access granted to this project:** VLAB-01 would target the first-generation native model and VLAB-02 Gemini Robotics 2. That needs a NEW action source that decodes native actions, not `er_conversion`. The `ER_MODEL_PREFIX` guard in `load_er_backend_config()` deliberately refuses a non-ER id, so a native model cannot silently go through the point-servo conversion.

## Verification
- `cd control && <main .venv>/bin/python -m pytest -q`: **943 passed**, against the fork base's 909, with 34 new tests in `test_backend_gemini_robotics.py`.
- `test_base_gemini.py::test_no_api_key_is_hardcoded_in_the_module` still passes; see deviation 3.
- `run_vla_episode.py --help`'s `--backend` choices now include `gemini-robotics`, 14 names in all.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Correctness] `convert_fn` is bound to the calibrated joint limits**
- **Found during:** Task 1.
- **Issue:** The plan passes the bare `er_conversion.convert_er_output_to_action`. `GeminiActionSourceBase` calls `convert_fn(er_output, joint_state)` with no limits, so the IK would clamp only to the URDF limits, not to the calibrated `joint_limits_deg` that `build()` receives. 14-13's integration notes call for `make_er_convert_fn(joint_limits_deg)`.
- **Fix:** Used `er_conversion.make_er_convert_fn(joint_limits_deg)`, which still calls `convert_er_output_to_action`. A test proves the limits are forwarded.
- **Commit:** e307728

**2. [Rule 2 - Correctness/Security] Config validation goes beyond "model/api_key_env present"**
- **Found during:** Task 1.
- **Issue:** The plan's check only ensured that the two keys exist. Several bad configs would still have passed:
  - a native-VLA or non-ER model id, which `er_conversion` cannot interpret;
  - the Live-API-only streaming id, which fails at the API;
  - a pasted key in `api_key_env`;
  - an out-of-range `thinking_level` or `timeout_s`.
- **Fix:** Added `base_gemini.load_er_backend_config()`. It lives in the shared base, outside this plan's `files_modified`, so that 14-15 reuses it instead of copying it. A `thinking_level`/`timeout_s` passthrough was also added, so latency tuning ("try `low` if slow", from 14-13's human check) is a config edit.
- **Commit:** e307728

**3. [Rule 1 - Bug, own code] The first key-paste guard tripped 14-13's "no key in module" test**
- **Found during:** the Task 1 full-suite run.
- **Issue:** The first version checked for the literal Google key prefix in `base_gemini.py`, so `test_no_api_key_is_hardcoded_in_the_module` failed.
- **Fix:** Removed the literal. `api_key_env` must now match `^[A-Z_][A-Z0-9_]*$`, the POSIX env-var convention, which rejects any Google key because those contain lowercase letters.
- **Commit:** e307728, fixed before committing.

**4. Plan's mocking target**
- The plan says to mock `Client.models.generate_content`. The backend uses the Interactions API, per 14-13's verified finding, so the round trip mocks HTTP under the real SDK instead.

## TDD Gate Compliance
- **Task 1:** RED `7ce7680`, where the test file failed at collection because the module was missing, then GREEN `e307728`.
- **Task 2:** a regression test of behaviour that already existed in 14-13 plus Task 1. It passed on its first run, because no new implementation was needed. It was committed as `test(...)` in `9ffc431`.

## Known Stubs
None.

## Threat Flags
None beyond the plan's register.
- **T-14-11 (API key):** the key comes only from the environment. The config can no longer hold a pasted key, and errors never echo it. The tests show it travels only in the `x-goog-api-key` header.
- **T-14-12 (spatial output):** unchanged. It is still covered by 14-13's 5 mm cap, the `solve_ik` clamp (now with the calibrated limits) and the safety validator.

## Human Checks Deferred
1. **Live hardware, VLAB-01 (Task 2 `<human-check>`)**
   - **test:** Export a *restricted* `GEMINI_API_KEY`, then run `python control/run_vla_episode.py --backend gemini-robotics --out control/outputs/<dir> --max-steps 20 --instruction "<task naming an object in the wrist view>"` against the real SO-ARM101. `--server-address` and `--checkpoint` are not needed.
     - **Wrist camera:** without `--server-address`, `main()` takes it from `device_map.json`'s raw `wrist` index (14-02's note). Pass `--camera <wrist index>` if that index is stale.
   - **expected:** At least one validated, non-held action moves the arm, tagged `gemini-robotics@gemini-er-conversion` in `episode.jsonl`, with `er_model` equal to `gemini-robotics-er-2-preview`. It should be a small, bounded nudge (at most 5 mm) toward the named object, not full manipulation.
   - **why_human:** This needs real hardware, a real restricted key and billing.
2. **Carried from 14-13:**
   - **Visual-servo direction:** the `IMAGE_UP_SIGN` / `IMAGE_RIGHT_SIGN` signs on the rig.
   - **Step latency:** latency at `thinking_level` "medium". If it is too slow, set `"thinking_level": "low"` in `gemini_robotics.json`.
   - **Wrist camera:** the camera must tolerate two `VideoCapture` opens.
3. **Unverifiable here:**
   - real ER 2 accuracy and latency on this rig;
   - whether this project's key is restricted;
   - rate limits, which the robotics pages do not publish.

## Self-Check: PASSED
- FOUND: control/vla_bridge/backends/gemini_robotics.py, control/vla_bridge/backend_configs/gemini_robotics.json, control/test_backend_gemini_robotics.py, control/vla_bridge/backends/base_gemini.py (load_er_backend_config)
- FOUND commits: 7ce7680, e307728, 9ffc431
