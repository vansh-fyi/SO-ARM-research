---
phase: 14-vla-style-pipeline-backends
plan: 15
subsystem: control/vla_bridge
status: complete
tags: [backends, gemini, gemini-robotics-er, vlab-02, er-conversion]
requires:
  - "14-13: base_gemini.GeminiActionSourceBase, build_gemini_backend(), ACTION_PATH; er_conversion.make_er_convert_fn()"
  - "14-14: base_gemini.load_er_backend_config(), and the binding VLAB-01 vs VLAB-02 decision"
  - "14-02: --backend registry dispatch (register(), load_backend_config(), BackendHandles teardown)"
provides:
  - "vla_bridge.backends.gemini_robotics_2: build(args, camera_names, joint_limits_deg), registered as --backend gemini-robotics-2 (plain register(), not the gRPC lane)"
  - "backend_configs/gemini_robotics_2.json: model gemini-robotics-er-2-preview (re-verified live 2026-10-02), api_key_env GEMINI_API_KEY, thinking_level medium, timeout_s 10, requirement VLAB-02"
  - "Parity tests that pin VLAB-01 and VLAB-02 to the same model and settings, so they cannot drift apart silently"
affects: [16]
tech-stack:
  added: []
  patterns:
    - "Same config-driven Gemini backend shape as 14-14: thin build() -> load_er_backend_config() -> build_gemini_backend(..., convert_fn=make_er_convert_fn(joint_limits_deg), **source_kwargs)"
key-files:
  created:
    - control/vla_bridge/backends/gemini_robotics_2.py
    - control/vla_bridge/backend_configs/gemini_robotics_2.json
    - control/test_backend_gemini_robotics_2.py
  modified: []
decisions:
  - "Followed 14-14's VLAB-01 vs VLAB-02 decision exactly: gemini-robotics-2 calls the same single public model gemini-robotics-er-2-preview with the same prompt, thinking level (medium) and conversion as gemini-robotics. Only the backend name, checkpoint_label/model_version prefix, config file, requirement field and live episode differ."
  - "convert_fn is er_conversion.make_er_convert_fn(joint_limits_deg), as in 14-14, not the bare convert_er_output_to_action the plan names, so the IK clamps to the calibrated limits"
  - "Added parity tests against VLAB-01's shipped config and builder call, so a future one-sided edit has to be deliberate"
metrics:
  duration: "~6 min"
  completed: 2026-10-02
  tasks: 2
  files: 3
---

# Phase 14 Plan 15: Gemini Robotics 2 Backend (VLAB-02) Summary

`--backend gemini-robotics-2` is now a registered backend. It calls the public Gemini Robotics-ER 2 model (`gemini-robotics-er-2-preview`, re-checked against the live docs on 2026-10-02) through 14-13's shared Gemini base, and 14-13's `er_conversion` turns the returned point into one small, capped joint step. Everywhere it is labelled ER plus conversion (`gemini-robotics-2@gemini-er-conversion`), not as the native Gemini Robotics 2 VLA. It follows 14-14's decision exactly: same model and settings as `gemini-robotics`, but a different identity. That gives this phase all 13 VLA-style backends.

## What Was Built

### Task 1: the `gemini-robotics-2` backend module and config (commits 38fa60c RED, bbbb6d8 GREEN)
- **`control/vla_bridge/backends/gemini_robotics_2.py`:** `@register("gemini-robotics-2") build(args, camera_names, joint_limits_deg)`. It calls `base_gemini.load_er_backend_config("gemini_robotics_2", "gemini-robotics-2")`, then returns `base_gemini.build_gemini_backend(model, api_key_env, "gemini-robotics-2", args, camera_names, convert_fn=er_conversion.make_er_convert_fn(joint_limits_deg), **source_kwargs)`.
  - The docstring covers:
    - the "what this is NOT" statement;
    - the model-id verification record (14-14 on 2026-10-01, re-fetched here on 2026-10-02);
    - the explicit equivalence with VLAB-01 and its difference table, pointing back to 14-14;
    - what would change if a newer ER generation or native API access appears.
  - The module has no top-level `google` import and no key material.
- **`control/vla_bridge/backend_configs/gemini_robotics_2.json`:**
  - `model: gemini-robotics-er-2-preview`, `api_key_env: GEMINI_API_KEY`, `thinking_level: medium`, `timeout_s: 10.0`. These match `gemini_robotics.json` exactly.
  - `requirement: VLAB-02`, `product_name: Gemini Robotics 2`, `native_vla_model: null`.
  - `action_path`, `model_verified` and `notes` fields record provenance.
- **Tests (34 in Task 1):**
  - Registration, through the plain lane and as a callable distinct from VLAB-01's.
  - Builder wiring with the `gemini-robotics-2` label.
  - `convert_fn` is bound to the calibrated limits.
  - Every `SystemExit` path: a missing, unparseable or incomplete config; blank or wrongly typed fields; a pasted key, which is never echoed; non-ER ids, including the native name `gemini-robotics-2`; the streaming id; and bad tuning values.
  - The shipped config's honesty fields, the no-key-material check, the lazy-import check and the docstring content.
  - **Parity:** the behavioural keys in the shipped config (`model`, `api_key_env`, `thinking_level`, `timeout_s`, `action_path`, `native_vla_model`) equal VLAB-01's. Both backends' builder calls are identical except for the label.
- **`run_vla_episode.py --help`:** `--backend` now lists `gemini-robotics-2`, for 15 choices.

### Task 2: mocked round-trip regression tests (commit 9fa5992)
- **Full path:** registry, then the shipped config, then the real pinned google-genai 2.25.0 SDK over an in-process `httpx.MockTransport`, then `er_conversion`, then the action. Only the hardware and the HTTP transport are faked. The test asserts that:
  - `model_version == "gemini-robotics-2@gemini-er-conversion"`;
  - the action covers the full `JOINT_ORDER`, is all-finite, is a real nudge, and leaves the gripper alone;
  - the safety validator passes it through unclamped;
  - the request goes to `/v1beta/interactions` with `model == gemini-robotics-er-2-preview` and `thinking_level == medium`;
  - the key travels only in the `x-goog-api-key` header.
- **403:** holds position as `gemini-robotics-2@gemini-error-holding-position`, never raises, and sends a single request.
- **Teardown handles:** `shutdown=None`, `skip_camera_names={"overhead"}`, and `close()` releases the wrist capture.
- **Cross-backend round trip:** VLAB-01 and VLAB-02 send identical JSON request bodies and return identical actions. Only the `model_version` tag differs.

## Model-id re-verification (2026-10-02, curl against the live official pages)
| Page | Finding |
|---|---|
| `ai.google.dev/gemini-api/docs/robotics-overview` (Last updated 2026-09-23) | The only ids are `gemini-robotics-er-2-preview` and `gemini-robotics-er-2-streaming-preview` ("Optimized for real-time streaming via the Live API"). The page also gives the upgrade note from ER 1.6. |
| `ai.google.dev/gemini-api/docs/deprecations` (Last updated 2026-10-01), "Robotics models" | ER 1.6 shut down 2026-08-31 and ER 1.5 shut down 2026-04-30. Both are replaced by `gemini-robotics-er-2-preview`. |
| `deepmind.google/models/gemini-robotics` | The native VLAs (Gemini Robotics 2 and On-Device 2) are open only to "100+ trusted testers", through "Join the waitlist for early access". There is no public id. |

**Conclusion:** unchanged from 14-14. Exactly one public request/response model exists, so per the plan's own branch ("if Plan 14-14 found a single shared public model string ... use that SAME string here and state the equivalence explicitly"), both backends use it.

## Verification
- `cd control && <main .venv>/bin/python -m pytest -q test_backend_gemini_robotics_2.py`: **38 passed**.
- Full suite `cd control && <main .venv>/bin/python -m pytest -q`: **981 passed**, against the fork base's 943, so 38 are new.
- `get_backend_names()` has 15 entries and includes `gemini-robotics-2`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Correctness] `convert_fn` is bound to the calibrated joint limits**
- **Found during:** Task 1. This is the same issue 14-14 fixed for VLAB-01, and the binding 14-14 handoff calls for this fix.
- **Issue:** The plan passes the bare `er_conversion.convert_er_output_to_action`. Called that way, the IK would clamp only to the URDF limits.
- **Fix:** Used `er_conversion.make_er_convert_fn(joint_limits_deg)`, which wraps `convert_er_output_to_action`. A test proves the limits are forwarded.
- **Commit:** bbbb6d8

**2. [Rule 2 - Correctness] Config validation goes through `base_gemini.load_er_backend_config()`**
- **Found during:** Task 1.
- **Issue:** The plan's check covered only "missing or lacking model/api_key_env".
- **Fix:** The shared 14-14 validator is reused instead of duplicated. It also rejects non-ER or streaming ids, pasted keys and bad tuning values, and passes `thinking_level`/`timeout_s` through.
- **Commit:** bbbb6d8

**3. Plan's mocking target**
- The round trip mocks HTTP under the real SDK's Interactions API, as 14-13 and 14-14 established. It does not mock `Client.models.generate_content`.

**4. Added (beyond the plan): VLAB-01/VLAB-02 parity tests**
- Three unit tests and one round-trip test pin the "no artificial variation" decision. A future one-sided config edit therefore fails a test and has to be made deliberately, together with a test update.

## TDD Gate Compliance
- **Task 1:** RED `38fa60c`, where the test file failed at collection with an `ImportError` because `gemini_robotics_2` did not exist yet, then GREEN `bbbb6d8`.
- **Task 2:** a regression test of behaviour that already existed after Task 1, committed as `test(...)` in `9fa5992`. It passed on its first run, because no new implementation was needed.

## Known Stubs
None.

## Threat Flags
None beyond the plan's register.
- **T-14-11 (API key):** the key comes only from the environment. A pasted key is rejected without being echoed. The tests show it travels only in the `x-goog-api-key` header.
- **T-14-12 (spatial output):** still covered by 14-13's 5 mm cap, the `solve_ik` clamp to the calibrated limits and the safety validator.

## Human Checks Deferred
1. **Live hardware, VLAB-02 (Task 2 `<human-check>`)**
   - **test:** Export a *restricted* `GEMINI_API_KEY`, then run `python control/run_vla_episode.py --backend gemini-robotics-2 --out control/outputs/<dir> --max-steps 20 --instruction "<task naming an object in the wrist view>"` against the real SO-ARM101.
     - This must be a separate episode from 14-14's `gemini-robotics` run (D-05).
     - `--server-address` and `--checkpoint` are not needed.
     - Pass `--camera <wrist index>` if `device_map.json`'s wrist index is stale.
   - **expected:** At least one validated, non-held action moves the arm, tagged `gemini-robotics-2@gemini-er-conversion` in `episode.jsonl`, with `er_model` equal to `gemini-robotics-er-2-preview`. It should be a small, bounded nudge (at most 5 mm) toward the named object.
   - **why_human:** This needs real hardware, a real restricted key and billing.
2. **Carried from 14-13 and 14-14:**
   - the `IMAGE_UP_SIGN` / `IMAGE_RIGHT_SIGN` servo direction on the rig;
   - step latency at `thinking_level` "medium". If it is too slow, change it to "low" in both configs to keep parity, and update the parity test if they are split on purpose;
   - the wrist camera must tolerate two `VideoCapture` opens.
3. **Unverifiable here:**
   - real ER 2 accuracy and latency on this rig;
   - whether the project's key is restricted;
   - rate limits, which the docs do not publish.

## Self-Check: PASSED
- FOUND: control/vla_bridge/backends/gemini_robotics_2.py, control/vla_bridge/backend_configs/gemini_robotics_2.json, control/test_backend_gemini_robotics_2.py
- FOUND commits: 38fa60c, bbbb6d8, 9fa5992
