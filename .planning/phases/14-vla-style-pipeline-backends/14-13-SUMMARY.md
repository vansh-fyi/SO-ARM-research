---
phase: 14-vla-style-pipeline-backends
plan: 13
subsystem: control/vla_bridge
status: complete
tags: [backends, gemini, gemini-robotics-er, google-genai, visual-servo, ik, vlab-01, vlab-02]
requires:
  - "14-01: BackendHandles, base_http_adapter.capture_observation_images() / _is_joint_action()"
  - "14-02: --backend registry dispatch; BackendHandles default robot.disconnect() teardown + action_source.close()"
  - "14-03: eef_to_joint.solve_ik(current, eef_delta, joint_limits_deg=...), forward_kinematics()"
provides:
  - "vla_bridge.backends.base_gemini: GeminiActionSourceBase, build_gemini_backend(model, api_key_env, checkpoint_label, args, camera_names, convert_fn, **source_kwargs), parse_er_json(), ACTION_PATH, CONVERSION_SUFFIX, DEFAULT_ER_PROMPT_TEMPLATE"
  - "vla_bridge.backends.er_conversion: convert_er_output_to_action(er_response, joint_state, image_shape=None, joint_limits_deg=None), make_er_convert_fn(joint_limits_deg, image_shape=None), extract_target_yx(), PIXELS_TO_EEF_GAIN, MAX_NUDGE_M"
  - "control/requirements.txt: google-genai==2.25.0"
affects: [14-14, 14-15]
tech-stack:
  added:
    - "google-genai==2.25.0 (pulls pydantic 2.13.5, google-auth 2.59.1, tenacity, websockets, cryptography; httpx/anyio/requests were already present)"
  patterns:
    - "Gemini Robotics-ER Interactions API (client.interactions.create) with an inline base64 PNG + text prompt; JSON points parsed from output_text"
    - "secret discipline: key only from the env var, SystemExit if missing, redacted from any logged error text"
    - "bounded proportional visual servo: pixel offset -> <=5 mm horizontal nudge along the FK-derived heading -> URDF IK"
key-files:
  created:
    - control/vla_bridge/backends/base_gemini.py
    - control/vla_bridge/backends/er_conversion.py
    - control/test_base_gemini.py
  modified:
    - control/requirements.txt
    - control/test_backend_registry.py
decisions:
  - "Calls go through the Interactions API (client.interactions.create), which the live ER 2 docs use, not models.generate_content (the plan's guess). Model id gemini-robotics-er-2-preview"
  - "Only the wrist frame is sent to ER, so 'offset from image centre' is meaningful for an eye-in-hand camera. Capture still goes through the shared capture_observation_images() (one wrist read plus one stereo pair), as on the HTTP lane"
  - "The image-plane offset is mapped onto the arm's FK-derived horizontal heading (image-up = radially outward, image-right = right tangent), not onto fixed base axes. At pan=0 this URDF's reach direction is about -y, not +x"
  - "Converted actions are always tagged <label>@gemini-er-conversion. Holds are tagged <label>@<reason>-holding-position. Every last_details carries action_path=ACTION_PATH (ER + local conversion, not native VLA)"
  - "A conversion that chose not to move keeps @gemini-er-conversion but adds validator flag gemini-er-conversion-held-position"
  - "build_gemini_backend() builds the action source (key check) before connecting the robot, so a missing key never touches hardware"
  - "google-genai pinned at 2.25.0 (the plan and research audit version, released 2026-09-22), not the less-than-a-day-old 2.26.0"
metrics:
  duration: "~40 min"
  completed: 2026-10-01
  tasks: 2
  files: 5
---

# Phase 14 Plan 13: Gemini Robotics-ER Foundation Summary

This plan adds the shared base for both Gemini backends. `GeminiActionSourceBase` calls the public Gemini Robotics-ER 2 model (`gemini-robotics-er-2-preview`) directly over HTTPS through the official `google-genai` SDK, which returns a target point. `er_conversion` then turns that point into one small, capped joint step via the URDF IK. This path is documented everywhere as ER plus a local conversion layer, not as a native Gemini Robotics VLA policy.

## What Was Built

### Task 1: `base_gemini.py`, `GeminiActionSourceBase` (commits e7d6b5a RED, 5dab568 GREEN)
- **Construction:**
  - The API key comes from `os.environ[api_key_env]`. If it is missing or blank, construction raises `SystemExit("Environment variable <X> is not set")` before any client is built.
  - `google.genai` is imported lazily inside `_make_genai_client()`, because backend discovery imports modules at startup.
  - The client gets `HttpOptions(timeout=<ms>, retry_options=HttpRetryOptions(attempts=1))`.
- **`get_action()`:**
  1. **Guard:** a non-blocking in-flight guard.
  2. **Capture:** the shared `capture_observation_images()` reads the wrist once and the stereo pair once, then downsizes to 640 and encodes as PNG.
  3. **Request:** `client.interactions.create(model, input=[{"type":"image","data":b64,"mime_type":"image/png"}, {"type":"text","text":prompt}], generation_config={"thinking_level": "medium"}, timeout=timeout_s)`.
  4. **Parse:** `parse_er_json(output_text)` accepts bare JSON or JSON in a code fence.
  5. **Convert:** `convert_fn({"er_entries", "image_shape"}, joint_state)`.
  6. **Check:** the result must be a full, finite `JOINT_ORDER` dict.
  7. **Return:** `(action, "<label>@gemini-er-conversion")`.
- **Hold reasons (it never raises):** `gemini-request-inflight`, `camera-read-failed`, `gemini-error` (any SDK, transport, parse or conversion exception), and `action-format-error`, each with the `-holding-position` suffix.
- **`last_details`:**
  - Normal contents: `raw_model_output` (parsed entries, image shape, model text truncated to 2000 chars), `bridge_flags`, `observation=None`, `action_path`, `er_model`.
  - On an error it also holds `error_type` and the error text with the key redacted.
  - It never holds the client, headers or the key. `__repr__` hides the key.
- **`build_gemini_backend()`:** same robot and camera construction as `build_http_adapter_backend()`. It checks for a wrist camera, then the key, before `SO101Follower.connect()`. It returns `skip_camera_names={"overhead"}` with `shutdown=None`, so 14-02's teardown runs `robot.disconnect()`, then `close()` on the wrist capture, then `stereo_camera.release()`.
- **Requirement pin:** `google-genai==2.25.0` added to `control/requirements.txt`, appended after 14-03's `ikpy`.

### Task 2: `er_conversion.py` and conformance (commits 82ad933 RED, 99c6001 GREEN)
- **`convert_er_output_to_action()`:**
  - **Target:** the first valid entry wins. Accepted forms are `point [y,x]`, a `{y,x,y2,x2}` box or a `box_2d` box (a box counts as its centre), all on the 0-1000 scale.
  - **Pixel offset:** computed from the image centre against the shape of the frame that was actually sent. An offset within 8 px counts as centred, so the step holds.
  - **Gain and cap:** `PIXELS_TO_EEF_GAIN=5e-5 m/px`, hard-capped at `MAX_NUDGE_M=5 mm`.
  - **Direction:** comes from URDF FK. Image-up maps to radially outward, image-right to the right tangent.
  - **IK delta:** `dz` and all rotation deltas are 0. The gripper stays at its current value.
  - **Output:** `eef_to_joint.solve_ik(..., joint_limits_deg=...)`.
  - **Holds:** any malformed input, bad shape or state, missing heading, or FK/IK failure returns `dict(joint_state)`.
- **`make_er_convert_fn(joint_limits_deg)`:** the closure Plans 14-14 and 14-15 pass as `convert_fn`.
- **`test_backend_registry.py`:**
  - `_action_source_bases()` now includes `GeminiActionSourceBase`, so the parametrized Protocol-signature test covers Bridge, HttpAdapter, Gemini and Scripted.
  - New checks: the conformance list covers all three shared bases, and neither foundation module registers a backend.
- **Tests:** 74 in `test_base_gemini.py`.
  - One drives the **real** pinned SDK against an in-process `httpx.MockTransport`. It checks the request path, body shape and `x-goog-api-key` header, and that the key appears in neither the body nor `last_details`.
  - No test ever calls the real API.

## Verification
- `cd control && <main .venv>/bin/python -m pytest -q`: **909 passed** (fork base 832; +74 `test_base_gemini.py`; `test_backend_registry.py` +1 parametrized case and +2 new tests).
- `run_vla_episode.py --help`'s `--backend` choices are unchanged (13 names), so the foundation modules registered nothing.

## Live-Source Verification Record (execution discipline)
Fetched 2026-10-01 with curl; the installed SDK source was read directly.
- **ai.google.dev/gemini-api/docs/robotics-overview and /robotics-spatial (both "Last updated 2026-09-04"):**
  - **Model ids:** `gemini-robotics-er-2-preview` and `gemini-robotics-er-2-streaming-preview` (Live API only; Structured outputs not supported). **ER 1.6 "will be shut down at the end of August"**, so it is gone.
  - **Request:** the Python example is `client.interactions.create(model=..., input=[{"type":"image",...},{"type":"text",...}], generation_config={"thinking_level":"high"})`, read via `.output_text`.
  - **REST:** `POST https://generativelanguage.googleapis.com/v1beta/interactions` with header `x-goog-api-key`.
  - **Output format:** points are "in [y, x] format normalized to 0-1000". Boxes are `{"y","x","y2","x2"}` on the same scale. Trajectories are `[{"step", "point": [y,x], "action"}]`.
  - **Limitations:** unrestricted API keys get **403**. "For thinking level use medium for a good balance between latency and performance." Input limit 131,072 tokens, output 65,536.
  - **Not published on these pages:** rate limits and latency figures. The pages link to the pricing page, which was not fetched.
- **google-genai 2.25.0 installed source:**
  - `google/genai/_gaos/interactions.py` `create(**body_kwargs, timeout=<s>)`.
  - `types/interactions/imagecontent.py`: `ImageContentParam{type:"image", data, mime_type, uri}`.
  - `ThinkingLevel ∈ {minimal, low, medium, high}`.
  - `types.HttpOptions.timeout` is in **milliseconds**.
  - `Interaction.output_text` comes from the trailing model_output text steps.
- **Measured with a MockTransport probe:** the request body is `{"model", "generation_config", "input":[{"type":"user_input","content":[image, text]}]}` and the key goes only in `x-goog-api-key`. SDK errors (`RateLimitError` / `InternalServerError` / `PermissionDeniedError`) do not echo the key.
  - **Retry behaviour:** with `HttpRetryOptions(attempts=1)`, and even with `attempts=0`, the Interactions path still made **one retry** on 429/5xx and none on 403. Worst-case blocking per step is therefore about 2 x `timeout_s` (10 s default) plus backoff. The robot holds still meanwhile.
- **Repo calibration check:** there are no wrist-camera intrinsics, no camera-to-robot extrinsics, and no camera link in `So-101/So-101.urdf`. `control/stereo_calibration.json` covers only the overhead stereo pair (K, D, R, T). The mapping is therefore an explicit, named assumption, not invented calibration. It is documented in `er_conversion`'s docstring.
- **FK finding:** at all joints 0, the URDF EEF is at (0.087, -0.319, 0.233) m, so the arm reaches along about **-y**. That is why the mapping is heading-relative rather than fixed to base x/y.
- **Unverifiable here:** real ER 2 accuracy and latency on this rig's wrist images, the real wrist-camera mount orientation, and whether this project's key is restricted. All need a live key or the hardware.

## Package Legitimacy Evidence (for the human check)
- **PyPI `google-genai`:**
  - **Publisher:** author "Google LLC <googleapis-packages@google.com>"; Homepage `https://github.com/googleapis/python-genai`; requires Python >= 3.10.
  - **History:** 126 releases since 0.0.1 (2024-12-10). The latest is 2.26.0 (2026-09-30).
  - **Pinned 2.25.0:** uploaded 2026-09-22T17:22:59. Wheel `google_genai-2.25.0-py3-none-any.whl` sha256 `2e8f3a5f76ed00d5ffc3153f6089bc7d3511054eed53ad9cfc6095a7dc59b028`; sdist sha256 `ab603baa5eee0205926ad0f8d7f93e0400df6d67650e99c33e01ab228ea16ad6`.
- **GitHub `googleapis/python-genai`:** created 2024-12-06, Apache-2.0, about 4003 stars and 1024 forks, not archived, pushed 2026-10-01.
- **Installed SDK self-identification:** the installed SDK reports `x-goog-api-client: google-genai-sdk/2.25.0`.
- **Install effect:** installed into the shared venv. `pip check` shows only the pre-existing `ultralytics`/numpy 2.2.6 conflict, which is unrelated.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Wrong SDK entry point in the plan**
- **Found during:** Task 1, while verifying the live docs.
- **Issue:** The plan says `client.models.generate_content(...)`. The current official ER 2 docs (all language samples and REST) use the Interactions API (`client.interactions.create`, `POST /v1beta/interactions`) and read `output_text`.
- **Fix:** Implemented against the documented Interactions call and checked the request shape against the real SDK.
- **Commit:** 5dab568

**2. [Rule 2 - Correctness] Image-plane to base-frame mapping uses the FK heading**
- **Found during:** Task 2.
- **Issue:** The plan maps the pixel offset straight to `(dx, dy)`. But `solve_ik` takes base-frame deltas, and this URDF's forward direction at pan=0 is about -y, rotating with `shoulder_pan`. A fixed x/y mapping would push sideways or backwards as soon as the arm panned.
- **Fix:** The offset maps onto the horizontal heading from FK (outward and right tangent). `dz` and the rotations stay 0, as planned. Sign constants (`IMAGE_UP_SIGN`, `IMAGE_RIGHT_SIGN`) are left for the hardware check. There is also an 8 px deadband, so a centred target does not jitter the IK.
- **Commit:** 99c6001

**3. [Rule 2 - Correctness] `joint_limits_deg` threaded through the conversion**
- **Found during:** Task 2.
- **Issue:** The handoff requires `solve_ik(..., joint_limits_deg=...)`, but the plan's `convert_er_output_to_action` signature had no limits.
- **Fix:** Added an optional `joint_limits_deg` and the `make_er_convert_fn(joint_limits_deg)` closure that subclasses pass as `convert_fn`. `image_shape` defaults to None, so the shape of the frame actually sent (carried in the ER output) wins, falling back to `(480, 640)`.
- **Commit:** 99c6001

**4. [Rule 2 - Security/Safety] Key check before hardware, redaction, latency bound**
- **Order:** `build_gemini_backend()` constructs the action source (key check) before `robot.connect()`.
- **Redaction:** error text is redacted before it enters `last_details`, and `__repr__` hides the key.
- **Latency:** the SDK's default retry policy (up to 4 retries, 30 s elapsed) is replaced by `attempts=1` plus an explicit timeout.
- **Visibility:** a held no-op conversion is flagged in `validator_flags`.
- **Commits:** 5dab568, 99c6001

**5. [Rule 2 - Correctness] Only the wrist image is sent**
- **Issue:** The plan says to capture wrist plus stereo, and that is kept. But sending three images to a pointing model makes "which image is the point in" ambiguous.
- **Fix:** Only the wrist frame is sent, because an image-centre servo only makes sense for the eye-in-hand camera. `er_camera` is a constructor option.
- **Commit:** 5dab568

**6. Version note**
- The plan's `google-genai==2.25.0` was kept, even though 2.26.0 (2026-09-30) is now the latest. 2.25.0 is the version the research audit covered, and it has a week of release history.

## Known Stubs
None.

## Threat Flags
None beyond the plan's register:
- **T-14-11 is mitigated:** the key comes only from the environment, a missing key raises SystemExit, the key is redacted, and tests prove it reaches neither the body nor `last_details`.
- **T-14-12 is mitigated:** the 5 mm hard cap, the URDF and calibrated clamp in `solve_ik`, and the unmodified `safety_validator` downstream.

The new outbound surface is HTTPS to `generativelanguage.googleapis.com`, which D-02 anticipated. Note that each step sends a wrist-camera image to Google, which the ER docs' Privacy Notice covers.

## Integration Notes for Plans 14-14 / 14-15
- **Subclass shape:** `@register("gemini") def build(args, camera_names, joint_limits_deg): return build_gemini_backend(model, "GEMINI_API_KEY", label, args, camera_names, er_conversion.make_er_convert_fn(joint_limits_deg))`. Keep the module free of top-level `google` imports.
- **Model ids:** only **one** public ER model id exists today, `gemini-robotics-er-2-preview`. ER 1.6 is shut down, and the streaming variant is Live-API only, which `interactions.create` does not support. VLAB-01 and VLAB-02 therefore cannot point at different public ER generations. 14-14 has to choose and document how "Gemini Robotics" differs from "Gemini Robotics 2" (for example prompt or thinking level), or share the model id and state that.
- **API key:** must be a *restricted* key, or the API returns 403 (per the docs).
- **`main()` checks keyed on `args.server_address`:** 14-02's note still applies. The device-map camera default and `--stereo-camera-index` name resolution don't apply to a Gemini backend, so the wrist camera must come from `--camera`/`device_map.json` as on the HTTP lane.

## Human Checks Deferred
1. **Package legitimacy (Task 1 `<human-check>`):**
   - **test:** Confirm `pypi.org/project/google-genai` shows version 2.25.0 (the pinned version) and that its source repo is `github.com/googleapis/python-genai`. Evidence is above.
   - **expected:** The installed version matches the official PyPI release, and the repo is Google's own `googleapis` org.
   - **why_human:** Package-legitimacy sign-off is a human call under this project's protocol.
2. **Live API smoke test (new):**
   - **test:** Export a *restricted* `GEMINI_API_KEY`. Construct `GeminiActionSourceBase` with a real wrist frame and `make_er_convert_fn(load_joint_limits_deg())`, or run 14-14's backend, for a few steps.
   - **expected:** `model_version` ends with `@gemini-er-conversion`, the ER point lands on the named object, and step latency is acceptable at `thinking_level="medium"`. If it is too slow, try `"low"`.
   - **why_human:** This needs a real key and billing. The tests never call the API.
3. **Visual-servo direction on hardware (new):**
   - **test:** With the gripper pointing down, wrist_roll at about 0, and the object off-centre in the wrist image, run single steps and watch the arm.
   - **expected:** The object drifts toward the image centre: image-up moves the arm outward and image-right moves it to its right. If an axis is mirrored, flip `er_conversion.IMAGE_UP_SIGN` or `IMAGE_RIGHT_SIGN`. Tune `PIXELS_TO_EEF_GAIN` if convergence is too slow.
   - **why_human:** The wrist-camera mount, the real joint signs against the URDF (carried from 14-03), and the image-centre-equals-gripper-line assumption can only be checked on the rig.
4. **Carried from 14-01/14-03:** the wrist camera must tolerate two independent `cv2.VideoCapture` opens (the backend's own plus IOLogger's), on this lane too.

## Self-Check: PASSED
- FOUND: control/vla_bridge/backends/base_gemini.py, control/vla_bridge/backends/er_conversion.py, control/test_base_gemini.py, control/requirements.txt (google-genai==2.25.0), control/test_backend_registry.py (GeminiActionSourceBase in the conformance list)
- FOUND commits: e7d6b5a, 5dab568, 82ad933, 99c6001
