---
phase: 15-mllm-style-pipeline-backends
plan: 05
subsystem: control/vla_bridge (MLLM inputs and static context)
status: complete
tags: [mllm, pipe-02, depth, urdf, prompt-cli, tdd]
requires:
  - control/vla_bridge/backends/base_mllm.py (MllmInputs, IMAGE_ORDER, load_mllm_settings; unmodified)
  - control/vla_bridge/mllm_prompt.py (build_system_prompt, build_user_text, lint_prompt_text, hashes)
  - control/vla_bridge/mllm_schema.py (build_action_chunk_schema, schema_sha256)
  - control/vla_bridge/depth_camera.py (DepthCameraClient.compute_depth_result, check_ready; unmodified)
  - control/vla_bridge/robot_client.py (_downsize_for_transport; unmodified)
  - So-101/So-101.urdf
provides:
  - vla_bridge.mllm_inputs (IMAGE_ROLES, DEPTH_UNAVAILABLE_NOTE, capture_mllm_inputs, encode_image, depth_stats, depth_grid, depth_grid_text, depth_heatmap, heatmap_legend_labels, fetch_depth, MllmInputCollector, main preview CLI)
  - vla_bridge.mllm_urdf (URDF_PATH, urdf_kinematics_text)
  - vla_bridge.mllm_prompt.main (prompt-inspection CLI) and EXAMPLE_DEPTH_NOTE
  - fake_providers.FakeDepthClient, fake_providers.make_depth_array
affects:
  - Plan 15-06 (hardware builder wires MllmInputCollector and urdf_kinematics_text into MllmActionSourceBase)
  - Plan 15-10 (live depth check via python -m vla_bridge.mllm_inputs; human URDF sign check)
  - Phase 16 ablations (left_image_source, depth_presentation, include_urdf are now real toggles)
tech-stack:
  added: []
  patterns:
    - one synchronous depth request per chunk on raw 800x600 frames, 429-only retries, fixed failure note
    - heatmap, grid text, depth.npy and aligned_left all derived from the same depth result
    - lazy MllmInputs / load_mllm_settings imports inside functions to avoid circular backend discovery
key-files:
  created:
    - control/vla_bridge/mllm_inputs.py
    - control/vla_bridge/mllm_urdf.py
    - control/test_mllm_inputs.py
    - control/test_mllm_urdf.py
  modified:
    - control/vla_bridge/mllm_prompt.py
    - control/test_mllm_prompt.py
    - control/fake_providers.py
decisions:
  - "The depth image sent to the model and the saved depth_heatmap.png are the same downsized heatmap (legend included, longest edge <= image_max_side), so the sidecar shows exactly what the model saw"
  - "The heatmap body is shrunk with INTER_NEAREST to fit image_max_side minus the 96 px legend strip; colour bar runs near (blue, top) to far (red, bottom) with 5 range_m ticks and a black n/a swatch"
  - "A 429 is detected with a word-boundary regex on the error text; only 429 is retried, request_ms covers all attempts and sleeps"
  - "Depth requests carry request_id mllm-depth-NNNNNN (echoed by FastFS and checked by the client); meta.depth.request_id and meta.image_shapes are extra keys beyond the 15-01 contract"
  - "URDF text includes fixed joints (no limits) and the gripper_right mimic; the jaw range in the header is derived from the prismatic limits, not hard-coded"
  - "Adding mllm_prompt.main() changes no template constant, so PROMPT_TEMPLATE_VERSION stays mllm-prompt-1 and the golden template hash is unchanged"
metrics:
  duration: "~25 min"
  completed: 2026-10-02
  tasks: 2
  tests_added: 62
  full_suite: "1195 passed (1133 baseline + 62 new)"
---

# Phase 15 Plan 05: MLLM inputs, URDF text and inspection CLIs Summary

`MllmInputCollector` turns one live capture into what the model is shown on each call: wrist, left and right frames downsized for transport, a turbo depth heatmap with a legend, and a 24x32 median-pooled depth grid in metres. All of this comes from one depth request on the raw 800x600 stereo pair. `urdf_kinematics_text()` renders So-101.urdf as a 1.9K-character joint table that states only verified conventions. Two keyless CLIs show what the model sees without calling it: `python -m vla_bridge.mllm_prompt` prints the prompt, and `python -m vla_bridge.mllm_inputs` captures one live set of inputs.

## What was built

- **`mllm_inputs.py`**
  - `capture_mllm_inputs`: calls `grab()` 3 times to flush the wrist camera before `read()` (Pitfall 8). Then calls `read_left()` and `read_right()` back to back, so both come from one physical AR0144 read. Returns the raw arrays, `captured_utc` and `stereo_pair_metadata`, or `None` if any read fails.
  - `fetch_depth`: one `compute_depth_result()` call on the unmodified raw arrays (an identity test checks this). HTTP 429 is retried up to `retry_429` times, with `sleep_fn(retry_sleep_s)` between tries. No other error is retried. It never raises, and it records `attempts` and `request_ms`.
  - `depth_grid` / `depth_grid_text`: median pooling over integer `linspace` block edges. A pixel is valid when it is finite and > 0. A cell with no valid pixels becomes `n/a`. The text has a header, a min/median/max/valid% line computed over the full array, and 24 `rNN:` rows. The real-shaped grid is about 4.2K characters.
  - `depth_heatmap`: clipped to `range_m`, invalid pixels pure black, resized with nearest-neighbour, plus a legend strip whose tick labels come from `range_m`.
  - `MllmInputCollector.collect`:
    - `depth_presentation` heatmap/grid/both decides whether the `depth` image is sent and what the note contains.
    - `left_image_source` raw/aligned picks the left frame; aligned falls back to raw when depth failed, and `meta.left_image_source_sent` records what was actually sent.
    - When depth succeeds, the sidecar always gets `depth.npy`, `depth_heatmap.png`, `depth_grid.txt` and `aligned_left.png`.
    - When depth fails or is not configured, the note is exactly `Depth unavailable this call.`; the error text goes only into meta, cut to 300 characters.
    - `meta` follows the 15-01 contract.
  - `main()` preview CLI: health-checks the depth endpoint, does one capture, and writes `images/*.png`, the sidecars, `depth_note.txt` and `meta.json`, printing a `Saved → ` line for each file.
- **`mllm_urdf.py`**: parses only `<joint>` elements with `xml.etree`. Each line has the joint type, `parent -> child`, origin xyz (m) and rpy (deg), axis, and limits (degrees for revolute, metres for prismatic). The header says: lengths are in metres and angles in degrees; URDF angle 0 is 0 degrees in the targets; the jaws slide 0 to 0.036 m; the gripper channel runs from 0 percent closed to 100 percent open. It says nothing about the direction of positive rotation (C8). A missing file raises `FileNotFoundError` with the path.
- **`mllm_prompt.main()`**: `--config`, `--chunk-len`, `--no-urdf`, `--joint-calibration`, `--instruction`, `--depth-presentation`. It prints the template version, the template, system-prompt and schema sha256 hashes, `include_urdf`, the full system prompt, an example user text with the joints at their midpoints, and the schema JSON.
- **`fake_providers.py`**: added `FakeDepthClient` (scripted results: None, an array, an error string, a dict or an exception; a call log; the received arrays; `check_ready`) and `make_depth_array`. The existing fakes are unchanged.

## Verification

- `pytest test_mllm_inputs.py test_mllm_urdf.py test_mllm_prompt.py`: 95 passed. That is 48 inputs tests, 8 URDF tests, and 6 new prompt tests on top of the existing ones.
- Full suite: 1195 passed in 9.2 s.
- `python -c "import vla_bridge.mllm_inputs"` in a fresh interpreter exits 0. An AST test also checks that the module has no top-level `backends` import.
- `python -m vla_bridge.mllm_prompt --chunk-len 5` with the real `soarm_follower_02.json` exits 0, and its output contains `ARM KINEMATICS (URDF)` and `mllm-prompt-1`.
- `len(urdf_kinematics_text()) == 1910`, with 10 table lines for the 10 joint elements.
- I rendered a heatmap and looked at it: the legend is legible, the NaN block is black, and the colours run blue (near) to red (far).

## TDD Gate Compliance

- Task 1: RED `b1c23d8` (`ImportError: cannot import name 'mllm_inputs'`), then GREEN `e7b044f`.
- Task 2: RED `06195eb` (`ModuleNotFoundError: vla_bridge.mllm_urdf`), then GREEN `fd39cc4`.

## Deviations from Plan

None. These choices fall within the plan's discretion:
- The collector also writes `meta.depth.request_id` and `meta.image_shapes`, beyond the 15-01 contract keys. The base record only copies `meta["depth"]` and the capture keys, so nothing downstream breaks.
- The preview CLI exits with `SystemExit` if the depth endpoint fails its health check, and returns 1 (writing nothing) if the camera read fails.

Environment: tests ran with the main checkout's venv interpreter from the worktree's `control/`. Nothing was installed, and no real camera, robot or depth service was contacted.

## Known Stubs

None.

## Threat Flags

None. No new network endpoints. The preview CLI calls the existing FastFS `/health` and `/depth` endpoints through the existing client, and writes files only under the `--out` directory the user gives. T-15-14 (failure note never contains the URL or error text) and T-15-15 (one depth call per chunk, 429-only retries, never blocking) are mitigated and tested. T-15-17: the URDF text makes no claim about joint sign, which a test asserts.

## Self-Check: PASSED

- Files exist: `control/vla_bridge/mllm_inputs.py`, `control/vla_bridge/mllm_urdf.py`, `control/test_mllm_inputs.py`, `control/test_mllm_urdf.py`, and the modified `mllm_prompt.py`, `test_mllm_prompt.py` and `fake_providers.py`.
- Commits `b1c23d8`, `e7b044f`, `06195eb` and `fd39cc4` are present in `git log`.
