# Phase 15: MLLM-Style Pipeline & Backends - Pattern Map

**Mapped:** 2026-10-02
**Files analyzed:** 27 (new/modified; list from 15-RESEARCH.md "Recommended Project Structure" plus two run_vla_episode.py edits)
**Analogs found:** 25 / 27 (2 partial or no analog)

All paths are under `/Users/hp/Desktop/Work/Repositories/SoARM-Research/control/` unless noted.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `vla_bridge/backends/base_mllm.py` (MllmActionSourceBase, build_mllm_backend, load_mllm_backend_config) | service (action source base) | request-response, chunk-queue | `vla_bridge/backends/base_gemini.py` | exact (role) / chunk-queue is new |
| `vla_bridge/backends/mllm_claude.py` (+ AnthropicAdapter) | backend module | request-response | `vla_bridge/backends/gemini_robotics.py` | exact |
| `vla_bridge/backends/mllm_gpt.py` (+ OpenAIResponsesAdapter) | backend module | request-response | `vla_bridge/backends/gemini_robotics.py` | exact |
| `vla_bridge/backends/mllm_hf_qwen.py` / `mllm_hf_deepseek.py` / `mllm_hf_kimi.py` | backend module | request-response | `vla_bridge/backends/gemini_robotics.py`, `gemini_robotics_2.py` (sibling variants) | exact |
| `vla_bridge/backends/base_mllm_chat.py` (OpenAIChatAdapter, optional) | service (adapter) | request-response | `base_gemini._make_genai_client` / `_request` | role-match |
| `vla_bridge/mllm_schema.py` | utility | transform | `vla_bridge/action_contract.py` (source of constants) | partial |
| `vla_bridge/mllm_prompt.py` | utility | transform | `base_gemini.DEFAULT_ER_PROMPT_TEMPLATE` + `action_contract.py` | partial |
| `vla_bridge/mllm_record.py` (redact, usage, cost, sidecar) | utility | file-I/O | `io_logger.IOLogger.write_depth_result` / `capture_*` | role-match |
| `vla_bridge/backend_configs/mllm_defaults.json` | config | n/a | `backend_configs/gemini_robotics.json` | role-match |
| `vla_bridge/backend_configs/mllm_{claude,gpt,hf_qwen,hf_deepseek,hf_kimi}.json` | config | n/a | `backend_configs/gemini_robotics.json` | exact |
| `run_vla_episode.py` (task_complete hook; depth parser rule) | modified entry | event-driven loop | itself, lines 276-296, 548-554 | exact (edit in place) |
| `test_mllm_schema.py`, `test_mllm_prompt.py`, `test_mllm_record.py` | test | transform | `test_base_gemini.py` (style) | role-match |
| `test_base_mllm.py` | test | request-response | `test_base_gemini.py` | exact |
| `test_backend_mllm_<name>.py` (x5) | test | request-response | `test_backend_gemini_robotics.py` | exact |
| `test_run_vla_episode.py` (update 1 test, add hook test) | test | modified | `test_main_errors_when_depth_flags_given_without_server_address` (line 1504) | exact |
| `requirements.txt` (add anthropic, openai) | config | n/a | Plan 14-13 google-genai pin comments in same file | exact |

## Pattern Assignments

### `vla_bridge/backends/base_mllm.py` (action source base, request-response with internal chunk deque)

**Analog:** `vla_bridge/backends/base_gemini.py` (450 lines). Reuse its structure wholesale; add a deque and the multi-image/depth capture.

**Imports pattern** (base_gemini.py lines 77-94):
```python
import base64, json, math, numbers, os, re, threading
import cv2
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
from vla_bridge import safety_validator
from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import BackendHandles, load_backend_config
from vla_bridge.backends.base_http_adapter import _is_joint_action, capture_observation_images
from vla_bridge.robot_client import OBSERVATION_MAX_SIDE
from vla_bridge.stereo_camera import StereoSplitCamera
```
Heavy SDKs (`anthropic`, `openai`) are imported lazily inside factory functions, as `_make_genai_client` does (lines 136-147).

**Secret discipline at construction** (lines 212-228):
```python
api_key = (os.environ.get(api_key_env) or "").strip()
if not api_key:
    raise SystemExit(f"Environment variable {api_key_env} is not set")
...
self._api_key = api_key  # only used to redact error text; never logged
self._client = _make_genai_client(api_key, timeout_s)
self._inflight_lock = threading.Lock()
self.last_details = self._details({})
```
Also copy `__repr__` (lines 231-233: never prints the key) and `_redact` (244-245).

**Hold-on-failure pattern** (lines 247-253):
```python
def _hold(self, joint_state, reason: str, error: BaseException | None = None):
    raw = self.last_details.get("raw_model_output", {})
    self.last_details = self._details(raw, [reason])
    if error is not None:
        self.last_details["error_type"] = type(error).__name__
        self.last_details["error"] = self._redact(str(error))[:MAX_LOGGED_TEXT_CHARS]
    return dict(joint_state), f"{self.checkpoint_label}@{reason}"
```
For MLLM: pop a `Hold(reason)` sentinel from the deque and return exactly this tuple shape (`@<reason>-holding-position`).

**get_action skeleton** (lines 269-304): non-blocking lock acquire -> `_hold("...-inflight-holding-position")`; `capture_observation_images(self.wrist_cap, self.stereo_camera, self.max_side)` returning `None` -> `camera-read-failed-holding-position`; `try/except Exception` around the provider call -> hold with error; `_is_joint_action(converted)` check before returning; `finally: self._inflight_lock.release()`. In base_mllm, only refill when the deque is empty; always pop one item per tick. Whole-call failure enqueues N `Hold` items (no retry, no per-tick paid calls).

**Per-step shape check** (base_http_adapter.py lines 93-100): use `_is_joint_action` (rejects bool and non-Real) plus `math.isfinite` for D-03 pre-validation; first bad index k makes k..N-1 `Hold`.

**Image capture** (base_http_adapter.py lines 71-90): `capture_observation_images` returns b64 PNG dict `{"wrist","overhead_left","overhead_right"}`, reads wrist plus ONE stereo pair. Depth needs raw left/right frames, which this helper does not return (it returns encoded strings). Planner: either add a sibling helper that returns raw frames and encodes afterward, or read `stereo_camera.read_left()/read_right()` directly in base_mllm (same calls as lines 79-81). Do not modify `capture_observation_images`'s existing contract (other backends use it).

**Config loader + validation** (base_gemini.py lines 313-376 `load_er_backend_config`): copy the structure for `load_mllm_backend_config`. Key idioms: `where = f"control/vla_bridge/backend_configs/{config_name}.json"`; `load_backend_config(name)` returns `None` on failure -> `SystemExit` naming the field; `_ENV_VAR_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")` rejects a pasted key without echoing it; numeric validation `isinstance(timeout_s, bool) or not isinstance(timeout_s, numbers.Real) or not math.isfinite(timeout_s) or timeout_s <= 0`. Merge `mllm_defaults` first (loads through `load_backend_config("mllm_defaults")`, which accepts the name pattern), then the per-backend overrides.

**Builder** (base_gemini.py lines 379-437 `build_gemini_backend`): copy ordering exactly. Resolve wrist index from `camera_names` (`next(idx for idx, name in camera_names.items() if name == "wrist")` else `SystemExit`); build the action source (key check) BEFORE `robot.connect()`; then:
```python
robot = SO101Follower(SOFollowerRobotConfig(port=args.port, id=args.robot_id,
        max_relative_target=safety_validator.MAX_RELATIVE_TARGET_DEG))
robot.connect(calibrate=False)
with robot.bus.torque_disabled():
    robot.bus.write_calibration(robot.calibration)
stereo_camera = StereoSplitCamera(index=args.stereo_camera_index)
wrist_cap = cv2.VideoCapture(wrist_idx)
...
return BackendHandles(action_source=action_source, robot=robot, stereo_camera=stereo_camera,
                      skip_camera_names=frozenset({"overhead"}))
```
For depth (D-06), additionally construct `DepthCameraClient(load_calibration(args.depth_calibration), args.depth_endpoint)` as `run_vla_episode.py` line 611-612 does, but inside the backend and only when `args.depth_endpoint` is set. `close()` pattern: release `wrist_cap` (lines 306-310).

**Depth call** (depth_camera.py lines 153-192): `DepthCameraClient.compute_depth_result(left_frame, right_frame, request_id=None)` returns a dict with `depth` (float32 metres, NaN invalid) or `depth=None` plus `error`; it never raises, and also returns `aligned_left`, `valid_fraction`, `inference_ms`. Use it synchronously per chunk; on `depth is None`, continue without depth and log the error in the call record (do not hold the entire chunk unless config says so). The FastFS server 429s on overlapping requests (RESEARCH C5).

### `vla_bridge/backends/mllm_claude.py`, `mllm_gpt.py`, `mllm_hf_*.py` (backend modules)

**Analog:** `vla_bridge/backends/gemini_robotics.py` (117 lines; ~20 lines of code after a long docstring). Sibling-variant precedent: `gemini_robotics_2.py` (same shape, different `BACKEND_NAME`/`CONFIG_NAME`/`CHECKPOINT_LABEL`).

**Full pattern** (gemini_robotics.py lines 79-117):
```python
from vla_bridge.backends import BackendHandles, base_gemini, er_conversion, register

BACKEND_NAME = "gemini-robotics"
CONFIG_NAME = "gemini_robotics"
CHECKPOINT_LABEL = "gemini-robotics"

@register(BACKEND_NAME)
def build(args, camera_names: dict[int, str], joint_limits_deg: dict) -> BackendHandles:
    model, api_key_env, source_kwargs = base_gemini.load_er_backend_config(CONFIG_NAME, BACKEND_NAME)
    return base_gemini.build_gemini_backend(model, api_key_env, CHECKPOINT_LABEL, args,
                                            camera_names, convert_fn=..., **source_kwargs)

__all__ = ["BACKEND_NAME", "CHECKPOINT_LABEL", "CONFIG_NAME", "build"]
```
Module docstring convention: document model-id verification with source/date (as gemini_robotics.py does with its "verified, not guessed" block), config keys, transport, and secret rules. Names per RESEARCH: `mllm-claude`, `mllm-gpt`, `mllm-hf-qwen`, `mllm-hf-deepseek`, `mllm-hf-kimi`. `joint_limits_deg` is passed to `build()` and must flow into the prompt (D-07). Module names must NOT start with `base_` or `_` (those are skipped by discovery; `base_mllm.py` and `base_mllm_chat.py` are correctly skipped).

### `vla_bridge/mllm_schema.py` (utility, transform)

**Analog:** `vla_bridge/action_contract.py` supplies inputs (`JOINT_ORDER` lines 28-35, `ACTION_UNITS` lines 39-46, `ACTION_MODE = "absolute"` line 51). No existing JSON-schema generator; follow RESEARCH Code Example 1 / Pattern 2 (loop over `JOINT_ORDER`; units in property `description`; all required; `additionalProperties:false`; no min/max; key order task_restatement, plan, scene_claims, actions, task_complete; `sha256` of canonical JSON). Style: module docstring, `UPPER_SNAKE_CASE` constants, no heavy imports.

### `vla_bridge/mllm_prompt.py` (utility, transform)

**Analog:** `base_gemini.py` lines 105-114 (`DEFAULT_ER_PROMPT_TEMPLATE`; substitute with `str.replace`, never `str.format`, because the template contains JSON braces) and lines 116-117 (`MAX_LOGGED_TEXT_CHARS`). Add `PROMPT_TEMPLATE_VERSION` and a hash. Units/limits come from `action_contract` + the `joint_limits_deg` dict. URDF text parsed with `xml.etree.ElementTree` from `/Users/hp/Desktop/Work/Repositories/SoARM-Research/So-101/So-101.urdf`. Include a unit test linting for forbidden words (reasoning/thinking/trace/scratchpad/chain-of-thought) per RESEARCH C4.

### `vla_bridge/mllm_record.py` (utility, file-I/O)

**Analog:** `vla_bridge/io_logger.py`. Sidecar precedent: `write_depth_result` (lines 79-98) writes under `self.out_dir` using a temp file then `replace` (atomic) and appends to a `.jsonl`; `capture_*` methods return small metadata dicts rather than inline blobs. Copy: `temporary = path.with_suffix('.json.tmp'); temporary.write_text(json.dumps(record, allow_nan=False)); temporary.replace(path)`. Note `IOLogger.write_step(..., action_details=...)` (lines 45-77) stores `action_details` inline in `episode.jsonl`, so the per-chunk record must be a compact dict in `last_details` on the first step of a chunk with `chunk_ref` on every step, and raw payloads go to sidecar files. `run_vla_episode.py` line 285-287 pops `observation` and `raw_model_output` from `last_details` and passes the remainder as `action_details`; keep the big stuff out of those keys. The base class has no out_dir, so the sidecar directory must be injected (new constructor/attribute set by the loop or derived from args.out); planner to decide.

### `run_vla_episode.py` (modified)

**Edit 1: termination hook** inside the loop (lines 276-296 currently):
```python
reason = "max_steps_reached"
try:
    for i in range(max_steps):
        ...
        raw_action, model_version = action_source.get_action(current, instruction)
        details = dict(getattr(action_source, "last_details", {}))
```
Add a duck-typed check using the same `getattr` style (e.g. `getattr(action_source, "task_complete_claimed", False)`) after the step is executed/logged, set `reason = "task_complete"` and `break`. The existing `finally` block (lines ~388-406) already handles return-to-start and calls `_write_termination(io_logger.out_dir, reason, steps_completed)` (def at 208-211).

**Edit 2: depth parser rule** (lines 548-554):
```python
if (args.depth_endpoint is None) != (args.depth_calibration is None):
    parser.error("--depth-endpoint and --depth-calibration must be given together")
if args.depth_endpoint and not args.server_address:
    parser.error("--server-address is required when --depth-endpoint is given ...")
```
Scope the second check to non-MLLM backends. Also lines 611-612 build `DepthCameraClient` and line 268 starts `DepthRecorder` for `depth_client`; for MLLM backends the backend owns depth, so avoid double calls (429 risk).

### Tests

**Analog:** `test_base_gemini.py` (header lines 1-50: fake wrist cap, `SECRET`, `JOINT_STATE`, `MockRobot` from `conftest`, fake SDK client, one test driving the real SDK via `httpx.MockTransport`) and `test_backend_gemini_robotics.py` (lines 1-60: `_args(**overrides)` SimpleNamespace, `config_dir` fixture monkeypatching `backends.BACKEND_CONFIG_DIR` to `tmp_path`, `MODULE_PATH`/`SHIPPED_CONFIG_PATH` constants, AST-based test of module imports to enforce lazy heavy imports, `CAMERA_NAMES = {5: "wrist", 3: "overhead"}`, `JOINT_LIMITS_DEG`). Also assert the secret never appears in `last_details`/`episode.jsonl`. `test_run_vla_episode.py` line 1504: `test_main_errors_when_depth_flags_given_without_server_address` (pins the rule that must be scoped); update it and add an MLLM-backend counterpart.

### Config JSON

**Analog:** `vla_bridge/backend_configs/gemini_robotics.json`: required `model`, `api_key_env` (env-var NAME only), optional tuning keys (`thinking_level`, `timeout_s`), plus provenance fields (`requirement`, `model_verified` with date and source, `notes`). Mirror: per-backend `model`, `api_key_env`, `effort`, `max_tokens`, `timeout_s`, `prices` (dated), HF `model:provider` pin, `chunk_len` override with derivation in `notes`, `step_dt_s: 0.785` with its evidence (D-24), `include_urdf`, `left_image_source`, `disclose_safety_caps`. Shared values go in `mllm_defaults.json`.

## Shared Patterns

### Secrets
**Source:** `base_gemini.py` lines 61-68 docstring, 212-214, 244-245, 351-361.
**Apply to:** `base_mllm`, all adapters, `mllm_record` (sidecar redaction), all tests.
Key read only from the named env var; `SystemExit` before any hardware is touched; never in `last_details`, `__repr__`, sidecars; error text passes through `_redact` and is truncated.

### Never-raises hold contract
**Source:** `base_gemini.py` lines 247-253 and 269-304.
**Apply to:** `get_action` in `base_mllm`. Failure returns `(dict(joint_state), "<label>@<reason>-holding-position")`; `run_episode()` still runs the unmodified `safety_validator.validate_action` (run_vla_episode.py lines 289-296).

### Registry / discovery
**Source:** `vla_bridge/backends/__init__.py` lines 36-120 (`BackendHandles`, `register`, `load_backend_config`, `_discover_backends` skipping `base_`/`_`).
**Apply to:** all backend modules. No edits to `__init__.py`. NOTE: its docstring (lines 31-33) says nothing in this package touches `depth_camera.py`; the MLLM base importing `DepthCameraClient` is a deliberate D-06 exception; update that docstring/comment in the same change.

### Config-error style
**Source:** `base_gemini.load_er_backend_config` (lines 330-376): `where` string, field-named `SystemExit` messages, never echoing a pasted key.
**Apply to:** `load_mllm_backend_config` and all config validation.

### Logging conventions (CLAUDE.md)
Module docstring with CLI usage and contract; `print()` progress; `Saved → path`; `pathlib.Path`; `snake_case`; `UPPER_SNAKE_CASE` constants.

## No Analog Found

| File / Concern | Role | Data Flow | Reason |
|---|---|---|---|
| Chunk deque with `Step`/`Hold` sentinels inside an action source | service | queue-drain | No existing source buffers multiple steps internally (Phase 12 chunk-drain lives in the gRPC bridge, `base_grpc.py`/`robot_client.py`, server side; planner may skim `robot_client.py` for naming but it is not the same pattern). Use RESEARCH Pattern 1 and 5. |
| Provider adapters (Anthropic Messages stream, OpenAI Responses stream, OpenAI chat via `with_raw_response`) | service | request-response | No existing Anthropic/OpenAI client in repo; only google-genai in `base_gemini._make_genai_client`. Use RESEARCH Standard Stack and Provider Facts; verify against installed SDK source (MEMORY: verify against live source). |
| JSON-schema generator and URDF text formatter | utility | transform | Nothing comparable; only the `action_contract` constants are reusable. |

## Metadata

**Analog search scope:** `control/vla_bridge/` (backends, backend_configs, io_logger, depth_camera, action_contract), `control/run_vla_episode.py`, `control/test_*gemini*.py`, `control/test_run_vla_episode.py`.
**Files read:** base_gemini.py (full), gemini_robotics.py, gemini_robotics.json, backends/__init__.py, parts of base_http_adapter.py, io_logger.py, depth_camera.py, run_vla_episode.py, action_contract.py, test_base_gemini.py, test_backend_gemini_robotics.py.
**Pattern extraction date:** 2026-10-02
