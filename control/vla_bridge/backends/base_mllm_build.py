"""Shared hardware builder and strict config validator for every MLLM backend (Phase 15, Plan 06).

What this is
============
Every MLLM backend module (`mllm_claude.py`, and the GPT and HF modules of
Plans 15-07/15-08) is one adapter class plus one config file. Its registered
`build(args, camera_names, joint_limits_deg)` is a one-line call to
`build_mllm_backend()`, which:

  1. validates `backend_configs/<config_name>.json` merged over
     `mllm_defaults.json` (`load_mllm_backend_config`);
  2. reads the API key from the environment variable NAMED in the config;
  3. builds the provider adapter through the registered factory;
  4. builds the optional FastFS depth client and checks that it is ready;
  5. renders the URDF kinematics text (unless `include_urdf` is false) and
     creates the per-call `CallRecorder` under `<args.out>/mllm_calls`;
  6. prints an operator NOTICE naming the provider, the model and the
     config's data-retention note (camera images and depth leave this machine);
  7. only then connects the robot, opens the stereo camera and the wrist
     capture, and assembles `MllmInputCollector` + `MllmActionSourceBase`.

Build-ordering guarantee (D-00b, research Pitfall 14)
-----------------------------------------------------
Every failure that can be detected without hardware raises `SystemExit`
BEFORE `SO101Follower` is constructed: an unknown backend name, an invalid
config, an unset or blank key variable, no camera named "wrist", only one of
`--depth-endpoint`/`--depth-calibration`, or a depth service that is not
ready. The robot is never touched on a run that cannot make a call.

Secret discipline (T-15-18)
---------------------------
The config holds only the NAME of the key variable. `api_key_env` must match
`^[A-Z_][A-Z0-9_]*$`, so a pasted key is rejected and its value is never
echoed. The key goes to exactly three places: the adapter factory (which
gives it to the provider SDK client), the base (`secrets=`, for error
redaction) and the recorder (`secrets=`, for record and sidecar redaction).
It is never printed, never in `last_details`, a `repr()`, a record or a
sidecar.

Depth (D-06)
------------
`run_vla_episode.main()` gives MLLM backends no depth client (FastFS answers
429 to overlapping requests). This builder owns the single client, built from
the same `--depth-endpoint`/`--depth-calibration` flags, and writes
`depth_calibration.json` into `--out` exactly as main() does for other
backends. Without the flags the model gets no depth image and the prompt
carries the fixed "Depth unavailable" note.

Unlike `base_mllm.py` (pure orchestration), this module imports `lerobot`
and `cv2` at top level, like `base_gemini.py`: it is the hardware edge. It
imports no provider SDK; adapters import theirs lazily inside their factory.
The `base_` prefix keeps backend discovery from importing it on its own.
"""

import json
import math
import numbers
import os
import re
from pathlib import Path

import cv2
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO101Follower

from vla_bridge import safety_validator
from vla_bridge.backends import BackendHandles, base_mllm, load_backend_config
from vla_bridge.depth_camera import DepthCameraClient
from vla_bridge.mllm_inputs import MllmInputCollector
from vla_bridge.mllm_record import CallRecorder
from vla_bridge.mllm_urdf import urdf_kinematics_text
from vla_bridge.stereo_calibration import load_calibration
from vla_bridge.stereo_camera import StereoSplitCamera

PROVIDERS = ("anthropic", "openai", "huggingface")
MIN_MAX_TOKENS = 1024

# An environment-variable NAME, never a key (same rule as base_gemini).
_ENV_VAR_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def _config_where(config_name: str) -> str:
    return f"control/vla_bridge/backend_configs/{config_name}.json"


def _is_bool(value) -> bool:
    return isinstance(value, bool)


def _is_price(value) -> bool:
    return (not isinstance(value, bool) and isinstance(value, numbers.Real)
            and math.isfinite(value) and value >= 0)


def _non_blank_str(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def load_mllm_backend_config(config_name: str, backend_name: str) -> dict:
    """Merged and validated settings for one MLLM backend.

    `backend_configs/<config_name>.json` is merged over `mllm_defaults.json`
    by `base_mllm.load_mllm_settings` (which validates the shared keys), then
    the provider keys are checked here. Raises `SystemExit` naming the file
    and the bad field; a pasted key in `api_key_env` is never echoed.
    """
    where = _config_where(config_name)
    if load_backend_config(config_name) is None:
        raise SystemExit(
            f"--backend {backend_name} needs {where} (a JSON object with 'provider', 'model', "
            "'api_key_env', 'prices', 'capabilities' and 'data_retention_note'); it is missing "
            "or unreadable."
        )
    settings = base_mllm.load_mllm_settings(config_name)

    def fail(field_name: str, requirement: str):
        raise SystemExit(f"{where}: '{field_name}' must be {requirement}.")

    if settings.get("provider") not in PROVIDERS:
        fail("provider", f"one of {', '.join(PROVIDERS)}")
    if not _non_blank_str(settings.get("model")):
        fail("model", "a non-blank model id string")

    api_key_env = settings.get("api_key_env")
    if not isinstance(api_key_env, str) or not _ENV_VAR_NAME_RE.match(api_key_env):
        # Never echo the value: it may be a pasted key.
        raise SystemExit(
            f"{where}: 'api_key_env' must be the uppercase NAME of an environment variable "
            "(e.g. ANTHROPIC_API_KEY), never the key itself. If a key was pasted here, remove "
            "it, rotate it, and export it under that variable instead."
        )

    capabilities = settings.get("capabilities")
    if not isinstance(capabilities, dict):
        fail("capabilities", "an object with a boolean 'sampling_supported'")
    if not _is_bool(capabilities.get("sampling_supported")):
        fail("capabilities.sampling_supported", "true or false")
    if "summary_supported" in capabilities and not _is_bool(capabilities["summary_supported"]):
        fail("capabilities.summary_supported", "true or false when present")
    effort_levels = capabilities.get("effort_levels")
    if effort_levels is not None and (
            not isinstance(effort_levels, list) or not all(_non_blank_str(v) for v in effort_levels)):
        fail("capabilities.effort_levels", "a list of effort names or null")

    effort = settings.get("effort")
    if effort is not None:
        if not _non_blank_str(effort):
            fail("effort", "a non-blank effort name or null")
        if effort_levels is not None and effort not in effort_levels:
            fail("effort", f"one of capabilities.effort_levels {effort_levels}")

    max_tokens = settings.get("max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < MIN_MAX_TOKENS:
        fail("max_tokens", f"an int >= {MIN_MAX_TOKENS}")

    prices = settings.get("prices")
    if not isinstance(prices, dict):
        fail("prices", "an object with 'input_per_mtok', 'output_per_mtok' and 'verified_on'")
    for key in ("input_per_mtok", "output_per_mtok"):
        if not _is_price(prices.get(key)):
            fail(f"prices.{key}", "a finite number >= 0 (USD per million tokens)")
    cached = prices.get("cached_input_per_mtok")
    if cached is not None and not _is_price(cached):
        fail("prices.cached_input_per_mtok", "a finite number >= 0 or null")
    if not _non_blank_str(prices.get("verified_on")):
        fail("prices.verified_on", "a non-blank date string (when the prices were checked)")

    if not _non_blank_str(settings.get("data_retention_note")):
        fail("data_retention_note", "a non-blank note on how the provider retains request data")
    return settings


def _build_depth_client(args, out_dir: Path, api_key: str):
    """The backend-owned FastFS client, or None with a WARNING (D-06)."""
    endpoint = getattr(args, "depth_endpoint", None)
    calibration_path = getattr(args, "depth_calibration", None)
    if (endpoint is None) != (calibration_path is None):
        raise SystemExit("--depth-endpoint and --depth-calibration must be given together (or both omitted).")
    if endpoint is None:
        print("WARNING: no --depth-endpoint/--depth-calibration given; no depth image will be sent "
              "to the model (the prompt says depth is unavailable).")
        return None
    depth_client = DepthCameraClient(load_calibration(calibration_path), endpoint)
    try:
        depth_client.check_ready()
    except Exception as exc:
        detail = str(exc).replace(api_key, "<redacted>") if api_key else str(exc)
        raise SystemExit(
            f"Depth service at --depth-endpoint is not ready ({type(exc).__name__}: {detail}); "
            "start it or omit the depth flags. The robot was not touched."
        ) from None
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "depth_calibration.json").write_text(json.dumps(depth_client.calibration_metadata, indent=2))
    return depth_client


def build_mllm_backend(backend_name: str, args, camera_names: dict[int, str],
                       joint_limits_deg: dict) -> BackendHandles:
    """Build the full MLLM action source for `backend_name` and connect the robot last.

    `args` needs `.port`, `.robot_id`, `.stereo_camera_index` and `.out`;
    `.depth_endpoint`/`.depth_calibration` are optional. See the module
    docstring for the fail-closed ordering.
    """
    spec = base_mllm.MLLM_BACKENDS.get(backend_name)
    if spec is None:
        raise SystemExit(f"{backend_name!r} is not a registered MLLM backend "
                         f"(known: {', '.join(sorted(base_mllm.MLLM_BACKENDS)) or 'none'}).")
    config = load_mllm_backend_config(spec.config_name, backend_name)

    wrist_idx = next((idx for idx, name in camera_names.items() if name == "wrist"), None)
    if wrist_idx is None:
        raise SystemExit(
            f"MLLM backend {backend_name!r} needs a wrist camera "
            "(first --camera index / device_map.json wrist entry)."
        )

    env_name = config["api_key_env"]
    api_key = (os.environ.get(env_name) or "").strip()
    if not api_key:
        raise SystemExit(
            f"Environment variable {env_name} is not set (or is blank); --backend {backend_name} "
            "needs it. Export the key under that name; never put it in the config file."
        )

    adapter = spec.adapter_factory(api_key, config)

    out_dir = Path(args.out)
    depth_client = _build_depth_client(args, out_dir, api_key)
    urdf_text = urdf_kinematics_text() if config["include_urdf"] else None
    recorder = CallRecorder(out_dir / "mllm_calls", secrets=[api_key], prices=config["prices"])

    print(f"NOTICE: --backend {backend_name} sends the task prompt, camera images and depth data "
          f"to provider {config['provider']!r} (model {config['model']!r}) over HTTPS.")
    print(f"NOTICE: data retention: {config['data_retention_note']}")

    robot = SO101Follower(
        SOFollowerRobotConfig(
            port=args.port,
            id=args.robot_id,
            max_relative_target=safety_validator.MAX_RELATIVE_TARGET_DEG,
        )
    )
    robot.connect(calibrate=False)
    with robot.bus.torque_disabled():
        robot.bus.write_calibration(robot.calibration)

    stereo_camera = StereoSplitCamera(index=args.stereo_camera_index)
    wrist_cap = cv2.VideoCapture(wrist_idx)
    if hasattr(wrist_cap, "isOpened") and not wrist_cap.isOpened():
        print(f"WARNING: wrist camera index {wrist_idx} did not open; every call will hold position.")

    collector = MllmInputCollector(wrist_cap, stereo_camera, depth_client, config)
    source = base_mllm.MllmActionSourceBase(
        adapter,
        collector,
        config,
        joint_limits_deg,
        backend_name,
        urdf_text=urdf_text,
        recorder=recorder,
        secrets=[api_key],
    )
    return BackendHandles(
        action_source=source,
        robot=robot,
        stereo_camera=stereo_camera,
        skip_camera_names=frozenset({"overhead"}),
    )


__all__ = [
    "MIN_MAX_TOKENS",
    "PROVIDERS",
    "build_mllm_backend",
    "load_mllm_backend_config",
]
