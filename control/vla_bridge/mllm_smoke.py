"""Live smoke check: ONE tiny paid call to one MLLM backend (Phase 15, Plan 15-09).

What this is
============
The first live call per provider (Plan 15-10, VALIDATION manual item 1) is
made with this tool, not with a robot episode. It runs the REAL registered
adapter (`base_mllm.MLLM_BACKENDS[name].adapter_factory`, the same factory
`--backend NAME` uses) through the real `MllmActionSourceBase`,
`MllmInputCollector` and `CallRecorder`, but every input is synthetic: three
generated camera images and a generated depth map (a plane at 0.8 m with a
nearer 0.5 m box). No robot, camera or depth service is touched, and no lab
image leaves the machine.

It makes exactly ONE `get_action()` call with a short chunk (default 3 steps),
writes the usual per-call sidecar under `<out>/mllm_calls/0000/`, and prints:

  - PASS/FAIL schema honoured (the answer parsed and every one of the N steps
    is valid);
  - PASS/WARN reasoning tokens reported (usage.reasoning_tokens present or null);
  - PASS/FAIL stop reason, plus the refusal category when there is one (a
    `reasoning_extraction` refusal means the provider read the prompt as asking
    for its internal deliberation: fix the prompt vocabulary before any episode);
  - api latency, estimated cost, request id, and the settings requested /
    applied / omitted.

It returns 0 only when the stop reason is complete and the schema was honoured.

SPEND WARNING
-------------
This is the only code in Phase 15 that spends money outside an episode. It
refuses to do so without `--confirm-spend`: without the flag it prints what it
would call (backend, request model string, chunk length, effort) and returns 2
without reading the key or building any client. Claude and GPT at xhigh
effort cost about $1 per call; the HF variants draw on purchased HF Inference
Providers credits (Kimi-K3 is about $0.35 per call).

The key is read only from the environment variable named in the backend
config, is given only to the adapter factory, the base and the recorder (for
redaction), and never appears in stdout or in any file under --out.

Usage::

    cd control
    export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY / HF_TOKEN
    python -m vla_bridge.mllm_smoke --backend mllm-claude --out outputs/smoke            # dry run
    python -m vla_bridge.mllm_smoke --backend mllm-claude --chunk-len 3 \\
        --out outputs/smoke --confirm-spend

Optional: --effort LEVEL (overrides the config effort for this run only),
--joint-calibration PATH (default action_contract.CALIBRATION_PATH; when it is
missing or unreadable the tool warns and uses neutral limits of -90 to 90
degrees for the five arm joints and 0 to 100 for the gripper).
"""

import argparse
import copy
import json
import os
from pathlib import Path

import cv2
import numpy as np

from vla_bridge import action_contract
from vla_bridge.action_contract import JOINT_ORDER
from vla_bridge.backends import base_mllm, base_mllm_build
from vla_bridge.backends.base_mllm import MAX_CHUNK_LEN, STOP_COMPLETE, MllmActionSourceBase
from vla_bridge.mllm_inputs import MllmInputCollector
from vla_bridge.mllm_record import CallRecorder
from vla_bridge.mllm_urdf import urdf_kinematics_text

DEFAULT_CHUNK_LEN = 3
SMOKE_INSTRUCTION = "Move the arm slightly toward the pen"
NEUTRAL_JOINT_LIMITS_DEG = {joint: (-90.0, 90.0) for joint in JOINT_ORDER}
NEUTRAL_JOINT_LIMITS_DEG["gripper"] = (0.0, 100.0)

# Synthetic inputs: frame sizes match the real rig (640x480 wrist, 800x600 per
# stereo half); the depth map matches the FastFS output shape.
WRIST_SHAPE = (480, 640)
STEREO_SHAPE = (600, 800)
DEPTH_SHAPE = (576, 800)
DEPTH_PLANE_M = 0.8
DEPTH_BOX_M = 0.5
SYNTHETIC_DEPTH_MODEL = "synthetic-plane-and-box"

SPEND_WARNING = (
    "WARNING: --confirm-spend makes ONE paid provider call. Claude and GPT at xhigh effort cost "
    "about $1 per call; HF variants draw on purchased HF Inference Providers credits (Kimi-K3 is "
    "about $0.35 per call)."
)


# --- Synthetic inputs ------------------------------------------------------------------------


def synthetic_frame(shape: tuple, base_bgr: tuple, rect_bgr: tuple, rect: tuple, label: str) -> np.ndarray:
    """A horizontal gradient tinted by `base_bgr`, a filled rectangle and a text label."""
    height, width = shape
    ramp = np.linspace(0.35, 1.0, width, dtype=np.float32).reshape(1, width, 1)
    frame = (ramp * np.array(base_bgr, dtype=np.float32).reshape(1, 1, 3)).repeat(height, axis=0)
    frame = np.ascontiguousarray(frame.astype(np.uint8))
    x0, y0, x1, y1 = rect
    cv2.rectangle(frame, (x0, y0), (x1, y1), rect_bgr, -1)
    cv2.putText(frame, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def synthetic_depth(shape: tuple = DEPTH_SHAPE) -> np.ndarray:
    """Metres: a flat plane at DEPTH_PLANE_M with a nearer DEPTH_BOX_M box in the middle."""
    rows, cols = shape
    depth = np.full(shape, DEPTH_PLANE_M, dtype=np.float32)
    depth[rows // 3: 2 * rows // 3, 2 * cols // 5: 3 * cols // 5] = DEPTH_BOX_M
    return depth


class _SyntheticWristCap:
    """`cv2.VideoCapture` stand-in returning one fixed synthetic wrist frame."""

    def __init__(self):
        self.frame = synthetic_frame(WRIST_SHAPE, (90, 160, 210), (40, 40, 220), (250, 200, 390, 300),
                                     "SYNTHETIC WRIST")

    def grab(self):
        return True

    def read(self):
        return True, self.frame.copy()

    def isOpened(self):
        return True

    def release(self):
        pass


class _SyntheticStereoCamera:
    """`StereoSplitCamera` stand-in: left/right views of the same box with a small disparity."""

    def __init__(self):
        self.left = synthetic_frame(STEREO_SHAPE, (150, 150, 150), (40, 200, 40), (330, 200, 470, 400),
                                    "SYNTHETIC LEFT")
        self.right = synthetic_frame(STEREO_SHAPE, (140, 140, 160), (40, 200, 40), (300, 200, 440, 400),
                                     "SYNTHETIC RIGHT")
        self.last_pair_metadata = {"pair_id": 0, "source": "synthetic"}

    def read_left(self):
        return self.left.copy()

    def read_right(self):
        return self.right.copy()


class _SyntheticDepthClient:
    """`DepthCameraClient` stand-in honouring the `compute_depth_result` dict contract."""

    def __init__(self):
        self.depth = synthetic_depth()

    def compute_depth_result(self, left, right, request_id=None):
        return {
            "depth": self.depth.copy(),
            "valid_fraction": 1.0,
            "calibration_sha256": None,
            "model": SYNTHETIC_DEPTH_MODEL,
            "inference_ms": 0.0,
            "request_id": request_id,
        }


class SyntheticCollector:
    """Collector for the smoke call: the REAL `MllmInputCollector` fed synthetic sources.

    So the images, depth heatmap/grid (per `settings["depth_presentation"]`),
    depth note, meta and sidecar files (depth.npy, depth_heatmap.png,
    depth_grid.txt) are produced exactly as on the robot.
    """

    def __init__(self, settings: dict):
        self._collector = MllmInputCollector(_SyntheticWristCap(), _SyntheticStereoCamera(),
                                             _SyntheticDepthClient(), settings, sleep_fn=lambda _s: None)

    def collect(self, joint_state):
        return self._collector.collect(joint_state)

    def close(self):
        self._collector.close()


# --- Helpers ---------------------------------------------------------------------------------


def _load_limits(path: Path) -> dict:
    try:
        return action_contract.load_joint_limits_deg(Path(path))
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"WARNING: no usable follower calibration at {path} ({type(exc).__name__}); using neutral "
              "joint limits (-90 to 90 degrees for the arm joints, 0 to 100 for the gripper). The limits "
              "only shape the prompt text; no robot is driven.")
        return dict(NEUTRAL_JOINT_LIMITS_DEG)


def _request_model(config: dict) -> str:
    hf_provider = config.get("hf_provider")
    return f"{config['model']}:{hf_provider}" if hf_provider else config["model"]


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _report(record: dict, details: dict, chunk_len: int, out: Path) -> bool:
    """Print the PASS/WARN/FAIL lines; True when the stop is complete and the schema honoured."""
    sidecar_dir = record.get("sidecar_dir")
    if sidecar_dir:
        print(f"Saved → {out / sidecar_dir}")
    else:
        print(f"WARN sidecar not written: {record.get('sidecar_error') or record.get('recorder_error')}")

    parse = record.get("parse") or {}
    n_valid = parse.get("n_valid") or 0
    schema_ok = parse.get("ok") is True and n_valid == chunk_len
    if schema_ok:
        print(f"PASS schema honoured: {n_valid}/{chunk_len} valid steps")
    else:
        print(f"FAIL schema not honoured: {n_valid}/{chunk_len} valid steps "
              f"(reason {parse.get('reason')!r}: {parse.get('reason_detail')})")

    usage = record.get("usage") or {}
    reasoning_tokens = usage.get("reasoning_tokens")
    if reasoning_tokens is not None:
        print(f"PASS reasoning tokens reported: {reasoning_tokens}")
    else:
        print("WARN reasoning tokens not reported (usage.reasoning_tokens is null)")

    stop = record.get("stop") or {}
    reason = stop.get("reason")
    stop_ok = reason == STOP_COMPLETE
    if stop_ok:
        print(f"PASS stop reason: {reason}")
    else:
        extra = {key: stop.get(key) for key in ("provider_stop_reason", "incomplete_reason") if stop.get(key)}
        print(f"FAIL stop reason: {reason}" + (f" {_json(extra)}" if extra else ""))
        if details.get("error"):
            print(f"  error: {details.get('error_type')}: {details['error']}")
    category = stop.get("refusal_category")
    if category == "reasoning_extraction":
        print("FAIL refusal category reasoning_extraction: the provider read the prompt as asking it to "
              "reproduce its internal deliberation. Check the prompt vocabulary "
              "(mllm_prompt.lint_prompt_text) before any episode.")
    elif category:
        print(f"FAIL refusal category {category}")

    latency = (record.get("latency_ms") or {}).get("api")
    print(f"api latency: {latency / 1000.0:.1f} s" if latency is not None else "api latency: n/a")
    cost = record.get("cost_usd_est")
    verified_on = (record.get("price_table") or {}).get("verified_on")
    print(f"estimated cost: ${cost:.4f} (prices verified {verified_on})" if cost is not None
          else "estimated cost: n/a")
    print(f"request id: {record.get('request_id')}")
    print(f"model returned: {record.get('model_returned')}")
    settings = record.get("settings") or {}
    for part in ("requested", "applied", "omitted"):
        print(f"settings {part}: {_json(settings.get(part))}")
    return schema_ok and stop_ok


# --- CLI -------------------------------------------------------------------------------------


def main(argv=None) -> int:
    """One confirmed, tiny, auditable provider call; see the module docstring."""
    parser = argparse.ArgumentParser(
        description="One tiny paid smoke call to an MLLM backend with synthetic inputs "
        "(refuses to spend without --confirm-spend)."
    )
    parser.add_argument("--backend", required=True, help="A registered MLLM backend, e.g. mllm-claude")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for the call sidecar")
    parser.add_argument("--chunk-len", type=int, default=DEFAULT_CHUNK_LEN,
                        help=f"Steps requested in the one call (default {DEFAULT_CHUNK_LEN})")
    parser.add_argument("--effort", type=str, default=None,
                        help="Override the config effort for this run only")
    parser.add_argument("--joint-calibration", type=Path, default=action_contract.CALIBRATION_PATH,
                        help="Follower calibration JSON for the prompt's joint ranges "
                        "(default: action_contract.CALIBRATION_PATH; neutral limits if missing)")
    parser.add_argument("--confirm-spend", action="store_true",
                        help="Actually make the paid call (otherwise print the plan and exit 2)")
    args = parser.parse_args(argv)
    if not 1 <= args.chunk_len <= MAX_CHUNK_LEN:
        parser.error(f"--chunk-len must be between 1 and {MAX_CHUNK_LEN}")

    name = args.backend
    spec = base_mllm.MLLM_BACKENDS.get(name)
    if spec is None:
        raise SystemExit(f"--backend {name!r} is not a registered MLLM backend "
                         f"(known: {', '.join(sorted(base_mllm.MLLM_BACKENDS)) or 'none'}).")

    loaded = base_mllm_build.load_mllm_backend_config(spec.config_name, name)
    config = copy.deepcopy(loaded)  # overrides apply to this run only
    config["chunk_len"] = args.chunk_len
    if args.effort is not None:
        levels = (config.get("capabilities") or {}).get("effort_levels")
        if not args.effort.strip() or (levels is not None and args.effort not in levels):
            raise SystemExit(f"--effort {args.effort!r} is not allowed for {name}; "
                             f"capabilities.effort_levels is {levels}.")
        config["effort"] = args.effort

    limits = _load_limits(args.joint_calibration)
    effort_note = (f" (--effort override; config has {loaded['effort']})" if args.effort is not None else "")
    print(f"MLLM smoke: backend {name}, provider {config['provider']}, model {_request_model(config)}")
    print(f"  chunk_len: {config['chunk_len']}  effort: {config['effort']}{effort_note}  "
          f"max_tokens: {config['max_tokens']}  depth_presentation: {config['depth_presentation']}")
    print("  inputs: synthetic wrist/left/right images and a synthetic depth map (no lab images sent)")
    print(SPEND_WARNING)
    if not args.confirm_spend:
        print("Not calling: re-run with --confirm-spend to make the one paid call.")
        return 2

    env_name = config["api_key_env"]
    api_key = (os.environ.get(env_name) or "").strip()
    if not api_key:
        raise SystemExit(f"Environment variable {env_name} is not set (or is blank); --backend {name} needs "
                         "it. Export the key under that name; never put it in a config file.")

    adapter = spec.adapter_factory(api_key, config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    urdf_text = urdf_kinematics_text() if config["include_urdf"] else None
    recorder = CallRecorder(out / "mllm_calls", secrets=[api_key], prices=config["prices"])
    source = MllmActionSourceBase(adapter, SyntheticCollector(config), config, limits, name,
                                  urdf_text=urdf_text, recorder=recorder, secrets=[api_key])
    joint_state = {joint: (float(lo) + float(hi)) / 2 for joint, (lo, hi) in limits.items()}
    try:
        source.get_action(joint_state, SMOKE_INSTRUCTION)
    finally:
        source.close()

    record = source.last_details.get("mllm_call")
    if not isinstance(record, dict):
        print(f"FAIL no call record ({source.last_details.get('error_type')}: {source.last_details.get('error')})")
        print("RESULT: FAIL")
        return 1
    passed = _report(record, source.last_details, config["chunk_len"], out)
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CHUNK_LEN",
    "NEUTRAL_JOINT_LIMITS_DEG",
    "SMOKE_INSTRUCTION",
    "SyntheticCollector",
    "main",
    "synthetic_depth",
    "synthetic_frame",
]
