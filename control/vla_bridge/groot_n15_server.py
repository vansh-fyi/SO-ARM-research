"""Resident GR00T N1.5 HTTP adapter (Tier C, Plan 14-12, VLAB-11). Run on Colab inside
the isolated Isaac-GR00T venv (see vla_bridge/notebooks/http_groot_n15.ipynb).

Same shape as `rdt1b_server.py` / `octo_server.py` / `fastfs_server.py`:
- the model loads ONCE
- one throwaway warmup inference runs before the socket opens
- `/health` reports ready
- a non-blocking `inference_lock` answers concurrent requests with 429
- `ValueError`/`KeyError`/`TypeError` on a request become 400
- a model failure (`RuntimeError`, e.g. a non-finite or wrong-shaped action)
  becomes 500, so the client holds position instead of parsing NaN JSON

The server stays model-format-only: it returns one step of the checkpoint's
un-normalized action chunk as 6 floats in `JOINT_ORDER`, and
`backends/groot_n15.py` reshapes them into the `JOINT_ORDER` dict.

Why this is NOT the GR00T N1.7 code path (14-RESEARCH Pitfall 3)
-----------------------------------------------------------------
lerobot 0.6.1's `groot` policy wraps N1.7 only and rejects N1.5 checkpoints
("GR00T N1.5 support was removed from LeRobot", Plan 14-07). N1.5 is served
through NVIDIA's own `NVIDIA/Isaac-GR00T` package, GitHub-only (not on PyPI;
14-RESEARCH audit), cloned at the tag `n1.5-release` = branch `n1d5` =
`ISAAC_GROOT_COMMIT`. Nothing here imports lerobot.

Verified against live upstream source (2026-10-01, execution discipline)
------------------------------------------------------------------------
github.com/NVIDIA/Isaac-GR00T @ 4af2b62 (Apache-2.0). Files read: `README.md`,
`pyproject.toml`, `gr00t/model/policy.py`, `gr00t/model/gr00t_n1.py`,
`gr00t/model/transforms.py`, `gr00t/model/backbone/eagle_backbone.py`,
`gr00t/model/backbone/eagle2_hg_model/config.json`,
`gr00t/experiment/data_config.py`, `gr00t/data/embodiment_tags.py`,
`gr00t/data/transform/{base,video,state_action,concat}.py`,
`gr00t/eval/service.py`, `scripts/inference_service.py`,
`examples/SO-100/{README.md,eval_lerobot.py,so100_dualcam__modality.json}`,
`getting_started/{3_0_new_embodiment_finetuning,4_deeper_understanding,5_policy_deployment}.md`.
Also NVIDIA's SO-101 tutorial (huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning).

- Loading: `Gr00tPolicy(model_path, embodiment_tag, modality_config,
  modality_transform, denoising_steps, device)` (README 2.1). Its own
  `snapshot_download(model_path)` and `GR00T_N1_5.from_pretrained()` take NO
  revision (they resolve `main`). So `_download_pinned` fetches the reviewed
  revision into a local dir, and the local path is passed instead. A local path
  is not a valid repo id, so both calls fall back to it (`HFValidationError`).
  The Eagle 2.5 VLM config and tokenizer are bundled in the pinned clone
  (`eagle2_hg_model/`); every weight comes from the checkpoint's safetensors, so
  the checkpoint revision is the only network fetch.
- Embodiment: SO-100/SO-101 were NOT in N1.5's pre-training. The tutorial
  fine-tunes them as `new_embodiment` with data config `so100_dualcam`
  (`video.front` + `video.wrist`, `state/action.single_arm` [5] +
  `.gripper` [1], `annotation.human.task_description`, 1 observation step,
  16-step action chunk). The base `nvidia/GR00T-N1.5-3B`'s
  `experiment_cfg/metadata.json` has only `gr1`, `oxe_droid` and
  `agibot_genie1`, so `Gr00tPolicy(..., "new_embodiment")` on the base raises
  "No metadata found for embodiment tag". A real SO-101 fine-tune is required.
- Action convention: `so100_dualcam` normalizes `single_arm`/`gripper` with
  `min_max` (`StateActionTransform`); `get_action()` un-applies it, so the chunk
  comes back in the dataset's raw units. Every SO-101 N1.5 fine-tune's metadata
  marks both action groups `absolute: true`. NVIDIA's own real-robot client
  (`examples/SO-100/eval_lerobot.py`) feeds `robot.get_observation()` values
  straight in as `state.*`, and sends each chunk step straight to
  `robot.send_action()`. So the actions are ABSOLUTE lerobot `.pos` joint
  targets, with no delta and no IK, in whatever unit mode the dataset was
  recorded in. That mode is checked per checkpoint (`classify_action_units`).
- Units: lerobot SO-101 data comes in several modes. Current lerobot
  (`so_follower`, `use_degrees=True`, the mode `action_contract.py` speaks) is
  degrees about the calibrated midpoint. Older SO-100 data is range-normalized
  (+/-100, clamped at the calibration limits), or uses the pre-0.2 offset-zero
  calibration (shoulder_lift up to ~199). Only degree mode is an identity here,
  so the reviewed profile pins it and `_check_checkpoint_layout` re-derives it
  from the downloaded `metadata.json` before the model loads.
- Images: `Gr00tPolicy` puts the transforms in eval mode: `VideoCrop` becomes a
  0.95 centre crop and `VideoColorJitter` is skipped. `VideoCrop.check_input`
  asserts the frame is exactly the training resolution from metadata (640x480),
  so frames are resized to it here. Colors are RGB (lerobot `OpenCVCamera`).
- flash-attn: the bundled Eagle config hard-codes `flash_attention_2` and
  asserts it for the Qwen3 LLM, so the GPU must be Ampere or newer (A100/L4). A
  T4 (sm75) cannot run it. `main()` refuses such GPUs.
- Pins (pyproject `[base]` + README): Python 3.10, `torch==2.5.1`,
  `transformers==4.51.3`, `flash-attn==2.7.1.post4`. `main()` refuses others.

Checkpoint (HF API, read 2026-10-01; see `CHECKPOINT_PROFILES`)
---------------------------------------------------------------
`quangnd58/GR00T-N1.5-3B-so101-multitask` @ 488f3b6. Model card: base
`nvidia/GR00T-N1.5-3B`, trained with Isaac-GR00T branch `n1d5`
(`scripts/gr00t_finetune.py`), `new_embodiment`, `so100_dualcam`,
`video.front` = top camera, `video.wrist`, 640x480, horizon 16, license
"other" (NVIDIA License, derived from the base). Its 899 weight keys equal the
official base's. Dataset `hungho77/so101-multitask` (Apache-2.0): lerobot v3.0,
`robot_type: so_follower`, 30 fps, 143 episodes, 3 tasks, names exactly
`shoulder_pan.pos .. gripper.pos`. Its stats are degree mode (wrist_flex
reaches 100.4, which is past the range mode's +/-100 clamp; no joint sits at
exactly +/-100). Task prompts (use verbatim): "Pick up the banana and place it
in the bot, then close the lid", "Pick blue cube and place on red cube", "Pick
all cubes and place into cup".

Known limitations (documented, not hidden)
------------------------------------------
- The default checkpoint is a small community fine-tune (16 downloads, published
  2026-09-29) on someone else's rig and scene. Expect pipeline-proving motion,
  not task success.
- `video.front` is the dataset's TOP camera, mapped to this rig's
  `overhead_left` by default (`--front-image-key` overrides).
- It returns one chunk step about one client control period ahead (14 at the
  default 2 Hz), clamped to the 16-step horizon, not NVIDIA's "execute 8 steps
  at 50 Hz" client loop. Every step is an absolute target, so the per-step
  safety clamp bounds the jump.

Colab launch: see vla_bridge/notebooks/http_groot_n15.ipynb.
"""
import argparse
import base64
import json
import math
import numbers
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

DEFAULT_CHECKPOINT = "quangnd58/GR00T-N1.5-3B-so101-multitask"
DEFAULT_LOCAL_DIR = "/content/groot_n15_checkpoints"
DEFAULT_GROOT_REPO = "/content/Isaac-GR00T"
DEFAULT_PORT = 3006  # 3000 FastFS, 3001 OpenVLA-OFT, 3002 Wall-X, 3003 OpenVLA, 3004 Octo, 3005 RDT-1B
DEFAULT_CLIENT_CONTROL_HZ = 2.0  # run_vla_episode.py's default --control-hz
DEFAULT_INSTRUCTION = "Pick blue cube and place on red cube"
# github.com/NVIDIA/Isaac-GR00T tag n1.5-release == branch n1d5 (GitHub API, 2026-10-01).
ISAAC_GROOT_COMMIT = "4af2b622892f7dcb5aae5a3fb70bcb02dc217b96"

# Reviewed checkpoints. `backend_configs/groot_n15.json["checkpoints"]` must agree
# (pinned by test_client_and_server_checkpoint_profiles_agree).
CHECKPOINT_PROFILES = {
    "quangnd58/GR00T-N1.5-3B-so101-multitask": dict(
        revision="488f3b6da322e55a316a562a00567fb4a1fe130a", subfolder="", data_config="so100_dualcam",
        embodiment_tag="new_embodiment", video_sources={"front": "overhead_left", "wrist": "wrist"},
        action_space="absolute_joint", action_units="degrees", control_hz=30, action_horizon=16,
        denoising_steps=4),
}

# 6 DOF in action_contract.JOINT_ORDER (re-declared: this file runs standalone on Colab).
SO101_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM_GROUP, GRIPPER_GROUP = "single_arm", "gripper"
GROUP_SHAPES = {ARM_GROUP: 5, GRIPPER_GROUP: 1}

MAX_IMAGE_SIDE = 1024
MAX_INSTRUCTION_CHARS = 1000
IMAGE_KEYS = ("overhead_left", "overhead_right", "wrist")
MIN_COMPUTE_CAPABILITY = (8, 0)  # flash-attn 2: Ampere or newer
PINNED_DEPENDENCIES = {"torch": {"2.5.1"}, "transformers": {"4.51.3"}, "flash_attn": {"2.7.1.post4"}}


def chunk_step_for(control_hz, client_hz, horizon):
    """Chunk step i is the absolute target i/control_hz s ahead; return the one
    about one client control period ahead, clamped to the chunk."""
    step = int(round(float(control_hz) / float(client_hz))) - 1
    return max(0, min(int(horizon) - 1, step))


def classify_action_units(action_stats):
    """Unit mode of an SO-101 checkpoint, from its own `single_arm` action stats.

    - `radians`: every bound within +/-3.5.
    - `legacy_offset`: some bound past 180 (the pre-0.2 lerobot SO-100 calibration).
    - `degrees`: some bound past 100, which the range mode's clamp cannot produce.
    - `range_m100_100`: a bound at exactly +/-100 (lerobot clamps there), none past it.
    - `ambiguous`: anything else. It is not accepted as degrees.
    """
    arm = action_stats.get(ARM_GROUP) or {}
    values = [abs(float(v)) for v in list(arm.get("min", [])) + list(arm.get("max", []))]
    if not values:
        return "ambiguous"
    peak = max(values)
    if peak <= 3.5:
        return "radians"
    if peak > 180.0:
        return "legacy_offset"
    if peak > 100.0 + 1e-6:
        return "degrees"
    if any(abs(v - 100.0) <= 1e-4 for v in values):
        return "range_m100_100"
    return "ambiguous"


def _download_pinned(checkpoint, revision, local_dir, subfolder):
    """Materialise the pinned revision's inference files (no optimizer state) as a
    real directory; return the model directory (`<snapshot>/<subfolder>`)."""
    from huggingface_hub import snapshot_download

    target = Path(local_dir) / checkpoint.replace("/", "--") / revision
    patterns = [f"{subfolder}{name}" for name in (
        "config.json", "experiment_cfg/*", "model*.safetensors", "model.safetensors.index.json")]
    snapshot_download(repo_id=checkpoint, revision=revision, local_dir=str(target), allow_patterns=patterns)
    return str(target / subfolder) if subfolder else str(target)


def _git_head(repo):
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _is_groot_checkout(repo):
    return (Path(repo) / "gr00t" / "model" / "policy.py").is_file()


def _check_groot_checkout(groot_repo):
    """Refuse to start unless `groot_repo` is the pinned Isaac-GR00T commit (T-14-06)."""
    if not _is_groot_checkout(groot_repo):
        raise SystemExit(f"{groot_repo} is not an Isaac-GR00T checkout (no gr00t/model/policy.py)")
    head = _git_head(groot_repo)
    if head != ISAAC_GROOT_COMMIT:
        raise SystemExit(f"{groot_repo} is at {head}, expected the reviewed Isaac-GR00T n1.5-release commit "
                         f"{ISAAC_GROOT_COMMIT}")


def _read_checkpoint_metadata(model_dir, embodiment_tag):
    path = Path(model_dir) / "experiment_cfg" / "metadata.json"
    metadata = json.loads(path.read_text())
    if embodiment_tag not in metadata:
        raise RuntimeError(f"{path} has no {embodiment_tag!r} entry (found {sorted(metadata)}); "
                           "Gr00tPolicy would refuse it")
    return metadata[embodiment_tag]


def _check_checkpoint_layout(metadata, profile):
    """The downloaded checkpoint must be the layout and unit mode its profile was
    reviewed as. Raises `RuntimeError` before the 3B model is loaded."""
    modalities = metadata.get("modalities", {})
    videos = sorted(modalities.get("video", {}))
    if videos != sorted(profile["video_sources"]):
        raise RuntimeError(f"checkpoint video keys {videos} != reviewed {sorted(profile['video_sources'])}")
    for modality in ("state", "action"):
        groups = modalities.get(modality, {})
        if sorted(groups) != sorted(GROUP_SHAPES):
            raise RuntimeError(f"checkpoint {modality} groups {sorted(groups)} != {sorted(GROUP_SHAPES)}")
        for group, size in GROUP_SHAPES.items():
            if list(groups[group].get("shape", [])) != [size]:
                raise RuntimeError(f"checkpoint {modality}.{group} shape {groups[group].get('shape')} != [{size}]")
            if groups[group].get("absolute") is not True:
                raise RuntimeError(f"checkpoint {modality}.{group} is not absolute; this adapter only "
                                   "serves absolute joint targets (Pitfall 3)")
    detected = classify_action_units(metadata.get("statistics", {}).get("action", {}))
    if detected != profile["action_units"]:
        raise RuntimeError(f"checkpoint action stats look like {detected!r}, but its profile says "
                           f"{profile['action_units']!r}; refusing to serve it as an identity mapping")
    return detected


def _load_policy(model_dir, profile, denoising_steps, groot_repo):
    """README 2.1: `Gr00tPolicy` with the profile's data config, from a LOCAL path."""
    import sys

    root = str(Path(groot_repo).resolve())
    if not _is_groot_checkout(root):
        raise RuntimeError(f"{root} is not an Isaac-GR00T checkout")
    if root not in sys.path:
        sys.path.insert(0, root)
    import gr00t

    if not str(Path(gr00t.__file__).resolve()).startswith(root):
        raise RuntimeError(f"imported gr00t from {gr00t.__file__}, not the pinned checkout at {root}")
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy

    data_config = DATA_CONFIG_MAP[profile["data_config"]]
    policy = Gr00tPolicy(
        model_path=str(model_dir),
        embodiment_tag=profile["embodiment_tag"],
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        denoising_steps=denoising_steps,
        device="cuda",
    )
    return policy, data_config


def _verify_weights(model, model_dir):
    """`GR00T_N1_5.from_pretrained` (transformers) only warns on mismatched keys.
    Fail instead: every model key must come from the checkpoint, or be tied to one
    that does (Qwen's lm_head), and the checkpoint must hold no unused key."""
    index = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())
    saved = set(index["weight_map"])
    state = model.state_dict()
    unexpected = sorted(saved - set(state))
    loaded_ptrs = {state[k].data_ptr() for k in saved & set(state)}
    missing = sorted(k for k in set(state) - saved if state[k].data_ptr() not in loaded_ptrs)
    if unexpected or missing:
        raise RuntimeError(f"checkpoint weights do not match the model: missing {missing[:5]} "
                           f"({len(missing)}), unexpected {unexpected[:5]} ({len(unexpected)})")


def _get_action(policy, observation):
    return policy.get_action(observation)


def parse_joint_state(joint_state):
    """`{JOINT_ORDER name: finite number}` -> floats in `SO101_JOINTS` order."""
    if not isinstance(joint_state, dict):
        raise TypeError("joint_state must be an object")
    values = []
    for joint in SO101_JOINTS:
        value = joint_state[joint]
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"joint_state[{joint!r}] must be a number")
        if not math.isfinite(value):
            raise ValueError(f"joint_state[{joint!r}] must be finite")
        values.append(float(value))
    return values


class GrootN15Model:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, revision=None, groot_repo=DEFAULT_GROOT_REPO,
                 local_dir=DEFAULT_LOCAL_DIR, client_control_hz=DEFAULT_CLIENT_CONTROL_HZ,
                 front_image_key=None, denoising_steps=None):
        profile = CHECKPOINT_PROFILES.get(checkpoint)
        if profile is None:
            raise ValueError(f"{checkpoint} has no reviewed profile (known: {sorted(CHECKPOINT_PROFILES)})")
        revision = revision or profile["revision"]
        video_sources = dict(profile["video_sources"])
        if front_image_key is not None:
            if front_image_key not in IMAGE_KEYS:
                raise ValueError(f"front_image_key must be one of {IMAGE_KEYS}")
            video_sources["front"] = front_image_key
        denoising_steps = int(denoising_steps or profile["denoising_steps"])

        model_dir = _download_pinned(checkpoint, revision, local_dir, profile["subfolder"])
        checkpoint_metadata = _read_checkpoint_metadata(model_dir, profile["embodiment_tag"])
        detected_units = _check_checkpoint_layout(checkpoint_metadata, profile)
        self.policy, data_config = _load_policy(model_dir, profile, denoising_steps, groot_repo)
        _verify_weights(self.policy.model, model_dir)

        self.video_keys = {key: key.split(".", 1)[1] for key in data_config.video_keys}
        if sorted(self.video_keys.values()) != sorted(video_sources):
            raise RuntimeError(f"data config {profile['data_config']} video keys {data_config.video_keys} do not "
                               f"match the profile's {sorted(video_sources)}")
        self.state_keys = list(data_config.state_keys)
        self.action_keys = list(data_config.action_keys)
        if self.state_keys != [f"state.{g}" for g in GROUP_SHAPES] or \
                self.action_keys != [f"action.{g}" for g in GROUP_SHAPES]:
            raise RuntimeError(f"data config {profile['data_config']} is not single_arm + gripper")
        self.language_key = data_config.language_keys[0]
        self.action_horizon = len(data_config.action_indices)
        self.video_sources = video_sources
        video_meta = checkpoint_metadata["modalities"]["video"]
        self.resolutions = {name: tuple(int(v) for v in video_meta[name]["resolution"])  # (width, height)
                            for name in video_sources}
        width, height = self.resolutions[next(iter(self.resolutions))]
        self.warmup_shape = (height, width, 3)
        self.required_image_keys = tuple(dict.fromkeys(video_sources.values()))
        self.chunk_step = chunk_step_for(profile["control_hz"], client_control_hz, self.action_horizon)
        self.model_version = f"{checkpoint}@{revision[:12]}"
        self.metadata = dict(
            checkpoint=checkpoint, revision=revision, isaac_groot_commit=ISAAC_GROOT_COMMIT,
            data_config=profile["data_config"], embodiment_tag=profile["embodiment_tag"],
            action_space=profile["action_space"], action_units=profile["action_units"],
            detected_action_units=detected_units, action_horizon=self.action_horizon,
            control_hz=profile["control_hz"], client_control_hz=float(client_control_hz),
            chunk_step=self.chunk_step, denoising_steps=denoising_steps, video_sources=video_sources,
            video_resolution={k: list(v) for k, v in self.resolutions.items()},
            joint_order=list(SO101_JOINTS),
            returns="one absolute joint-target step of the un-normalized action chunk, JOINT_ORDER",
        )

    def _frame(self, name, rgb):
        width, height = self.resolutions[name]
        if rgb.shape[:2] != (height, width):
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(rgb, dtype=np.uint8)[None]

    def __call__(self, images, instruction, joint_state):
        state = parse_joint_state(joint_state)
        observation = {key: self._frame(name, images[self.video_sources[name]])
                       for key, name in self.video_keys.items()}
        observation[f"state.{ARM_GROUP}"] = np.asarray([state[:5]], dtype=np.float64)
        observation[f"state.{GRIPPER_GROUP}"] = np.asarray([state[5:]], dtype=np.float64)
        observation[self.language_key] = [instruction]
        chunk = _get_action(self.policy, observation)
        try:
            arm = np.asarray(chunk[f"action.{ARM_GROUP}"], dtype=np.float64)
            gripper = np.asarray(chunk[f"action.{GRIPPER_GROUP}"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"model returned an unexpected action chunk ({exc!r})") from exc
        if gripper.ndim == 1:
            gripper = gripper[:, None]
        if arm.shape != (self.action_horizon, 5) or gripper.shape != (self.action_horizon, 1):
            raise RuntimeError(f"model returned single_arm {arm.shape} / gripper {gripper.shape}, expected "
                               f"({self.action_horizon}, 5) / ({self.action_horizon}, 1)")
        values = np.concatenate([arm[self.chunk_step], gripper[self.chunk_step]])
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"model returned a non-finite action {values.tolist()}")
        return [float(v) for v in values]


def _decode_rgb(b64_png):
    """base64 PNG (OpenCV BGR-encoded by the client) -> HxWx3 uint8 RGB."""
    if not isinstance(b64_png, str):
        raise TypeError("image must be a base64 string")
    raw = base64.b64decode(b64_png, validate=True)
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("invalid PNG")
    if max(frame.shape[:2]) > MAX_IMAGE_SIDE:
        raise ValueError(f"image dimensions must be <= {MAX_IMAGE_SIDE}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def create_app(infer):
    from flask import Flask, jsonify, request
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
    inference_lock = threading.Lock()
    image_keys = tuple(getattr(infer, 'required_image_keys', ('overhead_left', 'wrist')))

    @app.get('/health')
    def health():
        return jsonify(status='ready', protocol_version=1, model=infer.metadata)

    @app.post('/action')
    def action():
        if not inference_lock.acquire(blocking=False):
            return jsonify(error='groot n1.5 service busy'), 429
        try:
            body = request.get_json(silent=True)
            if not isinstance(body, dict):
                raise ValueError('expected a JSON object body')
            images_in = body['images']
            if not isinstance(images_in, dict):
                raise TypeError('images must be an object')
            images = {key: _decode_rgb(images_in[key]) for key in image_keys}
            instruction = body['instruction']
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError('instruction must be a non-empty string')
            if len(instruction) > MAX_INSTRUCTION_CHARS:
                raise ValueError(f'instruction must be <= {MAX_INSTRUCTION_CHARS} characters')
            joint_state = body['joint_state']
            parse_joint_state(joint_state)  # GR00T needs the proprio: reject before the model runs
            reset = body.get('reset', False)
            if not isinstance(reset, bool):
                raise TypeError('reset must be a boolean')  # accepted for wire parity; GR00T is stateless
            started = time.perf_counter()
            result = infer(images, instruction, joint_state)
            elapsed = (time.perf_counter() - started) * 1000
            return jsonify(action=[float(v) for v in result], model_version=infer.model_version,
                           inference_ms=elapsed)
        except (ValueError, KeyError, TypeError) as exc:
            return jsonify(error=str(exc)), 400
        except RuntimeError as exc:
            return jsonify(error=f'groot n1.5 inference failed: {exc}'), 500
        finally:
            inference_lock.release()
    return app


def _installed_versions():
    import importlib.metadata as metadata

    return {name: metadata.version(name).split("+")[0] for name in PINNED_DEPENDENCIES}


def _check_dependency_versions():
    versions = _installed_versions()
    bad = {name: version for name, version in versions.items() if version not in PINNED_DEPENDENCIES[name]}
    if bad:
        raise SystemExit(
            "GR00T N1.5 needs torch==2.5.1, transformers==4.51.3, flash-attn==2.7.1.post4 "
            f"(Isaac-GR00T n1.5-release pyproject/README); found {bad}. Use the isolated GR00T N1.5 venv."
        )


def _gpu_info():
    import torch

    if not torch.cuda.is_available():
        return dict(cuda=False, name=None, total_gib=0.0, capability=None, bf16=False)
    props = torch.cuda.get_device_properties(0)
    return dict(cuda=True, name=props.name, total_gib=props.total_memory / 2**30,
                capability=tuple(torch.cuda.get_device_capability(0)), bf16=bool(torch.cuda.is_bf16_supported()))


def _peak_vram_gib():
    import torch

    return torch.cuda.max_memory_allocated() / 2**30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT, choices=sorted(CHECKPOINT_PROFILES))
    parser.add_argument('--revision', default=None, help="pinned HF commit sha (defaults to the reviewed profile's)")
    parser.add_argument('--groot-repo', default=DEFAULT_GROOT_REPO,
                        help=f'NVIDIA/Isaac-GR00T checkout at {ISAAC_GROOT_COMMIT[:7]} (n1.5-release)')
    parser.add_argument('--local-dir', default=DEFAULT_LOCAL_DIR)
    parser.add_argument('--front-image-key', default=None, choices=IMAGE_KEYS,
                        help="rig camera fed as video.front (default: the profile's, overhead_left)")
    parser.add_argument('--denoising-steps', type=int, default=None, help="default: the profile's (4)")
    parser.add_argument('--client-control-hz', type=float, default=DEFAULT_CLIENT_CONTROL_HZ,
                        help="run_vla_episode.py's --control-hz; picks the returned chunk step")
    parser.add_argument('--warmup-instruction', default=DEFAULT_INSTRUCTION)
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    _check_dependency_versions()
    _check_groot_checkout(args.groot_repo)
    gpu = _gpu_info()
    if not gpu['cuda'] or gpu['capability'] is None or tuple(gpu['capability']) < MIN_COMPUTE_CAPABILITY \
            or not gpu['bf16']:
        raise SystemExit(f"GR00T N1.5's Eagle backbone requires flash_attention_2, which needs an Ampere+ GPU "
                         f"with bf16 (A100/L4); found {gpu}. A T4 cannot run it.")
    print(f"GPU {gpu['name']} {gpu['total_gib']:.1f} GiB, sm{gpu['capability'][0]}{gpu['capability'][1]}",
          flush=True)
    model = GrootN15Model(args.checkpoint, revision=args.revision, groot_repo=args.groot_repo,
                          local_dir=args.local_dir, client_control_hz=args.client_control_hz,
                          front_image_key=args.front_image_key, denoising_steps=args.denoising_steps)
    # First inference compiles kernels / allocates; it happens before readiness is reported.
    model({key: np.zeros(model.warmup_shape, dtype=np.uint8) for key in model.required_image_keys},
          args.warmup_instruction, {joint: 0.0 for joint in SO101_JOINTS})
    print(f'GR00T N1.5 model loaded and warmed ({model.model_version}); peak VRAM {_peak_vram_gib():.1f} GiB; ready',
          flush=True)
    create_app(model).run(host='0.0.0.0', port=args.port, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
