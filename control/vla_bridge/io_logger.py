"""JSON Lines I/O logger for a real-hardware VLA episode (VLAHW-03, D-04).

One JSON object per inference step, camera frames referenced by path (not
embedded) -- matches `control/record_episode.py`'s existing frame-storage
convention, but replaces its single shared per-tick timestamp with genuinely
independent per-camera timestamps (see `capture_camera_frame` below), per
11-RESEARCH.md's Pattern 4 schema and VLAHW-03's explicit requirement.

Durability: every `write_step()` call flushes the file handle immediately, so
a mid-episode crash (process killed, exception, power loss) still leaves a
readable, valid `episode.jsonl` containing every step written before the
crash -- no buffering, no "finalize" step required.
"""

import json
from datetime import datetime, timezone
from pathlib import Path


class IOLogger:
    """Writes one JSON Lines record per inference step to `out_dir/episode.jsonl`.

    Also owns per-camera frame capture (`capture_camera_frame` for a `cap`
    object owning its own `.read()`; `capture_stereo_frame` for an
    already-read frame, e.g. from a shared `StereoSplitCamera`), writing each
    frame as a PNG under `out_dir/camera_{name}/{step:06d}.png` and returning
    a path+timestamp reference for that step's JSONL record.
    """

    def __init__(self, out_dir: Path, camera_names: dict[int, str]):
        self.out_dir = Path(out_dir)
        self.camera_names = camera_names
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for name in camera_names.values():
            (self.out_dir / f"camera_{name}").mkdir(parents=True, exist_ok=True)
        self._file = open(self.out_dir / "episode.jsonl", "a")

    def __enter__(self) -> "IOLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._file.close()

    def write_step(
        self,
        step: int,
        instruction: str,
        camera_frames: dict[str, dict],
        joint_state: dict,
        raw_model_output: dict,
        validated_action: dict,
        validator_flags: list[str],
        executed_action: dict,
        latency_ms: dict,
        model_version: str,
        depth_frames: dict | None = None,
    ) -> None:
        record = {
            "step": step,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "instruction": instruction,
            "camera_frames": camera_frames,
            "joint_state": joint_state,
            "raw_model_output": raw_model_output,
            "validated_action": validated_action,
            "validator_flags": validator_flags,
            "executed_action": executed_action,
            "latency_ms": latency_ms,
            "model_version": model_version,
            "depth_frames": depth_frames if depth_frames is not None else {},
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def capture_camera_frame(self, cap, camera_name: str, step: int) -> dict:
        """Read one frame from `cap` and save it under this camera's subdir.

        Deliberately a single `cap.read()` call, not `record_episode.py`'s
        15-iteration warm-up loop -- that loop exists only for a one-off
        still-shot capture, not a per-step hot path running every control
        tick. Each camera's timestamp is captured independently, immediately
        after its own read -- never shared across cameras (unlike
        `record_episode.py`'s single `ts` variable).
        """
        ok, frame = cap.read()
        captured_at_utc = datetime.now(timezone.utc).isoformat()
        if not ok:
            return {"path": None, "captured_at_utc": captured_at_utc, "error": "capture failed"}

        return self._record_frame(frame, camera_name, step, captured_at_utc)

    def capture_stereo_frame(self, frame, camera_name: str, step: int) -> dict:
        """Records an ALREADY-read frame from the shared `StereoSplitCamera`
        feed the live-bridge policy itself receives (Gap 1 closure).

        Unlike `capture_camera_frame()`, which owns the read via a `cap.read()`
        call, this method takes a raw `np.ndarray` (or `None` on read failure)
        directly -- matching `StereoSplitCamera.read_left()`/`read_right()`'s
        actual return contract, never a `cv2.VideoCapture`-style `(ok, frame)`
        tuple. Exists specifically so the `camera_overhead` diagnostic log
        strictly mirrors what the policy actually saw, instead of a second,
        independently-opened, colliding capture of the same physical device.
        """
        captured_at_utc = datetime.now(timezone.utc).isoformat()
        if frame is None:
            return {"path": None, "captured_at_utc": captured_at_utc, "error": "capture failed"}

        return self._record_frame(frame, camera_name, step, captured_at_utc)

    def capture_depth_map(self, depth_map, camera_name: str, step: int) -> dict:
        """Saves an already-computed depth map (a float32-meters numpy array,
        e.g. from `depth_camera.DepthCameraClient.compute_depth()`) under
        `out_dir/depth_{camera_name}/{step:06d}.npy`.

        A depth map is NOT an 8-bit image `cv2.imwrite()` can losslessly
        encode -- this is a separate method from `_record_frame()`, never
        routed through it, and uses `np.save()` instead of PNG encoding.
        Mirrors `capture_stereo_frame()`'s fail-safe convention: never
        raises, returns a `{"path": None, ..., "error": ...}` shape when
        `depth_map` is `None` (e.g. a failed/timed-out FastFS request).
        """
        import numpy as np  # local import: keeps this module importable without numpy for pure-logic tests

        captured_at_utc = datetime.now(timezone.utc).isoformat()
        if depth_map is None:
            return {"path": None, "captured_at_utc": captured_at_utc, "error": "depth capture failed"}

        (self.out_dir / f"depth_{camera_name}").mkdir(parents=True, exist_ok=True)
        rel_path = Path(f"depth_{camera_name}") / f"{step:06d}.npy"
        np.save(self.out_dir / rel_path, depth_map)
        # np.save() only auto-appends ".npy" if the given path doesn't
        # already end in it -- rel_path already ends in ".npy", and passing
        # a Path object directly (as above) does not double-append.
        return {
            "path": str(rel_path),
            "captured_at_utc": captured_at_utc,
            "shape": list(depth_map.shape),
            "dtype": str(depth_map.dtype),
        }

    def _record_frame(self, frame, camera_name: str, step: int, captured_at_utc: str) -> dict:
        """Shared directory-ensure + PNG-write + result-dict step for both
        `capture_camera_frame()` and `capture_stereo_frame()`.

        Creates `camera_{camera_name}/` lazily on every call (not only in
        `__init__`) so names never declared in `__init__`'s `camera_names`
        argument (e.g. `"overhead_left"`/`"overhead_right"`) still get a
        subdirectory on first write.
        """
        import cv2  # local import: keeps this module importable without cv2 for pure-logic tests

        (self.out_dir / f"camera_{camera_name}").mkdir(parents=True, exist_ok=True)
        rel_path = Path(f"camera_{camera_name}") / f"{step:06d}.png"
        cv2.imwrite(str(self.out_dir / rel_path), frame)
        return {"path": str(rel_path), "captured_at_utc": captured_at_utc}


__all__ = ["IOLogger"]
