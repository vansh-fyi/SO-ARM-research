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

    Also owns per-camera frame capture (`capture_camera_frame`), writing each
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
        import cv2  # local import: keeps this module importable without cv2 for pure-logic tests

        ok, frame = cap.read()
        captured_at_utc = datetime.now(timezone.utc).isoformat()
        if not ok:
            return {"path": None, "captured_at_utc": captured_at_utc, "error": "capture failed"}

        rel_path = Path(f"camera_{camera_name}") / f"{step:06d}.png"
        cv2.imwrite(str(self.out_dir / rel_path), frame)
        return {"path": str(rel_path), "captured_at_utc": captured_at_utc}


__all__ = ["IOLogger"]
