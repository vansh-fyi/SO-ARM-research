"""Splits the AR0144 stereo camera's single 2560x720 physical frame into two
independent real camera feeds (`camera2`=left half, `camera3`=right half),
per Plan 11-03's split-stereo camera mapping (`policy_server_launch.md`).

Opens the AR0144 device (cv2 index 1 by default) exactly ONCE per process --
most webcam drivers reject a second concurrent open of the same index -- and
serves both halves from a single shared `.read()` per tick: `read_left()`
and `read_right()` called back-to-back for the same control-loop step share
one physical read (so the stereo pair never desyncs by reading two different
physical frames for what should be the same instant), while the NEXT call to
either method (after both halves of the current tick have been consumed)
triggers a fresh physical read.
"""

import cv2
import numpy as np

# AR0144 native side-by-side stereo resolution and the column split point,
# per policy_server_launch.md's Camera mapping table.
STEREO_WIDTH = 2560
STEREO_HEIGHT = 720
SPLIT_COL = 1280


class StereoSplitCamera:
    """Wraps a single AR0144 `cv2.VideoCapture` device, exposing independent
    left/right halves of its one physical 2560x720 frame as two feeds."""

    def __init__(self, index: int = 1, warmup_frames: int = 15):
        self._index = index
        self._cap = cv2.VideoCapture(index)
        self._opened = self._cap.isOpened()

        if not self._opened:
            print(f"WARNING: StereoSplitCamera: camera index {index} did not open, skipping")
        else:
            # Warm up, same lesson as record_episode.py's record_still().
            for _ in range(warmup_frames):
                ok, _ = self._cap.read()
                if ok:
                    break

        self._left: np.ndarray | None = None
        self._right: np.ndarray | None = None

    @property
    def is_opened(self) -> bool:
        return self._opened

    def _read_and_split(self) -> None:
        """One physical `.read()` shared by a `read_left()`/`read_right()`
        pair for the same tick. If a half is still cached (unconsumed) from
        the current tick's read, skip re-reading -- avoids desyncing the
        stereo pair with two separate physical reads for one instant.
        """
        if self._left is not None or self._right is not None:
            return

        if not self._opened:
            return

        ok, frame = self._cap.read()
        if not ok or frame is None:
            print(f"WARNING: StereoSplitCamera: camera index {self._index} read failed, skipping")
            return

        self._left = frame[:, 0:SPLIT_COL]
        self._right = frame[:, SPLIT_COL:STEREO_WIDTH]

    def read_left(self) -> np.ndarray | None:
        """Left half (columns [0:1280]) of the AR0144's 2560x720 frame, or
        `None` if the device failed to open or the read failed."""
        self._read_and_split()
        frame = self._left
        self._left = None
        return frame

    def read_right(self) -> np.ndarray | None:
        """Right half (columns [1280:2560]) of the AR0144's 2560x720 frame,
        or `None` if the device failed to open or the read failed."""
        self._read_and_split()
        frame = self._right
        self._right = None
        return frame

    def release(self) -> None:
        self._cap.release()


__all__ = ["StereoSplitCamera", "STEREO_WIDTH", "STEREO_HEIGHT", "SPLIT_COL"]
