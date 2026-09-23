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

Capture backend: `ffmpeg` subprocess, NOT `cv2.VideoCapture`. Found live
during 11-05 Task 3 prep: `cv2.VideoCapture`'s AVFoundation backend cannot be
made to report this camera's true 2560x720 frame on macOS -- it silently
serves whatever lower resolution (1920x1080, or 1280x720 once an explicit
size is requested) the backend happens to default to, regardless of
`CAP_PROP_FRAME_WIDTH`/`HEIGHT`/`FOURCC` requests. This is a known unfixed
OpenCV bug (opencv/opencv#23368), not a hardware or cable problem --
AVFoundation itself confirms 2560x720 is a genuinely supported mode for this
device (via `ffmpeg -video_size 9999x9999 ...`'s "Supported modes" error
listing). `ffmpeg -f avfoundation -pixel_format uyvy422 -video_size
2560x720` reliably captures the real frame where cv2 cannot, so this module
shells out to it instead. Feeding the VLA a silently-wrong-resolution/cropped
frame during a live episode (instead of failing loudly) would be a genuine
safety risk -- the model would act on corrupted visual input while still
commanding real robot motion.
"""

import subprocess

import numpy as np

# AR0144 native side-by-side stereo resolution and the column split point,
# per policy_server_launch.md's Camera mapping table.
STEREO_WIDTH = 2560
STEREO_HEIGHT = 720
SPLIT_COL = 1280

_FRAME_BYTES = STEREO_WIDTH * STEREO_HEIGHT * 3  # raw bgr24


class _FFmpegAVFoundationCapture:
    """Minimal `cv2.VideoCapture`-shaped wrapper (`isOpened()`/`read()`/
    `release()`) around an `ffmpeg` subprocess that streams the AR0144's real
    2560x720 `uyvy422` frame, converted to raw `bgr24`, over stdout."""

    def __init__(self, index: int | str, framerate: int = 30):
        """`index` is the ffmpeg avfoundation `-i` target -- prefer the
        device's AVFoundation NAME (e.g. `"CCB Camera"`) over a numeric
        index, since that numbering has been confirmed to drift between
        process launches on macOS (see this module's docstring)."""
        self._index = index
        cmd = [
            "ffmpeg", "-loglevel", "error",
            "-f", "avfoundation",
            "-pixel_format", "uyvy422",
            "-video_size", f"{STEREO_WIDTH}x{STEREO_HEIGHT}",
            "-framerate", str(framerate),
            "-i", str(index),
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=_FRAME_BYTES * 2
            )
        except OSError as e:
            print(f"WARNING: _FFmpegAVFoundationCapture: failed to start ffmpeg for index {index}: {e}")
            self._proc = None

    def isOpened(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def read(self):
        if not self.isOpened():
            return False, None
        raw = self._read_exact(_FRAME_BYTES)
        if raw is None:
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((STEREO_HEIGHT, STEREO_WIDTH, 3))
        return True, frame

    def _read_exact(self, n: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def release(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()


def _open_stereo_capture(index: int):
    """Factory seam for `StereoSplitCamera`'s capture backend -- injectable so
    tests never spawn a real `ffmpeg` subprocess."""
    return _FFmpegAVFoundationCapture(index)


class StereoSplitCamera:
    """Wraps a single AR0144 capture device, exposing independent left/right
    halves of its one physical 2560x720 frame as two feeds."""

    def __init__(self, index: int | str = 1, warmup_frames: int = 15, capture_factory=None):
        self._index = index
        factory = capture_factory if capture_factory is not None else _open_stereo_capture
        self._cap = factory(index)
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
