"""Splits the AR0144 stereo camera's single 1600x600 physical frame into two
independent real camera feeds (`camera2`=left half, `camera3`=right half),
per Plan 11-03's split-stereo camera mapping (`policy_server_launch.md`).

Capture resolution: 1600x600 total (800x600 per half), NOT the AR0144's
higher-resolution 2560x720 mode. 2560x720 was found persistently
stuck/frozen on this machine's camera+macOS combination, confirmed two
independent ways: (1) raw ffmpeg CLI captures returned a byte-for-byte
identical frame across multiple runs, both before and after a full OS
reboot and camera replug; (2) a real prior episode's recorded frames
(`outputs/vla_episode_003/camera_overhead_left/*.png`) were all
byte-for-byte identical across 300 ticks even though `run_vla_episode.py`
calls `read_left()`/`read_right()` fresh every tick (no caching bug on this
project's side) -- proving the freeze is inside ffmpeg/AVFoundation's own
capture pipe for that mode.

Resolution history: an initial fix landed 1280x360 (640x360 per half),
confirmed live via raw ffmpeg testing to capture genuinely changing
frame-to-frame pixel data in the same correct side-by-side layout. This was
then upgraded to the current 1600x600 (800x600 per half) because it is
meaningfully higher resolution (800x600/eye vs 640x360/eye) while remaining
confirmed stable: 1600x600 was confirmed live and stable via raw ffmpeg
testing across 20 consecutive frames, all showing genuine non-zero
frame-to-frame pixel differences (~4.9), with no freezing, and was visually
confirmed via a saved test frame to be the correct side-by-side stereo
layout -- robot workspace visible, correctly split into two 800x600 halves,
with visible parallax between them. The still-broken 2560x720 mode was
re-tested after applying macOS's legacy-camera-plugins-without-sw-camera-
indication system override (a recovery-mode fix) and a fresh reboot, and
remained frozen -- all known software-level fixes for that mode are now
exhausted. 1600x600 is the best available resolution on this hardware: the
camera's advertised mode list jumps directly from 1600x600 to the broken
2560x720 with no smaller intermediate step. Two other intermediate modes,
1280x480 and 1280x712, were also tested but ruled out as single-lens crops
(not the stereo pair), so they are not usable candidates.

This does not meaningfully affect VLA policy quality: the deployed
checkpoint resizes all camera inputs to 256x256 regardless of input
resolution, and 800x600->256x256 remains a smaller downsample ratio than a
native 1920x1080->256x256 feed would be. The one tradeoff is calibration
checkerboard-corner-detection precision, which now IMPROVES relative to the
1280x360 stepping stone (800x600/eye gives more pixels for corner detection
than 640x360/eye did) -- no compensating workaround is needed.

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
made to report this camera's true requested frame size on macOS -- it
silently serves whatever lower resolution (1920x1080, or 1280x720 once an
explicit size is requested) the backend happens to default to, regardless of
`CAP_PROP_FRAME_WIDTH`/`HEIGHT`/`FOURCC` requests. This is a known unfixed
OpenCV bug (opencv/opencv#23368), not a hardware or cable problem -- ffmpeg
reliably captures whatever `-video_size` is requested (now `1600x600`) where
cv2 cannot, and AVFoundation's own "Supported modes" listing (via `ffmpeg
-video_size 9999x9999 ...`'s error output) confirms 1600x600, 1280x360, and
2560x720 are all genuinely supported modes for this device (1280x480 and
1280x712 are also supported but are single-lens crops, not stereo pairs).
This module shells out to ffmpeg instead of relying on cv2. Feeding the VLA a silently-wrong-
resolution/cropped frame during a live episode (instead of failing loudly)
would be a genuine safety risk -- the model would act on corrupted visual
input while still commanding real robot motion.
"""

import subprocess
import threading
import time
from datetime import datetime, timezone

import numpy as np

# AR0144 stereo capture resolution and the column split point, per
# policy_server_launch.md's Camera mapping table. Upgraded from 1280x360 to
# 1600x600 -- see this module's docstring above for the full rationale.
STEREO_WIDTH = 1600
STEREO_HEIGHT = 600
SPLIT_COL = 800

_FRAME_BYTES = STEREO_WIDTH * STEREO_HEIGHT * 3  # raw bgr24


class _FFmpegAVFoundationCapture:
    """Minimal `cv2.VideoCapture`-shaped wrapper (`isOpened()`/`read()`/
    `release()`) around an `ffmpeg` subprocess that streams the AR0144's real
    1600x600 `uyvy422` frame, converted to raw `bgr24`, over stdout."""

    def __init__(self, index: int | str, framerate: int = 30):
        """`index` is the ffmpeg avfoundation `-i` target -- prefer the
        device's AVFoundation NAME (e.g. `"CCB Camera"`) over a numeric
        index, since that numbering has been confirmed to drift between
        process launches on macOS (see this module's docstring)."""
        self._index = index
        self._condition = threading.Condition()
        self._latest = None
        self._closed = False
        self._sequence = 0
        self.last_frame_metadata = {}
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
        self._reader = threading.Thread(target=self._drain_frames, name="stereo-capture", daemon=True)
        if self._proc is not None:
            self._reader.start()

    def isOpened(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _drain_frames(self):
        # Drain at the camera rate even while inference/control is slow. Keeping
        # only the latest complete frame prevents accumulated pipe latency.
        try:
            while not self._closed:
                raw = self._read_exact(_FRAME_BYTES)
                if raw is None:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((STEREO_HEIGHT, STEREO_WIDTH, 3))
                with self._condition:
                    self._sequence += 1
                    self._latest = (frame, time.monotonic(), dict(
                        frame_id=self._sequence,
                        received_at_utc=datetime.now(timezone.utc).isoformat()))
                    self._condition.notify_all()
        finally:
            with self._condition:
                self._closed = True
                self._condition.notify_all()

    def read(self):
        with self._condition:
            self._condition.wait_for(lambda: self._latest is not None or self._closed, timeout=2.0)
            if self._closed or self._latest is None:
                return False, None
            frame, received, metadata = self._latest
            if time.monotonic() - received > 1.0:
                return False, None
            self.last_frame_metadata = dict(metadata)
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
        self._closed = True
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=2)
        if self._reader.is_alive():
            self._reader.join(timeout=2)
        self._proc.stdout.close()


def _open_stereo_capture(index: int):
    """Factory seam for `StereoSplitCamera`'s capture backend -- injectable so
    tests never spawn a real `ffmpeg` subprocess."""
    return _FFmpegAVFoundationCapture(index)


class StereoSplitCamera:
    """Wraps a single AR0144 capture device, exposing independent left/right
    halves of its one physical 1600x600 frame as two feeds."""

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
        self.last_pair_metadata = {}

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

        self.last_pair_metadata = dict(getattr(self._cap, "last_frame_metadata", {}))
        self._left = frame[:, 0:SPLIT_COL]
        self._right = frame[:, SPLIT_COL:STEREO_WIDTH]

    def read_left(self) -> np.ndarray | None:
        """Left half (columns [0:800]) of the AR0144's 1600x600 frame, or
        `None` if the device failed to open or the read failed."""
        self._read_and_split()
        frame = self._left
        self._left = None
        return frame

    def read_right(self) -> np.ndarray | None:
        """Right half (columns [800:1600]) of the AR0144's 1600x600 frame,
        or `None` if the device failed to open or the read failed."""
        self._read_and_split()
        frame = self._right
        self._right = None
        return frame

    def release(self) -> None:
        self._cap.release()


__all__ = ["StereoSplitCamera", "STEREO_WIDTH", "STEREO_HEIGHT", "SPLIT_COL"]
