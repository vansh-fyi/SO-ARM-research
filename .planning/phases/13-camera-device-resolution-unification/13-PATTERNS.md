# Phase 13: Camera Device Resolution Unification - Pattern Map

**Mapped:** 2026-09-30
**Files analyzed:** 6 (new/modified) + 1 new field (`device_map.json`)
**Analogs found:** 6 / 6 (all analogs are in-repo, same codebase — this phase is pure generalization of existing patterns per RESEARCH.md)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|----------------|
| `control/detect_devices.py` (`resolve_cameras()` extension + new shared helper, e.g. `resolve_camera_capture()`) | utility / device-resolution | request-response (in-process function call, not I/O streaming itself) | `control/detect_devices.py::resolve_cameras()` (lines 345-424) — itself, extended | exact (self-extension) |
| `control/record_episode.py` (`open_cameras()`, `record_still()`, `record_episode()`, `main()`) | CLI utility / file-I/O | streaming (camera read loop) + file-I/O (PNG/MP4 write) | `control/run_vla_episode.py::_open_cameras()` (lines 78-105) and `_load_device_map()` (lines 196-208+) | role-match (same "open cams from device_map, skip by role" pattern, different script) |
| `control/vla_bridge/robot_client.py` (`connect_bridge()` wrist wiring ~line 166, `_wire_stereo_split_cameras()` ~line 193) | service / streaming | event-driven (bridge action loop) | itself — `_wire_stereo_split_cameras()` (lines 193-230ish) is the exact pattern to generalize for wrist-by-name | exact (self-extension) |
| `control/vla_bridge/io_logger.py` (`capture_stereo_frame()` / `_record_frame()` — add `sha256` field) | utility / file-I/O | transform (frame → hash + PNG) | itself — `capture_stereo_frame()` (lines 116-132) and `_record_frame()` (lines 165-180) | exact (self-extension) |
| `control/vla_bridge/depth_camera.py` (`calibration_sha256`, line 90) | utility | transform | N/A — this IS the analog for the hashlib pattern | exact (source of pattern) |
| `control/device_map.json` (new `wrist_name` field) | config | CRUD (static JSON, regenerated wholesale) | itself — existing `stereo_overhead_name` field | exact |
| `control/run_vla_episode.py` (`_open_cameras()`, `main()`'s device_map/`--stereo-camera-index` resolution ~546-566, `policy_observation_frames` ~340) | controller/CLI | request-response + streaming | itself — already-working Phase-12 pattern to reuse, not replace | exact (self-extension) |
| `control/test_record_episode.py` (NEW — does not exist yet) | test | n/a | `control/test_run_vla_episode.py` (`FakeCapture`, `_install_fake_capture()` pattern) and `control/test_stereo_camera.py` | role-match |

## Pattern Assignments

### `control/detect_devices.py` — shared `resolve_camera_capture()`-style helper + `wrist_name` field

**Analog:** itself, `resolve_cameras()` (lines 345-424) and `_correlate_names()` (lines 248-256)

**Existing name-first/index-fallback pattern to generalize** (lines 407-424):
```python
# `stereo_overhead` (cv2 index) is kept for backward compat / the IOLogger's
# own cv2-based recording caps in run_vla_episode.py. `stereo_overhead_name`
# is the AUTHORITATIVE identifier for StereoSplitCamera's ffmpeg capture
# backend ... Falls back to the numeric index only if no name could be
# correlated for it (best-effort correlation, see `_correlate_names()`).
stereo_name = names_by_index.get(stereo_idx)
return {
    "wrist": wrist_idx,
    "stereo_overhead": stereo_idx,
    "stereo_overhead_name": stereo_name if stereo_name is not None else stereo_idx,
}
```

**The exact one-line gap to close** — `resolve_cameras()` already computes `names_by_index` (via `_correlate_names()`, lines 248-256) which contains the wrist camera's name too, but never reads it. Add:
```python
wrist_name = names_by_index.get(wrist_idx)
return {
    "wrist": wrist_idx,
    "wrist_name": wrist_name if wrist_name is not None else wrist_idx,
    "stereo_overhead": stereo_idx,
    "stereo_overhead_name": stereo_name if stereo_name is not None else stereo_idx,
}
```

**Correlation source** (lines 248-256):
```python
def _correlate_names(candidates: list[dict], names: list[str]) -> dict[int, str | None]:
    """Best-effort POSITIONAL correlation between `system_profiler`'s device-name
    order and cv2's ascending-index probe order. macOS does NOT guarantee these two
    enumeration orders actually match -- this is a heuristic..."""
    indices = [c["index"] for c in candidates]
    return {idx: name for idx, name in zip_longest(indices, names, fillvalue=None) if idx is not None}
```

**Error handling pattern** — this module raises `DeviceDetectionError` for hard resolution failures (ambiguous/no candidates), but treats name-correlation misses as soft ("falls back to raw index"), never raising just because a name wasn't found. The new shared helper must follow the SAME two-tier convention: hard errors (no camera at all) raise `DeviceDetectionError`; soft failures (name not found / `system_profiler` unavailable) degrade silently to the raw index, per D-04.

**Recommended placement:** `detect_devices.py` (per RESEARCH.md's Architecture Patterns > Recommended Project Structure) — it already owns `_correlate_names()`/`system_profiler` invocation the helper's runtime name-lookup step depends on; avoids a `stereo_camera.py` → `detect_devices.py` import that would invert the existing dependency direction (`detect_devices.py` already imports `StereoSplitCamera` for `STEREO_WIDTH`/`STEREO_HEIGHT`).

---

### `control/record_episode.py` — route through shared helper (D-01)

**Analog:** `control/run_vla_episode.py::_open_cameras()` (lines 78-105) and `_load_device_map()` (lines 196-208+)

**Current state (to be replaced/extended)** — `record_episode.py:30-38`:
```python
def open_cameras(indices):
    caps = {}
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"WARNING: camera index {idx} did not open, skipping")
            continue
        caps[idx] = cap
    return caps
```
This is a bare, independent `cv2.VideoCapture(idx)` per D-01/Pitfall 3 — has ZERO `device_map.json` awareness today (unlike `run_vla_episode.py`). Must gain `_load_device_map()`-equivalent loading before the helper has anything to resolve names from.

**Target pattern to copy** — `run_vla_episode.py::_open_cameras()` (lines 78-105):
```python
def _open_cameras(
    camera_indices: list[int],
    camera_names: dict[int, str],
    skip_names: frozenset[str] = frozenset(),
) -> dict[int, "cv2.VideoCapture"]:
    """... Exists to stop this script's own diagnostic recording from opening a
    second, independent capture of the AR0144 -- the physical device backing
    the "overhead" index -- while `connect_bridge()`'s `StereoSplitCamera`
    already holds it open exclusively..."""
    caps: dict[int, "cv2.VideoCapture"] = {}
    for idx in camera_indices:
        if camera_names.get(idx) in skip_names:
            continue
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            caps[idx] = cap
        else:
            print(f"WARNING: camera index {idx} did not open, skipping")
    return caps
```

**Critical structural warning (RESEARCH.md Pitfall 2):** once the overhead role routes through `StereoSplitCamera` instead of `cv2.VideoCapture`, `record_still()`/`record_episode()`'s current homogeneous-`dict[int, cv2.VideoCapture]` assumption breaks — `StereoSplitCamera` has `.is_opened` (property, not method), `.read_left()`/`.read_right()` (no single `.read()`), and no `.get(cv2.CAP_PROP_FRAME_WIDTH)`. Per Pattern 2/A4 in RESEARCH.md, adapt `record_episode.py` to write `camera_overhead_left`/`camera_overhead_right` outputs (mirroring `IOLogger`'s naming convention), rather than collapsing to one eye.

**`StereoSplitCamera`'s actual interface** (`control/vla_bridge/stereo_camera.py`, confirmed via grep):
```python
class StereoSplitCamera:
    def __init__(self, index: int | str = 1, warmup_frames: int = 15, capture_factory=None): ...
    def is_opened(self) -> bool: ...
    def read_left(self) -> np.ndarray | None: ...
    def read_right(self) -> np.ndarray | None: ...
    def release(self) -> None: ...
```

---

### `control/vla_bridge/robot_client.py` — wrist by-name resolution

**Analog:** itself — `_wire_stereo_split_cameras()` (lines 193+) is the proven pattern; `connect_bridge()`'s wrist wiring (line 166) is what needs the new by-name resolution layered in front of it.

**Current wrist wiring** (line 166):
```python
if wrist_camera_index is not None:
    robot_config.cameras["camera1"] = OpenCVCameraConfig(index_or_path=wrist_camera_index)
```
Per D-03, `wrist_camera_index` should be resolved by the new shared helper (name-first, raw-index fallback) BEFORE reaching this line — this line itself likely stays unchanged; the resolution happens at the call site that currently passes a raw index into `connect_bridge(...)`.

**Overhead already-correct pattern to generalize, not reimplement** (`_wire_stereo_split_cameras()`, lines 193-230):
```python
def _wire_stereo_split_cameras(client, stereo_camera_index: int | str = 1, stereo_camera=None) -> None:
    """Make every observation this client's `control_loop_observation()`
    sends include real `camera2`/`camera3` split-stereo feeds, sourced from
    a single shared `StereoSplitCamera` -- not two independent
    `cv2.VideoCapture(1)` opens of the same physical AR0144 device...."""
```

**Provenance capture point (CAMFIX-03 attaches here, no new interception point needed)** — `BridgeActionSource.get_action()` (lines 372-391):
```python
class BridgeActionSource:
    def __init__(self):
        self.last_details = {}

    def get_action(self, ...):
        self.last_details = dict(raw_model_output={}, bridge_flags=[], observation=None)
        ...
        self.last_details["observation"] = observation
        self.last_details["observation_recorded_at_unix_s"] = time.time()
        self.last_details["stereo_frame"] = dict(getattr(getattr(self.client, "_stereo_camera", None), "last_pair_metadata", {}))
```

---

### `control/vla_bridge/io_logger.py` — sha256 hash field (CAMFIX-03)

**Analog:** itself — `capture_stereo_frame()` (lines 116-132) / `_record_frame()` (lines 165-180); hashing precedent from `control/vla_bridge/depth_camera.py:90`

**Fail-safe convention to preserve exactly** (lines 116-132):
```python
def capture_stereo_frame(self, frame, camera_name: str, step: int) -> dict:
    """Records an ALREADY-read frame from the shared `StereoSplitCamera`
    feed the live-bridge policy itself receives (Gap 1 closure)...."""
    captured_at_utc = datetime.now(timezone.utc).isoformat()
    if frame is None:
        return {"path": None, "captured_at_utc": captured_at_utc, "error": "capture failed"}
    return self._record_frame(frame, camera_name, step, captured_at_utc)
```

**Where the PNG write + result dict is built** (lines 165-180):
```python
def _record_frame(self, frame, camera_name: str, step: int, captured_at_utc: str) -> dict:
    import cv2  # local import: keeps this module importable without cv2 for pure-logic tests
    (self.out_dir / f"camera_{camera_name}").mkdir(parents=True, exist_ok=True)
    rel_path = Path(f"camera_{camera_name}") / f"{step:06d}.png"
    if not cv2.imwrite(str(self.out_dir / rel_path), frame):
        raise OSError(f"Failed to write camera frame: {rel_path}")
    return {"path": str(rel_path), "captured_at_utc": captured_at_utc}
```
Add a `sha256` key to this returned dict, computed from `frame.tobytes()` (raw numpy bytes, before PNG encoding — cheaper, deterministic, per RESEARCH.md's A2/Alternatives). Follow the exact same import-locally, no-raise convention already used here.

**Hashing precedent to replicate (pattern, not exact code)** — `control/vla_bridge/depth_camera.py:90`:
```python
self.calibration_sha256 = hashlib.sha256(
    json.dumps(self.calibration_metadata, sort_keys=True).encode()
).hexdigest()
```

**Docstring disclosure required on the hashing method itself (D-10/D-11)** — must state client-side-only provenance; never duplicated into `episode.jsonl` or any other output:
```python
# CLIENT-SIDE-ONLY PROVENANCE: this hash attests only to what this
# process's local copy of the frame was at capture time. It does NOT
# prove what bytes the Colab PolicyServer actually received or used --
# no round-trip confirmation mechanism exists between this client and
# the server (D-10). Disclosure lives here in this docstring only,
# never duplicated into episode.jsonl or any other output (D-11).
frame_sha256 = hashlib.sha256(frame.tobytes()).hexdigest()
```

---

### `control/run_vla_episode.py` — call-site wiring (unchanged shape, new resolved inputs)

**Analog:** itself — this file's existing Phase-12 mechanism is the thing the new helper feeds into, not something being replaced.

**`_load_device_map()` pattern to mirror in `record_episode.py`** (lines 196-208):
```python
def _load_device_map() -> dict | None:
    """Loads `control/device_map.json` ... Returns `None` -- never raises -- on
    any failure (missing file, unreadable, invalid JSON), so callers can
    fall through to requiring explicit CLI args instead of crashing."""
    if not DEVICE_MAP_PATH.exists():
        return None
    try:
        device_map = json.loads(DEVICE_MAP_PATH.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: {DEVICE_MAP_PATH} exists but could not be read ({e}); ignoring it.")
        return None
    detected_at = device_map.get("detected_at")
    ...
```

**`_build_camera_names()` semantic-labeling precedent** (lines 66-75):
```python
CAMERA_SEMANTIC_NAMES = ["wrist", "overhead"]

def _build_camera_names(camera_indices: list[int]) -> dict[int, str]:
    """Maps physical camera indices to semantic names positionally..."""
    return dict(zip(camera_indices, CAMERA_SEMANTIC_NAMES))
```

## Shared Patterns

### Name-first, index-fallback resolution (CAMFIX-01/D-03/D-04)
**Source:** `control/detect_devices.py::resolve_cameras()` (lines 407-424), `_correlate_names()` (lines 248-256)
**Apply to:** the new shared helper (wherever placed), `robot_client.py`'s wrist wiring call site, `record_episode.py`'s new `open_cameras()`, `run_vla_episode.py`'s existing `--stereo-camera-index` resolution.
```python
name = names_by_index.get(idx)
resolved = name if name is not None else idx  # fallback tier — never raise here
```

### Never open a second independent capture of the AR0144
**Source:** `run_vla_episode.py::_open_cameras()` docstring (lines 83-94) and `stereo_camera.py`'s own module docstring (AR0144 does not tolerate concurrent opens).
**Apply to:** any new call site touching the overhead camera — must reuse an existing `StereoSplitCamera` instance within a process, or construct exactly one, never two, per RESEARCH.md's Anti-Patterns.

### IOLogger fail-safe dict shape (`{"path": None, ..., "error": ...}`)
**Source:** `control/vla_bridge/io_logger.py` lines 111-113, 128-130, 149-150
**Apply to:** the new `sha256` field addition — degrade gracefully (omit/null field) rather than raise, matching every existing method in this class.

### hashlib.sha256 provenance pattern
**Source:** `control/vla_bridge/depth_camera.py:90` (`calibration_sha256`), `fastfs_server.py` (`checkpoint_sha256`)
**Apply to:** `io_logger.py`'s new frame-hash field — reuse stdlib `hashlib`, no new dependency, hash raw bytes not encoded PNG (implementer's discretion per D-08, raw bytes recommended).

### Subprocess invocation safety (list-form, never shell=True)
**Source:** `detect_devices.py`'s `subprocess.run(cmd, ...)` calls (e.g. `_verify_stereo_via_ffmpeg`, lines 326-338) and `stereo_camera.py`'s ffmpeg subprocess construction
**Apply to:** any generalized helper that re-invokes `system_profiler`/`ffmpeg` — always list-form `cmd`, never `shell=True`, never string-interpolated.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `control/test_record_episode.py` | test | n/a | Does not exist yet — `record_episode.py` currently has zero test coverage (confirmed in RESEARCH.md's Wave 0 Gaps). Closest structural analog for test-writing style is `control/test_run_vla_episode.py`'s `FakeCapture`/`_install_fake_capture(monkeypatch, ...)` injectable-fake-capture pattern and `control/test_stereo_camera.py`'s fake `capture_factory` pattern — use these as the template, not a literal pre-existing test of this file. |
| New shared `resolve_camera_capture()`-style helper (role-aware dispatch: `cv2.VideoCapture`-shaped for wrist vs. `StereoSplitCamera`-shaped for overhead) | utility | request-response | No existing function in the codebase currently dispatches a single call to two structurally different return types by role — `resolve_cameras()` returns only index/name data, never a capture object itself. This is genuinely new composition (of two fully-existing lower-level pieces), not a copy of an existing analog. |

## Metadata

**Analog search scope:** `control/`, `control/vla_bridge/` — no search outside `control/` needed (RESEARCH.md confirmed this phase's entire attack surface is within `control/`).
**Files scanned:** `detect_devices.py`, `record_episode.py`, `run_vla_episode.py`, `vla_bridge/robot_client.py`, `vla_bridge/io_logger.py`, `vla_bridge/stereo_camera.py`, `vla_bridge/depth_camera.py`, `device_map.json`.
**Pattern extraction date:** 2026-09-30
