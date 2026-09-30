# Phase 13: Camera Device Resolution Unification - Research

**Researched:** 2026-09-30
**Domain:** macOS AVFoundation/cv2 camera device resolution, single-process shared-capture pattern, Python stdlib hashing
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Unification scope**
- **D-01:** `record_episode.py` IS in scope. Route its camera opening (both wrist and overhead) through the same shared, name-based resolution helper this phase builds, giving it correct AR0144 stereo support for the first time (it never had this before — cv2.VideoCapture cannot address the AR0144's split-stereo format at all).
- **D-02:** The non-bridge `ScriptedActionSource` path in `run_vla_episode.py` (no `--server-address`) is explicitly OUT of scope — leave its camera-opening code as-is, no changes. `ScriptedActionSource` is a throwaway Phase-11 dry-run harness that toggles the gripper on a fixed schedule and never consumes camera frames for anything — no policy, no vision. There is no second concurrent consumer of the same physical device in this branch, so the original "two independent opens of one device" collision cannot occur here.
- `detect_devices.py`'s own camera-enumeration probing (which necessarily opens raw indices to discover what's connected) is unaffected — this is the bootstrapping step that produces the name↔index correlation the shared helper then consumes, not a candidate for name-based resolution itself.

**device_map.json / by-name resolution helper**
- **D-03:** Build ONE shared resolve-by-name helper used for BOTH cameras, not just the overhead/AR0144:
  - **Overhead/stereo (AR0144):** Already resolved by name via ffmpeg (`"CCB Camera"`) — this existing mechanism becomes (or is wrapped by) the shared helper.
  - **Wrist:** Currently index-only. `detect_devices.py` already correlates the wrist camera's real macOS device name (`"USB Camera"`) via `system_profiler` (see `_correlate_names()`), but never persists it as a `wrist_name` field in `device_map.json` and never uses it at runtime. This phase adds `wrist_name` to `device_map.json` and builds runtime by-name resolution for it: at each process launch, re-run the same `system_profiler`-based correlation to look up "USB Camera"'s CURRENT numeric index, then pass that resolved index to `cv2.VideoCapture` (cv2 itself cannot open a device by name string directly — the resolution step happens before the `cv2.VideoCapture()` call, not inside it).
  - Rationale for doing both, not overhead-only: the literal CAMFIX-01 language is "a single shared, name-based camera-resolution helper" — a helper that only handles one of the two cameras isn't genuinely unified.
- **D-04:** Raw numeric fields (`wrist`, `stereo_overhead`) STAY in `device_map.json` as diagnostic/fallback only — NOT removed. `detect_devices.py`'s initial camera probing still needs numeric indices to enumerate hardware at all, so these fields can't disappear entirely. The shared helper always tries name resolution FIRST and only falls back to the raw index if name correlation fails at runtime (e.g., `system_profiler` unavailable/fails) — same fallback pattern `stereo_overhead_name` already uses today. Document clearly that these raw indices are last-resort only, not for direct/independent use by new code.

**CAMFIX-02 re-verification**
- **D-05:** A FRESH live re-verification episode IS required before this phase closes, run by the user (Claude does not run physical hardware, per Phase 12's CONTEXT.md D-06 — same convention applies here). Do not rely solely on Phase 12's `12-UAT.md` Test 1 pass — this phase adds a NEW by-name resolution layer sitting between `detect_devices.py`'s name correlation and the actual `cv2.VideoCapture`/`StereoSplitCamera` calls (for both cameras). A regression in that new layer would reproduce exactly the bug this phase exists to prevent, and would NOT be caught by re-citing Phase 12's evidence.
- **D-06:** The re-verification's pass criteria cover BOTH cameras, not just overhead: `camera_overhead_left`/`camera_overhead_right` show the real robot workspace (same check as Phase 12's Test 1), AND `camera1`/wrist frames are visually confirmed to be the correct physical camera — since wrist camera resolution is newly changed this phase, a wrist-resolution regression needs its own explicit check, not just unit-test coverage of the new helper.

**CAMFIX-03 provenance (frame hash)**
- **D-07:** Much of "what was actually sent" is already captured by existing Phase-12 code, unrelated to this phase: `BridgeActionSource.get_action()` in `robot_client.py` already records the exact `observation` dict (post-`_downsize_for_transport()`, i.e. the real transmitted frame), and `run_vla_episode.py` already saves these as full-resolution `policy_camera1`/`policy_camera2`/`policy_camera3` PNGs every step (via `io_logger.capture_stereo_frame()`). The genuinely missing piece for CAMFIX-03 is narrower than "build a provenance log from scratch."
- **D-08:** Add a `sha256` hash field to the dict `capture_stereo_frame()` (or a thin wrapper around it) already returns for the `policy_camera1`/`policy_camera2`/`policy_camera3` records — computed from the same frame bytes already being PNG-encoded and saved. No new files, no new per-step image artifacts.
- **D-09 (explicit rejection):** Do NOT add a separate/redundant thumbnail image — declined, since the full-resolution frame is already saved as a PNG per step, a second smaller thumbnail copy adds no real capability.
- **D-10:** The hash can only ever attest to the CLIENT's own local copy of the frame — it structurally CANNOT prove Colab's `PolicyServer` received or used that exact frame (no round-trip confirmation mechanism exists). This is the "client-side-only provenance" disclosure CAMFIX-03 explicitly requires.
- **D-11:** The disclosure lives ONLY as a code docstring on the hashing method/helper — explicitly NOT duplicated into `episode.jsonl` or any new metadata file.

### Claude's Discretion
- Exact shape/naming of the shared resolve-by-name helper function/module (e.g., whether it lives in `detect_devices.py`, a new `camera_resolution.py`, or is added to `stereo_camera.py`) — no user preference expressed beyond "one shared helper for both cameras."
- Exact hash algorithm parameters beyond "sha256" (e.g., hashing the raw numpy array bytes vs. the encoded PNG bytes) — either satisfies D-08/D-10's intent as long as it's deterministic and cheap.
- Whether `wrist_name` resolution failure surfaces as a warning-and-fallback or a harder failure mode before falling back to the raw index — consistent with D-04's "try name first, fall back to raw index" framing; existing `stereo_overhead_name`'s fallback behavior (falls back to raw index, best-effort) is the closest existing precedent to follow unless there's a reason to diverge.

### Deferred Ideas (OUT OF SCOPE)
None raised — discussion stayed within phase scope throughout. The one significant reframing (much of CAMFIX-01/02 already being done by Phase 12's gap closure) narrowed the phase's remaining work rather than expanding it.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|---------------------|
| CAMFIX-01 | A single shared, name-based camera-resolution helper is used by both the model's inference-input path and the `IOLogger` recording path (reusing one `StereoSplitCamera` instance, not two independent camera opens) | Architecture Patterns (Pattern 1, Pattern 2) documents the existing Phase-12-proven name-resolution pattern to generalize, and the structural difference (cv2 vs. `StereoSplitCamera` return shapes) the shared helper must accommodate for both cameras. Code Examples cites the exact existing `resolve_cameras()`/`_correlate_names()` code to extend. Common Pitfalls 2-3 flag the concrete implementation traps in `record_episode.py`'s adaptation. |
| CAMFIX-02 | `camera_overhead` recordings are visually confirmed by a human to show the robot workspace (not the laptop webcam) on a live re-verification episode | Environment Availability and Validation Architecture document this as a manual-only, non-automatable human-judgment gate (D-05/D-06), not something research or planning can close further — flagged accordingly so the planner schedules a `checkpoint:human-verify`-equivalent task rather than an automated test. |
| CAMFIX-03 | The bridge logs a hash/thumbnail of the frame it actually sent to Colab per step, explicitly disclosed as client-side-only provenance | Architecture Patterns (Pattern 3) and Code Examples cite the exact existing `hashlib.sha256` precedent (`depth_camera.py`/`fastfs_server.py`) to replicate, the exact capture point (`capture_stereo_frame()`) to extend, and the fail-safe convention to preserve. Validation Architecture lists the concrete new unit test needed. |
</phase_requirements>

## Summary

This phase is pure backend plumbing on top of an already-working mechanism, not new-technology research. Phase 12's Gap 1 closure (`e7ff126`) already proved the core pattern — resolve the AR0144 overhead camera by AVFoundation name and share one `StereoSplitCamera` instance between the policy-input path and the diagnostic-recording path — and that pattern passed a live UAT visual check. What remains is: (1) generalize that name-first/index-fallback pattern into one shared helper function usable by both cameras (overhead AND wrist), (2) apply it to the one remaining raw-index opener (`record_episode.py`, which never worked correctly for the AR0144 at all), (3) persist the wrist camera's already-computed-but-discarded name correlation into `device_map.json` as `wrist_name`, and (4) add a `sha256` field to the frame-record dict `capture_stereo_frame()` already returns.

No new libraries, no new external dependencies, no new package installs are required — this is 100% assembled from primitives already present in the codebase (`system_profiler SPCameraDataType` parsing in `detect_devices.py`, `ffmpeg`-subprocess capture in `stereo_camera.py`, stdlib `hashlib.sha256` already used elsewhere in this codebase for a different provenance purpose in `depth_camera.py`/`fastfs_server.py`). The main research risk is not "what library to use" but "does macOS's `system_profiler`↔cv2-index positional correlation, already documented as a best-effort heuristic for the AR0144, hold with the same confidence for a second, generic USB camera (the wrist cam)?" — this is unverifiable without live hardware and must be tagged `[ASSUMED]`/flagged for the required live re-verification (D-05/D-06).

**Primary recommendation:** Build one new module-level function, `resolve_camera_capture(role, device_map, ...)` (exact name/location is Claude's discretion per CONTEXT.md), that takes a camera role (`"wrist"` or `"stereo_overhead"`), tries the corresponding `_name` field in `device_map.json` first, falls back to the raw numeric index field on any resolution failure, and returns a capture object appropriate to that camera's actual opening mechanism (a `cv2.VideoCapture`-shaped object for wrist, a `StereoSplitCamera`-shaped object for overhead — these are NOT the same shape, so the helper's return contract must be role-aware, not a single uniform type). Wire this helper into `record_episode.py`, `run_vla_episode.py`'s `_open_cameras()`/`main()`, and `robot_client.py`'s `connect_bridge()`/`_wire_stereo_split_cameras()`.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Camera name↔index correlation (`system_profiler` parsing) | Local process / OS bridge | — | Already owned by `detect_devices.py`; a subprocess call out to macOS's device database, not a network or storage tier concern |
| Shared name-resolution helper | Local process (control/) | — | Pure in-process function; consumed by 3 call sites within the same `control/` package, no service boundary crossed |
| Overhead camera capture (AR0144) | Local process (ffmpeg subprocess) | — | `StereoSplitCamera` already owns this; generalization wraps it, does not replace the transport |
| Wrist camera capture | Local process (cv2/AVFoundation) | — | `cv2.VideoCapture` against an OS-resolved index; unchanged transport, only the index-resolution step changes |
| `device_map.json` persistence | Local filesystem | — | Static JSON written once per `detect_devices.py` run, read at each process launch; no database tier involved |
| Frame provenance hash | Local process (client-side only) | — | Computed and stored entirely within the client process; explicitly cannot attest to server-side (Colab) receipt — no cross-tier proof exists |

## Standard Stack

### Core
No new libraries. This phase is a refactor/generalization of existing in-repo code using stdlib + already-installed dependencies:

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|---------------|
| `hashlib` (stdlib) | Python 3.12 stdlib | `sha256` hash of frame bytes for CAMFIX-03 | Already the established pattern in this codebase for provenance hashing (`vla_bridge/depth_camera.py`'s `calibration_sha256`, `vla_bridge/fastfs_server.py`'s `checkpoint_sha256`) — no reason to introduce a third-party hashing library for a one-line stdlib operation |
| `subprocess` (stdlib) | Python 3.12 stdlib | `system_profiler SPCameraDataType` invocation (already used, unchanged) | Already the established pattern (`detect_devices.py::get_camera_names()`) |
| `opencv-python` | 5.0.0.93 (confirmed installed via `control/.venv`) | Wrist camera capture (`cv2.VideoCapture`) | Unchanged — existing dependency, not new for this phase |

### Supporting
None — no new supporting libraries needed.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Positional `system_profiler`↔cv2-index correlation (existing heuristic) | `AVFoundation`-native device-UID enumeration via a PyObjC binding | PyObjC would give a genuinely stable device identifier (not position-dependent), but is a new third-party dependency for a research-only Colab-driven project where the existing heuristic already has a working, tested precedent for the overhead camera. Out of scope per CONTEXT.md — D-03 explicitly extends the EXISTING correlation mechanism to the wrist camera, not a new one. |
| `sha256` on raw frame bytes | `sha256` on encoded PNG bytes | Both are valid per D-08 ("implementer's choice ... either satisfies D-08/D-10's intent as long as it's deterministic and cheap"). Hashing the same bytes already being written to the PNG file (post-`cv2.imwrite`, i.e. re-reading the just-written file) is more work than hashing the in-memory array; hashing `frame.tobytes()` before encoding is simpler and cheaper. Recommend hashing the raw numpy array bytes (`frame.tobytes()`) directly in `capture_stereo_frame()`/`_record_frame()`, before/independent of the `cv2.imwrite()` call. |

**Installation:** None required — no `pip install` step for this phase.

**Version verification:** Not applicable (no new packages). `opencv-python==5.0.0.93` confirmed already installed via `control/.venv/bin/python -m pip show` (verified live this session).

## Package Legitimacy Audit

**Not applicable.** This phase installs zero new external packages — every capability needed (`hashlib`, `subprocess`, `cv2.VideoCapture`, `ffmpeg` subprocess invocation) is either Python stdlib or an already-installed, already-audited dependency from a prior phase. No `npm view`/`pip index versions` verification is needed because nothing new is being added to `control/requirements.txt`.

## Architecture Patterns

### System Architecture Diagram

```text
                    ┌─────────────────────────────┐
                    │      device_map.json          │
                    │ {wrist: 1, wrist_name: "USB   │  <- NEW: wrist_name field (D-03)
                    │  Camera", stereo_overhead: 2,  │
                    │  stereo_overhead_name:         │
                    │  "CCB Camera"}                 │
                    └───────────────┬─────────────┘
                                    │  read at process launch
                                    v
                    ┌─────────────────────────────┐
                    │   resolve_camera_capture()    │  <- NEW: shared helper (D-03/D-04)
                    │   (name lookup via             │
                    │    system_profiler, per-role)  │
                    │   name resolution FIRST,        │
                    │   raw index FALLBACK             │
                    └───────┬──────────────┬─────────┘
                            │              │
                 role="wrist"        role="stereo_overhead"
                            │              │
                            v              v
                ┌───────────────┐  ┌─────────────────────┐
                │ cv2.VideoCapture│  │ StereoSplitCamera     │
                │ (resolved index)│  │ (ffmpeg subprocess,    │
                │                 │  │  resolved name)        │
                └───────┬────────┘  └──────────┬───────────┘
                        │                       │
        ┌───────────────┼───────────────────────┼─────────────────┐
        │               │                       │                 │
        v               v                       v                 v
┌───────────────┐ ┌──────────────┐  ┌──────────────────┐ ┌───────────────┐
│record_episode.py│ │robot_client.py│  │robot_client.py     │ │run_vla_episode│
│open_cameras()   │ │connect_bridge │  │_wire_stereo_split_  │ │_open_cameras()│
│(D-01: NEW route)│ │(wrist camera1) │  │cameras() (ALREADY   │ │(IOLogger caps,│
│                 │ │                │  │shared per Phase 12) │ │overhead SKIPPED│
│                 │ │                │  │                      │ │in bridge path)│
└───────┬────────┘ └──────┬───────┘  └──────────┬───────────┘ └───────┬───────┘
        │                 │                     │                     │
        v                 v                     v                     v
   camera_{idx}.png  policy camera1        policy camera2/3      camera_wrist/*.png
   camera_{idx}.mp4  (to PolicyServer)      (to PolicyServer)     camera_overhead_left/right
                                                   │                (via shared instance,
                                                   v                 unchanged Phase 12)
                                        IOLogger.capture_stereo_frame()
                                              │
                                              v
                                   NEW: + sha256 hash field (D-08)
                                   (client-side-only provenance,
                                    docstring disclosure only, D-11)
```

### Recommended Project Structure
No new files/directories required (Claude's discretion on exact helper location, per CONTEXT.md). Two viable placements, both acceptable:
```
control/
├── detect_devices.py       # Option A: add resolve_camera_capture() here
│                            #   (co-located with the name-correlation logic it consumes)
├── device_map.json          # gets new "wrist_name" field
└── vla_bridge/
    ├── stereo_camera.py      # Option B: add the helper here instead
    │                          #   (co-located with StereoSplitCamera, which it wraps for overhead)
    ├── robot_client.py        # consumes the helper (wrist + overhead by-name resolution)
    ├── io_logger.py            # gets the sha256 hash field addition
    └── ...
```
Recommendation: **`detect_devices.py`** is the stronger fit — it already owns `_correlate_names()`/`system_profiler` invocation, which the helper's name-lookup step directly depends on at runtime (re-running the correlation at each process launch per D-03: "at each process launch, re-run the same `system_profiler`-based correlation"). Putting the helper in `stereo_camera.py` would create an import from `stereo_camera.py` back into `detect_devices.py`'s correlation logic, or force duplicating the `system_profiler` parsing — `detect_devices.py` importing `StereoSplitCamera` (as it already does, for `STEREO_WIDTH`/`STEREO_HEIGHT`) is the cleaner existing dependency direction.

### Pattern 1: Name-first, index-fallback resolution
**What:** Try resolving a camera by its persisted `*_name` field (re-correlating `system_profiler` output live, not trusting a stale persisted index) first; only fall back to the raw numeric index field if name resolution fails (correlation call fails, name not found among current candidates, etc.).
**When to use:** Every camera-opening call site in `control/` (D-04: "The shared helper always tries name resolution FIRST and only falls back to the raw index if name correlation fails at runtime").
**Example (existing precedent, `resolve_cameras()` in `detect_devices.py`, to be generalized):**
```python
# Source: control/detect_devices.py:407-424 (existing, established pattern)
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
**Gap this phase closes:** the exact same pattern does not exist for `wrist` — `_correlate_names()` computes `names_by_index` (including the wrist camera's real name, confirmed live as `"USB Camera"`), but `resolve_cameras()` never captures `names_by_index.get(wrist_idx)` into a returned `wrist_name` field. This is a one-line omission to fix in `detect_devices.py::resolve_cameras()`, in addition to building the shared runtime-resolution helper that consumes the new field.

### Pattern 2: Role-aware capture object return type
**What:** The shared helper cannot return a single uniform capture type across both cameras — cv2 cannot open the AR0144's split-stereo format at all (confirmed multiple times live, `opencv/opencv#23368`), so the overhead camera MUST go through `StereoSplitCamera`'s ffmpeg subprocess path, while the wrist camera has no such constraint and uses plain `cv2.VideoCapture`.
**When to use:** Any call site that needs "the overhead camera" vs. "the wrist camera" — the helper's contract should be documented as: for `role="wrist"`, returns a `cv2.VideoCapture`-shaped object (has `.isOpened()`/`.read()`/`.release()`); for `role="stereo_overhead"`, returns a `StereoSplitCamera`-shaped object (has `.is_opened`/`.read_left()`/`.read_right()`/`.release()` — a DIFFERENT interface, not drop-in compatible with a plain `cv2.VideoCapture`).
**Consequence for `record_episode.py` (D-01):** `open_cameras()`, `record_still()`, and `record_episode()` all currently assume every entry in their `caps` dict is a plain `cv2.VideoCapture`-shaped single-frame-per-`.read()` object (`camera_{idx}.png` / `camera_{idx}.mp4`, one file per camera). Once the overhead camera routes through `StereoSplitCamera`, these functions need either (a) special-case handling for the overhead role (write `camera_overhead_left`/`camera_overhead_right` outputs, mirroring `IOLogger`'s existing left/right split naming convention, or (b) a thin adapter that presents `StereoSplitCamera` as a single `cv2.VideoCapture`-shaped object by internally calling `read_left()` and discarding `read_right()` (loses the right eye's diagnostic value, not recommended). **Recommendation: adapt the recorder to write left/right outputs, matching `IOLogger`'s existing naming convention (`overhead_left`/`overhead_right`)** rather than collapsing to one eye — this keeps `record_episode.py`'s manual-CLI-tool output consistent with the live-episode logger's established schema.

### Pattern 3: `sha256` frame-hash addition, following existing provenance-hash precedent
**What:** Add a `sha256` field to `capture_stereo_frame()`'s (and/or `_record_frame()`'s) returned dict, computed from the frame's bytes.
**When to use:** Every call that logs a `policy_camera1`/`policy_camera2`/`policy_camera3` frame (D-08) — the existing capture point is `run_vla_episode.py::run_episode()`'s `policy_frames[camera] = io_logger.capture_stereo_frame(frame, "policy_" + camera, step=i)` call (line ~347).
**Example (existing hashlib precedent in this codebase, to replicate the pattern, not the exact code):**
```python
# Source: control/vla_bridge/depth_camera.py:90 (existing pattern in this codebase)
self.calibration_sha256 = hashlib.sha256(
    json.dumps(self.calibration_metadata, sort_keys=True).encode()
).hexdigest()
```
```python
# Recommended equivalent for capture_stereo_frame()/_record_frame(), per D-08/D-10:
# hash the raw frame bytes (before PNG encoding -- deterministic, cheap,
# no dependency on cv2.imwrite()'s compression settings).
#
# CLIENT-SIDE-ONLY PROVENANCE: this hash attests only to what this
# process's local copy of the frame was at capture time. It does NOT
# prove what bytes the Colab PolicyServer actually received or used --
# no round-trip confirmation mechanism exists between this client and
# the server (D-10). Disclosure lives here in this docstring only,
# never duplicated into episode.jsonl or any other output (D-11).
frame_sha256 = hashlib.sha256(frame.tobytes()).hexdigest()
```
**Fail-safe convention to follow (existing, D-08 note):** `IOLogger`'s established pattern never raises on capture failure — returns `{"path": None, ..., "error": ...}`. The hash field should degrade the same way: if hashing fails for any reason (unlikely for `frame.tobytes()`, but the `frame is None` case already short-circuits before reaching the hash step in the existing `capture_stereo_frame()`), do not raise; omit or null the field.

### Anti-Patterns to Avoid
- **Removing `device_map.json`'s raw numeric fields:** D-04 explicitly keeps them — `detect_devices.py`'s own initial probing still needs numeric indices to enumerate hardware in the first place, and they remain the fallback tier. Do not "clean up" by deleting `wrist`/`stereo_overhead`.
- **Treating `StereoSplitCamera` as drop-in-compatible with `cv2.VideoCapture`:** it is not — no `.isOpened()` alias matches exactly (`StereoSplitCamera` exposes `.is_opened` as a property, not a method, and has no single `.read()` that returns one frame). Any adapter code must respect this difference explicitly, not assume structural interchangeability.
- **Re-verifying identity via a servo/config write ("test the fix by writing to it"):** irrelevant to cameras directly, but the analogous anti-pattern for THIS phase is treating a successful `cv2.VideoCapture(idx).isOpened()` call as proof of correct camera identity — `isOpened()` returning `True` proves a device exists at that index, not that it's the RIGHT device (this is exactly the root-cause bug class this whole phase exists to close). The name-resolution step must run and be trusted BEFORE any `isOpened()` check is used as a success signal.
- **A second independent open of the AR0144 anywhere:** `stereo_camera.py`'s own docstring states the AR0144 does not tolerate concurrent opens on this rig ("most webcam drivers reject a second concurrent open of the same index"). Any new code path touching the overhead camera MUST reuse an existing `StereoSplitCamera` instance or, if it must construct its own (e.g. `record_episode.py`'s process is entirely separate from `run_vla_episode.py`'s process — they are never run concurrently against the same device by design), construct exactly one within that process, never two.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Camera name↔index correlation | A new PyObjC/AVFoundation binding, a new `system_profiler` parser | `detect_devices.py`'s existing `get_camera_names()`/`_correlate_names()` | Already built, already tested (`test_detect_devices.py` has 30+ passing tests covering this exact logic), already documented with its known limitations (positional heuristic, not guaranteed-stable) |
| AR0144 stereo-format capture | A new cv2-based capture attempt for the split-stereo format | `StereoSplitCamera` (`vla_bridge/stereo_camera.py`) | Already solves the exact problem (cv2's AVFoundation backend cannot address this device's format at all — confirmed via `opencv/opencv#23368`) via a tested `ffmpeg` subprocess wrapper |
| Frame content hashing | A custom rolling checksum, CRC32, or a new hashing dependency | `hashlib.sha256` (stdlib) | Already the established pattern in this exact codebase for provenance hashing (`depth_camera.py`, `fastfs_server.py`); no reason to diverge |

**Key insight:** Every piece this phase needs already exists in the codebase in some form — the work is exclusively "generalize an existing narrow solution to a second case" (wrist camera) and "wire an existing narrow fix into one more call site" (`record_episode.py`), not "invent a new mechanism."

## Common Pitfalls

### Pitfall 1: Assuming `system_profiler`↔cv2-index correlation is equally reliable for the wrist camera as it was confirmed for the AR0144
**What goes wrong:** The positional correlation heuristic (`_correlate_names()`) was "confirmed to hold on this rig this session with all cameras connected" for distinguishing the AR0144 from the built-in FaceTime camera — but that confirmation was specifically in the context of 2-3 cameras with the AR0144 identifiable by its distinctive stereo resolution as a secondary check. The wrist camera (a generic USB webcam) has no equivalent secondary verification signal (no distinctive resolution/format check like `_verify_stereo_via_ffmpeg()` exists for it) — if the positional correlation between `system_profiler`'s name-listing order and cv2's index-probing order silently misaligns for a generic camera, `wrist_name` resolution could silently point at the wrong physical device with no fallback detection.
**Why it happens:** `system_profiler`'s enumeration order and cv2's `VideoCapture` index-assignment order are two independently-implemented OS/driver-level orderings that macOS does not document as guaranteed-identical (the code's own comments confirm this: "macOS does NOT guarantee its enumeration order matches cv2's ascending VideoCapture index order").
**How to avoid:** The planner should treat wrist-name resolution as needing the SAME class of live re-verification as the overhead camera (D-05/D-06 already mandates this: "wrist camera resolution is newly changed this phase ... a wrist-resolution regression needs its own explicit check, not just unit-test coverage"). Do not treat unit-test-passing as sufficient evidence the wrist-name resolution is correct on real hardware.
**Warning signs:** A live re-verification episode where `camera_wrist`/`camera1` frames show something other than the wrist's actual mounted view (e.g., showing the overhead workspace, or the laptop's own camera) would be the direct signal this heuristic failed for the wrist role specifically.

### Pitfall 2: Silent shape mismatch when adapting `record_episode.py` to the shared helper
**What goes wrong:** If the shared helper is called generically (e.g., a loop over `{"wrist": ..., "stereo_overhead": ...}` roles) and the returned capture objects are used through a uniform interface without dispatching on role, code will call `.read()` on what might be a `StereoSplitCamera` (which has no single `.read()` method — it has `read_left()`/`read_right()`), raising an `AttributeError` at runtime, OR silently succeed if an adapter accidentally shadows an unrelated method.
**Why it happens:** `record_episode.py`'s current `open_cameras()`/`record_still()`/`record_episode()` are written assuming a homogeneous `dict[int, cv2.VideoCapture]` — introducing a second capture shape into that same dict without updating every consumer function is the most likely implementation mistake.
**How to avoid:** Treat the overhead camera as a structurally distinct case throughout `record_episode.py`'s call chain (see Pattern 2 above) — either a separate code path/parameter, or an explicit adapter class that both implements the plain-capture interface for the wrist case AND clearly signals when it's wrapping a stereo device.
**Warning signs:** `AttributeError: 'StereoSplitCamera' object has no attribute 'read'` (or `'get'`, since `record_episode.py`'s `record_episode()` also calls `cap.get(cv2.CAP_PROP_FRAME_WIDTH)`, which `StereoSplitCamera` also does not implement) during manual testing or in a new unit test.

### Pitfall 3: Forgetting `record_episode.py` has NO `device_map.json` integration today
**What goes wrong:** Unlike `run_vla_episode.py` (which already reads `device_map.json` for camera defaults when `--camera` is omitted) and `detect_devices.py` (which writes it), `record_episode.py` currently has zero awareness of `device_map.json` — it only accepts `--camera <idx>` CLI args or blind-probes indices 0-5 with no name-based logic at all. A plan that only "routes existing camera opening through the shared helper" without first adding `device_map.json` loading to `record_episode.py` would have nothing to resolve names FROM.
**Why it happens:** `record_episode.py` predates `detect_devices.py`/`device_map.json` (it's one of the oldest scripts in `control/`, per its own docstring referencing "Plan 11-05 Task 2" as a LATER addition).
**How to avoid:** The plan must include a task to add `device_map.json` loading to `record_episode.py` (mirroring `run_vla_episode.py`'s existing `_load_device_map()` pattern, including its freshness-check/staleness-warning behavior) BEFORE the shared helper can have anything to resolve names against, unless the helper itself owns loading `device_map.json` internally (a valid alternative — Claude's discretion).
**Warning signs:** A plan task that says "route `record_episode.py` through the shared helper" without an explicit sub-step for sourcing `device_map.json` (or equivalent) data into that script.

### Pitfall 4: Treating `calibrate_with_preview.py`/`preview_checkerboard.py` as in-scope "fourth paths"
**What goes wrong:** A cursory grep for `cv2.VideoCapture` or camera-opening code might surface these two files as candidates needing the same fix.
**Why it happens:** They are diagnostic/calibration tools that touch the same physical AR0144 camera.
**Verified during this research (not assumed):** Both files already import and use `StereoSplitCamera` directly (`from vla_bridge.stereo_camera import StereoSplitCamera`), already accept a `--stereo-camera-index` CLI flag documented as preferring the AVFoundation NAME over a numeric index, and neither opens the wrist camera at all. They do NOT independently open a raw `cv2.VideoCapture` against the overhead camera's index — they already follow the correct pattern. **These files are genuinely out of scope; no changes needed.** `test_stereo_calibration.py` also has an explicit structural test (`test_run_calibration_session_never_opens_second_capture_no_ffmpeg_reference`) confirming `stereo_calibration.py`'s own headless capture session has no ffmpeg/second-capture reference either.

## Runtime State Inventory

This is a partial-refactor phase (device-resolution logic changes, camera opening call sites change) but NOT a rename/rebrand/string-migration phase — no identifiers are being renamed across the codebase. The Runtime State Inventory trigger (rename/refactor/migration) is being interpreted narrowly here since the actual work is "add a resolution layer," not "rename an entity everywhere it appears." For completeness, the categories are addressed:

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | `device_map.json`'s `wrist`/`stereo_overhead` raw-int fields and `stereo_overhead_name` string field — no rename, but a NEW field (`wrist_name`) is added alongside them | Code edit only (extend `detect_devices.py::resolve_cameras()`'s return dict); no migration of the old fields needed, they stay as-is (D-04) |
| Live service config | None — `device_map.json` is a local file regenerated by `detect_devices.py` on demand, not a remote/UI-configured service | None |
| OS-registered state | None — no Task Scheduler/launchd/pm2 registrations involved; camera indices are OS-assigned at runtime, not registered by this project | None |
| Secrets/env vars | None — no secret or env var references the camera fields by name | None |
| Build artifacts | None — no installed packages or compiled artifacts reference camera device names/indices | None |

**Nothing found requiring data migration** — verified by reading `device_map.json`'s current contents directly and confirming it is entirely regenerated (not incrementally patched) by `detect_devices.py::main()`'s `args.out.write_text(json.dumps(device_map, indent=2))` each run.

## Code Examples

Verified patterns from the actual installed codebase (not external docs — this phase is entirely in-repo generalization):

### Existing name-first/index-fallback pattern (to generalize)
```python
# Source: control/detect_devices.py (live, current file, lines 407-424)
# `stereo_overhead` (cv2 index) is kept for backward compat / the IOLogger's
# own cv2-based recording caps in run_vla_episode.py. `stereo_overhead_name`
# is the AUTHORITATIVE identifier for StereoSplitCamera's ffmpeg capture
# backend: found live during 11-05 Task 3 prep that BOTH cv2's AND ffmpeg's
# own AVFoundation device index numbering can drift between process
# launches on macOS ... a persisted numeric index is not safe to reuse
# across process boundaries. Addressing the device by its AVFoundation NAME
# string (`ffmpeg -i "CCB Camera"`) sidesteps the instability entirely.
# Falls back to the numeric index only if no name could be correlated for
# it (best-effort correlation, see `_correlate_names()`).
stereo_name = names_by_index.get(stereo_idx)
return {
    "wrist": wrist_idx,
    "stereo_overhead": stereo_idx,
    "stereo_overhead_name": stereo_name if stereo_name is not None else stereo_idx,
}
```

### Existing correlation function (already computes wrist's name, currently discarded)
```python
# Source: control/detect_devices.py, lines 248-256 (live, current file)
def _correlate_names(candidates: list[dict], names: list[str]) -> dict[int, str | None]:
    """Best-effort POSITIONAL correlation between `system_profiler`'s device-name
    order and cv2's ascending-index probe order. ... Extra opened indices beyond
    the number of names found get `None` (never excluded by name, since we can't
    confirm either way)."""
    indices = [c["index"] for c in candidates]
    return {idx: name for idx, name in zip_longest(indices, names, fillvalue=None) if idx is not None}
```
Note: `resolve_cameras()` calls `names_by_index.get(stereo_idx)` but never `names_by_index.get(wrist_idx)` — this exact one-line gap is what D-03 closes.

### Existing sha256 provenance pattern (precedent for D-08)
```python
# Source: control/vla_bridge/depth_camera.py, line 90 (live, current file)
self.calibration_sha256 = hashlib.sha256(
    json.dumps(self.calibration_metadata, sort_keys=True).encode()
).hexdigest()
```

### Existing IOLogger fail-safe convention (to preserve for the hash field)
```python
# Source: control/vla_bridge/io_logger.py, lines 128-132 (live, current file)
def capture_stereo_frame(self, frame, camera_name: str, step: int) -> dict:
    captured_at_utc = datetime.now(timezone.utc).isoformat()
    if frame is None:
        return {"path": None, "captured_at_utc": captured_at_utc, "error": "capture failed"}
    return self._record_frame(frame, camera_name, step, captured_at_utc)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Two independent raw-index camera opens for the overhead camera (bridge policy path + IOLogger recording path) | One shared `StereoSplitCamera` instance, resolved by AVFoundation name | Phase 12 Gap 1 closure (`e7ff126`, 2026-09-27) | Already fixed and live-UAT-confirmed for the `--server-address` bridge path; this phase generalizes the PATTERN, doesn't redo the fix |
| Wrist camera: index-only, no name correlation used at runtime | (This phase) Wrist camera resolved by name first, index fallback | Phase 13 (this phase) | Closes the same bug class for the wrist camera before it manifests, rather than reactively |
| `record_episode.py`: raw `cv2.VideoCapture(idx)` for all cameras including overhead (never worked for AR0144's stereo format) | (This phase) Routed through the shared helper; overhead gets `StereoSplitCamera` | Phase 13 (this phase) | `record_episode.py` gains correct AR0144 support for the first time — a capability gain, not just a bug fix |

**Deprecated/outdated:** None — no external library APIs are being deprecated by this phase; it is entirely internal refactoring.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|----------------|
| A1 | The `system_profiler`↔cv2-index positional correlation heuristic, confirmed reliable for the AR0144 (2-3 camera scenario, disambiguated by a secondary resolution check), will hold with similar reliability for the wrist camera (a generic USB webcam with no secondary disambiguation signal) | Common Pitfalls > Pitfall 1; canonical D-03 discussion | If wrong, `wrist_name` could silently resolve to the wrong physical camera at runtime with no automatic detection — exactly the bug class this phase exists to close, but for a NEW camera role. This is why D-05/D-06 mandates live re-verification covering the wrist camera specifically, not just overhead. |
| A2 | Recommending `frame.tobytes()` (raw numpy array bytes) as the hash input, rather than the PNG-encoded bytes, satisfies D-08/D-10's intent | Architecture Patterns > Pattern 3; Alternatives Considered | Low risk — CONTEXT.md explicitly states this choice is "implementer's choice ... either satisfies D-08/D-10's intent as long as it's deterministic and cheap." If the planner disagrees, PNG-encoded bytes is an equally valid choice requiring no research correction, just a different one-line implementation. |
| A3 | `detect_devices.py` is the better location for the new shared helper vs. `vla_bridge/stereo_camera.py` | Recommended Project Structure | Low risk — explicitly "Claude's Discretion" per CONTEXT.md; this is a recommendation with a stated rationale (avoids an awkward import-direction), not a hard requirement. Either location is workable. |
| A4 | Recording the overhead camera in `record_episode.py` as split left/right outputs (mirroring `IOLogger`'s naming convention) is the right adaptation, rather than collapsing to one eye or another approach | Architecture Patterns > Pattern 2 | Medium risk if the planner picks a different split without realizing `record_episode.py`'s current per-camera single-output assumption needs ANY change at all — the actual requirement (from D-01) is just "route through the shared helper," the exact output-shape adaptation is not explicitly specified by CONTEXT.md and is a genuine open design point for the plan to resolve, not something this research can fully close. |

## Open Questions

1. **Does the shared helper own `device_map.json` loading internally, or does each caller load it and pass fields in?**
   - What we know: `run_vla_episode.py` already has its own `_load_device_map()` with freshness-checking logic; `record_episode.py` has none yet.
   - What's unclear: CONTEXT.md's canonical_refs lists `device_map.json`-reading logic changes across three call sites but doesn't mandate a single loading owner.
   - Recommendation: Have the shared helper accept an already-loaded `device_map: dict` (or `None`) as a parameter rather than loading the file itself — keeps it hermetic/testable (consistent with `run_episode()`'s own stated design philosophy: "keeps this function hermetic/testable against a mocked calibration file, not this machine's real hardware cache"), and lets each caller decide whether/how to apply `run_vla_episode.py`'s existing freshness-warning UX. `record_episode.py` should likely gain its own `_load_device_map()`-equivalent (or import/reuse `run_vla_episode.py`'s, if refactored into a shared location) to source the dict it hands the helper.

2. **Should the shared helper live-re-run `system_profiler` correlation on EVERY call, or cache it once per process?**
   - What we know: D-03 says "at each process launch, re-run the same `system_profiler`-based correlation" — this reads as once-per-process, not once-per-camera-open-call.
   - What's unclear: If both `record_episode.py`'s wrist AND overhead resolution, or `run_vla_episode.py`'s IOLogger-caps AND bridge-wiring resolution, each independently trigger a fresh `system_profiler` subprocess call, that's a redundant ~seconds-scale subprocess call per invocation (per `get_camera_names()`'s own 10s timeout budget) for data that won't have changed within one process's lifetime.
   - Recommendation: Have the helper (or its caller) invoke `get_camera_names()`/`probe_camera_candidates()`/`_correlate_names()` ONCE per process and resolve both roles from that single snapshot, rather than re-running per camera. This is a performance/cleanliness recommendation, not a correctness requirement — plan should decide the exact call shape (e.g., a `resolve_all_cameras(device_map) -> {"wrist": <capture>, "stereo_overhead": <capture>}` helper that runs correlation once, vs. one `resolve_camera_capture(role, ...)` call per role that quietly caches at module scope).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|--------------|-----------|---------|----------|
| `opencv-python` (cv2) | Wrist camera capture, cv2-based probing | Yes | 5.0.0.93 (confirmed via `control/.venv`) | — |
| `ffmpeg` CLI | `StereoSplitCamera`'s AR0144 capture backend | Not verifiable in this research environment (macOS-only live hardware dependency; this session's tool sandbox is not the target macOS rig) | — | None — `ffmpeg` is a hard requirement for the overhead camera on the target machine; already an established dependency from Phase 11/12, unchanged by this phase |
| `system_profiler` CLI | Camera name correlation | macOS-only, not verifiable in this session's sandbox | — | `get_camera_names()` already has a documented graceful-degradation fallback (empty list, warning printed, no crash) — unchanged by this phase |
| Real AR0144 + wrist USB camera hardware | Live re-verification (D-05/D-06) | Not available in this coding session (per Phase 12's CONTEXT.md D-06 convention: Claude does not run physical hardware) | — | User runs the live re-verification episode; this is explicitly out of scope for research/planning to resolve further |

**Missing dependencies with no fallback:** None blocking planning or implementation — `ffmpeg`/`system_profiler`/real hardware are all pre-existing, already-accepted constraints from prior phases (11/12), not new blockers introduced by this phase.

**Missing dependencies with fallback:** `system_profiler` unavailability already degrades gracefully (existing code); the shared helper's name-resolution-fails-fall-back-to-raw-index pattern (D-04) is itself the fallback strategy for the whole phase.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | `pytest` (confirmed: `control/.venv/bin/python -m pytest --collect-only` returns 170 tests across `control/test_*.py`) |
| Config file | None found (`pytest.ini`/`pyproject.toml`/`setup.cfg` with `[tool:pytest]` not present in `control/`) — pytest runs with defaults, test discovery via `test_*.py` naming |
| Quick run command | `cd control && .venv/bin/python -m pytest -q test_detect_devices.py test_run_vla_episode.py test_robot_client.py test_io_logger.py` |
| Full suite command | `cd control && .venv/bin/python -m pytest -q` (170 tests, all files) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|---------------------|--------------|
| CAMFIX-01 | Shared name-based helper used by both inference-input and IOLogger recording paths; one `StereoSplitCamera` instance | unit | `pytest test_run_vla_episode.py::test_server_address_never_opens_overhead_camera_index_while_wrist_still_does test_run_vla_episode.py::test_run_episode_records_overhead_left_and_right_from_shared_stereo_camera` (existing, Phase 12) + NEW tests for the generalized helper and `record_episode.py`'s new routing | ✅ existing (Phase 12) partially covers this; ❌ Wave 0 needed for: shared-helper unit tests, `record_episode.py` routing tests, wrist by-name resolution tests |
| CAMFIX-02 | Live re-verification episode's `camera_overhead/*.png` show the robot workspace | manual-only | N/A — human visual confirmation, per D-05/D-06 (Claude does not run physical hardware) | N/A — not automatable by design |
| CAMFIX-03 | `sha256` hash field on `policy_camera1/2/3` records, disclosed as client-side-only | unit | NEW test: assert `capture_stereo_frame()`'s returned dict contains a `sha256` key matching a known `hashlib.sha256(...)` value for a test frame | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cd control && .venv/bin/python -m pytest -q test_detect_devices.py test_run_vla_episode.py test_robot_client.py test_io_logger.py test_record_episode.py` (the last file does not yet exist — see Wave 0 gap below)
- **Per wave merge:** `cd control && .venv/bin/python -m pytest -q` (full 170+ test suite)
- **Phase gate:** Full suite green before `/gsd-verify-work`; live hardware re-verification (D-05/D-06) is a separate, user-run, non-automatable gate that must also pass before phase close

### Wave 0 Gaps
- [ ] `control/test_record_episode.py` — does **not currently exist** (confirmed: no `test_record_episode.py` in the file listing of `control/`). `record_episode.py` currently has ZERO test coverage. This phase's D-01 (route `record_episode.py` through the shared helper) is the natural trigger to create this file — any plan touching `record_episode.py` should include creating baseline tests for `open_cameras()`/`record_still()`/`record_episode()` alongside the new by-name-resolution behavior, following the existing `test_run_vla_episode.py`/`test_stereo_camera.py` injectable-fake-capture patterns (`FakeCapture`, `_install_fake_capture(monkeypatch, ...)`).
- [ ] New tests for the shared `resolve_camera_capture()`-style helper itself (name-first/fallback-to-index behavior, role-dispatch to the correct capture type) — no existing test file covers a function that doesn't exist yet.
- [ ] New test(s) for `detect_devices.py::resolve_cameras()` returning a `wrist_name` field — extend the existing `test_detect_devices.py` test suite (which already has dense coverage of `resolve_cameras()`'s other return-dict behavior, e.g. `test_2560x720_resolution_candidate_always_assigned_stereo_overhead_never_wrist`) with an analogous `wrist_name` assertion.
- [ ] New test(s) for the `sha256` field in `capture_stereo_frame()`'s/`_record_frame()`'s returned dict — extend `test_io_logger.py` (which already has `test_capture_stereo_frame_writes_png_and_returns_path_and_timestamp` as the closest existing precedent to extend).

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|----------------|---------|---------------------|
| V2 Authentication | No | This phase touches only local camera-device resolution and a local JSON file; no auth boundary involved |
| V3 Session Management | No | No session concept in this phase's scope |
| V4 Access Control | No | Single-user local research tooling; no access-control boundary |
| V5 Input Validation | Marginal | `device_map.json` is read as trusted local config (written by `detect_devices.py` in the same trust domain), not external/untrusted input — no new validation surface introduced. The one new external-ish input is `system_profiler`'s subprocess stdout, already parsed defensively (regex-anchored, never raises on unexpected output) by the existing `_parse_camera_names()` |
| V6 Cryptography | Marginal | `sha256` is used here purely as a content-integrity/dedup signal (provenance), not for any security/authentication purpose — `hashlib.sha256` (stdlib, not hand-rolled) is the correct standard tool regardless, consistent with V6's "never hand-roll crypto primitives" principle even though this isn't a security-sensitive use of hashing |

### Known Threat Patterns for this stack
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|-----------------------|
| Camera identity confusion (wrong physical device silently used) | Spoofing (of device identity, not a network actor) | Name-based resolution with a documented fallback (this phase's entire purpose) — the STRIDE-relevant point is that a MISIDENTIFIED local device is the actual threat model here, not a malicious external actor; the mitigation is the same regardless of framing |
| Subprocess command injection via `system_profiler`/`ffmpeg` invocation | Tampering | Both existing `subprocess.run()`/`subprocess.Popen()` calls already use a list-form `cmd` (never `shell=True`, never string-interpolated into a shell command) — this pattern is already correct and should be preserved unchanged in any generalized helper; do not introduce a shell=True call when generalizing |

This is a research-only, single-user, local-hardware project (per CLAUDE.md's Platform Requirements: "Research-only codebase; no production deployment target") — the ASVS lens is included per config's `security_enforcement: true` default, but most categories are correctly N/A for this phase's actual attack surface.

## Sources

### Primary (HIGH confidence)
- `control/detect_devices.py` (live file, read in full this session) — `_correlate_names()`, `resolve_cameras()`, `get_camera_names()`, `_verify_stereo_via_ffmpeg()`
- `control/vla_bridge/stereo_camera.py` (live file, read in full this session) — `StereoSplitCamera`, `_FFmpegAVFoundationCapture`
- `control/vla_bridge/robot_client.py` (live file, read in full this session) — `connect_bridge()`, `_wire_stereo_split_cameras()`, `BridgeActionSource`
- `control/vla_bridge/io_logger.py` (live file, read in full this session) — `capture_stereo_frame()`, `_record_frame()`
- `control/run_vla_episode.py` (live file, read in full this session) — `_open_cameras()`, `_load_device_map()`, `main()`
- `control/record_episode.py` (live file, read in full this session) — `open_cameras()`, `record_still()`, `record_episode()`, `main()`
- `control/calibrate_with_preview.py`, `control/preview_checkerboard.py` (live files, read this session) — confirmed already `StereoSplitCamera`-based, out of scope
- `control/device_map.json` (live file, read this session) — current schema
- `control/vla_bridge/depth_camera.py`, `control/vla_bridge/fastfs_server.py` (grepped this session) — existing `hashlib.sha256` precedent
- `control/test_detect_devices.py`, `control/test_run_vla_episode.py`, `control/test_robot_client.py`, `control/test_io_logger.py`, `control/test_stereo_camera.py`, `control/test_stereo_calibration.py` (test-name-listed this session) — existing test patterns and coverage
- `control/.venv/bin/python -m pytest --collect-only` (run this session) — confirmed 170 tests, pytest framework, no config file
- `pip show opencv-python` (run this session) — confirmed `5.0.0.93` installed
- `.planning/phases/12-bridge-tick-latency-fix/12-03-SUMMARY.md` (read this session) — the "Gap 1 closure" prior-work record, `affects: [13-camera-resolution-unification]` frontmatter
- `.planning/phases/13-camera-device-resolution-unification/13-CONTEXT.md` (read this session) — authoritative scoping decisions D-01 through D-11

### Secondary (MEDIUM confidence)
None — no external/web research was needed for this phase; the entire domain is this repository's own existing, already-diagnosed code.

### Tertiary (LOW confidence)
None.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new libraries; every tool used is already installed and verified present in `control/.venv`
- Architecture: HIGH — generalization of an already-implemented, already-UAT-passed pattern (Phase 12 Gap 1), verified by direct source reading, not inference
- Pitfalls: HIGH for structural/code pitfalls (verified via direct source reading); MEDIUM for the macOS `system_profiler` correlation-stability claim for the wrist camera specifically (A1 in Assumptions Log — genuinely unverifiable without live hardware, correctly flagged as requiring D-05/D-06's live re-verification rather than treated as settled)

**Research date:** 2026-09-30
**Valid until:** No hard expiry — this research is entirely derived from the current state of an in-repo codebase, not an external fast-moving ecosystem. Re-verify only if the underlying source files (`detect_devices.py`, `stereo_camera.py`, `robot_client.py`, `io_logger.py`, `run_vla_episode.py`, `record_episode.py`) change materially before this phase is planned/executed.
