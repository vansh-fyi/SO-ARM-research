# Phase 13: Camera Device Resolution Unification - Context

**Gathered:** 2026-09-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Unify camera device resolution behind one shared, name-based helper used by every code path that opens the AR0144 overhead/stereo camera or the wrist camera, close off the raw-numeric-index fallback so no future path can silently reproduce the "wrong physical camera" bug, and add client-side frame-provenance hashing (CAMFIX-03).

**Critical scoping finding (discovered during this discussion, not in ROADMAP.md's Context/Notes):** ROADMAP.md's Phase 13 description was written before Phase 12's gap-closure wave landed. Phase 12's Plan 03 (commit `e7ff126`, "Gap 1 closure") already implemented the core of CAMFIX-01 for the live `--server-address` bridge path: `run_vla_episode.py` now reuses the single `StereoSplitCamera` instance (`client._stereo_camera`) that feeds the SmolVLA policy for its own `camera_overhead_left`/`camera_overhead_right` diagnostic recording, resolved by device name (`"CCB Camera"`) via ffmpeg, not a second `cv2.VideoCapture` open. Phase 12's UAT (`12-UAT.md` Test 1) already visually confirmed this shows the real robot workspace — CAMFIX-02's literal ask already passed once, against the pre-Phase-13 code.

**What is genuinely still open for this phase, given that finding:**
1. `record_episode.py` — a standalone manual CLI tool, unrelated to the live episode path — still opens the overhead camera via a bare, independent `cv2.VideoCapture(idx)` in its own `open_cameras()`. It never worked correctly for the AR0144's split-stereo format at all (cv2's AVFoundation backend can't address it — this is why the live path went to ffmpeg in the first place). This is a real third path that can still reproduce the underlying bug class if anyone runs it against the overhead index.
2. `device_map.json` still exposes raw numeric fields (`wrist`, `stereo_overhead`) that a future consumer could pick up directly, bypassing name resolution.
3. The wrist camera has never been name-resolved at all — only index-based, even though `detect_devices.py` already correlates its real macOS device name (`"USB Camera"`) via `system_profiler`, it just doesn't persist or use that name anywhere today.
4. CAMFIX-03 (frame provenance hashing) has no existing implementation — genuinely new work, though most of the underlying "what frame was sent" data is already captured (see Specific Ideas).
5. A fresh live re-verification episode is still required this phase (not just relying on Phase 12's evidence), because this phase adds a new by-name resolution layer on top of the already-working Phase 12 mechanism, and a regression in that new layer (e.g., name correlation silently resolving to the wrong index) would reproduce exactly the bug this phase exists to prevent.

</domain>

<decisions>
## Implementation Decisions

### Unification scope
- **D-01:** `record_episode.py` IS in scope. Route its camera opening (both wrist and overhead) through the same shared, name-based resolution helper this phase builds, giving it correct AR0144 stereo support for the first time (it never had this before — cv2.VideoCapture cannot address the AR0144's split-stereo format at all).
- **D-02:** The non-bridge `ScriptedActionSource` path in `run_vla_episode.py` (no `--server-address`) is explicitly OUT of scope — leave its camera-opening code as-is, no changes. Confirmed via discussion: `ScriptedActionSource` is a throwaway Phase-11 dry-run harness that toggles the gripper on a fixed schedule and never consumes camera frames for anything — no policy, no vision. There is no second concurrent consumer of the same physical device in this branch, so the original "two independent opens of one device" collision cannot occur here. Not worth extending scope into pure dev/test scaffolding.
- **`detect_devices.py`'s own camera-enumeration probing** (which necessarily opens raw indices to discover what's connected in the first place) is unaffected — this is the bootstrapping step that produces the name↔index correlation the shared helper then consumes, not a candidate for name-based resolution itself.

### device_map.json / by-name resolution helper
- **D-03:** Build ONE shared resolve-by-name helper used for BOTH cameras, not just the overhead/AR0144:
  - **Overhead/stereo (AR0144):** Already resolved by name via ffmpeg (`"CCB Camera"`) — this existing mechanism becomes (or is wrapped by) the shared helper.
  - **Wrist:** Currently index-only. `detect_devices.py` already correlates the wrist camera's real macOS device name (`"USB Camera"`) via `system_profiler` (see `_correlate_names()`), but never persists it as a `wrist_name` field in `device_map.json` and never uses it at runtime. This phase adds `wrist_name` to `device_map.json` and builds runtime by-name resolution for it: at each process launch, re-run the same `system_profiler`-based correlation to look up "USB Camera"'s CURRENT numeric index, then pass that resolved index to `cv2.VideoCapture` (cv2 itself cannot open a device by name string directly — the resolution step happens before the `cv2.VideoCapture()` call, not inside it).
  - Rationale for doing both, not overhead-only: the literal CAMFIX-01 language is "a single shared, name-based camera-resolution helper" — a helper that only handles one of the two cameras isn't genuinely unified. The user explicitly chose this broader scope after clarifying that cv2's inability to address the AR0144 by name (a hard technical requirement) is different from the wrist camera's index-instability risk (unconfirmed but structurally the same class of risk, and the correlation data already exists to close it cheaply).
- **D-04:** Raw numeric fields (`wrist`, `stereo_overhead`) STAY in `device_map.json` as diagnostic/fallback only — NOT removed. `detect_devices.py`'s initial camera probing still needs numeric indices to enumerate hardware at all, so these fields can't disappear entirely. The shared helper always tries name resolution FIRST and only falls back to the raw index if name correlation fails at runtime (e.g., `system_profiler` unavailable/fails) — same fallback pattern `stereo_overhead_name` already uses today. Document clearly (docstring/comment on the raw fields in `device_map.json`-writing code, or wherever the helper's fallback branch lives) that these raw indices are last-resort only, not for direct/independent use by new code — this is the mechanism that satisfies the roadmap's warning that "device_map.json's raw numeric field must not remain available for a third path to reintroduce the same failure."

### CAMFIX-02 re-verification
- **D-05:** A FRESH live re-verification episode IS required before this phase closes, run by the user (Claude does not run physical hardware, per Phase 12's CONTEXT.md D-06 — same convention applies here). Do not rely solely on Phase 12's `12-UAT.md` Test 1 pass — that tested the Phase-12 mechanism (shared `StereoSplitCamera` instance) which is untouched by this phase, but this phase adds a NEW by-name resolution layer sitting between `detect_devices.py`'s name correlation and the actual `cv2.VideoCapture`/`StereoSplitCamera` calls (for both cameras). A regression in that new layer (e.g., name correlation silently resolving to the wrong physical index) would reproduce exactly the bug this phase exists to prevent, and would NOT be caught by re-citing Phase 12's evidence.
- **D-06:** The re-verification's pass criteria cover BOTH cameras, not just overhead: `camera_overhead_left`/`camera_overhead_right` show the real robot workspace (same check as Phase 12's Test 1), AND `camera1`/wrist frames are visually confirmed to be the correct physical camera — since wrist camera resolution is newly changed this phase (index-only → name-based-with-fallback per D-03), a wrist-resolution regression needs its own explicit check, not just unit-test coverage of the new helper.

### CAMFIX-03 provenance (frame hash)
- **D-07:** Much of "what was actually sent" is already captured by existing Phase-12 code, unrelated to this phase: `BridgeActionSource.get_action()` in `robot_client.py` already records the exact `observation` dict (post-`_downsize_for_transport()`, i.e. the real transmitted frame) passed to lerobot's `control_loop_observation()`, and `run_vla_episode.py` already saves these as full-resolution `policy_camera1`/`policy_camera2`/`policy_camera3` PNGs every step an observation is actually sent (`policy_observation_frames`, via `io_logger.capture_stereo_frame()`). This is pre-existing Phase 12 work, not new. The genuinely missing piece for CAMFIX-03 is narrower than "build a provenance log from scratch."
- **D-08:** Add a `sha256` hash field to the dict `capture_stereo_frame()` (or a thin wrapper around it) already returns for the `policy_camera1`/`policy_camera2`/`policy_camera3` records — computed from the same frame bytes already being PNG-encoded and saved. No new files, no new per-step image artifacts.
- **D-09 (explicit rejection):** Do NOT add a separate/redundant thumbnail image. Discussed and explicitly declined: since the full-resolution frame is already saved as a PNG per step, a second smaller "thumbnail" copy of the same content adds no real capability — only relevant if the full frames weren't already being saved, which they are.
- **D-10:** The hash can only ever attest to the CLIENT's own local copy of the frame (e.g., catching an accidental duplicate-frame send, or post-hoc file corruption) — it structurally CANNOT prove Colab's `PolicyServer` received or used that exact frame (no round-trip confirmation mechanism exists). This is the "client-side-only provenance" disclosure CAMFIX-03 explicitly requires.
- **D-11:** The disclosure lives ONLY as a code docstring on the hashing method/helper — explicitly NOT duplicated into `episode.jsonl` or any new metadata file. Rationale (discussed and confirmed): nothing will ever programmatically read a disclaimer field back — its only audience is a human reading the code or `REQUIREMENTS.md`/`ROADMAP.md` (which already state the "client-side-only provenance" requirement in the existing requirement text). Adding it to per-step JSON records would only add repetitive noise; adding it as a one-time episode-metadata note would require inventing a new metadata mechanism that doesn't exist today, for a caveat nothing consumes programmatically.

### Claude's Discretion
- Exact shape/naming of the shared resolve-by-name helper function/module (e.g., whether it lives in `detect_devices.py`, a new `camera_resolution.py`, or is added to `stereo_camera.py`) — implementer's choice, no user preference expressed beyond "one shared helper for both cameras."
- Exact hash algorithm parameters beyond "sha256" (e.g., hashing the raw numpy array bytes vs. the encoded PNG bytes) — implementer's choice; either satisfies D-08/D-10's intent as long as it's deterministic and cheap.
- Whether `wrist_name` resolution failure (e.g., `system_profiler` returns no match) surfaces as a warning-and-fallback or a harder failure mode before falling back to the raw index — implementer's choice, consistent with D-04's "try name first, fall back to raw index" framing; existing `stereo_overhead_name`'s fallback behavior (falls back to raw index, best-effort) is the closest existing precedent to follow unless the researcher/planner finds a reason to diverge.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Root-cause diagnosis and prior fix (read first — establishes what's already done vs. still open)
- `.planning/phases/12-bridge-tick-latency-fix/12-03-SUMMARY.md` — the "Gap 1 closure" that already implemented shared-instance camera recording for the bridge path; frontmatter explicitly notes `affects: [13-camera-resolution-unification]`.
- `.planning/phases/12-bridge-tick-latency-fix/12-UAT.md` Test 1 — the live visual confirmation already passed for the pre-Phase-13 mechanism; establishes the baseline this phase's fresh re-verification (D-05/D-06) must reconfirm, not just cite.
- `.planning/phases/12-bridge-tick-latency-fix/12-CONTEXT.md` — Phase 12's own context, including D-06 (Claude does not run physical hardware, user runs live verification) which this phase's D-05 follows.

### Source files to modify
- `control/vla_bridge/robot_client.py` — `_wire_stereo_split_cameras()` (~line 197), `connect_bridge()` (wrist camera wiring via `OpenCVCameraConfig`, ~line 165), `BridgeActionSource.get_action()` (~line 372, `self.last_details["observation"]` — the existing capture point CAMFIX-03's hash attaches to). Both the overhead and wrist by-name resolution helper calls land here.
- `control/run_vla_episode.py` — `_open_cameras()` (line 78), `_build_camera_names()` (line 68), `main()`'s device_map/`--stereo-camera-index` resolution (~lines 546-566), `run_episode()`'s `policy_observation_frames` construction (~line 340) where the hash field is added.
- `control/vla_bridge/io_logger.py` — `capture_stereo_frame()` (line 117) and `_record_frame()` (line 173) — hash field addition per D-08 lands here.
- `control/record_episode.py` — `open_cameras()` (line 30) — route through the shared helper per D-01.
- `control/detect_devices.py` — `_correlate_names()`, `resolve_cameras()` (line 345) — source of the existing wrist-name correlation data (`"USB Camera"`) that needs persisting as `wrist_name` per D-03; `_is_builtin_name()` (line 259) for exclusion logic precedent.
- `control/device_map.json` — gets a new `wrist_name` field (D-03); existing `wrist`/`stereo_overhead` raw int fields stay as fallback-only (D-04).

### Requirements / roadmap
- `.planning/REQUIREMENTS.md` lines 164-168 (CAMFIX-01..03).
- `.planning/ROADMAP.md` "Phase 13: Camera Device Resolution Unification" section (~line 411) — NOTE: this section's Context/Notes predate Phase 12's gap-closure fix and describe the bug as if none of it were fixed yet; this CONTEXT.md's `<domain>` section supersedes that framing for planning purposes, but the literal requirement text (CAMFIX-01/02/03) is still accurate and unchanged.

No other external specs — requirements fully captured in ROADMAP.md/REQUIREMENTS.md plus the decisions above.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `control/vla_bridge/stereo_camera.py`'s `StereoSplitCamera` — already resolves the AR0144 by device name via ffmpeg; the pattern to generalize into the shared helper, not reimplement.
- `control/detect_devices.py`'s `_correlate_names()` / `system_profiler SPCameraDataType` parsing — already produces the wrist camera's real device name; just needs its output persisted and reused at runtime instead of discarded after `device_map.json` is written.
- `control/vla_bridge/io_logger.py`'s `capture_stereo_frame()` / `_record_frame()` — the existing frame-save path CAMFIX-03's hash attaches to; no new capture mechanism needed.
- `control/vla_bridge/robot_client.py`'s existing `last_details["observation"]` capture in `BridgeActionSource.get_action()` — already holds the exact transmitted frame bytes; the hash computes from this, not a new interception point.

### Established Patterns
- Name-resolution-with-raw-index-fallback (`stereo_overhead_name` falling back to `stereo_overhead` if correlation fails) is the existing precedent D-04 extends to the wrist camera too.
- `IOLogger`'s fail-safe convention (`{"path": None, ..., "error": ...}` shape on failure, never raises) — the hash field should follow the same "degrade gracefully, log an error marker, don't crash the episode" convention if hashing a frame fails.

### Integration Points
- The shared resolve-by-name helper sits between `detect_devices.py`'s name↔index correlation and every camera-opening call site (`robot_client.py`'s wrist `OpenCVCameraConfig` wiring, `StereoSplitCamera`'s ffmpeg open, `run_vla_episode.py`'s `_open_cameras()`, `record_episode.py`'s `open_cameras()`).
- The hash field is additive to `capture_stereo_frame()`'s existing return dict — does not change any existing consumer of that dict's `path`/`captured_at_utc` fields.

</code_context>

<specifics>
## Specific Ideas

No UI/visual requirements — this is backend/device-resolution plumbing and a log-schema addition, verified via live hardware by the user (same convention as Phase 12), not a design surface.

</specifics>

<deferred>
## Deferred Ideas

None raised — discussion stayed within phase scope throughout. The one significant reframing (much of CAMFIX-01/02 already being done by Phase 12's gap closure) narrowed the phase's remaining work rather than expanding it; captured in `<domain>` above, not deferred to another phase.

### Reviewed Todos (not folded)
None — todo cross-reference step returned no matches for this phase.

</deferred>

---

*Phase: 13-Camera Device Resolution Unification*
*Context gathered: 2026-09-30*
