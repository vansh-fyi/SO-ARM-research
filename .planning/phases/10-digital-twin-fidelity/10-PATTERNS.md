
> **2026-09-20 established-model update:** [10-ESTABLISHED-MODEL.md](10-ESTABLISHED-MODEL.md)
> is authoritative for the accepted Coppelia-aligned assembly, black housing,
> yellow jaws, 0.036 m per-jaw cap, corrected starting pose and LIBERO wiring.
> Earlier settings and verification results below are historical and do not
> establish policy success for the accepted model.

# Phase 10: Digital-Twin Fidelity - Pattern Map

**Mapped:** 2026-09-18
**Files analyzed:** 6 (2 new scripts, 1 new test, 3 modified assets)
**Analogs found:** 6 / 6

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|----------------|
| `scripts/mjcf_to_urdf.py` (new) | utility / offline transform | file-I/O (parse MJCF XML, emit URDF XML) | `coppelia/export_model_library.py` | role-match (offline one-off transform script, top-level `scripts`-style dir, `Path(__file__).parent` convention) |
| `scripts/verify_urdf.py` (new) | utility / validation | file-I/O + assertions | `diagnostics/measure_object_depth.py` | role-match (standalone diagnostic CLI script: argparse-free/minimal, `print()` progress, `SystemExit` on failure, load-then-report shape) |
| `scripts/test_verify_urdf.py` (new) | test | request-response (load fixture, assert) | `LIBERO/libero/libero/envs/test_camera_config.py` | exact (same repo's only precedent for a real, no-mock integration-style pytest file against a generated/loaded artifact; also documents the `sys.path` repo-root-anchoring convention this new test will likely need) |
| `So-101/So-101.urdf` (generated output, overwritten) | config / generated artifact | transform output (XML) | `coppelia/soarm_parallel_gripper.urdf` | exact (same URDF dialect/conventions already used in this repo: `<link>`/`<joint type="prismatic">`, relative mesh `filename`, `<limit lower/upper effort velocity>`, `<dynamics damping/friction>`) |
| `LIBERO/libero/libero/assets/robots/soarm101/robot.xml` (source of truth, read + possibly limit-edited) | config / simulation asset | transform input | itself (`robot.xml`) | exact — this file's own header comment block is the canonical pattern for how joint/body edits must be documented (numbered rationale comments, explicit "why" per changed value) |
| `LIBERO/libero/libero/assets/grippers/soarm_gripper.xml` (modified: clamp visual offset fix) | config / simulation asset | transform input + visual-geom edit | itself (`soarm_gripper.xml`) — specifically the `main_frame_visual` geom's own asset-note comment | exact — the file already documents the exact "iterative offscreen-render dialing" method to use for the D-04 clamp fix, applied previously to `main_frame_visual` |

## Pattern Assignments

### `scripts/mjcf_to_urdf.py` (utility, file-I/O transform)

**Analog:** `coppelia/export_model_library.py`

**Module docstring + path-constant pattern** (lines 1-23):
```python
#!/Users/hp/.pyenv/versions/3.12.11/bin/python3
"""Create clean SO-ARM entries for CoppeliaSim's user model browser.

This script intentionally uses the verified CoppeliaSim scene as the arm
source. It does not import the hand-converted whole-arm URDF.
...
"""

import socket
import sys
from pathlib import Path


ROOT = Path(__file__).parent.resolve()
REFERENCE_SCENE = ROOT / "reference" / "so101_dev.ttt"
PARALLEL_URDF = ROOT / "soarm_parallel_gripper.urdf"
MESH_DIR = ROOT / "meshes"
```
Copy this shape for `mjcf_to_urdf.py`: module docstring stating exactly what
this script does and does NOT do (mirrors "does not import the
hand-converted whole-arm URDF" — state "generates the URDF fresh from MJCF,
does not patch `So-101/So-101.urdf` in place", per D-02), `ROOT =
Path(__file__).resolve().parent.parent` (repo root, matching
`explorations/`'s `ROOT` convention too), then explicit path constants for
`robot.xml`, `soarm_gripper.xml`, and the output `So-101/So-101.urdf`.

**Fail-loud precondition checks** (lines 36-38):
```python
for required in (REFERENCE_SCENE, PARALLEL_URDF, MESH_DIR):
    if not required.exists():
        sys.exit(f"ERROR: missing {required}")
```
Apply the same pattern before parsing: assert `robot.xml` and
`soarm_gripper.xml` exist and that expected bodies/joints
(`right_hand`, `wrist_roll`, `gripper_left`, `gripper_right`) are present in
the parsed tree before emitting URDF — RESEARCH.md's Security Domain
section explicitly calls for "fail loudly, not silently emit a malformed
URDF" on missing expected structure.

**Progress + completion print pattern** (lines 109-117, generalized project convention from CLAUDE.md "Logging" section — `Saved → {out_path}"` with arrow notation):
```python
print("Created CoppeliaSim model-browser entries:")
for name, destination in (...):
    if not destination.exists():
        sys.exit(f"Export reported success, but {destination} was not written.")
    print(f"  {name}: {destination}")
```
End `mjcf_to_urdf.py` with `print(f"Saved → {out_path}")` (exact repo-wide
convention per CLAUDE.md Logging section) after writing `So-101/So-101.urdf`.

**MJCF source structure to walk** — read directly from `robot.xml` (lines
54-120 read this session) and `soarm_gripper.xml` (full file, lines 1-104
read this session): explicit `<body name="..." pos="..." quat="...">` /
`<joint axis="..." name="..." type="hinge|slide" range="lo hi" .../>`
nesting, terminating the arm at `right_hand` and attaching the gripper's
`right_gripper` root there (per the documented "body split at the
wrist_roll->gripper boundary" comment at `robot.xml` lines 10-13). This is
the exact tree `mjcf_to_urdf.py` must translate hinge→revolute,
slide→prismatic (see RESEARCH.md Pattern 1 for the `quat`→`rpy` closed-form
conversion to use for each `<joint><origin rpy="...">`).

---

### `So-101/So-101.urdf` (generated output)

**Analog:** `coppelia/soarm_parallel_gripper.urdf` (full file, 47 lines, read this session)

This is the closest in-repo URDF dialect example and should be used as the
literal structural template for the generated file's `<link>`/`<joint>`
syntax:
```xml
<link name="parallel_gripper_left_jaw">
  <visual>
    <origin xyz="0.15241 -0.02808 0.15206" rpy="3.14159 0 1.5708"/>
    <geometry><mesh filename="meshes/clamp_1_visual.stl" scale="0.001 0.001 0.001"/></geometry>
    <material name="jaw"/>
  </visual>
  <collision><origin xyz="-0.052 -0.0092 0" rpy="0 0 0"/><geometry><box size="0.044 0.018 0.040"/></geometry></collision>
  <inertial><origin xyz="-0.052 -0.0092 0" rpy="0 0 0"/><mass value="0.05"/><inertia ixx="4e-5" ixy="0" ixz="0" iyy="4e-5" iyz="0" izz="1e-5"/></inertial>
</link>

<joint name="parallel_gripper_left" type="prismatic">
  <parent link="parallel_gripper_base"/><child link="parallel_gripper_left_jaw"/>
  <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 -1 0"/>
  <limit lower="0" upper="0.042" effort="60" velocity="0.5"/><dynamics damping="50" friction="0.5"/>
</joint>
```
Key transferable conventions for `mjcf_to_urdf.py`'s output:
- Mesh `filename` attributes are **relative** (`meshes/clamp_1_visual.stl`),
  never absolute — directly satisfies TWIN-04. Use paths relative to the
  URDF's own file location (`So-101/*.dae` / `coppelia/meshes/*.stl`, per
  D-02's mesh sourcing split).
- Prismatic joint `<limit>` includes `effort`/`velocity` (URDF requires
  these even though MJCF's `slide` joints don't have a direct equivalent —
  reuse `soarm_gripper.xml`'s `forcerange`/reasonable velocity as stand-ins,
  same values already used in this reference gripper URDF: `effort="60"
  velocity="0.5"`).
- `<dynamics damping="..." friction="..."/>` mirrors MJCF's `damping`/
  `frictionloss` on the corresponding `<joint>` (`soarm_gripper.xml` lines
  85, 93: `damping="50" ... frictionloss="0.5"` — exact match to this
  URDF's `damping="50" friction="0.5"`, confirming the gripper is a direct,
  already-correct precedent).
- Axis sign directly carries over: `axis xyz="0 -1 0"` for left jaw / `"0 1
  0"` for right jaw — identical to `soarm_gripper.xml`'s `axis="0 -1 0"` /
  `axis="0 1 0"` on `gripper_left`/`gripper_right` (TWIN-03's direction
  requirement is satisfied by a direct pass-through, not a re-derivation).

**Do not copy structurally:** `So-101/So-101.urdf`'s own current content
(298 lines) — it is the broken artifact being replaced (absolute
`file:///Users/hp/Downloads/...` paths, disconnected gripper root, no
`wrist_roll`/gripper joints, terminal `wrist_link_respondable`). Read only
to confirm what NOT to reproduce.

---

### `scripts/verify_urdf.py` (utility, validation)

**Analog:** `diagnostics/measure_object_depth.py` (full file, 169 lines, read this session)

**Module docstring + precondition-check pattern** (lines 1-21, 74-76):
```python
"""Measure a dark object's distance from the camera using the stereo calibration.
...
Requires diagnostics/.venv (plain OpenCV/numpy) and an existing
diagnostics/stereo_calibration.npz (run stereo_calibration_capture.py +
stereo_calibrate.py first if it doesn't exist yet).
"""
...
if not CALIB_PATH.exists():
    raise SystemExit(f"No calibration found at {CALIB_PATH}. Run "
                      "stereo_calibration_capture.py + stereo_calibrate.py first.")
```
Apply directly: `verify_urdf.py`'s docstring should state its dependency
(`yourdfpy`, `pip install yourdfpy` into the `libero` conda env) and that it
expects `So-101/So-101.urdf` to already exist (generated by
`mjcf_to_urdf.py` first) — `raise SystemExit(...)` with an actionable
fix-it message if missing, exact same idiom.

**Report + reliability-flag print pattern** (lines 141-155):
```python
degenerate = abs(p25 - median) < 0.5 and abs(p75 - median) < 0.5
if degenerate:
    print("\n⚠ UNRELIABLE: ...")
elif obj_px.mean() < 60:
    print("\n⚠ CAUTION: ...")
else:
    print("\n✓ Depth reading looks reliable ...")
```
Use this exact print-and-flag shape (not raised exceptions) for
`verify_urdf.py`'s TWIN-01..04 checks when run standalone/manually: load
via `yourdfpy`, walk parent→child chain, print `✓`/`⚠` lines per assertion
(gripper is child of wrist link, `wrist_roll` present, `gripper_left`/
`gripper_right` prismatic with correct axes, no absolute mesh paths). Keep
the actual pass/fail *test* logic in `scripts/test_verify_urdf.py` (pytest
asserts) — `verify_urdf.py` itself is the human-readable diagnostic CLI,
matching this repo's existing diagnostics-vs-tests split (`diagnostics/` has
no pytest files; LIBERO's `envs/` does).

**Output-path convention** (lines 160-164):
```python
out_dir = Path(__file__).resolve().parent / "outputs"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "..._last.png"
...
print(f"\nSaved disparity visualization: {out_path}")
```
If `verify_urdf.py` writes any rendered/dumped output (e.g. a printed tree
dump saved to a file), mirror `Saved → {path}` phrasing exactly (arrow
convention per CLAUDE.md).

---

### `scripts/test_verify_urdf.py` (test, request-response)

**Analog:** `LIBERO/libero/libero/envs/test_camera_config.py` (lines 1-60 read this session, full file structure confirmed via docstring)

**Docstring classification + import-path-anchoring pattern** (lines 1-38):
```python
"""Real (no-mock) local-sim integration proof for camera config threading.

Classified the same way as datasets/test_hdf5_writer.py: integration, real
local sim, small N. ...
"""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")

# repo root = four levels up from this file's dir
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
```
Copy directly for `scripts/test_verify_urdf.py`: since `scripts/` has no
`conftest.py` (RESEARCH.md Wave 0 Gaps explicitly says mirror this inline
`sys.path` pattern rather than adding one), compute repo root via
`os.path.dirname(__file__)` walk-up and insert into `sys.path` before
importing `yourdfpy` / referencing `So-101/So-101.urdf`'s absolute path.
Open with a classification docstring line ("integration, real no-mock URDF
load, small N") matching this repo's established self-documenting test
style.

**Test naming convention** (module docstring lines 8-14): one test function
per requirement, descriptive `test_<behavior>` names —
`test_rgb_cameras_non_degenerate_spatial`, `test_depth_cameras_non_degenerate_spatial`.
Apply the same for TWIN-01..05: `test_gripper_is_child_of_wrist`,
`test_wrist_roll_range`, `test_gripper_joint_axes`,
`test_no_absolute_mesh_paths`, `test_joint_limits_match_calibration` (exact
names already specified in RESEARCH.md's Phase Requirements → Test Map
table — treat that table as locked, not just illustrative).

---

### `LIBERO/libero/libero/assets/grippers/soarm_gripper.xml` (D-04 clamp visual fix)

**Analog:** itself — `main_frame_visual`'s own asset-note comment (lines 38-46, read this session)

```xml
<!-- Fixed Main frame (roboninecom RB9.01.062.010) that bolts to the wrist and
     houses the gear/racks holding the two clamps. VISUAL-ONLY (the physics is
     the box jaws + slide joints below); it was previously omitted, which left the
     clamps rendering with no connecting structure ("floating claws"). The CAD part
     is in its own corner-origin frame; a cyclic-axis rotation (quat 0.5 0.5 0.5 0.5)
     maps mesh X->local Y (jaw-travel/opening), mesh Y->local Z, mesh Z->local X
     (mount-face normal / approach). Placement dialed in via offscreen render so the
     frame sits at the wrist mount and bridges to the jaws. -->
```
This comment documents the exact prior fix method for a sibling bug on the
same file — the executor should use the identical *process* for D-04's
clamp-mesh offset (`left_jaw_visual`/`right_jaw_visual` geoms, lines 87 and
95: `pos="0.15241 -0.02808 0.15206" quat="0 0.707107 0.707107 0"` etc.):
iterative offscreen-render → inspect → adjust `pos`/`quat` → repeat, since
(per the file's own top-of-file "Fidelity note", lines 29-33) the upstream
clamp STLs are baked in full-arm-assembly coordinates with no closed-form
mapping to the mount frame. After fixing, add a comment in the same style
documenting the empirically-found `pos`/`quat` and referencing D-04.

---

## Shared Patterns

### Fail-loud precondition checks on file-I/O scripts
**Source:** `diagnostics/measure_object_depth.py` lines 74-76; `coppelia/export_model_library.py` lines 36-38
**Apply to:** `scripts/mjcf_to_urdf.py`, `scripts/verify_urdf.py`
```python
if not CALIB_PATH.exists():
    raise SystemExit(f"No calibration found at {CALIB_PATH}. Run ...")
```

### `Saved → {path}` output logging
**Source:** CLAUDE.md "Logging" convention; concretely demonstrated in `explorations/*` scripts (`print(f"Saved → {out_path}")`)
**Apply to:** `scripts/mjcf_to_urdf.py` (after writing `So-101/So-101.urdf`), `scripts/verify_urdf.py` (if it writes any diagnostic dump)

### `ROOT = Path(__file__).resolve().parent[.parent]` path anchoring
**Source:** `coppelia/export_model_library.py` line 16 (`ROOT = Path(__file__).parent.resolve()`); `explorations/*.py`'s established `ROOT = Path(__file__).resolve().parent.parent` convention (per CLAUDE.md Module-Level Path Setup / Architectural Constraints)
**Apply to:** `scripts/mjcf_to_urdf.py`, `scripts/verify_urdf.py` — resolve `robot.xml`, `soarm_gripper.xml`, `So-101/So-101.urdf` paths relative to repo root, not cwd

### Inline `sys.path` repo-root anchoring for pytest files with no conftest
**Source:** `LIBERO/libero/libero/envs/test_camera_config.py` lines 28-38
**Apply to:** `scripts/test_verify_urdf.py` (per RESEARCH.md Wave 0 Gaps, explicitly instructed to mirror this rather than add a new `conftest.py`)

### Numbered/rationale header comments for MJCF/URDF structural edits
**Source:** `LIBERO/libero/libero/assets/robots/soarm101/robot.xml` lines 2-19 (bulleted rationale list); `soarm_gripper.xml` lines 2-33 (prose rationale + Fidelity note)
**Apply to:** Any edits to `robot.xml`/`soarm_gripper.xml` joint limits (TWIN-05/TWIN-06) or the clamp visual fix (D-04) — document *why* each changed value was changed (e.g. "re-derived from soarm_follower_02.json calibration, see Phase 10 D-06"), matching this repo's existing self-documenting-asset convention. This is a strict project norm here, not optional style.

## No Analog Found

None — all 6 files/edits have a strong in-repo precedent (either an existing analogous script, or the file's own prior self-documented fix pattern).

## Metadata

**Analog search scope:** `coppelia/`, `diagnostics/`, `explorations/`, `LIBERO/libero/libero/envs/`, `LIBERO/libero/libero/assets/robots/soarm101/`, `LIBERO/libero/libero/assets/grippers/`
**Files scanned:** 10 (read in full or targeted range: `measure_object_depth.py`, `test_camera_config.py` (partial), `robot.xml` (partial), `soarm_gripper.xml`, `So-101/So-101.urdf` (metadata only), `coppelia/soarm_parallel_gripper.urdf`, `coppelia/export_model_library.py`)
**Pattern extraction date:** 2026-09-18
