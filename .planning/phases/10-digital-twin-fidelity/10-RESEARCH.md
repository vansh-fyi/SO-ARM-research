
> **2026-09-20 established-model update:** [10-ESTABLISHED-MODEL.md](10-ESTABLISHED-MODEL.md)
> is authoritative for the accepted Coppelia-aligned assembly, black housing,
> yellow jaws, 0.036 m per-jaw cap, corrected starting pose and LIBERO wiring.
> Earlier settings and verification results below are historical and do not
> establish policy success for the accepted model.

# Phase 10: Digital-Twin Fidelity - Research

**Researched:** 2026-09-18
**Domain:** Robot kinematic description generation (MJCF → URDF), servo calibration math, MuJoCo asset/mesh correctness
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** The MuJoCo side (`robot.xml` + `soarm_gripper.xml`) already has correct kinematic joints — `wrist_roll` and `gripper_left`/`gripper_right` prismatic joints already exist with plausible limits, built independently of the URDF (not derived from it). This was NOT known before this discussion and changes the phase's shape: it is not a from-scratch kinematic rebuild on both files, it's "generate a correct URDF from an already-correct MuJoCo model," plus a separate visual-mesh fix.
- **D-02:** Generate the URDF from the MuJoCo model (`robot.xml` + `soarm_gripper.xml`), rather than patching the broken CoppeliaSim-exported `So-101/So-101.urdf` in place. This guarantees the URDF and MuJoCo XML agree by construction instead of risking two independently-maintained kinematic definitions drifting apart. Visual meshes are still sourced from the CoppeliaSim export (`So-101/*.dae`, `coppelia/meshes/*.stl`).
- **D-03:** Do not lock the conversion tool/method now. The phase-researcher should survey available options (MuJoCo has no native URDF exporter; third-party mjcf-to-urdf converters exist) during Phase 10 research, and the planner should pick based on fidelity/maintainability vs. a small hand-written converter script.
- **D-04:** A known, distinct bug exists in `soarm_gripper.xml`'s clamp **visual** mesh placement (NOT the physics — physics uses separate box-jaw geoms and works correctly for grasping/motion): the clamp STL's gear-tooth detail visibly pokes outside the `main_frame_visual` housing during rendering in Python, instead of sitting flush in an L-shape against the frame. This is a mesh recentering/offset bug in the clamp geom placement (see the asset-note comment in `soarm_gripper.xml` around the `soarm_main_frame` mesh). **Fix this inside Phase 10** — it's directly part of gripper fidelity (TWIN-03/TWIN-07) and touches the same file being worked on.
- **D-05:** No calibration JSON lives in this repo — LeRobot writes/reads it from its local cache keyed to a device name (e.g. `soarm_follower_02`, referenced in `control/COMMANDS.md`). The source of truth for TWIN-05 is the **current follower arm's live LeRobot calibration file** — whatever `control/` presently uses to drive the real robot. Locate it (LeRobot's calibration cache dir) before deriving limits.
- **D-06:** Re-derive joint limits from that live calibration file (tick → radian conversion) and cross-check the result against `robot.xml`'s existing 4 arm-joint limits (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`) — these look like they were already derived from calibration in Phase 2 but have never been independently re-verified. If they match, `robot.xml` is confirmed correct; if not, correct it. `wrist_roll` and gripper limits are derived fresh from the same calibration file either way (TWIN-02/TWIN-05).
- **D-07:** Phase 10 is sim-side-only, but TWIN-07 ("driving the simulated gripper with a given joint command opens/closes it in the same direction as the real gripper under the identical command") cannot be verified from sim state alone. Verification method: **the user performs a manual side-by-side check** — send a known gripper command to the real robot via `control/` (e.g. `jog_gripper_raw.py`), drive the sim with the same command, visually confirm both open/close the same way. This requires the user's involvement at verification time during Phase 10 execution — flag this to the planner/executor as a human-in-the-loop checkpoint, not something the executor can auto-verify.
- **D-08:** Compare using **LeRobot's native gripper action value** (the actual command representation `control/` sends to the real servo), translated into the sim's `-1`/`+1` convention (`SoarmGripper.format_action`: `-1`=open, `+1`=closed) for the comparison — not the reverse. Keeps the real-hardware side of the comparison unambiguous.
- **D-09:** Nothing in the repo currently loads the URDF programmatically — the MuJoCo sim uses `robot.xml` directly, and no ROS/IK/visualization tooling in this repo consumes `So-101.urdf`. Treat the fixed URDF as a **documentation/reference artifact**: it must load cleanly (via a standard URDF parser) and be kinematically correct per TWIN-01..05, but does not need integration testing against any consuming system beyond that — no ROS stack or IK solver is currently expected to load it in this milestone.

### Claude's Discretion

- Exact conversion tool/script for MuJoCo→URDF generation (research + planning decide, per D-03).
- How to represent the visual-only clamp mesh fix in the generated URDF (the URDF's gripper geometry should reflect the corrected mesh placement once fixed in `soarm_gripper.xml`).

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope. (The clamp mesh bug, while newly discovered during this discussion, was folded into Phase 10 scope per D-04 rather than deferred, since it's directly part of the gripper fidelity requirements TWIN-03/TWIN-07 already cover.)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|---------------------|
| TWIN-01 | The URDF's gripper is attached at the end of the kinematic chain (child of the wrist link), not floating off `robot_base` | Hand-rolled converter walks the MJCF tree where the gripper attaches at `right_hand` (arm) / `right_gripper` (gripper root) by construction — see Architecture Patterns / System Architecture Diagram. Verified with `yourdfpy` per D-09's "standard URDF parser" requirement |
| TWIN-02 | The URDF includes a `wrist_roll` joint with range matching the real robot's calibrated servo limits | Located exactly why `wrist_roll` can't use the calibration-ticks formula (full-turn placeholder) — see Common Pitfalls Pitfall 1; recommend keeping `robot.xml`'s existing mechanical-limit value |
| TWIN-03 | The URDF includes `gripper_left`/`gripper_right` prismatic joints whose open/close direction matches the real gripper (fixes the mirrored-gears bug) | Confirmed `soarm_gripper.xml`'s existing prismatic joints/axes are already correct (D-01); confirmed the one candidate conversion library doesn't support prismatic joints at all, driving the hand-rolled-converter recommendation |
| TWIN-04 | All mesh file references in the URDF use in-repo relative paths, not absolute `~/Downloads/` paths | Confirmed exact count (30 absolute-path occurrences) and confirmed the actual mesh files already exist in-repo (`So-101/*.dae`) under matching filenames — a straight relative-path rewrite, not a re-export |
| TWIN-05 | URDF joint limits for all 6 joints match the real servo calibration ranges (converted from LeRobot calibration ticks to radians) | Located the live calibration file, read LeRobot's exact tick→radian formula from installed source, computed all 6 derived values and cross-checked against `robot.xml`'s current values (3 of 4 arm joints mismatch) — see Pattern 2 and Pitfall 3 |
| TWIN-06 | The MuJoCo XML is brought into agreement with the corrected URDF's kinematic structure | Identified the existing `test_camera_config.py::test_rgb_cameras_non_degenerate_spatial`-style `env.reset()` call as the concrete existing verification path (Validation Architecture section) |
| TWIN-07 | Driving the simulated gripper with a given joint command opens/closes it in the same direction as the real gripper under the same command | Documented as a human-in-the-loop checkpoint per D-07; clarified the `SoarmGripper.format_action` sign-convention subtlety that risks tripping up the comparison (Pitfall 2) |
</phase_requirements>

## Summary

This phase has one dominant fact that reshapes its scope: the MuJoCo side (`robot.xml` +
`soarm_gripper.xml`) is already kinematically correct and does not need to be rebuilt. What's
actually broken is `So-101/So-101.urdf` — a CoppeliaSim export with the gripper as a disconnected
root-level object, 30 absolute `file:///Users/hp/Downloads/...` mesh paths, and no `wrist_roll` or
gripper joints at all (verified by direct read: `wrist_link_respondable` is the terminal link). Per
D-02, the fix is to **generate** a new URDF from the two already-correct MJCF files rather than
patch the broken export in place.

The single hardest technical question — MJCF→URDF conversion tooling — has a clear answer after
investigation: **write a small hand-rolled Python converter**, not a library. The one actively
maintained MJCF→URDF PyPI tool (`mjcf-urdf-simple-converter`) explicitly does not support prismatic
joints, and this robot's gripper is exactly two prismatic joints (`gripper_left`/`gripper_right`).
Given the kinematic tree is 7 joints total across 2 files, both already fully documented with
explicit `pos`/`quat`/`range` per joint, a ~150-line script that walks the MJCF body tree directly
(via `xml.etree.ElementTree`, no MuJoCo runtime needed) is more reliable than adapting a
general-purpose converter around its gaps. Use `yourdfpy` only for the read-side verification step
(TWIN-01's "load the URDF and walk the parent-child chain") — that's exactly the kind of load-only
consumption D-09 describes, and it has no ROS dependency.

The second hard question — where LeRobot's calibration lives and what its tick→radian formula is —
is now fully resolved from the actually-installed package source and the actual on-disk calibration
file for `soarm_follower_02` (this repo's live follower). The file exists at
`~/.cache/huggingface/lerobot/calibration/robots/so_follower/soarm_follower_02.json` and the formula
is `degrees = (raw_tick - (range_min+range_max)/2) * 360 / 4095`. Applying it to the live file
produces joint limits that **diverge meaningfully** from `robot.xml`'s current values for
`shoulder_pan`, `shoulder_lift`, and `wrist_flex` — because those current values were copied
verbatim from TheRobotStudio's generic upstream vendor XML (`so101_new_calib.source.xml`, see
`VENDOR.txt`), not derived from this arm's actual servo calibration. `wrist_roll` cannot be handled
the same way at all: LeRobot's own calibration routine treats it as a full continuous-rotation joint
and never records a real range for it (hardcoded `0`–`4095` = a full 360° placeholder), so its real
operating limit must come from the mechanical/cable-wrap constraint that's already encoded in
`robot.xml` (`-2.744`..`2.841` rad), not from the calibration file.

**Primary recommendation:** Write a small hand-rolled MJCF→URDF Python script (ElementTree-based,
no new heavy dependency) driven directly off `robot.xml` + `soarm_gripper.xml`'s documented body/
joint tree; derive `shoulder_pan`/`shoulder_lift`/`elbow_flex`/`wrist_flex`/`gripper` limits from the
live `soarm_follower_02.json` calibration file using LeRobot's own DEGREES-mode formula, keep
`wrist_roll`'s existing mechanical-limit-derived range from `robot.xml` unchanged (flag, don't
silently recompute it from calibration), and verify the generated URDF with `yourdfpy` (installed
fresh into the `libero` conda env — the only env with `mujoco`/`robosuite` already present).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Kinematic chain definition (ground truth) | Simulation asset (MJCF: `robot.xml` + `soarm_gripper.xml`) | — | Already correct per D-01; MuJoCo consumes this directly, nothing else in-repo generates it |
| URDF generation | Offline tooling (one-off/repeatable Python script) | — | URDF is a derived, documentation/reference artifact (D-09) — no runtime consumer in this repo |
| Joint limit source of truth | Real-hardware calibration (`~/.cache/huggingface/lerobot/calibration/...`) | Vendor upstream MJCF (`so101_new_calib.source.xml`) as fallback only for `wrist_roll` | Live calibration reflects the actual attached servos/arm; vendor XML is a generic reference that Phase 2 copied wholesale, never re-verified (D-06) |
| Visual mesh correctness (clamp placement) | Simulation asset (MJCF geom `pos`/`quat` offsets) | — | Physics uses separate box-jaw geoms; only the rendered visual mesh is wrong (D-04) — a pure offset-tuning problem, not a kinematic one |
| URDF↔MJCF cross-verification tooling | Offline tooling (new script, doesn't exist yet) | — | Explicitly called out in CONTEXT.md as new tooling this phase must write |
| Real/sim gripper-direction comparison (TWIN-07) | Human-in-the-loop (manual, `control/` scripts) | Sim (`SoarmGripper.format_action`) | D-07 declares this un-automatable from sim state alone — needs the physical robot |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|---------------|
| `yourdfpy` | 0.0.60 (latest on PyPI, verified via `pip index versions`) [VERIFIED: pypi registry] | Load the generated URDF and walk/print its parent-child joint chain (TWIN-01 verification) | Pure Python, no ROS runtime dependency (matches D-09's "no ROS installed" constraint), actively maintained (github.com/clemense/yourdfpy), used widely in robotics/manipulation research code |
| `xml.etree.ElementTree` (stdlib) | — | Parse `robot.xml`/`soarm_gripper.xml`, write generated URDF XML | No dependency needed — both source files are already clean, hand-authored MJCF with a fully documented, small body tree; stdlib XML is sufficient and avoids adding any conversion-library dependency risk |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `numpy` | already installed (`libero` conda env) | Quaternion math if the converter needs to compose parent-child transforms manually | Only if hand-computing any transform composition beyond what's already explicit per-body in the MJCF (most joints/bodies already have explicit relative `pos`/`quat`, so this may not be needed at all) |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled ElementTree converter | `mjcf-urdf-simple-converter` (PyPI) | **Rejected** — confirmed via GitHub README (WebFetch) it does not support prismatic joints, only revolute/hinge; this robot's gripper is exactly 2 prismatic joints, so it would silently drop or mishandle TWIN-03's core requirement |
| Hand-rolled ElementTree converter | `mjcf2urdf` (PyPI, pybullet-based) | Not investigated in depth — pybullet-dependent (heavier install), and no evidence it handles prismatic joints either; same class of "hinge-only" tooling gap as above |
| `yourdfpy` for verification | `urchin` (PyPI, urdfpy fork) | Also viable — lighter dependency footprint (no `trimesh[easy]` pull), maintained fork of the abandoned `urdfpy`; use if `yourdfpy`'s `trimesh` dependency proves too heavy for the target env |
| `yourdfpy` for verification | `urdf_parser_py` (official ROS package, pip-installable standalone) | Works without a ROS install (pure parsing library), but the package name/origin ("ros/urdf_parser_py") risks confusion with D-09's explicit "no ROS" framing even though it has no ROS runtime dependency — prefer `yourdfpy` to keep the distinction unambiguous |

**Installation:**
```bash
# In the `libero` conda env (has mujoco/robosuite already, needed to run TWIN-06's env.reset() check)
conda activate libero
pip install yourdfpy
```

**Version verification:** `pip index versions yourdfpy` was run against the live PyPI registry during
this research session — confirmed `0.0.60` is current (published 2026-01-23 per registry metadata).
No package in the `libero` or `control/.venv` environments currently provides URDF parsing —
confirmed by `importlib.util.find_spec` checks against `yourdfpy`, `urdfpy`, `urchin`,
`urdf_parser_py` in both environments (all `False`).

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| `yourdfpy` | PyPI | published 2026-01-23 (latest release; project itself is older — first release 0.0.0 predates this) | unknown (registry download-stats lookup unavailable in this tool run) | github.com/clemense/yourdfpy | SUS (reason: `unknown-downloads` only — repo/registry existence both confirmed) | Approved, flagged for `checkpoint:human-verify` per protocol |
| `urchin` | PyPI | published 2025-10-21 | unknown | github.com/fishbotics/urchin | SUS (reason: `unknown-downloads` only) | Fallback candidate only — same flag if used |
| `urdf-parser-py` | PyPI | published 2021-11-11 | unknown | github.com/ros/urdf_parser_py | SUS (reason: `unknown-downloads` only) | Not recommended (see Alternatives Considered) — same flag if used |

**Packages removed due to [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** `yourdfpy`, `urchin`, `urdf-parser-py` — all three verdicts
are driven entirely by the legitimacy-check tool being unable to fetch PyPI download-count stats in
this environment (`unknown-downloads`), not by any red flag in the package itself. All three resolve
to real, well-known GitHub organizations/maintainers in the robotics tooling space (a solo maintainer
known for robotics research tooling, a named robotics-research GitHub org, and the official ROS
project respectively). The planner must still add a `checkpoint:human-verify` task before the `pip
install yourdfpy` step per protocol, since the automated verdict is SUS regardless of this manual
context.

*Note: `yourdfpy`'s package name was discovered via training knowledge and confirmed to exist via
`pip index versions` (an authoritative registry command) during this session — per the package
provenance rule this still counts as `[ASSUMED]` origin until a human confirms it against
`yourdfpy`'s own PyPI project page / GitHub README, which the `checkpoint:human-verify` step
satisfies.*

## Architecture Patterns

### System Architecture Diagram

```text
   robot.xml (MJCF, arm)          soarm_gripper.xml (MJCF, gripper)
   5 hinge joints, documented      2 slide joints, documented
   pos/quat per body               pos/quat per body
        |                                    |
        v                                    v
   +---------------------------------------------------------+
   |     NEW: mjcf_to_urdf.py (hand-rolled, ElementTree)      |
   |  1. parse both files' <worldbody> trees                 |
   |  2. walk body->joint->body chain (arm ends at            |
   |     right_hand; gripper attaches there per D-01's        |
   |     documented boundary split)                           |
   |  3. emit URDF <link>/<joint> elements, converting        |
   |     MJCF hinge->URDF revolute, slide->prismatic           |
   |  4. rewrite mesh refs: coppelia .stl/.dae relative        |
   |     paths (visual meshes trusted, only kinematics come    |
   |     from MJCF)                                            |
   |  5. substitute derived joint limits (see below)           |
   +---------------------------------------------------------+
        |
        v
   So-101/So-101.urdf (generated, replaces broken export)
        |
        v
   +---------------------------------------------------------+
   |   NEW: verify_urdf.py (yourdfpy-based)                  |
   |   - load URDF, walk parent->child chain, print it        |
   |   - assert: gripper is child of wrist link (TWIN-01)     |
   |   - assert: wrist_roll + gripper_left/right present       |
   |     with correct axes (TWIN-02/03)                        |
   |   - assert: no absolute paths in any mesh filename         |
   |     (TWIN-04)                                             |
   +---------------------------------------------------------+

   Joint-limit derivation (separate, feeds step 5 above):
   ~/.cache/huggingface/lerobot/calibration/robots/so_follower/
   soarm_follower_02.json  (live calibration, LOCATED this session)
        |  degrees = (raw_tick - mid) * 360 / 4095
        v
   shoulder_pan/shoulder_lift/elbow_flex/wrist_flex/gripper limits (radians)
   -- cross-checked against robot.xml's current (vendor-XML-derived) values --
   wrist_roll: NOT derivable this way (calibration hardcodes a full-turn
   placeholder) -- keep robot.xml's existing mechanical-limit value

   TWIN-06 (MJCF agreement) + TWIN-07 (direction, human-in-the-loop):
   robot.xml / soarm_gripper.xml edited in place (if TWIN-05 finds real
   mismatches) -> existing test:
   pytest LIBERO/libero/libero/envs/test_camera_config.py -x
   (constructs a real Soarm101 OffScreenRenderEnv and calls env.reset())
```

### Recommended Project Structure

No new permanent package — this is scripting work. Suggested location for the new one-off/
repeatable scripts (mirrors the existing `coppelia/` and `diagnostics/` precedent of small top-level
utility directories, not a new library layer):

```
scripts/                      # NEW (or reuse an existing top-level scripts-style dir if one exists)
├── mjcf_to_urdf.py            # generates So-101/So-101.urdf from the two MJCF files
└── verify_urdf.py             # yourdfpy-based load + parent-child chain print/assert
```

### Pattern 1: MJCF body/joint tree walk (hand-rolled)

**What:** Recursively walk `<worldbody><body>...<joint>...<body>...` nesting in the MJCF XML tree,
emitting one URDF `<link>` per MJCF `<body>` and one URDF `<joint>` per MJCF `<joint>`, translating
MJCF's relative parent-frame `pos`/`quat` on each `<body>` directly into the URDF `<joint><origin
xyz="..." rpy="..."/>` (URDF uses roll-pitch-euler, MJCF quat — a quat→rpy conversion is needed, not
just a pass-through).
**When to use:** This phase's converter script — the tree is fully known ahead of time (arm:
`base→shoulder→upper_arm→lower_arm→wrist→right_hand`; gripper: `right_gripper→gripper_left_jaw`/
`gripper_right_jaw` as two sibling children), so a generic recursive walker isn't even strictly
necessary — an explicit per-joint mapping table is simpler and easier to verify by hand than a truly
generic MJCF interpreter, given only 7 joints total.
**Example (quat→rpy, stdlib-only, no scipy needed for this simple case):**
```python
# Source: derived from MuJoCo's documented quat convention (w,x,y,z) and the
# standard Tait-Bryan intrinsic ZYX->rpy conversion used by URDF.
import math

def mjcf_quat_to_rpy(w, x, y, z):
    # MuJoCo quat order is (w, x, y, z); URDF wants roll,pitch,yaw (rad).
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw
```

### Pattern 2: LeRobot calibration tick→radian conversion (must match LeRobot exactly)

**What:** Reproduce LeRobot's own `MotorNormMode.DEGREES` formula so the derived URDF limits are
consistent with what `control/`'s live robot object would compute, not an independently-invented
formula that happens to look similar.
**When to use:** Deriving TWIN-05's joint limits for `shoulder_pan`, `shoulder_lift`, `elbow_flex`,
`wrist_flex` (and, separately, `gripper`'s calibration numbers for the D-07/D-08 direction check —
see Common Pitfalls for why gripper ticks do NOT convert into a prismatic meter range the same way).
**Example:**
```python
# Source: control/.venv/lib/python3.12/site-packages/lerobot/motors/motors_bus.py
# lines 874-877 (_normalize, DEGREES branch) -- read directly from the
# installed package this session. model_resolution_table["sts3215"] = 4096
# (control/.venv/.../lerobot/motors/feetech/tables.py line 190).
import math

def calibration_ticks_to_radians(range_min: int, range_max: int, max_res: int = 4095):
    """Mirrors LeRobot's MotorsBus._normalize() DEGREES branch exactly.
    Returns (neg_limit_rad, pos_limit_rad) -- symmetric around the calibrated
    midpoint, which is also MuJoCo's qpos=0 home pose per robot.xml's header
    comment ("no keyframe: qpos=0 IS the documented new-calib home pose")."""
    half_ticks = (range_max - range_min) / 2
    half_deg = half_ticks * 360 / max_res
    half_rad = math.radians(half_deg)
    return -half_rad, half_rad
```

**Live calibration values found this session** (`~/.cache/huggingface/lerobot/calibration/robots/
so_follower/soarm_follower_02.json` — the follower ID `control/COMMANDS.md` documents as the current
hardware):

| Joint | `range_min` | `range_max` | Derived limit (rad) | `robot.xml` current (rad) | Match? |
|-------|-----------:|-----------:|---------------------:|---------------------------:|--------|
| `shoulder_pan` | 1269 | 2869 | ±1.2275 | ±1.91986 | **No** — vendor XML value is ~40° wider per side |
| `shoulder_lift` | 904 | 3349 | ±1.8757 | ±1.74533 | **No** — derived is ~7.5° wider per side |
| `elbow_flex` | 832 | 3044 | ±1.6970 | ±1.69 | **Close** — within ~0.4° |
| `wrist_flex` | 899 | 3220 | ±1.7806 | ±1.65806 | **No** — derived is ~7° wider per side |
| `wrist_roll` | 0 | 4095 | ±3.1416 (meaningless — see Pitfall below) | -2.7438 to 2.8412 (asymmetric) | N/A — do not replace with the calibration-derived value |
| `gripper` (real servo, `RANGE_0_100` mode, not directly a URDF joint) | 48 | 3637 | n/a (percent mode, not degrees) | n/a — sim gripper limits (0/0.042 m) come from the 84mm hardware spec, not this | See Pitfall below |

`robot.xml`'s current 4 arm-joint limits (`shoulder_pan`, `shoulder_lift`, `elbow_flex`,
`wrist_flex`) were traced to `LIBERO/libero/libero/assets/robots/soarm101/
so101_new_calib.source.xml` (the pristine upstream TheRobotStudio vendor MJCF, see `VENDOR.txt` —
pinned commit `fda892c`), copied into `robot.xml` verbatim during the Phase 2 adaptation, **not**
independently derived from this repo's own `soarm_follower_02` calibration. This is exactly the "not
independently re-verified" situation CONTEXT.md's D-06 flags — the cross-check surfaces real,
non-trivial mismatches on 3 of 4 arm joints.

### Anti-Patterns to Avoid

- **Re-deriving `wrist_roll`'s limit from the calibration file:** LeRobot's own `calibrate()` method
  (`lerobot/robots/so_follower/so_follower.py`) hardcodes `range_mins["wrist_roll"] = 0` and
  `range_maxes["wrist_roll"] = 4095` — it is explicitly excluded from `record_ranges_of_motion()`
  because the real servo spins continuously past a mechanical stop check. Feeding this placeholder
  through the DEGREES formula produces exactly ±180° (a full circle), which is not a real joint
  limit — it would silently widen `wrist_roll`'s range to a full unrestricted turn, which is wrong
  (the real arm has a cable-wrap-limited stop, already captured in `robot.xml`'s asymmetric
  `-2.7438..2.8412` rad range).
- **Converting the `gripper` calibration entry's ticks straight into a prismatic meter range:** the
  calibration file's `gripper` entry describes the *real physical servo's rotation*, normalized by
  LeRobot into a `RANGE_0_100` percent-open representation for whatever gripper is currently
  attached — it is not a linear-distance measurement and has no defined mapping to the roboninecom
  84mm parallel-jaw's `0`–`0.042` m prismatic stroke (a completely different aftermarket mechanism
  installed in Phase 4). The gripper prismatic joint limits in `soarm_gripper.xml` (`0 0.042`) are
  already correct, sourced from the 84mm hardware spec — leave them as-is. The calibration file's
  `range_min`/`range_max`/`drive_mode` for `gripper` are still useful for D-07/D-08's direction check
  (which raw-tick extreme is physically open vs. closed), just not for a URDF *limit*.
- **Using a general-purpose MJCF→URDF converter without checking prismatic joint support first:**
  confirmed this session (WebFetch of the `mjcf-urdf-simple-converter` README) that the one
  actively-referenced PyPI tool in this space silently doesn't support prismatic joints at all.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| URDF *parsing*/loading for verification | A custom URDF XML reader | `yourdfpy` | TWIN-01's verification explicitly wants "a standard URDF parser" per D-09 — hand-rolling this defeats the point of an independent check |
| Quaternion→RPY math beyond the simple case above | A general quaternion/rotation library dependency (e.g. `scipy.spatial.transform`) | The small closed-form conversion in Pattern 1, OR `numpy`+manual if edge cases (gimbal lock near ±90° pitch) matter | This robot's joints are all single-axis Z hinges/Y or -Y slides with modest, well-documented relative rotations between links — not a general robotics IK/rotation problem needing a heavy library |

**Key insight:** Everything else in this phase is small, fully-observed, and already-documented
(the MJCF files' own header comments spell out every design decision). The temptation is to reach
for a general MJCF/URDF conversion framework; resist it — the real risk in that direction is
silently losing the prismatic gripper joints, exactly as demonstrated by the one real candidate
tool found.

## Common Pitfalls

### Pitfall 1: Treating `wrist_roll`'s calibration range as real

**What goes wrong:** Naively applying the tick→radian formula to `wrist_roll`'s calibration entry
produces ±180° (a full turn), which would replace `robot.xml`'s real, narrower, asymmetric
mechanical limit with an incorrect wide-open one.
**Why it happens:** LeRobot's calibration routine special-cases `wrist_roll` as a "full turn motor"
and never actually measures its real range — the `0`/`4095` values in the calibration JSON are a
placeholder, not a measurement.
**How to avoid:** For `wrist_roll` specifically, keep `robot.xml`'s existing value (`-2.7438` to
`2.8412` rad) as TWIN-02's source of truth, and only investigate further (e.g. real cable-wrap
measurement) if there's specific reason to distrust it — CONTEXT.md's D-06 already scoped
`wrist_roll` limits as "derived fresh... either way," but "fresh" here means confirming the existing
mechanical-stop-derived value is still correct, not blindly re-deriving via the ticks formula that
doesn't apply to this joint.
**Warning signs:** A derived `wrist_roll` range that comes out suspiciously close to exactly ±π rad
(±180°) is the tell — that's the full-turn placeholder leaking through, not a real limit.

### Pitfall 2: Gripper action-convention confusion (relevant to D-07/D-08's TWIN-07 check)

**What goes wrong:** `SoarmGripper.format_action`'s docstring states the external convention is
"-1 => open, +1 => closed" for the `action` argument — but the same docstring's internal explanation
describes `current_action=-1 => ctrl 0 (closed)`, i.e. the *internal* `current_action` state variable
has the **opposite** sign convention from the external `action` argument. Reading only the internal
explanation (or only skimming the class-level docstring) risks flipping the sign when wiring up
D-08's real/sim comparison.
**Why it happens:** `format_action` integrates the external action into an internal `current_action`
state via `current_action += [-1,-1] * speed * sign(action)` — a `+1` (CLOSE) external command
pushes `current_action` toward `-1`. The two variables (`action` vs. `current_action`) are inversely
related by design, but both get called "the action" in prose.
**How to avoid:** For D-08's comparison, use the *external* `action` argument's convention only
(`-1`=open, `+1`=closed, exactly as CONTEXT.md's D-08 states) and never read `current_action`'s sign
directly as if it meant the same thing.
**Warning signs:** If a sim/real side-by-side check shows the gripper opening when a "close" command
is sent, check whether `current_action` was read instead of the external `action` value before
assuming the axis/mesh direction itself is wrong.

### Pitfall 3: `robot.xml`'s existing joint limits are not "already verified real calibration"

**What goes wrong:** Assuming `robot.xml`'s 4 documented arm-joint limits (mentioned in the phase's
own framing as "look like they were already derived from calibration in Phase 2") are safe to leave
untouched.
**Why it happens:** They *were* copied from a calibration-flavored source — `so101_new_calib.
source.xml` — but that's TheRobotStudio's generic **upstream vendor** calibration reference (fixed
across all SO-101 units), not this specific arm's (`soarm_follower_02`) live, individually-run
calibration. The naming similarity ("new_calib") is misleading.
**How to avoid:** Always diff against the live `soarm_follower_02.json` file (or whichever
calibration file is current per `control/COMMANDS.md` at execution time — check the device ID/port
table hasn't changed), not the vendor XML.
**Warning signs:** This research already found 3 of 4 arm joints mismatch meaningfully (7-40° per
side) — expect the planner/executor to need to actually change `robot.xml`'s limits, not just
confirm them.

## Code Examples

### Existing gripper action contract (read directly, do not re-derive)

```python
# Source: LIBERO/libero/libero/envs/grippers/soarm_gripper.py (read this session)
def format_action(self, action):
    """-1 => open, +1 => closed (external convention, per class docstring
    and CONTEXT.md D-08). Internal current_action has the OPPOSITE sign
    relationship to ctrl -- see Pitfall 2 above."""
    assert len(action) == self.dof
    self.current_action = np.clip(
        self.current_action + np.array([-1.0, -1.0]) * self.speed * np.sign(action),
        -1.0, 1.0,
    )
    return self.current_action
```

### Existing gripper clamp mesh placement (asset-note comments to build the D-04 fix from)

```xml
<!-- Source: LIBERO/libero/libero/assets/grippers/soarm_gripper.xml lines 38-46, read this session -->
<!-- Fixed Main frame (roboninecom RB9.01.062.010) ... The CAD part
     is in its own corner-origin frame; a cyclic-axis rotation (quat 0.5 0.5 0.5 0.5)
     maps mesh X->local Y (jaw-travel/opening), mesh Y->local Z, mesh Z->local X
     (mount-face normal / approach). Placement dialed in via offscreen render so the
     frame sits at the wrist mount and bridges to the jaws. -->
<mesh name="soarm_main_frame" file="soarm_parallel/main_frame_visual.stl" scale="0.001 0.001 0.001"/>
...
<geom type="mesh" mesh="soarm_main_frame" ... pos="-0.015 -0.064 -0.02975" quat="0.5 0.5 0.5 0.5" name="main_frame_visual"/>
...
<!-- LEFT jaw clamp visual -- the D-04 bug is in this geom's pos/quat -->
<geom type="mesh" mesh="soarm_clamp_left" ... pos="0.15241 -0.02808 0.15206" quat="0 0.707107 0.707107 0" name="left_jaw_visual"/>
<geom type="mesh" mesh="soarm_clamp_right" ... pos="0.15242 0.01270 0.15171" quat="0 0.707107 0.707107 0" name="right_jaw_visual"/>
```

The file's own comment for `soarm_main_frame` documents the exact method already used to fix a
near-identical problem: **"Placement dialed in via offscreen render"** — i.e. an iterative
render-inspect-adjust-`pos`/`quat` loop (not a closed-form calculation), because the STL's local
origin/axes don't align with the mount frame by construction (confirmed in the file's own
"Fidelity note" at the top: "the upstream clamp STLs are baked in full-arm-assembly coordinates").
The same iterative offscreen-render method is the recommended approach for the clamp geoms — there
is no closed-form shortcut available since the STL's local coordinate frame relative to the mount
geometry isn't independently documented anywhere in-repo; the main-frame fix was itself
empirically dialed in, not derived.

### Existing verification test to point TWIN-06 at (do not write a new one from scratch)

```python
# Source: LIBERO/libero/libero/envs/test_camera_config.py (read this session)
# Constructs a real Soarm101 OffScreenRenderEnv and calls env.reset() --
# exactly TWIN-06's stated success criterion. Run:
#   conda activate libero && pytest LIBERO/libero/libero/envs/test_camera_config.py -x
from libero.envs import OffScreenRenderEnv
env = OffScreenRenderEnv(
    bddl_file_name=BDDL_PATH,      # existing put_the_cream_cheese_in_the_bowl.bddl
    robots=["Soarm101"],
    camera_names=["agentview", "robot0_eye_in_hand"],
    has_renderer=False,
    has_offscreen_renderer=True,
)
obs = env.reset()   # this is the exact call TWIN-06 requires to "complete without errors"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|-------------------|---------------|--------|
| Patch `So-101/So-101.urdf` (CoppeliaSim export) in place | Generate a new URDF from the already-correct MJCF | This phase (D-02) | Guarantees URDF/MJCF agreement by construction instead of two independently-drifting kinematic definitions |
| Assume `robot.xml`'s 4 documented joint limits are calibration-verified | Cross-check against live `soarm_follower_02.json` | This phase (D-06) | 3 of 4 arm joints need correction; the previous values came from generic vendor data, not this specific arm |

**Deprecated/outdated:** None — this is a first-time fix, not a migration away from a previously
working approach.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|----------------|
| A1 | `yourdfpy` is the right verification-parser choice (vs. `urchin`/`urdf_parser_py`) | Standard Stack / Don't Hand-Roll | Low — all three are viable; wrong choice just means a `pip install` swap, no design rework. Package identity itself still needs the `checkpoint:human-verify` gate per the Package Legitimacy Audit |
| A2 | `mjcf-urdf-simple-converter` truly lacks prismatic joint support (based on one WebFetch of its README, not a code read) | Architecture Patterns / Alternatives Considered | Medium — if this is stale/wrong, a viable off-the-shelf converter exists after all and the hand-rolled script is extra unneeded work. Low cost to double check: `pip install mjcf-urdf-simple-converter` and grep its source for "prismatic"/"slide" before committing to hand-rolling, as a first Wave-0-style sanity step |
| A3 | `soarm_follower_02` (port `/dev/cu.usbmodem5B8E1139151`) is still the correct/current follower device at execution time | Code Examples / Pitfall 3 | Medium — `control/COMMANDS.md` notes macOS reassigns port suffixes on replug; the **device ID** (`soarm_follower_02`) is what keys the calibration filename and should be stable, but confirm against `control/COMMANDS.md`'s current table before trusting the specific calibration file used in this research |

**If this table is empty:** N/A — see entries above.

## Open Questions (RESOLVED)

1. **Does the generated URDF replace `So-101/So-101.urdf` in place, or land at a new path?**
   - What we know: D-02 says "generate the URDF from the MuJoCo model... rather than patching the
     broken CoppeliaSim-exported `So-101/So-101.urdf` in place" — this reads as "don't patch it,"
     which could mean either "overwrite it with a generated replacement" or "write a new file
     elsewhere and leave the broken one as a CoppeliaSim-workflow artifact."
   - What's unclear: Whether anything (e.g. `coppelia/export_model_library.py`'s CoppeliaSim
     re-import workflow) still expects a file at exactly `So-101/So-101.urdf`.
   - Recommendation: Planner should default to overwriting `So-101/So-101.urdf` in place (simplest,
     matches "the URDF" being referred to as one artifact throughout REQUIREMENTS.md/CONTEXT.md), but
     confirm `coppelia/export_model_library.py` doesn't read `So-101/So-101.urdf` back in anywhere
     (a repo grep this session found no such reference — it only reads `soarm_parallel_gripper.urdf`
     from `coppelia/`, a separate file, unaffected).
   - **RESOLVED:** Yes, Plan 10-03 overwrites `So-101/So-101.urdf` in place (no other script reads
     it back in).

2. **Is `soarm_follower_02`'s calibration file the one that should drive TWIN-05, or should
   calibration be re-run fresh during Phase 10 execution?**
   - What we know: The file exists, is current per `control/COMMANDS.md`'s device table, and is
     presumably what's actually loaded whenever `control/` scripts connect to the follower today.
   - What's unclear: Whether this specific calibration run is trusted as accurate (vs. e.g. the
     gripper recalibration history in `recalibrate_gripper.py`'s own docstring, which describes a
     prior bug in the *original* full-arm calibration's gripper wraparound handling — already fixed
     for `gripper`, but raises the general question of whether the 5 arm-joint calibration values in
     this same file have ever been similarly double-checked).
   - Recommendation: Use this file as the source of truth per D-05 (CONTEXT.md is explicit: "current
     follower arm's live LeRobot calibration file... whatever `control/` presently uses to drive the
     real robot" — this **is** that file), but the plan should not assume the numbers are beyond
     doubt; if the derived limits look physically implausible (e.g. wider than the servo's mechanical
     travel), that's worth a sanity flag rather than blind trust.
   - **RESOLVED:** Yes, Plans 10-01/10-04 use `soarm_follower_02`'s live calibration file as the
     source of truth per CONTEXT.md D-05.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|--------------|-----------|---------|----------|
| `libero` conda env (mujoco, robosuite) | TWIN-06 verification (`env.reset()` test) | Yes | mujoco 2.3.7, robosuite 1.4.1 | — |
| `yourdfpy` | TWIN-01 URDF-load verification | No (not yet installed anywhere) | latest on PyPI: 0.0.60 | `urchin` or `urdf_parser_py` (see Alternatives) |
| `control/.venv` (LeRobot) | Locating/reading the live calibration file, D-07 manual verification scripts | Yes | lerobot package installed, calibration file present on disk | — |
| Physical SO-101 follower hardware | D-07's human-in-the-loop gripper-direction check | Assumed present (not verified this session — no hardware probe performed) | — | None — TWIN-07 cannot be completed without it; flag as a checkpoint requiring the user |
| ROS / any URDF-consuming stack | None (explicitly out of scope per D-09) | N/A | — | — |

**Missing dependencies with no fallback:**
- Physical robot access for the D-07 manual side-by-side gripper check — this is a human-in-the-loop
  checkpoint by design (D-07), not an environment gap to engineer around.

**Missing dependencies with fallback:**
- `yourdfpy` — not installed yet, but installation is a single `pip install` with two viable
  alternatives if it proves problematic.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (no `pytest.ini`/`conftest.py` at repo or `LIBERO/` root — relies on pytest's "prepend" import-mode auto-discovery, per `test_camera_config.py`'s own module docstring) |
| Config file | none — see Wave 0 gaps below for the one new test this phase should add |
| Quick run command | `pytest LIBERO/libero/libero/envs/test_camera_config.py -x` (run with `conda activate libero`) |
| Full suite command | `pytest LIBERO/libero/libero/envs/ LIBERO/libero/libero/datasets/ -x` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|---------------------|--------------|
| TWIN-01 | Generated URDF loads via `yourdfpy`; gripper is a child of the wrist link, not floating off `robot_base` | unit (new) | `pytest scripts/test_verify_urdf.py::test_gripper_is_child_of_wrist -x` | ❌ Wave 0 |
| TWIN-02 | URDF's `wrist_roll` joint range matches `robot.xml`'s calibrated-servo-limit value | unit (new) | `pytest scripts/test_verify_urdf.py::test_wrist_roll_range -x` | ❌ Wave 0 |
| TWIN-03 | URDF's `gripper_left`/`gripper_right` are prismatic with axes matching `soarm_gripper.xml`'s open/close direction | unit (new) | `pytest scripts/test_verify_urdf.py::test_gripper_joint_axes -x` | ❌ Wave 0 |
| TWIN-04 | No `file://` or absolute-path mesh references anywhere in the generated URDF | unit (new) | `pytest scripts/test_verify_urdf.py::test_no_absolute_mesh_paths -x` | ❌ Wave 0 |
| TWIN-05 | Derived-from-calibration joint limits (5 of 6) match a documented, reproducible formula; `wrist_roll` explicitly excepted | unit (new) | `pytest scripts/test_verify_urdf.py::test_joint_limits_match_calibration -x` | ❌ Wave 0 |
| TWIN-06 | `robot.xml`/`soarm_gripper.xml` edits (if any, from TWIN-05's cross-check) don't break `env.reset()` | integration (existing) | `pytest LIBERO/libero/libero/envs/test_camera_config.py -x` | ✅ already exists |
| TWIN-07 | Sim gripper direction matches real gripper direction under the same command | manual-only (human-in-the-loop, D-07) | N/A — documented checkpoint, not automatable | N/A by design |

### Sampling Rate

- **Per task commit:** `pytest LIBERO/libero/libero/envs/test_camera_config.py -x` (fast, local, no
  GPU needed — already the established Phase 7 pattern)
- **Per wave merge:** `pytest LIBERO/libero/libero/envs/ LIBERO/libero/libero/datasets/ -x`
- **Phase gate:** Full suite green before `/gsd-verify-work`, plus the D-07 human checkpoint
  completed and its outcome documented.

### Wave 0 Gaps

- [ ] `scripts/test_verify_urdf.py` — new test file covering TWIN-01 through TWIN-05 assertions
      against the generated URDF (via `yourdfpy`)
- [ ] `yourdfpy` install into the `libero` conda env (`pip install yourdfpy`)
- [ ] No `conftest.py` exists for `scripts/` — if the new test file needs repo-root path setup,
      mirror `test_camera_config.py`'s existing inline `sys.path` pattern rather than adding a new
      `conftest.py` (keeps consistency with the established no-conftest convention)

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|----------------|---------|--------------------|
| V2 Authentication | No | No auth surface in this phase — local file generation + offline sim verification |
| V3 Session Management | No | N/A |
| V4 Access Control | No | N/A |
| V5 Input Validation | Marginally | The MJCF→URDF converter should fail loudly (not silently emit a malformed URDF) if a body/joint it expects is missing — treat the two source MJCF files as a fixed, trusted, small input, not arbitrary untrusted input |
| V6 Cryptography | No | N/A |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|-----------------------|
| Physical hardware damage from an incorrect gripper-direction command during D-07's manual check | (physical safety, not a STRIDE category) | `control/`'s existing small-step, confirm-before-send scripts (`jog_gripper_raw.py`'s "one step per ENTER" design, `joint_jog.py`'s `--max-relative-target`) are the established mitigation pattern in this repo — reuse them for D-07 rather than sending large/unbounded raw commands |
| Path confusion if the generated URDF's mesh paths are relative to the wrong working directory | Tampering (of intent, not malicious) | Use paths relative to the URDF file's own location (standard URDF convention), and verify with TWIN-04's own no-absolute-path check plus an actual `yourdfpy` load (which will fail to resolve meshes if paths are wrong relative to cwd) |

This phase does not touch the real robot's control path, any network-facing code, or any
authentication/authorization surface — `security_enforcement` is satisfied by the two rows above,
both already covered by the phase's own TWIN-04 requirement and existing `control/` script patterns.

## Sources

### Primary (HIGH confidence — direct file reads / installed package source, this session)

- `LIBERO/libero/libero/assets/robots/soarm101/robot.xml` — full read, arm MJCF kinematic tree, joint limits, header comments
- `LIBERO/libero/libero/assets/grippers/soarm_gripper.xml` — full read, gripper MJCF, clamp mesh asset notes
- `LIBERO/libero/libero/envs/grippers/soarm_gripper.py` — `SoarmGripper.format_action` action convention
- `So-101/So-101.urdf` — full read, confirmed broken structure (absolute paths, no wrist_roll/gripper joints, terminal `wrist_link_respondable`)
- `coppelia/soarm_parallel_gripper.urdf`, `coppelia/export_model_library.py` — gripper-fragment reference, root-level-gripper bug confirmation
- `LIBERO/libero/libero/assets/robots/soarm101/so101_new_calib.source.xml` + `VENDOR.txt` — traced `robot.xml`'s current joint limits to this upstream vendor file
- `control/.venv/lib/python3.12/site-packages/lerobot/robots/robot.py` — `calibration_dir`/`calibration_fpath` location logic
- `control/.venv/lib/python3.12/site-packages/lerobot/utils/constants.py` — `HF_LEROBOT_CALIBRATION` default path resolution
- `control/.venv/lib/python3.12/site-packages/lerobot/robots/so_follower/so_follower.py` — `calibrate()` method, `wrist_roll` full-turn special-casing, `use_degrees` default
- `control/.venv/lib/python3.12/site-packages/lerobot/motors/motors_bus.py` — `_normalize`/`_unnormalize` DEGREES-mode formula (lines 854-911)
- `control/.venv/lib/python3.12/site-packages/lerobot/motors/feetech/feetech.py` + `tables.py` — `model_resolution_table["sts3215"] = 4096`, homing-offset-applied-at-firmware-level confirmation
- `~/.cache/huggingface/lerobot/calibration/robots/so_follower/soarm_follower_02.json` — actual live calibration file, located and read this session
- `control/COMMANDS.md`, `control/jog_gripper_raw.py`, `control/recalibrate_gripper.py` — hardware device table, D-07 manual-check scripts, gripper calibration history/caveats
- `LIBERO/libero/libero/envs/test_camera_config.py` — existing `env.reset()` verification test for TWIN-06
- `pip index versions yourdfpy` / `urchin` / `urdf-parser-py` (PyPI registry, this session) — version/existence confirmation
- `gsd-tools query package-legitimacy check --ecosystem pypi` (this session) — legitimacy verdicts for all three URDF-parser candidates

### Secondary (MEDIUM confidence)

- WebFetch of `github.com/Yasu31/mjcf_urdf_simple_converter` README — prismatic-joint support gap (drives the hand-rolled-converter recommendation)

### Tertiary (LOW confidence)

- WebSearch summary listing `mjcf2urdf` and a Bullet3-bundled converter as alternative tools — not independently verified beyond the search snippet; not recommended, listed only for completeness in Alternatives Considered

## Metadata

**Confidence breakdown:**
- Standard stack (URDF parser choice): MEDIUM — the recommendation itself (yourdfpy) is well-reasoned but the specific package's legitimacy signals came back SUS on download-count data availability alone; gate with `checkpoint:human-verify`
- Architecture (hand-rolled converter vs. library): HIGH — directly confirmed the one real candidate library's prismatic-joint gap, and the source MJCF files are fully read and understood
- Joint-limit derivation (calibration formula + live values): HIGH — formula read directly from installed LeRobot source, live calibration file read directly, cross-check math independently verified with a script
- Pitfalls: HIGH — all three pitfalls trace to specific, directly-read source code/comments, not inference

**Research date:** 2026-09-18
**Valid until:** 30 days for the joint-limit/calibration findings (stable, local hardware state
unlikely to change) — but re-verify the calibration file's mtime/content if significant time passes
before execution, since `control/recalibrate_gripper.py`'s own docstring shows this file has been
hand-edited/re-run before. 7 days for the `yourdfpy`/PyPI package version claims (fast-moving
ecosystem, re-run `pip index versions` at execution time).
