# Phase 2: SOARM Robot Integration - Research

**Researched:** 2026-07-11
**Domain:** robosuite 1.4.1 / LIBERO custom robot integration (MJCF adaptation, ManipulatorModel/GripperModel wiring)
**Confidence:** HIGH (nearly all load-bearing claims verified against installed robosuite 1.4.1 source and empirical MuJoCo 2.3.7 compilation of the actual SO101 MJCF)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### BDDL Task Selection
- **D-01:** Adapted tasks come from the `libero_spatial` suite — simplest scenes, sets up reuse for Phase 5 (SPAT-05 needs spatial-language BDDL tasks), lower physics-instability risk than `libero_goal`/`libero_object`.
- **D-02:** Reuse existing BDDL files as-is first (swap robot name only). Only modify scene/object placement if a specific task genuinely fails due to SOARM's reach/workspace during verification — no speculative pre-adjustment.
- **D-03:** Claude (researcher/planner) selects the specific 3+ `libero_spatial` task names during planning, informed by workspace constraints discovered while building the MJCF/robot class.

#### Robot Config & Mount
- **D-04:** Ground-mounted only — mirror `OnTheGroundPanda`, not `MountedPanda`. Physical SO101 is a tabletop arm; no mount-hardware use case exists yet.
- **D-05:** Model SOARM's actual gripper (geometry already exists in the source MJCF, not built from scratch).
- **D-06:** Split the gripper into a separate robosuite `GripperModel` class (e.g. `SoarmGripper`), not fused into the arm body — required for robosuite's gripper attachment/action-space pipeline (`default_gripper` property, LIBERO's actionable-gripper assumption).
- **D-07:** SOARM's `init_qpos` (home/rest pose) should start from SO101's documented default/calibration pose (if encoded in `so101_new_calib.xml`) rather than an arbitrary sim-tuned pose — adjust only if that pose proves unstable in sim.

#### Deliverable Format & Workspace
- **D-08:** Robot/gripper Python classes live as real git-tracked modules in the LIBERO fork — `LIBERO/libero/libero/envs/robots/soarm.py` and a gripper module (e.g. `LIBERO/libero/libero/envs/grippers/soarm_gripper.py`), mirroring `on_the_ground_panda.py`'s structure, registered in `ROBOT_CLASS_MAPPING` via `LIBERO/libero/libero/envs/robots/__init__.py` (same pattern as existing Panda variants).
- **D-09:** MJCF/robot-class editing and iteration happens **locally** (no GPU required — pure MuJoCo physics + Python; reuse the `MUJOCO_GL=glfw` macOS pattern from `explorations/create_scene.py` for any local rendering checks).
- **D-10:** Verification of ENV-04 through ENV-07 happens on **Colab**, via a new notebook (`LIBERO/notebooks/02-soarm-integration-check.ipynb`) with PASS/FAIL cells per success criterion — consistent with Phase 1's UAT pattern. This is the only part of this phase that needs the Colab GPU runtime.

#### Physics Tuning Scope
- **D-11:** Tune joint damping, `init_qpos`, and base offsets only to the point of passing success criteria (contact forces <10N at reset, stable rendering, 3+ BDDL tasks complete without crashing) — do not over-invest in hardware-accurate realism beyond that. Remaining physics issues can be refined in later phases (dataset collection, fine-tuning) if they surface.

### Claude's Discretion
- Exact joint damping values, base_xpos_offset per scene type, and horizontal_radius/top_offset tuning — same pattern as `OnTheGroundPanda`, tuned iteratively to pass success criteria.
- Which 3+ specific `libero_spatial` BDDL task files to adapt (D-03).
- Whether any BDDL scene/object repositioning is needed (only if reuse-as-is fails).
- Exact structure/cell layout of the Colab verification notebook, following Phase 1's established pattern.

### Deferred Ideas (OUT OF SCOPE)
- Mounted (wall/ceiling) SOARM variant — no current use case; only ground-mounted is in scope for this phase (D-04). Revisit if a future phase needs it.
- Hardware-accurate physics tuning beyond what success criteria require — deferred to whichever later phase first surfaces a concrete physics-fidelity problem (D-11).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ENV-04 | SOARM SO101 MJCF (from TheRobotStudio/SO-ARM100) is adapted for robosuite 1.4 and registered as a ManipulatorModel subclass | Full MJCF adaptation checklist below (kv strip, default-class inlining, motor actuators, XML split, mesh path fix); `ManipulatorModel` subclass property contract verified from installed robosuite 1.4.1 source; empirical MuJoCo 2.3.7 compile of SO101 XML confirms only `kv` blocks it |
| ENV-05 | SOARM robot is registered in LIBERO's ROBOT_CLASS_MAPPING and can be instantiated in a LIBERO environment | Registration mechanics verified: `REGISTERED_ROBOTS` auto-populates via metaclass on class definition; `ROBOT_CLASS_MAPPING.update()` in `LIBERO/libero/libero/envs/robots/__init__.py`; **critical naming discovery** — `Libero_Tabletop_Manipulation` prepends `"Mounted"` to the passed robot name (see Pitfall 1) |
| ENV-06 | At least 3 BDDL tasks are configured to use SOARM (replacing Panda arm) | Robot selection is an **env-creation kwarg** (`robots=[...]` on `OffScreenRenderEnv`), NOT stored in BDDL files — "configuring a task for SOARM" = passing the robot name at env creation; candidate task list from `libero_spatial` given below |
| ENV-07 | Rendered frames from SOARM environment are visually correct (right-side-up, correct camera angle, physics stable) | Camera requirements verified: default LIBERO `camera_names=["agentview", "robot0_eye_in_hand"]` means the arm XML MUST carry an `eye_in_hand` camera; contact-force measurement recipe for the <10 N criterion provided in Code Examples |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **Framework pins are hard:** LIBERO + robosuite 1.4.x + MuJoCo 2.3.7 + gym 0.25.2. robosuite 1.5+ is explicitly out of scope (SingleArmEnv removed). All MJCF adaptation must target MuJoCo **2.3.7** schema.
- **`MUJOCO_GL` must be set before any MuJoCo import** — `glfw` locally on macOS, `egl` on Colab (Phase 1 contract).
- **LIBERO first-import reads `~/.libero/config.yaml`** — must exist before any LIBERO-dependent script (verified present locally).
- **Python path pattern:** scripts importing LIBERO insert the `LIBERO` repo dir into `sys.path` first (see `explorations/create_scene.py`).
- **Conventions:** snake_case files/functions, UPPER_SNAKE_CASE module constants, `matplotlib.use("Agg")` before pyplot in visualization scripts, `print(f"Saved → {path}")` output pattern.
- **GSD workflow enforcement:** edits happen through GSD commands (this phase runs via `/gsd-execute-phase`).
- **Compute:** Colab GPU only for verification (D-09/D-10 split honors this); local work is CPU MuJoCo.
- **Required reading before Colab work:** `.planning/phases/01-colab-environment-setup/01-DEBUG-HISTORY.md` (7-invariant environment contract).

## Summary

This phase is a wiring/adaptation exercise with one genuinely novel artifact: a robosuite-1.4-compatible split of the SO101 MJCF. Everything else follows patterns that already exist in the codebase (`OnTheGroundPanda`, `ROBOT_CLASS_MAPPING.update`, Phase 1's PASS/FAIL notebook). Research verified every step of the pipeline against the **installed robosuite 1.4.1 source** in the `libero` conda env and **empirically compiled the actual `so101_new_calib.xml` under MuJoCo 2.3.7** — so the adaptation checklist below is grounded, not speculative.

Five hard facts drive the plan: (1) the stock SO101 XML **does not compile** under MuJoCo 2.3.7 — the `kv` attribute on its `<position>` actuators is a MuJoCo 3.x feature; (2) robosuite's `SingleArm.control()` writes **raw torques** into `sim.data.ctrl` for arm actuators, so the SO101's position actuators must become `<motor>` actuators for the arm (the gripper keeps a position actuator — robosuite rescales gripper actions into `ctrlrange`); (3) robosuite's XML `merge()` copies worldbody/assets/actuators/sensors but **silently drops `<default>` blocks**, so all of SO101's `class="sts3215"` defaults must be inlined onto elements; (4) LIBERO's `Libero_Tabletop_Manipulation` problem class (used by ALL `libero_spatial` tasks) **prepends `"Mounted"` to the robot name you pass** — so the registered class must be named `Mounted<Name>` even though, per D-04, it will have `default_mount=None` and no pedestal; (5) the SO101's maximum horizontal reach is **0.479 m** (measured via forward kinematics under MuJoCo), versus a LIBERO table surface at z=0.90 m with objects placed around table center — the robot base must therefore be placed **on the tabletop** (z≈0.90) near the table edge, not on the floor like Panda.

The riskiest part is not registration (mechanical, fully understood) but physics/controller behavior of a 0.632 kg, ±2.94 Nm arm under an OSC controller tuned for a Panda-scale robot — this is exactly the 1–2 day iteration budget STATE.md flags. Success criteria only require stable reset, correct rendering, and 3 tasks running without crashing (not task success), which is achievable with damping/init_qpos/base-offset tuning (D-11).

**Primary recommendation:** Build two hand-edited XMLs (arm + gripper) from `so101_new_calib.xml` following the adaptation checklist below, mirror `OnTheGroundPanda`/`PandaGripper` class shapes exactly, register `MountedSoarm101` (+ optional `OnTheGroundSoarm101`) and `SoarmGripper`, iterate locally with a CPU sanity script until reset is stable, then verify on Colab with the Phase-2 notebook.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| SO101 geometry/kinematics (meshes, joints, inertials) | MJCF asset layer (`LIBERO/libero/libero/assets/robots/soarm101/`) | — | Vendored from TheRobotStudio/SO-ARM100; edited XML + STL files are data, not code |
| Arm model contract (init_qpos, offsets, gripper name, damping) | robosuite model class (`envs/robots/soarm.py`) | MJCF asset layer | robosuite reads all placement/physics knobs from `ManipulatorModel` properties |
| Gripper model contract (jaw actuation, grip_site, sensors) | robosuite gripper class (`envs/grippers/soarm_gripper.py`) + gripper XML | — | robosuite's gripper pipeline requires a separate `GripperModel` (D-06) |
| Name → class resolution | LIBERO registration (`envs/robots/__init__.py`) | robosuite `REGISTERED_ROBOTS` (automatic via metaclass) | `ROBOT_CLASS_MAPPING` and `GRIPPER_MAPPING` are manual dict updates; model class registration is automatic on import |
| Task selection / robot swap | Env-creation config (notebook / scripts passing `robots=["Soarm101"]`) | — | BDDL files do not name the robot; the env kwarg does |
| Verification (ENV-04..07 PASS/FAIL) | Colab notebook (`LIBERO/notebooks/02-soarm-integration-check.ipynb`) | Local CPU sanity script | GPU rendering + full BDDL runs on Colab (D-10); physics iteration local (D-09) |

## Standard Stack

### Core (all already installed — no new packages this phase)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| robosuite | 1.4.1 | ManipulatorModel/GripperModel/SingleArm pipeline | Verified installed in local `libero` conda env; LIBERO hard dependency [VERIFIED: local site-packages inspection] |
| mujoco | 2.3.7 | Physics + MJCF compilation | Verified installed locally; project pin [VERIFIED: `python -c "import mujoco"` → 2.3.7] |
| LIBERO (vendored fork) | local | BDDL envs, TASK_MAPPING, ROBOT_CLASS_MAPPING extension point | The repo itself [VERIFIED: codebase] |
| numpy | 1.22.x | init_qpos / damping arrays | Existing pin [VERIFIED: codebase] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| matplotlib | 3.5.x | Local frame-render sanity checks (Agg backend) | Local iteration only (D-09) |
| imageio (on Colab, from Phase 1 env) | as installed | Notebook frame display/video | Colab verification notebook |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-editing `so101_new_calib.xml` | MuJoCo Menagerie `trs_so_arm100` model | Menagerie model targets MuJoCo 3.x and is SO-100 (not SO-101); the SO-ARM100 repo XML is the canonical SO101 source per prior STATE.md decision — hand-edit it [VERIFIED: STATE.md decision + empirical compile test] |
| `<motor>` torque actuators + OSC_POSE | Keep `<position>` actuators + JOINT_POSITION controller | Would break LIBERO's default OSC_POSE pipeline and Phase 3's 7-D delta-EEF action contract (VLA-01); torque motors are what every robosuite robot uses [VERIFIED: robosuite source — `control()` writes torques to ctrl] |
| Robot base on tabletop (z≈0.90) | Custom pedestal MountModel raising base from floor | Pedestal = new mount asset to build/tune; on-table placement matches the physical SO101 use case and needs zero new assets. Deferred per D-04 |

**Installation:** none — this phase adds zero packages. Colab env comes from Phase 1's notebook install cells; local env is the existing `libero` conda env.

## Package Legitimacy Audit

No external packages are installed in this phase. All work uses the already-pinned Phase 1 environment (robosuite 1.4.1, mujoco 2.3.7) and vendored LIBERO fork.

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

Vendored **assets** (not packages) are pulled from `github.com/TheRobotStudio/SO-ARM100` (`Simulation/SO101/so101_new_calib.xml` + `assets/*.stl`) — an established open-hardware repo already designated as the canonical source in STATE.md. Pin the commit SHA when vendoring and record it in the plan for reproducibility.

## Architecture Patterns

### System Architecture Diagram

```
                        env creation (notebook / script)
                        OffScreenRenderEnv(bddl_file_name=..., robots=["Soarm101"])
                                        |
                                        v
        BDDL file ---> get_problem_info() -> TASK_MAPPING["libero_tabletop_manipulation"]
        (libero_spatial)                      = Libero_Tabletop_Manipulation
                                        |
                                        |  kwargs["robots"] = ["Mounted" + "Soarm101"]   <-- STRING PREPEND (Pitfall 1)
                                        v
        BDDLBaseDomain(SingleArmEnv) --- robosuite RobotEnv._load_robots()
                                        |
                    ROBOT_CLASS_MAPPING["MountedSoarm101"] -> SingleArm
                                        |
                    SingleArm.load_model()
                        |-- create_robot("MountedSoarm101")  (REGISTERED_ROBOTS, auto via metaclass)
                        |       -> MountedSoarm101(ManipulatorModel)  reads soarm101 arm XML
                        |-- gripper_factory(robot.default_gripper)   (GRIPPER_MAPPING, manual update)
                        |       -> SoarmGripper(GripperModel)        reads soarm gripper XML
                        |-- robot_model.add_gripper(gripper)  merges gripper XML onto "right_hand" body
                                        |
                    BDDLBaseDomain._load_model()
                        |-- base_xpos_offset["table"](table_len) -> set_base_xpos()   (place base ON tabletop)
                        |-- ManipulationTask(arena, robot_model, objects)  -> merged final MJCF
                        |       (merge() drops <default> blocks -- Pitfall 2)
                                        v
                    MuJoCo 2.3.7 compile -> env.reset() -> OSC_POSE torques -> sim.data.ctrl
                                        v
                    offscreen render: "agentview" (arena) + "robot0_eye_in_hand" (arm XML camera)
```

### Recommended Project Structure

```
LIBERO/libero/libero/
├── assets/robots/soarm101/
│   ├── robot.xml                 # adapted arm MJCF (5 joints, motor actuators, right_hand body, eye_in_hand cam)
│   └── assets/*.stl              # 13 vendored STL meshes from SO-ARM100 (pin commit SHA)
├── assets/grippers/
│   └── soarm_gripper.xml         # adapted gripper MJCF (jaw joint+position actuator, eef body, grip sites, ft sensors)
├── envs/robots/
│   ├── __init__.py               # + import Soarm classes, ROBOT_CLASS_MAPPING.update, GRIPPER_MAPPING update
│   └── soarm.py                  # MountedSoarm101 (+ optional OnTheGroundSoarm101) ManipulatorModel subclasses
├── envs/grippers/
│   ├── __init__.py               # new package
│   └── soarm_gripper.py          # SoarmGripper(GripperModel)
LIBERO/notebooks/
└── 02-soarm-integration-check.ipynb   # Colab PASS/FAIL verification (D-10)
explorations/
└── soarm_sanity.py               # local CPU reset/contact-force/render iteration script (D-09)
```

### Pattern 1: MJCF adaptation checklist (SO101 → robosuite 1.4 / MuJoCo 2.3.7)

**What:** The exact set of edits to turn `so101_new_calib.xml` into a working robosuite arm+gripper pair. Every item below is verified against robosuite 1.4.1 source or empirical compile.

1. **Strip `kv` from all `<position>` actuators.** `kv` is rejected by MuJoCo 2.3.7's schema (`ValueError: unrecognized attribute: 'kv'`). Compensate lost velocity feedback with joint damping. [VERIFIED: empirical compile — stock XML fails, kv-stripped XML compiles clean under 2.3.7]
2. **Inline every `<default>` class onto elements, then delete the `<default>` blocks.** robosuite's `MujocoXML.merge()` copies `worldbody/asset/actuator/sensor/tendon/equality/contact` children only — `<default>` is dropped, so `class="sts3215"` references would break the merged task XML. Inline: joints get `damping="0.6" frictionloss="0.052" armature="0.028"`; visual geoms get `contype="0" conaffinity="0" group="1"`; collision geoms get `group="0"`. Inlining also protects the values from `RobotModel.__init__`'s `force=False` defaults stomp (it only skips attributes literally present on the element). [VERIFIED: robosuite/models/base.py `merge()` + robot_model.py `__init__`]
3. **Convert the 5 arm actuators to `<motor>` torque actuators** (`ctrlrange="-2.94 2.94"` per STS3215 forcerange, `ctrllimited="true"`). `SingleArm.control()` does `sim.data.ctrl[arm_actuator_idx] = torques` — position actuators would interpret torques as position targets. The gripper actuator **stays `<position>`** (kp≈998 from source, drop kv): `Manipulator.grip_action()` rescales the −1..1 gripper action into the actuator ctrlrange as a position target. [VERIFIED: robosuite/robots/single_arm.py line ~261, manipulator.py `grip_action`]
4. **Split into two XMLs at the wrist_roll→gripper boundary.** Arm XML keeps bodies `base→shoulder→upper_arm→lower_arm→wrist` plus the `gripper` body **renamed to `right_hand`** (keeping the `wrist_roll` joint and the servo geoms, minus the jaw parts). Gripper XML gets the fixed-jaw geoms (`wrist_roll_follower_so101_v1`), the `moving_jaw_so101_v1` body with its `gripper` joint, and the gripper position actuator. Renaming to `right_hand` matches `ManipulatorModel._eef_name` default so no override is needed. [VERIFIED: manipulator_model.py `_eef_name` = "right_hand"; SO101 body tree from compiled model]
5. **Gripper XML must contain (all verified hard requirements):**
   - a body named **`eef`** — `GripperModel.__init__` looks it up by name and crashes without it [VERIFIED: gripper_model.py `__init__`]
   - sites **`grip_site`** and **`grip_site_cylinder`** — `SingleArm._setup_references` resolves both via `site_name2id`; `grip_site` also becomes the OSC controller's `eef_name` [VERIFIED: single_arm.py lines ~123, 215–216]
   - sites `ee`, `ee_x`, `ee_y`, `ee_z` (referenced by the default `_important_sites`; cheap to include, mirrors panda_gripper.xml)
   - a site `ft_frame` with **`<force name="force_ee">` and `<torque name="torque_ee">` sensors** — `SingleArm.control()` reads `ee_force`/`ee_torque` **every policy step**, so missing sensors crash every `env.step()` [VERIFIED: single_arm.py line ~269 + gripper_model.py `_important_sensors`]
   - Good starting `grip_site` position: the source XML's `gripperframe` site pos `(-0.0079, -0.0002, -0.0981)` in the gripper body frame (between the jaws).
6. **Fix mesh asset paths.** The source uses `<compiler meshdir="assets"/>` with bare `file="x.stl"`. robosuite's `resolve_asset_dependency()` joins the XML's folder + the `file` attribute and **ignores meshdir** (and the compiler element is lost on merge anyway). Change every mesh to `file="assets/x.stl"` and drop `meshdir`. Give each `<mesh>` an explicit `name` attribute (robosuite convention). [VERIFIED: robosuite/models/base.py `resolve_asset_dependency`]
7. **Add an `eye_in_hand` camera to the `right_hand` body** in the arm XML. LIBERO's default `camera_names=["agentview", "robot0_eye_in_hand"]` — without this camera, every default env creation fails. Mirror Panda's: `<camera mode="fixed" name="eye_in_hand" pos=... quat=... fovy="75"/>`, orientation tuned so it looks along the gripper axis. [VERIFIED: env_wrapper.py lines 31–34 + panda robot.xml]
8. **Keep the base body free of collision geoms** (the source XML already has visual-only base geoms — SO-ARM100 removed base collision meshes deliberately). This lets the base sit flush on the tabletop without contact-force spikes. [VERIFIED: SO101 XML inspection + SO-ARM100 README]
9. **No keyframe exists in the source XML**; "new calibration" means qpos=0 is mid-range for every joint. So `init_qpos = np.zeros(5)` IS the documented calibration pose (satisfies D-07). Measured: at qpos=0 the gripper frame sits at (0.39, 0, 0.23) m from the base, arm extended forward. [VERIFIED: empirical FK under MuJoCo 2.3.7]

### Pattern 2: Robot class (mirror of on_the_ground_panda.py)

**What:** `ManipulatorModel` subclass with the 8 required properties.
**When to use:** exactly once, in `envs/robots/soarm.py`.

```python
# Source: mirrors LIBERO/libero/libero/envs/robots/on_the_ground_panda.py (verified local pattern)
import os
import numpy as np
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel

ASSETS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "assets")
)

class MountedSoarm101(ManipulatorModel):
    """SO-101 tabletop arm. Named 'Mounted...' because Libero_Tabletop_Manipulation
    prepends 'Mounted' to the robots kwarg; default_mount is still None (D-04)."""

    def __init__(self, idn=0):
        super().__init__(os.path.join(ASSETS, "robots/soarm101/robot.xml"), idn=idn)
        # 5 arm joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
        self.set_joint_attribute(
            attrib="damping", values=np.array((0.6, 0.6, 0.6, 0.6, 0.6))
        )  # starting point = STS3215 class value; tune per D-11

    @property
    def default_mount(self):
        return None            # D-04: no pedestal hardware

    @property
    def default_gripper(self):
        return "SoarmGripper"

    @property
    def default_controller_config(self):
        return "default_panda"  # fallback only; LIBERO always passes OSC_POSE explicitly

    @property
    def init_qpos(self):
        return np.zeros(5)      # D-07: new-calib zero = documented mid-range home

    @property
    def base_xpos_offset(self):
        # Table surface is at z=0.90 (workspace_offset in Libero_Tabletop_Manipulation).
        # SO101 reach is 0.479m max -> base must sit ON the table near its -x edge.
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.38 - 0.0 * table_length, 0, 0.90),  # tune x per D-11
            "kitchen_table": lambda table_length: (-0.38, 0, 0.90),
            "coffee_table": lambda table_length: (-0.30, 0, 0.41),
            "living_room_table": lambda table_length: (-0.30, 0, 0.42),
            "study_table": lambda table_length: (-0.38, 0, 0.90),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 0.30))   # arm is ~0.3m tall at home

    @property
    def _horizontal_radius(self):
        return 0.25                      # ~half of 0.479m max reach envelope

    @property
    def arm_type(self):
        return "single"
```

**Notes for planner:** the `table` lambda values above are *informed starting points*, not verified-stable values — expect D-11 iteration. `set_joint_attribute` asserts value count == number of joints in the arm XML (5). Panda's 7-element damping array will assert-fail if copied blindly.

### Pattern 3: Gripper class + registration

```python
# Source: mirrors robosuite/models/grippers/panda_gripper.py (verified local pattern)
import os
import numpy as np
from robosuite.models.grippers.gripper_model import GripperModel

ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets"))

class SoarmGripper(GripperModel):
    def __init__(self, idn=0):
        super().__init__(os.path.join(ASSETS, "grippers/soarm_gripper.xml"), idn=idn)

    def format_action(self, action):
        # -1 => open, 1 => close, integrated like PandaGripper
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def init_qpos(self):
        return np.array([0.8])   # jaw open (range -0.17..1.75 rad); tune

    @property
    def speed(self):
        return 0.10               # rad-scale jaw moves faster than Panda's 0.01 m fingers

    @property
    def dof(self):
        return 1                  # dof == len(actuators) == 1 -> total action dim 6+1=7

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["fixed_jaw_collision"],   # names from adapted gripper XML
            "right_finger": ["moving_jaw_collision"],
            "left_fingerpad": ["fixed_jaw_collision"],
            "right_fingerpad": ["moving_jaw_collision"],
        }
```

```python
# Source: LIBERO/libero/libero/envs/robots/__init__.py extension (verified registration mechanics)
from .mounted_panda import MountedPanda
from .on_the_ground_panda import OnTheGroundPanda
from .soarm import MountedSoarm101            # defining the class auto-registers it
                                              # in robosuite REGISTERED_ROBOTS (metaclass)
from robosuite.robots.single_arm import SingleArm
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.models.grippers import GRIPPER_MAPPING
from LIBERO.libero.libero.envs.grippers.soarm_gripper import SoarmGripper

ROBOT_CLASS_MAPPING.update(
    {
        "MountedPanda": SingleArm,
        "OnTheGroundPanda": SingleArm,
        "MountedSoarm101": SingleArm,
    }
)
GRIPPER_MAPPING["SoarmGripper"] = SoarmGripper   # gripper_factory asserts membership
```

[VERIFIED: robosuite robot_model.py `RobotModelMeta` auto-registration; gripper_factory.py asserts `name in GRIPPER_MAPPING`; LIBERO robots/__init__.py existing pattern]

### Pattern 4: Task creation with SOARM (ENV-06)

```python
# Source: verified against env_wrapper.py + libero_tabletop_manipulation.py
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import os

task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = task_suite.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

env = OffScreenRenderEnv(
    bddl_file_name=bddl,
    robots=["Soarm101"],        # Libero_Tabletop_Manipulation turns this into "MountedSoarm101"
    camera_heights=256,
    camera_widths=256,
)
obs = env.reset()
```

**Candidate `libero_spatial` tasks (D-03):** all 10 use the same problem class, arena, and table — physics risk is identical across them; they differ only in object regions. Recommend the three with object regions closest to table center/robot side (shortest required reach): `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`, `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`, `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`. Final pick is planner discretion after checking each BDDL's region coordinates against the tuned base position. [VERIFIED: all 10 libero_spatial BDDL files declare `(define (problem LIBERO_Tabletop_Manipulation)`]

### Anti-Patterns to Avoid

- **Copying Panda's floor-level base placement:** Panda reaches 0.855 m and stands on a pedestal; SO101 reaches 0.479 m. A floor-placed SO101 can never touch a 0.90 m tabletop. Base goes ON the table.
- **Keeping `<position>` actuators on arm joints:** torques written into position-actuator ctrl produce garbage motion that *looks* like a tuning problem but is a semantics bug.
- **Registering only `OnTheGroundSoarm101`:** libero_spatial would then look up `MountedOnTheGround...`? No — it looks up `Mounted` + the passed name, so `robots=["Soarm101"]` needs class `MountedSoarm101`. An `OnTheGround` variant is optional future-proofing for floor tasks only.
- **Editing robosuite site-packages** (e.g., dropping a controller JSON or gripper into robosuite's tree): keep everything in the LIBERO fork; robosuite dicts are mutated at LIBERO import time.
- **Testing renders only via `agentview`:** ENV-07's "right-side-up" criterion historically bites via raw `sim.render()` returning upside-down buffers; use `obs["agentview_image"]` (LIBERO/robosuite already flip it) and note `[::-1]` flipping in the notebook display cell.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Robot geometry/inertials | Any re-modeling | `so101_new_calib.xml` + STLs from SO-ARM100 | Locked (CONTEXT); inertials come from CAD via onshape-to-robot |
| EEF control | Custom IK/controller for 5-DOF arm | robosuite `OSC_POSE` (generic config LIBERO already loads) | OSC works for any DOF (J is 6×5); Phase 3 needs the standard 7-D action interface |
| Gripper action mapping | Custom gripper action pipeline | `GripperModel.format_action` + robosuite `grip_action` rescaling | robosuite already maps −1..1 to actuator ctrlrange |
| Robot registration | Custom factory/loader | `RobotModelMeta` auto-registration + `ROBOT_CLASS_MAPPING`/`GRIPPER_MAPPING` updates | Verified existing extension points |
| Scene/task assembly | Custom MJCF task composer | LIBERO `BDDLBaseDomain` + `ManipulationTask` merge | Whole point of registering into LIBERO |
| Contact-force measurement | Custom force estimation | `mujoco.mj_contactForce` / `sim.data.cfrc_ext` | Built into MuJoCo; recipe in Code Examples |

**Key insight:** every integration point in this phase is an existing dict/property/XML contract in robosuite 1.4.1 — the only creative work is the XML surgery and the numeric tuning.

## Common Pitfalls

### Pitfall 1: `libero_spatial` prepends "Mounted" to the robot name
**What goes wrong:** You register `Soarm101` or `OnTheGroundSoarm101`, pass `robots=["Soarm101"]`, and env creation raises `Robot MountedSoarm101 not found` (or KeyError in ROBOT_CLASS_MAPPING).
**Why it happens:** `Libero_Tabletop_Manipulation.__init__` does `kwargs.update({"robots": [f"Mounted{robot_name}" ...]})` — every libero_spatial task routes through it. `libero_floor_manipulation` similarly prepends `OnTheGround`. [VERIFIED: problems/libero_tabletop_manipulation.py line ~23]
**How to avoid:** Name the class `MountedSoarm101`, register it in both `REGISTERED_ROBOTS` (automatic) and `ROBOT_CLASS_MAPPING` (manual). This does NOT conflict with D-04: `default_mount=None` means no pedestal is attached regardless of the class name. Document the naming irony in the module docstring.
**Warning signs:** exception naming a `Mounted`-prefixed class you never wrote.

### Pitfall 2: robosuite's merge silently drops `<default>` blocks
**What goes wrong:** Arm XML compiles standalone but the merged task XML fails with "unknown default class 'sts3215'" — or worse, if classes were removed but values not inlined, `RobotModel.__init__` silently stomps damping to 0.1 and armature to `[5.0, 2.5, 1.67, 1.25, 1.0]` (absurd for 0.1 kg links → sluggish/unstable arm that looks like a mystery physics bug).
**Why it happens:** `MujocoXML.merge()` only copies worldbody/asset/actuator/sensor/tendon/equality/contact; `set_joint_attribute(force=False)` writes defaults for any attribute not literally present on the joint element. [VERIFIED: models/base.py + robot_model.py]
**How to avoid:** Inline all class attributes onto elements and delete `<default>` blocks entirely (Panda XMLs are fully inlined — that's why robosuite never hits this).
**Warning signs:** arm behaves like it's moving through honey; merged-XML compile errors mentioning class names.

### Pitfall 3: Stock SO101 XML is MuJoCo 3.x-flavored
**What goes wrong:** `mujoco.MjModel.from_xml_path("so101_new_calib.xml")` raises `Schema violation: unrecognized attribute: 'kv'` under 2.3.7.
**Why it happens:** `kv` on `<position>` actuators (and the repo's `scene.xml` conveniences) target current MuJoCo. [VERIFIED: empirical compile under 2.3.7 — only `kv` blocks compilation; `autolimits`, `fullinertia`, defaults all accepted]
**How to avoid:** Strip `kv` (arm actuators become `<motor>` anyway; gripper keeps position without kv). After the strip, the XML compiles clean — no other 3.x features present.
**Warning signs:** any schema-violation error at compile.

### Pitfall 4: Missing gripper sites/sensors crash at setup or on every step
**What goes wrong:** `env.reset()` raises site-not-found for `gripper0_grip_site`/`grip_site_cylinder`; or reset succeeds but the first `env.step()` fails looking up `force_ee`.
**Why it happens:** `SingleArm._setup_references` hard-resolves both grip sites; `SingleArm.control()` reads force/torque sensors every step; `GripperModel.__init__` requires a body literally named `eef`. [VERIFIED: single_arm.py, gripper_model.py]
**How to avoid:** Gripper XML must contain: body `eef`, sites `grip_site`/`grip_site_cylinder`/`ee`/`ee_x`/`ee_y`/`ee_z`/`ft_frame`, and `<force name="force_ee">`+`<torque name="torque_ee">` sensors — mirror panda_gripper.xml exactly.
**Warning signs:** KeyError/`site_name2id` errors mentioning `gripper0_` names.

### Pitfall 5: Missing `eye_in_hand` camera breaks default env creation
**What goes wrong:** `OffScreenRenderEnv(bddl_file_name=...)` fails because the default `camera_names` includes `robot0_eye_in_hand`, which is defined in Panda's arm XML, not the arena.
**Why it happens:** env_wrapper defaults `camera_names=["agentview", "robot0_eye_in_hand"]`. [VERIFIED: env_wrapper.py lines 31–34]
**How to avoid:** Add `<camera name="eye_in_hand" .../>` to the `right_hand` body of the arm XML. Also directly serves SPAT-01 (wrist camera) in Phase 5.
**Warning signs:** camera-name errors during observable setup, or all-black `robot0_eye_in_hand` frames (bad pos/quat — tune while checking the render).

### Pitfall 6: Reach vs. table geometry — reset interpenetration and unreachable objects
**What goes wrong:** Contact forces >10 N at reset (arm spawns intersecting the table/objects), or tasks "run" but the arm can't reach any object.
**Why it happens:** Table surface z=0.90, size (1.0, 1.2); libero_spatial object regions cluster around table center. SO101 max horizontal reach is 0.479 m (measured), home pose extends the gripper 0.39 m forward at z+0.23. If the base sits at the table edge (x≈−0.5), center objects at 0.35–0.55 m are marginal; if the base is too close, the extended home pose can spawn inside object regions → contact spike.
**How to avoid:** Base on the tabletop (z=0.90 in the `table` lambda), x between −0.45 and −0.30 tuned per D-11; if the zero home pose interpenetrates, tuck the arm (e.g., shoulder_lift back, elbow up) — D-07 explicitly allows adjusting when unstable. Measure with the contact-force recipe below after every change.
**Warning signs:** reset contact forces in the hundreds of N; objects flying at reset; arm frozen at joint limits.

### Pitfall 7: OSC gains tuned for Panda-scale dynamics
**What goes wrong:** Arm oscillates, drifts, or barely moves under OSC_POSE with default kp=150 — the SO101 weighs 0.632 kg total with ±2.94 Nm torque limits vs. Panda's ~18 kg and ±80 Nm.
**Why it happens:** LIBERO loads the *generic* `osc_pose.json` (`suite.load_controller_config(default_controller="OSC_POSE")`), not a robot-specific config. OSC uses the model's mass matrix, so it partially self-scales, but torque clipping to ±2.94 Nm and joint damping interact badly if damping is too low/high. [VERIFIED: env_wrapper.py line 47; robot torque limits come from actuator ctrlrange]
**How to avoid:** This is the D-11 iteration loop: adjust joint damping (start 0.6, the STS3215 value) and, if needed, pass a custom `controller_configs` dict in the notebook (LIBERO accepts it as a kwarg — no robosuite file edits). Phase 2 only needs no-crash stepping, not tracking quality.
**Warning signs:** frames show jitter/vibration; qacc warnings; MuJoCo instability resets.

### Pitfall 8: 5-DOF arm under a 6-DOF OSC controller
**What goes wrong:** Expectation mismatch — full 6-DOF pose commands can't be tracked (only 5 DOF); orientation error is structural, not a bug.
**Why it happens:** OSC builds a 6×5 Jacobian task; the least-squares solution sacrifices one rotational direction.
**How to avoid:** Accept for Phase 2 (criteria don't require tracking). Document for Phase 3: OpenVLA's 7-D delta-EEF actions will exhibit partial orientation-following on SOARM; action_dim stays 7 (OSC control_dim 6 + gripper dof 1) so the VLA interface contract holds. [VERIFIED: action_dim = controller.control_dim + gripper.dof from single_arm.py]
**Warning signs:** none in this phase; note in SUMMARY for Phase 3.

## Code Examples

### Contact-force check for success criterion 1 (<10 N at reset)

```python
# Source: MuJoCo 2.3.7 API (mj_contactForce), verified pattern against installed mujoco
import numpy as np, mujoco

def max_contact_force(env):
    sim = env.sim
    m, d = sim.model._model, sim.data._data
    peak = 0.0
    for i in range(d.ncon):
        f6 = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f6)
        peak = max(peak, float(np.linalg.norm(f6[:3])))
    return peak

obs = env.reset()
for _ in range(10):                       # let transients settle a few control steps
    obs, *_ = env.step(np.zeros(7))
assert max_contact_force(env) < 10.0, "physics unstable at init"
```

### Local sanity loop (D-09, macOS)

```python
# Source: explorations/create_scene.py pattern (verified local codebase)
import os
os.environ["MUJOCO_GL"] = "glfw"          # BEFORE any mujoco import
import sys; sys.path.insert(0, "/path/to/SoARM-Research/LIBERO")
from libero.libero.envs import OffScreenRenderEnv
# ... create env with robots=["Soarm101"], reset, save agentview frame via matplotlib Agg
```

### Rendering correctness check (ENV-07)

```python
frame = obs["agentview_image"]            # HxWx3 uint8
# robosuite offscreen frames are bottom-up; LIBERO consumers display frame[::-1]
# PASS cell: save both agentview and robot0_eye_in_hand, human-verify right-side-up,
# arm visible, meshes intact (no missing/garbled STL), gripper centered in wrist cam
```

### Verification notebook skeleton (D-10)

```
Cell 1: Phase 1 Block A/B install + env bootstrap (reuse 01-colab-env-setup pattern, MUJOCO_GL=egl)
Cell 2: ENV-04 PASS/FAIL — import soarm module; MjModel compiles; class in REGISTERED_ROBOTS
Cell 3: ENV-05 PASS/FAIL — "MountedSoarm101" in ROBOT_CLASS_MAPPING; env instantiates
Cell 4: SC-1  PASS/FAIL — env.reset() + max_contact_force < 10 N
Cell 5: ENV-07 PASS/FAIL — render agentview + eye_in_hand frames, display, visual check
Cell 6: ENV-06 PASS/FAIL — loop 3 selected libero_spatial BDDLs x N random-action steps, no crash
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| robosuite 1.4: custom robots via source-tree classes + manual dict updates | robosuite 1.5+: `robosuite_models` plugin package + `register_robot_class` decorator | robosuite 1.5 (2024) | **Not usable here** — LIBERO pins 1.4.x (1.5 removed SingleArmEnv). Use the 1.4 dict-update pattern [CITED: robosuite.ai/docs, github.com/ARISE-Initiative/robosuite_models] |
| SO-ARM100 old calibration (`so101_old_calib.xml`, zero = horizontal extension) | New calibration (`so101_new_calib.xml`, zero = joint mid-range) | SO-ARM100 repo, 2025 | Use new calib (already locked in STATE.md); init_qpos=zeros is the documented home [CITED: SO-ARM100 Simulation/SO101/README.md] |
| Hand-written MJCF for hobby arms | onshape-to-robot generated MJCF with CAD inertials | SO-ARM100 pipeline | Inertials are trustworthy; don't re-derive |

**Deprecated/outdated:**
- MuJoCo 3.x MJCF attributes (`kv` on position actuators, `actuatorfrcrange`): not available in 2.3.7 — strip on sight.
- TechLabs Aachen SO100+robosuite prior art (STATE.md reference): confirmed to exist and validates the URDF→MJCF + iterative damping/gain tuning + simplified-collision approach, but **published no code or concrete values** — treat as directional confirmation only, not a copy source. [VERIFIED: article fetched; MEDIUM confidence on details]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Base placement `(-0.38, 0, 0.90)` on the tabletop yields <10 N reset contacts and adequate reach for the 3 chosen tasks | Pattern 2 / Pitfall 6 | Placement iteration takes longer; may need per-task BDDL region checks (D-02 fallback) |
| A2 | Generic OSC_POSE config produces crash-free (not necessarily accurate) stepping on the 0.6 kg arm after damping tuning | Pitfall 7 | May need a custom controller_configs dict (kwarg-level fix, still no robosuite edits) |
| A3 | Gripper `init_qpos=[0.8]`, `speed=0.10` are reasonable jaw values | Pattern 3 | Trivial retune during local iteration |
| A4 | Convex-hull collisions on the full SO101 STLs are stable enough under MuJoCo 2.3.7 (TechLabs simplified theirs; SO101 meshes are simpler) | Pattern 1 item 8 | Fallback: replace collision geoms with primitive boxes/capsules — an extra half-day |
| A5 | `grip_site` at the source `gripperframe` position is a usable EEF control point | Pattern 1 item 5 | OSC control quality suffers; move site between jaw tips |
| A6 | The three recommended libero_spatial tasks have the most robot-proximal object regions | Pattern 4 | Planner swaps in different tasks after reading region coords — zero structural impact |

## Open Questions

1. **Does the extended home pose (gripper 0.39 m forward) interpenetrate object spawn regions at reset for the chosen tasks?**
   - What we know: home pose FK measured; object regions are per-BDDL and near table center; init noise is disabled by LIBERO default (`initialization_noise=None`).
   - What's unclear: exact overlap depends on the tuned base x-position and per-task regions.
   - Recommendation: plan a local iteration task that reads each candidate BDDL's region ranges and checks them against the arm's swept home volume before Colab verification; tuck init_qpos if needed (D-07 allows).
2. **Will OSC torque clipping at ±2.94 Nm cause MuJoCo instability warnings under random actions?**
   - What we know: torques are clipped to actuator ctrlrange; damping 0.6 per joint is the hardware-derived value.
   - What's unclear: interaction of kp=150 OSC gains with tiny link inertias at control_freq=20.
   - Recommendation: local random-action soak test (500 steps × 3 seeds) as an explicit plan task before the Colab run.
3. **Eye-in-hand camera pose** — needs a tune-by-render loop; no ground truth exists for SO101.
   - Recommendation: dedicate a local render-check iteration; acceptance = gripper jaws visible at frame bottom, workspace centered.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| conda env `libero` (python 3.9) | local iteration (D-09) | ✓ | robosuite 1.4.1, mujoco 2.3.7 | — |
| MUJOCO_GL=glfw rendering (macOS) | local render checks | ✓ (Phase-1-proven pattern) | — | skip local rendering, physics-only checks |
| `~/.libero/config.yaml` | any LIBERO import | ✓ | — | — |
| SO-ARM100 repo access (GitHub) | vendoring XML + 13 STLs | ✓ (fetched during research) | main branch | already-downloaded copies in scratchpad can be re-fetched; pin SHA |
| Colab GPU runtime + Phase 1 notebook env | ENV-04..07 verification (D-10) | ✓ (Phase 1 closed 4/4 PASS) | — | none — verification blocks without Colab |
| robosuite source extension points (ROBOT_CLASS_MAPPING, GRIPPER_MAPPING, REGISTERED_ROBOTS) | registration | ✓ verified in 1.4.1 | — | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** local rendering (physics-only fallback).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | None (no pytest infra in repo) — validation via executable scripts + Colab PASS/FAIL notebook cells (Phase 1 precedent) |
| Config file | none — see Wave 0 |
| Quick run command | `conda run -n libero python explorations/soarm_sanity.py` (local, CPU, ~30 s) |
| Full suite command | Run all PASS/FAIL cells in `LIBERO/notebooks/02-soarm-integration-check.ipynb` on Colab |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ENV-04 | Adapted MJCF compiles; ManipulatorModel subclass instantiates | smoke | `conda run -n libero python explorations/soarm_sanity.py --check compile` | ❌ Wave 0 |
| ENV-05 | `MountedSoarm101` in ROBOT_CLASS_MAPPING; LIBERO env instantiates + resets | integration | `conda run -n libero python explorations/soarm_sanity.py --check reset` (local headless) + notebook Cell 3/4 | ❌ Wave 0 |
| ENV-06 | 3 libero_spatial BDDLs run N steps without crash | integration | notebook Cell 6 (Colab); local variant in sanity script | ❌ Wave 0 |
| ENV-07 | Frames right-side-up, correct camera, no mesh artifacts; contact <10 N | smoke + manual-only (visual correctness needs human eyes) | notebook Cells 4–5; contact-force assert automated, visual check human | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `conda run -n libero python explorations/soarm_sanity.py` (compile + reset + contact-force asserts, <30 s CPU)
- **Per wave merge:** sanity script full mode (adds 500-step random-action soak on one BDDL task)
- **Phase gate:** all notebook PASS/FAIL cells green on Colab before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `explorations/soarm_sanity.py` — covers ENV-04, ENV-05, and the automatable half of ENV-07 (contact force); must exist before/with the first robot-class task
- [ ] `LIBERO/notebooks/02-soarm-integration-check.ipynb` — covers ENV-04..07 PASS/FAIL on Colab (built late in the phase, consuming the Phase 1 install-cell pattern)
- [ ] No framework install needed

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (offline research sim; only existing HF-token pattern from Phase 1, unchanged) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | marginal | BDDL/XML files are repo-local, not user-supplied; no untrusted input path |
| V6 Cryptography | no | — |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Supply-chain: vendored third-party assets (SO-ARM100 XML/STLs) | Tampering | Pin the upstream commit SHA in the plan/commit message; vendor files into git (reviewable diffs) rather than downloading at runtime |
| Notebook secrets (HF token on Colab) | Information disclosure | Reuse Phase 1's Drive-file token pattern; never inline tokens in the notebook |

No other security surface: no network services, no user input, no persistence beyond git-tracked files.

## Sources

### Primary (HIGH confidence — verified by direct inspection/execution this session)
- Installed robosuite 1.4.1 source (`/opt/homebrew/Caskroom/miniconda/base/envs/libero/lib/python3.9/site-packages/robosuite/`) — manipulator_model.py, robot_model.py, gripper_model.py, gripper_factory.py, single_arm.py, manipulator.py, models/base.py (merge/resolve_asset_dependency), robots/__init__.py, panda robot.xml, panda_gripper.xml, controller configs
- LIBERO fork source — envs/robots/*, envs/env_wrapper.py, envs/bddl_base_domain.py, envs/problems/*.py, bddl_files/libero_spatial/*
- `so101_new_calib.xml` + 13 STL assets fetched from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101) and **empirically compiled under local MuJoCo 2.3.7** (kv failure reproduced; FK reach/mass measured)
- `.planning/` docs: CONTEXT.md, REQUIREMENTS.md, STATE.md, Phase 1 artifacts

### Secondary (MEDIUM confidence)
- [SO-ARM100 Simulation/SO101 README](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/README.md) — calibration semantics, mesh layout, STS3215 gain provenance
- [TechLabs Aachen: Organizer Robot (SO100 + robosuite + SmolVLA)](https://techlabs-aachen.medium.com/organizer-robot-teaching-an-so100-to-restore-order-using-smolvla-and-robosuite-9b5f2d0558ed) — prior-art confirmation of the adaptation approach (no code published)

### Tertiary (LOW confidence)
- [robosuite 1.5 docs](https://robosuite.ai/docs/modules/robots.html) / [robosuite_models](https://github.com/ARISE-Initiative/robosuite_models) — noted only to rule out 1.5-era registration mechanisms

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — zero new packages; versions verified in the local env
- Architecture (registration/wiring): HIGH — every extension point read from installed 1.4.1 source; naming-prefix behavior read from LIBERO problem classes
- MJCF adaptation requirements: HIGH — stock-XML failure and post-fix compile reproduced empirically under MuJoCo 2.3.7; kinematics measured
- Pitfalls: HIGH for mechanics (1–5), MEDIUM for tuning outcomes (6–8) — numeric values are informed starting points requiring D-11 iteration
- Prior art (TechLabs): MEDIUM — approach confirmed, details unpublished

**Research date:** 2026-07-11
**Valid until:** 2026-08-11 (stable pinned stack; SO-ARM100 upstream may move — pin the vendored commit)
