# Phase 2: SOARM Robot Integration - Pattern Map

**Mapped:** 2026-07-11
**Files analyzed:** 8 new/modified files
**Analogs found:** 8 / 8 (2 analogs live in robosuite 1.4.1 site-packages — read-only references, never edit them)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `LIBERO/libero/libero/envs/robots/soarm.py` (new) | model (robot class) | config/property contract | `LIBERO/libero/libero/envs/robots/on_the_ground_panda.py` | exact |
| `LIBERO/libero/libero/envs/grippers/soarm_gripper.py` (new, new package) | model (gripper class) | config/property contract | robosuite site-packages `models/grippers/panda_gripper.py` | exact |
| `LIBERO/libero/libero/envs/robots/__init__.py` (modified) | config (registration) | dict mutation at import | itself (existing `ROBOT_CLASS_MAPPING.update` block) | exact |
| `LIBERO/libero/libero/assets/robots/soarm101/robot.xml` (new) | config (arm MJCF) | declarative asset | robosuite site-packages `models/assets/robots/panda/robot.xml` | exact |
| `LIBERO/libero/libero/assets/grippers/soarm_gripper.xml` (new) | config (gripper MJCF) | declarative asset | robosuite site-packages `models/assets/grippers/panda_gripper.xml` | exact |
| `LIBERO/libero/libero/assets/robots/soarm101/assets/*.stl` (new, 13 vendored meshes) | config (binary assets) | static data | Panda mesh layout (`meshes/` subdir convention) | role-match |
| `explorations/soarm_sanity.py` (new) | utility (local CPU sanity script) | request-response (env create → reset → render) | `explorations/create_scene.py` | exact |
| `LIBERO/notebooks/02-soarm-integration-check.ipynb` (new) | test (Colab PASS/FAIL verification) | batch (sequential PASS/FAIL cells) | `LIBERO/notebooks/01-colab-env-setup.ipynb` | exact |

Note: BDDL task files are **not** modified (D-02) — robot selection is an env-creation kwarg, not a BDDL field. No file entry needed unless verification fails.

## Pattern Assignments

### `LIBERO/libero/libero/envs/robots/soarm.py` (robot class)

**Analog:** `LIBERO/libero/libero/envs/robots/on_the_ground_panda.py` (entire file, 65 lines — copy the whole class shape)

**Imports pattern** (lines 1-4):
```python
import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion
```
Deviation: SOARM's XML lives in the LIBERO fork, not robosuite's assets tree, so replace `xml_path_completion(...)` with an absolute path built from `os.path.dirname(__file__)` (see RESEARCH.md Pattern 2's `ASSETS` constant). Do NOT use `xml_path_completion` — it resolves inside robosuite site-packages.

**Core pattern — `__init__` + damping** (lines 14-20):
```python
    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/panda/robot.xml"), idn=idn)

        # Set joint damping
        self.set_joint_attribute(
            attrib="damping", values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))
        )
```
SOARM version: 5-element array (start `0.6` per joint, STS3215 value). `set_joint_attribute` asserts len(values) == joint count — Panda's 7-element array will assert-fail if copied blindly.

**Property contract** (lines 22-64) — implement all 8, same decorator/order:
```python
    @property
    def default_mount(self):
        return None

    @property
    def default_gripper(self):
        return "PandaGripper"          # -> "SoarmGripper"

    @property
    def default_controller_config(self):
        return "default_panda"         # keep as-is (fallback only; LIBERO passes OSC_POSE)

    @property
    def init_qpos(self):
        return np.array([...])         # -> np.zeros(5)  (D-07: new-calib zero = home)

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "coffee_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.41),
            "living_room_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.42),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))   # -> ~(0, 0, 0.30)

    @property
    def _horizontal_radius(self):
        return 0.5                     # -> ~0.25

    @property
    def arm_type(self):
        return "single"
```
Critical deviation (RESEARCH Pitfall 6 / anti-pattern): do NOT copy Panda's floor-level `table` lambda z=0. SOARM reach is 0.479 m vs table surface z=0.90 — the `table`/`kitchen_table`/`study_table` lambdas must place the base ON the tabletop (z≈0.90, x≈−0.38, tune per D-11). RESEARCH.md Pattern 2 gives the full starting dict.

Critical naming (RESEARCH Pitfall 1): class must be named `MountedSoarm101` — `Libero_Tabletop_Manipulation` prepends `"Mounted"` to the `robots=[...]` kwarg. `default_mount` stays `None` (D-04).

---

### `LIBERO/libero/libero/envs/grippers/soarm_gripper.py` (gripper class)

**Analog:** `/opt/homebrew/Caskroom/miniconda/base/envs/libero/lib/python3.9/site-packages/robosuite/models/grippers/panda_gripper.py` (read-only reference)

**Imports pattern** (lines 4-7):
```python
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion   # replace with fork-local ASSETS path, as above
```

**Core pattern — integrated one-DOF action** (`PandaGripper`, lines 43-66):
```python
    def format_action(self, action):
        """Maps continuous action into binary output: -1 => open, 1 => closed"""
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + np.array([-1.0, 1.0]) * self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.01        # -> ~0.10 (rad-scale jaw vs Panda's 0.01 m slide fingers)

    @property
    def dof(self):
        return 1           # keep: total action dim = OSC 6 + gripper 1 = 7
```
SOARM has 1 actuator (single jaw joint), so the direction array `np.array([-1.0, 1.0])` collapses to `np.sign(action)` scaling only — see RESEARCH.md Pattern 3 for the exact 1-actuator variant.

**`init_qpos` + `_important_geoms` pattern** (`PandaGripperBase`, lines 24-35):
```python
    @property
    def init_qpos(self):
        return np.array([0.020833, -0.020833])   # -> np.array([0.8]) jaw-open, tune

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["finger1_collision", "finger1_pad_collision"],
            "right_finger": ["finger2_collision", "finger2_pad_collision"],
            "left_fingerpad": ["finger1_pad_collision"],
            "right_fingerpad": ["finger2_pad_collision"],
        }
```
Map to the adapted gripper XML's collision geom names (fixed jaw / moving jaw). No `Base` + subclass split needed — SOARM has one actuator, so a single class suffices.

---

### `LIBERO/libero/libero/envs/robots/__init__.py` (registration — MODIFY)

**Analog:** the file itself, entire current contents (13 lines):
```python
from .mounted_panda import MountedPanda
from .on_the_ground_panda import OnTheGroundPanda

from robosuite.robots.single_arm import SingleArm
from robosuite.robots import ROBOT_CLASS_MAPPING

ROBOT_CLASS_MAPPING.update(
    {
        "MountedPanda": SingleArm,
        "OnTheGroundPanda": SingleArm,
    }
)
```
Extend, don't replace: add `from .soarm import MountedSoarm101` (importing the class auto-registers it in robosuite `REGISTERED_ROBOTS` via metaclass), add `"MountedSoarm101": SingleArm` to the update dict, and add the gripper registration:
```python
from robosuite.models.grippers import GRIPPER_MAPPING
from ..grippers.soarm_gripper import SoarmGripper
GRIPPER_MAPPING["SoarmGripper"] = SoarmGripper   # gripper_factory asserts membership
```
(Use a relative import `..grippers.soarm_gripper`, not the absolute `LIBERO.libero...` path shown in RESEARCH.md Pattern 3 — the package is imported as `libero.libero.envs`.) The new `envs/grippers/` package needs an `__init__.py`.

---

### `LIBERO/libero/libero/assets/robots/soarm101/robot.xml` (arm MJCF)

**Analog:** `/opt/homebrew/Caskroom/miniconda/base/envs/libero/lib/python3.9/site-packages/robosuite/models/assets/robots/panda/robot.xml` (read-only reference)

Key structural elements to mirror (Panda robot.xml is fully inlined — no `<default>` blocks — which is exactly the state SOARM's XML must reach per RESEARCH Pitfall 2):

**`robotview` camera on base** (line 145):
```xml
<camera mode="fixed" name="robotview" pos="1.0 0 0.4" quat="0.653 0.271 0.271 0.653"/>
```

**`right_hand` terminal body + `eye_in_hand` camera** (lines 236-239) — mandatory (RESEARCH Pitfall 5):
```xml
<body name="right_hand" pos="0 0 0.1065" quat="0.924 0 0 -0.383">
    <!-- This camera points out from the eef. -->
    <camera mode="fixed" name="eye_in_hand" pos="0.05 0 0" quat="0 0.707108 0.707108 0" fovy="75"/>
```
SOARM's `gripper` body (minus jaw parts) is renamed `right_hand` to match `ManipulatorModel._eef_name`; camera pos/quat tuned by render loop (RESEARCH Open Question 3).

Full adaptation checklist (strip `kv`, inline defaults, `<motor>` arm actuators with `ctrlrange="-2.94 2.94"`, `file="assets/x.stl"` mesh paths, named meshes) is RESEARCH.md Pattern 1 items 1-9 — the planner should treat that checklist as the action list for this file.

---

### `LIBERO/libero/libero/assets/grippers/soarm_gripper.xml` (gripper MJCF)

**Analog:** `/opt/homebrew/Caskroom/miniconda/base/envs/libero/lib/python3.9/site-packages/robosuite/models/assets/grippers/panda_gripper.xml` (read-only reference, 54 lines)

**Mandatory skeleton to replicate exactly** (missing any item crashes reset or every step — RESEARCH Pitfall 4):

Root body + `ft_frame` site (lines 14-15):
```xml
<body name="right_gripper" pos="0 0 0" quat="0.707107 0 0 -0.707107">
    <site name="ft_frame" pos="0 0 0" size="0.01 0.01 0.01" rgba="1 0 0 1" type="sphere" group="1"/>
```

`eef` body with all required sites (lines 20-27):
```xml
<body name="eef" pos="0 0 0.097" quat="1 0 0 0">
    <site name="grip_site" pos="0 0 0" size="0.01 0.01 0.01" rgba="1 0 0 0.5" type="sphere" group="1"/>
    <site name="ee_x" pos="0.1 0 0" size="0.005 .1" quat="0.707105 0 0.707108 0" rgba="1 0 0 0" type="cylinder" group="1"/>
    <site name="ee_y" pos="0 0.1 0" size="0.005 .1" quat="0.707105 0.707108 0 0" rgba="0 1 0 0" type="cylinder" group="1"/>
    <site name="ee_z" pos="0 0 0.1" size="0.005 .1" quat="1 0 0 0" rgba="0 0 1 0" type="cylinder" group="1"/>
    <site name="grip_site_cylinder" pos="0 0 0" size="0.005 10" rgba="0 1 0 0.3" type="cylinder" group="1"/>
</body>
```
(`eef` pos for SOARM: start from source `gripperframe` site `(-0.0079, -0.0002, -0.0981)`.)

Force/torque sensors (lines 50-53):
```xml
<sensor>
    <force name="force_ee" site="ft_frame"/>
    <torque name="torque_ee" site="ft_frame"/>
</sensor>
```

Position actuator pattern (line 10) — SOARM keeps ONE position actuator (jaw), kp≈998 from source, `kv` stripped:
```xml
<position ctrllimited="true" ctrlrange="0.0 0.04" joint="finger_joint1" kp="1000" name="gripper_finger_joint1" forcelimited="true" forcerange="-20 20"/>
```

Collision geom style for jaws (line 32) — solref/friction/condim values worth copying for graspability:
```xml
<geom type="mesh" group="0" conaffinity="1" contype="0" solref="0.02 1" friction="1 0.005 0.0001" condim="4" mesh="finger" name="finger1_collision"/>
```

---

### `explorations/soarm_sanity.py` (local sanity script)

**Analog:** `explorations/create_scene.py` (entire file, 97 lines)

**Bootstrap pattern** (lines 8-23) — order is load-bearing (sys.path and MUJOCO_GL before any mujoco/libero import; `matplotlib.use("Agg")` before pyplot):
```python
import sys
import os
import numpy as np

# Point Python at the LIBERO package
LIBERO_PATH = os.path.join(os.path.dirname(__file__), "LIBERO")
sys.path.insert(0, LIBERO_PATH)

os.environ["MUJOCO_GL"] = "glfw"               # macOS headless via GLFW

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from libero.libero.envs import OffScreenRenderEnv
```

**Env creation + settle pattern** (lines 35-56):
```python
def build_env(camera_h=512, camera_w=512):
    env = OffScreenRenderEnv(
        bddl_file_name=BDDL,
        camera_names=["agentview", "frontview", "robot0_eye_in_hand"],
        camera_heights=camera_h,
        camera_widths=camera_w,
        has_renderer=False,
        has_offscreen_renderer=True,
    )
    return env
...
    obs = env.reset()
    # Let physics settle for a few steps
    for _ in range(10):
        obs, _, _, _ = env.step(np.zeros(7))
```
SOARM version adds `robots=["Soarm101"]` to the kwargs, plus the `max_contact_force` recipe from RESEARCH.md Code Examples, and `--check compile|reset` argparse modes per the Validation Architecture table.

**Frame-save + output conventions** (lines 68, 80-82):
```python
        img = obs[key][::-1]           # MuJoCo images are upside-down
...
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
```

---

### `LIBERO/notebooks/02-soarm-integration-check.ipynb` (Colab verification)

**Analog:** `LIBERO/notebooks/01-colab-env-setup.ipynb`

Structural pattern to copy:
- Cells 1-11 (Block A): install-cell sequence — reuse/import verbatim as the new notebook's Cell 1 bootstrap per RESEARCH.md notebook skeleton (MUJOCO_GL=**egl** on Colab, not glfw). Restart-runtime STOP markdown cell between install and verification blocks.
- Cell 14: EGL bootstrap (NVIDIA ICD JSON before setting `MUJOCO_GL`) — must be first post-restart cell.
- Cells 17-18: `~/.libero/config.yaml` bootstrap before any `import libero`, then sys.path setup.
- Cell 20 (ENV-02) is the canonical PASS/FAIL cell shape to replicate per criterion: comment header naming the requirement ID, `matplotlib.use("Agg")` before pyplot, the numba-stub compatibility shim (copy verbatim — robosuite imports break without it on Colab), disk-probing `LIBERO_PKG` resolution, hard `assert` for the automated part, explicit PASS/FAIL print, inline frame display.
- Final markdown summary table mapping requirement → status (Phase 1 Cell 24 pattern).

New PASS/FAIL cells per RESEARCH.md "Verification notebook skeleton": ENV-04 (compile + REGISTERED_ROBOTS), ENV-05 (ROBOT_CLASS_MAPPING + instantiate), SC-1 (contact force <10 N), ENV-07 (render both cameras), ENV-06 (3 BDDL tasks × N random steps).

**Required reading before writing this file:** `.planning/phases/01-colab-environment-setup/01-DEBUG-HISTORY.md` (7-invariant environment contract, `--no-deps` install contract).

## Shared Patterns

### MUJOCO_GL-before-import
**Source:** `explorations/create_scene.py` line 16 / Phase 1 notebook Cell 14
**Apply to:** `soarm_sanity.py` (glfw) and the notebook (egl). Env var must be set before ANY mujoco/robosuite/libero import.

### sys.path insertion for LIBERO
**Source:** `explorations/create_scene.py` lines 13-14
**Apply to:** any script outside the LIBERO package importing `libero.*`.
```python
LIBERO_PATH = os.path.join(os.path.dirname(__file__), "LIBERO")
sys.path.insert(0, LIBERO_PATH)
```

### Frame flip convention
**Source:** `explorations/create_scene.py` line 68 (`obs[key][::-1]  # MuJoCo images are upside-down`)
**Apply to:** sanity script and all notebook display cells (directly serves ENV-07's right-side-up criterion).

### Output logging convention
**Source:** `explorations/create_scene.py` line 82 — `print(f"Saved → {out}")` with arrow notation.
**Apply to:** all new scripts/notebook cells that write files.

### Fork-local asset path (replaces `xml_path_completion`)
**Source:** RESEARCH.md Patterns 2/3 (no in-repo precedent — Panda classes use robosuite's own asset tree)
**Apply to:** both `soarm.py` and `soarm_gripper.py`:
```python
ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets"))
```

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| — | — | — | Every file has a direct analog. The only pattern with no in-repo precedent is loading a robot XML from the LIBERO fork instead of robosuite's asset tree (covered by RESEARCH.md's ASSETS-constant pattern above). |

## Metadata

**Analog search scope:** `LIBERO/libero/libero/envs/robots/`, `explorations/`, `LIBERO/notebooks/`, robosuite 1.4.1 site-packages (`/opt/homebrew/Caskroom/miniconda/base/envs/libero/lib/python3.9/site-packages/robosuite/models/{grippers,assets}`)
**Files scanned:** 6 read in full + notebook cell survey
**Pattern extraction date:** 2026-07-11
