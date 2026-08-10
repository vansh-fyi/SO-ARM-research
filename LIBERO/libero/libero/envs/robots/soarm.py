"""SOARM SO101 robot model for robosuite 1.4 / LIBERO.

Mirrors on_the_ground_panda.py's class shape with SOARM values. The arm MJCF
lives in the LIBERO fork's asset tree (not robosuite site-packages), so the
XML path is built from a fork-local ASSETS constant instead of robosuite's
path-completion helper.
"""
import os

import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel

# Fork-local asset root (LIBERO/libero/libero/assets). Do NOT use robosuite's
# path-completion helper — it resolves inside robosuite site-packages.
ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets"))


class MountedSoarm101(ManipulatorModel):
    """SO-101 5-DOF tabletop arm (TheRobotStudio SO-ARM100, new calibration).

    Named "Mounted..." because Libero_Tabletop_Manipulation prepends "Mounted"
    to the robots kwarg (robots=["Soarm101"] looks up "MountedSoarm101") —
    despite the name, default_mount is None (D-04): the SO101 has no pedestal
    hardware and its base sits directly on the tabletop.

    Args:
        idn (int or str): Number or some other unique identification string
            for this robot instance
    """

    def __init__(self, idn=0):
        super().__init__(os.path.join(ASSETS, "robots/soarm101/robot.xml"), idn=idn)

        # 5 arm joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
        # wrist_roll. Starting damping = STS3215 class value; tune per D-11.
        self.set_joint_attribute(
            attrib="damping", values=np.array((0.6, 0.6, 0.6, 0.6, 0.6))
        )

    @property
    def default_mount(self):
        return None  # D-04: no pedestal hardware

    @property
    def default_gripper(self):
        return "SoarmGripper"

    @property
    def default_controller_config(self):
        return "default_panda"  # fallback only; LIBERO always passes OSC_POSE

    @property
    def init_qpos(self):
        return np.zeros(5)  # D-07: new-calib zero IS the documented home pose

    @property
    def base_xpos_offset(self):
        # Table surface sits at z=0.90 (Libero_Tabletop_Manipulation
        # workspace_offset). SO101 max reach is 0.479 m, so the base must sit
        # ON the tabletop near its -x edge (Pitfall 6) — never on the floor
        # like Panda. Starting values; tuned in plan 02-03 per D-11.
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.38, 0, 0.90),
            "kitchen_table": lambda table_length: (-0.38, 0, 0.90),
            "study_table": lambda table_length: (-0.38, 0, 0.90),
            "coffee_table": lambda table_length: (-0.30, 0, 0.41),
            "living_room_table": lambda table_length: (-0.30, 0, 0.42),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 0.30))  # arm is ~0.3 m tall at home

    @property
    def _horizontal_radius(self):
        return 0.25  # ~half of the 0.479 m max reach envelope

    @property
    def arm_type(self):
        return "single"
