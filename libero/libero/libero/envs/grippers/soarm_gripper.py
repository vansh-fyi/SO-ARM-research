"""SOARM SO101 gripper model for robosuite 1.4 / LIBERO.

Mirrors robosuite's PandaGripper integrated-action pattern, collapsed to the
SO101's single jaw actuator. The gripper MJCF lives in the LIBERO fork's
asset tree, so the XML path is built from a fork-local ASSETS constant
instead of robosuite's path-completion helper.
"""
import os

import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel

# Fork-local asset root (LIBERO/libero/libero/assets). Do NOT use robosuite's
# path-completion helper — it resolves inside robosuite site-packages.
ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets"))


class SoarmGripper(GripperModel):
    """SO101 1-DOF jaw gripper (fixed jaw + moving jaw on one hinge joint).

    Args:
        idn (int or str): Number or some other unique identification string
            for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(os.path.join(ASSETS, "grippers/soarm_gripper.xml"), idn=idn)

    def format_action(self, action):
        """Maps continuous action into an integrated jaw position target.

        -1 => open, 1 => closed. Single-actuator collapse of PandaGripper's
        two-finger direction array: only np.sign(action) scaling remains.

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def init_qpos(self):
        return np.array([0.8])  # jaw open (range -0.17..1.75 rad); tune in 02-03

    @property
    def speed(self):
        return 0.10  # rad-scale jaw moves faster than Panda's 0.01 m fingers

    @property
    def dof(self):
        return 1  # total action dim = OSC 6 + gripper 1 = 7 (Phase 3 contract)

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["fixed_jaw_collision"],
            "right_finger": ["moving_jaw_collision"],
            "left_fingerpad": ["fixed_jaw_collision"],
            "right_fingerpad": ["moving_jaw_collision"],
        }
