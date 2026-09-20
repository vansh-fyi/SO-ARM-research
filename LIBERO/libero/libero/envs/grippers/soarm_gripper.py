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
    """SO-ARM100/101 parallel-jaw gripper (roboninecom upgrade).

    Two symmetric prismatic jaws (``gripper_left`` / ``gripper_right``) driven by
    one actuator each, collapsed to a single 1-D gripper action exactly like
    robosuite's ``PandaGripper``. 36 mm travel per jaw (72 mm total change in
    aperture). Geometry is registered to the user's saved Coppelia assembly;
    its simplified collision pads span 6..78 mm, distinct from mesh-tip spacing.

    Both joint coordinates increase when opening: 0 is closed, 0.036 is open.
    The integrated action increases with external +1 (open), and decreases
    with external -1 (close). This preserves the collector's action contract
    while matching the positive-opening coordinate convention of Coppelia.

    Args:
        idn (int or str): Number or some other unique identification string
            for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(os.path.join(ASSETS, "grippers/soarm_gripper.xml"), idn=idn)

    def format_action(self, action):
        """Maps the 1-D gripper action into the two jaw position targets.

        External +1 opens and -1 closes. robosuite scales current_action
        from [-1, +1] to the actuator range [0, 0.036] metres: -1 is closed,
        +1 is open. Both symmetric jaws integrate in the same direction.

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + np.array([1.0, 1.0]) * self.speed * np.sign(action),
            -1.0,
            1.0,
        )
        return self.current_action

    @property
    def init_qpos(self):
        return np.array([0.036, 0.036])  # open endpoint; Coppelia q=0.044 on each jaw

    @property
    def speed(self):
        return 0.10  # jaw closes briskly within the FSM's GRASP_HOLD_STEPS budget

    @property
    def dof(self):
        return 1  # total action dim = OSC 6 + gripper 1 = 7 (Phase 3 contract)

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["left_jaw_collision"],
            "right_finger": ["right_jaw_collision"],
            "left_fingerpad": ["left_jaw_collision"],
            "right_fingerpad": ["right_jaw_collision"],
        }
