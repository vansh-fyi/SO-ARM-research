from .mounted_panda import MountedPanda
from .on_the_ground_panda import OnTheGroundPanda

# Defining the class auto-registers it in robosuite REGISTERED_ROBOTS (metaclass)
from .soarm import MountedSoarm101

from robosuite.robots.single_arm import SingleArm
from robosuite.robots import ROBOT_CLASS_MAPPING

from robosuite.models.grippers import GRIPPER_MAPPING
from ..grippers.soarm_gripper import SoarmGripper

ROBOT_CLASS_MAPPING.update(
    {
        "MountedPanda": SingleArm,
        "OnTheGroundPanda": SingleArm,
        "MountedSoarm101": SingleArm,
    }
)

GRIPPER_MAPPING["SoarmGripper"] = SoarmGripper  # gripper_factory asserts membership
