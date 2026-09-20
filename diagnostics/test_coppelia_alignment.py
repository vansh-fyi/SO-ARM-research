"""Regression checks against captured Coppelia geometry, not XML constants."""
import json
import mujoco
import numpy as np
from verify_coppelia_alignment import ASSETS, FIXTURE, scene_xml, verify


def test_arm_and_gripper_match_measured_reference_through_motion():
    model = mujoco.MjModel.from_xml_string(scene_xml())
    verify(model, mujoco.MjData(model), json.loads(FIXTURE.read_text()))


def test_single_actuator_still_drags_both_jaws():
    model = mujoco.MjModel.from_xml_path(str(ASSETS / "grippers/soarm_gripper.xml"))
    model.opt.gravity[:] = 0
    right = model.actuator("gripper_right").id
    model.actuator_gainprm[right, :] = 0
    model.actuator_biasprm[right, :] = 0
    data = mujoco.MjData(model)
    data.qpos[:] = [-.035, -.005]
    data.ctrl[model.actuator("gripper_left").id] = -.02
    for _ in range(1500):
        mujoco.mj_step(model, data)
    assert np.max(np.abs(data.qpos + .02)) < .002
    assert abs(data.qpos[0] - data.qpos[1]) < 1e-4
