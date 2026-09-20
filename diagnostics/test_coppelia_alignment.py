"""Regression checks against captured Coppelia geometry, not XML constants."""
import json
import importlib.util
from pathlib import Path
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
    data.qpos[:] = [.035, .005]
    data.ctrl[model.actuator("gripper_left").id] = .02
    for _ in range(1500):
        mujoco.mj_step(model, data)
    assert np.max(np.abs(data.qpos - .02)) < .002
    assert abs(data.qpos[0] - data.qpos[1]) < 1e-4


def test_positive_joint_position_and_external_action_open_jaws():
    path = Path(__file__).resolve().parents[1] / "LIBERO/libero/libero/envs/grippers/soarm_gripper.py"
    spec = importlib.util.spec_from_file_location("positive_open_gripper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gripper = module.SoarmGripper()
    np.testing.assert_allclose(gripper.init_qpos, [.036, .036])
    model = mujoco.MjModel.from_xml_path(str(ASSETS / "grippers/soarm_gripper.xml"))
    model.opt.gravity[:] = 0
    data = mujoco.MjData(model)
    gaps = []
    for action, endpoint in ((1, .036), (-1, 0)):
        for _ in range(25):
            normalized = gripper.format_action(np.array([action]))
        limits = model.actuator_ctrlrange
        targets = limits.mean(axis=1) + normalized * (limits[:, 1] - limits[:, 0]) / 2
        np.testing.assert_allclose(targets, endpoint, atol=1e-12)
        data.ctrl[:] = targets
        for _ in range(1500):
            mujoco.mj_step(model, data)
        np.testing.assert_allclose(data.qpos, endpoint, atol=.002)
        mujoco.mj_forward(model, data)
        gaps.append(np.linalg.norm(data.geom("left_jaw_collision").xpos - data.geom("right_jaw_collision").xpos))
    assert gaps[0] - gaps[1] > .068
