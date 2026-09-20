"""Inspect the captured reference in MuJoCo's arm and wrist frames."""
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def transform(values):
    matrix = np.eye(4)
    matrix[:3] = np.asarray(values).reshape(3, 4)
    return matrix


def load_reference():
    import cbor2
    with (ROOT / "diagnostics/outputs/coppelia_reference.cbor").open("rb") as f:
        snapshot = cbor2.load(f)
    for item in snapshot["objects"]:
        if isinstance(item["name"], bytes):
            item["name"] = item["name"].decode()
    return {item["name"]: item for item in snapshot["objects"]}


def body_transform(data, index):
    matrix = np.eye(4)
    matrix[:3, :3] = data.xmat[index].reshape(3, 3)
    matrix[:3, 3] = data.xpos[index]
    return matrix


def reference_alignment(reference):
    model = mujoco.MjModel.from_xml_path(str(ROOT / "LIBERO/libero/libero/assets/robots/soarm101/robot.xml"))
    data = mujoco.MjData(model)
    for j in range(model.njnt):
        data.qpos[model.jnt_qposadr[j]] = reference[model.joint(j).name]["position"]
    mujoco.mj_forward(model, data)
    # Root shape coordinates are not the MJCF base coordinates. Match the first
    # joint's base frame, then independently check the remaining arm chain.
    first = model.joint("shoulder_pan").id
    source = transform(reference["shoulder_pan"]["matrix_root"])
    moving = np.eye(4)
    moving[:3, :3] = Rotation.from_euler("z", reference["shoulder_pan"]["position"]).as_matrix()
    align = body_transform(data, model.jnt_bodyid[first]) @ np.linalg.inv(source @ moving)
    return model, data, align


def main():
    np.set_printoptions(precision=8, suppress=True)
    ref = load_reference()
    model, data, align = reference_alignment(ref)
    hand = body_transform(data, model.body("right_hand").id)
    for j in range(model.njnt):
        name = model.joint(j).name
        source = transform(ref[name]["matrix_root"])
        moving = np.eye(4)
        moving[:3, :3] = Rotation.from_euler("z", ref[name]["position"]).as_matrix()
        actual = body_transform(data, model.jnt_bodyid[j])
        expected = align @ source @ moving
        print(name, "position error mm", 1000*np.linalg.norm(actual[:3, 3]-expected[:3, 3]),
              "orientation error deg", Rotation.from_matrix(actual[:3,:3].T @ expected[:3,:3]).magnitude()*180/np.pi)
    for name, item in ref.items():
        if "gripper" not in name and name not in ("Gear", "sts3215_03a_v1"):
            continue
        local = np.linalg.inv(hand) @ align @ transform(item["matrix_root"])
        print(name, "hand frame", local)
        if "vertices" in item:
            v = np.array(item["vertices"]).reshape(-1,3) @ local[:3,:3].T + local[:3,3]
            print("bounds", np.array([v.min(0),v.max(0)]))


if __name__ == "__main__":
    main()
