"""Verify compiled MuJoCo vertices against Coppelia, and render/view the full arm.

conda run -n libero python diagnostics/verify_coppelia_alignment.py --render
conda run -n libero mjpython diagnostics/verify_coppelia_alignment.py --view

Normal runs use the checked-in measured reference. --capture reads the full
local CBOR snapshot; --write-reference records a compact independent fixture.
"""
import argparse
import copy
import hashlib
import json
import os
from itertools import product
from pathlib import Path
import time
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "glfw")
import mujoco
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from inspect_coppelia_reference import ROOT, load_reference, reference_alignment, transform

ASSETS = ROOT / "LIBERO/libero/libero/assets"
FIXTURE = ROOT / "diagnostics/reference/soarm_coppelia.json"
OUTPUT = ROOT / "diagnostics/outputs"
SHAPES = {
    "main_frame_visual": "parallel_gripper_base_visual",
    "left_jaw_visual": "parallel_gripper_left_jaw_visual",
    "right_jaw_visual": "parallel_gripper_right_jaw_visual",
    "right_hand_servo_visual": "sts3215_03a_v1",
    "left_jaw_collision": "parallel_gripper_left_jaw_respondable",
    "right_jaw_collision": "parallel_gripper_right_jaw_respondable",
}
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
ARM_SHAPES = {
    "base": "base_link_visual", "shoulder": "shoulder_link_visual",
    "upper_arm": "upper_arm_link_visual", "lower_arm": "lower_arm_link_visual",
    "wrist": "wrist_link_visual",
}


def scene_xml():
    robot_path = ASSETS / "robots/soarm101/robot.xml"
    gripper_path = ASSETS / "grippers/soarm_gripper.xml"
    robot, gripper = ET.parse(robot_path).getroot(), ET.parse(gripper_path).getroot()
    for root, directory in ((robot, robot_path.parent), (gripper, gripper_path.parent)):
        for mesh in root.findall("./asset/mesh"):
            mesh.set("file", str(directory / mesh.get("file")))
    for name in ("asset", "actuator", "equality", "sensor"):
        source = gripper.find(name)
        if source is not None:
            target = robot.find(name)
            if target is None:
                target = ET.SubElement(robot, name)
            target.extend(copy.deepcopy(list(source)))
    ET.SubElement(robot.find("asset"), "texture", name="inspection_sky", type="skybox",
                  builtin="gradient", rgb1=".45 .5 .56", rgb2=".7 .73 .77", width="512", height="3072")
    robot.find(".//body[@name='right_hand']").append(copy.deepcopy(gripper.find("./worldbody/body")))
    ET.SubElement(robot, "visual")
    ET.SubElement(robot.find("visual"), "global", offwidth="960", offheight="720")
    world = robot.find("worldbody")
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1", diffuse="0.9 0.9 0.9")
    ET.SubElement(world, "geom", name="floor", type="plane", size="1 1 .01", pos="0 0 -0.0024", rgba=".8 .81 .82 1", contype="0", conaffinity="0", group="1")
    return ET.tostring(robot, encoding="unicode")


def source_matrices(objects, positions):
    by_handle = {o["handle"]: o for o in objects.values()}
    result = {}

    def visit(item):
        name = item["name"]
        if name in result:
            return result[name]
        parent = by_handle.get(item["parent"])
        if parent is None:
            matrix = np.eye(4)
        else:
            matrix = visit(parent).copy()
            if "position" in parent:
                motion = np.eye(4)
                q = positions.get(parent["name"], parent["position"])
                if parent["name"].startswith("parallel_gripper_"):
                    motion[2, 3] = q
                else:
                    motion[:3,:3] = Rotation.from_euler("z", q).as_matrix()
                matrix = matrix @ motion
            matrix = matrix @ transform(item["matrix_parent"])
        result[name] = matrix
        return matrix

    for item in objects.values():
        visit(item)
    return result


def capture_reference(write=False):
    objects = load_reference()
    _, _, alignment = reference_alignment(objects)
    # Source data is recorded, never reconstructed from the edited MJCF.
    needed = set(SHAPES.values()) | set(ARM_SHAPES.values())
    compact = copy.deepcopy(objects)
    for name, obj in compact.items():
        obj.pop("indices", None)
        vertices = obj.pop("vertices", None)
        if name in needed:
            vertices = np.asarray(vertices).reshape(-1, 3)
            if write:
                selected = np.unique(np.r_[np.linspace(0,len(vertices)-1,min(256,len(vertices))).astype(int), vertices.argmin(0), vertices.argmax(0)])
                vertices = vertices[selected]
            obj["vertices"] = vertices.tolist()
    result = dict(objects=compact, alignment=alignment.tolist(),
                  source="SO-ARM 101-parallel.ttm, captured 2026-09-20 after wrist_roll reparenting",
                  snapshot_sha256=hashlib.sha256((OUTPUT / "coppelia_reference.cbor").read_bytes()).hexdigest())
    if write:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(result, indent=2) + "\n")
    return result


def geom_vertices(model, data, name):
    geom = model.geom(name).id if isinstance(name, str) else name
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
        vertices = np.array(list(product((-1, 1), repeat=3))) * model.geom_size[geom]
    else:
        mesh = model.geom_dataid[geom]
        vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
    return vertices @ data.geom_xmat[geom].reshape(3,3).T + data.geom_xpos[geom]


def set_pose(model, data, positions, jaw):
    for name in ARM_JOINTS:
        data.qpos[model.joint(name).qposadr[0]] = positions[name]
    for name in ("gripper_left", "gripper_right"):
        data.qpos[model.joint(name).qposadr[0]] = jaw
    mujoco.mj_forward(model, data)


def verify(model, data, reference):
    objects, align = reference["objects"], np.array(reference["alignment"])
    baseline = {name: objects[name]["position"] for name in ARM_JOINTS}
    results = []
    for pose_name, offset in (("reference", np.zeros(5)), ("wrist_rotated", [0,0,0,0,.8]), ("arm_moved", [.2,-.3,.4,-.5,-.4])):
        positions = {name: baseline[name]+offset[i] for i, name in enumerate(ARM_JOINTS)}
        for jaw in (0, .012, .018, .036):
            set_pose(model, data, positions, jaw)
            matrices = source_matrices(objects, dict(positions, parallel_gripper_left=.008+jaw, parallel_gripper_right=.008+jaw))
            for geom, source in SHAPES.items():
                pose = align @ matrices[source]
                points = np.array(objects[source]["vertices"]).reshape(-1,3) @ pose[:3,:3].T + pose[:3,3]
                actual = geom_vertices(model, data, geom)
                error = float(cKDTree(actual).query(points)[0].max())
                results.append(dict(pose=pose_name, jaw=jaw, geom=geom, max_error_mm=error*1000))
                if error > 5e-6:
                    raise AssertionError(f"{pose_name} jaw={jaw} {geom}: {error*1000:.6f} mm error")
            for body, source in ARM_SHAPES.items():
                pose = align @ matrices[source]
                points = np.array(objects[source]["vertices"]).reshape(-1,3) @ pose[:3,:3].T + pose[:3,3]
                actual = np.concatenate([geom_vertices(model, data, i) for i in range(model.ngeom)
                                         if model.geom_bodyid[i] == model.body(body).id and model.geom_group[i] == 1])
                error = float(cKDTree(actual).query(points)[0].max())
                results.append(dict(pose=pose_name, jaw=jaw, geom=body, max_error_mm=error*1000))
                if error > 5e-6:
                    raise AssertionError(f"{pose_name} {body}: {error*1000:.6f} mm error")
            midpoint = (data.geom_xpos[model.geom("left_jaw_collision").id]+data.geom_xpos[model.geom("right_jaw_collision").id])/2
            if np.linalg.norm(midpoint-data.site_xpos[model.site("grip_site").id]) > 1e-8:
                raise AssertionError("grip_site does not follow the contact midpoint")
    print(f"PASS: {len(results)} geometry comparisons; maximum {max(r['max_error_mm'] for r in results):.6f} mm")
    return baseline, results


def render(model, data, baseline):
    from PIL import Image, ImageDraw
    renderer = mujoco.Renderer(model, height=720, width=960)
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    options.geomgroup[0] = 0
    options.sitegroup[:] = 0
    model.site_rgba[:,3] = 0
    images = []
    try:
        for label, jaw in (("reference", .012), ("closed", 0), ("open", .036)):
            set_pose(model, data, baseline, jaw)
            for azimuth in (60, 150, 240, 330):
                camera = mujoco.MjvCamera()
                mujoco.mjv_defaultFreeCamera(model, camera)
                camera.lookat[:] = [.125, 0, .13]
                camera.distance, camera.azimuth, camera.elevation = .55, azimuth, -15
                renderer.update_scene(data, camera=camera, scene_option=options)
                picture = Image.fromarray(renderer.render())
                ImageDraw.Draw(picture).text((15,15), f"{label} / azimuth {azimuth}", fill="black")
                picture.save(OUTPUT/f"soarm_{label}_{azimuth}.png")
                images.append(picture.resize((480,360)))
        sheet = Image.new("RGB", (1920,1080), "white")
        for index, picture in enumerate(images):
            sheet.paste(picture, ((index%4)*480,(index//4)*360))
        sheet.save(OUTPUT/"soarm_coppelia_comparison.png")
        print("Rendered", OUTPUT/"soarm_coppelia_comparison.png")
    finally:
        # MuJoCo 2.3.7 predates Renderer.close().
        if hasattr(renderer, "close"):
            renderer.close()
        # On 2.3.7 leave cleanup to the context destructors. Explicitly freeing
        # private contexts here can crash macOS GLFW during process teardown.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--write-reference", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()
    reference = capture_reference(args.write_reference) if args.capture or args.write_reference else json.loads(FIXTURE.read_text())
    xml = scene_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    baseline, results = verify(model, data, reference)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/"soarm_alignment_report.json").write_text(json.dumps(results, indent=2)+"\n")
    (OUTPUT/"soarm_reference_scene.xml").write_text(xml)
    if args.render:
        render(model, data, baseline)
    if args.view:
        from mujoco import viewer as mj_viewer
        set_pose(model, data, baseline, .012)
        def keypress(key):
            if key in (ord("O"), ord("C")):
                jaw = .036 if key == ord("O") else 0
                set_pose(model, data, baseline, jaw)
        with mj_viewer.launch_passive(model, data, key_callback=keypress) as viewer:
            viewer.cam.lookat[:] = [.125,0,.13]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = .55, 60, -15
            model.site_rgba[:,3] = 0
            viewer.opt.geomgroup[0] = 0
            viewer.opt.sitegroup[:] = 0
            print("O = open, C = close. Inspection is kinematic; motors/hardware are not driven.")
            while viewer.is_running():
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(.02)


if __name__ == "__main__":
    main()
