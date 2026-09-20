"""
Render the Coppelia-aligned SOARM parallel gripper in open and closed states.

Loads `soarm_gripper.xml` directly via mujoco.MjModel.from_xml_path -- every
relevant geom follows the measured parallel_gripper_mechanism mounting frame.
For the full arm and numerical checks, use verify_coppelia_alignment.py.

Usage:
    conda activate libero && python diagnostics/render_gripper_clamp.py
"""
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
GRIPPER_XML = ROOT / "LIBERO" / "libero" / "libero" / "assets" / "grippers" / "soarm_gripper.xml"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

CAM_H, CAM_W = 480, 640

# The corrected fingers extend approximately along right_hand -Z.
CAM_LOOKAT = [0, 0, -0.075]
CAM_DISTANCE = 0.3
CAM_AZIMUTH = 30
CAM_ELEVATION = -25

# Head-on framing (both jaws visible side-by-side, teeth facing camera) for
# direct left-vs-right tooth-orientation comparison. Zoomed out enough to
# keep the full paddle-jaw tips in frame (0.18 cropped them at the edges).
CAM_LOOKAT_FRONTAL = [0, 0, -0.075]
CAM_DISTANCE_FRONTAL = 0.3
CAM_AZIMUTH_FRONTAL = 0
CAM_ELEVATION_FRONTAL = 0

# (jaw_value, output_filename_suffix) -- open (-0.036, jaws fully apart) and
# closed (0.0, jaws together). Post-TWIN-07-fix range is [-0.036, 0] (was
# [0, 0.036] before the gripper axis/range polarity flip); qpos=-0.036 is
# now the open end, qpos=0 is still closed (see soarm_gripper.xml).
JAW_STATES = [(-0.036, "open"), (0.0, "closed")]


def main():
    if not GRIPPER_XML.exists():
        raise FileNotFoundError(f"Gripper MJCF not found: {GRIPPER_XML}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(GRIPPER_XML))
    model.site_rgba[:, 3] = 0
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=CAM_H, width=CAM_W)
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    options.geomgroup[0] = 0
    options.sitegroup[:] = 0

    camera = mujoco.MjvCamera()
    camera.lookat[:] = CAM_LOOKAT
    camera.distance = CAM_DISTANCE
    camera.azimuth = CAM_AZIMUTH
    camera.elevation = CAM_ELEVATION

    frontal_camera = mujoco.MjvCamera()
    frontal_camera.lookat[:] = CAM_LOOKAT_FRONTAL
    frontal_camera.distance = CAM_DISTANCE_FRONTAL
    frontal_camera.azimuth = CAM_AZIMUTH_FRONTAL
    frontal_camera.elevation = CAM_ELEVATION_FRONTAL

    for jaw_value, suffix in JAW_STATES:
        data.qpos[:] = [jaw_value, jaw_value]
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera=camera, scene_option=options)
        pixels = renderer.render()
        out_path = OUT_DIR / f"gripper_clamp_{suffix}.png"
        Image.fromarray(np.asarray(pixels)).save(out_path)
        print(f"Saved -> {out_path}")

        renderer.update_scene(data, camera=frontal_camera, scene_option=options)
        pixels_frontal = renderer.render()
        out_path_frontal = OUT_DIR / f"gripper_clamp_{suffix}_frontal.png"
        Image.fromarray(np.asarray(pixels_frontal)).save(out_path_frontal)
        print(f"Saved -> {out_path_frontal}")


if __name__ == "__main__":
    main()
