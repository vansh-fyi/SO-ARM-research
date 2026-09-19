"""
Render the SOARM parallel-jaw gripper standalone (no arm attachment) in open
and closed jaw states, for iteratively dialing in the D-04 clamp visual mesh
fix (Phase 10 CONTEXT.md) -- the clamp STL's gear-tooth detail must sit flush
against `main_frame_visual`'s housing in an L-shape, not poke outside it.

Loads `soarm_gripper.xml` directly via mujoco.MjModel.from_xml_path -- every
relevant geom (main_frame_visual/left_jaw_visual/right_jaw_visual) is already
defined in this file's own `right_gripper`-relative body-local frame, so no
arm attachment is needed to inspect the mount-to-jaw junction.

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

# 3/4 side angle framing the main-frame-to-jaw junction. Tuned away from the
# plan's initial azimuth=110/elevation=-15/distance=0.25 guess (which framed
# the junction at a grazing angle that made a correctly-flush mesh look
# disconnected) to a tighter, more head-on angle that reads the frame<->jaw
# connection unambiguously -- confirmed against a multi-azimuth contact sheet
# during D-04 dialing.
CAM_LOOKAT = [-0.05, -0.02, 0]
CAM_DISTANCE = 0.2
CAM_AZIMUTH = 30
CAM_ELEVATION = -25

# (jaw_value, output_filename_suffix) -- open (0.042, jaws fully apart) and
# closed (0.0, jaws together).
JAW_STATES = [(0.042, "open"), (0.0, "closed")]


def main():
    if not GRIPPER_XML.exists():
        raise FileNotFoundError(f"Gripper MJCF not found: {GRIPPER_XML}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(GRIPPER_XML))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=CAM_H, width=CAM_W)

    camera = mujoco.MjvCamera()
    camera.lookat[:] = CAM_LOOKAT
    camera.distance = CAM_DISTANCE
    camera.azimuth = CAM_AZIMUTH
    camera.elevation = CAM_ELEVATION

    for jaw_value, suffix in JAW_STATES:
        data.qpos[:] = [jaw_value, jaw_value]
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()

        out_path = OUT_DIR / f"gripper_clamp_{suffix}.png"
        Image.fromarray(np.asarray(pixels)).save(out_path)
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
