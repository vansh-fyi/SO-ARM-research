"""Interactive eye_in_hand camera tuner.

Opens a live MuJoCo render of the wrist camera with sliders for position
(x,y,z, in robot0_right_hand-local coords -- same frame as robot.xml's
<camera pos=.../> attribute) and orientation (roll/pitch/yaw in degrees,
applied in that local frame). Move sliders, watch the render update live.
A pose selector lets you check the view at reset / early / mid / end of a
real recorded demo trajectory, since this arm's wrist barely reorients
across an episode -- one static check isn't representative.

When you're happy, read the printed pos/quat values in the terminal and
paste them into LIBERO/libero/libero/assets/robots/soarm101/robot.xml's
eye_in_hand <camera> element (pos="..." quat="w x y z").

Requires the `libero` conda env (MuJoCo/robosuite/scipy) -- NOT diagnostics/
.venv, which is for the hardware servo scripts in this same directory.

Run: conda activate libero && python diagnostics/tune_eye_in_hand.py
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LIBERO_LIBERO = os.path.join(_REPO_ROOT, "LIBERO", "libero")
for p in (_REPO_ROOT, _LIBERO_LIBERO):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

from libero.envs import OffScreenRenderEnv

BDDL_PATH = os.path.join(
    _LIBERO_LIBERO, "libero", "bddl_files", "libero_goal",
    "put_the_cream_cheese_in_the_bowl.bddl",
)
DEMO_H5 = os.path.join(
    _LIBERO_LIBERO, "datasets", "soarm_spatial", "teleop",
    "cream_cheese_teleop_demo_v2.hdf5",
)

CAM_H = CAM_W = 256

env = OffScreenRenderEnv(
    bddl_file_name=BDDL_PATH,
    robots=["Soarm101"],
    camera_names=["agentview", "robot0_eye_in_hand"],
    camera_heights=CAM_H,
    camera_widths=CAM_W,
    has_renderer=False,
    has_offscreen_renderer=True,
)
env.reset()
sim = env.env.sim
cam_id = sim.model.camera_name2id("robot0_eye_in_hand")

# Load a few real recorded states for pose-dependent preview.
POSES = {"reset (current sim state)": None}
try:
    import h5py
    with h5py.File(DEMO_H5, "r") as f:
        states = f["data/demo_1"]["states"][:]
    n = len(states)
    POSES["early (real demo)"] = states[n // 6]
    POSES["mid (real demo)"] = states[n // 2]
    POSES["end (real demo)"] = states[-1]
except Exception as e:
    print(f"Could not load demo states ({e}); only reset pose available.")

current_pose_key = "mid (real demo)" if "mid (real demo)" in POSES else "reset (current sim state)"

# Starting values: resume from the last session's chosen pos/quat (was too
# close to the slider edges on x/z before -- widened range below).
start_pos = np.array([0.0400, 0.0007, -0.0800])
start_quat_wxyz = np.array([0.295057, 0.639464, 0.644636, 0.297443])
start_rot = Rotation.from_quat([*start_quat_wxyz[1:], start_quat_wxyz[0]])  # xyzw for scipy
start_euler = start_rot.as_euler("xyz", degrees=True)  # roll, pitch, yaw about local x,y,z


def apply_pose(pose_key):
    state = POSES[pose_key]
    if state is not None:
        sim.set_state_from_flattened(state)
    sim.forward()


def set_camera(x, y, z, roll, pitch, yaw):
    sim.model.cam_pos[cam_id] = [x, y, z]
    rot = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=True)
    xq, yq, zq, wq = rot.as_quat()
    sim.model.cam_quat[cam_id] = [wq, xq, yq, zq]
    sim.forward()


apply_pose(current_pose_key)
set_camera(*start_pos, *start_euler)

fig, (ax_img, ax_ctx) = plt.subplots(1, 2, figsize=(9, 5))
plt.subplots_adjust(left=0.08, bottom=0.42, right=0.98, top=0.95, wspace=0.05)

img = sim.render(camera_name="robot0_eye_in_hand", width=CAM_W, height=CAM_H, depth=False)[::-1]
im_artist = ax_img.imshow(img)
ax_img.set_title("eye_in_hand (live)")
ax_img.axis("off")

img_ctx = sim.render(camera_name="agentview", width=CAM_W, height=CAM_H, depth=False)[::-1]
im_ctx_artist = ax_ctx.imshow(img_ctx)
ax_ctx.set_title("agentview (context)")
ax_ctx.axis("off")

slider_axes = {}
sliders = {}
specs = [
    ("x", start_pos[0] - 0.3, start_pos[0] + 0.3, start_pos[0]),
    ("y", start_pos[1] - 0.3, start_pos[1] + 0.3, start_pos[1]),
    ("z", start_pos[2] - 0.3, start_pos[2] + 0.3, start_pos[2]),
    ("roll", -180, 180, start_euler[0]),
    ("pitch", -180, 180, start_euler[1]),
    ("yaw", -180, 180, start_euler[2]),
]
for i, (name, lo, hi, init) in enumerate(specs):
    ax = plt.axes([0.15, 0.32 - i * 0.045, 0.65, 0.03])
    slider_axes[name] = ax
    sliders[name] = Slider(ax, name, lo, hi, valinit=init)

radio_ax = plt.axes([0.83, 0.05, 0.15, 0.25])
radio = RadioButtons(radio_ax, list(POSES.keys()), active=list(POSES.keys()).index(current_pose_key))

status_text = fig.text(0.08, 0.03, "", fontsize=8, family="monospace")


def redraw(_=None):
    apply_pose(radio.value_selected)
    vals = {name: s.val for name, s in sliders.items()}
    set_camera(vals["x"], vals["y"], vals["z"], vals["roll"], vals["pitch"], vals["yaw"])
    new_img = sim.render(camera_name="robot0_eye_in_hand", width=CAM_W, height=CAM_H, depth=False)[::-1]
    im_artist.set_data(new_img)
    new_ctx = sim.render(camera_name="agentview", width=CAM_W, height=CAM_H, depth=False)[::-1]
    im_ctx_artist.set_data(new_ctx)
    q = sim.model.cam_quat[cam_id]
    status_text.set_text(
        f'pos="{vals["x"]:.4f} {vals["y"]:.4f} {vals["z"]:.4f}"   '
        f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"'
    )
    fig.canvas.draw_idle()


for s in sliders.values():
    s.on_changed(redraw)
radio.on_clicked(redraw)

redraw()
print("Move sliders to adjust the wrist camera. Final pos/quat values shown live under the plot.")
print("Close the window when done -- final values will be printed here too.")
plt.show()

vals = {name: s.val for name, s in sliders.items()}
q = sim.model.cam_quat[cam_id]
print("\n=== FINAL VALUES ===")
print(f'pos="{vals["x"]:.4f} {vals["y"]:.4f} {vals["z"]:.4f}"')
print(f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"')
print("Paste these into robot.xml's eye_in_hand <camera> element.")

env.close()
