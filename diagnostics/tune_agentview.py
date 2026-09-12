"""Interactive agentview camera tuner.

agentview is attached to `world` (a fixed, static camera -- not on the arm),
so x/y/z and roll/pitch/yaw are plain world coordinates, no arm-pose replay
needed (unlike tune_eye_in_hand.py).

Opens a live MuJoCo render next to a static real reference photo (one lens
of the connected stereo camera) so you can drag sliders until the sim
framing matches. Sliders: x/y/z (world pos, meters), roll/pitch/yaw
(degrees), fovy (degrees).

When happy, read the printed pos/quat/fovy in the terminal and paste into
LIBERO/libero/libero/envs/problems/libero_tabletop_manipulation.py's
_setup_camera() agentview call (and mirror into bddl_base_domain.py per
07-01's existing defensive-mirror convention).

Requires the `libero` conda env (MuJoCo/robosuite/scipy) -- NOT diagnostics/
.venv, which is for the hardware servo scripts in this same directory.

To refresh the real reference photo with a new capture from the connected
camera: `ffmpeg -f avfoundation -video_size 1280x480 -framerate 30 -i "<device>"
-vframes 1 out.png` (device index varies -- run `ffmpeg -f avfoundation
-list_devices true -i ""` to find it), then crop the left half (this is a
side-by-side stereo pair) and overwrite REAL_PHOTO below.

Run: conda activate libero && python diagnostics/tune_agentview.py
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
from PIL import Image

from libero.envs import OffScreenRenderEnv

BDDL_PATH = os.path.join(
    _LIBERO_LIBERO, "libero", "bddl_files", "libero_goal",
    "put_the_cream_cheese_in_the_bowl.bddl",
)
DEMO_H5 = os.path.join(
    _LIBERO_LIBERO, "datasets", "soarm_spatial", "teleop",
    "cream_cheese_teleop_demo_v2.hdf5",
)
REAL_PHOTO = os.path.join(
    os.path.dirname(__file__), "outputs", "agentview_real_reference.png"
)

CAM_H = CAM_W = 384

env = OffScreenRenderEnv(
    bddl_file_name=BDDL_PATH,
    robots=["Soarm101"],
    camera_names=["agentview"],
    camera_heights=CAM_H,
    camera_widths=CAM_W,
    has_renderer=False,
    has_offscreen_renderer=True,
)
env.reset()
sim = env.env.sim
cam_id = sim.model.camera_name2id("agentview")

# Load real recorded states so the scene (arm/object) can be checked at
# different stages of a task, not just the frozen reset layout.
STAGES = {"reset (spawn layout)": None}
try:
    import h5py
    with h5py.File(DEMO_H5, "r") as f:
        states = f["data/demo_1"]["states"][:]
    n = len(states)
    STAGES["early (real demo)"] = states[n // 6]
    STAGES["mid (real demo)"] = states[n // 2]
    STAGES["end (real demo)"] = states[-1]
except Exception as e:
    print(f"Could not load demo states ({e}); only reset stage available.")

current_stage_key = "mid (real demo)" if "mid (real demo)" in STAGES else "reset (spawn layout)"


def apply_stage(stage_key):
    state = STAGES[stage_key]
    if state is not None:
        sim.set_state_from_flattened(state)
    sim.forward()


apply_stage(current_stage_key)

start_pos = sim.model.cam_pos[cam_id].copy()
start_quat_wxyz = sim.model.cam_quat[cam_id].copy()
start_rot = Rotation.from_quat([*start_quat_wxyz[1:], start_quat_wxyz[0]])
start_euler = start_rot.as_euler("xyz", degrees=True)
start_fovy = float(sim.model.cam_fovy[cam_id])


def set_camera(x, y, z, roll, pitch, yaw, fovy):
    sim.model.cam_pos[cam_id] = [x, y, z]
    rot = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=True)
    xq, yq, zq, wq = rot.as_quat()
    sim.model.cam_quat[cam_id] = [wq, xq, yq, zq]
    sim.model.cam_fovy[cam_id] = fovy
    sim.forward()


set_camera(*start_pos, *start_euler, start_fovy)

fig, (ax_sim, ax_real) = plt.subplots(1, 2, figsize=(9, 5))
plt.subplots_adjust(left=0.08, bottom=0.42, right=0.98, top=0.92, wspace=0.05)

img = sim.render(camera_name="agentview", width=CAM_W, height=CAM_H, depth=False)[::-1]
im_artist = ax_sim.imshow(img)
ax_sim.set_title("agentview (live sim)")
ax_sim.axis("off")

if os.path.exists(REAL_PHOTO):
    real_img = np.array(Image.open(REAL_PHOTO))
    ax_real.imshow(real_img)
    ax_real.set_title("real camera (static reference)")
else:
    ax_real.text(
        0.5, 0.5, f"No reference photo found at\n{REAL_PHOTO}\n\nSee module"
        " docstring for how to capture one.",
        ha="center", va="center", wrap=True, fontsize=8,
    )
    ax_real.set_title("real camera (missing)")
ax_real.axis("off")

sliders = {}
specs = [
    ("x", start_pos[0] - 0.5, start_pos[0] + 0.5, start_pos[0]),
    ("y", start_pos[1] - 0.5, start_pos[1] + 0.5, start_pos[1]),
    ("z", start_pos[2] - 0.5, start_pos[2] + 0.5, start_pos[2]),
    ("roll", -180, 180, start_euler[0]),
    ("pitch", -180, 180, start_euler[1]),
    ("yaw", -180, 180, start_euler[2]),
    ("fovy", 20, 100, start_fovy),
]
for i, (name, lo, hi, init) in enumerate(specs):
    ax = plt.axes([0.10, 0.34 - i * 0.045, 0.62, 0.03])
    sliders[name] = Slider(ax, name, lo, hi, valinit=init)

radio_ax = plt.axes([0.80, 0.06, 0.18, 0.28])
radio = RadioButtons(radio_ax, list(STAGES.keys()), active=list(STAGES.keys()).index(current_stage_key))

status_text = fig.text(0.08, 0.02, "", fontsize=8, family="monospace")


def redraw(_=None):
    apply_stage(radio.value_selected)
    vals = {name: s.val for name, s in sliders.items()}
    set_camera(vals["x"], vals["y"], vals["z"], vals["roll"], vals["pitch"], vals["yaw"], vals["fovy"])
    new_img = sim.render(camera_name="agentview", width=CAM_W, height=CAM_H, depth=False)[::-1]
    im_artist.set_data(new_img)
    q = sim.model.cam_quat[cam_id]
    status_text.set_text(
        f'pos="{vals["x"]:.4f} {vals["y"]:.4f} {vals["z"]:.4f}"   '
        f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"   '
        f'fovy="{vals["fovy"]:.2f}"'
    )
    fig.canvas.draw_idle()


for s in sliders.values():
    s.on_changed(redraw)
radio.on_clicked(redraw)

redraw()
print("Move sliders to match the real reference photo. Final values print under the plot.")
print("Close the window when done -- final values print here too.")
plt.show()

vals = {name: s.val for name, s in sliders.items()}
q = sim.model.cam_quat[cam_id]
print("\n=== FINAL VALUES ===")
print(f'pos=[{vals["x"]:.4f}, {vals["y"]:.4f}, {vals["z"]:.4f}]')
print(f'quat=[{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]')
print(f'camera_attribs={{"fovy": "{vals["fovy"]:.2f}"}}')
print("Paste into libero_tabletop_manipulation.py's _setup_camera() agentview call")
print("(and mirror into bddl_base_domain.py, per 07-01's existing convention).")

env.close()
