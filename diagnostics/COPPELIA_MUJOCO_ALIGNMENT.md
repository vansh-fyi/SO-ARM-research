# Coppelia reference → MuJoCo alignment

Resolved the gripper assembly mismatch on 2026-09-20 using the user's saved
`/Applications/coppeliaSim.app/Contents/Resources/models/robots/non-mobile/SO-ARM 101-parallel.ttm`.
The capture was taken **after the user reparented the gripper under wrist_roll**.
The source model was loaded and read, not edited by the conversion tools.

**Appearance/travel update:** the user subsequently requested a black housing,
yellow jaws matching the arm (`1 0.82 0.12`), and a 36 mm per-jaw travel cap.
The current open endpoint is MuJoCo +0.036 (Coppelia-equivalent 0.044), with
72 mm total aperture change and 6..78 mm collision-proxy separation. The table
and 42 mm source interval below describe the original captured reference.
Verification now samples the reduced range; the source fixture stays unchanged.

**Positive-opening coordinate update (2026-09-20):** at the user's request,
both jaw axes and coordinate signs were reversed without moving any geometry.
The current range is `[0, 0.036]`, closed to open; source mapping is now
`Coppelia q = 0.008 + MuJoCo q`. The LIBERO action adapter was also reversed,
preserving external `+1 = open`, `-1 = close`. Negative joint coordinates in
the historical investigation below describe the superseded convention.

## What was wrong

The previous handoff's assumed frame equivalences were incorrect. A Coppelia
shape origin is not necessarily the original STL origin, and matching one
assumed quaternion does not establish a common coordinate frame. The imported
Coppelia meshes already contain local transformations baked into their vertices.
The jaw STL coordinates also sit hundreds of millimetres from their own origins.
Applying object orientations directly to those STL vertices therefore does not
reproduce the Coppelia geometry.

The previous centroid correction compounded this by mixing MuJoCo's processed
mesh coordinates with authored XML geom transforms. MuJoCo centres and aligns
meshes during compilation; processed vertices must be paired with the compiled
geom transform, not the original XML transform.

The old collision boxes and grip_site assumed fingers extending along right_hand
-X. In the saved reference assembly they extend approximately along right_hand
-Z. The fixed frame and wrist-mounted servo also needed corrected placement.
The two jaw meshes **do use effectively the same rotation** after accounting for
their own CAD coordinates; arbitrarily rotating just one was not the solution.

The earlier URDF export also lacked the wrist/jaw hierarchy. Before the user's
last correction, a live read showed parallel_gripper_base_respondable parented
to wrist_link_respondable. The final capture confirms its parent is wrist_roll.
No exporter limitation needs to be assumed to explain that missing wrist link.

## How the fix was measured

1. Capture all 25 objects, their full 3×4 matrices, joint positions/intervals,
   and shape-local mesh vertices directly from Coppelia. No Euler conversion.
2. Align the two models' shoulder_pan frames to remove the arbitrary scene/root
   coordinate difference. Independently compare the remaining joint frames and
   all arm visual meshes. No arm joint definitions required changes.
3. Rigidly register the original frame/jaw STLs to the captured visual vertices.
   The maximum bidirectional registration residual is below 0.000004 mm.
4. Preserve the measured gripper base frame as the nested
   `parallel_gripper_mechanism` body. Apply registered STL transforms within it.
   Correct both visual and collision poses of the parallel-gripper servo.
5. Copy the source jaw collision-box placement, put grip_site at their midpoint,
   and include the source's cylindrical pinion visual. Preserve the existing
   joint equality, actuator strengths, ranges, and external action convention.

Joint correspondence:

| State | MuJoCo jaw q | Coppelia jaw q |
|---|---:|---:|
| Closed endpoint | 0 | 0.008 |
| Captured pose | -0.012 | 0.020 |
| Half travel | -0.021 | 0.029 |
| Open endpoint | -0.042 | 0.050 |

The Coppelia interval's second number is its **length**, not its upper endpoint.
The stored interval `[0.008, 0.042]` thus ends at 0.050 metres.

The saved arm pose is `[0, 0, 0, 1.5355350971152681, 0]` in joint order
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll. LIBERO's
`MountedSoarm101.init_qpos` now uses this downward-facing pose. Calibration zeros,
joint limits, and hardware control files were not changed.

## Verification and viewing

```sh
conda activate libero
python diagnostics/verify_coppelia_alignment.py
python -m pytest diagnostics/test_coppelia_alignment.py scripts/test_verify_urdf.py -q
python diagnostics/verify_coppelia_alignment.py --render
mjpython diagnostics/verify_coppelia_alignment.py --view
```

The viewer is kinematic so gravity does not collapse the arm while inspecting
geometry. Press **O** to open and **C** to close. Standard camera orbit/zoom is
available. This does not connect to or command physical hardware.

The verifier compares compiled MuJoCo vertices against independently captured
Coppelia points, including source forward kinematics through changed arm and
wrist configurations. It covers five arm mesh groups, frame, two jaws, servo,
and two collision boxes at four openings and three poses: **132 comparisons**.
The maximum measured difference is **0.0015 mm** (tolerance: 0.005 mm). It also
checks grip_site remains the midpoint of the contact boxes. A separate dynamics
test disables the right actuator and confirms the left still drives both jaws.

`reference/soarm_coppelia.json` stores sampled source points, transforms and the
full capture's SHA-256. Normal verification needs no running Coppelia instance
or CBOR dependency. The initial full-vertex comparison used every captured
vertex, including over 160,000 arm vertices.

Generated, ignored outputs:

- `outputs/soarm_coppelia_comparison.png`: four angles × captured/closed/open poses.
- `outputs/soarm_alignment_report.json`: all numerical residuals.
- `outputs/soarm_reference_scene.xml`: combined arm/gripper inspection scene.
- `outputs/coppelia_reference.cbor`: original full read-only scene snapshot.
- `outputs/soarm_gripper_before_coppelia_fix.xml` and
  `outputs/soarm_robot_before_coppelia_fix.xml`: pre-edit backups, including the
  user's previous uncommitted gripper edits.

To capture a future reference, run `capture_coppelia_reference.lua` using
`dofile('/absolute/path/to/diagnostics/capture_coppelia_reference.lua')` in the
Coppelia Lua console while stopped. The Python remote-API alternative is
`capture_coppelia_reference.py`. `inspect_coppelia_reference.py` and
`fit_coppelia_meshes.py` inspect the CBOR capture (require cbor2; the fitter also
requires trimesh). `--capture` verifies the full snapshot; `--write-reference`
explicitly replaces the compact reference fixture. Do not regenerate that
fixture merely to make a failing geometry check pass.

## What this does not establish

This verifies **geometry and kinematics against Coppelia**, not experimentally
identified hardware dynamics. Mass/inertia, friction, motor dynamics, and grasp
forces remain approximations. The source collision boxes are simplified solid
proxies; they do not follow mesh holes, rack teeth, or the exact fingertip
surfaces. Their endpoint separation is 6 mm closed and 90 mm open (84 mm change),
so “84 mm stroke” is not an exact claim about collision-pad aperture.

The old scripted pick/place controller was tuned with different gripper geometry
and an old zero-joint starting pose. Its success-rate validation must be rerun and
may need retuning; matching geometry does not by itself validate collection
success or previously tuned camera framing. The old 50-attempt success test was
interrupted while testing the pre-starting-pose-update process; no success-rate
claim is made here. Structural, geometry, coupling, and camera integration checks
are separate from that task-level evaluation.

The URDF converter preserves the nested mounting body, jaw colours, wrist
servo, cylindrical pinion, grasp frame and jaw mimic relation. Its portable
mesh copies and generated URDF are refreshed from the canonical MJCF assets.
The Coppelia measurement fixture remains the reference for MuJoCo verification.

Technical references: [Coppelia object matrices](https://manual.coppeliarobotics.com/en/sim/simGetObjectMatrix.htm),
[Coppelia shape mesh data](https://manual.coppeliarobotics.com/en/sim/simGetShapeMesh.htm),
[MuJoCo mesh centring/alignment](https://mujoco.readthedocs.io/en/2.3.6/XMLreference.html#asset-mesh).
