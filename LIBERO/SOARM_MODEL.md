# Established SO-101 simulation model

User accepted 2026-09-20: SO-101 arm with Robonine parallel gripper, registered
to the corrected Coppelia model. Black housing, arm-yellow jaws, **0.036 m travel
per jaw**. This is the authoritative model for this LIBERO fork.

## Runtime sources

- `libero/libero/assets/robots/soarm101/robot.xml` and its `assets/*.stl` meshes.
- `libero/libero/assets/grippers/soarm_gripper.xml` and `soarm_parallel/*.stl`.
- `libero/libero/envs/robots/soarm.py`: `MountedSoarm101`, initial joints
  `[0, 0, 0, 1.5355350971152681, 0]` radians.
- `libero/libero/envs/grippers/soarm_gripper.py`: `SoarmGripper`, initial jaws
  `[-0.036, -0.036]`; external action **+1 opens, -1 closes**.
- `libero/libero/envs/robots/__init__.py` registers both models with robosuite.

Use `robots=["Soarm101"]` in LIBERO environments. Both jaw joint ranges and
position-actuator ranges are `[-0.036, 0]`. Equality mechanically couples the
two simulated joints; observations retain two jaw coordinates. The 72 mm total
travel change is distinct from the simplified collision-pad aperture (6..78 mm).
The material RGBA values are housing `0.1 0.1 0.1 1`, jaws `1 0.82 0.12 1`.

Use this repository's fork, not a fresh upstream LIBERO checkout. From the
repository root, set `PYTHONPATH` to include both the root and `LIBERO/libero`
when running your own scripts (the verification commands below already do this).
No manual copies into robosuite's site-packages or Coppelia's application folder
are needed. The source-distribution manifest includes model code and resources.
This checkout retains legacy absolute imports and is verified as a repository
installation, not as a standalone wheel installed outside the checkout.

## Reproduce and check (repository root, libero environment)

```sh
python diagnostics/verify_coppelia_alignment.py
python -m pytest diagnostics/test_coppelia_alignment.py scripts/test_verify_urdf.py LIBERO/libero/libero/envs/test_camera_config.py -q
python scripts/mjcf_to_urdf.py
mjpython diagnostics/verify_coppelia_alignment.py --view
```

The viewer uses O/C to open/close and is a kinematic inspection tool. Real
simulation runs through LIBERO/robosuite. `So-101/So-101.urdf` is a generated,
portable derivative; `So-101-coppelia/` and the vendor `so101_new_calib.source.xml`
are historical/source references, not active model definitions.

The numerical Coppelia fixture and diagnostics live in `diagnostics/`; acceptance
is recorded in `.planning/phases/10-digital-twin-fidelity/10-ESTABLISHED-MODEL.md`.
Earlier datasets may embed older XML or states. Preserve them as historical data;
collect new data with this model and revalidate camera framing, normalization,
task reachability and pick/place success. Geometry acceptance does not calibrate
motor dynamics, friction or inertias, or establish policy success rates.
