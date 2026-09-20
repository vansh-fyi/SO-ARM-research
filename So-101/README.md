# SO-101 portable URDF export

Keep this directory: `So-101.urdf` is the updated portable export of the accepted
SO-101 / Robonine parallel-gripper model, not the superseded Coppelia URDF.
Its referenced per-link meshes are required by the URDF and verification tools.

## Source of truth

LIBERO runs the canonical MJCF assets directly:

- [Arm](../LIBERO/libero/libero/assets/robots/soarm101/robot.xml)
- [Gripper](../LIBERO/libero/libero/assets/grippers/soarm_gripper.xml)

The accepted configuration has black gripper housing, yellow jaws, and
**0.036 m travel per jaw** (joint range `[0, 0.036]`). See the
[runtime model guide](../LIBERO/SOARM_MODEL.md) and
[Phase 10 acceptance / Phase 11 handoff](../.planning/phases/10-digital-twin-fidelity/10-ESTABLISHED-MODEL.md).

Do not hand-edit the generated URDF. From the repository root, in the `libero`
environment, regenerate and verify it with:

```sh
python scripts/mjcf_to_urdf.py
python -m pytest scripts/test_verify_urdf.py -q
```

The converter reuses the arm's per-link DAE meshes here and copies the gripper
and servo STL assets from the canonical sources. Do not delete this directory
on the assumption that regeneration recreates every mesh.

## Historical artifacts

The whole-assembly `So-101.obj`, `So-101.mtl`, and `So-101.stl` are historical
exports, not the authoritative articulated model. Their presence does not mean
they were regenerated with the accepted gripper corrections.

The separate `So-101-coppelia/` directory contains an earlier problematic export
and is not required by the active model or these verification tools. It may be
archived independently; do not substitute its URDF for this generated one.
