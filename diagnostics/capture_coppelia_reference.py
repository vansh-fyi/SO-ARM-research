"""Read the stopped SO-101 scene without changing any CoppeliaSim objects.

Requires coppeliasim-zmqremoteapi-client (or its bundled Python source), numpy.
The snapshot includes mesh-local vertices AND complete object matrices, avoiding
Euler conventions and the distinction between visual/respondable shape frames.
"""
import argparse
import json
from pathlib import Path

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import zmq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = RemoteAPIClient()
    sim = client.require("sim")
    client.socket.setsockopt(zmq.RCVTIMEO, 60000)
    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError("Stop the simulation before capturing a stable reference.")
    root = sim.getObject("/so101")
    objects = []
    for handle in sim.getObjectsInTree(root, sim.handle_all, 0):
        kind = sim.getObjectType(handle)
        item = dict(
            handle=handle, name=sim.getObjectAlias(handle),
            parent=sim.getObjectParent(handle), type=kind,
            matrix_world=sim.getObjectMatrix(handle, sim.handle_world),
            matrix_root=sim.getObjectMatrix(handle, root),
            matrix_parent=sim.getObjectMatrix(handle, sim.handle_parent),
        )
        if kind == sim.sceneobject_joint:
            item.update(position=sim.getJointPosition(handle),
                        interval=sim.getJointInterval(handle),
                        joint_type=sim.getJointType(handle))
        if kind == sim.sceneobject_shape:
            vertices, indices, _ = sim.getShapeMesh(handle)
            item.update(vertices=vertices, indices=indices)
        objects.append(item)
        print(item["name"], "parent", item["parent"],
              "q=" + str(item["position"]) if "position" in item else
              "vertices=" + str(len(item.get("vertices", [])) // 3))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(root=root, objects=objects)))
    print("Saved", args.output)


if __name__ == "__main__":
    main()
