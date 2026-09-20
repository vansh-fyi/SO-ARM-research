"""Rigidly register original STL vertices to the captured Coppelia visuals.

No visual guessing: try proper principal-axis rotations, refine nearest-vertex
correspondences, and report bidirectional maximum vertex error in metres.
"""
from itertools import permutations, product
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from inspect_coppelia_reference import ROOT, load_reference, transform

PARTS = {
    "main_frame_visual": "parallel_gripper_base_visual",
    "clamp_1_visual": "parallel_gripper_left_jaw_visual",
    "clamp_2_visual": "parallel_gripper_right_jaw_visual",
}


def rigid_fit(source, target):
    sm, tm = source.mean(0), target.mean(0)
    _, sp = np.linalg.eigh(np.cov((source-sm).T))
    _, tp = np.linalg.eigh(np.cov((target-tm).T))
    tree = cKDTree(target)
    candidates = []
    for perm in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            rotation = tp[:, perm] @ np.diag(signs) @ sp.T
            if np.linalg.det(rotation) < 0:
                continue
            translation = tm-rotation@sm
            distance, _ = tree.query(source@rotation.T+translation)
            candidates.append((np.mean(distance**2), rotation, translation))
    _, rotation, translation = min(candidates, key=lambda item: item[0])
    for _ in range(15):
        _, match = tree.query(source@rotation.T+translation)
        target_match = target[match]
        match_mean = target_match.mean(0)
        u, _, vt = np.linalg.svd((source-sm).T@(target_match-match_mean))
        correction = np.diag([1, 1, np.linalg.det(vt.T@u.T)])
        rotation = vt.T@correction@u.T
        translation = match_mean-rotation@sm
    fitted = source@rotation.T+translation
    errors = np.concatenate([tree.query(fitted)[0], cKDTree(fitted).query(target)[0]])
    result = np.eye(4)
    result[:3,:3], result[:3,3] = rotation, translation
    return result, float(errors.max())


def main():
    ref = load_reference()
    base = np.linalg.inv(transform(ref["parallel_gripper_base_respondable"]["matrix_root"]))
    for mesh, obj in PARTS.items():
        source = trimesh.load(ROOT / "LIBERO/libero/libero/assets/grippers/soarm_parallel" / (mesh+".stl")).vertices*.001
        target = np.asarray(ref[obj]["vertices"]).reshape(-1,3)
        fit, error = rigid_fit(source, target)
        pose = base@transform(ref[obj]["matrix_root"])@fit
        if "left" in obj:
            pose[1,3] += ref["parallel_gripper_left"]["position"]-.008
        if "right" in obj:
            pose[1,3] -= ref["parallel_gripper_right"]["position"]-.008
        print(mesh, "max registration error mm",error*1000)
        print("pos", " ".join(f"{x:.10g}" for x in pose[:3,3]))
        print("quat", " ".join(f"{x:.10g}" for x in Rotation.from_matrix(pose[:3,:3]).as_quat()[[3,0,1,2]]))


if __name__ == "__main__":
    main()
