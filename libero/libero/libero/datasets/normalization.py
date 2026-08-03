"""SOARM-specific action/proprio normalization statistics (DATA-03, Phase 4).

Computes the OpenVLA q01/q99/mean/std/min/max normalization-statistics schema
directly from this project's own collected SOARM dataset (04-02's real HDF5
output), NOT copied from Panda's LIBERO stats. The schema matches exactly what
``oft_backend.py`` already consumes via its ``dataset_statistics.json``
overlay (``hf_hub_download`` + ``json.load`` + dict-key access, lines 99-119)
so Phase 6 can reuse that same loading path for this file instead of writing a
second stats-loading code path.

Pure numpy at module top (no MuJoCo/robosuite/torch/h5py import beyond
``numpy``/``json``) so ``compute_norm_stats`` stays importable even in a
context without the sim stack — matches this package's ``__init__.py``
convention of unconditional imports for dependency-light modules. The h5py
import needed to actually read a dataset off disk is confined to
``load_actions_and_proprio_from_hdf5``.
"""

from __future__ import annotations

import json
import os

import numpy as np


def _stats(x: np.ndarray) -> dict:
    """Per-dimension mean/std/max/min/q01/q99 stats, OpenVLA schema (verbatim)."""
    return {
        "mean": x.mean(0).tolist(),
        "std": x.std(0).tolist(),
        "max": x.max(0).tolist(),
        "min": x.min(0).tolist(),
        "q01": np.quantile(x, 0.01, axis=0).tolist(),
        "q99": np.quantile(x, 0.99, axis=0).tolist(),
    }


def compute_norm_stats(
    actions: np.ndarray,
    proprio: np.ndarray,
    dataset_key: str = "soarm_spatial",
    num_trajectories: int | None = None,
) -> dict:
    """Compute OpenVLA-schema normalization stats for actions + proprio.

    Args:
        actions: (T, 7) array of per-step actions (OSC_POSE's 7-D action space).
        proprio: (T, D) array of per-step proprioceptive observations (SOARM's
            joint_states concatenated with gripper_states — D depends on the
            gripper's DOF, NOT hardcoded here).
        dataset_key: top-level dict key (Phase 6 unnorm_key convention).
        num_trajectories: caller-supplied demo count (not inferred from T,
            since T is the total transition count across all demos).

    Returns:
        {dataset_key: {"action": {...}, "proprio": {...},
                       "num_transitions": int, "num_trajectories": int}}
    """
    return {
        dataset_key: {
            "action": _stats(actions),
            "proprio": _stats(proprio),
            "num_transitions": int(actions.shape[0]),
            "num_trajectories": num_trajectories,
        }
    }


def load_actions_and_proprio_from_hdf5(hdf5_paths: list) -> tuple:
    """Load + concatenate actions/proprio across every demo in every HDF5 path.

    Proprio = SOARM's ``joint_states`` concatenated with ``gripper_states``
    along the last axis (this repo's own obs schema, per ``hdf5_writer.py`` —
    NOT Panda's 7+2 convention). Returns
    ``(actions, proprio, total_demo_count)``.
    """
    import h5py  # local import: keep this module importable with no sim stack.

    actions_list = []
    proprio_list = []
    total_demo_count = 0

    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, "r") as f:
            grp = f["data"]
            demo_names = [k for k in grp.keys() if k.startswith("demo_")]
            total_demo_count += len(demo_names)
            for demo_name in demo_names:
                demo = grp[demo_name]
                actions_list.append(np.asarray(demo["actions"]))
                obs = demo["obs"]
                proprio_list.append(
                    np.concatenate(
                        [np.asarray(obs["joint_states"]), np.asarray(obs["gripper_states"])],
                        axis=-1,
                    )
                )

    actions = np.concatenate(actions_list, axis=0)
    proprio = np.concatenate(proprio_list, axis=0)
    return actions, proprio, total_demo_count


def write_dataset_statistics(
    hdf5_paths: list,
    out_json_path: str,
    dataset_key: str = "soarm_spatial",
) -> dict:
    """Compute norm stats from real HDF5 dataset(s) and write them to disk.

    Per T-04-04-01: the JSON write only happens AFTER stats are successfully
    computed over the full loaded array — a load failure (missing/corrupt
    HDF5) raises before any partial file is written.
    """
    actions, proprio, total_demo_count = load_actions_and_proprio_from_hdf5(hdf5_paths)
    stats = compute_norm_stats(actions, proprio, dataset_key, num_trajectories=total_demo_count)

    out_dir = os.path.dirname(out_json_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_json_path, "w") as f:
        json.dump(stats, f)

    return stats


if __name__ == "__main__":
    import argparse
    import glob as glob_module

    parser = argparse.ArgumentParser(
        description="Compute SOARM-specific action/proprio normalization "
        "statistics (DATA-03) from the real collected dataset."
    )
    parser.add_argument(
        "--hdf5-glob",
        default="LIBERO/libero/datasets/soarm_spatial/*_demo.hdf5",
        help="Glob pattern for input HDF5 file(s).",
    )
    parser.add_argument(
        "--out",
        default="LIBERO/libero/datasets/soarm_spatial/dataset_statistics.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    paths = sorted(glob_module.glob(args.hdf5_glob))
    result = write_dataset_statistics(paths, args.out)
    key = list(result.keys())[0]
    print(
        f"wrote stats for {result[key]['num_transitions']} transitions, "
        f"{result[key]['num_trajectories']} trajectories -> {args.out}"
    )
