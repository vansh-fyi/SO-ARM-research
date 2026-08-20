"""robomimic-HDF5 -> genuine TFDS/RLDS dataset converter (TUNE-01, D-04/D-05/D-06).

Reads this project's own ``put_the_cream_cheese_in_the_bowl_demo.hdf5`` schema
(``agentview_rgb``, ``eye_in_hand_rgb``, ``joint_states``+``gripper_states``,
``actions`` -- Phase 4's ``hdf5_writer.py`` output) and writes a genuine
``tensorflow_datasets``-loadable RLDS dataset directory that Plan 06-02's
``finetune.py`` invocation trains against.

Pure numpy at module top (no ``h5py``/``tensorflow``/``tensorflow_datasets``
import beyond ``numpy``) so this module -- including its schema-validation and
per-episode HDF5-loading logic -- stays importable in this project's local
``libero`` conda env (h5py available, no tensorflow_datasets). ``h5py``,
``tensorflow``, ``tensorflow_datasets``, and
``LIBERO.libero.libero.envs.bddl_utils.get_problem_info`` are ALL local
imports confined to the function bodies that need them, matching
``normalization.py``'s local-import convention (see this package's
``__init__.py`` for the unconditional-import rationale).
"""

from __future__ import annotations

import os

import numpy as np

# 5 SOARM arm joints + 2-DOF roboninecom 84mm parallel gripper (D-07) -- the
# gripper has two independently sliding jaws (robot0_gripper_qpos is 2-D, not
# Panda's single combined DOF), matching joint_states + gripper_states
# concatenation this dataset's ``state`` observation actually uses.
REQUIRED_PROPRIO_DIM = 7
# OSC_POSE's delta-XYZ(3) + delta-RPY(3) + gripper(1) action space.
REQUIRED_ACTION_DIM = 7


def validate_episode_arrays(agentview_rgb, eye_in_hand_rgb, state, actions) -> None:
    """Fail-loud schema/shape/dtype validation for one episode's four arrays.

    Raises ``ValueError`` (naming the offending array and the actual vs.
    expected shape/dtype) for any malformed input -- never silently pads or
    truncates (D-06, ASVS V5). Returns ``None`` for a well-formed episode.
    """
    arrays = {
        "agentview_rgb": agentview_rgb,
        "eye_in_hand_rgb": eye_in_hand_rgb,
        "state": state,
        "actions": actions,
    }
    for name, arr in arrays.items():
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"{name} must be a numpy array, got {type(arr)!r}")

    for name in ("agentview_rgb", "eye_in_hand_rgb"):
        arr = arrays[name]
        if arr.dtype != np.uint8:
            raise ValueError(f"{name} must be dtype=uint8, got dtype={arr.dtype}")
        if arr.ndim != 4:
            raise ValueError(
                f"{name} must be ndim==4 (T, H, W, 3), got ndim={arr.ndim} shape={arr.shape}"
            )
        if arr.shape[-1] != 3:
            raise ValueError(f"{name} last dim must be 3 (RGB), got shape={arr.shape}")

    if state.ndim != 2:
        raise ValueError(
            f"state must be ndim==2 (T, {REQUIRED_PROPRIO_DIM}), "
            f"got ndim={state.ndim} shape={state.shape}"
        )
    if state.shape[-1] != REQUIRED_PROPRIO_DIM:
        raise ValueError(
            f"state last dim must be {REQUIRED_PROPRIO_DIM} (5 joints + 2-DOF gripper), "
            f"got {state.shape[-1]} (shape={state.shape})"
        )

    if actions.ndim != 2:
        raise ValueError(
            f"actions must be ndim==2 (T, {REQUIRED_ACTION_DIM}), "
            f"got ndim={actions.ndim} shape={actions.shape}"
        )
    if actions.shape[-1] != REQUIRED_ACTION_DIM:
        raise ValueError(
            f"actions last dim must be {REQUIRED_ACTION_DIM} "
            f"(OSC_POSE delta-XYZ+delta-RPY+gripper), got {actions.shape[-1]} "
            f"(shape={actions.shape})"
        )

    lengths = {name: arr.shape[0] for name, arr in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"episode arrays have mismatched leading (timestep) dims: {lengths}")

    if agentview_rgb.shape[0] == 0:
        raise ValueError("episode has zero timesteps")

    if not np.all(np.isfinite(state)):
        raise ValueError("state contains NaN or Inf")
    if not np.all(np.isfinite(actions)):
        raise ValueError("actions contains NaN or Inf")


def load_episodes_from_hdf5(hdf5_paths: list) -> list:
    """Load every ``demo_*`` episode across the given HDF5 paths.

    Unlike ``normalization.py``'s ``load_actions_and_proprio_from_hdf5``, this
    does NOT flatten across demos -- each episode's own timestep count is
    preserved so episode boundaries (``is_first``/``is_last``/``is_terminal``)
    survive into the RLDS conversion (TUNE-01, D-04).

    Returns one dict per episode:
    ``{"agentview_rgb", "eye_in_hand_rgb", "state", "actions", "language_instruction"}``.
    """
    import h5py  # local import: keep this module importable with no sim stack.

    # LIBERO-anchored absolute import (repo ROOT on sys.path), same pattern
    # hdf5_writer.py already relies on for get_problem_info.
    from LIBERO.libero.libero.envs.bddl_utils import get_problem_info

    episodes = []
    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, "r") as f:
            grp = f["data"]

            bddl_file_name = grp.attrs["bddl_file_name"]
            if isinstance(bddl_file_name, bytes):
                bddl_file_name = bddl_file_name.decode()
            # One HDF5 file = one task (this project's dataset scope, D-04).
            language_instruction = get_problem_info(bddl_file_name)["language_instruction"]

            demo_names = sorted(
                (k for k in grp.keys() if k.startswith("demo_")),
                key=lambda k: int(k.split("_")[1]),
            )
            for demo_name in demo_names:
                demo = grp[demo_name]
                obs = demo["obs"]

                agentview_rgb = np.asarray(obs["agentview_rgb"], dtype=np.uint8)
                eye_in_hand_rgb = np.asarray(obs["eye_in_hand_rgb"], dtype=np.uint8)
                state = np.concatenate(
                    [
                        np.asarray(obs["joint_states"]),
                        np.asarray(obs["gripper_states"]),
                    ],
                    axis=-1,
                ).astype(np.float32)
                actions = np.asarray(demo["actions"], dtype=np.float32)

                episodes.append(
                    {
                        "agentview_rgb": agentview_rgb,
                        "eye_in_hand_rgb": eye_in_hand_rgb,
                        "state": state,
                        "actions": actions,
                        "language_instruction": language_instruction,
                    }
                )

    return episodes


def _episode_to_rlds_steps(episode: dict) -> list:
    """Convert one loaded episode dict into a list of per-step RLDS dicts."""
    num_steps = episode["actions"].shape[0]
    steps = []
    for t in range(num_steps):
        is_last = t == num_steps - 1
        steps.append(
            {
                "observation": {
                    "agentview_rgb": episode["agentview_rgb"][t],
                    "eye_in_hand_rgb": episode["eye_in_hand_rgb"][t],
                    "state": episode["state"][t],
                },
                "action": episode["actions"][t],
                "language_instruction": episode["language_instruction"],
                "discount": np.float32(1.0),
                "reward": np.float32(1.0) if is_last else np.float32(0.0),
                "is_first": t == 0,
                "is_last": is_last,
                "is_terminal": is_last,
            }
        )
    return steps


def hdf5_to_rlds(hdf5_paths: list, out_dir: str, dataset_name: str = "soarm_spatial") -> int:
    """Convert every episode in ``hdf5_paths`` into a genuine TFDS-loadable RLDS dataset.

    Loads and validates EVERY episode before writing ANY ``.tfrecord`` bytes --
    a validation failure on episode N raises before episode ``0..N-1`` are ever
    written to ``out_dir`` (validate-then-write ordering matches
    ``normalization.py``'s ``write_dataset_statistics``, extended here to
    per-episode granularity; D-06, ASVS V5).

    This function's body is untestable without ``tensorflow_datasets``
    installed (not available in this project's local libero conda env) --
    callers guard tests with ``pytest.importorskip("tensorflow_datasets")``
    and expect this to run for real only on Colab.
    """
    episodes = load_episodes_from_hdf5(hdf5_paths)
    if not episodes:
        raise ValueError(f"no demo_* episodes found in {hdf5_paths}")

    for i, episode in enumerate(episodes):
        try:
            validate_episode_arrays(
                episode["agentview_rgb"],
                episode["eye_in_hand_rgb"],
                episode["state"],
                episode["actions"],
            )
        except ValueError as exc:
            raise ValueError(f"episode {i} (of {len(episodes)}) failed validation: {exc}") from exc

    # Local imports: tensorflow/tensorflow_datasets are NOT available in this
    # project's local libero conda env (Colab-only, per 06-VALIDATION.md).
    import tensorflow as tf
    import tensorflow_datasets as tfds

    # Feature schema matches 06-RESEARCH.md Pattern 1's step_features exactly
    # (cross-checked against openvla/modified_libero_rlds's documented shapes).
    step_features = tfds.features.FeaturesDict(
        {
            "steps": tfds.features.Dataset(
                {
                    "observation": tfds.features.FeaturesDict(
                        {
                            "agentview_rgb": tfds.features.Image(
                                shape=(None, None, 3), dtype=np.uint8
                            ),
                            "eye_in_hand_rgb": tfds.features.Image(
                                shape=(None, None, 3), dtype=np.uint8
                            ),
                            "state": tfds.features.Tensor(
                                shape=(REQUIRED_PROPRIO_DIM,), dtype=np.float32
                            ),
                        }
                    ),
                    "action": tfds.features.Tensor(
                        shape=(REQUIRED_ACTION_DIM,), dtype=np.float32
                    ),
                    "language_instruction": tf.string,
                    "discount": tf.float32,
                    "reward": tf.float32,
                    "is_first": tf.bool,
                    "is_last": tf.bool,
                    "is_terminal": tf.bool,
                }
            )
        }
    )

    os.makedirs(out_dir, exist_ok=True)

    # NOTE [MEDIUM confidence, 06-RESEARCH.md Pattern 1/Standard Stack]: this
    # project's local libero conda env has no tensorflow_datasets installed,
    # so tfds.core.SequentialWriter's exact constructor/method signature could
    # not be verified against tensorflow_datasets==4.9.10's real source in
    # this session. If this call shape differs from the installed package on
    # Colab, adjust it there -- but keep the feature schema and the
    # validate-then-write ordering above unchanged (Pitfall 2 / D-06).
    dataset_identity = tfds.core.naming.DatasetIdentity(
        name=dataset_name,
        version=tfds.core.Version("1.0.0"),
        data_dir=out_dir,
        module_name=__name__,
    )
    dataset_info = tfds.core.DatasetInfo(builder=dataset_identity, features=step_features)
    writer = tfds.core.SequentialWriter(dataset_info, num_shards=1)
    writer.initialize_splits(["train"])
    examples = [
        (i, {"steps": _episode_to_rlds_steps(episode)}) for i, episode in enumerate(episodes)
    ]
    writer.add_examples({"train": examples})
    writer.close_all()

    # Round-trip proof (Pitfall 2) -- let any exception propagate, do not swallow.
    tfds.builder(dataset_name, data_dir=out_dir).info

    return len(episodes)


if __name__ == "__main__":
    import argparse
    import glob as glob_module

    parser = argparse.ArgumentParser(
        description="Convert SOARM robomimic HDF5 demo file(s) into a genuine "
        "TFDS-loadable RLDS dataset (TUNE-01)."
    )
    parser.add_argument(
        "--hdf5-glob",
        default="LIBERO/libero/datasets/soarm_spatial/*_demo.hdf5",
        help="Glob pattern for input HDF5 file(s).",
    )
    parser.add_argument(
        "--out-dir",
        default="LIBERO/libero/datasets/soarm_spatial/rlds/",
        help="Output RLDS/TFDS directory (data_dir).",
    )
    parser.add_argument(
        "--dataset-name",
        default="soarm_spatial",
        help="TFDS dataset name.",
    )
    args = parser.parse_args()

    paths = sorted(glob_module.glob(args.hdf5_glob))
    n = hdf5_to_rlds(paths, args.out_dir, dataset_name=args.dataset_name)
    print(f"wrote {n} episodes -> {args.out_dir}{args.dataset_name}/")
