"""Assemble raw DataCollectionWrapper episodes into a robomimic-schema HDF5.

Stage 2 of Phase 4's two-stage collect-then-extract pipeline (DATA-01). Reads
the per-episode ``state_*.npz`` chunks a ``DataCollectionWrapper`` flushed to a
tmp directory (states + actions only, no observations), and for each recorded
mujoco state regenerates real image + proprio observations by replaying the
state through this repo's own ``ControlEnv.set_init_state()`` primitive
(``env_wrapper.py``). The regenerated observations are renamed to LIBERO's
schema (``OBS_KEY_MAPPING``) and written into an ``obs/`` group so Phase 6's
robomimic ``SequenceDataset`` reader finds the exact keys it expects.

Key correctness details (all from RESEARCH.md / installed robosuite prior art):
  * Success-gating is OR-accumulated across ALL npz chunks of an episode, since
    ``DataCollectionWrapper._flush()`` resets ``self.successful = False`` after
    every flush — gating on only the last chunk would drop genuinely successful
    multi-flush episodes. Unsuccessful episodes are silently dropped (never
    written), so failed attempts do not pad the dataset.
  * The extra trailing state is dropped to fix the DataCollectionWrapper
    off-by-one (it records the state AFTER playing each action).
  * Obs keys are renamed to LIBERO's convention at write time — writing raw
    robosuite key names causes a silent KeyError at Phase 6 load time.

No ``robomimic`` import: ``env_type=1`` (robomimic's ROBOSUITE_TYPE enum) is
hardcoded rather than imported, per RESEARCH.md's no-robomimic-dependency rule.
"""

import json
import os
from glob import glob

import numpy as np

# Set the MuJoCo GL backend before the import that triggers MuJoCo (same
# reasoning as raw_recorder.py). setdefault so Linux osmesa is not clobbered.
os.environ.setdefault("MUJOCO_GL", "glfw")

import h5py

# LIBERO-anchored absolute import (repo ROOT on sys.path, per 02-02 STATE.md
# decision), NOT a relative ``from ..envs`` — see raw_recorder.py for the full
# rationale (relative import binds a separate, unregistered TASK_MAPPING copy).
from LIBERO.libero.libero.envs import OffScreenRenderEnv
from LIBERO.libero.libero.envs.bddl_utils import get_problem_info

# Copied verbatim from LIBERO/libero/configs/data/default.yaml's obs_key_mapping.
# Maps LIBERO's schema key (written into the HDF5) -> raw robosuite obs-dict key
# (what env._get_observations() returns). Writing the raw robosuite names instead
# causes a silent KeyError at Phase 6 load time, not at this phase's write time.
OBS_KEY_MAPPING = {
    "agentview_rgb": "agentview_image",
    "eye_in_hand_rgb": "robot0_eye_in_hand_image",
    "agentview_depth": "agentview_depth",  # robosuite's own key name, no rename needed
    "gripper_states": "robot0_gripper_qpos",
    "joint_states": "robot0_joint_pos",
}

# Which renamed keys are uint8 images vs. float proprio arrays vs. depth maps.
_RGB_KEYS = ("agentview_rgb", "eye_in_hand_rgb")
# agentview-only per D-05 (only the real overhead AR0144 is stereo/depth-capable;
# eye_in_hand's real IMX335 wrist cam has no depth channel).
_DEPTH_KEYS = ("agentview_depth",)
_STATE_KEYS = ("gripper_states", "joint_states")


def gather_demonstrations_as_hdf5(
    tmp_directory,
    hdf5_path,
    bddl_file_name,
    robots=("Soarm101",),
    camera_size=128,
):
    """Gather raw npz episodes into an obs-augmented robomimic-schema HDF5.

    Args:
        tmp_directory (str): Directory containing per-episode subdirectories of
            ``state_*.npz`` chunks + ``model.xml`` (DataCollectionWrapper output).
        hdf5_path (str): Output HDF5 path.
        bddl_file_name (str): BDDL task file the episodes were recorded against
            (validated fail-loudly; also stored in the file's ``data`` attrs so a
            later replay plan can rebuild the regeneration env without a separate
            argument).
        robots (tuple[str]): Robot class name(s); SOARM's ``Soarm101`` default.
        camera_size (int): Square camera height/width for regenerated images.

    Returns:
        int: Number of successful demo groups written.
    """
    assert os.path.exists(
        bddl_file_name
    ), f"[error] {bddl_file_name} does not exist!"

    problem_name = get_problem_info(bddl_file_name)["problem_name"]

    # ONE offscreen env, reused across every recorded state in every episode
    # (Phase 2/3 zero-custom-kwarg construction). reset() once to build the sim
    # so set_init_state()'s set_state_from_flattened has a live model to write into.
    regen_env = OffScreenRenderEnv(
        bddl_file_name=bddl_file_name,
        robots=list(robots),
        camera_heights=camera_size,
        camera_widths=camera_size,
        has_renderer=False,
        has_offscreen_renderer=True,
        # Renders depth for BOTH agentview and robot0_eye_in_hand globally —
        # robosuite has no per-camera depth toggle in 1.4.0. agentview-only
        # persistence (D-05) is enforced below, by which keys are written,
        # not by suppressing rendering here.
        camera_depths=True,
    )
    regen_env.reset()

    f = h5py.File(hdf5_path, "w")
    grp = f.create_group("data")

    num_eps = 0
    try:
        for ep_directory in sorted(os.listdir(tmp_directory)):
            ep_path = os.path.join(tmp_directory, ep_directory)
            if not os.path.isdir(ep_path):
                continue

            states = []
            actions = []
            success = False
            for state_file in sorted(glob(os.path.join(ep_path, "state_*.npz"))):
                dic = np.load(state_file, allow_pickle=True)
                states.extend(dic["states"])
                actions.extend(ai["actions"] for ai in dic["action_infos"])
                # OR-accumulate across flush chunks (see module docstring).
                success = success or bool(dic["successful"])

            # Silently drop unsuccessful / empty episodes — they must never pad
            # the dataset (RESEARCH.md Pitfall 3).
            if not success or len(states) == 0:
                continue

            # Off-by-one: DataCollectionWrapper records state AFTER the action.
            del states[-1]
            assert len(states) == len(
                actions
            ), f"state/action misalignment: {len(states)} vs {len(actions)}"

            # Regenerate observations for every recorded state.
            obs_acc = {k: [] for k in OBS_KEY_MAPPING}
            for state in states:
                obs = regen_env.set_init_state(state)
                for schema_key, robosuite_key in OBS_KEY_MAPPING.items():
                    obs_acc[schema_key].append(obs[robosuite_key])

            num_eps += 1
            ep_grp = grp.create_group("demo_{}".format(num_eps))
            ep_grp.attrs["num_samples"] = len(states)

            xml_path = os.path.join(ep_path, "model.xml")
            with open(xml_path, "r") as xf:
                ep_grp.attrs["model_file"] = xf.read()

            ep_grp.create_dataset(
                "states", data=np.array(states, dtype=np.float64)
            )
            ep_grp.create_dataset(
                "actions", data=np.array(actions, dtype=np.float64)
            )

            obs_grp = ep_grp.create_group("obs")
            for key in _RGB_KEYS:
                obs_grp.create_dataset(
                    key, data=np.array(obs_acc[key], dtype=np.uint8)
                )
            for key in _DEPTH_KEYS:
                # obs_acc[key] is already (H, W, 1)-shaped per step, in the
                # sim's native normalized [0, 1] range straight from
                # robosuite's offscreen renderer — this function performs no
                # unit conversion (e.g. to metric meters) before writing, so
                # depth_xyz.py's downstream consumer contract holds.
                obs_grp.create_dataset(
                    key, data=np.array(obs_acc[key], dtype=np.float32)
                )
            for key in _STATE_KEYS:
                # Coerce each per-step proprio value to at least 1-D before
                # stacking so the dataset is 2-D (N, D) as robomimic's obs schema
                # expects. SOARM's 84mm 2-DOF parallel gripper (D-07 upgrade)
                # returns robot0_gripper_qpos as shape (2,); atleast_1d also
                # guards any future gripper variant returning a 0-d scalar,
                # which would otherwise stack to a malformed (N,) instead of
                # (N, D).
                stacked = np.array(
                    [np.atleast_1d(v) for v in obs_acc[key]], dtype=np.float64
                )
                obs_grp.create_dataset(key, data=stacked)

        grp.attrs["total"] = num_eps
        grp.attrs["env_args"] = json.dumps(
            {
                "env_name": problem_name,
                # robomimic's EnvType.ROBOSUITE_TYPE == 1; hardcoded to avoid a
                # robomimic import for one constant (RESEARCH.md anti-pattern).
                "env_type": 1,
                "env_kwargs": {
                    "robots": list(robots),
                    "controller_name": "OSC_POSE",
                },
            }
        )
        grp.attrs["bddl_file_name"] = os.path.abspath(bddl_file_name)
    finally:
        f.close()
        regen_env.close()

    return num_eps
