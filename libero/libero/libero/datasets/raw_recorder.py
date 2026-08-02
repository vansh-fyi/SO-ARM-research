"""Shared raw-recording env builder for Phase 4's dataset pipeline.

Wraps a SOARM ``ControlEnv`` with robosuite's ``VisualizationWrapper`` and
``DataCollectionWrapper`` so that BOTH the scripted collector (later plan) and
the keyboard teleop interface (later plan) can log raw MuJoCo states + actions
per episode to a local tmp directory, via one identical construction path.

Adapted from ``LIBERO/scripts/collect_demonstration.py`` (lines 274-306), but
built directly against ``TASK_MAPPING`` with ``robots=["Soarm101"]`` and zero
custom controller kwargs beyond ``default_controller="OSC_POSE"`` — the
Phase 2/3 confirmed convention that a generic OSC_POSE config suffices for SOARM
(02-03 key-decisions). This is NOT a modification of the existing npz-based
prior-art scripts (D-05); it is new, reusable infrastructure.
"""

import os

# Set the MuJoCo GL backend BEFORE any import that triggers a MuJoCo import
# (i.e. before ``from ..envs import ...``). Matches the established macOS-headless
# convention in explorations/create_scene.py / explorations/soarm_sanity.py.
# Use setdefault (not a plain assignment) so an externally-set osmesa value on
# Linux is not clobbered.
os.environ.setdefault("MUJOCO_GL", "glfw")

import robosuite as suite
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

from ..envs import TASK_MAPPING
from ..envs.bddl_utils import get_problem_info


def build_recording_env(
    bddl_file_name,
    tmp_directory,
    robots=("Soarm101",),
    has_renderer=False,
    render_camera="agentview",
    controller="OSC_POSE",
):
    """Build a DataCollectionWrapper-wrapped SOARM env for raw episode recording.

    Args:
        bddl_file_name (str): Path to the BDDL task file (developer-supplied
            local path — validated with ``os.path.exists``, fail-loudly).
        tmp_directory (str): Directory where DataCollectionWrapper flushes
            per-episode ``state_*.npz`` chunks.
        robots (tuple[str]): Robot class name(s). Defaults to SOARM's
            ``Soarm101`` (Phase 2) — do NOT hardcode "Panda".
        has_renderer (bool): On-screen GLFW viewer (True for teleop, False for
            headless scripted collection).
        render_camera (str): Camera name for the on-screen viewer.
        controller (str): robosuite controller name; OSC_POSE per Phase 2/3.

    Returns:
        DataCollectionWrapper: env ready for ``reset()`` / ``step(action)``.
    """
    # Fail loudly on a missing task file (matches env_wrapper.py's convention),
    # rather than silently swallowing it downstream.
    assert os.path.exists(
        bddl_file_name
    ), f"[error] {bddl_file_name} does not exist!"

    controller_configs = suite.load_controller_config(default_controller=controller)

    problem_info = get_problem_info(bddl_file_name)
    problem_name = problem_info["problem_name"]

    env = TASK_MAPPING[problem_name](
        bddl_file_name=bddl_file_name,
        robots=list(robots),
        controller_configs=controller_configs,
        has_renderer=has_renderer,
        has_offscreen_renderer=False,
        render_camera=render_camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    env = VisualizationWrapper(env)
    env = DataCollectionWrapper(env, tmp_directory)
    return env
