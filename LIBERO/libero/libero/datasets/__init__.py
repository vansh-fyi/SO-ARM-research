"""SOARM dataset-collection package (Phase 4).

Mirrors ``libero.libero.vla``'s graceful-degradation import pattern: the
sim-dependent submodules (``raw_recorder``, ``hdf5_writer``, and — in later
Phase 4 plans — ``collector``, ``teleop``, ``replay``) pull in robosuite /
MuJoCo, which can fail to import in a context without a working MuJoCo GL
backend or offscreen renderer. Wrap each in its own ``try/except Exception``
so a broken sim backend degrades gracefully (the symbol becomes ``None``)
rather than making the whole package unimportable — exactly the rationale
``vla/__init__.py`` documents for its torch/openpi imports.

Later plans will add ``collector``, ``teleop``, and ``replay`` here using this
same guarded pattern, and ``normalization`` UNCONDITIONALLY (it is pure numpy
with no sim dependency, so it must always import cleanly for the local no-sim
pytest suite).
"""

try:
    # raw_recorder builds a DataCollectionWrapper-wrapped SOARM env, which
    # triggers robosuite/MuJoCo env construction on import of ``..envs``.
    # Degrade gracefully if the MuJoCo GL backend / robosuite import chain
    # is unavailable, matching vla/__init__.py's resilience convention.
    from .raw_recorder import build_recording_env
except Exception:
    build_recording_env = None

try:
    # hdf5_writer imports OffScreenRenderEnv (offscreen MuJoCo renderer) for
    # obs regeneration — same sim-dependency risk as raw_recorder.
    from .hdf5_writer import gather_demonstrations_as_hdf5
except Exception:
    gather_demonstrations_as_hdf5 = None

try:
    # collector drives the scripted waypoint FSM against a real SOARM env;
    # same robosuite/MuJoCo sim-dependency risk as the modules above. The pure
    # compute_waypoint_action FSM has no sim dependency but shares this module.
    from .collector import (
        compute_waypoint_action,
        run_scripted_episode,
        collect_task,
        collect_all,
        TASKS,
    )
except Exception:
    compute_waypoint_action = None
    run_scripted_episode = None
    collect_task = None
    collect_all = None
    TASKS = None

try:
    # replay proves state-based determinism (DATA-02) against a real SOARM
    # env, same robosuite/MuJoCo sim-dependency risk as the modules above.
    from .replay import verify_states_only, verify_full_obs_regeneration
except Exception:
    verify_states_only = None
    verify_full_obs_regeneration = None

try:
    # teleop drives a keyboard-only human teleoperation loop against a real
    # SOARM env (DATA-04, D-02) — same robosuite/MuJoCo sim-dependency risk
    # as the modules above, PLUS an even narrower requirement: a real
    # on-screen GLFW display (has_renderer=True), not just an offscreen
    # renderer. Degrades gracefully in headless/Colab contexts.
    from .teleop import run_teleop_episode, collect_teleop
except Exception:
    run_teleop_episode = None
    collect_teleop = None

# normalization is pure numpy (no sim dependency) — import UNCONDITIONALLY,
# matching this module's own design note (it must always import cleanly for
# the local no-sim pytest suite).
from .normalization import (
    compute_norm_stats,
    load_actions_and_proprio_from_hdf5,
    write_dataset_statistics,
)
