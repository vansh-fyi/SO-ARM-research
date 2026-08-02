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
