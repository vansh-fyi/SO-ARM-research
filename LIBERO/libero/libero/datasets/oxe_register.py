"""Runtime OXE dataset registration for ``soarm_spatial`` (TUNE-02, D-04).

``openvla-oft``'s ``vla-scripts/finetune.py`` resolves ``--dataset_name`` through
the INSTALLED ``prismatic`` package's ``OXE_DATASET_CONFIGS``/``OXE_NAMED_MIXTURES``
module-level dicts -- there is no CLI flag to point at an unregistered dataset
(06-RESEARCH.md's single highest-risk finding, Pitfall 1).

``finetune.py`` itself is launched via ``torchrun`` as a SEPARATE OS process
(``subprocess.Popen`` in 06a-finetune.ipynb) -- a fresh Python interpreter that
re-imports ``prismatic`` from disk. An in-memory-only monkeypatch of the
notebook kernel's dicts is invisible to that subprocess. So this module patches
the INSTALLED package's ``configs.py``/``mixtures.py``/``transforms.py``/
``constants.py`` files on disk (idempotent -- checks a marker comment before
appending), which any fresh process -- including torchrun's -- picks up on
import like any other built-in dataset.

Three registrations are required, not one:
- ``OXE_DATASET_CONFIGS[name]`` (configs.py) -- dataset shape/encoding.
- ``OXE_NAMED_MIXTURES[name]`` (mixtures.py) -- single-dataset mixture weight.
- ``OXE_STANDARDIZATION_TRANSFORMS[name]`` (transforms.py) -- materialize.py
  does a plain dict lookup with no default; without this, dataset loading
  raises ``KeyError`` immediately regardless of the configs/mixtures entries.
  ``rlds_converter.py``'s output already matches openvla-oft's expected
  post-transform schema (steps/observation/action/language_instruction/...),
  so the correct transform is the identity function -- same pattern as
  openvla-oft's own ``aloha_dataset_transform`` ("dataset is already in the
  correct format").

Also overrides ``PROPRIO_DIM`` in ``constants.py``: openvla-oft auto-detects
robot-platform constants from ``sys.argv`` (any "libero" substring -- which
``--data_root_dir``'s path contains -- selects ``LIBERO_CONSTANTS``, whose
``PROPRIO_DIM=8`` matches LIBERO's own EEF-pose(6) + gripper(2) state layout).
This project's actual proprio is joint-space: 5 SOARM joints + 2-DOF gripper =
``REQUIRED_PROPRIO_DIM=7`` (rlds_converter.py). Feeding 7-dim proprio into an
8-dim fixed-width embedding layer would crash or silently misalign.

``register_soarm_spatial`` is pure dict mutation with zero external imports --
callable and unit-testable with plain empty dicts and sentinel values, no
``prismatic`` package required. ``apply_soarm_spatial_registration`` is the
Colab-facing wrapper: patches the installed files, reloads the patched
modules so the CURRENT kernel's smoke-test sees the correction immediately,
and mirrors ``oft_backend.py``'s ``_ensure_prismatic`` fail-loud contract
(print full traceback, then raise a clear, actionable RuntimeError -- never
swallow).

No ``prismatic``/``tensorflow`` import at module top level -- only stdlib --
matching this package's other modules' local-import convention (see this
package's ``__init__.py`` for the unconditional-import rationale).
"""

from __future__ import annotations

REQUIRED_PROPRIO_DIM = 7  # must match rlds_converter.py's REQUIRED_PROPRIO_DIM


def register_soarm_spatial(
    oxe_dataset_configs: dict,
    oxe_named_mixtures: dict,
    state_encoding,
    action_encoding,
    dataset_name: str = "soarm_spatial",
) -> None:
    """Inject a ``dataset_name`` entry into the given OXE registry dicts.

    Pure dict mutation -- takes the two registry dicts AND the two encoding
    enum values as plain parameters (dependency injection) specifically so
    this function has zero import-time coupling to ``prismatic`` and is fully
    unit-testable with bare dicts/sentinels. Mutates ``oxe_dataset_configs``
    and ``oxe_named_mixtures`` in place; returns ``None``.

    ``state_obs_keys`` is a single-element list, not padded with ``None``
    entries: openvla-oft's RLDS dataset.py inserts one zero-padding element
    per ``None`` entry in this list, so ``["state", None, None]`` would
    produce a 9-dim proprio (7 real + 2 padding) instead of the intended
    7-dim (``REQUIRED_PROPRIO_DIM``).
    """
    oxe_dataset_configs[dataset_name] = {
        "image_obs_keys": {
            "primary": "agentview_rgb",
            "secondary": None,
            "wrist": "eye_in_hand_rgb",
        },
        "state_obs_keys": ["state"],
        "state_encoding": state_encoding,
        "action_encoding": action_encoding,
    }
    oxe_named_mixtures[dataset_name] = [(dataset_name, 1.0)]


def _patch_installed_file(file_path: str, marker: str, code_to_append: str) -> bool:
    """Idempotently append ``code_to_append`` to an installed package file.

    Returns True if the file was patched, False if ``marker`` was already
    present (a prior run/cell-rerun already patched it -- skip to avoid
    duplicate module-level statements).
    """
    with open(file_path, "r") as f:
        content = f.read()
    if marker in content:
        return False
    with open(file_path, "a") as f:
        f.write(f"\n\n{marker}\n{code_to_append}\n")
    return True


def apply_soarm_spatial_registration(dataset_name: str = "soarm_spatial") -> dict:
    """Colab-facing wrapper: register ``dataset_name`` against the REAL,
    installed ``prismatic`` OXE registry, on disk, so ``torchrun``'s separate
    subprocess (which re-imports ``prismatic`` fresh) sees it too -- not just
    this notebook kernel.

    Fails loudly (prints a full traceback, then raises ``RuntimeError``) if
    ``prismatic``'s OXE registry cannot be imported -- mirrors
    ``oft_backend.py``'s ``_ensure_prismatic`` fail-loud contract exactly.
    Returns the registered config dict on success.
    """
    try:
        import importlib

        import prismatic.vla.constants as constants_mod
        import prismatic.vla.datasets.rlds.oxe.configs as configs_mod
        import prismatic.vla.datasets.rlds.oxe.mixtures as mixtures_mod
        import prismatic.vla.datasets.rlds.oxe.transforms as transforms_mod
    except Exception:
        import traceback

        traceback.print_exc()
        raise RuntimeError(
            "prismatic OXE registry import failed -- see traceback above; run "
            "this only inside the training notebook's Colab kernel after "
            "openvla-oft/prismatic is installed"
        )

    # In-memory update for this kernel's own use before the on-disk patch is
    # (re)read below -- matches register_soarm_spatial's pure-mutation contract.
    register_soarm_spatial(
        configs_mod.OXE_DATASET_CONFIGS,
        mixtures_mod.OXE_NAMED_MIXTURES,
        configs_mod.StateEncoding.POS_EULER,
        configs_mod.ActionEncoding.EEF_POS,
        dataset_name=dataset_name,
    )
    transforms_mod.OXE_STANDARDIZATION_TRANSFORMS[dataset_name] = lambda trajectory: trajectory

    # On-disk patch -- this is what torchrun's subprocess actually needs.
    marker = f"# --- {dataset_name} runtime registration (SoARM-Research, oxe_register.py) ---"

    _patch_installed_file(
        configs_mod.__file__,
        marker,
        f'''OXE_DATASET_CONFIGS["{dataset_name}"] = {{
    "image_obs_keys": {{"primary": "agentview_rgb", "secondary": None, "wrist": "eye_in_hand_rgb"}},
    "state_obs_keys": ["state"],
    "state_encoding": StateEncoding.POS_EULER,
    "action_encoding": ActionEncoding.EEF_POS,
}}''',
    )
    _patch_installed_file(
        mixtures_mod.__file__,
        marker,
        f'OXE_NAMED_MIXTURES["{dataset_name}"] = [("{dataset_name}", 1.0)]',
    )
    _patch_installed_file(
        transforms_mod.__file__,
        marker,
        f'OXE_STANDARDIZATION_TRANSFORMS["{dataset_name}"] = lambda trajectory: trajectory',
    )
    _patch_installed_file(
        constants_mod.__file__,
        marker,
        f"PROPRIO_DIM = {REQUIRED_PROPRIO_DIM}  # override: {dataset_name}'s proprio is joint-space "
        f"(5 SOARM joints + 2-DOF gripper), not LIBERO's 8-dim EEF-pose+gripper layout",
    )

    # Reload so THIS kernel's smoke-test also reflects the on-disk patch
    # (not strictly required for torchrun's subprocess, which imports fresh).
    importlib.reload(configs_mod)
    importlib.reload(mixtures_mod)
    importlib.reload(transforms_mod)
    importlib.reload(constants_mod)

    return configs_mod.OXE_DATASET_CONFIGS[dataset_name]
