"""Runtime OXE dataset registration for ``soarm_spatial`` (TUNE-02, D-04).

``openvla-oft``'s ``vla-scripts/finetune.py`` resolves ``--dataset_name`` through
the INSTALLED ``prismatic`` package's ``OXE_DATASET_CONFIGS``/``OXE_NAMED_MIXTURES``
module-level dicts -- there is no CLI flag to point at an unregistered dataset
(06-RESEARCH.md's single highest-risk finding, Pitfall 1). This module injects a
``soarm_spatial`` entry into those dicts at runtime via a monkeypatch, never
editing the cloned/installed package's files on disk.

``register_soarm_spatial`` is pure dict mutation with zero external imports --
callable and unit-testable with plain empty dicts and sentinel values, no
``prismatic`` package required. ``apply_soarm_spatial_registration`` is the
Colab-facing wrapper that imports the real ``prismatic`` OXE registry dicts and
enum values, mirroring ``oft_backend.py``'s ``_ensure_prismatic`` fail-loud
contract (print full traceback, then raise a clear, actionable RuntimeError --
never swallow).

No ``prismatic``/``tensorflow`` import at module top level -- only stdlib --
matching this package's other modules' local-import convention (see this
package's ``__init__.py`` for the unconditional-import rationale).
"""

from __future__ import annotations


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
    """
    oxe_dataset_configs[dataset_name] = {
        "image_obs_keys": {
            "primary": "agentview_rgb",
            "secondary": None,
            "wrist": "eye_in_hand_rgb",
        },
        "state_obs_keys": ["state", None, None],
        "state_encoding": state_encoding,
        "action_encoding": action_encoding,
    }
    oxe_named_mixtures[dataset_name] = [(dataset_name, 1.0)]


def apply_soarm_spatial_registration(dataset_name: str = "soarm_spatial") -> dict:
    """Colab-facing wrapper: register ``dataset_name`` against the REAL,
    installed ``prismatic`` OXE registry dicts.

    Fails loudly (prints a full traceback, then raises ``RuntimeError``) if
    ``prismatic``'s OXE registry cannot be imported -- mirrors
    ``oft_backend.py``'s ``_ensure_prismatic`` fail-loud contract exactly.
    Returns the registered config dict on success.
    """
    try:
        from prismatic.vla.datasets.rlds.oxe.configs import (
            OXE_DATASET_CONFIGS,
            StateEncoding,
            ActionEncoding,
        )
        from prismatic.vla.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES
    except Exception:
        import traceback

        traceback.print_exc()
        raise RuntimeError(
            "prismatic OXE registry import failed -- see traceback above; run "
            "this only inside the training notebook's Colab kernel after "
            "openvla-oft/prismatic is installed"
        )

    # ASSUMED placeholder (06-RESEARCH.md Assumption A1 / Open Question 1):
    # StateEncoding.POS_EULER is openvla-oft's EEF-pose proprio encoding, not
    # verified against this project's actual joint-space (5 joints + 2-DOF
    # gripper) proprio. The training notebook's own smoke-test cell (Task 2)
    # is what actually confirms this live on Colab -- do not treat this as
    # settled until that cell's printed config has been reviewed.
    register_soarm_spatial(
        OXE_DATASET_CONFIGS,
        OXE_NAMED_MIXTURES,
        StateEncoding.POS_EULER,
        ActionEncoding.EEF_POS,
        dataset_name=dataset_name,
    )
    return OXE_DATASET_CONFIGS[dataset_name]
