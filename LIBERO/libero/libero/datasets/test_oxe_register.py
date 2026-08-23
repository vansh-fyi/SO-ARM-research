"""Tests for oxe_register.py -- OXE dataset registration (Phase 6, TUNE-02).

Import-path note (mirrors test_normalization.py / test_rlds_converter.py):
insert the repo root so ``LIBERO.libero.libero.envs`` absolute imports resolve
if ever needed, and ``LIBERO/libero`` so ``libero.datasets.*`` resolves
regardless of invocation dir. oxe_register.py itself has no sim dependency,
but this import-path setup is kept consistent with the package's sibling test
files.

``register_soarm_spatial`` is tested purely with bare dicts and ``object()``
sentinels -- no ``prismatic`` install required.
``apply_soarm_spatial_registration`` is tested for its REAL, unmocked
fail-loud behavior: this project's local ``libero`` conda env genuinely has no
``prismatic`` installed, so calling it directly proves the guard fires
end-to-end without any monkeypatching.
"""

import os
import sys

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_LIBERO_LIBERO = os.path.join(_REPO_ROOT, "LIBERO", "libero")
if _LIBERO_LIBERO not in sys.path:
    sys.path.insert(0, _LIBERO_LIBERO)

import pytest

from libero.datasets.oxe_register import (
    register_soarm_spatial,
    apply_soarm_spatial_registration,
)


def test_register_soarm_spatial_injects_expected_dict_shape():
    state_encoding_sentinel = object()
    action_encoding_sentinel = object()
    oxe_dataset_configs = {}
    oxe_named_mixtures = {}

    register_soarm_spatial(
        oxe_dataset_configs,
        oxe_named_mixtures,
        state_encoding=state_encoding_sentinel,
        action_encoding=action_encoding_sentinel,
    )

    entry = oxe_dataset_configs["soarm_spatial"]
    assert entry["image_obs_keys"] == {
        "primary": "agentview_rgb",
        "secondary": None,
        "wrist": "eye_in_hand_rgb",
    }
    assert entry["state_obs_keys"] == ["state", None, None]
    assert entry["state_encoding"] is state_encoding_sentinel
    assert entry["action_encoding"] is action_encoding_sentinel
    assert oxe_named_mixtures["soarm_spatial"] == [("soarm_spatial", 1.0)]


def test_register_soarm_spatial_uses_custom_dataset_name():
    oxe_dataset_configs = {}
    oxe_named_mixtures = {}

    register_soarm_spatial(
        oxe_dataset_configs,
        oxe_named_mixtures,
        state_encoding=object(),
        action_encoding=object(),
        dataset_name="custom_name",
    )

    assert "custom_name" in oxe_dataset_configs
    assert "soarm_spatial" not in oxe_dataset_configs
    assert "custom_name" in oxe_named_mixtures
    assert "soarm_spatial" not in oxe_named_mixtures
    assert oxe_named_mixtures["custom_name"] == [("custom_name", 1.0)]


def test_apply_soarm_spatial_registration_fails_loud_without_prismatic(capsys):
    with pytest.raises(RuntimeError, match="prismatic OXE registry import failed"):
        apply_soarm_spatial_registration()

    captured = capsys.readouterr()
    assert "Traceback" in captured.out or "Traceback" in captured.err
