"""Mock-based pytest suite for Pi0Backend (VLA-04, Phase 3 Plan 03 Task 1).

`openpi_client` is a Colab-only dependency (03-RESEARCH.md Environment
Availability table) — not installed in this project's local dev environment.
These tests inject fake `openpi_client`/`openpi_client.websocket_client_policy`/
`openpi_client.image_tools` modules into `sys.modules` before importing
`pi0_backend`, so the obs-dict construction and predict() plumbing can be
verified without any real network call, GPU, or openpi install.

NOTE on import path: mirrors test_eval_loop.py's documented convention —
pytest's default rootdir-walking collection mode treats this file's nearest
ancestor without an __init__.py (LIBERO/libero/, since
LIBERO/libero/libero/__init__.py exists) as the insertion point, making this
package resolvable as `libero.vla.*` when invoked via
`pytest LIBERO/libero/libero/vla/test_pi0_backend.py` from the repo root.
"""

import sys
import types
from unittest import mock

import numpy as np
import pytest


@pytest.fixture
def fake_openpi_client(monkeypatch):
    """Install fake openpi_client.websocket_client_policy / image_tools modules.

    Returns the fake `websocket_client_policy` module so tests can inspect
    the mock WebsocketClientPolicy class it defines.
    """
    fake_pkg = types.ModuleType("openpi_client")

    fake_wcp_mod = types.ModuleType("openpi_client.websocket_client_policy")
    mock_wcp_cls = mock.MagicMock(name="WebsocketClientPolicy")
    fake_wcp_mod.WebsocketClientPolicy = mock_wcp_cls

    fake_image_tools_mod = types.ModuleType("openpi_client.image_tools")
    # resize_with_pad / convert_to_uint8 are pass-through stand-ins — the
    # real openpi implementations do actual resizing/dtype conversion, but
    # for these plumbing tests we only need to confirm they're called with
    # the documented (224, 224) target size and that predict() flows their
    # output into the obs dict correctly.
    fake_image_tools_mod.resize_with_pad = mock.MagicMock(
        name="resize_with_pad", side_effect=lambda img, h, w: img
    )
    fake_image_tools_mod.convert_to_uint8 = mock.MagicMock(
        name="convert_to_uint8", side_effect=lambda img: img
    )

    fake_pkg.websocket_client_policy = fake_wcp_mod
    fake_pkg.image_tools = fake_image_tools_mod

    monkeypatch.setitem(sys.modules, "openpi_client", fake_pkg)
    monkeypatch.setitem(sys.modules, "openpi_client.websocket_client_policy", fake_wcp_mod)
    monkeypatch.setitem(sys.modules, "openpi_client.image_tools", fake_image_tools_mod)

    # pi0_backend imports these at module load time — force a fresh import
    # against the fake modules just installed above.
    monkeypatch.delitem(
        sys.modules, "libero.libero.libero.vla.pi0_backend", raising=False
    )
    monkeypatch.delitem(sys.modules, "libero.vla.pi0_backend", raising=False)

    return fake_wcp_mod, fake_image_tools_mod, mock_wcp_cls


def _import_pi0_backend():
    import importlib

    # Matches this package's own resolvable form under pytest's
    # rootdir-walking collection mode (see module docstring).
    return importlib.import_module("libero.vla.pi0_backend")


def test_init_constructs_websocket_client_policy(fake_openpi_client):
    """Test 1: __init__(host, port) constructs WebsocketClientPolicy(host, port)."""
    _, _, mock_wcp_cls = fake_openpi_client
    mod = _import_pi0_backend()

    backend = mod.Pi0Backend(host="localhost", port=8000)

    mock_wcp_cls.assert_called_once_with("localhost", 8000)
    assert backend.client is mock_wcp_cls.return_value


def test_predict_builds_obs_dict_with_required_keys(fake_openpi_client):
    """Test 2: predict() builds obs dict with the 4 required openpi keys.

    Both "observation/image" and "observation/wrist_image" are built from
    images["eye_in_hand"] via resize_with_pad(224, 224) + convert_to_uint8
    (per D-01, this phase only populates the "eye_in_hand" key).
    """
    _, fake_image_tools_mod, mock_wcp_cls = fake_openpi_client
    mod = _import_pi0_backend()

    mock_client = mock_wcp_cls.return_value
    mock_client.infer.return_value = {"actions": np.zeros((10, 7))}

    backend = mod.Pi0Backend(host="localhost", port=8000)
    fake_image = np.zeros((256, 256, 3), dtype=np.uint8)
    backend.predict({"eye_in_hand": fake_image}, "pick up the black bowl")

    assert mock_client.infer.call_count == 1
    (obs,), _kwargs = mock_client.infer.call_args

    assert set(obs.keys()) >= {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
    }
    assert obs["prompt"] == "pick up the black bowl"
    assert isinstance(obs["observation/image"], np.ndarray)
    assert isinstance(obs["observation/wrist_image"], np.ndarray)
    assert isinstance(obs["observation/state"], np.ndarray)

    # Confirm the documented 224x224 resize target was used to build the
    # shared image (per D-01/plan: both obs image keys reuse the single
    # "eye_in_hand" resize+convert result, since no separate wrist camera
    # exists in this phase's scope).
    resize_calls = fake_image_tools_mod.resize_with_pad.call_args_list
    assert len(resize_calls) >= 1
    for call in resize_calls:
        args, _ = call
        assert args[1] == 224 and args[2] == 224


def test_predict_returns_actions_from_infer_unchanged(fake_openpi_client):
    """Test 3: predict() calls client.infer(obs)["actions"] and returns as ndarray."""
    _, _, mock_wcp_cls = fake_openpi_client
    mod = _import_pi0_backend()

    mock_client = mock_wcp_cls.return_value
    expected_actions = np.zeros((10, 7))
    mock_client.infer.return_value = {"actions": expected_actions}

    backend = mod.Pi0Backend(host="localhost", port=8000)
    fake_image = np.zeros((256, 256, 3), dtype=np.uint8)
    result = backend.predict({"eye_in_hand": fake_image}, "pick up the black bowl")

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, expected_actions)
