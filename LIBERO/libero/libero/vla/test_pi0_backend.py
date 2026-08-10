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

    "observation/image" is built from images["agentview"] and
    "observation/wrist_image" from images["eye_in_hand"] -- independently,
    via resize_with_pad(224, 224) + convert_to_uint8 (Phase 5, D-02: real
    distinct base/wrist views, no duplication).
    """
    _, fake_image_tools_mod, mock_wcp_cls = fake_openpi_client
    mod = _import_pi0_backend()

    mock_client = mock_wcp_cls.return_value
    mock_client.infer.return_value = {"actions": np.zeros((10, 7))}

    backend = mod.Pi0Backend(host="localhost", port=8000)
    fake_image = np.zeros((256, 256, 3), dtype=np.uint8)
    backend.predict(
        {"eye_in_hand": fake_image, "agentview": fake_image}, "pick up the black bowl"
    )

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

    # Confirm the documented 224x224 resize target was used to build both
    # independently-sourced images (agentview -> observation/image,
    # eye_in_hand -> observation/wrist_image).
    resize_calls = fake_image_tools_mod.resize_with_pad.call_args_list
    assert len(resize_calls) >= 2
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
    result = backend.predict(
        {"eye_in_hand": fake_image, "agentview": fake_image}, "pick up the black bowl"
    )

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, expected_actions)


def test_predict_reconnects_and_retries_on_connection_loss(fake_openpi_client):
    """Test 4 (regression, 2026-07-26 L4 run): a transient connection drop is
    retried transparently — a NEW WebsocketClientPolicy is constructed and the
    same observation is re-sent — instead of crashing run_episode mid-suite
    (observed live as ConnectionClosedError 1011 keepalive ping timeout on
    episode 1's first inference)."""
    _, _, mock_wcp_cls = fake_openpi_client
    mod = _import_pi0_backend()

    expected_actions = np.zeros((10, 7))
    # First client: infer raises a retryable connection error.
    first_client = mock.MagicMock(name="first_client")
    first_client.infer.side_effect = ConnectionError("keepalive ping timeout")
    # Second client (built by _reconnect): infer succeeds.
    second_client = mock.MagicMock(name="second_client")
    second_client.infer.return_value = {"actions": expected_actions}
    mock_wcp_cls.side_effect = [first_client, second_client]

    backend = mod.Pi0Backend(host="localhost", port=8000, retry_backoffs=(0,))
    backend._wait_for_port = lambda timeout_s=30.0: True  # no real socket probe

    fake_image = np.zeros((256, 256, 3), dtype=np.uint8)
    result = backend.predict(
        {"eye_in_hand": fake_image, "agentview": fake_image}, "pick up the black bowl"
    )

    np.testing.assert_array_equal(result, expected_actions)
    assert mock_wcp_cls.call_count == 2  # original + reconnect
    assert backend.client is second_client
    assert first_client.infer.call_count == 1
    assert second_client.infer.call_count == 1


def test_predict_raises_runtime_error_after_exhausted_retries(fake_openpi_client):
    """Test 5 (regression): when every attempt fails with a connection error,
    predict() raises an actionable RuntimeError (pointing at the server log)
    rather than leaking the raw websocket exception."""
    _, _, mock_wcp_cls = fake_openpi_client
    mod = _import_pi0_backend()

    dead_client = mock.MagicMock(name="dead_client")
    dead_client.infer.side_effect = ConnectionError("keepalive ping timeout")
    mock_wcp_cls.side_effect = None
    mock_wcp_cls.return_value = dead_client

    backend = mod.Pi0Backend(host="localhost", port=8000, retry_backoffs=(0,))
    backend._wait_for_port = lambda timeout_s=30.0: True

    fake_image = np.zeros((256, 256, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="serve_policy.log"):
        backend.predict(
            {"eye_in_hand": fake_image, "agentview": fake_image}, "pick up the black bowl"
        )


def test_predict_fails_fast_when_server_port_dead(fake_openpi_client):
    """Test 6 (regression): if the port-liveness probe fails after a drop, the
    error names a dead server process instead of blocking forever inside
    WebsocketClientPolicy._wait_for_server()."""
    _, _, mock_wcp_cls = fake_openpi_client
    mod = _import_pi0_backend()

    dead_client = mock.MagicMock(name="dead_client")
    dead_client.infer.side_effect = ConnectionError("keepalive ping timeout")
    mock_wcp_cls.return_value = dead_client

    backend = mod.Pi0Backend(host="localhost", port=8000, retry_backoffs=(0,))
    backend._wait_for_port = lambda timeout_s=30.0: False  # server died

    fake_image = np.zeros((256, 256, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="died"):
        backend.predict(
            {"eye_in_hand": fake_image, "agentview": fake_image}, "pick up the black bowl"
        )
    # No reconnect attempt was made past the dead-port check.
    assert mock_wcp_cls.call_count == 1


def test_predict_uses_distinct_base_and_wrist_images_spatial(fake_openpi_client):
    """Test 7 (Phase 5, D-02): predict() sources observation/image from
    images["agentview"] and observation/wrist_image from
    images["eye_in_hand"] independently -- proves no duplication of a
    single view into both openpi obs keys."""
    _, _, mock_wcp_cls = fake_openpi_client
    mod = _import_pi0_backend()

    mock_client = mock_wcp_cls.return_value
    mock_client.infer.return_value = {"actions": np.zeros((10, 7))}

    backend = mod.Pi0Backend(host="localhost", port=8000)
    eye_in_hand_image = np.zeros((4, 4, 3), dtype=np.uint8)
    agentview_image = np.full((4, 4, 3), 255, dtype=np.uint8)

    backend.predict(
        {"eye_in_hand": eye_in_hand_image, "agentview": agentview_image},
        "pick up the black bowl",
    )

    (obs,), _kwargs = mock_client.infer.call_args
    np.testing.assert_array_equal(obs["observation/image"], agentview_image)
    np.testing.assert_array_equal(obs["observation/wrist_image"], eye_in_hand_image)
    assert not np.array_equal(obs["observation/image"], obs["observation/wrist_image"])
