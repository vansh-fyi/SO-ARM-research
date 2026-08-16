"""Mock-based pytest suite for OFTBackend's 05-UAT.md gap-closure fixes
(Phase 5 Plan 04, D-02, SPAT-02).

`torch`, `huggingface_hub`, and `transformers` are unconditionally imported
at module level by `oft_backend.py`, but none of them are installed in this
project's local `libero` conda env (confirmed: `conda run -n libero python
-c "import torch"` raises `ModuleNotFoundError`). These tests inject fake
`torch`/`huggingface_hub`/`transformers` modules into `sys.modules` before
importing `oft_backend`, mirroring `test_pi0_backend.py`'s sys.modules-
injection technique, so the two live-Colab regression fixes (05-UAT.md
Gap 1) can be verified locally without any GPU/network/real model weights.

NOTE on import path: mirrors test_pi0_backend.py's documented convention —
pytest's default rootdir-walking collection mode treats this file's nearest
ancestor without an __init__.py (LIBERO/libero/, since
LIBERO/libero/libero/__init__.py exists) as the insertion point, making this
package resolvable as `libero.vla.*` when invoked via
`pytest LIBERO/libero/libero/vla/test_oft_backend.py` from the repo root.
"""

import json
import sys
import types
from unittest import mock

import numpy as np
import pytest


@pytest.fixture
def fake_oft_deps(monkeypatch, tmp_path):
    """Install fake torch/huggingface_hub/transformers modules, then import
    a fresh `oft_backend` against them.

    Returns a dict of the key fakes tests need to assert against.
    """
    # --- fake torch ---
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"  # sentinel value, never inspected for type

    class _NoGradCtx:
        def __enter__(self):
            return None

        def __exit__(self, *exc_info):
            return False

    fake_torch.no_grad = mock.MagicMock(name="torch.no_grad", side_effect=_NoGradCtx)
    # Return value is only ever re-assigned to primary_inputs["pixel_values"]
    # and never inspected by these tests -- any placeholder is fine.
    fake_torch.cat = mock.MagicMock(name="torch.cat")

    # --- fake huggingface_hub ---
    stats_path = tmp_path / "dataset_statistics.json"
    stats_path.write_text(
        json.dumps({"libero_spatial_no_noops": {"action": {"q01": [0] * 7, "q99": [1] * 7}}})
    )
    fake_hf_hub = types.ModuleType("huggingface_hub")
    fake_hf_hub.hf_hub_download = mock.MagicMock(
        name="hf_hub_download", return_value=str(stats_path)
    )

    # --- fake transformers ---
    fake_transformers = types.ModuleType("transformers")

    class FakeBatchFeature(dict):
        """Stand-in for HF's BatchFeature -- a dict with a chainable .to()."""

        def to(self, device, dtype=None):
            return self

    fake_processor = mock.MagicMock(name="fake_processor")

    def _processor_call(prompt, image):
        bf = FakeBatchFeature()
        bf["pixel_values"] = np.array(image)
        return bf

    fake_processor.side_effect = _processor_call

    fake_auto_processor_cls = mock.MagicMock(name="AutoProcessor")
    fake_auto_processor_cls.from_pretrained = mock.MagicMock(
        name="AutoProcessor.from_pretrained", return_value=fake_processor
    )

    fake_model = mock.MagicMock(name="fake_model")
    # .to(device) returns itself, so self.model in the real code IS this
    # same mock -- what lets tests assert on
    # fake_model.vision_backbone.set_num_images_in_input.call_args_list.
    fake_model.to = mock.MagicMock(name="model.to", return_value=fake_model)
    fake_model.predict_action = mock.MagicMock(
        name="predict_action", return_value=(np.zeros((8, 7)), None)
    )

    fake_auto_model_cls = mock.MagicMock(name="AutoModelForVision2Seq")
    fake_auto_model_cls.from_pretrained = mock.MagicMock(
        name="AutoModelForVision2Seq.from_pretrained", return_value=fake_model
    )

    fake_transformers.AutoProcessor = fake_auto_processor_cls
    fake_transformers.AutoModelForVision2Seq = fake_auto_model_cls

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_hub)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    # oft_backend imports these at module load time -- force a fresh import
    # against the fake modules just installed above.
    monkeypatch.delitem(sys.modules, "libero.libero.libero.vla.oft_backend", raising=False)
    monkeypatch.delitem(sys.modules, "libero.vla.oft_backend", raising=False)

    import importlib

    mod = importlib.import_module("libero.vla.oft_backend")

    # prismatic isn't installed locally and tests must not touch the
    # network -- short-circuit _ensure_prismatic()'s clone-and-retry path.
    monkeypatch.setattr(mod, "_prismatic_ok", lambda: True)

    return {
        "mod": mod,
        "fake_processor": fake_processor,
        "fake_model": fake_model,
        "fake_torch": fake_torch,
        "fake_hf_hub": fake_hf_hub,
    }


def test_init_activates_num_images_in_input_two(fake_oft_deps):
    """Test 1 (05-UAT.md Gap 1, fix 1): __init__ calls
    self.model.vision_backbone.set_num_images_in_input(2) exactly once,
    after the model is loaded and moved to device -- proves the missing
    activation call that caused the live `split_with_sizes=[3, 3]`
    RuntimeError is now made."""
    mod = fake_oft_deps["mod"]
    fake_model = fake_oft_deps["fake_model"]

    mod.OFTBackend(checkpoint="fake/checkpoint", device="cpu")

    assert fake_model.vision_backbone.set_num_images_in_input.call_args_list == [mock.call(2)]


def test_predict_orders_agentview_as_primary_eye_in_hand_as_extra(fake_oft_deps):
    """Test 2 (05-UAT.md Gap 1, fix 2): predict() invokes self.processor with
    the agentview-derived image FIRST (as primary) and the eye_in_hand-
    derived image SECOND (as the sole extra view) -- proves the inverted
    primary/extra_views assignment is fixed. Also confirms the returned
    action chunk shape contract (8, 7) is unchanged."""
    mod = fake_oft_deps["mod"]
    fake_processor = fake_oft_deps["fake_processor"]

    backend = mod.OFTBackend(checkpoint="fake/checkpoint", device="cpu")

    eye_in_hand_image = np.zeros((4, 4, 3), dtype=np.uint8)
    agentview_image = np.full((4, 4, 3), 255, dtype=np.uint8)

    actions = backend.predict(
        {"eye_in_hand": eye_in_hand_image, "agentview": agentview_image},
        "pick up the bowl",
    )

    assert fake_processor.call_args_list is not None
    calls = fake_processor.call_args_list
    assert len(calls) == 2

    first_call_image = np.array(calls[0].args[1])
    second_call_image = np.array(calls[1].args[1])

    assert np.array_equal(first_call_image, agentview_image)
    assert np.array_equal(second_call_image, eye_in_hand_image)

    assert actions.shape == (8, 7)
