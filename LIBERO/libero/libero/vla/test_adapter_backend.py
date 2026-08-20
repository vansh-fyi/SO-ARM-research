"""Mock-based pytest suite for FinetunedOFTBackend (D-01, Pitfall 6).

Mirrors `test_oft_backend.py`'s `fake_oft_deps` fixture technique EXACTLY
(sys.modules injection of fake torch/huggingface_hub/transformers), extended
with a fake `peft` module so `adapter_backend.py`'s merge step can be
verified locally without any GPU/network/real model weights.

NOTE on import path: mirrors test_oft_backend.py's / test_eval_loop.py's
documented convention — pytest's default rootdir-walking collection mode
treats this file's nearest ancestor without an __init__.py (LIBERO/libero/,
since LIBERO/libero/libero/__init__.py exists) as the insertion point,
making this package resolvable as `libero.vla.*` when invoked via
`pytest LIBERO/libero/libero/vla/test_adapter_backend.py` from the repo root.
"""

import json
import sys
import types
from unittest import mock

import numpy as np
import pytest

BASE_CHECKPOINT = "fake/base-checkpoint"
ADAPTER_REPO_ID = "fake-user/soarm-oft-lora-20260820-000000"


@pytest.fixture
def fake_adapter_deps(monkeypatch, tmp_path):
    """Install fake torch/huggingface_hub/transformers/peft modules, then
    import a fresh `adapter_backend` (and its `oft_backend` dependency)
    against them.

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
    fake_torch.cat = mock.MagicMock(name="torch.cat")

    # --- fake huggingface_hub ---
    # Two DISTINCT dataset_statistics.json files -- base-checkpoint's (fetched
    # during super().__init__()) and the adapter's OWN (fetched afterward,
    # Pitfall 6) -- with different unnorm_key content so tests can prove the
    # adapter-specific file (not the base one) is what ends up in
    # self.model.norm_stats.
    base_stats_path = tmp_path / "base_dataset_statistics.json"
    base_stats_path.write_text(
        json.dumps({"libero_spatial_no_noops": {"action": {"q01": [0] * 7, "q99": [1] * 7}}})
    )
    adapter_stats_path = tmp_path / "adapter_dataset_statistics.json"
    adapter_stats_path.write_text(
        json.dumps({"libero_spatial": {"action": {"q01": [-1] * 7, "q99": [2] * 7}}})
    )

    def _hf_hub_download_side_effect(repo_id, filename):
        if repo_id == ADAPTER_REPO_ID:
            return str(adapter_stats_path)
        return str(base_stats_path)

    fake_hf_hub = types.ModuleType("huggingface_hub")
    fake_hf_hub.hf_hub_download = mock.MagicMock(
        name="hf_hub_download", side_effect=_hf_hub_download_side_effect
    )
    fake_hf_hub.snapshot_download = mock.MagicMock(
        name="snapshot_download", return_value="/fake/adapter/dir"
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
    # same mock during super().__init__() -- what lets tests assert on
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

    # --- fake peft ---
    # PeftModel.from_pretrained(base_model, adapter_dir) returns a merged-
    # model placeholder whose .merge_and_unload() returns fake_model itself
    # -- so self.model after the merge is STILL fake_model, letting tests
    # assert on the SAME mock's set_num_images_in_input call count (once
    # from super().__init__(), once from the post-merge re-application).
    fake_peft = types.ModuleType("peft")
    merged_placeholder = mock.MagicMock(name="PeftModel_instance")
    merged_placeholder.merge_and_unload = mock.MagicMock(
        name="merge_and_unload", return_value=fake_model
    )
    fake_peft_model_cls = mock.MagicMock(name="PeftModel")
    fake_peft_model_cls.from_pretrained = mock.MagicMock(
        name="PeftModel.from_pretrained", return_value=merged_placeholder
    )
    fake_peft.PeftModel = fake_peft_model_cls

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_hub)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    # oft_backend/adapter_backend import these at module load time -- force
    # a fresh import against the fake modules just installed above.
    monkeypatch.delitem(sys.modules, "libero.libero.libero.vla.oft_backend", raising=False)
    monkeypatch.delitem(sys.modules, "libero.vla.oft_backend", raising=False)
    monkeypatch.delitem(sys.modules, "libero.libero.libero.vla.adapter_backend", raising=False)
    monkeypatch.delitem(sys.modules, "libero.vla.adapter_backend", raising=False)

    import importlib

    oft_mod = importlib.import_module("libero.vla.oft_backend")
    # prismatic isn't installed locally and tests must not touch the
    # network -- short-circuit _ensure_prismatic()'s clone-and-retry path.
    monkeypatch.setattr(oft_mod, "_prismatic_ok", lambda: True)

    mod = importlib.import_module("libero.vla.adapter_backend")

    return {
        "mod": mod,
        "fake_processor": fake_processor,
        "fake_model": fake_model,
        "fake_torch": fake_torch,
        "fake_hf_hub": fake_hf_hub,
        "fake_peft_model_cls": fake_peft_model_cls,
        "merged_placeholder": merged_placeholder,
    }


def test_init_merges_adapter_and_reapplies_dual_image_mode(fake_adapter_deps):
    """Test 1 (D-01, Pitfall 6): constructing FinetunedOFTBackend calls
    super().__init__() first, merges the adapter via
    PeftModel.from_pretrained(base_model, adapter_dir).merge_and_unload(),
    re-applies set_num_images_in_input(2) after the merge, and overlays the
    ADAPTER's own dataset_statistics.json (not the base checkpoint's)."""
    mod = fake_adapter_deps["mod"]
    fake_model = fake_adapter_deps["fake_model"]
    fake_hf_hub = fake_adapter_deps["fake_hf_hub"]
    fake_peft_model_cls = fake_adapter_deps["fake_peft_model_cls"]

    backend = mod.FinetunedOFTBackend(
        adapter_repo_id=ADAPTER_REPO_ID, checkpoint=BASE_CHECKPOINT, device="cpu"
    )

    # PeftModel.from_pretrained was called with the base-checkpoint's loaded
    # model (fake_model, since fake_model.to() returns itself) and the
    # adapter_dir snapshot_download resolved.
    assert fake_peft_model_cls.from_pretrained.call_args_list == [
        mock.call(fake_model, "/fake/adapter/dir")
    ]

    # set_num_images_in_input(2) called TWICE: once from the inherited
    # OFTBackend.__init__, once from the re-application after merge (A4).
    assert fake_model.vision_backbone.set_num_images_in_input.call_args_list == [
        mock.call(2),
        mock.call(2),
    ]

    # The SECOND dataset_statistics.json fetch used adapter_repo_id (NOT the
    # base checkpoint) as its first positional arg.
    adapter_stats_calls = [
        c for c in fake_hf_hub.hf_hub_download.call_args_list
        if c.args and c.args[0] == ADAPTER_REPO_ID
    ]
    assert len(adapter_stats_calls) == 1
    assert adapter_stats_calls[0].args[1] == "dataset_statistics.json"

    # The adapter's OWN stats (unnorm_key "libero_spatial", q01=[-1]*7) are
    # what ends up in self.model.norm_stats -- not the base checkpoint's
    # ("libero_spatial_no_noops", q01=[0]*7).
    assert "libero_spatial" in backend.model.norm_stats
    assert backend.model.norm_stats["libero_spatial"]["action"]["q01"] == [-1] * 7
    assert backend.unnorm_key == "libero_spatial"


def test_predict_inherited_from_oft_backend_returns_8x7_chunk(fake_adapter_deps):
    """Test 2 (D-01): predict() is NOT overridden -- inherited verbatim from
    OFTBackend, proven via the same fake_model.predict_action mock
    OFTBackend's own tests already exercise, returning an (8, 7) chunk."""
    mod = fake_adapter_deps["mod"]

    backend = mod.FinetunedOFTBackend(
        adapter_repo_id=ADAPTER_REPO_ID, checkpoint=BASE_CHECKPOINT, device="cpu"
    )

    eye_in_hand_image = np.zeros((4, 4, 3), dtype=np.uint8)
    agentview_image = np.full((4, 4, 3), 255, dtype=np.uint8)

    actions = backend.predict(
        {"eye_in_hand": eye_in_hand_image, "agentview": agentview_image},
        "pick up the bowl",
    )

    assert actions.shape == (8, 7)
