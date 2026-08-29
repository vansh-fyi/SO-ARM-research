"""FinetunedOFTBackend: reload a HF Hub LoRA adapter through OFTBackend's
already-proven loading path (D-01, Pattern 4).

Subclasses OFTBackend and reuses its entire prismatic-guard + bf16-load +
`set_num_images_in_input(2)` + base-checkpoint norm_stats overlay unchanged
via `super().__init__()`. Additionally downloads the fine-tuned adapter from
`adapter_repo_id`, merges it via `peft.PeftModel.merge_and_unload()`,
re-applies dual-image mode (Assumption A4 — merge_and_unload()'s resulting
model tree may not expose `.vision_backbone` at the same location; this call
will fail loudly if so, which is the intended fail-fast behavior), and
overlays THIS adapter's OWN `dataset_statistics.json` (Pitfall 6 — must NOT
reuse Phase 4's raw-HDF5-derived stats file, which can silently diverge from
the post-RLDS-conversion stats the adapter was actually trained against).

`predict()` is inherited verbatim from `OFTBackend` — the entire point of
this design is that `eval_loop.run_suite` drives both backends through the
identical `predict(images, language)` call with zero branching.

Like `oft_backend.py`, this module does not run any loading/network/GPU
logic at import time — all of that happens inside `FinetunedOFTBackend.__init__`.
This file is Colab-only (torch/peft/huggingface_hub/transformers are Colab-only
GPU dependencies per this project's established convention) — `vla/__init__.py`'s
guarded try/except wraps the whole import, not this file internally.
"""

import json
import os

from huggingface_hub import hf_hub_download, snapshot_download
from peft import PeftModel

from .oft_backend import CHECKPOINT, OFTBackend


class FinetunedOFTBackend(OFTBackend):
    """OpenVLA-OFT + fine-tuned LoRA adapter VLA backend (TUNE-03, D-01)."""

    def __init__(self, adapter_repo_id: str, checkpoint: str = CHECKPOINT, device: str = "cuda"):
        # Reuse OFTBackend's entire loading path unchanged: prismatic guard,
        # bf16 model load, set_num_images_in_input(2), base-checkpoint
        # norm_stats overlay, unnorm_key resolution.
        super().__init__(checkpoint=checkpoint, device=device)

        print(f"Downloading fine-tuned adapter from {adapter_repo_id}...")
        checkpoint_dir = snapshot_download(adapter_repo_id)
        # push_checkpoint_to_hub (06a-finetune.ipynb) uploads finetune.py's
        # entire checkpoint directory verbatim -- the actual PEFT adapter
        # (adapter_config.json + adapter_model.safetensors) lives in that
        # checkpoint's own "lora_adapter/" subfolder, not at the repo root.
        # Confirmed live via HfApi().list_repo_files(adapter_repo_id).
        adapter_dir = os.path.join(checkpoint_dir, "lora_adapter")

        print(f"Merging LoRA adapter from {adapter_dir} into base checkpoint...")
        self.model = PeftModel.from_pretrained(self.model, adapter_dir).merge_and_unload()

        # Assumption A4 (06-RESEARCH.md): merge_and_unload()'s resulting
        # model tree may not expose `.vision_backbone` at the same location
        # as the pre-merge model. This call is left unguarded on purpose —
        # an AttributeError here is the intended fail-fast signal that A4's
        # assumption broke, not something to silently skip.
        self.model.vision_backbone.set_num_images_in_input(2)

        # Pitfall 6: overlay THIS adapter's OWN dataset_statistics.json —
        # NOT the base checkpoint's (which super().__init__() already set
        # and which we now overwrite). The adapter's stats reflect the
        # post-RLDS-conversion training data, not the checkpoint's OXE
        # pretraining stats or Phase 4's raw-HDF5 stats.
        stats_path = hf_hub_download(adapter_repo_id, "dataset_statistics.json")
        with open(stats_path) as f:
            self.model.norm_stats = json.load(f)
        print(
            f"norm_stats overlaid from adapter's own dataset_statistics.json "
            f"({adapter_repo_id}): {list(self.model.norm_stats.keys())}"
        )

        # Re-resolve unnorm_key against these NEW adapter-specific stats —
        # the base checkpoint's resolved key (OFTBackend's zero-shot
        # "libero_spatial", the stock LIBERO dataset name) does not exist
        # here. finetune.py's own dataset_statistics.json is keyed by
        # THIS project's actual registered dataset name (oxe_register.py's
        # "soarm_spatial", RLDS_DATASET_NAME in 06a-finetune.ipynb) --
        # confirmed live: Available: ['soarm_spatial'], not
        # 'libero_spatial'/'libero_spatial_no_noops'. A single-dataset LoRA
        # fine-tune (this project's setup, D-04 -- one HDF5 = one task)
        # always has exactly one key in this dict, so use whichever one is
        # actually present rather than hardcoding a name that only applies
        # to the zero-shot base checkpoint.
        available_keys = list(self.model.norm_stats.keys())
        if len(available_keys) != 1:
            raise KeyError(
                f"Expected exactly 1 key in adapter norm_stats (single-dataset "
                f"fine-tune), found {len(available_keys)}: {available_keys}"
            )
        self.unnorm_key = available_keys[0]
        print(f"Using unnorm_key: {self.unnorm_key} (adapter-specific stats)")

    # predict() is intentionally NOT overridden — inherited verbatim from
    # OFTBackend, which drives eval_loop.run_suite identically for both
    # backends.
