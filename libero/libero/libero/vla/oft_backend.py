"""OpenVLA-OFT backend implementing VLABackend (VLA-01).

Replicates Phase 1's confirmed Colab loading pattern (LIBERO/notebooks/
01-colab-env-setup.ipynb, cell 23 — ENV-03) as a reusable class:
prismatic-import guard, bf16 model load (D-05: no 4-bit quantization),
norm_stats overlay from `dataset_statistics.json`, and unnorm_key
resolution with the confirmed `libero_spatial_no_noops` fallback.

This module does not run any loading/network/GPU logic at import time —
all of that happens inside `OFTBackend.__init__`. Full GPU-backed
behavioral verification (actual model load + (8, 7) action shape) happens
on Colab in Plan 02's Notebook A VLA-01 cell; this project has no local
GPU (03-RESEARCH.md Environment Availability table).
"""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForVision2Seq, AutoProcessor

CHECKPOINT = "moojink/openvla-7b-oft-finetuned-libero-spatial"
PRISMATIC_REPO = "https://github.com/moojink/openvla-oft.git"


def _prismatic_ok() -> bool:
    try:
        importlib.import_module("prismatic.training.train_utils")
        return True
    except Exception:
        return False


def _ensure_prismatic() -> None:
    """Guard exactly as Notebook 01 cell 23 does: checkpoint's remote code
    (modeling_prismatic.py) imports prismatic.training.train_utils, which
    exists ONLY in the moojink/openvla-oft repo (NOT TRI-ML prismatic-vlms,
    NOT PyPI). If the installed prismatic lacks it, fall back to a repo
    clone on sys.path — the clone shadows any broken install.

    Guarded behind try/except so local no-GPU dev doesn't hard-fail if
    prismatic legitimately isn't installed, but fails loudly with the
    traceback if the clone-and-retry also fails.
    """
    if _prismatic_ok():
        return

    try:
        repo = Path("/content/openvla-oft")
        if not repo.exists():
            print(f"Cloning moojink/openvla-oft for prismatic package -> {repo}")
            subprocess.run(
                ["git", "clone", "--depth", "1", PRISMATIC_REPO, str(repo)],
                check=True,
            )
        for k in [k for k in list(sys.modules) if k.startswith("prismatic")]:
            del sys.modules[k]
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        importlib.import_module("prismatic.training.train_utils")
        print(f"prismatic resolved from {repo} (sys.path)")
    except Exception:
        # Surface the REAL failure — usually a missing dependency in
        # prismatic's import chain (e.g. wandb), not a missing prismatic
        # module itself. Matches Notebook 01's exact error-surfacing
        # behavior (fail loudly, do not swallow).
        import traceback

        traceback.print_exc()
        raise RuntimeError(
            "prismatic import failed — see traceback above for the missing dependency"
        )


class OFTBackend:
    """OpenVLA-OFT VLA backend satisfying the VLABackend Protocol."""

    def __init__(self, checkpoint: str = CHECKPOINT, device: str = "cuda"):
        _ensure_prismatic()

        print(f"Loading processor from {checkpoint}...")
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)

        print("Loading model in bf16...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device)
        self.device = device

        # Checkpoint config's norm_stats holds only the OXE PRETRAINING
        # datasets (bridge_orig, fractal, ...). LIBERO fine-tune statistics
        # live in a separate dataset_statistics.json in the HF repo — must
        # overlay it or unnorm_key="libero_spatial" won't exist (Phase 1
        # confirmed, 03-CONTEXT.md canonical_refs).
        stats_path = hf_hub_download(checkpoint, "dataset_statistics.json")
        with open(stats_path) as f:
            self.model.norm_stats = json.load(f)
        print(f"norm_stats overlaid from dataset_statistics.json: {list(self.model.norm_stats.keys())}")

        # Resolve unnorm_key — fine-tune datasets are often suffixed "_no_noops".
        self.unnorm_key = "libero_spatial"
        if self.unnorm_key not in self.model.norm_stats:
            if f"{self.unnorm_key}_no_noops" in self.model.norm_stats:
                self.unnorm_key = f"{self.unnorm_key}_no_noops"
            else:
                raise KeyError(
                    f"No libero_spatial key in norm_stats. "
                    f"Available: {list(self.model.norm_stats.keys())}"
                )
        print(f"Using unnorm_key: {self.unnorm_key}")

    def predict(self, images: dict, language: str) -> np.ndarray:
        """Return the full (8, 7) OFT action chunk for one eye_in_hand frame.

        Per D-01, `images` is a dict of named camera views; this backend
        only consumes the "eye_in_hand" key. Per D-03, the eval loop (not
        this backend) owns open-loop replay of all 8 returned steps — this
        method returns the full chunk, not just actions[0].
        """
        prompt = f"In: What action should the robot take to {language}?\nOut:"
        inputs = self.processor(prompt, images["eye_in_hand"]).to(
            self.device, dtype=torch.bfloat16
        )

        # OFT predict_action returns (actions, hidden_states) — unpack the
        # tuple. OFT predicts action CHUNKS: shape (8, 7), not a single
        # (7,) action (Phase 1 confirmed).
        with torch.no_grad():
            result = self.model.predict_action(
                **inputs, unnorm_key=self.unnorm_key, do_sample=False
            )

        actions = result[0] if isinstance(result, tuple) else result
        actions = np.asarray(actions)
        assert actions.shape == (8, 7), f"expected (8,7) chunk, got {actions.shape}"
        return actions
