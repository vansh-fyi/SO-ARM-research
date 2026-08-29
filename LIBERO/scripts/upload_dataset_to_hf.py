"""One-time upload of Phase 4's SOARM demo dataset to a private HF Hub dataset repo.

LIBERO/libero/datasets/soarm_spatial/ is gitignored (LIBERO/.gitignore:8), so a
fresh `git clone` on a Colab VM never includes the raw HDF5 demos. This script
pushes the *_demo.hdf5 files and dataset_statistics.json to a private HF Hub
dataset repo so 06a-finetune.ipynb can pull them back down before RLDS
conversion (see the "Download dataset from HF Hub" cell in that notebook).

Usage (run once, from the repo root, after `pip install huggingface_hub`):
    python LIBERO/scripts/upload_dataset_to_hf.py

Requires an HF Hub token with WRITE access (huggingface.co/settings/tokens).
You'll be prompted via getpass() -- the token is never written to disk here.
"""
import getpass
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "libero" / "datasets" / "soarm_spatial"
UPLOAD_PATTERNS = ["*_demo.hdf5", "dataset_statistics.json"]


def main() -> None:
    token = getpass.getpass("HF Hub token (Write role, huggingface.co/settings/tokens): ")
    api = HfApi(token=token)
    username = api.whoami()["name"]
    repo_id = f"{username}/soarm-spatial-demos"

    files = [f for pattern in UPLOAD_PATTERNS for f in sorted(DATASET_DIR.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files matching {UPLOAD_PATTERNS} found in {DATASET_DIR}")

    print(f"Creating/reusing private dataset repo: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)

    for f in files:
        size_mb = f.stat().st_size / 1e6
        print(f"Uploading {f.name} ({size_mb:.1f} MB) -> {repo_id} ...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=repo_id,
            repo_type="dataset",
        )

    print(f"Saved → https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
