# dlimp Install Patch — `01-colab-env-setup.ipynb`

> **CLOSED (2026-07-10).** All fixes baked into the notebook (commits 96cbb0d, b095ee6,
> dbcf69a); ENV-01/02/03 all PASS on a fresh Colab A100. The dlimp fix evolved beyond
> this doc: kvablack/dlimp also pins tensorflow==2.15.0, so the working install is
> `pip install --no-deps -e`, plus a protobuf restore and five more eager-chain deps
> (see Step 5b in the notebook and `.planning/notes/dlimp-install-forensics.md`).
> Kept for the historical import-chain analysis below.

## Context

This repo is `SoARM-Research` — a robotics research project running OpenVLA-OFT inference + LIBERO simulation on a Colab A100 (80 GB VRAM).

The primary setup notebook is:
```
LIBERO/notebooks/01-colab-env-setup.ipynb
```

It has two blocks:
- **Block A** — installs all dependencies (run once, then restart runtime)
- **Block B** — verifies the environment (ENV-01 / ENV-02 / ENV-03)

---

## The Problem

ENV-03 (the final verification cell) loads `moojink/openvla-7b-oft-finetuned-libero-spatial`
via HuggingFace and runs a forward pass to confirm a `(8, 7)` action chunk output.

It currently fails with:

```
ModuleNotFoundError: No module named 'dlimp'
```

### Root Cause

The import chain is:

```
prismatic/__init__.py
  → prismatic.models (line 1)
    → prismatic.models.load
      → prismatic.models.vlas.openvla
        → prismatic.vla.action_tokenizer
          → prismatic.vla.__init__           ← eager import of full dataset stack
            → prismatic.vla.materialize
              → prismatic.vla.datasets
                → prismatic.vla.datasets.rlds.dataset
                  → import dlimp             ← FAILS here
```

`dlimp` is a required transitive dependency of `openvla-oft` (declared in its `pyproject.toml`
as `dlimp @ git+https://github.com/moojink/dlimp_openvla`). However:

1. **Block A never installs it** — the openvla-oft editable install silently skips git-URL deps.
2. **`pip install git+https://github.com/moojink/dlimp_openvla` fails** — returns non-zero exit
   code (pip cannot build/install from that URL directly on Colab Python 3.12).

### What Does Work

> **Correction (2026-07-10):** the plain editable install below **no longer works** —
> `kvablack/dlimp`'s `setup.py` also pins `tensorflow==2.15.0`, so dependency resolution
> fails identically on Colab py3.12 (`CalledProcessError` observed live). The working
> form requires `--no-deps`:
>
> ```bash
> git clone --depth 1 https://github.com/kvablack/dlimp /content/dlimp_kvablack
> pip install --no-deps -e /content/dlimp_kvablack
> ```
>
> dlimp's runtime deps (tensorflow, tensorflow-datasets) come from Colab's preinstalled
> packages — the exact combination the passing ENV-03 actually ran on. The notebook's
> Step 5b cell verifies both are present and installs them only if missing.

Cloning the **parent repo** (`kvablack/dlimp`) and installing editably:

```bash
git clone --depth 1 https://github.com/kvablack/dlimp /content/dlimp_kvablack
pip install -e /content/dlimp_kvablack
```

This was confirmed in a live Colab A100 session — after running the above, ENV-03 produced:

```
Model dtype: torch.bfloat16  device: cuda:0
norm_stats overlaid from dataset_statistics.json: ['libero_spatial_no_noops']
Using unnorm_key: libero_spatial_no_noops
ENV-03: PASS — action shape: (8, 7), dtype: float64
First action step: [0.934... 0.872... 0.928... 0.103... 0.176... 0.145... 0.996...]
```

> **Note on unnorm_key:** The fine-tune checkpoint's `dataset_statistics.json` uses the key
> `libero_spatial_no_noops` (not `libero_spatial`). The notebook's existing fallback logic
> handles this correctly — no change needed there.

---

## The Fix

### Where to add it

In `LIBERO/notebooks/01-colab-env-setup.ipynb`, **Block A**, in or after the Step 5
cell that installs the openvla-oft packages (cell ID `cell-8-openvla-fork`).

> **Correction (2026-07-10):** this doc previously named cell ID `98b86f1a` as the
> Block A openvla-oft install cell — that ID is actually the Block B prismatic guard.

The dlimp install must happen **in Block A** (before the runtime restart), so it is present
on disk when Block B's ENV-03 cell runs its `prismatic` import.

### Code to prepend to cell `98b86f1a`

```python
# ── dlimp install (required transitive dep of openvla-oft) ─────────────────
# pip install git+https://... fails for this package; clone-then-editable is required.
# Must run in Block A so the package is on disk before Block B's prismatic import.
import subprocess as _sp, sys as _sys, pathlib as _pl

_DLIMP_DIR = _pl.Path("/content/dlimp_kvablack")
if not _DLIMP_DIR.exists():
    print("Cloning kvablack/dlimp ...")
    _sp.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/kvablack/dlimp", str(_DLIMP_DIR)],
        check=True
    )
_sp.run([_sys.executable, "-m", "pip", "install", "-q", "-e", str(_DLIMP_DIR)], check=True)
print("dlimp installed from kvablack/dlimp ✓")
# ───────────────────────────────────────────────────────────────────────────
```

### Alternative: add as a standalone Block A cell

If you prefer not to modify the existing openvla-oft cell, add a **new code cell** in Block A
immediately after the openvla-oft install cell and before the `--- STOP — Restart runtime now ---`
markdown cell.

---

## Files Changed

| File | Status | Action |
|------|--------|--------|
| `LIBERO/notebooks/01-colab-env-setup.ipynb` | ✅ Patched (2026-07-10) | New Block A cell `cell-8b-dlimp` (Step 5b) added after `cell-8-openvla-fork`; failing Block B `pip install git+.../dlimp_openvla` cell (`89cc2bac`) deleted; stale `print("dlimp installed")` removed from the prismatic guard |

> Applied via `json` scripting against the notebook. Root cause of the git-URL install
> failure confirmed from live output: `dlimp_openvla` pins `tensorflow==2.15.0`, which has
> no Python 3.12 wheel on Colab (TF ships cp312 wheels only from 2.16+). See
> `.planning/notes/dlimp-install-forensics.md` for the full evidence trail.

---

## Verification

After applying the patch, a clean Colab run should produce this in ENV-03:

```
dlimp installed from kvablack/dlimp ✓
...
ENV-03: PASS — action shape: (8, 7), dtype: float64
```

The `(8, 7)` shape is the OFT action chunk: 8 timesteps × 7 DoF (6 joint + gripper).

---

## UAT Record

Full test history is in:
```
.planning/phases/01-colab-environment-setup/01-UAT.md
```

Summary:
- ENV-01 ✅ — all pipeline packages installed, numpy 1.26.4, ABI canary OK
- ENV-02 ✅ — LIBERO EGL render, mean pixel 118.99 (non-black)
- ENV-03 ✅ — OpenVLA-OFT model load + action chunk (after manual dlimp fix)
- numpy ABI gate ✅ — numpy 1.26.4 coherent

~~Only remaining task: bake the dlimp fix permanently into Block A of the notebook.~~
**Done (2026-07-10).** Remaining verification: one clean run from a **deleted** runtime
(Runtime → Disconnect and delete runtime — a mere restart keeps the VM disk and would
invalidate the test): Block A → restart → Block B, expecting ENV-01/02/03 all PASS with
zero manual intervention.
