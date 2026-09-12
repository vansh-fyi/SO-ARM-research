# Phase 1 Debug History & Environment Contract

**Audience: future phase agents (Phase 2–6).** This document exists so you do not
re-learn Phase 1's lessons by crashing into them. Read the Environment Contract before
touching the Colab setup; read the Failure Timeline before "fixing" anything in it.

Related: [`01-UAT.md`](01-UAT.md) (test record) · [`DLIMP-PATCH.md`](DLIMP-PATCH.md)
(original dlimp analysis, superseded in parts) ·
[`.planning/notes/dlimp-install-forensics.md`](../../notes/dlimp-install-forensics.md)
(evidence trail)

---

## 1. The Environment Contract (invariants — do not break these)

The deliverable of Phase 1 is `libero/notebooks/01-colab-env-setup.ipynb`. Every future
phase starts a Colab session by running it: **Block A (installs) → Restart runtime →
Block B (verification gates ENV-01/02/03)**. These invariants are load-bearing:

1. **numpy is pinned to exactly 1.26.4** and enforced by the Block A final gate cell
   (`cell-9b-numpy-abi-gate`), which purges, reinstalls, and ABI-probes numpy as the
   LAST Block A operation. **Any pip install added later must run BEFORE this gate**
   (or be followed by re-running it). Installing packages in Block B / mid-session can
   silently pull numpy 2.x and poison the next kernel with
   `ValueError: numpy.dtype size changed`.

2. **Block B cell order is mandatory:** EGL bootstrap (sets `MUJOCO_GL` etc.) before
   ANY MuJoCo/robosuite import; `~/.libero/config.yaml` bootstrap before any
   `import libero`; `USE_TORCH=1 / USE_TF=0 / USE_FLAX=0` before the FIRST transformers
   import (transformers caches backend availability at first import and never
   re-checks).

3. **openvla-oft is installed `--no-deps`** — deliberately. Its declared dependency set
   is unsatisfiable on Colab Python 3.12 (`tensorflow==2.15.0` has no cp312 wheel;
   `dlimp @ git+.../dlimp_openvla` fails for the same reason). Consequence: **its
   dependencies must be provided explicitly.** Step 5b (`cell-8b-dlimp`) carries the
   complete set, enumerated from the prismatic source's eager import chain:
   - dlimp: `git clone kvablack/dlimp` + `pip install --no-deps -e` (the parent repo
     carries the SAME tensorflow pin — plain `-e` also fails)
   - `tensorflow-graphics==2021.12.3` `--no-deps` (its deps include OpenEXR, a C++
     source build that fails on Colab, and tensorflow-addons, deprecated with no cp312
     wheel; the only submodule prismatic uses — `geometry.transformation` — was
     verified from the wheel to need neither)
   - `draccus==0.8.0`, `jsonlines`, `wandb`, `diffusers==0.30.3`
   - tensorflow + tensorflow-datasets come from **Colab stock** — never reinstall or
     pin them

4. **protobuf must end Block A at Colab-stock level or newer.** Block A installs drag
   the protobuf runtime below the gencode version of Colab's `tensorflow_metadata`
   (observed: runtime 5.29.6 vs gencode 6.31.1 → `VersionError` inside the
   dlimp → tensorflow_datasets import). Step 5b's LAST pip operation is
   `pip install -U protobuf`. Keep it last; anything installed after it may downgrade
   protobuf again.

5. **If you add ANY new dependency in a future phase, it goes in Block A** (before the
   numpy gate), not in Block B, not ad-hoc mid-session. Check first whether it pins
   tensorflow, numpy, or protobuf.

6. **HF token** is optional (all Phase 1 downloads are public) and is read by the
   Block B bootstrap cell from `/content/drive/MyDrive/SoARM-Research/.hf_token`,
   mounting Drive if needed. **Never use `google.colab.userdata` (secrets vault)** —
   it blocks indefinitely from VS Code-attached sessions. Never write a literal token
   into the notebook: cell outputs get committed to git in this repo.

7. **Facts ENV-03 established that Phase 3 (inference loop) must honor:**
   - `predict_action` output is an **action chunk of shape (8, 7)** float64 —
     8 timesteps × 7 DoF; consume chunks, not single steps
   - norm stats must be overlaid from the checkpoint's `dataset_statistics.json`
     after `from_pretrained`; the actual unnorm key is **`libero_spatial_no_noops`**
     (not `libero_spatial`)
   - bf16 + `low_cpu_mem_usage=True` on A100; no quantization needed (D-05)

---

## 2. Failure Timeline (what actually happened, in order)

Every ENV-03 failure was **disk state**, never cell order. Each fix exposed the next
missing layer of the import chain
(`prismatic → dlimp → tensorflow_datasets → protobuf → tensorflow_graphics`).

| # | Symptom | Root cause | Fix | Commit |
|---|---------|-----------|-----|--------|
| 1 | `ModuleNotFoundError: dlimp` at ENV-03 | openvla-oft installed `--no-deps`; dlimp never arrives. `pip install git+.../dlimp_openvla` always fails: it pins `tensorflow==2.15.0`, which has no Python 3.12 wheel (Colab's TF options start at 2.16) | Clone kvablack/dlimp, install editable | 96cbb0d |
| 2 | A Block B "patch cell" appeared to fix #1 | **Illusion.** The cell failed (return code 1) and installed nothing; ENV-03 passed off a manual install persisting on the VM disk across "Restart runtime" | Cell deleted; forensics documented | 96cbb0d |
| 3 | `CalledProcessError` from the baked Step 5b (`pip install -e dlimp_kvablack`) | kvablack/dlimp's `setup.py` carries the **same** `tensorflow==2.15.0` pin — any deps-resolving install of either dlimp repo dies on py3.12 | `pip install --no-deps -e`; verify tensorflow/tfds present from Colab stock | 96cbb0d |
| 4 | `protobuf VersionError: gencode 6.31.1 runtime 5.29.6` at dlimp → tfds import | Block A installs (LIBERO editable / transformers fork resolution) downgraded protobuf below Colab's tensorflow_metadata gencode | `pip install -U protobuf` as Step 5b's last op (runtime ≥ gencode is always allowed) | 96cbb0d |
| 5 | `ModuleNotFoundError: tensorflow_graphics` on a genuinely fresh VM | More `--no-deps` fallout — and previous "passing" sessions had it from manual debug installs (illusion #2 again) | Stopped whack-a-mole: enumerated ALL third-party imports in `prismatic/` from source, diffed against Block A + Colab stock, installed the full missing set (`_OFT_DEPS` loop) | b095ee6 |
| 6 | HF token cell hung ("nothing is happening") | `userdata.get()` blocks outside the Colab UI; original cell tried the vault before the Drive file | Reordered env var → Drive file → (vault removed entirely); cell mounts Drive itself | b3c173a, dbcf69a |
| 7 | `FileNotFoundError` writing the token file | Drive not mounted / folder absent | Mount + `mkdir(parents=True)` in the setup snippet; automount baked into the bootstrap cell | dbcf69a |

Final state: **ENV-01, ENV-02, ENV-03 all PASS** on a fresh A100 VM (2026-07-10);
outputs committed as evidence (1a37159). UAT 4/4.

---

## 3. Meta-lessons (how to debug this environment without fooling yourself)

1. **"Restart runtime" is NOT a clean slate.** It restarts the Python kernel; the VM
   disk (site-packages, `/content`) survives. Only **"Disconnect and delete runtime"**
   gives a fresh machine. Twice in Phase 1, manual debug installs persisted across
   restarts and made broken cells look like working fixes. Any "it works now" claim
   is meaningless unless it comes from a deleted runtime.

2. **Disk state vs. kernel state.** pip changes disk; the running kernel keeps its
   already-imported (possibly stale) modules. After installing C-extension packages
   (numpy, protobuf), restart before trusting behavior. Conversely: an
   import-time failure that "goes away" after restart+reinstall was a disk problem,
   not a cell-ordering problem.

3. **Never trust a pip cell that only prints its return code.** The original patch
   cell used `capture_output=True` without `check=True` — it failed every run,
   silently, while taking credit for a pass. Use `check=True`, and avoid `-q` where a
   failure would need diagnosis.

4. **Kill whack-a-mole with source enumeration.** After the second
   `ModuleNotFoundError`, the winning move was grepping every `import` in
   `prismatic/` and diffing against installed packages — finding all five gaps at
   once — instead of fixing one crash per Colab round-trip (each costing a ~40 min
   Block A run).

5. **Verify claims against artifacts, not docs.** DLIMP-PATCH.md's "confirmed
   working" install no longer worked (the parent repo had the same broken pin), and
   its cell-ID reference was wrong. The wheel METADATA, the repo's `setup.py`/
   `pyproject.toml`, and live tracebacks are the ground truth.

6. **`--no-deps` is a contract, not a flag.** Every `--no-deps` install transfers
   responsibility for the dependency tree to you, permanently. Document the full dep
   set next to the install (as Step 5b does) or the next fresh VM will find the gap.

7. **Cell outputs are committed in this repo.** Great for evidence (ENV-03 PASS lives
   in git); dangerous for secrets — a pasted HF token in a traceback would be
   published. The exposed Phase 1 token was ordered revoked; `**/.env` is gitignored.

---

## 4. Quick reference: what a healthy session looks like

```
Block A (fresh VM, ~40 min):
  GPU assert → apt EGL → torch 2.2.0 cu121 → mujoco/robosuite/gym →
  LIBERO editable → openvla-oft pkgs + transformers fork (--no-deps for oft) →
  Step 5b: dlimp(--no-deps) + tfds/tf check + _OFT_DEPS + protobuf -U →
  flash-attn (optional) → numpy ABI gate: PASS → RESTART RUNTIME

Block B (after restart, ~5 min + model download):
  EGL bootstrap → HF token (Drive) → libero config.yaml → sys.path →
  ENV-01: PASS (versions + ABI canary) →
  ENV-02: PASS (non-black LIBERO render) →
  ENV-03: PASS — action shape: (8, 7), dtype: float64
```

Anything that deviates from this transcript is a regression — check the Failure
Timeline above before inventing a new fix.
