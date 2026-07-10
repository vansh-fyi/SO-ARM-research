---
title: dlimp install forensics — why the moojink pip cell looked like it worked
date: 2026-07-10
context: Phase 1 (01-colab-environment-setup), ENV-03 OpenVLA-OFT model load
---

## Summary

The Block B cell running `pip install git+https://github.com/moojink/dlimp_openvla`
**always fails and installs nothing**, yet ENV-03 passed in the UAT session. The pass
came from a manually installed `kvablack/dlimp` surviving on the VM disk, not from
the cell. Three findings, each verified:

## 1. The moojink URL is structurally unsatisfiable on Colab

Observed output (2026-07-10 session):

```
ERROR: Could not find a version that satisfies the requirement tensorflow==2.15.0
  (from dlimp) (from versions: 2.16.0rc0, 2.16.1, ... 2.21.0)
Return code: 1
```

`moojink/dlimp_openvla` pins `tensorflow==2.15.0`, but Colab runs Python 3.12 and
TensorFlow only ships cp312 wheels from 2.16 onward. This is not flaky — it can
never succeed on current Colab. And pip resolves the full dependency set *before*
installing anything, so a resolution failure is a total no-op: nothing lands in
site-packages.

## 2. dlimp is eagerly required by ENV-03 — no way around it

Verified against `moojink/openvla-oft` HEAD (cloned 2026-07-10). The prismatic
guard's `import prismatic.training.train_utils` triggers:

```
prismatic/__init__.py
  → prismatic.models → models/load.py → models/vlas/openvla.py
    → prismatic.vla.action_tokenizer   (executes prismatic/vla/__init__.py)
      → vla/materialize.py → vla/datasets → datasets/rlds/dataset.py:13
        → import dlimp
```

No lazy import, and the repo does not vendor a `dlimp/` folder — `pyproject.toml:55`
only declares `dlimp @ git+https://github.com/moojink/dlimp_openvla` as a dependency
(which the editable install silently skips). If dlimp is not importable, the guard
cell raises and ENV-03 never runs.

## 3. The Colab gotcha that created the illusion

**"Restart runtime" only restarts the Python kernel — the VM disk survives**,
including site-packages and `/content`. Only "Disconnect and delete runtime" gives
a fresh machine. The UAT session's manual fix
(`git clone kvablack/dlimp && pip install -e`) persisted across the restart, so
ENV-03 passed while the failing pip cell sat next to it taking the credit.

Diagnostic for any live session: `import dlimp; print(dlimp.__file__)` — a path
under `/content/dlimp_kvablack/` means the manual fix is what's active.

## 4. Update (2026-07-10): kvablack/dlimp has the same pin

The first baked-in Block A cell (`pip install -e /content/dlimp_kvablack`) failed live
with `CalledProcessError` — `kvablack/dlimp`'s `setup.py` **also** declares
`install_requires=["tensorflow==2.15.0", ...]`, so its dependency resolution dies the
same way as the moojink URL. The doc's claim that plain editable install "was confirmed
working" does not reproduce on current Colab py3.12.

Working form: `pip install --no-deps -e /content/dlimp_kvablack`, relying on Colab's
preinstalled tensorflow + tensorflow-datasets (which is what the passing ENV-03 actually
ran on — dlimp needs TF importable, not TF 2.15 specifically). The Step 5b cell now does
this and checks both runtime deps are present.

## 5. Update (2026-07-10): protobuf downgrade breaks tensorflow_datasets

With dlimp installing cleanly (`--no-deps`), ENV-03 progressed to the next layer and
failed at dlimp → `tensorflow_datasets` → `tensorflow_metadata` with
`google.protobuf.runtime_version.VersionError: gencode 6.31.1 runtime 5.29.6`.
Colab stock ships a protobuf runtime matching its tensorflow_metadata gencode; some
Block A install (LIBERO editable or the transformers fork resolution) drags protobuf
down to 5.29.6. Protobuf's guarantee is runtime >= gencode, so the fix is restoring
protobuf after the Block A installs: `pip install -U protobuf` (now baked into the
Step 5b cell, after the dlimp install).

Meta-lesson: every ENV-03 failure so far has been **disk state**, not cell order —
each fix moves the import chain one layer deeper
(prismatic → dlimp → tensorflow_datasets → protobuf).

## 6. Update (2026-07-10): the whack-a-mole root cause — openvla-oft --no-deps

Next failure layer: `ModuleNotFoundError: tensorflow_graphics` (from
`prismatic/vla/datasets/rlds/oxe/utils/droid_utils.py`). Root cause of the whole
sequence: Block A installs openvla-oft with `--no-deps` (necessary — its dep list
contains the unsatisfiable `tensorflow==2.15.0` and the broken dlimp git URL), so
NONE of its declared dependencies are ever pulled, and each ENV-03 run fails at the
first missing one.

Ended the game by enumerating every third-party import in prismatic/ from source
and diffing against Block A + Colab stock. Missing set: `tensorflow_graphics`,
`draccus`, `jsonlines`, `wandb`, `diffusers`. Step 5b now installs any of these
that are absent (`_OFT_DEPS` loop).

`tensorflow-graphics==2021.12.3` must itself be `--no-deps`: it declares OpenEXR
(C++ source build, fails on Colab) and tensorflow-addons (deprecated, no cp312
wheel), but the only submodule prismatic imports (`geometry.transformation`) was
verified from the wheel to reference neither — and its `__init__.py` gates heavy
imports behind a docs-only flag.

Sessions that "passed" before these deps were baked in had them from manual
debugging installs — the same VM-persistence illusion as finding no. 3.

## Consequence

The confirmed fix (kvablack clone + editable install) must be baked into Block A
and the moojink cell deleted — tracked in
[bake-dlimp-fix-into-block-a](../todos/pending/bake-dlimp-fix-into-block-a.md).
Any future "clean run" claim for Phase 1 must start from a deleted runtime, not a
restarted one.
