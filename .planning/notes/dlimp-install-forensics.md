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

## Consequence

The confirmed fix (kvablack clone + editable install) must be baked into Block A
and the moojink cell deleted — tracked in
[bake-dlimp-fix-into-block-a](../todos/pending/bake-dlimp-fix-into-block-a.md).
Any future "clean run" claim for Phase 1 must start from a deleted runtime, not a
restarted one.
