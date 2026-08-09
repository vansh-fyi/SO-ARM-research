"""Case-insensitive-checkout compatibility shim for this directory's tests.

Several modules under ``LIBERO/libero/libero/`` (``bddl_utils.py``,
``env_wrapper.py``, ``hdf5_writer.py``, ``raw_recorder.py``, ``replay.py``,
...) use the ABSOLUTE import form ``import LIBERO.libero.libero.envs...``
rather than a relative ``libero.envs...`` form. This resolves fine on a
normal checkout, where the top-level directory is literally named
``LIBERO`` (uppercase) on disk.

Some git worktrees on macOS (case-insensitive, case-preserving APFS) get
checked out with the top-level directory cased as ``libero`` (lowercase)
instead -- git's own ``core.ignorecase`` matching treats both spellings as
the same path, but CPython's import machinery does an exact, case-sensitive
directory-entry scan when resolving ``import LIBERO``, so it fails to find a
lowercase ``libero`` directory under that name. This is a checkout artifact
of this worktree, not a change to the tracked repo layout (see
``.planning/STATE.md``'s "mystery-sync" note on this class of issue).

This conftest.py runs before test collection in this directory (and its
subdirectories, e.g. ``envs/``, ``datasets/``) and transparently aliases
``sys.modules["LIBERO"]`` to the already-importable on-disk package ONLY
when the literal ``LIBERO`` import fails -- a no-op on a normal-cased
checkout.
"""

import os
import sys
import types

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if "LIBERO" not in sys.modules:
    try:
        import LIBERO  # noqa: F401  (succeeds on a normal-cased checkout)
    except ModuleNotFoundError:
        # os.path.isdir resolves "LIBERO" case-insensitively against the
        # on-disk `libero` directory (APFS); Python's import machinery does
        # not, so hand-build a namespace-package module object whose
        # __path__ points at the real directory. Everything nested below
        # this (``LIBERO.libero.libero.envs...``) is already lower-case in
        # both the import statements and on disk, so normal import
        # resolution handles the rest once this top-level alias exists.
        _outer_dir = os.path.join(_REPO_ROOT, "LIBERO")
        _ns_pkg = types.ModuleType("LIBERO")
        _ns_pkg.__path__ = [_outer_dir]
        sys.modules["LIBERO"] = _ns_pkg
