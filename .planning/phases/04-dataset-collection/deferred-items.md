# Deferred Items — Phase 4

Out-of-scope discoveries logged per the executor's scope boundary (do NOT fix
inline; recorded here for a future task).

## 2026-08-03 (04-02, 3rd resolution attempt)

- **`test_hdf5_writer.py::test_schema_and_obs_key_naming`** — pre-existing
  failure (`assert 2 == 1`), confirmed present BEFORE this session's changes
  (verified via `git stash` + re-run on the prior commit). Unrelated to the
  BDDL repositioning / `collector.py` `Z_TOL` tightening done in this attempt
  (no files touched in that test's dependency path). Not fixed here — out of
  scope per the executor's scope boundary rule (only fix issues directly
  caused by the current task's changes).
