---
status: testing
phase: 07-camera-depth-perception
source: [07-VERIFICATION.md]
started: 2026-09-12T07:05:00Z
updated: 2026-09-12T15:10:00Z
---

## Current Test

number: 1
name: Colab TFDS spot check (DEPTH-03 runtime proof)
expected: |
  Run on Colab (or any environment with `tensorflow_datasets` installed):
  `pytest LIBERO/libero/libero/datasets/test_rlds_converter.py::test_hdf5_to_rlds_writes_tfds_loadable_dataset LIBERO/libero/libero/datasets/test_rlds_converter.py::test_hdf5_to_rlds_against_real_dataset -v`
  Both tests should pass. The first confirms the persisted `dataset_info.json` schema
  declares `agentview_depth` as a `(None, None, 1)` float32 `Tensor`. The second, run
  against the real pre-Phase-7 file
  `LIBERO/libero/datasets/soarm_spatial/put_the_cream_cheese_in_the_bowl_demo.hdf5`
  (confirmed present, confirmed to lack `agentview_depth`), should hit the new
  pre-flight `h5py.File` check and `pytest.skip("real dataset predates Phase 7 depth
  persistence...")` rather than crash with an unhandled `KeyError`.
awaiting: user response

## Tests

### 1. Colab TFDS spot check (DEPTH-03 runtime proof)
expected: |
  Both `tensorflow_datasets`-gated tests in `test_rlds_converter.py` pass on Colab —
  the TFDS schema write/read round-trip works, and the legacy-HDF5-skip branch
  triggers cleanly instead of crashing.
result: [pending]

### 2. Real-camera side-by-side photo comparison for agentview
expected: |
  Once the physical overhead-camera mount (D-01/D-02/D-03) is built and photographed,
  render `agentview` with the current recalibrated pos/quat/fovy and visually compare
  against the real photo — sim framing should resemble the real overhead view.
result: PASSED (2026-09-12) — physical mount installed mid-phase, ahead of the original D-04 timeline. Captured a live still from the connected AR0144 stereo camera, built an interactive live-render tuning tool (`diagnostics/tune_agentview.py`) with a real-demo-stage selector, and iterated pos/quat/fovy against the real photo until framing matched. Applied `pos=[0.3244,-0.0081,1.4509] quat=[0.663562,0.250002,0.244879,0.661226] fovy=35.63` to `libero_tabletop_manipulation.py` (+ `bddl_base_domain.py` mirror), user-approved side by side, both camera tests re-passed. Commit `1be5f28`.

## Summary

total: 2
passed: 1
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps
