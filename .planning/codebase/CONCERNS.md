# Codebase Concerns

**Analysis Date:** 2026-07-07

## Tech Debt

**Hardcoded relative paths in `render_scenes.py`:**
- Issue: `pc_dir = "Real-3DQA/point_clouds"` and `output_dir = "perspective_images"` are bare relative strings, not constructed from `__file__`. The script breaks silently if not invoked from `explorations/`.
- Files: `explorations/render_scenes.py:101-102`
- Impact: Produces `FileNotFoundError` or writes output to an unexpected location when the working directory differs.
- Fix approach: Replace with `Path(__file__).resolve().parent / "Real-3DQA" / "point_clouds"` and a matching output path, matching the pattern used in `explorations/real3dqa/explore.py` and `explorations/lib/explore.py`.

**Duplicated exploration scripts for the same dataset:**
- Issue: Three near-identical LIBERO exploration scripts exist: `explorations/lib/explore.py` (current), `explorations/real3dqa/explore.py`, and the now-deleted `libero/explore.py` / `real3dqa/explore.py` at repo root (still tracked as deleted in git: ` D libero/explore.py`, ` D real3dqa/explore.py`). The deleted files linger in git history without a clean migration note.
- Files: `explorations/lib/explore.py`, `explorations/real3dqa/explore.py`
- Impact: Confusion about the canonical exploration entry point; future contributors may edit the wrong file.
- Fix approach: Commit the deletions of `libero/explore.py` and `real3dqa/explore.py` that are currently staged. Add a top-level `README.md` or `explorations/README.md` pointing to the correct scripts.

**`sys.path` mutation in `create_scene.py`:**
- Issue: `sys.path.insert(0, LIBERO_PATH)` at module level in `explorations/create_scene.py:14` overrides the import order for any process that imports this module. There is also a stale, deleted `requirements.txt` in the repo root (` D requirements.txt` in git status) while `explorations/requirements.txt` and `LIBERO/requirements.txt` define separate incompatible dependency sets.
- Files: `explorations/create_scene.py:13-16`
- Impact: Silent import shadowing; version conflicts between `explorations/requirements.txt` (loose `torch>=1.11.0`) and `LIBERO/requirements.txt` (pinned `numpy==1.22.4`, `gym==0.25.2`).
- Fix approach: Package LIBERO as an editable install (`pip install -e LIBERO/`) and remove the path hack. Consolidate or clearly document which `requirements.txt` governs which script.

**Unimplemented predicate logic in LIBERO:**
- Issue: `LIBERO/libero/libero/envs/predicates/base_predicates.py:72` contains a TODO for center-of-mass region checking that is commented out. This means certain spatial predicates may pass incorrectly during task evaluation.
- Files: `LIBERO/libero/libero/envs/predicates/base_predicates.py:72`
- Impact: Incorrect task success signals for tasks requiring center-of-mass containment checks.
- Fix approach: Implement the region check or open a tracked issue if the upstream LIBERO library is expected to provide this.

**TODO stubs in PackNet and lifelong evaluation:**
- Issue: `LIBERO/libero/lifelong/algos/packnet.py:251` and `LIBERO/libero/lifelong/evaluate.py:5` both contain `TODO: find a better way` comments indicating unresolved design decisions in the lifelong learning path management code.
- Files: `LIBERO/libero/lifelong/algos/packnet.py:251`, `LIBERO/libero/lifelong/evaluate.py:5`
- Impact: The current approach may not scale to larger task suites or alternate model architectures.
- Fix approach: Document the intended design, or replace with a clean abstraction once the research direction is clearer.

## Known Bugs

**`render_scenes.py` unpacks 4-tuple without validation:**
- Symptoms: `coords, colors, _, _ = data` at line 115 assumes each `.pth` file always contains exactly a 4-element tuple. A malformed or differently-structured file raises a `ValueError` with no diagnostic message.
- Files: `explorations/render_scenes.py:115`
- Trigger: Loading any `.pth` file that does not match the `(xyz, rgb, labels, instance_ids)` schema.
- Workaround: None; the script crashes silently mid-batch.

**`bird_eye_grid` in `real3dqa/explore.py` loads annotations unconditionally after grid rendering:**
- Symptoms: `main()` always calls `load_annotations()` and prints a sample, even when `--scene` is provided and no annotations are relevant to the output. If the annotations directory is absent (e.g., only point clouds downloaded), it silently returns an empty list rather than informing the user.
- Files: `explorations/real3dqa/explore.py:120-124`
- Trigger: Running with `--scene` argument and no `annotations/` directory present.
- Workaround: None; misleading output (`Total QA pairs: 0`) with no explanation.

## Security Considerations

**`torch.load` with `weights_only=False`:**
- Risk: `torch.load(path, weights_only=False)` in `explorations/render_scenes.py:114` and `explorations/real3dqa/explore.py:34` deserializes arbitrary Python objects. Malicious `.pth` files can execute code at load time.
- Files: `explorations/render_scenes.py:114`, `explorations/real3dqa/explore.py:34`
- Current mitigation: None. Files are downloaded from Hugging Face (`Oliver-Ma/Real-3DQA`) which is a public, untrusted source.
- Recommendations: If the `.pth` files store only tensors and numpy arrays, switch to `weights_only=True`. If that is not possible because the files contain non-tensor objects, add a checksum verification step after download (the LIBERO benchmark already ships `LIBERO/benchmark_scripts/shasum_files.py` as a pattern to follow).

**No checksum verification for downloaded datasets:**
- Risk: `explorations/download_real3dqa.py` downloads ~300 MB from Hugging Face with no integrity check after download.
- Files: `explorations/download_real3dqa.py`
- Current mitigation: None.
- Recommendations: After `snapshot_download`, verify file checksums using a manifest, similar to `LIBERO/benchmark_scripts/shasum_files.py`.

## Performance Bottlenecks

**Point cloud rendering via Python PIL loop:**
- Problem: `render_scenes.py:92-94` renders each point as an individual `draw.ellipse()` call inside a Python `for` loop. For large scenes with millions of points, this is extremely slow.
- Files: `explorations/render_scenes.py:92-94`
- Cause: PIL has no vectorized drawing API; every point is a separate Python-level call.
- Improvement path: Use `matplotlib` scatter (already a dependency) with `s=1` and `rasterized=True`, or render directly to a numpy array using array indexing (`image[y_int, x_int] = color`).

**`collect_first_frames` iterates entire parquet rows:**
- Problem: `explorations/lib/explore.py:64` uses `df.iterrows()` over every row of each parquet file to find the first frame of each episode. `iterrows()` is the slowest pandas iteration method.
- Files: `explorations/lib/explore.py:64`
- Cause: Row-by-row Python iteration instead of a vectorized group-by or `drop_duplicates` on `episode_index`.
- Improvement path: Replace with `df.drop_duplicates(subset="episode_index", keep="first")` after reading only the needed columns with `pd.read_parquet(..., columns=[...])`.

## Fragile Areas

**LIBERO path bootstrap in `create_scene.py`:**
- Files: `explorations/create_scene.py:13-16`
- Why fragile: The path `os.path.join(os.path.dirname(__file__), "LIBERO")` assumes `create_scene.py` is located in `explorations/` and that `explorations/LIBERO/` is a symlink or copy of the LIBERO package. Moving the file or cloning without the `LIBERO/` subtree present causes a silent import failure.
- Safe modification: Install LIBERO as a proper package and remove the path injection.
- Test coverage: No tests; only manual visual inspection.

**Fixed action dimension in `create_scene.py`:**
- Files: `explorations/create_scene.py:56`
- Why fragile: `np.zeros(7)` hard-codes a 7-DOF action space. If the task or robot configuration changes this dimensionality, the physics settling loop fails with a MuJoCo dimension mismatch error.
- Safe modification: Read action dimension from `env.action_space.shape[0]` instead.
- Test coverage: None.

## Missing Critical Features

**No unified project README or setup guide:**
- Problem: There is no top-level `README.md` (only `LIBERO/README.md` and `explorations/Real-3DQA/README.md`). New contributors have no documented path for: setting up the environment, which `requirements.txt` to use, how to download datasets, or which scripts to run in what order.
- Blocks: Reproducibility; onboarding additional collaborators.

**No test suite:**
- Problem: There are zero test files (`*.test.py`, `*_test.py`, `test_*.py`) in the custom code under `explorations/`. All verification is manual and visual.
- Blocks: Automated CI, regression detection, confident refactoring.

## Test Coverage Gaps

**All exploration scripts untested:**
- What is not tested: Dataset loading, point-cloud rendering, LIBERO environment initialization, annotation parsing.
- Files: `explorations/render_scenes.py`, `explorations/real3dqa/explore.py`, `explorations/lib/explore.py`, `explorations/create_scene.py`, `explorations/download_real3dqa.py`
- Risk: Regressions in data loading or rendering go undetected until manual inspection.
- Priority: Medium — these are currently research/exploration scripts, but any promotion to pipeline components will need coverage.

---

*Concerns audit: 2026-07-07*
