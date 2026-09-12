---
phase: 01-colab-environment-setup
verified: 2026-07-09T17:56:28Z
status: human_needed
score: 2/6 must-haves verified
behavior_unverified: 4
overrides_applied: 0
behavior_unverified_items:
  - truth: "All simulation dependencies install in order on a fresh Colab runtime without conflicts (SC1 / ENV-01)"
    test: "Fresh Colab A100 runtime: run Block A cells top-to-bottom (cells 0-10). After restart, run Block B in order and observe ENV-01 output."
    expected: "Block A completes without pip resolver errors; ENV-01 prints all version rows OK (mujoco 3.3.2, robosuite 1.4.0, gym 0.25.2, torch 2.2.0, numpy 1.26.4, ...), 'numpy ABI canary: OK', and 'ENV-01: PASS'"
    why_human: "Requires a live Colab runtime — pip resolution against Colab's preinstalled package set and the post-purge on-disk numpy state cannot be reproduced locally"
  - truth: "A default LIBERO Panda environment renders non-black RGB frames using EGL headless rendering (SC2 / ENV-02)"
    test: "Post-restart, run EGL bootstrap → config.yaml → sys.path → ENV-02 cell. Observe render output and saved libero_render_check.png."
    expected: "ENV-02 prints 'ENV-02: PASS — mean pixel value: <n>' with n > 5.0, non-black inline image, 'Saved → .../libero_render_check.png', and NO 'numpy.dtype size changed' ValueError"
    why_human: "Requires Colab GPU + EGL driver stack + the post-restart kernel state; this truth has NEVER been observed passing (UAT Test 3 was a blocker). Watch for any NEW failure at env.reset() — robosuite 1.4.0 with mujoco 3.3.2's render/reset path has never executed past the (now-repaired) import crash"
  - truth: "OpenVLA-OFT loads onto Colab GPU without OOM and returns a 7-D action given a test image and prompt (SC3 / ENV-03)"
    test: "Post-restart with guards set (EGL bootstrap cell ran first), run the ENV-03 cell."
    expected: "'ENV-03: PASS — action shape: (8, 7)' (or (7,)) with a 7-D per-step action printed; no jax/tensorflow in any traceback; no IPython ultratb infinite loop"
    why_human: "Requires A100 GPU + ~14GB HF model download; the jax/TF import-severing behavior of USE_TORCH/USE_TF/USE_FLAX depends on transformers' import-time caching in a live kernel"
  - truth: "Block A ends with 'numpy ABI gate: PASS' printed BEFORE the restart instruction; the gate raises RuntimeError (do NOT restart) when on-disk numpy is binary-incoherent"
    test: "Run Block A top-to-bottom on fresh Colab; observe the final gate cell (after Step 6, before the STOP markdown)."
    expected: "Probe stdout ending 'probe-ok: gym + numpy.random ABI coherent' followed by 'numpy ABI gate: PASS — numpy 1.26.4 coherent on disk; safe to restart runtime'. If it raises instead: do NOT restart; capture diagnostics"
    why_human: "The purge → clean-reinstall → fresh-subprocess probe operates on Colab's on-disk site-packages state (mixed numpy 2.x extensions under a 1.26.4 core) that does not exist locally"
human_verification:
  - test: "Fresh Colab A100: run Block A cells 0-10 top-to-bottom; observe the final numpy ABI gate cell output"
    expected: "'numpy ABI gate: PASS — numpy 1.26.4 coherent on disk; safe to restart runtime' printed after the probe output. On RuntimeError: do NOT restart — report the diagnostic (numpy paths + mtrand path + traceback)"
    why_human: "Executor cannot run Colab; gate behavior depends on Colab's on-disk package state"
  - test: "Restart runtime; run Block B cells in order; observe ENV-01 output"
    expected: "numpy row shows 1.26.4 OK, 'numpy ABI canary: OK', 'ENV-01: PASS'"
    why_human: "Post-restart kernel state and Colab package set required (FAIL branch already behaviorally verified locally)"
  - test: "Run ENV-02 cell; observe render result and any exception"
    expected: "'ENV-02: PASS' with mean pixel > 5.0, non-black frame displayed and saved; no numpy dtype-size ValueError. If a NEW error appears at env.reset() (e.g. anything mujoco-API related), report it — the robosuite 1.4.0 + mujoco 3.3.2 reset/render path has never executed"
    why_human: "Requires Colab GPU EGL rendering; UAT blocker gap closure must be confirmed at runtime"
  - test: "Run ENV-03 cell; observe model load and action output"
    expected: "'ENV-03: PASS' with a 7-D per-step action (chunk shape (8, 7) accepted); no jax/tensorflow import in any traceback path; no IPython ultratb loop"
    why_human: "Requires A100 + HF model download; guard effectiveness only observable in a live kernel"
---

# Phase 1: Colab Environment Setup — Verification Report

**Phase Goal:** A working Colab notebook environment where all dependencies install without version conflicts, GPU is accessible for VLA inference, and headless rendering produces valid RGB frames.
**Verified:** 2026-07-09T17:56:28Z
**Status:** human_needed
**Re-verification:** No — initial verification (post gap-closure plan 01-04)

## MVP Mode Note

Phase 1 is `mode: mvp`. The ROADMAP goal text predates user-story formatting and fails the user-story format check; plan 01-04's derived user story ("As a researcher, I want to run the setup notebook top-to-bottom on a fresh Colab A100 runtime and see ENV-01, ENV-02, and ENV-03 all print PASS, so that the language-to-embodied-action pipeline has a verified working environment to build on.") validates and was used for the user-flow framing. Roadmap Success Criteria remain the verification contract.

## User Flow Coverage

| Step | Expected | Evidence in Codebase | Status |
|------|----------|----------------------|--------|
| 1. Researcher opens notebook, sets REPO_ROOT | Single user-editable constant; all paths derive | Cell 1: REPO_ROOT + derived LIBERO_ROOT/LIBERO_PKG/OUT_DIR/BDDL_FILE, os.makedirs(OUT_DIR) | ✓ VERIFIED (+ UAT Test 6 pass) |
| 2. GPU check confirms A100 | Loud warning on non-A100, proceed allowed | Cell 2: torch.cuda.get_device_name(0), VRAM print, WARNING block | ✓ VERIFIED (+ UAT log pass) |
| 3. Block A installs full stack in order | apt→torch→sim stack→LIBERO→OFT fork→flash-attn→numpy gate | Cells 4-10 in exact order; fork last in cell 8; gate is last pip-invoking cell | ✓ present; runtime = human |
| 4. numpy ABI gate prints PASS before restart | Gate purges/pins/probes; PASS or RuntimeError(do-not-restart) | Cell 10: purge loop, 3-pattern leftover sweep, --no-cache-dir reinstall, fresh-subprocess probe of exact crash path | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED |
| 5. Restart → ENV-01 PASS | Version table + canary + verdict | Cell 16 (FAIL branch behaviorally verified locally) | ⚠️ runtime PASS = human |
| 6. ENV-02 PASS non-black render | EGL chain + numpy repair | Cells 13-17 ordering verified | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED |
| 7. ENV-03 PASS 7-D action | Guards + bf16 load + chunk handling | Cells 13, 20 | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED |

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SC1: All simulation dependencies install in order on fresh Colab runtime without conflicts | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | All install cells present in correct order (apt cell 4 → torch cell 5 → sim stack cell 6 → LIBERO cell 7 → OFT+fork cell 8 → flash-attn cell 9 → numpy gate cell 10). Pre-01-04 UAT observed Block A completing and ENV-01: PASS on Colab; 01-04 modified Block A (gate cell), so a fresh run is required. Note: versions deviate from SC1's literal list (mujoco 3.3.2 not 2.3.7 — no cp312 wheel for 2.3.7; torch 2.2.0 not 2.1.x — openvla-oft requirement; both documented, human-witnessed UAT facts) |
| 2 | SC2: Default LIBERO Panda env renders non-black RGB frames via EGL (MUJOCO_GL=egl before any MuJoCo import) | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | Full wiring verified: ICD JSON write precedes MUJOCO_GL=egl (cell 13 code lines 2 vs 7); config.yaml (cell 14) and sys.path (cell 15) precede the only libero import (cell 17); ENV-02 cell has render, [::-1] flip, mean>5.0 gate, save, env.close(). Root-cause repair (numpy gate) present. NEVER observed passing at runtime (UAT Test 3 blocker) |
| 3 | SC3: OpenVLA-OFT loads onto GPU without OOM, returns 7-D action | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | ENV-03 cell substantive: bf16 + low_cpu_mem_usage + trust_remote_code, norm_stats overlay from dataset_statistics.json, unnorm_key fallback, (8,7) chunk handling, per-step 7-D contract. USE_TORCH/USE_TF/USE_FLAX guards at cell top precede the prismatic importlib probe (offset-verified). NEVER observed passing (UAT Test 4 blocker) |
| 4 | 01-04 T1: numpy ABI gate is last executable Block A cell, prints PASS before restart, raises RuntimeError (do NOT restart) on incoherence | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | Gate cell (index 10) verified: purge loop (max 3, PackageNotFoundError-guarded), leftover sweep restricted to exactly numpy/numpy-*.dist-info/numpy.libs in purelib, --no-cache-dir exact-pin reinstall, fresh-subprocess _ABI_PROBE compiles clean and walks the exact ENV-02 crash path (numpy.random.rand → mtrand → gym → gym.spaces.Box), no mujoco/robosuite/libero imports in probe, RuntimeError branch with do-not-restart guidance. Gate is adjacent to restart markdown (10→11); no pip operation after it. Runtime purge/probe behavior needs Colab |
| 5 | 01-04 T4: ENV-01 verifies numpy==1.26.4, runs in-kernel ABI canary, prints FAIL on any version mismatch (not only pip conflicts) | ✓ VERIFIED | Behaviorally exercised locally: executed the ENV-01 cell source verbatim in an env with numpy 2.4.6 (MISMATCH) and clean `pip check` — printed "ENV-01: FAIL — version mismatch/missing package or numpy ABI canary failure". This is exactly the latent bug's trigger condition (old logic keyed only on our_conflicts and would have printed PASS). numpy row, canary OK/FAIL literals, all_ok=False in canary except-branch, and `all_ok and not our_conflicts` verdict all confirmed in source |
| 6 | Notebook structural contract from plans 01-01/01-02/01-03 holds (22 cells, ordering constraints, conventions) | ✓ VERIFIED | All four plan verify snippets pass (Task 1/2/3 OK, nb OK 22); nbformat 4 valid; all code cells AST-compile (shell-magic cells are valid IPython); restart STOP markdown (cell 11); Block A/B header docs updated with rule 4 and guard note; no stale Cell-14/18 refs in Block B header; matplotlib.use("Agg") before pyplot; Saved → convention; UPPER_SNAKE_CASE constants; no bitsandbytes/load_in_4bit in code |

**Score:** 2/6 truths verified (4 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `libero/notebooks/01-colab-env-setup.ipynb` | 22 cells; new Block A gate cell between flash-attn and STOP markdown | ✓ VERIFIED | 22 cells, valid nbformat 4; gate at index 10, flash-attn 9, STOP 11. Tracked clean at HEAD (git tracks lowercase `libero/`) |
| Block A gate cell | PURELIB, NUMPY_PIN, _ABI_PROBE, purge + sweep + reinstall + probe | ✓ VERIFIED | All identifiers present; logic substantive (3332 chars); probe string compiles as valid Python |
| Guard assignments (EGL bootstrap + ENV-03) | USE_TORCH=1/USE_TF=0/USE_FLAX=0 in both cells | ✓ VERIFIED | Cell 13 Step (b2) and cell 20 top; ENV-03 guards at lower char offset than both `import importlib, subprocess, sys as _sys` and `from transformers import` |
| ENV-01 numpy row + canary | EXPECTED["numpy"]="1.26.4"; canary OK/FAIL; verdict fix | ✓ VERIFIED | All present; FAIL branch behaviorally exercised |
| `LIBERO/notebooks/outputs/libero_render_check.png` | Runtime artifact | N/A — runtime | Produced only when ENV-02 runs on Colab; save call wired in cell 17 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| EGL bootstrap + ENV-03 cell tops | transformers import | USE_TORCH/USE_TF/USE_FLAX env vars set textually before any transformers import | ✓ WIRED | Offset-verified in both cells; transformers 4.40.1 fork's import_utils.py reads these |
| numpy ABI gate | STOP restart markdown | Gate is LAST pip-invoking cell in Block A | ✓ WIRED | Gate index 10 immediately precedes restart markdown 11; zero code cells / `pip install` between |
| NVIDIA ICD JSON | MUJOCO_GL=egl | Write ordering inside cell 13 | ✓ WIRED | Code-line order: ICD write (line 2) before os.environ["MUJOCO_GL"] (line 7); no sim imports in cell |
| config.yaml + sys.path | `from libero.libero.envs import OffScreenRenderEnv` | Cells 14, 15 precede cell 17 | ✓ WIRED | Only libero import in notebook is in cell 17 |
| Cell 1 constants | Block B cells | LIBERO_PKG/LIBERO_ROOT/OUT_DIR referenced, not re-hardcoded | ✓ WIRED | Cell 14 uses LIBERO_ROOT; cell 15 uses LIBERO_PKG; cell 17 uses OUT_DIR and probes LIBERO_PKG with /content/libero fallback (established UAT fact) |
| Local mj_kinematics shim (bddl_base_domain.py) | Runtime LIBERO install | Drive sync OR GitHub clone fallback | ⚠️ PARTIAL (inert) | Shim exists ONLY in this repo (commit 489c22a); cell 7's fallback clones upstream Lifelong-Robot-Learning/LIBERO which lacks it, and UAT proved the fallback is the path taken. HOWEVER: verified against mujoco 3.3.2 source that `mj_step1` still exists (mujoco.h line 134 + python functions.cc) — commit 489c22a's premise ("3.x removed mj_step1") is factually wrong, so the shim's hasattr branch and upstream's direct call are behaviorally identical. Cell 6's "shim is active" print is misleading but harmless |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| ENV-01 FAIL branch on version mismatch with clean pip check | `exec(ENV-01 cell source)` locally (numpy 2.4.6, pipeline pkgs missing) | Printed "ENV-01: FAIL — version mismatch/missing package or numpy ABI canary failure" | ✓ PASS |
| _ABI_PROBE is valid Python with correct import hygiene | `ast.parse(probe)` + import scan | Compiles; imports only numpy/numpy.random/gym/gym.spaces; no mujoco/robosuite/libero | ✓ PASS |
| All plan 01-04 task verifications | 3 JSON-assertion scripts from 01-04-PLAN.md | Task 1 OK / Task 2 OK / Task 3 OK | ✓ PASS |
| Whole-notebook structural check | nbformat/outputs/execution_count assertion | `nb OK 22` | ✓ PASS |
| All code cells parse | AST compile per cell (magics stripped) | All pure-Python cells OK; cells 4/5/8 are valid IPython shell cells | ✓ PASS |
| ENV-01/02/03 runtime PASS | — | Requires Colab A100 | ? SKIP → human |

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes exist in this repo and none are declared by the phase plans. The plan-declared automated verifications (the three per-task JSON assertion commands plus the whole-notebook check) were executed in this verifier's own process — all PASS (see Behavioral Spot-Checks).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| ENV-01 | 01-01, 01-02, 01-04 | Deps install in correct order without conflicts | ? NEEDS HUMAN | Install cells + verification cell + gate all present and wired; FAIL path behaviorally verified; runtime PASS on fresh Colab pending (pre-01-04 UAT observed ENV-01: PASS) |
| ENV-02 | 01-02, 01-04 | EGL headless rendering, non-black frames | ? NEEDS HUMAN | Full EGL/config/sys.path/render chain wired; numpy root-cause repair in place; never observed passing at runtime |
| ENV-03 | 01-03, 01-04 | OpenVLA-OFT loads on Colab GPU | ? NEEDS HUMAN | Load cell substantive with guards, bf16, chunk handling; never observed passing at runtime |

Orphaned requirements: none — REQUIREMENTS.md maps exactly ENV-01/02/03 to Phase 1 and all three are claimed by plans.

Commits verified: `638a17e`, `89e03ef`, `ee04e3c` (plan 01-04 tasks) all exist and are ancestors of HEAD; working-tree notebook is clean against HEAD.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| notebook cell 6 | print | "bddl_base_domain.py mj_kinematics shim is active" | ℹ️ Info | Misleading: shim is local-only and does not ship via the /content/libero clone fallback (the path UAT proved is taken). Behaviorally inert since mujoco 3.3.2 retains mj_step1 — but the print asserts something the runtime cannot guarantee |
| notebook cell 16 | pip-check block | `all_conflicts` treats "No broken requirements found." as truthy conflict output | ℹ️ Info | Cosmetic only: prints the "pip check output (all)" block (and the pre-existing-conflicts parenthetical on PASS) even when pip check is clean. Verdict logic unaffected — first-token filter never matches "No" |
| notebook cell 21 | summary table | "Run Cell 15 / Cell 16" numeric refs | ℹ️ Info | Stale after the 01-04 cell insertion (ENV-01 is now index 16, ENV-02 index 17). 01-04 only mandated fixing the Block B header, which was done |

No TBD/FIXME/XXX/TODO/HACK/placeholder debt markers in the notebook. No bitsandbytes/load_in_4bit in any code cell (D-05 honored; the string appears once in a markdown cell explaining the exclusion).

### Human Verification Required

#### 1. Block A numpy ABI gate — PASS before restart

**Test:** Fresh Colab A100 runtime. Run Block A (cells 0-10) top-to-bottom. Observe the final gate cell.
**Expected:** Probe output ending `probe-ok: gym + numpy.random ABI coherent`, then `numpy ABI gate: PASS — numpy 1.26.4 coherent on disk; safe to restart runtime`. If it raises RuntimeError: do NOT restart — capture and report the diagnostic output (numpy paths, mtrand path, traceback).
**Why human:** The purge/reinstall/probe operates on Colab's on-disk site-packages state (mixed numpy 2.x C extensions under a 1.26.4 core) which cannot be reproduced locally.

#### 2. ENV-01 — PASS post-restart

**Test:** Restart runtime; run Block B cells in order; observe ENV-01.
**Expected:** numpy row `1.26.4 ... OK`, `numpy ABI canary: OK`, `ENV-01: PASS`.
**Why human:** Post-restart kernel + Colab package set required. (The FAIL branch is already behaviorally verified locally.)

#### 3. ENV-02 — PASS with non-black render (UAT Test 3 closure)

**Test:** Run the ENV-02 cell.
**Expected:** `ENV-02: PASS — mean pixel value: <n>` with n > 5.0, non-black inline frame, `Saved → .../libero_render_check.png`, and no `numpy.dtype size changed` ValueError.
**Why human:** Requires Colab GPU EGL. **Watch specifically for a NEW failure at `env.reset()`:** the robosuite 1.4.0 + mujoco 3.3.2 reset/render path has never executed (all prior runs crashed at import). The local mj_kinematics shim does not ship via the clone fallback, but mujoco 3.3.2 retains mj_step1 so upstream code should work — report any mujoco-API AttributeError immediately.

#### 4. ENV-03 — PASS with 7-D action (UAT Test 4 closure)

**Test:** Run the ENV-03 cell (after the EGL bootstrap cell ran first this session).
**Expected:** `ENV-03: PASS — action shape: (8, 7)` (or `(7,)`) with a 7-D per-step action printed; no jax/tensorflow anywhere in a traceback; no IPython ultratb infinite loop.
**Why human:** Requires A100 + ~14GB HF checkpoint download; the USE_TF/USE_FLAX guard effect on transformers' import-time backend caching is only observable in a live kernel.

### Gaps Summary

No blocking gaps found in the codebase. All artifacts from all four plans exist, are substantive, and are correctly wired; all plan 01-04 static verifications pass against the current notebook; the ENV-01 verdict-gate bug fix was behaviorally confirmed locally; the two UAT blocker root-cause repairs (numpy ABI gate, transformers backend guards) are present and textually ordered exactly as the plan requires.

The phase goal itself, however, is runtime-behavioral on a platform this machine cannot reach: ENV-01/ENV-02/ENV-03 printing PASS on a fresh Colab A100 has not yet been observed post-gap-closure (ENV-02/ENV-03 have never been observed passing at all). Status is therefore `human_needed` — a fresh-Colab UAT re-run is the remaining step, per plan 01-04's own human-check contract.

Noteworthy finding for the human tester (documented in Key Links and Human Verification item 3): commit 489c22a's premise that mujoco 3.x removed `mj_step1` is factually wrong (verified against mujoco 3.3.2 source — mujoco.h line 134 and python bindings both retain it), and the local shim never reaches the Colab runtime when the /content/libero clone fallback fires. This is currently harmless, but the first-ever execution of the LIBERO reset/render path under mujoco 3.3.2 happens during human verification — any mujoco-API failure there is a new gap, not a regression of this phase's fixes.

Also noted: ROADMAP SC1's literal version list (MuJoCo 2.3.7, PyTorch 2.1.x) is stale relative to established, human-witnessed UAT facts (mujoco 3.3.2 — no cp312 wheel for 2.3.7; torch 2.2.0 — openvla-oft requirement). If desired, record this as an override in this file's frontmatter or refresh the ROADMAP SC text.

---

_Verified: 2026-07-09T17:56:28Z_
_Verifier: Claude (gsd-verifier)_
