# Phase 3: VLA Inference Loop - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-18
**Phase:** 3-VLA Inference Loop
**Areas discussed:** VLA interface & action consumption, π0 integration depth, Task & prompt scope, Success detection & episode termination

---

## VLA Interface & Action Consumption

| Option | Description | Selected |
|--------|-------------|----------|
| Open-loop: replay all 8 steps | Run inference once, execute the full 8-step chunk, then re-infer | |
| Closed-loop: execute fewer steps then re-infer | More robust to drift, more VLA calls | |
| You decide | Claude picks based on LIBERO eval protocol / OFT paper conventions | ✓ |

**User's choice:** You decide.
**Notes:** Claude researches actual LIBERO eval / OFT paper convention before locking this in.

| Option | Description | Selected |
|--------|-------------|----------|
| Single wrist/eye-in-hand image + language string | Matches Phase 2's only verified camera | |
| Dict of named camera views + language string | Future-proofs for Phase 5 multi-camera | ✓ |

**User's choice:** Dict of named camera views + language string in, action array out.
**Notes:** Only one key populated in Phase 3; signature won't need to change for Phase 5.

| Option | Description | Selected |
|--------|-------------|----------|
| `LIBERO/libero/libero/vla/` | Sits in the vendored LIBERO fork alongside SOARM robot classes | |
| `explorations/vla/` | Keeps it out of the LIBERO fork | |
| You decide | Claude picks placement during planning | ✓ |

**User's choice:** You decide.

| Option | Description | Selected |
|--------|-------------|----------|
| Each backend handles its own normalization internally | Keeps shared interface minimal | ✓ |
| Shared normalization step outside the backend | More upfront structure, risks forcing OFT's format onto π0 | |

**User's choice:** Each backend handles its own normalization internally.

---

## π0 Integration Depth

| Option | Description | Selected |
|--------|-------------|----------|
| Real working π0 inference on Colab | Proves success criterion #3 literally | ✓ |
| Interface-only stub for π0 | Lower risk, faster, doesn't fully prove swap | |

**User's choice:** Real working π0 inference on Colab.

| Option | Description | Selected |
|--------|-------------|----------|
| Two separate Colab notebooks/kernels | Mirrors existing LIBERO-training-vs-VLA-inference kernel split | |
| Single notebook, single kernel | Simpler demo, real dependency-conflict risk | |
| You decide | Claude confirms actual openpi dependencies during research first | ✓ |

**User's choice:** You decide.

| Option | Description | Selected |
|--------|-------------|----------|
| π0-FAST or base LIBERO-finetuned checkpoint (smallest available) | Mirrors OFT's Colab-compatibility-driven checkpoint choice | |
| You decide | Researcher determines during research | ✓ |

**User's choice:** You decide.

---

## Task & Prompt Scope

| Option | Description | Selected |
|--------|-------------|----------|
| All 3 tasks, looped automatically | One run produces videos for all 3, 3 data points for VLA-03 | ✓ |
| Single fixed task for the core demo | Simplest to build/debug first | |
| User-selectable at run time | Most flexible, more plumbing | |

**User's choice:** All 3 tasks, looped automatically.

| Option | Description | Selected |
|--------|-------------|----------|
| 1 episode per task (3 total) | Fastest, pass/fail smoke signal only | |
| Multiple episodes per task (5-10) for a real success rate | Closer to LIBERO's standard eval protocol | ✓ |

**User's choice:** Multiple episodes per task (e.g. 5-10).

| Option | Description | Selected |
|--------|-------------|----------|
| Both backends run the full suite | Fair like-for-like comparison, doubles GPU time | |
| OFT gets full suite; π0 gets lighter smoke-test | OFT is primary/proven; π0 just proves the swap | ✓ |

**User's choice:** OFT gets the full suite; π0 gets a lighter smoke-test (e.g. 1 task, 1-2 episodes).

| Option | Description | Selected |
|--------|-------------|----------|
| One video file per episode | Simplest, matches LIBERO's video_utils.py pattern | ✓ |
| One combined video per task | Fewer files, extra stitching logic needed | |

**User's choice:** One video file per episode.

---

## Success Detection & Episode Termination

| Option | Description | Selected |
|--------|-------------|----------|
| Poll check_success() every step, stop early on success | Standard LIBERO eval pattern, saves sim time | ✓ |
| Only check success once, at episode end / max steps | Simpler logic, wastes sim steps | |

**User's choice:** Poll check_success() every step, stop early on success.

| Option | Description | Selected |
|--------|-------------|----------|
| LIBERO's standard eval horizon (task-suite default) | Keeps success rate comparable to published numbers | |
| You decide | Researcher confirms exact standard horizon | ✓ |

**User's choice:** You decide.

| Option | Description | Selected |
|--------|-------------|----------|
| Printed PASS/FAIL per episode + aggregated success-rate summary table | Mirrors Phase 1/2's PASS/FAIL convention | ✓ |
| Printed PASS/FAIL per episode only | Simpler, no aggregate | |

**User's choice:** Printed PASS/FAIL per episode + aggregated success-rate summary table at the end.

| Option | Description | Selected |
|--------|-------------|----------|
| Hard failure only — binary success/fail | Simplest, matches LIBERO's binary metric | |
| You decide | Claude decides if extra diagnostic signal is worth it | ✓ |

**User's choice:** You decide.

---

## Claude's Discretion

- OFT action-chunk consumption strategy (open-loop replay vs. closed-loop re-inference) — research LIBERO eval protocol / OFT paper convention.
- Exact module location for the shared VLA interface (`LIBERO/libero/libero/vla/` vs. `explorations/vla/` vs. other).
- Single vs. split notebook/kernel structure for OFT vs. π0 — confirm openpi's actual dependency stack first.
- Specific π0 checkpoint/variant to target for Colab GPU compatibility.
- Exact max-step cap value for episode timeout (use LIBERO's standard `libero_spatial` horizon).
- Whether to capture a near-miss/partial-credit diagnostic signal on timeout, beyond binary success/fail.

## Deferred Ideas

None — discussion stayed within phase scope.
