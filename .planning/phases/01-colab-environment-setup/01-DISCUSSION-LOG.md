# Phase 1: Colab Environment Setup - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-08
**Phase:** 1-Colab Environment Setup
**Areas discussed:** Notebook structure, Transformers conflict fix, GPU tier assumption, Verification artifact

---

## Notebook Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Single notebook, restart-aware | One notebook with clearly marked restart cells | ✓ |
| Two separate notebooks | 01a-env-setup.ipynb + 01b-vla-verify.ipynb for kernel isolation | |
| Colab form / section-based | Single notebook with collapsible section headers | |

**User's choice:** Single notebook, restart-aware

---

| Option | Description | Selected |
|--------|-------------|----------|
| LIBERO/notebooks/ | Co-located with existing LIBERO notebooks | ✓ |
| colab/ at repo root | New top-level directory for all Colab notebooks | |
| notebooks/ at repo root | Centralized outside the vendored LIBERO subtree | |

**User's choice:** LIBERO/notebooks/

---

## Transformers Conflict Fix

| Option | Description | Selected |
|--------|-------------|----------|
| Override to 4.40.1 | Install OpenVLA-OFT's version, override LIBERO's pin | ✓ |
| Pin to 4.21.1, defer conflict | Satisfy LIBERO, risk silent VLA failure | |
| Isolate with cell-level venv | True isolation, complex Colab setup | |

**User's choice:** Override to 4.40.1 — Phase 1 only needs LIBERO rendering, not training

---

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — render LIBERO frame as post-install check | Explicitly verify LIBERO rendering under 4.40.1 | ✓ |
| No — skip LIBERO compat check in Phase 1 | Trust backward compatibility | |

**User's choice:** Yes — render a LIBERO frame as the post-install check (ENV-02 verification)

---

## GPU Tier Assumption

| Option | Description | Selected |
|--------|-------------|----------|
| Colab Pro A100 | bf16, no quantization, 40GB VRAM | ✓ |
| Free T4 with 4-bit quantization | bitsandbytes 4-bit, 15GB VRAM, tight fit | |
| Write for both tiers | Auto-detect at runtime | |

**User's choice:** Colab Pro A100

---

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — assert A100 or warn loudly | Early GPU check cell with loud warning | ✓ |
| No — skip the assertion | Markdown documentation only | |

**User's choice:** Yes — assert A100 or warn loudly

---

## Verification Artifact

| Option | Description | Selected |
|--------|-------------|----------|
| Structured pass/fail cells | Each ENV requirement gets its own PASS/FAIL cell | ✓ |
| Summary output cell only | Single final cell with summary table | |
| Saved outputs to Google Drive | Persistent cross-session, good for sharing | |

**User's choice:** Structured pass/fail cells (one per ENV-01, ENV-02, ENV-03)

---

| Option | Description | Selected |
|--------|-------------|----------|
| Display inline + save to disk | plt.imshow() + save to LIBERO/notebooks/outputs/ | ✓ |
| Save to disk only | Save PNG, print path | |
| Display inline only | Ephemeral, lost on session end | |

**User's choice:** Display inline + save to disk

---

## Claude's Discretion

- Exact pip install ordering and version pinning within install cells
- Which specific LIBERO BDDL task/camera config to use for the ENV-02 render check
- Whether to use IPython.display.Image or matplotlib for inline display

## Deferred Ideas

- 4-bit quantization / bitsandbytes path for T4 compatibility — not needed for A100 target
- Google Drive output persistence — deferred; local outputs/ dir sufficient for Phase 1
- π0 (openpi) model loading in Colab — deferred to Phase 3 (VLA Inference Loop)
