# Phase 13: Camera Device Resolution Unification - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-30
**Phase:** 13-Camera Device Resolution Unification
**Areas discussed:** Unification scope, device_map.json raw field, CAMFIX-02 re-verification, Provenance log design (CAMFIX-03)

---

## Unification scope

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, unify it too | Give record_episode.py access to the same name-based resolution + StereoSplitCamera helper | ✓ |
| Leave it alone — legacy/dev tool | Document as wrist-only, known-unsupported for overhead | |
| Remove it | Delete record_episode.py entirely | |

**User's choice:** Route record_episode.py through the shared helper.
**Notes:** Claude explained record_episode.py predates the bridge, isn't called by any live-episode flow, and its `cv2.VideoCapture`-based overhead capture never actually worked for the AR0144's split-stereo format in the first place (cv2's AVFoundation backend can't address it). User chose to give it correct support via the shared helper rather than leave it broken or delete it.

| Option | Description | Selected |
|--------|-------------|----------|
| Leave as-is, no changes | ScriptedActionSource is Phase-11 scripted stand-in, no vision consumer | ✓ |
| Also share the helper here for consistency | Route it through the helper anyway | |

**User's choice:** Leave as-is, no changes.
**Notes:** User initially asked "What is this scripted action source for exactly?" — Claude explained it's a throwaway dry-run harness from Phase 11 that only toggles the gripper on a fixed schedule and never reads camera frames at all (not a real VLA, per its own docstring). After clarification, user confirmed no changes needed — no collision risk exists since there's no second consumer of the same device in this branch.

---

## device_map.json raw field

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — build one shared resolve-by-name helper for both | Add wrist_name to device_map.json, resolve wrist by name at runtime same as overhead | ✓ |
| No — name-resolve overhead only, wrist stays index-based | Keep wrist on raw index, scope narrowly to the AR0144 | |

**User's choice:** Build one shared resolve-by-name helper for both cameras.
**Notes:** User pushed back on the initial framing ("shouldn't both be resolved using name? Wrist camera is called 'USB Camera'"), correctly identifying that detect_devices.py already correlates the wrist camera's real device name but never persists/uses it. Claude confirmed this and explained the AR0144 went name-based primarily because cv2 can't open its stereo format at all (a hard technical constraint), while the wrist camera's index-instability risk is a lower-stakes but structurally similar risk that the existing correlation data could close cheaply. User chose the broader unification.

| Option | Description | Selected |
|--------|-------------|----------|
| Keep as diagnostic/fallback only, name is authoritative | Raw indices stay as last-resort fallback if name correlation fails | ✓ |
| Remove raw fields entirely once name resolution is proven reliable | No fallback path at all | |

**User's choice:** Keep raw fields as diagnostic/fallback only.
**Notes:** detect_devices.py's initial probing still needs numeric indices to enumerate hardware, so raw fields can't disappear entirely — they become explicitly fallback-only, documented as not for direct use by new code.

---

## CAMFIX-02 re-verification

| Option | Description | Selected |
|--------|-------------|----------|
| Fresh re-verification required | User runs a new live episode against this phase's new by-name resolution code | ✓ |
| Rely on Phase 12's evidence, unit tests only | Treat CAMFIX-02 as already satisfied by 12-UAT.md Test 1 | |

**User's choice:** Fresh re-verification required.
**Notes:** Claude surfaced that Phase 12's UAT already passed this exact visual check (camera_overhead shows real workspace) against the pre-Phase-13 code, but this phase adds a new resolution layer on top that could itself regress the fix. User agreed a fresh check is warranted.

| Option | Description | Selected |
|--------|-------------|----------|
| Both overhead AND wrist frames confirmed | Wrist resolution also newly changed this phase, needs its own visual check | ✓ |
| Overhead only, same scope as Phase 12's Test 1 | Leave wrist correctness to unit tests | |

**User's choice:** Both overhead and wrist frames confirmed.
**Notes:** Since wrist camera resolution is newly changed (index-only → name-based-with-fallback), user chose to have the live episode confirm both cameras, not just overhead.

---

## Provenance log design (CAMFIX-03)

| Option | Description | Selected |
|--------|-------------|----------|
| Content hash only | sha256 of exact sent bytes per step, written into episode.jsonl | |
| Thumbnail only | Small downsized image per step for visual spot-check | |
| Both hash and thumbnail | Most complete provenance | ✓ (initial) → later revised to hash-only |

**User's choice:** Initially selected "both", but revised to hash-only after Claude surfaced that full-resolution `policy_camera1/2/3` PNGs are already saved per step by existing Phase-12 code (`policy_observation_frames`), making a separate thumbnail redundant.
**Notes:** User asked for clarification twice ("I think i need more clarification here", then a request for a concrete example of what this is useful for) before settling. Claude explained the hash only ever attests to the client's own local copy — it cannot prove Colab received/used that exact frame, since there's no round-trip confirmation mechanism. User's own restatement ("this is for confirming... was actually used by colab or not") was corrected by Claude: the hash explicitly cannot answer that question; it can only catch client-side issues (duplicate sends, post-hoc file corruption).

| Option | Description | Selected |
|--------|-------------|----------|
| Hash only — no redundant thumbnail | Add sha256 field to existing policy_camera1/2/3 records | ✓ |
| Still want a separate smaller thumbnail file too | Additional lower-res artifact | |

**User's choice:** Hash only, no redundant thumbnail.

| Option | Description | Selected |
|--------|-------------|----------|
| Docstring + one-time log note | Code docstring plus a new episode-level metadata note | |
| Per-step field in every record | Disclaimer repeated on every JSON line | |
| Docstring only, nothing in output data | No machine-readable disclaimer at all | ✓ (final) |

**User's choice:** Docstring only, nothing in output data.
**Notes:** User asked for clarification on placement twice before deciding. Claude explained that since nothing reads a disclaimer field back programmatically, its only audience is a human reading code or REQUIREMENTS.md/ROADMAP.md (which already state the caveat in the existing requirement text) — adding it to output data would only add noise or require inventing a new metadata mechanism for no functional gain. User agreed on docstring-only placement.

---

## Claude's Discretion

- Exact shape/naming of the shared resolve-by-name helper function/module (file location, function signature) — no user preference expressed beyond "one shared helper for both cameras."
- Exact hash algorithm parameters (hashing raw numpy bytes vs. PNG-encoded bytes) — implementer's choice as long as deterministic and cheap.
- Wrist-name resolution failure handling (warn-and-fallback vs. harder failure) before falling back to the raw index — follow the existing `stereo_overhead_name` fallback precedent unless research/planning finds a reason to diverge.

## Deferred Ideas

None — discussion stayed within phase scope throughout. The major reframing (discovering that much of CAMFIX-01/02 was already implemented by Phase 12's gap-closure wave) narrowed remaining scope rather than introducing new capabilities; captured directly in CONTEXT.md's `<domain>` section, not deferred.
