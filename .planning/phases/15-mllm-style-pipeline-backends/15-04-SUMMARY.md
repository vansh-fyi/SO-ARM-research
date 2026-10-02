---
phase: 15-mllm-style-pipeline-backends
plan: 04
subsystem: control/vla_bridge (MLLM per-call audit)
status: complete
tags: [mllm, mllm-06, pipe-02, audit, redaction, sidecar, cost, tdd]
requires:
  - control/vla_bridge/backends/base_mllm.py (recorder hook from Plan 15-01, unmodified)
  - control/fake_providers.py (ScriptedAdapter, FakeCollector, make_chunk_doc, make_provider_result, png_bytes; unmodified)
  - control/run_vla_episode.py::run_episode and control/vla_bridge/io_logger.py::IOLogger (unmodified)
provides:
  - vla_bridge.mllm_record (REDACTED, SECRET_HEADER_KEYS, redact_text, redact, error_text, to_json_safe, sha256_hex, estimate_cost_usd, SIDECAR_FILE_NAMES, SIDECAR_IMAGE_ROLES, IMAGE_EXTENSIONS, RECORDER_FILE_NAMES, COLLECTOR_FILE_NAMES, write_call_sidecar, CallRecorder)
  - "Record fields: cost_usd_est, price_table, sidecar_dir, sidecar_files, images[].path, sidecar_error"
  - "Sidecar layout: <out>/mllm_calls/<NNNN>/{prompt.txt, request_meta.json, response_raw.json, thinking.txt, depth.npy, aligned_left.png, depth_heatmap.png, depth_grid.txt, images/<role>.<png|jpg>}"
affects:
  - Plan 15-06 (hardware builder should construct CallRecorder(out_dir / "mllm_calls", secrets=[api_key], prices=config["prices"]))
  - Plans 15-07/15-08 (adapters put the wire body or model_dump object in raw_payload; config carries a dated prices table)
  - Phase 16 (audits calls from sidecars + cost_usd_est)
tech-stack:
  added: []
  patterns:
    - recorder dependency-injected into MllmActionSourceBase (no edit to the shared base)
    - one central redact()/redact_text() pass over every record, sidecar text and error string
    - whitelist-validated paths, atomic temp-then-replace writes (io_logger idiom)
key-files:
  created:
    - control/vla_bridge/mllm_record.py
    - control/test_mllm_record.py
  modified: []
decisions:
  - "An unknown (None) cached_input_tokens bills all input at the input price; only a None/invalid input or output count (or a missing required price) yields cost None"
  - "The base passes error as the exception object, not a string; the recorder formats it as '<Type>: <message>', redacted and capped at 2000 chars"
  - "Collector sidecar_files may only use depth.npy, aligned_left.png, depth_heatmap.png, depth_grid.txt; recorder-owned names (prompt.txt, request_meta.json, response_raw.json, thinking.txt) from a collector are ignored"
  - "write_call_sidecar returns files relative to the sidecar dir and image paths relative to the episode output dir, so record images[i].path joins directly onto <out>"
  - "Redaction also strips header-style values (Authorization/x-api-key/api-key: ...) in free text and JSON-escaped forms of each secret, as defence in depth"
  - "Non-finite floats become strings in to_json_safe so request_meta.json is written with allow_nan=False"
metrics:
  duration: "~14 min"
  completed: 2026-10-02
  tasks: 2
  tests_added: 71
  full_suite: "1204 passed (1133 baseline + 71 new)"
---

# Phase 15 Plan 04: MLLM per-call audit recorder Summary

Every MLLM provider call now leaves an audit trail with the key provably absent. A `CallRecorder` plugs into the existing `recorder` hook of `MllmActionSourceBase`. For each call it adds an estimated cost and a snapshot of the dated price table to the record. It also writes a sidecar directory, `mllm_calls/NNNN/`, holding the exact prompt, the image bytes sent, the redacted raw response, the full thinking text and the request metadata (settings requested, applied and omitted).

## What was built

- **`mllm_record.py`, Task 1 helpers:**
  - `redact_text` and `redact` replace each secret (4 or more characters) and its JSON-escaped form. They also replace the whole value of any `authorization`, `x-api-key`, `api-key`, `api_key` or `apikey` key, matched case-insensitively. The walk covers nested dicts, lists and tuples, dict keys included. Header-style `key: value` pairs in free text are also redacted.
  - `to_json_safe` converts the following to strict JSON:
    - objects with `model_dump(mode="json")`
    - numpy scalars and arrays (numpy is imported lazily)
    - paths, and bytes (stored as `{bytes, sha256}`)
    - datetimes
    - non-finite floats, which become strings
    - anything else becomes a repr capped at 2000 characters.
  - `estimate_cost_usd` bills `(input - cached)` tokens at the input price and cached tokens at the cached price. If no cached price is set, cached tokens use the input price. Output tokens use the output price, and reasoning tokens are not counted a second time. For the plan's acceptance case the result is `0.0332`.
  - `write_call_sidecar` checks the chunk id first: it must be an int of 0 or more, and not a bool or numpy int. It then checks every filename and image role against a whitelist, along with the media type and content types. All of this happens before any directory is created. Each file is written to a `.tmp` sibling and then moved into place with `Path.replace`.
- **`CallRecorder.finalize`, Task 2.** It works in this order:
  1. Deep-copies the record.
  2. Adds `cost_usd_est` and `price_table`.
  3. Builds the sidecar contents: `prompt.txt` (system prompt, a `USER MESSAGE` separator, the user text, then the images in send order with sha256 and size), `request_meta.json`, `response_raw.json`, `thinking.txt` and the whitelisted collector files.
  4. Writes the sidecar and sets `sidecar_dir`, `sidecar_files` and `images[i].path`.
  5. Ends with one more `redact(to_json_safe(record))` pass over the whole record.

  Two details about the files:
  - `response_raw.json` is skipped when the result is None. A string payload is the wire body, written verbatim after redaction. Any other payload is dumped as JSON.
  - `thinking.txt` holds the full text. The inline record keeps only the first 20000 characters.

  An `OSError`, `ValueError` or `TypeError` while building or writing becomes a redacted `sidecar_error`. Any other exception becomes a `recorder_error`. `finalize` never raises.

## Verification

- `pytest test_mllm_record.py test_base_mllm.py -k "record or redact or sidecar or cost or secret"`: 76 passed.
- `pytest test_mllm_record.py test_base_mllm.py`: 136 passed.
- Full suite: 1204 passed in about 9.5 s (1133 baseline plus 71 new).
- Acceptance checks:
  - `write_call_sidecar(tmp/"mllm_calls", 3, files={"../x": "y"}, ...)` raises `ValueError` and leaves `tmp_path` empty.
  - No `.tmp` files are left behind.
  - The integration run through `run_episode`, `IOLogger` and `MockRobot` creates `mllm_calls/0000/{prompt.txt, request_meta.json, response_raw.json, thinking.txt, images/{wrist,left,right}.png}`.
  - `cost_usd_est` matches the hand-computed `(1000*3 + 200*0.3 + 300*15)/1e6`.
  - Every file under `tmp_path` was read as bytes, and none contains the secret, its JSON-escaped form, or a header value that was not itself the secret.
  - In the integration tests the secrets were passed only to the recorder, not to the base. That proves the recorder alone keeps the key out of `episode.jsonl` and the sidecars.
- `git diff --stat 08be61d HEAD` touches only `control/vla_bridge/mllm_record.py` and `control/test_mllm_record.py`. `base_mllm.py` and `fake_providers.py` are unchanged.
- `test_mllm_failure.py` (named in the plan's acceptance command) is created by sibling plan 15-03 in a parallel worktree, so it does not exist at this base and could not be run here.

## TDD Gate Compliance

- Task 1: RED `0e16fb5` (ModuleNotFoundError), then GREEN `a512048`.
- Task 2: RED `7e7d496` (ImportError: `CallRecorder`), then GREEN `cec5a30`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] The base passes an exception object as `error`, not a string**
- **Found during:** Task 2
- **Issue:** The plan interface describes `error=<redacted error string or None>`, but `base_mllm._refill` passes the raw exception (`error=error`).
- **Fix:** Added `error_text()`. It accepts an exception or a string and returns `"<Type>: <message>"`, redacted and capped at 2000 characters. `request_meta.json` and `sidecar_error` both use it. The base is not edited.
- **Files modified:** control/vla_bridge/mllm_record.py
- **Commit:** cec5a30

**2. [Rule 1 - Bug] A test expectation was wrong**
- **Found during:** Task 2 GREEN
- **Issue:** The collector-sidecar test expected `thinking.txt`, but its scripted result has no thinking text.
- **Fix:** Corrected the expected file set in the test.
- **Commit:** cec5a30

### Interpretation choices

- When `cached_input_tokens` is None (unknown), all input is billed at the input price instead of returning None. A None input or output count still returns None. Without this, every OpenAI-compatible call would have no cost, because those providers do not report cached tokens.
- The collector cannot supply recorder-owned file names (prompt.txt and the other three). Names that are not whitelisted are silently ignored and never touch disk.

## Known Stubs

None.

## Threat Flags

None beyond the plan's threat register. The sidecar file writes are the surface T-15-10..T-15-13 already describes, and the mitigations are tested: central redaction plus an all-artifact scan, whitelisted paths with traversal tests, and `finalize` never raising.

## Self-Check: PASSED

- control/vla_bridge/mllm_record.py and control/test_mllm_record.py exist.
- Commits 0e16fb5, a512048, 7e7d496 and cec5a30 are present in `git log`.
