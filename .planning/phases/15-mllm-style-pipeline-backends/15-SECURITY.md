---
phase: 15
slug: mllm-style-pipeline-backends
status: complete
# threats_open = count of OPEN threats at or above workflow.security_block_on severity (the blocking gate)
threats_open: 0
asvs_level: 1
created: 2026-10-03
audit: none (closed by user decision; see Audit Trail)
---

# Phase 15 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

> **Not a verified audit.** On 2026-10-03, `/gsd-secure-phase 15` exited at its enforcement gate: no `secure-phase` step hook is active at `verify:post`, although `workflow.security_enforcement` is true in `.planning/config.json`. The user then chose to mark the security step complete. Every threat below is CLOSED by that user decision. A `gsd-security-auditor` has NOT checked the mitigations against the code, except for the live evidence cited in the Audit Trail.

---

## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| model output -> MllmActionSourceBase.parse_chunk | Untrusted JSON text from a cloud/open model crosses here before becoming joint targets |
| MllmActionSourceBase -> safety_validator (unmodified) | Parsed absolute targets are the only thing that can reach servos |
| exception text -> last_details / console | Provider SDK errors may echo request headers or keys |
| PyPI -> control/.venv | Third-party code, including freshly released versions, enters the environment that talks to robot hardware |
| model/provider response -> hold queue | Failure responses must never become motion or paid retries |
| run_episode loop <- action source flag | A duck-typed attribute decides episode termination |
| FastFS depth service <- two possible callers | Overlapping depth requests are rejected with 429 |
| provider response/exception text -> record and sidecar files | Third-party payloads and error strings may echo credentials |
| model output -> filesystem | Anything model-derived must not influence paths |
| lab images -> disk | Sidecars retain exact camera frames sent to a third party |
| control machine -> FastFS Colab endpoint | Raw stereo frames leave the machine; only one request may be in flight |
| control machine -> third-party model provider (via later plans) | Camera images, depth heatmap and grid are sent to a provider |
| depth error text -> prompt | Network errors can contain the rotating endpoint URL |
| process environment -> adapter | The only place the Anthropic key exists |
| control machine -> api.anthropic.com | Lab camera images and depth data leave the building |
| Claude response -> parse -> validator | Untrusted model output (covered by Plans 15-01/15-03) |
| control machine -> api.openai.com | Lab camera images and depth data leave the building |
| GPT response -> parse -> validator | Untrusted model output (Plans 15-01/15-03) |
| control machine -> router.huggingface.co -> pinned third-party provider | Lab images and depth data reach HF and then a provider whose retention policy differs per provider |
| model output -> parse -> validator | Untrusted output (Plans 15-01/15-03), more likely to be non-conforming when structured_mode is json_object |
| operator command line -> paid provider API (smoke tool) | A mistyped command could spend money |
| repo configs/docs -> secrets | Config and docs must never contain keys |
| operator shell environment -> provider APIs | Real keys exist only in the operator's shell during live runs |
| real SO-ARM101 hardware | Live model output ultimately moves a physical arm |
| lab camera images -> three provider families | Real workspace images are sent to Anthropic, OpenAI and HF-routed providers |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Plan | Status |
|-----------|----------|-----------|----------|-------------|------------|------|--------|
| T-15-01 | Tampering | parse_chunk / hold queue | high | mitigate | Per-step key whitelist from JOINT_ORDER, bool/NaN/Infinity/overflow rejected, first bad step and tail become hold of the live joint_state (D-03); well-formed out-of-range values still hit the unmodified validator (D-04); test matrix in test_base_mllm.py | 15-01 | closed (user decision 2026-10-03, not audited) |
| T-15-02 | Denial of service (economic) | refill failure path | high | mitigate | Every failure enqueues chunk_len holds so the next paid call happens only after N ticks; adapter script exhaustion fails tests loudly; test asserts adapter.calls stays 1 per chunk | 15-01 | closed (user decision 2026-10-03, not audited) |
| T-15-03 | Information disclosure | error text in last_details | high | mitigate | Exception messages pass through secret redaction and truncation to MAX_LOGGED_TEXT_CHARS before logging; repr omits secrets; fuller redaction of records and sidecars in Plan 04 | 15-01 | closed (user decision 2026-10-03, not audited) |
| T-15-04 | Tampering | prompt injection from scene text in camera views | medium | accept | Model can only emit schema-bound JSON (no tools, nothing executed); every value still bounded by the validator; scene claims logged for audit only | 15-01 | closed (accepted risk) |
| T-15-05 | Tampering | unpinned transitive upgrade on a later pip run | medium | accept | Direct pins fix the two SDKs; transitive versions recorded in provenance comments; a future lockfile is out of scope for this research repo | 15-02 | closed (accepted risk) |
| T-15-06 | Denial of service (economic) | persistent provider failure / refusal loop | high | mitigate | chunk_len holds per failure, consecutive-failure cap ends the episode with mllm_backend_failed, post-complete guard blocks any call after a terminal flag; tests assert adapter.calls | 15-03 | closed (user decision 2026-10-03, not audited) |
| T-15-07 | Tampering | model claims task_complete then emits motion | medium | mitigate | A strict boolean true holds the entire chunk (D-25) so no unexamined motion follows a done claim; claim logged for Phase 16 false-completion analysis (D-19) | 15-03 | closed (user decision 2026-10-03, not audited) |
| T-15-08 | Tampering | termination hook triggered by a non-string or mock attribute | low | mitigate | Hook requires a non-empty str; tests cover None, empty and object values | 15-03 | closed (user decision 2026-10-03, not audited) |
| T-15-09 | Denial of service | backend depth call collides with a second depth client (HTTP 429) | medium | mitigate | main() never builds a depth client or DepthRecorder for MLLM backends; the backend owns one synchronous call per chunk (Plan 15-05, 15-06) | 15-03 | closed (user decision 2026-10-03, not audited) |
| T-15-10 | Information disclosure | records, sidecars, error strings | high | mitigate | Central redact()/redact_text() on every record, sidecar text and error; header-key value redaction; test scans every produced file for the secret | 15-04 | closed (user decision 2026-10-03, not audited) |
| T-15-11 | Tampering | sidecar path construction | high | mitigate | Integer chunk id, fixed filename and role whitelists validated before any write; ValueError on anything else (test with traversal attempts) | 15-04 | closed (user decision 2026-10-03, not audited) |
| T-15-12 | Denial of service | disk full or unwritable output directory | medium | mitigate | finalize never raises; failure becomes sidecar_error, loop keeps holding/running | 15-04 | closed (user decision 2026-10-03, not audited) |
| T-15-13 | Information disclosure | camera images retained on disk | low | accept | Sidecars live under git-ignored control/outputs; provider retention is disclosed at build time (Plan 15-06) | 15-04 | closed (accepted risk) |
| T-15-14 | Information disclosure | depth failure text reaching the model prompt | medium | mitigate | Failed or missing depth always yields the fixed note "Depth unavailable this call."; error text goes only into truncated meta (tested with a URL-bearing error) | 15-05 | closed (user decision 2026-10-03, not audited) |
| T-15-15 | Denial of service | HTTP 429 from overlapping depth requests | medium | mitigate | One synchronous depth call per chunk, 2 retries with 2 s sleeps, failure never blocks the chunk; main() never starts a second depth caller (Plan 15-03) | 15-05 | closed (user decision 2026-10-03, not audited) |
| T-15-16 | Information disclosure | lab images and depth sent to a third party | medium | accept | Operator notice printed at build time (Plan 15-06) and retention documented per provider; sidecars stay in git-ignored control/outputs | 15-05 | closed (accepted risk) |
| T-15-17 | Tampering | misleading kinematics context (unverified joint sign) | medium | mitigate | URDF text omits any sign claim; include_urdf toggle changes the prompt hash and is logged per call; human sign check scheduled in Plan 15-10 | 15-05 | closed (user decision 2026-10-03, not audited) |
| T-15-18 | Information disclosure | API key in logs, repr, config, sidecars | high | mitigate | Env-var NAME only in config (regex rejects a pasted key without echoing it); SystemExit before hardware if unset; secrets passed to base and recorder for redaction; tests scan every artifact | 15-06 | closed (user decision 2026-10-03, not audited) |
| T-15-19 | Denial of service (economic) | runaway or retried paid calls | high | mitigate | max_retries 0 on the SDK client, config-pinned max_tokens 64000 and effort, failure = chunk_len holds plus consecutive-failure cap (Plan 15-03), per-call cost estimate in every record | 15-06 | closed (user decision 2026-10-03, not audited) |
| T-15-20 | Information disclosure | images retained by Anthropic (30 days, not ZDR) | medium | mitigate | NOTICE printed at build time with the config's data_retention_note; operator must accept before live runs (Plan 15-10) | 15-06 | closed (user decision 2026-10-03, not audited) |
| T-15-21 | Tampering | reasoning-extraction refusal or prefill/forced-tool errors | medium | mitigate | Lint-clean prompt/schema field names (Plan 15-01), no tools or prefill sent, refusal mapped to a held failed call with stop details logged | 15-06 | closed (user decision 2026-10-03, not audited) |
| T-15-22 | Tampering | request kwargs drifting from the installed SDK | low | mitigate | Test binds the exact kwargs to the installed Messages.stream signature | 15-06 | closed (user decision 2026-10-03, not audited) |
| T-15-23 | Information disclosure | OpenAI key in logs, repr, config, sidecars | high | mitigate | Env-var NAME only in config; SystemExit before hardware; secrets redacted by base and recorder; end-to-end test scans every produced file | 15-07 | closed (user decision 2026-10-03, not audited) |
| T-15-24 | Denial of service (economic) | runaway or retried paid calls, large reasoning budgets | high | mitigate | max_retries 0, config-pinned max_output_tokens 64000 and effort, failure = chunk_len holds plus cap, per-call cost estimate recorded | 15-07 | closed (user decision 2026-10-03, not audited) |
| T-15-25 | Information disclosure | lab images processed by OpenAI | medium | mitigate | store=false on every request, NOTICE with retention note at build time, operator acknowledges before live runs (Plan 15-10) | 15-07 | closed (user decision 2026-10-03, not audited) |
| T-15-26 | Tampering | unsupported parameter negotiation causing silent behavior change | low | mitigate | Static capabilities table decides what is sent; omitted params logged with reasons; request kwargs bound to the installed SDK signature in tests | 15-07 | closed (user decision 2026-10-03, not audited) |
| T-15-27 | Information disclosure | HF token in logs, repr, config, sidecars, wire-body sidecar | high | mitigate | Env-var NAME only; SystemExit before hardware; secrets redacted from the wire body before it is written; end-to-end tests scan every produced file | 15-08 | closed (user decision 2026-10-03, not audited) |
| T-15-28 | Tampering | silent loss of JSON-schema mode via default :fastest routing | high | mitigate | Config validation requires a pinned lowercase provider and rejects routing-policy names; request model string asserted exactly; structured_mode applied is logged per call | 15-08 | closed (user decision 2026-10-03, not audited) |
| T-15-29 | Denial of service (economic) | expensive Kimi-K3 runs, large reasoning budgets, retries | medium | mitigate | max_retries 0, max_tokens 32000 with context-window reserve check, per-call cost from dated prices, failure = chunk_len holds plus cap; Kimi's cost flagged in config notes (research C1) | 15-08 | closed (user decision 2026-10-03, not audited) |
| T-15-30 | Information disclosure | images forwarded to third-party providers with varying retention | medium | mitigate | NOTICE with the config's retention note at build time; operator acknowledges before live runs (Plan 15-10) | 15-08 | closed (user decision 2026-10-03, not audited) |
| T-15-31 | Tampering | json_object fallback yields non-conforming output | medium | mitigate | Output still passes per-step pre-validation and the unmodified validator; fenced JSON tolerated and flagged; the switch is human-set after a live check, never automatic | 15-08 | closed (user decision 2026-10-03, not audited) |
| T-15-32 | Denial of service (economic) | smoke tool spending without intent | medium | mitigate | Requires --confirm-spend; one call, chunk_len default 3; prints the plan and exits 2 without reading the key otherwise | 15-09 | closed (user decision 2026-10-03, not audited) |
| T-15-33 | Information disclosure | key-shaped strings in configs or smoke output | high | mitigate | Config scan test for key shapes; api_key_env NAME validation; smoke output and files scanned for the secret in tests | 15-09 | closed (user decision 2026-10-03, not audited) |
| T-15-34 | Tampering | a backend quietly gaining tool-use or dynamic code execution | medium | mitigate | AST scans forbid tools/tool_choice keywords, eval/exec and top-level SDK imports in all MLLM modules | 15-09 | closed (user decision 2026-10-03, not audited) |
| T-15-35 | Information disclosure | keys exported or pasted into files during live runs | high | mitigate | Export in the shell only; check every sidecar and the results file for key shapes before committing; no key ever written to configs | 15-10 | closed (user decision 2026-10-03, not audited) |
| T-15-36 | Tampering (physical safety) | live model output moving the real arm | high | mitigate | Unmodified validator hard limits (joint clamping, NaN/inf rejection) and software e-stop stay active; hand on Ctrl-C; clear workspace; first run low-risk instruction; hold and Ctrl-C paths verified in this plan | 15-10 | closed (user decision 2026-10-03, not audited) |
| T-15-37 | Denial of service (economic) | spend during live runs | medium | mitigate | --confirm-spend gate, tiny chunk smoke first, cheapest-first order, per-call cost estimate recorded, explicit dated deferral allowed for unfunded variants | 15-10 | closed (user decision 2026-10-03, not audited) |
| T-15-38 | Information disclosure | lab images retained by providers | medium | accept | Operator was shown the retention NOTICE at build time; recorded in the results file that the operator accepted it before live runs | 15-10 | closed (accepted risk) |

*Status: open · closed · open — below high threshold (non-blocking)*
*Severity: critical > high > medium > low — only open threats at or above workflow.security_block_on count toward threats_open*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-15-01 | T-15-04 | Model can only emit schema-bound JSON (no tools, nothing executed); every value still bounded by the validator; scene claims logged for audit only | plan-time disposition | plan time |
| AR-15-02 | T-15-05 | Direct pins fix the two SDKs; transitive versions recorded in provenance comments; a future lockfile is out of scope for this research repo | plan-time disposition | plan time |
| AR-15-03 | T-15-13 | Sidecars live under git-ignored control/outputs; provider retention is disclosed at build time (Plan 15-06) | plan-time disposition | plan time |
| AR-15-04 | T-15-16 | Operator notice printed at build time (Plan 15-06) and retention documented per provider; sidecars stay in git-ignored control/outputs | plan-time disposition | plan time |
| AR-15-05 | T-15-38 | Operator was shown the retention NOTICE at build time; recorded in the results file that the operator accepted it before live runs | plan-time disposition + operator (retention notice accepted) | 2026-10-02 |
| AR-15-06 | all mitigate-disposition threats (T-15-01..T-15-37) | Mitigations were not independently audited because the secure-phase gate is inactive; the user chose to mark the security step complete | user | 2026-10-03 |

---

## Audit Trail

## Security Audit 2026-10-03

| Metric | Count |
|--------|-------|
| Threats found | 38 |
| Closed | 38 |
| Open | 0 |
| Formally audited | 0 |

Live evidence recorded in 15-LIVE-RESULTS.md (Plan 15-10) bears on the high-severity live threats, but it is not a code audit:
- **T-15-35 (keys):** keys were exported in the operator's shell only. Key-shape scans found nothing in the results file or in any Task 3 episode output. The phase verifier's scan of all live output folders also found nothing.
- **T-15-36 (physical safety):** every executed step went through the unmodified `safety_validator.py`, and the verifier confirmed zero changes to it across the phase. The hold path was observed (empty-response and truncated holds). Ctrl-C returned the arm to start; the termination-reason defect was fixed in 9ac4472 and re-tested.
- **T-15-37 (spend):** cheapest backend first, the `--confirm-spend` gate on smoke calls, and per-call cost estimates were recorded.
- **T-15-38 (retention):** the operator accepted the retention notice on 2026-10-02.

Incident outside the register: on 2026-10-03, an Anthropic API key from the user's gitignored `.env` (never committed) was attached to the Claude Code session through an editor selection. The user was advised to rotate it.

Follow-up: either enable the secure-phase step hook via `/gsd-settings` or set `workflow.security_enforcement: false`, so the config and the hook registry agree.
