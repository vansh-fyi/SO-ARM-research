"""The one action-chunk JSON schema every MLLM backend sends (Phase 15, MLLM-04).

Generated, never hand-written: the per-step object is built by looping over
`action_contract.JOINT_ORDER`, and each joint's unit comes from
`action_contract.ACTION_UNITS`. Every provider (Claude structured outputs,
OpenAI `json_schema`, HF router `response_format`) receives this same object,
so it uses only the intersection of their schema subsets:

  - types object / array / string / number / boolean, `enum`, `required`,
    `additionalProperties: false`, `description`;
  - NO numeric bounds (`minimum`/`maximum`): D-04 leaves out-of-range values
    to the unmodified `safety_validator`, so raw model behaviour stays
    visible in the logs;
  - NO `minItems`/`maxItems`: the chunk length N is stated in the prompt and
    in the `actions` description, and enforced after parsing (a short chunk
    holds the remainder, extra steps are dropped; see `base_mllm.parse_chunk`).

Top-level keys follow the staged order (D-10): `task_restatement`, `plan`,
`scene_claims`, `actions`, `task_complete`. Every property is required, so
both OpenAI (schema order) and Claude (required properties first, in schema
order) emit the staged fields before the actions. Scene claims are
`{claim, source, confidence}` objects with enum sources and confidences
(D-11).

The field descriptions are phrased as task deliverables. They never ask the
model to reproduce its internal deliberation, which Claude refuses with
`stop_reason: "refusal"` (research conflict C4); `mllm_prompt.lint_prompt_text`
checks this.

Usage:
    from vla_bridge.mllm_schema import build_action_chunk_schema, schema_sha256
    schema = build_action_chunk_schema(20)
    print(schema_sha256(schema))
"""

import hashlib
import json

from vla_bridge.action_contract import ACTION_UNITS, JOINT_ORDER

SCHEMA_NAME = "action_chunk"
SCENE_CLAIM_SOURCES = ("wrist", "left", "right", "depth")
CONFIDENCE_LEVELS = ("low", "medium", "high")
TOP_LEVEL_FIELD_ORDER = ("task_restatement", "plan", "scene_claims", "actions", "task_complete")

# Keywords outside the cross-provider subset (Claude structured outputs has no
# numeric or length constraints and no recursion; D-04 forbids per-joint bounds).
UNSUPPORTED_SCHEMA_KEYWORDS = frozenset({
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "minItems",
    "maxItems",
    "uniqueItems",
    "$ref",
    "anyOf",
    "oneOf",
    "allOf",
    "not",
})


def _closed_object(properties: dict) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def build_action_chunk_schema(chunk_len: int) -> dict:
    """Build the action-chunk schema for a fixed chunk length N (D-17)."""
    if isinstance(chunk_len, bool) or not isinstance(chunk_len, int) or chunk_len < 1:
        raise ValueError(f"chunk_len must be an int >= 1, got {chunk_len!r}")

    step = _closed_object({
        joint: {"type": "number", "description": f"{joint} absolute target, {ACTION_UNITS[joint]}"}
        for joint in JOINT_ORDER
    })
    claim = _closed_object({
        "claim": {"type": "string", "description": "One statement about the scene."},
        "source": {
            "type": "string",
            "enum": list(SCENE_CLAIM_SOURCES),
            "description": "The image this claim comes from.",
        },
        "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
    })
    properties = {
        "task_restatement": {"type": "string", "description": "Restate the task in one sentence."},
        "plan": {"type": "string", "description": f"Your plan for the next {chunk_len} steps."},
        "scene_claims": {
            "type": "array",
            "items": claim,
            "description": "Claims about the scene, each citing the image it came from.",
        },
        "actions": {
            "type": "array",
            "items": step,
            "description": f"Exactly {chunk_len} steps, one per control tick.",
        },
        "task_complete": {
            "type": "boolean",
            "description": "true only when the task goal is already achieved in the CURRENT camera views.",
        },
    }
    assert tuple(properties) == TOP_LEVEL_FIELD_ORDER
    return _closed_object(properties)


def schema_sha256(schema: dict) -> str:
    """Stable hash of a schema (sorted keys, compact separators) for per-call provenance."""
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _lint_node(node: dict, path: str, violations: list[str]) -> None:
    for keyword in sorted(set(node) & UNSUPPORTED_SCHEMA_KEYWORDS):
        violations.append(f"{path}: unsupported keyword {keyword!r}")
    if node.get("type") == "object":
        properties = node.get("properties", {})
        if node.get("additionalProperties") is not False:
            violations.append(f"{path}: additionalProperties must be false")
        if node.get("required") != list(properties):
            violations.append(f"{path}: required must list every property in order")
    for name, child in node.get("properties", {}).items():
        _lint_node(child, f"{path}.properties.{name}", violations)
    if isinstance(node.get("items"), dict):
        _lint_node(node["items"], f"{path}.items", violations)


def lint_schema_subset(schema: dict) -> None:
    """Raise `ValueError` listing every node outside the cross-provider subset."""
    violations: list[str] = []
    _lint_node(schema, "$", violations)
    if violations:
        raise ValueError("schema outside the cross-provider subset: " + "; ".join(violations))


__all__ = [
    "CONFIDENCE_LEVELS",
    "SCENE_CLAIM_SOURCES",
    "SCHEMA_NAME",
    "TOP_LEVEL_FIELD_ORDER",
    "UNSUPPORTED_SCHEMA_KEYWORDS",
    "build_action_chunk_schema",
    "lint_schema_subset",
    "schema_sha256",
]
