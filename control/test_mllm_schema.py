"""Tests for the shared MLLM output schema (Phase 15, MLLM-04, D-04, D-10, D-11).

`vla_bridge.mllm_schema` builds the one action-chunk JSON schema every MLLM
provider receives, generated from `action_contract.JOINT_ORDER` and
`ACTION_UNITS`. Pure-python; never touches hardware or the network.
"""

import json

import pytest

from vla_bridge.action_contract import ACTION_UNITS, JOINT_ORDER
from vla_bridge.mllm_schema import (
    CONFIDENCE_LEVELS,
    SCENE_CLAIM_SOURCES,
    SCHEMA_NAME,
    TOP_LEVEL_FIELD_ORDER,
    UNSUPPORTED_SCHEMA_KEYWORDS,
    build_action_chunk_schema,
    lint_schema_subset,
    schema_sha256,
)

# Pinned after the first green run. Bump mllm_prompt.PROMPT_TEMPLATE_VERSION
# whenever this (or GOLDEN_TEMPLATE_SHA256 in test_mllm_prompt.py) changes, so
# per-call records stay comparable across runs.
GOLDEN_SCHEMA_SHA256 = "PIN_AFTER_FIRST_GREEN_RUN"


def _walk(node):
    """Yield every schema node (dict) reachable through properties/items."""
    yield node
    for child in node.get("properties", {}).values():
        yield from _walk(child)
    if isinstance(node.get("items"), dict):
        yield from _walk(node["items"])


def test_schema_name_constant():
    assert SCHEMA_NAME == "action_chunk"


def test_step_properties_are_generated_from_joint_order_with_units():
    schema = build_action_chunk_schema(20)
    step = schema["properties"]["actions"]["items"]
    assert list(step["properties"]) == list(JOINT_ORDER)
    for joint in JOINT_ORDER:
        prop = step["properties"][joint]
        assert prop["type"] == "number"
        assert ACTION_UNITS[joint] in prop["description"]
        assert "absolute" in prop["description"]


def test_schema_has_no_per_joint_bounds_or_unsupported_keywords():
    schema = build_action_chunk_schema(20)
    banned = {"minimum", "maximum", "minItems", "maxItems", "format", "pattern",
              "anyOf", "oneOf", "allOf", "$ref"}
    assert banned <= UNSUPPORTED_SCHEMA_KEYWORDS
    for node in _walk(schema):
        assert not (set(node) & UNSUPPORTED_SCHEMA_KEYWORDS), node


def test_every_object_node_is_closed_and_fully_required():
    schema = build_action_chunk_schema(20)
    objects = [node for node in _walk(schema) if node.get("type") == "object"]
    assert len(objects) == 3  # top level, step, scene claim
    for node in objects:
        assert node["additionalProperties"] is False
        assert node["required"] == list(node["properties"])


def test_top_level_field_order_is_the_staged_order():
    schema = build_action_chunk_schema(20)
    expected = ("task_restatement", "plan", "scene_claims", "actions", "task_complete")
    assert TOP_LEVEL_FIELD_ORDER == expected
    assert tuple(schema["properties"]) == expected
    assert tuple(schema["required"]) == expected
    assert schema["properties"]["task_complete"]["type"] == "boolean"


def test_scene_claim_source_and_confidence_enums():
    schema = build_action_chunk_schema(20)
    claim = schema["properties"]["scene_claims"]["items"]
    assert list(claim["properties"]) == ["claim", "source", "confidence"]
    assert claim["properties"]["source"]["enum"] == ["wrist", "left", "right", "depth"]
    assert claim["properties"]["confidence"]["enum"] == ["low", "medium", "high"]
    assert SCENE_CLAIM_SOURCES == ("wrist", "left", "right", "depth")
    assert CONFIDENCE_LEVELS == ("low", "medium", "high")


def test_actions_description_states_the_chunk_length():
    for n in (1, 7, 20):
        description = build_action_chunk_schema(n)["properties"]["actions"]["description"]
        assert f"Exactly {n} steps" in description
        assert "one per control tick" in description


@pytest.mark.parametrize("bad", [0, -3, True, False, 2.0, "20", None])
def test_invalid_chunk_len_is_rejected(bad):
    with pytest.raises(ValueError, match="chunk_len"):
        build_action_chunk_schema(bad)


def test_schema_sha256_is_deterministic_and_changes_with_chunk_len():
    a = schema_sha256(build_action_chunk_schema(20))
    b = schema_sha256(build_action_chunk_schema(20))
    c = schema_sha256(build_action_chunk_schema(21))
    assert a == b
    assert a != c
    assert len(a) == 64


def test_schema_sha256_matches_the_pinned_golden_hash_for_chunk_20():
    assert schema_sha256(build_action_chunk_schema(20)) == GOLDEN_SCHEMA_SHA256


def test_lint_schema_subset_accepts_the_built_schema():
    assert lint_schema_subset(build_action_chunk_schema(20)) is None


def test_lint_schema_subset_rejects_bounds_open_objects_and_partial_required():
    schema = build_action_chunk_schema(3)
    schema["properties"]["actions"]["items"]["properties"]["shoulder_pan"]["minimum"] = -90
    schema["properties"]["actions"]["minItems"] = 3
    schema["properties"]["scene_claims"]["items"]["additionalProperties"] = True
    schema["required"] = schema["required"][:-1]
    with pytest.raises(ValueError) as excinfo:
        lint_schema_subset(schema)
    message = str(excinfo.value)
    assert "minimum" in message
    assert "minItems" in message
    assert "additionalProperties" in message
    assert "required" in message


def test_schema_is_plain_json():
    schema = build_action_chunk_schema(20)
    assert json.loads(json.dumps(schema)) == schema
