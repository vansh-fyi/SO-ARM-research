"""Tests for the shared MLLM prompt builder and defaults config (Phase 15).

`vla_bridge.mllm_prompt` (system prompt, per-call user text, vocabulary lint,
provenance hashes) and `backend_configs/mllm_defaults.json` (chunk_len and
its derivation, measured step duration). Decisions covered: D-01, D-09, D-16,
D-17, D-24, D-26, D-28 and research conflict C4. Pure-python; never touches
hardware or the network.
"""

import itertools
import json

import pytest

from vla_bridge import safety_validator
from vla_bridge.action_contract import ACTION_UNITS, JOINT_ORDER
from vla_bridge.backends import load_backend_config
from vla_bridge.mllm_prompt import (
    CAPS_HEADING,
    FORBIDDEN_PROMPT_TERMS,
    PRIMITIVE_TERMS,
    PROMPT_TEMPLATE_VERSION,
    URDF_HEADING,
    build_system_prompt,
    build_user_text,
    depth_scale_text,
    lint_prompt_text,
    template_sha256,
    text_sha256,
)
from vla_bridge.mllm_schema import build_action_chunk_schema

# Pinned after the first green run. Bump PROMPT_TEMPLATE_VERSION whenever
# this (or GOLDEN_SCHEMA_SHA256 in test_mllm_schema.py) changes.
GOLDEN_TEMPLATE_SHA256 = "ce29b4ed253d87e07e6341e6b85bb001ea839cb96c6400a5a7bfa817a0202b37"

LIMITS = {
    "shoulder_pan": (-70.33, 70.33),
    "shoulder_lift": (-107.47, 107.47),
    "elbow_flex": (-97.23, 97.23),
    "wrist_flex": (-102.02, 102.02),
    "wrist_roll": (-157.21, 162.79),
    "gripper": (0.0, 100.0),
}
JOINT_STATE = {joint: 1.23456 * (i + 1) for i, joint in enumerate(JOINT_ORDER)}
FAKE_URDF = '<robot name="so101"><joint name="shoulder_pan" type="revolute"/></robot>'


def _settings(**overrides):
    settings = load_backend_config("mllm_defaults")
    assert settings is not None
    settings = json.loads(json.dumps(settings))  # deep copy
    settings.update(overrides)
    return settings


def _section(prompt: str, heading: str) -> str:
    """Text of the headed section (up to the next blank-line-separated heading)."""
    start = prompt.index(heading)
    end = prompt.find("\n\n", start)
    return prompt[start:] if end == -1 else prompt[start:end]


# --- mllm_defaults.json (MLLM-05, D-02, D-17, D-24) ----------------------------


def test_defaults_chunk_len_and_measured_step_duration():
    config = load_backend_config("mllm_defaults")
    assert isinstance(config["chunk_len"], int) and not isinstance(config["chunk_len"], bool)
    assert config["chunk_len"] >= 1
    assert config["chunk_len"] == 20
    assert config["step_dt_s"] == 0.785
    assert "236" in config["step_dt_evidence"]
    assert config["depth_presentation"] == "both"
    assert config["depth"]["grid_rows"] == 24
    assert config["depth"]["grid_cols"] == 32
    assert config["left_image_source"] == "raw"
    assert config["disclose_safety_caps"] is False
    assert config["include_urdf"] is True


def test_defaults_chunk_len_derivation_is_documented():
    derivation = load_backend_config("mllm_defaults")["chunk_len_derivation"]
    for key in ("measured", "lower_bound", "upper_bound", "chosen", "duty_cycle_table", "not_measured"):
        assert key in derivation, key
    assert derivation["chosen"] == 20
    assert derivation["measured"]["tick_period_s_median"] == 0.785
    assert derivation["measured"]["depth_roundtrip_s"]["median"] == 6.15


# --- System prompt content (D-01, D-16, D-24, D-26) ---------------------------


def test_system_prompt_names_every_joint_with_unit_and_calibrated_limit():
    prompt = build_system_prompt(_settings(), LIMITS)
    joints = _section(prompt, "JOINTS AND UNITS")
    for joint in JOINT_ORDER:
        lo, hi = LIMITS[joint]
        line = next(line for line in joints.splitlines() if line.startswith(f"- {joint}:"))
        assert ACTION_UNITS[joint] in line
        assert f"{lo:.2f}" in line and f"{hi:.2f}" in line
    assert "absolute" in joints
    assert "0 = fully closed" in joints and "100 = fully open" in joints
    assert "calibrated midpoint" in joints


def test_system_prompt_states_measured_step_and_chunk_seconds():
    prompt = build_system_prompt(_settings(), LIMITS)
    assert "0.785" in prompt
    assert "20 steps" in prompt
    assert "15.7 s" in prompt
    assert "0.5 s" not in prompt  # D-24: never the nominal --control-hz step


def test_system_prompt_chunk_length_follows_settings():
    prompt = build_system_prompt(_settings(chunk_len=10), LIMITS)
    assert "10 steps" in prompt
    assert "7.9 s" in prompt  # 10 * 0.785 rounded to one decimal
    assert "exactly 10 steps" in prompt


def test_system_prompt_depth_alignment_completion_and_control_statements():
    prompt = build_system_prompt(_settings(), LIMITS)
    assert "approximately, not pixel-exactly, aligned" in prompt
    assert "rectified, resized left image" in prompt
    assert "already achieved in the CURRENT camera views" in prompt
    assert "no helper routines" in prompt
    assert "raw absolute joint targets" in prompt
    assert "wrist, left, right, depth" in prompt
    assert "previous plan" in prompt


def test_caps_disclosed_only_under_heading_when_enabled():
    hidden = build_system_prompt(_settings(disclose_safety_caps=False), LIMITS)
    shown = build_system_prompt(_settings(disclose_safety_caps=True), LIMITS)
    assert CAPS_HEADING == "PER-STEP CHANGE CAPS"
    assert CAPS_HEADING not in hidden
    assert CAPS_HEADING in shown
    caps_block = _section(shown, CAPS_HEADING)
    for joint, cap in safety_validator.MAX_RELATIVE_TARGET_DEG.items():
        assert f"- {joint}: {cap:g}" in caps_block
        assert f"- {joint}: {cap:g}" not in hidden
    outside = shown.replace(caps_block, "")
    for joint, cap in safety_validator.MAX_RELATIVE_TARGET_DEG.items():
        assert f"- {joint}: {cap:g}" not in outside


def test_caps_qualitative_clamp_sentence_always_present():
    for disclose in (False, True):
        prompt = build_system_prompt(_settings(disclose_safety_caps=disclose), LIMITS)
        assert "cannot be commanded to jump arbitrarily" in prompt
        assert "limited by a safety layer" in prompt


# --- Depth presentation (D-28) -------------------------------------------------


def test_depth_presentation_changes_prompt_text_and_hash():
    prompts = {p: build_system_prompt(_settings(depth_presentation=p), LIMITS)
               for p in ("heatmap", "grid", "both")}
    hashes = {p: text_sha256(text) for p, text in prompts.items()}
    assert len(set(hashes.values())) == 3


def test_depth_presentation_heatmap_mentions_colour_scale_from_range():
    settings = _settings(depth_presentation="heatmap")
    prompt = build_system_prompt(settings, LIMITS)
    scale = depth_scale_text(settings)
    assert scale in prompt
    assert "0.20 m" in scale and "1.50 m" in scale
    assert "blue = near" in scale and "red = far" in scale and "black = no valid depth" in scale
    assert "numeric depth grid" not in prompt


def test_depth_presentation_scale_follows_configured_range():
    settings = _settings()
    settings["depth"]["range_m"] = [0.3, 2.0]
    scale = depth_scale_text(settings)
    assert "0.30 m" in scale and "2.00 m" in scale


def test_depth_presentation_grid_mentions_numeric_grid_only():
    settings = _settings(depth_presentation="grid")
    prompt = build_system_prompt(settings, LIMITS)
    assert "numeric depth grid" in prompt
    assert "min/median/max" in prompt
    assert depth_scale_text(settings) not in prompt


def test_depth_presentation_both_mentions_heatmap_and_grid():
    settings = _settings(depth_presentation="both")
    prompt = build_system_prompt(settings, LIMITS)
    assert depth_scale_text(settings) in prompt
    assert "numeric depth grid" in prompt


# --- URDF block (D-07) -----------------------------------------------------------


def test_urdf_block_only_when_text_passed_and_changes_hash():
    without = build_system_prompt(_settings(), LIMITS)
    with_urdf = build_system_prompt(_settings(), LIMITS, urdf_text=FAKE_URDF)
    assert URDF_HEADING == "ARM KINEMATICS (URDF)"
    assert URDF_HEADING not in without
    assert URDF_HEADING in with_urdf
    assert FAKE_URDF in with_urdf
    assert with_urdf.index(URDF_HEADING) < with_urdf.index(FAKE_URDF)
    assert text_sha256(without) != text_sha256(with_urdf)


# --- Vocabulary lint (research C4, anti-features) ---------------------------------


@pytest.mark.parametrize(
    "presentation,disclose,urdf",
    list(itertools.product(("heatmap", "grid", "both"), (False, True), (None, FAKE_URDF))),
)
def test_forbidden_terms_absent_from_prompt_user_text_and_schema(presentation, disclose, urdf):
    settings = _settings(depth_presentation=presentation, disclose_safety_caps=disclose)
    prompt = build_system_prompt(settings, LIMITS, urdf_text=urdf)
    user = build_user_text(JOINT_STATE, "Pick up the pen", settings,
                           previous_plan="Lower the arm.", depth_note="Depth unavailable this call.")
    first_user = build_user_text(JOINT_STATE, "Pick up the pen", settings)
    schema_text = json.dumps(build_action_chunk_schema(settings["chunk_len"]))
    assert lint_prompt_text(prompt) == []
    assert lint_prompt_text(user) == []
    assert lint_prompt_text(first_user) == []
    assert lint_prompt_text(schema_text) == []


def test_forbidden_terms_lint_flags_every_term_case_insensitively():
    bad = ("Show your REASONING and Thinking, a trace, a scratchpad, chain-of-thought; "
           "then call move_to(x), grasp(obj), pick(obj), place(obj) via a waypoint "
           "or a trajectory helper.")
    hits = lint_prompt_text(bad)
    for term in ("reasoning", "thinking", "trace", "scratchpad", "chain-of-thought"):
        assert term in hits
    for term in PRIMITIVE_TERMS:
        assert term in hits
    assert set(FORBIDDEN_PROMPT_TERMS) >= {"reasoning", "thinking", "trace", "scratchpad",
                                           "chain-of-thought"}
    assert lint_prompt_text("Pick up the pen.") == []


# --- User text (D-09, D-17) -----------------------------------------------------------


def test_user_text_carries_task_joint_state_and_exact_chunk_reminder():
    text = build_user_text(JOINT_STATE, "Pick up the pen", _settings())
    assert "TASK: Pick up the pen" in text
    state_line = next(line for line in text.splitlines() if line.startswith("CURRENT JOINT STATE:"))
    state = json.loads(state_line.split(":", 1)[1])
    assert list(state) == list(JOINT_ORDER)
    for joint in JOINT_ORDER:
        assert state[joint] == round(JOINT_STATE[joint], 3)
    assert "Return exactly 20 steps." in text


def test_user_text_previous_plan_or_explicit_none_marker():
    first = build_user_text(JOINT_STATE, "Pick up the pen", _settings())
    assert "PREVIOUS PLAN: none (this is the first call)" in first
    later = build_user_text(JOINT_STATE, "Pick up the pen", _settings(), previous_plan="Lower the arm.")
    assert "PREVIOUS PLAN: Lower the arm." in later
    long_plan = "x" * 5000
    truncated = build_user_text(JOINT_STATE, "t", _settings(), previous_plan=long_plan)
    assert "x" * 2000 in truncated and "x" * 2001 not in truncated


def test_user_text_includes_depth_note_only_when_given():
    without = build_user_text(JOINT_STATE, "t", _settings())
    with_note = build_user_text(JOINT_STATE, "t", _settings(), depth_note="DEPTH GRID (m): ...")
    assert "DEPTH GRID (m): ..." in with_note
    assert "DEPTH GRID" not in without


# --- Provenance hashes (D-22b) -----------------------------------------------------------


def test_text_sha256_and_template_sha256():
    assert text_sha256("abc") == text_sha256("abc")
    assert text_sha256("abc") != text_sha256("abd")
    assert len(template_sha256()) == 64
    assert template_sha256() == template_sha256()
    assert PROMPT_TEMPLATE_VERSION == "mllm-prompt-1"


def test_template_sha256_matches_the_pinned_golden_hash():
    assert template_sha256() == GOLDEN_TEMPLATE_SHA256


def test_system_prompt_is_deterministic():
    assert build_system_prompt(_settings(), LIMITS) == build_system_prompt(_settings(), LIMITS)


# --- URDF block with the real So-101 text (D-07) -----------------------------------------


def test_system_prompt_with_real_urdf_text_is_lint_clean_and_changes_hash():
    from vla_bridge.mllm_urdf import urdf_kinematics_text

    urdf = urdf_kinematics_text()
    with_urdf = build_system_prompt(_settings(), LIMITS, urdf_text=urdf)
    without = build_system_prompt(_settings(), LIMITS)
    assert URDF_HEADING + "\n" + urdf in with_urdf
    assert lint_prompt_text(with_urdf) == []
    assert text_sha256(with_urdf) != text_sha256(without)


# --- Prompt-inspection CLI (keyless) --------------------------------------------------------


def _cli_expected(calibration, chunk_len=5, urdf=True, **overrides):
    from vla_bridge.action_contract import load_joint_limits_deg
    from vla_bridge.backends.base_mllm import load_mllm_settings
    from vla_bridge.mllm_urdf import urdf_kinematics_text

    settings = load_mllm_settings()
    settings.update(chunk_len=chunk_len, **overrides)
    limits = load_joint_limits_deg(calibration)
    return build_system_prompt(settings, limits, urdf_kinematics_text() if urdf else None)


def _cli_run(capsys, *argv):
    from vla_bridge import mllm_prompt

    assert mllm_prompt.main(list(argv)) == 0
    return capsys.readouterr().out


def test_prompt_cli_dumps_version_hashes_prompt_user_text_and_schema(mock_calibration_file, capsys):
    from vla_bridge.mllm_schema import schema_sha256

    out = _cli_run(capsys, "--joint-calibration", str(mock_calibration_file), "--chunk-len", "5")
    expected = _cli_expected(mock_calibration_file)
    schema = build_action_chunk_schema(5)
    assert f"prompt_template_version: {PROMPT_TEMPLATE_VERSION}" in out
    assert f"template_sha256: {template_sha256()}" in out
    assert f"system_prompt_sha256: {text_sha256(expected)}" in out
    assert f"schema_sha256: {schema_sha256(schema)}" in out
    assert "include_urdf: True" in out
    assert expected in out
    assert URDF_HEADING in out
    assert json.dumps(schema, indent=2) in out
    assert "TASK: Pick up the pen" in out
    assert "supplied per call" in out
    assert "Return exactly 5 steps." in out
    assert lint_prompt_text(out) == []


def test_prompt_cli_example_user_text_uses_joint_midpoints(mock_calibration_file, capsys):
    from vla_bridge.action_contract import load_joint_limits_deg

    out = _cli_run(capsys, "--joint-calibration", str(mock_calibration_file))
    limits = load_joint_limits_deg(mock_calibration_file)
    state_line = [line for line in out.splitlines() if line.startswith("CURRENT JOINT STATE: ")][0]
    state = json.loads(state_line.split(": ", 1)[1])
    for joint, (lo, hi) in limits.items():
        assert state[joint] == pytest.approx(round((lo + hi) / 2, 3))
    assert "Return exactly 20 steps." in out


def test_prompt_cli_no_urdf_removes_block_and_changes_printed_hash(mock_calibration_file, capsys):
    with_urdf = _cli_run(capsys, "--joint-calibration", str(mock_calibration_file), "--chunk-len", "5")
    without = _cli_run(capsys, "--joint-calibration", str(mock_calibration_file), "--chunk-len", "5",
                       "--no-urdf")
    assert URDF_HEADING not in without
    assert "include_urdf: False" in without
    expected = _cli_expected(mock_calibration_file, urdf=False)
    assert f"system_prompt_sha256: {text_sha256(expected)}" in without

    def sha_line(text):
        return [line for line in text.splitlines() if line.startswith("system_prompt_sha256: ")][0]

    assert sha_line(with_urdf) != sha_line(without)


def test_prompt_cli_instruction_and_depth_presentation_overrides(mock_calibration_file, capsys):
    out = _cli_run(capsys, "--joint-calibration", str(mock_calibration_file),
                   "--instruction", "Push the cup left", "--depth-presentation", "grid")
    expected = _cli_expected(mock_calibration_file, chunk_len=20, depth_presentation="grid")
    assert "TASK: Push the cup left" in out
    assert f"system_prompt_sha256: {text_sha256(expected)}" in out


def test_prompt_cli_missing_joint_calibration_exits(tmp_path):
    from vla_bridge import mllm_prompt

    with pytest.raises(SystemExit):
        mllm_prompt.main(["--joint-calibration", str(tmp_path / "missing.json")])
