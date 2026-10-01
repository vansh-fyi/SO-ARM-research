"""Tests for `vla_bridge/check_checkpoint.py`'s per-backend sections.

`checkpoint_candidates.md` is shared by several plans (14-04 pi0/pi05,
14-05 ACT, 14-06 Diffusion, plus Phase 11's SmolVLA table). Re-running the
script for one backend must only rewrite that backend's section. No network
access here: the HF fetch is never called.
"""

import pytest

from vla_bridge import check_checkpoint as cc


def test_candidate_groups_keep_smolvla_and_add_pi0_and_pi05():
    assert "victorvanhalst/smolvla_so101_cube" in cc.CANDIDATE_GROUPS["smolvla"]
    # Backward-compatible alias for the Phase 11 list.
    assert cc.CANDIDATES == cc.CANDIDATE_GROUPS["smolvla"]
    assert "amindt/pi0_so101_pickandplace_redblue" in cc.CANDIDATE_GROUPS["pi0"]
    assert "yathAg/pi05_so101_pick_place" in cc.CANDIDATE_GROUPS["pi05"]


def test_replace_section_appends_a_new_section_and_keeps_existing_ones():
    doc = (
        "# header\n\n"
        "<!-- backend:smolvla:start -->\nSMOLVLA ROWS\n<!-- backend:smolvla:end -->\n"
    )

    out = cc.replace_section(doc, "pi0", "PI0 ROWS")

    assert "SMOLVLA ROWS" in out
    assert out.index("SMOLVLA ROWS") < out.index("PI0 ROWS")
    assert "<!-- backend:pi0:start -->\nPI0 ROWS\n<!-- backend:pi0:end -->" in out


def test_replace_section_rewrites_only_its_own_section():
    doc = cc.replace_section("# header\n", "smolvla", "OLD SMOLVLA")
    doc = cc.replace_section(doc, "pi0", "OLD PI0")
    doc = cc.replace_section(doc, "pi05", "OLD PI05")

    out = cc.replace_section(doc, "pi0", "NEW PI0")

    assert "OLD PI0\n" not in out
    assert "NEW PI0" in out
    assert "OLD SMOLVLA" in out and "OLD PI05" in out
    assert out.count("<!-- backend:pi0:start -->") == 1
    # Section order is stable: rewriting pi0 doesn't move it after pi05.
    assert out.index("NEW PI0") < out.index("OLD PI05")


def test_replace_section_does_not_confuse_pi0_with_pi05_markers():
    doc = cc.replace_section("# header\n", "pi05", "PI05 ROWS")

    out = cc.replace_section(doc, "pi0", "PI0 ROWS")

    assert "PI05 ROWS" in out and "PI0 ROWS" in out
    assert out.count("<!-- backend:pi05:start -->") == 1


def test_replace_section_rejects_unsafe_backend_names():
    with pytest.raises(ValueError):
        cc.replace_section("", "bad name -->", "x")


@pytest.mark.parametrize(
    "image_keys, expected_prefix",
    [
        (["observation.images.camera1", "observation.images.camera2"], "OK"),
        (["observation.images.camera1", "observation.images.camera2", "observation.images.camera3"], "OK"),
        (["observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb"], "INCOMPATIBLE"),
        ([], "UNKNOWN"),
    ],
)
def test_bridge_key_fit_requires_camera1(image_keys, expected_prefix):
    assert cc._bridge_key_fit(image_keys).startswith(expected_prefix)


def test_render_section_has_one_row_per_candidate_and_escapes_pipes():
    rows = [
        dict(repo_id="a/b", revision="abc", policy_type="pi0", fetch_status="OK",
             camera_keys="x | y", action_dim="[6]", task_description="t",
             rig_fit="GOOD", bridge_fit="OK"),
        dict(repo_id="c/d", revision="n/a", policy_type="n/a", fetch_status="fetch failed:\nboom",
             camera_keys="n/a", action_dim="n/a", task_description="n/a",
             rig_fit="UNKNOWN", bridge_fit="UNKNOWN"),
    ]

    section = cc.render_section("pi0", rows)

    table_rows = [line for line in section.splitlines() if line.startswith("| `")]
    assert len(table_rows) == 2
    assert "x \\| y" in section
    assert "fetch failed: boom" in section


# --- ACT (Plan 14-05): no missing-camera masking --------------------------------------
#
# Installed lerobot 0.6.1 `ACTPolicy.predict_action_chunk()` builds
# `[batch[key] for key in self.config.image_features]`, so unlike pi0/pi05/SmolVLA
# a declared camera that the bridge never delivers is a KeyError, not a masked
# empty camera. For ACT the bridge fit is OK only when camera1 is the ONLY image key.


def test_candidate_groups_add_act():
    assert cc.CANDIDATE_GROUPS["act"]
    assert "act" in cc.SECTION_TITLES
    assert "act" in cc.NO_CAMERA_MASKING_BACKENDS
    assert "pi0" not in cc.NO_CAMERA_MASKING_BACKENDS


@pytest.mark.parametrize(
    "image_keys, expected_prefix",
    [
        (["observation.images.camera1"], "OK"),
        (["observation.images.camera1", "observation.images.camera2"], "INCOMPATIBLE"),
        (["observation.images.front"], "INCOMPATIBLE"),
        ([], "UNKNOWN"),
    ],
)
def test_bridge_key_fit_without_camera_masking_requires_camera1_only(image_keys, expected_prefix):
    assert cc._bridge_key_fit(image_keys, masks_missing_cameras=False).startswith(expected_prefix)


def test_bridge_key_fit_without_masking_names_the_undelivered_keys():
    fit = cc._bridge_key_fit(
        ["observation.images.camera1", "observation.images.camera2"], masks_missing_cameras=False
    )
    assert "observation.images.camera2" in fit
    assert "KeyError" in fit
