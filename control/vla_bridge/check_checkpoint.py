"""Fetch and compare candidate fine-tuned checkpoints on the HF Hub, per backend.

Phase 11 (VLA Hardware Connection), Plan 11-03, Task 1 wrote this for
SmolVLA. Phase 14 (Plan 14-04 onward) extends it to every gRPC-bridge-lane
backend (pi0/pi05 here; ACT in 14-05, Diffusion Policy in 14-06): add a
`CANDIDATE_GROUPS["<backend>"]` list and run it with `--backend <backend>`.

For each candidate repo id this script:

1. Attempts `hf_hub_download(repo_id, filename="config.json")` and loads the
   JSON.
2. Reports the lerobot policy `type`, the HF revision sha it inspected, and
   whatever camera-key/count and action-dimension information is present
   under `input_features`/`output_features` (the exact keys vary by
   author/training run -- this script does not assume a fixed schema).
3. Falls back to `HfApi().list_repo_files(repo_id)` if `config.json` is
   missing, reporting the file listing instead of crashing.
4. Judges two kinds of fit: the Phase 11 camera-count fit against the
   2-physical-camera rig, and the gRPC-bridge key fit (see `_bridge_key_fit`):
   the bridge only ever delivers `observation.images.camera1` to the policy
   server, so a checkpoint without that key fails server-side.

Writes ONLY that backend's section of
`control/vla_bridge/checkpoint_candidates.md`, between
`<!-- backend:<name>:start -->` / `<!-- backend:<name>:end -->` markers.
Every other backend's section is left byte-for-byte untouched, so plans can
append their own sections independently. Every candidate gets a row (real
fetched data, or an explicit "fetch failed: <reason>" note -- none is
silently omitted).

Run with:
    cd control && .venv/bin/python vla_bridge/check_checkpoint.py --backend pi0
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

# One list per backend. Candidates come from HF Hub model searches
# (`/api/models?search=...`); the script records what each one really is.
CANDIDATE_GROUPS: dict[str, list[str]] = {
    # Phase 11, Plan 11-03.
    "smolvla": [
        "victorvanhalst/smolvla_so101_cube",
        "majinwakeup30/smolvla_so101_stack_cube_v3_2_cameras",
        "lerobot/svla_so100_pickplace",
        "cn0303/smolvla-so101-strawberry-v3",
    ],
    # Phase 14, Plan 14-04 (searches "pi0 so101", "pi0 so100", 2026-10-01).
    "pi0": [
        "amindt/pi0_so101_pickandplace_redblue",
        "Harrysunshine/so101-pi0-lora-sim-real-10task",
        "heyunzhenwhat/pi0_so101-single-transfer",
        "SkieyFly/pi0-so101_block_to_container_all-state_proj_true-interval_32-rand_true-state_cond_true",
        "lerobot/pi0_base",
    ],
    # Phase 14, Plan 14-04 (searches "pi05 so101", "pi0.5 so101", 2026-10-01).
    "pi05": [
        "yathAg/pi05_so101_pick_place",
        "zrgong/pi05_so101_pnp_blue_cube_yellow_cup",
        "majinwakeup30/pi05_so101_stack_cube_v3_2_cameras",
        "pdhinaka/pi05_so101_blue_cube_gray_box_cleaned_20k",
        "lerobot/pi05_base",
    ],
}

# Backward-compatible name for the Phase 11 SmolVLA list.
CANDIDATES = CANDIDATE_GROUPS["smolvla"]

SECTION_TITLES = {
    "smolvla": "SmolVLA (Phase 11, Plan 11-03)",
    "pi0": "pi0 (Phase 14, Plan 14-04)",
    "pi05": "pi0.5 / pi05 (Phase 14, Plan 14-04)",
}

OUT_PATH = Path(__file__).resolve().parent / "checkpoint_candidates.md"

_SAFE_BACKEND_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

# The key every gRPC-bridge checkpoint must declare. Installed lerobot 0.6.1:
# `RobotClient.__init__` builds `lerobot_features` from `robot.observation_features`,
# which only lists cameras in `robot_config.cameras` -- just `camera1` (wrist,
# added by `connect_bridge()`). The `camera2`/`camera3` stereo halves that
# `_wire_stereo_split_cameras()` adds to the raw observation are then dropped by
# `build_dataset_frame()` on the server. And `prepare_raw_observation()` indexes
# `policy_image_features[key]` for every key that does arrive, so a checkpoint
# without `observation.images.camera1` raises KeyError on every inference.
BRIDGE_IMAGE_KEY = "observation.images.camera1"


def _oneline(text: str) -> str:
    """Collapse a possibly-multi-line error message into a single line safe
    for embedding in a markdown table cell (newlines in a cell break the
    table's row structure and corrupt every subsequent row when rendered)."""
    return " ".join(str(text).split())

AR0144_CAVEAT = (
    "**Important caveat — the AR0144 \"overhead\" camera is one physical "
    "2560x720 side-by-side stereo frame, not two independent camera feeds** "
    "(confirmed in `diagnostics/UAT/function/basic/UAT.md`, Step 4). Any "
    "candidate that expects exactly 2 named camera inputs could plausibly map "
    "to `wrist` + one-half-of-stereo (or the whole stereo frame passed as a "
    "single wide image) — a candidate expecting 3 named inputs would need a "
    "3rd feed (a split of the stereo pair, or a dummy/empty camera) to fit "
    "this project's real 2-physical-camera rig."
)


def _extract_camera_and_action_info(config: dict) -> tuple[str, str]:
    """Best-effort extraction of camera keys/count and action dim from a
    LeRobot policy config.json. Schema is not assumed fixed across authors —
    report whatever keys actually exist.
    """
    camera_bits = []
    action_bits = []

    input_features = config.get("input_features")
    output_features = config.get("output_features")

    if isinstance(input_features, dict):
        image_keys = [
            k
            for k, v in input_features.items()
            if isinstance(v, dict) and str(v.get("type", "")).upper() == "VISUAL"
        ]
        if image_keys:
            camera_bits.append(f"input_features image keys: {image_keys} ({len(image_keys)} camera(s))")
        state_keys = [
            k
            for k, v in input_features.items()
            if isinstance(v, dict) and str(v.get("type", "")).upper() == "STATE"
        ]
        if state_keys:
            for k in state_keys:
                shape = input_features[k].get("shape")
                camera_bits.append(f"input_features state key `{k}` shape={shape}")

    if isinstance(output_features, dict):
        for k, v in output_features.items():
            if isinstance(v, dict):
                shape = v.get("shape")
                action_bits.append(f"output_features `{k}` type={v.get('type')} shape={shape}")

    # Fallback: some configs may nest under different top-level keys.
    if not camera_bits:
        for key in ("image_features", "cameras", "camera_keys"):
            if key in config:
                camera_bits.append(f"`{key}`: {config[key]}")

    if not action_bits:
        for key in ("action_dim", "action_feature", "output_shapes"):
            if key in config:
                action_bits.append(f"`{key}`: {config[key]}")

    camera_summary = "; ".join(camera_bits) if camera_bits else "none found in config.json"
    action_summary = "; ".join(action_bits) if action_bits else "none found in config.json"
    return camera_summary, action_summary


def _image_keys(config: dict) -> list[str]:
    input_features = config.get("input_features")
    if not isinstance(input_features, dict):
        return []
    return [
        k
        for k, v in input_features.items()
        if isinstance(v, dict) and str(v.get("type", "")).upper() == "VISUAL"
    ]


def _fetch_candidate(repo_id: str) -> dict:
    result = {
        "repo_id": repo_id,
        "revision": "n/a",
        "policy_type": "n/a",
        "fetch_status": None,
        "camera_keys": "n/a",
        "action_dim": "n/a",
        "task_description": "n/a",
        "rig_fit": "n/a",
        "bridge_fit": "n/a",
        "image_keys": None,
    }

    # Pin the exact revision inspected, so the row stays checkable later.
    revision = None
    files: list[str] = []
    try:
        info = HfApi().model_info(repo_id)
        revision = info.sha
        files = [s.rfilename for s in (info.siblings or [])]
        result["revision"] = revision or "n/a"
    except Exception:  # noqa: BLE001 - the config.json fetch below reports the failure
        pass

    try:
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)
    except Exception as e:  # noqa: BLE001 - report any fetch failure, don't crash
        # Fallback: list repo files instead of crashing.
        try:
            files = HfApi().list_repo_files(repo_id)
            result["fetch_status"] = _oneline(
                f"config.json missing ({e.__class__.__name__}); file listing: {files}"
            )
        except Exception as e2:  # noqa: BLE001
            result["fetch_status"] = _oneline(
                f"fetch failed: {e.__class__.__name__}: {e}; file listing also failed: "
                f"{e2.__class__.__name__}: {e2}"
            )
        return result

    has_processors = "policy_preprocessor.json" in files and "policy_postprocessor.json" in files
    result["fetch_status"] = (
        "OK (config.json fetched; "
        + ("policy_pre/postprocessor.json present" if has_processors
           else "NO policy_pre/postprocessor.json -- PolicyServer's make_pre_post_processors(pretrained_path=...) needs them")
        + ")"
    )

    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:  # noqa: BLE001
        result["fetch_status"] = f"fetch failed: config.json fetched but could not parse JSON: {e}"
        return result

    result["policy_type"] = str(config.get("type", "n/a"))
    result["image_keys"] = _image_keys(config)
    camera_summary, action_summary = _extract_camera_and_action_info(config)
    result["camera_keys"] = camera_summary
    result["action_dim"] = action_summary

    # Best-effort task description: LeRobot policy config.json does not
    # standardly carry a natural-language task description field; check a
    # couple of plausible keys, otherwise report "not present in config.json".
    task_desc = config.get("task") or config.get("dataset_repo_id") or config.get("dataset")
    result["task_description"] = str(task_desc) if task_desc else "not present in config.json"

    return result


def _rig_fit_judgment(camera_summary: str) -> str:
    """Human-readable judgment of how the reported camera count fits this
    project's real 2-camera (wrist + one-frame-stereo-overhead) rig.
    """
    if "none found" in camera_summary:
        return "UNKNOWN — config.json had no recognizable camera-key info; verify manually before selecting"
    # Count image keys mentioned, if the "(N camera(s))" pattern is present.
    m = re.search(r"\((\d+) camera\(s\)\)", camera_summary)
    if not m:
        return "UNKNOWN — could not parse camera count from config.json"
    n = int(m.group(1))
    if n == 2:
        return "GOOD FIT — expects exactly 2 camera inputs, matches wrist + overhead(-stereo-as-one-frame)"
    if n == 3:
        return "PARTIAL FIT — expects 3 camera inputs; needs a dummy/empty 3rd feed or a split of the AR0144 stereo frame into two halves"
    if n == 1:
        return "PARTIAL FIT — expects only 1 camera input; would need to pick a single feed (likely wrist) and drop the other"
    return f"UNCLEAR FIT — expects {n} camera inputs, does not cleanly map to this project's 2-physical-camera rig"


def _bridge_key_fit(image_keys: list[str] | None) -> str:
    """Will the gRPC bridge's observation load on this checkpoint at all?

    Only `observation.images.camera1` (wrist) reaches the policy server (see
    `BRIDGE_IMAGE_KEY`). The checkpoint must declare it; any other declared
    camera arrives missing, which pi0/pi05/SmolVLA mask as an empty camera.
    """
    if not image_keys:
        return "UNKNOWN — no image keys found in config.json"
    if BRIDGE_IMAGE_KEY in image_keys:
        others = [k for k in image_keys if k != BRIDGE_IMAGE_KEY]
        note = f"; {len(others)} other key(s) arrive missing (masked)" if others else ""
        return f"OK — declares camera1, which receives our wrist frame{note}"
    return "INCOMPATIBLE — no observation.images.camera1; PolicyServer raises KeyError on every observation"


def _section_markers(backend: str) -> tuple[str, str]:
    if not _SAFE_BACKEND_NAME.match(backend):
        raise ValueError(f"unsafe backend section name: {backend!r}")
    return f"<!-- backend:{backend}:start -->", f"<!-- backend:{backend}:end -->"


def replace_section(document: str, backend: str, section_body: str) -> str:
    """Replace `backend`'s marked section in `document` with `section_body`,
    or append it at the end if the section doesn't exist yet. Every other
    section is left untouched."""
    start, end = _section_markers(backend)
    block = f"{start}\n{section_body.strip()}\n{end}"
    if start in document and end in document:
        before, rest = document.split(start, 1)
        _old, after = rest.split(end, 1)
        return before + block + after
    if document and not document.endswith("\n"):
        document += "\n"
    return document + ("\n" if document else "") + block + "\n"


def render_section(backend: str, rows: list[dict]) -> str:
    title = SECTION_TITLES.get(backend, backend)
    lines = [
        f"## {title}",
        "",
        f"Generated by `check_checkpoint.py --backend {backend}`.",
        "",
        "| Repo ID | Revision | Policy type | Fetch status | Camera keys found | Action dim found | "
        "Task description | 2-camera rig fit | gRPC-bridge key fit |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        # Sanitize every cell: collapse newlines (which would otherwise break
        # the table's row structure) and escape literal `|` (which would
        # otherwise be misread as a column separator).
        cells = {k: _oneline(v).replace("|", "\\|") for k, v in row.items() if k != "image_keys"}
        lines.append(
            f"| `{cells['repo_id']}` | `{cells['revision']}` | {cells['policy_type']} | "
            f"{cells['fetch_status']} | {cells['camera_keys']} | {cells['action_dim']} | "
            f"{cells['task_description']} | {cells['rig_fit']} | {cells['bridge_fit']} |"
        )
    return "\n".join(lines)


FILE_HEADER = "\n".join([
    "# Checkpoint Candidates — Fetched Comparison",
    "",
    "Generated per backend by `control/vla_bridge/check_checkpoint.py --backend <name>`. "
    "Each backend owns one section between `<!-- backend:<name>:start/end -->` markers; "
    "re-running the script for one backend rewrites only that section.",
    "",
    AR0144_CAVEAT,
    "",
    "**Note:** `config.json` schema is not standardized across community "
    "checkpoint authors — the camera-key/action-dim columns report "
    "whatever was actually present in each fetched file, not a fixed "
    "assumed schema. Where a fetch failed, the fetch-failure reason (or "
    "the repo's raw file listing, if `config.json` itself was missing) is "
    "reported verbatim rather than omitting the candidate.",
    "",
])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="smolvla", choices=sorted(CANDIDATE_GROUPS))
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    rows = []
    for repo_id in CANDIDATE_GROUPS[args.backend]:
        print(f"Fetching {repo_id} ...")
        row = _fetch_candidate(repo_id)
        row["rig_fit"] = _rig_fit_judgment(row["camera_keys"])
        row["bridge_fit"] = _bridge_key_fit(row["image_keys"])
        rows.append(row)
        print(f"  -> {row['fetch_status']}")

    document = args.out.read_text() if args.out.exists() else FILE_HEADER
    args.out.write_text(replace_section(document, args.backend, render_section(args.backend, rows)))
    print(f"Wrote the {args.backend} section of {args.out}")


if __name__ == "__main__":
    main()
