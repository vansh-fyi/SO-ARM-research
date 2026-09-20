"""Fetch and compare candidate fine-tuned SmolVLA checkpoints on the HF Hub.

Phase 11 (VLA Hardware Connection), Plan 11-03, Task 1.

For each candidate repo id (see RESEARCH.md Pitfall 2's "Candidate fine-tuned
checkpoints" table, all `[ASSUMED]` — found via WebSearch, not independently
inspected before this session), this script:

1. Attempts `hf_hub_download(repo_id, filename="config.json")` and loads the
   JSON.
2. Reports whatever camera-key/count and action-dimension information is
   present under `input_features`/`output_features` (LeRobot policy configs
   typically expose these, but the exact keys vary by author/training run —
   this script does not assume a fixed schema).
3. Falls back to `HfApi().list_repo_files(repo_id)` if `config.json` is
   missing, reporting the file listing instead of crashing.

Writes `control/vla_bridge/checkpoint_candidates.md` with one row per
candidate (real fetched data, or an explicit "fetch failed: <reason>" note —
no candidate silently omitted), plus an explicit callout that the AR0144
overhead camera is one physical 2560x720 stereo frame, not two independent
camera feeds.

Run with:
    cd control && .venv/bin/python vla_bridge/check_checkpoint.py
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

CANDIDATES = [
    "victorvanhalst/smolvla_so101_cube",
    "majinwakeup30/smolvla_so101_stack_cube_v3_2_cameras",
    "lerobot/svla_so100_pickplace",
    "cn0303/smolvla-so101-strawberry-v3",
]

OUT_PATH = Path(__file__).resolve().parent / "checkpoint_candidates.md"


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


def _fetch_candidate(repo_id: str) -> dict:
    result = {
        "repo_id": repo_id,
        "fetch_status": None,
        "camera_keys": "n/a",
        "action_dim": "n/a",
        "task_description": "n/a",
        "rig_fit": "n/a",
    }

    try:
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
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

    result["fetch_status"] = f"OK (config.json fetched to {config_path})"

    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:  # noqa: BLE001
        result["fetch_status"] = f"fetch failed: config.json fetched but could not parse JSON: {e}"
        return result

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
    import re

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


def main() -> None:
    rows = []
    for repo_id in CANDIDATES:
        print(f"Fetching {repo_id} ...")
        row = _fetch_candidate(repo_id)
        row["rig_fit"] = _rig_fit_judgment(row["camera_keys"])
        rows.append(row)
        print(f"  -> {row['fetch_status']}")

    lines = [
        "# SmolVLA Checkpoint Candidates — Fetched Comparison",
        "",
        "Generated by `control/vla_bridge/check_checkpoint.py` (Phase 11, Plan 11-03, Task 1).",
        "Replaces the `[ASSUMED]` WebSearch-only candidate table in "
        "`11-RESEARCH.md` with real fetched `config.json` data (or an explicit "
        "fetch-failure note) for each candidate.",
        "",
        AR0144_CAVEAT,
        "",
        "| Repo ID | Fetch status | Camera keys found | Action dim found | Task description | 2-camera rig fit |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        # Sanitize every cell: collapse newlines (which would otherwise break
        # the table's row structure) and escape literal `|` (which would
        # otherwise be misread as a column separator).
        cells = {k: _oneline(v).replace("|", "\\|") for k, v in row.items()}
        lines.append(
            f"| `{cells['repo_id']}` | {cells['fetch_status']} | {cells['camera_keys']} | "
            f"{cells['action_dim']} | {cells['task_description']} | {cells['rig_fit']} |"
        )

    lines.append("")
    lines.append(
        "**Note:** `config.json` schema is not standardized across community "
        "checkpoint authors — the camera-key/action-dim columns above report "
        "whatever was actually present in each fetched file, not a fixed "
        "assumed schema. Where a fetch failed, the fetch-failure reason (or "
        "the repo's raw file listing, if `config.json` itself was missing) is "
        "reported verbatim rather than omitting the candidate."
    )

    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
