---
phase: quick-260927-ndz
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - control/vla_bridge/policy_server_launch.md
  - control/vla_bridge/policy_server.ipynb
autonomous: true
requirements: []

must_haves:
  truths:
    - "The Flask /depth handler in both policy_server_launch.md's Step 8 and policy_server.ipynb's matching cell invokes FastFS's real, documented scripts/run_demo.py CLI (per docs/Fast Foundation Stereo Readme.md's 'Run demo' section) via subprocess, writing the intrinsic file in the exact documented format (line 1 = flattened 1x9 intrinsics matrix, line 2 = baseline in meters) -- no fabricated FastFS Python class/function names are introduced."
    - "The one genuinely undocumented FastFS detail (run_demo.py's exact output filename inside --out_dir) is marked honestly via a single, clearly-labeled DEPTH_OUTPUT_GLOB constant with a TODO comment for a human to confirm after a real Colab run, not guessed with false confidence."
    - "Both files' Step 9 depth-tunnel cell comment, and the .md's own Summary table 'Tunnel type' row, correctly state Step 3's gRPC PolicyServer port as 5173, matching the actual PolicyServerConfig(port=5173) code and Step 4's ngrok.connect(5173, \"tcp\") call -- no remaining reference to the stale, incorrect port 8080 for that purpose."
    - "policy_server.ipynb's cell 0 Camera mapping section matches policy_server_launch.md's current wording, documenting the AR0144 at 1600x600 (800x600 per half) split into two 800-wide halves -- no remaining reference to the stale, broken 2560x720/1280-split resolution."
    - "policy_server.ipynb contains exactly one tunnel-open cell for the gRPC PolicyServer (the commented cell, id fc7fd0ee), not two -- running the notebook top-to-bottom opens only one gRPC tunnel."
    - "control/vla_bridge/depth_camera.py is unmodified, and its request/response contract (left_png_b64/right_png_b64/intrinsics_flat/baseline_m request, depth_npy_b64 response) still exactly matches both files' documented Flask handler after the edits."
    - "policy_server.ipynb remains valid, parseable JSON after all edits (json.load succeeds), with nbformat/kernelspec/metadata and every untouched cell's formatting unchanged."
  artifacts:
    - control/vla_bridge/policy_server_launch.md
    - control/vla_bridge/policy_server.ipynb
  key_links:
    - "Step 7's git-clone cell -> new pip-install cell(s) -> Step 8's run_fastfs_inference() subprocess call, all referencing the same /content/Fast-FoundationStereo clone directory and weights/23-36-37/model_best_bp2_serialize.pth checkpoint path, kept consistent across both files."
    - "depth_camera.py's compute_depth() request/response field names <-> both files' Flask /depth handler -- unchanged in this task, re-verified not to have drifted."
---

<objective>
Close Phase 12 UAT test-5's gap in `control/vla_bridge/policy_server_launch.md` and `control/vla_bridge/policy_server.ipynb`: wire real Fast-FoundationStereo inference into the Flask depth cell using only the documented `scripts/run_demo.py` CLI contract, fix a wrong port-8080 comment (should be 5173) in both files, sync the notebook's stale 2560x720/1280-split Camera mapping cell to the .md's current 1600x600/800-split content, and remove the notebook's duplicate tunnel-open cell.

Purpose: `depth_camera.DepthCameraClient` (Plan 12-06) already POSTs real rectified stereo pairs to the documented Colab Flask endpoint, but that endpoint's own handler currently calls an undefined stub (`run_fastfs_inference`) -- the depth pipeline cannot actually run end-to-end until the Colab-side cell has a real implementation. The stale notebook camera-mapping text and duplicate tunnel cell are separate but adjacent doc-drift bugs flagged by the same UAT test.
Output: Both files updated so a human running the notebook (or following the .md) top-to-bottom gets a working FastFS depth inference cell, correct port documentation, and a notebook that matches the .md and the real hardware.
</objective>

<execution_context>
@/Users/hp/Desktop/Work/Repositories/SoARM-Research/.claude/gsd-core/workflows/execute-plan.md
@/Users/hp/Desktop/Work/Repositories/SoARM-Research/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md

Files to edit (read in full before editing):
@control/vla_bridge/policy_server_launch.md
@control/vla_bridge/policy_server.ipynb

Reference only -- do NOT modify. Its request/response contract already matches both target files and must remain unchanged after this task:
@control/vla_bridge/depth_camera.py

The ONLY source of truth for FastFS's actual API/CLI -- do not fabricate any class/function/flag not present here:
@docs/Fast Foundation Stereo Readme.md

Background (do not edit, context only):
@.planning/phases/12-bridge-tick-latency-fix/12-UAT.md

**Notebook cell-id map (query by id at runtime, never by a hardcoded index -- Task 1's insertion shifts every later index):**
- `5da4b39e` — cell 0, the full preamble markdown cell (title, checkpoint decision, pyngrok gate, and the stale Camera mapping section) — Task 2 replaces its entire `source` with the block below.
- `6e04ff98` — Step 7's `!git clone .../Fast-FoundationStereo` code cell — Task 1 inserts the new pip-install cell immediately after this one.
- `2e55f487` — the markdown cell right after the clone cell ("Then follow that repo's own README...") — Task 1 replaces its `source`.
- `8473a692` — Step 8's Flask cell (currently calls the undefined `run_fastfs_inference` stub) — Task 1 replaces its `source` in full.
- `553ee9ec` — Step 9's `depth_tunnel = ngrok.connect(3000, "http")` cell, containing the wrong "(8080)" comment — Task 2 fixes this one line.
- `wd94PYZ53t39` — the duplicate, uncommented `tunnel = ngrok.connect(5173, "tcp")` cell (current index 13) — Task 2 **deletes** this cell.
- `fc7fd0ee` — the commented `tunnel = ngrok.connect(5173, "tcp")` cell (current index 14) — keep unchanged, this is the one that survives.

**Notebook JSON formatting to preserve exactly (verified against the live file):** `nbformat: 4`, `nbformat_minor: 5`; top-level keys `cells`/`metadata`/`nbformat`/`nbformat_minor` in that order; write with `json.dump(nb, f, indent=2, ensure_ascii=False)` followed by a single trailing `"\n"` (the file currently ends in `}\n`). Each markdown/code cell's `source` is a list of strings, each ending in `"\n"` except the **last** line of the cell, which has no trailing newline (verified live on cell 0 and others) -- when building a new/replacement `source` list from a multi-line text block, use `text.splitlines(keepends=True)` and strip the trailing `"\n"` from the final element only. New code cells need the same keys as existing unrun cells in this file: `cell_type: "code"`, `execution_count: None`, `id: <new unique id>`, `metadata: {}`, `outputs: []`, `source: [...]`. Pick a new id with e.g. `uuid.uuid4().hex[:8]` and confirm it does not collide with any existing cell id.

**Target Step 7 addition — new pip-install cell content (used in both files, inserted right after the git-clone cell/fence, still under the existing "DO NOT RUN until the human has completed the Fast-FoundationStereo legitimacy check above" gate -- no new checkpoint needed):**
```python
# Colab notebook cell -- gated by the same Fast-FoundationStereo legitimacy
# check above (do not run before that check and before Step 7's clone cell).
# FastFS's documented "Option 2: pip" environment setup (docs/Fast Foundation
# Stereo Readme.md, "Environment setup" section), run inside the just-cloned
# repo directory.
%cd /content/Fast-FoundationStereo
!pip install torch==2.6.0 torchvision==0.21.0 xformers --index-url https://download.pytorch.org/whl/cu124
!pip install -r requirements.txt
```

**Target Step 7 follow-up paragraph (replaces the existing "Then follow that repo's own README..." paragraph, in both files):**
```
Then manually download the `23-36-37` checkpoint from the README's Google
Drive folder (https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap)
and place `model_best_bp2_serialize.pth` at
`/content/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth`
in the Colab runtime -- a manual human step (a gated/shared Google Drive
folder isn't reliably scriptable inside a single Colab cell), not something
this notebook automates.

Step 8 below invokes FastFS's own documented `scripts/run_demo.py` CLI
(README's "Run demo" section) directly, so the input/output CONTRACT (flags,
intrinsic-file format) is fully documented there -- no fabricated
function/class names. The one genuinely unresolved detail is
`run_demo.py`'s exact output filename inside `--out_dir`, which the README
does not state -- Step 8's `DEPTH_OUTPUT_GLOB` marks that spot for a human
to correct after a first real Colab run, matching this file's own existing
precedent for an honestly-unresolved implementation detail (see the
`camera2`/`camera3` `type: <split-stereo-left/right>` placeholder note in
the Camera mapping section above).
```

**Target Step 8 Flask cell (full replacement of the existing code fence/cell, in both files -- the `/depth` route body's request parsing and JSON response stay exactly as they are today, only the stub call becomes a real implementation):**
```python
# Colab notebook cell
!pip install flask  # widely-used, long-established -- no dedicated
                     # legitimacy check needed, unlike pyngrok/FastFS above

import base64
import glob
import io
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

FASTFS_DIR = "/content/Fast-FoundationStereo"
FASTFS_MODEL = f"{FASTFS_DIR}/weights/23-36-37/model_best_bp2_serialize.pth"

# run_demo.py's README does not state its output filename inside --out_dir --
# this glob picks the most plausible depth output file. Confirm the exact
# filename by running once in Colab and inspecting out_dir; narrow this
# pattern if more than one .npy file is written (e.g. an intermediate file).
DEPTH_OUTPUT_GLOB = "*.npy"  # TODO: confirm exact filename by running once in Colab and inspecting out_dir


def run_fastfs_inference(left_img, right_img, intrinsics_flat, baseline_m):
    """Invokes Fast-FoundationStereo's documented `scripts/run_demo.py` CLI
    (docs/Fast Foundation Stereo Readme.md, "Run demo" section) as a
    subprocess, then loads the resulting depth map back into this process."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        left_path = str(Path(tmp_dir) / "left.png")
        right_path = str(Path(tmp_dir) / "right.png")
        intrinsic_path = str(Path(tmp_dir) / "K.txt")
        out_dir = str(Path(tmp_dir) / "out")
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(left_path, left_img)
        cv2.imwrite(right_path, right_img)

        # README: line 1 = flattened 1x9 intrinsics matrix (space-separated),
        # line 2 = baseline in meters.
        with open(intrinsic_path, "w") as f:
            f.write(" ".join(str(v) for v in intrinsics_flat) + "\n")
            f.write(f"{baseline_m}\n")

        subprocess.run(
            [
                "python", "scripts/run_demo.py",
                "--model_dir", FASTFS_MODEL,
                "--left_file", left_path,
                "--right_file", right_path,
                "--intrinsic_file", intrinsic_path,
                "--out_dir", out_dir,
                "--remove_invisible", "0",
                "--denoise_cloud", "0",
                "--scale", "1",
                "--get_pc", "0",
                "--valid_iters", "8",
                "--max_disp", "192",
                "--zfar", "100",
            ],
            cwd=FASTFS_DIR,
            check=True,
        )

        npy_files = sorted(glob.glob(str(Path(out_dir) / DEPTH_OUTPUT_GLOB)))
        if not npy_files:
            raise RuntimeError(
                f"No files matching {DEPTH_OUTPUT_GLOB!r} found in {out_dir} "
                "after run_demo.py -- adjust DEPTH_OUTPUT_GLOB above."
            )
        # Picks the first alphabetical match if run_demo.py writes more than
        # one .npy file -- narrow DEPTH_OUTPUT_GLOB above if this is wrong.
        return np.load(npy_files[0])


@app.route("/depth", methods=["POST"])
def depth():
    body = request.get_json()
    left_bytes = base64.b64decode(body["left_png_b64"])
    right_bytes = base64.b64decode(body["right_png_b64"])
    left_img = cv2.imdecode(np.frombuffer(left_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    right_img = cv2.imdecode(np.frombuffer(right_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    intrinsics_flat = body["intrinsics_flat"]
    baseline_m = body["baseline_m"]

    depth_map = run_fastfs_inference(left_img, right_img, intrinsics_flat, baseline_m)

    buf = io.BytesIO()
    np.save(buf, depth_map.astype(np.float32))
    depth_npy_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return jsonify({"depth_npy_b64": depth_npy_b64})


# Run in the background, same convention as Step 3's PolicyServerConfig/serve() cell.
app.run(host="0.0.0.0", port=3000)
```

**Target notebook cell-0 replacement source (verbatim, from `policy_server_launch.md` lines 1 through the `---` divider that precedes `## Step 1` -- the .md is already correct here, the notebook is stale):**
```
# Colab PolicyServer + Tunnel Launch Instructions

Phase 11, Plan 11-03, Task 3. Documents how to stand up the Colab-hosted
`lerobot.async_inference.PolicyServer` and connect this project's local
`RobotClient` to it over a tunnel, before Plan 11-04 writes any code that
assumes this bridge exists.

**Selected checkpoint (Task 2's `checkpoint:decision`):** `victorvanhalst/smolvla_so101_cube`
— chosen for its exact task-instruction match to D-02's red-cube task. See
`checkpoint_candidates.md` for the fetched `config.json` data and
`11-03-SUMMARY.md` for the full decision rationale.

**pyngrok legitimacy gate:** this file's step 2 installs `pyngrok`, flagged
`SUS` in `11-RESEARCH.md`'s Package Legitimacy Audit. Do **not** run that
install cell until a human has completed the verification in this plan's
Task 3 `checkpoint:human-verify` (visit pypi.org/project/pyngrok, confirm
the `ngrok`/`alexdlaird` maintainer namespace and that `8.1.2` is a real
published version).

---

## Camera mapping (Task 2's split-stereo resolution)

`victorvanhalst/smolvla_so101_cube`'s `config.json` expects 3 named camera
inputs (`camera1`, `camera2`, `camera3` — none marked `empty_camera_*`).
This project's real rig has only 2 physical cameras:

- IMX335 wrist camera, cv2 index 0, native 1920x1080
- AR0144 stereo camera, cv2 index 1, captured at 1600x600 (800x600 per
  half) — **one physical side-by-side stereo frame, not two independent
  feeds** (confirmed in `diagnostics/UAT/function/basic/UAT.md`, Step 4).
  1600x600 was chosen over the previously-implemented 1280x360 for
  meaningfully higher per-eye resolution while remaining confirmed stable;
  the still-broken 2560x720 mode remains frozen even after exhausting the
  last known macOS system-level fix — see `stereo_camera.py`'s module
  docstring for the full diagnosis.

Resolution: split the AR0144's single 1600x600 frame into left/right
halves, giving **3 genuinely distinct real camera views** (not a dummy or
duplicated 3rd slot):

| Checkpoint camera key | Real source                 | Crop                                         |
| --------------------- | --------------------------- | -------------------------------------------- |
| `camera1`           | IMX335 wrist                | as-is, full 1920x1080 frame                  |
| `camera2`           | AR0144 stereo — LEFT half  | columns`[0:800]` of the 1600x600 frame    |
| `camera3`           | AR0144 stereo — RIGHT half | columns`[800:1600]` of the 1600x600 frame |

This is a better structural fit than an empty/dummy 3rd slot: all 3 of the
checkpoint's expected inputs receive real image data during inference,
none is a zero/black placeholder. (Community precedent: other SmolVLA
fine-tunes such as `bklassen3434/smolvla_pick_pen_v2_frozen` and
`Yilin1001/smolvla-pen-v2-030000` pair a 2-real-camera rig with an
inherited 3rd config slot by leaving it a dummy — this project instead has
a 3rd genuinely distinct real view available via the stereo split, which
is a strictly better fit than a dummy slot would be.)

`lerobot.async_inference.robot_client`'s `--robot.cameras` flag takes a
dict of named camera configs, each independently opened by `cv2` — it has
no built-in "crop a region of a wider frame" option. Two ways to produce
`camera2`/`camera3` as independent feeds from the one AR0144 device:

1. **Preferred for this bridge:** a thin wrapper camera process/script that
   opens cv2 index 1 once, splits each frame into left/right halves, and
   republishes them as two separate virtual camera endpoints the
   `RobotClient`'s camera config can point at (e.g. two loopback
   `cv2.VideoCapture`-compatible sources, or if `lerobot`'s camera config
   supports a custom `type` plugin, register a "split-half" camera type
   that does the crop inline). This avoids opening the same physical
   device twice (most webcam drivers do not support concurrent opens of
   one index).
2. **Fallback if (1) isn't scriptable in time:** open cv2 index 1 once in
   the `RobotClient`'s local process, capture the full 1600x600 frame per
   step, crop it into `camera2`/`camera3` numpy arrays in the observation
   dict before it's sent to the `PolicyServer` — this requires a small
   patch/subclass of the stock `robot_client.py` observation-building step
   rather than pure CLI flags, since the stock CLI only supports
   independent `cv2` device opens per named camera.

Plan 11-04 (which writes `robot_client.py`-adjacent bridge code) is
responsible for implementing whichever of these two options it lands on;
this file only fixes the *mapping* (which checkpoint key gets which real
data), not the implementation mechanism.

---
```

**Extra port-8080 occurrence found during read-first verification (not named in the original task brief, but the identical bug):** `policy_server_launch.md`'s own Summary table has a `Tunnel type` row reading `TCP (`ngrok.connect(8080, "tcp")`) — not HTTP` -- this is wrong for the same reason as the Step 9 comment (Step 3/4 use 5173 everywhere in actual code). The notebook's matching Summary cell (id `55c5a929`) already correctly says `5173` and needs no change. Fix the `.md`'s row to `5173` too, in Task 2.
</context>

<tasks>

<task type="auto">
  <name>Task 1: Wire real Fast-FoundationStereo inference into the Step 8 Flask cell (both files)</name>
  <files>control/vla_bridge/policy_server_launch.md, control/vla_bridge/policy_server.ipynb</files>
  <action>
In `control/vla_bridge/policy_server_launch.md`: immediately after the existing Step 7 git-clone code fence, insert a new python code fence containing exactly the "Target Step 7 addition" block from this plan's context section. Then replace the very next paragraph ("Then follow that repo's own README to install its dependencies and download its pretrained checkpoint...") with the "Target Step 7 follow-up paragraph" block from context, verbatim. Then replace the entire existing Step 8 code fence (the one defining the Flask app and the `run_fastfs_inference` stub) with the "Target Step 8 Flask cell" block from context, verbatim -- this turns the undefined stub call into a real function that writes temp PNG/intrinsic files, invokes FastFS's documented `scripts/run_demo.py` CLI as a subprocess with the exact documented flags, and loads the resulting depth map via the honestly-marked `DEPTH_OUTPUT_GLOB`.

In `control/vla_bridge/policy_server.ipynb`: write and run a small Python script (using `json.load`/mutate/`json.dump`, per this plan's context section's formatting notes on indent/trailing-newline/source-list conventions) that, querying cells by id (never by a hardcoded index): (1) locates the cell with id `6e04ff98` and inserts a new code cell immediately after it in the `cells` list, with `source` built from the same "Target Step 7 addition" text and a freshly generated unique id; (2) replaces the `source` of the cell with id `2e55f487` with the "Target Step 7 follow-up paragraph" text; (3) replaces the `source` of the cell with id `8473a692` with the "Target Step 8 Flask cell" text. Write the file back exactly as described in context (indent=2, ensure_ascii=False, single trailing newline, per-line `source` convention).

Do not modify `control/vla_bridge/depth_camera.py` -- its request/response field names (`left_png_b64`/`right_png_b64`/`intrinsics_flat`/`baseline_m` request, `depth_npy_b64` response) are unchanged by this task; confirm this remains true by inspecting it once, not by editing it.
  </action>
  <verify>
    <automated>cd /Users/hp/Desktop/Work/Repositories/SoARM-Research && grep -q "run_demo.py" control/vla_bridge/policy_server_launch.md && grep -q "DEPTH_OUTPUT_GLOB" control/vla_bridge/policy_server_launch.md && grep -q "torchvision==0.21.0" control/vla_bridge/policy_server_launch.md && python3 -c "
import json
nb = json.load(open('control/vla_bridge/policy_server.ipynb'))
full = chr(10).join(''.join(c['source']) for c in nb['cells'])
assert 'run_demo.py' in full, 'run_demo.py CLI call missing from notebook'
assert 'DEPTH_OUTPUT_GLOB' in full, 'DEPTH_OUTPUT_GLOB marker missing from notebook'
assert 'torchvision==0.21.0' in full, 'pip install cell missing from notebook'
assert 'run_fastfs_inference(left_img, right_img, intrinsics_flat, baseline_m)' in full
print('OK')
"</automated>
  </verify>
  <done>Both files' Step 8 cell defines a real `run_fastfs_inference()` that writes temp PNG/intrinsic files, invokes FastFS's documented `scripts/run_demo.py` CLI via subprocess with the exact documented flags, and loads the resulting depth map via the honestly-marked `DEPTH_OUTPUT_GLOB`; the `/depth` route body is otherwise unchanged; both files have a new pip-install cell/fence for FastFS's Option 2 environment right after Step 7's clone, plus an updated follow-up paragraph noting the manual checkpoint-download step; `policy_server.ipynb` still parses as valid JSON; `depth_camera.py` is untouched.</done>
</task>

<task type="auto">
  <name>Task 2: Fix the wrong port-8080 references and sync the notebook's stale camera-mapping + duplicate tunnel cell</name>
  <files>control/vla_bridge/policy_server_launch.md, control/vla_bridge/policy_server.ipynb</files>
  <action>
In `control/vla_bridge/policy_server_launch.md`: change "(8080)" to "(5173)" on the Step 9 depth-tunnel cell's comment line ("# A DIFFERENT local port than Step 3's gRPC server (8080), and a plain HTTP"). Also fix the Summary table's `Tunnel type` row, which independently has the identical wrong value (`TCP (`ngrok.connect(8080, "tcp")`) — not HTTP`) -- change its `8080` to `5173` too, matching Step 3/4's actual code and the notebook's own Summary row (which already correctly says 5173 and needs no change). Do not touch any other "8080"-shaped or "5173"-shaped text in this file.

In `control/vla_bridge/policy_server.ipynb`: load the file (as already modified by Task 1) and, querying cells by id (never by a hardcoded index, since Task 1's insertion shifted every index after `6e04ff98`): (1) in the cell with id `553ee9ec`, change "(8080)" to "(5173)" in its source, same single-word fix as the .md; (2) replace the entire `source` of the cell with id `5da4b39e` with the "Target notebook cell-0 replacement source" block from this plan's context section, using the same splitlines/trailing-newline convention described there; (3) delete the cell with id `wd94PYZ53t39` from the `cells` list entirely (the duplicate, uncommented tunnel-open cell) -- confirm the cell with id `fc7fd0ee` (the commented, kept tunnel-open cell) is still present afterward and unmodified. Write the file back using the same formatting conventions as Task 1 (indent=2, ensure_ascii=False, single trailing newline).
  </action>
  <verify>
    <automated>cd /Users/hp/Desktop/Work/Repositories/SoARM-Research && ! grep -q "8080" control/vla_bridge/policy_server_launch.md && grep -q "gRPC server (5173)" control/vla_bridge/policy_server_launch.md && ! grep -q "2560x720" control/vla_bridge/policy_server.ipynb && python3 -c "
import json
nb = json.load(open('control/vla_bridge/policy_server.ipynb'))
ids = [c['id'] for c in nb['cells']]
assert 'wd94PYZ53t39' not in ids, 'duplicate tunnel cell still present'
assert 'fc7fd0ee' in ids, 'kept tunnel cell missing'
assert len(nb['cells']) == 36, f'expected 36 cells after +1/-1, got {len(nb[\"cells\"])}'
by_id = {c['id']: ''.join(c['source']) for c in nb['cells']}
assert '8080' not in by_id['553ee9ec'], 'Step 9 comment still says 8080'
assert '5173' in by_id['553ee9ec'], 'Step 9 comment missing corrected 5173'
assert '1600x600' in by_id['5da4b39e'] and '800' in by_id['5da4b39e'], 'cell 0 camera mapping not synced'
assert '2560x720' not in by_id['5da4b39e'] and '1280x360' not in by_id['5da4b39e'], 'cell 0 still stale'
print('OK')
"</automated>
  </verify>
  <done>Both files correctly document Step 3's gRPC PolicyServer port as 5173 in every location that previously said 8080 (Step 9 comment in both files, plus the .md's own Summary table row); the notebook's cell 0 Camera mapping section matches the .md's current 1600x600/800-split wording with no stale 2560x720/1280-split text; the notebook has exactly one gRPC tunnel-open cell (id `fc7fd0ee`) with the duplicate (id `wd94PYZ53t39`) removed; the notebook still parses as valid JSON with 36 total cells.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|--------------|
| Colab-hosted Flask `/depth` endpoint -> `subprocess.run` invocation of FastFS's `scripts/run_demo.py` | The Flask handler now shells out to a subprocess running code from the already-cloned, human-legitimacy-checked NVlabs repo; this task adds the actual invocation, not a new untrusted code source. |
| New Colab pip installs (`torch==2.6.0`, `torchvision==0.21.0`, `xformers`, and the repo's own `requirements.txt`) -> Colab runtime | Package-manager installs added by this task, run inside the same already-gated Step 7 legitimacy-check block. |
| Local temp files (`left.png`/`right.png`/`K.txt`) -> subprocess `--out_dir` glob | `run_fastfs_inference()` writes/reads through Colab's ephemeral local filesystem inside a `tempfile.TemporaryDirectory()`; no data crosses a network boundary beyond the existing Flask request/response. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-quick260927ndz-01 | Tampering (documentation drift — port/camera-mapping mismatch) | `policy_server_launch.md` Step 9 + Summary table, `policy_server.ipynb` cell 0 + Step 9 cell | low | mitigate | Task 2 corrects every remaining "8080"/stale-resolution reference so a human following either file cannot misconfigure the actual tunnel port or camera crop. |
| T-quick260927ndz-02 | Tampering (supply chain) — new `pip install torch==2.6.0 torchvision==0.21.0 xformers` + repo `requirements.txt` | Colab runtime | medium | accept | `torch`/`torchvision` are already core, long-standing dependencies of this project (per CLAUDE.md stack) at the exact versions the vetted NVlabs README itself specifies; `xformers` is a well-known Meta/Facebook Research package; `requirements.txt` belongs to the repo already gated by Step 7's existing human legitimacy checkpoint, which explicitly instructs skimming `requirements.txt`/`setup.py` before installing -- no new checkpoint needed. |
| T-quick260927ndz-03 | Integrity (silently loading the wrong depth-output file) | `run_fastfs_inference()`'s `DEPTH_OUTPUT_GLOB` file pick | medium | mitigate | Raises `RuntimeError` (fails loud) if no `.npy` match is found, rather than silently returning garbage; the single `DEPTH_OUTPUT_GLOB` constant with an explicit TODO comment is the one, clearly-labeled spot a human corrects after confirming the real filename on a live Colab run, instead of the code guessing silently. |
| T-quick260927ndz-04 | Elevation of privilege (subprocess execution) | `subprocess.run([...], cwd=FASTFS_DIR, check=True)` | low | accept | Invokes only the already-cloned, human-legitimacy-checked repo's own documented entry point with a fixed, hardcoded argument list (no user-controlled string is passed as a shell command or format string); `check=True` fails loud on a non-zero exit rather than silently continuing. |

No new package-manager install in this task requires a separate blocking legitimacy checkpoint beyond the existing Step 7 gate -- see T-quick260927ndz-02's rationale.
</threat_model>

<verification>
- `python3 -c "import json; json.load(open('control/vla_bridge/policy_server.ipynb'))"` succeeds (valid JSON) after both tasks.
- `git diff --stat control/vla_bridge/depth_camera.py` shows no changes.
- `grep -c "8080" control/vla_bridge/policy_server_launch.md` is 0.
- `grep -c "2560x720" control/vla_bridge/policy_server.ipynb` is 0.
- Notebook cell count is 36 (unchanged net: +1 pip-install cell from Task 1, -1 duplicate tunnel cell from Task 2).
</verification>

<success_criteria>
- The Colab Flask `/depth` handler in both files calls a real, documented FastFS `run_demo.py` subprocess instead of an undefined stub, with the one genuinely undocumented detail (output filename) honestly marked via `DEPTH_OUTPUT_GLOB`.
- Both files correctly document Step 3's gRPC PolicyServer port as 5173 everywhere (Step 9 comment and the .md's Summary table row), with no remaining "8080" reference.
- The notebook's Camera mapping cell and tunnel-cell count match the .md and the real 1600x600/800-split hardware -- no stale 2560x720/1280-split text, no duplicate tunnel-open cell.
- `depth_camera.py` is unmodified and its contract still matches both files.
- `policy_server.ipynb` remains valid, parseable JSON with formatting/metadata unchanged outside the intended edits.
</success_criteria>

<output>
Create `.planning/quick/260927-ndz-fix-phase-12-uat-test-5-gaps-in-the-cola/260927-ndz-SUMMARY.md` when done
</output>
