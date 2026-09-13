# Functional UAT — Real Depth Camera (AR0144 Stereo)

Verifies the physical Waveshare AR0144 stereo module (overhead camera) can actually
produce usable **metric depth** on real hardware — not the simulated MuJoCo depth
already shipped in Phase 7 (`.planning/phases/07-camera-depth-perception/`), which is
a separate, already-complete, unrelated pipeline. This UAT is about the real-world
sim-to-real / physical-data-collection track: can this camera measure real object
dimensions well enough to be useful later (demo collection, safety/spatial
reasoning, etc.).

**Status: IN PROGRESS — not signed off.** Stereo calibration is done and good.
Object-dimension measurement is not yet reliable — lighting is the next thing to
fix before re-testing. See "Resume here" below.

**Prerequisites:**
- [`UAT/function/basic/UAT.md`](../basic/UAT.md) — camera recognized by control
  software (Step 4 there already confirms the AR0144 opens as a 2560x720 side-by-side
  stereo pair at cv2 index 1, alongside the wrist IMX335 at index 0). This UAT goes
  deeper on that same camera specifically for depth quality.

**Hardware note (from this session):** the AR0144 module has **no autofocus and no
software-controllable exposure** — confirmed empirically via
`cv2.VideoCapture.get()` on `CAP_PROP_AUTOFOCUS` / `CAP_PROP_FOCUS` /
`CAP_PROP_AUTO_EXPOSURE` / `CAP_PROP_EXPOSURE`, all of which return `-1.0` (not
supported by the driver). Two consequences that matter throughout this doc:
1. Lighting must be adjusted **physically** (room/desk lights) — there is no camera
   setting to fix over/under-exposure.
2. Moving/re-aiming the whole camera module does **not** invalidate calibration
   (fixed focus = fixed intrinsics; the two lenses' relative geometry doesn't change
   just because the whole rigid unit points somewhere else) — verified directly in
   Step 3.

**macOS quirk:** the camera's `avfoundation`/`cv2` device index is **not stable**
across reconnects — it has flipped between index 0 and 1 (with the built-in FaceTime
camera taking the other slot) multiple times this session. Always confirm with
`ffmpeg -f avfoundation -list_devices true -i ""` (look for `CCB Camera`) before
assuming an index.

---

## Step 1 — Confirm the camera is genuinely stereo and produces real disparity

Grabbed a still frame (`ffmpeg -f avfoundation -video_size 1280x480 -i "<idx>"
-vframes 1 out.png`), split it down the middle (left/right lens, same convention as
`diagnostics/camera_test.py`), and ran real OpenCV `StereoSGBM` block matching — not
just visual inspection — to prove actual stereo hardware, not two independent
cameras glued together.

| Check | Result | Date | Notes |
|---|---|---|---|
| Frame shape is a genuine side-by-side pair | ✅ PASS | 2026-09-12 | 1280x480 combined = two real 640x480 lenses. Visible horizontal disparity shift between halves confirmed by eye before running any matching. |
| Disparity computation produces coherent (non-random) output | ✅ PASS | 2026-09-12 | 55% of pixels got a valid disparity estimate; spatially coherent with scene geometry (closest object = warmest/highest disparity color, background = coolest) — confirmed via colorized disparity map, not just a pixel-count metric. |

---

## Step 2 — Stereo calibration (intrinsics + extrinsics)

**Target used:** `diagnostics/checkerboard_9x6_20mm.png` (9x6 internal corners,
designed for 20mm squares at 300 DPI, printed at "actual size"). **Measured actual
printed square size: 19mm** (printer scaling drift from the 20mm design) — used
`--square-mm 19` for the real calibration run, not the designed value. **Always
measure the printed target with a ruler before trusting a calibration** — this
project's own printer introduced a 5% scale error that would have silently
corrupted every metric depth number downstream if unmeasured.

**Tools built this session** (both require **`diagnostics/.venv`**, NOT the
`libero` conda env used elsewhere in this repo — these are plain OpenCV/numpy,
no MuJoCo dependency):
- `diagnostics/stereo_calibration_capture.py` — live dual-lens preview with
  real-time chessboard-corner overlay (green when both lenses detect it), SPACE to
  save a pair, auto-detects the camera index.
- `diagnostics/stereo_calibrate.py` — loads captured pairs, runs per-lens
  `calibrateCamera` + `stereoCalibrate` + `stereoRectify`, and includes an
  **automatic per-pair outlier-rejection pass**: computes per-image reprojection
  error, drops any pair exceeding 2x the median (e.g. a bent/non-flat board capture)
  before the final calibration. Also renders a sanity disparity map from the first
  pair as a visual gut-check.

**Result saved to** `diagnostics/stereo_calibration.npz` + `.json` (both committed —
real hardware config worth keeping, unlike the gitignored raw capture images in
`diagnostics/outputs/`).

| Attempt | Pairs captured | Pairs used (after outlier rejection) | Stereo reprojection RMS | Baseline | Result | Date | Notes |
|---|---|---|---|---|---|---|---|
| 1 (discarded) | 22 | 21 (1 dropped) | 0.46px (looked fine) | **523.88mm** | ❌ REJECTED | 2026-09-12 | Reprojection error alone looked good, but the baseline is physically absurd (a real module this size can't have a 52cm baseline) and the sanity disparity check gave nonsense negative-million-mm depths. Low reprojection error does **not** guarantee a physically correct calibration — always sanity-check baseline magnitude and a real depth number too. Root cause not fully diagnosed (suspected left/right correspondence issue); discarded rather than debugged since a clean recapture was faster. |
| 2 (accepted) | 30 | 29 (1 dropped, 3.43px vs 0.51px threshold — a bent-paper capture, confirmed visually) | **0.25px** | **57.79mm** | ✅ PASS | 2026-09-12 | Baseline is physically plausible for a small stereo module. Rectified block-matching disparity (58.5% valid, sane range) and a visual epipolar-line check (corresponding features land on the same row in both rectified images) both confirm the rectification is genuinely correct, not just numerically low-error. |

**Derived depth precision** (from this calibration: baseline 57.79mm, focal length
837.7px — formula `dZ = Z² / (f·B) · d_disparity`):

| Working distance | Precision (integer-px disparity) | Precision (sub-pixel, ~0.25px) |
|---|---|---|
| 300mm | ±1.86mm | ±0.46mm |
| 500mm | ±5.16mm | ±1.29mm |
| 700mm | ±10.12mm | ±2.53mm |
| 1000mm | ±20.66mm | ±5.16mm |

SO-ARM101's reach (~45cm) keeps real work naturally in the sub-2mm-precision range
— good news for this camera being useful for real measurement, *if* the surface/
lighting problems below are solved.

**Recalibration policy confirmed:** re-aiming the camera (angle change) does **not**
require recalibration, since intrinsics/extrinsics are properties of the rigid
module itself, not its orientation in the world. Verified directly: after a large
angle change (Step 4), computing disparity on the newly-rectified pair still worked
correctly (block matching is extremely sensitive to bad rectification, so this is a
strong test) — confirmed the *existing* calibration file still applies. Only
recalibrate if the housing itself gets physically flexed/bent, or focus/zoom
changes (not applicable — no autofocus).

**Pitfall found:** don't sanity-check rectification with ORB/feature-point matching
on a repetitive grid-pattern scene — it gives false alarms. A first attempt using
ORB matches + epipolar row-offset gave a scary ~112px "misalignment," but this was
the matcher confidently pairing up the *wrong* (visually identical) grid
intersections between lenses, not a real problem — repetitive patterns are a
classic failure case for sparse feature matching. The disparity-based and visual
line-crossing checks (which is what block-matching stereo actually depends on) are
the trustworthy tests.

---

## Step 3 — Workspace surface texture

**Problem found:** the original plain white/cream cloth surface (visible in early
capture pairs) gave almost no usable depth on its own — passive stereo needs visual
texture to correlate between lenses; a flat, featureless surface gives it nothing to
match. This matters for real tasks because things like "how tall is this object"
need a reliable **table-plane** depth reading right next to the object, not just
depth on the object itself.

**Fix iterated through several stages** (dot-grid notebook paper taped flat, then
lighting correction, then hand-drawn pencil lines added):

| Stage | Surface | Lighting | Valid depth over surface | Date | Notes |
|---|---|---|---|---|---|
| 1 | Plain white/cream cloth | normal | ~0% (not measured directly, visually obvious failure) | 2026-09-12 | Motivated the whole texture investigation. |
| 2 | Dot-grid paper (faint printed dots) | normal | 37.9% | 2026-09-13 | Real improvement, but dots are low-contrast/fine relative to camera resolution at working distance — visible banding in the disparity map shows the grid lines *are* contributing, just not enough on their own. |
| 3 | Dot-grid paper | **overexposed** (room light increased too much) | 18.0% (worse) | 2026-09-13 | 87.5% of the paper region was fully clipped to white (pixel value ≥250) — overexposure destroys contrast exactly like underexposure does, just from the other direction. No auto-exposure to compensate (Step 0 hardware note), so this is purely a physical-lighting mistake. |
| 4 | Dot-grid paper + hand-drawn pencil lines (**half the mat only**) | corrected (no clipping) | 39.3% overall, but **18.9% on dots-only half vs 59.6% on dots+pencil half** | 2026-09-13 | Direct, controlled proof that bolder linear marks help far more than the faint printed dots alone — visible as a stark left/right split in the disparity map. |
| 5 | Dot-grid paper + pencil lines (**full coverage**) | corrected | **64.3%** | 2026-09-13 | Final state. Good, usable background depth across the whole workspace. |

**Takeaway for future surfaces:** bold, evenly-spread linear texture (a fine grid,
hand-drawn or printed) works well. Avoid: plain colors (any color, not just white —
the fix is about *texture*, not hue), and avoid picking a target exposure by feel —
check for clipping (`(pixels>=250).mean()`) explicitly, since the AR0144 has no
auto-exposure to save you from a bad guess.

---

## Step 4 — Real object dimension measurement

With calibration (Step 2) and a textured surface (Step 3) both working, tested
whether real objects can actually be measured — this is the part that's still
**not reliable** and is where to resume.

### Attempt 1 — Small shiny metal part (a servo/disc-shaped part)

| Result | Date | Notes |
|---|---|---|
| ❌ FAIL | 2026-09-13 | Essentially zero valid depth *on the object itself* (a handful of pixels out of thousands), despite the surrounding mat working fine. Root cause: specular reflection off the shiny metal surface — a highlight looks different (or is in a different place) between the two lenses depending on viewing angle, breaking correspondence entirely. This is a fundamental limitation of passive stereo on shiny/reflective objects, not a setup mistake. |

### Attempt 2 — Black plastic TV remote (matte, textured with buttons)

| Sub-attempt | Camera angle | Object surface brightness | Specular glint (% pixels >200) | Height reading | Length x Width | Date | Notes |
|---|---|---|---|---|---|---|---|
| 1 | original | mean 139, std 80.9 | **26.5%** | **0.0mm (degenerate/fake)** | **164.9mm x 45.0mm** — plausible! | 2026-09-13 | XY dimensions came out very believable (real remotes are roughly this size), proving the calibration + mat combo *does* work for reasonable objects. But height was clearly broken: 75% of "valid" object-region pixels returned the *exact identical* Z value — a smoothness-penalty artifact (the matcher filling in a fake flat plateau across a corrupted low-confidence region), not a real measurement. Root cause: the specular glint (>1 in 4 pixels) breaks real matching. |
| 2 | changed a lot (to kill the glint) | mean 21.6, std 39.1 | **0.0%** (fixed) | still degenerate (min=p25=median=p75=451.9, only a max outlier differs) | not remeasured | 2026-09-13 | Confirmed the angle-change hypothesis: glint is genuinely gone. But traded it for severe **underexposure** (object nearly black) — same degenerate-plateau failure mode, different root cause. Net result: still not usable. |

**Methodology pitfall hit and fixed along the way:** naive "most elongated contour"
object-picking accidentally grabbed the gripper's dark fingers (also thin +
elongated + dark) instead of the remote, and a later size-cutoff (`<500px`) excluded
the real remote by a handful of pixels and fell through to a false match near the
top of frame. Fixed by filtering on aspect ratio (2.5-5:1, matching a remote's real
proportions) **and** picking the largest-area match among candidates, then
confirming visually. **Always visually confirm an auto-detected bounding box before
trusting any numbers derived from it** — this cost real time twice in one session.

---

## Resume here next session

**Immediate next step (user's own words: "we can fix the lighting"):** find a
lighting setup at the *current* (post-angle-change, no-glint) camera position that
gives the remote enough diffuse light to have real contrast, without reintroducing
a specular hotspot. Concretely: check `(pixel >= 200).mean()` for glint and
`gray.mean()` for overall brightness on the object itself — target something like
mean brightness 100-180 with near-zero glint, instead of the two failure extremes
already found (mean 139/26.5% glint vs mean 21.6/0% glint).

**After that's fixed, re-run Step 4's remote test** and confirm: (1) glint stays
near 0%, (2) object Z distribution is no longer degenerate (should show real
variance across the object's surface, not repeated identical values), (3) a height
estimate comes back in a plausible range for a remote (~15-20mm).

**Then broaden testing** to 2-3 more everyday matte objects (a box, a fabric item,
etc.) before trusting this pipeline for anything real — one working remote
measurement isn't enough evidence yet.

**Deferred, not blocking:** whether to adopt a trained object-detector (SAM2 /
Grounding DINO) to replace the ad hoc contour-based object-picking used throughout
Step 4, instead of writing better heuristics each time. Decided *not* to use Meta's
linked `Detectron` tool specifically — confirmed via fetching the actual page that
it's the original 2018-era Caffe2/Python2 tool, not even Detectron2, and is not
worth adopting today. Revisit only if a concrete need shows up (this project's core
VLA pipeline doesn't need a separate object detector for the main control loop).

**Files to know about:**
- `diagnostics/checkerboard_9x6_20mm.png` — calibration target (measure before
  reprinting/reusing — see Step 2).
- `diagnostics/stereo_calibration_capture.py`, `diagnostics/stereo_calibrate.py` —
  reusable tools, run under `diagnostics/.venv`.
- `diagnostics/stereo_calibration.{npz,json}` — the current good calibration
  (baseline 57.79mm). No need to redo this unless the housing gets physically
  flexed.
- Raw captured calibration pairs live in `diagnostics/outputs/stereo_calib/`
  (gitignored, local only — matches this project's existing convention of not
  versioning real camera photos, see `progress-documentation/images/`).

---

## Sign-off

| All steps pass? | Date | Tested by |
|---|---|---|
| ⏳ NOT YET — Steps 1-3 pass, Step 4 (object measurement) open | 2026-09-13 | vanshux23 |
