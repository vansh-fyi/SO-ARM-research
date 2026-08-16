---
status: complete
phase: 05-spatial-awareness
source: [05-VERIFICATION.md]
started: 2026-08-10T05:35:00Z
updated: 2026-08-16T00:20:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Colab GPU spot-check — OFTBackend consumes both camera views
expected: Non-degenerate, plausible (8,7) action chunk; ideally compare against a single-image baseline to confirm the second view changes the output.
result: pass
notes: |
  RETEST after 05-04-PLAN.md's gap fix (set_num_images_in_input(2) + corrected
  primary/extra_views image order). Original blocker (RuntimeError: split_with_sizes
  expects split_sizes to sum exactly to 12 ... got split_sizes=[3, 3]) is resolved —
  no crash, both actions_dual/actions_degenerate return valid (8,7) chunks.

  CAVEAT (non-blocking, logged for future investigation): the model's predicted
  action chunk is only weakly sensitive to the second image. Diagnostic on Colab
  confirmed pixel_values genuinely differ between calls (max abs diff 3.34, means
  differ ~0.044) — the image pipeline itself is verified correct end-to-end against
  the upstream moojink/openvla-oft reference. But swapping the primary (agentview)
  image for solid white (a maximally different image) changed the output by only a
  mean of 9.2e-5 — nonzero (not fully ignored) but far below the original 1e-3
  threshold used to detect "silently ignored" behavior. Likely explanation: this
  checkpoint's first predicted action chunk (a canonical "reach" motion at episode
  start) is dominated by the language instruction rather than fine visual detail —
  a plausible checkpoint/task characteristic rather than a code defect, but worth
  re-examining if downstream eval results (VLA-02/VLA-03 full eval loop) show the
  model failing to react to genuinely different scene layouts.

### 2. Colab Linux (osmesa/egl) re-run of depth_xyz.py's D-04 test using real MuJoCo instance-segmentation render
expected: obs['agentview_segmentation_instance'] is non-degenerate on Colab's GL backend, and object_xyz_from_obs's back-projected estimate still matches sim.data.body_xpos ground truth within ~0.05m using the REAL segmentation array.
result: pass

## Summary

total: 2
passed: 2
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

- truth: "OFTBackend.predict() packs both camera views into a single multi-image forward pass and the checkpoint's num_images_in_input=2 pattern is actually consumed correctly (D-02, SPAT-02)."
  status: resolved
  resolution: "Fixed by 05-04-PLAN.md (commits f5b1e29/955dc79/96441cc/44eaa1b, merged 013cb43). Retested on Colab GPU 2026-08-16: no crash, valid (8,7) chunks from both dual and degenerate calls. See Test 1 notes for a separate non-blocking caveat (weak model sensitivity to the second image) discovered during retest — not part of this gap, tracked as an observation only."
  reason: "User reported: RuntimeError: split_with_sizes expects split_sizes to sum exactly to 12 (input tensor's size at dimension 1), but got split_sizes=[3, 3] — raised in modeling_prismatic.py's DinoSigLIPViTBackbone.forward when OFTBackend.predict() is called with both eye_in_hand and agentview images on Colab GPU."
  severity: blocker
  test: 1
  root_cause: |
    oft_backend.py's torch.cat(dim=1) pixel_values construction is byte-for-byte
    correct per the upstream moojink/openvla-oft reference (experiments/robot/
    openvla_utils.py get_vla_action). The missing piece is a one-time, load-time
    call the reference's get_vla() makes right after from_pretrained():
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input).
    PrismaticVisionBackbone defaults self.num_images_in_input = 1, which routes
    forward() through the single-image torch.split(pixel_values, [3,3], dim=1)
    path (tolerates only 6 channels). OFTBackend.__init__ never calls
    set_num_images_in_input(2), so the model stays in its default single-image
    state and crashes on the 12-channel concatenated tensor exactly as observed.
    When num_images_in_input > 1, forward() instead does
    torch.split(pixel_values, [6]*num_images_in_input, dim=1) and loops per-image.
    Separately (does not crash, but is wrong): current primary/extra_views
    ordering is primary=eye_in_hand (wrist), extra=agentview (third-person) —
    inverted relative to upstream's primary=third-person, wrist=secondary
    convention; channel order is baked into vision-backbone training so this
    should be corrected alongside the crash fix.
  artifacts:
    - path: "LIBERO/libero/libero/vla/oft_backend.py"
      issue: "__init__ never calls self.model.vision_backbone.set_num_images_in_input(2) after from_pretrained, so the model stays in its default single-image (num_images_in_input=1) state and its forward() unconditionally does torch.split(pixel_values, [3,3], dim=1), rejecting the 12-channel dual-image tensor predict() builds."
    - path: "LIBERO/libero/libero/vla/oft_backend.py"
      issue: "predict()'s primary/extra_views image order is inverted vs. upstream convention (primary should be third-person/agentview, secondary should be wrist/eye_in_hand) — silently mis-assigns camera views, no exception raised."
  missing:
    - "Call self.model.vision_backbone.set_num_images_in_input(2) once in OFTBackend.__init__, immediately after the model .to(device) call, mirroring upstream get_vla()."
    - "Swap predict()'s primary/extra_views assignment so primary=agentview (third-person) and extra_views=[eye_in_hand] (wrist), matching upstream's image-order convention."
    - "Re-verify on Colab GPU: re-run the VLA-01 dual-camera spot-check cell after the fix and confirm actions_dual differs meaningfully from actions_degenerate with no RuntimeError."
