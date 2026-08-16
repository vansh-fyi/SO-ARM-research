---
status: complete
phase: 05-spatial-awareness
source: [05-VERIFICATION.md]
started: 2026-08-10T05:35:00Z
updated: 2026-08-16T00:00:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Colab GPU spot-check — OFTBackend consumes both camera views
expected: Non-degenerate, plausible (8,7) action chunk; ideally compare against a single-image baseline to confirm the second view changes the output.
result: issue
reported: "RuntimeError: split_with_sizes expects split_sizes to sum exactly to 12 (input tensor's size at dimension 1), but got split_sizes=[3, 3]. Raised inside moojink/openvla-7b-oft-finetuned-libero-spatial's modeling_prismatic.py DinoSigLIPViTBackbone.forward when backend.predict() is called with both eye_in_hand and agentview views. oft_backend.py's torch.cat([...], dim=1) on two per-image processor outputs produces a 12-channel pixel_values tensor, but the checkpoint's vision backbone does torch.split(pixel_values, [3,3], dim=1) expecting exactly 6 channels (3ch DINO + 3ch SigLIP fused encoder for ONE image) — the checkpoint does not support channel-dim concatenation of two images' processor outputs the way oft_backend.py assumes."
severity: blocker

### 2. Colab Linux (osmesa/egl) re-run of depth_xyz.py's D-04 test using real MuJoCo instance-segmentation render
expected: obs['agentview_segmentation_instance'] is non-degenerate on Colab's GL backend, and object_xyz_from_obs's back-projected estimate still matches sim.data.body_xpos ground truth within ~0.05m using the REAL segmentation array.
result: pass

## Summary

total: 2
passed: 1
issues: 1
pending: 0
skipped: 0
blocked: 0

## Gaps

- truth: "OFTBackend.predict() packs both camera views into a single multi-image forward pass and the checkpoint's num_images_in_input=2 pattern is actually consumed correctly (D-02, SPAT-02)."
  status: failed
  reason: "User reported: RuntimeError: split_with_sizes expects split_sizes to sum exactly to 12 (input tensor's size at dimension 1), but got split_sizes=[3, 3] — raised in modeling_prismatic.py's DinoSigLIPViTBackbone.forward when OFTBackend.predict() is called with both eye_in_hand and agentview images on Colab GPU."
  severity: blocker
  test: 1
  artifacts: []
  missing: []
