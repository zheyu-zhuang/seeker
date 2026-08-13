# Checkpoint provenance, compatibility, and licensing

Checkpoints are distributed via `seeker setup` (see the top-level
[README](../../README.md#4-set-up-suite-dependencies-and-assets)), not committed to git.

## `seeker.mimicgen.pth`

**What it is**: the pretrained Seeker checkpoint (`Seeker` class in
[`seeker_model.py`](./seeker_model.py)), used both to resume/continue Seeker
pretraining and as the frozen `method=seeker` focus source for downstream
policy training.

**Training recipe**: `train_visual_focus_seeker`, all 6 canonical MimicGen
tasks (`coffee_preparation_d1`, `pick_place_d0`, `square_d2`,
`stack_three_d1`, `threading_d2`, `three_piece_assembly_d2`), 100 demos/task,
seed 0 — the defaults in
[`train_visual_focus_seeker.yaml`](../config/train_visual_focus_seeker.yaml).

**License**: MIT, same as the rest of Seeker's original code. See the
top-level [`LICENSE`](../../LICENSE). The published checkpoint contains only
`normalizer.*` and `view_branches.*` tensors — the frozen DINOv3 backbone
(`vit.*`) is stripped out via
[`strip_dinov3_backbone.py`](../scripts/strip_dinov3_backbone.py), since
`load_pretrained_weights` already loads the backbone separately from
`dinov3.vits16plus.pth` and excludes `vit.*` from its strict-compatibility
check. This keeps the file's contents unambiguously MIT (no Meta-licensed
weights inside) and cuts its size from ~146MB to ~31MB.

**Compatibility contract**: loading is `strict_weights` by default (see
`load_pretrained_weights` in [`seeker_model.py`](./seeker_model.py)) — the
checkpoint's tensor shapes depend on these values from
[`seeker_default_params.yaml`](../config/seeker_default_params.yaml) and
[`method/seeker.yaml`](../config/method/seeker.yaml):

| Key | Value baked into this checkpoint |
|---|---|
| `query_composer.disable_proprio` | `false` |
| `intent_refiner.disable_head_gating` | `false` |
| `intent_refiner.num_refinement_iters` | `3` |
| `query_composer.task_emb_dim` | `512` |
| `entmax_alpha` | `1.3` |
| `checkpoint_views` | `[agentview, eye_in_hand]` |
| DINOv3 backbone | frozen `dinov3_vits16plus` |

Changing any of the shape-affecting values above (`disable_proprio`,
`disable_head_gating`, `num_refinement_iters`, `task_emb_dim`) before loading
this checkpoint will fail with a `RuntimeError` from `load_pretrained_weights`
rather than silently mismatching.

## `rvt2_heatmap.mimicgen.pth`

**What it is**: the pretrained RVT2-heatmap baseline checkpoint
(`RVT2Heatmap` class in [`rvt2_heatmap.py`](./rvt2_heatmap.py)), used as the
focus source for `method=rvt2`.

Downloaded by `seeker setup-assets` to `.weights/rvt2_heatmap.mimicgen.pth`
along with the other checkpoints — only needed if you're running
`method=rvt2`; `seeker`/`mirroraug`/`oracle` don't use it.

**License**: MIT, same as `seeker.mimicgen.pth` above. Like that
checkpoint, the frozen DINOv3 backbone is stripped out (verified
byte-identical to `dinov3.vits16plus.pth` before stripping) via
[`strip_dinov3_backbone.py`](../scripts/strip_dinov3_backbone.py) — see
`RVT2Heatmap.__init__`, which tolerates a missing
`patch_backbone_state_dict` for a `dino`-type backbone and loads it from
`dinov3.vits16plus.pth` instead. Cuts the file from ~156MB to ~42MB.

## `dinov3.vits16plus.pth`

The frozen DINOv3 ViT-S+/16 backbone weights. License and redistribution
terms: [`seeker/model/dinov3_core/LICENSE.md`](./dinov3_core/LICENSE.md)
(Meta's DINOv3 License, reproduced verbatim).

## Checksums

`setup-assets` verifies all three checkpoints against these sha256 digests
after download (see `EXPECTED_SHA256` in
[`setup_assets.py`](../scripts/setup_assets.py)):

| File | sha256 |
|---|---|
| `seeker.mimicgen.pth` | `b0aa9d7272e8b93ddccc402959969eb52ae075e8595d0ddd478a1b39c5aacda1` |
| `dinov3.vits16plus.pth` | `4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea` |
| `rvt2_heatmap.mimicgen.pth` | `996ea845bcbb1fc4b5a9e8c66671d83acafcbb89113f74009bf81bc94696212e` |
