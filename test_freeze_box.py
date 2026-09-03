"""Local, model-free unit tests for the --freeze_box path of static_range_edit.py.

The two halves of a hard freeze are tested separately, because they have to
agree exactly or the box is only half held:

  * `frame_mapping.build_freeze_regen_mask` decides *where* the sampler is
    forbidden to regenerate, and
  * `latent_paste.pin_anchor_content` decides *what* is held there.

A mask that snapped outward while the content pin snapped inward would leave a
ring of cells frozen to the drifted source instead of the anchor -- visible as
a stuck copy of the drift, which is the failure this whole file guards.

Runs on CPU with no checkpoints. Run with: python test_freeze_box.py
"""

import numpy as np
import torch

from frame_mapping import (
    build_freeze_regen_mask,
    build_latent_regen_mask,
    build_regeneration_window,
    anchor_latent_frame,
    snap_box_outward,
)
from latent_paste import pin_anchor_content, target_latent_frames

STRIDE, PATCH = 16, 2
POOL = STRIDE * PATCH  # 32: one DiT token
H, W = 704, 1248  # initial3.mp4, both exact multiples of POOL
BOX = (330, 230, 630, 440)


def _fixture(box=BOX, a=24, b=120, video_len=121):
    """The real case: a box held across an edit that runs to the last frame."""
    window = build_regeneration_window(a, b, video_len)
    anchor = anchor_latent_frame(window)
    snapped = snap_box_outward(box, POOL, W, H)
    mask = build_freeze_regen_mask(window, [snapped], H, W,
                                   vae_spatial_stride=STRIDE, patch_spatial=PATCH)
    temporal = build_latent_regen_mask(window)
    latent_box = tuple(v // STRIDE for v in (snapped[0], snapped[1], snapped[2], snapped[3]))
    return window, anchor, snapped, mask, temporal, latent_box


def test_snap_rounds_outward_never_inward():
    assert snap_box_outward(BOX, POOL, W, H) == (320, 224, 640, 448)
    x1, y1, x2, y2 = snap_box_outward(BOX, POOL, W, H)
    assert x1 <= BOX[0] and y1 <= BOX[1] and x2 >= BOX[2] and y2 >= BOX[3]


def test_snap_clips_to_frame():
    assert snap_box_outward((-5, -5, W + 40, H + 40), POOL, W, H) == (0, 0, W, H)


def test_snap_is_idempotent_on_aligned_boxes():
    aligned = (320, 224, 640, 448)
    assert snap_box_outward(aligned, POOL, W, H) == aligned


def test_mask_freezes_the_box_and_nothing_else():
    _, _, _, mask, temporal, (lx1, ly1, lx2, ly2) = _fixture()
    assert mask.shape == (len(temporal), H // STRIDE, W // STRIDE)
    for i, regen in enumerate(temporal):
        if not regen:
            assert not mask[i].any(), f"frozen frame {i} should regenerate nothing"
            continue
        assert not mask[i, ly1:ly2, lx1:lx2].any(), f"frame {i}: box is not frozen"
        outside = mask[i].copy()
        outside[ly1:ly2, lx1:lx2] = True
        assert outside.all(), f"frame {i}: something outside the box was frozen too"


def test_mask_is_uniform_within_each_dit_token():
    """The per-token timestep trick reads the mask as `mask[:, ::2, ::2]`, so a
    2x2 latent block that mixed frozen and regenerate cells would be sampled at
    an arbitrary corner and the other three cells would get the wrong timestep."""
    _, _, _, mask, _, _ = _fixture()
    n = mask.shape[0]
    blocks = mask.reshape(n, H // POOL, PATCH, W // POOL, PATCH)
    all_on = blocks.all(axis=(2, 4))
    all_off = ~blocks.any(axis=(2, 4))
    assert (all_on | all_off).all()


def test_mask_never_widens_the_temporal_mask():
    _, _, _, mask, temporal, _ = _fixture()
    for i, regen in enumerate(temporal):
        if not regen:
            assert not mask[i].any()


def test_whole_frame_freeze_leaves_nothing_to_regenerate():
    window = build_regeneration_window(24, 120, 121)
    mask = build_freeze_regen_mask(window, [(0, 0, W, H)], H, W,
                                   vae_spatial_stride=STRIDE, patch_spatial=PATCH)
    assert not mask.any()


def test_pin_broadcasts_anchor_content_to_targets_only():
    _, anchor, _, mask, temporal, (lx1, ly1, lx2, ly2) = _fixture()
    z = torch.randn(4, len(temporal), H // STRIDE, W // STRIDE)
    targets = target_latent_frames(temporal, anchor)
    pinned = pin_anchor_content(z, [(lx1, ly1, lx2, ly2)], anchor, targets)

    for t in targets:
        assert torch.equal(pinned[:, t, ly1:ly2, lx1:lx2], z[:, anchor, ly1:ly2, lx1:lx2])
    assert torch.equal(pinned[:, anchor], z[:, anchor]), "anchor frame was modified"
    for i, regen in enumerate(temporal):
        if not regen:
            assert torch.equal(pinned[:, i], z[:, i]), f"frozen frame {i} was modified"

    restored = pinned.clone()
    restored[:, targets, ly1:ly2, lx1:lx2] = z[:, targets, ly1:ly2, lx1:lx2]
    assert torch.equal(restored, z), "pin touched cells outside the box"


def test_pin_is_a_noop_without_boxes_or_targets():
    z = torch.randn(4, 6, 8, 10)
    assert pin_anchor_content(z, [], 0, [1, 2]) is z
    assert pin_anchor_content(z, [(0, 0, 2, 2)], 0, []) is z


def test_freeze_holds_exactly_at_every_noise_level():
    """The whole point of --freeze_box over --static_box: the held content is
    independent of sigma, where `write_delta`'s correction is scaled by
    (1 - sigma) and so writes nothing at all near sigma = 1."""
    _, anchor, _, mask, temporal, (lx1, ly1, lx2, ly2) = _fixture()
    z = torch.randn(4, len(temporal), H // STRIDE, W // STRIDE)
    targets = target_latent_frames(temporal, anchor)
    pinned = pin_anchor_content(z, [(lx1, ly1, lx2, ly2)], anchor, targets)
    mask2 = torch.tensor(mask, dtype=z.dtype).unsqueeze(0).expand_as(z)

    for sigma in (1.0, 0.8333, 0.2083, 0.0):
        latent = torch.randn_like(z)  # whatever the sampler produced this step
        held = (1.0 - mask2) * pinned + mask2 * latent
        for t in targets:
            assert torch.equal(held[:, t, ly1:ly2, lx1:lx2], z[:, anchor, ly1:ly2, lx1:lx2]), sigma
        # and outside the box the sampler's output survives untouched
        assert torch.equal(held[:, targets, :, lx2:], latent[:, targets, :, lx2:]), sigma


def test_multiple_freeze_boxes_compose():
    window = build_regeneration_window(24, 120, 121)
    anchor = anchor_latent_frame(window)
    boxes = [snap_box_outward((330, 230, 630, 440), POOL, W, H),
             snap_box_outward((40, 60, 200, 260), POOL, W, H)]
    mask = build_freeze_regen_mask(window, boxes, H, W,
                                   vae_spatial_stride=STRIDE, patch_spatial=PATCH)
    temporal = build_latent_regen_mask(window)
    live = [i for i, r in enumerate(temporal) if r]
    for x1, y1, x2, y2 in boxes:
        sl = (slice(y1 // STRIDE, y2 // STRIDE), slice(x1 // STRIDE, x2 // STRIDE))
        for i in live:
            assert not mask[i][sl].any()

    z = torch.randn(4, len(temporal), H // STRIDE, W // STRIDE)
    targets = target_latent_frames(temporal, anchor)
    latent_boxes = [tuple(v // STRIDE for v in b) for b in boxes]
    pinned = pin_anchor_content(z, latent_boxes, anchor, targets)
    for lx1, ly1, lx2, ly2 in latent_boxes:
        for t in targets:
            assert torch.equal(pinned[:, t, ly1:ly2, lx1:lx2], z[:, anchor, ly1:ly2, lx1:lx2])


def test_mask_and_pin_agree_on_the_same_cells():
    """The load-bearing invariant: every cell the mask freezes is a cell the
    pin filled with anchor content. Any disagreement holds drifted source
    content instead, which looks like a freeze but reproduces the drift."""
    _, anchor, _, mask, temporal, latent_box = _fixture()
    lx1, ly1, lx2, ly2 = latent_box
    z = torch.randn(4, len(temporal), H // STRIDE, W // STRIDE)
    targets = target_latent_frames(temporal, anchor)
    pinned = pin_anchor_content(z, [latent_box], anchor, targets)

    for t in targets:
        frozen_cells = ~mask[t]
        pinned_cells = np.zeros_like(frozen_cells)
        pinned_cells[ly1:ly2, lx1:lx2] = True
        assert np.array_equal(frozen_cells, pinned_cells), f"frame {t}: mask and pin disagree"
        # and the held content really is the anchor's, not frame t's own
        held = pinned[:, t][:, frozen_cells]
        assert torch.equal(held, z[:, anchor][:, frozen_cells])


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} tests passed")
