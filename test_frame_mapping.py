"""Local, model-free unit tests for frame_mapping.py's index math.

Run with: python inpainting/test_frame_mapping.py
"""

import numpy as np

from frame_mapping import (
    anchor_latent_frame,
    build_feathered_box_weight,
    build_latent_regen_mask,
    build_regeneration_window,
    build_spatial_latent_regen_mask,
    latent_frame_to_pixel_range,
    num_latent_frames,
    pixel_box_to_latent_box,
)


def test_num_latent_frames():
    assert num_latent_frames(81) == 21
    assert num_latent_frames(1) == 1
    assert num_latent_frames(13) == 4


def test_latent_frame_to_pixel_range():
    assert latent_frame_to_pixel_range(0) == (0, 0)
    assert latent_frame_to_pixel_range(1) == (1, 4)
    assert latent_frame_to_pixel_range(2) == (5, 8)
    assert latent_frame_to_pixel_range(3) == (9, 12)


def test_build_regeneration_window_block_aligned():
    # (b - a) % 4 == 1 -> the naive [a-2, b+2] window already lands on a
    # block boundary, so this case worked even before the fix.
    window = build_regeneration_window(a=10, b=15, video_len=100)
    assert window.edit_start == 9
    assert window.edit_end == 16
    assert window.window_start == 8
    assert window.window_end == 20
    assert (window.num_pixel_frames - 1) % 4 == 0


def test_build_regeneration_window_rejects_insufficient_context():
    try:
        build_regeneration_window(a=1, b=5, video_len=100)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a-2 < 0")

    try:
        build_regeneration_window(a=10, b=95, video_len=100, context_blocks=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when there isn't a full trailing context block")


def test_build_latent_regen_mask():
    window = build_regeneration_window(a=10, b=15, video_len=100)
    mask = build_latent_regen_mask(window)
    # window is [8, 20] (13 frames -> 4 latent frames); latent ranges are
    # (0,0) (1,4) (5,8) (9,12); local edit range is [1, 8]
    assert mask == [False, True, True, False]


def test_mask_length_matches_num_latent_frames():
    window = build_regeneration_window(a=2, b=2, video_len=9)
    mask = build_latent_regen_mask(window)
    assert len(mask) == num_latent_frames(window.num_pixel_frames)


def _frozen_block_is_guaranteed_after_last_regen(a: int, b: int, video_len: int, context_blocks: int = 1) -> None:
    """Regression check for the block-alignment bug: whatever the alignment
    of (b - a), there must be a fully frozen, real latent block immediately
    after the last regenerated block, so the model always has genuine
    context to smoothly continue into -- not just whatever happened to
    survive the old fixed-width padding.

    Only applies when `b != video_len - 1` -- when the edit reaches the true
    end of the video there is no real trailing footage to freeze, so no
    frozen block is required or produced; see the dedicated at-video-end
    tests below instead.
    """
    window = build_regeneration_window(a, b, video_len, context_blocks=context_blocks)
    mask = build_latent_regen_mask(window)

    last_regen_idx = max(i for i, regen in enumerate(mask) if regen)
    trailing_frozen = mask[last_regen_idx + 1:]
    assert len(trailing_frozen) >= context_blocks, (
        f"expected >= {context_blocks} frozen block(s) after the last regen block, got {len(trailing_frozen)}")
    assert all(not regen for regen in trailing_frozen), "trailing context block(s) must be entirely frozen"
    assert window.window_end <= video_len - 1


def test_regression_previously_buggy_alignment():
    # a=10, b=14 -> (b - a) % 4 == 0, not 1: the case that exposed the bug,
    # since the naive [a-2, b+2] window used to end exactly at the last
    # regenerated latent block, with no frozen block beyond it at all.
    window = build_regeneration_window(a=10, b=14, video_len=100)
    assert window.window_start == 8
    assert window.window_end == 20  # old buggy code would have stopped at 16
    _frozen_block_is_guaranteed_after_last_regen(a=10, b=14, video_len=100)


def test_regression_all_alignments():
    for offset in range(8):  # covers every (b - a) % 4 value at least twice
        a, b = 20, 20 + offset
        _frozen_block_is_guaranteed_after_last_regen(a, b, video_len=200)


def test_context_blocks_extends_required_trailing_context():
    window1 = build_regeneration_window(a=10, b=14, video_len=200, context_blocks=1)
    window2 = build_regeneration_window(a=10, b=14, video_len=200, context_blocks=2)
    assert window2.window_end == window1.window_end + 4
    _frozen_block_is_guaranteed_after_last_regen(a=10, b=14, video_len=200, context_blocks=2)


def test_edit_starting_at_frame_zero():
    window = build_regeneration_window(a=0, b=5, video_len=100)
    assert window.window_start == 0
    assert window.edit_start == 0
    assert window.pad_end == 0  # right-side math is unaffected by window_start, so no padding needed
    mask = build_latent_regen_mask(window)
    assert mask[0] is True  # no frozen anchor before frame 0 -- it's part of the regen region


def test_edit_ending_at_last_frame():
    video_len = 100
    window = build_regeneration_window(a=10, b=video_len - 1, video_len=video_len)
    assert window.edit_end == video_len - 1
    assert window.window_end == video_len - 1  # never extends past the real video
    assert 0 <= window.pad_end < 4  # at most a sub-block alignment remainder, never a full frozen context block
    assert window.num_pixel_frames % 4 == 1  # VAE 4n+1 requirement still holds
    mask = build_latent_regen_mask(window)
    assert mask[-1] is True  # tail is regenerated, never a frozen fabricated anchor


def test_edit_reaching_video_end_no_padding_needed():
    # Exact numbers from the reported bug: editing through the true end of
    # a 121-frame video. Regression guard for the trailing-anchor bug --
    # previously this fabricated `pad_end` frozen frames that were literal
    # duplicates of the original (uncorrected) ending, biasing the edit
    # back toward the old motion even at noise_strength=1.0.
    window = build_regeneration_window(a=54, b=120, video_len=121)
    assert window.window_end == 120
    assert window.edit_end == 120
    assert window.pad_end == 0  # this case happens to land exactly on a block boundary
    mask = build_latent_regen_mask(window)
    assert mask[-1] is True  # no frozen block appended after the edit region
    assert window.num_pixel_frames % 4 == 1


def test_edit_reaching_video_end_padding_is_bounded_and_never_frozen():
    # a=10, b=99, video_len=100 is NOT block-aligned at the tail -> pad_end
    # is a small (< temporal_stride) alignment remainder, not zero. The key
    # invariant is that it's never a full frozen context block and never
    # marked frozen -- unlike the pre-fix behavior.
    window = build_regeneration_window(a=10, b=99, video_len=100)
    assert 0 <= window.pad_end < 4
    mask = build_latent_regen_mask(window)
    assert mask[-1] is True


def test_full_video_edit():
    video_len = 50
    window = build_regeneration_window(a=0, b=video_len - 1, video_len=video_len)
    assert window.window_start == 0
    assert window.edit_start == 0
    assert window.edit_end == video_len - 1
    assert window.num_pixel_frames % 4 == 1


def test_full_video_edit_needs_no_trailing_anchor():
    # a==0 and b==video_len-1 together: neither boundary needs an anchor.
    video_len = 50
    window = build_regeneration_window(a=0, b=video_len - 1, video_len=video_len)
    assert window.window_start == 0
    assert window.window_end == video_len - 1
    mask = build_latent_regen_mask(window)
    assert mask[0] is True  # no leading anchor either (already covered by test_edit_starting_at_frame_zero)
    assert mask[-1] is True  # no trailing anchor


def test_build_spatial_latent_regen_mask_never_widens_temporal():
    # window is [8, 20] (13 frames -> 4 latent frames); latent ranges are
    # (0,0) (1,4) (5,8) (9,12); temporal mask is [False, True, True, False]
    window = build_regeneration_window(a=10, b=15, video_len=100)
    pixel_mask = np.ones((window.num_pixel_frames, 64, 64), dtype=bool)  # fully permissive spatially
    mask = build_spatial_latent_regen_mask(window, pixel_mask)
    assert mask.shape == (4, 4, 4)  # 64 / vae_spatial_stride(16) == 4
    assert not mask[0].any()  # temporally frozen latent frame stays frozen regardless of spatial mask
    assert not mask[3].any()
    assert mask[1].all()  # temporally regen + spatially unrestricted -> fully regen
    assert mask[2].all()


def test_build_spatial_latent_regen_mask_narrows_within_regen_region():
    window = build_regeneration_window(a=10, b=15, video_len=100)
    pixel_mask = np.zeros((window.num_pixel_frames, 64, 64), dtype=bool)
    pixel_mask[:, :32, :32] = True  # only the top-left quadrant is editable
    mask = build_spatial_latent_regen_mask(window, pixel_mask)
    for t in (1, 2):  # temporally regen latent frames
        assert mask[t][:2, :2].all()
        assert not mask[t][:2, 2:].any()
        assert not mask[t][2:, :].any()
    for t in (0, 3):  # temporally frozen latent frames -- unaffected by spatial mask
        assert not mask[t].any()


def test_build_spatial_latent_regen_mask_constant_within_patch_blocks():
    window = build_regeneration_window(a=10, b=15, video_len=100)
    pixel_mask = np.zeros((window.num_pixel_frames, 64, 64), dtype=bool)
    pixel_mask[:, 10:54, 10:54] = True  # arbitrary region, not aligned to any pooling boundary
    mask = build_spatial_latent_regen_mask(window, pixel_mask)
    for t in range(mask.shape[0]):
        for i in range(0, mask.shape[1], 2):
            for j in range(0, mask.shape[2], 2):
                block = mask[t, i:i + 2, j:j + 2]
                assert block.all() or not block.any(), (
                    "mask must be constant within every 2x2 latent block, or the DiT's "
                    "stride-2 per-token timestep subsample picks an arbitrary corner")


def test_build_spatial_latent_regen_mask_rejects_frame_count_mismatch():
    window = build_regeneration_window(a=10, b=15, video_len=100)
    pixel_mask = np.ones((window.num_pixel_frames + 1, 64, 64), dtype=bool)
    try:
        build_spatial_latent_regen_mask(window, pixel_mask)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for pixel_mask frame count mismatch")


def test_build_spatial_latent_regen_mask_rejects_non_multiple_spatial_size():
    window = build_regeneration_window(a=10, b=15, video_len=100)
    pixel_mask = np.ones((window.num_pixel_frames, 50, 50), dtype=bool)  # not a multiple of 32
    try:
        build_spatial_latent_regen_mask(window, pixel_mask)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for spatial size not a multiple of vae_spatial_stride * patch_spatial")


def test_pixel_box_to_latent_box_divides_by_vae_stride_not_patch_pool():
    # 1280x704 -> 80x44 latent grid. A direct latent write addresses individual
    # latent cells (stride 16), unlike build_spatial_latent_regen_mask's 32-px
    # pooling, which exists only for the DiT's stride-2 timestep subsample.
    assert pixel_box_to_latent_box((160, 96, 320, 192), latent_h=44, latent_w=80) == (10, 6, 20, 12)


def test_pixel_box_to_latent_box_rounds_outward():
    # Any latent cell the pixel box touches must be included, so start floors
    # and end ceils -- a box covering one pixel of a cell affects that cell.
    assert pixel_box_to_latent_box((17, 17, 33, 33), latent_h=44, latent_w=80) == (1, 1, 3, 3)


def test_pixel_box_to_latent_box_clips_to_grid():
    lx1, ly1, lx2, ly2 = pixel_box_to_latent_box((-50, -50, 5000, 5000), latent_h=44, latent_w=80)
    assert (lx1, ly1, lx2, ly2) == (0, 0, 80, 44)


def test_pixel_box_to_latent_box_rejects_degenerate_and_offgrid():
    for box in [(10, 10, 10, 20), (10, 10, 20, 10), (20, 10, 10, 20)]:
        try:
            pixel_box_to_latent_box(box, latent_h=44, latent_w=80)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for degenerate box {box}")

    try:  # entirely off the right edge of the grid
        pixel_box_to_latent_box((5000, 10, 5010, 20), latent_h=44, latent_w=80)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a box that maps to an empty latent region")


def test_anchor_latent_frame_is_the_frozen_frame_before_the_edit():
    # For a > 0 the causal singleton guarantees latent frame 0 is frozen, so
    # the anchor is real pre-drift source footage that the edit never touches.
    for a, b, video_len in [(30, 60, 121), (20, 40, 121), (50, 120, 121), (10, 15, 100)]:
        window = build_regeneration_window(a, b, video_len)
        mask = build_latent_regen_mask(window)
        idx = anchor_latent_frame(window)
        assert idx == 0, f"a={a} b={b}: expected anchor at latent frame 0, got {idx}"
        assert mask[idx] is False, "the anchor must be a frozen latent frame"
        assert mask[idx + 1] is True, "the anchor must be immediately before the regen region"


def test_anchor_latent_frame_is_none_when_edit_starts_at_frame_zero():
    window = build_regeneration_window(a=0, b=5, video_len=100)
    assert anchor_latent_frame(window) is None


def test_feathered_box_weight_is_one_in_the_interior_and_zero_outside():
    w = build_feathered_box_weight([(160, 96, 480, 288)], latent_h=44, latent_w=80, feather=2)
    # latent box is (10, 6, 30, 18); with feather=2 the interior stays at 1
    assert w[6 + 2:18 - 2, 10 + 2:30 - 2].min() == 1.0
    assert w[:6, :].max() == 0.0
    assert w[18:, :].max() == 0.0
    assert w[:, :10].max() == 0.0
    assert w[:, 30:].max() == 0.0


def test_feathered_box_weight_ramps_monotonically_at_the_border():
    w = build_feathered_box_weight([(160, 96, 480, 288)], latent_h=44, latent_w=80, feather=3)
    row = w[12, 10:30]  # a horizontal cut through the middle of the box
    assert 0.0 < row[0] < row[1] < row[2] < 1.0, "leading edge must ramp up smoothly"
    assert 0.0 < row[-1] < row[-2] < row[-3] < 1.0, "trailing edge must ramp down smoothly"


def test_feathered_box_weight_zero_feather_is_a_hard_box():
    w = build_feathered_box_weight([(160, 96, 480, 288)], latent_h=44, latent_w=80, feather=0)
    assert set(np.unique(w)) == {0.0, 1.0}
    assert w[6:18, 10:30].all()


def test_feathered_box_weight_overlapping_boxes_take_the_max():
    boxes = [(160, 96, 480, 288), (320, 96, 640, 288)]  # overlap in x
    w = build_feathered_box_weight(boxes, latent_h=44, latent_w=80, feather=2)
    assert w.max() <= 1.0, "overlap must not accumulate past 1"
    # the seam between the two boxes is interior to their union -> fully weighted
    assert w[12, 20] == 1.0


def test_feathered_box_weight_survives_a_box_narrower_than_the_feather():
    w = build_feathered_box_weight([(160, 96, 192, 128)], latent_h=44, latent_w=80, feather=4)
    assert w.max() <= 1.0
    assert w.max() > 0.0  # degrades to an all-ramp bump rather than vanishing


def test_out_of_range_frame_indices_still_rejected():
    for kwargs in [dict(a=-1, b=5, video_len=100), dict(a=10, b=100, video_len=100)]:
        try:
            build_regeneration_window(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for out-of-range indices: {kwargs}")


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} tests passed")
