"""Local, model-free unit tests for frame_mapping.py's index math.

Run with: python inpainting/test_frame_mapping.py
"""

from frame_mapping import (
    build_latent_regen_mask,
    build_regeneration_window,
    latent_frame_to_pixel_range,
    num_latent_frames,
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
    assert window.pad_end > 0  # synthetic frames fill the missing trailing anchor
    assert window.num_pixel_frames % 4 == 1  # VAE 4n+1 requirement still holds


def test_full_video_edit():
    video_len = 50
    window = build_regeneration_window(a=0, b=video_len - 1, video_len=video_len)
    assert window.window_start == 0
    assert window.edit_start == 0
    assert window.edit_end == video_len - 1
    assert window.num_pixel_frames % 4 == 1


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
