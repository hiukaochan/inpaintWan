"""Pixel-frame <-> latent-frame index mapping for Wan2.2's causal 3D VAE.

Wan2.2's VAE has temporal stride 4 with a causal convention: latent frame 0
encodes pixel frame 0 alone, and every following latent frame encodes a block
of 4 pixel frames. This mirrors the `(F - 1) // vae_stride[0] + 1` formula
used throughout `Wan2.2/wan/textimage2video.py` (`WanTI2V.t2v`/`i2v`) for
`target_shape`/`seq_len`, and `Wan2.2/wan/configs/wan_ti2v_5B.py` which sets
`vae_stride = (4, 16, 16)`.

Pure Python, no torch/model dependency, so it can be unit-tested without a
GPU or checkpoints.
"""

from dataclasses import dataclass

import numpy as np

DEFAULT_TEMPORAL_STRIDE = 4
CONTEXT_MARGIN = 2  # frames A-2 / B+2 are the frozen anchors (per spec)


def num_latent_frames(num_pixel_frames: int, temporal_stride: int = DEFAULT_TEMPORAL_STRIDE) -> int:
    if num_pixel_frames < 1:
        raise ValueError("num_pixel_frames must be >= 1")
    return (num_pixel_frames - 1) // temporal_stride + 1


def latent_frame_to_pixel_range(latent_idx: int, temporal_stride: int = DEFAULT_TEMPORAL_STRIDE) -> tuple[int, int]:
    """Inclusive [start, end] pixel-frame indices a latent frame covers."""
    if latent_idx < 0:
        raise ValueError("latent_idx must be >= 0")
    if latent_idx == 0:
        return (0, 0)
    start = temporal_stride * (latent_idx - 1) + 1
    end = temporal_stride * latent_idx
    return (start, end)


@dataclass(frozen=True)
class FrameRangeWindow:
    """A crop window of the source video used for one edit.

    All indices are in the source video's pixel-frame coordinate space.
    `window_end` is always a real frame index (`<= video_len - 1`); it is
    never itself padded. `pad_end` counts synthetic frames (replicated from
    the real frame at `window_end`) appended after it purely to satisfy the
    VAE's `4n+1` pixel-count requirement when there isn't enough real
    trailing footage -- i.e. when the edit region extends through the true
    last frame of the video. At most `temporal_stride - 1` such frames are
    ever needed, and they always fall inside the same latent block that
    already overlaps `edit_end`, so `build_latent_regen_mask` always marks
    that block regenerated, never frozen -- there is no fabricated anchor.
    They're discarded before the final splice regardless, since
    `edit_end <= window_end` always holds.
    """

    window_start: int  # first pixel frame included in the crop
    window_end: int  # last REAL pixel frame included in the crop
    edit_start: int  # first frame that may be regenerated
    edit_end: int  # last frame that may be regenerated
    pad_end: int = 0  # synthetic frames appended after window_end, VAE-alignment only

    @property
    def num_pixel_frames(self) -> int:
        return (self.window_end - self.window_start + 1) + self.pad_end

    def local(self, global_idx: int) -> int:
        return global_idx - self.window_start


def build_regeneration_window(
    a: int,
    b: int,
    video_len: int,
    temporal_stride: int = DEFAULT_TEMPORAL_STRIDE,
    context_blocks: int = 1,
) -> FrameRangeWindow:
    """Build a crop window for editing pixel frames [a, b].

    Regenerates [a-1, b+1]; a-2 (and everything further left) is frozen
    context and never modified. `window_start = a - 2` always: latent frame
    0 is a causal singleton exactly at `a-2` (Wan2.2's VAE has temporal
    stride `temporal_stride` but latent frame 0 covers only pixel frame 0
    of the window), so the left side is automatically clean -- it never
    shares a latent block with the regenerate region.

    The right side needs care: the latent block containing `b+1` generally
    also covers a few pixels past it (block size == temporal_stride), so a
    fixed `+2` margin can leave the intended `b+2` anchor sharing a block
    with the regenerate region -- i.e. not actually frozen. To guarantee a
    real, untouched anchor immediately after the edit, the regenerate region
    is rounded outward to whole latent blocks and `context_blocks` full
    blocks (`temporal_stride` real frames each) of guaranteed-frozen content
    are required immediately after it. This also makes the window length
    automatically satisfy Wan2.2's `4n+1` frame-count requirement, so no
    iterative padding is needed.

    Two exact boundary cases are special-cased because they need no anchor
    at all on that side -- there's nothing before pixel frame 0 or after the
    true last frame to freeze in the first place:

    - `a == 0`: skip the left context-margin requirement; `window_start` and
      `edit_start` both become 0. The right-side math only depends on
      `edit_end - window_start`, so the `4n+1` window length is unaffected
      and no padding is needed.
    - `b == video_len - 1`: skip the trailing context-block requirement
      entirely (no `context_blocks * temporal_stride` anchor is required or
      fabricated); `edit_end` clamps to `video_len - 1` instead of `b + 1`
      (which would be out of bounds). Rounding the regenerate region out to
      a whole latent block can still leave a small (`< temporal_stride`)
      alignment remainder past the true last frame, which is recorded as
      `pad_end` and filled with synthetic (replicated) frames by the caller
      -- see `FrameRangeWindow`. That remainder always lands inside the
      already-regenerated final block, never a frozen anchor.

    Any other `a`/`b` that doesn't clear the normal margin still raises,
    unchanged.
    """
    if a > b:
        raise ValueError("a must be <= b")
    if context_blocks < 1:
        raise ValueError("context_blocks must be >= 1")
    if not (0 <= a < video_len) or not (0 <= b < video_len):
        raise ValueError(f"a and b must be valid frame indices in [0, {video_len - 1}]")

    at_video_start = a == 0
    at_video_end = b == video_len - 1

    if a - CONTEXT_MARGIN < 0 and not at_video_start:
        raise ValueError(
            f"frame range too close to the video start: need at least "
            f"{CONTEXT_MARGIN} frames of context before a")

    edit_start = 0 if at_video_start else a - 1
    edit_end = (video_len - 1) if at_video_end else b + 1
    window_start = 0 if at_video_start else a - CONTEXT_MARGIN

    local_edit_end = edit_end - window_start
    aligned_regen_end = -(-local_edit_end // temporal_stride) * temporal_stride  # round up to a block end
    trailing_context = 0 if at_video_end else temporal_stride * context_blocks
    desired_window_end = window_start + aligned_regen_end + trailing_context

    if at_video_end:
        window_end = video_len - 1
        pad_end = desired_window_end - window_end
        print(f"[DEBUG] build_regeneration_window: window_end={window_end}, desired_window_end={desired_window_end}, pad_end={pad_end}")
    else:
        if desired_window_end > video_len - 1:
            raise ValueError(
                "frame range too close to the video end: need "
                f"{desired_window_end - (video_len - 1)} more frame(s) of trailing context after b")
        window_end = desired_window_end
        pad_end = 0

    return FrameRangeWindow(window_start, window_end, edit_start, edit_end, pad_end)


def build_latent_regen_mask(
    window: FrameRangeWindow,
    temporal_stride: int = DEFAULT_TEMPORAL_STRIDE,
) -> list[bool]:
    """Per-latent-frame mask: True = regenerate, False = frozen context.

    A latent frame is frozen only if the pixel-frame block it covers falls
    entirely outside [edit_start, edit_end]; otherwise it's regenerated. The
    final splice back into the full video (not this mask) is what guarantees
    byte-identical output outside [edit_start, edit_end] -- this mask only
    controls what the model is allowed to change internally.
    """
    n_pixel = window.num_pixel_frames
    n_latent = num_latent_frames(n_pixel, temporal_stride)
    local_edit_start = window.local(window.edit_start)
    local_edit_end = window.local(window.edit_end)

    mask = []
    for i in range(n_latent):
        lo, hi = latent_frame_to_pixel_range(i, temporal_stride)
        hi = min(hi, n_pixel - 1)
        overlaps_edit = not (hi < local_edit_start or lo > local_edit_end)
        mask.append(overlaps_edit)
    return mask


def build_spatial_latent_regen_mask(
    window: FrameRangeWindow,
    pixel_mask: np.ndarray,
    temporal_stride: int = DEFAULT_TEMPORAL_STRIDE,
    vae_spatial_stride: int = 16,
    patch_spatial: int = 2,
) -> np.ndarray:
    """Per-latent-position mask (True = regenerate) that also varies spatially.

    `pixel_mask` is a bool array `(num_pixel_frames, H, W)` at the model-input
    resolution, aligned to `window` the same way the window's pixel frames
    are. A latent position is regenerated only if it's both temporally inside
    `[edit_start, edit_end]` (same rule as `build_latent_regen_mask`) AND its
    corresponding pixel block contains at least one True pixel -- the spatial
    mask can only narrow what the temporal mask already allows, never widen it.

    The DiT groups `patch_spatial x patch_spatial` latent pixels into one
    token and (per `frame_range_edit.py`'s per-token timestep trick) reads
    the mask via a stride-`patch_spatial` subsample -- so the returned mask
    must be constant within every such block, or that subsample would pick
    an arbitrary corner of an inconsistent block. Pooling at
    `vae_spatial_stride * patch_spatial` (32 for ti2v-5B) and re-expanding to
    latent resolution guarantees this by construction.
    """
    n_pixel = window.num_pixel_frames
    if pixel_mask.shape[0] != n_pixel:
        raise ValueError(
            f"pixel_mask has {pixel_mask.shape[0]} frames but the window covers {n_pixel}")
    h, w = pixel_mask.shape[1:]
    pool = vae_spatial_stride * patch_spatial
    if h % pool != 0 or w % pool != 0:
        raise ValueError(f"pixel_mask spatial size ({h}, {w}) must be a multiple of {pool}")
    h_pool, w_pool = h // pool, w // pool

    pooled = pixel_mask.reshape(n_pixel, h_pool, pool, w_pool, pool).any(axis=(2, 4))

    n_latent = num_latent_frames(n_pixel, temporal_stride)
    temporal_mask = build_latent_regen_mask(window, temporal_stride)

    mask = np.zeros((n_latent, h_pool, w_pool), dtype=bool)
    for i in range(n_latent):
        if not temporal_mask[i]:
            continue
        lo, hi = latent_frame_to_pixel_range(i, temporal_stride)
        hi = min(hi, n_pixel - 1)
        mask[i] = pooled[lo:hi + 1].any(axis=0)

    return mask.repeat(patch_spatial, axis=1).repeat(patch_spatial, axis=2)
