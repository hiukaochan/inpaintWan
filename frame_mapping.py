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
    """

    window_start: int  # first pixel frame included in the crop
    window_end: int  # last pixel frame included in the crop
    edit_start: int  # A-1: first frame that may be regenerated
    edit_end: int  # B+1: last frame that may be regenerated

    @property
    def num_pixel_frames(self) -> int:
        return self.window_end - self.window_start + 1

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
    """
    if a > b:
        raise ValueError("a must be <= b")
    if context_blocks < 1:
        raise ValueError("context_blocks must be >= 1")
    if a - CONTEXT_MARGIN < 0:
        raise ValueError(
            f"frame range too close to the video start: need at least "
            f"{CONTEXT_MARGIN} frames of context before a")

    edit_start, edit_end = a - 1, b + 1
    window_start = a - CONTEXT_MARGIN

    local_edit_end = edit_end - window_start
    aligned_regen_end = -(-local_edit_end // temporal_stride) * temporal_stride  # round up to a block end
    window_end = window_start + aligned_regen_end + temporal_stride * context_blocks

    if window_end > video_len - 1:
        raise ValueError(
            "frame range too close to the video end: need "
            f"{window_end - (video_len - 1)} more frame(s) of trailing context after b")

    return FrameRangeWindow(window_start, window_end, edit_start, edit_end)


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
