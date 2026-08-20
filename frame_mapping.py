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


def pixel_box_to_latent_box(
    box: tuple[int, int, int, int],
    latent_h: int,
    latent_w: int,
    vae_spatial_stride: int = 16,
) -> tuple[int, int, int, int]:
    """Map a pixel-space box `(x1, y1, x2, y2)` to latent-grid `(lx1, ly1, lx2, ly2)`.

    Ends are exclusive, matching Python slicing. The box is rounded *outward*
    (floor the start, ceil the end) so every latent cell the pixel box touches
    is included -- a box that covers even one pixel of a cell affects that
    cell's encoded content.

    The divisor is `vae_spatial_stride` (16 for ti2v-5B), NOT the 32 used by
    `build_spatial_latent_regen_mask`. That function pools at
    `vae_spatial_stride * patch_spatial` only because the mask it returns is
    consumed by the DiT's stride-`patch_spatial` per-token timestep subsample
    (see its docstring). A direct write into the latent tensor never goes
    through that path, so it is free to address individual latent cells --
    which is what makes feathering (`build_feathered_box_weight`) possible.
    """
    x1, y1, x2, y2 = box
    if not (x1 < x2 and y1 < y2):
        raise ValueError(f"box must have x1<x2 and y1<y2, got {box}")

    lx1 = max(0, x1 // vae_spatial_stride)
    ly1 = max(0, y1 // vae_spatial_stride)
    lx2 = min(latent_w, -(-x2 // vae_spatial_stride))  # ceil
    ly2 = min(latent_h, -(-y2 // vae_spatial_stride))

    if lx1 >= lx2 or ly1 >= ly2:
        raise ValueError(
            f"box {box} maps to an empty latent region on a {latent_h}x{latent_w} grid "
            f"(stride {vae_spatial_stride}) -- it is either off-grid or smaller than one latent cell")
    return lx1, ly1, lx2, ly2


def anchor_latent_frame(
    window: FrameRangeWindow,
    temporal_stride: int = DEFAULT_TEMPORAL_STRIDE,
) -> int | None:
    """Index of the last frozen latent frame before the regenerate region.

    This is the source of truth for "what did the scene look like before the
    drift": a latent frame the edit is guaranteed never to touch, holding real
    encoded source footage. `build_regeneration_window` makes one exist by
    construction whenever `a > 0` -- it sets `window_start = a - 2` and
    `edit_start = a - 1`, and latent frame 0 is a causal singleton covering
    only local pixel 0, so latent frame 0 is always frozen.

    Returns `None` when the regenerate region starts at latent frame 0 (i.e.
    `a == 0`), where no such frame exists and the caller must supply its own
    reference.
    """
    mask = build_latent_regen_mask(window, temporal_stride)
    first_regen = next((i for i, regen in enumerate(mask) if regen), None)
    if first_regen is None:
        raise ValueError("window has no regenerated latent frames")
    return None if first_regen == 0 else first_regen - 1


def feathered_map_from_latent_box(
    latent_h: int,
    latent_w: int,
    lbox: tuple[int, int, int, int],
    feather: int = 2,
) -> np.ndarray:
    """A `(latent_h, latent_w)` weight map for one box given in *latent cells*.

    Unlike `build_feathered_box_weight`, `lbox` may extend outside the grid --
    the raised-cosine patch is built at the box's full size and then cropped,
    so a box hanging off an edge keeps the ramp geometry it would have had if
    the frame were larger, instead of ramping against the frame border. That
    matters for trajectories, where a moving box legitimately runs off-frame.
    """
    lx1, ly1, lx2, ly2 = lbox
    weight = np.zeros((latent_h, latent_w), dtype=np.float32)
    cx1, cy1 = max(0, lx1), max(0, ly1)
    cx2, cy2 = min(latent_w, lx2), min(latent_h, ly2)
    if cx1 >= cx2 or cy1 >= cy2:
        return weight

    patch = _cosine_ramp(ly2 - ly1, feather)[:, None] * _cosine_ramp(lx2 - lx1, feather)[None, :]
    weight[cy1:cy2, cx1:cx2] = patch[cy1 - ly1:cy2 - ly1, cx1 - lx1:cx2 - lx1]
    return weight


def build_feathered_box_weight(
    boxes: list[tuple[int, int, int, int]],
    latent_h: int,
    latent_w: int,
    vae_spatial_stride: int = 16,
    feather: int = 2,
) -> np.ndarray:
    """A `(latent_h, latent_w)` float weight map in [0, 1], 1 inside the boxes.

    A hard rectangular write leaves a seam on the latent grid that the VAE
    decoder turns into a visible `vae_spatial_stride`-pixel blocky edge. This
    ramps the weight from 0 to 1 over `feather` latent cells at each box
    border with a cosine (raised-cosine) profile, which has a continuous first
    derivative and so leaves no ridge for the decoder to sharpen.

    Overlapping boxes take the element-wise maximum, so an overlap is fully
    weighted rather than double-counted.

    `feather=0` gives hard edges. Note that the ramp grows *inward* from the
    box border, so a box narrower than `2 * feather` cells never reaches
    weight 1 anywhere -- deliberate, since a box that small has no interior to
    protect.
    """
    weight = np.zeros((latent_h, latent_w), dtype=np.float32)
    if feather < 0:
        raise ValueError(f"feather must be >= 0, got {feather}")

    for box in boxes:
        lbox = pixel_box_to_latent_box(box, latent_h, latent_w, vae_spatial_stride)
        weight = np.maximum(weight, feathered_map_from_latent_box(latent_h, latent_w, lbox, feather))

    return weight


def _cosine_ramp(length: int, feather: int) -> np.ndarray:
    """1-D window of `length` cells: raised-cosine ramp up over `feather`
    cells, flat 1 in the middle, ramp back down. Degrades gracefully when
    `length < 2 * feather` (the plateau just disappears)."""
    w = np.ones(length, dtype=np.float32)
    if feather == 0:
        return w
    # +1 so the first cell inside the box gets a nonzero weight rather than 0
    ramp = 0.5 * (1.0 - np.cos(np.pi * (np.arange(feather) + 1) / (feather + 1)))
    n = min(feather, length)
    w[:n] = np.minimum(w[:n], ramp[:n])
    w[length - n:] = np.minimum(w[length - n:], ramp[:n][::-1])
    return w


@dataclass(frozen=True)
class LatentShift:
    """One latent frame's copy instruction, in latent-cell coordinates.

    `src` and `dst` are `(y1, y2, x1, x2)` half-open slices of identical size,
    already clipped to the grid. `weight` is the feathered patch matching that
    clipped size; `vacated` is a full `(H, W)` map marking cells the box moved
    *out of* -- zero wherever the destination still covers them.
    """

    frame: int
    src: tuple[int, int, int, int]
    dst: tuple[int, int, int, int]
    weight: np.ndarray
    vacated: np.ndarray
    cell_shift: tuple[int, int]  # (dy, dx) in latent cells, for reporting


def load_trajectory_npy(
    path,
    orig_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[np.ndarray]:
    """Load SG-I2V's `[N, 2+F, 2]` trajectory format.

    Rows `[:, :2]` are the box's top-left/bottom-right corners as `(w, h)`;
    rows `[:, 2:]` are the box *centre* in each of F frames, also `(w, h)`.
    Coordinates are rescaled from `orig_size` to `target_size` exactly as
    `SG-I2V/inference.py:36-37` does.

    Returns one `(F, 4)` array of `(x1, y1, x2, y2)` boxes per object. As in
    SG-I2V, the box keeps its frame-0 size throughout and only the centre
    moves (`SG-I2V/inference.py:46-49`).
    """
    ret = np.load(path).astype(np.float32)
    if ret.ndim != 3 or ret.shape[1] < 3 or ret.shape[2] != 2:
        raise ValueError(
            f"trajectory must have shape [N, 2+F, 2] with F >= 1, got {ret.shape}")

    ow, oh = orig_size
    tw, th = target_size
    ret[:, :, 0] *= tw / ow
    ret[:, :, 1] *= th / oh

    out = []
    for obj in ret:
        (x1, y1), (x2, y2) = obj[0], obj[1]
        centres = obj[2:]
        delta = centres - centres[0]  # (F, 2) as (dw, dh)
        boxes = np.stack([
            x1 + delta[:, 0], y1 + delta[:, 1],
            x2 + delta[:, 0], y2 + delta[:, 1],
        ], axis=1)
        out.append(boxes.astype(np.float32))
    return out


def linear_trajectory(box: tuple[int, int, int, int], move_to: tuple[float, float], num_frames: int) -> np.ndarray:
    """`(num_frames, 4)` boxes sweeping `box` linearly by `move_to = (dx, dy)`.

    Frame 0 is `box` itself and the last frame is `box` displaced by the full
    `(dx, dy)`, so `move_to` is a *displacement*, not a destination corner.
    """
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    x1, y1, x2, y2 = box
    dx, dy = move_to
    t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32) if num_frames > 1 else np.zeros(1, np.float32)
    return np.stack([x1 + t * dx, y1 + t * dy, x2 + t * dx, y2 + t * dy], axis=1).astype(np.float32)


def resample_trajectory(boxes: np.ndarray, num_frames: int) -> np.ndarray:
    """Linearly resample an `(F, 4)` box path to `num_frames` entries.

    A trajectory authored against the whole video, or against the edit range,
    will not generally have one entry per frame of the crop window the model
    actually sees. Resampling is more forgiving than demanding an exact length,
    and the caller reports when it happens.
    """
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must be (F, 4), got {boxes.shape}")
    if len(boxes) == num_frames:
        return boxes.astype(np.float32)
    if len(boxes) == 1:
        return np.repeat(boxes.astype(np.float32), num_frames, axis=0)
    src = np.linspace(0.0, 1.0, len(boxes))
    dst = np.linspace(0.0, 1.0, num_frames)
    return np.stack([np.interp(dst, src, boxes[:, k]) for k in range(4)], axis=1).astype(np.float32)


def trajectory_to_latent_plan(
    window: FrameRangeWindow,
    per_frame_boxes: np.ndarray,
    anchor_latent_idx: int,
    target_frames: list[int],
    latent_h: int,
    latent_w: int,
    vae_spatial_stride: int = 16,
    temporal_stride: int = DEFAULT_TEMPORAL_STRIDE,
    feather: int = 2,
) -> list[LatentShift]:
    """Turn a per-pixel-frame box path into per-latent-frame copy instructions.

    `per_frame_boxes` is `(window.num_pixel_frames, 4)` in model-input pixel
    coordinates, indexed by *window-local* pixel frame.

    Two things this has to get right, both of which are easy to get wrong:

    **The latent box size is computed once, from the anchor.** Rounding a pixel
    box outward to latent cells does not preserve its size -- a 140px-wide box
    starting at x=170 spans 10 cells, but starting at x=176 it spans 9. Mapping
    each frame's box independently would therefore produce mismatched shapes
    partway along the path. Instead the anchor's box fixes the size, and every
    other frame places that same-sized box at a rounded cell *offset*.

    **A latent frame covers four pixel frames**, so there is no single "the"
    box for it. The mean box centre over the covered pixel range is used, which
    is the least arbitrary choice available; motion faster than the temporal
    stride is averaged away regardless.

    Motion is therefore quantised to whole `vae_spatial_stride`-pixel cells.
    `cell_shift` on each returned entry exposes that, so a path too slow to
    register as anything but zeros is visible before any GPU time is spent.
    """
    n_pixel = window.num_pixel_frames
    if per_frame_boxes.shape != (n_pixel, 4):
        raise ValueError(
            f"per_frame_boxes must be ({n_pixel}, 4) to match the window, got {per_frame_boxes.shape}")

    def mean_box(latent_idx: int) -> np.ndarray:
        lo, hi = latent_frame_to_pixel_range(latent_idx, temporal_stride)
        lo, hi = min(lo, n_pixel - 1), min(hi, n_pixel - 1)
        return per_frame_boxes[lo:hi + 1].mean(axis=0)

    anchor_box = mean_box(anchor_latent_idx)
    src_lbox = pixel_box_to_latent_box(
        tuple(int(round(v)) for v in anchor_box), latent_h, latent_w, vae_spatial_stride)
    sx1, sy1, sx2, sy2 = src_lbox
    bw, bh = sx2 - sx1, sy2 - sy1
    anchor_centre = np.array([(anchor_box[0] + anchor_box[2]) / 2, (anchor_box[1] + anchor_box[3]) / 2])
    src_map = feathered_map_from_latent_box(latent_h, latent_w, src_lbox, feather)
    full_patch = _cosine_ramp(bh, feather)[:, None] * _cosine_ramp(bw, feather)[None, :]

    plan: list[LatentShift] = []
    for t in target_frames:
        box = mean_box(t)
        centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        dx, dy = (int(v) for v in np.round((centre - anchor_centre) / vae_spatial_stride))

        dst_full = (sx1 + dx, sy1 + dy, sx2 + dx, sy2 + dy)
        cx1, cy1 = max(0, dst_full[0]), max(0, dst_full[1])
        cx2, cy2 = min(latent_w, dst_full[2]), min(latent_h, dst_full[3])
        if cx1 >= cx2 or cy1 >= cy2:
            continue  # box has left the frame entirely at this point in the path

        ox1, oy1 = cx1 - dst_full[0], cy1 - dst_full[1]
        ox2, oy2 = cx2 - dst_full[0], cy2 - dst_full[1]

        # The destination is excluded by its HARD footprint, not its feathered
        # one. Subtracting the feathered map would leave `vacated > 0` inside
        # the destination's own ramp, so the softening would partially erase
        # content the copy had just written there -- visible as a dimmed halo
        # around the object whenever source and destination overlap. The copy
        # still uses the feathered `full_patch`; only this exclusion is hard.
        dst_hard = feathered_map_from_latent_box(latent_h, latent_w, dst_full, feather=0)
        plan.append(LatentShift(
            frame=t,
            src=(sy1 + oy1, sy1 + oy2, sx1 + ox1, sx1 + ox2),
            dst=(cy1, cy2, cx1, cx2),
            weight=np.ascontiguousarray(full_patch[oy1:oy2, ox1:ox2]),
            vacated=np.clip(src_map - dst_hard, 0.0, 1.0),
            cell_shift=(dy, dx),
        ))
    return plan


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
