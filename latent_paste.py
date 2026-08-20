"""Latent-space region pasting: pin a box to its pre-drift appearance.

The problem this solves: Wan2.2 sometimes lets scene furniture drift when it
should be bolted in place (e.g. the whole cabinet slides while the robot arm
closes a drawer). Text conditioning has almost no purchase on that -- "the
cabinet stays still" is a statement about pixel geometry -- so the correction
is applied directly to the latent during denoising instead.

This is the approach SG-I2V (ICLR 2025) evaluated as its `FreeTraj` baseline
and rejected: pasting latent content scored the worst FID in their Table 1,
and Appendix A reports the modified latents "fell out of distribution". Three
things differ here, and they are the reason it is worth doing anyway:

  * they pasted into *initial noise* of a text-to-video model, relocating
    content that did not exist yet; this pastes into a partially denoised
    latent of a real encoded video, where the content exists and is merely in
    the wrong place;
  * the trajectory is degenerate -- holding something still means source and
    destination coincide spatially, so nothing is teleported and no hole is
    opened behind it;
  * `write_delta` below corrects the *difference* from a reference rather
    than overwriting, which is a much smaller perturbation than a raw paste.

`fft_restore` ports SG-I2V's own remedy for off-distribution latents (their
Sec. 3.4 / Eq. 2), which matters more here than in the paper, not less.
"""

from __future__ import annotations

import numpy as np
import torch


def target_latent_frames(regen_mask: list[bool], anchor_idx: int | None) -> list[int]:
    """Latent frames the paste may write to: the regenerated ones, minus the
    anchor itself. Frozen frames are excluded because the denoising loop
    re-pastes clean content over them after every step anyway -- writing there
    would be silently discarded work.
    """
    return [i for i, regen in enumerate(regen_mask) if regen and i != anchor_idx]


def write_delta(
    latent: torch.Tensor,
    z: torch.Tensor,
    weight_hw: torch.Tensor,
    anchor_idx: int,
    target_frames: list[int],
    sigma: float,
    strength: float = 1.0,
) -> torch.Tensor:
    """Pull the boxed region of `target_frames` toward the anchor frame's content.

    `latent` is the noisy latent at noise level `sigma`, `z` the clean VAE
    encoding of the source video, both `(C, T, H, W)`. `weight_hw` is a
    `(H, W)` map in [0, 1] (see `frame_mapping.build_feathered_box_weight`).

    Wan's flow-matching convention is `x = (1 - sigma) * x0 + sigma * eps`
    (the same formula `frame_range_edit.py` uses to build its SDEdit init).
    Replacing the content while keeping the destination's own noise means
    changing only the `x0` term, i.e. adding `(1 - sigma) * (x0_anchor - x0_dst)`.
    Taking `z` as the estimate of `x0` at both positions gives

        latent[t] += (1 - sigma) * weight * strength * (z[anchor] - z[t])

    Three properties make this preferable to overwriting the region outright:

    1. **The noise realisation is preserved exactly**, by construction. A
       literal copy of `latent[anchor]` into `latent[t]` would carry
       `eps_anchor` along with it, and identical noise at two temporal
       positions reads to the model as "these are the same frame" -- which can
       freeze everything in the box, the occluding gripper included.
    2. **No division by sigma**, so nothing is amplified as sigma shrinks.
       Recovering the noise explicitly, as `(x - (1-sigma) z) / sigma`, blows
       up exactly where the estimate is least reliable.
    3. **It is a no-op wherever the scene already matches.** The correction is
       proportional to `z[anchor] - z[t]`, so it acts only where there is
       genuine drift and leaves everything else untouched.

    The `x0 ~ z` approximation is exact when the latent still encodes the
    source video and degrades as denoising moves the content away from it --
    which is why callers should apply this only at high sigma (early steps),
    where it holds well and where layout is still being decided. That timing
    matches SG-I2V's own finding (their Figs. 10, 15, 16) that late edits
    wreck visual quality.

    Frozen frames and everything outside the boxes are untouched: `weight_hw`
    is zero there, and `target_frames` excludes frozen positions.
    """
    if not target_frames:
        return latent
    if latent.shape != z.shape:
        raise ValueError(f"latent {tuple(latent.shape)} and z {tuple(z.shape)} must have the same shape")
    if weight_hw.shape != latent.shape[-2:]:
        raise ValueError(
            f"weight_hw {tuple(weight_hw.shape)} must match the latent spatial size "
            f"{tuple(latent.shape[-2:])}")
    if not (0.0 <= strength <= 1.0):
        raise ValueError(f"strength must be in [0, 1], got {strength}")

    out = latent.clone()
    w = weight_hw.to(device=latent.device, dtype=latent.dtype) * float(strength)
    anchor_content = z[:, anchor_idx]  # (C, H, W)

    idx = torch.tensor(target_frames, device=latent.device, dtype=torch.long)
    delta = anchor_content.unsqueeze(1) - z[:, idx]  # (C, len(idx), H, W)
    out[:, idx] = out[:, idx] + (1.0 - sigma) * w * delta
    return out


def soften_toward_noise(
    latent: torch.Tensor,
    z: torch.Tensor,
    frame: int,
    vacated_hw: torch.Tensor,
    sigma: float,
    fill: float = 1.0,
) -> torch.Tensor:
    """Erase the content bias in `vacated_hw` cells of one frame, in place.

    This answers "what is left where the object moved away from" -- the
    unfinished line 4 of the drag formulation. It asserts *no* replacement
    content; it removes the existing content's influence and leaves the region
    for the denoiser to fill from surrounding context and the prompt.

    With `x = (1 - sigma) * x0 + sigma * eps` and `x0 ~ z`, subtracting the
    content term drives `x0` to zero while leaving the noise untouched:

        latent[frame] -= (1 - sigma) * vacated * fill * z[frame]

    at `fill = 1` this leaves exactly `sigma * eps`.

    Zero is the right target to shrink toward, not an arbitrary one:
    `Wan2_2_VAE.encode` normalises latents by `(raw - mean) / std` per channel
    (`Wan2.2/wan/modules/vae2_2.py`), so the latent distribution is centred on
    zero. Shrinking `x0` toward 0 is shrinking toward the distribution's mean --
    the least-committal content that can be written.

    Note this **keeps the region's own noise realisation** rather than drawing
    fresh noise. Resampling would put a discontinuity in the noise field
    exactly at the box border, which is the kind of thing a denoiser reads as
    out-of-distribution; it would also need an RNG threaded through, making the
    result harder to reproduce. Removing the content term achieves the same
    `x0 -> 0` with none of that, and in the same algebraic form (a
    `(1 - sigma)`-scaled subtraction) as `write_delta` itself.
    """
    if not (0.0 <= fill <= 1.0):
        raise ValueError(f"fill must be in [0, 1], got {fill}")
    if fill == 0.0:
        return latent
    v = vacated_hw.to(device=latent.device, dtype=latent.dtype) * float(fill)
    latent[:, frame] = latent[:, frame] - (1.0 - sigma) * v * z[:, frame]
    return latent


def write_delta_trajectory(
    latent: torch.Tensor,
    z: torch.Tensor,
    anchor_idx: int,
    plan: list,
    sigma: float,
    strength: float = 1.0,
    vacated_fill: float = 1.0,
) -> torch.Tensor:
    """Move a box's content along a trajectory, gradient-free.

    `plan` is a list of `frame_mapping.LatentShift` -- per latent frame, where
    the box sits now (`dst`) versus where it sat at the anchor (`src`), both
    already clipped to the grid and guaranteed the same size.

    Two writes per frame, which are the two lines of the drag formulation:

        Z[t, dst] += (1 - sigma) * w * (z[anchor, src] - z[t, dst])   # line 3
        Z[t, vacated] -= (1 - sigma) * fill * z[t, vacated]           # line 4

    The first is `write_delta` with a spatial offset: the destination takes the
    anchor's content while keeping its own noise. The second removes the
    content the object left behind. They never overlap -- `vacated` is built as
    the difference of the source and destination weight maps, so it is zero
    wherever the destination still covers a cell.

    Unlike the static case, content genuinely *is* being relocated here, so the
    ghosting risk is real: line 4 removes the *bias* toward redrawing the
    object at its old position, but cannot guarantee the model won't put it
    back when context strongly implies it. Run with `vacated_fill=0` to see
    the uncorrected ghost, then compare.
    """
    if latent.shape != z.shape:
        raise ValueError(f"latent {tuple(latent.shape)} and z {tuple(z.shape)} must have the same shape")
    if not (0.0 <= strength <= 1.0):
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    if not plan:
        return latent

    out = latent.clone()
    num_frames = latent.shape[1]

    for shift in plan:
        if not (0 <= shift.frame < num_frames):
            raise ValueError(f"plan references latent frame {shift.frame}, but T={num_frames}")
        sy1, sy2, sx1, sx2 = shift.src
        dy1, dy2, dx1, dx2 = shift.dst

        w = torch.from_numpy(np.ascontiguousarray(shift.weight)).to(
            device=latent.device, dtype=latent.dtype) * float(strength)
        src = z[:, anchor_idx, sy1:sy2, sx1:sx2]
        dst = z[:, shift.frame, dy1:dy2, dx1:dx2]
        out[:, shift.frame, dy1:dy2, dx1:dx2] = (
            out[:, shift.frame, dy1:dy2, dx1:dx2] + (1.0 - sigma) * w * (src - dst))

        if vacated_fill > 0.0:
            vacated = torch.from_numpy(np.ascontiguousarray(shift.vacated))
            soften_toward_noise(out, z, shift.frame, vacated, sigma, fill=vacated_fill)

    return out


def butterworth_low_pass_filter(
    shape_hw: tuple[int, int],
    d_s: float = 0.5,
    n: int = 4,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """2-D Butterworth low-pass mask over the (H, W) frequency plane.

    Vectorised port of `SG-I2V/src/utils.py:83-97` (MIT, `SG-I2V/LICENSE`),
    which itself follows FreeTraj. `d_s` is the cut-off: 0 passes nothing.
    """
    h, w = shape_hw
    if d_s == 0:
        return torch.zeros((h, w), device=device, dtype=dtype)
    yy = (2 * torch.arange(h, device=device, dtype=dtype) / h - 1) ** 2
    xx = (2 * torch.arange(w, device=device, dtype=dtype) / w - 1) ** 2
    d_square = yy[:, None] + xx[None, :]
    return 1.0 / (1.0 + (d_square / d_s ** 2) ** n)


def fft_restore(edited: torch.Tensor, original: torch.Tensor, d_s: float = 0.5) -> torch.Tensor:
    """Keep `edited`'s low spatial frequencies, restore `original`'s high ones.

    SG-I2V Eq. 2 (`SG-I2V/src/pipeline.py:165-184`). The motivating
    observation is that the layout/motion signal lives almost entirely in the
    low frequencies of the latent, while the artifacts introduced by editing
    it live in the high ones -- so this discards the damage nearly for free.
    Their Fig. 8 sweep found quality degrades sharply as `d_s` approaches 1
    (keep everything edited) while motion control is barely affected until
    `d_s` approaches 0, hence the 0.5 default.

    Both tensors are `(C, T, H, W)`; the transform is per (channel, frame)
    over the spatial axes only.
    """
    if edited.shape != original.shape:
        raise ValueError(
            f"edited {tuple(edited.shape)} and original {tuple(original.shape)} must have the same shape")
    axes = (-2, -1)
    lpf = butterworth_low_pass_filter(
        edited.shape[-2:], d_s=d_s, device=edited.device, dtype=torch.float32)

    edited_freq = torch.fft.fftshift(torch.fft.fftn(edited.float(), dim=axes), dim=axes)
    original_freq = torch.fft.fftshift(torch.fft.fftn(original.float(), dim=axes), dim=axes)

    merged = edited_freq * lpf + original_freq * (1.0 - lpf)
    restored = torch.fft.ifftn(torch.fft.ifftshift(merged, dim=axes), dim=axes).real
    return restored.to(dtype=edited.dtype)


def weight_to_tensor(weight_hw: np.ndarray, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(weight_hw)).to(device=device, dtype=dtype)
