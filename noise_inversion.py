"""Noise inversion for Wan2.2: run the sampling ODE backwards from a real video.

Wan2.2 is a rectified-flow model, not a DDPM. Its forward process is

    x_sigma = (1 - sigma) * z0 + sigma * eps

and the DiT predicts the velocity `v = eps - z0 = dx/dsigma`
(`training/conditioning.py::velocity_target`; UniPC's `x0 = x - sigma * v`,
`Wan2.2/wan/utils/fm_solvers_unipc.py:323`). Sampling integrates that ODE
from sigma=1 down to 0, and one Euler step of it is exactly

    x_{sigma - d} = x_sigma - d * v(x_sigma, sigma)            # denoise

The ODE is deterministic, so it can be run the other way:

    x_{sigma + d} = x_sigma + d * v(x_sigma, sigma)            # invert

Starting from the clean VAE encoding `z` (sigma=0) and stepping up to some
sigma_start gives a noisy latent that denoises back to *this* video -- unlike
SDEdit's `(1 - sigma) z + sigma eps` with a random eps, which denoises to a
plausible neighbour of it, and unlike `--cache_path`, which only exists for
videos generated with `--cache_output` and is invalidated by any later edit.

Why Euler and not UniPC/DPM++: those are multistep solvers whose internal
history has no clean reverse, so inversion and the matching denoise both use
plain Euler steps on the *same* sigma grid. The inversion is still only
approximate: a step evaluates `v` at `x_sigma` when the denoise step it has
to undo will evaluate it at `x_{sigma+d}`. `fixed_point_iters` closes that
gap by iterating `x_{sigma+d} = x_sigma + d * v(x_{sigma+d}, sigma+d)`, which
is the exact inverse of the Euler denoise step whenever it converges. CFG
above 1 makes the mismatch worse, so callers should invert at guide_scale 1.

Everything here is model-free: the caller supplies `velocity_fn(x, sigma)`
(per-token timestep + CFG), so the module runs and is tested on CPU.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from frame_mapping import LatentShift, feathered_map_from_latent_box


def shifted_sigmas(sampling_steps: int, shift: float) -> np.ndarray:
    """Same schedule as `Wan2.2/wan/utils/fm_solvers.py::get_sampling_sigmas`,
    restated here so this module does not import the vendored `wan` package
    (whose `__init__` pulls in the whole model stack). The test file asserts
    the two agree wherever `wan` is importable."""
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    return shift * sigma / (1 + (shift - 1) * sigma)


class EulerFlowScheduler:
    """First-order Euler solver for Wan's flow ODE.

    Exposes the three things `StaticPasteEditor._denoise` uses from Wan's own
    schedulers -- `.timesteps`, `.sigmas` and `.step(...)` -- with the same
    semantics, so the existing loop (paste, freeze, per-token timestep) runs
    unchanged on top of it. `sigmas` has one more entry than `timesteps`, the
    trailing 0, exactly as `FlowUniPCMultistepScheduler` does with
    `final_sigmas_type="zero"`.
    """

    def __init__(
        self,
        sampling_steps: int,
        shift: float,
        num_train_timesteps: int = 1000,
        device: torch.device | str = "cpu",
    ):
        if sampling_steps < 1:
            raise ValueError(f"sampling_steps must be >= 1, got {sampling_steps}")
        sigmas = np.append(shifted_sigmas(sampling_steps, shift), 0.0)
        self.num_train_timesteps = num_train_timesteps
        # sigmas stay on CPU (read as Python floats), timesteps go to the
        # device the model runs on -- same split as the UniPC scheduler.
        self.sigmas = torch.from_numpy(sigmas).to(torch.float32)
        self.timesteps = (self.sigmas[:-1] * num_train_timesteps).to(device)

    def index_for_timestep(self, timestep) -> int:
        t = float(timestep)
        return int(torch.argmin((self.timesteps.cpu() - t).abs()).item())

    def step(self, model_output, timestep, sample, return_dict: bool = False, generator=None):
        """`sample + (sigma_next - sigma) * v` -- the mentor's
        `x - v * delta_step`, with `delta_step = sigma - sigma_next > 0`."""
        i = self.index_for_timestep(timestep)
        d = float(self.sigmas[i + 1] - self.sigmas[i])
        prev = (sample.float() + d * model_output.float()).to(sample.dtype)
        return (prev,) if not return_dict else {"prev_sample": prev}

    def inversion_sigmas(self, start_idx: int) -> torch.Tensor:
        """Ascending grid from 0 to `sigmas[start_idx]`: the denoise path from
        `start_idx`, walked backwards, so inversion and denoise share nodes."""
        return self.sigmas[start_idx:].flip(0)


def euler_invert(
    velocity_fn: Callable[[torch.Tensor, float], torch.Tensor],
    z: torch.Tensor,
    sigmas_up: torch.Tensor,
    mask2: torch.Tensor | None = None,
    z_hold: torch.Tensor | None = None,
    fixed_point_iters: int = 0,
    on_step: Callable[[int, float, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Integrate `dx/dsigma = v` upward from `z` along `sigmas_up`.

    `sigmas_up` must start at 0 and be increasing. `mask2` / `z_hold` play
    the same role as in the denoising loop: mask-0 positions are re-held at
    `z_hold` after every step (and the caller's `velocity_fn` gives them a
    per-token timestep of 0), so conditioning frames stay clean context
    throughout, exactly as they will during the denoise that follows.

    With `fixed_point_iters = K > 0` each step is refined K times with
    `x_next <- x + d * v(x_next, sigma_next)`, costing K extra model calls per
    step but making the step the exact inverse of `EulerFlowScheduler.step`
    once converged.
    """
    if sigmas_up.ndim != 1 or len(sigmas_up) < 2:
        raise ValueError("sigmas_up must be a 1-D grid with at least two nodes")
    if float(sigmas_up[0]) != 0.0 or not bool((sigmas_up[1:] > sigmas_up[:-1]).all()):
        raise ValueError("sigmas_up must start at 0 and be strictly increasing")
    if fixed_point_iters < 0:
        raise ValueError(f"fixed_point_iters must be >= 0, got {fixed_point_iters}")
    if (mask2 is None) != (z_hold is None):
        raise ValueError("mask2 and z_hold must be given together")

    def hold(x):
        return x if mask2 is None else (1.0 - mask2) * z_hold + mask2 * x

    x = hold(z.clone())
    for i in range(len(sigmas_up) - 1):
        s, s_next = float(sigmas_up[i]), float(sigmas_up[i + 1])
        d = s_next - s
        x_next = hold(x + d * velocity_fn(x, s))
        for _ in range(fixed_point_iters):
            x_next = hold(x + d * velocity_fn(x_next, s_next))
        x = x_next
        if on_step is not None:
            on_step(i, s_next, x)
    return x


def _channel_stats(x_frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std of one `(C, H, W)` latent frame, as `(C, 1, 1)`."""
    flat = x_frame.float().flatten(1)
    return flat.mean(1)[:, None, None], flat.std(1)[:, None, None]


def shift_inverted_region(
    x_inv: torch.Tensor,
    plan: list[LatentShift],
    *,
    source: str = "self",
    anchor_idx: int | None = None,
    vacated: str = "noise",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Move a box inside the inverted latent, noise realisation and all.

    `plan` is `frame_mapping.trajectory_to_latent_plan`'s output -- the same
    `--object_box/--move_to/--object_traj` inputs Method G uses. Per entry:

      * the box at `shift.src` is copied to `shift.dst`, feathered by
        `shift.weight`. `source="self"` reads each frame's *own* inverted
        content at the source box, so the object keeps its own motion and is
        offset; `source="anchor"` reads the anchor frame for every frame, as
        Method G does (the anchor must have been inverted too -- see
        `StaticPasteEditor.edit_from_inversion`);
      * cells the box moved out of (`shift.vacated`) are refilled with fresh
        Gaussian noise matched to that frame's per-channel mean/std
        (`vacated="noise"`), or left alone (`vacated="keep"`, the ghost
        control). The fill is variance-preserving -- `sqrt(1-v)` old,
        `sqrt(v)` new -- since the two are independent by construction, so the
        feathered border does not open a low-variance ring.

    Unlike Method G's `write_delta_trajectory`, nothing here is clean content
    written into a noisy latent: at high sigma the inverted latent is close to
    Gaussian, so the moved box is still something the model could have been
    handed as input. Copying the noise along with it is deliberate -- the
    README's warning about duplicated eps is about the same noise at two
    *temporal* positions; this is a spatial move within one frame.

    Every read is from the unmodified `x_inv`, so overlapping source and
    destination boxes do not chain.
    """
    if source not in ("self", "anchor"):
        raise ValueError(f"source must be 'self' or 'anchor', got {source!r}")
    if vacated not in ("noise", "keep"):
        raise ValueError(f"vacated must be 'noise' or 'keep', got {vacated!r}")
    if source == "anchor" and anchor_idx is None:
        raise ValueError("source='anchor' needs anchor_idx")

    out = x_inv.clone()
    for s in plan:
        f = s.frame
        if vacated == "noise":
            v = torch.from_numpy(np.ascontiguousarray(s.vacated)).to(device=x_inv.device, dtype=torch.float32)
            if bool((v > 0).any()):
                mean, std = _channel_stats(x_inv[:, f])
                fresh = torch.randn(
                    x_inv[:, f].shape, generator=generator, dtype=torch.float32,
                    device=generator.device if generator is not None else x_inv.device,
                ).to(x_inv.device) * std + mean
                old = out[:, f].float()
                mixed = mean + (1.0 - v).sqrt() * (old - mean) + v.sqrt() * (fresh - mean)
                out[:, f] = mixed.to(out.dtype)

        sy1, sy2, sx1, sx2 = s.src
        dy1, dy2, dx1, dx2 = s.dst
        src_frame = f if source == "self" else anchor_idx
        src = x_inv[:, src_frame, sy1:sy2, sx1:sx2]
        w = torch.from_numpy(np.ascontiguousarray(s.weight)).to(device=x_inv.device, dtype=x_inv.dtype)
        dst = out[:, f, dy1:dy2, dx1:dx2]
        out[:, f, dy1:dy2, dx1:dx2] = (1.0 - w) * dst + w * src
    return out


def box_weight(
    latent_boxes: list[tuple[int, int, int, int]],
    latent_h: int,
    latent_w: int,
    feather: int,
) -> np.ndarray:
    """Union of feathered `(x1, y1, x2, y2)` latent-cell boxes, `(H, W)` in [0, 1]."""
    weight = np.zeros((latent_h, latent_w), dtype=np.float32)
    for lbox in latent_boxes:
        weight = np.maximum(weight, feathered_map_from_latent_box(latent_h, latent_w, lbox, feather))
    return weight


def copy_region_across_frames(
    x: torch.Tensor,
    weight_hw: torch.Tensor,
    src_frame: int,
    target_frames: list[int],
) -> torch.Tensor:
    """Blend frame `src_frame`'s content into `target_frames` under `weight_hw`.

    The noise-space freeze (`--freeze_mode noise`). Applied once to the
    inverted latent with the anchor as source, it hands every target frame the
    anchor's box *including its noise*, which the model reads as "this region
    is the same in every frame". Re-applied after each of the first few
    denoise steps (source: the first target frame), it keeps the box identical
    across frames at the *current* noise level -- rather than clamping to clean
    content with a timestep of 0 as the hard freeze does, which could not be
    released mid-schedule without handing the model clean content labelled as
    noisy. Once released, the remaining steps are free to blend the box edge
    into its surroundings.

    A linear blend, not a variance-preserving one: after the first copy the
    frames are strongly correlated inside the box, and `sqrt` weights would
    inflate correlated values.
    """
    targets = [t for t in target_frames if t != src_frame]
    if not targets:
        return x
    out = x.clone()
    idx = torch.tensor(targets, device=x.device, dtype=torch.long)
    w = weight_hw.to(device=x.device, dtype=x.dtype)
    src = x[:, src_frame].unsqueeze(1)
    out[:, idx] = (1.0 - w) * out[:, idx] + w * src
    return out
