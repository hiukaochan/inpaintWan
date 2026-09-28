"""Flow-matching noising and I2V conditioning for TI2V-5B training.

TI2V-5B does not use a `y`-channel concat for image conditioning -- `in_dim ==
out_dim == 48`. Instead `WanTI2V.i2v()` (Wan2.2/wan/textimage2video.py):

  * hard-pastes the clean first-frame latent into latent frame 0, before and
    after every denoising step:  `latent = (1-mask2)*z + mask2*latent`
  * zeroes the *per-token timestep* on that frame's tokens:
    `temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()`

So training conditioning is expressible with the same masks, and this module
reproduces them exactly. `masks_like()` in Wan2.2/wan/utils/utils.py is already
upstream's training-time sampler (it shipped in the inference repo): given a
generator it picks, per sample, either

  * conditioned  -- mask1[:,0] = exp(N(-3.5, 0.5)) ~ 0.03, mask2[:,0] = 0
  * unconditioned -- both left at 1, i.e. plain t2v

mask1 is unused at inference; it is the noise-level scale that makes the
conditioning frame slightly noisy during training, so the model tolerates the
VAE reconstruction error of a real first frame at inference.

Setting mask1[:,0] = 0 (upstream's `zero=True` inference path) makes
`add_noise` reduce exactly to `latent = (1-mask2)*z0 + mask2*noisy`, which is
what `test_conditioning.py` asserts.
"""

import torch


def sample_sigma(
    shift: float,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
) -> torch.Tensor:
    """Draw one flow-matching sigma in (0,1), logit-normal then shifted.

    The shift matches FlowUniPCMultistepScheduler's construction
    (fm_solvers_unipc.py:116) so training and inference agree on what a given
    noise level means; the scheduler's timesteps are `sigma * 1000`.
    """
    u = torch.randn(1, generator=generator, device=device).mul_(logit_std).add_(logit_mean).sigmoid()
    return shift * u / (1.0 + (shift - 1.0) * u)


def add_noise(
    z0: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
    mask1: torch.Tensor,
) -> torch.Tensor:
    """Rectified-flow interpolation with a per-frame noise scale.

    `sigma * mask1` is the effective noise level per element, so the
    conditioning frame (mask1[:,0] near 0) stays essentially clean while every
    other frame sits at `sigma`.
    """
    sigma_map = sigma * mask1
    return (1.0 - sigma_map) * z0 + sigma_map * noise


def velocity_target(z0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """d x_t / d sigma for x_t = (1-sigma) z0 + sigma noise."""
    return noise - z0


def build_per_token_timestep(mask2: torch.Tensor, t_scalar: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Per-token timesteps, matching `WanTI2V.i2v()` element for element.

    `mask2` is [C, T, h, w]; channel 0 is representative, and the `::2` strides
    mirror the (1,2,2) patchify. Tokens of the conditioning frame get 0, the
    rest get `t_scalar`, then the tail is padded up to `seq_len`.

    Returns [1, seq_len].
    """
    temp_ts = (mask2[0][:, ::2, ::2] * t_scalar).flatten()
    if temp_ts.numel() < seq_len:
        pad = temp_ts.new_ones(seq_len - temp_ts.numel()) * t_scalar
        temp_ts = torch.cat([temp_ts, pad])
    return temp_ts.unsqueeze(0)


def build_loss_mask(mask2: torch.Tensor) -> torch.Tensor:
    """Weight per latent element: 0 on the conditioning frame.

    Its target is meaningless there -- the input is clean z0 and the model is
    told t=0 -- so including it would train the model against noise.
    """
    return mask2
