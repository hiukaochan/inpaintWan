"""Local, model-free unit tests for the I2V conditioning used in training.

A silent mismatch between how the conditioning frame is presented at training
time and at inference time is the classic way a video fine-tune degrades
without ever showing a bad loss curve -- the loss stays plausible because the
model is learning a task that is merely *adjacent* to the one it will be asked
to do. So these tests pin training against the real thing:
`WanTI2V.i2v()` in Wan2.2/wan/textimage2video.py, whose construction is
reproduced verbatim here and compared element for element.

The three pieces that have to agree exactly:

  * `build_per_token_timestep` -- *which tokens* are told they are clean
  * `add_noise` (via mask1)    -- *how clean* the conditioning frame actually is
  * `build_loss_mask`          -- *where* the loss is allowed to look

If the timestep mask said frame 0 while the noise mask said frame 1, every
sample would train the model to denoise a frame it was told was already clean.

Runs on CPU with no checkpoints. Run with: python test_conditioning.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "Wan2.2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.conditioning import (
    add_noise,
    build_loss_mask,
    build_per_token_timestep,
    sample_sigma,
    velocity_target,
)
from wan.utils.utils import masks_like

# 640x352 @ 49 frames: the primary training config
C, T, H, W = 48, 13, 22, 40
SEQ_LEN = T * (H // 2) * (W // 2)
TOKENS_PER_FRAME = (H // 2) * (W // 2)


def _rand(seed=0, scale=1.0):
    return torch.randn(C, T, H, W, generator=torch.Generator().manual_seed(seed)) * scale


def test_seq_len_matches_latent_shape():
    """The token count the DiT sees, from the cached latent shape."""
    assert SEQ_LEN == 2860, SEQ_LEN


def test_per_token_timestep_matches_inference():
    """Reproduce textimage2video.py:i2v's timestep construction line for line."""
    noise = _rand()
    _, mask2 = masks_like([noise], zero=True)
    t = torch.tensor([731.0])

    # verbatim from WanTI2V.i2v()
    expected = (mask2[0][0][:, ::2, ::2] * t).flatten()
    expected = torch.cat([expected, expected.new_ones(SEQ_LEN - expected.size(0)) * t])
    expected = expected.unsqueeze(0)

    got = build_per_token_timestep(mask2[0], t, SEQ_LEN)

    assert got.shape == (1, SEQ_LEN), got.shape
    assert torch.equal(got, expected)


def test_conditioning_frame_tokens_carry_zero_timestep():
    noise = _rand()
    _, mask2 = masks_like([noise], zero=True)
    ts = build_per_token_timestep(mask2[0], torch.tensor([500.0]), SEQ_LEN)[0]

    assert torch.all(ts[:TOKENS_PER_FRAME] == 0), "frame 0 tokens must carry t=0"
    assert torch.all(ts[TOKENS_PER_FRAME:] == 500.0), "all later tokens carry t"


def test_add_noise_reduces_to_inference_paste():
    """With mask1[:,0]=0, add_noise must equal i2v's `(1-mask2)*z0 + mask2*noisy`."""
    z0, noise = _rand(0), _rand(1, 0.5)
    mask1, mask2 = masks_like([noise], zero=True)  # zero=True -> mask1[:,0]=0
    sigma = torch.tensor(0.3)

    got = add_noise(z0, noise, sigma, mask1[0])

    plain_noisy = (1.0 - sigma) * z0 + sigma * noise
    expected = (1.0 - mask2[0]) * z0 + mask2[0] * plain_noisy

    assert torch.allclose(got, expected, atol=1e-6)
    assert torch.allclose(got[:, 0], z0[:, 0], atol=1e-6), "frame 0 must stay clean"


def test_noise_augmented_conditioning_frame_stays_nearly_clean():
    """The p-branch of masks_like perturbs frame 0 slightly, not fully.

    mask1[:,0] = exp(N(-3.5,0.5)) ~ 0.03 is what makes the model tolerate the
    VAE reconstruction error of a real first frame at inference.
    """
    z0, noise = _rand(0), _rand(1, 0.5)
    mask1 = torch.ones_like(noise)
    mask1[:, 0] = 0.03
    sigma = torch.tensor(1.0)

    got = add_noise(z0, noise, sigma, mask1)
    drift = (got[:, 0] - z0[:, 0]).abs().mean()
    full = (noise[:, 0] - z0[:, 0]).abs().mean()

    assert drift < 0.05 * full, f"frame 0 drifted too far: {drift} vs {full}"
    assert torch.allclose(got[:, 1:], noise[:, 1:], atol=1e-6), "other frames fully noised"


def test_unconditioned_sample_is_plain_t2v():
    """When masks_like leaves frame 0 at 1, nothing is special-cased."""
    z0, noise = _rand(0), _rand(1, 0.5)
    mask1 = torch.ones_like(noise)
    mask2 = torch.ones_like(noise)
    sigma = torch.tensor(0.4)

    got = add_noise(z0, noise, sigma, mask1)
    assert torch.allclose(got, (1.0 - sigma) * z0 + sigma * noise, atol=1e-6)

    ts = build_per_token_timestep(mask2, torch.tensor([400.0]), SEQ_LEN)
    assert torch.all(ts == 400.0)
    assert torch.all(build_loss_mask(mask2) == 1.0)


def test_loss_mask_excludes_only_the_conditioning_frame():
    noise = _rand()
    _, mask2 = masks_like([noise], zero=True)
    mask = build_loss_mask(mask2[0])
    assert torch.all(mask[:, 0] == 0.0)
    assert torch.all(mask[:, 1:] == 1.0)


def test_masks_like_p_branch_agrees_between_the_two_masks():
    """Whenever mask2 zeroes frame 0, mask1 must also mark it near-clean.

    These are sampled by the same coin flip inside masks_like; if they ever
    disagreed, a sample would be told t=0 on a fully noised frame.
    """
    g = torch.Generator().manual_seed(3)
    noise = _rand()
    conditioned = 0
    for _ in range(200):
        mask1, mask2 = masks_like([noise], zero=True, generator=g, p=0.5)
        m1_zeroed = mask1[0][0, 0, 0, 0].item() < 0.5
        m2_zeroed = mask2[0][0, 0, 0, 0].item() == 0.0
        assert m1_zeroed == m2_zeroed, "mask1 and mask2 disagree about frame 0"
        conditioned += m2_zeroed
    assert 60 < conditioned < 140, f"p=0.5 should split roughly evenly, got {conditioned}/200"


def test_velocity_target_is_recoverable_from_the_interpolation():
    """x_t + (1-sigma)*target == noise, i.e. the target really is d x_t/d sigma."""
    z0, noise = _rand(0), _rand(1, 0.5)
    sigma = torch.tensor(0.37)
    x_t = (1.0 - sigma) * z0 + sigma * noise
    target = velocity_target(z0, noise)
    assert torch.allclose(x_t + (1.0 - sigma) * target, noise, atol=1e-5)


def test_sample_sigma_is_in_range_and_shifted():
    for shift in (3.0, 5.0):
        g = torch.Generator().manual_seed(7)
        vals = torch.cat([sample_sigma(shift, generator=g) for _ in range(2000)])
        assert torch.all(vals > 0) and torch.all(vals < 1)
        # shift>1 pushes mass toward higher noise, so the median exceeds the
        # unshifted logit-normal median of 0.5
        assert vals.median() > 0.5, (shift, vals.median())


def test_sample_sigma_matches_scheduler_shift_formula():
    """Pin the shift convention sample_sigma depends on.

    Training and inference have to agree on what a given noise level means. If
    the scheduler's map (fm_solvers_unipc.py:116) ever changed, sample_sigma
    would quietly be training on a different noise schedule than the sampler
    uses, so this asserts the closed form against the scheduler itself.
    """
    import numpy as np

    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    shift = 5.0
    alphas = np.linspace(1, 1 / 1000, 1000)[::-1].copy()
    base = torch.from_numpy(1.0 - alphas).float()
    expected = shift * base / (1 + (shift - 1) * base)

    sched = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)
    assert torch.allclose(sched.sigmas, expected, atol=1e-5), "scheduler shift map changed"

    # and sample_sigma's draws must land inside that same range
    g = torch.Generator().manual_seed(11)
    vals = torch.cat([sample_sigma(shift, generator=g) for _ in range(500)])
    assert vals.min() >= expected.min() and vals.max() <= expected.max()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(tests)} passed")
