"""Local, model-free unit tests for latent_paste.py.

Runs on CPU with no checkpoints. Run with: python test_latent_paste.py
"""

import numpy as np
import torch

from frame_mapping import build_feathered_box_weight
from latent_paste import (
    butterworth_low_pass_filter,
    fft_restore,
    target_latent_frames,
    weight_to_tensor,
    write_delta,
)

C, T, H, W = 4, 6, 8, 10
SIGMA = 0.8


def _fixture(drift: float = 3.0):
    """A latent built exactly the way the sampler builds one --
    x = (1-sigma) z + sigma eps.

    The clean content `z` is a *static* scene (every frame identical), which
    is the situation being corrected toward; `drift` then offsets the boxed
    region of every frame after the anchor, standing in for the cabinet
    sliding away from where it should be. `drift=0` therefore means "already
    static", and the correction must do nothing at all.
    """
    g = torch.Generator().manual_seed(0)
    static_frame = torch.randn(C, 1, H, W, generator=g)
    z = static_frame.repeat(1, T, 1, 1)
    eps = torch.randn(C, T, H, W, generator=g)
    box = (slice(None), slice(1, None), slice(2, 6), slice(3, 7))
    z[box] = z[box] + drift  # frames 1.. drift away from the anchor inside the box
    latent = (1.0 - SIGMA) * z + SIGMA * eps
    return z, eps, latent


def _box_weight():
    # pixel box -> latent cells [2:6) rows, [3:7) cols on an 8x10 grid
    w = build_feathered_box_weight([(48, 32, 112, 96)], latent_h=H, latent_w=W, feather=0)
    assert w[2:6, 3:7].all() and w.sum() == 16
    return weight_to_tensor(w, "cpu", torch.float32)


def test_target_latent_frames_skips_frozen_and_anchor():
    mask = [False, True, True, True, False]
    assert target_latent_frames(mask, anchor_idx=0) == [1, 2, 3]


def test_target_latent_frames_handles_missing_anchor():
    mask = [True, True, False]
    assert target_latent_frames(mask, anchor_idx=None) == [0, 1]


def test_write_delta_swaps_content_and_keeps_destination_noise():
    z, eps, latent = _fixture()
    out = write_delta(latent, z, _box_weight(), anchor_idx=0, target_frames=[1, 2, 3], sigma=SIGMA)

    # Inside the box, the result must be the anchor's content carried at the
    # destination's own noise -- that is the whole point of the delta form.
    for t in (1, 2, 3):
        expected = (1.0 - SIGMA) * z[:, 0, 2:6, 3:7] + SIGMA * eps[:, t, 2:6, 3:7]
        assert torch.allclose(out[:, t, 2:6, 3:7], expected, atol=1e-5), f"frame {t}"


def test_write_delta_leaves_everything_outside_the_box_untouched():
    z, _, latent = _fixture()
    out = write_delta(latent, z, _box_weight(), anchor_idx=0, target_frames=[1, 2, 3], sigma=SIGMA)
    outside = torch.ones(H, W, dtype=torch.bool)
    outside[2:6, 3:7] = False
    assert torch.equal(out[:, :, outside], latent[:, :, outside])


def test_write_delta_leaves_untargeted_frames_untouched():
    z, _, latent = _fixture()
    out = write_delta(latent, z, _box_weight(), anchor_idx=0, target_frames=[1, 2], sigma=SIGMA)
    for t in (0, 3, 4, 5):
        assert torch.equal(out[:, t], latent[:, t]), f"frame {t} should not have been written"


def test_write_delta_is_a_noop_where_there_is_no_drift():
    # The correction is proportional to (z[anchor] - z[t]), so a region that
    # already matches the anchor must come back bit-identical.
    z, _, latent = _fixture(drift=0.0)
    out = write_delta(latent, z, _box_weight(), anchor_idx=0, target_frames=[1, 2, 3], sigma=SIGMA)
    assert torch.allclose(out, latent, atol=1e-6)


def test_write_delta_strength_scales_the_correction_linearly():
    z, _, latent = _fixture()
    w = _box_weight()
    full = write_delta(latent, z, w, 0, [1, 2, 3], SIGMA, strength=1.0)
    half = write_delta(latent, z, w, 0, [1, 2, 3], SIGMA, strength=0.5)
    assert torch.allclose(half - latent, (full - latent) * 0.5, atol=1e-6)


def test_write_delta_zero_strength_and_empty_targets_are_noops():
    z, _, latent = _fixture()
    w = _box_weight()
    assert torch.allclose(write_delta(latent, z, w, 0, [1, 2, 3], SIGMA, strength=0.0), latent, atol=1e-6)
    assert torch.equal(write_delta(latent, z, w, 0, [], SIGMA), latent)


def test_write_delta_does_not_mutate_its_input():
    z, _, latent = _fixture()
    before = latent.clone()
    write_delta(latent, z, _box_weight(), 0, [1, 2, 3], SIGMA)
    assert torch.equal(latent, before)


def test_write_delta_feathered_weight_gives_partial_correction():
    z, _, latent = _fixture()
    w = weight_to_tensor(
        build_feathered_box_weight([(48, 32, 112, 96)], latent_h=H, latent_w=W, feather=1),
        "cpu", torch.float32)
    out = write_delta(latent, z, w, 0, [1, 2, 3], SIGMA)
    changed = (out - latent).abs().amax(dim=(0, 1))
    edge, interior = changed[2, 3], changed[3, 4]
    assert 0 < edge < interior, "feathered border must be corrected less than the interior"


def test_write_delta_rejects_shape_mismatches():
    z, _, latent = _fixture()
    for bad in [
        lambda: write_delta(latent, z[:, :, :4], _box_weight(), 0, [1], SIGMA),
        lambda: write_delta(latent, z, torch.ones(3, 3), 0, [1], SIGMA),
        lambda: write_delta(latent, z, _box_weight(), 0, [1], SIGMA, strength=1.5),
    ]:
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_butterworth_is_low_pass_and_bounded():
    lpf = butterworth_low_pass_filter((16, 16), d_s=0.5)
    assert lpf.shape == (16, 16)
    assert 0.0 <= lpf.min() and lpf.max() <= 1.0
    # fftshift puts DC at the centre, so the centre must pass and corners must not
    assert lpf[8, 8] > 0.99
    assert lpf[0, 0] < 0.05
    assert torch.equal(butterworth_low_pass_filter((8, 8), d_s=0.0), torch.zeros(8, 8))


def test_fft_restore_endpoints():
    g = torch.Generator().manual_seed(1)
    edited = torch.randn(C, T, H, W, generator=g)
    original = torch.randn(C, T, H, W, generator=g)

    # d_s = 0 -> the low-pass mask is empty, so nothing of `edited` survives
    assert torch.allclose(fft_restore(edited, original, d_s=0.0), original, atol=1e-4)
    # a very wide cut-off keeps essentially all of `edited`
    assert torch.allclose(fft_restore(edited, original, d_s=50.0), edited, atol=1e-3)


def test_fft_restore_is_a_noop_when_both_inputs_agree():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(C, T, H, W, generator=g)
    assert torch.allclose(fft_restore(x, x, d_s=0.5), x, atol=1e-4)


def test_fft_restore_keeps_the_low_frequency_content_of_edited():
    # A constant offset is pure DC, i.e. the lowest frequency there is, so it
    # must survive the restore intact.
    g = torch.Generator().manual_seed(3)
    original = torch.randn(C, T, H, W, generator=g)
    edited = original + 5.0
    out = fft_restore(edited, original, d_s=0.5)
    assert abs(out.mean().item() - edited.mean().item()) < 1e-3


def test_fft_restore_rejects_shape_mismatch():
    try:
        fft_restore(torch.zeros(C, T, H, W), torch.zeros(C, T, H, W + 1), d_s=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched shapes")


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} tests passed")
