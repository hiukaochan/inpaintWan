"""Local, model-free unit tests for latent_paste.py.

Runs on CPU with no checkpoints. Run with: python test_latent_paste.py
"""

import numpy as np
import torch

from frame_mapping import LatentShift, build_feathered_box_weight, feathered_map_from_latent_box
from latent_paste import (
    butterworth_low_pass_filter,
    fft_restore,
    soften_toward_noise,
    target_latent_frames,
    weight_to_tensor,
    write_delta,
    write_delta_trajectory,
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


def _shift(frame, src, dst, feather=0):
    """Build one LatentShift by hand. `src`/`dst` are (y1, y2, x1, x2)."""
    sy1, sy2, sx1, sx2 = src
    dy1, dy2, dx1, dx2 = dst
    assert (sy2 - sy1, sx2 - sx1) == (dy2 - dy1, dx2 - dx1)
    weight = np.ones((sy2 - sy1, sx2 - sx1), dtype=np.float32)
    src_map = feathered_map_from_latent_box(H, W, (sx1, sy1, sx2, sy2), feather)
    dst_map = feathered_map_from_latent_box(H, W, (dx1, dy1, dx2, dy2), feather)
    return LatentShift(frame, src, dst, weight,
                       np.clip(src_map - dst_map, 0.0, 1.0), (dy1 - sy1, dx1 - sx1))


def _traj_fixture():
    """z with per-frame content, latent built as (1-s)z + s*eps."""
    g = torch.Generator().manual_seed(7)
    z = torch.randn(C, T, H, W, generator=g)
    eps = torch.randn(C, T, H, W, generator=g)
    return z, eps, (1.0 - SIGMA) * z + SIGMA * eps


SRC = (2, 5, 1, 4)          # y 2:5, x 1:4
DST_DISJOINT = (3, 6, 4, 7)  # shifted (+1, +3) -- no overlap in x


def test_trajectory_moves_content_to_the_destination_keeping_its_noise():
    z, eps, latent = _traj_fixture()
    plan = [_shift(t, SRC, DST_DISJOINT) for t in (1, 2)]
    out = write_delta_trajectory(latent, z, anchor_idx=0, plan=plan, sigma=SIGMA, vacated_fill=0.0)

    for t in (1, 2):
        expected = (1.0 - SIGMA) * z[:, 0, 2:5, 1:4] + SIGMA * eps[:, t, 3:6, 4:7]
        assert torch.allclose(out[:, t, 3:6, 4:7], expected, atol=1e-5), f"frame {t}"


def test_trajectory_softening_removes_the_content_left_behind():
    z, eps, latent = _traj_fixture()
    plan = [_shift(1, SRC, DST_DISJOINT)]
    out = write_delta_trajectory(latent, z, 0, plan, SIGMA, vacated_fill=1.0)
    # the vacated source box keeps its own noise but loses its content entirely
    assert torch.allclose(out[:, 1, 2:5, 1:4], SIGMA * eps[:, 1, 2:5, 1:4], atol=1e-5)


def test_trajectory_vacated_fill_zero_leaves_the_old_position_untouched():
    z, _, latent = _traj_fixture()
    plan = [_shift(1, SRC, DST_DISJOINT)]
    out = write_delta_trajectory(latent, z, 0, plan, SIGMA, vacated_fill=0.0)
    assert torch.equal(out[:, 1, 2:5, 1:4], latent[:, 1, 2:5, 1:4])


def test_trajectory_partial_overlap_only_vacates_the_uncovered_strip():
    z, eps, latent = _traj_fixture()
    dst_overlap = (2, 5, 3, 6)          # shifted (+0, +2): x 3:4 still covered
    plan = [_shift(1, SRC, dst_overlap)]
    out = write_delta_trajectory(latent, z, 0, plan, SIGMA, vacated_fill=1.0)
    # x 1:3 vacated -> content stripped; x 3:4 is inside the destination -> not stripped
    assert torch.allclose(out[:, 1, 2:5, 1:3], SIGMA * eps[:, 1, 2:5, 1:3], atol=1e-5)
    assert not torch.allclose(out[:, 1, 2:5, 3:4], SIGMA * eps[:, 1, 2:5, 3:4], atol=1e-3)


def test_trajectory_leaves_other_frames_and_cells_alone():
    z, _, latent = _traj_fixture()
    plan = [_shift(1, SRC, DST_DISJOINT)]
    out = write_delta_trajectory(latent, z, 0, plan, SIGMA, vacated_fill=1.0)
    for t in (0, 2, 3, 4, 5):
        assert torch.equal(out[:, t], latent[:, t]), f"frame {t} should be untouched"
    touched = torch.zeros(H, W, dtype=torch.bool)
    touched[2:5, 1:4] = True   # vacated
    touched[3:6, 4:7] = True   # destination
    assert torch.equal(out[:, 1, ~touched], latent[:, 1, ~touched])


def test_trajectory_strength_scales_the_copy():
    z, _, latent = _traj_fixture()
    plan = [_shift(1, SRC, DST_DISJOINT)]
    full = write_delta_trajectory(latent, z, 0, plan, SIGMA, strength=1.0, vacated_fill=0.0)
    half = write_delta_trajectory(latent, z, 0, plan, SIGMA, strength=0.5, vacated_fill=0.0)
    assert torch.allclose(half - latent, (full - latent) * 0.5, atol=1e-6)


def test_trajectory_is_deterministic_and_does_not_mutate_input():
    z, _, latent = _traj_fixture()
    before = latent.clone()
    plan = [_shift(1, SRC, DST_DISJOINT)]
    a = write_delta_trajectory(latent, z, 0, plan, SIGMA)
    b = write_delta_trajectory(latent, z, 0, plan, SIGMA)
    assert torch.equal(a, b), "no RNG is involved -- repeated calls must agree exactly"
    assert torch.equal(latent, before)


def test_trajectory_empty_plan_and_bad_inputs():
    z, _, latent = _traj_fixture()
    assert torch.equal(write_delta_trajectory(latent, z, 0, [], SIGMA), latent)
    for bad in [
        lambda: write_delta_trajectory(latent, z[:, :3], 0, [_shift(1, SRC, DST_DISJOINT)], SIGMA),
        lambda: write_delta_trajectory(latent, z, 0, [_shift(1, SRC, DST_DISJOINT)], SIGMA, strength=2.0),
        lambda: write_delta_trajectory(latent, z, 0, [_shift(99, SRC, DST_DISJOINT)], SIGMA),
    ]:
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_soften_toward_noise_drives_content_to_zero():
    z, eps, latent = _traj_fixture()
    vacated = torch.zeros(H, W)
    vacated[2:5, 1:4] = 1.0
    out = soften_toward_noise(latent.clone(), z, 1, vacated, SIGMA, fill=1.0)
    assert torch.allclose(out[:, 1, 2:5, 1:4], SIGMA * eps[:, 1, 2:5, 1:4], atol=1e-5)
    # half fill removes half the content term
    half = soften_toward_noise(latent.clone(), z, 1, vacated, SIGMA, fill=0.5)
    expected = latent[:, 1, 2:5, 1:4] - 0.5 * (1 - SIGMA) * z[:, 1, 2:5, 1:4]
    assert torch.allclose(half[:, 1, 2:5, 1:4], expected, atol=1e-6)


def test_soften_toward_noise_zero_fill_is_a_noop_and_rejects_bad_fill():
    z, _, latent = _traj_fixture()
    vacated = torch.ones(H, W)
    assert torch.equal(soften_toward_noise(latent.clone(), z, 1, vacated, SIGMA, fill=0.0), latent)
    try:
        soften_toward_noise(latent.clone(), z, 1, vacated, SIGMA, fill=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for fill outside [0, 1]")


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
