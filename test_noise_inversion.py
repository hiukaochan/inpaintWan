"""Local, model-free unit tests for `noise_inversion.py`.

The model is replaced by toy velocity fields whose behaviour under Euler is
known in closed form, so these pin down the solver itself: that the sigma
grid is Wan's, that invert-then-denoise round-trips, that held positions
never move, and that the region moves land where the plan says.

Runs on CPU with no checkpoints. Run with: python test_noise_inversion.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "Wan2.2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frame_mapping import LatentShift
from noise_inversion import (
    EulerFlowScheduler,
    box_weight,
    copy_region_across_frames,
    euler_invert,
    shift_inverted_region,
    shifted_sigmas,
)

C, T, H, W = 4, 5, 16, 16


def _rand(seed=0, shape=(C, T, H, W)):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _denoise(sched, velocity_fn, x, start_idx, mask2=None, z_hold=None):
    """The editor's loop, minus the model: what `_denoise` does per step."""
    for t in sched.timesteps[start_idx:]:
        i = sched.index_for_timestep(t)
        v = velocity_fn(x, float(sched.sigmas[i]))
        x = sched.step(v, t, x)[0]
        if mask2 is not None:
            x = (1.0 - mask2) * z_hold + mask2 * x
    return x


def test_sigma_grid_is_wans():
    sched = EulerFlowScheduler(20, 5.0)
    assert len(sched.sigmas) == 21 and len(sched.timesteps) == 20
    assert float(sched.sigmas[0]) == 1.0 and float(sched.sigmas[-1]) == 0.0
    assert bool((sched.sigmas[1:] < sched.sigmas[:-1]).all())
    assert torch.allclose(sched.timesteps, sched.sigmas[:-1] * 1000)
    try:
        from wan.utils.fm_solvers import get_sampling_sigmas
    except Exception as e:  # vendored wan needs easydict/diffusers
        print(f"    (skipped comparison with get_sampling_sigmas: {type(e).__name__})")
        return
    assert np.allclose(shifted_sigmas(20, 5.0), get_sampling_sigmas(20, 5.0))


def test_step_is_mentors_equation():
    """x_next = x - v * delta_step, delta_step = sigma - sigma_next."""
    sched = EulerFlowScheduler(10, 5.0)
    x, v = _rand(1), _rand(2)
    i = 3
    out = sched.step(v, sched.timesteps[i], x)[0]
    delta = float(sched.sigmas[i] - sched.sigmas[i + 1])
    assert delta > 0
    assert torch.allclose(out, x - v * delta, atol=1e-6)


def test_inversion_grid_mirrors_denoise():
    sched = EulerFlowScheduler(20, 5.0)
    up = sched.inversion_sigmas(4)
    assert float(up[0]) == 0.0 and float(up[-1]) == float(sched.sigmas[4])
    assert len(up) == len(sched.timesteps[4:]) + 1


def test_constant_field_round_trips_exactly():
    """Rectified flow on a single data point: v = eps - z0, constant along the
    path, so Euler is exact both ways and the inverted latent is the true
    forward-noised one."""
    z0, eps = _rand(3), _rand(4)
    v_true = eps - z0
    sched = EulerFlowScheduler(20, 5.0)
    start = 2
    x_inv = euler_invert(lambda x, s: v_true, z0, sched.inversion_sigmas(start))
    s0 = float(sched.sigmas[start])
    assert torch.allclose(x_inv, (1 - s0) * z0 + s0 * eps, atol=1e-5)
    back = _denoise(sched, lambda x, s: v_true, x_inv, start)
    assert torch.allclose(back, z0, atol=1e-5)


def test_fixed_point_closes_the_round_trip_gap():
    """With a state-dependent field, plain Euler inversion does not exactly
    undo Euler denoising; fixed-point refinement makes it the exact inverse."""
    z0 = _rand(5)
    fn = lambda x, s: 0.8 * torch.tanh(x) * (1.0 + s)  # noqa: E731

    def err(steps, iters):
        sched = EulerFlowScheduler(steps, 5.0)
        x_inv = euler_invert(fn, z0, sched.inversion_sigmas(1), fixed_point_iters=iters)
        return (_denoise(sched, fn, x_inv, 1) - z0).abs().max().item()

    plain_10, plain_40 = err(10, 0), err(40, 0)
    fp_10 = err(10, 20)
    assert plain_40 < plain_10, (plain_10, plain_40)
    assert fp_10 < 1e-3 and fp_10 < 0.05 * plain_10, (fp_10, plain_10)


def test_held_positions_never_move():
    z = _rand(6)
    z_hold = _rand(7)
    mask2 = torch.ones_like(z)
    mask2[:, 0] = 0.0  # conditioning frame
    mask2[:, 2, 4:8, 4:8] = 0.0  # a frozen box
    sched = EulerFlowScheduler(10, 5.0)
    x_inv = euler_invert(lambda x, s: x + 1.0, z, sched.inversion_sigmas(0),
                         mask2=mask2, z_hold=z_hold, fixed_point_iters=2)
    held = mask2 == 0
    assert torch.equal(x_inv[held], z_hold[held])
    assert not torch.allclose(x_inv[~held], z[~held])


def _shift(frame=2, src=(2, 6, 2, 6), dst=(8, 12, 2, 6)):
    vac = np.zeros((H, W), dtype=np.float32)
    vac[src[0]:src[1], src[2]:src[3]] = 1.0
    vac[dst[0]:dst[1], dst[2]:dst[3]] = 0.0
    return LatentShift(frame=frame, src=src, dst=dst, weight=np.ones((4, 4), dtype=np.float32),
                       vacated=vac, cell_shift=(dst[0] - src[0], 0))


def test_shift_self_moves_own_content_and_keeps_rest():
    x = _rand(8)
    out = shift_inverted_region(x, [_shift()], source="self", vacated="keep")
    assert torch.equal(out[:, 2, 8:12, 2:6], x[:, 2, 2:6, 2:6])
    assert torch.equal(out[:, 2, 2:6, 2:6], x[:, 2, 2:6, 2:6])  # 'keep' leaves the ghost
    others = [f for f in range(T) if f != 2]
    assert torch.equal(out[:, others], x[:, others])


def test_shift_anchor_reads_anchor_frame():
    x = _rand(9)
    out = shift_inverted_region(x, [_shift()], source="anchor", anchor_idx=0, vacated="keep")
    assert torch.equal(out[:, 2, 8:12, 2:6], x[:, 0, 2:6, 2:6])


def test_vacated_noise_is_fresh_and_matches_stats():
    x = _rand(10, shape=(48, T, 32, 32)) * 1.7 + 0.3
    src, dst = (0, 16, 0, 16), (16, 32, 0, 16)
    vac = np.zeros((32, 32), dtype=np.float32)
    vac[0:16, 0:16] = 1.0
    shift = LatentShift(frame=1, src=src, dst=dst, weight=np.ones((16, 16), dtype=np.float32),
                        vacated=vac, cell_shift=(16, 0))
    out = shift_inverted_region(x, [shift], vacated="noise", generator=torch.Generator().manual_seed(0))
    filled = out[:, 1, 0:16, 0:16]
    assert not torch.allclose(filled, x[:, 1, 0:16, 0:16])
    assert abs(filled.std().item() / x[:, 1].std().item() - 1.0) < 0.1
    assert abs(filled.mean().item() - x[:, 1].mean().item()) < 0.1
    assert torch.equal(out[:, 1, 16:32, 0:16], x[:, 1, 0:16, 0:16])  # copy still lands


def test_copy_region_across_frames():
    x = _rand(11)
    w = torch.from_numpy(box_weight([(4, 4, 10, 10)], H, W, feather=0))
    out = copy_region_across_frames(x, w, src_frame=1, target_frames=[1, 2, 3])
    for f in (2, 3):
        assert torch.equal(out[:, f, 4:10, 4:10], x[:, 1, 4:10, 4:10])
        assert torch.equal(out[:, f, :4], x[:, f, :4])  # outside the box untouched
    assert torch.equal(out[:, [0, 1, 4]], x[:, [0, 1, 4]])


def test_box_weight_feathers_inward():
    w = box_weight([(2, 2, 12, 12)], H, W, feather=2)
    assert w[7, 7] == 1.0 and w[0, 0] == 0.0
    assert 0.0 < w[2, 7] < 1.0  # border cell ramps


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(tests)} passed")
