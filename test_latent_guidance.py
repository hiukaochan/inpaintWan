"""Local, model-free unit tests for latent_guidance.py.

Runs on CPU with no checkpoints and no Wan2.2 weights: `forward_to_block` is
exercised against a mock DiT that mimics `WanModel`'s structure closely enough
to test the parts that are easy to get silently wrong -- early exit, gradient
flow to the latent, and checkpointed-vs-plain equivalence.

Run with: python test_latent_guidance.py
"""

import sys
import types

import numpy as np
import torch
import torch.nn as nn

from frame_mapping import LatentShift, feathered_map_from_latent_box

# latent_guidance imports `sinusoidal_embedding_1d` from wan.modules.model
# inside forward_to_block. Stub just that symbol so the module is importable
# without the Wan2.2 dependency stack.
if "wan.modules.model" not in sys.modules:
    for name in ("wan", "wan.modules", "wan.modules.model"):
        sys.modules.setdefault(name, types.ModuleType(name))

    def _sinusoidal_embedding_1d(dim, position):
        half = dim // 2
        position = position.type(torch.float64)
        sinusoid = torch.outer(
            position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
        return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)

    sys.modules["wan.modules.model"].sinusoidal_embedding_1d = _sinusoidal_embedding_1d
    sys.modules["wan.modules.model"].rope_apply = lambda x, g, f: x

from latent_guidance import (  # noqa: E402
    estimate_tokens,
    forward_to_block,
    gaussian_heatmap,
    optimize_latent,
    rope_in_fp32,
    tokens_to_feature_grid,
    trajectory_loss,
)

DIM, N_BLOCKS = 32, 8
C, T, H, W = 4, 6, 8, 12          # latent (C, T, H, W)
FH, FW = H // 2, W // 2           # feature grid after patch_size (1, 2, 2)


class MockBlock(nn.Module):
    """A stand-in for WanAttentionBlock: same call signature, records calls."""

    def __init__(self, idx, log):
        super().__init__()
        self.idx, self.log = idx, log
        self.lin = nn.Linear(DIM, DIM)
        with torch.no_grad():                       # deterministic, non-trivial
            self.lin.weight.copy_(torch.eye(DIM) * 0.9 + 0.01 * (idx + 1))
            self.lin.bias.fill_(0.01 * (idx + 1))

    def forward(self, x, e=None, seq_lens=None, grid_sizes=None,
                freqs=None, context=None, context_lens=None):
        self.log.append(self.idx)
        return torch.tanh(self.lin(x))


class MockDiT(nn.Module):
    """Mimics the parts of WanModel that forward_to_block touches."""

    def __init__(self):
        super().__init__()
        self.calls = []
        self.head_calls = []
        self.dim, self.freq_dim, self.text_len = DIM, DIM, 8
        self.patch_embedding = nn.Conv3d(C, DIM, (1, 2, 2), stride=(1, 2, 2))
        self.time_embedding = nn.Sequential(nn.Linear(DIM, DIM), nn.SiLU(), nn.Linear(DIM, DIM))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(DIM, DIM * 6))
        self.text_embedding = nn.Sequential(nn.Linear(DIM, DIM), nn.GELU(), nn.Linear(DIM, DIM))
        self.blocks = nn.ModuleList([MockBlock(i, self.calls) for i in range(N_BLOCKS)])
        self.register_buffer("freqs", torch.zeros(1024, DIM // 2, dtype=torch.complex64))

    def head(self, x, e):
        self.head_calls.append(1)
        return x


def _inputs(seq_len=None, requires_grad=False):
    torch.manual_seed(0)
    model = MockDiT()
    latent = torch.randn(C, T, H, W, requires_grad=requires_grad)
    seq_len = seq_len or T * FH * FW
    t_tok = torch.zeros(1, seq_len)
    context = [torch.randn(4, DIM)]
    return model, latent, t_tok, context, seq_len


def _plan(frame, src, dst):
    sy1, sy2, sx1, sx2 = src
    dy1, dy2, dx1, dx2 = dst
    src_map = feathered_map_from_latent_box(FH, FW, (sx1, sy1, sx2, sy2), 0)
    dst_map = feathered_map_from_latent_box(FH, FW, (dx1, dy1, dx2, dy2), 0)
    return LatentShift(frame, src, dst, np.ones((sy2 - sy1, sx2 - sx1), np.float32),
                       np.clip(src_map - dst_map, 0, 1), (dy1 - sy1, dx1 - sx1))


# --------------------------------------------------------------------------
# forward_to_block
# --------------------------------------------------------------------------

def test_forward_to_block_stops_at_the_requested_block():
    for k in (0, 3, N_BLOCKS - 1):
        model, latent, t_tok, context, seq_len = _inputs()
        forward_to_block(model, [latent], t_tok, context, seq_len, k, grad_checkpoint=False)
        assert model.calls == list(range(k + 1)), f"k={k}: ran blocks {model.calls}"
        assert model.head_calls == [], "the head must never run"


def test_forward_to_block_returns_the_right_shapes():
    model, latent, t_tok, context, seq_len = _inputs()
    hidden, grid = forward_to_block(model, [latent], t_tok, context, seq_len, 2,
                                    grad_checkpoint=False)
    assert hidden.shape == (1, seq_len, DIM)
    assert [int(v) for v in grid[0]] == [T, FH, FW]


def test_forward_to_block_gradient_reaches_the_latent():
    model, latent, t_tok, context, seq_len = _inputs(requires_grad=True)
    hidden, _ = forward_to_block(model, [latent], t_tok, context, seq_len, 4,
                                 grad_checkpoint=False)
    hidden.sum().backward()
    assert latent.grad is not None and latent.grad.abs().sum() > 0


def test_checkpointed_and_plain_give_identical_gradients():
    # If checkpointing silently dropped or duplicated a block, the gradients
    # would diverge -- this is the cheapest way to catch that without a GPU.
    grads = []
    for ckpt in (False, True):
        model, latent, t_tok, context, seq_len = _inputs(requires_grad=True)
        hidden, _ = forward_to_block(model, [latent], t_tok, context, seq_len, 5,
                                     grad_checkpoint=ckpt)
        hidden.pow(2).sum().backward()
        grads.append(latent.grad.clone())
        if ckpt:
            assert model.calls == list(range(6)) * 2, (
                "checkpointing should run blocks twice (forward + recompute), got "
                f"{model.calls}")
    assert torch.allclose(grads[0], grads[1], atol=1e-6), "checkpointing changed the gradient"


def test_forward_to_block_rejects_out_of_range_block():
    model, latent, t_tok, context, seq_len = _inputs()
    for k in (-1, N_BLOCKS, 99):
        try:
            forward_to_block(model, [latent], t_tok, context, seq_len, k)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for block_idx={k}")


# --------------------------------------------------------------------------
# tokens_to_feature_grid
# --------------------------------------------------------------------------

def test_tokens_to_feature_grid_drops_padding_and_reshapes():
    padded = T * FH * FW + 37
    hidden = torch.arange(padded * DIM, dtype=torch.float32).view(1, padded, DIM)
    grid = torch.tensor([[T, FH, FW]])
    feat = tokens_to_feature_grid(hidden, grid)
    assert feat.shape == (T, FH, FW, DIM)
    # token index f*(H'*W') + h*W' + w must land at [f, h, w]
    assert torch.equal(feat[2, 1, 3], hidden[0, 2 * FH * FW + 1 * FW + 3])


def test_tokens_to_feature_grid_rejects_a_short_sequence():
    hidden = torch.zeros(1, 4, DIM)
    try:
        tokens_to_feature_grid(hidden, torch.tensor([[T, FH, FW]]))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when the grid needs more tokens than exist")


# --------------------------------------------------------------------------
# gaussian_heatmap
# --------------------------------------------------------------------------

def test_gaussian_heatmap_peaks_at_the_centre_and_is_normalised():
    g = gaussian_heatmap(8, 12)
    assert g.shape == (8, 12)
    assert abs(g.max().item() - 1.0) < 1e-6
    assert g[4, 6] > g[0, 0] and g[4, 6] > g[7, 11]
    assert torch.allclose(g, g.flip(0), atol=1e-6), "must be symmetric vertically"
    assert torch.allclose(g, g.flip(1), atol=1e-6), "must be symmetric horizontally"


def test_gaussian_heatmap_sigma_controls_falloff():
    tight, wide = gaussian_heatmap(16, 16, 0.2), gaussian_heatmap(16, 16, 0.8)
    assert tight.sum() < wide.sum(), "a smaller sigma must concentrate weight"


# --------------------------------------------------------------------------
# trajectory_loss
# --------------------------------------------------------------------------

def test_loss_is_zero_when_the_crops_already_match():
    feats = torch.randn(T, FH, FW, DIM)
    feats[3] = feats[0]                       # frame 3 identical to the anchor
    plan = [_plan(3, (1, 3, 1, 4), (1, 3, 1, 4))]
    assert trajectory_loss(feats, plan, anchor_idx=0).item() == 0.0


def test_loss_matches_a_hand_computed_gaussian_weighted_mean():
    feats = torch.zeros(T, FH, FW, DIM)
    feats[2, 1:3, 1:4] = 2.0                  # constant residual of 2 -> per_cell = 4
    plan = [_plan(2, (1, 3, 1, 4), (1, 3, 1, 4))]
    got = trajectory_loss(feats, plan, anchor_idx=0).item()
    assert abs(got - 4.0) < 1e-5, f"a constant residual must give exactly its square, got {got}"


def test_loss_normaliser_makes_it_invariant_to_box_count():
    feats = torch.zeros(T, FH, FW, DIM)
    feats[2, 1:3, 1:4] = 2.0
    feats[3, 1:3, 1:4] = 2.0
    one = trajectory_loss(feats, [_plan(2, (1, 3, 1, 4), (1, 3, 1, 4))], 0)
    two = trajectory_loss(feats, [_plan(2, (1, 3, 1, 4), (1, 3, 1, 4)),
                                  _plan(3, (1, 3, 1, 4), (1, 3, 1, 4))], 0)
    assert abs(one.item() - two.item()) < 1e-6, "adding an identical box must not change the mean"


def test_loss_gradient_is_zero_outside_the_boxes():
    feats = torch.randn(T, FH, FW, DIM, requires_grad=True)
    plan = [_plan(2, (1, 3, 1, 4), (1, 3, 1, 4))]
    trajectory_loss(feats, plan, anchor_idx=0).backward()
    touched = torch.zeros(T, FH, FW, dtype=torch.bool)
    touched[2, 1:3, 1:4] = True
    touched[0, 1:3, 1:4] = True               # the anchor crop is read (but detached)
    grad = feats.grad.abs().sum(-1)
    assert grad[~touched].max() == 0.0, "cells outside the boxes must get no gradient"
    assert grad[2, 1:3, 1:4].min() > 0
    assert grad[0, 1:3, 1:4].max() == 0.0, "the anchor target must be stop-gradiented"


def test_huber_agrees_with_mse_for_small_residuals_and_diverges_for_large():
    feats_small = torch.zeros(T, FH, FW, DIM)
    feats_small[2, 1:3, 1:4] = 0.02
    plan = [_plan(2, (1, 3, 1, 4), (1, 3, 1, 4))]
    mse_s = trajectory_loss(feats_small, plan, 0, loss="mse").item()
    hub_s = trajectory_loss(feats_small, plan, 0, loss="huber").item()
    assert abs(mse_s - 2 * hub_s) < 1e-6, "Huber is 0.5*x^2 below delta, so mse == 2*huber"

    feats_big = torch.zeros(T, FH, FW, DIM)
    feats_big[2, 1:3, 1:4] = 50.0             # an occluding gripper, in effect
    mse_b = trajectory_loss(feats_big, plan, 0, loss="mse").item()
    hub_b = trajectory_loss(feats_big, plan, 0, loss="huber").item()
    assert mse_b / hub_b > 25, (
        f"Huber must dampen a large outlier hard (mse={mse_b:.1f}, huber={hub_b:.1f})")


def test_loss_rejects_bad_inputs():
    feats = torch.randn(T, FH, FW, DIM)
    plan = [_plan(2, (1, 3, 1, 4), (1, 3, 1, 4))]
    for bad in [
        lambda: trajectory_loss(feats[0], plan, 0),
        lambda: trajectory_loss(feats, plan, 0, loss="l1"),
        lambda: trajectory_loss(feats, [_plan(99, (1, 3, 1, 4), (1, 3, 1, 4))], 0),
    ]:
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_empty_plan_gives_zero_loss():
    feats = torch.randn(T, FH, FW, DIM)
    assert trajectory_loss(feats, [], 0).item() == 0.0


# --------------------------------------------------------------------------
# optimize_latent
# --------------------------------------------------------------------------

def test_optimize_latent_reduces_the_loss_and_respects_the_mask():
    model, latent, t_tok, context, seq_len = _inputs()
    mask2 = torch.zeros_like(latent)
    mask2[:, 2:] = 1.0                                  # frames 0,1 frozen
    plan = [_plan(f, (1, 3, 1, 4), (1, 3, 1, 4)) for f in (3, 4)]

    def loss_of(x):
        hidden, grid = forward_to_block(model, [x], t_tok, context, seq_len, 3,
                                        grad_checkpoint=False)
        return trajectory_loss(tokens_to_feature_grid(hidden, grid), plan, 0).item()

    before = loss_of(latent)
    out = optimize_latent(latent, mask2, model, t_tok, context, seq_len, plan, 0,
                          block_idx=3, iters=6, lr=0.05, grad_checkpoint=False, verbose=False)
    assert loss_of(out) < before, f"loss did not fall: {before} -> {loss_of(out)}"
    assert torch.equal(out[:, :2], latent[:, :2]), "masked (frozen) frames must not move"
    assert not torch.equal(out[:, 2:], latent[:, 2:]), "unmasked frames should have moved"
    assert out.dtype == latent.dtype and not out.requires_grad


def test_optimize_latent_is_a_noop_without_work_to_do():
    model, latent, t_tok, context, seq_len = _inputs()
    mask2 = torch.ones_like(latent)
    plan = [_plan(3, (1, 3, 1, 4), (1, 3, 1, 4))]
    assert torch.equal(
        optimize_latent(latent, mask2, model, t_tok, context, seq_len, [], 0,
                        block_idx=3, grad_checkpoint=False, verbose=False), latent)
    assert torch.equal(
        optimize_latent(latent, mask2, model, t_tok, context, seq_len, plan, 0,
                        block_idx=3, iters=0, grad_checkpoint=False, verbose=False), latent)


def test_optimize_latent_does_not_mutate_its_input():
    model, latent, t_tok, context, seq_len = _inputs()
    before = latent.clone()
    plan = [_plan(3, (1, 3, 1, 4), (1, 3, 1, 4))]
    optimize_latent(latent, torch.ones_like(latent), model, t_tok, context, seq_len,
                    plan, 0, block_idx=3, iters=3, grad_checkpoint=False, verbose=False)
    assert torch.equal(latent, before)


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------

def test_rope_in_fp32_patches_and_restores():
    mod = sys.modules["wan.modules.model"]
    original = mod.rope_apply
    with rope_in_fp32(mod):
        assert mod.rope_apply is not original
        assert mod.rope_apply.__name__ == "rope_apply_fp32"
    assert mod.rope_apply is original, "the patch must be reverted on exit"

    try:                                    # and on exception too
        with rope_in_fp32(mod):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert mod.rope_apply is original


def test_estimate_tokens_matches_the_real_grid():
    assert estimate_tokens((48, 18, 44, 80)) == 18 * 22 * 40 == 15840
    assert estimate_tokens((C, T, H, W)) == T * FH * FW


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} tests passed")
