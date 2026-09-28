"""Local, model-free unit tests for the LoRA training step.

Exercises the real `WanModel` forward and backward on a deliberately tiny
model, so the whole training path -- conditioning, DiT forward, masked loss,
backward, LoRA wiring, gradient checkpointing -- is checked without a GPU or
the 10 GB checkpoint. This matters because the actual run is gated on hardware
that is often busy; nothing here should have to wait for a free card.

Two traps this file exists to document, both of which look like bugs the first
time you hit them:

  * `WanModel.init_weights` zero-initializes the output layer
    (model.py:546, standard DiT practice). A *freshly constructed* model
    therefore predicts exactly zero and no gradient reaches any block, so LoRA
    gradients are all zero and the loss sits at E[(noise-z0)^2] = 2. That is
    correct behaviour for an untrained model, not a broken training loop, and
    it does not happen with the real pretrained checkpoint. The tests below
    perturb the head to simulate a trained one.

  * At LoRA initialization `lora_B` is zero, so on the first step only the
    `lora_B` matrices receive gradient -- `lora_A`'s gradient is
    `dL/dout * lora_B`, i.e. zero until lora_B moves. Seeing exactly half the
    adapter parameters with nonzero grad is expected.

`flash_attention` requires CUDA, so these patch in an SDPA equivalent.

Runs on CPU with no checkpoints. Run with: python test_training_step.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "Wan2.2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import wan.modules.model as wan_model


def _sdpa(q, k, v, q_lens=None, k_lens=None, window_size=(-1, -1), **kwargs):
    """CPU stand-in for flash_attention: [B,L,N,C] in and out.

    Padding masks are ignored, which is exact here because these tests use a
    single unpadded sample.
    """
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    )
    return out.transpose(1, 2)


wan_model.flash_attention = _sdpa

from peft import LoraConfig, get_peft_model  # noqa: E402

from training.conditioning import (  # noqa: E402
    add_noise,
    build_loss_mask,
    build_per_token_timestep,
    sample_sigma,
    velocity_target,
)
from training.droid_dataset import seq_len_for  # noqa: E402
from training.lora_train import LORA_TARGETS, enable_grad_checkpointing  # noqa: E402
from wan.modules.model import WanModel  # noqa: E402
from wan.utils.utils import masks_like  # noqa: E402

C, T, H, W = 48, 13, 22, 40  # 640x352 @ 49 frames
TEXT_LEN, TEXT_DIM = 130, 4096


def _tiny_model(lora=True, trained_head=True, grad_checkpointing=False):
    torch.manual_seed(0)
    model = WanModel(
        model_type="ti2v", dim=64, ffn_dim=128, num_heads=4,
        num_layers=2, in_dim=C, out_dim=C,
    )
    if trained_head:
        # undo the DiT zero-init so gradients can reach the blocks, as they do
        # with the real pretrained weights
        torch.nn.init.normal_(model.head.head.weight, std=0.02)
    if not lora:
        return model

    model.requires_grad_(False)
    model = get_peft_model(
        model, LoraConfig(r=8, lora_alpha=8, target_modules=LORA_TARGETS, bias="none")
    )
    if grad_checkpointing:
        enable_grad_checkpointing(model.base_model.model)
    return model


def _batch(seed=1, cond_prob=0.9):
    z0 = torch.randn(C, T, H, W)
    context = torch.randn(TEXT_LEN, TEXT_DIM)
    seq_len = seq_len_for(z0)
    g = torch.Generator().manual_seed(seed)

    noise = torch.randn(z0.shape, generator=g)
    sigma = sample_sigma(5.0, generator=g)
    mask1, mask2 = masks_like([noise], zero=True, generator=g, p=cond_prob)

    return {
        "x_t": add_noise(z0, noise, sigma, mask1[0]),
        "t_in": build_per_token_timestep(mask2[0], sigma * 1000, seq_len),
        "target": velocity_target(z0, noise),
        "mask": build_loss_mask(mask2[0]),
        "context": context,
        "seq_len": seq_len,
        "z0": z0,
    }


def _loss(model, b):
    pred = model([b["x_t"]], t=b["t_in"], context=[b["context"]], seq_len=b["seq_len"])[0]
    return pred, ((pred.float() - b["target"]) ** 2 * b["mask"]).sum() / b["mask"].sum()


def test_forward_shapes_round_trip():
    """The DiT must return a latent the same shape it was given."""
    model = _tiny_model()
    b = _batch()
    pred, _ = _loss(model, b)
    assert pred.shape == b["z0"].shape, (pred.shape, b["z0"].shape)
    assert pred.dtype == torch.float32, "WanModel returns fp32; keep the target fp32 too"


def test_untrained_head_predicts_zero_and_loss_equals_target_power():
    """Documents the zero-init head trap: loss == E[(noise-z0)^2] == 2."""
    model = _tiny_model(lora=False, trained_head=False)
    b = _batch()
    pred, loss = _loss(model, b)
    assert pred.abs().max() == 0.0, "zero-init head must predict exactly zero"
    assert abs(loss.item() - 2.0) < 0.1, loss.item()


def test_gradients_reach_lora_and_not_the_frozen_base():
    model = _tiny_model()
    b = _batch()
    _, loss = _loss(model, b)
    loss.backward()

    adapters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert len(adapters) == 40, f"expected 20 wrapped Linears x 2 matrices, got {len(adapters)}"

    nonzero = [n for n, p in adapters if p.grad is not None and p.grad.abs().sum() > 0]
    # only lora_B moves on the first step; lora_A's grad is dL/dout * lora_B == 0
    assert len(nonzero) == 20, f"expected 20 lora_B grads, got {len(nonzero)}"
    assert all("lora_B" in n for n in nonzero), "only lora_B should have grad at init"

    frozen_with_grad = [n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
    assert not frozen_with_grad, frozen_with_grad


def test_gradient_checkpointing_gives_the_same_gradients():
    """Checkpointing must only trade compute for memory, never change results."""
    grads = {}
    for use_ckpt in (False, True):
        model = _tiny_model(grad_checkpointing=use_ckpt)
        _, loss = _loss(model, _batch())
        loss.backward()
        grads[use_ckpt] = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None
        }

    assert grads[False].keys() == grads[True].keys()
    for name in grads[False]:
        assert torch.allclose(grads[False][name], grads[True][name], atol=1e-6), name


def test_lora_excludes_modulation_patch_embedding_and_head():
    """These break the conditioning path or the latent<->token mapping."""
    model = _tiny_model()
    wrapped = {n.split(".lora_A")[0] for n, _ in model.named_parameters() if "lora_A" in n}
    assert wrapped, "no modules wrapped"
    for forbidden in ("modulation", "patch_embedding", "time_embedding", "time_projection"):
        assert not any(forbidden in w for w in wrapped), forbidden
    assert not any(w.endswith("head") for w in wrapped)


def test_conditioning_frame_is_excluded_from_the_loss():
    """A conditioned sample must not be scored on the frame it was handed."""
    model = _tiny_model()
    b = _batch()
    assert torch.all(b["mask"][:, 0] == 0), "expected a conditioned sample"

    pred, loss = _loss(model, b)
    # corrupting the prediction on frame 0 alone must not move the loss
    pred_bad = pred.clone()
    pred_bad[:, 0] += 1000.0
    loss_bad = ((pred_bad.float() - b["target"]) ** 2 * b["mask"]).sum() / b["mask"].sum()
    assert torch.allclose(loss, loss_bad), "loss is looking at the conditioning frame"


def test_unconditioned_sample_scores_every_frame():
    model = _tiny_model()
    b = _batch(cond_prob=0.0)  # never conditioned -> pure t2v
    assert torch.all(b["mask"] == 1), "expected an unconditioned sample"
    assert torch.all(b["t_in"] > 0), "t2v samples carry t on every token"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(tests)} passed")
