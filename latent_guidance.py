"""Gradient-based latent guidance: SG-I2V's Eq. 1 on Wan2.2's DiT.

The alternative to `latent_paste.py`. That module corrects the latent by
*algebra* -- a closed-form write, no model call. This one *searches*: it
measures the error in the DiT's own feature space and backpropagates it to the
latent, exactly as SG-I2V (ICLR 2025) does for Stable Video Diffusion.

Two things follow that a closed-form write structurally cannot do:

  * **it is selective about what it objects to** -- a latent-space difference
    fires on a shadow crossing the box, a feature-space one does not, which
    matters when a gripper occludes the region being controlled;
  * **it never has to say what fills a vacated region** -- a loss asks for
    content at the new position and stays silent about the old one, leaving
    the model's prior to resolve it. That is the unanswered "line 4" of the
    drag formulation, and it is why SG-I2V optimises rather than assigns.

The price is a forward *and* backward pass through half the DiT per iteration,
so this mode costs minutes where `latent_paste` costs milliseconds.

What is ported and what is not (see `SG-I2V-method-notes.md` for the method):

  * Part 1 (replacing each frame's K/V with frame 1's) is **not** ported.
    It exists because SVD's spatial self-attention runs per-frame and so
    produces features that do not correspond across frames. Wan's DiT applies
    full 3D attention over the flattened (F, H, W) token sequence
    (`Wan2.2/wan/modules/model.py:126-155`), so every token already attends to
    every frame. `tools/probe_dit_features.py` checks whether that is enough.
  * Part 2 (Eq. 1) is ported here.
  * Part 3 (Eq. 2, the frequency restore) is already implemented as
    `latent_paste.fft_restore` and is reused unchanged.
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn.functional as F

_HEATMAP_CACHE: dict[tuple[int, int, float], torch.Tensor] = {}


def gaussian_heatmap(h: int, w: int, sigma: float = 0.4, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """`(h, w)` anisotropic Gaussian, peak 1 at the centre.

    Port of `SG-I2V/src/pipeline.py:23-43` (MIT, `SG-I2V/LICENSE`), which
    follows DragAnything. The rationale is that a rectangle around an object
    inevitably catches background near its corners which should not be dragged
    along, so the loss weights the centre more. `sigma` scales with each box
    dimension independently, so tall and wide boxes are treated alike.

    Vectorised and cached; SG-I2V rebuilds it with a Python double loop.
    """
    key = (h, w, sigma)
    if key not in _HEATMAP_CACHE:
        sy, sx = sigma * (h / 2), sigma * (w / 2)
        yy = torch.arange(h, dtype=torch.float32) + 0.5 - h / 2
        xx = torch.arange(w, dtype=torch.float32) + 0.5 - w / 2
        g = torch.exp(-0.5 * ((xx / sx) ** 2)[None, :] - 0.5 * ((yy / sy) ** 2)[:, None])
        _HEATMAP_CACHE[key] = g / g.max()
    return _HEATMAP_CACHE[key].to(device=device, dtype=dtype)


@contextlib.contextmanager
def rope_in_fp32(model_module):
    """Temporarily run Wan's rotary embedding in fp32 instead of fp64.

    `rope_apply` (`Wan2.2/wan/modules/model.py:39-66`) casts queries and keys
    to `float64` before the complex multiply. At ~16k tokens that is roughly
    390MB *per tensor*, and it happens twice per block -- a real share of the
    budget once activations must be retained for backward. fp64 precision is
    meaningless for a guidance signal, so this swaps in an fp32 version for the
    duration of the optimisation pass and restores the original afterwards.

    Patching the vendored module is deliberate but scoped: it is reversed on
    exit, including on exception, and never active during ordinary sampling.
    """
    original = model_module.rope_apply

    def rope_apply_fp32(x, grid_sizes, freqs):
        n, c = x.size(2), x.size(3) // 2
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        output = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w
            x_i = torch.view_as_complex(
                x[i, :seq_len].to(torch.float32).reshape(seq_len, n, -1, 2))
            freqs_i = torch.cat([
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ], dim=-1).reshape(seq_len, 1, -1).to(torch.complex64)
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])
            output.append(x_i)
        return torch.stack(output).float()

    model_module.rope_apply = rope_apply_fp32
    try:
        yield
    finally:
        model_module.rope_apply = original


def forward_to_block(
    model,
    x: list[torch.Tensor],
    t: torch.Tensor,
    context: list[torch.Tensor],
    seq_len: int,
    block_idx: int,
    grad_checkpoint: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `WanModel` only as far as `blocks[block_idx]`; return `(hidden, grid_sizes)`.

    A project-local reimplementation of `WanModel.forward`
    (`Wan2.2/wan/modules/model.py:410-497`), following the convention
    `generate_initial_video.py` documents: reimplement rather than patch the
    vendored class. A forward hook would not be enough, for two reasons:

    * `WanModel` has no gradient-checkpointing support of its own (no
      `_supports_gradient_checkpointing`, no `torch.utils.checkpoint`), so each
      block has to be wrapped here;
    * blocks after `block_idx`, and the head, must not run at all -- they build
      graph and burn memory for an output nobody reads.

    The returned hidden state is `(1, seq_len, dim)` with real tokens first and
    zero padding after; `grid_sizes` gives `(F, H//2, W//2)` for reshaping.
    """
    if not (0 <= block_idx < len(model.blocks)):
        raise ValueError(f"block_idx must be in [0, {len(model.blocks) - 1}], got {block_idx}")

    from wan.modules.model import sinusoidal_embedding_1d

    device = model.patch_embedding.weight.device
    if model.freqs.device != device:
        model.freqs = model.freqs.to(device)

    x = [model.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x
    ])

    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast("cuda", dtype=torch.float32):
        bt = t.size(0)
        e = model.time_embedding(
            sinusoidal_embedding_1d(model.freq_dim, t.flatten()).unflatten(0, (bt, seq_len)).float())
        e0 = model.time_projection(e).unflatten(2, (6, model.dim))

    context = model.text_embedding(torch.stack([
        torch.cat([u, u.new_zeros(model.text_len - u.size(0), u.size(1))]) for u in context
    ]))

    kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
                  freqs=model.freqs, context=context, context_lens=None)

    for i, block in enumerate(model.blocks):
        if grad_checkpoint and torch.is_grad_enabled():
            # use_reentrant=False so kwargs survive and .grad_fn stays sane
            x = torch.utils.checkpoint.checkpoint(
                lambda inp, b=block: b(inp, **kwargs), x, use_reentrant=False)
        else:
            x = block(x, **kwargs)
        if i == block_idx:
            return x, grid_sizes  # never touch the remaining blocks or the head

    raise AssertionError("unreachable: block_idx was validated above")


def tokens_to_feature_grid(hidden: torch.Tensor, grid_sizes: torch.Tensor) -> torch.Tensor:
    """`(1, seq_len, dim)` -> `(F, H', W', dim)`, dropping the sequence padding.

    `WanModel.forward` pads the token sequence out to `seq_len` with zeros
    after patchifying, so only the first `prod(grid_sizes)` entries are real.
    `patch_size=(1, 2, 2)` means `H' = H//2` and `W' = W//2`, i.e. one token
    covers `vae_stride * patch_size = 32` source pixels.
    """
    f, h, w = (int(v) for v in grid_sizes[0].tolist())
    real = f * h * w
    if hidden.shape[1] < real:
        raise ValueError(f"hidden has {hidden.shape[1]} tokens but the grid needs {real}")
    return hidden[0, :real].view(f, h, w, hidden.shape[-1])


def trajectory_loss(
    features: torch.Tensor,
    plan: list,
    anchor_idx: int,
    sigma: float = 0.4,
    loss: str = "mse",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """SG-I2V Eq. 1 over a feature grid.

    `features` is `(F, H', W', dim)` from `tokens_to_feature_grid`; `plan` is a
    list of `frame_mapping.LatentShift` built at the feature grid's stride
    (32px), so `src` is the box at the anchor frame and `dst` its position at
    that frame -- identical for a static box, displaced along a trajectory.

        per_cell = (source - stopgrad(target)) ** 2, averaged over `dim`
        loss     = sum(gaussian * per_cell) / sum(gaussian)

    The normaliser is accumulated across *all* boxes and frames, making the
    result one weighted mean: adding boxes dilutes rather than amplifies, and
    box area does not inflate it. That matches SG-I2V's
    `loss / max(1e-8, loss_cnt)` (`SG-I2V/src/pipeline.py:144`).

    `loss="huber"` is a deliberate deviation, available because occlusion is
    expected here in a way it was not in the paper: where a gripper crosses the
    box those cells genuinely *should* differ from the anchor, and under a
    squared penalty they become the largest residuals and dominate the
    gradient -- the loss would actively fight the arm. Huber caps their
    influence so they read as outliers instead.
    """
    if features.ndim != 4:
        raise ValueError(f"features must be (F, H, W, dim), got {tuple(features.shape)}")
    if loss not in ("mse", "huber"):
        raise ValueError(f"loss must be 'mse' or 'huber', got {loss!r}")

    total = features.new_zeros(())
    count = features.new_zeros(())
    frames = features.shape[0]

    for shift in plan:
        if not (0 <= shift.frame < frames):
            raise ValueError(f"plan references frame {shift.frame} but features have {frames}")
        sy1, sy2, sx1, sx2 = shift.src
        dy1, dy2, dx1, dx2 = shift.dst

        target = features[anchor_idx, sy1:sy2, sx1:sx2].detach()
        source = features[shift.frame, dy1:dy2, dx1:dx2]

        diff = source - target
        if loss == "mse":
            per_cell = diff.pow(2).mean(dim=-1)
        else:
            per_cell = F.huber_loss(
                source, target, reduction="none", delta=huber_delta).mean(dim=-1)

        g = gaussian_heatmap(per_cell.shape[0], per_cell.shape[1], sigma,
                             device=features.device, dtype=per_cell.dtype)
        total = total + (g * per_cell).sum()
        count = count + g.sum()

    if count == 0:
        return total
    return total / count


def optimize_latent(
    latent: torch.Tensor,
    mask2: torch.Tensor,
    model,
    t_tok: torch.Tensor,
    context: list[torch.Tensor],
    seq_len: int,
    plan: list,
    anchor_idx: int,
    *,
    block_idx: int = 15,
    iters: int = 5,
    lr: float = 0.21,
    heatmap_sigma: float = 0.4,
    loss: str = "mse",
    grad_checkpoint: bool = True,
    verbose: bool = True,
) -> torch.Tensor:
    """SG-I2V's inner optimisation loop: `iters` AdamW steps on the latent.

    Returns a detached latent. The caller applies `fft_restore` and re-asserts
    the frozen-context mask, matching how `latent_paste`-mode corrections are
    finished off.

    Differences from `SG-I2V/src/pipeline.py:45-195`, all forced by the
    backbone rather than chosen:

    * **No `GradScaler`.** They autocast to fp16, which needs loss scaling to
      keep gradients from flushing to zero. Wan runs bf16, which has fp32
      exponent range, so a scaler would be cargo-culted complexity.
    * **The gradient is masked by `mask2`** before each step, so frozen context
      positions never move. The denoising loop re-pastes clean content over
      them anyway, but masking keeps AdamW's moments from accumulating on
      positions whose updates are discarded.
    * **One conditional forward per iteration, no CFG** -- same as SG-I2V,
      which optimises against `image_embeddings[1:]` only.
    """
    if iters < 1 or not plan:
        return latent.detach()

    original_dtype = latent.dtype
    work = latent.detach().to(torch.float32).requires_grad_(True)
    optimizer = torch.optim.AdamW([work], lr=lr)
    mask = mask2.to(dtype=torch.float32)
    first = last = None

    with torch.enable_grad():
        for it in range(iters):
            hidden, grid_sizes = forward_to_block(
                model, [work], t_tok, context, seq_len, block_idx,
                grad_checkpoint=grad_checkpoint)
            features = tokens_to_feature_grid(hidden, grid_sizes)
            value = trajectory_loss(features, plan, anchor_idx,
                                    sigma=heatmap_sigma, loss=loss)

            optimizer.zero_grad(set_to_none=True)
            value.backward()
            if work.grad is not None:
                work.grad.mul_(mask)
            optimizer.step()

            if it == 0:
                first = float(value.detach())
            last = float(value.detach())
            del hidden, features, value

    if verbose and first is not None:
        print(f"[gradient] loss {first:.5f} -> {last:.5f} over {iters} iters "
              f"({'no change' if abs(first - last) < 1e-9 else f'{100 * (first - last) / max(first, 1e-12):+.1f}%'})")

    return work.detach().to(original_dtype)


def estimate_tokens(latent_shape: tuple[int, ...], patch_spatial: int = 2) -> int:
    """Token count for a latent `(C, T, H, W)` -- the thing memory scales with."""
    _, f, h, w = latent_shape
    return f * math.ceil(h / patch_spatial) * math.ceil(w / patch_spatial)
