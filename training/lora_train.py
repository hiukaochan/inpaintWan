"""LoRA fine-tune Wan2.2 TI2V-5B on DROID (first frame + caption -> video).

    python training/lora_train.py --cache_dir training/cache/640x352_f49s1
    torchrun --nproc_per_node=2 training/lora_train.py --cache_dir ...

Runs entirely off the precomputed cache, so neither T5 (11.4 GB) nor the VAE
(2.8 GB) is loaded here -- together with a 10 GB DiT they do not fit on a 24 GB
card. Sampling is likewise a separate process (`training/sample_eval.py`), not
an in-loop hook, for the same reason.

Memory at 640x352 / 49 frames (2860 tokens), bf16 + gradient checkpointing:
~10.0 GB DiT + ~1.3 GB LoRA optimizer state + ~1.1 GB activations ~= 13.6 GB.

Wan2.2/wan/ stays vendored and untouched: gradient checkpointing is installed
by wrapping each block's bound `forward`, not by editing the model.
"""

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Wan2.2"))
sys.path.insert(0, str(ROOT))

from training.conditioning import (  # noqa: E402
    add_noise,
    build_loss_mask,
    build_per_token_timestep,
    sample_sigma,
    velocity_target,
)
from training.droid_dataset import DroidLatentDataset, collate, seq_len_for  # noqa: E402
from wan.configs import WAN_CONFIGS  # noqa: E402
from wan.modules.model import WanModel  # noqa: E402
from wan.utils.utils import masks_like  # noqa: E402

# Every nn.Linear in a WanAttentionBlock. Deliberately excluded:
#   modulation      -- a raw nn.Parameter; peft cannot wrap it
#   patch_embedding -- Conv3d defining the latent->token mapping, which must
#                      stay compatible with the inference path
#   head, time_embedding, time_projection, all norms
# Those last are exactly the parameters that, when perturbed, break the
# per-token-timestep conditioning that i2v depends on.
# Smooth bars in a terminal; one line every 30s when redirected to a log,
# so a nohup run stays readable instead of megabytes of carriage returns.
BAR_INTERVAL = 0.1 if sys.stderr.isatty() else 30.0

LORA_TARGETS = [
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
    "ffn.0", "ffn.2",
]


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def log(msg: str) -> None:
    # tqdm.write rather than print: a bare print tears the progress bar in half
    if rank() == 0:
        tqdm.write(msg)


def enable_grad_checkpointing(model: WanModel) -> None:
    """Recompute each transformer block's activations during backward.

    Wraps the bound `forward` rather than patching the vendored module.
    `use_reentrant=False` is required for this to compose with DDP.
    """
    for block in model.blocks:
        original = block.forward

        def wrapped(*args, _original=original, **kwargs):
            return torch.utils.checkpoint.checkpoint(_original, *args, use_reentrant=False, **kwargs)

        block.forward = wrapped


def build_model(args, device):
    from peft import LoraConfig, get_peft_model

    log(f"loading WanModel from {args.ckpt_dir}")
    model = WanModel.from_pretrained(args.ckpt_dir).to(torch.bfloat16)
    model.requires_grad_(False)

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=LORA_TARGETS,
        ),
    )

    # Adapter weights in fp32 for optimizer stability; the frozen base stays
    # bf16 and autocast casts the adapters down for the matmuls.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"trainable {trainable/1e6:.1f}M / {total/1e6:.1f}M params ({100*trainable/total:.2f}%)")

    model = model.to(device)
    if args.grad_checkpointing:
        enable_grad_checkpointing(model.base_model.model)
    return model


def training_step(model, batch, num_train_timesteps, args, device, generator, null_context):
    latents = [x.to(device, torch.float32, non_blocking=True) for x in batch["latent"]]
    contexts = [c.to(device, torch.bfloat16, non_blocking=True) for c in batch["context"]]
    seq_len = seq_len_for(latents[0])

    x_in, t_in, targets, masks = [], [], [], []
    for z0 in latents:
        noise = torch.randn(z0.shape, generator=generator, device=device, dtype=torch.float32)
        sigma = sample_sigma(args.shift, generator=generator, device=device)

        # p is P(i2v-conditioned). Upstream's default of 0.2 reflects TI2V-5B's
        # joint t2v/i2v pretraining; for an I2V adapt we invert it, keeping a
        # little t2v so the unconditional branch CFG uses at inference does not
        # drift away from the conditioned one.
        mask1, mask2 = masks_like([noise], zero=True, generator=generator, p=args.cond_prob)

        x_in.append(add_noise(z0, noise, sigma, mask1[0]).to(torch.bfloat16))
        t_in.append(build_per_token_timestep(mask2[0], sigma * num_train_timesteps, seq_len))
        targets.append(velocity_target(z0, noise))
        masks.append(build_loss_mask(mask2[0]))

    if null_context is not None and args.caption_dropout > 0:
        keep = torch.rand(len(contexts), generator=generator, device=device)
        contexts = [
            null_context if keep[i].item() < args.caption_dropout else c
            for i, c in enumerate(contexts)
        ]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        preds = model(x_in, t=torch.cat(t_in, dim=0), context=contexts, seq_len=seq_len)

    # WanModel returns a list of fp32 tensors; keep the target in fp32 rather
    # than casting the prediction down.
    numer = sum(((p.float() - tgt) ** 2 * m).sum() for p, tgt, m in zip(preds, targets, masks))
    denom = sum(m.sum() for m in masks)
    return numer / denom


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache_dir", type=Path, required=True)
    ap.add_argument("--index", type=Path, default=Path("training/index.json"))
    ap.add_argument("--ckpt_dir", type=str, default="Wan2.2/checkpoints/Wan2.2-TI2V-5B")
    ap.add_argument("--output_dir", type=Path, default=Path("training/runs/droid_lora"))
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--cond_prob", type=float, default=0.9, help="P(i2v-conditioned sample)")
    ap.add_argument("--caption_dropout", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_grad_checkpointing", dest="grad_checkpointing", action="store_false")
    ap.add_argument(
        "--overfit_one",
        action="store_true",
        help="train on a single clip; loss should collapse toward zero",
    )
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(args.seed + rank())
    generator = torch.Generator(device=device).manual_seed(args.seed + rank())

    num_train_timesteps = WAN_CONFIGS["ti2v-5B"].num_train_timesteps

    dataset = DroidLatentDataset(
        args.cache_dir,
        split="train",
        index_path=args.index if args.index.exists() else None,
    )
    if args.overfit_one:
        dataset.clips = dataset.clips[:1]
    log(f"{len(dataset)} training clips")

    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if is_dist() else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None and not args.overfit_one),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(args, device)
    if is_dist():
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    def lr_scale(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank() == 0:
        with open(args.output_dir / "args.json", "w") as fh:
            json.dump({k: str(v) for k, v in vars(args).items()}, fh, indent=1)

    null_context = dataset.null_context
    if null_context is not None:
        null_context = null_context.to(device, torch.bfloat16)

    def save(tag: str) -> None:
        if rank() != 0:
            return
        dst = args.output_dir / tag
        (model.module if is_dist() else model).save_pretrained(str(dst))
        log(f"saved adapter to {dst}")

    step = 0
    running = 0.0
    model.train()

    # disabled off rank 0 so the two DDP processes do not fight over the line
    bar = tqdm(total=args.max_steps, disable=rank() != 0, unit="step",
               dynamic_ncols=True, mininterval=BAR_INTERVAL)

    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in loader:
            accumulating = (step + 1) % args.grad_accum != 0
            sync_ctx = model.no_sync() if (is_dist() and accumulating) else nullcontext()
            with sync_ctx:
                loss = training_step(
                    model, batch, num_train_timesteps, args, device, generator, null_context
                )
                (loss / args.grad_accum).backward()

            running += loss.item()
            if not accumulating:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            bar.update(1)
            if step % args.log_every == 0:
                bar.set_postfix(
                    loss=f"{running/args.log_every:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    peak=f"{torch.cuda.max_memory_allocated()/2**30:.1f}GiB",
                    refresh=False,
                )
                running = 0.0

            if step % args.save_every == 0:
                save(f"step{step:06d}")
            if step >= args.max_steps:
                break

    bar.close()
    save("final")
    log("done")
    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
