"""Precompute VAE latents and T5 embeddings for DROID clips.

Caching is mandatory, not an optimization: T5 (11.4 GB) + VAE (2.8 GB) + the DiT
(10 GB bf16) is 24.2 GB, i.e. over a 4090's budget before a single activation.
After this runs, training loads only the DiT.

The two encoders also cannot be co-resident comfortably, so this runs in stages:

    python training/precompute_cache.py --stage text  --index training/index.json
    python training/precompute_cache.py --stage video --index training/index.json

Both stages are resumable (existing shards are skipped) and sharded via
`--rank/--world_size`, so you can run one process per GPU.

Video preprocessing per clip: center-crop 1280x720 -> 1280x704 (8 px off top and
bottom; squashing to 704 would distort scene geometry), resize to the target,
then take `49` frames at `--temporal_stride`. Clips are non-overlapping and each
is encoded independently -- the Wan VAE is causal over time, so slicing a
long cached latent at an arbitrary offset would not reproduce a standalone
encode of those frames.

Note on frame rate: DROID runs at ~14.9 fps and Wan's prior is 24 fps, but you
cannot resample up without a frame interpolator -- duplicating frames only makes
motion stutter. Output fps is a decode-time label that does not affect training;
what does affect it is `--temporal_stride`, which sets how much real time (and
hence how much motion) a 49-frame clip spans: stride 1 = 3.3 s, stride 2 = 6.6 s.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Wan2.2"))

from wan.configs import WAN_CONFIGS  # noqa: E402

# Smooth bars in a terminal; one line every 30s when redirected to a log,
# so a nohup run stays readable instead of megabytes of carriage returns.
BAR_INTERVAL = 0.1 if sys.stderr.isatty() else 30.0

SRC_W, SRC_H = 1280, 720
CROP_H = 704  # 1280x704 is TI2V-5B's native bucket; 640x352 is exactly half


def resolve_paths(entry: dict, root: Path) -> Path:
    return Path(entry.get("path") or root / entry["session"] / entry["serial"])


def shard_name(entry: dict) -> str:
    return f"{entry['session']}__{entry['serial']}"


# ---------------------------------------------------------------- text stage


def run_text_stage(entries, cache_dir: Path, cfg, ckpt_dir: Path, device, batch_size: int):
    from wan.modules.t5 import T5EncoderModel

    out_dir = cache_dir / "text"
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [e for e in entries if not (out_dir / f"{shard_name(e)}.safetensors").exists()]
    null_path = cache_dir / "null.safetensors"
    if not todo and null_path.exists():
        print("text stage: nothing to do")
        return

    encoder = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=device,
        checkpoint_path=str(ckpt_dir / cfg.t5_checkpoint),
        tokenizer_path=str(ckpt_dir / cfg.t5_tokenizer),
    )

    # The unconditional embedding used for CFG dropout in training and for the
    # negative branch at inference. Wan uses a fixed negative prompt, not "".
    if not null_path.exists():
        null_ctx = encoder([cfg.sample_neg_prompt], device)[0]
        save_file(
            {"context": null_ctx.to(torch.float16).cpu().contiguous()},
            str(null_path),
            metadata={"prompt": cfg.sample_neg_prompt},
        )

    # Batched: the encoder pads to text_len internally and returns one
    # variable-length tensor per caption, so batching costs nothing in storage
    # but is several times faster -- decisively so on CPU.
    bar = tqdm(total=len(todo), unit="caption", dynamic_ncols=True, desc="text",
               mininterval=BAR_INTERVAL)
    for start in range(0, len(todo), batch_size):
        chunk = todo[start : start + batch_size]
        captions = []
        for entry in chunk:
            with open(resolve_paths(entry, Path("")) / "caption.pickle", "rb") as fh:
                captions.append(pickle.load(fh)["caption"])

        for entry, caption, ctx in zip(chunk, captions, encoder(captions, device)):
            save_file(
                {"context": ctx.to(torch.float16).cpu().contiguous()},
                str(out_dir / f"{shard_name(entry)}.safetensors"),
                metadata={"caption": caption},
            )
        bar.update(len(chunk))
    bar.close()

    print(f"text stage: wrote {len(todo)} shards")


# --------------------------------------------------------------- video stage


def preprocess_clip(frames_u8: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """[F,H,W,3] uint8 -> [3,F,h,w] float in [-1,1], center-cropped and resized."""
    x = frames_u8.permute(0, 3, 1, 2).float().div_(255.0)  # [F,3,H,W]
    top = (x.shape[2] - CROP_H) // 2
    x = x[:, :, top : top + CROP_H, :]
    x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
    x = x.sub_(0.5).div_(0.5)
    return x.permute(1, 0, 2, 3).contiguous()  # [3,F,h,w]


def run_video_stage(entries, cache_dir: Path, cfg, ckpt_dir: Path, device, args):
    from decord import VideoReader, cpu
    from wan.modules.vae2_2 import Wan2_2_VAE

    out_dir = cache_dir / "video"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loaded on first use, not up front: an episode's chunk count is only known
    # after opening its video, so there is no cheap way to pre-filter the work
    # list. Deferring means a re-run over an already-complete cache costs no
    # GPU memory at all, which matters when the cards are contended.
    vae = None

    def encode(clip):
        nonlocal vae
        if vae is None:
            vae = Wan2_2_VAE(vae_pth=str(ckpt_dir / cfg.vae_checkpoint), device=device)
        with torch.no_grad():
            # Wan2_2_VAE.encode applies the per-channel mean/std itself
            # (self.scale), so this latent is already training-ready.
            return vae.encode([clip])[0]

    span = (args.frames - 1) * args.temporal_stride + 1  # source frames per clip
    written = skipped = 0

    bar = tqdm(entries, unit="ep", dynamic_ncols=True, desc="video",
               mininterval=BAR_INTERVAL)
    for entry in bar:
        name = shard_name(entry)
        src = resolve_paths(entry, Path(""))
        try:
            vr = VideoReader(str(src / "rgb.mp4"), ctx=cpu(0), num_threads=2)
        except Exception as exc:  # corrupt container; drop the episode
            # tqdm.write so a bad episode stays legible above the bar
            tqdm.write(f"skip {name}: {exc}")
            skipped += 1
            continue

        n_src = len(vr)
        n_chunks = min(n_src // span, args.max_chunks) if span else 0
        for c in range(n_chunks):
            dst = out_dir / f"{name}__{c:03d}.safetensors"
            if dst.exists():
                continue
            start = c * span
            idx = list(range(start, start + span, args.temporal_stride))[: args.frames]
            frames = torch.from_numpy(vr.get_batch(idx).asnumpy())
            latent = encode(preprocess_clip(frames, args.width, args.height).to(device))
            save_file(
                {"latent": latent.to(torch.float16).cpu().contiguous()},
                str(dst),
                metadata={"source_start": str(start), "stride": str(args.temporal_stride)},
            )
            written += 1

        # clips-per-episode varies with video length, so surface the running total
        bar.set_postfix(clips=written, skipped=skipped, refresh=False)

    print(f"video stage: wrote {written} clips, skipped {skipped} episodes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("text", "video"), required=True)
    ap.add_argument("--index", type=Path, default=Path("training/index.json"))
    ap.add_argument("--cache_dir", type=Path, default=None)
    ap.add_argument("--ckpt_dir", type=Path, default=Path("Wan2.2/checkpoints/Wan2.2-TI2V-5B"))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=352)
    ap.add_argument("--frames", type=int, default=49, help="must be 4n+1")
    ap.add_argument("--temporal_stride", type=int, default=1)
    ap.add_argument("--max_chunks", type=int, default=4, help="cap clips per episode")
    ap.add_argument("--text_batch_size", type=int, default=16)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if (args.frames - 1) % 4 != 0:
        ap.error("--frames must be 4n+1 to align with the VAE's temporal stride of 4")
    for name, val in (("width", args.width), ("height", args.height)):
        if val % 32 != 0:
            ap.error(f"--{name} must be divisible by 32 (VAE stride 16 x patch 2)")

    with open(args.index) as fh:
        index = json.load(fh)
    entries = index["entries"]
    entries = entries[args.rank :: args.world_size]

    cache_dir = args.cache_dir or Path("training/cache") / f"{args.width}x{args.height}_f{args.frames}s{args.temporal_stride}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cfg = WAN_CONFIGS["ti2v-5B"]
    device = torch.device(args.device)

    if args.stage == "text":
        run_text_stage(entries, cache_dir, cfg, args.ckpt_dir, device, args.text_batch_size)
    else:
        run_video_stage(entries, cache_dir, cfg, args.ckpt_dir, device, args)


if __name__ == "__main__":
    main()
