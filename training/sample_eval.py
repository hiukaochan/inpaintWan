"""Sample from a DROID-fine-tuned TI2V-5B adapter, for qualitative comparison.

    python training/sample_eval.py --adapter training/runs/droid_lora/final \
        --index training/index_smoke.json --num 4

Run as its own process, never as an in-loop hook inside training: this loads
T5 + VAE + DiT together, which alone is ~24 GB.

Takes the first frame and caption straight from held-out DROID episodes, so
generations are directly comparable against the ground-truth video. Pass
`--base_too` to emit the frozen model's output alongside the adapter's -- the
adapter is only interesting relative to what the base model already did.

Note the caption style matters: DROID captions are dense 60-110 word VLM
descriptions, and the adapter is tuned on exactly that. Prompting it with a
short instruction ("pick up the can") is off-distribution and will underperform.
"""

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Wan2.2"))
sys.path.insert(0, str(ROOT))

import wan  # noqa: E402
from wan.configs import WAN_CONFIGS  # noqa: E402
from wan.utils.utils import save_video  # noqa: E402

CROP_H = 704


def first_frame(video_path: Path) -> Image.Image:
    """Frame 0, center-cropped 1280x720 -> 1280x704 to match training."""
    from decord import VideoReader, cpu

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
    frame = vr[0].asnumpy()
    top = (frame.shape[0] - CROP_H) // 2
    return Image.fromarray(frame[top : top + CROP_H])


def load_episode(entry: dict) -> tuple[Image.Image, str]:
    path = Path(entry["path"])
    with open(path / "caption.pickle", "rb") as fh:
        caption = pickle.load(fh)["caption"]
    return first_frame(path / "rgb.mp4"), caption


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", type=Path, default=None, help="omit to sample the base model")
    ap.add_argument("--index", type=Path, default=Path("training/index.json"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--ckpt_dir", type=str, default="Wan2.2/checkpoints/Wan2.2-TI2V-5B")
    ap.add_argument("--output_dir", type=Path, default=Path("training/samples"))
    ap.add_argument("--num", type=int, default=4)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=352)
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guide_scale", type=float, default=5.0)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--fps", type=int, default=15, help="decode-time label; DROID is ~14.9 fps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base_too", action="store_true", help="also sample without the adapter")
    ap.add_argument("--device_id", type=int, default=0)
    args = ap.parse_args()

    with open(args.index) as fh:
        entries = [e for e in json.load(fh)["entries"] if e["split"] == args.split]
    if not entries:
        raise SystemExit(f"no entries with split={args.split} in {args.index}")
    random.Random(args.seed).shuffle(entries)
    entries = entries[: args.num]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = WAN_CONFIGS["ti2v-5B"]

    pipe = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device_id,
        rank=0,
        t5_cpu=False,
        convert_model_dtype=True,
    )

    variants = [("base", None)] if args.base_too else []
    if args.adapter is not None:
        variants.append(("lora", args.adapter))
    if not variants:
        variants = [("base", None)]

    episodes = [(e, *load_episode(e)) for e in entries]

    for tag, adapter in variants:
        if adapter is not None:
            from peft import PeftModel

            print(f"merging adapter {adapter}")
            # merge_and_unload keeps peft out of the forward path, so the
            # freeze/inpaint hooks in the project scripts still apply cleanly.
            pipe.model = PeftModel.from_pretrained(pipe.model, str(adapter)).merge_and_unload()

        for i, (entry, image, caption) in enumerate(episodes):
            print(f"[{tag}] {i+1}/{len(episodes)} {entry['session']}")
            video = pipe.generate(
                caption,
                img=image,
                size=(args.width, args.height),
                max_area=args.width * args.height,
                frame_num=args.frames,
                shift=args.shift,
                sampling_steps=args.steps,
                guide_scale=args.guide_scale,
                seed=args.seed,
                offload_model=True,
            )
            dst = args.output_dir / f"{i:02d}_{tag}_{entry['serial']}.mp4"
            save_video(
                video[None].cpu(),
                str(dst),
                fps=args.fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            (args.output_dir / f"{i:02d}_prompt.txt").write_text(caption)
            print(f"  wrote {dst}")

        if adapter is not None:
            # a merged adapter cannot be un-merged; reload for any later variant
            del pipe.model
            torch.cuda.empty_cache()
            from wan.modules.model import WanModel

            pipe.model = WanModel.from_pretrained(args.ckpt_dir).eval().requires_grad_(False)
            pipe.model.to(pipe.device)

    print(f"samples in {args.output_dir}")


if __name__ == "__main__":
    main()
