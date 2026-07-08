"""Generate the initial video: first-frame + text conditioned, via Wan2.2.

Thin CLI wrapper around Wan2.2's already-shipped `WanTI2V.i2v()` (reached via
`WanTI2V.generate(prompt, img=...)`), mirroring `Wan2.2/generate.py`'s own
`ti2v` branch. This is step 1 of the workflow: produce a starting video from
a first frame + prompt; step 2 is picking a frame range in it and correcting
it with `frame_range_edit.py`.

Run on a machine with the Wan2.2 checkpoints and a CUDA GPU -- see README.md.
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "Wan2.2"))

import wan  # noqa: E402  (vendored at inpainting/Wan2.2)
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS  # noqa: E402
from wan.utils.utils import save_video  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, required=True, help="first-frame conditioning image")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--negative_prompt", type=str, default="")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--task", type=str, default="ti2v-5B", choices=["ti2v-5B"])
    ap.add_argument("--size", type=str, default="1280*704", choices=list(SIZE_CONFIGS.keys()))
    ap.add_argument("--frame_num", type=int, default=None, help="defaults to the task config's frame_num (4n+1)")
    ap.add_argument("--fps", type=int, default=None, help="defaults to the task config's sample_fps")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--guide_scale", type=float, default=5.0)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument("--t5_cpu", action="store_true")
    ap.add_argument("--no_offload", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    cfg = WAN_CONFIGS[args.task]
    frame_num = args.frame_num or cfg.frame_num
    fps = args.fps or cfg.sample_fps

    img = Image.open(args.image).convert("RGB")

    pipe = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device_id,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=True,
    )

    video = pipe.generate(
        args.prompt,
        img=img,
        size=SIZE_CONFIGS[args.size],
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=frame_num,
        shift=args.shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.num_steps,
        guide_scale=args.guide_scale,
        n_prompt=args.negative_prompt,
        seed=args.seed,
        offload_model=not args.no_offload,
    )

    save_video(video.unsqueeze(0), save_file=str(args.output), fps=fps)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
