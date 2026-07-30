"""Generate the initial video: first-frame + text conditioned, via Wan2.2.

Thin CLI wrapper around Wan2.2's already-shipped `WanTI2V.i2v()` (reached via
`WanTI2V.generate(prompt, img=...)`), mirroring `Wan2.2/generate.py`'s own
`ti2v` branch. This is step 1 of the workflow: produce a starting video from
a first frame + prompt; step 2 is picking a frame range in it and correcting
it with `frame_range_edit.py`.

If `--cache_output` is given, generation instead goes through
`generate_i2v_with_step_cache()`, a project-local reimplementation of
`WanTI2V.i2v()` (Wan2.2/wan/textimage2video.py) that saves every denoising
step's latent to disk -- so a deformed generation can later be inspected, or
(in a future script) re-denoised from an earlier step with a different
prompt, instead of restarting from scratch.

Run on a machine with the Wan2.2 checkpoints and a CUDA GPU -- see README.md.
"""

import argparse
import math
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "Wan2.2"))

import wan  # noqa: E402  (vendored at inpainting/Wan2.2)
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS  # noqa: E402
from wan.utils.fm_solvers import (  # noqa: E402
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402
from wan.utils.utils import best_output_size, masks_like, save_video  # noqa: E402


@contextmanager
def _noop():
    yield


def generate_i2v_with_step_cache(
    pipe: "wan.WanTI2V",
    input_prompt: str,
    img: Image.Image,
    step_callback: Callable[[int, torch.Tensor, torch.Tensor], None],
    max_area: int = 704 * 1280,
    frame_num: int = 121,
    shift: float = 5.0,
    sample_solver: str = "unipc",
    sampling_steps: int = 40,
    guide_scale: float = 5.0,
    n_prompt: str = "",
    seed: int = -1,
    offload_model: bool = True,
) -> torch.Tensor:
    """Reimplementation of `WanTI2V.i2v()` (Wan2.2/wan/textimage2video.py)

    that calls `step_callback(step_idx, timestep, latent)` after every
    denoising step, so callers can cache the trajectory. Wan2.2/wan/ is
    vendored and left untouched (see README.md); this mirrors the same
    project-level-reimplementation approach `frame_range_edit.py` uses for
    its own custom denoising loop, rather than patching the vendored class.

    Note: `sample_scheduler` (UniPC/DPM++) is a stateful multistep solver --
    it keeps a short rolling history of previous model outputs internally.
    `step_callback` only receives the latent, not that internal history, so
    resuming denoising from a cached step later will be a close but not
    bit-exact approximation of an uninterrupted run for the first step or two
    after the resume point.
    """
    device = pipe.device

    # preprocess (identical to WanTI2V.i2v)
    ih, iw = img.height, img.width
    dh, dw = pipe.patch_size[1] * pipe.vae_stride[1], pipe.patch_size[2] * pipe.vae_stride[2]
    ow, oh = best_output_size(iw, ih, dw, dh, max_area)

    scale = max(ow / iw, oh / ih)
    img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

    x1 = (img.width - ow) // 2
    y1 = (img.height - oh) // 2
    img = img.crop((x1, y1, x1 + ow, y1 + oh))
    assert img.width == ow and img.height == oh

    img_t = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device).unsqueeze(1)

    F = frame_num
    seq_len = ((F - 1) // pipe.vae_stride[0] + 1) * (oh // pipe.vae_stride[1]) * (
        ow // pipe.vae_stride[2]) // (pipe.patch_size[1] * pipe.patch_size[2])
    seq_len = int(math.ceil(seq_len / pipe.sp_size)) * pipe.sp_size

    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)
    noise = torch.randn(
        pipe.vae.model.z_dim, (F - 1) // pipe.vae_stride[0] + 1,
        oh // pipe.vae_stride[1],
        ow // pipe.vae_stride[2],
        dtype=torch.float32,
        generator=seed_g,
        device=device)

    if n_prompt == "":
        n_prompt = pipe.sample_neg_prompt

    if not pipe.t5_cpu:
        pipe.text_encoder.model.to(device)
        context = pipe.text_encoder([input_prompt], device)
        context_null = pipe.text_encoder([n_prompt], device)
        if offload_model:
            pipe.text_encoder.model.cpu()
    else:
        context = pipe.text_encoder([input_prompt], torch.device("cpu"))
        context_null = pipe.text_encoder([n_prompt], torch.device("cpu"))
        context = [t.to(device) for t in context]
        context_null = [t.to(device) for t in context_null]

    z = pipe.vae.encode([img_t])

    no_sync = getattr(pipe.model, "no_sync", _noop)

    with (
            torch.amp.autocast("cuda", dtype=pipe.param_dtype),
            torch.no_grad(),
            no_sync(),
    ):
        if sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=pipe.num_train_timesteps, shift=1, use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == "dpm++":
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=pipe.num_train_timesteps, shift=1, use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        latent = noise
        _, mask2 = masks_like([noise], zero=True)
        latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

        arg_c = {"context": [context[0]], "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}

        if offload_model or pipe.init_on_cpu:
            pipe.model.to(device)
            torch.cuda.empty_cache()

        for step_idx, t in enumerate(timesteps):
            latent_model_input = [latent.to(device)]
            timestep = torch.stack([t]).to(device)

            temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
            temp_ts = torch.cat([temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep])
            timestep_tok = temp_ts.unsqueeze(0)

            noise_pred_cond = pipe.model(latent_model_input, t=timestep_tok, **arg_c)[0]
            if offload_model:
                torch.cuda.empty_cache()
            noise_pred_uncond = pipe.model(latent_model_input, t=timestep_tok, **arg_null)[0]
            if offload_model:
                torch.cuda.empty_cache()
            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False, generator=seed_g)[0]
            latent = temp_x0.squeeze(0)
            latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

            step_callback(step_idx, t, latent)

            x0 = [latent]

        if offload_model:
            pipe.model.cpu()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if pipe.rank == 0:
            videos = pipe.vae.decode(x0)

    del noise, latent, x0, sample_scheduler
    if offload_model:
        torch.cuda.synchronize()

    return videos[0] if pipe.rank == 0 else None


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
    ap.add_argument(
        "--cache_output", type=Path, default=None,
        help="if given, also save every denoising step's latent (keyed by step index) to this .pt path, "
             "via torch.save -- for later inspection or resuming denoising from an earlier step")
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

    if args.cache_output is not None:
        trajectory_tape = {}

        def capture_step(step_idx, timestep, latent):
            trajectory_tape[step_idx] = {
                "timestep": timestep.item(),
                "latent": latent.detach().cpu().clone(),
            }

        video = generate_i2v_with_step_cache(
            pipe,
            args.prompt,
            img=img,
            step_callback=capture_step,
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

        torch.save(trajectory_tape, args.cache_output)
        print(f"wrote {args.cache_output} ({len(trajectory_tape)} steps)")
    else:
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
