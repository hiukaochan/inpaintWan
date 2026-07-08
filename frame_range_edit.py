"""Regenerate an interior frame range of an existing video with Wan2.2.

Given a video and a frame range [A, B], this crops a local window around it,
regenerates pixel frames [A-1, B+1] conditioned on a new text prompt, and
splices the result back into the source video. Frames A-2 and B+2 (and
everything further out) are frozen context anchors and are never modified --
the model only sees them as clean conditioning, and the final splice copies
everything outside [A-1, B+1] byte-for-byte from the source regardless of
what the model produced there.

The conditioning mechanism generalizes the first-frame-only trick already
shipped in `Wan2.2/wan/textimage2video.py::WanTI2V.i2v` (paste the clean VAE
latent into frozen positions -- both the initial latent and after every
denoising step, and zero out the per-token diffusion timestep at those
positions) to an arbitrary interior sub-range instead of just frame 0.

Run on a machine with the Wan2.2 checkpoints and a CUDA GPU -- see README.md.
"""

import argparse
import math
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "Wan2.2"))

import wan  # noqa: E402  (vendored at inpainting/Wan2.2)
from wan.configs import WAN_CONFIGS  # noqa: E402
from wan.utils.fm_solvers import (  # noqa: E402
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402

from frame_mapping import build_latent_regen_mask, build_regeneration_window, FrameRangeWindow

SPATIAL_MULTIPLE = 32  # patch_size[1]*vae_stride[1] == patch_size[2]*vae_stride[2] for ti2v-5B


@contextmanager
def _noop():
    yield


def read_video_frames(path: Path) -> tuple[np.ndarray, float]:
    """Read all frames as an (T, H, W, 3) uint8 RGB array + fps."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"no frames read from {path}")
    return np.stack(frames), fps


def write_video_frames(path: Path, frames: np.ndarray, fps: float) -> None:
    h, w = frames.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def round_down_to_multiple(x: int, m: int) -> int:
    return max(m, (x // m) * m)


def resize_frames(frames: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_LANCZOS4) for f in frames])


def frames_to_vae_input(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    """(T, H, W, 3) uint8 -> (3, T, H, W) float in [-1, 1]."""
    t = torch.from_numpy(frames).to(device=device, dtype=torch.float32)
    t = t.permute(3, 0, 1, 2) / 255.0
    return t.sub_(0.5).div_(0.5)


def vae_output_to_frames(video: torch.Tensor) -> np.ndarray:
    """(3, T, H, W) float in [-1, 1] -> (T, H, W, 3) uint8."""
    video = video.clamp_(-1, 1).add_(1).div_(2).mul_(255)
    return video.permute(1, 2, 3, 0).round().to(torch.uint8).cpu().numpy()


def splice_edited_frames(full_frames: np.ndarray, edited_window: np.ndarray, window: FrameRangeWindow) -> np.ndarray:
    out = full_frames.copy()
    local_start = window.local(window.edit_start)
    local_end = window.local(window.edit_end)
    out[window.edit_start:window.edit_end + 1] = edited_window[local_start:local_end + 1]
    return out


def assert_outside_range_intact(original: np.ndarray, edited: np.ndarray, edit_start: int, edit_end: int) -> None:
    before_ok = np.array_equal(original[:edit_start], edited[:edit_start])
    after_ok = np.array_equal(original[edit_end + 1:], edited[edit_end + 1:])
    if not (before_ok and after_ok):
        raise AssertionError("frames outside the edit range were modified -- splice logic bug")


class WanFrameRangeEditor:
    """Wraps a `wan.WanTI2V` pipeline to regenerate an interior frame range
    of an existing video, conditioned on frozen context frames further out.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        task: str = "ti2v-5B",
        device_id: int = 0,
        t5_cpu: bool = False,
        convert_model_dtype: bool = True,
        offload_model: bool = True,
    ):
        cfg = WAN_CONFIGS[task]
        self.pipe = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=checkpoint_dir,
            device_id=device_id,
            t5_cpu=t5_cpu,
            convert_model_dtype=convert_model_dtype,
        )
        self.offload_model = offload_model
        self.device = self.pipe.device

    def edit(
        self,
        window_frames: np.ndarray,
        regen_mask: list,
        prompt: str,
        n_prompt: str = "",
        sampling_steps: int = 50,
        shift: float = 5.0,
        guide_scale: float = 5.0,
        sample_solver: str = "unipc",
        seed: int = -1,
        noise_strength: float = 0.6,
    ) -> np.ndarray:
        """`noise_strength` in (0, 1]: how much of the regenerate-mask region

        is re-noised before denoising, SDEdit/img2img-style. 1.0 regenerates
        that region from scratch (pure noise, like WanTI2V.i2v's boundary-only
        conditioning); lower values start from the region's own original
        content partially noised, so the edit corrects the existing scene
        (per the new prompt) instead of hallucinating a new one. Frozen
        (mask=0) positions are always exactly `z` regardless of this value.
        """
        pipe = self.pipe
        device = self.device

        video_in = frames_to_vae_input(window_frames, device)
        z = pipe.vae.encode([video_in])[0]  # (z_dim, T_latent, H_latent, W_latent)

        mask = torch.tensor(regen_mask, dtype=z.dtype, device=device)
        if mask.shape[0] != z.shape[1]:
            raise ValueError(
                f"regen_mask has {mask.shape[0]} entries but the VAE produced "
                f"{z.shape[1]} latent frames -- pixel/latent mapping mismatch")
        mask2 = mask.view(1, -1, 1, 1).expand_as(z)  # 1 = regenerate, 0 = frozen context

        seed = seed if seed >= 0 else torch.seed()
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)

        noise = torch.randn(*z.shape, dtype=torch.float32, device=device, generator=seed_g)

        ph, pw = pipe.patch_size[1], pipe.patch_size[2]
        seq_len = math.ceil((z.shape[1] * z.shape[2] * z.shape[3]) / (ph * pw) / pipe.sp_size) * pipe.sp_size

        if n_prompt == "":
            n_prompt = pipe.sample_neg_prompt
        if not pipe.t5_cpu:
            pipe.text_encoder.model.to(device)
            context = pipe.text_encoder([prompt], device)
            context_null = pipe.text_encoder([n_prompt], device)
            if self.offload_model:
                pipe.text_encoder.model.cpu()
        else:
            context = pipe.text_encoder([prompt], torch.device("cpu"))
            context_null = pipe.text_encoder([n_prompt], torch.device("cpu"))
            context = [c.to(device) for c in context]
            context_null = [c.to(device) for c in context_null]

        if not (0.0 < noise_strength <= 1.0):
            raise ValueError(f"noise_strength must be in (0, 1], got {noise_strength}")

        if sample_solver == "unipc":
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=pipe.num_train_timesteps, shift=1, use_dynamic_shifting=False)
            scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
        elif sample_solver == "dpm++":
            scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=pipe.num_train_timesteps, shift=1, use_dynamic_shifting=False)
            sigmas = get_sampling_sigmas(sampling_steps, shift)
            retrieve_timesteps(scheduler, device=device, sigmas=sigmas)
        else:
            raise NotImplementedError(f"unsupported solver: {sample_solver}")

        # SDEdit: only re-noise the regenerate-mask region, and only down to
        # an intermediate sigma (not full noise) -- so the edit starts from
        # the region's own original content, letting the prompt correct it
        # rather than replace it. Frozen positions always start at clean z.
        start_idx = round(len(scheduler.timesteps) * (1.0 - noise_strength))
        start_idx = min(start_idx, len(scheduler.timesteps) - 1)
        sigma0 = scheduler.sigmas[start_idx].to(device=device, dtype=z.dtype)
        timesteps = scheduler.timesteps[start_idx:]

        regen_init = (1.0 - sigma0) * z + sigma0 * noise
        latent = (1.0 - mask2) * z + mask2 * regen_init

        arg_c = {"context": context, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}

        no_sync = getattr(pipe.model, "no_sync", _noop)
        if self.offload_model or pipe.init_on_cpu:
            pipe.model.to(device)
            torch.cuda.empty_cache()

        with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad(), no_sync():
            for t in timesteps:
                latent_model_input = [latent]
                timestep = torch.stack([t]).to(device)

                # per-token timestep: 0 at frozen positions (tells the DiT
                # they're already clean), real t elsewhere -- same trick as
                # WanTI2V.i2v's temp_ts construction, generalized to our mask.
                temp_ts = (mask2[0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep])
                timestep_tok = temp_ts.unsqueeze(0)

                noise_pred_cond = pipe.model(latent_model_input, t=timestep_tok, **arg_c)[0]
                if self.offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = pipe.model(latent_model_input, t=timestep_tok, **arg_null)[0]
                if self.offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                temp_x0 = scheduler.step(
                    noise_pred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False, generator=seed_g)[0]
                latent = temp_x0.squeeze(0)
                latent = (1.0 - mask2) * z + mask2 * latent

        if self.offload_model:
            pipe.model.cpu()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        video = pipe.vae.decode([latent])[0]
        return vae_output_to_frames(video)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_video", type=Path, required=True)
    ap.add_argument("--start_frame", type=int, required=True, help="A: first frame of the range to edit")
    ap.add_argument("--end_frame", type=int, required=True, help="B: last frame of the range to edit")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument(
        "--prompt_file", type=Path, default=None,
        help="one prompt per line; runs the edit once per prompt so results are directly comparable")
    ap.add_argument("--negative_prompt", type=str, default="")
    ap.add_argument("--output", type=Path, required=True, help="output path; used as a stem when sweeping prompts")
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--task", type=str, default="ti2v-5B", choices=["ti2v-5B"])
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--guide_scale", type=float, default=5.0)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument(
        "--noise_strength", type=float, default=0.6,
        help="SDEdit strength in (0, 1]: how much of the edit region is re-noised/regenerated. "
             "1.0 = regenerate from scratch; lower values correct the existing content instead of replacing it")
    ap.add_argument(
        "--context_blocks", type=int, default=1,
        help="number of guaranteed-frozen 4-frame latent blocks required immediately after the edit region")
    ap.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument("--t5_cpu", action="store_true")
    ap.add_argument("--no_offload", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.prompt and not args.prompt_file:
        raise ValueError("must pass --prompt or --prompt_file")
    prompts = [args.prompt] if args.prompt else [
        line.strip() for line in args.prompt_file.read_text().splitlines() if line.strip()
    ]

    full_frames, fps = read_video_frames(args.input_video)
    window = build_regeneration_window(
        args.start_frame, args.end_frame, len(full_frames), context_blocks=args.context_blocks)
    regen_mask = build_latent_regen_mask(window)

    raw_window_frames = full_frames[window.window_start:window.window_end + 1]
    orig_h, orig_w = raw_window_frames.shape[1:3]
    valid_w = round_down_to_multiple(orig_w, SPATIAL_MULTIPLE)
    valid_h = round_down_to_multiple(orig_h, SPATIAL_MULTIPLE)
    model_input_frames = resize_frames(raw_window_frames, valid_w, valid_h)

    editor = WanFrameRangeEditor(
        checkpoint_dir=args.ckpt_dir, task=args.task, device_id=args.device_id,
        t5_cpu=args.t5_cpu, offload_model=not args.no_offload)

    for prompt in prompts:
        edited_window = editor.edit(
            model_input_frames, regen_mask, prompt,
            n_prompt=args.negative_prompt, sampling_steps=args.num_steps,
            shift=args.shift, guide_scale=args.guide_scale,
            sample_solver=args.sample_solver, seed=args.seed,
            noise_strength=args.noise_strength)
        edited_window = resize_frames(edited_window, orig_w, orig_h)

        full_edited = splice_edited_frames(full_frames, edited_window, window)
        assert_outside_range_intact(full_frames, full_edited, window.edit_start, window.edit_end)

        out_path = args.output
        if len(prompts) > 1:
            out_path = args.output.with_name(f"{args.output.stem}_{slugify(prompt)}{args.output.suffix}")
        write_video_frames(out_path, full_edited, fps)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
