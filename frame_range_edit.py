"""Regenerate an interior frame range of an existing video with Wan2.2.

Given a video and a frame range [A, B], this crops a local window around it,
regenerates pixel frames [A-1, B+1] conditioned on a new text prompt, and
splices the result back into the source video. Frames A-2 and B+2 (and
everything further out) are frozen context anchors and are never modified --
the model only sees them as clean conditioning, and the final splice copies
everything outside [A-1, B+1] byte-for-byte from the source regardless of
what the model produced there.

`A == 0` and `B == the last frame of the video` are supported as exact
boundary cases -- there's no anchor before frame 0 or after the last frame
to freeze, so none is required on that side. See
`frame_mapping.py::build_regeneration_window` for the details; the latter
case pads the model's input with replicated edge frames purely to satisfy
the VAE's frame-count alignment, which are discarded before the final
splice and never appear in the output.

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

from frame_mapping import (
    build_latent_regen_mask,
    build_regeneration_window,
    build_spatial_latent_regen_mask,
    FrameRangeWindow,
)

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


def resize_mask_frames(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    """Nearest-neighbor resize for a bool (T, H, W) mask -- masks are categorical,
    so Lanczos (used for pixel frames) would blur edges into fractional values."""
    resized = np.stack([
        cv2.resize(f.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) for f in mask
    ])
    return resized.astype(bool)


def parse_box(box: str) -> tuple[int, int, int, int]:
    parts = [int(p) for p in box.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--mask_box must be 'x1,y1,x2,y2', got {box!r}")
    x1, y1, x2, y2 = parts
    if not (x1 < x2 and y1 < y2):
        raise ValueError(f"--mask_box must have x1<x2 and y1<y2, got {box!r}")
    return x1, y1, x2, y2


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
        regen_mask: np.ndarray,
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
        if mask.shape != z.shape[1:]:
            raise ValueError(
                f"regen_mask has shape {tuple(mask.shape)} but the VAE produced latent "
                f"shape {tuple(z.shape[1:])} -- pixel/latent mapping mismatch")
        mask2 = mask.unsqueeze(0).expand_as(z)  # 1 = regenerate, 0 = frozen context

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

    def edit_from_cache(
        self,
        cache_path: Path,
        regen_mask: np.ndarray,
        prompt: str,
        n_prompt: str = "",
        guide_scale: float = 5.0,
        seed: int = -1,
        resume_noise_pct: float = 0.4,
    ) -> np.ndarray:
        """Resume denoising from a real generation-time latent instead of

        `edit()`'s SDEdit fresh-noise formula. `cache_path` must be a `.pt`
        file written by `generate_initial_video.py --cache_output` for the
        *exact* video passed to `main()` as `--input_video` -- the cache no
        longer matches a video's pixels once any other edit has been
        applied to it.

        Unlike `edit()`, this always operates on the whole cached
        video-length latent, never a cropped window: Wan2.2's causal VAE has
        exactly one singleton latent frame, at true pixel 0 of the whole
        video, so a re-encoded crop's own "position 0" doesn't correspond to
        anything in the originally cached latent unless the crop starts at
        true frame 0. `regen_mask` must therefore be shaped for the whole
        cached video's latent, not a cropped window (see `main()`).
        """
        pipe = self.pipe
        device = self.device

        trajectory_tape = torch.load(cache_path, map_location="cpu")
        if "_meta" not in trajectory_tape:
            raise ValueError(
                f"{cache_path} has no '_meta' entry -- re-generate it with the current "
                f"generate_initial_video.py --cache_output")
        meta = trajectory_tape["_meta"]
        num_steps = len(trajectory_tape) - 1  # exclude "_meta"
        if num_steps != meta["sampling_steps"]:
            raise ValueError(
                f"cache has {num_steps} step entries but _meta says sampling_steps="
                f"{meta['sampling_steps']} -- cache file looks corrupted or truncated")

        z = trajectory_tape[num_steps - 1]["latent"].to(device=device, dtype=torch.float32)

        mask = torch.tensor(regen_mask, dtype=z.dtype, device=device)
        if mask.shape != z.shape[1:]:
            raise ValueError(
                f"regen_mask has shape {tuple(mask.shape)} but the cached latent has shape "
                f"{tuple(z.shape[1:])} -- regen_mask must cover the whole cached video, and "
                f"--input_video must be the exact video this cache was generated for")
        mask2 = mask.unsqueeze(0).expand_as(z)

        seed = seed if seed >= 0 else torch.seed()
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)

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

        if not (0.0 < resume_noise_pct <= 1.0):
            raise ValueError(f"resume_noise_pct must be in (0, 1], got {resume_noise_pct}")

        sample_solver = meta["sample_solver"]
        if sample_solver == "unipc":
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=pipe.num_train_timesteps, shift=1, use_dynamic_shifting=False)
            scheduler.set_timesteps(meta["sampling_steps"], device=device, shift=meta["shift"])
        elif sample_solver == "dpm++":
            scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=pipe.num_train_timesteps, shift=1, use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(meta["sampling_steps"], meta["shift"])
            retrieve_timesteps(scheduler, device=device, sigmas=sampling_sigmas)
        else:
            raise NotImplementedError(f"unsupported solver: {sample_solver}")

        # sigmas[i+1] is (approximately) the noise level of the cached latent
        # at step i, i.e. the state produced right after that step ran
        sigmas = scheduler.sigmas.to(device=device, dtype=z.dtype)
        candidates = sigmas[1:num_steps + 1]
        step_idx = int(torch.argmin((candidates - resume_noise_pct).abs()).item())
        step_idx = min(step_idx, num_steps - 2)  # keep at least one step to run

        print(f"[edit_from_cache] resume_noise_pct={resume_noise_pct} -> cached step {step_idx} "
              f"(timestep={trajectory_tape[step_idx]['timestep']}, sigma~{candidates[step_idx].item():.4f}), "
              f"{num_steps - (step_idx + 1)} steps remaining")

        latent = trajectory_tape[step_idx]["latent"].to(device=device, dtype=torch.float32)
        latent = (1.0 - mask2) * z + mask2 * latent
        timesteps = scheduler.timesteps[step_idx + 1:]

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


def full_video_edit_bounds(a: int, b: int, video_len: int) -> tuple[int, int]:
    """edit_start/edit_end for `edit_from_cache`'s whole-video "window".

    Same `[a-1, b+1]`-unless-at-a-boundary rule `build_regeneration_window`
    uses, minus its crop-edge `CONTEXT_MARGIN`/`context_blocks` requirements
    -- there's no crop here, so no minimum-context validation is needed.
    """
    if a > b:
        raise ValueError("a must be <= b")
    if not (0 <= a < video_len) or not (0 <= b < video_len):
        raise ValueError(f"a and b must be valid frame indices in [0, {video_len - 1}]")
    edit_start = 0 if a == 0 else a - 1
    edit_end = (video_len - 1) if b == video_len - 1 else b + 1
    return edit_start, edit_end


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_video", type=Path, required=True)
    ap.add_argument(
        "--start_frame", type=int, required=True,
        help="A: first frame of the range to edit; 0 is allowed (edits through the start of the video)")
    ap.add_argument(
        "--end_frame", type=int, required=True,
        help="B: last frame of the range to edit; the video's last frame index is allowed "
             "(edits through the end of the video)")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument(
        "--prompt_file", type=Path, default=None,
        help="one prompt per line; runs the edit once per prompt so results are directly comparable")
    ap.add_argument("--negative_prompt", type=str, default="")
    ap.add_argument("--output", type=Path, required=True, help="output path; used as a stem when sweeping prompts")
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--task", type=str, default="ti2v-5B", choices=["ti2v-5B"])
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--num_steps", type=int, default=20)
    ap.add_argument("--guide_scale", type=float, default=5.0)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument(
        "--noise_strength", type=float, default=0.6,
        help="how much of the edit region is regenerated, in (0, 1]. Without --cache_path: SDEdit "
             "strength -- 1.0 regenerates from scratch, lower values re-noise the existing content less. "
             "With --cache_path: how far back into the real generation trajectory to resume from -- "
             "e.g. 0.4 resumes from the cached step closest to 40% noise, then re-denoises with --prompt")
    ap.add_argument(
        "--cache_path", type=Path, default=None,
        help="resume denoising from a per-step latent cache written by generate_initial_video.py "
             "--cache_output, instead of SDEdit's fresh-noise re-noising. --input_video must be the "
             "exact video that cache was generated for -- it no longer matches after any other edit. "
             "Operates on the whole video's latent (no window crop), so --context_blocks is unused")
    ap.add_argument(
        "--context_blocks", type=int, default=1,
        help="number of guaranteed-frozen 4-frame latent blocks required immediately after the edit "
             "region; unused with --cache_path (no crop, so no crop-edge context is needed)")
    ap.add_argument(
        "--edit_mode", type=str, default="full", choices=["full", "spatial"],
        help="'full' regenerates entire frames in the range (default). 'spatial' regenerates only "
             "the pixels inside --mask_box, keeping the rest of each frame biased toward its original "
             "content (soft latent-level masking, not a byte-exact guarantee)")
    ap.add_argument(
        "--mask_box", type=str, default=None,
        help="'x1,y1,x2,y2' in source-video pixel coordinates; the region to regenerate in --edit_mode "
             "spatial. Required iff --edit_mode is 'spatial'")
    ap.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument("--t5_cpu", action="store_true")
    ap.add_argument("--no_offload", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.prompt and not args.prompt_file:
        raise ValueError("must pass --prompt or --prompt_file")
    if args.edit_mode == "spatial" and args.mask_box is None:
        raise ValueError("--edit_mode spatial requires --mask_box")
    if args.edit_mode == "full" and args.mask_box is not None:
        raise ValueError("--mask_box is only used with --edit_mode spatial")
    prompts = [args.prompt] if args.prompt else [
        line.strip() for line in args.prompt_file.read_text().splitlines() if line.strip()
    ]

    full_frames, fps = read_video_frames(args.input_video)
    cfg = WAN_CONFIGS[args.task]

    editor = WanFrameRangeEditor(
        checkpoint_dir=args.ckpt_dir, task=args.task, device_id=args.device_id,
        t5_cpu=args.t5_cpu, offload_model=not args.no_offload)

    if args.cache_path is not None:
        # Whole-video path: no crop, no resize -- the cached latent already
        # covers the video at its native (VAE-aligned) resolution.
        edit_start, edit_end = full_video_edit_bounds(args.start_frame, args.end_frame, len(full_frames))
        window = FrameRangeWindow(0, len(full_frames) - 1, edit_start, edit_end, pad_end=0)
        temporal_mask = build_latent_regen_mask(window)

        h, w = full_frames.shape[1:3]
        if args.edit_mode == "spatial":
            x1, y1, x2, y2 = parse_box(args.mask_box)
            pixel_mask = np.zeros((full_frames.shape[0], h, w), dtype=bool)
            pixel_mask[:, y1:y2, x1:x2] = True
            regen_mask = build_spatial_latent_regen_mask(
                window, pixel_mask, vae_spatial_stride=cfg.vae_stride[1], patch_spatial=cfg.patch_size[1])
        else:
            h_latent, w_latent = h // cfg.vae_stride[1], w // cfg.vae_stride[2]
            regen_mask = np.broadcast_to(
                np.array(temporal_mask, dtype=bool)[:, None, None], (len(temporal_mask), h_latent, w_latent))

        for prompt in prompts:
            edited_video = editor.edit_from_cache(
                args.cache_path, regen_mask, prompt,
                n_prompt=args.negative_prompt, guide_scale=args.guide_scale,
                seed=args.seed, resume_noise_pct=args.noise_strength)

            full_edited = splice_edited_frames(full_frames, edited_video, window)
            assert_outside_range_intact(full_frames, full_edited, window.edit_start, window.edit_end)

            out_path = args.output
            if len(prompts) > 1:
                out_path = args.output.with_name(f"{args.output.stem}_{slugify(prompt)}{args.output.suffix}")
            write_video_frames(out_path, full_edited, fps)
            print(f"wrote {out_path}")
        return

    window = build_regeneration_window(
        args.start_frame, args.end_frame, len(full_frames), context_blocks=args.context_blocks)
    temporal_mask = build_latent_regen_mask(window)

    raw_window_frames = full_frames[window.window_start:window.window_end + 1]
    if window.pad_end > 0:
        # No real footage exists beyond window_end (edit range reaches the
        # true end of the video) -- pad with the last real frame, replicated,
        # purely to satisfy the VAE's 4n+1 alignment requirement. Discarded
        # before the final splice; never appears in the output.
        pad_block = np.repeat(raw_window_frames[-1:], window.pad_end, axis=0)
        raw_window_frames = np.concatenate([raw_window_frames, pad_block], axis=0)
    orig_h, orig_w = raw_window_frames.shape[1:3]
    valid_w = round_down_to_multiple(orig_w, SPATIAL_MULTIPLE)
    valid_h = round_down_to_multiple(orig_h, SPATIAL_MULTIPLE)
    model_input_frames = resize_frames(raw_window_frames, valid_w, valid_h)

    if args.edit_mode == "spatial":
        x1, y1, x2, y2 = parse_box(args.mask_box)
        pixel_mask = np.zeros((raw_window_frames.shape[0], orig_h, orig_w), dtype=bool)
        pixel_mask[:, y1:y2, x1:x2] = True
        pixel_mask = resize_mask_frames(pixel_mask, valid_w, valid_h)
        regen_mask = build_spatial_latent_regen_mask(
            window, pixel_mask, vae_spatial_stride=cfg.vae_stride[1], patch_spatial=cfg.patch_size[1])
    else:
        h_latent, w_latent = valid_h // cfg.vae_stride[1], valid_w // cfg.vae_stride[2]
        regen_mask = np.broadcast_to(
            np.array(temporal_mask, dtype=bool)[:, None, None], (len(temporal_mask), h_latent, w_latent))

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
