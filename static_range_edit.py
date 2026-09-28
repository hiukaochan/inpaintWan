"""Regenerate a frame range while pinning chosen regions to a pre-drift anchor.

Motivating failure: Wan2.2 generates "robot arm closes the drawer", but the
whole cabinet drifts across the frame instead of staying bolted to the scene.
Only the drawer front should move. A correction *prompt* has almost no
purchase on that -- "the cabinet stays still" is a claim about pixel geometry
-- so this applies the constraint directly to the latent during denoising,
following SG-I2V's idea of steering the sampling process rather than the
weights (zero-shot; the model is never fine-tuned).

Relationship to `frame_range_edit.py`: same window/masking/splice machinery
(imported, not copied) and the same two entry points -- SDEdit-style partial
re-noising, or resuming from a `generate_initial_video.py --cache_output`
latent cache. The only addition is that at high-noise steps, the boxed
regions are pulled back toward how they looked in a frozen anchor frame from
before the drift. `frame_range_edit.py` itself is untouched and remains the
baseline to compare against.

The correction is a *soft delta*, not an overwrite -- see
`latent_paste.write_delta` for why that distinction carries the whole method.

Run on a machine with the Wan2.2 checkpoints and a CUDA GPU -- see README.md.
"""

import argparse
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "Wan2.2"))

from wan.configs import WAN_CONFIGS  # noqa: E402
from wan.utils.fm_solvers import (  # noqa: E402
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402

from frame_mapping import (  # noqa: E402
    FrameRangeWindow,
    anchor_latent_frame,
    build_feathered_box_weight,
    build_latent_regen_mask,
    build_freeze_regen_mask,
    build_regeneration_window,
    latent_frame_to_pixel_range,
    linear_trajectory,
    load_trajectory_npy,
    resample_trajectory,
    snap_box_outward,
    trajectory_to_latent_plan,
)
from frame_range_edit import (  # noqa: E402
    SPATIAL_MULTIPLE,
    WanFrameRangeEditor,
    _noop,
    assert_outside_range_intact,
    frames_to_vae_input,
    full_video_edit_bounds,
    parse_box,
    read_video_frames,
    resize_frames,
    round_down_to_multiple,
    slugify,
    splice_edited_frames,
    vae_output_to_frames,
    write_video_frames,
)
from latent_paste import (  # noqa: E402
    fft_restore,
    pin_anchor_content,
    target_latent_frames,
    weight_to_tensor,
    write_delta,
    write_delta_trajectory,
)
from noise_inversion import (  # noqa: E402
    EulerFlowScheduler,
    box_weight,
    copy_region_across_frames,
    euler_invert,
    shift_inverted_region,
)


class PasteConfig:
    """What to correct in the latent, how hard, and for how long.

    Carries both modes, which compose: `weight_hw` pins static boxes in place,
    `plan` moves a box along a trajectory. A run may use either or both -- e.g.
    hold the cabinet still *and* move the gripper -- since both are latent
    writes gated by the same sigma window.
    """

    def __init__(
        self,
        anchor_idx: int,
        target_frames: list[int],
        weight_hw: np.ndarray | None = None,
        plan: list | None = None,
        strength: float = 1.0,
        object_strength: float = 1.0,
        vacated_fill: float = 1.0,
        sigma_range: tuple[float, float] = (0.30, 1.00),
        max_steps: int = 4,
        fft_ratio: float | None = 0.5,
    ):
        lo, hi = sigma_range
        if lo > hi:
            raise ValueError(f"sigma_range must be (low, high), got {sigma_range}")
        if weight_hw is None and not plan:
            raise ValueError("PasteConfig needs static boxes, a trajectory plan, or both")
        self.weight_hw = weight_hw
        self.plan = plan or []
        self.anchor_idx = anchor_idx
        self.target_frames = target_frames
        self.strength = strength
        self.object_strength = object_strength
        self.vacated_fill = vacated_fill
        self.sigma_lo, self.sigma_hi = lo, hi
        self.max_steps = max_steps
        self.fft_ratio = fft_ratio
        self.applied = 0

    def written_frames(self) -> list[int]:
        """Latent frames either mode touches -- the scope for `fft_restore`."""
        frames = set(self.plan and [s.frame for s in self.plan] or [])
        if self.weight_hw is not None:
            frames |= set(self.target_frames)
        return sorted(frames)

    def should_apply(self, sigma: float) -> bool:
        """Only at high noise, and only for the first `max_steps` such steps.

        Both bounds are load-bearing. Editing late in the schedule wrecks
        visual quality (SG-I2V Figs. 10/15/16 -- their Fig. 15 shows the last
        frame dissolving into coloured blobs), and `write_delta`'s `x0 ~ z`
        approximation is only accurate while the latent still encodes
        something close to the source video. Stopping after a few steps leaves
        the remaining ones free to harmonise the seam and re-assert anything
        the correction stepped on, such as an occluding gripper.
        """
        return self.applied < self.max_steps and self.sigma_lo <= sigma <= self.sigma_hi


class StaticPasteEditor(WanFrameRangeEditor):
    """`WanFrameRangeEditor` plus a latent-space anchor correction in the loop."""

    def _encode_prompts(self, prompt: str, n_prompt: str):
        pipe = self.pipe
        if n_prompt == "":
            n_prompt = pipe.sample_neg_prompt
        if not pipe.t5_cpu:
            pipe.text_encoder.model.to(self.device)
            context = pipe.text_encoder([prompt], self.device)
            context_null = pipe.text_encoder([n_prompt], self.device)
            if self.offload_model:
                pipe.text_encoder.model.cpu()
        else:
            context = pipe.text_encoder([prompt], torch.device("cpu"))
            context_null = pipe.text_encoder([n_prompt], torch.device("cpu"))
            context = [c.to(self.device) for c in context]
            context_null = [c.to(self.device) for c in context_null]
        return context, context_null

    def _build_scheduler(self, sample_solver: str, sampling_steps: int, shift: float):
        pipe = self.pipe
        if sample_solver == "unipc":
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=pipe.num_train_timesteps, shift=1, use_dynamic_shifting=False)
            scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
        elif sample_solver == "dpm++":
            scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=pipe.num_train_timesteps, shift=1, use_dynamic_shifting=False)
            sigmas = get_sampling_sigmas(sampling_steps, shift)
            retrieve_timesteps(scheduler, device=self.device, sigmas=sigmas)
        else:
            raise NotImplementedError(f"unsupported solver: {sample_solver}")
        return scheduler

    @contextmanager
    def _sampling(self):
        """Model on the GPU, autocast, no_grad -- the setup every sampling loop
        here needs. Re-entrant, so inversion and the denoise that follows it
        share one model load instead of offloading in between."""
        if getattr(self, "_sampling_active", False):
            yield
            return
        pipe = self.pipe
        no_sync = getattr(pipe.model, "no_sync", _noop)
        if self.offload_model or pipe.init_on_cpu:
            pipe.model.to(self.device)
            torch.cuda.empty_cache()
        self._sampling_active = True
        try:
            with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad(), no_sync():
                yield
        finally:
            self._sampling_active = False
        if self.offload_model:
            pipe.model.cpu()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _velocity(self, latent, t, mask2, seq_len, context, context_null, guide_scale):
        """One CFG velocity prediction, with the per-token timestep that tells
        the DiT mask-0 positions are clean. Shared by the denoise loop and
        inversion so both call the model identically; guide_scale 1 skips the
        unconditional pass (the same result, half the cost)."""
        pipe = self.pipe
        timestep = torch.stack([t]).to(self.device)
        temp_ts = (mask2[0][:, ::2, ::2] * timestep).flatten()
        temp_ts = torch.cat([temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep])
        timestep_tok = temp_ts.unsqueeze(0)

        noise_pred_cond = pipe.model([latent], t=timestep_tok, context=context, seq_len=seq_len)[0]
        if self.offload_model:
            torch.cuda.empty_cache()
        if guide_scale == 1.0:
            return noise_pred_cond
        noise_pred_uncond = pipe.model([latent], t=timestep_tok, context=context_null, seq_len=seq_len)[0]
        if self.offload_model:
            torch.cuda.empty_cache()
        return noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

    def _denoise(
        self,
        latent: torch.Tensor,
        z: torch.Tensor,
        mask2: torch.Tensor,
        scheduler,
        first_step: int,
        context,
        context_null,
        seq_len: int,
        guide_scale: float,
        seed_g: torch.Generator,
        paste: PasteConfig | None,
        z_frozen: torch.Tensor | None = None,
        post_step=None,
    ) -> torch.Tensor:
        """The `frame_range_edit.py` denoising loop with one insertion.

        `first_step` indexes into `scheduler.timesteps`; sigma for the step at
        absolute index `k` is `scheduler.sigmas[k]`, so selecting the paste
        window by sigma stays meaningful regardless of where the resume landed
        or how many sampling steps were requested.

        `post_step(j, latent) -> latent`, if given, runs after each step's
        re-hold, with `j` counting steps actually run (used by
        `--freeze_mode noise`).
        """
        timesteps = scheduler.timesteps[first_step:]
        sigmas = scheduler.sigmas
        # What mask-0 positions are held at. Defaults to `z` (the source
        # encoding), and is the anchor-pinned copy when --freeze_box is used;
        # `z` itself stays the source, which `write_delta` needs for its delta.
        z_hold = z if z_frozen is None else z_frozen

        weight_t = None
        if paste is not None and paste.weight_hw is not None:
            weight_t = weight_to_tensor(paste.weight_hw, latent.device, latent.dtype)

        with self._sampling():
            for j, t in enumerate(timesteps):
                abs_idx = first_step + j
                sigma = float(sigmas[min(abs_idx, len(sigmas) - 1)])

                if paste is not None and paste.should_apply(sigma):
                    before = latent
                    if weight_t is not None:
                        latent = write_delta(
                            latent, z, weight_t, paste.anchor_idx, paste.target_frames,
                            sigma, strength=paste.strength)
                    if paste.plan:
                        latent = write_delta_trajectory(
                            latent, z, paste.anchor_idx, paste.plan, sigma,
                            strength=paste.object_strength, vacated_fill=paste.vacated_fill)
                    written = paste.written_frames()
                    if paste.fft_ratio is not None and written:
                        # Restrict the frequency mix to the frames actually
                        # written: it is a global spatial operation, so running
                        # it over untouched frames would be pure round-trip
                        # error. Within a written frame the smearing is wanted
                        # -- it is what softens the box seam.
                        idx = torch.tensor(written, device=latent.device, dtype=torch.long)
                        latent[:, idx] = fft_restore(latent[:, idx], before[:, idx], d_s=paste.fft_ratio)
                    latent = (1.0 - mask2) * z_hold + mask2 * latent
                    paste.applied += 1
                    print(f"[paste] step {abs_idx} sigma={sigma:.4f} "
                          f"({paste.applied}/{paste.max_steps})")

                noise_pred = self._velocity(
                    latent, t, mask2, seq_len, context, context_null, guide_scale)

                temp_x0 = scheduler.step(
                    noise_pred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False, generator=seed_g)[0]
                latent = temp_x0.squeeze(0)
                latent = (1.0 - mask2) * z_hold + mask2 * latent
                if post_step is not None:
                    latent = post_step(j, latent)

        if paste is not None and paste.applied == 0:
            print(f"[paste] WARNING: no step fell in sigma range "
                  f"[{paste.sigma_lo}, {paste.sigma_hi}] -- nothing was corrected. "
                  f"Widen --paste_sigma_range or raise --noise_strength.")
        return latent

    def edit(
        self,
        window_frames: np.ndarray,
        regen_mask: np.ndarray,
        prompt: str,
        *,
        paste_spec: dict | None = None,
        freeze_spec: dict | None = None,
        n_prompt: str = "",
        sampling_steps: int = 20,
        shift: float = 5.0,
        guide_scale: float = 5.0,
        sample_solver: str = "unipc",
        seed: int = -1,
        noise_strength: float = 0.6,
    ) -> np.ndarray:
        """SDEdit path: re-noise the regenerate region, then denoise with the paste."""
        pipe = self.pipe
        z = pipe.vae.encode([frames_to_vae_input(window_frames, self.device)])[0]

        mask2, seq_len = self._mask_and_seq_len(regen_mask, z)
        seed_g = self._generator(seed)
        noise = torch.randn(*z.shape, dtype=torch.float32, device=self.device, generator=seed_g)

        context, context_null = self._encode_prompts(prompt, n_prompt)
        if not (0.0 < noise_strength <= 1.0):
            raise ValueError(f"noise_strength must be in (0, 1], got {noise_strength}")

        scheduler = self._build_scheduler(sample_solver, sampling_steps, shift)
        start_idx = min(round(len(scheduler.timesteps) * (1.0 - noise_strength)),
                        len(scheduler.timesteps) - 1)
        sigma0 = scheduler.sigmas[start_idx].to(device=self.device, dtype=z.dtype)
        z_frozen = _make_frozen(freeze_spec, regen_mask, z)
        z_hold = z if z_frozen is None else z_frozen
        latent = (1.0 - mask2) * z_hold + mask2 * ((1.0 - sigma0) * z + sigma0 * noise)

        paste = _make_paste(paste_spec, regen_mask, z)
        latent = self._denoise(
            latent, z, mask2, scheduler, start_idx, context, context_null,
            seq_len, guide_scale, seed_g, paste, z_frozen=z_frozen)
        return vae_output_to_frames(pipe.vae.decode([latent])[0])

    def edit_from_cache(
        self,
        cache_path: Path,
        regen_mask: np.ndarray,
        prompt: str,
        *,
        paste_spec: dict | None = None,
        freeze_spec: dict | None = None,
        n_prompt: str = "",
        guide_scale: float = 5.0,
        seed: int = -1,
        resume_noise_pct: float = 0.4,
    ) -> np.ndarray:
        """Resume from a real generation-time latent (see `frame_range_edit.py`
        for the cache contract) and denoise with the paste applied."""
        pipe = self.pipe
        trajectory_tape = torch.load(cache_path, map_location="cpu")
        if "_meta" not in trajectory_tape:
            raise ValueError(
                f"{cache_path} has no '_meta' entry -- re-generate it with the current "
                f"generate_initial_video.py --cache_output")
        meta = trajectory_tape["_meta"]
        num_steps = len(trajectory_tape) - 1
        if num_steps != meta["sampling_steps"]:
            raise ValueError(
                f"cache has {num_steps} step entries but _meta says sampling_steps="
                f"{meta['sampling_steps']} -- cache file looks corrupted or truncated")

        z = trajectory_tape[num_steps - 1]["latent"].to(device=self.device, dtype=torch.float32)
        mask2, seq_len = self._mask_and_seq_len(regen_mask, z)
        seed_g = self._generator(seed)
        context, context_null = self._encode_prompts(prompt, n_prompt)

        if not (0.0 < resume_noise_pct <= 1.0):
            raise ValueError(f"resume_noise_pct must be in (0, 1], got {resume_noise_pct}")

        scheduler = self._build_scheduler(
            meta["sample_solver"], meta["sampling_steps"], meta["shift"])
        sigmas = scheduler.sigmas.to(device=self.device, dtype=z.dtype)
        candidates = sigmas[1:num_steps + 1]
        step_idx = min(int(torch.argmin((candidates - resume_noise_pct).abs()).item()), num_steps - 2)
        print(f"[edit_from_cache] resume_noise_pct={resume_noise_pct} -> cached step {step_idx} "
              f"(timestep={trajectory_tape[step_idx]['timestep']}, sigma~{candidates[step_idx].item():.4f}), "
              f"{num_steps - (step_idx + 1)} steps remaining")

        latent = trajectory_tape[step_idx]["latent"].to(device=self.device, dtype=torch.float32)
        z_frozen = _make_frozen(freeze_spec, regen_mask, z)
        z_hold = z if z_frozen is None else z_frozen
        latent = (1.0 - mask2) * z_hold + mask2 * latent

        paste = _make_paste(paste_spec, regen_mask, z)
        latent = self._denoise(
            latent, z, mask2, scheduler, step_idx + 1, context, context_null,
            seq_len, guide_scale, seed_g, paste, z_frozen=z_frozen)
        return vae_output_to_frames(pipe.vae.decode([latent])[0])

    def edit_from_inversion(
        self,
        window_frames: np.ndarray,
        regen_mask: np.ndarray,
        inversion_mask: np.ndarray,
        prompt: str,
        *,
        paste_spec: dict | None = None,
        freeze_spec: dict | None = None,
        move_spec: dict | None = None,
        noise_freeze_spec: dict | None = None,
        n_prompt: str = "",
        inversion_prompt: str | None = None,
        sampling_steps: int = 20,
        shift: float = 5.0,
        guide_scale: float = 5.0,
        inversion_guide_scale: float = 1.0,
        fixed_point_iters: int = 0,
        seed: int = -1,
        noise_strength: float = 0.9,
        save_inversion: Path | None = None,
    ) -> np.ndarray:
        """Invert the source up to sigma ~ `noise_strength`, edit there, denoise.

        See `noise_inversion.py` for the math. Same crop window as the SDEdit
        path, and no cache: any video can be inverted, including one that has
        already been edited. Both halves use Euler on one shared sigma grid
        (`EulerFlowScheduler`), so the denoise retraces the inversion.

        `inversion_mask` is what gets inverted -- the edit range, without the
        freeze holes (the source is inverted as it is; the freeze is applied
        afterwards), plus the anchor frame when something needs to read its
        inverted latent. `regen_mask` is what gets denoised, as in `edit()`.

        Edits applied to the inverted latent, in order:
          * `move_spec`: `shift_inverted_region`, the object moved in noise
            space rather than pasted as clean content (Method G).
          * `noise_freeze_spec`: the anchor's inverted box copied into every
            target frame, then kept identical across frames for the first
            `hold_steps` denoise steps and released (`--freeze_mode noise`).
          * `freeze_spec`: the ordinary hard freeze, via `regen_mask`/`z_hold`.
        """
        pipe = self.pipe
        z = pipe.vae.encode([frames_to_vae_input(window_frames, self.device)])[0]

        mask2, seq_len = self._mask_and_seq_len(regen_mask, z)
        inv_mask2, _ = self._mask_and_seq_len(inversion_mask, z)
        seed_g = self._generator(seed)

        context, context_null = self._encode_prompts(prompt, n_prompt)
        inv_context = context
        if inversion_prompt is not None and inversion_prompt != prompt:
            inv_context, _ = self._encode_prompts(inversion_prompt, n_prompt)

        if not (0.0 < noise_strength <= 1.0):
            raise ValueError(f"noise_strength must be in (0, 1], got {noise_strength}")
        scheduler = EulerFlowScheduler(
            sampling_steps, shift, num_train_timesteps=pipe.num_train_timesteps, device=self.device)
        start_idx = min(round(len(scheduler.timesteps) * (1.0 - noise_strength)),
                        len(scheduler.timesteps) - 1)
        sigmas_up = scheduler.inversion_sigmas(start_idx)
        sigma0 = float(sigmas_up[-1])
        passes = (1 if inversion_guide_scale == 1.0 else 2) * (1 + fixed_point_iters)
        print(f"[invert] 0 -> sigma {sigma0:.4f} in {len(sigmas_up) - 1} Euler steps "
              f"(guide_scale={inversion_guide_scale}, fixed_point={fixed_point_iters}, "
              f"{passes} model pass(es)/step), then {len(scheduler.timesteps) - start_idx} "
              f"denoise steps back")

        num_train = pipe.num_train_timesteps

        def inv_velocity(x, sigma):
            t = torch.tensor(sigma * num_train, device=self.device, dtype=torch.float32)
            return self._velocity(x, t, inv_mask2, seq_len, inv_context, context_null, inversion_guide_scale)

        temporal = [bool(inversion_mask[i].any()) for i in range(inversion_mask.shape[0])]
        with self._sampling():
            x_inv = euler_invert(inv_velocity, z, sigmas_up, mask2=inv_mask2, z_hold=z,
                                 fixed_point_iters=fixed_point_iters)
            if save_inversion is not None:
                torch.save({"latent": x_inv.cpu(), "z": z.cpu(), "sigma": sigma0,
                            "sampling_steps": sampling_steps, "shift": shift,
                            "start_idx": start_idx}, save_inversion)
                print(f"[invert] wrote {save_inversion}")

            latent = x_inv
            if move_spec is not None:
                anchor_idx = move_spec["anchor_idx"]
                targets = target_latent_frames(
                    [bool(regen_mask[i].any()) for i in range(regen_mask.shape[0])], anchor_idx)
                plan = _build_object_plan(
                    move_spec["object_paths"], move_spec["window"], anchor_idx, targets,
                    z.shape[-2], z.shape[-1], move_spec["vae_spatial_stride"], move_spec["feather"])
                if move_spec["source"] == "anchor" and not temporal[anchor_idx]:
                    raise ValueError("--move_source anchor needs the anchor frame inverted -- "
                                     "it must be in inversion_mask")
                latent = shift_inverted_region(
                    latent, plan, source=move_spec["source"], anchor_idx=anchor_idx,
                    vacated=move_spec["vacated"], generator=seed_g)
                print(f"[move] shifted in inverted latent at sigma {sigma0:.4f} "
                      f"(source={move_spec['source']}, vacated={move_spec['vacated']})")

            post_step = None
            if noise_freeze_spec is not None:
                latent, post_step = _noise_freeze(noise_freeze_spec, latent, regen_mask, temporal)

            z_frozen = _make_frozen(freeze_spec, regen_mask, z)
            z_hold = z if z_frozen is None else z_frozen
            latent = (1.0 - mask2) * z_hold + mask2 * latent

            paste = _make_paste(paste_spec, regen_mask, z)
            latent = self._denoise(
                latent, z, mask2, scheduler, start_idx, context, context_null,
                seq_len, guide_scale, seed_g, paste, z_frozen=z_frozen, post_step=post_step)
        return vae_output_to_frames(pipe.vae.decode([latent])[0])

    def _mask_and_seq_len(self, regen_mask: np.ndarray, z: torch.Tensor):
        mask = torch.tensor(regen_mask, dtype=z.dtype, device=self.device)
        if mask.shape != z.shape[1:]:
            raise ValueError(
                f"regen_mask has shape {tuple(mask.shape)} but the latent has shape "
                f"{tuple(z.shape[1:])} -- pixel/latent mapping mismatch")
        mask2 = mask.unsqueeze(0).expand_as(z)
        ph, pw = self.pipe.patch_size[1], self.pipe.patch_size[2]
        seq_len = math.ceil(
            (z.shape[1] * z.shape[2] * z.shape[3]) / (ph * pw) / self.pipe.sp_size) * self.pipe.sp_size
        return mask2, seq_len

    def _generator(self, seed: int) -> torch.Generator:
        seed = seed if seed >= 0 else torch.seed()
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        return g


def _make_frozen(freeze_spec: dict | None, regen_mask: np.ndarray, z: torch.Tensor):
    """Bind a `--freeze_box` spec to the latent in hand, like `_make_paste`.

    Returns None when nothing is frozen, so callers keep using `z` unchanged.
    """
    if freeze_spec is None or not freeze_spec["latent_boxes"]:
        return None
    temporal = [bool(regen_mask[i].any()) for i in range(regen_mask.shape[0])]
    anchor_idx = freeze_spec["anchor_idx"]
    targets = target_latent_frames(temporal, anchor_idx)
    if not targets:
        raise ValueError("no latent frames left to freeze -- the regenerate region is empty")
    held = int((regen_mask[targets] == 0).sum())
    print(f"[freeze] anchor=latent frame {anchor_idx}, targets={targets[0]}..{targets[-1]} "
          f"({len(targets)} frames), held cells={held // len(targets)}/"
          f"{regen_mask.shape[1] * regen_mask.shape[2]} per frame")
    return pin_anchor_content(z, freeze_spec["latent_boxes"], anchor_idx, targets)


def _make_paste(paste_spec: dict | None, regen_mask: np.ndarray, z: torch.Tensor) -> PasteConfig | None:
    """Bind a CLI-level paste spec to the latent actually in hand.

    Deferred to here because the latent's spatial size is only known after the
    VAE encode (or after the cache is loaded), and the feathered weight map
    must match it exactly.
    """
    if paste_spec is None:
        return None
    latent_h, latent_w = z.shape[-2], z.shape[-1]
    stride, feather = paste_spec["vae_spatial_stride"], paste_spec["feather"]

    temporal = [bool(regen_mask[i].any()) for i in range(regen_mask.shape[0])]
    anchor_idx = paste_spec["anchor_idx"]
    targets = target_latent_frames(temporal, anchor_idx)
    if not targets:
        raise ValueError(
            f"no latent frames left to correct: the regenerate region is "
            f"{[i for i, r in enumerate(temporal) if r]} and the anchor is {anchor_idx}")

    weight = None
    if paste_spec["boxes"]:
        weight = build_feathered_box_weight(
            paste_spec["boxes"], latent_h, latent_w, vae_spatial_stride=stride, feather=feather)
        if weight.max() <= 0:
            raise ValueError("--static_box produced an all-zero weight map -- boxes are off-grid")
        print(f"[static] anchor=latent frame {anchor_idx}, targets={targets[0]}..{targets[-1]} "
              f"({len(targets)} frames), weighted cells={int((weight > 0).sum())}/{latent_h * latent_w}")

    plan = _build_object_plan(
        paste_spec["object_paths"], paste_spec["window"], anchor_idx, targets,
        latent_h, latent_w, stride, feather)

    return PasteConfig(
        anchor_idx, targets, weight_hw=weight, plan=plan,
        strength=paste_spec["strength"], object_strength=paste_spec["object_strength"],
        vacated_fill=paste_spec["vacated_fill"], sigma_range=paste_spec["sigma_range"],
        max_steps=paste_spec["max_steps"], fft_ratio=paste_spec["fft_ratio"])


def _build_object_plan(object_paths, window, anchor_idx, targets, latent_h, latent_w, stride, feather):
    """Per-latent-frame `LatentShift`s for every object path, with the same
    reporting either consumer needs: Method G's `write_delta_trajectory`, or
    `shift_inverted_region` on the `--inversion` path."""
    plan = []
    for obj_idx, boxes in enumerate(object_paths):
        obj_plan = trajectory_to_latent_plan(
            window, boxes, anchor_idx, targets,
            latent_h, latent_w, vae_spatial_stride=stride, feather=feather)
        if not obj_plan:
            raise ValueError(
                f"object {obj_idx}'s trajectory never lands inside the frame -- check the box "
                f"coordinates and --move_to against tools/draw_box.py")
        shifts = [s.cell_shift for s in obj_plan]
        dropped = len(targets) - len(obj_plan)
        print(f"[object {obj_idx}] latent-cell shift {shifts[0]} -> {shifts[-1]} over "
              f"{len(obj_plan)} frames"
              + (f" ({dropped} dropped: box left the frame)" if dropped else ""))
        span = max(abs(shifts[-1][0] - shifts[0][0]), abs(shifts[-1][1] - shifts[0][1]))
        if span <= 1:
            print(f"  WARNING: the whole path spans {span} latent cell(s). Motion is quantised to "
                  f"{stride}px, so this trajectory will barely register -- use a larger --move_to "
                  f"or accept that this edit cannot express it.")
        plan.extend(obj_plan)
    return plan


def _noise_freeze(spec: dict, latent: torch.Tensor, regen_mask: np.ndarray, inverted: list[bool]):
    """`--freeze_mode noise`: freeze a box in the inverted latent, not in `z`.

    Copies the anchor frame's *inverted* box (content and noise) into every
    target frame, and returns a `post_step` that keeps the box identical across
    frames for the first `hold_steps` denoise steps before releasing it, so the
    rest of the schedule can blend the border. Soft, unlike the hard freeze:
    the box is held by its content, not clamped, once released.
    """
    anchor_idx = spec["anchor_idx"]
    if not inverted[anchor_idx]:
        raise ValueError("--freeze_mode noise needs the anchor frame inverted -- it must be in "
                         "inversion_mask")
    targets = target_latent_frames(
        [bool(regen_mask[i].any()) for i in range(regen_mask.shape[0])], anchor_idx)
    if not targets:
        raise ValueError("no latent frames left to freeze -- the regenerate region is empty")
    h, w = latent.shape[-2:]
    weight = weight_to_tensor(box_weight(spec["latent_boxes"], h, w, spec["feather"]),
                              latent.device, latent.dtype)
    latent = copy_region_across_frames(latent, weight, anchor_idx, targets)
    hold_steps, ref = spec["hold_steps"], targets[0]
    print(f"[freeze noise] anchor=latent frame {anchor_idx} (inverted) -> targets "
          f"{targets[0]}..{targets[-1]}, held for the first {hold_steps} denoise step(s), "
          f"cells={int((weight > 0).sum())}/{h * w} per frame")

    def post_step(j, x):
        return copy_region_across_frames(x, weight, ref, targets) if j < hold_steps else x

    return latent, post_step


def scale_boxes(boxes: list[tuple[int, int, int, int]], sx: float, sy: float):
    """Boxes are given in source-video pixels; the model may run on a resized
    crop, so they have to follow. Same convention as SG-I2V's own input
    handling (`SG-I2V/inference.py:36-37`)."""
    return [(round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy))
            for (x1, y1, x2, y2) in boxes]


def build_object_paths(
    args,
    window: FrameRangeWindow,
    orig_size: tuple[int, int],
    model_size: tuple[int, int],
) -> list[np.ndarray]:
    """One `(window.num_pixel_frames, 4)` box path per object, in model pixels.

    Both input forms land here: `--object_traj` reads SG-I2V's `[N, 2+F, 2]`
    files (so their `examples/` transfer unchanged), and
    `--object_box` + `--move_to` synthesises a straight-line sweep without
    needing a file. Either way the result is one box per *pixel* frame of the
    crop window, which is what `trajectory_to_latent_plan` consumes.

    A `.npy` authored against the whole video, or against just the edit range,
    will not have one entry per window frame; rather than demand an exact
    length, the path is linearly resampled and the resample is reported.
    """
    n_pixel = window.num_pixel_frames
    ow, oh = orig_size
    mw, mh = model_size
    paths: list[np.ndarray] = []

    for traj_path in (args.object_traj or []):
        for obj in load_trajectory_npy(Path(traj_path), (ow, oh), (mw, mh)):
            if len(obj) != n_pixel:
                print(f"[object] resampling {traj_path} from {len(obj)} to {n_pixel} frames "
                      f"to match the crop window")
                obj = resample_trajectory(obj, n_pixel)
            paths.append(obj)

    if args.object_box:
        box = parse_box(args.object_box)
        parts = args.move_to.split(",")
        if len(parts) != 2:
            raise ValueError(f"--move_to must be 'dx,dy', got {args.move_to!r}")
        try:
            dx, dy = (float(v) for v in parts)
        except ValueError:
            raise ValueError(f"--move_to components must be numbers, got {args.move_to!r}") from None
        scaled = scale_boxes([box], mw / ow, mh / oh)[0]
        paths.append(linear_trajectory(scaled, (dx * mw / ow, dy * mh / oh), n_pixel))

    return paths


def edit_range_psnr(original: np.ndarray, edited: np.ndarray, window: FrameRangeWindow) -> float:
    """Mean PSNR over the regenerated frames -- how faithfully the edit
    reproduced the source there. The inversion reconstruction check."""
    a = original[window.edit_start:window.edit_end + 1].astype(np.float64)
    b = edited[window.edit_start:window.edit_end + 1].astype(np.float64)
    mse = ((a - b) ** 2).reshape(len(a), -1).mean(axis=1)
    return float(np.mean(10.0 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-10))))


def visualize_boxes(frame: np.ndarray, boxes, save_path: Path, stride: int = 16) -> None:
    """Draw the static boxes on the anchor frame. Worth looking at before
    spending GPU time: a box that overlaps the drawer front will fight the
    motion the video is supposed to show.

    Delegates to `tools/draw_box.py` so the preview cannot drift out of
    agreement with this script -- in particular it also draws the *effective*
    region after the outward snap to whole latent cells, which is larger than
    the box as typed. `tools/draw_box.py` is also runnable standalone, without
    the model or a checkpoint directory.
    """
    from tools.draw_box import draw_boxes, describe, save_image

    h, w = frame.shape[:2]
    for box in boxes:
        print(describe(box, h, w, stride))
    save_image(save_path, draw_boxes(frame, list(boxes), grid=False, stride=stride))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_video", type=Path, required=True)
    ap.add_argument("--start_frame", type=int, required=True, help="A: first frame of the range to edit")
    ap.add_argument("--end_frame", type=int, required=True, help="B: last frame of the range to edit")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--prompt_file", type=Path, default=None, help="one prompt per line; runs the edit once per prompt")
    ap.add_argument("--negative_prompt", type=str, default="")
    ap.add_argument("--output", type=Path, required=True, help="output path; used as a stem when sweeping prompts")
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--task", type=str, default="ti2v-5B", choices=["ti2v-5B"])
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--num_steps", type=int, default=20)
    ap.add_argument("--guide_scale", type=float, default=5.0)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--noise_strength", type=float, default=0.6)
    ap.add_argument("--cache_path", type=Path, default=None,
                    help="resume from a generate_initial_video.py --cache_output latent cache; "
                         "--input_video must be the exact video that cache was generated for")
    ap.add_argument("--context_blocks", type=int, default=1)
    ap.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument("--t5_cpu", action="store_true")
    ap.add_argument("--no_offload", action="store_true")

    ap.add_argument("--static_box", type=str, action="append", default=None,
                    help="'x1,y1,x2,y2' in source-video pixels: a region to hold at its pre-drift "
                         "appearance. Repeatable. Prefer several small boxes on unambiguously static "
                         "structure (cabinet top, side panel, wall behind) over one big box -- a box "
                         "straddling something that legitimately moves, like the drawer front, will "
                         "fight that motion")
    ap.add_argument("--freeze_box", type=str, action="append", default=None,
                    help="'x1,y1,x2,y2' in source-video pixels: a region to hold HARD at the anchor "
                         "frame's appearance while everything else regenerates. Repeatable. Unlike "
                         "--static_box this is not a soft correction that decays with sigma -- the "
                         "region is excluded from the regenerate mask entirely, held at the anchor's "
                         "latent content at every step, and fed to the DiT with a timestep of 0 as "
                         "clean context. Nothing inside it can move, an occluding arm included, so it "
                         "pairs badly with a box containing something that should move. Snaps outward "
                         "to 32px (the DiT's 2x2-cell token grid), not 16px")
    ap.add_argument("--object_traj", type=str, action="append", default=None,
                    help="path to a trajectory .npy in SG-I2V's [N, 2+F, 2] format (two corner rows "
                         "as (w,h), then F box centres as (w,h)), in source-video pixels. Repeatable. "
                         "The box keeps its size and only the centre moves, as in SG-I2V")
    ap.add_argument("--object_box", type=str, default=None,
                    help="'x1,y1,x2,y2' in source-video pixels: the object to move, as an alternative "
                         "to --object_traj for a simple straight-line sweep. Requires --move_to")
    ap.add_argument("--move_to", type=str, default=None,
                    help="'dx,dy' total pixel DISPLACEMENT (not a destination corner) applied linearly "
                         "across the window. Requires --object_box")
    ap.add_argument("--object_strength", type=float, default=1.0,
                    help="[0,1] how strongly to move the object toward its trajectory position")
    ap.add_argument("--vacated_fill", type=float, default=1.0,
                    help="[0,1] how hard to erase the content the object left behind. 1.0 removes the "
                         "content term entirely (keeping that region's own noise) and lets the model "
                         "refill it; 0.0 writes nothing there, which is the control condition for "
                         "seeing how bad the ghost actually is")
    ap.add_argument("--paste_strength", type=float, default=1.0,
                    help="[0,1] fraction of the anchor-vs-current difference to correct per step")
    ap.add_argument("--paste_steps", type=int, default=4,
                    help="number of denoising steps to apply the correction on; the rest are left free "
                         "to harmonise the seam and re-assert an occluding gripper")
    ap.add_argument("--paste_sigma_range", type=float, nargs=2, default=(0.30, 1.00), metavar=("LO", "HI"),
                    help="only correct while the noise level is in [LO, HI]. Selected by sigma rather "
                         "than step index so it is unaffected by --num_steps and --noise_strength. "
                         "The default is deliberately wide, since --paste_steps already confines the "
                         "correction to the first few (highest-noise) steps; tighten LO upward only to "
                         "explicitly forbid late correction, and note that a LO above the resume sigma "
                         "(~--noise_strength) disables the correction entirely")
    ap.add_argument("--feather", type=int, default=2,
                    help="latent cells over which the box weight ramps in; 0 for hard edges")
    ap.add_argument("--fft_ratio", type=float, default=0.5,
                    help="Butterworth cutoff for SG-I2V's high-frequency restore after each correction")
    ap.add_argument("--no_fft_restore", action="store_true",
                    help="skip the frequency restore entirely (ablation)")
    ap.add_argument("--anchor_latent_frame", type=int, default=None,
                    help="override the auto-selected anchor latent frame (the last frozen one before "
                         "the edit region)")
    ap.add_argument("--visualize_boxes", type=Path, default=None,
                    help="write the anchor frame with the boxes drawn on it, then exit without "
                         "loading the model")

    inv = ap.add_argument_group("noise inversion (Method H)")
    inv.add_argument("--inversion", action="store_true",
                     help="invert the source window up to sigma ~ --noise_strength with Euler steps and "
                          "denoise back from there, instead of SDEdit's random re-noise or --cache_path. "
                          "Works on any video. Both halves use Euler on the --num_steps/--shift grid, so "
                          "--sample_solver is ignored. With no boxes it is the reconstruction check")
    inv.add_argument("--inversion_prompt", type=str, default=None,
                     help="prompt the source is inverted under; defaults to the edit prompt")
    inv.add_argument("--inversion_guide_scale", type=float, default=1.0,
                     help="CFG during inversion. 1.0 (default) is one model pass per step and inverts "
                          "most faithfully; higher values make the round trip drift")
    inv.add_argument("--inversion_fixed_point", type=int, default=0,
                     help="fixed-point refinements per inversion step (each one an extra model pass); "
                          "makes the inversion the exact inverse of the Euler denoise step")
    inv.add_argument("--move_source", type=str, default="self", choices=["self", "anchor"],
                     help="with --object_box/--object_traj: 'self' moves each frame's own inverted content "
                          "(the object keeps its motion, offset along the path); 'anchor' copies the "
                          "anchor frame's, as Method G does")
    inv.add_argument("--vacated_noise", type=str, default="noise", choices=["noise", "keep"],
                     help="what fills the cells a moved object left: fresh noise matched to the inverted "
                          "latent's statistics, or 'keep' (the ghost control)")
    inv.add_argument("--freeze_mode", type=str, default="hard", choices=["hard", "noise"],
                     help="'hard' holds --freeze_box at clean anchor content every step (exact). 'noise' "
                          "copies the anchor's *inverted* box into every frame instead, keeps it identical "
                          "across frames for --freeze_hold_steps, then releases it so the model can blend "
                          "the border (soft; aimed at seams at the box edge). 'noise' needs --inversion")
    inv.add_argument("--freeze_hold_steps", type=int, default=3,
                     help="with --freeze_mode noise: denoise steps to keep the box identical across frames")
    inv.add_argument("--save_inversion", type=Path, default=None,
                     help="write the inverted latent (before any edit) to this .pt")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.prompt and not args.prompt_file:
        raise ValueError("must pass --prompt or --prompt_file")
    if bool(args.object_box) != bool(args.move_to):
        raise ValueError("--object_box and --move_to must be given together")
    has_boxes = bool(args.static_box or args.object_traj or args.object_box or args.freeze_box)
    if not has_boxes and not args.inversion:
        raise ValueError(
            "must pass at least one of --static_box, --freeze_box, --object_traj or --object_box -- "
            "without any of them this script is exactly frame_range_edit.py, so use that instead "
            "(or pass --inversion alone for the reconstruction check)")
    if args.inversion and args.cache_path is not None:
        raise ValueError("--inversion and --cache_path are alternative entry paths -- pass one")
    if args.freeze_mode == "noise" and not (args.inversion and args.freeze_box):
        raise ValueError("--freeze_mode noise needs --inversion and at least one --freeze_box")
    prompts = [args.prompt] if args.prompt else [
        line.strip() for line in args.prompt_file.read_text().splitlines() if line.strip()
    ]
    boxes = [parse_box(b) for b in (args.static_box or [])]
    freeze_boxes = [parse_box(b) for b in (args.freeze_box or [])]

    full_frames, fps = read_video_frames(args.input_video)
    cfg = WAN_CONFIGS[args.task]
    stride = cfg.vae_stride[1]

    use_cache = args.cache_path is not None
    if use_cache:
        edit_start, edit_end = full_video_edit_bounds(args.start_frame, args.end_frame, len(full_frames))
        window = FrameRangeWindow(0, len(full_frames) - 1, edit_start, edit_end, pad_end=0)
        model_input_frames = None
        orig_h, orig_w = full_frames.shape[1:3]
        valid_h, valid_w = orig_h, orig_w
    else:
        window = build_regeneration_window(
            args.start_frame, args.end_frame, len(full_frames), context_blocks=args.context_blocks)
        raw_window_frames = full_frames[window.window_start:window.window_end + 1]
        if window.pad_end > 0:
            pad_block = np.repeat(raw_window_frames[-1:], window.pad_end, axis=0)
            raw_window_frames = np.concatenate([raw_window_frames, pad_block], axis=0)
        orig_h, orig_w = raw_window_frames.shape[1:3]
        valid_w = round_down_to_multiple(orig_w, SPATIAL_MULTIPLE)
        valid_h = round_down_to_multiple(orig_h, SPATIAL_MULTIPLE)
        model_input_frames = resize_frames(raw_window_frames, valid_w, valid_h)

    model_boxes = scale_boxes(boxes, valid_w / orig_w, valid_h / orig_h)
    patch = cfg.patch_size[1]
    model_freeze_boxes = [
        snap_box_outward(b, stride * patch, valid_w, valid_h)
        for b in scale_boxes(freeze_boxes, valid_w / orig_w, valid_h / orig_h)]

    temporal_mask = build_latent_regen_mask(window)
    h_latent, w_latent = valid_h // stride, valid_w // cfg.vae_stride[2]
    for given, snapped in zip(freeze_boxes, model_freeze_boxes):
        print(f"[freeze] {given}  ->  held region {snapped} (snapped out to {stride * patch}px)")
    # --freeze_mode noise holds the box through the inverted latent instead, so
    # the box stays in the regenerate mask and is denoised like everything else.
    if model_freeze_boxes and args.freeze_mode == "hard":
        regen_mask = build_freeze_regen_mask(
            window, model_freeze_boxes, valid_h, valid_w,
            vae_spatial_stride=stride, patch_spatial=patch)
        if not regen_mask.any():
            raise ValueError("--freeze_box covers the whole frame -- nothing left to regenerate")
    else:
        regen_mask = np.broadcast_to(
            np.array(temporal_mask, dtype=bool)[:, None, None], (len(temporal_mask), h_latent, w_latent))

    anchor_idx = args.anchor_latent_frame
    if anchor_idx is None:
        anchor_idx = anchor_latent_frame(window)
        if anchor_idx is None and has_boxes:
            raise ValueError(
                "the edit region starts at latent frame 0, so there is no frozen anchor frame "
                "before it (this happens when --start_frame is 0). Pass --anchor_latent_frame "
                "explicitly to choose a reference, or start the edit later in the video")
    if anchor_idx is not None and temporal_mask[anchor_idx]:
        print(f"[paste] WARNING: anchor latent frame {anchor_idx} is inside the regenerate region, "
              f"so it is not frozen and may itself drift during the edit")

    object_paths = build_object_paths(args, window, (orig_w, orig_h), (valid_w, valid_h))

    if args.visualize_boxes is not None:
        anchor_pixel = window.window_start + latent_frame_to_pixel_range(anchor_idx)[0]
        preview = list(boxes) + [
            tuple(int(round(v * (orig_w / valid_w if k % 2 == 0 else orig_h / valid_h)))
                  for k, v in enumerate(b))
            for b in model_freeze_boxes]
        for path in object_paths:  # show each object where it starts and ends
            src_scale = (orig_w / valid_w, orig_h / valid_h)
            for frame_box in (path[0], path[-1]):
                preview.append(tuple(int(round(v * src_scale[k % 2])) for k, v in enumerate(frame_box)))
        visualize_boxes(full_frames[anchor_pixel], preview, args.visualize_boxes)
        print(f"anchor latent frame {anchor_idx} -> source pixel frame {anchor_pixel}")
        return

    freeze_latent_boxes = [(x1 // stride, y1 // stride, x2 // stride, y2 // stride)
                           for (x1, y1, x2, y2) in model_freeze_boxes]
    freeze_spec = noise_freeze_spec = None
    if model_freeze_boxes and args.freeze_mode == "hard":
        freeze_spec = {"latent_boxes": freeze_latent_boxes, "anchor_idx": anchor_idx}
    elif model_freeze_boxes:
        noise_freeze_spec = {"latent_boxes": freeze_latent_boxes, "anchor_idx": anchor_idx,
                             "hold_steps": args.freeze_hold_steps, "feather": args.feather}

    # On the --inversion path objects move in the inverted latent instead of
    # through write_delta_trajectory; --static_box still goes through the paste.
    move_spec = None
    if args.inversion and object_paths:
        move_spec = {"object_paths": object_paths, "window": window, "anchor_idx": anchor_idx,
                     "source": args.move_source, "vacated": args.vacated_noise,
                     "feather": args.feather, "vae_spatial_stride": stride}
    paste_objects = [] if args.inversion else object_paths

    inversion_mask = None
    if args.inversion:
        # Invert the edit range as the source has it (no freeze holes), plus the
        # anchor whenever something reads the anchor's *inverted* latent.
        inversion_mask = np.broadcast_to(
            np.array(temporal_mask, dtype=bool)[:, None, None],
            (len(temporal_mask), h_latent, w_latent)).copy()
        if noise_freeze_spec is not None or (move_spec is not None and args.move_source == "anchor"):
            inversion_mask[anchor_idx] = True

    paste_spec = None if not (model_boxes or paste_objects) else {
        "boxes": model_boxes,
        "object_paths": paste_objects,
        "window": window,
        "anchor_idx": anchor_idx,
        "strength": args.paste_strength,
        "object_strength": args.object_strength,
        "vacated_fill": args.vacated_fill,
        "sigma_range": tuple(args.paste_sigma_range),
        "max_steps": args.paste_steps,
        "fft_ratio": None if args.no_fft_restore else args.fft_ratio,
        "feather": args.feather,
        "vae_spatial_stride": stride,
    }

    editor = StaticPasteEditor(
        checkpoint_dir=args.ckpt_dir, task=args.task, device_id=args.device_id,
        t5_cpu=args.t5_cpu, offload_model=not args.no_offload)

    for prompt in prompts:
        if use_cache:
            edited = editor.edit_from_cache(
                args.cache_path, regen_mask, prompt, paste_spec=paste_spec,
                freeze_spec=freeze_spec,
                n_prompt=args.negative_prompt, guide_scale=args.guide_scale,
                seed=args.seed, resume_noise_pct=args.noise_strength)
        elif args.inversion:
            edited = editor.edit_from_inversion(
                model_input_frames, regen_mask, inversion_mask, prompt,
                paste_spec=paste_spec, freeze_spec=freeze_spec, move_spec=move_spec,
                noise_freeze_spec=noise_freeze_spec,
                n_prompt=args.negative_prompt, inversion_prompt=args.inversion_prompt,
                sampling_steps=args.num_steps, shift=args.shift, guide_scale=args.guide_scale,
                inversion_guide_scale=args.inversion_guide_scale,
                fixed_point_iters=args.inversion_fixed_point, seed=args.seed,
                noise_strength=args.noise_strength, save_inversion=args.save_inversion)
            edited = resize_frames(edited, orig_w, orig_h)
        else:
            edited = editor.edit(
                model_input_frames, regen_mask, prompt, paste_spec=paste_spec,
                freeze_spec=freeze_spec,
                n_prompt=args.negative_prompt, sampling_steps=args.num_steps,
                shift=args.shift, guide_scale=args.guide_scale,
                sample_solver=args.sample_solver, seed=args.seed,
                noise_strength=args.noise_strength)
            edited = resize_frames(edited, orig_w, orig_h)

        full_edited = splice_edited_frames(full_frames, edited, window)
        assert_outside_range_intact(full_frames, full_edited, window.edit_start, window.edit_end)
        if args.inversion:
            print(f"[invert] edit-range PSNR vs source: "
                  f"{edit_range_psnr(full_frames, full_edited, window):.2f} dB "
                  f"(the reconstruction check: high with no boxes, lower where you moved things)")

        out_path = args.output
        if len(prompts) > 1:
            out_path = args.output.with_name(f"{args.output.stem}_{slugify(prompt)}{args.output.suffix}")
        write_video_frames(out_path, full_edited, fps)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
