# Frame-range video editing on Wan2.2

Generate a robot-manipulation video from a first frame and a prompt, spot
something wrong in it, and fix *that part* without regenerating the rest.

This folder vendors the official [Wan2.2](https://github.com/Wan-Video/Wan2.2)
repo at `Wan2.2/` (cloned, not a submodule, and never patched) and adds custom
scripts on top of it. Every editing method here is zero-shot -- the model is
never fine-tuned; only the sampling process is steered.

Two properties hold for every method below:

- **Everything outside the edited range is byte-identical to the source.** The
  edit runs on a cropped window, and the final splice copies frames outside
  `[A-1, B+1]` from the original regardless of what the model produced there.
  `assert_outside_range_intact` verifies this after every run and raises if it
  ever fails.
- **GPU + checkpoints are required** for generation and editing. The
  inspection tools (`tools/`) and the unit tests are pure CPU and run on a
  laptop.

Companion docs in this tree: `static-method-workflow.md` (a tensor-by-tensor
walkthrough of one `static_range_edit.py` run -- every shape from input video
to spliced output) and `Wan2.2-architecture-notes.md` (the model itself: VAE,
DiT, schedulers).

---

## Setup (run on the server)

```bash
cd inpainting

conda create -n wan22 python=3.11 -y
conda activate wan22

pip install -r requirements.txt   # installs Wan2.2/requirements.txt

# TI2V-5B checkpoint (~5B params, runs on a single 24GB GPU at 720P)
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
  hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2/checkpoints/Wan2.2-TI2V-5B
```

### Smoke-test the stock install

Confirms Wan2.2 itself works before touching any custom script:

```bash
cd Wan2.2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 \
python generate.py --task ti2v-5B --size 1280*704 \
    --ckpt_dir checkpoints/Wan2.2-TI2V-5B \
    --offload_model True --convert_model_dtype --t5_cpu \
    --frame_num 121 \
    --image firstframe.png \
    --prompt "A white robotic arm with black joints and cables extends from a base on a wooden table, positioned near a closed white door. The arm's gripper slowly moves toward the door's handle, adjusting its angle as it approaches. The camera perspective remains completely unchanged."
```

Only `ti2v-5B` is supported by the custom scripts (`--task` is a one-choice
argument). The mapping constants -- temporal stride 4, spatial stride 16,
DiT patch 2 -- are that config's.

---

## Pick a method

| Symptom | Method | Flag |
|---|---|---|
| Wrong action/content over a stretch of frames | [A. Re-noise and regenerate](#method-a--add-noise-and-regenerate-a-frame-range) | `frame_range_edit.py` (default) |
| Wrong content in one *area* of the frame | [B. Spatial mask](#method-b--regenerate-only-part-of-the-frame) | `--edit_mode spatial --mask_box` |
| Arm/gripper deformed partway through generation | [C. Resume the cached trajectory](#method-c--resume-the-real-generation-trajectory-cached-latents) | `--cache_path` |
| Not sure which phrasing fixes it | [D. Prompt sweep](#method-d--sweep-several-prompt-phrasings) | `--prompt_file` |
| Background furniture drifts when it should be bolted down | [E. Soft pin](#method-e--soft-pin-a-drifting-region---static_box) | `static_range_edit.py --static_box` |
| A region must not change **at all** | [F. Hard freeze](#method-f--hard-freeze-a-region---freeze_box) | `static_range_edit.py --freeze_box` |
| Something should move along a path | [G. Object drag](#method-g--move-an-object-along-a-trajectory) | `--object_box` / `--object_traj` |
| Like C, but no cache (or the video was already edited); or G/F leave artifacts | [H. Noise inversion](#method-h--noise-inversion---inversion) | `static_range_edit.py --inversion` |

E, F and G compose in a single run. A/B/C/H are entry paths that E/F/G sit on
top of.

---

## Concepts every method shares

Read this once; the method sections assume it.

### The window and the frozen anchor

You pass `--start_frame A` and `--end_frame B`. The scripts regenerate pixel
frames `[A-1, B+1]` -- one frame of slack on each side, so the edit has room
to blend -- and crop a window around that range rather than feeding the whole
video to the model (`build_regeneration_window`, [frame_mapping.py:71](frame_mapping.py#L71)).

- `window_start = A - 2` always. Wan's causal VAE makes latent frame 0 a
  singleton covering only the window's own pixel frame 0, so the left side is
  clean by construction -- it never shares a latent block with the regenerate
  region.
- The right side needs `--context_blocks` (default 1) **whole 4-frame latent
  blocks** of guaranteed-frozen real content after the edit region -- not a
  fixed pixel margin, because the block containing `B+1` generally extends
  past it. Without this the intended anchor would silently share a block with
  the regenerated content. Too little trailing footage raises with a message
  saying how many more frames are needed.
- Two boundary cases are always allowed regardless of context: `--start_frame 0`
  and `--end_frame <last index>`. Neither needs an anchor on that side --
  there is nothing before frame 0 or after the last frame to freeze. The
  end case pads the model input with replicated edge frames purely to satisfy
  the VAE's `4n+1` alignment (`pad_end`); those are discarded before the splice
  and never reach the output.

The **anchor latent frame** is the last frozen latent frame before the
regenerate region (`anchor_latent_frame`). It holds real, un-edited source
footage from before whatever went wrong, and it is the reference that
`--static_box` and `--freeze_box` hold regions to. It exists by construction
whenever `A > 0`; at `A == 0` there is none and `static_range_edit.py` asks
for an explicit `--anchor_latent_frame` rather than guessing.

### Pixel frames vs. latent frames

Wan2.2's VAE is causal with temporal stride 4: **latent frame 0 encodes pixel
frame 0 alone**, and every later latent frame encodes a block of 4 pixel
frames. Spatially the stride is 16, so one latent cell is a 16x16 pixel block;
the DiT then groups 2x2 latent cells into one token, so a *token* covers 32
pixels.

That 16-vs-32 distinction is load-bearing and is why boxes snap differently
depending on what consumes them:

| Consumer | Snap | Why |
|---|---|---|
| `--static_box`, `--object_box` | 16px (`pixel_box_to_latent_box`) | a direct latent write can address individual cells, which is what makes feathering possible |
| `--mask_box`, `--freeze_box` | 32px (`build_spatial_latent_regen_mask`) | the per-token timestep reads the mask with a stride-2 subsample, so the mask must be constant within a token |

Every box rounds **outward**. The region actually affected is never smaller
than the one you typed, and up to 15 (or 31) pixels larger per side. Check it
with `tools/draw_box.py` before spending GPU time.

### How a region is held

Frozen positions are pinned by two lines the sampler runs every step
([frame_range_edit.py:271](frame_range_edit.py#L271)):

```python
temp_ts = (mask2[0][:, ::2, ::2] * timestep).flatten()   # per-token timestep: 0 where frozen
latent  = (1.0 - mask2) * z_hold + mask2 * latent         # re-paste clean content after each step
```

This generalises the first-frame-only trick already shipped in
`Wan2.2/wan/textimage2video.py::WanTI2V.i2v` (paste the clean VAE latent into
frozen positions and zero their diffusion timestep) from just frame 0 to an
arbitrary interior sub-range, and -- with `--mask_box`/`--freeze_box` -- to
arbitrary spatial regions. Neither line depends on the noise level, which is
why a freeze is exact where a `--static_box` correction is not.

### `--noise_strength` means two different things

Same flag, different semantics depending on the entry path:

- **Without `--cache_path` (SDEdit):** how much of the edit region is
  re-noised before denoising starts. `1.0` regenerates from pure noise,
  ignoring the original content; lower values start from the region's own
  content partially noised, so the prompt corrects the scene instead of
  replacing it. Default `0.6`. Frozen positions are always exactly `z`
  regardless.
- **With `--cache_path`:** how far back into the real generation trajectory to
  resume. `0.4` resumes from the cached step closest to 40% noise remaining.
  The run prints which step it picked and how many remain.

---

## Step 1: generate the initial video

```bash
python generate_initial_video.py \
    --image input/episode_000000.png \
    --prompt "The robotic arm closes its grippers and pushes the drawer back to the closed position. Throughout the entire sequence, the gripper undergoes no structural deformation, only the opening angle of its jaws changes; the rotational movements of the robotic arm joints strictly adhere to its inherent mechanical structure; and the camera perspective remains completely unchanged." \
    --output initial.mp4 \
    --ckpt_dir Wan2.2/checkpoints/Wan2.2-TI2V-5B
```

Without `--cache_output` this is a thin wrapper around Wan2.2's own
first-frame-conditioned generation (`WanTI2V.generate(img=...)` →
`WanTI2V.i2v()`), called unmodified, mirroring `Wan2.2/generate.py`'s `ti2v`
branch.

| Flag | Default | Notes |
|---|---|---|
| `--image` | required | first-frame conditioning image |
| `--prompt` / `--negative_prompt` | required / `""` | empty negative falls back to Wan's `sample_neg_prompt` |
| `--output` | required | `.mp4` |
| `--ckpt_dir` | required | `Wan2.2/checkpoints/Wan2.2-TI2V-5B` |
| `--size` | `1280*704` | one of Wan's `SIZE_CONFIGS` |
| `--frame_num` | task config | must be `4n+1` |
| `--fps` | task config | output framerate |
| `--num_steps` | `50` | denoising steps |
| `--guide_scale` / `--shift` | `5.0` / `5.0` | CFG scale, flow shift |
| `--sample_solver` | `unipc` | or `dpm++` |
| `--seed` | `-1` | `-1` picks a random seed |
| `--device_id`, `--t5_cpu`, `--no_offload` | | VRAM tuning, same as `generate.py` |
| `--cache_output` | none | see below |

### `--cache_output`: save the whole denoising trajectory

```bash
python generate_initial_video.py \
    --image input/episode_000000.png \
    --prompt "..." \
    --output initial.mp4 \
    --cache_output trajectory.pt \
    --ckpt_dir Wan2.2/checkpoints/Wan2.2-TI2V-5B
```

Saves **every denoising step's latent** to a `.pt` file, keyed by step index
(`{step: {"timestep": float, "latent": tensor}}`), plus a `_meta` entry
recording `sampling_steps`, `shift`, `sample_solver` and `frame_num` so a
later resume can rebuild the identical scheduler.

This is what [Method C](#method-c--resume-the-real-generation-trajectory-cached-latents)
consumes: instead of approximating "the video before it went wrong" by
re-noising the finished result, you resume from the actual intermediate state
of the run that produced it.

Two things to know:

- The flag switches generation to `generate_i2v_with_step_cache()`, a
  project-local reimplementation of `WanTI2V.i2v()`, because the vendored code
  has no callback hook and this repo's rule is to reimplement rather than patch
  `Wan2.2/`. Without the flag, behaviour is Wan's own, unchanged.
- UniPC/DPM++ are **stateful multistep solvers** -- they keep a short rolling
  history of previous model outputs that the cache does not capture. Resuming
  from a cached step is therefore a close but not bit-exact approximation of
  an uninterrupted run, for the first step or two after the resume point.

The cache is tied to the exact video it produced. Any later edit to that video
invalidates it.

---

## Method A -- add noise and regenerate a frame range

The baseline edit. SDEdit-style: re-noise the region partway, then denoise
with a correction prompt, so the model fixes the existing scene rather than
hallucinating a new one.

```bash
CUDA_VISIBLE_DEVICES=1 \
python frame_range_edit.py \
    --input_video initial.mp4 \
    --start_frame 54 --end_frame 120 \
    --prompt "Keep the arm's skeleton and structure identical -- only move/rotate it as a rigid body so the gripper reaches toward the door handle. No change to shape, geometry, or proportions, pose change only." \
    --output out.mp4 \
    --noise_strength 0.6 \
    --ckpt_dir Wan2.2/checkpoints/Wan2.2-TI2V-5B
```

| Flag | Default | Notes |
|---|---|---|
| `--input_video` | required | source video |
| `--start_frame` / `--end_frame` | required | `A` / `B`; regenerates `[A-1, B+1]` |
| `--prompt` / `--prompt_file` | one required | see [Method D](#method-d--sweep-several-prompt-phrasings) |
| `--negative_prompt` | `""` | |
| `--output` | required | used as a stem when sweeping prompts |
| `--noise_strength` | `0.6` | SDEdit strength in `(0, 1]` |
| `--context_blocks` | `1` | frozen 4-frame blocks required after the range |
| `--num_steps` | `20` | note: lower than `generate_initial_video.py`'s 50 |
| `--guide_scale` / `--shift` | `5.0` / `5.0` | |
| `--sample_solver` | `unipc` | or `dpm++` |
| `--seed` | `-1` | fix it to compare runs |
| `--edit_mode` / `--mask_box` | `full` / none | [Method B](#method-b--regenerate-only-part-of-the-frame) |
| `--cache_path` | none | [Method C](#method-c--resume-the-real-generation-trajectory-cached-latents) |
| `--device_id`, `--t5_cpu`, `--no_offload` | | VRAM tuning |

**Tuning `--noise_strength`.** If the edit looks like a wholesale scene
replacement, lower it. If the output is indistinguishable from the original,
raise it. `1.0` discards the region's content entirely and regenerates from
pure noise, conditioned only on the frozen context frames and the prompt.

The window is resized down to a multiple of 32 before encoding and resized
back after decoding, so the output keeps the source resolution.

---

## Method B -- regenerate only part of the frame

`--edit_mode spatial --mask_box x1,y1,x2,y2` narrows the edit to a rectangle:
inside the box is regenerated, outside stays biased toward its original
content.

```bash
python frame_range_edit.py \
    --input_video initial.mp4 \
    --start_frame 24 --end_frame 120 \
    --edit_mode spatial --mask_box 330,230,630,440 \
    --prompt "the gripper closes around the drawer handle" \
    --output spatial.mp4 \
    --noise_strength 0.8 \
    --ckpt_dir Wan2.2/checkpoints/Wan2.2-TI2V-5B
```

- `--mask_box` is required with `--edit_mode spatial` and rejected without it.
  Coordinates are source-video pixels, `x1<x2` and `y1<y2`.
- The spatial mask can only **narrow** what the temporal mask already allows,
  never widen it (`build_spatial_latent_regen_mask`, [frame_mapping.py:514](frame_mapping.py#L514)).
  A latent position regenerates only if it is both inside `[A-1, B+1]` *and*
  inside the box.
- **The box snaps outward to 32px.** The mask is pooled at
  `vae_stride * patch_size` because the DiT groups 2x2 latent cells into one
  token and the per-token timestep reads the mask with a stride-2 subsample --
  a mask varying inside a token would be sampled at an arbitrary corner.
- **Outside the box is soft, not guaranteed.** This is latent-level masking:
  positions outside the box are held at the source encoding each step, which
  strongly biases them but is not the byte-exact promise the temporal splice
  gives. Frames outside `[A-1, B+1]` are still byte-exact; *pixels* outside the
  box within those frames are not.
- Works on both entry paths -- combine it with `--cache_path` freely.

Use `tools/draw_box.py --grid` first to see where the box really lands.

---

## Method C -- resume the real generation trajectory (cached latents)

For the failure where the arm comes out deformed: the deformation did not
exist at step 5, it appeared somewhere in the middle. SDEdit can only
approximate that earlier state by re-noising the *finished* (already deformed)
video with fresh random noise. If you generated with `--cache_output`, you can
resume from the actual latent instead.

```bash
python frame_range_edit.py \
    --input_video initial.mp4 \
    --cache_path trajectory.pt \
    --start_frame 54 --end_frame 120 \
    --prompt "..." \
    --output out.mp4 \
    --noise_strength 0.4 \
    --ckpt_dir Wan2.2/checkpoints/Wan2.2-TI2V-5B
```

- `--input_video` must be the **exact** video that cache was generated for. It
  no longer matches after any other edit has been applied. The script checks
  the latent shape and the `_meta` step count, which catches gross mismatches
  but not a same-shape different video.
- `--noise_strength` is the resume depth. The run prints what it chose:

  ```
  [edit_from_cache] resume_noise_pct=0.4 -> cached step 7 (timestep=..., sigma~0.4012), 12 steps remaining
  ```

  Try a few values (`0.1`-`0.4`) to find one that resumes from *before* the
  deformation appeared. At least one step is always left to run.
- This mode always operates on the **whole video's latent**, never a cropped
  window. Wan's causal VAE has exactly one singleton latent frame, at true
  pixel 0 of the whole video, so a re-encoded crop's own "frame 0" would not
  line up with the cached latent unless the crop started at the real start.
  `--context_blocks` is therefore unused here, and no resize happens (the
  cached latent is already at the VAE-aligned resolution).
- The scheduler is rebuilt from `_meta`, not from your CLI flags, so
  `--num_steps`, `--shift` and `--sample_solver` are ignored on this path.

---

## Method D -- sweep several prompt phrasings

Put one phrasing per line in a text file (see `prompts.txt`) and pass
`--prompt_file` instead of `--prompt`:

```bash
python frame_range_edit.py \
    --input_video initial.mp4 \
    --start_frame 20 --end_frame 40 \
    --prompt_file prompts.txt \
    --output out.mp4 \
    --ckpt_dir Wan2.2/checkpoints/Wan2.2-TI2V-5B
```

Reuses the same crop and seed across prompts and writes one file per prompt
(`out_<slugified_prompt>.mp4`), so results are directly comparable. The model
is loaded once. Both `frame_range_edit.py` and `static_range_edit.py` accept
it, on either entry path.

---

## Method E -- soft-pin a drifting region (`--static_box`)

Some failures are not reachable by prompting at all. If Wan generates "robot
arm closes the drawer" but the whole cabinet slides across the frame, no
phrasing of "the cabinet stays still" reliably fixes it -- that is a statement
about pixel geometry, not about semantics.

`static_range_edit.py` applies the constraint to the latent instead, following
SG-I2V's idea of steering the sampling process rather than the weights. You
mark regions that must not move; at high-noise denoising steps they are pulled
back toward how they looked in the frozen anchor frame from before the drift.

### First, measure it (no GPU or checkpoints needed)

```bash
python tools/measure_drift.py \
    --video initial.mp4 \
    --box 620,180,900,540 \
    --box 40,60,200,260 \
    --overlay drift.mp4
```

Worth doing before spending GPU time, because it answers the three questions
that decide whether the edit can work at all:

- **Is it really translation?** The tool reports the best rigid offset *and*
  how well the region still matches there. A high match score at a large
  offset means it slid (fixable); a low score everywhere means it is
  deforming, which pinning will not fix, and it says so.
- **Where does it start?** It prints the first frame exceeding `--threshold`
  and the `--start_frame` to use.
- **Is the camera moving instead?** Pass a second `--box` on distant
  background. If every region drifts by the same vector the camera panned,
  holding one region still is the wrong remedy, and it warns you.

Flags: `--ref_frame` (the last known-good frame to measure against, default 0),
`--search_radius` (default 96px), `--threshold` (default 4.0px),
`--per_frame`, `--overlay`, `--csv`. Re-run it on the output to score the fix
-- it is the local stand-in for the ObjMC metric SG-I2V reports.

### Then check where the boxes land (no GPU or checkpoints needed)

```bash
python tools/draw_box.py \
    --video initial.mp4 --frame 52 \
    --box 620,180,900,540 --box 40,60,200,260 \
    --output boxes.png --grid
```

Draws each box on one frame and saves a PNG. It draws **two** rectangles per
box, and the second one is the point:

- **red** -- the box exactly as you typed it;
- **green** -- the region that will *actually* be affected, after the outward
  snap to whole latent cells.

A box that looks clear of the drawer front can overlap it once snapped.
`--grid` overlays the latent lattice to make that visible. It also prints the
cell range and how much area the snap added:

```
  (170, 100, 310, 190)  ->  latent cells x[10:20] y[6:12]  ->  effective pixels (160, 96, 320, 192)  (+2760 px^2)
```

`--frame` accepts negative indices (`-1` is the last frame); `--stride`
defaults to 16 (pass 32 to preview a `--freeze_box`); `--no_effective` draws
only the box as typed; `--no_labels` drops the text. `draw_boxes()` and
`draw_trajectory()` are importable if you would rather annotate frames from
your own code.

`static_range_edit.py --visualize_boxes boxes.png` does the same on the
auto-selected anchor frame and exits without loading the model, but it needs
the full set of edit arguments, so the standalone tool is usually quicker.

### Then run it

```bash
python static_range_edit.py \
    --input_video initial.mp4 \
    --cache_path trajectory.pt \
    --start_frame 54 --end_frame 120 \
    --prompt "the robot arm closes the drawer" \
    --static_box 620,180,900,540 \
    --static_box 40,60,200,260 \
    --output pinned.mp4 \
    --noise_strength 0.4 \
    --ckpt_dir Wan2.2/checkpoints/Wan2.2-TI2V-5B
```

`static_range_edit.py` accepts every `frame_range_edit.py` flag except
`--edit_mode`/`--mask_box` (its own boxes replace that mechanism), on both the
SDEdit and `--cache_path` paths, plus:

| Flag | Default | Notes |
|---|---|---|
| `--static_box x1,y1,x2,y2` | none | repeatable, source-video pixels |
| `--paste_strength` | `1.0` | fraction of the anchor-vs-current difference corrected per step |
| `--paste_steps` | `4` | how many denoising steps to correct on |
| `--paste_sigma_range LO HI` | `0.30 1.00` | only correct while the noise level is in range |
| `--feather` | `2` | latent cells over which the box weight ramps in; `0` for hard edges |
| `--fft_ratio` | `0.5` | Butterworth cutoff for the high-frequency restore |
| `--no_fft_restore` | off | skip the restore entirely (ablation) |
| `--anchor_latent_frame` | auto | override the auto-selected anchor |
| `--visualize_boxes` | none | write the annotated anchor frame and exit |

**Prefer several small boxes on unambiguously static structure** (cabinet top,
side panel, wall behind) over one large box. A box that straddles something
which legitimately moves -- the drawer front -- will fight that motion. This
is SG-I2V's own trick for camera control, anchoring static background,
repurposed.

### What the correction actually is

A soft *delta*, not an overwrite ([latent_paste.py:69](latent_paste.py#L69)).
Wan's flow-matching convention is `x = (1 - sigma) * x0 + sigma * eps`, so
changing content while keeping the destination's own noise means editing only
the `x0` term:

```
latent[t] += (1 - sigma) * weight * strength * (z[anchor] - z[t])
```

Three properties make this preferable to pasting the region outright:

1. **The noise realisation is preserved exactly.** Copying `latent[anchor]`
   into `latent[t]` would carry `eps_anchor` with it, and identical noise at
   two temporal positions reads to the model as "these are the same frame" --
   which can freeze everything in the box, the occluding gripper included.
2. **No division by sigma**, so nothing is amplified where the estimate is
   least reliable.
3. **It is a no-op wherever the scene already matches**, since the correction
   is proportional to `z[anchor] - z[t]`.

**Timing is deliberate.** The correction fires only at high noise and only for
the first `--paste_steps` such steps. The `x0 ~ z` approximation is accurate
only while the latent still encodes something close to the source video, and
SG-I2V's own ablations show late latent edits wreck visual quality. Leaving
the remaining steps free is what lets the sampler harmonise the box seam and
re-assert anything the correction stepped on, such as an occluding gripper.
Each firing prints:

```
[paste] step 8 sigma=0.6213 (1/4)
```

If no step falls in the sigma window, it warns and corrects nothing -- widen
`--paste_sigma_range` or raise `--noise_strength`.

After each correction, `fft_restore` keeps the edited latent's low spatial
frequencies and restores the original's high ones (SG-I2V Eq. 2), restricted
to the frames actually written. The motion/layout signal lives in the low
frequencies and the editing artifacts in the high ones, so this discards the
damage nearly for free.

### The limit worth knowing before you use it

The write is scaled by `(1 - sigma)`, which is forced by the flow-matching
convention -- at high sigma there is barely any `x0` term to edit. At
`--noise_strength 1.0` the first correction step sits at `sigma = 1.0` and
writes **nothing**; the four default steps together apply roughly 6% of the
anchor delta. That is a property of the mechanism, not a setting. Use
`--freeze_box` when you need the region genuinely held at high noise.

**A caveat from the source method.** SG-I2V evaluated latent copy-pasting as a
baseline and rejected it -- worst FID in their Table 1, with Appendix A
reporting that the modified latents "fell out of distribution". This
implementation differs in ways that should matter: it corrects a *difference*
from an anchor rather than overwriting, preserves each position's own noise,
and edits a real encoded video rather than initial noise. But if results come
out over-smoothed or artifact-heavy, that is the known failure mode. Compare
against a plain `frame_range_edit.py` run before concluding it helped.

---

## Method F -- hard-freeze a region (`--freeze_box`)

When you want a region genuinely held and everything else free:

```bash
python static_range_edit.py \
    --input_video initial.mp4 \
    --start_frame 24 --end_frame 120 \
    --prompt "the robot arm closes the drawer" \
    --freeze_box 330,230,630,440 \
    --output frozen.mp4 \
    --noise_strength 1.0 \
    --ckpt_dir Wan2.2/checkpoints/Wan2.2-TI2V-5B
```

This is the complement of [Method B](#method-b--regenerate-only-part-of-the-frame),
which regenerates the *inside* of its box. Here the inside is excluded from
the regenerate mask entirely (`build_freeze_regen_mask`, [frame_mapping.py:564](frame_mapping.py#L564))
and held by the two structural lines the sampler already runs every step --
the re-paste and the per-token timestep of 0. Neither depends on sigma, so the
freeze is **exact even at `--noise_strength 1.0`**.

The held content is the **anchor frame's**, not each frame's own
(`pin_anchor_content`, [latent_paste.py:43](latent_paste.py#L43)). This matters:
the source video is the drifted one, so holding the box at `z` would reproduce
the drift faithfully. The run reports both halves:

```
[freeze] (330, 230, 630, 440)  ->  held region (320, 224, 640, 448) (snapped out to 32px)
[freeze] anchor=latent frame 5, targets=6..30 (25 frames), held cells=380/3432 per frame
```

Three consequences worth accepting before using it:

- **Nothing inside the box can move.** Not the drawer front, not an arm that
  passes in front of it. A frozen region occluded by the gripper will clip it.
  `--static_box` deliberately leaves later steps free to re-assert an
  occluder; a freeze does not.
- **The box snaps outward to 32px**, not 16px, for the same token-grid reason
  as `--mask_box`.
- **It composes with `--static_box`.** Freeze the rigid shell, soft-pin the
  parts that still need to move a little.

Covering the whole frame raises -- there would be nothing left to regenerate.

Rule of thumb: `--static_box` when the region should mostly hold but still
respond to the scene; `--freeze_box` when it should not change at all.

---

## Method G -- move an object along a trajectory

The non-zero-trajectory case: give a box a path and its content is carried
along it. Composes with E and F, so you can pin the cabinet *and* move the
gripper in one run.

```bash
# straight-line sweep, no file needed -- 'dx,dy' is a DISPLACEMENT, not a destination
python static_range_edit.py \
    --input_video initial.mp4 \
    --start_frame 24 --end_frame 120 \
    --prompt "..." --output moved.mp4 \
    --ckpt_dir Wan2.2/checkpoints/Wan2.2-TI2V-5B \
    --object_box 620,180,900,540 --move_to 320,160 \
    --static_box 40,60,200,260

# or SG-I2V's own [N, 2+F, 2] format, so their examples/ transfer unchanged
python static_range_edit.py ... --object_traj traj.npy
```

- `--object_box` and `--move_to` must be given together.
- `--object_traj` is repeatable and takes SG-I2V's format: two corner rows as
  `(w, h)`, then F box centres as `(w, h)`. As in SG-I2V, the box keeps its
  size and only the centre moves. A path authored against the whole video (or
  just the edit range) is linearly resampled to the crop window's length, and
  the resample is reported.
- `--object_strength` (default `1.0`) -- how strongly to move the object.
- `--vacated_fill` (default `1.0`) -- what happens where the object moved
  *away from*. This is the part with no obviously right answer. At `1.0` the
  content term in those cells is removed entirely (keeping their own noise),
  so the denoiser refills them from surrounding context and the prompt; at
  `0.0` nothing is written there. **Run `0.0` first** to see how bad the ghost
  is, then `1.0` to see whether softening actually clears it -- that contrast
  is the only honest way to tell whether this is working.

Preview a path before spending GPU time, with the same flag names the editor
takes:

```bash
python tools/draw_box.py --video initial.mp4 --frame 52 \
    --object_box 620,180,900,540 --move_to 320,160 --output path.png
# or: --object_traj traj.npy
```

Blue is the start box and the centre track, amber is where it ends.

Two constraints worth knowing before authoring a path:

- **Motion is quantised to 16px latent cells.** Sub-16px motion does not
  register at all. Wan's temporal stride of 4 helps (each latent frame spans 4
  pixel frames, so per-latent-frame displacement is 4x the per-frame motion),
  but a slow path still staircases. Both the tool and the editor print the
  latent-cell displacement and warn when the whole path spans one cell or
  fewer:

  ```
  [object 0] latent-cell shift (0, 0) -> (10, 20) over 25 frames
  ```

- **Content is relocated, not duplicated-then-erased.** `--vacated_fill`
  removes the *bias* toward redrawing the object at its old position; it
  cannot guarantee the model will not put it back where context strongly
  implies it (an arm still attached to it, say).

---

## Method H -- noise inversion (`--inversion`)

A third way to reach an intermediate noisy latent, next to SDEdit (A) and the
cache (C). SDEdit re-noises with a *random* `eps`, so the result denoises to a
plausible neighbour of the source. The cache is the real trajectory but only
exists for videos generated with `--cache_output`, and dies with the first
edit. Inversion runs the sampler **backwards** from the clean encoding and
gets a noisy latent that denoises back to *this* video -- any video, edited
or not. Edits (moving the arm, freezing the cabinet) are then made in that
latent, at high noise, where the model has the whole schedule to make them
look natural.

### Why it works on Wan2.2

Wan2.2 is rectified flow, not DDPM: `x_sigma = (1 - sigma) z0 + sigma eps`,
and the DiT predicts the velocity `v = eps - z0 = dx/dsigma`
(`training/conditioning.py::velocity_target`). Sampling integrates that ODE,
and the ODE is deterministic, so it runs both ways:

```
denoise  x_{sigma-d} = x_sigma - d * v(x_sigma, sigma)     # one Euler sampling step
invert   x_{sigma+d} = x_sigma + d * v(x_sigma, sigma)     # the same step, mirrored
```

The first line is the familiar "`v = f(x_0.9, t=0.9)`, `x = x_0.9 - v * delta`":
it is the *denoising* half. Where `x_0.9` comes from is what makes it
inversion: here it is the output of the second line, walked up from `z`, not
`0.1 z + 0.9 eps` with a random `eps`. The model is called with timestep
`sigma * 1000`.

Both halves use plain **Euler** on the same sigma grid
(`noise_inversion.EulerFlowScheduler`), because UniPC/DPM++ carry a multistep
history with no clean reverse. `--sample_solver` is ignored on this path.

### Check the reconstruction first

```bash
python static_range_edit.py \
    --input_video initial.mp4 \
    --start_frame 54 --end_frame 120 \
    --prompt "the robot arm closes the drawer" \
    --inversion --noise_strength 0.9 \
    --output recon.mp4 \
    --ckpt_dir Wan2.2/checkpoints/Wan2.2-TI2V-5B
```

With no boxes this should give back the input. The run prints the edit-range
PSNR against the source; compare it with an SDEdit run (`frame_range_edit.py`)
at the same strength, which should be clearly lower. If it is not high:

- Euler inversion is approximate -- each step evaluates `v` at `x_sigma`, but
  the denoise step it must undo evaluates it at `x_{sigma+d}`.
  `--inversion_fixed_point K` iterates the step to the exact inverse, at `K`
  extra model passes per step (`2`-`3` is usually plenty).
- CFG breaks the symmetry. Inversion runs at `--inversion_guide_scale 1`
  (default; also one model pass instead of two). The denoise uses
  `--guide_scale`; if reconstruction suffers, try lowering it toward 1-2.
- More `--num_steps` shrinks the per-step error.

`--noise_strength` keeps its SDEdit meaning -- the fraction of steps run, not
sigma -- so with `--shift 5` a value of `0.9` starts at sigma ~0.98. The run
prints the actual sigma.

### Move a region in the inverted latent

```bash
python static_range_edit.py ... --inversion --noise_strength 0.9 \
    --freeze_box 620,180,900,540 \
    --object_box 330,230,630,440 --move_to 0,64
```

Same `--object_box/--move_to/--object_traj` inputs as Method G, but instead
of `write_delta_trajectory`'s clean-content paste, the box is moved *inside
the inverted latent* -- content and noise together
(`noise_inversion.shift_inverted_region`). At high sigma that latent is close
to Gaussian, so the moved box is still something the model could have been
handed; that is the hypothesis for why this should leave fewer
out-of-distribution artifacts than G.

- `--move_source self` (default) moves each frame's *own* content: the arm
  keeps the motion it had and is offset along the path. `anchor` copies the
  anchor frame into every frame, as G does.
- `--vacated_noise noise` (default) refills the cells the object left with
  fresh noise matched to the inverted latent's per-channel statistics, so the
  model redraws them from context. `keep` leaves them alone -- the ghost
  control, same role as G's `--vacated_fill 0`.
- `--static_box` still composes (it goes through the usual paste).

### Freeze boxes under inversion (`--freeze_mode`)

- `hard` (default) -- Method F unchanged: clamped to the anchor's clean
  content every step, timestep 0, exact. Caveat: inversion reproduces
  everything *outside* the box faithfully, drift included, so if the source
  drifted, a pinned box can tear against its surroundings *more* than with
  SDEdit at 1.0 (which redraws the surroundings to fit).
- `noise` -- aimed at that seam. The anchor frame is inverted too, and its
  inverted box (content and noise) is copied into every target frame. Identical
  noise across frames reads to the model as "this region is static". The box
  is kept identical across frames for the first `--freeze_hold_steps` (default
  3) denoise steps -- at the current noise level, not clamped to clean
  content -- then released, so the remaining steps can blend its border.
  Soft: the box is held by its content, not clamped, so measure it with
  `tools/measure_drift.py`. Try `--freeze_hold_steps` 2/3/5 against `hard`
  on the same seed.

Because the SDEdit-style window puts the anchor at the window's latent frame
0, `noise` (and `--move_source anchor`) invert that frame as well; the
trailing `--context_blocks` frames stay clean context during inversion.

| Flag | Default | Notes |
|---|---|---|
| `--inversion` | off | this entry path; exclusive with `--cache_path` |
| `--inversion_prompt` | `--prompt` | prompt the source is inverted under |
| `--inversion_guide_scale` | `1.0` | CFG while inverting |
| `--inversion_fixed_point` | `0` | refinements per inversion step |
| `--move_source` | `self` | `self` / `anchor` |
| `--vacated_noise` | `noise` | `noise` / `keep` |
| `--freeze_mode` | `hard` | `hard` / `noise` |
| `--freeze_hold_steps` | `3` | `noise` freeze only |
| `--save_inversion` | none | write the inverted latent (pre-edit) to a `.pt` |

---

## `latent_guidance.py` -- implemented, not wired up

The alternative to `latent_paste.py`, and the part of SG-I2V that their own
ablations call load-bearing. **No script imports it and there is no CLI flag
for it today** -- it is complete, unit-tested code with no entry point.

`latent_paste` corrects the latent by *algebra*: a closed-form write, no model
call. `latent_guidance` *searches*: it measures the error in the DiT's own
feature space and backpropagates it to the latent (SG-I2V Eq. 1). Two things
follow that a closed-form write structurally cannot do:

- **it is selective about what it objects to** -- a latent-space difference
  fires on a shadow crossing the box, a feature-space one does not, which
  matters when a gripper occludes the controlled region;
- **it never has to say what fills a vacated region** -- a loss asks for
  content at the new position and stays silent about the old one, leaving the
  model's prior to resolve it. That is the unanswered "line 4" of the drag
  formulation, and it is why SG-I2V optimises rather than assigns.

The price is a forward *and* backward pass through half the DiT per iteration,
so this costs minutes where `latent_paste` costs milliseconds.

The pieces, if you wire it in:

- `forward_to_block(model, ..., block_idx=15)` -- runs `WanModel` only as far
  as one block, with gradient checkpointing (`WanModel` has none of its own)
  and no head, so later blocks never build graph.
- `tokens_to_feature_grid` -- `(1, seq_len, dim)` → `(F, H', W', dim)`,
  dropping the sequence padding.
- `trajectory_loss` -- Gaussian-weighted MSE (or Huber) between each frame's
  box features and the anchor's. Huber is a deliberate deviation from the
  paper: where a gripper crosses the box those cells genuinely *should*
  differ, and under a squared penalty they would dominate the gradient and
  fight the arm.
- `optimize_latent` -- AdamW on the latent, gradient masked by `mask2` so
  frozen context never moves. Caller applies `fft_restore` and re-asserts the
  mask afterwards, the same way `latent_paste` corrections are finished off.
- `rope_in_fp32` -- scoped patch of the vendored `rope_apply` (fp64 → fp32) for
  the duration of the optimisation; ~390MB per tensor at 16k tokens, and fp64
  precision is meaningless for a guidance signal. Reversed on exit, including
  on exception, and never active during ordinary sampling.

What is ported and what is not: SG-I2V's Part 1 (replacing each frame's K/V
with frame 1's) is **not** ported -- it exists because SVD's spatial
self-attention runs per-frame, whereas Wan's DiT applies full 3D attention over
the flattened `(F, H, W)` sequence, so every token already attends to every
frame. Part 2 (Eq. 1) is what this module is. Part 3 (Eq. 2) is already
`latent_paste.fft_restore` and is reused unchanged.

Note: the module docstring references `tools/probe_dit_features.py`, which does
not exist in this tree.

---

## Module map

| File | Role |
|---|---|
| `generate_initial_video.py` | step 1; also `generate_i2v_with_step_cache()` for `--cache_output` |
| `frame_range_edit.py` | `WanFrameRangeEditor`: window crop, masked denoise (both entry paths), splice, verification. The baseline every other method is compared against |
| `static_range_edit.py` | `StaticPasteEditor(WanFrameRangeEditor)`: same machinery plus the in-loop latent correction. Imports `frame_range_edit.py`, never copies it |
| `frame_mapping.py` | pixel↔latent index math, window construction, all mask/box/trajectory builders. Pure Python + numpy, no torch, no model |
| `latent_paste.py` | the latent writes themselves: `write_delta`, `write_delta_trajectory`, `soften_toward_noise`, `pin_anchor_content`, `fft_restore` |
| `noise_inversion.py` | Method H: `EulerFlowScheduler`, `euler_invert`, `shift_inverted_region`, `copy_region_across_frames`. Model-free; tested by `test_noise_inversion.py` |
| `latent_guidance.py` | gradient guidance (above). Not wired to any CLI |
| `tools/draw_box.py` | box and trajectory preview; owns the shared `read_video_frames`/`parse_box` helpers |
| `tools/measure_drift.py` | drift measurement, before and after |

`tools/` deliberately does **not** import `frame_range_edit.py`, which would
drag in the whole Wan2.2 dependency stack at module scope. That is what keeps
the tools runnable on a laptop and over SSH.