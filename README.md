# Frame-range video editing on Wan2.2

Workflow:

1. **Generate an initial video** from a first frame + text prompt (`generate_initial_video.py`), e.g. "robot arm picking up an apple."
2. **Spot a problem in a frame range** (e.g. the gripper isn't close enough to the apple in frames 20-40) and **correct it with a new prompt** (`frame_range_edit.py`), e.g. "make the gripper close to the apple," regenerating only that range.

`frame_range_edit.py` regenerates frames `[A-1, B+1]` of an existing video
conditioned on a *correction* prompt, while frames `A-2`/`B+2` (and
everything further out) stay byte-identical to the source, and the edit
starts from the region's own original content (SDEdit-style partial
re-noising) rather than pure noise, so it corrects the existing scene
instead of hallucinating a new one. See `frame_range_edit.py`'s module
docstring for the exact algorithm.

This folder vendors the official [Wan2.2](https://github.com/Wan-Video/Wan2.2)
repo at `Wan2.2/` (cloned, not a submodule) and adds custom scripts on top
of it. Everything below (env, checkpoints, GPU runs) is meant to run on the
server -- this machine has no GPU/checkpoints.

## Setup (run on the server)

```bash
cd inpainting

conda create -n wan22 python=3.11 -y
conda activate wan22

pip install -r requirements.txt   # installs Wan2.2/requirements.txt

# TI2V-5B checkpoint (~5B params, runs on a single 24GB GPU at 720P)
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2/checkpoints/Wan2.2-TI2V-5B
```

### Smoke-test the stock install

Confirms Wan2.2 itself works before touching our custom script:


<!-- ```bash
cd Wan2.2
WAN_SKIP_DECODE=1 WAN_LATENT_PATH=/tmp/wan_latent.pt \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=4 generate.py --task ti2v-5B --size 1280*704 \
    --ckpt_dir checkpoints/Wan2.2-TI2V-5B \
    --dit_fsdp --t5_fsdp \
    --ulysses_size 4 \
    --frame_num 81 \
    --offload_model True \
    --image firstframe.png \
    --prompt "A white robotic arm with black joints and cables extends from a base on a wooden table, positioned near a closed white door. The arm's gripper, equipped with a small black device, slowly moves toward the door's handle, adjusting its angle as it approaches. The background includes a wall with a framed picture and a mounted camera, suggesting a tech-focused environment. The robotic arm's movements are smooth and deliberate, showcasing precision and control. A medium shot captures the entire setup, emphasizing the interaction between the robot and the door." 2>&1 | tee /tmp/run.log
``` -->
```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=1 \
python generate.py --task ti2v-5B --size 1280*704 \
    --ckpt_dir checkpoints/Wan2.2-TI2V-5B \
    --offload_model True --convert_model_dtype --t5_cpu \
    --frame_num 121 \
    --image firstframe.png \
    --prompt "A white robotic arm with black joints and cables extends from a base on a wooden table, positioned near a closed white door. The arm's gripper, equipped with a small black device, slowly moves toward the door's handle, adjusting its angle as it approaches. The background includes a wall with a framed picture and a mounted camera, suggesting a tech-focused environment. The robotic arm's movements are smooth and deliberate, showcasing precision and control. A medium shot captures the entire setup, emphasizing the interaction between the robot and the door."
```
## Step 1: generate an initial video

```bash
cd inpainting
python generate_initial_video.py \
    --image first_frame.jpg \
    --prompt "robot arm picking up an apple" \
    --output initial.mp4 \
    --ckpt_dir Wan2.2/checkpoints/Wan2.2-TI2V-5B
```

Thin wrapper around Wan2.2's own first-frame-conditioned generation
(`WanTI2V.i2v()`, called as-is, unmodified) -- see `generate_initial_video.py`'s
module docstring.

Add `--cache_output trajectory.pt` to also save every denoising step's
latent (keyed by step index, plus a `_meta` entry with the scheduler config)
to disk, for later inspection or -- via `frame_range_edit.py --cache_path`,
see Step 2 below -- resuming denoising from an earlier step with a
different prompt instead of restarting from scratch (useful when e.g. the
robot arm comes out deformed and you want to branch off a pre-deformation
step). This path runs a project-local reimplementation of `WanTI2V.i2v()`
(`generate_i2v_with_step_cache()`), since the vendored Wan2.2 code has no
callback hook; without the flag, behavior is unchanged.

## Step 2: correct a frame range

```bash
CUDA_VISIBLE_DEVICES=1 \
python frame_range_edit.py \
    --input_video initial.mp4 \
    --start_frame 54 --end_frame 120 \
    --prompt "Keep the arm's skeleton and structure identical — only move/rotate it as a rigid body so the gripper reaches toward the door handle. No change to shape, geometry, or proportions, pose change only." \
    --output out_8.mp4 \
    --noise_strength 1.0 \
    --ckpt_dir Wan2.2/checkpoints/Wan2.2-TI2V-5B
```

- `--start_frame`/`--end_frame` are `A`/`B`. The script regenerates
  `[A-1, B+1]`; frames `A-2`, `B+2`, and everything further out are left
  untouched (verified automatically after each run), with at least
  `--context_blocks` (default 1) full 4-frame blocks of guaranteed-frozen
  real content immediately after the edit region so the model always has a
  genuine anchor to blend into -- not just a fixed pixel margin, since
  Wan2.2's causal VAE groups frames into blocks of 4.
- `--noise_strength` (default `0.6`) controls how much of the edit region is
  re-noised: `1.0` regenerates it from scratch (ignoring the original
  content), lower values preserve more of the original composition while
  still letting the prompt drive a correction. Tune this if edits look too
  much like a wholesale scene replacement (lower it) or too similar to the
  original (raise it).
- The video needs enough real frames of context before `A` and after `B`;
  otherwise the script raises with a clear message. Two exact boundary
  cases are always allowed regardless of context availability:
  `--start_frame 0` (edit through the start of the video) and
  `--end_frame <last frame index>` (edit through the end of the video) --
  neither needs a frozen anchor on that side, since there's nothing before
  frame 0 or after the last frame to freeze.
- Add `--t5_cpu` / drop `--no_offload` flags to tune VRAM usage the same way
  Wan2.2's own `generate.py` does.

### Resuming from the real generation trajectory instead of SDEdit

If `initial.mp4` was generated with `generate_initial_video.py --cache_output
trajectory.pt`, pass that cache to correct a deformation by resuming
denoising from a real intermediate state of that same generation run,
instead of SDEdit's approximation (re-noising a re-encoded crop with fresh
random noise):

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

- `--input_video` must be the *exact* video that cache was generated for --
  it no longer matches after any other edit has been applied to it.
- `--noise_strength` is reused here as "how far back to resume": `0.4`
  resumes from whichever cached step is closest to 40% noise remaining, then
  re-denoises the rest with `--prompt`. Try a few values (e.g. `0.1`-`0.4`)
  to find one that resumes from before the deformation appeared.
- This mode always operates on the *whole* video's latent, never a cropped
  window -- Wan2.2's causal VAE has exactly one singleton latent frame, at
  true pixel 0 of the whole video, so a re-encoded crop's own "frame 0"
  wouldn't line up with the originally cached latent unless the crop starts
  at the real start of the video. `--context_blocks` is therefore unused
  in this mode.

### Trying multiple prompt phrasings (step 3)

Put one phrasing per line in a text file (see `prompts.txt` for an example)
and pass `--prompt_file` instead of `--prompt`:

```bash
python frame_range_edit.py \
    --input_video initial.mp4 \
    --start_frame 20 --end_frame 40 \
    --prompt_file prompts.txt \
    --output out.mp4 \
    --ckpt_dir Wan2.2/checkpoints/Wan2.2-TI2V-5B
```

This reuses the same crop/seed across prompts and writes one file per prompt
(`out_<slugified_prompt>.mp4`) so results are directly comparable.

## Step 3: pin a drifting region (`static_range_edit.py`)

Some failures aren't reachable by prompting at all. If Wan generates "robot
arm closes the drawer" but the whole cabinet slides across the frame, no
phrasing of "the cabinet stays still" reliably fixes it -- that's a statement
about pixel geometry, not about semantics.

`static_range_edit.py` applies the constraint to the latent instead, using
SG-I2V's idea of steering the sampling process rather than the weights (still
zero-shot -- the model is never fine-tuned). See `SG-I2V-method-notes.md` for
the method it's derived from. You mark regions that must not move; at
high-noise denoising steps they're pulled back toward how they looked in a
frozen anchor frame from before the drift.

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
  offset means it slid (fixable); a low score everywhere means it's
  deforming, which pinning will not fix, and it says so.
- **Where does it start?** It prints the first frame exceeding `--threshold`
  and the `--start_frame` to use.
- **Is the camera moving instead?** Pass a second `--box` on distant
  background. If every region drifts by the same vector the camera panned,
  holding one region still is the wrong remedy, and it warns you.

Re-run it on the output afterwards to score the fix.

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
- **green** -- the region that will *actually* be pinned.

They differ because `--static_box` is mapped through
`frame_mapping.pixel_box_to_latent_box`, which rounds **outward** to whole
16-pixel latent cells. The effective region is therefore always at least as
large as what you asked for, and up to 15px larger on each side -- so a box
that looks clear of the drawer front can overlap it once snapped. `--grid`
overlays the latent lattice to make that visible. The tool also prints the
latent cell range and how much area the snap added:

```
  (170, 100, 310, 190)  ->  latent cells x[10:20] y[6:12]  ->  effective pixels (160, 96, 320, 192)  (+2760 px^2)
```

`--frame` accepts negative indices (`-1` is the last frame). `draw_boxes()` is
importable if you'd rather annotate frames from your own code.

`static_range_edit.py --visualize_boxes boxes.png` does the same thing on the
auto-selected anchor frame and exits without loading the model, but it needs
the full set of edit arguments, so the standalone tool is usually quicker.

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

Accepts every `frame_range_edit.py` flag (both the SDEdit and `--cache_path`
paths), plus:

- `--static_box x1,y1,x2,y2` -- repeatable, in source-video pixels.
  **Prefer several small boxes on unambiguously static structure** (cabinet
  top, side panel, wall behind) over one large box. A box that straddles
  something which legitimately moves -- the drawer front -- will fight that
  motion. This is SG-I2V's own trick for camera control, anchoring static
  background, repurposed.
- `--paste_strength` (default `1.0`) -- fraction of the anchor-vs-current
  difference corrected per step. Lower it if the correction visibly drags
  the gripper along with the cabinet.
- `--paste_steps` (default `4`) -- how many denoising steps to correct on.
  The remaining steps are deliberately left free to harmonise the seam and
  re-assert anything the correction stepped on, such as an occluding arm.
- `--feather` (default `2`) -- latent cells over which the box weight ramps
  in, so the box edge doesn't leave a blocky 16px seam.
- `--fft_ratio` (default `0.5`) / `--no_fft_restore` -- SG-I2V's
  high-frequency restore (their Eq. 2), which repairs the artifacts that
  editing a latent otherwise introduces.

The anchor is chosen automatically: `build_regeneration_window` guarantees
that latent frame 0 of the window is frozen whenever `--start_frame > 0`, so
it holds real, un-edited source footage from before the drift. Editing from
frame 0 leaves no such frame, and the script asks for an explicit
`--anchor_latent_frame` rather than guessing.

**A caveat worth knowing.** SG-I2V evaluated latent copy-pasting as a
baseline and rejected it -- it scored the worst FID in their Table 1, with
Appendix A reporting that the modified latents "fell out of distribution".
This implementation differs in ways that should matter (it corrects a
*difference* from an anchor rather than overwriting, preserves each
position's own noise, and edits a real encoded video rather than initial
noise -- see `latent_paste.write_delta`), but if results come out
over-smoothed or artifact-heavy, that's the known failure mode. Compare
against a plain `frame_range_edit.py` run before concluding it helped.

## Local, model-free checks

The pixel-frame <-> latent-frame index math (which frames get frozen vs.
regenerated, and where a pixel box lands in the latent grid) and the latent
correction itself are both pure Python/PyTorch-CPU, testable without a GPU or
checkpoints:

```bash
python test_frame_mapping.py   # 32 tests
python test_latent_paste.py    # 16 tests
```

`tools/measure_drift.py` and `tools/draw_box.py` also run here -- they're
plain OpenCV (plus the torch-free `frame_mapping.py`) and deliberately do not
import `frame_range_edit.py`, which would drag in the whole Wan2.2 dependency
stack. `draw_box.py` owns the shared `read_video_frames`/`parse_box` helpers
that `measure_drift.py` imports.
