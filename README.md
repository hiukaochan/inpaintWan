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
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
    --local-dir Wan2.2/checkpoints/Wan2.2-TI2V-5B
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

## Local, model-free checks

The pixel-frame <-> latent-frame index math (which frames get frozen vs.
regenerated) is pure Python and can be tested without a GPU or checkpoints:

```bash
python inpainting/test_frame_mapping.py
```
