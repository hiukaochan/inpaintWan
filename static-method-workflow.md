# `static_range_edit.py`: the static method, end to end

What actually happens to the tensors when you run

```bash
python static_range_edit.py \
    --input_video initial3.mp4 \
    --start_frame 24 --end_frame 120 \
    --prompt "..." \
    --static_box 330,230,630,440 \
    --output pinned.mp4 \
    --noise_strength 0.5 \
    --ckpt_dir Wan2.2/checkpoints/Wan2.2-TI2V-5B
```

Every shape below is the real one for `initial3.mp4` (121 frames, 1248x704,
24 fps) with that command, computed from `frame_mapping.py` rather than
estimated. Companion docs: `Wan2.2-architecture-notes.md` for the model,
`SG-I2V-method-notes.md` for the method this borrows from.

---

## 0. The one-paragraph version

The video is cropped to a window around the edit range, encoded to a latent,
and partially re-noised. During denoising, the boxed region of every
regenerated latent frame is nudged toward how that box looked in **one frozen
anchor frame from before the drift** -- a small additive correction, applied
on the first few (highest-noise) steps only. The result is decoded and spliced
back into the untouched original. The model is never fine-tuned; only the
sampling process is steered.

---

## 1. Input: from pixels to a latent

### 1.1 The source video

```
full_frames   (121, 704, 1248, 3)  uint8 RGB     read by cv2, in frame order
```

### 1.2 The crop window

`build_regeneration_window(a=24, b=120, video_len=121)` decides which slice of
the video the model actually sees. For this run:

```
FrameRangeWindow(window_start=22, window_end=120, edit_start=23, edit_end=120, pad_end=2)
num_pixel_frames = (120 - 22 + 1) + 2 = 101
```

Read that as four different things:

| field | value | meaning |
|---|---|---|
| `window_start` | 22 | first frame handed to the VAE |
| `edit_start` | 23 | first frame the model may change |
| `edit_end` | 120 | last frame the model may change |
| `window_end` | 120 | last *real* frame in the window |
| `pad_end` | 2 | synthetic frames (last frame replicated) for VAE alignment |

Two rules produce these numbers:

- **`window_start = a - 2`, always.** Wan2.2's causal VAE has temporal stride
  4, but latent frame 0 is a singleton covering only pixel frame 0 of the
  window. Starting two frames early guarantees latent frame 0 is a real,
  untouched frame that shares no latent block with the edit region. That frame
  is what becomes the anchor.
- **The right edge is rounded out to whole latent blocks**, plus
  `--context_blocks` (default 1) full 4-frame blocks of guaranteed-frozen
  content. Here `b == video_len - 1`, so that trailing requirement is skipped
  (there is nothing after the last frame to freeze) and the leftover alignment
  remainder becomes `pad_end = 2`. Those 2 frames are copies of frame 120,
  they fall inside an already-regenerated block, and they are discarded before
  the splice.

```
raw_window_frames  (99, 704, 1248, 3)   = full_frames[22:121]
      + 2 replicated frames
                   (101, 704, 1248, 3)  uint8
```

### 1.3 Resize to the model grid

```
valid_w = round_down_to_multiple(1248, 32) = 1248
valid_h = round_down_to_multiple( 704, 32) =  704      -> no resize happens here
model_input_frames  (101, 704, 1248, 3)  uint8
```

32 is `vae_stride * patch_size` = 16 x 2. Both dimensions are already
multiples, so this run is a no-op -- but when it is not, `--static_box`
coordinates are scaled to match by `scale_boxes()`, the same convention
SG-I2V uses.

### 1.4 VAE encode

```
frames_to_vae_input:  (101, 704, 1248, 3) uint8
                   -> (3, 101, 704, 1248) float32 in [-1, 1]

pipe.vae.encode:      (3, 101, 704, 1248)
                   -> z (48, 26, 44, 78) float32
                       ^   ^   ^   ^
                       |   |   |   +-- 1248 / 16
                       |   |   +------  704 / 16
                       |   +---------- (101 - 1) // 4 + 1
                       +-------------- z_dim
```

`z` is the **clean encoding of the source video**, and it never changes for
the rest of the run. Note what that means: the source video is the *drifted*
one, so `z` contains the drift. Fixing the drift is entirely a matter of how
`z` is used, not of altering `z`.

Compression: 266M pixel values become 4.28M latent values, **62x fewer**
(4x temporally, 256x spatially, 16x more channels -- `4 * 256 / 16 = 64`, and
the singleton latent frame 0 accounts for the small shortfall). In bytes that
is 266 MB of uint8 pixels to 17 MB of fp32 latent, 15.5x.

### 1.5 Temporal mapping (the part that trips people up)

Latent frame `i` covers pixel frames:

```
i == 0 :  [0, 0]                    <- singleton, causal
i >= 1 :  [4(i-1) + 1, 4i]
```

so for our 26 latent frames, with the edit region at window-local pixels
[1, 98]:

```
latent frame:  0  1  2  3  4 ... 25
mask:          F  R  R  R  R ... R      F = frozen, R = regenerate
```

Only latent frame 0 is frozen. It maps to window-local pixel 0, i.e. **source
frame 22** -- three frames before drift first exceeds 4px at frame 25, as
measured by `tools/measure_drift.py`. That is the anchor.

```
anchor_idx    = 0
target_frames = [1, 2, ..., 25]      the 25 frames the correction may write to
```

`target_latent_frames()` excludes the anchor itself and every frozen frame:
writing to a frozen frame would be silently discarded, because the loop
re-imposes clean content there after every step.

### 1.6 The box

`--static_box 330,230,630,440` is in **source-video pixels**. It is mapped to
latent cells by rounding **outward**, so every cell the box touches is
included:

```
(330, 230, 630, 440) px
   -> latent cells x[20:40] y[14:28]        20 x 14 = 280 cells
   -> effective pixels (320, 224, 640, 448)     +8680 px^2 vs. what you typed
```

That growth is why `tools/draw_box.py` draws two rectangles. The region
actually affected is always at least as large as the one you asked for, and
up to 15px larger per side.

From those cells `build_feathered_box_weight` builds a `(44, 78)` map in
[0, 1]:

```
feather=0 : 280 cells at weight 1, hard edge
feather=2 : 280 cells nonzero, 160 at weight 1
            across the left border: 0.00  0.25  0.75  1.00  1.00
```

The raised-cosine ramp has a continuous first derivative, so the decoder finds
no ridge to sharpen into a blocky 16px seam. It grows *inward*, so a box
narrower than `2 * feather` cells never reaches weight 1 anywhere.

### 1.7 The regenerate mask

```
regen_mask  (26, 44, 78)  bool     1 = regenerate, 0 = frozen
mask2       (48, 26, 44, 78) float  same, broadcast over channels
```

In the plain static run this is just the temporal mask broadcast spatially:
every position of latent frames 1..25 is 1, all of frame 0 is 0. (With
`--freeze_box` it also varies spatially -- see section 7.)

`seq_len = 26 * 22 * 39 = 22308` tokens, since the DiT groups 2x2 latent cells
into one token.

---

## 2. `--noise_strength`: two different meanings

This one flag means different things on the two entry paths, which is worth
being precise about.

### 2.1 SDEdit path (no `--cache_path`)

The sampler's 20 steps have a fixed sigma schedule (`shift=5.0`):

```
step:   0      1      2      3   ...  10     ...  19
sigma:  1.000  0.990  0.978  0.966 ... 0.833 ...  0.208
```

`--noise_strength` selects **where in that schedule to start**:

```python
start_idx = round(20 * (1.0 - noise_strength))
sigma0    = scheduler.sigmas[start_idx]
```

| `--noise_strength` | start_idx | sigma0 | steps run |
|---|---|---|---|
| 1.0 | 0 | 1.0000 | 20 |
| 0.6 | 8 | 0.8824 | 12 |
| 0.5 | 10 | 0.8333 | 10 |
| 0.4 | 12 | 0.7692 | 8 |

The initial latent is then built as

```python
latent = (1 - mask2) * z + mask2 * ((1 - sigma0) * z + sigma0 * noise)
         \_____________/   \_______________________________________/
          frozen frames:    regenerated frames: the source content,
          exactly z         partially re-noised to level sigma0
```

At `noise_strength = 1.0`, `sigma0 = 1.0` and that second term is
`0 * z + 1 * noise` -- **pure noise**. The source composition inside the edit
region is erased outright and the model re-invents it. Lower values keep
progressively more of the original scene.

### 2.2 Cache path (`--cache_path trajectory.pt`)

Here it means "how far back to rewind an existing generation". The cache is a
tape written by `generate_initial_video.py --cache_output`: one entry per
sampling step, each holding that step's real intermediate latent. The code
picks the cached step whose sigma is closest to `noise_strength` and resumes
from it. No re-noising and no fresh random noise are involved -- it is the
genuine intermediate state of the run that produced the video, which is
strictly better information than SDEdit's approximation.

This mode always operates on the whole video's latent, never a crop, because
the causal VAE's singleton latent frame 0 only lines up if the window starts
at the true frame 0.

---

## 3. The denoising loop

Per step, in order:

```
for each step, with sigma = sigmas[abs_idx]:

  1. if paste.should_apply(sigma):
         latent = write_delta(...)          <- the correction
         latent = fft_restore(...)          <- repair the damage it caused
         latent = (1-mask2)*z + mask2*latent   <- re-impose frozen frames
  2. build the per-token timestep tensor
  3. noise_pred = uncond + guide_scale * (cond - uncond)
  4. latent = scheduler.step(noise_pred, t, latent)
  5. latent = (1-mask2)*z + mask2*latent      <- re-impose frozen frames again
```

Steps 1 and 5 are the two writes that matter. Everything else is stock Wan2.2.

### 3.1 When the correction fires

```python
should_apply(sigma) = applied < max_steps  and  sigma_lo <= sigma <= sigma_hi
```

Defaults: `--paste_steps 4`, `--paste_sigma_range 0.30 1.00`. So the first
four steps at or above sigma 0.30 get corrected, and the remaining sixteen do
not. Both bounds are load-bearing:

- **Early only**, because `write_delta`'s `x0 ~ z` approximation holds while
  the latent still encodes something close to the source, and degrades as
  denoising moves away from it. SG-I2V's Figs. 10/15/16 show late latent edits
  dissolving the frame into coloured blobs.
- **Only a few steps**, because the remaining ones are deliberately left free
  to harmonise the box seam and re-assert anything the correction stepped on
  -- most importantly a gripper passing in front of the pinned region.

### 3.2 The correction itself

`latent_paste.write_delta`, one line:

```python
latent[t] += (1 - sigma) * weight * strength * (z[anchor] - z[t])
```

for every `t` in `target_frames`. Three things to notice.

**Why a delta and not an overwrite.** Wan's flow-matching convention is
`x = (1 - sigma) * x0 + sigma * eps`. Replacing content while keeping the
destination's own noise means changing only the `x0` term, whose weight is
`(1 - sigma)`. Taking `z` as the estimate of `x0` at both positions gives
exactly the line above. Copying `latent[anchor]` into `latent[t]` outright
would drag `eps_anchor` along with it, and identical noise at two temporal
positions reads to the model as "these are the same frame" -- which freezes
everything in the box, the occluding gripper included.

**It is a no-op where the scene already matches.** The correction is
proportional to `z[anchor] - z[t]`, so it acts only where there is genuine
drift.

**It decays with sigma, and that is not tunable.** The `(1 - sigma)` factor
comes from the convention, not from a design choice. Per-step gains for the
four default paste steps:

| `--noise_strength` | gains on the 4 paste steps | total pull |
|---|---|---|
| **1.0** | 0.000, 0.010, 0.022, 0.034 | **0.066** |
| 0.6 | 0.118, 0.141, 0.167, 0.196 | 0.62 |
| 0.5 | 0.167, 0.196, 0.231, 0.271 | 0.87 |
| 0.4 | 0.231, 0.271, 0.318, 0.375 | 1.19 |

**At `--noise_strength 1.0` the static method barely does anything** -- about
6% of the anchor delta over the whole run, with the very first paste step
writing mathematically nothing. This is the single most common way to get a
run that looks correctly configured and drifts anyway. Either lower
`--noise_strength`, or cap the top of the window with
`--paste_sigma_range 0.30 0.85` so the correction skips the dead steps.

### 3.3 `fft_restore`

Editing a latent pushes it off-distribution -- SG-I2V's Appendix A reports
exactly this for their rejected copy-paste baseline. Their remedy (Eq. 2) is
applied here after every correction: keep the *edited* low spatial
frequencies, restore the *original* high ones, split by a Butterworth filter
at `--fft_ratio 0.5`. Layout and motion live in the low frequencies; the
artifacts live in the high ones, so this discards the damage nearly for free.

It runs only on the frames actually written -- a global spatial operation over
untouched frames would be pure round-trip error.

### 3.4 The per-token timestep

```python
temp_ts = (mask2[0][:, ::2, ::2] * timestep).flatten()
```

Frozen positions get timestep 0, so the DiT reads them as clean conditioning
rather than noise to be denoised. The `::2` subsample is the 2x2 latent cells
per token -- which is why any spatially varying mask must be constant within
each 32px block (see section 7).

---

## 4. Which tensors exist, and what happens to each

| tensor | shape | lifetime |
|---|---|---|
| `z` | (48, 26, 44, 78) | the clean source encoding. **Never modified.** Used for the frozen-frame re-imposition, for the SDEdit init, and as both endpoints of the correction's delta |
| `latent` | (48, 26, 44, 78) | the working noisy latent. Rebuilt every step |
| `weight_t` | (44, 78) | the feathered box map. Constant |
| `mask2` | (48, 26, 44, 78) | the regenerate mask. Constant |
| `noise` | (48, 26, 44, 78) | one fixed draw from `seed_g`, used only to build the init |
| `z_frozen` | (48, 26, 44, 78) | **only with `--freeze_box`**: a clone of `z` with the box's cells replaced by the anchor's. `z` itself stays the honest source encoding, because `write_delta` still needs it |

`write_delta` and `write_delta_trajectory` both `clone()` and return a new
tensor rather than mutating in place, so the pre-correction latent stays
available -- `fft_restore` needs it as its "original".

There is no separate "static latent" object. The pin is not stored anywhere;
it is recomputed each step from `z[anchor]`, which is why nothing accumulates
and why the correction is exactly reproducible.

---

## 5. Output: back to pixels

```
pipe.vae.decode(latent)   -> (3, 101, 704, 1248) float32 in [-1, 1]
vae_output_to_frames      -> (101, 704, 1248, 3) uint8
resize_frames             -> (101, 704, 1248, 3)   no-op here
splice_edited_frames      -> (121, 704, 1248, 3)   uint8
```

The splice writes only `[edit_start, edit_end]` = frames 23..120, taking them
from window-local indices 1..98. The 2 padding frames are past index 98 and
are dropped. Frames 0..22 come straight from the original array.

`assert_outside_range_intact` then checks byte-equality outside the edit range
and raises if the splice logic ever drifts. That check -- not the latent mask
-- is what actually guarantees the rest of the video is untouched.

---

## 6. Order of operations in `main()`

```
read video                     (121, 704, 1248, 3)
  |
build_regeneration_window      window(22, 120, edit 23..120, pad 2)
  |
crop + pad + resize            (101, 704, 1248, 3)
  |
build_latent_regen_mask        26 booleans -> broadcast to (26, 44, 78)
  |
anchor_latent_frame            0  -> source frame 22
  |
scale_boxes                    source px -> model px
  |
[--visualize_boxes exits here, before the model is ever loaded]
  |
StaticPasteEditor.edit()
  |-- vae.encode               z (48, 26, 44, 78)
  |-- _make_paste              feathered weight (44, 78), targets [1..25]
  |-- SDEdit init              latent at sigma0
  |-- _denoise                 20 - start_idx steps, 4 of them corrected
  |-- vae.decode               (3, 101, 704, 1248)
  |
splice + assert                (121, 704, 1248, 3)
  |
write_video_frames             pinned.mp4
```

With `--prompt_file`, everything above the `edit()` call is done once and the
edit is repeated per prompt, so the model loads once.

---

## 7. How `--freeze_box` differs

Same script, different mechanism. `--static_box` nudges a region that is
still being regenerated; `--freeze_box` removes the region from regeneration.

| | `--static_box` | `--freeze_box` |
|---|---|---|
| where it acts | inside the loop, additive | in the mask, structural |
| sigma dependence | scaled by `(1 - sigma)` | none |
| at `--noise_strength 1.0` | ~6% of the delta applied | exact |
| applies on | first 4 steps | every step |
| model can override | yes, later steps | no |
| occluding arm | can re-assert itself | gets clipped |
| box edges | feathered | hard |
| snaps to | 16px latent cells | 32px token grid |

The freeze changes two things in the pipeline above:

- **The mask becomes spatial.** `build_freeze_regen_mask` marks the box 0
  inside the edit range, so `regen_mask` is no longer constant across a frame.
  It pools at 32px (`vae_stride * patch_size`) because the per-token timestep
  reads `mask2[0][:, ::2, ::2]` -- a mask varying inside a token would be
  sampled at an arbitrary corner.
- **The held content becomes the anchor's.** `pin_anchor_content` builds
  `z_frozen`, and the two re-imposition lines use it instead of `z`. This is
  essential: `z` is the drifted source, so freezing to `z` would reproduce the
  drift perfectly.

Everything else -- window, anchor selection, splice -- is identical, and the
two modes compose.

---

## 8. Checking it worked

```bash
# before: confirm it is translation, not deformation, and find --start_frame
python tools/measure_drift.py --video initial3.mp4 \
    --box 330,230,630,440 --box 60,480,300,660

# after: the same command on the output
python tools/measure_drift.py --video pinned.mp4 --box 330,230,630,440
```

The second `--box` on static background is the control: if both regions drift
by the same vector the camera moved, and pinning one region is the wrong
remedy. For `initial3.mp4` the background box never exceeds 2.2px while the
cabinet reaches 135.8px, so the camera is static and the cabinet is what
moves.

Watch the run's own log too:

```
[static] anchor=latent frame 0, targets=1..25 (25 frames), weighted cells=280/3432
[paste] step 10 sigma=0.8333 (1/4)
[paste] step 11 sigma=0.8036 (2/4)
...
```

A `[paste]` line printing at `sigma=1.0000` is the warning sign from section
3.2: the step fired, and wrote nothing.
