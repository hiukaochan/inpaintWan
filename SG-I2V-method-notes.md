# SG-I2V: how the trajectory control works

Notes on *SG-I2V: Self-Guided Trajectory Control in Image-to-Video Generation*
(Namekata, Bahmani, Wu, Kant, Gilitschenski, Lindell — ICLR 2025), read
alongside the reference implementation in `SG-I2V/`.

**Scope:** explanation only. No implementation, no port to this repo.

---

## 1. The problem and the control interface

Given an input image, an image-to-video model animates it — but *how* it
animates it is decided by the seed. Getting a specific motion means
re-rolling seeds and re-phrasing prompts until something acceptable appears.
Prior fixes fine-tune the generator on motion-annotated data, which is
expensive and needs datasets that are hard to procure.

SG-I2V is **zero-shot**: it never touches the weights. It wraps a frozen
Stable Video Diffusion (SVD) and steers the *sampling process* instead.

**The user provides** (Sec. 3.1): a set of `B` bounding boxes drawn on the
input image, each with a per-frame centre position over the `N` output
frames. Box `b` in frame `n` is `B_{b,n} = {h_b, w_b, c_{b,n}}` — note the
height/width are fixed; **only the centre moves**. The promise is that
whatever falls inside the box in the input image ends up at the box's
position in each output frame.

One formulation covers both kinds of motion:

- **Object motion** — put a box around the object, draw where it should go.
- **Camera motion** — put boxes on *static background* regions and give them
  the trajectory *opposite* to the desired camera movement.
- **Hold something still** — give it the zero trajectory.

This unification is a small but real contribution; earlier controllable-I2V
frameworks treated object and camera control as separate mechanisms.

### How the input is actually encoded

`SG-I2V/inference.py:17-51` — a directory with `img.png` and `traj.npy`,
where `traj.npy` has shape `[N, 2+F, 2]`:

- `[:, :2]` → the box's top-left and bottom-right corners as `(w, h)`
- `[:, 2:]` → the box centre in each of the `F` frames, as `(w, h)`

The per-frame box is then built by **translating the first-frame box** by the
centre delta `d = centre[i] - centre[0]` (`inference.py:46-49`). So the box
never changes size; the trajectory is pure translation. Coordinates are
rescaled from the source image's resolution to the generation resolution
first (`inference.py:36-37`).

---

## 2. Why the obvious approach fails

The natural zero-shot recipe, borrowed from image editing (DragDiffusion,
DIFT): diffusion feature maps are *semantically aligned* — pixels on the same
object have similar feature vectors — so you can move an object by optimizing
the latent until the features at the target location match the features at
the source location.

In **image** diffusion models this works, using the output of mid-resolution
upsampling blocks. **In SVD it does not.** The paper's central empirical
finding (Sec. 3.2, Fig. 2, and Figs. 11-12 in the appendix) is that SVD's
feature maps are only *weakly* correlated **across frames** — PCA-visualised,
the same object takes on a different colour in each frame. Optimizing against
a signal that doesn't correspond across frames gives no useful layout control
(Fig. 13: the flower simply fails to follow the path).

The architectural reason is specific and worth stating precisely:

- SVD's **spatial** self-attention is applied *per frame independently* — it
  has no cross-frame information at all, so nothing forces frame 5's features
  into the same space as frame 1's.
- SVD's **temporal** attention does cross frames, but only attends to *the
  same pixel position* across time. That's inadequate for capturing semantics
  spatially — which is why Fig. 5 shows temporal-layer features score poorly
  on motion fidelity (ObjMC).

Meanwhile, the layout of the video is settled at *early* denoising steps, so
this is exactly where the guidance needs to act. Hence the dilemma: you must
optimize early, but early features are not comparable across frames.

---

## 3. The method, in three parts

### Part 1 — Manufacture aligned features (Sec. 3.2)

Rather than searching for a naturally aligned feature, they *force* alignment
with a modification to spatial self-attention.

Normally, for frame `n`: `F_n = Softmax(Q_n K_nᵀ / √D) · V_n`.

SG-I2V replaces frame `n`'s key and value with **frame 1's**:

> `F̃_n = Softmax(Q_n · SG(K_1)ᵀ / √D) · SG(V_1)`

where `SG(·)` is stop-gradient. Every frame's output is now a weighted
combination of the *same* value vectors `V_1`, so a crop from frame 5 is
directly comparable to a crop from frame 1 — while `Q_n` preserves each
frame's own layout. The stop-gradient on `K_1`/`V_1` keeps the subsequent
optimization stable.

Two properties matter:

1. **It runs only during loss computation.** The actual denoising pass uses
   the unmodified attention. In the code
   (`SG-I2V/src/model.py:149-159`) the modified attention is computed
   *in addition to* the real one, guarded by `self.training and (layer,
   sublayer) in self.record_layer_sublayer`; the value returned to the
   network at `model.py:161-163` is always the original. `train(True)` is
   being used purely as an "am I in the optimization pass" flag, not to
   enable training.
2. **What gets recorded is the raw attention output**, reshaped to
   `(frames, h, w, heads·dim)` — captured *before* the output projection
   `to_out`.

Fig. 12 ranks the candidates: self-attention features beat temporal-attention
and upsampling-block features, and the modified self-attention is the most
aligned of all, by construction.

### Part 2 — Optimize the latent (Sec. 3.3)

With comparable features `F̃_n(z_t)` in hand, the noisy latent `z_t` itself
becomes the optimization variable (**not** the weights — this is what makes
the method zero-shot):

> `z_t* = argmin Σ_{b,n} ‖ G_b ⊙ ( F̃_n(z_t)[B_{b,n}] − SG( F̃_1(z_t)[B_{b,1}] ) ) ‖₂`

In words: *the features inside box `b` at its frame-`n` position should look
like the features inside that box at its frame-1 position.* The frame-1 crop
is stop-gradiented — it's a fixed target, not something being pulled toward
the others.

`G_b` is a **Gaussian heatmap** the size of the box, weighting the centre
higher than the edges. The rationale (from DragAnything) is that a rectangle
around an object inevitably includes background pixels near its corners that
should *not* be dragged along. σ = 0.4·(h/2), 0.4·(w/2). Table 2 shows it is
a small but consistent win (ObjMC 14.43 vs 14.72) — a refinement, not a
load-bearing component.

Implementation notes from `SG-I2V/src/pipeline.py:45-163`:

- All recorded feature maps are bilinearly upsampled to the **latent**
  resolution and concatenated along the channel axis (lines 78-85), so boxes
  map onto the grid by a single division by 8 (the VAE stride, line 89).
- The frame-1 target is captured on the first loop iteration and detached
  (line 119).
- When the box has drifted, the frame-1 crop is bilinearly **resized** to the
  current box's size before comparison (lines 123-126) — which for
  translation-only trajectories only matters at the frame edges.
- Boxes that run off the frame are cropped to the visible part and the
  off-frame remainder is skipped (lines 104-115); a box that is *already*
  outside on frame 1 is a hard error.
- The loss is a sum of `mask · per-pixel MSE`, normalised by the total mask
  weight (line 144) — a weighted mean, so it doesn't scale with box area.
- Optimization runs on the **conditional branch only** — no classifier-free
  guidance (`pipeline.py:267` passes `image_latents[1:]`, `image_embeddings[1:]`).
- AdamW, lr 0.21, 5 iterations, re-instantiated per timestep (line 148).

**Where and when this is applied** is not incidental — two selections carry
much of the method's performance:

- **Which layers.** Features come from mid-resolution self-attention layers
  in the *upsampling* path — specifically layers 2 and 3 of resolution level
  2, `record_layer_sublayer = [(2,1), (2,2)]`. Fig. 6 shows the bottom and
  top levels are both worse, and the top level is *catastrophic* (FID jumps
  to ~70). Combining two mid layers beats either alone.
- **Which timesteps.** Only `t ∈ [30, 45]` of 50, i.e. **early** denoising
  (`t=50` is pure noise). Figs. 10, 15 and 16 show both boundaries are real:
  optimizing late (`t = 10, 20`) shreds visual quality — FID rises past 270,
  and Fig. 15's last frame dissolves into coloured blobs — while at `t > 45`
  the noise is so heavy there is no semantic signal to guide with. Stopping
  at `t = 40` gives insufficient motion control; a *range* beats any single
  timestep.

### Part 3 — Repair the damage with an FFT filter (Sec. 3.4)

The optimized latent `z_t*` is no longer a plausible sample from the
diffusion process — it has drifted off-distribution, and the video comes out
over-smoothed and artifact-ridden (Fig. 7, left).

The fix, inspired by FreeInit's observation that motion lives in the *low*
frequencies of the noisy latent: **keep the optimized latent's low
frequencies, restore the original latent's high frequencies.**

> `z̃_t = IFFT₂D( FFT₂D(z_t*) ⊙ H_γ + FFT₂D(z_t) ⊙ (1 − H_γ) )`

`H_γ` is a Butterworth low-pass filter (order 4, cutoff γ = 0.5), applied per
frame over the spatial `(H, W)` axes (`pipeline.py:165-184`,
`utils.py:83-97`).

Fig. 8 shows both extremes fail and that the trade-off is unusually benign:
γ = 1 (keep everything optimized) badly degrades FID/FVD; γ = 0 (discard the
optimization) leaves ObjMC at ~40, i.e. no control at all. In between, motion
control is nearly flat while quality improves — the motion signal really does
live almost entirely in the low frequencies, so this discards artifacts
almost for free.

---

## 4. The loop, end to end

Per denoising step (`SG-I2V/src/pipeline.py:262-283`):

```
for i, t in enumerate(timesteps):            # 50 steps
    if (50 - i) in range(30, 46):            # early steps only
        # ---- optimize_latent ----
        for _ in range(5):                   # 5 AdamW iterations
            forward UNet up to the recorded layer   (early-exit)
              └─ recorded layers also compute modified self-attn (K/V ← frame 1)
            crop features per box per frame, Gaussian-weighted MSE vs frame 1
            backprop to z_t, AdamW step (lr 0.21)
        z_t ← FFT mix: low freq from optimized, high freq from original
    # ---- normal denoising, unmodified ----
    noise_pred = UNet(z_t, ...)   with CFG
    z_t ← scheduler.step(...)
```

Two efficiency details in the recording pass: the UNet **returns early**
once it is past the deepest recorded layer, skipping the remaining upsampling
blocks and the output convolutions entirely (`model.py:90-91`, `109-110`), and
gradient checkpointing is on (`pipeline.py:53`). Only the latent carries
gradients — the weights are frozen throughout.

---

## 5. Hyperparameters

From `SG-I2V/inference.py:123-137`:

| Parameter | Value | Notes |
|---|---|---|
| Resolution / frames | 576×1024, 14 | Full native SVD resolution — supervised baselines fine-tune at 320×576 |
| Sampling steps | 50 | Euler discrete |
| Optimized timesteps | `range(30, 46)` | 15 of 50, early |
| Iterations per timestep | 5 | 75 optimization iterations total |
| Learning rate | 0.21 | Fig. 9: higher → artifacts, lower → weak control |
| Recorded layers | `[(2,1), (2,2)]` | mid-resolution up-path self-attention |
| Gaussian σ | 0.4 | as a fraction of half the box size |
| FFT cutoff γ | 0.5 | Butterworth order 4 |

**Cost** (Appendix C): ~305 s per video on an A6000 48GB, peak GPU memory
**~30 GB**, driven by backpropagation. Runtime scales with the number of
trajectory conditions.

---

## 6. Results, honestly read

On VIPSeg (Table 1), against **supervised** baselines that were fine-tuned on
motion-annotated data:

| Method | FID ↓ | FVD ↓ | ObjMC ↓ | Zero-shot |
|---|---|---|---|---|
| DragNUWA v1.5 | 30.73 | 253.57 | 10.84 | |
| DragAnything | 30.81 | 268.47 | 11.64 | |
| SVD (no control) | 30.50 | 340.52 | 39.59 | ✓ |
| FreeTraj† | 46.61 | 394.14 | 36.43 | ✓ |
| DragDiffusion† | 30.93 | 458.29 | 31.49 | ✓ |
| **SG-I2V** | **28.87** | 298.10 | 14.43 | ✓ |

Reading this fairly:

- **Best visual quality in the table** (FID 28.87) — but partly because it
  keeps SVD's native 576×1024, while supervised baselines were fine-tuned at
  lower resolution.
- **Motion fidelity trails the supervised methods** (14.43 vs 10.84) — the
  paper's claim is "narrows the gap", not "closes it". Against the no-control
  baseline at 39.59, though, the control is clearly doing real work.
- **The adapted zero-shot baselines are far behind** (31-36 ObjMC), which is
  the paper's main argument: text-to-video zero-shot tricks do not simply
  transfer to image-to-video models.
- Qualitatively (Fig. 4) they claim DragNUWA tends to *distort* objects
  rather than move them, and DragAnything is weak at part-level control (it
  was trained on entity-level masks), whereas SG-I2V handles arbitrary
  control-region granularity.

---

## 7. What is actually load-bearing

If you were to keep only some of this, the ablations say the ranking is:

1. **The modified self-attention** — without aligned features the
   optimization simply doesn't steer (Fig. 13). This is the paper.
2. **Optimizing early, over a range of timesteps** — the difference between a
   working method and coloured mush (Figs. 10, 15, 16).
3. **The mid-resolution layer choice** — the top level is catastrophic (Fig. 6).
4. **The FFT post-processing** — not needed for *control*, but the difference
   between usable and unusable output (Figs. 7, 8).
5. **The Gaussian weighting** — a genuine but marginal refinement (Table 2).

## 8. Limitations the authors state

- Quality is upper-bounded by the frozen base model; large motions and
  complex physical interactions remain hard.
- The optimized latent is fundamentally out-of-distribution. The FFT step
  mitigates this rather than solving it, and artifacts still appear at higher
  learning rates. They call keeping latents in-distribution an open problem.
- Everything is validated on SVD (a UNet with *separate* spatial and temporal
  attention). Whether the approach transfers to other architectures is left
  as future work — and note that Part 1 exists specifically to patch a
  UNet-shaped hole, so the analysis would need redoing on any backbone with
  full spatio-temporal attention.

---

## Source map

| Concept | Location |
|---|---|
| Input format, box-trajectory construction | `SG-I2V/inference.py:17-51` |
| Hyperparameters | `SG-I2V/inference.py:123-137` |
| Attention processor injection | `SG-I2V/src/model.py:21-28` |
| Modified self-attention + feature recording | `SG-I2V/src/model.py:149-159` |
| Early exit during the recording pass | `SG-I2V/src/model.py:90-91`, `109-110` |
| Gaussian heatmap | `SG-I2V/src/pipeline.py:23-43` |
| Latent optimization (Eq. 1) | `SG-I2V/src/pipeline.py:45-163` |
| FFT post-processing (Eq. 2) | `SG-I2V/src/pipeline.py:165-184` |
| Butterworth filter | `SG-I2V/src/utils.py:83-97` |
| Denoising loop / where optimization fires | `SG-I2V/src/pipeline.py:262-283` |
| Trajectory visualisation | `SG-I2V/src/utils.py:51-81` |
