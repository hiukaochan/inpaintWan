# Wan2.2 — Architecture & Generation Pipeline

Notes from reading `Wan2.2/` in this repo. Every claim below is traceable to a file/line in the
vendored source, not to the paper.

---

## 1. Repository layout

```
Wan2.2/
├── generate.py                  # single CLI entrypoint for all 5 tasks
└── wan/
    ├── text2video.py            # WanT2V      — MoE (2× 14B experts), Wan2.1 VAE
    ├── image2video.py           # WanI2V      — MoE (2× 14B experts), Wan2.1 VAE
    ├── textimage2video.py       # WanTI2V     — dense 5B, Wan2.2 VAE (high compression)
    ├── speech2video.py          # WanS2V      — audio-driven
    ├── animate.py               # WanAnimate  — character animation / replacement
    ├── configs/                 # per-task hyperparameters (EasyDict)
    ├── modules/
    │   ├── model.py             # ★ WanModel — the DiT backbone
    │   ├── attention.py         # FlashAttention-3 → FA2 → SDPA dispatch
    │   ├── t5.py                # umT5-XXL encoder (text conditioning)
    │   ├── tokenizers.py        # HF tokenizer wrapper
    │   ├── vae2_1.py            # Wan2.1 VAE   (4×8×8,  z=16)
    │   ├── vae2_2.py            # Wan2.2 VAE   (4×16×16, z=48)
    │   ├── s2v/                 # audio encoder, motioner, audio injector
    │   └── animate/             # CLIP, face/motion encoders, face adapter
    ├── distributed/             # FSDP sharding + Ulysses sequence parallel
    └── utils/
        ├── fm_solvers.py        # Flow-matching DPM-Solver++
        ├── fm_solvers_unipc.py  # Flow-matching UniPC (default)
        └── prompt_extend.py     # Qwen / DashScope prompt rewriting
```

The five tasks are dispatched by string matching on `--task` in
`Wan2.2/generate.py:403-540`.

---

## 2. The five pipelines at a glance

| Task | Class | DiT | VAE | latent stride (T,H,W) | z | Native res / fps |
|---|---|---|---|---|---|---|
| `t2v-A14B` | `WanT2V` | 2 × 14B MoE experts | Wan2.1 | 4, 8, 8 | 16 | 480P / 720P @ 16fps |
| `i2v-A14B` | `WanI2V` | 2 × 14B MoE experts | Wan2.1 | 4, 8, 8 | 16 | 480P / 720P @ 16fps |
| `ti2v-5B` | `WanTI2V` | 1 × 5B dense | Wan2.2 | 4, 16, 16 | 48 | 720P @ 24fps |
| `s2v-14B` | `WanS2V` | 1 × 14B + audio blocks | Wan2.1 | 4, 8, 8 | 16 | 480P / 720P |
| `animate-14B` | `WanAnimate` | 1 × 14B + face/pose blocks | Wan2.1 | 4, 8, 8 | 16 | 720P @ 30fps |

All five share the same skeleton: **umT5 text encoder → 3D causal VAE encoder → DiT denoiser
under flow-matching → VAE decoder → mp4**. What differs is the conditioning route into the DiT.

DiT shape parameters (from `wan/configs/`):

| | dim | ffn_dim | heads | head_dim | layers | patch_size |
|---|---|---|---|---|---|---|
| A14B (t2v/i2v/s2v/animate) | 5120 | 13824 | 40 | 128 | 40 | (1,2,2) |
| TI2V-5B | 3072 | 14336 | 24 | 128 | 30 | (1,2,2) |

---

## 3. End-to-end flow

### 3.1 Data path (I2V-A14B, the most instructive case)

Reference: `wan/image2video.py:256-431`.

```
 prompt (str)                          image (PIL)
     │                                      │
     ▼                                      ▼
 umT5-XXL encoder                    resize to (h,w) snapped
 → context   [L≤512, 4096]           to vae_stride × patch_size
 → context_null (neg prompt)                │
     │                                      ▼
     │                          [img , zeros(3, F-1, h, w)]  ← image is frame 0 only
     │                                      │
     │                                 VAE.encode
     │                                      ▼
     │                            y   [16, T', h/8, w/8]
     │                                      │
     │                        concat 4-ch temporal mask (1 for frame 0)
     │                                      ▼
     │                            y   [20, T', h/8, w/8]
     ▼                                      │
 ┌───────────────────────────────────────────────────────────┐
 │  for t in timesteps (UniPC / DPM++, flow-matching):        │
 │    x = concat(latent[16], y[20])  →  36 channels           │
 │    expert = high_noise if t >= boundary else low_noise     │
 │    ε_c = expert(x, t, context)                             │
 │    ε_u = expert(x, t, context_null)                        │
 │    ε   = ε_u + w·(ε_c − ε_u)          ← classifier-free    │
 │    latent = scheduler.step(ε, t, latent)                   │
 └───────────────────────────────────────────────────────────┘
                          │
                          ▼
                    VAE.decode → [3, F, h, w] in [-1,1]
                          │
                          ▼
                 save_video (imageio, libx264)
```

The mask construction (`image2video.py:289-296`) is worth noting: a `[1, F, lat_h, lat_w]` mask
with `1` on frame 0 is *temporally folded* into 4 channels so it matches the VAE's 4× temporal
downsampling — `msk.view(1, T', 4, h, w).transpose(1,2)` gives `[4, T', h, w]`. So the DiT's
input channel count is **16 (noisy latent) + 4 (mask) + 16 (clean cond latent) = 36**.

Note: unlike Wan2.1-I2V, **no CLIP image encoder is used** in Wan2.2 I2V. `image2video.py` imports
only T5, the VAE, and `WanModel`. CLIP survives only in the Animate variant.

### 3.2 T2V (`wan/text2video.py:249-378`)

Same loop, minus the image branch: pure noise `[16, T', H/8, W/8]`, `y=None`, `in_dim=16`.

### 3.3 TI2V-5B — conditioning by latent replacement, not channel concat

`wan/textimage2video.py:461-619` takes a different route. There is no mask, no channel
concatenation, no `y`:

```python
z = self.vae.encode([img])                          # first-frame latent
mask1, mask2 = masks_like([noise], zero=True)       # mask2[:,0] = 0, rest = 1
latent = (1. - mask2[0]) * z[0] + mask2[0] * latent # frame 0 ← clean latent
...
for t in timesteps:
    temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()   # per-token timestep
    temp_ts = cat([temp_ts, ones(seq_len - len) * timestep])
    timestep = temp_ts.unsqueeze(0)                             # [1, seq_len]
    ...
    latent = (1. - mask2[0]) * z[0] + mask2[0] * latent         # re-inject each step
```

Two mechanisms are at play:

1. **Latent replacement** — the first latent frame is overwritten with the clean encoded image
   before and after every solver step.
2. **Per-token timesteps** — tokens belonging to frame 0 get `t = 0`, all other tokens get the
   current `t`. The `[:, ::2, ::2]` slice is the patchify stride (patch_size = (1,2,2)), so the
   mask is downsampled into token space.

This is why `WanModel.forward` accepts a timestep of either shape `[B]` or `[B, seq_len]`
(`model.py:460-469`), and why the AdaLN modulation tensor `e` carries a token axis. The same 5B
weights therefore serve T2V and I2V with no architectural switch — that is the "hybrid TI2V"
claim in the README.

---

## 4. Text encoder — umT5-XXL (`wan/modules/t5.py`)

- **Encoder only.** `T5EncoderModel` builds `umt5_xxl(encoder_only=True)`, so the decoder half of
  the checkpoint is never instantiated (`t5.py:472-512`).
- Config (`t5.py:456-469`): vocab 256384, dim 4096, ffn 10240, 64 heads, **24 encoder layers**,
  `shared_pos=False`.
- Components: `T5SelfAttention` + `T5FeedForward` (gated GELU: `gate * fc1 → fc2`),
  `T5LayerNorm` (RMS-style, no bias/mean subtraction), and `T5RelativeEmbedding` — T5 bucketed
  relative position bias, re-computed per layer because `shared_pos=False`.
- Loaded in **bf16** (`wan_shared_cfg.t5_dtype`), tokenized to a fixed `text_len = 512`, output
  masked to the true sequence length. It can live on CPU (`--t5_cpu`) or be FSDP-sharded.

Output: a list of `[L, 4096]` tensors, one for the prompt and one for the negative prompt.

---

## 5. The 3D causal VAE

Two variants, both `WanVAE_` = `Encoder3d` + `conv1` + `conv2` + `Decoder3d`, both **causal in
time** and both chunked so memory stays flat over long clips.

### 5.1 Shared machinery

- **`CausalConv3d`** — a `nn.Conv3d` with asymmetric temporal padding (all padding on the past
  side), so frame *t* never sees frame *t+1*.
- **Feature caching** — `clear_cache()` allocates one slot per `CausalConv3d`
  (`vae2_1.py:582-589`). The encoder processes frames in chunks of `1, 4, 4, 4, …`
  (`vae2_1.py:516-534`) and the decoder one latent frame at a time
  (`vae2_1.py:552-566`), each conv reading its cached tail from the previous chunk. This makes
  encode/decode streaming rather than whole-clip.
- **`Resample`** — spatial 2× and `upsample3d`/`downsample3d` variants that also handle time.
- **`ResidualBlock`** (RMS_norm → SiLU → CausalConv3d ×2 + shortcut) and a single
  **`AttentionBlock`** (spatial self-attention) in the middle stage.
- **Latent normalization** — a per-channel `mean`/`std` baked into the class
  (`vae2_1.py:629-639`, `vae2_2.py:904-1012`); `encode` returns `(mu - mean) / std`, `decode`
  inverts it. Only `mu` is used at inference; `log_var` is discarded.

### 5.2 Wan2.1 VAE (`vae2_1.py`) — used by T2V/I2V/S2V/Animate

- `dim=96`, `dim_mult=[1,2,4,4]`, `z_dim=16`, `temperal_downsample=[False, True, True]`.
- Compression **4×8×8**, 16 channels.

### 5.3 Wan2.2 VAE (`vae2_2.py`) — used by TI2V-5B

Everything above plus three additions:

- **Pixel-space patchify** (`vae2_2.py:280-313`): `encode` first does
  `patchify(x, patch_size=2)`, folding a 2×2 spatial block into channels (3 → 12 channels,
  hence `CausalConv3d(12, dims[0], 3)` at `vae2_2.py:525`). `decode` unpatchifies at the end.
- **`AvgDown3D` / `DupUp3D`** (`vae2_2.py:316-413`): parameter-free shuffle-style down/up
  sampling used inside `Down_ResidualBlock` / `Up_ResidualBlock`, folding space-time into
  channels rather than strided convolution.
- Larger widths: `dim=160`, `dec_dim=256`, **`z_dim=48`**.

Net compression **4×16×16 = 1024×** on pixels, 48 channels. With the DiT's own (1,2,2) patch
embedding the total is **4×32×32**, which is what makes a 5B model viable at 720p/24fps.

---

## 6. ★ The DiT — `WanModel` (`wan/modules/model.py`)

This is the core. `WanModel(ModelMixin, ConfigMixin)` is loaded via
`from_pretrained(ckpt_dir, subfolder=...)`, so its hyperparameters come from the checkpoint's
`config.json`, not from `wan/configs/`.

### 6.1 Input stage (`model.py:437-478`)

| Step | Code | Shape |
|---|---|---|
| optional concat of conditioning | `x = cat([u, v], dim=0)` | `[36, T', h, w]` for I2V |
| **patch embedding** — `nn.Conv3d(in_dim, dim, k=(1,2,2), s=(1,2,2))` | `patch_embedding` | `[1, dim, T', h/2, w/2]` |
| flatten to tokens | `u.flatten(2).transpose(1,2)` | `[1, L, dim]`, `L = T'·(h/2)·(w/2)` |
| zero-pad to `seq_len` | | `[B, seq_len, dim]` |
| **time embedding** | `sinusoidal_embedding_1d(256, t)` → MLP(SiLU) | `e: [B, seq_len, dim]` |
| **time projection** | `SiLU → Linear(dim, 6·dim)` | `e0: [B, seq_len, 6, dim]` |
| **text embedding** | `Linear(4096,dim) → GELU → Linear(dim,dim)` | `[B, 512, dim]` |

Note the patch is `(1,2,2)` — **no temporal patching**; temporal compression is entirely the
VAE's job. `grid_sizes` `[B,3]` records `(F,H,W)` in token units and is threaded through to RoPE
and to unpatchify.

### 6.2 The transformer block — `WanAttentionBlock` (`model.py:183-259`)

Each of the 40 (or 30) blocks is:

```
        x ──────────────────────────────────────────┐
        │                                           │
   LayerNorm(x)·(1+e₁) + e₀      ← AdaLN modulation │
        │                                           │
   WanSelfAttention  (3D RoPE, full attention)      │
        │                                           │
        └── × e₂ ──────────────────────────────► (+)┘
                                                    │
        ┌───────────────────────────────────────────┤
        │                                           │
   LayerNorm(x)  → WanCrossAttention(context) ──► (+)┘   ← text conditioning
                                                    │
        ┌───────────────────────────────────────────┤
   LayerNorm(x)·(1+e₄) + e₃                         │
        │                                           │
   FFN: Linear(dim,ffn) → GELU(tanh) → Linear       │
        │                                           │
        └── × e₅ ──────────────────────────────► (+)┘
```

Details that matter:

- **Modulation / AdaLN-Zero.** Each block owns `self.modulation`, a learned
  `nn.Parameter(1, 6, dim)`. The six chunks are
  `(shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn)` and are computed as
  `modulation + e0` — i.e. a **per-block learned bias added to the per-token timestep
  projection**. All modulation math is forced to `float32` under
  `torch.amp.autocast('cuda', dtype=torch.float32)` with `assert e.dtype == torch.float32`
  (`model.py:237-247`). Because `e0` carries a token axis, the scale/shift/gate are *per token*,
  which is exactly what makes TI2V's mixed-timestep conditioning work.
- **Normalization.** `WanLayerNorm` = LayerNorm with `elementwise_affine=False` (affine is
  supplied by the modulation instead), computed in fp32 and cast back. Cross-attention gets an
  affine `norm3` when `cross_attn_norm=True` (it is, in every config).
- **QK-norm.** `WanRMSNorm` on q and k inside both attentions (`qk_norm=True` everywhere) — the
  standard stability fix for large-scale attention in bf16.
- **Self-attention is global**, not windowed: `window_size=(-1,-1)` in all configs.
- **Cross-attention** has no RoPE and no length mask (`context_lens=None`); text is padded to a
  fixed 512 and the padding is zeros through the embedding MLP.

### 6.3 3D RoPE (`model.py:28-66`)

Rotary embeddings are split across the three axes of the token grid. With `head_dim = 128`,
`c = 64` complex pairs are split as `[c - 2·(c//3), c//3, c//3] = [22, 21, 21]` →
**22 pairs for frames, 21 for height, 21 for width**. `self.freqs` is built once as a
concatenation of three `rope_params(1024, …)` tables (`model.py:398-405`), so the maximum
extent per axis is 1024.

`rope_apply` reconstructs the per-sample frequency tensor by broadcasting each axis table over
the `(F,H,W)` grid, views q/k as complex, multiplies, and views back as real. It runs in
`float64` with autocast disabled — precision here is deliberate. Padding tokens beyond
`seq_len` are passed through untouched.

### 6.4 Attention kernel (`wan/modules/attention.py`)

`flash_attention()` dispatches, in order: **FlashAttention-3** (`flash_attn_interface`) →
**FlashAttention-2** (`flash_attn`) → **`F.scaled_dot_product_attention`**. It uses the *varlen*
API, packing the batch into one ragged sequence via `cu_seqlens`, which is how the zero-padding
to `seq_len` is made free. q/k/v are cast to bf16 for the kernel and the result cast back.

Caveat visible in the code: FA3 does not support `dropout_p` or `window_size`, and the SDPA
fallback drops the padding mask entirely (it warns).

### 6.5 Output stage

- **`Head`** (`model.py:262-291`): its own 2-way modulation (`shift, scale`), a `WanLayerNorm`,
  then `Linear(dim, prod(patch_size) · out_dim)` = `Linear(dim, 4·16)`. Zero-initialized
  (`init_weights`, `model.py:546`) — standard DiT zero-init so the model starts as identity.
- **`unpatchify`** (`model.py:499-522`): `einsum('fhwpqrc->cfphqwr')` then reshape back to
  `[C_out, F, H, W]` in latent space.

### 6.6 Weight init (`model.py:524-546`)

Xavier-uniform on all `nn.Linear`, `normal_(std=0.02)` on the text and time embedding MLPs,
Xavier on the flattened patch-embedding kernel, zeros on the output head.

---

## 7. The MoE — two timestep experts

This is Wan2.2's headline change, and in code it is remarkably simple. There is **no router and
no per-token gating**. `WanT2V.__init__` / `WanI2V.__init__` load *two complete `WanModel`
instances*:

```python
self.low_noise_model  = WanModel.from_pretrained(ckpt_dir, subfolder='low_noise_model')
self.high_noise_model = WanModel.from_pretrained(ckpt_dir, subfolder='high_noise_model')
```

and `_prepare_model_for_timestep` (`text2video.py:169-201`, `image2video.py:172-204`) picks one
per denoising step:

```python
boundary = config.boundary * num_train_timesteps        # t2v: 0.875·1000 = 875
                                                        # i2v: 0.900·1000 = 900
model = high_noise_model if t >= boundary else low_noise_model
```

- **High-noise expert**: early steps (large `t`) — global layout and motion.
- **Low-noise expert**: later steps (small `t`) — detail refinement.
- 27B total parameters, **14B active per step** — compute and VRAM per step are unchanged
  versus a single 14B model.
- The inactive expert is moved to CPU on every switch when `offload_model=True`, so the boundary
  crossing costs one host↔device transfer per generation, not per step.
- **Guidance scale is also per-expert**: `t2v_A14B.sample_guide_scale = (3.0, 4.0)` — 3.0 for
  low-noise, 4.0 for high-noise (`text2video.py:343-344`). I2V uses `(3.5, 3.5)`.

TI2V-5B has no MoE — one dense `self.model`.

---

## 8. Sampling — flow matching

`num_train_timesteps = 1000`, but the schedule is **flow matching / rectified flow**, not DDPM.

In `FlowUniPCMultistepScheduler` (`utils/fm_solvers_unipc.py`):

- `sigmas = linspace(sigma_max, sigma_min, steps)`, and `alpha_t = 1 - sigma`, `sigma_t = sigma`.
- **Timestep shift**: `sigmas = shift · sigmas / (1 + (shift - 1) · sigmas)` — the `--sample_shift`
  knob. Defaults: T2V **12.0**, I2V **5.0**, TI2V **5.0**; the I2V docstring recommends 3.0 for
  480p.
- `timesteps = sigmas · 1000`.
- `prediction_type = "flow_prediction"`, and the conversion is
  `x0_pred = sample - sigma_t · model_output` — the network predicts a **velocity**, not noise.

Two solvers are selectable via `--sample_solver`:

- `unipc` (default) — `FlowUniPCMultistepScheduler`, multistep predictor–corrector, `solver_order=2`.
- `dpm++` — `FlowDPMSolverMultistepScheduler` + `get_sampling_sigmas` / `retrieve_timesteps`.

Default step counts: 40 (T2V/I2V/S2V), 50 (TI2V), 20 (Animate).

**Classifier-free guidance** is done as two full forward passes per step (conditional and
null-prompt) rather than a batched pass — this halves peak activation memory at the cost of
sequencing, and it lets `torch.cuda.empty_cache()` run between them.

---

## 9. Distributed execution

Three orthogonal switches, all set in `_configure_model`:

**FSDP** (`distributed/fsdp.py`) — `--t5_fsdp` / `--dit_fsdp`. `FULL_SHARD` with an
auto-wrap policy of exactly `lambda m: m in model.blocks`, i.e. one FSDP unit per transformer
block. Mixed precision: bf16 params, fp32 reduce and buffers.

**Ulysses sequence parallel** (`--ulysses_size`, `distributed/sequence_parallel.py` +
`ulysses.py`) — monkey-patches the model at load time:

```python
for block in model.blocks:
    block.self_attn.forward = types.MethodType(sp_attn_forward, block.self_attn)
model.forward = types.MethodType(sp_dit_forward, model)
```

- `sp_dit_forward` chunks the token sequence *and* `e`/`e0` across ranks after embedding
  (`torch.chunk(x, world_size, dim=1)[rank]`), then `gather_forward` before unpatchify.
- `sp_attn_forward` → `distributed_attention`: **all-to-all** to swap the sharding axis from
  sequence to heads, run FlashAttention on full-length/partial-head tensors, all-to-all back.
  This is why `cfg.num_heads % ulysses_size == 0` is asserted in `generate.py:363`.
- The SP variant of `rope_apply` pads the frequency table to `s · sp_size` and slices this
  rank's window (`sequence_parallel.py:50-55`).

**Model offload** (`--offload_model`, default true on single GPU) — text encoder, and each MoE
expert, shuttled between CPU and GPU around their use.

`--convert_model_dtype` casts DiT params to bf16 outright (only valid without FSDP).

---

## 10. Task-specific extensions on top of the DiT

### 10.1 S2V — `wan/modules/s2v/model_s2v.py`

`WanModel_S2V` reuses `WanSelfAttention`/`WanAttentionBlock` and adds:

- **`CausalAudioEncoder`** (`casual_audio_encoder`) over wav2vec2 features
  (`s2v_14B.wav2vec = "wav2vec2-large-xlsr-53-english"`, `audio_dim = 1024`), producing
  `num_audio_token` tokens per frame. `audio_encoder.py` handles fps resampling
  (`get_sample_indices`, `linear_interpolation`) between audio rate and video rate.
- **`AudioInjector_WAN`** — extra cross-attention modules spliced into a *subset* of blocks:
  `audio_inject_layers = [0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39]`, discovered by walking the
  block tree (`torch_dfs`). Optional **AdaIN** conditioning (`enable_adain=True`,
  `adain_mode="attn_norm"`).
- **FramePack motioner** (`enable_framepack=True`, `motion_frames=73`) — compresses previously
  generated frames into motion tokens so long videos can be generated clip-by-clip.
  `MotionerTransformers` (the alternative, mutually exclusive path) is its own 13-layer
  transformer with Swin/causal attention variants.
- **`cond_encoder`** — a second `Conv3d` patch embedding for pose video (`cond_dim = 16`).
- **`trainable_cond_mask = nn.Embedding(3, dim)`** — a learned token-type embedding
  distinguishing generated / reference / motion tokens.
- `zero_timestep=True` — reference and motion tokens are given timestep 0, the same trick as TI2V.

### 10.2 Animate — `wan/modules/animate/model_animate.py`

`WanAnimateModel(ModelMixin, ConfigMixin, PeftAdapterMixin)` — `in_dim=36`, plus:

- **`pose_patch_embedding`** — a second `Conv3d(16, dim, (1,2,2))`; pose latents are *added* to
  the video tokens from frame 1 onward (`x_[:, :, 1:] += pose_latents_`).
- **Motion / face path**: `Generator` (StyleGAN-style `motion_encoder`, `motion_dim=20`) extracts
  per-frame face motion vectors → `FaceEncoder` → **`FaceAdapter`** injected as a residual every
  5th block (`after_transformer_block`, `num_adapter_layers = num_layers // 5 = 8`).
- **CLIP image conditioning**: `XLMRobertaCLIP` (ViT-H/14 + XLM-RoBERTa,
  `modules/animate/clip.py`) → `MLPProj(1280, dim)` → an extra image cross-attention inside
  `WanAnimateCrossAttention`. This is the only Wan2.2 task still using CLIP.
- **LoRA**: `relighting_lora.ckpt` via `PeftAdapterMixin`, enabled by `--use_relighting_lora` for
  the character-replacement mode.
- Preprocessing lives in `modules/animate/preprocess/`: 2D pose estimation, SAM-based masking,
  and pose retargeting.

---

## 11. Prompt extension (`wan/utils/prompt_extend.py`)

Optional (`--use_prompt_extend`) rewriting of the user prompt into the long, cinematic style the
model was trained on. Two backends:

- **`DashScopePromptExpander`** — Alibaba's hosted API (`qwen-plus` for text, `qwen-vl-max` when
  an image is supplied).
- **`QwenPromptExpander`** — a local Qwen / Qwen-VL model.

Rank 0 does the rewrite and broadcasts the result (`generate.py:380-401`). System prompts live in
`wan/utils/system_prompt.py`.

The **negative prompt** is a fixed Chinese string in `shared_config.py`
(over-exposure, blur, subtitles, static frames, malformed hands/limbs, …) used as `context_null`
for CFG whenever `--sample_neg_prompt` is not given.

---

## 12. Worked shape example

**T2V-A14B, 1280×720, 81 frames:**

| Stage | Shape |
|---|---|
| Output video | `[3, 81, 720, 1280]` |
| Latent (`vae_stride = 4,8,8`) | `[16, 21, 90, 160]` — `T' = (81-1)/4 + 1 = 21` |
| After patch embed `(1,2,2)` | `[5120, 21, 45, 80]` |
| Token sequence `seq_len` | `21 × 45 × 80 = 75,600` |
| Text context | `[512, 4096]` → `[512, 5120]` |
| DiT | 40 blocks, self-attn over 75,600 tokens, cross-attn to 512 |

**TI2V-5B, 1280×704, 121 frames:**

| Stage | Shape |
|---|---|
| Output video | `[3, 121, 704, 1280]` |
| Latent (`vae_stride = 4,16,16`) | `[48, 31, 44, 80]` |
| After patch embed | `[3072, 31, 22, 40]` |
| `seq_len` | `31 × 22 × 40 = 27,280` |

The 2.8× shorter sequence at higher resolution and frame count is the whole point of the Wan2.2
VAE — self-attention cost drops roughly 7.7× while the frame count goes *up*.

---

## 13. Summary — component inventory

| Component | File | Role |
|---|---|---|
| `generate.py` | `Wan2.2/generate.py` | CLI, task dispatch, distributed init, saving |
| `WanT2V` / `WanI2V` / `WanTI2V` / `WanS2V` / `WanAnimate` | `wan/*.py` | Pipeline orchestration |
| `T5EncoderModel` (umT5-XXL enc, 24L, d=4096) | `modules/t5.py` | Text → `[512, 4096]` |
| `HuggingfaceTokenizer` | `modules/tokenizers.py` | Text → ids/mask, len 512 |
| `Wan2_1_VAE` (4×8×8, z=16) | `modules/vae2_1.py` | Pixels ↔ latents |
| `Wan2_2_VAE` (4×16×16, z=48) | `modules/vae2_2.py` | Pixels ↔ latents, high compression |
| `CausalConv3d`, feature cache | both VAEs | Causal, chunked, O(1)-memory streaming |
| **`WanModel`** | `modules/model.py` | The DiT denoiser |
| ├ `patch_embedding` `Conv3d(in,dim,(1,2,2))` | | Latents → tokens |
| ├ `time_embedding` + `time_projection` | | Sinusoidal → 6-way AdaLN modulation |
| ├ `text_embedding` MLP | | 4096 → dim |
| ├ `WanAttentionBlock` ×40 | | self-attn + cross-attn + FFN, all AdaLN-gated |
| │ ├ `WanSelfAttention` + 3D RoPE (22/21/21) | | Spatiotemporal mixing |
| │ ├ `WanCrossAttention` | | Text conditioning |
| │ └ `WanRMSNorm` / `WanLayerNorm` | | QK-norm and affine-free LN |
| └ `Head` (zero-init) + `unpatchify` | | Tokens → latent velocity |
| `flash_attention` | `modules/attention.py` | FA3 → FA2 → SDPA, varlen |
| `FlowUniPCMultistepScheduler` | `utils/fm_solvers_unipc.py` | Default flow-matching solver |
| `FlowDPMSolverMultistepScheduler` | `utils/fm_solvers.py` | Alternative solver |
| `shard_model` (FSDP) | `distributed/fsdp.py` | Per-block FULL_SHARD |
| `sp_dit_forward` / `distributed_attention` | `distributed/` | Ulysses sequence parallel |
| `PromptExpander` | `utils/prompt_extend.py` | Qwen-based prompt rewriting |
| `save_video` | `utils/utils.py` | Tensor → mp4 (imageio, libx264) |
