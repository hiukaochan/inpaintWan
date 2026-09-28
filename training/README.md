# Fine-tuning Wan2.2 TI2V-5B on DROID

LoRA fine-tune of TI2V-5B for **first frame + caption -> robot rollout video**,
trained on the [DROID](https://droid-dataset.github.io/) dataset.

The point is domain adaptation. The scripts at the repo root
(`generate_initial_video.py`, `static_range_edit.py`) are all zero-shot -- as
`static_range_edit.py`'s docstring puts it, *"the model is never fine-tuned."*
They work considerably better on a model that already knows what a Franka arm
on a lab bench looks like, which is what this produces.

Upstream Wan2.2 ships no training code, and neither did this repo before
`training/`. Everything here is written against the vendored
`Wan2.2/wan/` modules, which stay untouched.

## How TI2V-5B does image conditioning

Worth understanding before changing anything, because it is not the usual
design and the training code is shaped entirely around it.

TI2V-5B has **no `y`-channel concat** for the conditioning image --
`in_dim == out_dim == 48`, so there is nowhere to put one. Instead
`WanTI2V.i2v()` (`Wan2.2/wan/textimage2video.py`) does two things:

1. **Hard-pastes** the clean first-frame latent into latent frame 0, before and
   after every denoising step:
   `latent = (1-mask2)*z + mask2*latent`
2. **Zeroes the per-token timestep** on that frame's tokens:
   `temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()`

So the model is *told* frame 0 is already clean, and it is handed a clean
frame 0. Training has to present exactly the same thing, or the model learns a
task subtly adjacent to the one it is asked to perform at inference -- a
failure that never shows up as a bad loss curve.

`masks_like()` in `Wan2.2/wan/utils/utils.py:172` turns out to be upstream's
**training-time** conditioning sampler, shipped inside the inference repo.
Given a `generator` it picks, per sample, either:

- **conditioned** -- `mask1[:,0] = exp(N(-3.5,0.5))` (~0.03, a small noise
  augmentation so the model tolerates the VAE reconstruction error of a real
  first frame), and `mask2[:,0] = 0`
- **unconditioned** -- both left at 1, i.e. plain t2v

`training/conditioning.py` reuses it verbatim, and `test_conditioning.py`
asserts the result is `torch.equal` to what `i2v()` builds. Note the default
`p=0.2` means 80% *unconditioned*, reflecting TI2V-5B's joint t2v/i2v
pretraining; for an I2V adapt we invert it to `--cond_prob 0.9`, keeping a
little t2v so the unconditional branch CFG relies on does not drift.

## Hardware

Tokens are `L = T_lat * (H/32) * (W/32)` with `T_lat = (F-1)/4 + 1`
(VAE stride `(4,16,16)`, patch `(1,2,2)`). The default 640x352 @ 49 frames
gives a `[48,13,22,40]` latent and **L = 2860**.

Fixed floor is about **12.5 GB**: 10.0 GB bf16 DiT + ~1.3 GB LoRA optimizer
state (r=32, 80.6M params) + ~1.2 GB CUDA/NCCL context. Activations with
gradient checkpointing run ~0.4 MB/token.

| config | L | total @ bs1 |
|---|---|---|
| **640x352 @ 49f** (default) | 2,860 | **13.6 GB** |
| 1280x704 @ 49f | 11,440 | 17.0 GB |
| 1280x704 @ 81f | 18,480 | 19.7 GB |
| 1280x704 @ 121f (native) | 27,280 | OOM on 24 GB |

640x352 is exactly half of TI2V's native 1280x704, so the RoPE grid stays
aspect-consistent. Self-attention is O(L^2) in compute, but flash-attn 2 keeps
it O(L) in memory.

A single 24 GB card is enough. Two are better.

This project targets **GPUs 0 and 1**. `~/.bashrc` pins
`CUDA_VISIBLE_DEVICES=0,1`, but note that `.bashrc` returns early for
non-interactive shells, so anything launched via `ssh host '<cmd>'`, cron, or a
job runner will **not** pick it up. The commands below set it explicitly for
that reason.

## Run this when the GPUs are free

### 0. Check they are actually free

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

Training needs **~12.5 GB free** on a card before it will even start (that is
the floor: bf16 DiT + LoRA optimizer state + CUDA context). A card showing
21 GB used is unusable regardless of who owns it. Check who is on it before
assuming it is stale:

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
ps -o user=,etime=,comm= -p <pid>
```

### 1. Smoke test first (already cached)

The 525-episode smoke cache is **already built** at
`training/cache/640x352_f49s1` — 1,630 train + 88 val clips, 2.3 GB. Nothing to
re-run. Go straight to:

```bash
export CUDA_VISIBLE_DEVICES=0,1

# does the loop actually run, and at what step time / peak VRAM?
python training/lora_train.py \
    --cache_dir training/cache/640x352_f49s1 \
    --index training/index_smoke.json \
    --overfit_one --max_steps 200 --log_every 10

# then a short real run on the smoke set
python training/lora_train.py \
    --cache_dir training/cache/640x352_f49s1 \
    --index training/index_smoke.json \
    --max_steps 1000 --save_every 500
```

`--overfit_one` trains on a single clip; the loss should collapse toward zero.
If it plateaus, the loss mask or the velocity sign is wrong — see Gotchas for
the one case where a flat loss is *expected*.

Confirm peak VRAM lands near the predicted 13.6 GB before committing to a long
run.

### 2. Full run

```bash
export CUDA_VISIBLE_DEVICES=0,1

# ~30k episodes; takes a few seconds
python training/build_index.py --output training/index.json

# both stages skip existing shards, so re-running is safe and resumable
for r in 0 1; do
  python training/precompute_cache.py --stage video --index training/index.json \
      --device cuda:$r --rank $r --world_size 2 &
done; wait

python training/precompute_cache.py --stage text --index training/index.json \
    --device cuda:0 --text_batch_size 16

torchrun --nproc_per_node=2 training/lora_train.py \
    --cache_dir training/cache/640x352_f49s1 \
    --index training/index.json \
    --max_steps 20000 --save_every 1000

python training/sample_eval.py \
    --adapter training/runs/droid_lora/final \
    --index training/index.json --num 4 --base_too
```

Rough costs, and which are measured:

| stage | disk | time |
|---|---|---|
| index | trivial | ~1 s (measured) |
| video cache, full | ~200 GB | not measured; the 525-episode smoke set took ~25 min on one contended card |
| text cache, full | ~39 GB | not measured; runs fine on CPU (`--device cpu`) if the GPUs are busy |
| training | ~320 MB per adapter | **not measured** — no run has had a free card yet |

Step time and time-to-convergence are deliberately absent rather than guessed.
Fill them in from the smoke run's bar.

Long runs are worth backgrounding. The progress bars throttle themselves to one
line every 30 s when stderr is not a terminal, so a redirected log stays
readable:

```bash
nohup torchrun --nproc_per_node=2 training/lora_train.py \
    --cache_dir training/cache/640x352_f49s1 --index training/index.json \
    > train.log 2>&1 &
tail -f train.log
```

## Workflow

### 1. Build the index

```bash
python training/build_index.py --output training/index.json
# 30005 episodes (29874 train / 131 val), 17360 sessions
```

Walks DROID at depth 2 (`<session>/<camera_serial>/`) and writes `index.json`.
Three things it handles that a naive glob does not:

- The dataset root contains a **self-symlink** `droid -> <root>`. Any recursive
  walk that follows symlinks loops forever; the scan uses
  `follow_symlinks=False`.
- Of 38,656 `rgb.mp4` files, ~1271 have no caption and ~275 no metadata, so the
  index is built by **intersecting** all three files rather than globbing
  videos.
- Container frame rates are a bogus `179/12`. True fps comes from
  `../download_log.csv` (94% coverage; the rest fall back to DROID's nominal
  14.9254).

Sessions in `../benchmark_manifest.json["kept"]` go to the `val` split so the
curated benchmark stays genuinely held out. `--all_outcomes` adds the
`__failure__` sessions; `--limit N` cuts a smoke-test subset.

### 2. Precompute latents and text embeddings

Caching is **mandatory, not an optimization**: T5 (11.4 GB) + VAE (2.8 GB) +
DiT (10 GB) is 24.2 GB, over budget before a single activation. After this
runs, training loads only the DiT.

```bash
python training/precompute_cache.py --stage video --index training/index.json
python training/precompute_cache.py --stage text  --index training/index.json
```

Separate stages because the two encoders should not be co-resident either.
Both are resumable (existing shards are skipped) and shard via
`--rank/--world_size`, so you can run one process per GPU:

```bash
for r in 0 1; do
  python training/precompute_cache.py --stage video --index training/index.json \
      --device cuda:$r --rank $r --world_size 2 &
done; wait
```

(`--device cuda:N` indexes into whatever `CUDA_VISIBLE_DEVICES` already
exposes — with the expected `0,1`, `cuda:0` and `cuda:1` are physical GPUs 0
and 1. It composes with the restriction instead of overriding it, which is why
these examples do not set `CUDA_VISIBLE_DEVICES` per command.)

Re-running either stage over an already-complete cache is cheap: the video
stage loads the VAE lazily, so a pass that finds every shard present allocates
no GPU memory at all and takes seconds. Use it to verify a cache is intact.

The text stage also runs fine **on CPU** (`--device cpu`) when the GPUs are
busy -- useful, since it only needs to happen once. Keep `--text_batch_size`
at 16 or so; encoding one caption at a time is several times slower.

Video preprocessing per clip: center-crop 1280x720 -> 1280x704 (8 px off top
and bottom -- squashing to 704 distorts scene geometry), resize, then take 49
frames at `--temporal_stride`. Clips are non-overlapping and each is encoded
**independently**, because the Wan VAE is causal over time: slicing a long
cached latent at an arbitrary offset does not reproduce a standalone encode of
those frames.

`Wan2_2_VAE.encode()` applies the per-channel mean/std itself, so cached
latents are training-ready. **Do not rescale them again.**

Footprint at 640x352: ~1.1 MB per clip and ~1 MB per caption, so roughly
200 GB of latents and 39 GB of text for the full 30k episodes.

### 3. Train

```bash
# single GPU
python training/lora_train.py --cache_dir training/cache/640x352_f49s1

# two GPUs
torchrun --nproc_per_node=2 training/lora_train.py \
    --cache_dir training/cache/640x352_f49s1
```

The step, in full:

```python
sigma = sample_sigma(shift=5.0)                  # logit-normal, then Wan's shift
mask1, mask2 = masks_like([noise], zero=True, generator=g, p=0.9)
x_t    = add_noise(z0, noise, sigma, mask1[0])   # per-frame noise scale
target = noise - z0                              # flow-matching velocity
t_in   = build_per_token_timestep(mask2[0], sigma*1000, seq_len)
loss   = ((pred - target)**2 * mask2[0]).sum() / mask2[0].sum()
```

The loss mask matters: the conditioning frame's target is meaningless there,
since the input is clean `z0` and the model was told `t=0`. Scoring it would
train the model against noise.

LoRA covers every `nn.Linear` in a `WanAttentionBlock` --
`self_attn.{q,k,v,o}`, `cross_attn.{q,k,v,o}`, `ffn.0`, `ffn.2` (`WanCrossAttention`
subclasses `WanSelfAttention`, so both expose plain q/k/v/o). Deliberately
excluded: `modulation` (a raw `nn.Parameter` peft cannot wrap),
`patch_embedding` (the Conv3d defining the latent->token mapping, which must
stay inference-compatible), `head`, `time_embedding`, `time_projection`, and
all norms. Those last are precisely the parameters that, perturbed, break the
per-token-timestep conditioning path.

Adapter weights are held in fp32 for optimizer stability while the frozen base
stays bf16; autocast casts them down for the matmuls. Gradient checkpointing is
installed by wrapping each block's bound `forward`
(`use_reentrant=False`, required to compose with DDP) rather than editing the
vendored model.

Knobs worth knowing:

| flag | default | what it does |
|---|---|---|
| `--cond_prob` | 0.9 | P(i2v-conditioned sample); see the mask discussion above |
| `--caption_dropout` | 0.1 | swaps in the cached null embedding, for CFG |
| `--shift` | 5.0 | must match the sampler's shift |
| `--lora_rank` | 32 | 80.6M trainable params (1.6%) |
| `--grad_accum` | 8 | effective batch 8 per GPU |
| `--overfit_one` | off | train on one clip; loss should collapse |

### 4. Sample

```bash
python training/sample_eval.py --adapter training/runs/droid_lora/final \
    --index training/index.json --num 4 --base_too
```

Its own process, never an in-loop hook -- it loads T5 + VAE + DiT together,
which is ~24 GB on its own. Takes first frames and captions straight from
held-out episodes so output is directly comparable to ground truth.
`--base_too` emits the frozen model's version alongside; the adapter is only
interesting relative to what the base model already did.

The adapter is applied with `PeftModel.from_pretrained(...).merge_and_unload()`,
which keeps peft out of the forward path so the freeze/inpaint hooks in the
root scripts still apply cleanly.

## Gotchas

**Loss pinned at exactly 2.0 with zero LoRA gradients.** `WanModel.init_weights`
zero-initializes the output head (`model.py:546`, standard DiT practice), so a
*freshly constructed* model predicts exactly zero and no gradient reaches any
block. The loss then sits at `E[(noise-z0)^2] = 2`. This is correct behaviour
for an untrained model and does not happen with the real checkpoint -- but it
is worth recognizing, because it looks exactly like a broken training loop.

**Only half the LoRA parameters get gradients on step 1.** `lora_B` is
zero-initialized, so `lora_A`'s gradient is `dL/dout * lora_B`, i.e. zero until
`lora_B` moves. Expected, not a bug.

**Frame rate.** DROID runs at ~14.9 fps and Wan's prior is 24 fps, but you
cannot resample up without a frame interpolator -- duplicating frames only
makes motion stutter. **Output fps is a decode-time label that does not affect
training at all.** What does affect it is `--temporal_stride`, which sets how
much real time a 49-frame clip spans: stride 1 = 3.3 s, stride 2 = 6.6 s. Most
DROID tasks take 10-17 s, so a larger stride captures more of the task per clip
at the cost of coarser motion.

**Caption style.** DROID captions are dense 60-110 word VLM descriptions
("A white and black robotic arm with a claw-like gripper descends from the
ceiling in a kitchen setting..."), not short task instructions. `text_len=512`
means no truncation, but the adapter is tuned on that distribution -- prompting
it with "pick up the can" at inference is off-distribution and underperforms.

**Stereo views.** Sessions have 1-3 camera serials: different views of the same
episode, each with its own caption. Fine as independent samples at bs=1.

**Dtypes.** `WanModel.forward` returns a list of **fp32** tensors. Cast the
target up, not the prediction down.

**Progress bars and logs.** All three long-running scripts use tqdm. Bars go to
stderr and throttle to one update per 30 s when stderr is not a terminal, so
`nohup ... > train.log` produces a readable log rather than megabytes of
carriage returns. Log lines go through `tqdm.write()`; if you add a bare
`print()` to the train loop it will tear the bar in half. Under DDP only rank 0
draws a bar.

## Local, model-free checks

Both run on CPU with no checkpoints, in the style of the repo's other tests:

```bash
python test_conditioning.py    # 11 tests
python test_training_step.py   #  7 tests
```

`test_conditioning.py` pins the training-time conditioning against
`WanTI2V.i2v()` element for element -- the per-token timestep, the noise
interpolation, and the loss mask all have to agree or the model trains on the
wrong task. `test_training_step.py` runs the real `WanModel` forward and
backward on a deliberately tiny model, checking that gradients reach LoRA and
only LoRA, that gradient checkpointing changes nothing numerically, and that
the loss ignores the conditioning frame. It patches in an SDPA stand-in because
`flash_attention` requires CUDA.
