"""Dataset over the precomputed DROID latent/text cache.

Reads what `precompute_cache.py` wrote, so this does no video decoding and
touches neither the VAE nor T5. Each item is one 49-frame clip:

    latent  [48, 13, h, w]  fp16, already VAE-normalized
    context [L, 4096]       fp16, L varies with caption length

`WanModel.forward` takes *lists* of unbatched tensors, so `collate` returns
lists rather than stacked tensors.
"""

import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset


def _split_shard_name(stem: str) -> tuple[str, str, str]:
    """'<session>__<serial>__<chunk>' -> parts.

    Session names contain '__' themselves (e.g. 'AUTOLab__success__2023-...'),
    so this splits from the right.
    """
    session, serial, chunk = stem.rsplit("__", 2)
    return session, serial, chunk


class DroidLatentDataset(Dataset):
    def __init__(self, cache_dir: str | Path, split: str = "train", index_path: str | Path | None = None):
        self.cache_dir = Path(cache_dir)
        self.video_dir = self.cache_dir / "video"
        self.text_dir = self.cache_dir / "text"

        keep: set[str] | None = None
        if index_path is not None:
            with open(index_path) as fh:
                entries = json.load(fh)["entries"]
            keep = {f"{e['session']}__{e['serial']}" for e in entries if e["split"] == split}

        self.clips: list[tuple[str, str]] = []  # (video shard stem, episode key)
        with os.scandir(self.video_dir) as it:
            for entry in it:
                if not entry.name.endswith(".safetensors"):
                    continue
                stem = entry.name[: -len(".safetensors")]
                session, serial, _ = _split_shard_name(stem)
                episode = f"{session}__{serial}"
                if keep is not None and episode not in keep:
                    continue
                # A clip is only usable if its caption was cached too.
                if (self.text_dir / f"{episode}.safetensors").exists():
                    self.clips.append((stem, episode))
        self.clips.sort()

        if not self.clips:
            raise RuntimeError(f"no clips found under {self.video_dir} for split={split}")

        null_path = self.cache_dir / "null.safetensors"
        self.null_context = load_file(str(null_path))["context"] if null_path.exists() else None

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int) -> dict:
        stem, episode = self.clips[i]
        latent = load_file(str(self.video_dir / f"{stem}.safetensors"))["latent"]
        context = load_file(str(self.text_dir / f"{episode}.safetensors"))["context"]
        return {"latent": latent, "context": context, "episode": episode}


def collate(batch: list[dict]) -> dict:
    return {
        "latent": [b["latent"] for b in batch],
        "context": [b["context"] for b in batch],
        "episode": [b["episode"] for b in batch],
    }


def seq_len_for(latent: torch.Tensor, patch_size=(1, 2, 2)) -> int:
    """Token count for one latent: T * (h/2) * (w/2) for TI2V-5B."""
    _, t, h, w = latent.shape
    return (t // patch_size[0]) * (h // patch_size[1]) * (w // patch_size[2])
