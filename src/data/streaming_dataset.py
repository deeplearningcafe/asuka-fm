import io
import math
import random
import torch
import json
from pathlib import Path
from torch.utils.data import IterableDataset
from huggingface_hub import hf_hub_download
from torchvision.transforms import v2
from PIL import Image
from datasets import load_dataset
import fsspec.spec
import fsspec.utils
from datasets.distributed import split_dataset_by_node
from src.models.text_encoders.tokenizer import BaseTokenizer


def compute_aspect_coordinates(
    target_h: int,
    target_w: int,
    crop_y: int,
    crop_x: int,
    crop_size: int,
    patch_size: int,
) -> torch.Tensor:
    """Computes continuous 2D RoPE position map for cropped patch."""
    r_h = math.sqrt(target_h / target_w)
    r_w = math.sqrt(target_w / target_h)

    num_patches = crop_size // patch_size
    y_centers = (
        torch.arange(num_patches, dtype=torch.float32) + 0.5
    ) * patch_size + crop_y
    x_centers = (
        torch.arange(num_patches, dtype=torch.float32) + 0.5
    ) * patch_size + crop_x

    # Normalize to [-r_h, r_h] and [-r_w, r_w]
    y_norm = (y_centers / target_h) * (2.0 * r_h) - r_h
    x_norm = (x_centers / target_w) * (2.0 * r_w) - r_w

    grid_y, grid_x = torch.meshgrid(y_norm, x_norm, indexing="ij")
    return torch.stack((grid_y, grid_x), dim=-1).flatten(0, 1)


class StreamingImageDataset(IterableDataset):
    """HuggingFace Streaming Dataset with Shifted Square Crop."""

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str = None,
        resolution: int = 512,
        patch_size: int = 2,
        vae_downsample_factor: int = 8,
        max_seq_len: int = 256,
        tokenizer: BaseTokenizer | None = None,
        cfg_dropout_prob: float = 0.0,
        tag_dropout_prob: float = 0.0,
        shuffle_tags: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.resolution = resolution
        self.patch_size = patch_size
        self.latent_patch = patch_size * vae_downsample_factor
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.cfg_dropout_prob = cfg_dropout_prob
        self.tag_dropout_prob = tag_dropout_prob
        self.shuffle_tags = shuffle_tags
        self.rank = rank
        self.world_size = world_size

        self.metadata = self._load_metadata(dataset_name)
        self.total_samples = int(self.metadata.get("total_samples", 0))
        self.num_shards = int(
            self.metadata.get("num_shards", len(self.metadata.get("shards", [])))
        )
        self.samples_per_shard = int(self.metadata.get("samples_per_shard", 10000))
        self.length_tiers = self.metadata.get("length_tiers", [77, 152, 227])
        self.bucket_info = self.metadata.get("bucket_info", [])

        if self.world_size > 1:
            self.num_samples = self.total_samples // self.world_size
        else:
            self.num_samples = self.total_samples

        fsspec.spec.AbstractBufferedFile.DEFAULT_BLOCK_SIZE = 128 * 1024 * 1024
        fsspec.utils.DEFAULT_BLOCK_SIZE = 128 * 1024 * 1024

        storage_options = {
            "block_size": 128 * 1024 * 1024,
            "cache_type": "readahead",
        }

        dataset_path = Path(dataset_path)
        if dataset_path.is_dir():
            print(f"Reading data from {dataset_path}")
            parquet_files = sorted([str(p) for p in dataset_path.glob("*.parquet")])
            if not parquet_files:
                raise FileNotFoundError(
                    f"No parquet files found in directory: {dataset_name}"
                )
            self.hf_dataset = load_dataset(
                "parquet",
                data_files={"train": parquet_files},
                split="train",
                streaming=True,
            )
        else:
            self.hf_dataset = load_dataset(
                dataset_name,
                split="train",
                streaming=True,
                storage_options=storage_options,
            )

        if world_size > 1:
            self.hf_dataset = split_dataset_by_node(
                self.hf_dataset, rank=rank, world_size=world_size
            )

        self.normalize = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                    inplace=True,
                ),
            ]
        )

    def _load_metadata(self, dataset_name: str) -> dict:
        """Loads metadata.json from local path or downloads from HF Hub."""
        local_path = Path(dataset_name) / "metadata.json"
        if local_path.exists():
            with open(local_path, "r", encoding="utf-8") as f:
                return json.load(f)

        try:
            downloaded = hf_hub_download(
                repo_id=dataset_name,
                filename="metadata.json",
                repo_type="dataset",
            )
            with open(downloaded, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            warnings_msg = f"Could not load metadata.json for {dataset_name}: {e}"
            print(f"[Warning] {warnings_msg}")
            return {}

    def _process_sample(
        self, sample: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        img_data = sample.get("image") or sample.get("bytes")
        if isinstance(img_data, bytes):
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
        elif isinstance(img_data, Image.Image):
            image = img_data.convert("RGB")
        else:
            raise ValueError(f"Unsupported image format: {type(img_data)}")

        orig_w, orig_h = image.size
        scale = self.resolution / min(orig_h, orig_w)
        resized_w = max(self.resolution, int(round(orig_w * scale)))
        resized_h = max(self.resolution, int(round(orig_h * scale)))

        image = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)

        crop_y = random.randint(0, resized_h - self.resolution)
        crop_x = random.randint(0, resized_w - self.resolution)

        cropped = image.crop(
            (
                crop_x,
                crop_y,
                crop_x + self.resolution,
                crop_y + self.resolution,
            )
        )
        img_tensor = self.normalize(cropped)

        pos_map = compute_aspect_coordinates(
            target_h=resized_h,
            target_w=resized_w,
            crop_y=crop_y,
            crop_x=crop_x,
            crop_size=self.resolution,
            patch_size=self.latent_patch,
        )
        tier_len = int(sample.get("tier", 227))
        prompt = sample.get("prompt") or sample.get("text", "")
        # TODO: SHUFFLE TAGS
        # sample until tier len
        tokens, mask = self.tokenizer.encode(
            prompt,
            max_len=tier_len,
            cfg_dropout_prob=self.cfg_dropout_prob,
            tag_dropout_prob=self.tag_dropout_prob,
            shuffle_tags=self.shuffle_tags,
        )
        tag_weight = torch.tensor(
            float(sample.get("tag_weight", 1.0)), dtype=torch.float32
        )
        # TODO: use int?
        aes_tier = torch.tensor(
            float(sample.get("aesthetic_tier", -1.0)), dtype=torch.float32
        )

        return img_tensor, tokens, mask, pos_map, tag_weight, aes_tier

    def iter_raw(self):
        """Yields raw sample dictionaries without image/text processing."""
        for sample in self.hf_dataset:
            yield sample

    def __iter__(self):
        for sample in self.hf_dataset:
            try:
                yield self._process_sample(sample)
            except Exception:
                continue
