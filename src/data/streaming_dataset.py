import io
import math
import struct
import logging
import random
import argparse
from pathlib import Path
from collections import Counter

import torch
import json
from torch.utils.data import IterableDataset
from huggingface_hub import hf_hub_download
from torchvision.transforms import v2
from PIL import Image, ImageFile, PngImagePlugin
from datasets import load_dataset
import fsspec.spec
import fsspec.utils
from datasets.distributed import split_dataset_by_node
from tqdm import tqdm

from src.models.text_encoders.tokenizer import BaseTokenizer


PngImagePlugin.MAX_TEXT_CHUNK = 64 * 1024 * 1024
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def sniff_image_header(
    data: bytes,
) -> tuple[str, tuple[int, int] | None]:
    """
    Fast magic-byte format and dimension parser without raster decompression.
    """
    if len(data) < 16:
        return "CORRUPT", None

    # PNG Header: IHDR chunk width/height located at offsets 16:24
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(data) >= 24:
            w, h = struct.unpack(">II", data[16:24])
            return "PNG", (w, h)
        return "PNG", None

    # GIF Header: Logical Screen Descriptor dimensions at offsets 6:10
    if data.startswith((b"GIF87a", b"GIF89a")):
        if len(data) >= 10:
            w, h = struct.unpack("<HH", data[6:10])
            return "GIF", (w, h)
        return "GIF", None

    # JPEG Header
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG", None

    # WebP Header
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP", None

    # AVIF / HEIF Header: ISO Base Media File Format (ftyp box)
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"avif", b"avis", b"mif1"):
            return "AVIF", None
        return "HEIF", None

    return "UNKNOWN", None


def compute_aspect_coordinates(
    target_h: int,
    target_w: int,
    crop_y: int,
    crop_x: int,
    crop_size: int,
    patch_size: int,
) -> torch.Tensor:
    """Computes continuous patch-unit 2D RoPE position map for cropped patch."""
    num_patches_h = crop_size // patch_size
    num_patches_w = crop_size // patch_size

    crop_y_patch = crop_y / float(patch_size)
    crop_x_patch = crop_x / float(patch_size)

    y_pos = torch.arange(num_patches_h, dtype=torch.float32) + crop_y_patch
    x_pos = torch.arange(num_patches_w, dtype=torch.float32) + crop_x_patch

    grid_y, grid_x = torch.meshgrid(y_pos, x_pos, indexing="ij")
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
        low_ram: bool = False,
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

        storage_options = {
            "block_size": 4 * 1024 * 1024,
            "cache_type": "first",
        }
        if not low_ram:
            # 128 is using > 30gb ram
            fsspec.spec.AbstractBufferedFile.DEFAULT_BLOCK_SIZE = 32 * 1024 * 1024
            fsspec.utils.DEFAULT_BLOCK_SIZE = 32 * 1024 * 1024

            storage_options = {
                "block_size": 32 * 1024 * 1024,
                "cache_type": "readahead",
            }
            logging.info("Using high ram settings!")

        if dataset_path is not None:
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

        if self.world_size > 1:
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
        # sample until tier len
        tokens, mask = self.tokenizer.encode(
            prompt,
            max_len=tier_len,
            cfg_dropout_prob=self.cfg_dropout_prob,
            tag_dropout_prob=self.tag_dropout_prob,
            shuffle_tags=self.shuffle_tags,
        )
        is_uncond = mask.sum().item() <= 1
        raw_tag_weight = float(sample.get("tag_weight", 1.0))
        tag_weight = torch.tensor(
            1.0 if is_uncond else raw_tag_weight, dtype=torch.float32
        )
        # TODO: use int?
        aes_tier = torch.tensor(
            float(sample.get("aesthetic_tier", -1.0)), dtype=torch.float32
        )

        return img_tensor, tokens, mask, pos_map, tag_weight, aes_tier

    def _get_worker_stream(self):
        """Splits streaming dataset across DataLoader worker processes."""
        worker_info = torch.utils.data.get_worker_info()
        dataset = self.hf_dataset
        if worker_info is not None and worker_info.num_workers > 1:
            dataset = split_dataset_by_node(
                dataset,
                rank=worker_info.id,
                world_size=worker_info.num_workers,
            )
        return dataset

    def iter_raw(self):
        """Yields raw sample dictionaries without image/text processing."""
        for sample in self._get_worker_stream():
            yield sample

    def __iter__(self):
        for sample in self._get_worker_stream():
            try:
                yield self._process_sample(sample)
            except Exception:
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect streaming dataset for corrupt or invalid images."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="aipracticecafe/curated-danbooru-2026",
        help="HuggingFace dataset repository or local metadata directory.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Optional local path containing raw parquet shards.",
    )
    parser.add_argument(
        "--inspect_indices",
        type=str,
        default=None,
        help="Comma-separated sample indices to inspect (e.g. '92743,429,855').",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50000,
        help="Maximum samples to scan (set -1 for complete dataset).",
    )
    parser.add_argument(
        "--test_full_decode",
        action="store_true",
        default=True,
        help="Perform complete RGB raster decoding to check corruption.",
    )
    args = parser.parse_args()

    print(f"Initializing stream inspection for: {args.dataset_name}")
    dataset = StreamingImageDataset(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        resolution=256,
        low_ram=True,
    )

    if args.inspect_indices:
        target_indices = set(int(i.strip()) for i in args.inspect_indices.split(","))
        max_target = max(target_indices)
        print(f"Directly inspecting {len(target_indices)} target indices...")

        for idx, sample in enumerate(dataset.iter_raw(), start=1):
            if idx in target_indices:
                sample_id = sample.get("booru_id")
                img_data = sample.get("image") or sample.get("bytes")
                data_len = len(img_data) if isinstance(img_data, bytes) else 0

                fmt, dims = "UNKNOWN", None
                if isinstance(img_data, bytes):
                    fmt, dims = sniff_image_header(img_data)

                print("\n" + "=" * 50)
                print(f"INDEX: {idx} | SAMPLE ID: {sample_id}")
                print(f"Format: {fmt} | Byte Size: {data_len / 1024:.1f} KB")
                if dims:
                    print(
                        f"Dimensions: {dims} | Aspect Ratio: {dims[0] / dims[1]:.2f}:1"
                    )
                for k, v in sample.items():
                    if k not in ("image", "bytes"):
                        val_str = str(v)
                        if len(val_str) > 100:
                            val_str = val_str[:97] + "..."
                        print(f"  {k}: {val_str}")

            if idx >= max_target:
                break
        exit(0)

    format_counts: Counter = Counter()
    error_counts: Counter = Counter()
    corrupt_samples = []
    aspect_outliers = []
    total_scanned = 0

    pbar = tqdm(
        total=args.max_samples if args.max_samples > 0 else None,
        desc="Scanning Shards",
        dynamic_ncols=True,
    )

    for sample in dataset.iter_raw():
        total_scanned += 1
        img_data = sample.get("image") or sample.get("bytes")
        sample_id = sample.get("id") or sample.get("booru_id") or f"idx_{total_scanned}"

        if not isinstance(img_data, bytes):
            if isinstance(img_data, Image.Image):
                format_counts[img_data.format or "PIL_IMAGE"] += 1
                pbar.update(1)
                continue
            error_counts["INVALID_TYPE"] += 1
            corrupt_samples.append((sample_id, f"Type: {type(img_data)}"))
            pbar.update(1)
            continue

        # 1. Fast byte-level format and dimension sniffing (< 5 microseconds)
        fmt, fast_dims = sniff_image_header(img_data)
        format_counts[fmt] += 1

        # 2. Complete raster decode verification
        if args.test_full_decode:
            try:
                with Image.open(io.BytesIO(img_data)) as im:
                    w, h = im.size
                    ar = max(w, h) / max(1, min(w, h))
                    if ar > 3.5:
                        aspect_outliers.append((sample_id, fmt, (w, h), ar))
                    # Trigger full raster decompression
                    im.convert("RGB")
            except Exception as e:
                err_type = type(e).__name__
                error_counts[err_type] += 1
                corrupt_samples.append((sample_id, f"{err_type}: {str(e)}"))

        pbar.update(1)
        if 0 < args.max_samples <= total_scanned:
            break

    pbar.close()

    print("\n" + "=" * 60)
    print("DATASET DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Total Samples Scanned : {total_scanned}")
    print(f"Format Distribution   : {dict(format_counts)}")
    print(f"Total Errors Detected : {sum(error_counts.values())}")
    if error_counts:
        print(f"Error Breakdown       : {dict(error_counts)}")
    if corrupt_samples:
        print("\nFirst 10 Corrupted Samples:")
        for s_id, err in corrupt_samples[:10]:
            print(f"  [ID: {s_id}] -> {err}")
    if aspect_outliers:
        print(f"\nAspect Ratio Outliers (>3.5:1) Count: {len(aspect_outliers)}")
        for s_id, s_fmt, s_dims, s_ar in aspect_outliers[:5]:
            print(
                f"  [ID: {s_id}] Format: {s_fmt}, Dims: {s_dims}, Ratio: {s_ar:.2f}:1"
            )
    print("=" * 60)
