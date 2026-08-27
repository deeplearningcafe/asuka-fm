import io
import math
import struct
import logging
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import warnings

import torch
import json
from torch.utils.data import IterableDataset, DataLoader
from huggingface_hub import hf_hub_download
from torchvision.transforms import v2
from PIL import Image, ImageFile, PngImagePlugin
from datasets import load_dataset
import pyarrow
import pyarrow.dataset
import fsspec.spec
import fsspec.utils
from datasets.distributed import split_dataset_by_node
from tqdm import tqdm
from huggingface_hub.utils import _http

from src.models.text_encoders.tokenizer import BaseTokenizer


PngImagePlugin.MAX_TEXT_CHUNK = 64 * 1024 * 1024
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def _worker_init_fn(worker_id: int) -> None:
    """Resets HTTP session pools per worker to prevent socket collisions."""
    try:
        _http.reset_sessions()
    except Exception:
        pass

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

def bytes_to_tensor(
    raw_bytes: bytes,
    shape: list[int] | tuple[int, ...],
    dtype_str: str = "bf16",
) -> torch.Tensor:
    """Zero-copy bitcast deserialization of raw latent bytes."""
    dt = dtype_str.lower()
    
    # latent is read only but pytorch modifies it only strides and offsets so no problem
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given buffer is not writable.*",
            category=UserWarning,
        )
        if dt in ("bf16", "bfloat16"):
            tensor = torch.frombuffer(raw_bytes, dtype=torch.int16).view(
                torch.bfloat16
            )
        elif dt in ("fp16", "float16"):
            tensor = torch.frombuffer(raw_bytes, dtype=torch.float16)
        else:
            tensor = torch.frombuffer(raw_bytes, dtype=torch.float32)

    return tensor.view(*shape)


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
        is_latent: bool = False,
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
        self.is_latent = is_latent

        self.metadata = self._load_metadata(dataset_name)
        self.total_samples = int(self.metadata.get("total_samples", 0))
        self.num_shards = int(
            self.metadata.get("num_shards", len(self.metadata.get("shards", [])))
        )
        self.samples_per_shard = int(self.metadata.get("samples_per_shard", 10000))
        self.length_tiers = self.metadata.get("length_tiers", [77, 152, 227])
        self.bucket_info = self.metadata.get("bucket_info", [])
        self.crop_latent_size = resolution // vae_downsample_factor
        self.vae_downsample_factor = vae_downsample_factor

        if self.world_size > 1:
            self.num_samples = self.total_samples // self.world_size
        else:
            self.num_samples = self.total_samples

        storage_options = {
            "block_size": 4 * 1024 * 1024,
            "cache_type": "first",
        }
        cache_opts = pyarrow.CacheOptions(
            prefetch_limit=1 if low_ram else 2,
            range_size_limit=(32 << 20) if low_ram else (128 << 20),
        )
        scan_options = pyarrow.dataset.ParquetFragmentScanOptions(
            cache_options=cache_opts
        )
        if not low_ram:
            # 128 is using > 30gb ram
            fsspec.spec.AbstractBufferedFile.DEFAULT_BLOCK_SIZE = 128 * 1024 * 1024
            fsspec.utils.DEFAULT_BLOCK_SIZE = 128 * 1024 * 1024

            storage_options = {
                "block_size": 128 * 1024 * 1024,
                "cache_type": "readahead",
            }
            logging.info("Using high ram settings!")

        if dataset_path is not None:
            dataset_path = Path(dataset_path)
            if dataset_path.is_dir():
                parquet_files = sorted(
                    [
                        str(p)
                        for p in dataset_path.glob("data_shard_*.parquet")
                    ]
                )
                if not parquet_files:
                    parquet_files = sorted(
                        [str(p) for p in dataset_path.rglob("*.parquet")]
                    )
                if not parquet_files:
                    raise FileNotFoundError(
                        f"No parquet shards found in: {dataset_path}"
                    )
                logging.info(
                    f"Found {len(parquet_files)} local parquet shards."
                )
                self.hf_dataset = load_dataset(
                    "parquet",
                    data_files={"train": parquet_files},
                    split="train",
                    streaming=True,
                )
        else:
            # Explicitly match all 34 root data_shard_*.parquet files on Hub
            self.hf_dataset = load_dataset(
                dataset_name,
                data_files={"train": "data_shard_*.parquet"},
                split="train",
                streaming=True,
                storage_options=storage_options,
                    fragment_scan_options=scan_options,
            )

        detected_shards = getattr(
            self.hf_dataset,
            "num_shards",
            getattr(self.hf_dataset, "n_shards", "unknown"),
        )
        logging.info(
            f"Initialized HF Streaming Dataset with {detected_shards} shards."
        )

        # Split across DDP nodes/ranks ONCE at initialization
        if self.world_size > 1:
            self.hf_dataset = split_dataset_by_node(
                self.hf_dataset,
                rank=self.rank,
                world_size=self.world_size,
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

    def _process_latent_sample(
        self, sample: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ...]:
        """Processes precomputed latent sample with dynamic shift cropping."""
        raw_bytes = sample.get("latent")
        shape = sample.get("latent_shape", [32, 64, 64])
        dtype_str = sample.get("latent_dtype", "bf16")
        latent = bytes_to_tensor(raw_bytes, shape, dtype_str)

        _, h_lat, w_lat = latent.shape
        max_y = max(0, h_lat - self.crop_latent_size)
        max_x = max(0, w_lat - self.crop_latent_size)
        crop_y_lat = random.randint(0, max_y) if max_y > 0 else 0
        crop_x_lat = random.randint(0, max_x) if max_x > 0 else 0

        cropped_latent = latent[
            :,
            crop_y_lat : crop_y_lat + self.crop_latent_size,
            crop_x_lat : crop_x_lat + self.crop_latent_size,
        ].clone()

        pos_map = compute_aspect_coordinates(
            target_h=h_lat * self.vae_downsample_factor,
            target_w=w_lat * self.vae_downsample_factor,
            crop_y=crop_y_lat * self.vae_downsample_factor,
            crop_x=crop_x_lat * self.vae_downsample_factor,
            crop_size=self.resolution,
            patch_size=self.latent_patch,
        )

        tier_len = int(sample.get("tier", 227))
        prompt = sample.get("prompt") or sample.get("text", "")
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
        aes_tier = torch.tensor(
            float(sample.get("aesthetic_tier", -1.0)), dtype=torch.float32
        )

        return cropped_latent, tokens, mask, pos_map, tag_weight, aes_tier

    def _process_sample(
        self, sample: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.is_latent or "latent" in sample:
            return self._process_latent_sample(sample)

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
    
    def _extract_resized_image(
        self, sample: dict
    ) -> tuple[torch.Tensor, int, int]:
        """
        Resizes raw image to aspect ratio bucket preserving dimensions
        without cropping. Aligns to 8-pixel boundaries for VAE downsampling.
        """
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

        # Ensure spatial dimensions are divisible by VAE downsample factor
        down = self.patch_size * (self.latent_patch // self.patch_size)
        resized_w = (resized_w // down) * down
        resized_h = (resized_h // down) * down

        image = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
        return self.normalize(image), resized_h, resized_w


    def set_epoch(self, epoch: int) -> None:
        """Updates epoch state and refreshes HTTP sessions between epochs."""
        self.epoch = epoch
        try:
            from huggingface_hub.utils import _http

            _http.reset_sessions()
        except Exception:
            pass

        if hasattr(self.hf_dataset, "set_epoch"):
            self.hf_dataset.set_epoch(epoch)

    def _get_worker_stream(self):
        """
        Returns the HuggingFace streaming dataset. HF IterableDataset
        automatically detects PyTorch worker_info and splits its assigned
        shards across DataLoader workers.
        """
        return self.hf_dataset

    def iter_raw(self):
        """Yields raw samples with automatic recovery from closed sessions."""
        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                for sample in self._get_worker_stream():
                    yield sample
                break
            except RuntimeError as e:
                if "client has been closed" in str(e) or "closed" in str(e):
                    retries += 1
                    logging.warning(
                        f"HTTP client closed in worker stream. "
                        f"Resetting session (attempt {retries}/{max_retries})."
                    )
                    try:
                        from huggingface_hub.utils import _http

                        _http.reset_sessions()
                    except Exception:
                        pass
                else:
                    raise e

    def __iter__(self):
        for sample in self._get_worker_stream():
            try:
                yield self._process_sample(sample)
            except Exception:
                continue

class RAMCachedDataset(torch.utils.data.Dataset):
    """
    Dataset storing uncropped latents in POSIX shared memory and raw text
    metadata in Python lists. Executes latent shift-cropping and dynamic
    prompt tokenization (tag shuffling, tag dropout, CFG) on-the-fly.
    """

    def __init__(
        self,
        latents_flat: torch.Tensor,
        offsets: torch.Tensor,
        shapes: torch.Tensor,
        prompts: list[str],
        tiers: list[int],
        tag_weights: list[float],
        aes_tiers: list[int],
        tokenizer: BaseTokenizer,
        resolution: int = 256,
        patch_size: int = 2,
        vae_downsample_factor: int = 8,
        in_channels: int = 32,
        cfg_dropout_prob: float = 0.0,
        tag_dropout_prob: float = 0.0,
        shuffle_tags: bool = True,
    ):
        self.latents_flat = latents_flat
        self.offsets = offsets
        self.shapes = shapes
        self.prompts = prompts
        self.tiers = tiers
        self.tag_weights = tag_weights
        self.aes_tiers = aes_tiers
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.patch_size = patch_size
        self.vae_downsample_factor = vae_downsample_factor
        self.latent_patch = patch_size * vae_downsample_factor
        self.crop_latent_size = resolution // vae_downsample_factor
        self.in_channels = in_channels
        self.cfg_dropout_prob = cfg_dropout_prob
        self.tag_dropout_prob = tag_dropout_prob
        self.shuffle_tags = shuffle_tags
        self.num_samples = offsets.shape[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        offset = self.offsets[idx].item()
        h_lat = self.shapes[idx, 0].item()
        w_lat = self.shapes[idx, 1].item()
        numel = self.in_channels * h_lat * w_lat

        latent = self.latents_flat[offset : offset + numel].view(
            self.in_channels, h_lat, w_lat
        )

        # Dynamic random shift crop in latent space
        max_y = max(0, h_lat - self.crop_latent_size)
        max_x = max(0, w_lat - self.crop_latent_size)
        crop_y_lat = random.randint(0, max_y) if max_y > 0 else 0
        crop_x_lat = random.randint(0, max_x) if max_x > 0 else 0

        cropped_latent = latent[
            :,
            crop_y_lat : crop_y_lat + self.crop_latent_size,
            crop_x_lat : crop_x_lat + self.crop_latent_size,
        ].clone()

        # Dynamic continuous 2D RoPE position map for selected crop
        pos_map = compute_aspect_coordinates(
            target_h=h_lat * self.vae_downsample_factor,
            target_w=w_lat * self.vae_downsample_factor,
            crop_y=crop_y_lat * self.vae_downsample_factor,
            crop_x=crop_x_lat * self.vae_downsample_factor,
            crop_size=self.resolution,
            patch_size=self.latent_patch,
        )

        # Dynamic text tokenization on-the-fly
        prompt = self.prompts[idx]
        tier_len = self.tiers[idx]
        tokens, mask = self.tokenizer.encode(
            prompt,
            max_len=tier_len,
            cfg_dropout_prob=self.cfg_dropout_prob,
            tag_dropout_prob=self.tag_dropout_prob,
            shuffle_tags=self.shuffle_tags,
        )
        is_uncond = mask.sum().item() <= 1
        raw_tag_weight = self.tag_weights[idx]
        tag_weight = torch.tensor(
            1.0 if is_uncond else raw_tag_weight, dtype=torch.float32
        )
        aes_tier = torch.tensor(
            float(self.aes_tiers[idx]), dtype=torch.float32
        )

        return (
            cropped_latent,
            tokens,
            mask,
            pos_map,
            tag_weight,
            aes_tier,
        )


class PrecomputeExtractDataset(IterableDataset):
    """
    Worker-parallel iterable dataset for concurrent AVIF decoding and resizing.
    HuggingFace IterableDataset automatically shards streams across workers.
    """

    def __init__(self, streaming_dataset: "StreamingImageDataset"):
        super().__init__()
        self.streaming_dataset = streaming_dataset

    def __iter__(self):
        # Iterating hf_dataset within a DataLoader worker automatically
        # slices the assigned shard subset via get_worker_info().
        for raw_sample in self.streaming_dataset.hf_dataset:
            try:
                img_tensor, res_h, res_w = (
                    self.streaming_dataset._extract_resized_image(raw_sample)
                )
            except Exception as e:
                logging.warning(
                    f"Worker failed to decode image: {e}. Skipping."
                )
                continue

            prompt = raw_sample.get("prompt") or raw_sample.get("text", "")
            tier = int(raw_sample.get("tier", 227))
            tag_weight = float(raw_sample.get("tag_weight", 1.0))
            aes_tier = int(raw_sample.get("aesthetic_tier", -1))

            yield img_tensor, res_h, res_w, prompt, tier, tag_weight, aes_tier


@torch.no_grad()
def precompute_latents_to_ram(
    dataset: "StreamingImageDataset",
    vae: torch.nn.Module,
    batch_size: int = 64,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    device: torch.device = torch.device("cuda"),
    autocast_dtype: torch.dtype = torch.bfloat16,
    vae_mean: float = 0.0,
    vae_std: float = 1.0,
    in_channels: int = 32,
    rank: int = 0,
    world_size: int = 1,
) -> RAMCachedDataset:
    """
    High-throughput multi-worker precomputation of latents into POSIX shared
    memory using multi-worker DataLoader prefetching and resolution bucketing.
    """
    max_samples = getattr(dataset, "num_samples", 0)
    if max_samples <= 0:
        max_samples = dataset.total_samples // max(1, world_size)
    if max_samples <= 0:
        max_samples = 340000

    offsets_buf = torch.empty(
        (max_samples,), dtype=torch.long, device="cpu"
    ).share_memory_()

    shapes_buf = torch.empty(
        (max_samples, 2), dtype=torch.int32, device="cpu"
    ).share_memory_()

    base_elements = in_channels * (dataset.resolution // 8) ** 2
    est_elements = int(max_samples * base_elements * 1.5)
    latents_flat_buf = torch.empty(
        (est_elements,), dtype=autocast_dtype, device="cpu"
    ).share_memory_()

    prompts_list: list[str] = []
    tiers_list: list[int] = []
    tag_weights_list: list[float] = []
    aes_tiers_list: list[int] = []

    vae_mean_t = torch.tensor(vae_mean, device=device, dtype=autocast_dtype)
    vae_std_t = torch.tensor(vae_std, device=device, dtype=autocast_dtype)

    if rank == 0:
        logging.info(
            f"Precomputing latents with {num_workers} workers "
            f"(Batch Size: {batch_size}, Prefetch: {prefetch_factor})..."
        )
        pbar = tqdm(total=max_samples, desc=f"RAM Latent Encoding (Rank {rank})")
    else:
        pbar = None

    extract_dataset = PrecomputeExtractDataset(dataset)
    prefetch_loader = DataLoader(
        extract_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
        worker_init_fn=_worker_init_fn,
    )

    bucket_buffers: dict[tuple[int, int], list] = defaultdict(list)
    sample_idx = 0
    current_flat_offset = 0

    def _flush_bucket(items: list) -> None:
        nonlocal current_flat_offset, sample_idx, latents_flat_buf
        nonlocal offsets_buf, shapes_buf
        if not items:
            return

        b_size = len(items)
        imgs = torch.stack([it[0] for it in items], dim=0)
        imgs_gpu = imgs.to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda", dtype=autocast_dtype, enabled=True
        ):
            enc = vae.encode(imgs_gpu)
            dist = getattr(enc, "latent_dist", enc)
            lats = dist.sample() if hasattr(dist, "sample") else dist
            lats = (lats - vae_mean_t) / vae_std_t

        lats_cpu = lats.to(dtype=autocast_dtype, device="cpu")
        _, _, h_lat, w_lat = lats_cpu.shape
        sample_numel = in_channels * h_lat * w_lat
        total_elements = b_size * sample_numel

        # Dynamic buffer expansion
        if current_flat_offset + total_elements > latents_flat_buf.numel():
            new_size = int((current_flat_offset + total_elements) * 1.3)
            new_buf = torch.empty(
                (new_size,), dtype=autocast_dtype, device="cpu"
            ).share_memory_()
            new_buf[:current_flat_offset] = latents_flat_buf[
                :current_flat_offset
            ]
            latents_flat_buf = new_buf

        if sample_idx + b_size > offsets_buf.shape[0]:
            new_cap = int((sample_idx + b_size) * 1.3)
            new_offsets = torch.empty(
                (new_cap,), dtype=torch.long, device="cpu"
            ).share_memory_()
            new_offsets[:sample_idx] = offsets_buf[:sample_idx]
            offsets_buf = new_offsets

            new_shapes = torch.empty(
                (new_cap, 2), dtype=torch.int32, device="cpu"
            ).share_memory_()
            new_shapes[:sample_idx] = shapes_buf[:sample_idx]
            shapes_buf = new_shapes

        for i in range(b_size):
            flat_lat = lats_cpu[i].flatten()
            latents_flat_buf[
                current_flat_offset : current_flat_offset + sample_numel
            ] = flat_lat
            offsets_buf[sample_idx] = current_flat_offset
            shapes_buf[sample_idx, 0] = h_lat
            shapes_buf[sample_idx, 1] = w_lat

            _, prompt, tier, tag_weight, aes_tier = items[i]
            prompts_list.append(prompt)
            tiers_list.append(tier)
            tag_weights_list.append(tag_weight)
            aes_tiers_list.append(aes_tier)

            current_flat_offset += sample_numel
            sample_idx += 1

        if pbar:
            pbar.update(b_size)
        items.clear()

    for item in prefetch_loader:
        if sample_idx >= max_samples:
            break

        img_tensor, res_h, res_w, prompt, tier, tag_weight, aes_tier = item
        res_key = (int(res_h), int(res_w))
        bucket_buffers[res_key].append(
            (img_tensor, prompt, tier, tag_weight, aes_tier)
        )

        if len(bucket_buffers[res_key]) >= batch_size:
            _flush_bucket(bucket_buffers[res_key])

    # Flush all remaining partial buckets
    for res_key, items in list(bucket_buffers.items()):
        if items:
            _flush_bucket(items)

    if pbar:
        pbar.close()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    torch.cuda.empty_cache() 
    return RAMCachedDataset(
        latents_flat=latents_flat_buf[:current_flat_offset],
        offsets=offsets_buf[:sample_idx],
        shapes=shapes_buf[:sample_idx],
        prompts=prompts_list,
        tiers=tiers_list,
        tag_weights=tag_weights_list,
        aes_tiers=aes_tiers_list,
        tokenizer=dataset.tokenizer,
        resolution=dataset.resolution,
        patch_size=dataset.patch_size,
        vae_downsample_factor=dataset.latent_patch // dataset.patch_size,
        in_channels=in_channels,
        cfg_dropout_prob=dataset.cfg_dropout_prob,
        tag_dropout_prob=dataset.tag_dropout_prob,
        shuffle_tags=dataset.shuffle_tags,
    )

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
