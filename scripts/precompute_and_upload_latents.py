"""Module for precomputing VAE latents using StreamingImageDataset."""

import argparse
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import time

from diffusers import AutoencoderKL
from huggingface_hub import HfApi, hf_hub_download
from omegaconf import DictConfig, OmegaConf
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from src.data.streaming_dataset import (
    StreamingImageDataset,
    _worker_init_fn,
)
from src.utils.logging_utils import Logger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_parquet_schema() -> pa.Schema:
    """Defines PyArrow schema for precomputed latent shards."""
    return pa.schema(
        [
            ("booru_id", pa.string()),
            ("latent", pa.binary()),
            ("latent_shape", pa.list_(pa.int32())),
            ("latent_dtype", pa.string()),
            ("prompt", pa.string()),
            ("bucket_idx", pa.int32()),
            ("target_width", pa.int32()),
            ("target_height", pa.int32()),
            ("original_width", pa.int32()),
            ("original_height", pa.int32()),
            ("aspect_ratio", pa.float32()),
            ("tier", pa.int32()),
            ("aesthetic_tier", pa.int32()),
            ("tag_weight", pa.float32()),
        ]
    )


def tensor_to_bytes(
    tensor: torch.Tensor, dtype_str: str = "fp32"
) -> bytes:
    """Fast conversion of a single tensor to raw bytes via NumPy."""
    t_cpu = tensor.detach().contiguous().cpu()
    dt = dtype_str.lower()
    if dt in ("bf16", "bfloat16"):
        return t_cpu.to(torch.bfloat16).view(torch.int16).numpy().tobytes()
    elif dt in ("fp16", "float16"):
        return t_cpu.to(torch.float16).numpy().tobytes()
    return t_cpu.to(torch.float32).numpy().tobytes()


def batch_tensors_to_bytes(
    tensors: torch.Tensor, dtype_str: str = "fp32"
) -> List[bytes]:
    """Converts a batch of latents to contiguous raw bytes in C."""
    t_cpu = tensors.detach().contiguous().cpu()
    dt = dtype_str.lower()
    if dt in ("bf16", "bfloat16"):
        arr = t_cpu.to(torch.bfloat16).view(torch.int16).numpy()
    elif dt in ("fp16", "float16"):
        arr = t_cpu.to(torch.float16).numpy()
    else:
        arr = t_cpu.to(torch.float32).numpy()
    return [arr[i].tobytes() for i in range(arr.shape[0])]

def load_existing_shards(
    dst_repo_id: str, hf_token: Optional[str] = None
) -> Tuple[set, List[Dict[str, Any]], int]:
    """Loads completed sample IDs and metadata, deleting shards immediately."""
    api = HfApi(token=hf_token)
    try:
        repo_files = api.list_repo_files(
            repo_id=dst_repo_id, repo_type="dataset"
        )
    except Exception:
        return set(), [], 0

    shard_files = sorted(
        [
            f
            for f in repo_files
            if f.startswith("data_shard_") and f.endswith(".parquet")
        ]
    )
    if not shard_files:
        return set(), [], 0

    existing_ids = set()
    shard_meta = []
    logging.info(
        f"Found {len(shard_files)} existing shards in {dst_repo_id}. "
        "Fetching sample IDs with single-file disk staging..."
    )

    with tempfile.TemporaryDirectory(prefix="shard_inspect_") as tmp_dir:
        for sf in shard_files:
            local_path = None
            try:
                local_path = hf_hub_download(
                    repo_id=dst_repo_id,
                    filename=sf,
                    repo_type="dataset",
                    token=hf_token,
                    local_dir=tmp_dir,
                )
                tbl = pq.read_table(local_path, columns=["booru_id"])
                ids = tbl["booru_id"].to_pylist()
                existing_ids.update(ids)
                shard_meta.append(
                    {
                        "shard_file": sf,
                        "sample_count": len(ids),
                    }
                )
                logging.info(
                    f"Parsed {len(ids)} IDs from {sf}. "
                    "Releasing disk space."
                )
            finally:
                if local_path and os.path.exists(local_path):
                    os.remove(local_path)

    logging.info(
        f"Resuming: {len(existing_ids)} samples already uploaded "
        f"across {len(shard_meta)} shards."
    )
    next_shard_idx = len(shard_meta)
    return existing_ids, shard_meta, next_shard_idx

def create_empty_columns() -> Dict[str, List[Any]]:
    """Creates columnar structure for fast PyArrow table instantiation."""
    return {
        "booru_id": [],
        "latent": [],
        "latent_shape": [],
        "latent_dtype": [],
        "prompt": [],
        "bucket_idx": [],
        "target_width": [],
        "target_height": [],
        "original_width": [],
        "original_height": [],
        "aspect_ratio": [],
        "tier": [],
        "aesthetic_tier": [],
        "tag_weight": [],
    }


class LatentExtractDataset(IterableDataset):
    """Worker-parallel dataset with ID-skip resume and session recovery."""

    def __init__(
        self,
        streaming_dataset: StreamingImageDataset,
        existing_ids: Optional[set] = None,
    ):
        super().__init__()
        self.streaming_dataset = streaming_dataset
        self.existing_ids = existing_ids or set()

    def __iter__(self):
        max_retries = 5
        retries = 0
        while retries < max_retries:
            try:
                for raw_sample in self.streaming_dataset.hf_dataset:
                    booru_id = str(raw_sample.get("booru_id", ""))
                    if booru_id in self.existing_ids:
                        continue

                    try:
                        img_tensor, res_h, res_w = (
                            self.streaming_dataset._extract_resized_image(
                                raw_sample
                            )
                        )
                    except Exception as e:
                        logging.warning(
                            f"Worker failed to decode image: {e}"
                        )
                        continue

                    orig_w = int(raw_sample.get("original_width", res_w))
                    orig_h = int(raw_sample.get("original_height", res_h))
                    ar = float(
                        raw_sample.get(
                            "aspect_ratio", orig_w / max(1, orig_h)
                        )
                    )

                    meta = {
                        "booru_id": booru_id,
                        "prompt": str(
                            raw_sample.get("prompt")
                            or raw_sample.get("text", "")
                        ),
                        "bucket_idx": int(raw_sample.get("bucket_idx", -1)),
                        "target_width": int(res_w),
                        "target_height": int(res_h),
                        "original_width": orig_w,
                        "original_height": orig_h,
                        "aspect_ratio": ar,
                        "tier": int(raw_sample.get("tier", 227)),
                        "aesthetic_tier": int(
                            raw_sample.get("aesthetic_tier", -1)
                        ),
                        "tag_weight": float(
                            raw_sample.get("tag_weight", 1.0)
                        ),
                    }
                    yield img_tensor, res_h, res_w, meta
                break
            except RuntimeError as e:
                if "client has been closed" in str(e) or "closed" in str(e):
                    retries += 1
                    logging.warning(
                        f"HTTP client closed in worker stream. Resetting "
                        f"session (attempt {retries}/{max_retries})..."
                    )
                    try:
                        from huggingface_hub.utils import _http

                        _http.reset_sessions()
                    except Exception:
                        pass
                else:
                    raise e


class ShardUploader:
    """Handles Parquet serialization and async upload with local cleanup."""

    def __init__(
        self,
        dst_repo_id: str,
        hf_token: Optional[str] = None,
        compression: str = "SNAPPY",
        max_workers: int = 2,
    ):
        self.dst_repo_id = dst_repo_id
        self.compression = compression
        self.api = HfApi(token=hf_token)
        self.schema = get_parquet_schema()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="latent_shards_"))
        self.upload_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures: List[Future] = []

        self.api.create_repo(
            repo_id=self.dst_repo_id,
            repo_type="dataset",
            exist_ok=True,
        )

    def _write_and_upload(
        self, table: pa.Table, local_path: Path, filename: str
    ) -> None:
        try:
            pq.write_table(table, local_path, compression=self.compression)
            self.api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=filename,
                repo_id=self.dst_repo_id,
                repo_type="dataset",
            )
            logging.info(f"Successfully uploaded {filename} to HF.")
        finally:
            if local_path.exists():
                os.remove(local_path)
                logging.info(f"Removed local staging file {filename}.")

    def stage_and_upload(
        self, columns: Dict[str, List[Any]], shard_idx: int
    ) -> Dict[str, Any]:
        shard_filename = f"data_shard_{shard_idx:05d}.parquet"
        local_path = self.temp_dir / shard_filename
        sample_count = len(columns["booru_id"])

        table = pa.Table.from_pydict(columns, schema=self.schema)
        fut = self.upload_executor.submit(
            self._write_and_upload, table, local_path, shard_filename
        )
        self.futures.append(fut)

        return {
            "shard_file": shard_filename,
            "sample_count": sample_count,
        }

    def finalize(self, metadata: Dict[str, Any]) -> None:
        self.upload_executor.shutdown(wait=True)
        for fut in self.futures:
            fut.result()

        meta_path = self.temp_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        self.api.upload_file(
            path_or_fileobj=str(meta_path),
            path_in_repo="metadata.json",
            repo_id=self.dst_repo_id,
            repo_type="dataset",
        )
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        logging.info("Metadata uploaded and staging directory cleaned.")


def setup_cuda_environment(
    device_str: str,
) -> Tuple[torch.device, torch.dtype]:
    """Applies trainer-level Tensor Core and TF32 optimizations."""
    device_obj = torch.device(device_str)
    if device_obj.type != "cuda":
        return device_obj, torch.float32

    capability = torch.cuda.get_device_capability(device_obj)
    if capability[0] >= 8:
        autocast_dtype = torch.bfloat16
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        logging.info("Enabled TF32 and Tensor Core optimizations (SM >= 8.0).")
    elif capability[0] >= 7:
        autocast_dtype = torch.float16
        torch.set_float32_matmul_precision("high")
        logging.info("Enabled FP16 Tensor Core optimizations (SM 7.x).")
    else:
        autocast_dtype = torch.float32
        logging.info("Running standard FP32 precision.")

    return device_obj, autocast_dtype


@torch.no_grad()
def precompute_and_upload(
    cfg: DictConfig,
    src_dataset: Optional[str] = None,
    dst_repo_id: Optional[str] = None,
    vae_pretrained: Optional[str] = None,
    batch_size: Optional[int] = None,
    dtype_str: Optional[str] = None,
    samples_per_shard: int = 10000,
    num_workers: Optional[int] = None,
    device: str = "cuda",
    hf_token: Optional[str] = None,
) -> None:
    """Executes high-throughput VAE precomputation and shard streaming."""
    device_obj, default_autocast_dtype = setup_cuda_environment(device)

    src_ds = src_dataset or cfg.data.get(
        "streaming_dataset_name", "aipracticecafe/curated-danbooru-2026"
    )
    dst_repo = (
        dst_repo_id
        or cfg.logging.get("hf_repo", None)
        or f"{src_ds}-latents"
    )
    vae_path = (
        vae_pretrained
        or cfg.models.get("hf_vae", None)
        or cfg.models.get("vae_pretrained", "black-forest-labs/FLUX.1-dev")
    )
    res = cfg.data.get("resolution", 512)
    bs = batch_size or cfg.models.get(
        "vae_batch_size", cfg.train.get("batch_size", 32)
    )
    workers = (
        num_workers
        if num_workers is not None
        else cfg.data.get(
            "precompute_num_workers", cfg.train.get("num_workers", 8)
        )
    )
    prefetch = cfg.data.get(
        "precompute_prefetch_factor", cfg.train.get("prefetch_factor", 4)
    )
    logging.info(
        f"Precomputing latents with {workers} workers "
        f"(Batch Size: {bs}, Prefetch: {prefetch})..."
    )
    dt_str = dtype_str or cfg.train.get("dtype", "bf16")
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dt_str.lower(), default_autocast_dtype)

    vae_mean = getattr(cfg.models, "vae_mean", 0.0)
    vae_std = getattr(cfg.models, "vae_std", 1.0)
    in_channels = cfg.models.get("in_channels", 32)
    patch_size = cfg.data.get("patch_size", 2)
    vae_downsample = cfg.data.get("vae_downsample_factor", 8)

    logging.info(f"Loading VAE from {vae_path} on {device_obj} ({torch_dtype})")
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        torch_dtype=torch_dtype,
    ).to(device_obj)
    vae.eval()
    vae.requires_grad_(False)

    vae_mean_t = torch.tensor(vae_mean, device=device_obj, dtype=torch_dtype)
    vae_std_t = torch.tensor(vae_std, device=device_obj, dtype=torch_dtype)

    streaming_dataset = StreamingImageDataset(
        dataset_name=src_ds,
        dataset_path=cfg.data.get("dataset_path", None),
        resolution=res,
        patch_size=patch_size,
        vae_downsample_factor=vae_downsample,
        rank=0,
        world_size=1,
        low_ram=getattr(cfg.data, "low_ram", False),
    )

    uploader = ShardUploader(
        dst_repo_id=dst_repo,
        hf_token=hf_token,
    )
    shard_metadata_list: List[Dict[str, Any]] = []
    shard_counter = 0

    # Resume state check: query existing shards
    existing_ids, shard_metadata_list, shard_counter = load_existing_shards(
        dst_repo_id=dst_repo,
        hf_token=hf_token,
    )

    extract_dataset = LatentExtractDataset(
        streaming_dataset, existing_ids=existing_ids
    )
    prefetch_loader = DataLoader(
        extract_dataset,
        batch_size=None,
        num_workers=workers,
        pin_memory=(device_obj.type == "cuda"),
        prefetch_factor=prefetch if workers > 0 else None,
        persistent_workers=(workers > 0),
        worker_init_fn=_worker_init_fn,
    )

    total_samples = streaming_dataset.total_samples or 338011
    pbar = tqdm(
        total=total_samples,
        initial=len(existing_ids),
        desc="Precomputing & Streaming Latents",
    )

    bucket_buffers: Dict[Tuple[int, int], List[Any]] = defaultdict(list)
    current_shard_columns: Dict[str, List[Any]] = create_empty_columns()
    current_shard_count = 0

    
    def _flush_bucket(items: List[Any]) -> None:
        nonlocal shard_counter, current_shard_columns, current_shard_count
        if not items:
            return

        b_size = len(items)
        imgs = torch.stack([item[0] for item in items], dim=0)
        imgs_gpu = imgs.to(device_obj, non_blocking=True)

        with torch.autocast(
            device_type="cuda", dtype=torch_dtype, enabled=True
        ):
            enc = vae.encode(imgs_gpu)
            dist = getattr(enc, "latent_dist", enc)
            lats = dist.sample() if hasattr(dist, "sample") else dist
            lats = (lats - vae_mean_t) / vae_std_t

        # Batch convert to raw binary bytes in C memory
        raw_bytes_list = batch_tensors_to_bytes(lats, dt_str)
        c, h_lat, w_lat = lats.shape[1], lats.shape[2], lats.shape[3]
        shape_list = [int(c), int(h_lat), int(w_lat)]

        # Vectorized batch extension to avoid per-element Python overhead
        metas = [item[1] for item in items]
        current_shard_columns["booru_id"].extend(
            [m["booru_id"] for m in metas]
        )
        current_shard_columns["latent"].extend(raw_bytes_list)
        current_shard_columns["latent_shape"].extend(
            [shape_list] * b_size
        )
        current_shard_columns["latent_dtype"].extend([dt_str] * b_size)
        current_shard_columns["prompt"].extend(
            [m["prompt"] for m in metas]
        )
        current_shard_columns["bucket_idx"].extend(
            [m["bucket_idx"] for m in metas]
        )
        current_shard_columns["target_width"].extend(
            [m["target_width"] for m in metas]
        )
        current_shard_columns["target_height"].extend(
            [m["target_height"] for m in metas]
        )
        current_shard_columns["original_width"].extend(
            [m["original_width"] for m in metas]
        )
        current_shard_columns["original_height"].extend(
            [m["original_height"] for m in metas]
        )
        current_shard_columns["aspect_ratio"].extend(
            [m["aspect_ratio"] for m in metas]
        )
        current_shard_columns["tier"].extend(
            [m["tier"] for m in metas]
        )
        current_shard_columns["aesthetic_tier"].extend(
            [m["aesthetic_tier"] for m in metas]
        )
        current_shard_columns["tag_weight"].extend(
            [m["tag_weight"] for m in metas]
        )

        current_shard_count += b_size
        pbar.update(b_size)

        if current_shard_count >= samples_per_shard:
            meta_info = uploader.stage_and_upload(
                current_shard_columns, shard_counter
            )
            shard_metadata_list.append(meta_info)
            shard_counter += 1
            current_shard_columns = create_empty_columns()
            current_shard_count = 0

        items.clear()

    for item in prefetch_loader:
        img_tensor, res_h, res_w, meta = item
        res_key = (int(res_h), int(res_w))
        bucket_buffers[res_key].append((img_tensor, meta))

        if len(bucket_buffers[res_key]) >= bs:
            _flush_bucket(bucket_buffers[res_key])

    for res_key, items in list(bucket_buffers.items()):
        if items:
            _flush_bucket(items)

    if current_shard_count > 0:
        meta_info = uploader.stage_and_upload(
            current_shard_columns, shard_counter
        )
        shard_metadata_list.append(meta_info)
        shard_counter += 1
        current_shard_columns = create_empty_columns()
        current_shard_count = 0


    pbar.close()

    final_metadata = {
        "total_samples": sum(s["sample_count"] for s in shard_metadata_list),
        "num_shards": len(shard_metadata_list),
        "samples_per_shard": samples_per_shard,
        "length_tiers": streaming_dataset.length_tiers,
        "shards": shard_metadata_list,
        "bucket_info": streaming_dataset.bucket_info,
        "vae_config": {
            "pretrained": vae_path,
            "mean": vae_mean,
            "std": vae_std,
            "channels": in_channels,
            "dtype": dt_str,
        },
    }
    uploader.finalize(final_metadata)
    logging.info("All shards encoded, uploaded, and cleaned successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Fast precomputation of VAE latents using StreamingImageDataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to training configuration YAML file.",
    )
    parser.add_argument(
        "--src_dataset",
        type=str,
        default=None,
        help="Source Hugging Face dataset ID.",
    )
    parser.add_argument(
        "--dst_repo_id",
        type=str,
        default=None,
        help="Destination Hugging Face dataset ID.",
    )
    parser.add_argument(
        "--vae_pretrained",
        type=str,
        default=None,
        help="Pretrained VAE model name or path.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for GPU VAE encoding.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Latent data type ('bf16', 'fp16', 'fp32').",
    )
    parser.add_argument(
        "--samples_per_shard",
        type=int,
        default=10000,
        help="Samples per Parquet shard.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader worker processes for prefetching.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN", None),
        help="Hugging Face write token.",
    )
    args = parser.parse_args()

    cfg = (
        OmegaConf.load(args.config)
        if os.path.exists(args.config)
        else OmegaConf.create({})
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("results", "encode", timestamp)
    Logger.setup_logging(
        save_dir=save_dir,
        logging_name="encode_latents",
    )
    logging.info(cfg)

    precompute_and_upload(
        cfg=cfg,
        src_dataset=args.src_dataset,
        dst_repo_id=args.dst_repo_id,
        vae_pretrained=args.vae_pretrained,
        batch_size=args.batch_size,
        dtype_str=args.dtype,
        samples_per_shard=args.samples_per_shard,
        num_workers=args.num_workers,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()