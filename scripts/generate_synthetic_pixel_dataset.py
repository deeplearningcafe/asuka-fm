"""Generates synthetic pixel dataset from latent model and streams to HF."""

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
import io
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import HfApi, hf_hub_download
from omegaconf import DictConfig, OmegaConf
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from src.data.streaming_dataset import StreamingImageDataset
from src.diffusion.sampling import generate_samples
from src.diffusion.schedules import DDPMSchedule, LinearSchedule
from src.models.factory import load_trainable_model
from src.utils.logging_utils import Logger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def make_preview_grid(
    images: List[Image.Image], cols: int = 4, rows: int = 4
) -> Image.Image:
    """Stitches sample images into a preview grid."""
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h))
    for idx, img in enumerate(images[: cols * rows]):
        logging.info(f"Image shape: {img.size}")
        x = (idx % cols) * w
        y = (idx // cols) * h
        grid.paste(img, (x, y))
    return grid


def get_parquet_schema() -> pa.Schema:
    """PyArrow schema for synthetic pixel image shards."""
    return pa.schema(
        [
            ("booru_id", pa.string()),
            ("image", pa.binary()),
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


def create_empty_columns() -> Dict[str, List[Any]]:
    """Columnar dictionary for fast PyArrow table creation."""
    return {
        "booru_id": [],
        "image": [],
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


def image_to_avif_bytes(img: Image.Image, quality: int = 80, speed: int = 6) -> bytes:
    """Encodes PIL Image to AVIF format in memory."""
    if img.mode in ("P", "1", "RGBA"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="AVIF", quality=quality, speed=speed)
    return buf.getvalue()


def load_existing_shards(
    dst_repo_id: str, hf_token: Optional[str] = None
) -> Tuple[set, List[Dict[str, Any]], int]:
    """Loads existing sample IDs to enable resuming without disk bloat."""
    api = HfApi(token=hf_token)
    try:
        repo_files = api.list_repo_files(repo_id=dst_repo_id, repo_type="dataset")
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
        f"Found {len(shard_files)} shards in {dst_repo_id}. "
        "Inspecting sample IDs with single-file staging..."
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
            finally:
                if local_path and os.path.exists(local_path):
                    os.remove(local_path)

    logging.info(
        f"Resuming: {len(existing_ids)} samples already uploaded "
        f"across {len(shard_meta)} shards."
    )
    return existing_ids, shard_meta, len(shard_meta)


class ShardUploader:
    """Handles Parquet serialization and async upload with immediate cleanup."""

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
        self.temp_dir = Path(tempfile.mkdtemp(prefix="pixel_shards_"))
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
    """Applies TF32 and Tensor Core optimizations."""
    device_obj = torch.device(device_str)
    if device_obj.type != "cuda":
        return device_obj, torch.float32

    capability = torch.cuda.get_device_capability(device_obj)
    if capability[0] >= 8:
        autocast_dtype = torch.bfloat16
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        autocast_dtype = torch.float32

    return device_obj, autocast_dtype


@torch.inference_mode()
def generate_and_upload_synthetic(
    cfg: DictConfig,
    src_dataset: Optional[str] = None,
    dst_repo_id: Optional[str] = None,
    checkpoint: Optional[str] = None,
    max_samples: int = 100000,
    samples_per_shard: int = 10000,
    batch_size: int = 8,
    sample_steps: int = 30,
    cfg_scale: float = 6.0,
    shift: float = 1.0,
    negative_prompt: str = (
        "worst quality, low quality, displeasing, bad score, worse score"
    ),
    resolution: Optional[int] = None,
    target_aesthetic_tiers: Optional[List[int]] = None,
    vae_batch_size: int = 4,
    cfg_interval: Tuple[float, float] = (0.08, 0.92),
    compile_vae: bool = True,
    preview_dir: Optional[str] = None,
    avif_quality: int = 80,
    avif_speed: int = 6,
    hf_token: Optional[str] = None,
    device: str = "cuda",
) -> None:
    """Generates synthetic images via latent model and streams to HF."""
    device_obj, default_autocast_dtype = setup_cuda_environment(device)

    models_path = cfg.paths.models
    ckpt_path = checkpoint or cfg.models.get("resume_from_checkpoint", None)
    model_type = getattr(cfg.models, "model_type", "unet")
    dtype_str = cfg.train.get("dtype", "bf16")
    torch_dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float32
    is_dit = model_type in ["dual_stream", "sprint_dual"]

    logging.info(f"Loading base latent model ({model_type}) from {models_path}...")
    unet, text_encoder, vae, tokenizer, _ = load_trainable_model(
        models_path=models_path,
        device=device_obj,
        dtype=torch_dtype,
        train_te=False,
        use_checkpointing=False,
        resume_from_checkpoint=ckpt_path,
        train_only_output=False,
        global_rank=0,
        model_type=model_type,
        model_cfg=cfg.models,
        autocast_dtype=default_autocast_dtype,
    )
    unet.eval().requires_grad_(False)
    text_encoder.eval().requires_grad_(False)
    vae.eval().requires_grad_(False)
    unet = torch.compile(unet)
    text_encoder = torch.compile(text_encoder)

    if compile_vae and hasattr(torch, "compile"):
        try:
            logging.info("Compiling VAE decode stage with torch.compile...")
            vae.decode = torch.compile(vae.decode)
        except Exception as e:
            logging.warning(f"Could not compile VAE decode: {e}")

    diffusion_type = cfg.train.get("objective", "flow_matching")
    if diffusion_type == "flow_matching":
        schedule = LinearSchedule(device=device_obj)
    else:
        schedule = DDPMSchedule(device=device_obj)

    vae_mean = getattr(cfg.models, "vae_mean", 0.0)
    vae_std = getattr(cfg.models, "vae_std", 1.0 / 0.18215)
    vae_mean_t = torch.tensor(vae_mean, device=device_obj, dtype=torch_dtype).view(
        1, -1, 1, 1
    )
    vae_std_t = torch.tensor(vae_std, device=device_obj, dtype=torch_dtype).view(
        1, -1, 1, 1
    )
    in_channels = cfg.models.get("in_channels", 4)
    coord_system = (
        "aspect_norm"
        if getattr(cfg.models, "use_calibrated_spatial", False)
        else "discrete"
    )

    src_ds = src_dataset or cfg.data.get(
        "streaming_dataset_name", "aipracticecafe/curated-danbooru-2026"
    )
    dst_repo = (
        dst_repo_id
        or cfg.logging.get("synthetic_hf_repo", None)
        or f"{src_ds}-synthetic-pixel"
    )
    target_res = resolution or cfg.data.get("resolution", 512)

    streaming_dataset = StreamingImageDataset(
        dataset_name=src_ds,
        dataset_path=cfg.data.get("dataset_path", None),
        resolution=target_res,
        rank=0,
        world_size=1,
        low_ram=True,
        target_aesthetic_tiers=target_aesthetic_tiers,
    )

    uploader = ShardUploader(
        dst_repo_id=dst_repo,
        hf_token=hf_token,
    )
    existing_ids, shard_metadata_list, shard_counter = load_existing_shards(
        dst_repo_id=dst_repo,
        hf_token=hf_token,
    )
    total_uploaded = len(existing_ids)
    if total_uploaded >= max_samples:
        logging.info(
            f"Dataset target reached: {total_uploaded}/{max_samples} uploaded."
        )
        return

    pbar = tqdm(
        total=max_samples,
        initial=total_uploaded,
        desc="Generating & Streaming Synthetic Pixel Dataset",
    )

    current_shard_columns = create_empty_columns()
    current_shard_count = 0
    total_generated = total_uploaded

    batch_configs: List[Dict[str, Any]] = []
    batch_metas: List[Dict[str, Any]] = []
    has_saved_preview = False

    pipeline_executor = ThreadPoolExecutor(max_workers=1)
    pending_encode_future: Optional[Future] = None

    def _sync_encode_and_stage(
        images: List[Image.Image],
        metas: List[Dict[str, Any]],
    ) -> None:
        """Encodes images to AVIF and stages columns in background thread."""
        nonlocal shard_counter, current_shard_columns, current_shard_count
        nonlocal total_generated

        with ThreadPoolExecutor(max_workers=min(len(images), 8)) as ex:
            avif_bytes = list(
                ex.map(
                    lambda im: image_to_avif_bytes(
                        im, quality=avif_quality, speed=avif_speed
                    ),
                    images,
                )
            )

        n_done = len(metas)
        current_shard_columns["booru_id"].extend([m["booru_id"] for m in metas])
        current_shard_columns["image"].extend(avif_bytes)
        current_shard_columns["prompt"].extend([m["prompt"] for m in metas])
        current_shard_columns["bucket_idx"].extend([m["bucket_idx"] for m in metas])
        current_shard_columns["target_width"].extend([m["target_width"] for m in metas])
        current_shard_columns["target_height"].extend(
            [m["target_height"] for m in metas]
        )
        current_shard_columns["original_width"].extend(
            [m["original_width"] for m in metas]
        )
        current_shard_columns["original_height"].extend(
            [m["original_height"] for m in metas]
        )
        current_shard_columns["aspect_ratio"].extend([m["aspect_ratio"] for m in metas])
        current_shard_columns["tier"].extend([m["tier"] for m in metas])
        current_shard_columns["aesthetic_tier"].extend(
            [m["aesthetic_tier"] for m in metas]
        )
        current_shard_columns["tag_weight"].extend([m["tag_weight"] for m in metas])

        current_shard_count += n_done
        total_generated += n_done
        pbar.update(n_done)

        if current_shard_count >= samples_per_shard:
            meta_info = uploader.stage_and_upload(current_shard_columns, shard_counter)
            shard_metadata_list.append(meta_info)
            shard_counter += 1
            current_shard_columns = create_empty_columns()
            current_shard_count = 0

    def _process_batch():
        nonlocal pending_encode_future, has_saved_preview

        if not batch_configs:
            return

        images = generate_samples(
            unet=unet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            schedule=schedule,
            sample_configs=batch_configs,
            global_batch_size=batch_size,
            diffusion_type=diffusion_type,
            device=device_obj,
            dtype=torch_dtype,
            autocast_dtype=default_autocast_dtype,
            use_unet_mult=False if is_dit else True,
            vae_mean=vae_mean_t,
            vae_std=vae_std_t,
            in_channels=in_channels,
            coord_system=coord_system,
            pixel_sampling=False,
            vae_batch_size=vae_batch_size,
            cfg_interval=cfg_interval,
        )

        if preview_dir and not has_saved_preview and len(images) >= 16:
            p_dir = Path(preview_dir)
            p_dir.mkdir(parents=True, exist_ok=True)
            grid_img = make_preview_grid(images[:16], cols=4, rows=4)
            grid_img.save(p_dir / "preview_grid.png")
            with open(p_dir / "first_batch_meta.json", "w", encoding="utf-8") as f:
                json.dump(batch_metas[:16], f, indent=4)
            logging.info(f"Saved 16-sample preview grid to {p_dir}.")
            has_saved_preview = True

        if pending_encode_future is not None:
            pending_encode_future.result()

        # Dispatch current batch encoding to background thread
        pending_encode_future = pipeline_executor.submit(
            _sync_encode_and_stage,
            images,
            list(batch_metas),
        )

        batch_configs.clear()
        batch_metas.clear()

    sample_seed = cfg.train.get("seed", 42)
    patch_size = getattr(cfg.models, "patch_size", 2)
    align_down = patch_size * 8

    for raw_sample in streaming_dataset.iter_raw():
        if total_generated + len(batch_configs) >= max_samples:
            break

        booru_id = str(raw_sample.get("booru_id", ""))
        if booru_id in existing_ids:
            continue

        prompt = str(raw_sample.get("prompt") or raw_sample.get("text", "")).strip()
        if not prompt:
            continue

        orig_w = int(raw_sample.get("original_width", target_res))
        orig_h = int(raw_sample.get("original_height", target_res))
        ar = float(raw_sample.get("aspect_ratio", orig_w / max(1, orig_h)))
        tier = int(raw_sample.get("tier", 227))
        aesthetic_tier = int(raw_sample.get("aesthetic_tier", -1))
        tag_weight = float(raw_sample.get("tag_weight", 1.0))
        bucket_idx = int(raw_sample.get("bucket_idx", -1))

        #res_w = int(raw_sample.get("target_width", target_res))
        #res_h = int(raw_sample.get("target_height", target_res))

        target_width = int(raw_sample.get("target_width", target_res))
        target_height = int(raw_sample.get("target_height", target_res))
        min_orig = max(1, min(target_height, target_width))
        scale = target_res / min_orig
        scaled_w = max(target_res, int(round(target_width * scale)))
        scaled_h = max(target_res, int(round(target_height * scale)))

        # Align dimensions to patch_size * vae_downsample_factor (16px)
        res_w = (scaled_w // align_down) * align_down
        res_h = (scaled_h // align_down) * align_down
        sample_seed += 1
        sample_cfg = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": res_h,
            "width": res_w,
            "sample_steps": sample_steps,
            "cfg_scale": cfg_scale,
            "shift": shift,
            "seed": sample_seed,
        }
        meta = {
            "booru_id": booru_id,
            "prompt": prompt,
            "bucket_idx": bucket_idx,
            "target_width": res_w,
            "target_height": res_h,
            "original_width": orig_w,
            "original_height": orig_h,
            "aspect_ratio": ar,
            "tier": tier,
            "aesthetic_tier": aesthetic_tier,
            "tag_weight": tag_weight,
        }

        batch_configs.append(sample_cfg)
        batch_metas.append(meta)

        if len(batch_configs) >= batch_size:
            _process_batch()

    if batch_configs and total_generated < max_samples:
        _process_batch()

    if pending_encode_future is not None:
        pending_encode_future.result()
    pipeline_executor.shutdown(wait=True)

    if current_shard_count > 0:
        meta_info = uploader.stage_and_upload(current_shard_columns, shard_counter)
        shard_metadata_list.append(meta_info)
        shard_counter += 1
        current_shard_columns = create_empty_columns()
        current_shard_count = 0

    pbar.close()

    final_metadata = {
        "total_samples": sum(s["sample_count"] for s in shard_metadata_list),
        "num_shards": len(shard_metadata_list),
        "samples_per_shard": samples_per_shard,
        "format": "avif",
        "resolution": target_res,
        "shards": shard_metadata_list,
        "generator_checkpoint": str(ckpt_path),
    }
    uploader.finalize(final_metadata)
    logging.info(f"Finished generating {total_generated} samples to {dst_repo}.")


def main():
    parser = argparse.ArgumentParser(
        description="Stream synthetic pixel dataset to Hugging Face."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to training configuration YAML file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Model checkpoint path (overrides config).",
    )
    parser.add_argument(
        "--src_dataset",
        type=str,
        default=None,
        help="Source Hugging Face prompt dataset ID.",
    )
    parser.add_argument(
        "--dst_repo_id",
        type=str,
        default=None,
        help="Destination Hugging Face dataset ID.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100000,
        help="Maximum total synthetic samples to generate.",
    )
    parser.add_argument(
        "--samples_per_shard",
        type=int,
        default=10000,
        help="Number of samples per uploaded Parquet shard.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Inference batch size for sampling.",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=30,
        help="Sampling steps for Flow Matching Euler ODE.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=6.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--cfg_interval",
        nargs=2,
        type=float,
        default=[0.08, 0.92],
        help="Active interval [min, max] for CFG application.",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=1.0,
        help="Time SNR shift schedule parameter.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Target square image resolution.",
    )
    parser.add_argument(
        "--target_aesthetic_tiers",
        nargs="+",
        type=int,
        default=[2, 3],
        help="Whitelist of aesthetic tiers to sample (e.g. 2 3).",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=("worst quality, low quality, displeasing, bad score, worse score"),
        help="Default negative prompt for CFG inference.",
    )
    parser.add_argument(
        "--vae_batch_size",
        type=int,
        default=16,
        help="Batch size for VAE decoding.",
    )
    parser.add_argument(
        "--compile_vae",
        action="store_true",
        default=True,
        help="Compile VAE decoder to optimize memory bandwidth on A40.",
    )
    parser.add_argument(
        "--avif_quality",
        type=int,
        default=80,
        help="AVIF quality (1-100).",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN", None),
        help="Hugging Face write token.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Target execution device.",
    )
    args = parser.parse_args()

    cfg = (
        OmegaConf.load(args.config)
        if os.path.exists(args.config)
        else OmegaConf.create({})
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("results", "generate_synthetic", timestamp)
    Logger.setup_logging(
        save_dir=save_dir,
        logging_name="generate_synthetic_pixel",
    )
    logging.info(cfg)

    generate_and_upload_synthetic(
        cfg=cfg,
        src_dataset=args.src_dataset,
        dst_repo_id=args.dst_repo_id,
        checkpoint=args.checkpoint,
        max_samples=args.max_samples,
        samples_per_shard=args.samples_per_shard,
        batch_size=args.batch_size,
        sample_steps=args.sample_steps,
        cfg_scale=args.cfg_scale,
        shift=args.shift,
        negative_prompt=args.negative_prompt,
        resolution=args.resolution,
        target_aesthetic_tiers=args.target_aesthetic_tiers,
        vae_batch_size=args.vae_batch_size,
        preview_dir=save_dir,
        avif_quality=args.avif_quality,
        hf_token=args.hf_token,
        device=args.device,
    )


if __name__ == "__main__":
    main()
