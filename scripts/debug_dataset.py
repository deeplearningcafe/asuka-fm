"""
Dataset and DataLoader Verification & Visualization Script.

Validates batch shapes across 10 iterations and generates visual inspections:
1. 4x4 Image Grid of Shifted Square Crops (denormalized to [0, 1]).
2. Continuous 2D RoPE Position Maps (Y and X normalized spatial grids).
3. Token Attention Mask & Padding Heatmap (verifying Tier Batching).
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
import random
import logging
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import torch
import torchvision.utils as vutils
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import create_dataloader
from src.data.streaming_dataset import StreamingImageDataset
from src.models.text_encoders.tokenizer import HFLLMTokenizer
from src.utils.logging_utils import Logger

def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify, profile, and debug dataset and dataloader."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to training configuration YAML file",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=10,
        help="Number of batches to inspect in visual mode",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="debug_output",
        help="Directory to save inspection plots",
    )
    parser.add_argument(
        "--compute_vae_stats",
        action="store_true",
        help="Compute empirical mean and std for VAE latents",
    )
    parser.add_argument(
        "--num_stat_samples",
        type=int,
        default=80000,
        help="Max samples to scan for empirical VAE stats",
    )
    parser.add_argument(
        "--stat_chunk_size",
        type=int,
        default=64,
        help="Chunk size for batching VAE encode during stat calculation",
    )
    parser.add_argument(
        "--profile_full_epoch",
        action="store_true",
        help="Profile DataLoader over an entire epoch to track stall cycles",
    )
    parser.add_argument(
        "--stall_threshold",
        type=float,
        default=1.5,
        help="Latency threshold in seconds to flag a batch delivery stall",
    )
    parser.add_argument(
        "--benchmark_raw_stream",
        action="store_true",
        help="Directly benchmark raw HF stream I/O without DataLoader workers",
    )
    parser.add_argument(
        "--max_stream_samples",
        type=int,
        default=10000,
        help="Maximum raw stream samples to benchmark",
    )
    return parser.parse_args()

def compute_empirical_vae_stats(
    cfg,
    dataloader,
    num_samples: int = 80000,
    chunk_size: int = 64,
):
    """
    Computes online empirical mean and std of VAE latents using
    Chan's parallel algorithm, matching toy-diffusion.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg.train.dtype == "bf16" else torch.float32

    hf_vae_id = getattr(cfg.models, "hf_vae", None) or getattr(
        cfg.models, "vae_pretrained", None
    )
    models_path = cfg.paths.models

    if hf_vae_id:
        logging.info(f"Loading HuggingFace VAE for stats: {hf_vae_id}")
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(
            hf_vae_id,
            torch_dtype=dtype,
            cache_dir=f"{models_path}/vae",
        ).eval()
    else:
        from src.models.vae import Vae, VaeConfig

        vae_path = f"{models_path}/vae/diffusion_pytorch_model.safetensors"
        vae = Vae.from_pretrained(VaeConfig(), vae_path).eval()

    vae.requires_grad_(False)
    vae.to(device)

    logging.info(
        f"Computing empirical stats over max {num_samples} samples "
        f"on {device} ({dtype})..."
    )

    count = 0
    mean = 0.0
    m2 = 0.0
    total_processed = 0

    pbar = tqdm(
        total=num_samples,
        desc="Computing VAE Stats",
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for batch in dataloader:
            if total_processed >= num_samples:
                break

            if len(batch) >= 5:
                images = batch[0]
                batch_latents = []
                for i in range(0, images.shape[0], chunk_size):
                    chunk = images[i : i + chunk_size].to(device, dtype=dtype)
                    enc = vae.encode(chunk)
                    dist = getattr(enc, "latent_dist", enc)
                    lat = dist.sample() if hasattr(dist, "sample") else dist
                    batch_latents.append(lat.cpu())
                latents_tensor = torch.cat(batch_latents, dim=0)
                n_imgs = images.shape[0]
            else:
                latents_tensor = batch[0].cpu()
                n_imgs = latents_tensor.shape[0]

            # Online Chan update in float64 for precision
            chunk_f64 = latents_tensor.to(torch.float64)
            n_b = chunk_f64.numel()
            mean_b = chunk_f64.mean().item()
            m2_b = ((chunk_f64 - mean_b) ** 2).sum().item()

            if count == 0:
                count = n_b
                mean = mean_b
                m2 = m2_b
            else:
                delta = mean_b - mean
                count_next = count + n_b
                mean = mean + delta * (n_b / count_next)
                m2 = m2 + m2_b + (delta**2) * (count * n_b / count_next)
                count = count_next

            total_processed += n_imgs
            pbar.update(n_imgs)

    pbar.close()

    if count > 1:
        empirical_mean = mean
        empirical_std = (m2 / (count - 1)) ** 0.5
    else:
        empirical_mean = 0.0
        empirical_std = 1.0

    vae_shift = empirical_mean
    vae_std = empirical_std
    vae_scale = 1.0 / empirical_std if empirical_std > 0 else 1.0

    logging.info("\n" + "=" * 60)
    logging.info("EMPIRICAL VAE NORMALIZATION STATISTICS")
    logging.info("=" * 60)
    logging.info(f"Total samples processed : {total_processed}")
    logging.info(f"Total latent elements   : {count}")
    logging.info(f"Empirical Mean (Shift)  : {vae_shift:.6f}")
    logging.info(f"Empirical Std           : {vae_std:.6f}")
    logging.info(f"Empirical Scale (1/Std) : {vae_scale:.6f}")
    logging.info("=" * 60)
    logging.info("Suggested config.yaml settings:")
    logging.info("models:")
    logging.info(f"  vae_mean: {vae_shift:.6f}")
    logging.info(f"  vae_std: {vae_std:.6f}")
    logging.info("=" * 60 + "\n")

    return vae_shift, vae_std, vae_scale


def plot_image_grid(images: torch.Tensor, save_path: Path):
    """
    Plots a 4x4 grid of denormalized cropped images.
    """
    num_samples = min(16, images.shape[0])
    imgs = images[:num_samples].clone().detach().cpu().float()

    # Denormalize from [-1, 1] to [0, 1]
    imgs = torch.clamp((imgs * 0.5) + 0.5, 0.0, 1.0)

    grid = vutils.make_grid(imgs, nrow=4, padding=4, normalize=False)
    np_grid = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(np_grid)
    plt.axis("off")
    plt.title(
        f"Shifted Square Crops (4x4 Grid, N={num_samples})",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved crop grid to: {save_path}")


def plot_position_maps(pos_maps: torch.Tensor, save_path: Path):
    """
    Plots 2D RoPE position maps for Y and X coordinates across 4x4 grid.
    """
    num_samples = min(16, pos_maps.shape[0])
    pos = pos_maps[:num_samples].detach().cpu().float()  # [B, N, 2]

    num_patches = pos.shape[1]
    grid_size = int(math.isqrt(num_patches))

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    for idx in range(16):
        ax = axes[idx]
        if idx < num_samples:
            sample_pos = pos[idx].view(grid_size, grid_size, 2)
            y_coords = sample_pos[..., 0].numpy()
            x_coords = sample_pos[..., 1].numpy()

            ax.imshow(y_coords, cmap="coolwarm", origin="upper")
            ax.set_title(
                f"S{idx}: Y[{y_coords.min():.2f}, {y_coords.max():.2f}]\n"
                f"X[{x_coords.min():.2f}, {x_coords.max():.2f}]",
                fontsize=8,
            )
        ax.axis("off")

    fig.suptitle(
        "RoPE Position Maps (Y-coord Heatmap [-r_h, r_h])",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved position maps to: {save_path}")


def plot_token_padding(masks: torch.Tensor, tokens: torch.Tensor, save_path: Path):
    """
    Visualizes attention masks and padding distribution across the batch.
    """
    num_samples = min(16, masks.shape[0])
    mask_np = masks[:num_samples].detach().cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.imshow(
        mask_np,
        cmap="Blues",
        aspect="auto",
        interpolation="nearest",
    )
    plt.colorbar(
        label="Token Mask (1=Active Tag, 0=Pad)",
        ticks=[0, 1],
    )
    plt.xlabel("Token Sequence Position (Tier Dimension)", fontsize=11)
    plt.ylabel("Batch Sample Index", fontsize=11)
    plt.title(
        f"Token Tier Attention Masks (Batch Size={masks.shape[0]}, "
        f"Seq Len={masks.shape[1]})",
        fontsize=13,
    )

    total_tokens = mask_np.size
    active_tokens = int(mask_np.sum())
    pad_tokens = total_tokens - active_tokens
    efficiency = (active_tokens / total_tokens) * 100.0

    plt.figtext(
        0.15,
        0.02,
        f"Active Tokens: {active_tokens} | Pad Tokens: {pad_tokens} | "
        f"Packing Efficiency: {efficiency:.1f}%",
        fontsize=10,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved padding visualization to: {save_path}")



def benchmark_raw_stream(cfg, max_samples: int = 10000):
    """
    Measures isolated raw Hugging Face streaming speed and network throughput.
    Helps separate network bandwidth bottlenecks from DataLoader multi-worker
    contention.
    """
    logging.info("=" * 78)
    logging.info("BENCHMARKING RAW HUGGINGFACE STREAM (ISOLATED I/O)")
    logging.info("=" * 78)

    dataset = StreamingImageDataset(
        dataset_name=cfg.data.streaming_dataset_name,
        dataset_path=cfg.data.get("dataset_path", None),
        resolution=cfg.data.get("resolution", 512),
        low_ram=getattr(cfg.data, "low_ram", False),
        is_latent=cfg.data.get("is_latent", False),
    )

    t_start = time.perf_counter()
    sample_count = 0
    total_bytes = 0
    latencies = []
    last_time = t_start

    pbar = tqdm(
        total=max_samples,
        desc="Raw Stream Fetching",
        dynamic_ncols=True,
    )

    for sample in dataset.iter_raw():
        curr_time = time.perf_counter()
        delta = curr_time - last_time
        latencies.append(delta)
        last_time = curr_time

        sample_count += 1
        raw_latent = sample.get("latent")
        raw_img = sample.get("image") or sample.get("bytes")
        if isinstance(raw_latent, bytes):
            total_bytes += len(raw_latent)
        elif isinstance(raw_img, bytes):
            total_bytes += len(raw_img)

        pbar.update(1)
        if sample_count >= max_samples:
            break

    pbar.close()
    total_elapsed = time.perf_counter() - t_start

    avg_sps = sample_count / max(total_elapsed, 1e-6)
    mb_transferred = total_bytes / (1024 * 1024)
    mb_rate = mb_transferred / max(total_elapsed, 1e-6)

    logging.info("-" * 78)
    logging.info(f"Samples Processed   : {sample_count}")
    logging.info(f"Total Time Elapsed  : {total_elapsed:.2f} s")
    logging.info(f"Throughput          : {avg_sps:.2f} samples/sec")
    logging.info(f"Payload Transferred : {mb_transferred:.2f} MB")
    logging.info(f"Bandwidth           : {mb_rate:.2f} MB/sec")
    if latencies:
        p50 = np.percentile(latencies, 50) * 1000.0
        p95 = np.percentile(latencies, 95) * 1000.0
        p99 = np.percentile(latencies, 99) * 1000.0
        logging.info(
            f"Latency per sample  : P50={p50:.2f}ms | P95={p95:.2f}ms | "
            f"P99={p99:.2f}ms"
        )
    logging.info("=" * 78)


def run_dataloader_diagnostics(dataloader, stall_threshold: float = 1.5):
    """
    Monitors DataLoader execution across batches, detecting stalls and
    measuring throughput stability to pinpoint buffer exhaustion.
    """
    logging.info("=" * 78)
    logging.info("DATALOADER STALL & THROUGHPUT PROFILER")
    logging.info(f"Stall Latency Threshold: {stall_threshold:.2f} seconds")
    logging.info("=" * 78)

    batch_latencies = []
    stall_events = []
    batches_since_stall = 0
    total_samples = 0
    start_time = time.perf_counter()
    prev_time = start_time

    pbar = tqdm(
        dataloader,
        desc="Profiling DataLoader",
        dynamic_ncols=True,
    )

    for step, batch in enumerate(pbar):
        curr_time = time.perf_counter()
        delta = curr_time - prev_time
        batch_latencies.append(delta)

        bsz = batch[0].shape[0] if isinstance(batch, (list, tuple)) else 1
        total_samples += bsz
        batches_since_stall += 1

        if delta >= stall_threshold:
            stall_events.append(
                {
                    "step": step,
                    "pause_time": delta,
                    "batches_since_last": batches_since_stall,
                    "samples_since_last": batches_since_stall * bsz,
                }
            )
            logging.warning(
                f"[STALL DETECTED] Step {step:05d} | "
                f"Pause: {delta:.2f}s | "
                f"Interval: {batches_since_stall} batches "
                f"(~{batches_since_stall * bsz} samples)"
            )
            batches_since_stall = 0

        prev_time = curr_time
        inst_rate = 1.0 / max(delta, 1e-6)
        pbar.set_postfix({"it/s": f"{inst_rate:.2f}", "last_dt": f"{delta:.2f}s"})

    pbar.close()
    total_time = time.perf_counter() - start_time
    total_batches = len(batch_latencies)

    if total_batches == 0:
        logging.error("No batches were yielded by the DataLoader.")
        return

    latencies_np = np.array(batch_latencies)
    avg_it_s = total_batches / max(total_time, 1e-6)
    avg_sps = total_samples / max(total_time, 1e-6)

    logging.info("\n" + "=" * 78)
    logging.info("PROFILING SUMMARY & DIAGNOSTIC REPORT")
    logging.info("=" * 78)
    logging.info(f"Total Batches Yielded : {total_batches}")
    logging.info(f"Total Samples Yielded : {total_samples}")
    logging.info(f"Total Time Elapsed    : {total_time:.2f} s")
    logging.info(f"Mean Throughput       : {avg_it_s:.2f} it/s ({avg_sps:.2f} samples/s)")
    logging.info(
        f"Batch Latencies       : Min={latencies_np.min():.4f}s | "
        f"Mean={latencies_np.mean():.4f}s | "
        f"P50={np.percentile(latencies_np, 50):.4f}s | "
        f"P95={np.percentile(latencies_np, 95):.4f}s | "
        f"Max={latencies_np.max():.4f}s"
    )
    logging.info(f"Total Stalls Detected : {len(stall_events)}")

    if stall_events:
        intervals = [e["batches_since_last"] for e in stall_events[1:]]
        pauses = [e["pause_time"] for e in stall_events]
        logging.info(
            f"Mean Stall Duration   : {np.mean(pauses):.2f}s "
            f"(Max: {np.max(pauses):.2f}s)"
        )
        if intervals:
            logging.info(
                f"Mean Batches/Stall    : {np.mean(intervals):.1f} batches "
                f"(~{np.mean(intervals) * (total_samples / total_batches):.0f} samples)"
            )
            logging.info("Diagnosed Root Cause  : Multi-worker buffer exhaustion cycle.")
    else:
        logging.info("Diagnosis             : Pipeline is streaming continuously.")
    logging.info("=" * 78 + "\n")

def main():
    args = parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    cfg = OmegaConf.load(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Logger.setup_logging(
        save_dir=output_dir,
        logging_name="debug_latents",
    )
    logging.info(cfg)

    current_seed = cfg.train.seed
    torch.manual_seed(current_seed)
    random.seed(current_seed)
    np.random.seed(current_seed)

    if args.benchmark_raw_stream:
        benchmark_raw_stream(
            cfg=cfg,
            max_samples=args.max_stream_samples,
        )
        return

    logging.info("=" * 78)
    logging.info("ASUKA-FM: Dataset & DataLoader Debug Inspection")
    logging.info("=" * 78)
    logging.info(f"Dataset Type:   {getattr(cfg.data, 'dataset_type', 'h5')}")
    logging.info(f"Base Batch Size:{cfg.train.batch_size}")
    logging.info(f"Use Shift Crop: {cfg.data.get('use_shift_crop', False)}")
    tokenizer = HFLLMTokenizer(cfg.models.hf_text_encoder)

    dataloader = create_dataloader(cfg, rank=0, tokenizer=tokenizer)
    try:
        total_batches = len(dataloader)
        logging.info(f"Total Batches:  {total_batches}")
    except TypeError:
        logging.info("Total Batches:  Dynamic (Streaming Iterable)")
    logging.info("-" * 78)

    if args.compute_vae_stats:
        compute_empirical_vae_stats(
            cfg=cfg,
            dataloader=dataloader,
            num_samples=args.num_stat_samples,
            chunk_size=args.stat_chunk_size,
        )
        return

    if args.profile_full_epoch:
        run_dataloader_diagnostics(
            dataloader=dataloader,
            stall_threshold=args.stall_threshold,
        )
        return

    sample_batch = None

    for step, batch in enumerate(dataloader):
        if step >= args.num_batches:
            break

        logging.info(f"[Batch {step + 1:02d}/{args.num_batches}]")

        if len(batch) >= 5:
            images, tokens, mask, pos_map, tag_weights, *rest = batch
            aes_tier = rest[0] if rest else None

            logging.info(
                f"  Images:      {tuple(images.shape)} | "
                f"dtype={images.dtype} | "
                f"range=[{images.min():.2f}, {images.max():.2f}]"
            )
            logging.info(
                f"  Tokens:      {tuple(tokens.shape)} | "
                f"dtype={tokens.dtype}"
            )
            logging.info(
                f"  Mask:        {tuple(mask.shape)} | "
                f"Active={mask.sum().item()}/{mask.numel()}"
            )
            logging.info(
                f"  Pos Map:     {tuple(pos_map.shape)} | "
                f"range=[{pos_map.min():.2f}, {pos_map.max():.2f}]"
            )
            logging.info(
                f"  Tag Weights: {tuple(tag_weights.shape)} | "
                f"mean={tag_weights.mean():.3f}"
            )
            if aes_tier is not None:
                logging.info(f"  Aes Tiers:   {tuple(aes_tier.shape)}")

            if tokens.shape[1] != mask.shape[1]:
                raise ValueError(
                    f"Mismatch between token len ({tokens.shape[1]}) "
                    f"and mask len ({mask.shape[1]})."
                )

            if sample_batch is None and images.shape[0] >= 4:
                sample_batch = batch
        else:
            latents, cond, tag_weights, mask = batch[:4]
            logging.info(
                f"  Latents:     {tuple(latents.shape)} | "
                f"dtype={latents.dtype}"
            )
            logging.info(
                f"  Cond:        {tuple(cond.shape)} | "
                f"dtype={cond.dtype}"
            )
            logging.info(f"  Tag Weights: {tuple(tag_weights.shape)}")
            if mask is not None:
                logging.info(f"  Mask:        {tuple(mask.shape)}")

            if sample_batch is None:
                sample_batch = batch

    logging.info("-" * 78)

    if sample_batch is not None and len(sample_batch) >= 5:
        images, tokens, mask, pos_map, *_ = sample_batch
        plot_image_grid(images, output_dir / "grid_crops.png")
        plot_position_maps(pos_map, output_dir / "position_maps.png")
        plot_token_padding(mask, tokens, output_dir / "token_padding.png")
        logging.info("All visual inspections completed successfully.")
    else:
        logging.info("Raw image batch not found; visual inspection skipped.")

    logging.info("=" * 78)

if __name__ == "__main__":
    main()
