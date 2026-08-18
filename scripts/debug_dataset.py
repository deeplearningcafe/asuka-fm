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
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import torch
import torchvision.utils as vutils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import create_dataloader
from src.models.text_encoders.tokenizer import HFLLMTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify and visualize dataset, cropping, and padding."
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
        help="Number of batches to iterate through and inspect",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="debug_output",
        help="Directory to save inspection plots",
    )
    return parser.parse_args()


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
    print(f"Saved crop grid to: {save_path}")


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
    print(f"Saved position maps to: {save_path}")


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
    print(f"Saved padding visualization to: {save_path}")


def main():
    args = parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    cfg = OmegaConf.load(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_seed = cfg.train.seed
    torch.manual_seed(current_seed)
    random.seed(current_seed)
    np.random.seed(current_seed)

    print("=" * 78)
    print("ASUKA-FM: Dataset & DataLoader Debug Inspection")
    print("=" * 78)
    print(f"Dataset Type:   {getattr(cfg.data, 'dataset_type', 'h5')}")
    print(f"Base Batch Size:{cfg.train.batch_size}")
    print(f"Use Shift Crop: {cfg.data.get('use_shift_crop', False)}")
    tokenizer = HFLLMTokenizer(cfg.models.hf_text_encoder)

    dataloader = create_dataloader(cfg, rank=0, tokenizer=tokenizer)
    print(f"Total Batches:  {len(dataloader)}")
    print("-" * 78)

    sample_batch = None

    for step, batch in enumerate(dataloader):
        if step >= args.num_batches:
            break

        print(f"[Batch {step + 1:02d}/{args.num_batches}]")

        if len(batch) >= 5:
            images, tokens, mask, pos_map, tag_weights, *rest = batch
            aes_tier = rest[0] if rest else None

            print(
                f"  Images:      {tuple(images.shape)} | "
                f"dtype={images.dtype} | "
                f"range=[{images.min():.2f}, {images.max():.2f}]"
            )
            print(f"  Tokens:      {tuple(tokens.shape)} | dtype={tokens.dtype}")
            print(
                f"  Mask:        {tuple(mask.shape)} | "
                f"Active={mask.sum().item()}/{mask.numel()}"
            )
            print(
                f"  Pos Map:     {tuple(pos_map.shape)} | "
                f"range=[{pos_map.min():.2f}, {pos_map.max():.2f}]"
            )
            print(
                f"  Tag Weights: {tuple(tag_weights.shape)} | "
                f"mean={tag_weights.mean():.3f}"
            )
            if aes_tier is not None:
                print(f"  Aes Tiers:   {tuple(aes_tier.shape)}")

            # Validate sequence length alignment within the batch
            if tokens.shape[1] != mask.shape[1]:
                raise ValueError(
                    f"Mismatch between token len ({tokens.shape[1]}) "
                    f"and mask len ({mask.shape[1]})."
                )

            if sample_batch is None and images.shape[0] >= 4:
                sample_batch = batch
        else:
            latents, cond, tag_weights, mask = batch[:4]
            print(f"  Latents:     {tuple(latents.shape)} | dtype={latents.dtype}")
            print(f"  Cond:        {tuple(cond.shape)} | dtype={cond.dtype}")
            print(f"  Tag Weights: {tuple(tag_weights.shape)}")
            if mask is not None:
                print(f"  Mask:        {tuple(mask.shape)}")

            if sample_batch is None:
                sample_batch = batch

    print("-" * 78)

    if sample_batch is not None and len(sample_batch) >= 5:
        images, tokens, mask, pos_map, *_ = sample_batch
        plot_image_grid(images, output_dir / "grid_crops.png")
        plot_position_maps(pos_map, output_dir / "position_maps.png")
        plot_token_padding(mask, tokens, output_dir / "token_padding.png")
        print("All visual inspections completed successfully.")
    else:
        print("Raw image batch not found; visual inspection skipped.")

    print("=" * 78)


if __name__ == "__main__":
    main()
