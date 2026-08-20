"""RoPE Spatial Geometry and Phase Collapse Debugger.

Compares Discrete, Broken Continuous, and Corrected Patch-Unit RoPE.
"""

import argparse
import math
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw


def compute_rope_inv_freqs(dim: int = 16, theta: float = 10000.0) -> torch.Tensor:
    """Computes RoPE inverse frequency scales for a given axis dimension."""
    steps = torch.arange(0, dim, 2, dtype=torch.float32)
    return 1.0 / (theta ** (steps / dim))


def get_discrete_coords(h_patches: int, w_patches: int) -> torch.Tensor:
    """Case 1: Standard integer patch coordinates [0, H-1] and [0, W-1]."""
    y = torch.arange(h_patches, dtype=torch.float32)
    x = torch.arange(w_patches, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((grid_y, grid_x), dim=-1).flatten(0, 1)


def get_broken_continuous_coords(
    h_patches: int,
    w_patches: int,
    crop_y: int,
    crop_x: int,
    target_h: int,
    target_w: int,
    patch_size_px: int,
    apply_long_cast: bool = False,
) -> torch.Tensor:
    """Case 2: Broken normalized coordinates [-r_h, r_h] with optional

    .long() truncation bug.
    """
    r_h = math.sqrt(target_h / target_w)
    r_w = math.sqrt(target_w / target_h)

    y_centers = (
        torch.arange(h_patches, dtype=torch.float32) + 0.5
    ) * patch_size_px + crop_y
    x_centers = (
        torch.arange(w_patches, dtype=torch.float32) + 0.5
    ) * patch_size_px + crop_x

    y_norm = (y_centers / target_h) * (2.0 * r_h) - r_h
    x_norm = (x_centers / target_w) * (2.0 * r_w) - r_w

    grid_y, grid_x = torch.meshgrid(y_norm, x_norm, indexing="ij")
    coords = torch.stack((grid_y, grid_x), dim=-1).flatten(0, 1)

    if apply_long_cast:
        # Simulates the .long() cast bug in _forward_discrete
        coords = coords.long().float()
    return coords


def get_corrected_continuous_coords(
    h_patches: int,
    w_patches: int,
    crop_y: int,
    crop_x: int,
    patch_size_px: int,
) -> torch.Tensor:
    """Case 3: Corrected continuous coordinates in patch units."""
    crop_y_patch = crop_y / float(patch_size_px)
    crop_x_patch = crop_x / float(patch_size_px)

    y_pos = torch.arange(h_patches, dtype=torch.float32) + crop_y_patch
    x_pos = torch.arange(w_patches, dtype=torch.float32) + crop_x_patch

    grid_y, grid_x = torch.meshgrid(y_pos, x_pos, indexing="ij")
    return torch.stack((grid_y, grid_x), dim=-1).flatten(0, 1)


def compute_relative_kernel(
    coords: torch.Tensor, inv_freqs: torch.Tensor, probe_idx: int
) -> torch.Tensor:
    """Computes mean relative rotary similarity kernel cos(delta_pos *

    inv_freqs) relative to probe index.
    """
    probe_coord = coords[probe_idx]  # [2]
    diff = coords - probe_coord  # [N, 2]

    # Diff Y and Diff X evaluated against inv_freqs
    angles_y = diff[:, 0:1] * inv_freqs.unsqueeze(0)  # [N, D/2]
    angles_x = diff[:, 1:2] * inv_freqs.unsqueeze(0)  # [N, D/2]

    cos_y = angles_y.cos().mean(dim=-1)  # [N]
    cos_x = angles_x.cos().mean(dim=-1)  # [N]
    return 0.5 * (cos_y + cos_x)


def load_or_create_image(
    image_path: Optional[str], size: Tuple[int, int] = (512, 256)
) -> Image.Image:
    """Loads image from path or creates a synthetic test card."""
    if image_path and os.path.isfile(image_path):
        return Image.open(image_path).convert("RGB")

    w, h = size
    img = Image.new("RGB", (w, h), color=(30, 30, 45))
    draw = ImageDraw.Draw(img)

    # Draw synthetic grid and pattern
    for x in range(0, w, 32):
        draw.line([(x, 0), (x, h)], fill=(70, 70, 95), width=1)
    for y in range(0, h, 32):
        draw.line([(0, y), (w, y)], fill=(70, 70, 95), width=1)

    draw.ellipse([w // 4, h // 6, 3 * w // 4, 5 * h // 6], fill=(180, 50, 60))
    draw.text((20, 20), "Synthetic Test Pattern", fill=(240, 240, 240))
    return img


def print_structured_summary(
    inv_freqs: torch.Tensor,
    delta_discrete: float,
    delta_broken: float,
    delta_corrected: float,
) -> None:
    """Prints tabular RoPE rotation phase differences to stdout."""
    print("=" * 78)
    print(" RoPE Phase Shift (Delta Theta) Across Frequency Channels")
    print("=" * 78)
    print(
        f"{'Channel (k)':<12} | {'InvFreq (rad)':<15} | "
        f"{'Discrete (deg)':<15} | {'Broken (deg)':<15} | {'Corrected (deg)'}"
    )
    print("-" * 78)

    for k, freq in enumerate(inv_freqs.tolist()):
        deg_disc = math.degrees(delta_discrete * freq)
        deg_broken = math.degrees(delta_broken * freq)
        deg_corr = math.degrees(delta_corrected * freq)
        print(
            f"k={k:<10} | {freq:<15.5f} | {deg_disc:<15.2f} | "
            f"{deg_broken:<15.2f} | {deg_corr:.2f}"
        )
    print("=" * 78)
    print("Diagnosis:")
    print(" - Broken Continuous collapses phase shifts (Channel 0 is only")
    print("   ~7.16 deg; lower channels are ~0 deg).")
    print(" - Attention cannot resolve patch boundaries -> square artifacts.")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="RoPE Continuous vs Discrete Debugger")
    parser.add_argument(
        "--image_path", type=str, default=None, help="Path to reference image"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/debug_rope", help="Output dir"
    )
    parser.add_argument("--crop_size", type=int, default=256, help="Square crop size")
    parser.add_argument(
        "--patch_size", type=int, default=2, help="DiT latent patch size"
    )
    parser.add_argument(
        "--vae_scale", type=int, default=8, help="VAE spatial downsample scale"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    patch_px = args.patch_size * args.vae_scale  # e.g., 2 * 8 = 16px
    h_patches = args.crop_size // patch_px
    w_patches = args.crop_size // patch_px
    n_patches = h_patches * w_patches

    # 1. Load image and apply crop
    orig_img = load_or_create_image(args.image_path)
    orig_w, orig_h = orig_img.size
    scale = args.crop_size / min(orig_h, orig_w)
    res_w = max(args.crop_size, int(round(orig_w * scale)))
    res_h = max(args.crop_size, int(round(orig_h * scale)))
    resized_img = orig_img.resize((res_w, res_h), Image.Resampling.BICUBIC)

    crop_y = (res_h - args.crop_size) // 3
    crop_x = (res_w - args.crop_size) // 3
    cropped_img = resized_img.crop(
        (crop_x, crop_y, crop_x + args.crop_size, crop_y + args.crop_size)
    )

    # 2. Compute Coordinate Grids
    coords_discrete = get_discrete_coords(h_patches, w_patches)
    coords_broken_raw = get_broken_continuous_coords(
        h_patches, w_patches, crop_y, crop_x, res_h, res_w, patch_px, False
    )
    coords_broken_long = get_broken_continuous_coords(
        h_patches, w_patches, crop_y, crop_x, res_h, res_w, patch_px, True
    )
    coords_corrected = get_corrected_continuous_coords(
        h_patches, w_patches, crop_y, crop_x, patch_px
    )

    # 3. Compute RoPE Frequencies & Relative Rotary Kernels
    inv_freqs = compute_rope_inv_freqs(dim=16, theta=10000.0)
    probe_idx = (h_patches // 2) * w_patches + (w_patches // 2)

    kernel_disc = compute_relative_kernel(coords_discrete, inv_freqs, probe_idx).view(
        h_patches, w_patches
    )
    kernel_broken = compute_relative_kernel(
        coords_broken_raw, inv_freqs, probe_idx
    ).view(h_patches, w_patches)
    kernel_corr = compute_relative_kernel(coords_corrected, inv_freqs, probe_idx).view(
        h_patches, w_patches
    )

    # 4. Print Summary Table
    delta_discrete = 1.0
    delta_broken = (patch_px / res_h) * (2.0 * math.sqrt(res_h / res_w))
    delta_corrected = 1.0
    print_structured_summary(inv_freqs, delta_discrete, delta_broken, delta_corrected)

    # 5. Generate Multi-Panel Comparison Plot
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    fig.suptitle(
        "RoPE Coordinate & Phase Collapse Analysis",
        fontsize=16,
        fontweight="bold",
    )

    # Row 0: Image & Crop Context
    axes[0, 0].imshow(resized_img)
    axes[0, 0].set_title(f"Resized Source ({res_w}x{res_h})")
    rect = plt.Rectangle(
        (crop_x, crop_y),
        args.crop_size,
        args.crop_size,
        edgecolor="red",
        facecolor="none",
        lw=2,
    )
    axes[0, 0].add_patch(rect)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cropped_img)
    axes[0, 1].set_title(f"Crop Window ({args.crop_size}x{args.crop_size})")
    axes[0, 1].axis("off")

    # Row 0: Channel 0 Phase Shifts across adjacent patches
    channels = np.arange(len(inv_freqs))
    deg_disc = [math.degrees(delta_discrete * f) for f in inv_freqs]
    deg_broken = [math.degrees(delta_broken * f) for f in inv_freqs]
    deg_corr = [math.degrees(delta_corrected * f) for f in inv_freqs]

    axes[0, 2].plot(channels, deg_disc, "g-o", label="Discrete (Ground Truth)")
    axes[0, 2].plot(channels, deg_broken, "r-x", label="Broken Normalized")
    axes[0, 2].plot(channels, deg_corr, "b--^", label="Corrected Patch-Unit")
    axes[0, 2].set_xlabel("Frequency Channel k (dim=16)")
    axes[0, 2].set_ylabel("Phase Step (deg)")
    axes[0, 2].set_title("Adjacent Patch Phase Shift (Delta Theta)")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, linestyle="--", alpha=0.6)

    # Row 0: Coordinate Spread along Center Row
    row_start = (h_patches // 2) * w_patches
    row_end = row_start + w_patches
    axes[0, 3].plot(
        coords_discrete[row_start:row_end, 1].numpy(),
        "g-o",
        label="Discrete [0, 15]",
    )
    axes[0, 3].plot(
        coords_corrected[row_start:row_end, 1].numpy(),
        "b--^",
        label="Corrected Float",
    )
    axes[0, 3].plot(
        coords_broken_raw[row_start:row_end, 1].numpy(),
        "r-x",
        label="Broken [-1, 1]",
    )
    axes[0, 3].set_xlabel("Patch Index X")
    axes[0, 3].set_ylabel("Coordinate Value")
    axes[0, 3].set_title("X-Coordinates of Center Patch Row")
    axes[0, 3].legend(fontsize=8)
    axes[0, 3].grid(True, linestyle="--", alpha=0.6)

    # Row 1: 2D Position Grid Scatters
    scatter_data = [
        (coords_discrete, "1. Discrete Integer Grid", "Greens"),
        (coords_broken_raw, "2. Broken Normalized [-r_h, r_h]", "Reds"),
        (coords_broken_long, "2b. Broken with .long() Cast", "Oranges"),
        (coords_corrected, "3. Corrected Patch-Unit Grid", "Blues"),
    ]
    for col_idx, (coord_tensor, title, cmap_name) in enumerate(scatter_data):
        ax = axes[1, col_idx]
        y_pts = coord_tensor[:, 0].numpy()
        x_pts = coord_tensor[:, 1].numpy()
        sc = ax.scatter(x_pts, y_pts, c=np.arange(n_patches), cmap=cmap_name, s=35)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("X Coord")
        ax.set_ylabel("Y Coord")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.invert_yaxis()

    # Row 2: Relative RoPE Attention Kernel Heatmaps
    heatmap_data = [
        (kernel_disc.numpy(), "1. Discrete (Sharp Distance Decay)"),
        (kernel_broken.numpy(), "2. Broken (Phase Collapse: Flat ~1.0)"),
        (
            compute_relative_kernel(coords_broken_long, inv_freqs, probe_idx)
            .view(h_patches, w_patches)
            .numpy(),
            "2b. Broken .long() (Quantization Artifacts)",
        ),
        (kernel_corr.numpy(), "3. Corrected (Preserved Radial Decay)"),
    ]
    for col_idx, (k_map, title) in enumerate(heatmap_data):
        ax = axes[2, col_idx]
        im = ax.imshow(k_map, cmap="inferno", vmin=-0.2, vmax=1.0)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.plot(w_patches // 2, h_patches // 2, "c*", markersize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(args.output_dir, "rope_debug_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Visualization successfully saved to: {save_path}")


if __name__ == "__main__":
    main()
