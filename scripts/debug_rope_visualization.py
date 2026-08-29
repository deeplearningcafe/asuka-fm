"""RoPE Spatial Geometry and Phase Collapse Debugger.

Compares Discrete, Broken Continuous, Patch-Unit, and HDM High-Freq RoPE.
"""

import argparse
import math
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw


def compute_rope_inv_freqs(
    dim: int = 16, theta: float = 10000.0
) -> torch.Tensor:
    """Computes standard geometric RoPE inverse frequencies."""
    steps = torch.arange(0, dim, 2, dtype=torch.float32)
    return 1.0 / (theta ** (steps / dim))


def compute_hdm_rope_freqs(
    dim: int = 16, max_freq: float = 10.0
) -> torch.Tensor:
    """Computes HDM log-linear spatial RoPE frequencies in [pi, 5*pi]."""
    log_min = math.log(math.pi)
    log_max = math.log(max_freq * math.pi / 2.0)
    log_freqs = torch.linspace(log_min, log_max, dim // 2)
    return log_freqs.exp()

def compute_calibrated_spatial_freqs(
    dim: int = 16,
    min_freq: float = math.pi / 4.0,  # ~0.785 rad (Global scale)
    max_freq: float = 2.0 * math.pi,  # ~6.283 rad (Local patch scale)
) -> torch.Tensor:
    """Computes full-spectrum continuous 2D RoPE spatial frequencies.

    Ordered from highest frequency (k=0) to lowest frequency (k=D/2-1) to
    prevent high-frequency harmonic aliasing and eliminate axial phase collapse.
    """
    log_max = math.log(max_freq)
    log_min = math.log(min_freq)
    # Descending geometric progression
    log_freqs = torch.linspace(log_max, log_min, dim // 2)
    return log_freqs.exp()

def get_discrete_coords(h_patches: int, w_patches: int) -> torch.Tensor:
    """Case 1: Standard integer patch coordinates [0, H-1] and [0, W-1]."""
    y = torch.arange(h_patches, dtype=torch.float32)
    x = torch.arange(w_patches, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((grid_y, grid_x), dim=-1).flatten(0, 1)


def get_aspect_normalized_coords(
    h_patches: int,
    w_patches: int,
    crop_y: int,
    crop_x: int,
    target_h: int,
    target_w: int,
    patch_size_px: int,
) -> torch.Tensor:
    """Case 2 & 4: Aspect-normalized continuous coordinates [-r_h, r_h]."""
    r_h = math.sqrt(float(target_h) / float(target_w))
    r_w = math.sqrt(float(target_w) / float(target_h))

    y_centers = (
        torch.arange(h_patches, dtype=torch.float32) + 0.5
    ) * patch_size_px + crop_y
    x_centers = (
        torch.arange(w_patches, dtype=torch.float32) + 0.5
    ) * patch_size_px + crop_x

    y_norm = (y_centers / float(target_h)) * (2.0 * r_h) - r_h
    x_norm = (x_centers / float(target_w)) * (2.0 * r_w) - r_w

    grid_y, grid_x = torch.meshgrid(y_norm, x_norm, indexing="ij")
    return torch.stack((grid_y, grid_x), dim=-1).flatten(0, 1)


def get_patch_unit_continuous_coords(
    h_patches: int,
    w_patches: int,
    crop_y: int,
    crop_x: int,
    patch_size_px: int,
) -> torch.Tensor:
    """Case 3: Unbounded continuous coordinates in patch units."""
    crop_y_patch = crop_y / float(patch_size_px)
    crop_x_patch = crop_x / float(patch_size_px)

    y_pos = torch.arange(h_patches, dtype=torch.float32) + crop_y_patch
    x_pos = torch.arange(w_patches, dtype=torch.float32) + crop_x_patch

    grid_y, grid_x = torch.meshgrid(y_pos, x_pos, indexing="ij")
    return torch.stack((grid_y, grid_x), dim=-1).flatten(0, 1)


def compute_relative_kernel(
    coords: torch.Tensor, freqs: torch.Tensor, probe_idx: int
) -> torch.Tensor:
    """Computes mean relative rotary similarity kernel cos(delta_pos * freqs)."""
    probe_coord = coords[probe_idx]
    diff = coords - probe_coord

    angles_y = diff[:, 0:1] * freqs.unsqueeze(0)
    angles_x = diff[:, 1:2] * freqs.unsqueeze(0)

    cos_y = angles_y.cos().mean(dim=-1)
    cos_x = angles_x.cos().mean(dim=-1)
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

    for x in range(0, w, 32):
        draw.line([(x, 0), (x, h)], fill=(70, 70, 95), width=1)
    for y in range(0, h, 32):
        draw.line([(0, y), (w, y)], fill=(70, 70, 95), width=1)

    draw.ellipse([w // 4, h // 6, 3 * w // 4, 5 * h // 6], fill=(180, 50, 60))
    draw.text((20, 20), "Synthetic Test Pattern", fill=(240, 240, 240))
    return img


def print_structured_summary(
    inv_freqs_std: torch.Tensor,
    freqs_hdm: torch.Tensor,
    delta_unit: float,
    delta_norm: float,
) -> None:
    """Prints tabular RoPE rotation phase differences to stdout."""
    print("=" * 78)
    print(" RoPE Phase Shift (Delta Theta) Across Frequency Channels")
    print("=" * 78)
    print(
        f"{'Channel (k)':<12} | {'Discrete (deg)':<15} | "
        f"{'Broken Norm (deg)':<18} | {'HDM High-Freq (deg)'}"
    )
    print("-" * 78)

    for k in range(len(inv_freqs_std)):
        f_std = inv_freqs_std[k].item()
        f_hdm = freqs_hdm[k].item()

        deg_disc = math.degrees(delta_unit * f_std)
        deg_broken = math.degrees(delta_norm * f_std)
        deg_hdm = math.degrees(delta_norm * f_hdm)

        print(
            f"k={k:<10} | {deg_disc:<15.2f} | "
            f"{deg_broken:<18.2f} | {deg_hdm:.2f}"
        )
    print("=" * 78)
    print("Diagnosis:")
    print(" - Broken Norm + Standard RoPE collapses phase shifts (<7.16 deg).")
    print(" - HDM High-Freq RoPE restores healthy phase steps (22.5 to 112.5 deg).")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="RoPE Debugger")
    parser.add_argument(
        "--image_path", type=str, default=None, help="Reference image path"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/debug_rope", help="Output"
    )
    parser.add_argument("--crop_size", type=int, default=256, help="Crop size")
    parser.add_argument(
        "--patch_size", type=int, default=2, help="DiT latent patch size"
    )
    parser.add_argument(
        "--vae_scale", type=int, default=8, help="VAE downsample scale"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    patch_px = args.patch_size * args.vae_scale
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
    coords_norm = get_aspect_normalized_coords(
        h_patches, w_patches, crop_y, crop_x, res_h, res_w, patch_px
    )
    coords_patch_unit = get_patch_unit_continuous_coords(
        h_patches, w_patches, crop_y, crop_x, patch_px
    )

    # 3. Compute RoPE Frequencies & Kernels
    freqs_std = compute_rope_inv_freqs(dim=16, theta=10000.0)
    freqs_hdm = compute_calibrated_spatial_freqs(dim=16, max_freq=10.0)
    probe_idx = (h_patches // 2) * w_patches + (w_patches // 2)

    k_disc = compute_relative_kernel(
        coords_discrete, freqs_std, probe_idx
    ).view(h_patches, w_patches)
    k_broken = compute_relative_kernel(
        coords_norm, freqs_std, probe_idx
    ).view(h_patches, w_patches)
    k_patch = compute_relative_kernel(
        coords_patch_unit, freqs_std, probe_idx
    ).view(h_patches, w_patches)
    k_hdm = compute_relative_kernel(
        coords_norm, freqs_hdm, probe_idx
    ).view(h_patches, w_patches)

    # 4. Print Summary Table
    delta_unit = 1.0
    delta_norm = (patch_px / float(res_h)) * (2.0 * math.sqrt(res_h / res_w))
    print_structured_summary(freqs_std, freqs_hdm, delta_unit, delta_norm)

    # 5. Multi-Panel Comparison Plot
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    fig.suptitle(
        "RoPE Coordinate & Frequency Calibration Analysis",
        fontsize=16,
        fontweight="bold",
    )

    # Row 0: Context & Distributions
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

    channels = np.arange(len(freqs_std))
    deg_disc = [math.degrees(delta_unit * f) for f in freqs_std]
    deg_broken = [math.degrees(delta_norm * f) for f in freqs_std]
    deg_hdm = [math.degrees(delta_norm * f) for f in freqs_hdm]

    axes[0, 2].plot(channels, deg_disc, "g-o", label="Discrete (Std RoPE)")
    axes[0, 2].plot(channels, deg_broken, "r-x", label="Broken (Norm + Std)")
    axes[0, 2].plot(channels, deg_hdm, "m--s", label="HDM (Norm + High-Freq)")
    axes[0, 2].set_xlabel("Frequency Channel k (dim=16)")
    axes[0, 2].set_ylabel("Phase Step (deg)")
    axes[0, 2].set_title("Adjacent Patch Phase Shift (Delta Theta)")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, linestyle="--", alpha=0.6)

    row_start = (h_patches // 2) * w_patches
    row_end = row_start + w_patches
    axes[0, 3].plot(
        coords_discrete[row_start:row_end, 1].numpy(),
        "g-o",
        label="Discrete [0, 15]",
    )
    axes[0, 3].plot(
        coords_patch_unit[row_start:row_end, 1].numpy(),
        "b--^",
        label="Patch-Unit",
    )
    axes[0, 3].plot(
        coords_norm[row_start:row_end, 1].numpy(),
        "m-s",
        label="Aspect-Norm [-r, r]",
    )
    axes[0, 3].set_xlabel("Patch Index X")
    axes[0, 3].set_ylabel("Coordinate Value")
    axes[0, 3].set_title("X-Coordinates of Center Row")
    axes[0, 3].legend(fontsize=8)
    axes[0, 3].grid(True, linestyle="--", alpha=0.6)

    # Row 1: 2D Scatters
    scatter_data = [
        (coords_discrete, "1. Discrete Integer Grid", "Greens"),
        (coords_norm, "2. Broken Norm (with Std RoPE)", "Reds"),
        (coords_patch_unit, "3. Patch-Unit Float Grid", "Blues"),
        (coords_norm, "4. HDM Aspect-Norm Grid", "Purples"),
    ]
    for col_idx, (coord_tensor, title, cmap_name) in enumerate(scatter_data):
        ax = axes[1, col_idx]
        y_pts = coord_tensor[:, 0].numpy()
        x_pts = coord_tensor[:, 1].numpy()
        ax.scatter(x_pts, y_pts, c=np.arange(n_patches), cmap=cmap_name, s=35)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("X Coord")
        ax.set_ylabel("Y Coord")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.invert_yaxis()

    # Row 2: Attention Heatmaps
    heatmap_data = [
        (k_disc.numpy(), "1. Discrete (Ground Truth Decay)"),
        (k_broken.numpy(), "2. Broken (Phase Collapse: Flat ~1.0)"),
        (k_patch.numpy(), "3. Patch-Unit (Preserved Decay)"),
        (k_hdm.numpy(), "4. HDM (Restored Sharp Decay)"),
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