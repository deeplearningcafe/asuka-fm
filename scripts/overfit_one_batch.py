import os
import math
import random
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.utils.logging_utils import Logger

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf, DictConfig

from src.models.factory import (
    load_trainable_model,
    create_optim,
    create_scheduler,
)
from src.diffusion.schedules import LinearSchedule, DDPMSchedule
from src.diffusion.objectives import (
    FlowMatchingObjective,
    DDPMObjective,
)
from src.diffusion.sampling import generate_samples
from src.data.loader import create_dataloader


def save_pil_grid(
    pil_images: List[Image.Image],
    save_path: str,
    nrow: int = 4,
) -> None:
    """Saves a list of PIL Images as a stitched grid image."""
    if not pil_images:
        return
    n = len(pil_images)
    ncols = min(nrow, n)
    nrows = (n + ncols - 1) // ncols
    w, h = pil_images[0].size
    grid = Image.new("RGB", (ncols * w, nrows * h))
    for idx, img in enumerate(pil_images):
        x = (idx % ncols) * w
        y = (idx // ncols) * h
        grid.paste(img, (x, y))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    grid.save(save_path)


def decode_latents_to_pil(
    latents: torch.Tensor,
    vae: torch.nn.Module,
    device: torch.device,
    vae_mean: Any = 0.0,
    vae_std: Any = 1.0 / 0.18215,
    vae_batch_size: int = 4,
) -> List[Image.Image]:
    """Decodes latent representations to PIL images with batch chunking."""
    images = []
    vae.eval()
    vae_dtype = next(vae.parameters()).dtype
    with torch.no_grad():
        # Inverse transform: z_raw = (z_norm * std) + mean
        scaled = (latents.to(device) * vae_std) + vae_mean
        for i in range(0, scaled.shape[0], vae_batch_size):
            chunk = scaled[i : i + vae_batch_size].to(
                device=device, dtype=vae_dtype
            )
            out = vae.decode(chunk)
            sample = out.sample if hasattr(out, "sample") else out
            sample = (sample.to(torch.float32) / 2.0 + 0.5).clamp(0, 1)
            img_np = sample.cpu().permute(0, 2, 3, 1).numpy()
            for j in range(img_np.shape[0]):
                img_uint8 = (img_np[j] * 255.0).round().astype(np.uint8)
                images.append(Image.fromarray(img_uint8))
    return images


def encode_vae_latents(
    images: torch.Tensor,
    vae: torch.nn.Module,
    device: torch.device,
    autocast_dtype: torch.dtype,
    vae_mean: torch.Tensor,
    vae_std: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Encodes raw image batch to latents using dynamic chunking."""
    latents = []
    vae_dtype = next(vae.parameters()).dtype
    for i in range(0, images.shape[0], chunk_size):
        chunk = images[i : i + chunk_size].to(
            device=device, dtype=vae_dtype, non_blocking=True
        )
        with torch.autocast(
            device_type="cuda", dtype=autocast_dtype, enabled=True
        ):
            enc = vae.encode(chunk)
            dist = getattr(enc, "latent_dist", enc)
            lat = dist.sample() if hasattr(dist, "sample") else dist
            latents.append(lat)
    lat = torch.cat(latents, dim=0)
    # Forward transform: z_norm = (z_raw - mean) / std
    return (lat - vae_mean) / vae_std


def run_overfit_experiment(cfg: DictConfig, args: argparse.Namespace) -> None:
    """Executes single-batch overfitting optimization and sampling."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # Keep master weights in FP32

    model_type = getattr(cfg.models, "model_type", "unet")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("results", "overfit", timestamp)
    Logger.setup_logging(
        save_dir=save_dir,
        logging_name=f"{model_type}_loss_{cfg.train['objective']}",
    )
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)

    logging.info(f"Initializing model architecture: {model_type}")
    logging.info(cfg)

    # Hardware detection and TF32/autocast configuration
    autocast_dtype = torch.bfloat16
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 7 and capability[0] < 8:
            autocast_dtype = torch.float16
            torch.set_float32_matmul_precision("high")
            logging.info(
                "Using high precision for float32 matmul (Volta/Turing)."
            )
        elif capability[0] >= 8:
            torch.set_float32_matmul_precision("medium")
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                True
            )
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
                True
            )
            logging.info("Using TF32 and tensor cores (Ampere+).")
        else:
            logging.info("Legacy GPU detected; standard precision used.")

    logging.info(f"Using {autocast_dtype} for autocast mixed precision")


    unet, text_encoder, vae, tokenizer, ema = load_trainable_model(
        models_path=cfg.paths.models,
        device=device,
        dtype=dtype,
        train_te=cfg.train.train_te,
        use_checkpointing=cfg.train.use_checkpointing,
        resume_from_checkpoint=None,
        train_only_output=cfg.train.train_only_output,
        output_head_path=None,
        use_ema=False,
        global_rank=0,
        model_type=model_type,
        model_cfg=cfg.models,
        autocast_dtype=autocast_dtype,
    )

    # Configure schedule and loss objective
    is_dit = model_type in ["dual_stream", "sprint_dual"]
    if cfg.train.objective == "flow_matching":
        schedule = LinearSchedule(device=device)
        objective = FlowMatchingObjective(
            schedule=schedule,
            timestep_sampling=cfg.train.get("timestep_fn", "uniform"),
            shift=cfg.train.shift,
            use_ot=cfg.train.get("use_ot", False),
            use_unet_mult=False if is_dit else True,
        )
    else:
        schedule = DDPMSchedule(device=device)
        objective = DDPMObjective(
            schedule=schedule, min_snr_gamma=cfg.train.snr_gamma
        )

    is_latent = (
            cfg.data.get("is_latent", False)
            or getattr(cfg.data, "dataset_type", "") == "streaming_latents"
            or cfg.data.get("cache_latents_to_ram", False)
        )

    cfg.train.cfg_dropout_prob = 0.0
    cfg.train.tag_dropout = 0.0
    dataloader = create_dataloader(cfg, rank=0, tokenizer=tokenizer, vae=vae, autocast_dtype=autocast_dtype,)
    raw_batch = next(iter(dataloader))

    # Parse and cache single batch on GPU
    vae_mean = getattr(cfg.models, "vae_mean", 0.0)
    vae_std = getattr(cfg.models, "vae_std", 1.0 / 0.18215)
    vae_mean_t = torch.tensor(
        vae_mean, device=device, dtype=dtype
    ).view(1, -1, 1, 1)
    vae_std_t = torch.tensor(
        vae_std, device=device, dtype=dtype
    ).view(1, -1, 1, 1)
    vae_batch_size = cfg.train.get(
        "vae_batch_size", cfg.models.get("vae_batch_size", 64)
    )

    pos_map = None
    if len(raw_batch) >= 5:
        images_or_lats, cond, mask, pos_map, tag_weights, *rest = raw_batch
        if not is_latent:
            with torch.no_grad():
                with torch.autocast(
                    device_type="cuda", dtype=autocast_dtype, enabled=True
                ):
                    latents = encode_vae_latents(
                        images=images_or_lats,
                        vae=vae,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        vae_mean=vae_mean_t,
                        vae_std=vae_std_t,
                        chunk_size=vae_batch_size,
                    )
        else:
            # Precomputed latents (streaming shards or RAM buffer)
            latents = images_or_lats.to(
                device, dtype=dtype, non_blocking=True
            )
        torch.cuda.empty_cache()
        attention_mask = mask.to(device)
        pos_map = pos_map.to(device, dtype=dtype)
        tag_weights = tag_weights.to(device, dtype=dtype)
    else:
        latents = raw_batch[0].to(device, dtype=dtype) * 0.18215
        cond = raw_batch[1].to(device)
        tag_weights = raw_batch[2].to(device, dtype=dtype)
        attention_mask = raw_batch[3].to(device)

    bsz = latents.shape[0]
    logging.info(
        f"Overfitting batch size: {bsz}, Latents shape: {latents.shape}"
    )

    gt_images = decode_latents_to_pil(
        latents[:16],
        vae,
        device,
        vae_mean=vae_mean,
        vae_std=vae_std,
        vae_batch_size=min(4, vae_batch_size),
    )
    save_pil_grid(gt_images, os.path.join(save_dir, "ground_truth.png"))
    in_channels = cfg.models.get("in_channels", 4)

    cfg.train.epochs = 1
    cfg.train.warmup = 0.005
    optimizer = create_optim(unet, text_encoder, cfg)
    lr_scheduler = create_scheduler(optimizer, dataloader, cfg)
    scaler = (
        torch.cuda.amp.GradScaler()
        if autocast_dtype == torch.float16
        else None
    )

    # Setup sampling prompts from the batch
    sample_configs = []
    for i in range(latents.shape[0]):
        cfg_dict = {
            "prompt": tokenizer.decode(cond[i]),
            "negative_prompt": "",
            "height": latents.shape[-2] * 8,
            "width": latents.shape[-1] * 8,
            "sample_steps": cfg.sampling.get("steps", 25),
            "cfg_scale": 1.0,
            "seed": cfg.train.seed + i,
            "shift": cfg.train.shift,
        }
        if pos_map is not None:
            cfg_dict["pos_map"] = pos_map[i : i + 1]
        logging.info(cfg_dict["prompt"])
        sample_configs.append(cfg_dict)

    num_steps = args.steps
    sample_interval = args.sample_interval
    debug_params = args.debug_params

    logging.info(f"Beginning overfit loop for {num_steps} iterations...")
    unet.train()
    with torch.autocast(
        device_type="cuda", dtype=autocast_dtype, enabled=True
    ):
        with torch.set_grad_enabled(cfg.train.train_te):
            encoder_hidden_states, attention_mask = text_encoder(
                cond, mask=attention_mask, drop_mask=None
            )

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda", dtype=autocast_dtype, enabled=True
        ):
            loss, metrics = objective.forward(
                unet,
                latents,
                encoder_hidden_states,
                tag_weights,
                attention_mask=attention_mask,
                pos_map=pos_map,
            )

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if debug_params:
            for name, param in unet.named_parameters():
                if param.requires_grad and param.grad is None:
                    print(f"[UNUSED PARAMETER] {name} | shape: {param.shape}")

        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        if (step + 1) % 50 == 0 or step == 0:
            logging.info(
                f"[Step {step + 1:05d}/{num_steps:05d}] "
                f"Loss: {loss.item():.6f} | "
                f"Pred Norm: {metrics['pred_norm'].item():.3f} | "
                f"Target Norm: {metrics['target_norm'].item():.3f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        if (step + 1) % sample_interval == 0 or (step + 1) == num_steps:
            unet.eval()
            text_encoder.eval()

            logging.info(f"Generating samples at step {step + 1}...")

            with torch.no_grad():
                samples = generate_samples(
                    unet=unet,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    vae=vae,
                    schedule=schedule,
                    sample_configs=sample_configs,
                    global_batch_size=min(4, len(sample_configs)),
                    diffusion_type=cfg.train.objective,
                    device=str(device),
                    dtype=dtype,
                    autocast_dtype=autocast_dtype,
                    use_unet_mult=False if is_dit else True,
                    vae_mean=vae_mean_t,
                    vae_std=vae_std_t,
                    in_channels=in_channels
                )

            step_path = os.path.join(save_dir, f"sample_step_{step + 1:05d}.png")
            save_pil_grid(samples, step_path)
            unet.train()
            if cfg.train.train_te:
                text_encoder.train()

    logging.info(f"Overfit batch complete. Visualizations saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Overfit single batch")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--steps", type=int, default=3000, help="Number of optimization steps"
    )
    parser.add_argument(
        "--sample_interval", type=int, default=250, help="Sampling interval"
    )
    parser.add_argument("--debug_params", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    base_cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_cli(args.opts)
    cfg = OmegaConf.merge(base_cfg, cli_cfg)

    run_overfit_experiment(cfg, args)


if __name__ == "__main__":
    main()
