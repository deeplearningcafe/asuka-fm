import sys
import os
import argparse
import torch
import toml
from datetime import datetime
from omegaconf import OmegaConf
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.factory import load_trainable_model
from src.diffusion.schedules import LinearSchedule, DDPMSchedule
from src.diffusion.sampling import generate_samples
from src.utils.logging_utils import Logger


def load_toml_configs(toml_path):
    """Parses the TOML file into a list of config dictionaries."""
    with open(toml_path, "r", encoding="utf-8") as f:
        data = toml.load(f)

    defaults = data.get("prompt", {})
    subsets = defaults.pop("subset", [])

    configs = []
    for sub in subsets:
        c = defaults.copy()
        c.update(sub)
        configs.append(c)
    return configs


def main():
    parser = argparse.ArgumentParser(description="Sample from Flow Matching/DDPM Model")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the main config YAML (for model paths/types)",
    )
    parser.add_argument(
        "--sample_config",
        type=str,
        default="sample.toml",
        help="Path to the sampling TOML configuration",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to a specific checkpoint to load (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/sampling",
        help="Root directory for saving samples",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    cfg = OmegaConf.load(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg.train.dtype == "bf16" else torch.float32
    print(f"Using device: {device}, Precision: {dtype}")

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


    checkpoint_path = (
        args.checkpoint if args.checkpoint else cfg.models.resume_from_checkpoint
    )

    print(f"Loading models from: {cfg.paths.models}")
    if checkpoint_path:
        print(f"Resuming weights from: {checkpoint_path}")

    model_type = getattr(cfg.models, "model_type", "unet")
    unet, text_encoder, vae, tokenizer, _ = load_trainable_model(
        models_path=cfg.paths.models,
        device=device,
        dtype=dtype,
        train_te=False,
        use_checkpointing=False,
        resume_from_checkpoint=checkpoint_path,
        train_only_output=False,
        global_rank=0,
        model_type=model_type,
        model_cfg=cfg.models,
        autocast_dtype=autocast_dtype,
    )
    in_channels = cfg.models.get("in_channels", 4)
    vae_mean = getattr(cfg.models, "vae_mean", 0.0)
    vae_std = getattr(cfg.models, "vae_std", 1.0 / 0.18215)
    vae_mean = torch.tensor(
        vae_mean, device=device, dtype=dtype
    ).view(1, -1, 1, 1)
    vae_std = torch.tensor(vae_std, device=device, dtype=dtype).view(
        1, -1, 1, 1
    )
    is_dit = model_type in ["dual_stream", "sprint_dual"]

    # Ensure everything is in Eval mode and gradients are disabled
    unet.eval()
    text_encoder.eval()
    vae.eval()
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    # 4. Setup Diffusion Schedule
    if cfg.train.objective == "flow_matching":
        print(f"Using Flow Matching Schedule (Shift: {cfg.train.shift})")
        schedule = LinearSchedule(device=device)
    else:
        print("Using DDPM Schedule")
        schedule = DDPMSchedule(device=device)

    if not os.path.exists(args.sample_config):
        raise FileNotFoundError(f"Sample config not found: {args.sample_config}")
    sample_configs = load_toml_configs(args.sample_config)
    print(f"Loaded {len(sample_configs)} prompts to generate.")

    images = generate_samples(
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        schedule=schedule,
        sample_configs=sample_configs,
        global_batch_size=sample_configs[0].get("batch_size", 4),
        diffusion_type=cfg.train.objective,
        device=device,
        dtype=dtype,
        use_unet_mult=False if is_dit else True,
        vae_mean=vae_mean,
        vae_std=vae_std,
        in_channels=in_channels
    )

    # Create datetime folder: sampling/YYYY-MM-DD-HH-MM
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    save_path = os.path.join(args.output_dir, timestamp)
    os.makedirs(save_path, exist_ok=True)

    print(f"Saving images to {save_path}...")
    for i, img in enumerate(images):
        prompt_slug = sample_configs[i].get("prompt", "sample")[:30].replace(" ", "_")
        filename = f"{i:03d}.png"
        img.save(os.path.join(save_path, filename))

    print("Sampling complete.")


if __name__ == "__main__":
    main()
