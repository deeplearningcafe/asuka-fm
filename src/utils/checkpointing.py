import json
import logging
import os
from typing import Any, Dict, Optional, Union
from safetensors.torch import save_file
import torch
from omegaconf import DictConfig, OmegaConf

import src.utils.logging as logging_utils


def save_checkpoint(
    epoch: int,
    global_step: int,
    unet: torch.nn.Module,
    text_encoder: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    train_te: bool = False,
    hf_repo: Optional[str] = None,
    base_dir: str = ".",
    train_only_output: bool = False,
    ema: Any = None,
    config: Optional[Union[Dict[str, Any], Any]] = None,
) -> None:
    """Saves training checkpoint in bf16 with architecture configuration.

    Weights are cast to bfloat16 to optimize storage and I/O throughput.
    """
    save_dir = os.path.join(base_dir, f"epoch_{epoch}_step_{global_step}")
    checkpoint_dir = os.path.join(
        save_dir, f"epoch_{epoch}_step_{global_step}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    logging.info(f"Saving checkpoint to {checkpoint_dir}...")

    # 1. Clean compilation prefixes and cast floating point weights to bf16
    unet_state_dict = unet.state_dict()
    clean_unet_dict = {
        k.replace("_orig_mod.", ""): (
            v.to(torch.bfloat16) if v.is_floating_point() else v
        )
        for k, v in unet_state_dict.items()
    }

    if train_only_output:
        logging.info("Filtering state dict: Saving only output head.")
        keys_to_save = [
            k
            for k in clean_unet_dict.keys()
            if "conv_out" in k or "conv_norm_out" in k or "final_layer" in k
        ]
        clean_unet_dict = {k: clean_unet_dict[k] for k in keys_to_save}

    save_file(
        clean_unet_dict, os.path.join(checkpoint_dir, "unet.safetensors")
    )

    # 2. Save EMA model weights in bf16 if active
    if ema is not None and getattr(ema, "use_ema", False):
        logging.info("Saving EMA weights in bfloat16...")
        ema_state = (
            ema.ema_model.state_dict()
            if hasattr(ema, "ema_model") and ema.ema_model is not None
            else ema.state_dict()
        )
        clean_ema_dict = {
            k.replace("_orig_mod.", ""): (
                v.to(torch.bfloat16) if v.is_floating_point() else v
            )
            for k, v in ema_state.items()
        }
        if train_only_output:
            keys_to_save = [
                k
                for k in clean_ema_dict.keys()
                if "conv_out" in k
                or "conv_norm_out" in k
                or "final_layer" in k
            ]
            clean_ema_dict = {k: clean_ema_dict[k] for k in keys_to_save}
        save_file(
            clean_ema_dict,
            os.path.join(checkpoint_dir, "unet_ema.safetensors"),
        )

    # 3. Save text encoder if trained
    if train_te and text_encoder is not None:
        te_state = text_encoder.state_dict()
        clean_te_dict = {
            k.replace("_orig_mod.", ""): (
                v.to(torch.bfloat16) if v.is_floating_point() else v
            )
            for k, v in te_state.items()
        }
        save_file(
            clean_te_dict,
            os.path.join(checkpoint_dir, "text_encoder.safetensors"),
        )

    # 4. Save optimizer and scheduler states
    if optimizer is not None:
        torch.save(
            optimizer.state_dict(),
            os.path.join(checkpoint_dir, "optimizer.pt"),
        )
    if scheduler is not None:
        torch.save(
            scheduler.state_dict(),
            os.path.join(checkpoint_dir, "scheduler.pt"),
        )

    # 5. Save training metadata
    training_state = {
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(
        training_state,
        os.path.join(checkpoint_dir, "training_state.pt"),
    )

    # 6. Save HuggingFace-style model configuration JSON
    if config is not None:
        if OmegaConf is not None and isinstance(config, DictConfig):
            cfg_dict = OmegaConf.to_container(config, resolve=True)
        elif isinstance(config, dict):
            cfg_dict = config
        else:
            cfg_dict = dict(config)

        models_cfg = (
            cfg_dict.get("models", {})
            if "models" in cfg_dict
            else cfg_dict
        )

        hf_config = {
            "_class_name": models_cfg.get("model_type", "dual_stream"),
            "model_type": models_cfg.get("model_type", "dual_stream"),
            "in_channels": models_cfg.get("in_channels", 4),
            "hidden_size": models_cfg.get("hidden_size", 768),
            "depth": models_cfg.get("depth", 13),
            "num_heads": models_cfg.get("num_heads", 12),
            "encoder_depth": models_cfg.get("encoder_depth", 2),
            "decoder_depth": models_cfg.get("decoder_depth", 2),
            "drop_ratio": models_cfg.get("drop_ratio", 0.0),
            "drop_target": models_cfg.get("drop_target", "image"),
            "residual_type": models_cfg.get(
                "residual_type", "concat_linear"
            ),
            "hf_text_encoder": models_cfg.get("hf_text_encoder", ""),
            "hf_vae": models_cfg.get("hf_vae", ""),
            "vae_mean": models_cfg.get("vae_mean", 0.0),
            "vae_std": models_cfg.get("vae_std", 1.0 / 0.18215),
        }

        # Include remaining primitive metadata
        for section in [models_cfg, cfg_dict]:
            for k, v in section.items():
                if k not in hf_config and isinstance(
                    v, (int, float, str, bool, list, dict)
                ):
                    hf_config[k] = v

        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(hf_config, f, indent=2)

    logging.info("Checkpoint saved successfully.")

    if logging_utils.is_hfapi_initialized() and hf_repo:
        logging.info(
            f"Uploading checkpoint to Hugging Face repo: {hf_repo}"
        )
        logging_utils.log_folder(save_dir, hf_repo)
        logging.info("Upload complete.")


def load_checkpoint_config(checkpoint_dir: str) -> dict:
    """Loads config.json from checkpoint directory if present."""
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}