import json
import logging
import os
from typing import Any, Dict, Optional, Union
from safetensors.torch import save_file, load_file
import torch
from omegaconf import DictConfig, OmegaConf
import math

import src.utils.logging as logging_utils

PIXEL_EXPLICIT_MODULES = {
    "x_embedder",
    "conv_in",
    "pixel_decoder",
    "proj_out",
    "conv_out",
    "norm_final",
    "norm_out",
    "conv_norm_out",
    "final_layer",
}


def _normalize_param_key(k: str) -> str:
    """Removes torch.compile wrappers and module prefixes for matching."""
    k = k.replace("_orig_mod.", "")
    if k.startswith("unet."):
        k = k[5:]
    return k


def adapt_patch_embed_weight(
    src_weight: torch.Tensor,
    target_shape: tuple[int, ...],
) -> torch.Tensor:
    """
    Adapts patch embedding weights when expanding patch size (e.g. ps16->ps32).
    Follows Jiang et al. (2026): W' = (1/k) * [W, ..., W], replicating
    kernel weights across sub-patches to preserve output variance.
    """
    if src_weight.shape == target_shape:
        return src_weight

    # 4D Conv2d weights: [out_channels, in_channels, kh, kw]
    if src_weight.ndim == 4 and len(target_shape) == 4:
        k_h = target_shape[2] // src_weight.shape[2]
        k_w = target_shape[3] // src_weight.shape[3]
        if (
            k_h == k_w
            and k_h > 1
            and target_shape[2] == src_weight.shape[2] * k_h
            and target_shape[3] == src_weight.shape[3] * k_w
        ):
            return src_weight.repeat(1, 1, k_h, k_w) / float(k_h)

    # 2D Linear weights: [hidden_size, in_channels * p^2]
    if src_weight.ndim == 2 and len(target_shape) == 2:
        ratio = target_shape[1] // src_weight.shape[1]
        k = int(math.isqrt(ratio))
        if k * k == ratio and k > 1:
            d, in_features = src_weight.shape
            p_old = int(math.isqrt(in_features // 3))
            w_2d = src_weight.view(d, 3, p_old, p_old)
            w_tiled = w_2d.repeat(1, 1, k, k) / float(k)
            return w_tiled.reshape(d, target_shape[1])

    raise ValueError(
        f"Cannot adapt patch embed weight from {src_weight.shape} to {target_shape}."
    )


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
    checkpoint_dir = os.path.join(save_dir, f"epoch_{epoch}_step_{global_step}")
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
        logging.info("Filtering state dict: Saving only output head / pixel modules.")
        clean_unet_dict = {
            k: v
            for k, v in clean_unet_dict.items()
            if set(k.split(".")) & PIXEL_EXPLICIT_MODULES
        }

    save_file(clean_unet_dict, os.path.join(checkpoint_dir, "unet.safetensors"))

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
            clean_ema_dict = {
                k: v
                for k, v in clean_ema_dict.items()
                if set(k.split(".")) & PIXEL_EXPLICIT_MODULES
            }
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

        models_cfg = cfg_dict.get("models", {}) if "models" in cfg_dict else cfg_dict

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
            "residual_type": models_cfg.get("residual_type", "concat_linear"),
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
        logging.info(f"Uploading checkpoint to Hugging Face repo: {hf_repo}")
        logging_utils.log_folder(save_dir, hf_repo)
        logging.info("Upload complete.")


def load_checkpoint_config(checkpoint_dir: str) -> dict:
    """Loads config.json from checkpoint directory if present."""
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_latent_to_pixel_weights(
    model: torch.nn.Module,
    checkpoint_path: str,
    ema: Any = None,
    prefer_ema: bool = True,
) -> torch.nn.Module:
    """
    Transfers Transformer backbone and conditioning weights from a latent
    checkpoint to a pixel DiT model. Explicitly skips patch embedders and
    pixel decoders with granular logging.
    """
    logging.info(f"Transferring latent priors from: {checkpoint_path}")
    cand_path = checkpoint_path
    if os.path.isdir(checkpoint_path):
        for fname in ["unet.safetensors", "model.safetensors"]:
            p = os.path.join(checkpoint_path, fname)
            if os.path.exists(p):
                cand_path = p
                break

    if not os.path.isfile(cand_path):
        raise FileNotFoundError(f"Latent checkpoint not found at: {checkpoint_path}")

    sd = load_file(cand_path, device="cpu")
    if "ema_model" in sd and prefer_ema:
        sd = sd["ema_model"]
    elif "model" in sd:
        sd = sd["model"]
    elif "state_dict" in sd:
        sd = sd["state_dict"]

    target_state = model.state_dict()
    target_key_map = {_normalize_param_key(k): k for k in target_state.keys()}

    clean_sd = {}
    transferred = []
    skipped = []

    for k, v in sd.items():
        if k.startswith("text_enc."):
            continue
        norm_k = _normalize_param_key(k)
        if norm_k not in target_key_map:
            continue

        real_k = target_key_map[norm_k]
        target_p = target_state[real_k]

        # Check for pixel-specific module names
        parts = set(norm_k.split("."))
        is_pixel_mod = bool(parts & PIXEL_EXPLICIT_MODULES)

        # 1. Exact shape match: load if not an explicit pixel module
        if target_p.shape == v.shape:
            if is_pixel_mod and ("x_embedder" in norm_k or "conv_in" in norm_k):
                # Guard against reusing latent patch embed bias across spaces
                skipped.append(f"{real_k} (isolated pixel embedder bias reset)")
                continue
            clean_sd[real_k] = v
            transferred.append(real_k)

        # 2. Patch embedding spatial adaptation (ps16 -> ps32)
        elif "x_embedder.weight" in norm_k or "conv_in.weight" in norm_k:
            if v.ndim == 4 and target_p.ndim == 4 and v.shape[1] != target_p.shape[1]:
                skipped.append(
                    f"{real_k} (in_channels mismatch: ckpt "
                    f"{v.shape[1]} vs model {target_p.shape[1]})"
                )
                continue
            try:
                adapted_v = adapt_patch_embed_weight(v, target_p.shape)
                if adapted_v.shape == target_p.shape:
                    clean_sd[real_k] = adapted_v
                    transferred.append(real_k)
                else:
                    skipped.append(
                        f"{real_k} (adapted {adapted_v.shape} != {target_p.shape})"
                    )
            except Exception as e:
                skipped.append(f"{real_k} ({e})")
        else:
            skipped.append(f"{real_k} (shape {v.shape} != target {target_p.shape})")

    model.load_state_dict(clean_sd, strict=False)

    missing_in_ckpt = [
        k for k in target_state.keys() if _normalize_param_key(k) not in clean_sd
    ]

    logging.info(
        f"Successfully transferred {len(transferred)} layers from {checkpoint_path}."
    )
    logging.info(
        f"Skipped {len(skipped)} non-matching layers (expected for patch "
        f"embedder and pixel decoder): {skipped[:3]}"
    )
    logging.info(
        f"{len(missing_in_ckpt)} model layers freshly initialized (not "
        f"present in latent checkpoint)."
    )

    if ema is not None and getattr(ema, "use_ema", False):
        logging.info("Synchronizing target EMA shadow weights with model...")
        if hasattr(ema, "initialize"):
            ema.initialize(model)

    return model


def load_pixel_weights(
    model: torch.nn.Module,
    checkpoint_path: str,
    ema: Any = None,
) -> torch.nn.Module:
    """Loads only explicit pixel adaptation layers into model and EMA."""
    logging.info(f"Loading pixel adaptation weights from: {checkpoint_path}")
    cand_path = checkpoint_path
    if os.path.isdir(checkpoint_path):
        for fname in ["unet.safetensors", "model.safetensors"]:
            p = os.path.join(checkpoint_path, fname)
            if os.path.exists(p):
                cand_path = p
                break

    if not os.path.isfile(cand_path):
        raise FileNotFoundError(f"No pixel weights found at path: {checkpoint_path}")

    state_dict = load_file(cand_path, device="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    target_state = model.state_dict()
    target_key_map = {_normalize_param_key(k): k for k in target_state.keys()}
    pixel_dict = {}

    for k, v in state_dict.items():
        norm_k = _normalize_param_key(k)
        parts = set(norm_k.split("."))
        if not (parts & PIXEL_EXPLICIT_MODULES):
            continue

        if norm_k in target_key_map:
            target_key = target_key_map[norm_k]
            target_p = target_state[target_key]
            if target_p.shape == v.shape:
                pixel_dict[target_key] = v
            elif "x_embedder.weight" in norm_k or "conv_in.weight" in norm_k:
                pixel_dict[target_key] = adapt_patch_embed_weight(v, target_p.shape)
                logging.info(
                    f"Adapted pixel weight '{target_key}' from {v.shape} "
                    f"to {target_p.shape}."
                )
            else:
                logging.warning(
                    f"Skipping pixel layer {target_key} due to shape mismatch: "
                    f"{target_p.shape} vs {v.shape}"
                )

    if not pixel_dict:
        raise KeyError(
            f"No matching pixel adaptation weights found in {checkpoint_path}."
        )

    model.load_state_dict(pixel_dict, strict=False)
    logging.info(f"Loaded {len(pixel_dict)} pixel adaptation layers successfully.")

    if ema is not None and getattr(ema, "use_ema", False):
        ema_target = (
            ema.ema_model.state_dict()
            if hasattr(ema, "ema_model") and ema.ema_model is not None
            else ema.state_dict()
        )
        ema_key_map = {_normalize_param_key(k): k for k in ema_target.keys()}
        clean_ema = {}
        for k, v in pixel_dict.items():
            norm_k = _normalize_param_key(k)
            if norm_k in ema_key_map:
                target_k = ema_key_map[norm_k]
                clean_ema[target_k] = v

        if clean_ema:
            if hasattr(ema, "ema_model") and ema.ema_model is not None:
                ema.ema_model.load_state_dict(clean_ema, strict=False)
            elif hasattr(ema, "load_state_dict"):
                ema.load_state_dict(clean_ema, strict=False)

    return model
