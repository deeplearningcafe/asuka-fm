import torch
import logging
from safetensors.torch import load_file
from transformers import CLIPTokenizer
import os
import torch.nn as nn
from functools import partial
import gc
from typing import Any, List, Dict, Optional
import omegaconf
from diffusers import AutoencoderKL
from src.models.unet import Unet, UnetConfig
from src.models.dual_stream import DualStreamDiT
from src.models.sprint import SprintDualStreamDiT
from src.models.text_encoders.clip import Clip, ClipConfig
from src.models.vae import Vae, VaeConfig
from src.utils.ema import EMAModel
from src.models.text_encoders.text_encoders import (
    HFTextEncoder,
    CLIPTextEncoderWrapper,
)
from src.models.text_encoders.tokenizer import HFLLMTokenizer
from src.utils.checkpointing import (
    load_pixel_weights,
    load_latent_to_pixel_weights,
    _normalize_param_key,
    PIXEL_EXPLICIT_MODULES,
    adapt_patch_embed_weight,
)


class ModelInspector:
    """
    A modular class to inspect model stability by logging activations and
    gradients using PyTorch hooks.
    """

    def __init__(self, logging_fn, model_dtype=torch.float32):
        self.logging_fn = logging_fn
        self.model_dtype = model_dtype
        self.activation_tensors = {}
        self.gradient_tensors = {}
        self.hooks = []

    def _forward_hook(self, name, module, args, output):
        if torch.is_tensor(output):
            self.activation_tensors[name] = output.detach()

    def _backward_hook(self, name, module, grad_input, grad_output):
        grad = grad_output[0]
        if torch.is_tensor(grad):
            self.gradient_tensors[name] = grad.detach()

    def register_hooks(self, model: nn.Module):
        # Strategic subset of layers to monitor across UNet and DualStreamDiT
        target_layer_names = {
            # --- UNet Target Layers ---
            "down_blocks.0.resnets.0",
            "down_blocks.0.attentions.0",
            "down_blocks.2.attentions.1",
            "down_blocks.3.resnets.1",
            "mid_block.attentions.0",
            "mid_block.resnets.1",
            "up_blocks.0.resnets.2",
            "up_blocks.0.attentions.2",
            "up_blocks.1.resnets.0",
            "up_blocks.1.attentions.0",
            "up_blocks.3.resnets.2",
            "up_blocks.3.attentions.2",
            "conv_out",
            "conv_in",
            # --- DualStreamDiT Target Layers ---
            "x_embedder",
            "time_token_proj",
            "text_adapter.proj_in",
            "text_adapter.blocks.0.ff",
            "text_adapter.blocks.1.ff",
            "in_blocks.0.attn",
            "in_blocks.0.mlp_image",
            "in_blocks.0.mlp_text",
            "in_blocks.2.attn",
            "in_blocks.2.mlp_image",
            "in_blocks.3.attn",
            "in_blocks.3.mlp_image",
            "mid_block.attn",
            "mid_block.mlp_image",
            "mid_block.mlp_text",
            "out_blocks.0.skip_linear_image",
            "out_blocks.0.attn",
            "out_blocks.0.mlp_image",
            "out_blocks.2.attn",
            "out_blocks.2.mlp_image",
            "out_blocks.3.attn",
            "out_blocks.3.mlp_image",
            "norm_final",
            "proj_out",
        }

        for name, module in model.named_modules():
            clean_name = (
                name[len("_orig_mod.") :] if name.startswith("_orig_mod.") else name
            )
            if clean_name in target_layer_names:
                f_hook = module.register_forward_hook(
                    partial(self._forward_hook, clean_name)
                )
                b_hook = module.register_full_backward_hook(
                    partial(self._backward_hook, clean_name)
                )
                self.hooks.extend([f_hook, b_hook])

        logging.info(f"Registered {len(self.hooks)} hooks for stability checks.")

    def log_stats(self, step: int):
        if not self.activation_tensors and not self.gradient_tensors:
            return

        log_payload = {}
        for name, tensor in self.activation_tensors.items():
            log_payload[f"activations/{name}/mean"] = tensor.mean().item()
            log_payload[f"activations/{name}/std"] = tensor.std().item()
            log_payload[f"activations/{name}/max"] = tensor.abs().max().item()

        for name, tensor in self.gradient_tensors.items():
            log_payload[f"gradients/{name}/mean"] = tensor.mean().item()
            log_payload[f"gradients/{name}/std"] = tensor.std().item()
            log_payload[f"gradients/{name}/max"] = tensor.abs().max().item()

        if self.logging_fn and log_payload:
            self.logging_fn(log_payload, step=step, commit=False)

        self.activation_tensors.clear()
        self.gradient_tensors.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        logging.info("Removed all stability check hooks.")


def set_trainable_layers(
    model: nn.Module,
    train_output_only: bool = False,
    shallow_tuning: bool = False,
) -> nn.Module:
    """
    Configures parameter gradients for two-stage adaptation.
    - If train_output_only: trains only patch embedder and pixel decoder.
    - If shallow_tuning (L2P): freezes mid_blocks, trains in/out blocks.
    - Otherwise: trains all layers.
    """
    if not train_output_only and not shallow_tuning:
        return model

    trainable_count = 0
    frozen_count = 0

    target_model = (
        model["unet"]
        if isinstance(model, (dict, nn.ModuleDict)) and "unet" in model
        else model
    )

    for name, param in target_model.named_parameters():
        clean_name = (
            name[len("_orig_mod.") :] if name.startswith("_orig_mod.") else name
        )
        parts = set(clean_name.split("."))
        if train_output_only:
            is_trainable = bool(parts & PIXEL_EXPLICIT_MODULES)
        elif shallow_tuning:
            # L2P recipe: freeze deep intermediate blocks and text adapter
            is_frozen = (
                clean_name.startswith("mid_blocks.")
                or clean_name.startswith("text_adapter.")
                or "renoise_linear" in clean_name
            )
            is_trainable = not is_frozen
        else:
            is_trainable = True

        param.requires_grad = is_trainable
        if is_trainable:
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()

    mode_str = "Stage-1 Output-Only" if train_output_only else "L2P Shallow"
    logging.info(
        f"{mode_str} Adaptation: {trainable_count / 1e6:.2f}M trainable, "
        f"{frozen_count / 1e6:.2f}M frozen params."
    )
    return model


def load_trainable_model(
    models_path,
    device,
    dtype=torch.float32,
    train_te: bool = True,
    use_checkpointing: bool = True,
    resume_from_checkpoint: str = None,
    train_only_output: bool = False,
    output_head_path: str = None,
    use_ema: bool = True,
    ema_decay: float = 0.99,
    global_rank: int = 0,
    model_type: str = "unet",
    model_cfg: omegaconf.DictConfig = None,
    autocast_dtype=torch.float32,
    latent_checkpoint: str = None,
    pixel_dir: str = None,
    shallow_tuning: bool = False,
):
    """
    Loads models (UNet, TE, VAE) and configures them for training (gradients, dtype).
    Handles fallback logic: Checkpoint -> Base Model.
    """
    is_dit = model_type in ["dual_stream", "sprint_dual"]
    unet_path = f"{models_path}/unet/diffusion_pytorch_model.safetensors"
    te_path = f"{models_path}/clip/model.safetensors"

    if resume_from_checkpoint and os.path.isdir(resume_from_checkpoint):
        if global_rank == 0:
            logging.info(f"Resolved checkpoint directory: {resume_from_checkpoint}")
        for fname in ["unet.safetensors", "model.safetensors"]:
            candidate = os.path.join(resume_from_checkpoint, fname)
            if os.path.exists(candidate):
                unet_path = candidate
                if global_rank == 0:
                    logging.info(f"  -> Found model weights: {unet_path}")
                break

        ckpt_te_path = os.path.join(resume_from_checkpoint, "text_encoder.safetensors")
        hf_te_id = getattr(model_cfg, "hf_text_encoder", None)
        if not hf_te_id and train_te and os.path.exists(ckpt_te_path):
            te_path = ckpt_te_path
            if global_rank == 0:
                logging.info(f"  -> Found Text Encoder weights: {te_path}")
    elif resume_from_checkpoint:
        raise FileNotFoundError(
            f"Checkpoint path '{resume_from_checkpoint}' could not be resolved."
        )

    hf_te_id = getattr(model_cfg, "hf_text_encoder", None)
    if hf_te_id:
        if global_rank == 0:
            logging.info(f"Loading HuggingFace Text Encoder: {hf_te_id}")
        text_encoder = HFTextEncoder(
            hf_te_id,
            torch_dtype=autocast_dtype,
            cache_dir=f"{models_path}/text_encoder",
        )
        tokenizer = HFLLMTokenizer(
            hf_te_id,
            cache_dir=f"{models_path}/tokenizer",
        )
        text_embed_dim = text_encoder.embed_dim
    else:
        # TODO: make a wrapper in tokenizers.py
        raw_clip = Clip.from_pretrained(ClipConfig(), te_path).eval()
        tokenizer = CLIPTokenizer.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="tokenizer",
            cache_dir=f"{models_path}/tokenizer",
        )
        text_encoder = CLIPTextEncoderWrapper(raw_clip, tokenizer)
        text_embed_dim = 768

    in_channels = getattr(model_cfg, "in_channels", 32) if model_cfg else 4
    if global_rank == 0:
        logging.info(f"Loading {model_type} model from {models_path}...")
    try:
        hidden_size = getattr(model_cfg, "hidden_size", 768) if model_cfg else 768
        depth = getattr(model_cfg, "depth", 16) if model_cfg else 16
        num_heads = getattr(model_cfg, "num_heads", 12) if model_cfg else 12
        patch_size = getattr(model_cfg, "patch_size", 2) if model_cfg else 2
        skip_checkpointing_layers = (
            getattr(model_cfg, "skip_checkpointing_layers", 0) if model_cfg else 0
        )
        use_rope = (
            getattr(model_cfg, "use_rope_text_adapter", False) if model_cfg else False
        )
        use_calibrated_spatial = (
            getattr(model_cfg, "use_calibrated_spatial", False) if model_cfg else False
        )

        use_pixel_decoder = (
            getattr(model_cfg, "use_pixel_decoder", False) if model_cfg else False
        )
        input_level = (
            getattr(model_cfg, "input_level", "patch_level")
            if model_cfg
            else "patch_level"
        )
        upsample_mode = (
            getattr(model_cfg, "upsample_mode", "ConvTranspose")
            if model_cfg
            else "ConvTranspose"
        )
        if global_rank == 0:
            logging.info(
                f"Creating model with {hidden_size} hs, {depth} layers, and spatial rope {use_calibrated_spatial}"
                f"Use pixel decoder {use_pixel_decoder} and input level {input_level}"
            )
        if model_type == "dual_stream":
            # TODO: channels dynamically from vae meta
            unet = DualStreamDiT(
                in_channels=in_channels,
                out_channels=in_channels,
                patch_size=patch_size,
                hidden_size=hidden_size,
                depth=depth,
                num_heads=num_heads,
                text_embed_dim=text_embed_dim,
                use_checkpointing=use_checkpointing,
                use_rope_text_adapter=use_rope,
                skip_checkpointing_layers=skip_checkpointing_layers,
                use_pixel_decoder=use_pixel_decoder,
                input_level=input_level,
            )
            # if os.path.exists(unet_path):
            #     sd = load_file(unet_path, device="cpu")
            #     sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
            #     unet.load_state_dict(sd, strict=False)
        elif model_type == "sprint_dual":
            encoder_depth = getattr(model_cfg, "encoder_depth", 2) if model_cfg else 2
            decoder_depth = getattr(model_cfg, "decoder_depth", 2) if model_cfg else 2
            drop_ratio = getattr(model_cfg, "drop_ratio", 0.75) if model_cfg else 0.0
            drop_target = (
                getattr(model_cfg, "drop_target", "image") if model_cfg else "image"
            )
            residual_type = (
                getattr(model_cfg, "residual_type", "concat_linear")
                if model_cfg
                else "concat_linear"
            )
            cfg_mask_prob = (
                getattr(model_cfg, "cfg_mask_prob", 0.1) if model_cfg else 0.0
            )
            use_random_drop = (
                getattr(model_cfg, "use_random_drop", True) if model_cfg else True
            )
            if global_rank == 0:
                logging.info(
                    f"Sprint with {drop_ratio} drop ratio, {residual_type} residual type and {drop_target} target"
                )
            unet = SprintDualStreamDiT(
                in_channels=in_channels,
                out_channels=in_channels,
                patch_size=patch_size,
                hidden_size=hidden_size,
                depth=depth,
                num_heads=num_heads,
                text_embed_dim=text_embed_dim,
                encoder_depth=encoder_depth,
                decoder_depth=decoder_depth,
                drop_ratio=drop_ratio,
                drop_target=drop_target,
                residual_type=residual_type,
                cfg_mask_prob=cfg_mask_prob,
                use_checkpointing=use_checkpointing,
                use_rope_text_adapter=use_rope,
                skip_checkpointing_layers=skip_checkpointing_layers,
                use_random_drop=use_random_drop,
                use_calibrated_spatial=use_calibrated_spatial,
                use_pixel_decoder=use_pixel_decoder,
                input_level=input_level,
            )
        else:
            unet = Unet.from_pretrained(
                UnetConfig(use_checkpointing=use_checkpointing),
                unet_path,
                output_head_path=output_head_path,
            ).eval()

        # EMA initialization
        actual_use_ema = use_ema and (global_rank == 0)
        ema = EMAModel(
            unet,
            decay=ema_decay,
            use_ema=actual_use_ema,
            device=torch.device("cpu"),
        )

        # 1. Transfer latent priors if latent_checkpoint is provided
        if is_dit and latent_checkpoint:
            if global_rank == 0:
                logging.info(f"Transferring latent priors from: {latent_checkpoint}")
            load_latent_to_pixel_weights(
                unet, latent_checkpoint, ema=ema, prefer_ema=True
            )

        # 2. Resume full pixel training checkpoint if explicitly provided
        elif is_dit and resume_from_checkpoint:
            ckpt_dit_path = None
            if os.path.isdir(resume_from_checkpoint):
                for fname in ["unet.safetensors", "model.safetensors"]:
                    cand = os.path.join(resume_from_checkpoint, fname)
                    if os.path.exists(cand):
                        ckpt_dit_path = cand
                        break
            if ckpt_dit_path and os.path.exists(ckpt_dit_path):
                if global_rank == 0:
                    logging.info(f"  -> Resuming DiT weights from: {ckpt_dit_path}")
                sd = load_file(ckpt_dit_path, device="cpu")
                sd = {
                    _normalize_param_key(k): v
                    for k, v in sd.items()
                    if not k.startswith("text_enc.")
                }
                unet.load_state_dict(sd, strict=False)

                # Load EMA weights
                for ema_name in [
                    "unet_ema.safetensors",
                    "ema_model.safetensors",
                ]:
                    ema_path = os.path.join(resume_from_checkpoint, ema_name)
                    if os.path.exists(ema_path) and ema.use_ema:
                        ema_dict = load_file(ema_path, device="cpu")
                        ema_dict = {
                            _normalize_param_key(k): v
                            for k, v in ema_dict.items()
                            if not k.startswith("text_enc.")
                        }
                        if hasattr(ema, "ema_model") and ema.ema_model is not None:
                            ema.ema_model.load_state_dict(ema_dict, strict=False)
                        break

        # 3. Load Stage-1 pixel adaptation weights if provided
        active_pixel_dir = pixel_dir or output_head_path
        if active_pixel_dir and os.path.exists(active_pixel_dir):
            if global_rank == 0:
                logging.info(
                    f"Loading Stage-1 pixel adaptation weights: {active_pixel_dir}"
                )
            load_pixel_weights(unet, active_pixel_dir, ema=ema)

        hf_vae_id = getattr(model_cfg, "hf_vae", None) or getattr(
            model_cfg, "vae_pretrained", None
        )
        if hf_vae_id:
            if global_rank == 0:
                logging.info(f"Loading HuggingFace VAE: {hf_vae_id}")
            vae = AutoencoderKL.from_pretrained(
                hf_vae_id,
                torch_dtype=autocast_dtype,
                cache_dir=None,  # f"{models_path}/vae",
            ).eval()
        else:
            vae_path = f"{models_path}/vae/diffusion_pytorch_model.safetensors"
            vae = Vae.from_pretrained(VaeConfig(), vae_path).eval()
        # TODO: dynamically move to cpu
        vae.to(device)

        for param in vae.parameters():
            param.requires_grad = True

        if global_rank == 0:
            logging.info(f"Moving models to {device} and converting to {dtype}")
        unet.to(device)
        text_encoder.to(device)

        if dtype != torch.float32:
            unet.to(dtype=dtype)
            text_encoder.to(dtype=dtype)

        # clear ram
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        logging.info(f"ERROR: Could not load model: {e}")
        raise

    if train_only_output or shallow_tuning:
        logging.info("Configuring for Stage-1 Output/Pixel Head training only.")
        set_trainable_layers(
            unet, train_output_only=train_only_output, shallow_tuning=shallow_tuning
        )

    unet.train()

    # Text Encoder
    if train_te:
        for param in text_encoder.parameters():
            param.requires_grad = True
        text_encoder.train()
    else:
        logging.info("Freezing Text Encoder (converting to bf16)")
        text_encoder.to(dtype=autocast_dtype)
        for param in text_encoder.parameters():
            param.requires_grad = False
        text_encoder.eval()

    return unet, text_encoder, vae, tokenizer, ema


def load_training_state(
    checkpoint_path: str,
    optimizer,
    scheduler,
    device,
    global_rank,
    reset_scheduler: bool = False,
    reset_optimizer: bool = False,
):
    """Loads optimizer, scheduler, and training state (epoch/step)."""
    start_epoch = 0
    global_step = 0

    if not checkpoint_path or not os.path.isdir(checkpoint_path):
        return optimizer, scheduler, start_epoch, global_step

    if global_rank == 0:
        logging.info(f"Resuming training state from: {checkpoint_path}")

    # Load Epoch/Step with fallback to directory name parsing
    state_path = os.path.join(checkpoint_path, "training_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=device)
        start_epoch = state.get("epoch", 0)
        global_step = state.get("global_step", 0)
        if global_rank == 0:
            logging.info(
                f"  -> Resuming from epoch {start_epoch}, global step {global_step}"
            )
    else:
        base_name = os.path.basename(os.path.normpath(checkpoint_path))
        if base_name.startswith("epoch_"):
            parts = base_name.split("_")
            try:
                start_epoch = int(parts[1])
                if len(parts) >= 4 and parts[2] == "step":
                    global_step = int(parts[3])
            except (ValueError, IndexError):
                pass
            if global_rank == 0:
                logging.info(
                    f"  -> Resumed from directory name: "
                    f"epoch {start_epoch}, step {global_step}"
                )

    if not reset_optimizer:
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            # Capture target LR and weight decay configured from cfg
            target_lrs = [g["lr"] for g in optimizer.param_groups]
            target_wds = [g.get("weight_decay", 0.0) for g in optimizer.param_groups]

            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))

            if reset_scheduler:
                # When resetting scheduler, apply target LR from config
                if len(optimizer.param_groups) == len(target_lrs):
                    for group, new_lr, new_wd in zip(
                        optimizer.param_groups, target_lrs, target_wds
                    ):
                        group["lr"] = new_lr
                        group["initial_lr"] = new_lr
                        group["weight_decay"] = new_wd
                if global_rank == 0:
                    logging.info(
                        f"  -> Optimizer state loaded (re-applied target lr: "
                        f"{target_lrs[0]:.2e})"
                    )
            else:
                # Resuming: keep scheduled LR
                if len(optimizer.param_groups) == len(target_wds):
                    for group, new_wd in zip(optimizer.param_groups, target_wds):
                        group["weight_decay"] = new_wd
                if global_rank == 0:
                    logging.info(
                        f"  -> Optimizer state loaded (restored lr: "
                        f"{optimizer.param_groups[0]['lr']:.2e})"
                    )

            if global_rank == 0:
                logging.info(
                    f"  -> Optimizer state loaded (re-applied target lr: "
                    f"{target_lrs[0]:.2e})"
                )
    else:
        if global_rank == 0:
            logging.info("  -> Optimizer reset: starting fresh optimizer.")

    if not reset_scheduler:
        scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
        if scheduler and os.path.exists(scheduler_path):
            try:
                scheduler.load_state_dict(
                    torch.load(scheduler_path, map_location=device)
                )
                # Synchronize optimizer LRs with scheduler.
                if hasattr(scheduler, "get_last_lr"):
                    for group, last_lr in zip(
                        optimizer.param_groups, scheduler.get_last_lr()
                    ):
                        group["lr"] = last_lr

                # Restore base_lrs into initial_lr to ensure subsequent phases work
                base_lrs = getattr(scheduler, "base_lrs", None)
                if (
                    base_lrs is None
                    and hasattr(scheduler, "_schedulers")
                    and scheduler._schedulers
                ):
                    base_lrs = getattr(scheduler._schedulers[0], "base_lrs", None)
                if base_lrs is not None:
                    for group, b_lr in zip(optimizer.param_groups, base_lrs):
                        group["initial_lr"] = b_lr

                if global_rank == 0:
                    logging.info(
                        f"  -> Scheduler state loaded (resumed lr: "
                        f"{optimizer.param_groups[0]['lr']:.2e})."
                    )

            except Exception as e:
                if global_rank == 0:
                    logging.warning(
                        f"Could not load scheduler state: {e}. "
                        f"Proceeding with initialized scheduler."
                    )
    else:
        if global_rank == 0:
            logging.info("  -> Scheduler reset: will build new scheduler.")

    torch.cuda.empty_cache()
    return optimizer, scheduler, start_epoch, global_step


def create_optimizer_param_groups(
    unet_model: Any,
    text_encoder_model: Any,
    base_lr: float,
    weight_decay: float,
    train_te: bool = False,
    unet_output_lr_multiplier: float = 2.0,
    unet_high_lr_multiplier: float = 1.75,
    unet_backbone_lr_multiplier: float = 1.25,
    unet_low_lr_multiplier: float = 1.0,
    text_encoder_lr_multiplier: float = 0.5,
    model_type: str = "unet",
    dit_pixel_lr_multiplier: Optional[float] = None,
    dit_backbone_lr_multiplier: Optional[float] = None,
) -> List[Dict]:
    """Creates parameter groups with specific LRs and Weight Decay rules."""
    no_decay_keywords = ["bias", "norm"]
    param_groups = []

    is_dit = (
        model_type in ["dual_stream", "sprint_dual"]
        or hasattr(unet_model, "x_embedder")
        or hasattr(getattr(unet_model, "module", None), "x_embedder")
    )

    if is_dit:
        pixel_mult = (
            dit_pixel_lr_multiplier
            if dit_pixel_lr_multiplier is not None
            else unet_output_lr_multiplier
        )
        backbone_mult = (
            dit_backbone_lr_multiplier
            if dit_backbone_lr_multiplier is not None
            else 1.0
        )

        pix_d, pix_nd = [], []
        bb_d, bb_nd = [], []

        for name, param in unet_model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("_orig_mod."):
                name = name[len("_orig_mod.") :]

            parts = set(name.split("."))
            is_pixel_mod = bool(parts & PIXEL_EXPLICIT_MODULES)
            is_no_decay = any(k in name for k in no_decay_keywords)

            if is_pixel_mod:
                if is_no_decay:
                    pix_nd.append(param)
                else:
                    pix_d.append(param)
            else:
                if is_no_decay:
                    bb_nd.append(param)
                else:
                    bb_d.append(param)

        if pix_d:
            param_groups.append(
                {
                    "params": pix_d,
                    "lr": base_lr * pixel_mult,
                    "weight_decay": weight_decay,
                    "name": "dit_pixel_decay",
                }
            )
        if pix_nd:
            param_groups.append(
                {
                    "params": pix_nd,
                    "lr": base_lr * pixel_mult,
                    "weight_decay": 0.0,
                    "name": "dit_pixel_no_decay",
                }
            )

        if bb_d:
            param_groups.append(
                {
                    "params": bb_d,
                    "lr": base_lr * backbone_mult,
                    "weight_decay": weight_decay,
                    "name": "dit_backbone_decay",
                }
            )
        if bb_nd:
            param_groups.append(
                {
                    "params": bb_nd,
                    "lr": base_lr * backbone_mult,
                    "weight_decay": 0.0,
                    "name": "dit_backbone_no_decay",
                }
            )
    else:
        unet_output_prefixes = ("conv_out.", "conv_norm_out.", "down_blocks.0.")
        unet_high_lr_prefixes = (
            "time_embedding.",
            "down_blocks.1.",
            "down_blocks.2.",
        )
        unet_low_lr_prefixes = ("up_blocks.2.", "up_blocks.3.")

        u_out_d, u_out_nd = [], []
        u_high_d, u_high_nd = [], []
        u_low_d, u_low_nd = [], []
        u_base_d, u_base_nd = [], []

        for name, param in unet_model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("_orig_mod."):
                name = name[len("_orig_mod.") :]

            is_no_decay = any(k in name for k in no_decay_keywords)
            target_list = None

            if name.startswith(unet_output_prefixes):
                target_list = u_out_nd if is_no_decay else u_out_d
            elif name.startswith(unet_high_lr_prefixes):
                target_list = u_high_nd if is_no_decay else u_high_d
            elif name.startswith(unet_low_lr_prefixes):
                target_list = u_low_nd if is_no_decay else u_low_d
            else:
                target_list = u_base_nd if is_no_decay else u_base_d

            target_list.append(param)

        groups_config = [
            (
                u_out_d,
                u_out_nd,
                base_lr * unet_output_lr_multiplier,
                "unet_output",
            ),
            (
                u_high_d,
                u_high_nd,
                base_lr * unet_high_lr_multiplier,
                "unet_high",
            ),
            (
                u_low_d,
                u_low_nd,
                base_lr * unet_low_lr_multiplier,
                "unet_low",
            ),
            (
                u_base_d,
                u_base_nd,
                base_lr * unet_backbone_lr_multiplier,
                "unet_backbone",
            ),
        ]

        for decay, no_decay, lr, name in groups_config:
            if decay:
                param_groups.append(
                    {
                        "params": decay,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "name": f"{name}_decay",
                    }
                )
            if no_decay:
                param_groups.append(
                    {
                        "params": no_decay,
                        "lr": lr,
                        "weight_decay": 0.0,
                        "name": f"{name}_no_decay",
                    }
                )

    if train_te:
        # Freeze unused last layers
        num_layers = text_encoder_model.config.n_layer
        unused_prefixes = (
            f"text_model.encoder.{num_layers - 1}.",
            "text_model.final_layer_norm.",
        )
        for name, param in text_encoder_model.named_parameters():
            if name.startswith(unused_prefixes):
                param.requires_grad_(False)

        te_high_prefixes = (
            "text_model.embeddings.",
            "text_model.encoder.0.",
            "text_model.encoder.1.",
            f"text_model.encoder.{num_layers - 3}.",
            f"text_model.encoder.{num_layers - 2}.",
        )

        te_high_d, te_high_nd = [], []
        te_low_d, te_low_nd = [], []

        for name, param in text_encoder_model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("_orig_mod."):
                name = name[len("_orig_mod.") :]

            is_no_decay = any(k in name for k in no_decay_keywords)

            if name.startswith(te_high_prefixes):
                target_list = te_high_nd if is_no_decay else te_high_d
            else:
                target_list = te_low_nd if is_no_decay else te_low_d
            target_list.append(param)

        param_groups.append(
            {
                "params": te_high_d,
                "lr": base_lr * text_encoder_lr_multiplier,
                "weight_decay": weight_decay,
                "name": "te_high_decay",
            }
        )
        param_groups.append(
            {
                "params": te_high_nd,
                "lr": base_lr * text_encoder_lr_multiplier,
                "weight_decay": 0.0,
                "name": "te_high_no_decay",
            }
        )
        param_groups.append(
            {
                "params": te_low_d,
                "lr": base_lr * text_encoder_lr_multiplier * 0.5,
                "weight_decay": weight_decay,
                "name": "te_low_decay",
            }
        )
        param_groups.append(
            {
                "params": te_low_nd,
                "lr": base_lr * text_encoder_lr_multiplier * 0.5,
                "weight_decay": 0.0,
                "name": "te_low_no_decay",
            }
        )

    return [g for g in param_groups if g["params"]]


def create_optim(unet, text_encoder, conf: omegaconf.DictConfig):
    model_type = getattr(conf.models, "model_type", "unet")
    dit_pixel_mult = conf.train.get(
        "dit_pixel_lr_multiplier",
        conf.train.get("pixel_lr_multiplier", 2.0),
    )
    dit_bb_mult = conf.train.get(
        "dit_backbone_lr_multiplier",
        conf.train.get("backbone_lr_multiplier", 1.0),
    )
    param_groups = create_optimizer_param_groups(
        unet_model=unet,
        text_encoder_model=text_encoder,
        base_lr=conf.train.lr,
        weight_decay=conf.train.wd,
        train_te=conf.train.train_te,
        unet_output_lr_multiplier=1.15,
        unet_high_lr_multiplier=1.05,
        unet_backbone_lr_multiplier=1.0,
        unet_low_lr_multiplier=1.0,
        model_type=model_type,
        dit_pixel_lr_multiplier=dit_pixel_mult,
        dit_backbone_lr_multiplier=dit_bb_mult,
    )

    if conf.train.use_bitsandbytes:
        import bitsandbytes as bnb

        optim = bnb.optim.AdamW8bit(param_groups, lr=conf.train.lr, betas=(0.9, 0.95))
    elif conf.train.use_kahan_sum:
        from src.optimizer.adamw_8bit import AdamW8bitKahan

        optim = AdamW8bitKahan(param_groups, lr=conf.train.lr, betas=(0.9, 0.95))
    else:
        optim = torch.optim.AdamW(
            param_groups, lr=conf.train.lr, betas=(0.9, 0.95), fused=True
        )

    return optim


def create_scheduler(
    optim,
    train_loader,
    conf: omegaconf.DictConfig,
    total_steps_override: Optional[int] = None,
):
    """Creates a flexible LR scheduler supporting WSD, Cosine, and Constant."""
    # TODO: is this redundant?
    for group in optim.param_groups:
        group["initial_lr"] = group["lr"]

    if total_steps_override is not None:
        total_steps = max(1, total_steps_override)
    elif not hasattr(train_loader, "__len__"):
        total_steps = conf.train.epochs * 10000
    else:
        grad_accum = conf.train.gradient_accumulation_steps
        update_steps_epoch = len(train_loader) // grad_accum
        total_steps = conf.train.epochs * update_steps_epoch

    warmup_ratio = conf.train.get("warmup", 0.04)
    warmup_steps = int(warmup_ratio * total_steps)
    warmup_steps = min(warmup_steps, total_steps - 1) if total_steps > 1 else 0

    decay_ratio = conf.train.get("decay_ratio", 0.0)
    decay_steps = int(decay_ratio * total_steps)
    decay_steps = min(decay_steps, total_steps - warmup_steps)

    stable_steps = max(0, total_steps - warmup_steps - decay_steps)

    # Resolve schedule type with backward compatibility
    sched_type = conf.train.get("scheduler_type", None)
    if sched_type is None:
        if conf.train.get("use_cos_scheduler", False):
            sched_type = "cosine"
        elif decay_ratio > 0.0:
            sched_type = "wsd"
        else:
            sched_type = "constant"

    min_lr_ratio = conf.train.get("min_lr_ratio", 0.05)
    decay_type = conf.train.get("decay_type", "cosine")
    base_lr = conf.train.lr

    schedulers = []
    milestones = []
    current_step = 0

    # 1. Warmup stage
    if warmup_steps > 0:
        s_warmup = torch.optim.lr_scheduler.LinearLR(
            optim,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        schedulers.append(s_warmup)
        current_step += warmup_steps
        milestones.append(current_step)

    # 2. Main schedule branches
    if sched_type == "cosine":
        logging.info("Using Warmup-Cosine lr scheduler")
        cosine_steps = max(1, total_steps - warmup_steps)
        s_cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=cosine_steps,
            eta_min=base_lr * min_lr_ratio,
        )
        schedulers.append(s_cos)

    elif sched_type == "wsd":
        logging.info(
            f"Using WSD scheduler (Warmup: {warmup_steps}, "
            f"Stable: {stable_steps}, Decay: {decay_steps})"
        )
        if stable_steps > 0:
            s_stable = torch.optim.lr_scheduler.ConstantLR(
                optim,
                factor=1.0,
                total_iters=stable_steps,
            )
            schedulers.append(s_stable)
            current_step += stable_steps
            if decay_steps > 0:
                milestones.append(current_step)

        if decay_steps > 0:
            if decay_type == "cosine":
                s_decay = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim,
                    T_max=decay_steps,
                    eta_min=base_lr * min_lr_ratio,
                )
            else:
                s_decay = torch.optim.lr_scheduler.LinearLR(
                    optim,
                    start_factor=1.0,
                    end_factor=min_lr_ratio,
                    total_iters=decay_steps,
                )
            schedulers.append(s_decay)

    else:
        logging.info("Using constant lr scheduler")
        const_steps = max(1, total_steps - warmup_steps)
        s_const = torch.optim.lr_scheduler.ConstantLR(
            optim,
            factor=1.0,
            total_iters=const_steps,
        )
        schedulers.append(s_const)

    if len(schedulers) == 1:
        return schedulers[0]

    valid_milestones = milestones[: len(schedulers) - 1]
    return torch.optim.lr_scheduler.SequentialLR(
        optim,
        schedulers,
        milestones=valid_milestones,
    )
