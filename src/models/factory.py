import torch
import logging
from safetensors.torch import load_file
from transformers import CLIPTokenizer
import os
import torch.nn as nn
from functools import partial
import gc
from typing import Any, List, Dict
import omegaconf
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
):
    """
    Loads models (UNet, TE, VAE) and configures them for training (gradients, dtype).
    Handles fallback logic: Checkpoint -> Base Model.
    """
    unet_path = f"{models_path}/unet/diffusion_pytorch_model.safetensors"
    te_path = f"{models_path}/clip/model.safetensors"

    if resume_from_checkpoint and os.path.isdir(resume_from_checkpoint):
        if global_rank == 0:
            logging.info(
                f"Attempting to load weights from checkpoint: {resume_from_checkpoint}"
            )
        ckpt_unet_path = os.path.join(resume_from_checkpoint, "unet.safetensors")
        ckpt_te_path = os.path.join(resume_from_checkpoint, "text_encoder.safetensors")

        if os.path.exists(ckpt_unet_path):
            unet_path = ckpt_unet_path
            if global_rank == 0:
                logging.info(f"  -> Found UNet weights: {unet_path}")

        if train_te and os.path.exists(ckpt_te_path):
            te_path = ckpt_te_path
            if global_rank == 0:
                logging.info(f"  -> Found Text Encoder weights: {te_path}")

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
        if model_type == "dual_stream":
            # TODO: channels dynamically from vae meta
            unet = DualStreamDiT(
                in_channels=4,
                out_channels=4,
                patch_size=patch_size,
                hidden_size=hidden_size,
                depth=depth,
                num_heads=num_heads,
                text_embed_dim=text_embed_dim,
                use_checkpointing=use_checkpointing,
                use_rope_text_adapter=use_rope,
                skip_checkpointing_layers=skip_checkpointing_layers,
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

            unet = SprintDualStreamDiT(
                in_channels=4,
                out_channels=4,
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
            )
        else:
            unet = Unet.from_pretrained(
                UnetConfig(use_checkpointing=use_checkpointing),
                unet_path,
                output_head_path=output_head_path,
            ).eval()

        # Offload to CPU
        actual_use_ema = use_ema and (global_rank == 0)
        ema = EMAModel(
            unet, decay=ema_decay, use_ema=actual_use_ema, device=torch.device("cpu")
        )

        if resume_from_checkpoint and os.path.isdir(resume_from_checkpoint):
            ema_path = os.path.join(resume_from_checkpoint, "unet_ema.safetensors")
            # Only rank 0 has ema.use_ema = True
            if os.path.exists(ema_path) and ema.use_ema:
                logging.info(f"  -> Found EMA weights: {ema_path}")
                ema.ema_model.load_state_dict(load_file(ema_path, device="cpu"))

        # text_encoder = Clip.from_pretrained(ClipConfig(), te_path).eval()

        # TODO: add hf vae support
        vae = Vae.from_pretrained(
            VaeConfig(), f"{models_path}/vae/diffusion_pytorch_model.safetensors"
        ).eval()
        # TODO: dynamically move to cpu
        vae.to(device)

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

    if train_only_output:
        logging.info("Configuring for Output Head training only.")
        # nn.module default is true
        for param in unet.parameters():
            param.requires_grad = False

        if hasattr(unet, "proj_out"):
            # DiT / Sprint
            for param in unet.proj_out.parameters():
                param.requires_grad = True
            if hasattr(unet, "norm_final"):
                for param in unet.norm_final.parameters():
                    param.requires_grad = True
        elif hasattr(unet, "conv_out"):
            # UNet
            unet.conv_norm_out.bias.requires_grad = True
            unet.conv_norm_out.weight.requires_grad = True
            unet.conv_out.bias.requires_grad = True
            unet.conv_out.weight.requires_grad = True

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
):
    """Loads optimizer, scheduler, and training state (epoch/step) from checkpoint."""
    start_epoch = 0
    global_step = 0

    if not checkpoint_path or not os.path.isdir(checkpoint_path):
        return optimizer, scheduler, start_epoch, global_step

    if global_rank == 0:
        logging.info(f"Resuming training state from: {checkpoint_path}")

    # Load Epoch/Step
    state_path = os.path.join(checkpoint_path, "training_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=device)
        start_epoch = state.get("epoch", 0)
        global_step = state.get("global_step", 0)
        if global_rank == 0:
            logging.info(
                f"  -> Resuming from epoch {start_epoch}, global step {global_step}"
            )

    optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
    if os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
        if global_rank == 0:
            logging.info("  -> Optimizer state loaded.")

    scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
    if scheduler and os.path.exists(scheduler_path):
        scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))
        if global_rank == 0:
            logging.info("  -> Scheduler state loaded.")

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
) -> List[Dict]:
    """Creates parameter groups with specific LRs and Weight Decay rules."""
    no_decay_keywords = ["bias", "norm"]
    param_groups = []

    def get_params(model, prefixes, invert_prefix=False):
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("_orig_mod."):
                name = name[len("_orig_mod.") :]

            match = any(name.startswith(p) for p in prefixes)
            if invert_prefix:
                match = not match

            if match:
                if any(k in name for k in no_decay_keywords):
                    no_decay.append(param)
                else:
                    decay.append(param)
        return decay, no_decay

    if model_type in ["dual_stream", "sprint_dual"]:
        decay, no_decay = [], []
        for name, param in unet_model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("_orig_mod."):
                name = name[len("_orig_mod.") :]
            if any(k in name for k in no_decay_keywords):
                no_decay.append(param)
            else:
                decay.append(param)

        param_groups.append(
            {
                "params": decay,
                "lr": base_lr,
                "weight_decay": weight_decay,
                "name": "dit_decay",
            }
        )
        param_groups.append(
            {
                "params": no_decay,
                "lr": base_lr,
                "weight_decay": 0.0,
                "name": "dit_no_decay",
            }
        )
    else:
        unet_output_prefixes = ("conv_out.", "conv_norm_out.", "down_blocks.0.")
        unet_high_lr_prefixes = ("time_embedding.", "down_blocks.1.", "down_blocks.2.")
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
            (u_out_d, u_out_nd, base_lr * unet_output_lr_multiplier, "unet_output"),
            (u_high_d, u_high_nd, base_lr * unet_high_lr_multiplier, "unet_high"),
            (u_low_d, u_low_nd, base_lr * unet_low_lr_multiplier, "unet_low"),
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


def create_scheduler(optim, train_loader, conf: omegaconf.DictConfig):
    if not hasattr(train_loader, "__len__"):
        total_steps = conf.train.epochs * 10000
    else:
        update_steps_epoch = len(train_loader) // conf.train.gradient_accumulation_steps
        total_steps = conf.train.epochs * update_steps_epoch

    warmup_steps = int(conf.train.warmup * total_steps)
    warmup_steps = min(warmup_steps, total_steps - 1) if total_steps > 0 else 0

    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=0.001, end_factor=1.0, total_iters=max(1, warmup_steps)
    )

    if conf.train.get("use_cos_scheduler", False):
        logging.info("Using cosine lr scheduler")
        cosine_steps = max(1, total_steps - warmup_steps)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=cosine_steps, eta_min=conf.train.lr * 0.1
        )
    else:
        logging.info("Using constant lr scheduler")
        constant_steps = max(1, total_steps - warmup_steps)
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optim, factor=1.0, total_iters=constant_steps
        )

    if warmup_steps > 0:
        return torch.optim.lr_scheduler.SequentialLR(
            optim, [scheduler_warmup, scheduler], milestones=[warmup_steps]
        )
    return scheduler
