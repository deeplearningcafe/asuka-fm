import torch
import os
import contextlib
import toml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import random
import wandb
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

from src.diffusion.schedules import LinearSchedule, DDPMSchedule
from src.diffusion.objectives import FlowMatchingObjective, DDPMObjective
from src.diffusion.sampling import generate_samples, encode_tokens_batch
from src.models.factory import (
    load_trainable_model,
    create_optim,
    create_scheduler,
    load_training_state,
    ModelInspector,
)
from src.data.loader import create_dataloader
from src.utils.logging import (
    log_metrics,
    log_image,
    init_wandb,
    init_hfapi,
    is_wandb_initialized,
)
from src.utils.checkpointing import save_checkpoint
from src.utils.act_grad_checkpointing import (
    patch_unsloth_smart_gradient_checkpointing,
    CPUGradientAccumulator,
)


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.dtype = torch.bfloat16 if cfg.train.dtype == "bf16" else torch.float32
        self.setup_device()
        self.train_te = cfg.train.train_te
        self.cfg_dropout_prob = cfg.train.cfg_dropout_prob

        self.dataloader = create_dataloader(cfg, self.global_rank)

        self.unet, self.text_encoder, self.vae, self.tokenizer, self.ema = (
            load_trainable_model(
                models_path=cfg.paths.models,
                device=self.device,
                dtype=self.dtype,
                train_te=cfg.train.train_te,
                use_checkpointing=cfg.train.use_checkpointing,
                resume_from_checkpoint=cfg.models.resume_from_checkpoint,
                train_only_output=cfg.train.train_only_output,
                output_head_path=cfg.models.output_head_path,
                use_ema=cfg.train.get("use_ema", True),
                ema_decay=cfg.train.get("ema_decay", 0.99),
                global_rank=self.global_rank,
            )
        )

        self.optimizer = create_optim(self.unet, self.text_encoder, cfg)
        self.grad_offloader = None
        # only for unet and bf16
        if self.cfg.train.gradient_accumulation_steps > 1 and not self.train_te:
            self.grad_offloader = CPUGradientAccumulator(self.unet)
        self.lr_scheduler = create_scheduler(self.optimizer, self.dataloader, cfg)

        # Resume Training State (Epoch, Step, Optim State)
        self.optimizer, self.lr_scheduler, self.start_epoch, self.global_step = (
            load_training_state(
                cfg.models.resume_from_checkpoint,
                self.optimizer,
                self.lr_scheduler,
                self.device,
                self.global_rank,
            )
        )

        if self.ema is not None and self.ema.use_ema:
            self.ema.step = self.global_step

        if cfg.train.objective == "flow_matching":
            self.schedule = LinearSchedule(device=self.device)
            self.objective = FlowMatchingObjective(
                self.schedule,
                shift=cfg.train.shift,
                use_ot=cfg.train.get("use_ot", False),
            )
        else:
            self.schedule = DDPMSchedule(device=self.device)
            self.objective = DDPMObjective(
                self.schedule, min_snr_gamma=cfg.train.snr_gamma
            )

        if self.is_ddp:
            self.unet = DDP(
                self.unet, device_ids=[self.local_rank], output_device=self.local_rank
            )
            if cfg.train.train_te:
                self.text_encoder = DDP(
                    self.text_encoder,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                )

        # Precision & Utils
        self.dtype = torch.bfloat16 if cfg.train.dtype == "bf16" else torch.float32
        self.scaler = (
            torch.cuda.amp.GradScaler() if self.dtype == torch.float16 else None
        )

        # Precompute unconditional embeddings for CFG
        self.uncond_tokens_dict = (
            self._precompute_uncond() if cfg.train.cfg_dropout_prob > 0 else {}
        )

        self.sample_configs = self._load_sample_configs()

    def setup_device(self):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_ddp = self.world_size > 1

        if self.is_ddp and not dist.is_initialized():
            print(
                f"Initializing DDP: Rank {self.global_rank}/{self.world_size}, Local Rank {self.local_rank}"
            )
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)

        self.device = torch.device(f"cuda:{self.local_rank}")
        # Set seeds: Offset by rank to ensure different noise on different GPUs
        current_seed = self.cfg.train.seed + self.global_rank
        torch.manual_seed(current_seed)
        random.seed(current_seed)
        np.random.seed(current_seed)

        if self.global_rank == 0:
            print(
                f"Training on {self.world_size} GPUs. Precision: {self.cfg.train.dtype}"
            )

        print(f"CUDA version: {torch.version.cuda}")
        capability = torch.cuda.get_device_capability()
        if self.dtype == torch.bfloat16 and capability[0] < 8:
            print(
                f"Warning: bfloat16 specified but GPU capability "
                f"({capability[0]}.{capability[1]}) may not fully support it. "
                f"Consider float16 or float32."
            )

        self.autocast_dtype = torch.bfloat16
        if capability[0] >= 7 and capability[0] < 8:
            self.autocast_dtype = torch.float16
            torch.set_float32_matmul_precision("high")
            print("Using high precision for float32 matmul (tensor cores).")
        elif capability[0] >= 8:
            torch.set_float32_matmul_precision("medium")
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            print("Using half precision for float32 matmul (tensor cores).")
        else:
            print(
                "Tensor cores for float32 matmul not optimally supported "
                "or GPU is older."
            )

        print(f"Using {self.autocast_dtype} for autocast")
        patch_unsloth_smart_gradient_checkpointing(self.autocast_dtype)

    def _load_sample_configs(self):
        configs = []
        if self.cfg.sampling.config_file and os.path.exists(
            self.cfg.sampling.config_file
        ):
            if self.global_rank == 0:
                print(f"Loading sample configs from {self.cfg.sampling.config_file}")
            with open(self.cfg.sampling.config_file, "r") as f:
                toml_conf = toml.load(f)
                defaults = toml_conf.get("prompt", {})
                subsets = defaults.pop("subset", [])
                for sub in subsets:
                    c = defaults.copy()
                    c.update(sub)
                    configs.append(c)
        return configs

    def _precompute_uncond(self):
        # Helper to compute empty prompt embeddings for CFG
        uncond_dict = {}
        for tier in [77, 152, 227]:
            tokens = self.tokenizer(
                [""],
                padding="max_length",
                max_length=tier,
                truncation=True,
                return_tensors="pt",
            ).input_ids

            with torch.autocast(
                device_type="cuda", dtype=self.autocast_dtype, enabled=True
            ):
                with torch.no_grad():
                    te = (
                        self.text_encoder.module
                        if isinstance(self.text_encoder, DDP)
                        else self.text_encoder
                    )
                    embeds = encode_tokens_batch(
                        tokens.to(self.device), te, self.tokenizer, tier, self.device
                    )
                    uncond_dict[tier] = embeds.detach()
        return uncond_dict

    def train_step(self, batch):
        latents = (
            batch[0].to(self.device, dtype=self.dtype, non_blocking=True) * 0.18215
        )
        # this are the tokens not the embeddings
        cond = batch[1].to(self.device, non_blocking=True)
        tag_weights = batch[2].to(self.device, dtype=self.dtype, non_blocking=True)
        attention_mask = batch[3].to(self.device, non_blocking=True)

        with torch.autocast(
            device_type="cuda", dtype=self.autocast_dtype, enabled=True
        ):
            with torch.set_grad_enabled(self.train_te):
                te_model = (
                    self.text_encoder.module
                    if isinstance(self.text_encoder, DDP)
                    else self.text_encoder
                )
                encoder_hidden_states = encode_tokens_batch(
                    cond,
                    te_model,
                    self.tokenizer,
                    max_length=cond.shape[-1],
                    device=self.device,
                )

                # replace cond embeds with the pre-computed uncond embed
                tier = encoder_hidden_states.shape[1]

                if self.cfg_dropout_prob > 0:
                    # random mask for dropping prompts
                    bs = encoder_hidden_states.shape[0]
                    drop_mask = (
                        torch.rand(bs, device=self.device) < self.cfg_dropout_prob
                    )

                    uncond_to_use = self.uncond_tokens_dict[tier].expand(bs, tier, -1)

                    encoder_hidden_states[drop_mask] = uncond_to_use[drop_mask]
                    tag_weights[drop_mask] = 1.0

                    # empty prompt only has 2 valid tokens: BOS and EOS.
                    # mask out the rest to prevent padding dilution
                    empty_prompt_mask = torch.zeros(
                        tier, dtype=torch.bool, device=self.device
                    )
                    empty_prompt_mask[0:2] = True
                    attention_mask[drop_mask] = empty_prompt_mask

            loss, metrics = self.objective.forward(
                self.unet,
                latents,
                encoder_hidden_states,
                tag_weights,
                attention_mask=attention_mask,
            )
            scaled_loss = loss / self.cfg.train.gradient_accumulation_steps

        if self.scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        del scaled_loss

        return metrics

    def fit(self):
        # Prepare W&B run name and config
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        current_run_name = f"{self.cfg.logging.wandb_run_name_prefix}-{timestamp}"
        if self.global_rank == 0:
            config = {
                **OmegaConf.to_container(self.cfg.logging),
                **OmegaConf.to_container(self.cfg.data),
                **OmegaConf.to_container(self.cfg.train),
                **OmegaConf.to_container(self.cfg.models),
                **OmegaConf.to_container(self.cfg.sampling),
            }
            init_wandb(
                self.cfg.logging.project,
                run_name=current_run_name,
                entity=self.cfg.logging.wandb_entity,
                config=config,
            )
            init_hfapi()

        if is_wandb_initialized():
            wandb.define_metric("epoch")
            wandb.define_metric("train/image_log_step")

            wandb.define_metric("train/avg_epoch_loss", step_metric="epoch")

            wandb.define_metric("epoch_samples/*", step_metric="train/image_log_step")

        inspector = None
        if self.cfg.train.run_stability_check and self.global_rank == 0:
            inspector = ModelInspector(logging_fn=log_metrics, model_dtype=self.dtype)
            # Unwrap DDP for inspection
            unet_inspect = self.unet.module if self.is_ddp else self.unet
            inspector.register_hooks(unet_inspect)

        self.unet.train()

        # Thread pool for async image logging
        with ThreadPoolExecutor(max_workers=1) as io_executor:
            cumulative_images, cumulative_time = 0.0, 0.0
            images_total, last_images = 0, 0

            # CUDA-specific timing setup
            t_start = torch.cuda.Event(enable_timing=True)
            t_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            t_start.record()

            for epoch in range(self.start_epoch, self.cfg.train.epochs):
                if self.is_ddp and hasattr(self.dataloader.sampler, "set_epoch"):
                    self.dataloader.sampler.set_epoch(epoch)

                self.unet.train()
                if self.train_te:
                    self.text_encoder.train()
                epoch_loss_accum = torch.tensor(
                    [0.0], device=self.device, dtype=self.dtype
                )
                grad_loss_accum = torch.tensor(
                    [0.0], device=self.device, dtype=self.dtype
                )
                grad_unscaled_loss_accum = torch.tensor(
                    [0.0], device=self.device, dtype=self.dtype
                )

                batch_size_accum = 0
                num_batches_epoch = 0
                pbar = tqdm(
                    self.dataloader,
                    desc=f"Epoch {epoch + 1}/{self.cfg.train.epochs}",
                    leave=False,
                    disable=self.global_rank != 0,
                )

                self.optimizer.zero_grad(set_to_none=True)
                for step, batch in enumerate(pbar):
                    # DDP Context: Only sync gradients on the accumulation step
                    is_update_step = (
                        step + 1
                    ) % self.cfg.train.gradient_accumulation_steps == 0

                    if self.is_ddp:
                        # ddp sync is done in the grad accumulator
                        if self.grad_offloader is not None:
                            context = self.unet.no_sync()
                        elif not is_update_step:
                            context = self.unet.no_sync()
                        else:
                            context = contextlib.nullcontext()
                    else:
                        context = contextlib.nullcontext()

                    batch_size_accum += batch[0].shape[0] * self.world_size

                    with context:
                        metrics = self.train_step(batch)
                        grad_loss_accum += metrics["loss"]
                        grad_unscaled_loss_accum += metrics["raw_loss"]
                        num_batches_epoch += 1

                    if is_update_step:
                        if inspector:
                            inspector.log_stats(self.global_step)

                        norm_text_encoder = None
                        if self.grad_offloader is None:
                            if self.scaler is None:
                                norm_unet = torch.nn.utils.clip_grad_norm_(
                                    self.unet.parameters(), 2.0
                                )
                                if self.train_te:
                                    norm_text_encoder = torch.nn.utils.clip_grad_norm_(
                                        self.text_encoder.parameters(), max_norm=1.0
                                    )
                                self.optimizer.step()
                            else:
                                self.scaler.unscale_(self.optimizer)
                                norm_unet = torch.nn.utils.clip_grad_norm_(
                                    self.unet.parameters(), 2.0
                                )
                                if self.train_te:
                                    norm_text_encoder = torch.nn.utils.clip_grad_norm_(
                                        self.text_encoder.parameters(),
                                        max_norm=1.0,
                                    )
                                self.scaler.step(self.optimizer)
                        else:
                            norm_unet = self.grad_offloader.finalize_and_step(
                                self.optimizer, scaler=self.scaler, max_norm=2.0
                            )

                        self.lr_scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)

                        # Update EMA using the base UNet (unwrapped from DDP)
                        unet_base = self.unet.module if self.is_ddp else self.unet
                        self.ema.update(unet_base)

                        # only update global step when params updated
                        self.global_step += 1

                        if self.global_rank == 0:
                            avg_grad_loss = (
                                grad_loss_accum
                                / self.cfg.train.gradient_accumulation_steps
                            )
                            avg_grad_unscaled_loss = (
                                grad_unscaled_loss_accum
                                / self.cfg.train.gradient_accumulation_steps
                            )

                            log_payload = {
                                "train/loss_step": avg_grad_loss.detach().item(),
                                "train/unscaled_loss_step": avg_grad_unscaled_loss.detach().item(),
                                "train/grad_norm_unet_step": norm_unet.detach().item(),
                                "learning_rate": self.optimizer.param_groups[0]["lr"],
                                "batch_size": batch_size_accum,
                            }
                            if norm_text_encoder:
                                log_payload["train/grad_norm_text_encoder_step"] = (
                                    norm_text_encoder.detach().item()
                                )
                            for i, pg in enumerate(self.optimizer.param_groups):
                                log_payload[f"lr_group_{pg.get('name', i)}"] = pg["lr"]
                            log_payload["train/pred_norm"] = metrics["pred_norm"].item()
                            log_payload["train/target_norm"] = metrics[
                                "target_norm"
                            ].item()
                            log_payload["train/pred_mean_abs"] = metrics[
                                "pred_mean_abs"
                            ].item()
                            log_payload["train/target_mean_abs"] = metrics[
                                "target_mean_abs"
                            ].item()
                            log_metrics(
                                log_payload, step=self.global_step, commit=False
                            )

                        images_total += batch_size_accum
                        # Reset the accumulator
                        epoch_loss_accum += grad_loss_accum
                        grad_loss_accum = torch.tensor(
                            [0.0], device=self.device, dtype=self.dtype
                        )
                        grad_unscaled_loss_accum = torch.tensor(
                            [0.0], device=self.device, dtype=self.dtype
                        )
                        batch_size_accum = 0

                        if self.global_step % self.cfg.logging.save_interval == 0:
                            if self.global_rank == 0:
                                num_saves = (
                                    self.global_step // self.cfg.logging.save_interval
                                )
                                if num_saves > self.cfg.train.skip_save_n_times:
                                    unet_save = (
                                        self.unet.module if self.is_ddp else self.unet
                                    )
                                    te_save = (
                                        self.text_encoder.module
                                        if self.is_ddp and self.train_te
                                        else self.text_encoder
                                    )
                                    save_checkpoint(
                                        epoch,
                                        self.global_step,
                                        unet_save,
                                        te_save,
                                        self.optimizer,
                                        self.lr_scheduler,
                                        self.train_te,
                                        self.cfg.logging.hf_repo,
                                        train_only_output=(
                                            self.cfg.train.train_only_output
                                        ),
                                        ema=self.ema,
                                    )

                        # Sampling
                        if self.sample_configs and (
                            self.global_step % self.cfg.sampling.interval == 0
                        ):
                            t_end.record()
                            torch.cuda.synchronize()
                            elapsed = t_start.elapsed_time(t_end) / 1000

                            images_interval = images_total - last_images
                            last_images = images_total
                            imagesps = images_interval / elapsed if elapsed > 0 else 0

                            cumulative_images += images_interval
                            cumulative_time += elapsed

                            avg_imagesps = (
                                cumulative_images / cumulative_time
                                if cumulative_time > 0
                                else 0
                            )
                            if self.global_rank == 0:
                                print(
                                    f"Ep {epoch + 1}, Step {self.global_step:06d}, "
                                    f"Step imgs/sec: {imagesps}, Avg imgs/sec: {avg_imagesps}"
                                )

                                log_payload = {
                                    "train/imagesps": imagesps,
                                    "train/avg_imagesps": avg_imagesps,
                                }
                                log_metrics(
                                    log_payload, step=self.global_step, commit=False
                                )

                                print(
                                    f"Generating samples at step {self.global_step}..."
                                )
                                unet_infer = (
                                    self.unet.module if self.is_ddp else self.unet
                                )
                                del batch

                                # Use EMA weights for evaluation
                                with self.ema.average_parameters(unet_infer):
                                    images = generate_samples(
                                        unet=unet_infer,
                                        text_encoder=self.text_encoder,
                                        tokenizer=self.tokenizer,
                                        vae=self.vae,
                                        schedule=self.schedule,
                                        sample_configs=self.sample_configs,
                                        diffusion_type=self.cfg.train.objective,
                                        device=self.device,
                                        dtype=self.dtype,
                                    )
                                    prompts = [
                                        c.get("prompt") for c in self.sample_configs
                                    ]
                                # Offload upload to thread
                                io_executor.submit(
                                    log_image, images, prompts, epoch, self.global_step
                                )
                                self.unet.train()
                                if self.train_te:
                                    self.text_encoder.train()

                            t_start.record()

            # End of Epoch
            avg_epoch_loss = (
                (epoch_loss_accum / num_batches_epoch).item()
                if num_batches_epoch > 0
                else float("inf")
            )
            if self.global_rank == 0:
                print(
                    f"End of Epoch {epoch + 1}/{self.cfg.train.epochs}: Avg Train Loss: {avg_epoch_loss:.4f}"
                )
                log_metrics(
                    {
                        "train/avg_epoch_loss": avg_epoch_loss,
                        "epoch": epoch,
                    },
                    step=self.global_step,
                    commit=False,
                )

            t_end.record()
            torch.cuda.synchronize()
            elapsed = t_start.elapsed_time(t_end) / 1000

            images_interval = images_total - last_images
            last_images = images_total
            imagesps = images_interval / elapsed if elapsed > 0 else 0

            cumulative_images += images_interval
            cumulative_time += elapsed

            avg_imagesps = (
                cumulative_images / cumulative_time if cumulative_time > 0 else 0
            )
            if self.global_rank == 0:
                print(
                    f"Ep {epoch + 1}, Step {self.global_step:06d}, "
                    f"Step imgs/sec: {imagesps}, Avg imgs/sec: {avg_imagesps}"
                )

                log_payload = {
                    "train/imagesps": imagesps,
                    "train/avg_imagesps": avg_imagesps,
                }
                log_metrics(log_payload, step=self.global_step, commit=False)

                print(f"Generating samples at step {self.global_step}...")
                unet_infer = self.unet.module if self.is_ddp else self.unet
                del batch

                with self.ema.average_parameters(unet_infer):
                    images = generate_samples(
                        unet=unet_infer,
                        text_encoder=self.text_encoder,
                        tokenizer=self.tokenizer,
                        vae=self.vae,
                        sample_configs=self.sample_configs,
                        diffusion_type=self.cfg.train.objective,
                        device=self.device,
                        dtype=self.dtype,
                    )
                prompts = [c.get("prompt") for c in self.sample_configs]
                io_executor.submit(log_image, images, prompts, epoch, self.global_step)

                unet_save = self.unet.module if self.is_ddp else self.unet
                save_checkpoint(
                    epoch,
                    self.global_step,
                    unet_save,
                    self.text_encoder,
                    self.optimizer,
                    self.lr_scheduler,
                    self.train_te,
                    self.cfg.logging.hf_repo,
                    train_only_output=self.cfg.train.train_only_output,
                    ema=self.ema,
                )

                self.unet.train()
                if self.train_te:
                    self.text_encoder.train()

                t_start.record()

        # Cleanup DDP
        if self.world_size > 1:
            dist.destroy_process_group()
        if inspector:
            inspector.remove_hooks()
        if self.global_rank == 0:
            print("Training Complete.")
            wandb.finish()
