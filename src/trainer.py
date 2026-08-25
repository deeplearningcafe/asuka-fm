import torch
import os
import contextlib
import toml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default
import numpy as np
import random
import wandb
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import logging

from src.diffusion.schedules import LinearSchedule, DDPMSchedule
from src.diffusion.objectives import FlowMatchingObjective, DDPMObjective
from src.diffusion.sampling import generate_samples
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
    patch_torch_compile,
    patch_compiled_autograd,
)


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.dtype = torch.bfloat16 if cfg.train.dtype == "bf16" else torch.float32
        self.setup_device()
        self.train_te = cfg.train.train_te
        self.cfg_dropout_prob = cfg.train.cfg_dropout_prob
        self.model_type = getattr(cfg.models, "model_type", "unet")

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
                model_type=self.model_type,
                model_cfg=cfg.models,
                autocast_dtype=self.autocast_dtype,
            )
        )
        self.dataloader = create_dataloader(
            cfg, self.global_rank, tokenizer=self.tokenizer
        )

        # VAE empirical scaling buffers
        vae_mean = getattr(cfg.models, "vae_mean", 0.0)
        vae_std = getattr(cfg.models, "vae_std", 1.0 / 0.18215)
        self.vae_mean = torch.tensor(
            vae_mean, device=self.device, dtype=self.dtype
        ).view(1, -1, 1, 1)
        self.vae_std = torch.tensor(vae_std, device=self.device, dtype=self.dtype).view(
            1, -1, 1, 1
        )
        self.vae_batch_size = cfg.train.get("vae_batch_size", 64)
        self.in_channels = cfg.models.get("in_channels", 4)

        self.optimizer = create_optim(self.unet, self.text_encoder, cfg)
        self.grad_offloader = None
        # Default to False (VRAM gradient accumulation for small models <1B params)
        self.use_cpu_accumulator = self.cfg.train.get("use_cpu_accumulator", False)
        # only for unet and bf16
        if (
            self.use_cpu_accumulator
            and self.cfg.train.gradient_accumulation_steps > 1
            and not self.train_te
        ):
            self.grad_offloader = CPUGradientAccumulator(self.unet)

        # TODO: how compute total steps with streaming datasets? use compute like nanomagi
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

        self.is_dit = self.model_type in ["dual_stream", "sprint_dual"]
        if cfg.train.objective == "flow_matching":
            # not invert
            self.schedule = LinearSchedule(
                device=self.device,
            )
            timestep_sampling = self.cfg.train.get("timestep_fn", "uniform")

            self.objective = FlowMatchingObjective(
                self.schedule,
                timestep_sampling=timestep_sampling,
                shift=cfg.train.shift,
                use_ot=cfg.train.get("use_ot", False),
                use_unet_mult=False if self.is_dit else True,
            )
        else:
            self.schedule = DDPMSchedule(device=self.device)
            self.objective = DDPMObjective(
                self.schedule, min_snr_gamma=cfg.train.snr_gamma
            )

        if self.is_ddp:
            self.unet = DDP(
                self.unet,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                gradient_as_bucket_view=True,
            )
            if cfg.train.train_te:
                self.text_encoder = DDP(
                    self.text_encoder,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    gradient_as_bucket_view=True,
                )
            # Select DDP communication hook based on autocast precision
            if self.autocast_dtype == torch.bfloat16:
                comm_hook = default.bf16_compress_hook
            elif self.autocast_dtype == torch.float16:
                comm_hook = default.fp16_compress_hook
            else:
                comm_hook = None

            if comm_hook is not None:
                self.unet.register_comm_hook(state=None, hook=comm_hook)
                if cfg.train.train_te:
                    self.text_encoder.register_comm_hook(state=None, hook=comm_hook)

        self.compile_model = cfg.train.get("compile_model", True)
        if hasattr(torch, "compile") and self.compile_model:
            patch_torch_compile()
            patch_compiled_autograd()
            if self.global_rank == 0:
                logging.info("Compiling model and loss step with torch.compile...")
            # TODO: compile text_encoder and vae
            self.unet = torch.compile(self.unet)
            self.text_encoder = torch.compile(self.text_encoder)
            # self.vae = torch.compile(self.vae)
            # if hasattr(self.objective, "_compiled_loss_step"):
            #     self.objective._compiled_loss_step = torch.compile(
            #         self.objective._compiled_loss_step
            #     )

        # Precision & Utils
        self.scaler = (
            torch.cuda.amp.GradScaler() if self.dtype == torch.float16 else None
        )
        self.clip_norm = cfg.train.get("clip_norm", 1.0)

        self.sample_configs = self._load_sample_configs()

        unet_base = self.unet.module if self.is_ddp else self.unet
        self.patch_size = getattr(unet_base, "patch_size", 2)
        self.num_params = sum(
            p.numel() for p in unet_base.parameters() if p.requires_grad
        )
        self.peak_tflops = cfg.train.get("gpu_peak_tflops", 165.2)
        self.flops_factor = 7 if cfg.train.get("use_checkpointing", True) else 6
        if self.global_rank == 0:
            logging.info(f"Training {self.num_params / 1e6}M parameters")

    def setup_device(self):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_ddp = self.world_size > 1

        if self.is_ddp and not dist.is_initialized():
            logging.info(
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
            logging.info(
                f"Training on {self.world_size} GPUs. Precision: {self.cfg.train.dtype}"
            )

        logging.info(f"CUDA version: {torch.version.cuda}")
        capability = torch.cuda.get_device_capability()
        if self.dtype == torch.bfloat16 and capability[0] < 8:
            logging.info(
                f"Warning: bfloat16 specified but GPU capability "
                f"({capability[0]}.{capability[1]}) may not fully support it. "
                f"Consider float16 or float32."
            )

        self.autocast_dtype = torch.bfloat16
        if capability[0] >= 7 and capability[0] < 8:
            self.autocast_dtype = torch.float16
            torch.set_float32_matmul_precision("high")
            logging.info("Using high precision for float32 matmul (tensor cores).")
        elif capability[0] >= 8:
            torch.set_float32_matmul_precision("medium")
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            logging.info("Using half precision for float32 matmul (tensor cores).")
        else:
            logging.info(
                "Tensor cores for float32 matmul not optimally supported "
                "or GPU is older."
            )

        logging.info(f"Using {self.autocast_dtype} for autocast")
        # patch_unsloth_smart_gradient_checkpointing(self.autocast_dtype)

    def _load_sample_configs(self):
        configs = []
        if self.cfg.sampling.config_file and os.path.exists(
            self.cfg.sampling.config_file
        ):
            if self.global_rank == 0:
                logging.info(
                    f"Loading sample configs from {self.cfg.sampling.config_file}"
                )
            with open(self.cfg.sampling.config_file, "r") as f:
                toml_conf = toml.load(f)
                defaults = toml_conf.get("prompt", {})
                subsets = defaults.pop("subset", [])
                for sub in subsets:
                    c = defaults.copy()
                    c.update(sub)
                    configs.append(c)
        return configs

    @torch.no_grad()
    def _encode_vae_latents(self, images: torch.Tensor) -> torch.Tensor:
        """In-place VAE encoding with dynamic batch slicing."""
        latents = []
        chunk_size = self.vae_batch_size
        for i in range(0, images.shape[0], chunk_size):
            chunk = images[i : i + chunk_size].to(self.device, non_blocking=True)
            enc = self.vae.encode(chunk)
            dist = getattr(enc, "latent_dist", enc)
            lat = dist.sample() if hasattr(dist, "sample") else dist
            latents.append(lat)
        lat = torch.cat(latents, dim=0)
        return (lat - self.vae_mean) / self.vae_std

    def train_step(self, batch):
        pos_map = None
        # Check if raw image streaming batch (len >= 5) or precomputed latents
        if len(batch) >= 5:
            images, cond, mask, pos_map, tag_weights, *rest = batch
            torch._dynamo.maybe_mark_dynamic(images, 0)
            with torch.no_grad():
                with torch.autocast(
                    device_type="cuda", dtype=self.autocast_dtype, enabled=True
                ):
                    latents = self._encode_vae_latents(images)
            attention_mask = mask.to(self.device, non_blocking=True)
            pos_map = pos_map.to(self.device, non_blocking=True)
            tag_weights = tag_weights.to(
                self.device, dtype=self.dtype, non_blocking=True
            )
            torch._dynamo.maybe_mark_dynamic(pos_map, 0)
            torch._dynamo.maybe_mark_dynamic(tag_weights, 0)
            # tokenizer pipeline applies CFG
            drop_mask = None
        else:
            latents = (
                batch[0].to(self.device, dtype=self.dtype, non_blocking=True) * 0.18215
            )
            cond = batch[1]
            tag_weights = batch[2].to(self.device, dtype=self.dtype, non_blocking=True)
            attention_mask = batch[3].to(self.device, non_blocking=True)
            bs = latents.shape[0]
            # TODO: cache embeds of cfg
            drop_mask = None
            if self.cfg_dropout_prob > 0:
                drop_mask = torch.rand(bs, device=self.device) < self.cfg_dropout_prob

        # mark as dynamic batch size, not resolution
        torch._dynamo.maybe_mark_dynamic(latents, 0)
        torch._dynamo.maybe_mark_dynamic(cond, 0)
        torch._dynamo.maybe_mark_dynamic(cond, 1)
        torch._dynamo.maybe_mark_dynamic(attention_mask, 0)
        torch._dynamo.maybe_mark_dynamic(attention_mask, 1)

        with torch.autocast(
            device_type="cuda", dtype=self.autocast_dtype, enabled=True
        ):
            with torch.set_grad_enabled(self.train_te):
                encoder_hidden_states, attention_mask = self.text_encoder(
                    cond,
                    mask=attention_mask,
                    drop_mask=drop_mask,
                )
                if drop_mask is not None and drop_mask.any():
                    tag_weights[drop_mask] = 1.0

            loss, metrics = self.objective.forward(
                self.unet,
                latents,
                encoder_hidden_states,
                tag_weights,
                attention_mask=attention_mask,
                pos_map=pos_map,
            )
            scaled_loss = loss / self.cfg.train.gradient_accumulation_steps

        if self.scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return metrics

    def fit(self, timestamp=None):
        # Prepare W&B run name and config
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        current_run_name = f"{self.cfg.logging.wandb_run_name_prefix}-{timestamp}"
        base_save_dir = self.cfg.logging.get("save_dir", "results")
        self.output_dir = os.path.join(base_save_dir, "train", timestamp)
        self.sample_dir = os.path.join(self.output_dir, "samples")
        if self.global_rank == 0:
            os.makedirs(self.sample_dir, exist_ok=True)
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
            tokens_total, last_tokens = 0, 0

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
                epoch_tokens_accum = 0
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

                    # Batch size and token calculations
                    # TODO: clean logic and optimize
                    is_raw_image = len(batch) >= 5
                    raw_bsz = batch[0].shape[0]
                    bsz = raw_bsz * self.world_size
                    batch_size_accum += bsz

                    h_dim, w_dim = batch[0].shape[2], batch[0].shape[3]
                    if is_raw_image:
                        h_lat, w_lat = h_dim // 8, w_dim // 8
                        img_tokens = (h_lat // self.patch_size) * (
                            w_lat // self.patch_size
                        )
                        text_tokens = batch[2].bool().sum().item()
                    else:
                        # Precomputed latents
                        img_tokens = (h_dim // self.patch_size) * (
                            w_dim // self.patch_size
                        )
                        text_tokens = batch[3].bool().sum().item()

                    batch_tokens = (bsz * img_tokens) + (text_tokens * self.world_size)
                    tokens_total += batch_tokens
                    epoch_tokens_accum += batch_tokens

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
                                    self.unet.parameters(), self.clip_norm
                                )
                                if self.train_te:
                                    norm_text_encoder = torch.nn.utils.clip_grad_norm_(
                                        self.text_encoder.parameters(),
                                        max_norm=self.clip_norm / 2.0,
                                    )
                                self.optimizer.step()
                            else:
                                self.scaler.unscale_(self.optimizer)
                                norm_unet = torch.nn.utils.clip_grad_norm_(
                                    self.unet.parameters(), self.clip_norm
                                )
                                if self.train_te:
                                    norm_text_encoder = torch.nn.utils.clip_grad_norm_(
                                        self.text_encoder.parameters(),
                                        max_norm=self.clip_norm / 2.0,
                                    )
                                self.scaler.step(self.optimizer)
                        else:
                            norm_unet = self.grad_offloader.finalize_and_step(
                                self.optimizer,
                                scaler=self.scaler,
                                max_norm=self.clip_norm,
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

                            cpu_avg_grad_loss = avg_grad_loss.detach().item()
                            cpu_norm_unet = norm_unet.detach().item()
                            actual_lr = self.optimizer.param_groups[0]["lr"]

                            # Update progress bar locally (independent of W&B)
                            pbar.set_postfix(
                                {
                                    "loss": f"{cpu_avg_grad_loss:.4f}",
                                    "norm": f"{cpu_norm_unet:.3f}",
                                    "lr": f"{actual_lr:.2e}",
                                }
                            )
                            log_payload = {
                                "train/loss_step": cpu_avg_grad_loss,
                                "train/unscaled_loss_step": avg_grad_unscaled_loss.detach().item(),
                                "train/grad_norm_unet_step": cpu_norm_unet,
                                "learning_rate": actual_lr,
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
                            # if sprint multi images tokens by drop_ratio
                            # TODO: take into account enc and dec layers and drop target
                            images_interval = (
                                images_interval * self.cfg.models.drop_ratio
                                if self.model_type == "sprint_dual"
                                else images_interval
                            )
                            imagesps = images_interval / elapsed

                            tokens_interval = tokens_total - last_tokens
                            last_tokens = tokens_total
                            tokensps = tokens_interval / elapsed

                            cumulative_images += images_interval
                            cumulative_time += elapsed

                            avg_imagesps = (
                                cumulative_images / cumulative_time
                                if cumulative_time > 0
                                else 0
                            )

                            # Total peak TFLOPS across all GPUs in DDP
                            total_peak_tflops = self.peak_tflops * self.world_size

                            model_flops = 6 * self.num_params * tokens_interval
                            hw_flops = (
                                self.flops_factor * self.num_params * tokens_interval
                            )
                            tflops = (model_flops / elapsed) / 1e12
                            hw_tflops = (hw_flops / elapsed) / 1e12

                            mfu = (
                                (tflops / total_peak_tflops) * 100.0
                                if total_peak_tflops > 0
                                else 0.0
                            )
                            hfu = (
                                (hw_tflops / total_peak_tflops) * 100.0
                                if total_peak_tflops > 0
                                else 0.0
                            )

                            if self.global_rank == 0:
                                logging.info(
                                    f"Ep {epoch + 1}, Step {self.global_step:06d}, "
                                    f"Step imgs/sec: {imagesps}, Avg imgs/sec: {avg_imagesps}, "
                                    f"mfu: {mfu}, hfu: {hfu}, tokens/s {tokensps}"
                                )

                                log_payload = {
                                    "train/imagesps": imagesps,
                                    "train/avg_imagesps": avg_imagesps,
                                    "train/tflops": tflops,
                                    "train/mfu": mfu,
                                    "train/hfu": hfu,
                                    "train/tokens_per_sec": tokensps,
                                }
                                log_metrics(
                                    log_payload, step=self.global_step, commit=False
                                )

                                logging.info(
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
                                        use_unet_mult=False if self.is_dit else True,
                                        vae_mean=self.vae_mean,
                                        vae_std=self.vae_std,
                                        in_channels=self.in_channels
                                    )
                                    prompts = [
                                        c.get("prompt") for c in self.sample_configs
                                    ]
                                # Offload upload to thread
                                io_executor.submit(
                                    log_image,
                                    images,
                                    prompts,
                                    epoch,
                                    self.global_step,
                                    False,
                                    self.sample_dir,
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
                logging.info(
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
                logging.info(
                    f"Ep {epoch + 1}, Step {self.global_step:06d}, "
                    f"Step imgs/sec: {imagesps}, Avg imgs/sec: {avg_imagesps}"
                    f"Trained {images_total} images, Tokens {tokens_total}"
                )

                log_payload = {
                    "train/imagesps": imagesps,
                    "train/avg_imagesps": avg_imagesps,
                }
                log_metrics(log_payload, step=self.global_step, commit=False)

                logging.info(f"Generating samples at step {self.global_step}...")
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
                        use_unet_mult=False if self.is_dit else True,
                        vae_mean=self.vae_mean,
                        vae_std=self.vae_std,
                        in_channels=self.in_channels
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
            logging.info("Training Complete.")
            wandb.finish()
