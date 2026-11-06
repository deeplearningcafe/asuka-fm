import copy
import os
import math
import random
import logging
import contextlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.trainer import Trainer
from src.diffusion.sampling import (
    CFGModelWrapper,
    compute_inference_pos_map,
    generate_samples,
)
from src.rl.objectives import (
    FlowGRPOObjective,
    sde_step_with_logprob,
    cps_step_with_logprob,
)
from src.rl.rewards import multi_score
from src.utils.logging import (
    log_metrics,
    log_image,
    init_wandb,
    init_hfapi,
    is_wandb_initialized,
)
from src.utils.checkpointing import save_checkpoint
from src.models.factory import create_scheduler


# from https://github.com/yifan123/flow_grpo/blob/main/scripts/train_sd3.py#L133
def calculate_zero_std_ratio(
    prompts: List[str],
    rewards: np.ndarray,
) -> Tuple[float, float, List[str]]:
    """
    Computes zero reward std ratio across prompt groups and identifies
    prompts with zero variance.
    """
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )
    grouped_rewards = rewards[np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)

    prompt_stds = np.array([float(np.std(g)) for g in reward_groups])
    zero_std_mask = prompt_stds < 1e-6
    zero_std_count = int(np.count_nonzero(zero_std_mask))
    zero_std_ratio = (
        float(zero_std_count / len(prompt_stds)) if len(prompt_stds) > 0 else 0.0
    )
    zero_std_prompts = unique_prompts[zero_std_mask].tolist()
    mean_std = float(prompt_stds.mean()) if len(prompt_stds) > 0 else 0.0

    return zero_std_ratio, mean_std, zero_std_prompts


class RLTrainer(Trainer):
    """
    Flow-GRPO Trainer for Flow Matching Diffusion Models.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.rollout_num_prompts = getattr(cfg.rl, "rollout_num_prompts", 4)
        self.group_num_samples = getattr(cfg.rl, "group_num_samples", 24)
        self.num_sde_steps = getattr(cfg.rl, "num_sde_steps", 10)
        self.noise_level = getattr(cfg.rl, "noise_level", 0.7)
        self.timestep_fraction = getattr(cfg.rl, "timestep_fraction", 0.99)
        self.sde_type = getattr(cfg.rl, "sde_type", "sde").lower()
        self.clip_range = getattr(cfg.rl, "clip_range", 0.2)
        self.adv_clip_max = getattr(cfg.rl, "adv_clip_max", 5.0)
        self.beta = getattr(cfg.rl, "beta", 0.0)
        self.cfg_scale = getattr(cfg.rl, "cfg_scale", 4.5)
        self.num_inner_epochs = getattr(cfg.rl, "num_inner_epochs", 1)
        self.train_batch_size = getattr(
            cfg.train, "batch_size", getattr(cfg.rl, "train_batch_size", 4)
        )
        self.sample_interval = getattr(cfg.sampling, "interval", 2500)
        self.rollout_log_interval = getattr(cfg.logging, "rollout_log_interval", 10)

        # MixGRPO & Flow-GRPO-Fast
        self.window_size = int(getattr(cfg.rl, "window_size", 0))
        self.shift_interval = int(getattr(cfg.rl, "shift_interval", 25))
        self.window_stride = int(getattr(cfg.rl, "window_stride", 1))
        self.window_strategy = getattr(cfg.rl, "window_strategy", "progressive").lower()
        self.sde_window_range = getattr(
            cfg.rl, "sde_window_range", (0, self.num_sde_steps)
        )

        if self.window_size > 0:
            self.num_train_timesteps = self.window_size
        else:
            self.num_train_timesteps = max(
                1, int(self.num_sde_steps * self.timestep_fraction)
            )
        self.dt = 1.0 / self.num_sde_steps

        # Recalibrate LR scheduler
        total_rollout = self.rollout_num_prompts * self.group_num_samples
        batches_per_inner = math.ceil(total_rollout / self.train_batch_size)
        accum_steps = max(1, self.cfg.train.gradient_accumulation_steps)
        updates_per_inner = math.ceil(batches_per_inner / accum_steps)
        updates_per_epoch = updates_per_inner * self.num_inner_epochs

        rl_epochs = getattr(
            self.cfg.rl,
            "epochs",
            getattr(self.cfg.rl, "max_iterations", None),
        )
        if rl_epochs is not None:
            num_rl_epochs = int(rl_epochs)
        else:
            total_epochs = getattr(self.cfg.train, "epochs", 300)
            num_rl_epochs = max(1, total_epochs - self.start_epoch)

        total_rl_steps = getattr(
            self.cfg.rl,
            "max_steps",
            num_rl_epochs * updates_per_epoch,
        )

        self.lr_scheduler = create_scheduler(
            self.optimizer,
            self.dataloader,
            self.cfg,
            total_steps_override=total_rl_steps,
        )

        if self.beta > 0.0:
            raw_unet = self.unet.module if self.is_ddp else self.unet
            if hasattr(raw_unet, "_orig_mod"):
                raw_unet = raw_unet._orig_mod

            self.ref_model = copy.deepcopy(raw_unet).to(self.device)
            self.ref_model.eval()
            self.ref_model.requires_grad_(False)

            if self.compile_model and hasattr(torch, "compile"):
                self.ref_model = torch.compile(self.ref_model)
            if self.global_rank == 0:
                logging.info(
                    f"Initialized frozen reference model for KL penalty "
                    f"(beta={self.beta})."
                )
        else:
            self.ref_model = None

        self.objective = FlowGRPOObjective(
            clip_range=self.clip_range,
            adv_clip_max=self.adv_clip_max,
            beta=self.beta,
            noise_level=self.noise_level,
            sde_type=self.sde_type,
            use_unet_mult=False if self.is_dit else True,
        )

        reward_cfg = cfg.rl.get("reward_config", {})
        self.reward_fn = multi_score(self.device, reward_cfg)

        self.max_seq_len = getattr(self.cfg.data, "max_seq_len", 227)
        self.negative_prompt = getattr(
            self.cfg.rl,
            "negative_prompt",
            "worst quality, low quality, displeasing, bad score, worse score",
        )
        self._prompt_stream = self._prompt_generator()

        if self.global_rank == 0:
            total_rollout = self.rollout_num_prompts * self.group_num_samples
            kl_status = (
                f"Enabled (beta={self.beta})"
                if self.beta > 0.0
                else "Disabled (beta=0.0)"
            )
            logging.info("=" * 60)
            logging.info("Flow-GRPO RL Configuration:")
            logging.info(f"  Rollout Prompts per Epoch : {self.rollout_num_prompts}")
            logging.info(f"  Samples per Prompt (Group): {self.group_num_samples}")
            logging.info(f"  Total Rollout Samples     : {total_rollout}")
            logging.info(
                f"  SDE Steps (Total / Train) : {self.num_sde_steps} / "
                f"{self.num_train_timesteps}"
            )
            logging.info(f"  SDE Noise Level           : {self.noise_level}")
            logging.info(
                f"  Clip Range / Adv Clip Max : {self.clip_range} / {self.adv_clip_max}"
            )
            logging.info(f"  KL Regularization         : {kl_status}")
            logging.info(f"  Rollout CFG Scale         : {self.cfg_scale}")
            logging.info(
                f"  Train Batch / Accum Steps : {self.train_batch_size} / "
                f"{self.cfg.train.gradient_accumulation_steps}"
            )
            logging.info(f"  Evaluation Sample Interval: {self.sample_interval}")
            logging.info("=" * 60)

    def _get_active_window(self, iteration: int) -> Tuple[int, int]:
        """Calculates sliding window interval [l, r) along trajectory."""
        if self.window_size <= 0:
            return 0, self.num_train_timesteps

        max_l = max(0, self.num_sde_steps - self.window_size)
        if self.window_strategy == "random":
            # Flow-GRPO-Fast: randomly sampled window per iteration
            low = max(0, int(self.sde_window_range[0]))
            high = min(max_l, int(self.sde_window_range[1]) - self.window_size)
            l_start = random.randint(low, high) if high >= low else low
        elif self.window_strategy == "frozen":
            # MixGRPO-Frozen: static start at beginning
            l_start = min(getattr(self.cfg.rl, "frozen_start", 0), max_l)
        else:
            # MixGRPO-Progressive: shifts along trajectory
            shift_idx = (iteration - 1) // self.shift_interval
            l_start = min(shift_idx * self.window_stride, max_l)

        return l_start, l_start + self.window_size

    def _prompt_generator(self):
        """
        Streams prompts continuously from the dataset like generate_synthetic.
        """
        is_i1 = self.cfg.data.get("i1_dataset", False)
        target_tiers = self.cfg.data.get(
            "target_aesthetic_tiers",
            self.cfg.rl.get("target_aesthetic_tiers", None),
        )
        if target_tiers is not None:
            target_tiers = set(int(t) for t in target_tiers)

        ds = getattr(self.dataloader, "dataset", None)
        while ds is not None and not hasattr(ds, "iter_raw"):
            if hasattr(ds, "streaming_dataset"):
                ds = ds.streaming_dataset
            elif hasattr(ds, "dataset"):
                ds = ds.dataset
            else:
                break

        if ds is None or not hasattr(ds, "iter_raw"):
            from src.data.streaming_dataset import StreamingImageDataset

            src_ds = self.cfg.data.get(
                "streaming_dataset_name",
                "aipracticecafe/curated-danbooru-2026",
            )
            ds = StreamingImageDataset(
                dataset_name=src_ds,
                dataset_path=self.cfg.data.get("dataset_path", None),
                resolution=getattr(self.cfg.data, "resolution", 512),
                rank=self.global_rank,
                world_size=self.world_size,
                low_ram=True,
                target_aesthetic_tiers=list(target_tiers) if target_tiers else None,
            )

        while True:
            for raw_sample in ds.iter_raw():
                if target_tiers is not None:
                    tier = raw_sample.get("aesthetic_tier")
                    if tier is not None and int(tier) not in target_tiers:
                        continue

                raw_prompt = raw_sample.get("prompt") or raw_sample.get("text", "")
                if isinstance(raw_prompt, (list, tuple)):
                    valid = [str(p).strip() for p in raw_prompt if str(p).strip()]
                    if not valid:
                        continue
                    gen_prompt = valid[0] if is_i1 else ", ".join(valid)
                else:
                    gen_prompt = str(raw_prompt).strip()

                if gen_prompt:
                    yield gen_prompt

    @torch.no_grad()
    def sample_and_evaluate(
        self,
        epoch: int,
        step: int,
        io_executor: Optional[ThreadPoolExecutor] = None,
    ):
        """
        Samples validation images from sample_configs, computes reward metrics,
        and logs preview images asynchronously.
        """
        if self.global_rank != 0 or not self.sample_configs:
            return

        logging.info(f"Generating evaluation samples at step {step}...")
        unet_infer = self.unet.module if self.is_ddp else self.unet
        unet_infer.eval()
        if self.train_te:
            self.text_encoder.eval()

        ema_ctx = (
            self.ema.average_parameters(unet_infer)
            if self.ema is not None and getattr(self.ema, "use_ema", False)
            else contextlib.nullcontext()
        )

        with ema_ctx:
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
                vae_batch_size=1,
                in_channels=self.in_channels,
                pixel_sampling=self.pixel_training,
                prediction_target=self.prediction_target,
                noise_scale=self.noise_scale,
                clip_prediction=getattr(
                    self.cfg.sampling,
                    "clip_prediction",
                    self.pixel_training,
                ),
                dynamic_thresholding=getattr(
                    self.cfg.sampling,
                    "dynamic_thresholding",
                    self.pixel_training,
                ),
                cfg_interval=getattr(
                    self.cfg.sampling,
                    "cfg_interval",
                    (0.0, 0.90) if self.pixel_training else (0.0, 1.0),
                ),
            )

        prompts = [c.get("prompt", "") for c in self.sample_configs]
        num_images = len(images)

        avg_r = np.zeros(num_images, dtype=np.float32)
        tag_r = np.zeros(num_images, dtype=np.float32)
        aes_r = np.zeros(num_images, dtype=np.float32)
        cnt_r = np.zeros(num_images, dtype=np.float32)

        # Bucket images and prompts
        shape_buckets = defaultdict(list)
        for idx, (img, prompt) in enumerate(zip(images, prompts)):
            shape_key = (img.height, img.width)
            shape_buckets[shape_key].append((idx, img, prompt))

        for shape_key, bucket_items in shape_buckets.items():
            b_indices = [item[0] for item in bucket_items]
            b_imgs = [item[1] for item in bucket_items]
            b_prompts = [item[2] for item in bucket_items]
            b_meta = [{"tag_weights": {}} for _ in b_prompts]

            tensor_list = []
            for img in b_imgs:
                arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
                tensor_list.append(torch.from_numpy(arr).permute(2, 0, 1))

            img_tensor = torch.stack(tensor_list, dim=0)
            rewards_dict, _ = self.reward_fn(img_tensor, b_prompts, b_meta)

            b_avg = rewards_dict.get("avg", np.zeros(len(b_imgs)))
            b_tag = rewards_dict.get("tag_alignment", np.zeros(len(b_imgs)))
            b_aes = rewards_dict.get("aesthetic", np.zeros(len(b_imgs)))
            b_cnt = rewards_dict.get("face_count", np.zeros(len(b_imgs)))

            for i, global_idx in enumerate(b_indices):
                avg_r[global_idx] = float(b_avg[i])
                tag_r[global_idx] = float(b_tag[i])
                aes_r[global_idx] = float(b_aes[i])
                cnt_r[global_idx] = float(b_cnt[i])

        eval_metrics = {
            "eval/mean_reward": float(np.mean(avg_r)),
            "eval/mean_tag_alignment": float(np.mean(tag_r)),
            "eval/mean_aesthetic": float(np.mean(aes_r)),
            "eval/mean_face_count": float(np.mean(cnt_r)),
        }
        log_metrics(eval_metrics, step=step, commit=False)
        torch.cuda.empty_cache()

        caption_prompts = [
            f"{p[:240]} | R_avg: {avg_r[i]:.2f} "
            f"(Tag: {tag_r[i]:.2f}, Aes: {aes_r[i]:.2f}, Cnt: {cnt_r[i]:.2f})"
            for i, p in enumerate(prompts)
        ]

        if io_executor is not None:
            io_executor.submit(
                log_image,
                images,
                caption_prompts,
                epoch + 1,
                step,
                False,
                self.sample_dir,
            )
        else:
            log_image(
                images,
                caption_prompts,
                epoch + 1,
                step,
                False,
                self.sample_dir,
            )

        unet_infer.train()
        if self.train_te:
            self.text_encoder.train()

    @torch.no_grad()
    def rollout_sampling(self, iteration: int) -> Dict[str, torch.Tensor]:
        """Generates mixed ODE-SDE rollouts and computes rewards."""
        unet_base = self.unet.module if self.is_ddp else self.unet
        unet_base.eval()
        if self.train_te:
            self.text_encoder.eval()

        l_win, r_win = self._get_active_window(iteration)
        prompts = [next(self._prompt_stream) for _ in range(self.rollout_num_prompts)]
        if self.global_rank == 0:
            logging.info(f"Rollout Prompts Sampled ({len(prompts)}): {prompts[:2]}")

        expanded_prompts = [p for p in prompts for _ in range(self.group_num_samples)]
        num_samples = len(expanded_prompts)

        all_latents, all_next_latents = [], []
        all_log_probs, all_timesteps = [], []
        all_enc_states, all_attn_masks, all_pos_maps = [], [], []
        final_latents_cpu = []

        h = getattr(self.cfg.data, "resolution", 512)
        w = getattr(self.cfg.data, "resolution", 512)
        h_lat, w_lat = h // 8, w // 8

        chunk_size = self.train_batch_size
        for start_idx in range(0, num_samples, chunk_size):
            chunk_prompts = expanded_prompts[start_idx : start_idx + chunk_size]
            curr_bs = len(chunk_prompts)

            is_nl = getattr(self.cfg.data, "i1_dataset", False)
            tokens_list, mask_list = [], []
            for p in chunk_prompts:
                t_tok, t_m = self.tokenizer.encode(
                    p,
                    max_len=self.max_seq_len,
                    cfg_dropout_prob=0.0,
                    tag_dropout_prob=0.0,
                    shuffle_tags=False,
                    is_natural_language=is_nl,
                )
                tokens_list.append(t_tok)
                mask_list.append(t_m)

            tokens = torch.stack(tokens_list).to(self.device)
            mask = torch.stack(mask_list).to(self.device)

            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                enc_states, attn_mask = self.text_encoder(tokens, mask=mask)

            pos_map = (
                compute_inference_pos_map(
                    height=h,
                    width=w,
                    patch_size=self.patch_size,
                    vae_scale=8,
                    coord_system=self.coord_system,
                    device=self.device,
                    dtype=self.dtype,
                )
                .unsqueeze(0)
                .repeat(curr_bs, 1, 1)
            )

            if self.cfg_scale != 1.0:
                u_tok, u_m = self.tokenizer.encode(
                    self.negative_prompt,
                    max_len=self.max_seq_len,
                    cfg_dropout_prob=0.0,
                    tag_dropout_prob=0.0,
                    shuffle_tags=False,
                    is_natural_language=False,
                )
                u_tokens = u_tok.unsqueeze(0).repeat(curr_bs, 1).to(self.device)
                u_mask = u_m.unsqueeze(0).repeat(curr_bs, 1).to(self.device)

                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    u_enc_states, u_attn_mask = self.text_encoder(u_tokens, mask=u_mask)

                combined_embeddings = torch.cat([u_enc_states, enc_states], dim=0)
                combined_attn_mask = torch.cat([u_attn_mask, attn_mask], dim=0)
            else:
                combined_embeddings = enc_states
                combined_attn_mask = attn_mask

            z = torch.randn(
                (curr_bs, self.in_channels, h_lat, w_lat),
                device=self.device,
                dtype=self.dtype,
            )

            wrapper = CFGModelWrapper(
                unet=unet_base,
                combined_embeddings=combined_embeddings,
                cfg_scale=self.cfg_scale,
                device=self.device,
                autocast_dtype=self.autocast_dtype,
                attention_mask=combined_attn_mask,
                use_unet_mult=False if self.is_dit else True,
                pos_map=pos_map,
            )

            step_fn = (
                cps_step_with_logprob
                if self.sde_type == "cps"
                else sde_step_with_logprob
            )

            chunk_lats, chunk_next_lats = [], []
            chunk_lps, chunk_ts = [], []

            z_curr = z
            for step_idx in range(self.num_sde_steps):
                t_val = step_idx * self.dt
                t_tensor = torch.full((curr_bs,), t_val, device=self.device)

                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    v_pred = wrapper(z_curr, t_tensor)

                in_window = l_win <= step_idx < r_win
                cur_noise = self.noise_level if in_window else 0.0

                z_next, log_p, _, _ = step_fn(
                    sample=z_curr,
                    model_output=v_pred,
                    t=t_tensor,
                    dt=self.dt,
                    noise_level=cur_noise,
                )

                if in_window:
                    chunk_lats.append(z_curr.cpu())
                    chunk_next_lats.append(z_next.cpu())
                    chunk_lps.append(log_p.cpu())
                    chunk_ts.append(t_tensor.cpu())

                z_curr = z_next

            final_latents_cpu.append(z_curr.cpu())

            all_latents.append(torch.stack(chunk_lats, dim=1))
            all_next_latents.append(torch.stack(chunk_next_lats, dim=1))
            all_log_probs.append(torch.stack(chunk_lps, dim=1))
            all_timesteps.append(torch.stack(chunk_ts, dim=1))
            all_enc_states.append(enc_states.cpu())
            all_attn_masks.append(attn_mask.cpu())
            all_pos_maps.append(pos_map.cpu())

            del z, wrapper, z_curr, z_next
            torch.cuda.empty_cache()

        vae_bs = getattr(self.cfg.models, "vae_batch_size", 2)
        vae_dtype = next(self.vae.parameters()).dtype
        all_final_lats = torch.cat(final_latents_cpu, dim=0)
        decoded_images_cpu = []

        for v_i in range(0, num_samples, vae_bs):
            lat_chunk = all_final_lats[v_i : v_i + vae_bs].to(self.device)
            raw_lat = (lat_chunk.float() * self.vae_std) + self.vae_mean
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                dec = self.vae.decode(raw_lat.to(vae_dtype)).sample
            dec = (dec.float() / 2.0 + 0.5).clamp(0, 1)
            decoded_images_cpu.append(dec.cpu())
            del lat_chunk, raw_lat, dec
            torch.cuda.empty_cache()

        images = torch.cat(decoded_images_cpu, dim=0)
        meta = [{"tag_weights": {}} for _ in expanded_prompts]

        rewards_dict, _ = self.reward_fn(images, expanded_prompts, meta)
        raw_rewards = torch.as_tensor(rewards_dict["avg"], dtype=torch.float32)

        raw_rewards_np = raw_rewards.cpu().numpy()
        zero_std_ratio, mean_reward_std, zero_prompts = calculate_zero_std_ratio(
            expanded_prompts, raw_rewards_np
        )

        if zero_prompts and self.global_rank == 0:
            logging.info(
                f"[RL] Zero reward std detected on {len(zero_prompts)} "
                f"prompts: {zero_prompts[:2]}"
            )
        # Prompt-grouped advantage normalization
        advantages = torch.zeros_like(raw_rewards)
        for i in range(len(prompts)):
            s_i = i * self.group_num_samples
            e_i = s_i + self.group_num_samples
            grp = raw_rewards[s_i:e_i]
            std = grp.std()
            advantages[s_i:e_i] = (grp - grp.mean()) / (std + 1e-4)

        adv_matrix = advantages.unsqueeze(1).repeat(1, self.num_train_timesteps)
        torch.cuda.empty_cache()

        return {
            "latents": torch.cat(all_latents, dim=0),
            "next_latents": torch.cat(all_next_latents, dim=0),
            "log_probs": torch.cat(all_log_probs, dim=0),
            "timesteps": torch.cat(all_timesteps, dim=0),
            "advantages": adv_matrix,
            "encoder_hidden_states": torch.cat(all_enc_states, dim=0),
            "attention_mask": torch.cat(all_attn_masks, dim=0),
            "pos_map": torch.cat(all_pos_maps, dim=0),
            "rewards": raw_rewards,
            "zero_std_ratio": zero_std_ratio,
            "mean_reward_std": mean_reward_std,
            "preview_images": images[: min(4, len(images))],
            "preview_prompts": expanded_prompts[: min(4, len(images))],
        }

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        j: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Executes a single policy gradient update step on timestep j.
        """
        lats = batch["latents"][:, j].to(self.device, non_blocking=True)
        nlats = batch["next_latents"][:, j].to(self.device, non_blocking=True)
        ts = batch["timesteps"][:, j].to(self.device, non_blocking=True)
        old_lp = batch["log_probs"][:, j].to(self.device, non_blocking=True)
        adv = batch["advantages"][:, j].to(self.device, non_blocking=True)
        cond = batch["encoder_hidden_states"].to(self.device, non_blocking=True)
        mask = batch["attention_mask"].to(self.device, non_blocking=True)
        pos = batch["pos_map"].to(self.device, non_blocking=True)

        with torch.autocast(
            device_type="cuda", dtype=self.autocast_dtype, enabled=True
        ):
            loss, metrics = self.objective(
                model=self.unet,
                latents=lats,
                next_latents=nlats,
                timesteps=ts,
                dt=self.dt,
                old_log_probs=old_lp,
                advantages=adv,
                condition=cond,
                attention_mask=mask,
                pos_map=pos,
                ref_model=self.ref_model,
            )
            total_accum = (
                self.cfg.train.gradient_accumulation_steps * self.num_train_timesteps
            )
            scaled_loss = loss / total_accum

        if self.scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return metrics

    def fit(self, timestamp: Optional[str] = None):
        """Main Flow-GRPO outer optimization loop."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        current_run_name = f"{self.cfg.logging.wandb_run_name_prefix}-{timestamp}"
        base_save_dir = self.cfg.logging.get("save_dir", "results")
        self.output_dir = os.path.join(base_save_dir, "rl", timestamp)
        self.sample_dir = os.path.join(self.output_dir, "samples")

        if self.global_rank == 0:
            os.makedirs(self.sample_dir, exist_ok=True)
            cfg_dict = {
                **OmegaConf.to_container(self.cfg.logging),
                **OmegaConf.to_container(self.cfg.data),
                **OmegaConf.to_container(self.cfg.train),
                **OmegaConf.to_container(self.cfg.models),
                **OmegaConf.to_container(self.cfg.sampling),
                **OmegaConf.to_container(self.cfg.rl),
            }
            init_wandb(
                self.cfg.logging.project,
                run_name=current_run_name,
                entity=self.cfg.logging.wandb_entity,
                config=cfg_dict,
            )
            init_hfapi()

        if is_wandb_initialized():
            wandb.define_metric("epoch")
            wandb.define_metric("train/iteration")
            wandb.define_metric("train/image_log_step")
            wandb.define_metric("epoch_samples/*", step_metric="train/image_log_step")

        dataset_len = getattr(self.dataloader.dataset, "total_samples", 20000)
        total_epochs = getattr(self.cfg.train, "epochs", 300)
        prompts_seen = 0

        with ThreadPoolExecutor(max_workers=1) as io_executor:
            for epoch in range(self.start_epoch, total_epochs):
                iteration = epoch - self.start_epoch + 1
                prompts_seen += self.rollout_num_prompts
                prompt_epoch_float = prompts_seen / max(1, dataset_len)
                l_win, r_win = self._get_active_window(iteration)

                if self.global_rank == 0:
                    logging.info(
                        f"--- Epoch {epoch + 1}/{total_epochs} "
                        f"[Coverage: {prompt_epoch_float:.3f} | "
                        f"Window: [{l_win}:{r_win})] ---"
                    )

                rollout_data = self.rollout_sampling(iteration)

                preview_images = rollout_data.pop("preview_images", None)
                preview_prompts = rollout_data.pop("preview_prompts", None)
                zero_std_ratio = rollout_data.pop("zero_std_ratio", 0.0)
                mean_reward_std = rollout_data.pop("mean_reward_std", 0.0)

                if (
                    self.global_rank == 0
                    and is_wandb_initialized()
                    and iteration % self.rollout_log_interval == 0
                ):
                    rollout_log = {
                        "train/rollout_mean_reward": (
                            rollout_data["rewards"].mean().item()
                        ),
                        "train/zero_std_ratio": zero_std_ratio,
                        "train/reward_std_mean": mean_reward_std,
                        "epoch": epoch + 1,
                    }
                    if preview_images is not None and preview_prompts is not None:
                        preview_wandb = [
                            wandb.Image(
                                img.float().permute(1, 2, 0).numpy(),
                                caption=f"Epoch {epoch + 1}: {p[:100]}",
                            )
                            for img, p in zip(preview_images, preview_prompts)
                        ]
                        rollout_log["train/rollout_previews"] = preview_wandb

                    log_metrics(rollout_log, step=self.global_step, commit=False)

                self.unet.train()
                if self.train_te:
                    self.text_encoder.train()

                total_samples = rollout_data["latents"].shape[0]

                for inner in range(self.num_inner_epochs):
                    perm = torch.randperm(total_samples, device="cpu")
                    shuffled = {
                        k: v[perm]
                        for k, v in rollout_data.items()
                        if isinstance(v, torch.Tensor) and v.shape[0] == total_samples
                    }

                    batch_slices = list(range(0, total_samples, self.train_batch_size))
                    accum_steps = self.cfg.train.gradient_accumulation_steps

                    pbar = tqdm(
                        batch_slices,
                        desc=f"Epoch {epoch + 1}.{inner + 1} Training",
                        disable=self.global_rank != 0,
                        leave=False,
                        dynamic_ncols=True,
                    )

                    self.optimizer.zero_grad(set_to_none=True)

                    for idx, b_idx in enumerate(pbar):
                        b_slice = {
                            k: v[b_idx : b_idx + self.train_batch_size]
                            for k, v in shuffled.items()
                        }
                        is_accum_step = (idx + 1) % accum_steps == 0 or (
                            idx + 1
                        ) == len(batch_slices)

                        for j in range(self.num_train_timesteps):
                            sync_context = (
                                self.unet.no_sync()
                                if self.is_ddp
                                and not (
                                    is_accum_step and j == self.num_train_timesteps - 1
                                )
                                else contextlib.nullcontext()
                            )
                            with sync_context:
                                step_metrics = self.train_step(b_slice, j)

                        if is_accum_step:
                            if self.scaler:
                                self.scaler.unscale_(self.optimizer)
                                norm = torch.nn.utils.clip_grad_norm_(
                                    self.unet.parameters(), self.clip_norm
                                )
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                norm = torch.nn.utils.clip_grad_norm_(
                                    self.unet.parameters(), self.clip_norm
                                )
                                self.optimizer.step()

                            self.optimizer.zero_grad(set_to_none=True)
                            if self.lr_scheduler:
                                self.lr_scheduler.step()
                            if self.ema:
                                base = self.unet.module if self.is_ddp else self.unet
                                self.ema.update(base)

                            # Increment global_step after optimizer step
                            self.global_step += 1

                            if self.global_rank == 0:
                                pbar.set_postfix(
                                    {
                                        "step": f"{self.global_step}",
                                        "loss": f"{step_metrics['loss']:.4f}",
                                        "kl": f"{step_metrics['kl_loss']:.4f}",
                                        "norm": f"{norm:.3f}",
                                    }
                                )

                                if is_wandb_initialized():
                                    log_payload = {
                                        "train/loss": (step_metrics["loss"].item()),
                                        "train/policy_loss": (
                                            step_metrics["policy_loss"].item()
                                        ),
                                        "train/kl_loss": (
                                            step_metrics["kl_loss"].item()
                                        ),
                                        "train/approx_kl": (
                                            step_metrics["approx_kl"].item()
                                        ),
                                        "train/clipfrac": (
                                            step_metrics["clipfrac"].item()
                                        ),
                                        "train/ratio_mean": (
                                            step_metrics["ratio_mean"].item()
                                        ),
                                        "train/adv_mean": (
                                            step_metrics["adv_mean"].item()
                                        ),
                                        "train/grad_norm": (
                                            norm.item()
                                            if isinstance(norm, torch.Tensor)
                                            else float(norm)
                                        ),
                                        "train/iteration": iteration,
                                        "epoch": epoch + 1,
                                        "learning_rate": (
                                            self.optimizer.param_groups[0]["lr"]
                                        ),
                                    }
                                    log_metrics(
                                        log_payload,
                                        step=self.global_step,
                                        commit=False,
                                    )

                                if (
                                    self.sample_configs
                                    and self.global_step % self.sample_interval == 0
                                ):
                                    self.sample_and_evaluate(
                                        epoch,
                                        self.global_step,
                                        io_executor=io_executor,
                                    )

                save_interval = getattr(self.cfg.logging, "save_iteration_interval", 50)
                if iteration % save_interval == 0 and self.global_rank == 0:
                    u_save = self.unet.module if self.is_ddp else self.unet
                    t_save = (
                        self.text_encoder.module
                        if self.is_ddp and self.train_te
                        else self.text_encoder
                    )
                    save_checkpoint(
                        epoch=epoch + 1,
                        global_step=self.global_step,
                        unet=u_save,
                        text_encoder=t_save,
                        optimizer=self.optimizer,
                        scheduler=self.lr_scheduler,
                        train_te=self.train_te,
                        hf_repo=self.cfg.logging.hf_repo,
                        ema=self.ema,
                        config=self.cfg,
                    )

        if self.global_rank == 0 and is_wandb_initialized():
            wandb.finish()
