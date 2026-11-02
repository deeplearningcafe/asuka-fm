import os
import math
import random
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from tqdm import tqdm

from src.trainer import Trainer
from src.diffusion.sampling import (
    CFGModelWrapper,
    compute_inference_pos_map,
)
from src.rl.objectives import FlowGRPOObjective, sde_step_with_logprob
from src.rl.rewards import multi_score
from src.utils.logging import log_metrics, is_wandb_initialized
from src.utils.checkpointing import save_checkpoint


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
        self.clip_range = getattr(cfg.rl, "clip_range", 0.2)
        self.adv_clip_max = getattr(cfg.rl, "adv_clip_max", 5.0)
        self.beta = getattr(cfg.rl, "beta", 0.0)
        self.cfg_scale = getattr(cfg.rl, "cfg_scale", 4.5)
        self.num_inner_epochs = getattr(cfg.rl, "num_inner_epochs", 1)
        self.train_batch_size = getattr(
            cfg.train, "batch_size", getattr(cfg.rl, "train_batch_size", 4)
        )

        # Avoid degenerate zero-variance at final boundary (t=1, sigma=0)
        self.num_train_timesteps = max(
            1, int(self.num_sde_steps * self.timestep_fraction)
        )
        self.dt = 1.0 / self.num_sde_steps

        self.objective = FlowGRPOObjective(
            clip_range=self.clip_range,
            adv_clip_max=self.adv_clip_max,
            beta=self.beta,
            noise_level=self.noise_level,
        )

        reward_cfg = cfg.rl.get("reward_config", {})
        self.reward_fn = multi_score(self.device, reward_cfg)

        # Prompt bank for rollout sampling
        self.prompt_pool = self._init_prompt_pool()

    def _init_prompt_pool(self) -> List[str]:
        """Retrieves aesthetic tier-3 prompts for RL exploration."""
        # TODO: neg prompt?
        prompt_file = self.cfg.rl.get("prompt_file", None)
        if prompt_file and os.path.exists(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()][:10000]

        ds = getattr(self.dataloader, "dataset", None)
        if hasattr(ds, "prompts") and hasattr(ds, "aes_tiers"):
            tier3 = [p for p, tier in zip(ds.prompts, ds.aes_tiers) if int(tier) == 3]
            if tier3:
                return tier3[:10000]

        return [
            "1girl, solo, masterpiece, best quality, aesthetic",
            "1girl, souryuu asuka langley, neon genesis evangelion",
        ]

    @torch.no_grad()
    def rollout_sampling(self, epoch: int) -> Dict[str, torch.Tensor]:
        """
        Samples SDE trajectories across prompt groups and computes rewards.
        """
        unet_base = self.unet.module if self.is_ddp else self.unet
        unet_base.eval()
        if self.train_te:
            self.text_encoder.eval()

        # Select prompts deterministically per epoch and rank
        rng = random.Random(self.cfg.train.seed + epoch * 1000 + self.global_rank)
        prompts = rng.sample(self.prompt_pool, self.rollout_num_prompts)

        # Replicate each prompt for group relative comparison
        expanded_prompts = [p for p in prompts for _ in range(self.group_num_samples)]
        num_samples = len(expanded_prompts)

        all_latents = []
        all_next_latents = []
        all_log_probs = []
        all_timesteps = []
        all_enc_states = []
        all_attn_masks = []
        all_pos_maps = []
        all_decoded = []

        h = getattr(self.cfg.data, "resolution", 512)
        w = getattr(self.cfg.data, "resolution", 512)
        h_lat = h // 8
        w_lat = w // 8

        chunk_size = self.train_batch_size
        for start_idx in range(0, num_samples, chunk_size):
            # TODO: to reduce padding use group size instead of bs
            chunk_prompts = expanded_prompts[start_idx : start_idx + chunk_size]
            curr_bs = len(chunk_prompts)

            tokens_list, mask_list = [], []
            for p in chunk_prompts:
                tok, m = self.tokenizer.encode(
                    p,
                    max_len=227,
                    cfg_dropout_prob=0.0,
                    tag_dropout_prob=0.0,
                    shuffle_tags=False,
                )
                tokens_list.append(tok)
                mask_list.append(m)

            tokens = torch.stack(tokens_list).to(self.device)
            mask = torch.stack(mask_list).to(self.device)

            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                enc_states, attn_mask = self.text_encoder(tokens, mask=mask)

            # Positional map for RoPE
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

            # Initial Gaussian noise (t=0)
            # each sample in group diff latent
            z = torch.randn(
                (curr_bs, self.in_channels, h_lat, w_lat),
                device=self.device,
                dtype=self.dtype,
            )

            wrapper = CFGModelWrapper(
                unet=unet_base,
                combined_embeddings=enc_states,
                cfg_scale=self.cfg_scale,
                device=self.device,
                autocast_dtype=self.autocast_dtype,
                attention_mask=attn_mask,
                use_unet_mult=False if self.is_dit else True,
                pos_map=pos_map,
            )

            chunk_lats, chunk_next_lats = [], []
            chunk_log_probs, chunk_ts = [], []

            # SDE rollout trajectory loop
            z_curr = z
            for step_idx in range(self.num_sde_steps):
                t_val = step_idx * self.dt
                t_tensor = torch.full((curr_bs,), t_val, device=self.device)
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    v_pred = wrapper(z_curr, t_tensor)

                z_next, log_p, _, _ = sde_step_with_logprob(
                    sample=z_curr,
                    model_output=v_pred,
                    t=t_tensor,
                    dt=self.dt,
                    noise_level=self.noise_level,
                )

                chunk_lats.append(z_curr)
                chunk_next_lats.append(z_next)
                chunk_log_probs.append(log_p)
                chunk_ts.append(t_tensor)
                z_curr = z_next

            # Decode final clean latent with VAE for reward scoring
            raw_lat = (z_curr.float() * self.vae_std) + self.vae_mean
            vae_dtype = next(self.vae.parameters()).dtype
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                decoded = self.vae.decode(raw_lat.to(vae_dtype)).sample
            decoded = (decoded.float() / 2.0 + 0.5).clamp(0, 1)

            all_decoded.append(decoded)
            all_latents.append(torch.stack(chunk_lats, dim=1))
            all_next_latents.append(torch.stack(chunk_next_lats, dim=1))
            all_log_probs.append(torch.stack(chunk_log_probs, dim=1))
            all_timesteps.append(torch.stack(chunk_ts, dim=1))
            all_enc_states.append(enc_states)
            all_attn_masks.append(attn_mask)
            all_pos_maps.append(pos_map)

        images = torch.cat(all_decoded, dim=0)
        meta = [{"tag_weights": {}} for _ in expanded_prompts]

        rewards_dict, _ = self.reward_fn(images, expanded_prompts, meta)

        raw_rewards = torch.as_tensor(
            rewards_dict["avg"], device=self.device, dtype=torch.float32
        )

        # Compute group-relative normalized advantages
        advantages = torch.zeros_like(raw_rewards)
        for i in range(len(prompts)):
            s_idx = i * self.group_num_samples
            e_idx = s_idx + self.group_num_samples
            grp = raw_rewards[s_idx:e_idx]
            advantages[s_idx:e_idx] = (grp - grp.mean()) / (grp.std() + 1e-4)

        adv_matrix = advantages.unsqueeze(1).repeat(1, self.num_train_timesteps)

        return {
            "latents": torch.cat(all_latents, dim=0)[:, : self.num_train_timesteps],
            "next_latents": torch.cat(all_next_latents, dim=0)[
                :, : self.num_train_timesteps
            ],
            "log_probs": torch.cat(all_log_probs, dim=0)[:, : self.num_train_timesteps],
            "timesteps": torch.cat(all_timesteps, dim=0)[:, : self.num_train_timesteps],
            "advantages": adv_matrix,
            "encoder_hidden_states": torch.cat(all_enc_states, dim=0),
            "attention_mask": torch.cat(all_attn_masks, dim=0),
            "pos_map": torch.cat(all_pos_maps, dim=0),
            "rewards": raw_rewards,
        }

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        j: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Executes a single policy gradient update step on timestep j.
        """
        with torch.autocast(
            device_type="cuda", dtype=self.autocast_dtype, enabled=True
        ):
            loss, metrics = self.objective(
                model=self.unet,
                latents=batch["latents"][:, j],
                next_latents=batch["next_latents"][:, j],
                timesteps=batch["timesteps"][:, j],
                dt=self.dt,
                old_log_probs=batch["log_probs"][:, j],
                advantages=batch["advantages"][:, j],
                condition=batch["encoder_hidden_states"],
                attention_mask=batch["attention_mask"],
                pos_map=batch["pos_map"],
                use_unet_mult=False if self.is_dit else True,
            )
            accum_steps = self.cfg.train.gradient_accumulation_steps
            scaled_loss = loss / accum_steps

        if self.scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return metrics

    def fit(self, timestamp: Optional[str] = None):
        """
        Main Flow-GRPO outer optimization loop.
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        for epoch in range(self.start_epoch, self.cfg.train.epochs):
            if self.global_rank == 0:
                logging.info(
                    f"--- Epoch {epoch + 1}/{self.cfg.train.epochs}: "
                    f"Generating Rollouts ---"
                )

            # Trajectory sampling
            rollout_data = self.rollout_sampling(epoch)

            if self.global_rank == 0 and is_wandb_initialized():
                log_metrics(
                    {"train/mean_reward": rollout_data["rewards"].mean().item()},
                    step=self.global_step,
                    commit=False,
                )

            # Policy training over collected trajectories
            self.unet.train()
            if self.train_te:
                self.text_encoder.train()

            total_samples = rollout_data["latents"].shape[0]

            for inner in range(self.num_inner_epochs):
                perm = torch.randperm(total_samples, device=self.device)
                shuffled = {
                    k: v[perm]
                    for k, v in rollout_data.items()
                    if isinstance(v, torch.Tensor)
                }

                pbar = tqdm(
                    range(0, total_samples, self.train_batch_size),
                    desc=f"Epoch {epoch + 1}.{inner + 1} Training",
                    disable=self.global_rank != 0,
                )

                self.optimizer.zero_grad(set_to_none=True)
                for b_idx in pbar:
                    b_slice = {
                        k: v[b_idx : b_idx + self.train_batch_size]
                        for k, v in shuffled.items()
                    }

                    # Accumulate across active SDE timesteps
                    for j in range(self.num_train_timesteps):
                        step_metrics = self.train_step(b_slice, j)

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

                    self.global_step += 1

                    if self.global_rank == 0:
                        pbar.set_postfix(
                            {
                                "loss": f"{step_metrics['loss']:.4f}",
                                "norm": f"{norm:.3f}",
                            }
                        )
                        if is_wandb_initialized():
                            log_metrics(
                                {
                                    "train/policy_loss": step_metrics[
                                        "policy_loss"
                                    ].item(),
                                    "train/clipfrac": step_metrics["clipfrac"].item(),
                                    "train/approx_kl": step_metrics["approx_kl"].item(),
                                    "train/grad_norm": norm.item(),
                                },
                                step=self.global_step,
                                commit=False,
                            )

            if (
                epoch + 1
            ) % self.cfg.logging.save_interval == 0 and self.global_rank == 0:
                u_save = self.unet.module if self.is_ddp else self.unet
                t_save = (
                    self.text_encoder.module
                    if self.is_ddp and self.train_te
                    else self.text_encoder
                )
                save_checkpoint(
                    epoch=epoch,
                    step=self.global_step,
                    unet=u_save,
                    text_encoder=t_save,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    train_te=self.train_te,
                    repo_id=self.cfg.logging.hf_repo,
                    ema=self.ema,
                    config=self.cfg,
                )
