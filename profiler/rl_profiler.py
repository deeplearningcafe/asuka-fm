import os
import sys
import argparse
import socket
import time
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from omegaconf import OmegaConf, DictConfig
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.models.factory import load_trainable_model, create_optim
from src.diffusion.schedules import LinearSchedule
from src.diffusion.sampling import (
    CFGModelWrapper,
    compute_inference_pos_map,
)
from src.rl.objectives import FlowGRPOObjective, sde_step_with_logprob
from src.rl.rewards import multi_score

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


def setup_cuda_env(device_str: str) -> Tuple[torch.device, torch.dtype]:
    device_obj = torch.device(device_str)
    if device_obj.type != "cuda":
        return device_obj, torch.float32

    capability = torch.cuda.get_device_capability(device_obj)
    if capability[0] >= 8:
        autocast_dtype = torch.bfloat16
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        autocast_dtype = torch.float16
    return device_obj, autocast_dtype


def trace_handler(
    prof: torch.profiler.profile,
    output_dir: str = "profiling_results",
    trace_name: str = "rl_trace",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    host = socket.gethostname()
    ts = datetime.now().strftime(TIME_FORMAT_STR)
    prefix = os.path.join(output_dir, f"{trace_name}_{host}_{ts}")
    prof.export_chrome_trace(f"{prefix}.json.gz")
    dev_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    prof.export_memory_timeline(f"{prefix}_memory.html", device=dev_str)
    print(f"[PROFILER] Chrome trace and memory timeline exported to: {prefix}")


def profile_rl_breakdown(
    cfg: DictConfig,
    num_prompts: int = 2,
    group_samples: int = 4,
    sde_steps: int = 10,
    offload_to_cpu: bool = True,
    output_dir: str = "profiling_results",
) -> None:
    device, autocast_dtype = setup_cuda_env("cuda:0")
    dtype = (
        torch.bfloat16
        if getattr(cfg.train, "dtype", "bf16") == "bf16"
        else torch.float32
    )
    model_type = getattr(cfg.models, "model_type", "sprint_dual")
    is_dit = model_type in ["dual_stream", "sprint_dual"]

    unet, text_encoder, vae, tokenizer, _ = load_trainable_model(
        models_path=cfg.paths.models,
        device=device,
        dtype=dtype,
        train_te=False,
        use_checkpointing=getattr(cfg.train, "use_checkpointing", True),
        model_type=model_type,
        model_cfg=cfg.models,
        autocast_dtype=autocast_dtype,
    )
    unet.eval()
    text_encoder.eval()
    vae.eval()

    optimizer = create_optim(unet, text_encoder, cfg)
    objective = FlowGRPOObjective(
        clip_range=cfg.rl.clip_range,
        adv_clip_max=cfg.rl.adv_clip_max,
        beta=cfg.rl.beta,
        noise_level=cfg.rl.noise_level,
        use_unet_mult=False if is_dit else True,
    )

    reward_cfg = cfg.rl.get("reward_config", {})
    reward_fn = multi_score(device, reward_cfg)

    # Resolution and dimensions
    res = getattr(cfg.data, "resolution", 512)
    h_lat, w_lat = res // 8, res // 8
    total_samples = num_prompts * group_samples
    dt = 1.0 / sde_steps
    num_train_timesteps = max(1, int(sde_steps * cfg.rl.timestep_fraction))

    prompts = [
        "1girl, hatsune miku, masterpiece, best quality",
        "1boy, solo, detailed anime aesthetic, vibrant",
    ][:num_prompts]
    expanded_prompts = [p for p in prompts for _ in range(group_samples)]

    print("\n" + "=" * 70)
    print(f"FLOW-GRPO RL PIPELINE PROFILING BREAKDOWN ({model_type.upper()})")
    print(f"Total Rollouts: {total_samples} ({num_prompts} x {group_samples})")
    print(f"SDE Steps     : {sde_steps} (Train timesteps: {num_train_timesteps})")
    print(f"Resolution    : {res}x{res} | Latents: {h_lat}x{w_lat}")
    print(
        f"RAM Offload   : {'ENABLED (Zero VRAM Trajectories)' if offload_to_cpu else 'DISABLED (Full VRAM)'}"
    )
    print("=" * 70)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_enc_end = torch.cuda.Event(enable_timing=True)
    ev_sde_end = torch.cuda.Event(enable_timing=True)
    ev_vae_end = torch.cuda.Event(enable_timing=True)
    ev_rew_end = torch.cuda.Event(enable_timing=True)
    ev_train_end = torch.cuda.Event(enable_timing=True)

    ev_start.record()

    # 1. Text Encoding Phase
    tokens_list, mask_list = [], []
    for p in expanded_prompts:
        tok, m = tokenizer.encode(
            p,
            max_len=getattr(cfg.data, "max_seq_len", 227),
            cfg_dropout_prob=0.0,
            tag_dropout_prob=0.0,
            shuffle_tags=False,
        )
        tokens_list.append(tok)
        mask_list.append(m)

    tokens = torch.stack(tokens_list).to(device)
    attn_mask = torch.stack(mask_list).to(device)

    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        enc_states, attn_mask = text_encoder(tokens, mask=attn_mask)

    pos_map = (
        compute_inference_pos_map(
            height=res,
            width=res,
            patch_size=getattr(unet, "patch_size", 2),
            vae_scale=8,
            coord_system="aspect_norm",
            device=device,
            dtype=dtype,
        )
        .unsqueeze(0)
        .repeat(total_samples, 1, 1)
    )

    ev_enc_end.record()

    # 2. SDE Rollout Generation Loop
    chunk_size = getattr(cfg.rl, "train_batch_size", 4)
    all_lats, all_next_lats, all_log_probs, all_ts = [], [], [], []
    final_latents = []

    for idx in range(0, total_samples, chunk_size):
        b_slice = slice(idx, idx + chunk_size)
        curr_b = len(expanded_prompts[b_slice])
        z = torch.randn(
            (curr_b, cfg.models.in_channels, h_lat, w_lat),
            device=device,
            dtype=dtype,
        )

        wrapper = CFGModelWrapper(
            unet=unet,
            combined_embeddings=enc_states[b_slice],
            cfg_scale=cfg.rl.cfg_scale,
            device=device,
            autocast_dtype=autocast_dtype,
            attention_mask=attn_mask[b_slice],
            use_unet_mult=False if is_dit else True,
            pos_map=pos_map[b_slice],
        )

        chunk_l, chunk_nl, chunk_lp, chunk_t = [], [], [], []
        z_curr = z
        for s in range(sde_steps):
            t_val = s * dt
            t_ten = torch.full((curr_b,), t_val, device=device)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                v_pred = wrapper(z_curr, t_ten)
            z_next, log_p, _, _ = sde_step_with_logprob(
                sample=z_curr,
                model_output=v_pred,
                t=t_ten,
                dt=dt,
                noise_level=cfg.rl.noise_level,
            )
            target_device = "cpu" if offload_to_cpu else device
            chunk_l.append(z_curr.to(target_device, non_blocking=True))
            chunk_nl.append(z_next.to(target_device, non_blocking=True))
            chunk_lp.append(log_p.to(target_device, non_blocking=True))
            chunk_t.append(t_ten.to(target_device, non_blocking=True))
            z_curr = z_next

        final_latents.append(z_curr)
        all_lats.append(torch.stack(chunk_l, dim=1))
        all_next_lats.append(torch.stack(chunk_nl, dim=1))
        all_log_probs.append(torch.stack(chunk_lp, dim=1))
        all_ts.append(torch.stack(chunk_t, dim=1))

    ev_sde_end.record()
    mem_after_sde = torch.cuda.memory_allocated(device) / (1024**2)

    # 3. VAE Latent Decode Phase (Dedicated Profiling)
    torch.cuda.reset_peak_memory_stats(device)
    vae_start_mem = torch.cuda.memory_allocated(device) / (1024**2)

    vae_mean = torch.tensor(cfg.models.vae_mean, device=device, dtype=dtype)
    vae_std = torch.tensor(cfg.models.vae_std, device=device, dtype=dtype)
    vae_dtype = next(vae.parameters()).dtype

    all_decoded = []
    # Decode in micro-chunks of 2 to preserve VRAM
    v_bs = getattr(cfg.models, "vae_batch_size", 2)
    cat_final_lats = torch.cat(final_latents, dim=0)

    for v_idx in range(0, total_samples, v_bs):
        lat_slice = cat_final_lats[v_idx : v_idx + v_bs]
        raw_lat = (lat_slice.float() * vae_std) + vae_mean
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            dec = vae.decode(raw_lat.to(vae_dtype)).sample
        dec = (dec.float() / 2.0 + 0.5).clamp(0, 1)
        all_decoded.append(dec.cpu() if offload_to_cpu else dec)

    ev_vae_end.record()
    vae_peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)

    # 4. Multi-Head Reward Evaluation Phase
    # isn't this contrary?
    if offload_to_cpu:
        images_for_reward = torch.cat(all_decoded, dim=0).to(device)
    else:
        images_for_reward = torch.cat(all_decoded, dim=0)

    meta = [{"tag_weights": {}} for _ in expanded_prompts]
    rewards_dict, _ = reward_fn(images_for_reward, expanded_prompts, meta)
    raw_rewards = torch.as_tensor(rewards_dict["avg"], device=device)

    # Calculate group normalized advantages
    advantages = torch.zeros_like(raw_rewards)
    for p_idx in range(num_prompts):
        s_i = p_idx * group_samples
        e_i = s_i + group_samples
        grp = raw_rewards[s_i:e_i]
        advantages[s_i:e_i] = (grp - grp.mean()) / (grp.std() + 1e-4)

    adv_matrix = advantages.unsqueeze(1).repeat(1, num_train_timesteps)
    if offload_to_cpu:
        adv_matrix = adv_matrix.cpu()

    ev_rew_end.record()

    # 5. Policy Optimization Step (Forward + Backward + FlopCounter)
    unet.train()
    optimizer.zero_grad(set_to_none=True)

    latents_batch = torch.cat(all_lats, dim=0)[:, :num_train_timesteps]
    next_latents_batch = torch.cat(all_next_lats, dim=0)[:, :num_train_timesteps]
    log_probs_batch = torch.cat(all_log_probs, dim=0)[:, :num_train_timesteps]
    timesteps_batch = torch.cat(all_ts, dim=0)[:, :num_train_timesteps]

    # Measure FLOPs over micro-batch during policy forward pass
    flop_counter = FlopCounterMode(display=False)
    sub_slice = slice(0, chunk_size)

    # Transfer micro-batch tensors to GPU if offloaded
    sub_l = latents_batch[sub_slice, 0].to(device)
    sub_nl = next_latents_batch[sub_slice, 0].to(device)
    sub_t = timesteps_batch[sub_slice, 0].to(device)
    sub_lp = log_probs_batch[sub_slice, 0].to(device)
    sub_adv = adv_matrix[sub_slice, 0].to(device)

    with flop_counter:
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            loss, _ = objective(
                model=unet,
                latents=sub_l,
                next_latents=sub_nl,
                timesteps=sub_t,
                dt=dt,
                old_log_probs=sub_lp,
                advantages=sub_adv,
                condition=enc_states[sub_slice],
                attention_mask=attn_mask[sub_slice],
                pos_map=pos_map[sub_slice],
            )
        loss.backward()

    backbone_flops = flop_counter.get_total_flops()
    torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    ev_train_end.record()
    torch.cuda.synchronize()

    # Timing metrics
    t_enc = ev_start.elapsed_time(ev_enc_end)
    t_sde = ev_enc_end.elapsed_time(ev_sde_end)
    t_vae = ev_sde_end.elapsed_time(ev_vae_end)
    t_rew = ev_vae_end.elapsed_time(ev_rew_end)
    t_train = ev_rew_end.elapsed_time(ev_train_end)
    t_rollout = t_enc + t_sde + t_vae + t_rew
    t_total = t_rollout + t_train

    peak_total_mem = torch.cuda.max_memory_allocated(device) / (1024**2)

    # Compute MFU
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    t_train_s = t_train / 1000.0
    isolated_tflops = (backbone_flops / t_train_s) / 1e12 if t_train_s > 0 else 0
    peak_tflops = getattr(cfg.train, "gpu_peak_tflops", 165.2)
    isolated_mfu = (isolated_tflops / peak_tflops) * 100.0

    print("\n" + "=" * 70)
    print("RL BOTTLENECK & COMPONENT TIMING SUMMARY")
    print("=" * 70)
    print(f"Total Iteration Latency       : {t_total:8.2f} ms")
    print(
        f"  -> Total Rollout Phase      : {t_rollout:8.2f} ms ({t_rollout / t_total * 100:5.1f}%)"
    )
    print(
        f"     * Text Encoding          : {t_enc:8.2f} ms ({t_enc / t_total * 100:5.1f}%)"
    )
    print(
        f"     * SDE Denoising Loop     : {t_sde:8.2f} ms ({t_sde / t_total * 100:5.1f}%)"
    )
    print(
        f"     * VAE Decode to Pixels   : {t_vae:8.2f} ms ({t_vae / t_total * 100:5.1f}%) [Breakdown below]"
    )
    print(
        f"     * Multi-Head Reward Eval : {t_rew:8.2f} ms ({t_rew / t_total * 100:5.1f}%)"
    )
    print(
        f"  -> Total Policy Train Phase : {t_train:8.2f} ms ({t_train / t_total * 100:5.1f}%)"
    )
    print("-" * 70)
    print("DEDICATED VAE DECODE BREAKDOWN & BOTTLENECK ANALYSIS:")
    print(f"  - VAE Decode Wall Time      : {t_vae:8.2f} ms")
    print(f"  - Share of Rollout Time     : {(t_vae / t_rollout) * 100:5.1f}%")
    print(f"  - Share of Total Epoch Time : {(t_vae / t_total) * 100:5.1f}%")
    print(
        f"  - Peak VRAM during VAE      : {vae_peak_mem:8.2f} MB (Delta: +{vae_peak_mem - vae_start_mem:.2f} MB)"
    )
    print("-" * 70)
    print(f"Global Peak VRAM Allocated    : {peak_total_mem:8.2f} MB")
    print(f"Post-SDE Latent Memory Footprint: {mem_after_sde:8.2f} MB")
    print(f"Backbone Training FLOPs       : {backbone_flops / 1e9:8.2f} GFLOPs")
    print(f"Backbone Isolated Throughput  : {isolated_tflops:8.2f} TFLOPS")
    print(
        f"Backbone Isolated MFU         : {isolated_mfu:8.2f}% (Peak: {peak_tflops:.1f} TFLOPS)"
    )
    print("=" * 70)

    if (t_vae / t_rollout) > 0.30:
        print("[DIAGNOSTIC WARNING] VAE decoding consumes >30% of rollout time.")
        print(
            "  -> Use VAE tiling or reduce vae_batch_size to minimize peak activation spikes."
        )
    if peak_total_mem > 38000.0:
        print("[DIAGNOSTIC WARNING] Peak VRAM is dangerously close to 44GB limit.")
        print(
            "  -> Keep offload_to_cpu=True and micro-batch VAE decoding to avoid OOM."
        )


def profile_rl_trace(
    cfg: DictConfig,
    num_prompts: int = 1,
    group_samples: int = 2,
    sde_steps: int = 5,
    output_dir: str = "profiling_results",
) -> None:
    device, autocast_dtype = setup_cuda_env("cuda:0")
    dtype = (
        torch.bfloat16
        if getattr(cfg.train, "dtype", "bf16") == "bf16"
        else torch.float32
    )
    model_type = getattr(cfg.models, "model_type", "sprint_dual")
    is_dit = model_type in ["dual_stream", "sprint_dual"]

    unet, text_encoder, vae, tokenizer, _ = load_trainable_model(
        models_path=cfg.paths.models,
        device=device,
        dtype=dtype,
        train_te=False,
        use_checkpointing=True,
        model_type=model_type,
        model_cfg=cfg.models,
        autocast_dtype=autocast_dtype,
    )
    optimizer = create_optim(unet, text_encoder, cfg)
    objective = FlowGRPOObjective(
        clip_range=cfg.rl.clip_range,
        adv_clip_max=cfg.rl.adv_clip_max,
        beta=cfg.rl.beta,
        noise_level=cfg.rl.noise_level,
        use_unet_mult=False if is_dit else True,
    )
    reward_fn = multi_score(device, cfg.rl.get("reward_config", {}))

    res = getattr(cfg.data, "resolution", 512)
    dt = 1.0 / sde_steps
    prompts = ["1girl, hatsune miku, masterpiece"] * (num_prompts * group_samples)

    print(f"[PROFILER] Running PyTorch Profiler trace for Flow-GRPO RL...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=lambda p: trace_handler(p, output_dir, "rl_grpo_trace"),
    ) as prof:
        for _ in range(4):
            with torch.profiler.record_function("## rl_text_encode ##"):
                tok, m = tokenizer.encode(prompts[0], max_len=77)
                tokens = tok.unsqueeze(0).repeat(len(prompts), 1).to(device)
                mask = m.unsqueeze(0).repeat(len(prompts), 1).to(device)
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    enc_states, attn_mask = text_encoder(tokens, mask=mask)

            with torch.profiler.record_function("## rl_sde_rollout ##"):
                z = torch.randn(
                    (len(prompts), cfg.models.in_channels, res // 8, res // 8),
                    device=device,
                    dtype=dtype,
                )
                wrapper = CFGModelWrapper(
                    unet=unet,
                    combined_embeddings=enc_states,
                    cfg_scale=1.0,
                    device=device,
                    autocast_dtype=autocast_dtype,
                    attention_mask=attn_mask,
                    use_unet_mult=False if is_dit else True,
                )
                for s in range(sde_steps):
                    t_ten = torch.full((len(prompts),), s * dt, device=device)
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        v_pred = wrapper(z, t_ten)
                    z, log_p, _, _ = sde_step_with_logprob(
                        sample=z,
                        model_output=v_pred,
                        t=t_ten,
                        dt=dt,
                        noise_level=cfg.rl.noise_level,
                    )

            with torch.profiler.record_function("## rl_vae_decode ##"):
                raw_lat = (z.float() * cfg.models.vae_std) + cfg.models.vae_mean
                vae_dt = next(vae.parameters()).dtype
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    decoded = vae.decode(raw_lat.to(vae_dt)).sample
                decoded = (decoded.float() / 2.0 + 0.5).clamp(0, 1)

            with torch.profiler.record_function("## rl_reward_eval ##"):
                rewards_dict, _ = reward_fn(decoded, prompts, [{} for _ in prompts])

            with torch.profiler.record_function("## rl_policy_step ##"):
                optimizer.zero_grad(set_to_none=True)
                adv = torch.zeros(len(prompts), device=device)
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    loss, _ = objective(
                        model=unet,
                        latents=z,
                        next_latents=z,
                        timesteps=t_ten,
                        dt=dt,
                        old_log_probs=log_p,
                        advantages=adv,
                        condition=enc_states,
                        attention_mask=attn_mask,
                    )
                loss.backward()
                optimizer.step()

            prof.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flow-GRPO RL Compute, Memory & VAE Profiler"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["breakdown", "trace"],
        default="breakdown",
    )
    parser.add_argument("--num_prompts", type=int, default=2)
    parser.add_argument("--group_samples", type=int, default=4)
    parser.add_argument("--sde_steps", type=int, default=10)
    parser.add_argument("--no_cpu_offload", action="store_true")
    parser.add_argument("--output_dir", type=str, default="profiling_results")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    base_cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_cli(args.opts)
    cfg = OmegaConf.merge(base_cfg, cli_cfg)

    if args.mode == "breakdown":
        profile_rl_breakdown(
            cfg=cfg,
            num_prompts=args.num_prompts,
            group_samples=args.group_samples,
            sde_steps=args.sde_steps,
            offload_to_cpu=not args.no_cpu_offload,
            output_dir=args.output_dir,
        )
    elif args.mode == "trace":
        profile_rl_trace(
            cfg=cfg,
            num_prompts=args.num_prompts,
            group_samples=args.group_samples,
            sde_steps=args.sde_steps,
            output_dir=args.output_dir,
        )
