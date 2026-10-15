import os
import sys
import argparse
import socket
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
import torch.multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from omegaconf import OmegaConf, DictConfig
import numpy as np
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.models.factory import (
    load_trainable_model,
    create_optim,
    create_scheduler,
)
from src.data.loader import create_dataloader
from src.diffusion.schedules import LinearSchedule, DDPMSchedule
from src.diffusion.objectives import (
    FlowMatchingObjective,
    DDPMObjective,
)
from src.utils.act_grad_checkpointing import (
    patch_unsloth_smart_gradient_checkpointing,
    patch_torch_compile,
    patch_compiled_autograd,
    CPUGradientAccumulator,
)

from src.diffusion.sampling import generate_samples

MAX_BATCH_SIZE = 64
TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


def unpack_batch(
    batch: Any,
    vae: Optional[nn.Module],
    device: torch.device,
    dtype: torch.dtype,
    autocast_dtype: torch.dtype,
    vae_mean: torch.Tensor,
    vae_std: torch.Tensor,
    vae_batch_size: int = 64,
    is_latent: bool = False,
    pixel_training: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    """
    Unpacks batch tensors with support for online VAE, precomputed latents,
    and pixel-space training (DiP). Fully backward-compatible.
    """
    pos_map = None
    if len(batch) >= 5:
        # Streaming dataset format: images_or_lats, cond, mask, pos_map, ...
        images_or_lats, cond, mask, pos_map, tag_weights, *rest = batch

        if not is_latent and not pixel_training and vae is not None:
            # Online VAE encoding branch
            with torch.no_grad():
                latents = encode_vae_latents_profiling(
                    images=images_or_lats,
                    vae=vae,
                    device=device,
                    autocast_dtype=autocast_dtype,
                    vae_mean=vae_mean,
                    vae_std=vae_std,
                    chunk_size=vae_batch_size,
                )
        else:
            # Precomputed latents or direct pixel images [-1, 1]
            latents = images_or_lats.to(device=device, dtype=dtype, non_blocking=True)

        attention_mask = mask.to(device, non_blocking=True)
        cond = cond.to(device, non_blocking=True)
        if pos_map is not None:
            pos_map = pos_map.to(device, non_blocking=True)
        tag_weights = tag_weights.to(device, dtype=dtype, non_blocking=True)
    else:
        # H5LatentDataset batch format
        latents = batch[0].to(device, dtype=dtype, non_blocking=True) * 0.18215
        cond = batch[1].to(device, non_blocking=True)
        tag_weights = batch[2].to(device, dtype=dtype, non_blocking=True)
        attention_mask = batch[3].to(device, non_blocking=True)

    return latents, cond, attention_mask, tag_weights, pos_map


def encode_vae_latents_profiling(
    images: torch.Tensor,
    vae: nn.Module,
    device: torch.device,
    autocast_dtype: torch.dtype,
    vae_mean: torch.Tensor,
    vae_std: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Encodes raw RGB images to normalized VAE latents with chunking."""
    latents = []
    vae_dtype = next(vae.parameters()).dtype
    for i in range(0, images.shape[0], chunk_size):
        chunk = images[i : i + chunk_size].to(
            device=device, dtype=vae_dtype, non_blocking=True
        )
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True):
            enc = vae.encode(chunk)
            dist = getattr(enc, "latent_dist", enc)
            lat = dist.sample() if hasattr(dist, "sample") else dist
            latents.append(lat)
    lat = torch.cat(latents, dim=0)
    return (lat - vae_mean) / vae_std


def trace_handler(
    prof: torch.profiler.profile,
    output_dir: str = "profiling_results",
    trace_name: str = "trace",
) -> None:
    """
    Exports Chrome trace and CUDA memory timeline to the output directory.
    Maintains backward compatibility with 1-arg or 3-arg invocations.
    """
    os.makedirs(output_dir, exist_ok=True)
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = os.path.join(output_dir, f"{trace_name}_{host_name}_{timestamp}")

    prof.export_chrome_trace(f"{file_prefix}.json.gz")
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    prof.export_memory_timeline(f"{file_prefix}_memory.html", device=device_str)


def profile_pipeline_breakdown(
    cfg: DictConfig,
    warmup_iters: int = 5,
    profile_iters: int = 10,
    output_dir: str = "profiling_results",
    compile_model: bool = False,
    pixel_training: bool = False,
    train_only_output: bool = False,
) -> None:
    """
    Benchmarks execution times, FLOPs, and MFU across components:
    DataLoader, Online VAE, TextEncoder, DiT backbone, and DiP LocalDecoder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if getattr(cfg.train, "dtype", "bf16") == "bf16"
        else torch.float32
    )
    autocast_dtype = (
        torch.bfloat16
        if getattr(cfg.train, "dtype", "bf16") == "bf16"
        else torch.float16
    )

    model_type = getattr(cfg.models, "model_type", "dual_stream")
    is_dit = model_type in ["dual_stream", "sprint_dual"]
    is_latent = (
        cfg.data.get("is_latent", False)
        or getattr(cfg.data, "dataset_type", "") == "streaming_latents"
        or cfg.data.get("cache_latents_to_ram", False)
    )

    if pixel_training:
        cfg.train.pixel_training = True
        cfg.models.use_pixel_decoder = True

    unet, text_encoder, vae, tokenizer, ema = load_trainable_model(
        models_path=cfg.paths.models,
        device=device,
        dtype=dtype,
        train_te=cfg.train.train_te,
        use_checkpointing=cfg.train.use_checkpointing,
        train_only_output=train_only_output,
        model_type=model_type,
        model_cfg=cfg.models,
        autocast_dtype=autocast_dtype,
    )

    dataloader = create_dataloader(cfg, rank=0, tokenizer=tokenizer)

    vae_mean_val = getattr(cfg.models, "vae_mean", 0.0)
    vae_std_val = getattr(cfg.models, "vae_std", 1.0 / 0.18215)
    vae_mean = torch.tensor(vae_mean_val, device=device, dtype=dtype).view(1, -1, 1, 1)
    vae_std = torch.tensor(vae_std_val, device=device, dtype=dtype).view(1, -1, 1, 1)
    vae_batch_size = cfg.train.get(
        "vae_batch_size", cfg.models.get("vae_batch_size", 64)
    )

    if cfg.train.objective == "flow_matching":
        schedule = LinearSchedule(device=device)
        objective = FlowMatchingObjective(
            schedule=schedule,
            prediction_target=getattr(cfg.train, "prediction_target", "v"),
            loss_target=getattr(cfg.train, "loss_target", "v"),
            timestep_sampling=cfg.train.get("timestep_fn", "uniform"),
            shift=cfg.train.shift,
            use_ot=cfg.train.get("use_ot", False),
            use_unet_mult=False if is_dit else True,
        )
    else:
        schedule = DDPMSchedule(device=device)
        objective = DDPMObjective(schedule=schedule, min_snr_gamma=cfg.train.snr_gamma)

    optimizer = create_optim(unet, text_encoder, cfg)
    scaler = torch.cuda.amp.GradScaler() if autocast_dtype == torch.float16 else None

    if compile_model and hasattr(torch, "compile"):
        print("Compiling backbone and text encoder with torch.compile...")
        patch_torch_compile()
        patch_compiled_autograd()
        unet = torch.compile(unet)
        if cfg.train.train_te and text_encoder is not None:
            text_encoder = torch.compile(text_encoder)

    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    peak_tflops = cfg.train.get("gpu_peak_tflops", 165.2)
    patch_size = getattr(unet, "patch_size", 2)

    t_data_list: List[float] = []
    t_vae_list: List[float] = []
    t_te_list: List[float] = []
    t_fwd_list: List[float] = []
    t_bwd_list: List[float] = []
    t_opt_list: List[float] = []
    t_total_list: List[float] = []
    tokens_list: List[int] = []

    print(cfg)
    print("\n" + "=" * 70)
    print(f"PIPELINE PROFILING BREAKDOWN ({model_type.upper()})")
    print(f"Mode: {'PIXEL (DiP)' if pixel_training else 'LATENT'}")
    print(f"Using latents {is_latent} and {cfg.data.get('is_latent', False)}")
    print(
        f"Trainable Params: {trainable_params / 1e6:.2f}M / {total_params / 1e6:.2f}M"
    )
    print(f"Warmup: {warmup_iters} steps | Active: {profile_iters} steps")
    print("=" * 70)

    data_iter = iter(dataloader)
    total_steps = warmup_iters + profile_iters

    for step in range(total_steps):
        t_data_start = time.perf_counter()
        batch = next(data_iter)
        t_data_end = time.perf_counter()
        data_time_ms = (t_data_end - t_data_start) * 1000.0

        torch.cuda.synchronize()
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_vae_end = torch.cuda.Event(enable_timing=True)
        ev_te_end = torch.cuda.Event(enable_timing=True)
        ev_fwd_end = torch.cuda.Event(enable_timing=True)
        ev_bwd_end = torch.cuda.Event(enable_timing=True)
        ev_opt_end = torch.cuda.Event(enable_timing=True)

        ev_start.record()

        # Unpack batch tensors
        (
            latents,
            cond,
            attention_mask,
            tag_weights,
            pos_map,
        ) = unpack_batch(
            batch=batch,
            vae=vae,
            device=device,
            dtype=dtype,
            autocast_dtype=autocast_dtype,
            vae_mean=vae_mean,
            vae_std=vae_std,
            vae_batch_size=vae_batch_size,
            is_latent=is_latent,
            pixel_training=pixel_training,
        )
        ev_vae_end.record()

        # Dynamic shape marking for torch.compile
        if compile_model:
            torch._dynamo.mark_dynamic(latents, 0, min=1, max=MAX_BATCH_SIZE)
            torch._dynamo.mark_dynamic(cond, 0, min=1, max=MAX_BATCH_SIZE)
            torch._dynamo.mark_dynamic(cond, 1, min=16, max=512)
            if attention_mask is not None:
                torch._dynamo.mark_dynamic(attention_mask, 0, min=1, max=MAX_BATCH_SIZE)
                torch._dynamo.mark_dynamic(attention_mask, 1, min=16, max=512)
            if pos_map is not None:
                torch._dynamo.mark_dynamic(pos_map, 0, min=1, max=MAX_BATCH_SIZE)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True):
            with torch.set_grad_enabled(cfg.train.train_te):
                encoder_hidden_states, attention_mask = text_encoder(
                    cond,
                    mask=attention_mask,
                    drop_mask=None,
                )
        ev_te_end.record()

        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True):
            loss, metrics = objective.forward(
                unet,
                latents,
                encoder_hidden_states,
                tag_weights,
                attention_mask=attention_mask,
                pos_map=pos_map,
            )
        ev_fwd_end.record()

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        ev_bwd_end.record()

        if scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
        ev_opt_end.record()

        torch.cuda.synchronize()

        if step >= warmup_iters:
            t_vae = ev_start.elapsed_time(ev_vae_end)
            t_te = ev_vae_end.elapsed_time(ev_te_end)
            t_fwd = ev_te_end.elapsed_time(ev_fwd_end)
            t_bwd = ev_fwd_end.elapsed_time(ev_bwd_end)
            t_opt = ev_bwd_end.elapsed_time(ev_opt_end)
            t_total_gpu = ev_start.elapsed_time(ev_opt_end)
            t_total_step = data_time_ms + t_total_gpu

            bsz, _, h_sp, w_sp = latents.shape
            img_tokens = (h_sp // patch_size) * (w_sp // patch_size)
            txt_tokens = (
                attention_mask.bool().sum().item() if attention_mask is not None else 0
            )
            step_tokens = (bsz * img_tokens) + txt_tokens

            t_data_list.append(data_time_ms)
            t_vae_list.append(t_vae)
            t_te_list.append(t_te)
            t_fwd_list.append(t_fwd)
            t_bwd_list.append(t_bwd)
            t_opt_list.append(t_opt)
            t_total_list.append(t_total_step)
            tokens_list.append(step_tokens)

            print(
                f"Step {step - warmup_iters + 1:02d}/{profile_iters} | "
                f"Total: {t_total_step:6.1f}ms | "
                f"Data: {data_time_ms:5.1f}ms | "
                f"VAE: {t_vae:5.1f}ms | "
                f"TE: {t_te:5.1f}ms | "
                f"Fwd: {t_fwd:5.1f}ms | "
                f"Bwd: {t_bwd:5.1f}ms | "
                f"Opt: {t_opt:4.1f}ms"
            )

    print("\nMeasuring FLOPs using PyTorch FlopCounterMode...")
    flop_counter = FlopCounterMode(display=False)
    optimizer.zero_grad(set_to_none=True)
    with flop_counter:
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True):
            loss, _ = objective.forward(
                unet,
                latents,
                encoder_hidden_states,
                tag_weights,
                attention_mask=attention_mask,
                pos_map=pos_map,
            )
        loss.backward()

    backbone_flops = flop_counter.get_total_flops()

    avg_data = float(np.mean(t_data_list))
    avg_vae = float(np.mean(t_vae_list))
    avg_te = float(np.mean(t_te_list))
    avg_fwd = float(np.mean(t_fwd_list))
    avg_bwd = float(np.mean(t_bwd_list))
    avg_opt = float(np.mean(t_opt_list))
    avg_total = float(np.mean(t_total_list))
    avg_tokens = float(np.mean(tokens_list))

    t_backbone_s = (avg_fwd + avg_bwd) / 1000.0
    t_total_s = avg_total / 1000.0

    isolated_tflops = (backbone_flops / t_backbone_s) / 1e12 if t_backbone_s > 0 else 0
    isolated_mfu = (isolated_tflops / peak_tflops) * 100.0 if peak_tflops > 0 else 0

    # Calculate system MFU factoring in SPRINT token dropping if active
    drop_factor = (
        (1.0 - getattr(cfg.models, "drop_ratio", 0.0))
        if model_type == "sprint_dual"
        else 1.0
    )
    std_model_flops = 6 * trainable_params * avg_tokens * drop_factor
    system_tflops = (std_model_flops / t_total_s) / 1e12 if t_total_s > 0 else 0
    system_mfu = (system_tflops / peak_tflops) * 100.0 if peak_tflops > 0 else 0

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    print("\n" + "=" * 70)
    print("PROFILING & BOTTLENECK DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"Average Step Latency          : {avg_total:.2f} ms")
    print(f"Peak GPU Memory Allocated     : {peak_mem_mb:.2f} MB")
    print(
        f"Data Loading Time (CPU/IO)    : {avg_data:6.2f} ms "
        f"({(avg_data / avg_total) * 100:4.1f}%)"
    )
    print(
        f"VAE Encode / Image Ingest     : {avg_vae:6.2f} ms "
        f"({(avg_vae / avg_total) * 100:4.1f}%)"
    )
    print(
        f"Text Encoder Forward Time     : {avg_te:6.2f} ms "
        f"({(avg_te / avg_total) * 100:4.1f}%)"
    )
    print(
        f"Backbone Forward Pass Time    : {avg_fwd:6.2f} ms "
        f"({(avg_fwd / avg_total) * 100:4.1f}%)"
    )
    print(
        f"Backbone Backward Pass Time   : {avg_bwd:6.2f} ms "
        f"({(avg_bwd / avg_total) * 100:4.1f}%)"
    )
    print(
        f"Optimizer & Norm Clip Time    : {avg_opt:6.2f} ms "
        f"({(avg_opt / avg_total) * 100:4.1f}%)"
    )
    print("-" * 70)
    print(f"Backbone FLOPs per Step       : {backbone_flops / 1e9:.2f} GFLOPs")
    print(f"Backbone Isolated Throughput  : {isolated_tflops:.2f} TFLOPS")
    print(
        f"Backbone Isolated MFU         : {isolated_mfu:.2f}% "
        f"(Peak: {peak_tflops:.1f} TFLOPS)"
    )
    print(f"End-to-End System MFU         : {system_mfu:.2f}%")
    print("=" * 70)

    if (avg_data / avg_total) > 0.25:
        print(
            "[DIAGNOSTIC WARNING] DataLoader is consuming >25% of step time. "
            "Increase num_workers or cache latents to RAM/NVMe."
        )
    if not pixel_training and not is_latent and (avg_vae / avg_total) > 0.20:
        print(
            "[DIAGNOSTIC WARNING] Online VAE encoding is consuming >20% "
            "of step time. Precomputing latents to shards eliminates this cost."
        )


def profile_pipeline_trace(
    cfg: DictConfig,
    warmup_iters: int = 3,
    profile_iters: int = 5,
    output_dir: str = "profiling_results",
    compile_model: bool = False,
    pixel_training: bool = False,
    train_only_output: bool = False,
) -> None:
    """Captures an end-to-end PyTorch Profiler trace with operator scopes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if getattr(cfg.train, "dtype", "bf16") == "bf16"
        else torch.float32
    )
    autocast_dtype = (
        torch.bfloat16
        if getattr(cfg.train, "dtype", "bf16") == "bf16"
        else torch.float16
    )

    model_type = getattr(cfg.models, "model_type", "dual_stream")
    is_dit = model_type in ["dual_stream", "sprint_dual"]
    is_latent = (
        cfg.data.get("is_latent", False)
        or getattr(cfg.data, "dataset_type", "") == "streaming_latents"
        or cfg.data.get("cache_latents_to_ram", False)
    )

    if pixel_training:
        cfg.train.pixel_training = True
        cfg.models.use_pixel_decoder = True

    unet, text_encoder, vae, tokenizer, ema = load_trainable_model(
        models_path=cfg.paths.models,
        device=device,
        dtype=dtype,
        train_te=cfg.train.train_te,
        use_checkpointing=cfg.train.use_checkpointing,
        train_only_output=train_only_output,
        model_type=model_type,
        model_cfg=cfg.models,
        autocast_dtype=autocast_dtype,
    )

    dataloader = create_dataloader(cfg, rank=0, tokenizer=tokenizer)

    vae_mean = torch.tensor(
        getattr(cfg.models, "vae_mean", 0.0), device=device, dtype=dtype
    ).view(1, -1, 1, 1)
    vae_std = torch.tensor(
        getattr(cfg.models, "vae_std", 1.0 / 0.18215),
        device=device,
        dtype=dtype,
    ).view(1, -1, 1, 1)
    vae_batch_size = cfg.train.get(
        "vae_batch_size", cfg.models.get("vae_batch_size", 64)
    )

    if cfg.train.objective == "flow_matching":
        schedule = LinearSchedule(device=device)
        objective = FlowMatchingObjective(
            schedule=schedule,
            prediction_target=getattr(cfg.train, "prediction_target", "v"),
            loss_target=getattr(cfg.train, "loss_target", "v"),
            timestep_sampling=cfg.train.get("timestep_fn", "uniform"),
            shift=cfg.train.shift,
            use_ot=cfg.train.get("use_ot", False),
            use_unet_mult=False if is_dit else True,
        )
    else:
        schedule = DDPMSchedule(device=device)
        objective = DDPMObjective(schedule=schedule, min_snr_gamma=cfg.train.snr_gamma)

    optimizer = create_optim(unet, text_encoder, cfg)
    scaler = torch.cuda.amp.GradScaler() if autocast_dtype == torch.float16 else None

    if compile_model and hasattr(torch, "compile"):
        patch_torch_compile()
        patch_compiled_autograd()
        unet = torch.compile(unet)
        if cfg.train.train_te and text_encoder is not None:
            text_encoder = torch.compile(text_encoder)

    mode_prefix = "pixel" if pixel_training else "latent"
    trace_name = f"{model_type}_{mode_prefix}_{'compile' if compile_model else 'eager'}"
    data_iter = iter(dataloader)

    print(f"Running PyTorch Profiler trace for {trace_name}...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=warmup_iters,
            active=profile_iters,
            repeat=1,
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=not compile_model,
        on_trace_ready=lambda p: trace_handler(p, output_dir, trace_name),
    ) as prof:
        for _ in range(1 + warmup_iters + profile_iters):
            with torch.profiler.record_function("## dataloader ##"):
                batch = next(data_iter)

            with torch.profiler.record_function("## ingest_or_vae ##"):
                (
                    latents,
                    cond,
                    attention_mask,
                    tag_weights,
                    pos_map,
                ) = unpack_batch(
                    batch=batch,
                    vae=vae,
                    device=device,
                    dtype=dtype,
                    autocast_dtype=autocast_dtype,
                    vae_mean=vae_mean,
                    vae_std=vae_std,
                    vae_batch_size=vae_batch_size,
                    is_latent=is_latent,
                    pixel_training=pixel_training,
                )

            optimizer.zero_grad(set_to_none=True)

            with torch.profiler.record_function("## text_encoder ##"):
                with torch.autocast(
                    device_type="cuda", dtype=autocast_dtype, enabled=True
                ):
                    with torch.set_grad_enabled(cfg.train.train_te):
                        encoder_hidden_states, attention_mask = text_encoder(
                            cond,
                            mask=attention_mask,
                            drop_mask=None,
                        )

            with torch.profiler.record_function("## backbone_forward ##"):
                with torch.autocast(
                    device_type="cuda", dtype=autocast_dtype, enabled=True
                ):
                    loss, metrics = objective.forward(
                        unet,
                        latents,
                        encoder_hidden_states,
                        tag_weights,
                        attention_mask=attention_mask,
                        pos_map=pos_map,
                    )

            with torch.profiler.record_function("## backward ##"):
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            with torch.profiler.record_function("## optimizer ##"):
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                    optimizer.step()

            prof.step()

    print(f"Trace profiling completed. Results exported to: {output_dir}")


def flow_matching_step_min_rf(
    unet_model: torch.nn.Module,
    x1: torch.Tensor,
    cond_input: torch.Tensor,
    dtype=torch.float32,
    log=False,
    return_norms: bool = False,
):
    if log:
        torch.cuda.nvtx.range_push("prepare_inputs")

    batch_size = x1.shape[0]

    nt = torch.randn((batch_size,)).to(x1.device, non_blocking=True)
    t = torch.sigmoid(nt)

    texp = t.view([batch_size, *([1] * len(x1.shape[1:]))])
    x0 = torch.randn_like(x1)
    xt = (1 - texp) * x0 + texp * x1
    # timestep for the UNet: Invert then Scale
    t_unet_input = (1.0 - t) * 999.0
    if log:
        torch.cuda.nvtx.range_pop()

    if log:
        torch.cuda.nvtx.range_push("forward")
    v_pred = unet_model(
        xt,
        t_unet_input,
        encoder_hidden_states=cond_input,
    )
    if log:
        torch.cuda.nvtx.range_pop()

    v_pred_metrics = []
    v_true_metrics = []
    if return_norms:
        v_true = x1 - x0
        loss = F.mse_loss(
            v_true.to(torch.float32), v_pred.to(torch.float32), reduction="none"
        )
        with torch.no_grad():
            v_pred_metrics.append(torch.norm(v_pred.detach()))
            v_true_metrics.append(torch.norm(v_true.detach()))
            v_pred_metrics.append(torch.mean(torch.abs(v_pred.detach())))
            v_true_metrics.append(torch.mean(torch.abs(v_true.detach())))

    else:
        loss = F.mse_loss(
            (x1 - x0).to(torch.float32), v_pred.to(torch.float32), reduction="none"
        )
    batchwise_mse = loss.mean(dim=list(range(1, len(x1.shape))))

    return batchwise_mse.mean(), v_pred_metrics, v_true_metrics


def train_flow_matching_min_rf(
    model: torch.nn.Module,
    train_loader,
    epochs: int = 10,
    lr: float = 1e-4,
    wd: float = 1e-6,
    device: str = "cuda",
    seed: int = None,
    dtype=torch.bfloat16,
    warmup_iters=10,
):
    torch.cuda.empty_cache()
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = bnb.optim.AdamW8bit(
        params,
        lr=lr,
        weight_decay=wd,
    )
    if hasattr(train_loader, "__len__") and len(train_loader) > 0:
        t_max_steps = epochs * len(train_loader)
    else:
        print("Warning: train_loader length unknown, using estimated T_max.")
        estimated_steps_per_epoch = 500000 // (
            train_loader.batch_size
            if hasattr(train_loader, "batch_size") and train_loader.batch_size
            else 16
        )
        t_max_steps = epochs * estimated_steps_per_epoch
        if t_max_steps == 0:
            t_max_steps = 10000

    warmup_steps_sched = round(0.02 * t_max_steps) if t_max_steps > 0 else 0
    warmup_steps_sched = (
        min(warmup_steps_sched, t_max_steps - 1) if t_max_steps > 0 else 0
    )

    scheduler = None
    if t_max_steps > 0 and warmup_steps_sched < t_max_steps:
        scheduler_1 = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps_sched if warmup_steps_sched > 0 else 1,
        )
        cosine_t_max = t_max_steps - warmup_steps_sched
        if cosine_t_max <= 0:
            cosine_t_max = 1
        scheduler_2 = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_t_max, eta_min=lr * 0.01
        )
        if warmup_steps_sched > 0:
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer, [scheduler_1, scheduler_2], milestones=[warmup_steps_sched]
            )
        else:
            scheduler = scheduler_2

    generator = torch.Generator(device=device)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        generator.manual_seed(seed)

    model.to(device)
    global_step = 0

    try:
        print(f"Starting Flow Matching training for SongUNet (EDM pre-trained)...")
        for epoch in range(epochs):
            model.train()
            epoch_loss_accum = 0.0
            num_batches_epoch = 0

            for i, batch in enumerate(train_loader):
                x1_batch = batch[0].to(dtype=dtype, non_blocking=True).to(device)
                torch._dynamo.mark_dynamic(x1_batch, (2, 3), min=32, max=128)
                torch._dynamo.mark_dynamic(x1_batch, 0, min=1, max=MAX_BATCH_SIZE)

                cond_input_batch = batch[1]
                print(f"X shape {x1_batch.shape} and text {cond_input_batch.shape}")
                cond_input_batch = torch.randn(x1_batch.shape[0], 77, 768).to(
                    device, non_blocking=True
                )
                torch._dynamo.mark_dynamic(cond_input_batch, 1, min=77, max=227)

                optimizer.zero_grad()
                # start profiling after warmup iterations
                if global_step == warmup_iters:
                    torch.cuda.cudart().cudaProfilerStart()
                # push range for current iteration
                if global_step >= warmup_iters:
                    torch.cuda.nvtx.range_push("iteration{}".format(global_step))

                # Use the flow_matching_step_profiled version
                loss, _, _ = flow_matching_step_min_rf(
                    unet_model=model,
                    x1=x1_batch,
                    cond_input=cond_input_batch,
                    dtype=dtype,
                    log=global_step >= warmup_iters,
                )

                if global_step >= warmup_iters:
                    torch.cuda.nvtx.range_push("backward")
                loss.backward()
                if global_step >= warmup_iters:
                    torch.cuda.nvtx.range_pop()

                if global_step >= warmup_iters:
                    torch.cuda.nvtx.range_push("clip_norm")
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if global_step >= warmup_iters:
                    torch.cuda.nvtx.range_pop()

                if global_step >= warmup_iters:
                    torch.cuda.nvtx.range_push("opt.step()")
                optimizer.step()
                if global_step >= warmup_iters:
                    torch.cuda.nvtx.range_pop()
                if scheduler:
                    scheduler.step()

                epoch_loss_accum += loss.detach().item()
                num_batches_epoch += 1

                global_step += 1
                if global_step >= warmup_iters:
                    torch.cuda.nvtx.range_pop()

                # Stop training after the first profiling cycle
                if global_step > warmup_iters + 10:
                    print(
                        "First profiling cycle complete. Exiting for analysis as profile_repeat_cycles is 1."
                    )
                    return

            avg_epoch_loss = (
                epoch_loss_accum / num_batches_epoch if num_batches_epoch > 0 else 0
            )
            print(f"End of Epoch {epoch + 1}: Avg Train Loss: {avg_epoch_loss:.4f}")

    except StopIteration:
        print("Training stopped early due to patience.")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        torch.cuda.cudart().cudaProfilerStop()
        print("Training finished.")


def train_fm_memory_flops(
    model: torch.nn.Module,
    train_loader,
    epochs: int = 1,
    lr: float = 1e-4,
    wd: float = 1e-6,
    device: str = "cuda",
    seed: int = None,
    dtype=torch.bfloat16,
    use_autocast: bool = False,
    warmup_iters: int = 10,
    profile_iters: int = 5,
):
    """
    Measures and reports the average FLOPs, TFLOPS, and peak memory usage
    over a specified number of profiling steps after a warmup period.
    """
    torch.cuda.empty_cache()
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
    )

    generator = torch.Generator(device=device)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        generator.manual_seed(seed)

    model.to(device)
    global_step = 0

    flops_list = []
    peak_memory_list = []
    time_list = []
    batch_sizes = []

    try:
        print(f"Starting memory and FLOPs profiling...")
        print(
            f"Warmup Steps: {warmup_iters}, Profile Steps: {profile_iters} Using autocast: {use_autocast}"
        )
        for epoch in range(epochs):
            model.train()
            flop_counter = FlopCounterMode(model, display=False)

            autocast_context_manager = torch.amp.autocast(
                device_type="cuda",
                enabled=use_autocast,
                dtype=torch.float16,
                cache_enabled=False,
            )
            grad_scaler = torch.amp.GradScaler("cuda") if use_autocast else None

            for i, batch in enumerate(train_loader):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                x1_batch = batch[0].to(dtype=dtype, device=device)
                cond_input_batch = torch.randn(x1_batch.shape[0], 77, 768).to(device)

                torch.cuda.reset_peak_memory_stats(device)

                optimizer.zero_grad(set_to_none=True)

                start_event.record()

                # forward pass
                with flop_counter:
                    with autocast_context_manager:
                        loss, _, _ = flow_matching_step_min_rf(
                            unet_model=model,
                            x1=x1_batch,
                            cond_input=cond_input_batch,
                            dtype=dtype,
                            log=False,
                        )

                    if grad_scaler is None:
                        loss.backward()
                    else:
                        grad_scaler.scale(loss).backward()

                    if grad_scaler is None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    else:
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        grad_scaler.step(optimizer)
                        grad_scaler.update()

                end_event.record()
                torch.cuda.synchronize()
                batch_sizes.append(x1_batch.shape[0])

                if global_step >= warmup_iters:
                    total_flops = flop_counter.get_total_flops()
                    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
                    iter_time_ms = start_event.elapsed_time(end_event)

                    flops_list.append(total_flops)
                    peak_memory_list.append(peak_mem_mb)
                    time_list.append(iter_time_ms / 1000.0)

                    print(
                        f"Profile Step {global_step - warmup_iters + 1}/{profile_iters} - "
                        f"Time: {iter_time_ms:.2f}ms, "
                        f"Peak Mem: {peak_mem_mb:.2f}MB, "
                        f"GFLOPs: {total_flops / 1e9:.2f},"
                        f"Batch size: {batch_sizes[-1]}"
                    )

                global_step += 1

                if global_step >= (warmup_iters + profile_iters):
                    avg_flops = np.mean(flops_list)
                    avg_peak_memory = np.mean(peak_memory_list)
                    avg_time = np.mean(time_list)
                    tflops = (avg_flops / avg_time) / 1e12 if avg_time > 0 else 0

                    print("\n--- Profiling Summary ---")
                    print(f"Average Iteration Time: {avg_time * 1000:.2f} ms")
                    print(f"Average Peak Memory: {avg_peak_memory:.2f} MB")
                    print(f"Average FLOPs per Iteration: {avg_flops / 1e9:.2f} GFLOPs")
                    print(f"Achieved Performance: {tflops:.2f} TFLOPS")
                    print(f"The batch sizes used were {batch_sizes}")
                    print("-------------------------\n")
                    return

    except KeyboardInterrupt:
        print("Profiling interrupted by user.")
    finally:
        print("Profiling finished.")


def train_fm_memory_view(
    model: torch.nn.Module,
    train_loader,
    epochs: int = 1,
    lr: float = 1e-4,
    wd: float = 1e-6,
    device: str = "cuda",
    seed: int = None,
    dtype=torch.bfloat16,
    warmup_iters: int = 8,
    profile_iters: int = 10,
    gradient_accumulation_steps: int = 1,
):
    """
    Measures and reports the average FLOPs, TFLOPS, and peak memory usage
    over a specified number of profiling steps after a warmup period.
    """
    torch.cuda.empty_cache()
    offloader = CPUGradientAccumulator(model)
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
    )

    generator = torch.Generator(device=device)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        generator.manual_seed(seed)

    model.to(device)
    global_step = 0

    try:
        print(f"Starting memory and FLOPs profiling...")
        print(f"Warmup Steps: {warmup_iters}, Profile Steps: {profile_iters}")
        for epoch in range(epochs):
            model.train()

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    skip_first=warmup_iters // 2,
                    wait=0,
                    warmup=warmup_iters,
                    active=profile_iters,
                    repeat=1,
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                on_trace_ready=trace_handler,
            ) as prof:
                for i, batch in enumerate(train_loader):
                    x1_batch = batch[0].to(dtype=dtype, device=device)
                    cond_input_batch = torch.randn(x1_batch.shape[0], 77, 768).to(
                        device
                    )
                    is_update_step = (i + 1) % gradient_accumulation_steps == 0
                    with record_function("## forward ##"):
                        loss, _, _ = flow_matching_step_min_rf(
                            unet_model=model,
                            x1=x1_batch,
                            cond_input=cond_input_batch,
                            dtype=dtype,
                            log=False,
                        )
                    # Scale loss for backward pass to average gradients
                    scaled_loss = loss / gradient_accumulation_steps
                    with record_function("## backward ##"):
                        scaled_loss.backward()
                    if is_update_step:
                        with record_function("## optimizer ##"):
                            # the offloader performs the optim and clip
                            norm = offloader.finalize_and_step(optimizer)

                    prof.step()

                    global_step += 1
                    # Exit after collecting enough samples
                    if global_step >= (warmup_iters + profile_iters):
                        return  # Exit after profiling is complete

    except KeyboardInterrupt:
        print("Profiling interrupted by user.")
    finally:
        print("Profiling finished.")


def train_flow_matching_logging(
    model: torch.nn.Module,
    train_loader,
    epochs: int = 10,
    lr: float = 1e-4,
    wd: float = 1e-6,
    device: str = "cuda",
    seed: int = None,
    dtype=torch.bfloat16,
    warmup_iters=10,
):
    torch.cuda.empty_cache()
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
    )
    if hasattr(train_loader, "__len__") and len(train_loader) > 0:
        t_max_steps = epochs * len(train_loader)
    else:
        print("Warning: train_loader length unknown, using estimated T_max.")
        estimated_steps_per_epoch = 500000 // (
            train_loader.batch_size
            if hasattr(train_loader, "batch_size") and train_loader.batch_size
            else 16
        )
        t_max_steps = epochs * estimated_steps_per_epoch
        if t_max_steps == 0:
            t_max_steps = 10000

    warmup_steps_sched = round(0.02 * t_max_steps) if t_max_steps > 0 else 0
    warmup_steps_sched = (
        min(warmup_steps_sched, t_max_steps - 1) if t_max_steps > 0 else 0
    )

    scheduler = None
    if t_max_steps > 0 and warmup_steps_sched < t_max_steps:
        scheduler_1 = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps_sched if warmup_steps_sched > 0 else 1,
        )
        cosine_t_max = t_max_steps - warmup_steps_sched
        if cosine_t_max <= 0:
            cosine_t_max = 1
        scheduler_2 = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_t_max, eta_min=lr * 0.01
        )
        if warmup_steps_sched > 0:
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer, [scheduler_1, scheduler_2], milestones=[warmup_steps_sched]
            )
        else:
            scheduler = scheduler_2

    dummy_logger = lambda *args, **kwargs: None
    inspector = ModelInspector(logging_fn=dummy_logger, model_dtype=dtype)
    inspector.register_hooks(unet)

    generator = torch.Generator(device=device)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        generator.manual_seed(seed)

    model.to(device)
    global_step = 0

    try:
        print(f"Starting Flow Matching training for SongUNet (EDM pre-trained)...")
        for epoch in range(epochs):
            model.train()
            epoch_loss_accum = 0.0
            num_batches_epoch = 0

            for i, batch in enumerate(train_loader):
                x1_batch = batch[0].to(dtype=dtype, non_blocking=True).to(device)
                torch._dynamo.mark_dynamic(x1_batch, (2, 3), min=32, max=128)
                torch._dynamo.mark_dynamic(x1_batch, 0, min=1, max=MAX_BATCH_SIZE)

                cond_input_batch = batch[1]
                print(f"X shape {x1_batch.shape} and text {cond_input_batch.shape}")
                cond_input_batch = torch.randn(x1_batch.shape[0], 77, 768).to(
                    device, non_blocking=True
                )
                torch._dynamo.mark_dynamic(cond_input_batch, 1, min=77, max=227)

                optimizer.zero_grad()
                if global_step == warmup_iters:
                    torch.cuda.cudart().cudaProfilerStart()
                if global_step >= warmup_iters:
                    torch.cuda.nvtx.range_push("iteration{}".format(global_step))

                loss, v_pred_metrics, v_true_metrics = flow_matching_step_min_rf(
                    unet_model=model,
                    x1=x1_batch,
                    cond_input=cond_input_batch,
                    dtype=dtype,
                    log=global_step >= warmup_iters,
                    return_norms=True,
                )

                norm_unet = torch.nn.utils.clip_grad_norm_(
                    unet.parameters(), max_norm=2.0
                )
                print(f"Loss {loss.item()} Norm: {norm_unet.item()}")
                print(
                    f"V true norm {v_true_metrics[0].item()} V true abs {v_true_metrics[1].item()}. V Pred norm {v_pred_metrics[0].item()} V pred abs {v_pred_metrics[1].item()}"
                )
                if global_step >= warmup_iters:
                    torch.cuda.nvtx.range_push("backward")
                loss.backward()
                if global_step >= warmup_iters:
                    torch.cuda.nvtx.range_pop()

                if global_step >= warmup_iters:
                    torch.cuda.nvtx.range_push("clip_norm")
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if global_step >= warmup_iters:
                    torch.cuda.nvtx.range_pop()

                if global_step >= warmup_iters:
                    torch.cuda.nvtx.range_push("opt.step()")
                optimizer.step()
                if global_step >= warmup_iters:
                    torch.cuda.nvtx.range_pop()
                if scheduler:
                    scheduler.step()

                epoch_loss_accum += loss.detach().item()
                num_batches_epoch += 1

                global_step += 1
                if global_step >= warmup_iters:
                    torch.cuda.nvtx.range_pop()

                if global_step > warmup_iters + 10:
                    print(
                        "First profiling cycle complete. Exiting for analysis as profile_repeat_cycles is 1."
                    )
                    return

            avg_epoch_loss = (
                epoch_loss_accum / num_batches_epoch if num_batches_epoch > 0 else 0
            )
            print(f"End of Epoch {epoch + 1}: Avg Train Loss: {avg_epoch_loss:.4f}")

    except StopIteration:
        print("Training stopped early due to patience.")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        torch.cuda.cudart().cudaProfilerStop()
        print("Training finished.")


@torch.inference_mode()
def profile_sampling(
    cfg: DictConfig,
    num_samples: int = 16,
    batch_size: int = 4,
    steps: int = 30,
    resolution: int = 512,
    output_dir: str = "profiling_results",
    compile_model: bool = False,
    pixel_sampling: bool = False,
) -> None:
    """
    Benchmarks generation latency across ODE trajectory steps, CFG
    evaluations, VAE decoding, and host transfer to identify generation bottlenecks.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if getattr(cfg.train, "dtype", "bf16") == "bf16"
        else torch.float32
    )
    autocast_dtype = (
        torch.bfloat16
        if getattr(cfg.train, "dtype", "bf16") == "bf16"
        else torch.float16
    )

    model_type = getattr(cfg.models, "model_type", "dual_stream")
    is_dit = model_type in ["dual_stream", "sprint_dual"]

    unet, text_encoder, vae, tokenizer, _ = load_trainable_model(
        models_path=cfg.paths.models,
        device=device,
        dtype=dtype,
        train_te=False,
        use_checkpointing=False,
        model_type=model_type,
        model_cfg=cfg.models,
        autocast_dtype=autocast_dtype,
    )
    unet.eval().requires_grad_(False)
    text_encoder.eval().requires_grad_(False)
    if vae is not None:
        vae.eval().requires_grad_(False)

    if compile_model and hasattr(torch, "compile"):
        print("Compiling sampling modules with torch.compile...")
        patch_torch_compile()
        unet = torch.compile(unet)
        text_encoder = torch.compile(text_encoder)

    schedule = LinearSchedule(device=device)

    sample_configs = [
        {
            "prompt": (
                "masterpiece, best quality, 1girl, hatsune miku, "
                "vocaloid, looking at viewer, detailed background"
            ),
            "negative_prompt": "worst quality, low quality",
            "height": resolution,
            "width": resolution,
            "sample_steps": steps,
            "cfg_scale": 6.0,
            "shift": 1.0,
            "seed": 42 + i,
        }
        for i in range(num_samples)
    ]

    vae_mean = torch.tensor(
        getattr(cfg.models, "vae_mean", 0.0), device=device, dtype=dtype
    ).view(1, -1, 1, 1)
    vae_std = torch.tensor(
        getattr(cfg.models, "vae_std", 1.0 / 0.18215),
        device=device,
        dtype=dtype,
    ).view(1, -1, 1, 1)
    in_channels = cfg.models.get("in_channels", 4)
    print(f"Using channels {in_channels}")
    prediction_target = getattr(cfg.train, "prediction_target", "v")

    print("\n" + "=" * 70)
    print(f"SAMPLING BENCHMARK ({num_samples} images, {steps} steps, {resolution}px)")
    print(f"Mode: {'PIXEL' if pixel_sampling else 'LATENT + VAE DECODE'}")
    print(f"Batch Size: {batch_size} | Device: {device}")
    print("=" * 70)

    # Warmup pass (2 samples)
    _ = generate_samples(
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        schedule=schedule,
        sample_configs=sample_configs[:2],
        global_batch_size=min(2, batch_size),
        device=str(device),
        dtype=dtype,
        diffusion_type=cfg.train.objective,
        use_unet_mult=False if is_dit else True,
        vae_mean=vae_mean,
        vae_std=vae_std,
        vae_batch_size=1,
        in_channels=in_channels,
        pixel_sampling=pixel_sampling,
        prediction_target=prediction_target,
        noise_scale=1.0 if not pixel_sampling else 2.0,
        clip_prediction=getattr(
            cfg.sampling,
            "clip_prediction",
            pixel_sampling,
        ),
        dynamic_thresholding=getattr(
            cfg.sampling,
            "dynamic_thresholding",
            pixel_sampling,
        ),
        cfg_interval=getattr(
            cfg.sampling,
            "cfg_interval",
            (0.0, 0.90) if pixel_sampling else (0.0, 1.0),
        ),
    )
    torch.cuda.synchronize()

    # Active timed profiling
    t_start = time.perf_counter()
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats(device)
    ev_start.record()

    images = generate_samples(
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        schedule=schedule,
        sample_configs=sample_configs,
        global_batch_size=batch_size,
        device=str(device),
        dtype=dtype,
        autocast_dtype=autocast_dtype,
        use_unet_mult=False if is_dit else True,
        pixel_sampling=pixel_sampling,
    )

    ev_end.record()
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    gpu_elapsed_ms = ev_start.elapsed_time(ev_end)
    wall_elapsed_s = t_end - t_start

    images_per_sec = num_samples / wall_elapsed_s
    sec_per_image = wall_elapsed_s / num_samples
    ms_per_step = (gpu_elapsed_ms / (num_samples * steps)) * batch_size
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    print("\n" + "=" * 70)
    print("SAMPLING PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Total Wall Clock Time     : {wall_elapsed_s:.2f} s")
    print(f"Total GPU Execution Time   : {gpu_elapsed_ms / 1000.0:.2f} s")
    print(f"Overall Generation Rate    : {images_per_sec:.2f} images/sec")
    print(f"Latency per Image          : {sec_per_image:.3f} s/image")
    print(
        f"Per-Step Trajectory Latency: {ms_per_step:.2f} ms/step (batch={batch_size})"
    )
    print(f"Peak VRAM Allocation       : {peak_mem_mb:.2f} MB")
    print("=" * 70)

    # Diagnostic projection for 97k dataset generation
    projected_hours_97k = (97000.0 * sec_per_image) / 3600.0
    print(
        f"[SYSTEM PROJECTION] Generating 97,000 synthetic samples at this "
        f"throughput requires: {projected_hours_97k:.2f} hours."
    )
    if not pixel_sampling and ms_per_step > 35.0:
        print(
            "[DIAGNOSTIC WARNING] High latency per step observed. Ensure "
            "CFG interval clipping is active and batch_size is maximized."
        )


if __name__ == "__main__":
    # Force 'fork' on Linux to prevent Python 3.14 forkserver crash
    # with DataLoader workers (PyTorch reduction rebuild_storage_fd EINVAL)
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    try:
        mp.set_sharing_strategy("file_system")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="CUDA & Pipeline Profiler for DiT / Pixel Flow Matching."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to training config YAML.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["breakdown", "trace", "sampling"],
        default="breakdown",
        help="Profile mode: component breakdown, Chrome trace, or sampling.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="profiling_results",
        help="Directory to save traces and summaries.",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=5,
        help="Number of warmup iterations before measuring.",
    )
    parser.add_argument(
        "--profile_iters",
        type=int,
        default=10,
        help="Number of active profiling iterations.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile during profiling.",
    )
    parser.add_argument(
        "--pixel_training",
        action="store_true",
        help="Enable pixel-space adaptation (DiP) profiling.",
    )
    parser.add_argument(
        "--stage1",
        action="store_true",
        help="Profile Stage 1: train only x_embedder and pixel decoder.",
    )
    parser.add_argument(
        "--online_vae",
        action="store_true",
        help="Force online VAE encoding profiling for raw images.",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=30,
        help="Euler ODE steps for sampling mode benchmark.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution for profiling runs.",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if os.path.exists(args.config):
        base_cfg = OmegaConf.load(args.config)
        cli_cfg = OmegaConf.from_cli(args.opts)
        cfg = OmegaConf.merge(base_cfg, cli_cfg)
    else:
        cfg = OmegaConf.create(
            {
                "paths": {"models": "models"},
                "models": {
                    "model_type": "dual_stream",
                    "hidden_size": 768,
                    "depth": 16,
                    "num_heads": 12,
                    "patch_size": 2,
                    "in_channels": 32,
                },
                "data": {
                    "dataset_type": "streaming",
                    "streaming_dataset_name": ("aipracticecafe/curated-danbooru-2026"),
                    "resolution": args.resolution,
                },
                "train": {
                    "dtype": "fp32",
                    "batch_size": 4,
                    "train_te": False,
                    "use_checkpointing": True,
                    "objective": "flow_matching",
                    "shift": 1.0,
                    "lr": 1e-4,
                    "wd": 1e-2,
                    "gpu_peak_tflops": 165.2,
                },
            }
        )
        print("Using default config.")

    if args.online_vae:
        cfg.data.is_latent = False

    if args.mode == "breakdown":
        profile_pipeline_breakdown(
            cfg=cfg,
            warmup_iters=args.warmup_iters,
            profile_iters=args.profile_iters,
            output_dir=args.output_dir,
            compile_model=args.compile,
            pixel_training=args.pixel_training,
            train_only_output=args.stage1,
        )
    elif args.mode == "trace":
        profile_pipeline_trace(
            cfg=cfg,
            warmup_iters=args.warmup_iters,
            profile_iters=args.profile_iters,
            output_dir=args.output_dir,
            compile_model=args.compile,
            pixel_training=args.pixel_training,
            train_only_output=args.stage1,
        )
    elif args.mode == "sampling":
        profile_sampling(
            cfg=cfg,
            num_samples=16,
            batch_size=cfg.train.batch_size,
            steps=args.sample_steps,
            resolution=args.resolution,
            output_dir=args.output_dir,
            compile_model=args.compile,
            pixel_sampling=args.pixel_training,
        )

    exit()
    # train_flow_matching_min_rf(
    # model=unet_model, # Pass the U-Net part
    # train_loader=dataloader,
    # epochs=2,
    # lr=5e-5,
    # wd=0.01,
    # device=str(device), # Pass device as string
    # seed=seed,
    # dtype=dtype,
    # warmup_iters=10,
    # )
    # train_fm_memory_flops(
    # model=unet_model,
    # train_loader=dataloader,
    # epochs=2,
    # lr=5e-5,
    # wd=0.01,
    # device=str(device),
    # seed=seed,
    # dtype=dtype,
    # use_autocast=True,
    # warmup_iters=10,
    # )
    train_fm_memory_view(
        model=unet_model,
        train_loader=dataloader,
        epochs=2,
        lr=5e-5,
        wd=0.01,
        device=str(device),
        seed=seed,
        dtype=dtype,
        warmup_iters=8,
        gradient_accumulation_steps=2,
    )
    # train_flow_matching_logging(
    #     model=unet_model, # Pass the U-Net part
    #     train_loader=dataloader,
    #     epochs=2,
    #     lr=5e-5,
    #     wd=0.01,
    #     device=str(device), # Pass device as string
    #     seed=seed,
    #     dtype=dtype,
    #     warmup_iters=10,
    # )

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # # Print the top 20 functions by cumulative time
    # stats.print_stats(20)
    # # Save the full report for later analysis
    # stats.dump_stats("training_profile.prof")
