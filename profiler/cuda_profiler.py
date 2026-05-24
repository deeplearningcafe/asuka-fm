import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from torch.utils.flop_counter import FlopCounterMode
import socket
from datetime import datetime, timedelta
import bitsandbytes as bnb
import cProfile, pstats

import sys

sys.path.append("..")

import os

current_file_path = os.path.abspath(__file__)

current_dir = os.path.dirname(current_file_path)

common_parent_dir = os.path.dirname(current_dir)

folder_A_path = os.path.join(common_parent_dir, "models")
print(folder_A_path)
sys.path.insert(0, folder_A_path)
sys.path.insert(0, common_parent_dir)

from unet import UnetConfig, Unet
from h5_latent_dataset import H5LatentDataset, BucketBatchSampler, worker_init_fn
from torch.utils.data import DataLoader
from utils import (
    get_torch_compile_options,
    patch_unsloth_smart_gradient_checkpointing,
    patch_torch_compile,
    patch_compiled_autograd,
    CPUGradientAccumulator,
)
from train_fm_sd import ModelInspector


MAX_BATCH_SIZE = 64
TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


def trace_handler(prof: torch.profiler.profile):
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{host_name}_{timestamp}"

    prof.export_chrome_trace(f"{file_prefix}.json.gz")

    prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")


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


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    seed = 46
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")

    # Load model (U-Net part)
    dtype = torch.float32
    torch.cuda.empty_cache()
    unet_model = unet = (
        Unet(
            UnetConfig(
                block_out_channels=[dim // 2 for dim in [320, 640, 1280, 1280]],
                cross_attention_dim=768,
                norm_num_groups=16,
                use_checkpointing=True,
            )
        )
        .to(device)
        .train()
    )
    # Enable gradients for all parameters in the U-Net part

    for param in unet_model.parameters():
        param.requires_grad = True
    # only finetune output head
    # unet_model.conv_norm_out.bias.requires_grad = True
    # unet_model.conv_norm_out.weight.requires_grad = True
    # unet_model.conv_out.bias.requires_grad = True
    # unet_model.conv_out.weight.requires_grad = True

    torch_compile_options = get_torch_compile_options(
        epilogue_fusion=True,
        max_autotune=True,
        shape_padding=True,
        debug=True,
        cudagraphs=False,
        coordinate_descent_tuning=False,
        logging=True,
        combo_kernels=False,
        group_fusion=False,
        memory_planning=False,
        multi_kernel=False,
        use_block_ptr=False,
    )
    patch_torch_compile(True)
    patch_unsloth_smart_gradient_checkpointing(dtype=dtype)
    patch_compiled_autograd()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Trainable parameters in unet: {count_parameters(unet) / 1e6} M")

    # Memory stats
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3  # Convert to GB

        print(f"\nAllocated memory: {allocated:.4f} GB")
        print(f"Reserved memory: {reserved:.4f} GB\n")

    # Dataset
    METADATA_PATH = "latents/metadata.json"
    H5_ROOT_DIR = "latents"
    BASE_BATCH_SIZE = 4
    BASE_AREA = 64 * 96  # latents base area 64*64
    LENGTH_PENALTY = 0.1
    DROP_LAST = False
    SEED = 42
    NUM_WORKERS = 1
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2
    PERSISTENT_WORKERS = True
    INITIAL_EPOCH_FOCUS_LOW_RES = 2
    LOW_RES_FOCUS_FACTOR = 3.0
    LOW_RES_AREA_PERCENTILE = 0.4

    world_size = torch.cuda.device_count()
    rank = 0

    dataset = H5LatentDataset(
        metadata_path=METADATA_PATH,
        h5_root_dir=H5_ROOT_DIR,
    )

    batch_sampler = BucketBatchSampler(
        dataset=dataset,
        base_batch_size=BASE_BATCH_SIZE,
        base_resolution_area=BASE_AREA,
        length_penalty_power=LENGTH_PENALTY,
        drop_last=DROP_LAST,
        seed=SEED,
        world_size=world_size,
        rank=rank,
        initial_epoch_focus_low_res=INITIAL_EPOCH_FOCUS_LOW_RES,
        low_res_focus_factor=LOW_RES_FOCUS_FACTOR,
        low_res_area_percentile=LOW_RES_AREA_PERCENTILE,
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
        worker_init_fn=worker_init_fn,
    )

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
