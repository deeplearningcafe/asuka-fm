import time
from typing import Callable, Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
import os, logging

os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
os.environ["TORCH_LOGS"] = (
    "dynamo,graph_breaks,recompiles,graph_code,aot_joint_graph,aot_graphs,compiled_autograd_verbose"
)
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
torch._logging.set_logs(
    dynamo=logging.WARN,
    inductor=logging.WARN,
    graph_breaks=True,
    recompiles=True,
    recompiles_verbose=True,
    compiled_autograd_verbose=False,
    aot_joint_graph=False,
    aot_graphs=False,
    perf_hints=True,
)
torch._dynamo.config.verbose = True

import os
import sys

current_file_path = os.path.abspath(__file__)

current_dir = os.path.dirname(current_file_path)

common_parent_dir = os.path.dirname(current_dir)
folder_A_path = os.path.join(common_parent_dir, "models")
sys.path.insert(0, folder_A_path)
from unet import ResnetBlock, AttentionBlock, UnetConfig, Unet


def mean(x: list[float]) -> float:
    """Calculates the mean of a list of floats."""
    return sum(x) / len(x) if x else 0.0


def benchmark(
    description: str, run: Callable, num_warmups: int = 5, num_trials: int = 20
):
    """
    Benchmark a function by running it multiple times and returning the mean
    execution time. This function is adapted directly from lecture_06.py.

    Args:
        description (str): A description of the benchmark, printed to the
                           console.
        run (Callable): A zero-argument function that executes the code to
                        be benchmarked.
        num_warmups (int): The number of warmup runs before timing.
        num_trials (int): The number of timed trials to average.

    Returns:
        float: The mean execution time in milliseconds.
    """
    # Warmup
    for _ in range(num_warmups):
        run()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Perform the actual timed trials.
    times = []
    for _ in range(num_trials):
        start_time = time.time()
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        times.append((end_time - start_time) * 1000)

    mean_time = mean(times)
    return mean_time


def benchmark_module(
    module: nn.Module,
    input_args: Dict[str, Tuple[int, ...]],
    description: str,
    num_warmups: int = 5,
    num_trials: int = 20,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    A reusable utility to benchmark the forward and backward pass of any
    nn.Module.

    Args:
        module (nn.Module): The neural network module to benchmark.
        input_args (Dict[str, Tuple[int, ...]]): A dictionary mapping
            input argument names to their tensor shapes.
        description (str): A description for the benchmark run.
        num_warmups (int): Number of warmup iterations.
        num_trials (int): Number of timed iterations.
        dtype (torch.dtype): The data type for the module and inputs.
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return

    device = "cuda"
    module.to(device, dtype=dtype)

    dummy_inputs = {
        name: torch.randn(shape, device=device, dtype=dtype)
        for name, shape in input_args.items()
    }

    print(f"--- Benchmarking {description} ---")

    # forward pass only
    module.eval()
    run_forward = lambda: module(**dummy_inputs)
    forward_time = benchmark(description, run_forward, num_warmups, num_trials)
    print(f"  Forward Pass: {forward_time:.3f} ms")

    # forward + backward pass
    module.train()

    def run_forward_backward():
        module.zero_grad()

        output = module(**dummy_inputs)

        loss_tensor = output[0] if isinstance(output, tuple) else output

        # simple scalar loss
        loss = loss_tensor.mean()

        loss.backward()

    total_time = benchmark(description, run_forward_backward, num_warmups, num_trials)
    print(f"  Forward + Backward Pass: {total_time:.3f} ms")

    # backward pass time
    backward_time = total_time - forward_time
    print(f"  Calculated Backward Pass: {backward_time:.3f} ms")
    print("-" * (len(description) + 25))


if __name__ == "__main__":
    batch_size = 4
    height, width = 64, 64
    time_embed_dim = 1280
    cross_attn_dim = 768
    config = UnetConfig()

    unet_inputs = {
        "x": (batch_size, 4, height, width),
        "timestep": (batch_size),
        "encoder_hidden_states": (batch_size, 227, cross_attn_dim),
    }

    resnet_channels = 320
    resnet = ResnetBlock(
        input_channels=resnet_channels,
        output_channels=resnet_channels,
        time_embeddings=time_embed_dim,
        num_groups=config.norm_num_groups,
        eps=config.norm_eps,
    )
    resnet_inputs = {
        "x": (batch_size, resnet_channels, height, width),
        "temb": (batch_size, time_embed_dim),
    }
    benchmark_module(
        module=resnet,
        input_args=resnet_inputs,
        description="ResnetBlock (320 channels)",
        dtype=torch.float32,
    )

    attn_channels = 640
    n_head = attn_channels // config.attention_head_dim
    attention = AttentionBlock(
        input_channels=attn_channels,
        cross_attention_dim=cross_attn_dim,
        n_head=n_head,
        num_groups=config.norm_num_groups,
        eps=config.norm_eps,
    )
    # https://github.com/unslothai/unsloth-zoo/blob/26615eb3021b92abbfc8f895da4cd6803322b658/unsloth_zoo/patching_utils.py#L85
    torch_compile_options = {
        "epilogue_fusion": True,
        "max_autotune": True,
        "shape_padding": True,
        "triton.cudagraphs": False,
        "guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe,
    }
    attention = torch.compile(
        attention,
        dynamic=True,
        options=torch_compile_options,
        fullgraph=False,
    )

    attention_inputs = {
        "x": (batch_size, attn_channels, height // 2, width // 2),
        "encoder_hidden_states": (batch_size, 77, cross_attn_dim),
    }
    benchmark_module(
        module=attention,
        input_args=attention_inputs,
        description="AttentionBlock (640 channels)",
        dtype=torch.float32,
    )

