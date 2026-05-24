# CUDA Profiling and Benchmarking

This document provides an overview of the tools available in this directory
for profiling the U-Net model's performance, measuring computational load
(FLOPs), and analyzing memory usage.

## Table of Contents
- [High-Level Profiling (`cuda_profiler.py`)](#high-level-profiling-cuda_profilerpy)
    - [`train_flow_matching_min_rf`](#train_flow_matching_min_rf)
    - [`train_fm_memory_flops`](#train_fm_memory_flops)
    - [`train_fm_memory_view`](#train_fm_memory_view)
- [Module-Level Benchmarking (`benchmark_unet.py`)](#module-level-benchmarking-benchmark_unetpy)
    - [`benchmark_module`](#benchmark_module)
- [Benchmark Results](#benchmark-results)
    - [Batch Size ~2](#batch-size-2)
    - [Batch Size ~4](#batch-size-4)
- [Performance Analysis (from Nsight Systems)](#performance-analysis-from-nsight-systems)
- [How to Run](#how-to-run)

## High-Level Profiling (`cuda_profiler.py`)

This script is designed for end-to-end profiling of the entire training
loop. It provides three different functions to analyze performance from
different perspectives.

#### `train_flow_matching_min_rf`
This function is the primary tool for deep profiling with **NVIDIA Nsight
Systems**. It is instrumented with `torch.cuda.nvtx` ranges, which label
different parts of the training step (e.g., `forward`, `backward`,
`opt.step()`). This allows Nsight Systems to generate a detailed timeline,
correlating GPU kernel executions with specific parts of your code.

#### `train_fm_memory_flops`
This function provides a quick and easy way to get a high-level summary of
performance without external tools. It measures:
-   **Average Iteration Time**: The wall-clock time per training step.
-   **Peak Memory Usage**: The maximum GPU memory allocated during a step.
-   **GFLOPs**: Giga-Floating Point Operations per iteration.
-   **TFLOPS**: Tera-Floating Point Operations per second, indicating the
    achieved computational throughput.

#### `train_fm_memory_view`
This function uses the built-in `torch.profiler` to generate a detailed
trace of CPU and CUDA activities. It is configured to produce two key
artifacts:
1.  **Chrome Trace (`.json.gz`)**: A file that can be loaded into Chrome's
    `chrome://tracing` or Perfetto for an interactive, operator-level view
    of the execution timeline.
2.  **Memory Timeline (`.html`)**: A visual representation of memory
    allocations and deallocations over time, which is invaluable for
    debugging memory leaks or unexpected usage spikes.

## Module-Level Benchmarking (`benchmark_unet.py`)

This script is used to benchmark individual `nn.Module` components of the
U-Net in isolation. This is useful for identifying which specific layers
are the most computationally expensive and for testing the impact of
optimizations like `torch.compile` on a smaller scale.

#### `benchmark_module`
A utility function that takes any `nn.Module` and a set of dummy inputs,
then measures and reports the average execution time for its forward pass
and forward + backward pass.

## Benchmark Results

The following benchmarks were run on a **GTX 1660 Super** to compare
different activation checkpointing strategies.

### Batch Size ~2

| Checkpointing Strategy | Avg. Time (ms) | Peak Memory (MB) | GFLOPs | TFLOPS |
| :--- | :--- | :--- | :--- | :--- |
| **Unsloth (Smart)** | 2073.53 | **2788.40** | 2704.64 | **1.30** |
| **None** | 1918.11 | 3887.01 | 2301.76 | 1.20 |

**Observation**: Unsloth's smart activation checkpointing reduces peak
memory by over **1 GB** with a minimal impact on throughput (TFLOPS).

### Batch Size ~4

| Checkpointing Strategy | Avg. Time (ms) | Peak Memory (MB) | GFLOPs | TFLOPS |
| :--- | :--- | :--- | :--- | :--- |
| **Unsloth (Smart)** | 4363.02 | **3670.82** | 5671.04 | 1.30 |
| **Torch Naive** | 4043.75 | 4066.84 | 5404.14 | **1.34** |

**Observation**: With a larger batch size, naive Torch checkpointing is
slightly faster, but Unsloth still maintains a significant **~400 MB**
memory advantage.

## Performance Analysis (from Nsight Systems)

-   **`torch.compile` Issues**: On Turing-architecture GPUs (like the GTX
    1660 Super), `torch.compile` is not advantageous for this model. The
    dynamic shapes (variable batch size and image dimensions from the
    bucket sampler) lead to excessive recompilation and extremely long
    initial compile times (10-30 minutes), negating any potential runtime
    gains.
-   **Backward Pass is the Bottleneck**: The `backward()` call is the most
    expensive part of the training step, taking between 2.5 and 3.1
    seconds for a batch size of 3-5. The optimizer step follows, taking
    1.2 to 1.6 seconds.
-   **Attention Dominates Compute Time**: The attention mechanism is
    responsible for approximately **70%** of the total GPU computation time.
    -   The backward pass of the attention layers alone accounts for ~60%
        of the compute time.
    -   The forward pass of attention accounts for another ~10%.
-   **Low Warp Occupancy in Attention**: The attention kernels exhibit low
    compute warp utilization. This is likely due to non-optimal hidden
    dimensions (e.g., 40, 80), which are not multiples of hardware-friendly
    numbers (like 64 or 128). This forces the GPU to pad the inputs,
    leading to allocated but unused warps and a low theoretical occupancy
    of around 12.5%.

## How to Run

### Benchmarking Modules
To benchmark the individual U-Net components:
```bash
python benchmark_unet.py
```

### Profiling the Full Training Loop
1.  Open `cuda_profiler.py`.
2.  In the `if __name__ == "__main__":` block, uncomment one of the three
    profiling function calls (`train_flow_matching_min_rf`,
    `train_fm_memory_flops`, or `train_fm_memory_view`).
3.  Run the script:
    ```bash
    python cuda_profiler.py
    ```

### Profiling with NVIDIA Nsight Systems
To capture a detailed trace for analysis in Nsight Systems, use the
following command. This requires `train_flow_matching_min_rf` to be active
in the script.
```bash
sudo env "PATH=$PATH" nsys profile \
--gpu-metrics-device=0 \
-t cuda,nvtx,osrt,cudnn,cublas \
--cudabacktrace=true \
--capture-range=cudaProfilerApi \
--capture-range-end=stop \
python cuda_profiler.py
```