# ASUKA-FM

**ASUKA-FM** (Anime Shift Using Kinetic Alignment - Flow Matching) is an experimental research repository aimed at fine-tuning the Stable Diffusion 1.5 (SD1.5) architecture using modern Flow Matching objectives.

The name is a backronym inspired by Asuka Langley Soryu from *Neon Genesis Evangelion*, symbolizing kinetic nature of Flow Matching velocity precition shifting toward a modern anime aesthetic.

## 📖 Overview

Recent advancements in text-to-image generation have shifted from classical DDPM (Denoising Diffusion Probabilistic Models) to Flow Matching (FM) and Rectified Flow objectives, often paired with DiT architectures.

Focusing on environments with budget constraints, this project investigates whether applying a Flow Matching objective to an already pre-trained DDPM model (NovelAI's nai-v2 SD 1.5 from 2023) can accelerate the learning of new concepts and shift the distribution to modern anime styles.

## ✨ Key Features & Architecture

### 1. Flow Matching Objective
Instead of predicting noise ($\epsilon$) like standard SD1.5, this repository modifies the training loop to predict the velocity vector ($v = x_1 - x_0$) using a Conditional Flow Matching objective. This results in straighter sampling trajectories and allows for efficient ODE-based sampling (e.g., Euler, Heun).

### 2. CPU-Offloaded Gradient Accumulation
To train with larger effective batch sizes on consumer GPUs, we implemented a custom `CPUGradientAccumulator` (`src/utils/act_grad_checkpointing.py`). It asynchronously offloads gradients to the CPU via pinned memory and CUDA streams, accumulating them in system RAM before applying the optimizer step. This drastically reduces VRAM spikes at the cost of CPU/PCIe bandwidth.

### 3. High-Performance HDF5 Data Loading
Training on 200k-1M high-resolution images requires extreme I/O optimization.
* **Precomputed Latents:** VAE latents and CLIP text embeddings are precomputed for reducing disk space and computation costs.
* **Sharding & Bucketing:** The dataset is sharded across multiple >2GB `.h5` files. A custom `BucketBatchSampler` groups images by aspect ratio and sequence length to minimize padding.

### 4. Prompt Upsampling & Tag Weighting
Longer prompts generally improve convergence. We utilize a SwinV2 Danbooru tagger to upsample missing tags. To combat the severe data imbalance inherent in Danbooru (where some tags appear millions of times and others only a few dozen), the loss is multiplied by a computed tag weight based on rarity.

## 📦 Dataset Generation

This repository expects data to be pre-processed into sharded HDF5 files containing latents and tokenized prompts.

To generate the dataset required for this training pipeline, please use our companion repository: **[danbooru-dataset-toolkit](https://github.com/aipracticecafe/danbooru-dataset-toolkit)**. It handles sampling, downloading, deduplication, SwinV2 prompt upsampling, and HDF5 latent encoding.

## 🚀 Usage

### Prerequisites
* Python 3.10+
* PyTorch 2.5+
* `bitsandbytes`, `h5py`, `wandb`, `safetensors`

### Training
Configure your paths and hyperparameters in `config.yaml`, then launch the training script using `torchrun` for Distributed Data Parallel (DDP):

For background execution a script is provided with different commands like `tail` and `stop` to view and kill the process. Recommended when working with cloud VM.
```bash
# Example using the provided management script
./manage_train.sh start
```

Or manually:
```bash
torchrun --nproc_per_node=2 scripts/train.py --config config.yaml
```

### Profiling
The repository includes a comprehensive profiling suite in the `profiler/` directory to measure FLOPs, memory bandwidth, and kernel execution times using NVIDIA Nsight Systems.
```bash
python profiler/cuda_profiler.py
```

## 🔬 Experimental Results & Post-Mortem

This repository serves as a transparent log of our research into adapting DDPM models to Flow Matching. Below are our findings, theoretical justifications, and the results of our successful alpha run.

The result of the experiments is published at [aipracticecafe/nai-anime-v2-fm-alpha](https://huggingface.co/aipracticecafe/nai-anime-v2-fm-alpha).

### The 2-Stage Adaptation Strategy
Through our ablation studies, we found that forcing a DDPM-pretrained model to immediately learn a Flow Matching vector field across all layers causes catastrophic forgetting and slow convergence. To successfully adapt the model, **training must be done in two stages**:
1. **Stage 1 (Head Adaptation):** Fine-tune *only* the output head and its normalization layer (using the `train_only_output` flag in the config). This allows the model to learn the basic scale and distribution of the new velocity target without destroying the deep feature representations.
2. **Stage 2 (Full Fine-tuning):** Once the model starts generating coherent anime images from Stage 1, load the adapted output head and proceed to fine-tune the entire UNet.

### Theoretical Advantages: Why Flow Matching over v-prediction?
While v-prediction (used in models like NAI-v3 and NoobAI) solves zero-terminal SNR and color shifting issues, Rectified Flow / Flow Matching offers two distinct mathematical advantages:
1. **Straight Sampling Paths:** The diffusion path that transforms noise to data in a variance-preserving DDPM scheduler is curved (an arc). Flow Matching utilizes Optimal Transport to create a straight path between noise and data. This shorter, linear path is significantly easier to approximate using ODE solvers (like Euler), leading to faster and higher-quality sampling.
2. **Time-Independent Velocity:** In standard v-prediction, the target is $v = \alpha\epsilon - \sigma x_0$. This means the prediction target changes drastically depending on the timestep $t$ (predicting mostly data at $t \approx 1$ and mostly noise at $t \approx 0$). In Flow Matching, the velocity target $v = \epsilon - x_0$ is theoretically time-independent, making the neural network's prediction task much more stable across the entire sampling trajectory.

*(Note: Recent papers like "Diff2Flow" suggest using LoRA for this adaptation. While LoRA plateaus quickly—which is great for fast training—our experiments indicate that full fine-tuning provides the necessary capacity for continuous improvement over longer training runs.)*

### Alpha Model Release: Evangelion 260K
Following the 2-stage strategy, we successfully trained an alpha model on a curated dataset of 260,000 samples (focusing on modern anime and specific series like *Neon Genesis Evangelion*).

**Training Details:**
* **Hardware:** 2x RTX 4090 (~3 days)
* **Epochs:** 20
* **Effective Batch Size:** ~336 (12 per GPU, 14 gradient accumulation steps). *High batch sizes are crucial for learning new concepts effectively.*
* **Learning Rate:** 2.5e-5 (Constant with 0.05 warmup)
* **Shift:** 2.0

**Inference & Limitations:**
* **Samplers:** Because the objective is Rectified Flow, standard DDPM/DDIM samplers will not work. You must use Rectified Flow compatible nodes (e.g., `ModelSamplingRectifiedFlow` in ComfyUI or our updated [Forge WebUI fork](https://codeberg.org/aipracticecafe/stable-diffusion-webui-forge)) with Euler/DPM++ SDE and a Shift of 3-4.
* **Resolution Degradation:** The base NAI-v2 model could natively generate 1 Megapixel images. Because our fine-tuning was strictly bucketed around a 512px base area, the native 1MP capacity has degraded (proportions break). We recommend generating at `768x1024` and using an anime upscaler (like 4x NMKD YandereNeoXL) with Hires.fix.


## References
*   **Danbooru2024 Dataset**: [p1atdev/danbooru-2024](https://huggingface.co/datasets/p1atdev/danbooru-2024)
*   **SwinV2 Tagger Model**: [SmilingWolf/wd-swinv2-tagger-v3](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3)
*   The custom aesthetic classification model's data and training is done with the following repository [Aesthetic](https://github.com/deeplearningcafe/aesthetic).
*   **Relevant papers**: [Illustrious paper](https://arxiv.org/abs/2409.19946), [NAI v3 paper](https://arxiv.org/abs/2409.15997)
*   **Relevant implementations**: [CabalResearch/NoobAI-RectifiedFlow-Experimental](https://huggingface.co/CabalResearch/NoobAI-RectifiedFlow-Experimental), [CabalResearch/Mugen](https://huggingface.co/CabalResearch/Mugen), [ChenkinRF/ChenkinNoob-XL-v0.3-Rectified-Flow](https://huggingface.co/ChenkinRF/ChenkinNoob-XL-v0.3-Rectified-Flow) and [nyanko7/nyaflow-xl-alpha](https://huggingface.co/nyanko7/nyaflow-xl-alpha).
*   **Gradient checkpointing from unsloth-zoo**: [unsloth-zoo](https://github.com/unslothai/unsloth-zoo)
*   **AdamW with kahan summantion and optimal transport path from bluvoll's fork of sd-scripts**: [bluvoll/sd-scripts-f2vae](https://github.com/bluvoll/sd-scripts-f2vae)

## Author
[aipracticecafe](https://github.com/deeplearningcafe)
[aipracticecafe-codeberg](https://codeberg.org/aipracticecafe)

## License
This project is licensed under the `MIT`. Details are in the [LICENSE](LICENSE.txt) file. I don't own the data from the `danbooru` website nor the pretrained checkpoints used for the experiments.

