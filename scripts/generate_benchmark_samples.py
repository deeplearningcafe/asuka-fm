"""Generates benchmark images using prompt layouts and logic from i1."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, List, Tuple

from omegaconf import OmegaConf
import torch
from tqdm import tqdm

from src.diffusion.sampling import generate_samples
from src.diffusion.schedules import LinearSchedule
from src.models.factory import load_trainable_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# from https://github.com/zlab-princeton/i1/blob/main/jax/inference/generate.py
def get_caption_list(name: str, prompt_dir: Path) -> List[str]:
    """Loads prompt lists matching the exact i1 prompt catalog layout."""

    def _prompt_path(filename: str) -> Path:
        return prompt_dir / filename

    if name == "geneval":
        with open(_prompt_path("geneval.jsonl"), "r", encoding="utf-8") as fp:
            metadatas = [json.loads(line) for line in fp]
        return [metadata["prompt"] for metadata in metadatas]

    if name == "geneval_simple_rewrite":
        with open(_prompt_path("geneval_simple_rewrite.txt"), "r") as fp:
            return [caption.strip() for caption in fp.readlines()]

    if name == "geneval_complex_rewrite":
        p = _prompt_path("geneval_complex_rewrite.jsonl")
        with open(p, "r", encoding="utf-8") as fp:
            metadatas = [json.loads(line) for line in fp]
        return [metadata["prompt"] for metadata in metadatas]

    if name.startswith("geneval_repeated"):
        with open(_prompt_path(f"{name}.txt"), "r", encoding="utf-8") as fp:
            return [caption.strip() for caption in fp.readlines()]

    if name in ("dpg", "dpg_simple_rewrite", "dpg_complex_rewrite"):
        with open(_prompt_path(f"{name}.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        return [item[1].strip() for item in data]

    if name in ("prism", "prism_simple_rewrite", "prism_complex_rewrite"):
        with open(_prompt_path(f"{name}.json"), "r", encoding="utf-8") as f:
            return json.load(f)

    if name in (
        "CVTG-2K",
        "CVTG-2K_simple_rewrite",
        "CVTG-2K_complex_rewrite",
    ):
        with open(_prompt_path(f"{name}.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        return [item[1].strip() for item in data]

    if name in (
        "longtext",
        "longtext_simple_rewrite",
        "longtext_complex_rewrite",
    ):
        caption_list = []
        with open(_prompt_path(f"{name}.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                caption_list.append(obj["prompt"])
        return caption_list

    raise ValueError(f"No prompt list found for prompt_type: {name}")


def build_work_items(prompt_type: str, captions: List[str]) -> List[Tuple[Any, str]]:
    """Builds (metadata_idx, caption) tuples according to i1 grouping."""
    if (
        ("geneval" in prompt_type)
        or ("dpg" in prompt_type)
        or ("longtext" in prompt_type)
    ):
        # 4 samples per prompt
        return [
            ((it, s_idx), caption)
            for it, caption in enumerate(captions)
            for s_idx in range(4)
        ]
    return [(it, caption) for it, caption in enumerate(captions)]


def save_image_sample(
    img: Any,
    prompt_type: str,
    meta_idx: Any,
    target_dir: Path,
) -> None:
    """Saves image according to benchmark-specific hierarchy in i1."""
    if ("geneval" in prompt_type) or ("dpg" in prompt_type):
        p_idx, s_idx = meta_idx
        save_folder = target_dir / f"{p_idx:05d}" / "samples"
        save_folder.mkdir(parents=True, exist_ok=True)
        img.save(save_folder / f"{s_idx:05d}.png")
    elif "longtext" in prompt_type:
        p_idx, s_idx = meta_idx
        target_dir.mkdir(parents=True, exist_ok=True)
        img.save(target_dir / f"{p_idx:04d}_{s_idx}.png")
    elif ("prism" in prompt_type) or ("CVTG-2K" in prompt_type):
        p_idx = meta_idx
        target_dir.mkdir(parents=True, exist_ok=True)
        img.save(target_dir / f"{p_idx:05d}.png")
    else:
        raise ValueError(f"Unsupported prompt type layout: {prompt_type}")


def generate_benchmark_images(
    cfg: Any,
    prompt_types: List[str],
    prompt_dir: Path,
    output_dir: Path,
    resolution: int = 256,
    sample_steps: int = 250,
    cfg_scale: float = 12.0,
    shift: float = 1.0,
    prediction_target: str = "v",
    noise_scale: float = 1.0,
    clip_prediction: bool = False,
    dynamic_thresholding: bool = False,
    cfg_interval: Tuple[float, float] = (0.0, 1.0),
    pixel_sampling: bool = False,
    batch_size: int = 8,
    vae_batch_size: int = 4,
    global_seed: int = 99,
    device: str = "cuda",
) -> None:
    """Generates benchmark images with full sampling control."""
    device_obj = torch.device(device)
    torch_dtype = (
        torch.bfloat16 if cfg.train.get("dtype", "bf16") == "bf16" else torch.float32
    )

    autocast_dtype = torch.bfloat16
    capability = torch.cuda.get_device_capability()
    if capability[0] >= 7 and capability[0] < 8:
        autocast_dtype = torch.float16
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
            "Tensor cores for float32 matmul not optimally supported or GPU is older."
        )
    logging.info("Loading model weights and components...")
    unet, text_encoder, vae, tokenizer, _ = load_trainable_model(
        models_path=cfg.paths.models,
        device=device_obj,
        dtype=torch_dtype,
        train_te=False,
        use_checkpointing=False,
        resume_from_checkpoint=cfg.models.get("resume_from_checkpoint", None),
        train_only_output=False,
        global_rank=0,
        model_type=getattr(cfg.models, "model_type", "dual_stream"),
        model_cfg=cfg.models,
        autocast_dtype=autocast_dtype,
    )
    unet.eval().requires_grad_(False)
    text_encoder.eval().requires_grad_(False)
    vae.eval().requires_grad_(False)
    unet = torch.compile(unet)
    text_encoder = torch.compile(text_encoder)

    schedule = LinearSchedule(device=device_obj)
    vae_mean = getattr(cfg.models, "vae_mean", 0.0)
    vae_std = getattr(cfg.models, "vae_std", 1.0 / 0.18215)
    vae_mean_t = torch.tensor(vae_mean, device=device_obj, dtype=torch_dtype).view(
        1, -1, 1, 1
    )
    vae_std_t = torch.tensor(vae_std, device=device_obj, dtype=torch_dtype).view(
        1, -1, 1, 1
    )
    in_channels = cfg.models.get("in_channels", 4)
    coord_system = (
        "aspect_norm"
        if getattr(cfg.models, "use_calibrated_spatial", False)
        else "discrete"
    )

    for p_type in prompt_types:
        logging.info(f"Processing benchmark prompt set: {p_type}")
        captions = get_caption_list(p_type, prompt_dir)
        work_items = build_work_items(p_type, captions)

        type_output_dir = output_dir / p_type
        type_output_dir.mkdir(parents=True, exist_ok=True)

        for b_start in tqdm(range(0, len(work_items), batch_size), desc=p_type):
            chunk = work_items[b_start : b_start + batch_size]
            sample_configs = []
            for item_idx, (meta_idx, text_prompt) in enumerate(chunk):
                # Deterministic per-sample seed calculation
                global_item_idx = b_start + item_idx
                seed = global_seed + global_item_idx
                sample_configs.append(
                    {
                        "prompt": text_prompt,
                        "negative_prompt": "",
                        "height": resolution,
                        "width": resolution,
                        "sample_steps": sample_steps,
                        "cfg_scale": cfg_scale,
                        "shift": shift,
                        "seed": seed,
                    }
                )

            images = generate_samples(
                unet=unet,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                vae=vae,
                schedule=schedule,
                sample_configs=sample_configs,
                global_batch_size=len(chunk),
                diffusion_type="flow_matching",
                device=device_obj,
                dtype=torch_dtype,
                autocast_dtype=torch.bfloat16,
                use_unet_mult=False,
                vae_mean=vae_mean_t,
                vae_std=vae_std_t,
                in_channels=in_channels,
                coord_system=coord_system,
                pixel_sampling=pixel_sampling,
                prediction_target=prediction_target,
                noise_scale=noise_scale,
                clip_prediction=clip_prediction,
                dynamic_thresholding=dynamic_thresholding,
                cfg_interval=cfg_interval,
                vae_batch_size=vae_batch_size,
            )

            for (meta_idx, _), img in zip(chunk, images):
                save_image_sample(img, p_type, meta_idx, type_output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark images using i1 prompt structures."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--prompt-type",
        nargs="+",
        default=["dpg_complex_rewrite", "geneval_complex_rewrite", "longtext"],
        help="Prompt set name matching i1 catalog.",
    )
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=Path("benchmark_eval/prompts"),
        help="Directory containing benchmark prompt files from i1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_output"),
        help="Root destination path for generated samples.",
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--sample-steps", type=int, default=250)
    parser.add_argument("--cfg-scale", type=float, default=12.0)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--prediction-target", type=str, default="v")
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--clip-prediction", action="store_true", default=False)
    parser.add_argument("--dynamic-thresholding", action="store_true", default=False)
    parser.add_argument("--cfg-interval", nargs=2, type=float, default=[0.0, 1.0])
    parser.add_argument("--pixel-sampling", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vae-batch-size", type=int, default=4)
    parser.add_argument("--global-seed", type=int, default=99)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.checkpoint:
        cfg.models.resume_from_checkpoint = args.checkpoint
    cfg.models.drop_ratio = 0.0

    generate_benchmark_images(
        cfg=cfg,
        prompt_types=args.prompt_type,
        prompt_dir=args.prompt_dir,
        output_dir=args.output_dir,
        resolution=args.resolution,
        sample_steps=args.sample_steps,
        cfg_scale=args.cfg_scale,
        shift=args.shift,
        prediction_target=args.prediction_target,
        noise_scale=args.noise_scale,
        clip_prediction=args.clip_prediction,
        dynamic_thresholding=args.dynamic_thresholding,
        cfg_interval=tuple(args.cfg_interval),
        pixel_sampling=args.pixel_sampling,
        batch_size=args.batch_size,
        vae_batch_size=args.vae_batch_size,
        global_seed=args.global_seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
