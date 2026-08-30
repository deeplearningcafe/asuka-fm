import argparse
from datetime import datetime
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.diffusion.sampling import generate_samples
from src.diffusion.schedules import DDPMSchedule, LinearSchedule
from src.models.factory import load_trainable_model


def make_image_grid(
    images: List[Image.Image],
    rows: Optional[int] = None,
    cols: Optional[int] = None,
) -> Optional[Image.Image]:
    """Combines a list of PIL Images into a single grid Image."""
    if not images:
        return None
    n = len(images)
    if cols is None:
        cols = int(math.ceil(math.sqrt(n)))
    if rows is None:
        rows = int(math.ceil(n / cols))

    w, h = images[0].size
    grid_w = cols * w
    grid_h = rows * h
    grid_img = Image.new("RGB", (grid_w, grid_h))

    for i, img in enumerate(images):
        x = (i % cols) * w
        y = (i // cols) * h
        grid_img.paste(img, (x, y))
    return grid_img


def generate_images_ui(
    model_bundle: Dict[str, Any],
    prompt: str,
    neg_prompt: str,
    height: int,
    width: int,
    num_samples: int,
    steps: int,
    cfg_scale: float,
    batch_size: int,
    seed: int,
    zoom: float,
    x_shift: float,
    y_shift: float,
    shift: float,
    output_dir: str,
) -> List[Image.Image]:
    """Runs sampling through the diffusion pipeline and embeds metadata."""
    cfg = model_bundle["cfg"]
    device = model_bundle["device"]
    dtype = model_bundle["dtype"]
    autocast_dtype = model_bundle["autocast_dtype"]
    unet = model_bundle["unet"]
    text_encoder = model_bundle["text_encoder"]
    tokenizer = model_bundle["tokenizer"]
    vae = model_bundle["vae"]
    schedule = model_bundle["schedule"]
    is_dit = model_bundle["is_dit"]
    vae_mean = model_bundle["vae_mean"]
    vae_std = model_bundle["vae_std"]
    in_channels = model_bundle["in_channels"]
    coord_system = model_bundle["coord_system"]
    objective = cfg.train.get("objective", "flow_matching")

    base_seed = seed if (seed is not None and seed >= 0) else random.randint(
        0, 2**31 - 1
    )

    sample_configs = []
    for i in range(num_samples):
        sample_configs.append(
            {
                "prompt": prompt,
                "negative_prompt": neg_prompt,
                "height": int(height),
                "width": int(width),
                "sample_steps": int(steps),
                "cfg_scale": float(cfg_scale),
                "shift": float(shift),
                "zoom": float(zoom),
                "x_shift": float(x_shift),
                "y_shift": float(y_shift),
                "seed": base_seed + i,
            }
        )

    all_images = generate_samples(
        unet=unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        schedule=schedule,
        sample_configs=sample_configs,
        global_batch_size=int(batch_size),
        diffusion_type=objective,
        device=device,
        dtype=dtype,
        autocast_dtype=autocast_dtype,
        use_unet_mult=not is_dit,
        vae_mean=vae_mean,
        vae_std=vae_std,
        in_channels=in_channels,
        coord_system=coord_system,
    )

    grid_img = make_image_grid(all_images)

    # Save to timestamp directory
    save_path = output_dir
    os.makedirs(save_path, exist_ok=True)
    ts_ms = int(time.time() * 1000)

    for idx, (img, c) in enumerate(zip(all_images, sample_configs)):
        meta_payload = {
            "prompt": c["prompt"],
            "neg_prompt": c["negative_prompt"],
            "height": c["height"],
            "width": c["width"],
            "steps": c["sample_steps"],
            "cfg_scale": c["cfg_scale"],
            "shift": c["shift"],
            "zoom": c["zoom"],
            "x_shift": c["x_shift"],
            "y_shift": c["y_shift"],
            "seed": c["seed"],
            "objective": objective,
            "model_type": getattr(cfg.models, "model_type", "unet"),
            "checkpoint": getattr(
                cfg.models, "resume_from_checkpoint", "base"
            ),
        }
        png_info = PngInfo()
        png_info.add_text("parameters", json.dumps(meta_payload))

        img_filename = f"sample_{ts_ms}_{idx:03d}.png"
        img.save(os.path.join(save_path, img_filename), pnginfo=png_info)

    if grid_img is not None:
        grid_info = PngInfo()
        grid_info.add_text(
            "parameters",
            json.dumps({"prompt": prompt, "neg_prompt": neg_prompt, "base_seed": base_seed}),
        )
        grid_img.save(
            os.path.join(save_path, f"grid_{ts_ms}.png"), pnginfo=grid_info
        )

    return [grid_img] + all_images if grid_img is not None else all_images


def read_metadata(image: Optional[Image.Image]) -> str:
    """Extracts and formats JSON parameter string from PNG metadata."""
    if image is None:
        return "Upload an image to inspect generation metadata."
    try:
        parameters = image.info.get("parameters", "")
        if not parameters:
            return "No metadata found in this image."
        data = json.loads(parameters)
        return json.dumps(data, indent=4)
    except Exception:
        raw_data = image.info.get("parameters", "None")
        return f"Raw parameters:\n{raw_data}"


def send_to_generation(image: Optional[Image.Image]) -> List[Any]:
    """Parses PNG metadata dictionary to populate UI generation sliders."""
    if image is None:
        return [gr.skip()] * 11
    try:
        parameters = image.info.get("parameters", "")
        if not parameters:
            return [gr.skip()] * 11
        data = json.loads(parameters)
        return [
            data.get("prompt", gr.skip()),
            data.get("neg_prompt", gr.skip()),
            data.get("height", gr.skip()),
            data.get("width", gr.skip()),
            data.get("steps", gr.skip()),
            data.get("cfg_scale", gr.skip()),
            data.get("seed", gr.skip()),
            data.get("zoom", gr.skip()),
            data.get("x_shift", gr.skip()),
            data.get("y_shift", gr.skip()),
            data.get("shift", gr.skip()),
        ]
    except Exception:
        return [gr.skip()] * 11


def create_ui(model_bundle: Dict[str, Any], default_out_dir: str):
    """Builds Gradio UI Blocks application."""
    theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
    ).set(
        body_background_fill="#1e1e1e",
        block_background_fill="#2d2d2d",
        body_text_color="white",
        block_label_text_color="white",
        block_title_text_color="white",
    )
    js_func = "document.body.classList.toggle('dark', true);"

    with gr.Blocks(
        title="Asuka Flow Matching Playground", theme=theme, js=js_func
    ) as demo:
        gr.Markdown("# Flow Matching DiT / UNet Sampling Studio")

        with gr.Tabs():
            with gr.Tab("Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt = gr.Textbox(
                            value=(
                                "1girl, souryuu asuka langley, neon genesis "
                                "evangelion, masterpiece, absurdres"
                            ),
                            label="Prompt",
                            lines=3,
                        )
                        neg_prompt = gr.Textbox(
                            value=(
                                "very displeasing, displeasing, bad score, "
                                "worse score"
                            ),
                            label="Negative Prompt",
                            lines=2,
                        )

                        with gr.Row():
                            height = gr.Slider(
                                128, 1024, value=256, step=64, label="Height"
                            )
                            width = gr.Slider(
                                128, 1024, value=256, step=64, label="Width"
                            )

                        with gr.Row():
                            steps = gr.Slider(
                                1, 150, value=30, step=1, label="Sampling Steps"
                            )
                            cfg_scale = gr.Slider(
                                1.0, 20.0, value=6.0, step=0.5, label="CFG Scale"
                            )

                        with gr.Row():
                            num_samples = gr.Slider(
                                1, 32, value=4, step=1, label="Number of Samples"
                            )
                            batch_size = gr.Slider(
                                1, 16, value=4, step=1, label="Batch Size"
                            )

                        with gr.Accordion("Camera & Viewport Controls", open=True):
                            with gr.Row():
                                zoom = gr.Slider(
                                    0.5, 2.0, value=1.0, step=0.05, label="Zoom"
                                )
                                shift = gr.Slider(
                                    0.1, 5.0, value=1.0, step=0.1, label="Time Shift"
                                )
                            with gr.Row():
                                x_shift = gr.Slider(
                                    -1.0, 1.0, value=0.0, step=0.05, label="X Shift"
                                )
                                y_shift = gr.Slider(
                                    -1.0, 1.0, value=0.0, step=0.05, label="Y Shift"
                                )

                        seed = gr.Number(
                            value=-1, label="Seed (-1 for Random)", precision=0
                        )
                        output_dir = gr.Textbox(
                            value=default_out_dir, label="Output Root Directory"
                        )
                        generate_btn = gr.Button(
                            "Generate Samples", variant="primary"
                        )

                    with gr.Column(scale=1):
                        gallery = gr.Gallery(
                            label="Generated Output (First image is Grid)",
                            columns=2,
                            rows=2,
                            object_fit="contain",
                            height="auto",
                        )

            with gr.Tab("PNG Info"):
                with gr.Row():
                    with gr.Column(scale=1):
                        info_image = gr.Image(type="pil", label="Upload Generated PNG")
                    with gr.Column(scale=1):
                        metadata_text = gr.Textbox(
                            label="Extracted Metadata",
                            interactive=False,
                            lines=14,
                        )
                        send_btn = gr.Button("Send Parameters to Generation Tab")

        def _run_ui_gen(
            p_val,
            np_val,
            h_val,
            w_val,
            samples_val,
            steps_val,
            cfg_val,
            bsz_val,
            seed_val,
            zoom_val,
            xs_val,
            ys_val,
            shift_val,
            out_val,
        ):
            return generate_images_ui(
                model_bundle=model_bundle,
                prompt=p_val,
                neg_prompt=np_val,
                height=h_val,
                width=w_val,
                num_samples=int(samples_val),
                steps=int(steps_val),
                cfg_scale=float(cfg_val),
                batch_size=int(bsz_val),
                seed=int(seed_val),
                zoom=float(zoom_val),
                x_shift=float(xs_val),
                y_shift=float(ys_val),
                shift=float(shift_val),
                output_dir=out_val,
            )

        generate_btn.click(
            fn=_run_ui_gen,
            inputs=[
                prompt,
                neg_prompt,
                height,
                width,
                num_samples,
                steps,
                cfg_scale,
                batch_size,
                seed,
                zoom,
                x_shift,
                y_shift,
                shift,
                output_dir,
            ],
            outputs=[gallery],
        )

        info_image.change(
            fn=read_metadata, inputs=[info_image], outputs=[metadata_text]
        )

        send_btn.click(
            fn=send_to_generation,
            inputs=[info_image],
            outputs=[
                prompt,
                neg_prompt,
                height,
                width,
                steps,
                cfg_scale,
                seed,
                zoom,
                x_shift,
                y_shift,
                shift,
            ],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch Asuka-FM Sampling WebUI")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to training config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/sampling",
        help="Directory to save generated samples",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to serve Gradio application",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable public Gradio share link",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg.train.dtype == "bf16" else torch.float32

    autocast_dtype = torch.bfloat16
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 7 and capability[0] < 8:
            autocast_dtype = torch.float16
            torch.set_float32_matmul_precision("high")
        elif capability[0] >= 8:
            torch.set_float32_matmul_precision("medium")
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

    checkpoint_path = args.checkpoint or cfg.models.resume_from_checkpoint
    model_type = getattr(cfg.models, "model_type", "unet")

    unet, text_encoder, vae, tokenizer, _ = load_trainable_model(
        models_path=cfg.paths.models,
        device=device,
        dtype=dtype,
        train_te=False,
        use_checkpointing=False,
        resume_from_checkpoint=checkpoint_path,
        train_only_output=False,
        global_rank=0,
        model_type=model_type,
        model_cfg=cfg.models,
        autocast_dtype=autocast_dtype,
    )

    unet.eval().requires_grad_(False)
    text_encoder.eval().requires_grad_(False)
    vae.eval().requires_grad_(False)

    if cfg.train.objective == "flow_matching":
        schedule = LinearSchedule(device=device)
    else:
        schedule = DDPMSchedule(device=device)

    vae_mean = getattr(cfg.models, "vae_mean", 0.0)
    vae_std = getattr(cfg.models, "vae_std", 1.0 / 0.18215)
    vae_mean = torch.tensor(vae_mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    vae_std = torch.tensor(vae_std, device=device, dtype=dtype).view(1, -1, 1, 1)

    model_bundle = {
        "cfg": cfg,
        "device": device,
        "dtype": dtype,
        "autocast_dtype": autocast_dtype,
        "unet": unet,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "vae": vae,
        "schedule": schedule,
        "is_dit": model_type in ["dual_stream", "sprint_dual"],
        "vae_mean": vae_mean,
        "vae_std": vae_std,
        "in_channels": cfg.models.get("in_channels", 4),
        "coord_system": (
            "aspect_norm"
            if getattr(cfg.models, "use_calibrated_spatial", False)
            else "discrete"
        ),
    }

    timestamp_folder = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = os.path.join(args.output_dir, timestamp_folder)
    demo = create_ui(model_bundle, output_dir)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()