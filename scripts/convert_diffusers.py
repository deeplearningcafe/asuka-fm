import os
import re
import argparse
from omegaconf import OmegaConf
from safetensors.torch import load_file
from diffusers import UNet2DConditionModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert custom UNet to Diffusers format"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the custom model directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the training config YAML",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="diffusers_unet",
        help="Output directory for the diffusers UNet",
    )
    return parser.parse_args()


def resolve_model_path(model_path, config_path):
    """
    Resolves the model path based on priority:
    1. model_path CLI argument.
    2. config's resume_from_checkpoint.
    Throws an error if neither is valid.
    """
    if model_path is not None:
        return model_path

    if not os.path.exists(config_path):
        raise ValueError("Neither model_path was provided nor config file was found.")

    cfg = OmegaConf.load(config_path)
    ckpt_path = cfg.models.get("resume_from_checkpoint", None)

    if not ckpt_path:
        raise ValueError(
            "No model_path provided and resume_from_checkpoint is empty "
            "in config. Exporting the original model makes no sense."
        )

    return ckpt_path


def map_custom_to_diffusers(custom_sd):
    """
    Reverses the mapping applied in unet.py's from_pretrained method
    to restore standard diffusers keys.
    """
    diffusers_sd = {}
    for k, v in custom_sd.items():
        new_k = k

        # 1. Reverse Attention to_out mapping
        new_k = re.sub(
            r"\.attn([12])\.to_out\.weight", r".attn\1.to_out.0.weight", new_k
        )
        new_k = re.sub(r"\.attn([12])\.to_out\.bias", r".attn\1.to_out.0.bias", new_k)

        # 2. Reverse FeedForward/MLP block mapping
        new_k = new_k.replace("ff.geglu.proj", "ff.net.0.proj")
        new_k = new_k.replace("ff.proj_out", "ff.net.2")

        # 3. Reverse Downsamplers nesting
        # Custom uses downsamplers.0.weight, diffusers uses .0.conv.weight
        new_k = re.sub(r"downsamplers\.0\.weight", r"downsamplers.0.conv.weight", new_k)
        new_k = re.sub(r"downsamplers\.0\.bias", r"downsamplers.0.conv.bias", new_k)

        diffusers_sd[new_k] = v
    return diffusers_sd


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model_path, args.config)

    if os.path.isdir(model_path):
        if os.path.exists(os.path.join(model_path, "unet.safetensors")):
            ckpt_path = os.path.join(model_path, "unet.safetensors")
        elif os.path.exists(
            os.path.join(model_path, "unet", "diffusion_pytorch_model.safetensors")
        ):
            ckpt_path = os.path.join(
                model_path, "unet", "diffusion_pytorch_model.safetensors"
            )
        else:
            raise FileNotFoundError(f"Could not find UNet safetensors in {model_path}")
    else:
        ckpt_path = model_path

    print(f"Loading custom weights from {ckpt_path}")
    custom_sd = load_file(ckpt_path)

    print("Mapping keys to Diffusers format...")
    diffusers_sd = map_custom_to_diffusers(custom_sd)

    print("Initializing standard SD1.5 UNet2DConditionModel...")
    unet = UNet2DConditionModel(
        sample_size=64,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(320, 640, 1280, 1280),
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        cross_attention_dim=768,
        attention_head_dim=8,
        use_linear_projection=False,
    )

    print("Loading mapped weights into Diffusers model...")
    missing, unexpected = unet.load_state_dict(diffusers_sd, strict=False)

    if missing:
        print(f"Warning: Missing keys during load: {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys during load: {unexpected}")

    os.makedirs(args.output_path, exist_ok=True)
    print(f"Saving Diffusers UNet to {args.output_path}")
    unet.save_pretrained(args.output_path)
    print("Conversion complete!")


if __name__ == "__main__":
    main()
